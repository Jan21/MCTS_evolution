"""Self-play data generation: fresh lean boards -> instances -> MCTS with the
current nets -> certified plans -> labeled decision records (PROBLEM.md 5).

    PYTHONPATH=supervised_valuenet:self_play_robots python -m spr.selfplay \
        --config g24r4 --policy P.ckpt --value V.ckpt \
        --boards-dir runs/spr/boards/g24r4_iter1 --board-ids 5000-5099 \
        --per-board 8 --expansions 600 --stop-after 150 --root-noise 0.25 \
        --workers 8 --device cuda --out runs/spr/selfplay/g24r4_iter1/records.jsonl

Contract (sections 4.3/4.6): a candidate gets a label ONLY if a complete plan
through it was strictly realized (`eval.realize.strict_moves`) somewhere in
the search tree (or by the optional greedy sibling completion) --
`cost_to_go = plan_cost(certified plan) - fixed_g(decision plan)`, a certified
upper bound in the exact/descent labelers' own units (descent.py:249; the
strict realized total rides along as `strict_total`); `is_optimal` = argmin
of those certified bounds. The search itself minimizes STRICT moves.
Records use the frozen 18-field schema (drop-in for spr.train and the
supervised trainers) plus provenance (`label_source`, `strict_total`,
`visits`, `prior`, `ctg_hat`, `n`, `iter`, `label_model*`). Instances the
search cannot certify produce nothing. Board ids are config-namespaced and
DISJOINT from the pinned pool (0-1199), so leakage into the bench is
structurally impossible; the trainer takes them via `--splits`.

Boards are written first (`nn_labeler.leanboard.write_board`, seed = --seed
so an iteration's boards are reproducible), then W worker processes (spawn;
each owns a copy of the nets on the device) solve instances and stream
records back; the parent writes atomically (tmp + rename) with a manifest.
Last line: `SPR SELFPLAY DONE <out>`.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import signal
import sys
import time
from pathlib import Path

from spr import REPO, SV

_W = {}      # per-worker state


class _Timeout(Exception):
    pass


def _alarm(_s, _f):
    raise _Timeout()


def _init_worker(cfg_name, boards_dir, policy_path, value_path, device, opts):
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, int(opts.get("threads", 2))))
    from scaling.configs import get, apply_env
    cfg = get(cfg_name)
    apply_env(cfg)                       # one config per process
    from nn_labeler import leanboard
    from skeleton.astar import AStar
    from skeleton import heuristics
    from spr.nets import load_policy, load_value
    from spr.search import Evaluator
    _W["cfg"] = cfg
    _W["boards_dir"] = Path(boards_dir)
    _W["leanboard"] = leanboard
    _W["solver"] = AStar(propose=heuristics.propose, max_iters=4000, max_frontier=40_000)
    _W["policy"] = load_policy(policy_path, device)
    _W["value"] = load_value(value_path, device)
    _W["ev"] = Evaluator(_W["policy"], _W["value"], device)
    _W["opts"] = opts
    _W["device"] = device
    signal.signal(signal.SIGALRM, _alarm)


def _greedy_complete(env, state, solver, ev, env_id, n, plan, k, prefix_filter, max_depth=16):
    """Greedy net-guided completion (descent.py::nn_complete with policy top-k
    + value argmin). Returns the complete plan or None."""
    from spr.search import expand, forced_fixes
    plan = forced_fixes(env, state, solver, plan)
    for _ in range(max_depth):
        if plan.is_complete():
            return plan
        kids, _pr = expand(env, state, solver, plan, env_id, n, ev, k, prefix_filter)
        if not kids:
            return None
        best = min(kids, key=lambda c: c.ctg_hat)
        plan = forced_fixes(env, state, solver, best.plan)
    return plan if plan.is_complete() else None


def _extract_records(root, env, state, solver, ev, env_id, n, opts, cert, label_meta):
    """Walk the tree; label candidates by the min certified strict cost in
    their subtree (plus optional greedy sibling completion); emit groups with
    >= 2 labeled candidates along the principal path (or all expanded nodes)."""
    from nn.generate import _fixed_g
    from eval.realize import prefix_playable
    k = opts["k"]
    emit_all = opts.get("emit") == "all"
    complete_sib = bool(opts.get("complete_siblings"))
    prefix_filter = None
    if opts.get("prefix_check", True):
        pfx = {}
        from eval.realize import prefix_key
        def prefix_filter(pl):
            key = prefix_key(pl)
            hit = pfx.get(key)
            if hit is None:
                hit = pfx[key] = prefix_playable(env, state, pl)
            return hit

    # principal path: follow best_cert downward from root
    path_nodes = []
    node = root
    while node is not None and node.expanded:
        path_nodes.append(node)
        nxt = None
        for c in node.children:
            if c.best_cert is not None and c.best_cert == node.best_cert:
                nxt = c
                break
        node = nxt
    todo = path_nodes if not emit_all else [nd for nd in _iter_nodes(root) if nd.expanded]
    records = []
    stats = {"decisions": 0, "cand_labeled": 0, "cand_unlabeled": 0,
             "sibling_completed": 0, "sibling_failed": 0, "groups_emitted": 0}
    for depth, nd in enumerate(todo):
        if not nd.children:
            continue
        fixed_g = float(_fixed_g(nd.plan))
        labeled = []
        for c in nd.children:
            strict, abs_cost = c.best_cert, c.best_abs
            if strict is None and complete_sib and not c.dead:
                done = _greedy_complete(env, state, solver, ev, env_id, n, c.plan, k,
                                        prefix_filter)
                if done is not None:
                    m, _mv = cert(done)
                    if m is not None:
                        strict, abs_cost = m, float(done.cost())
                        stats["sibling_completed"] += 1
                    else:
                        stats["sibling_failed"] += 1
                else:
                    stats["sibling_failed"] += 1
            if strict is None:
                stats["cand_unlabeled"] += 1
                continue
            labeled.append((c, int(strict), int(round(abs_cost))))
        stats["decisions"] += 1
        stats["cand_labeled"] += len(labeled)
        if len(labeled) < 2:
            continue
        # label units = the exact/descent labelers' (abstract plan cost of the
        # certified plan minus the decision's fixed cost, descent.py:249); the
        # strict realized total rides along for audits and the metric.
        best = min(ab for _, _, ab in labeled)
        for c, strict, ab in labeled:
            rec = dict(c.child_obj.rec)
            rec.update({
                "cost_to_go": int(ab - fixed_g), "is_optimal": ab == best,
                # group key = (..., seg_start, seg_end, depth): off-path nodes at
                # the same tree depth could alias, so `all` mode namespaces
                # them by node index (path mode keeps the corpus depth semantics)
                "depth": nd.depth if not emit_all else 1000 * depth + nd.depth,
                "n": n, "label_source": "spr_mcts", "ctg_certified": True,
                "strict_total": int(strict), "abstract_total": int(ab), "fixed_g": fixed_g,
                "visits": int(c.N), "prior": round(float(c.prior), 5),
                "ctg_hat": round(float(c.child_obj.ctg_hat), 3),
                **label_meta,
            })
            records.append(rec)
        stats["groups_emitted"] += 1
    return records, stats


def _iter_nodes(root):
    stack = [root]
    while stack:
        nd = stack.pop()
        yield nd
        stack.extend(nd.children)


def _work(task):
    """task = (board_id, per_board, seed) -> (records, per-instance stats)."""
    board_id, per_board, seed = task
    from GridEnv import State
    from nn.generate import random_instance
    from simulate import _board_size
    from spr.search import mcts, Certifier
    opts = _W["opts"]
    lb = _W["leanboard"]
    env, s0 = lb.from_env(board_id, env_dir=_W["boards_dir"])
    n = _board_size(env.grid_data, None)
    colors = [s0.target_robot.color] + [h.color for h in s0.helpers]
    rng = random.Random(seed * 1000003 + board_id)
    solver, ev = _W["solver"], _W["ev"]
    label_meta = {"iter": opts["iter"], "label_model": opts["label_model"],
                  "search_budget": opts["expansions"], "boards_dir": str(_W["boards_dir"])}
    out_recs, inst_stats = [], []
    attempts = 0
    kept = 0
    while kept < per_board and attempts < per_board * 3:
        attempts += 1
        st = random_instance(env, colors, rng)
        acct = {}
        cert = Certifier(env, st, acct, dump=False)
        t0 = time.time()
        signal.alarm(int(opts["timeout"]))
        try:
            prefix_filter = None
            if opts.get("prefix_check", True):
                from eval.realize import prefix_playable, prefix_key
                pfx = {}
                def prefix_filter(pl, _e=env, _s=st, _c=pfx):
                    key = prefix_key(pl)
                    hit = _c.get(key)
                    if hit is None:
                        hit = _c[key] = prefix_playable(_e, _s, pl)
                    return hit
            res = mcts(env, st, solver, ev, board_id, n, opts["k"], opts["expansions"],
                       prefix_filter=prefix_filter, best_at_budget=True,
                       c_puct=opts["c_puct"], backup=opts["backup"], acct=acct,
                       dump_moves=False, root_noise=opts["root_noise"],
                       rng=random.Random(rng.randrange(1 << 30)),
                       stop_after_certified=opts["stop_after"],
                       root_k=(0 if opts.get("root_all") else None))
            recs, st_ = ([], {})
            if res.strict is not None:
                first = (res.extra or {}).get("first_certified_expansion")
                if first is not None and first < opts.get("min_expansions", 0):
                    status = "trivial"      # hard-instance focus (PROBLEM.md 6.4)
                else:
                    recs, st_ = _extract_records(res.root, env, st, solver, ev, board_id, n,
                                                 opts, cert, label_meta)
                    status = "ok"
            else:
                status = "ok"
        except _Timeout:
            res, recs, st_, status = None, [], {}, "timeout"
        except Exception as e:  # noqa: BLE001 -- one bad instance must not kill the worker
            res, recs, st_, status = None, [], {}, f"error:{type(e).__name__}"
        finally:
            signal.alarm(0)
        row = {"env_id": board_id, "status": status,
               "target": list(st.target), "target_robot": [list(st.target_robot.position),
                                                           st.target_robot.color],
               "helpers": [[list(h.position), h.color] for h in st.helpers],
               "seconds": round(time.time() - t0, 2), "records": len(recs)}
        if res is not None:
            row.update({"solved": res.strict is not None, "strict": res.strict,
                        "expansions": res.expansions, "rejected": res.rejected,
                        **{k2: v for k2, v in res.extra.items()}, **st_})
        inst_stats.append(row)
        if recs:
            kept += 1
            out_recs.extend(recs)
    return out_recs, inst_stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--value", required=True)
    p.add_argument("--boards-dir", required=True, help="lean boards for this iteration")
    p.add_argument("--board-ids", required=True, help="e.g. 5000-5099 (config-namespaced, "
                                                     "must not overlap 0-1199)")
    p.add_argument("--per-board", type=int, default=8)
    p.add_argument("--seed", type=int, default=1, help="boards + instances")
    p.add_argument("--iter", type=int, default=1)
    p.add_argument("--expansions", type=int, default=600)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--stop-after", type=int, default=150,
                   help="stop when the best certified plan has not improved for this "
                        "many expansions")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--backup", choices=["min", "mean"], default="min")
    p.add_argument("--root-noise", type=float, default=0.25)
    p.add_argument("--no-prefix-check", action="store_true")
    p.add_argument("--emit", choices=["path", "all"], default="path")
    p.add_argument("--complete-siblings", action="store_true",
                   help="greedy value-descent completion + certification for "
                        "top-k siblings the tree never certified")
    p.add_argument("--root-all", action="store_true",
                   help="score ALL candidates at the root (not top-k) so depth-0 "
                        "labels cover the full candidate set (fidelity gauge "
                        "comparability); one value pass over all root candidates")
    p.add_argument("--timeout", type=int, default=300, help="per-instance wall cap (s)")
    p.add_argument("--min-expansions", type=int, default=0,
                   help="drop instances whose first certified plan needed fewer "
                        "expansions than this (hard-instance focus, PROBLEM.md 6.4; "
                        "0 = keep all)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    from scaling.configs import get, parse_ids
    from nn_labeler import leanboard
    cfg = get(a.config)
    ids = parse_ids(a.board_ids)
    bad = [i for i in ids if i <= 1199]
    if bad:
        raise SystemExit(f"board ids overlap the pinned pool 0-1199: {bad[:5]}")
    boards_dir = Path(a.boards_dir)
    if not boards_dir.is_absolute():
        boards_dir = REPO / boards_dir
    boards_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    written = 0
    for i in ids:
        if not (boards_dir / f"env_{i}.pkl").exists():
            leanboard.write_board(boards_dir, i, cfg.grid, walls=cfg.walls,
                                  robots=cfg.robots, seed=a.seed)
            written += 1
    print(f"[selfplay] boards: {len(ids)} ids in {boards_dir} ({written} written, "
          f"{time.time() - t0:.0f}s)", flush=True)

    opts = dict(k=a.k, expansions=a.expansions, stop_after=a.stop_after, c_puct=a.c_puct,
                backup=a.backup, root_noise=a.root_noise,
                prefix_check=not a.no_prefix_check, emit=a.emit,
                complete_siblings=a.complete_siblings, root_all=a.root_all,
                timeout=a.timeout, min_expansions=a.min_expansions,
                iter=a.iter, threads=a.threads,
                label_model=f"{Path(a.policy).name}|{Path(a.value).name}")
    tasks = [(i, a.per_board, a.seed) for i in ids]
    out = Path(a.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    stats_all = []
    n_rec = n_inst = n_solved = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init_worker,
                  initargs=(a.config, str(boards_dir), a.policy, a.value, a.device, opts)) as pool, \
            open(tmp, "w") as f:
        for bi, (recs, inst_stats) in enumerate(pool.imap_unordered(_work, tasks, chunksize=1)):
            for r in recs:
                f.write(json.dumps(r) + "\n")
            n_rec += len(recs)
            n_inst += len(inst_stats)
            n_solved += sum(1 for s in inst_stats if s.get("solved"))
            stats_all.extend(inst_stats)
            if (bi + 1) % 5 == 0 or bi + 1 == len(tasks):
                print(f"[selfplay] {bi + 1}/{len(tasks)} boards, {n_inst} instances, "
                      f"{n_solved} solved, {n_rec} records ({time.time() - t0:.0f}s)",
                      flush=True)
    os.replace(tmp, out)
    solved = [s for s in stats_all if s.get("solved")]
    manifest = {
        "config": cfg.name, "grid": cfg.grid, "robots": cfg.robots, "iter": a.iter,
        "policy": a.policy, "value": a.value, "boards_dir": str(boards_dir),
        "board_ids": a.board_ids, "per_board": a.per_board, "seed": a.seed,
        "search": {k2: opts[k2] for k2 in ("k", "expansions", "stop_after", "c_puct",
                                            "backup", "root_noise", "prefix_check",
                                            "emit", "complete_siblings", "root_all",
                                            "timeout", "min_expansions")},
        "workers": a.workers, "device": a.device,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "seconds": round(time.time() - t0, 1),
        "instances": n_inst, "solved": len(solved), "records": n_rec,
        "mean_strict": (sum(s["strict"] for s in solved) / len(solved)) if solved else None,
        "mean_expansions": (sum(s.get("expansions", 0) for s in stats_all) / n_inst) if n_inst else None,
        "mean_seconds": (sum(s["seconds"] for s in stats_all) / n_inst) if n_inst else None,
        "status_counts": {k2: sum(1 for s in stats_all if s["status"] == k2)
                          for k2 in sorted({s["status"] for s in stats_all})},
    }
    out.with_suffix(out.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=1))
    out.with_suffix(out.suffix + ".instances.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in stats_all))
    print(f"[selfplay] manifest: {json.dumps({k2: v for k2, v in manifest.items() if k2 != 'search'})}",
          flush=True)
    print(f"SPR SELFPLAY DONE {out}", flush=True)


if __name__ == "__main__":
    main()
