"""Self-play generation for the PRIMITIVE-MOVE arm: fresh boards -> MCTS -> records.

    PYTHONPATH=supervised_valuenet:self_play_robots python -m spr.fwd.selfplay \
        --ckpt move_planner/checkpoints/candidate_scored.ckpt \
        --boards-dir runs/spr/fwd/boards/g16r4_iter1 --board-ids 5000-5119 \
        --per-board 8 --expansions 600 --stop-after 150 --root-noise 0.25 \
        --workers 8 --device cuda --out runs/spr/fwd/selfplay/g16r4_iter1/records.jsonl

WHAT IS NEW vs `move_planner_v2` (the existing forward expert-iteration loop):
the expert is PUCT MCTS (`spr.fwd.mcts`, best-at-budget) instead of the net's own
budgeted A*. Everything else is deliberately theirs, so the two are comparable
and `spr.fwd.train` / `move_planner.net` consume the output verbatim:

  * record schema = `move_planner_v2.selfplay._record` / `exit_records`, i.e. the
    SUPERVISED generator's JSONL schema
    `{env_id, robots, target, target_idx, cost_to_go, best_moves, legal_moves,
      depth, full:false}`;
  * value target = the plain REMAINING PATH LENGTH along the best path the search
    actually found (`T - j`), a Monte-Carlo return that is a certified upper
    bound, plus a `cost_to_go = 0` goal anchor (DESIGN.md section 4);
  * policy target = the COMMITTED move at each decision state;
  * unsolved instances emit nothing and are dropped -- net-solvability is the
    implicit curriculum (DESIGN.md section 5.2).

ON TOP of that, each decision record also carries `mcts_visits`
(`[[slot, dir, N], ...]` from the tree node at that state) -- the AlphaZero
policy target proper, kept as an EXTRA field so it is available to a future
visit-distribution trainer while `MoveDataset` (which reads only `best_moves`)
is unaffected.

CERTIFICATION (PROBLEM.md 4.6): the forward search's path is a certificate by
construction (every edge is a legal non-no-op slide), and it is replayed through
`apply_move` before any record is written (`spr.fwd.mcts.verify_path`).
Uncertifiable paths are impossible here, but the check is kept as the structural
guard the subgoal arm needs and this one must not silently lose.

INSTANCES ARE ORACLE-FREE. `--sampler mix` (default) draws
`move_planner_v2.start_states.sample_solvable_instance` (the eval distribution
under a `relaxed_target_dist` reachability filter -- NO oracle solve) with
probability `--rand-mix-prob`, else `forward_walk_relabel` (target-biased forward
walk + goal relabel, whose walk length is itself the solvability witness).
`--d-star diag` additionally runs `move_planner.oracle.solve` and stores the
result as `d_star` with `d_star_source="oracle_diagnostic"`: a DIAGNOSTIC ONLY
(the fidelity gauge of PROBLEM.md 4.1), never a training label -- no field the
trainer reads is derived from it.

BOARDS. Fresh lean boards (`nn_labeler.leanboard.write_board`) in a
per-iteration dir; ids must be >= 5000 so they are disjoint from the pinned pool
(0-2999 at g16r4, whose 2400-2549 carry the bench). The forward stack reads
boards from `$RR_ENV_DIR` AT IMPORT TIME, so this module sets RR_ENV_DIR before
the first forward-stack import and the workers (spawn) inherit it. Lean pkls
carry `grid_data` and `grid_graph`, which is exactly what `train.encode.walls_for`,
`_slide_fields` and `_graph`/`_adj` read.

Writes `<out>`, `<out>.manifest.json`, `<out>.instances.jsonl` atomically.
Last line: `SPR FWD SELFPLAY DONE <out>`.
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

_W = {}      # per-worker state


class _Timeout(Exception):
    pass


def _alarm(_s, _f):
    raise _Timeout()


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

def _init_worker(boards_dir, ckpt, device, opts):
    os.environ["RR_ENV_DIR"] = str(boards_dir)
    for k, v in (opts.get("rr_env") or {}).items():
        os.environ[k] = v
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, int(opts.get("threads", 2))))
    from move_planner.evaluate import Guide
    from eval.compare import _CountingGuide
    from nn.gen_grids import GRID
    _W["guide"] = _CountingGuide(Guide(ckpt, device))
    _W["grid"] = GRID
    _W["opts"] = opts
    _W["boards_dir"] = str(boards_dir)
    signal.signal(signal.SIGALRM, _alarm)


def _sample_instance(env_id, rng, wr, wd, opts):
    """(positions, target_idx, target, sampler) -- ORACLE-FREE."""
    from move_planner_v2.start_states import (forward_walk_relabel,
                                              sample_solvable_instance)
    mode = opts["sampler"]
    if mode == "mix":
        mode = "random" if rng.random() < opts["rand_mix_prob"] else "walk"
    if mode == "random":
        got = sample_solvable_instance(env_id, rng, wr, wd)
        return (*got, "random") if got else None
    k = rng.randint(1, opts["walk_k_max"])
    got = forward_walk_relabel(env_id, k, rng, wr, wd, opts["walk_target_bias"])
    return (*got, f"walk{k}") if got else None


def _visits_along_path(root, path):
    """[[ [slot, dir, N], ... ] per decision state] following `path` from the root."""
    out, node = [], root
    for (s, d) in path:
        out.append([[int(a), int(b), int(n)] for a, b, n, _p, _q in node.child_visits()])
        nxt = next((c for c in node.children if c.move == (s, d)), None)
        if nxt is None:                     # cannot happen for a tree path; be safe
            break
        node = nxt
    return out


def _work(task):
    """task = (board_id, per_board, seed) -> (records, per-instance stats)."""
    board_id, per_board, seed = task
    from train.encode import walls_for
    from move_planner_v2.selfplay import exit_records, path_to_states
    from spr.fwd.mcts import mcts, verify_path
    opts, guide, grid = _W["opts"], _W["guide"], _W["grid"]
    wr, wd = walls_for(board_id)
    rng = random.Random(seed * 1000003 + board_id)
    label_meta = {"label_source": "spr_fwd_mcts", "iter": opts["iter"],
                  "label_model": Path(opts["ckpt"]).name,
                  "search_budget": opts["expansions"], "search_k": opts["k"],
                  "boards_dir": _W["boards_dir"], "n": grid,
                  "certified": True}
    out_recs, inst_stats = [], []
    attempts = kept = 0
    while kept < per_board and attempts < per_board * 3:
        attempts += 1
        got = _sample_instance(board_id, rng, wr, wd, opts)
        if got is None:
            continue
        start, tidx, target, sampler = got
        c0 = guide.calls
        t0 = time.time()
        acct = {}
        signal.alarm(int(opts["timeout"]))
        try:
            res = mcts(guide, board_id, start, tidx, target, wr, wd, size=grid,
                       k=opts["k"], max_expansions=opts["expansions"],
                       c_puct=opts["c_puct"], backup=opts["backup"],
                       best_at_budget=True, root_noise=opts["root_noise"],
                       rng=random.Random(rng.randrange(1 << 30)),
                       stop_after_certified=opts["stop_after"],
                       root_k=(0 if opts.get("root_all") else None),
                       max_depth=opts["max_depth"], acct=acct)
            status = "ok"
        except _Timeout:
            res, status = None, "timeout"
        except Exception as e:                # one bad instance must not kill a worker
            res, status = None, f"error:{type(e).__name__}:{e}"
        finally:
            signal.alarm(0)
        recs = []
        d_star = None
        if res is not None and res.cost is not None:
            ok, why = verify_path(start, res.path, tidx, target, wr, wd, grid)
            if not ok:
                status = f"uncertified:{why}"
            else:
                states = path_to_states(start, res.path, wr, wd, grid)
                recs = exit_records(board_id, states, res.path, tidx, target, wr, wd)
                visits = _visits_along_path(res.root, res.path)
                if opts["d_star"] == "diag":
                    d_star = _oracle_d_star(start, tidx, target, wr, wd, grid, opts)
                for j, r in enumerate(recs):
                    r.update(label_meta)
                    r["strict_total"] = int(res.cost)
                    r["sampler"] = sampler
                    r["mcts_root_N"] = int(res.extra.get("root_N", 0))
                    r["mcts_expansions"] = int(res.expansions)
                    if j < len(visits):
                        r["mcts_visits"] = visits[j]
                    if d_star is not None:
                        r["d_star"] = int(d_star)          # DIAGNOSTIC, never a label
                        r["d_star_source"] = "oracle_diagnostic"
        row = {"env_id": board_id, "status": status, "sampler": sampler,
               "target": list(target), "target_idx": int(tidx),
               "positions": [list(p) for p in start],
               "solved": bool(recs), "moves": (res.cost if res is not None else None),
               "expansions": (res.expansions if res is not None else None),
               "nn_calls": guide.calls - c0, "records": len(recs),
               "seconds": round(time.time() - t0, 2)}
        if d_star is not None:
            row["d_star"] = int(d_star)
            row["regret"] = None if res is None or res.cost is None else res.cost - d_star
        if res is not None:
            row["search"] = res.extra
        inst_stats.append(row)
        if recs:
            kept += 1
            out_recs.extend(recs)
    return out_recs, inst_stats


def _oracle_d_star(start, tidx, target, wr, wd, grid, opts):
    """EXACT optimum, DIAGNOSTIC ONLY (fidelity gauge). Never a training label."""
    from move_planner import oracle as move_bfs
    hd = move_bfs.relaxed_target_dist(target, wr, wd, grid)
    return move_bfs.solve(start, tidx, target, wr, wd, grid, hd,
                          max_expansions=opts["oracle_expansions"])


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="g16r4", help="board config (grid/robots/walls)")
    p.add_argument("--ckpt", required=True, help="MoveNet checkpoint (the current net)")
    p.add_argument("--boards-dir", required=True, help="lean boards for this iteration")
    p.add_argument("--board-ids", required=True,
                   help="e.g. 5000-5119 (must not overlap the pinned pool 0-2999)")
    p.add_argument("--per-board", type=int, default=8)
    p.add_argument("--seed", type=int, default=1, help="boards + instances")
    p.add_argument("--iter", type=int, default=1)
    p.add_argument("--expansions", type=int, default=600)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--stop-after", type=int, default=150,
                   help="stop when the best path has not improved for N expansions")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--backup", choices=["min", "mean"], default="min")
    p.add_argument("--root-noise", type=float, default=0.25)
    p.add_argument("--root-all", action="store_true",
                   help="score ALL legal moves at the root (not top-k) so the "
                        "depth-0 visit distribution covers the full action set")
    p.add_argument("--max-depth", type=int, default=64)
    p.add_argument("--sampler", choices=["mix", "random", "walk"], default="mix")
    p.add_argument("--rand-mix-prob", type=float, default=0.25)
    p.add_argument("--walk-k-max", type=int, default=16)
    p.add_argument("--walk-target-bias", type=float, default=0.6)
    p.add_argument("--d-star", choices=["off", "diag"], default="off",
                   help="diag: record the exact optimum per instance as a "
                        "DIAGNOSTIC (never a label); cheap at 16x16")
    p.add_argument("--oracle-expansions", type=int, default=60_000)
    p.add_argument("--timeout", type=int, default=300, help="per-instance wall cap (s)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    from spr import REPO, SV
    boards_dir = Path(a.boards_dir)
    if not boards_dir.is_absolute():
        boards_dir = REPO / boards_dir
    boards_dir.mkdir(parents=True, exist_ok=True)
    # BEFORE any forward-stack import: the board pool is read at import time
    os.environ["RR_ENV_DIR"] = str(boards_dir)

    from scaling.configs import get, parse_ids, env as cfg_env
    cfg = get(a.config)
    rr_env = {"RR_ENV_DIR": str(boards_dir)}
    if not cfg.legacy:                       # legacy g16r4 == RR_* unset
        rr_env = dict(cfg_env(cfg), RR_ENV_DIR=str(boards_dir))
        os.environ.update(rr_env)
    ids = parse_ids(a.board_ids)
    bad = [i for i in ids if i < 5000]
    if bad:
        raise SystemExit(f"board ids must be >= 5000 (pinned pool / bench boards "
                         f"live below): {bad[:5]}")
    ckpt = a.ckpt if Path(a.ckpt).is_absolute() else str(SV / a.ckpt)
    if not Path(ckpt).is_file():
        raise SystemExit(f"missing checkpoint {ckpt}")

    from nn_labeler import leanboard
    t0 = time.time()
    written = 0
    for i in ids:
        if not (boards_dir / f"env_{i}.pkl").exists():
            leanboard.write_board(boards_dir, i, cfg.grid, walls=cfg.walls,
                                  robots=cfg.robots, seed=a.seed)
            written += 1
    print(f"[fwd-selfplay] boards: {len(ids)} ids in {boards_dir} ({written} written, "
          f"{time.time() - t0:.0f}s)", flush=True)

    opts = dict(k=a.k, expansions=a.expansions, stop_after=a.stop_after,
                c_puct=a.c_puct, backup=a.backup, root_noise=a.root_noise,
                root_all=a.root_all, max_depth=a.max_depth, sampler=a.sampler,
                rand_mix_prob=a.rand_mix_prob, walk_k_max=a.walk_k_max,
                walk_target_bias=a.walk_target_bias, d_star=a.d_star,
                oracle_expansions=a.oracle_expansions, timeout=a.timeout,
                iter=a.iter, threads=a.threads, ckpt=ckpt, rr_env=rr_env)
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
                  initargs=(str(boards_dir), ckpt, a.device, opts)) as pool, \
            open(tmp, "w") as f:
        for bi, (recs, inst_stats) in enumerate(pool.imap_unordered(_work, tasks,
                                                                   chunksize=1)):
            for r in recs:
                f.write(json.dumps(r) + "\n")
            n_rec += len(recs)
            n_inst += len(inst_stats)
            n_solved += sum(1 for s in inst_stats if s.get("solved"))
            stats_all.extend(inst_stats)
            if (bi + 1) % 5 == 0 or bi + 1 == len(tasks):
                print(f"[fwd-selfplay] {bi + 1}/{len(tasks)} boards, {n_inst} instances, "
                      f"{n_solved} solved, {n_rec} records ({time.time() - t0:.0f}s)",
                      flush=True)
    os.replace(tmp, out)

    solved = [s for s in stats_all if s.get("solved")]
    reg = [s["regret"] for s in solved if s.get("regret") is not None]
    ctgs = []
    with open(out) as f:
        for line in f:
            ctgs.append(int(round(float(json.loads(line)["cost_to_go"]))))
    manifest = {
        "arm": "forward", "config": cfg.name, "grid": cfg.grid, "robots": cfg.robots,
        "iter": a.iter, "ckpt": ckpt, "boards_dir": str(boards_dir),
        "board_ids": a.board_ids, "per_board": a.per_board, "seed": a.seed,
        "search": {kk: opts[kk] for kk in ("k", "expansions", "stop_after", "c_puct",
                                           "backup", "root_noise", "root_all",
                                           "max_depth", "timeout")},
        "sampler": {kk: opts[kk] for kk in ("sampler", "rand_mix_prob", "walk_k_max",
                                            "walk_target_bias")},
        "d_star": a.d_star, "workers": a.workers, "device": a.device,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "seconds": round(time.time() - t0, 1),
        "instances": n_inst, "solved": len(solved), "records": n_rec,
        "solve_rate": (len(solved) / n_inst) if n_inst else None,
        "mean_moves": (sum(s["moves"] for s in solved) / len(solved)) if solved else None,
        "mean_expansions": (sum(s["expansions"] or 0 for s in stats_all) / n_inst)
                           if n_inst else None,
        "mean_seconds": (sum(s["seconds"] for s in stats_all) / n_inst) if n_inst else None,
        # PROBLEM.md 4.1 / DESIGN.md 5.2: the deep-state share is the falsification
        # signal for the curriculum; mean regret (diag only) is the fidelity gauge.
        "ctg_ge5_frac": round(sum(1 for c in ctgs if c >= 5) / max(1, len(ctgs)), 3),
        "mean_ctg": round(sum(ctgs) / max(1, len(ctgs)), 3),
        "diag_mean_regret": (sum(reg) / len(reg)) if reg else None,
        "diag_pct_optimal": (100.0 * sum(1 for r in reg if r == 0) / len(reg))
                            if reg else None,
        "status_counts": {kk: sum(1 for s in stats_all if s["status"].split(":")[0] == kk)
                          for kk in sorted({s["status"].split(":")[0] for s in stats_all})},
    }
    out.with_suffix(out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n")
    out.with_suffix(out.suffix + ".instances.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in stats_all))
    print(f"[fwd-selfplay] manifest: "
          f"{json.dumps({k2: v for k2, v in manifest.items() if k2 not in ('search', 'sampler')})}",
          flush=True)
    print(f"SPR FWD SELFPLAY DONE {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
