"""Stage 3a of PLAN_HYBRID_ACTION_SPACE.md: the mixed action space, portfolio form.

Stage 1 measured the CEILING of "a macro-move head start, then the old
vocabulary finishes": +0.108 at 16x16 and +0.172 at 24x24, against +0.838 and
+1.276 for the old vocabulary alone. That ceiling was computed with a PERFECT
chooser, trying every prefix with a generous exhaustive search from each. This
module asks the question that matters: how much of it survives when a real
planner has to FIND the prefix inside a fixed budget?

The search, per instance:

  1. finish from the start        run the old vocabulary's search from the
                                  untouched position under `b0` iterations;
                                  whatever it certifies is the incumbent;
  2. enumerate macro moves        for each robot, the exact joint-state slide
                                  BFS gives every cell it can come to rest on
                                  and the true slide count (physics, never a
                                  network) -- ~150 candidates at 16x16;
  3. rank them                    by `prefix_cost + h(child)`, where h is the
                                  old planner's own abstract plan cost
                                  (`--h struct`, Stage 2's arm, no network) or
                                  the any-stop relaxation (`--h relaxed`, the
                                  control);
  4. finish from the best M       run the same old-vocabulary search from each,
                                  under `b_sub` iterations, and score
                                  `prefix_cost + realized strict moves`;
  5. keep the shortest total.     Ties go to the incumbent, so a prefix wins
                                  only by being strictly shorter.

Every candidate answer is replayed under the real joint-game rules before it
counts (`spr.ceiling.probe` realizes with `eval.realize.strict_moves`), and the
budget is charged in the old planner's own iteration unit so the row is
comparable with its recorded 22.0 mean.

This arm is deliberately NETWORK-FREE end to end: physics enumerates, the
abstract plan cost ranks, the old vocabulary finishes. If it beats the old
planner's recorded 53.3% optimal, it does so with no learned component at all.

    PYTHONPATH=supervised_valuenet:self_play_robots python -m subgoal.hybrid \
        --config g16r4 --instances eval/data/bench450.jsonl \
        --h struct --b0 400 --b-sub 200 --top-m 8 --out results/subgoal/hybrid/x.json
"""
from __future__ import annotations

import argparse, json, multiprocessing as mp, os, sys, time
from pathlib import Path
from spr import RESULTS, SV

_G = {}


def _init(config, vocab, b0, b_sub, top_m, hmode, frontier):
    sys.path.insert(0, str(SV))
    from GridEnv import GridEnv
    from skeleton.astar import AStar
    from skeleton import heuristics
    _G.update(GridEnv=GridEnv, vocab=vocab, b0=b0, b_sub=b_sub, top_m=top_m,
              hmode=hmode, envs={}, frontier=frontier,
              solver=AStar(propose=heuristics.propose_b1 if vocab in ("b1", "b2")
                           else heuristics.propose,
                           max_iters=max(b0, b_sub), max_frontier=frontier,
                           beam=None, by_reference=(vocab == "b2")))


def _state(inst, positions):
    from GridEnv import State, Robot_at
    from move_planner.state import COLOR_ORDER
    t = inst["target_idx"]
    return State(target=tuple(inst["target"]),
                 target_robot=Robot_at(position=positions[t], color=COLOR_ORDER[t]),
                 helpers=[Robot_at(position=positions[j], color=COLOR_ORDER[j])
                          for j in range(len(positions)) if j != t])


def _finish(env, st, iters):
    """Old vocabulary, bounded. Returns (strict_moves or None, iterations spent)."""
    from spr.ceiling import probe
    r = probe(env, st, _G["solver"], _G["vocab"], iters, _G["frontier"],
              time_cap=1e9, park_cap=2)
    return r["best_realizable_moves"], r["iters"]


def _work(item):
    idx, inst = item
    from simulate import wall_sets, _board_size
    from subgoal.space import rest_cells, relaxed_h
    from skeleton.astar import _initial_plan
    from spr.search import forced_fixes

    env = _G["envs"].get(inst["env_id"])
    if env is None:
        env, _ = _G["GridEnv"].from_env(inst["env_id"])
        _G["envs"].clear(); _G["envs"][inst["env_id"]] = env
    size = _board_size(env.grid_data, None)
    wr, wd = wall_sets(env.grid_data, size)
    pos = tuple(tuple(p) for p in inst["positions"])
    tidx, goal = inst["target_idx"], tuple(inst["target"])
    t0 = time.time()

    # 1. finish from the start
    spent = 0
    m0, it = _finish(env, _state(inst, pos), _G["b0"]); spent += it
    best = (m0, 0, None, None) if m0 is not None else None

    # 2. enumerate macro moves, 3. rank them
    H = relaxed_h(goal, wr, wd, size) if _G["hmode"] == "relaxed" else None
    cands = []
    for r in range(len(pos)):
        blockers = frozenset(pos[j] for j in range(len(pos)) if j != r)
        for cell, (cost, _d, _p) in rest_cells(pos[r], blockers, wr, wd, size).items():
            child = pos[:r] + (cell,) + pos[r + 1:]
            st2 = _state(inst, child)
            if _G["hmode"] == "struct":
                try:
                    h = float(forced_fixes(env, st2, _G["solver"], _initial_plan(env, st2)).cost())
                except Exception:
                    continue
            else:
                h = float(H.get(child[tidx], 1e6))
            cands.append((cost + h, cost, r, cell, st2))
    cands.sort(key=lambda c: (c[0], c[1]))

    # 4. finish from the best M
    tried = 0
    for rank, (_f, cost, r, cell, st2) in enumerate(cands[:_G["top_m"]]):
        if best is not None and cost >= best[0]:
            continue                       # the prefix alone already loses
        m, it = _finish(env, st2, _G["b_sub"]); spent += it; tried += 1
        if m is None:
            continue
        total = cost + m
        if best is None or total < best[0]:
            best = (total, cost, r, list(cell))

    d = inst.get("d_star")
    return dict(idx=idx, env_id=inst["env_id"], d_star=d,
                moves=(best[0] if best else None),
                solved=best is not None,
                extra=((best[0] - d) if (best and d is not None) else None),
                prefix_cost=(best[1] if best else None),
                prefix_robot=(best[2] if best else None),
                prefix_cell=(best[3] if best else None),
                baseline_moves=m0,
                iters=spent, subsearches=tried, candidates=len(cands),
                seconds=round(time.time() - t0, 2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--instances", required=True)
    p.add_argument("--vocab", choices=["base", "b1", "b2"], default="b2")
    p.add_argument("--h", choices=["struct", "relaxed"], default="struct")
    p.add_argument("--b0", type=int, default=400, help="iterations for the start-position finish")
    p.add_argument("--b-sub", type=int, default=200, help="iterations per prefix finish")
    p.add_argument("--top-m", type=int, default=8, help="how many prefixes get a finish")
    p.add_argument("--frontier", type=int, default=100_000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    sys.path.insert(0, str(SV))
    from scaling.configs import get, env as cfg_env
    cfg = get(a.config)
    for k in ("RR_GRID", "RR_ROBOTS", "RR_WALLS", "RR_ENV_DIR"):
        os.environ.pop(k, None)
    if not cfg.legacy:
        os.environ.update(cfg_env(cfg))

    ip = Path(a.instances)
    if not ip.is_absolute():
        ip = SV / ip
    insts = [json.loads(l) for l in ip.read_text().splitlines() if l.strip()]
    if a.limit:
        insts = insts[:a.limit]
    items = sorted(enumerate(insts), key=lambda t: t[1]["env_id"])
    print(f"[hybrid] {a.config} vocab={a.vocab} h={a.h} b0={a.b0} b_sub={a.b_sub} "
          f"top_m={a.top_m} n={len(items)} workers={a.workers}", flush=True)

    t0 = time.time(); rows = []
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers, initializer=_init,
                  initargs=(a.config, a.vocab, a.b0, a.b_sub, a.top_m, a.h, a.frontier)) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, items, chunksize=2)):
            rows.append(r)
            if (i + 1) % 50 == 0 or i + 1 == len(items):
                ok = sum(1 for x in rows if x["extra"] == 0)
                print(f"[hybrid] {i+1}/{len(items)} optimal={ok} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: r["idx"])

    graded = [r for r in rows if r["d_star"] is not None]
    solved = [r for r in graded if r["solved"]]
    base = [r for r in graded if r["baseline_moves"] is not None]
    helped = [r for r in solved if (r["prefix_cost"] or 0) > 0]
    summary = dict(
        n=len(rows), solved=len(solved),
        optimal=sum(1 for r in solved if r["extra"] == 0),
        pct_optimal_of_all=100.0 * sum(1 for r in solved if r["extra"] == 0) / max(1, len(graded)),
        mean_extra=(sum(r["extra"] for r in solved) / len(solved)) if solved else None,
        baseline_solved=len(base),
        baseline_optimal=sum(1 for r in base if r["baseline_moves"] == r["d_star"]),
        baseline_mean_extra=(sum(r["baseline_moves"] - r["d_star"] for r in base) / len(base)) if base else None,
        n_helped_by_prefix=len(helped),
        mean_iters=sum(r["iters"] for r in rows) / len(rows),
        median_iters=sorted(r["iters"] for r in rows)[len(rows) // 2],
        mean_subsearches=sum(r["subsearches"] for r in rows) / len(rows),
        mean_candidates=sum(r["candidates"] for r in rows) / len(rows),
        mean_seconds=sum(r["seconds"] for r in rows) / len(rows),
        wall_seconds=round(time.time() - t0, 1))
    print("\n[hybrid] summary:", json.dumps(summary, indent=1), flush=True)

    out = Path(a.out)
    if not out.is_absolute():
        out = (RESULTS.parent / out) if str(out).startswith("results/") else (RESULTS / out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        config=a.config, vocab=a.vocab, h=a.h, b0=a.b0, b_sub=a.b_sub,
        top_m=a.top_m, instances=str(ip), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        date=time.strftime("%Y-%m-%dT%H:%M:%S"), summary=summary, rows=rows), indent=1) + "\n")
    print(f"SPR HYBRID DONE {out}")


if __name__ == "__main__":
    main()
