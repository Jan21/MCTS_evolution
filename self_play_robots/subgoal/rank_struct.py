"""Stage 2 of PLAN_HYBRID_ACTION_SPACE.md, cheapest form: does the OLD
planner's abstract plan cost rank macro-move children well?

FINDINGS 32(c) already ranked three scorers on the same held-out corpus (289
decisions, boards 1800-1849). This adds a fourth arm, `struct`, computed exactly
as `variants/v07_hybrid_actions.py::plan_cost_of` computes it:

    h_struct(child) = forced_fixes(env, child, solver, _initial_plan(env, child)).cost()

no tree, no network, the old vocabulary's own admissible-ish estimate of what
remains. If it does not rank better than the any-stop relaxation, the old
vocabulary's value is in its ACTION SPACE and not in its cost estimates, and
family A of the plan dies here for a few CPU-minutes instead of a planner run.

    PYTHONPATH=supervised_valuenet:self_play_robots python -m subgoal.rank_struct \
        --data self_play_robots/results/subgoal/stage3/val_b1800-1849.npz \
        --limit-groups 20 --out /dev/null        # timing mode
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--vocab", choices=["base", "b1", "b2"], default="b2")
    p.add_argument("--limit-groups", type=int, default=None,
                   help="time/smoke on the first N (state,robot) groups")
    p.add_argument("--max-iters", type=int, default=20000)
    p.add_argument("--max-frontier", type=int, default=50000)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    from subgoal.table import ENV_DIR, SIZE, COLOR_ORDER
    from subgoal.ctgnet import CtgDataset, group_metrics
    from subgoal.space import board as _board, relaxed_h
    from GridEnv import State, Robot_at
    from skeleton.astar import AStar, _initial_plan
    from skeleton import heuristics
    from spr.search import forced_fixes
    import GridEnv as _GE

    solver = AStar(propose=heuristics.propose_b1 if a.vocab in ("b1", "b2") else heuristics.propose,
                   max_iters=a.max_iters, max_frontier=a.max_frontier,
                   beam=None, by_reference=(a.vocab == "b2"))

    store = np.load(a.data, allow_pickle=False)
    ds = CtgDataset(store, ENV_DIR, SIZE)
    n = SIZE
    envs, Hcache, pcache = {}, {}, {}
    hits = misses = 0

    def env_of(eid):
        e = envs.get(eid)
        if e is None:
            e, _ = _GE.GridEnv.from_env(eid)
            envs.clear(); envs[eid] = e
        return e

    def h_struct(eid, positions, tidx, target):
        """abstract plan cost of the child state; cached on the full position tuple"""
        nonlocal hits, misses
        key = (eid, positions, tidx, target)
        v = pcache.get(key)
        if v is not None:
            hits += 1
            return v
        misses += 1
        env = env_of(eid)
        st = State(target=target,
                   target_robot=Robot_at(position=positions[tidx], color=COLOR_ORDER[tidx]),
                   helpers=[Robot_at(position=positions[j], color=COLOR_ORDER[j])
                            for j in range(len(positions)) if j != tidx])
        try:
            v = float(forced_fixes(env, st, solver, _initial_plan(env, st)).cost())
        except Exception:
            v = 1e6
        pcache[key] = v
        return v

    index = ds.index if a.limit_groups is None else ds.index[:a.limit_groups]
    ARMS = ("struct", "relaxed", "combo")
    out, pool = {k: [] for k in ARMS}, {k: {} for k in ARMS}
    t0 = time.time(); ncalls = 0
    for gi, (si, r) in enumerate(index):
        eid = int(ds.env_id[si]); tidx = int(ds.tidx[si])
        tgt = (int(ds.target[si][0]), int(ds.target[si][1]))
        H = Hcache.get((eid, tgt))
        if H is None:
            wr, wd = _board(eid)
            Hd = relaxed_h(tgt, wr, wd, n)
            H = Hcache[(eid, tgt)] = [Hd.get((c % n, c // n), 1e6) for c in range(n * n)]
        pos = tuple((int(ds.pos[si][j][0]), int(ds.pos[si][j][1])) for j in range(ds.pos.shape[1]))
        y = ds.ctg[si, r].astype(np.float64)
        c = ds.cost[si, r].astype(np.float64)
        m = ds.ctg[si, r] >= 0
        cells = np.flatnonzero(m)
        hs = np.empty(len(cells), np.float64)
        for k, cc in enumerate(cells):
            cell = (int(cc) % n, int(cc) // n)
            child = pos[:r] + (cell,) + pos[r + 1:]
            hs[k] = h_struct(eid, child, tidx, tgt); ncalls += 1
        if r == tidx:
            hrel = np.array([H[int(cc)] for cc in cells], np.float64)
        else:
            hrel = np.full(len(cells), float(H[pos[tidx][1] * n + pos[tidx][0]]))
        # combo: both are optimistic estimates of the same quantity, so the
        # larger is the tighter one and still under-estimates if both do
        for key, hh in (("struct", hs), ("relaxed", hrel),
                        ("combo", np.maximum(hs, hrel))):
            out[key].append(group_metrics(hh, y[m], c[m]))
            pool[key].setdefault(si, []).append((hh, y[m], c[m]))
        if (gi + 1) % 100 == 0:
            print(f"[rank-struct] {gi+1}/{len(index)} groups, {ncalls} children, "
                  f"{time.time()-t0:.0f}s, cache {hits}/{hits+misses}", flush=True)

    el = time.time() - t0
    print(f"\n[rank-struct] {len(index)} groups, {ncalls} children scored, {el:.1f}s")
    print(f"[rank-struct] {1000*el/max(1,ncalls):.2f} ms per child, "
          f"cache hit rate {100*hits/max(1,hits+misses):.1f}%")
    print(f"[rank-struct] projected for all {len(ds.index)} groups: "
          f"{el*len(ds.index)/len(index)/60:.1f} min single-core")

    summary = {}
    for k, v in out.items():
        summary[k] = {mm + "_group": float(np.nanmean([d[mm] for d in v]))
                      for mm in ("mae", "spearman", "top1", "top5")}
        summary[k]["groups"] = len(v)
        dec = [group_metrics(np.concatenate([x[0] for x in parts]),
                             np.concatenate([x[1] for x in parts]),
                             np.concatenate([x[2] for x in parts]))
               for parts in pool[k].values()]
        summary[k]["decisions"] = len(dec)
        summary[k]["top1"] = float(np.mean([d["top1"] for d in dec]))
        summary[k]["top5"] = float(np.mean([d["top5"] for d in dec]))
        summary[k]["mae"] = summary[k]["mae_group"]
        summary[k]["spearman"] = summary[k]["spearman_group"]
    print("\n| scorer | rho vs true cost-to-go | top-1 of f=c+h | top-5 | MAE |")
    print("|---|---|---|---|---|")
    for k, lbl in (("struct", "old planner abstract cost (NEW)"),
                   ("combo", "max(struct, relaxation) (NEW)"),
                   ("relaxed", "any-stop relaxation (recomputed control)")):
        d = summary[k]
        print(f"| {lbl} | {d['spearman']:.3f} | {d['top1']*100:.1f}% | "
              f"{d['top5']*100:.1f}% | {d['mae']:.2f} |")
    print(f"\n(recorded reference, FINDINGS 32(c), same corpus: learned CtgNet "
          f"0.378 / 69.9% / 93.8% / 2.01; relaxation 0.518 / 47.8% / 83.4% / 5.44)")

    if a.out and a.out != "/dev/null":
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            dict(data=a.data, vocab=a.vocab, groups=len(index), children=ncalls,
                 seconds=round(el, 1), summary=summary,
                 date=time.strftime("%Y-%m-%dT%H:%M:%S")), indent=1) + "\n")
        print(f"SPR RANK STRUCT DONE {a.out}")


if __name__ == "__main__":
    main()
