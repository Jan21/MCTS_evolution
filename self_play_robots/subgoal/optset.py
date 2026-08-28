"""Which subgoals keep an optimal solution alive? (input to Stage 2's beam test)

Stage 1 showed the state subgoal space CONTAINS the optimum on 40/40 probed
instances, but it found those optima by popping up to 60,685 nodes. Stage 3 gets
1200 expansions and a top-k=5 beam. The number that predicts Stage 3 is
therefore not the network's error but:

    at a state on an optimal path, is ANY subgoal that preserves optimality
    inside the top 5 of the 1024 candidates, under a given ranking?

This module computes the ground truth that question needs. For one instance it
runs the same A* as `subgoal/space.py::search` but

  * with `d*` known in advance (the bench instance carries it, and Stage 1
    proved the space attains it), so every child with f = g + c + h > d* is
    dropped unexpanded and unstored -- the search explores exactly the "optimal
    cone" and stays small enough to keep parent lists in memory;
  * keeping ALL tight parents of every state (`ng == best_g[nxt]`), not one;
  * running until the frontier's f exceeds d*, so every state on every optimal
    path is closed.

A reverse pass from the goal states with g = d* through the tight edges marks
the states that lie on some optimal path and, for each, the FULL set of
optimal-preserving next subgoals -- not just the one subgoal some particular
optimal plan happened to use.

Output (JSON): one row per instance with `d_star`, whether the search was
capped, a canonical optimal path of states, and for every state on that path the
complete optimal-preserving subgoal set as (robot, flat cell) pairs.

    PYTHONPATH=supervised_valuenet:self_play_robots python -m subgoal.optset \
        --n 20 --stride 23 --out results/subgoal/stage2_optsets.json
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPR = HERE.parent
REPO = SPR.parent
SV = REPO / "supervised_valuenet"
for _p in (str(SV), str(SPR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from subgoal.space import board, rest_cells, relaxed_h           # noqa: E402
from subgoal.table import load_bench, SIZE                       # noqa: E402

BIG = 10 ** 9


def optimal_cone(inst, max_pops=600_000, time_cap=300.0):
    """A* over the state subgoal space restricted to f <= d*, with all tight
    parents kept. Returns the row described in the module docstring."""
    wr, wd = board(inst["env_id"])
    start = tuple(tuple(p) for p in inst["positions"])
    tidx, goal = inst["target_idx"], tuple(inst["target"])
    dstar = int(inst["d_star"])
    H = relaxed_h(goal, wr, wd)

    def h(state):
        return H.get(state[tidx], BIG)

    t0 = time.time()
    best_g = {start: 0}
    parents: dict = {}
    heap = [(h(start), 0, 0, start)]
    tie, pops, expansions = 1, 0, 0
    goals = set()
    below = False          # a goal cheaper than d* would contradict the bench
    capped = False
    rc_cache: dict = {}
    while heap:
        f, g, _, st = heapq.heappop(heap)
        if f > dstar:
            break
        pops += 1
        if g > best_g.get(st, BIG):
            continue
        if st[tidx] == goal:
            if g == dstar:
                goals.add(st)
            elif g < dstar:
                below = True
            continue                      # goal states are terminal
        if pops > max_pops or (time.time() - t0) > time_cap:
            capped = True
            break
        expansions += 1
        for i in range(len(st)):
            blockers = frozenset(c for j, c in enumerate(st) if j != i)
            key = (st[i], blockers)
            cells = rc_cache.get(key)
            if cells is None:
                cells = rc_cache[key] = rest_cells(st[i], blockers, wr, wd)
            for cell, (cost, _d, _p) in cells.items():
                nxt = st[:i] + (cell,) + st[i + 1:]
                ng = g + cost
                if ng + h(nxt) > dstar:
                    continue              # cannot lie on an optimal path
                old = best_g.get(nxt, BIG)
                if ng < old:
                    best_g[nxt] = ng
                    parents[nxt] = [(st, i, cell)]
                    heapq.heappush(heap, (ng + h(nxt), ng, tie, nxt))
                    tie += 1
                elif ng == old:
                    parents.setdefault(nxt, []).append((st, i, cell))

    row = dict(idx=inst.get("idx"), env_id=inst["env_id"], d_star=dstar,
               target_idx=tidx, target=list(goal), start=[list(c) for c in start],
               pops=pops, expansions=expansions, states_stored=len(best_g),
               seconds=round(time.time() - t0, 2), capped=capped,
               n_goal_states=len(goals), goal_below_dstar=below)
    if capped or not goals or below:
        row.update(ok=False, path=None,
                   reason="capped" if capped else
                          ("a goal state was reached in fewer than d* moves"
                           if below else "no goal state at d*"))
        return row

    # reverse pass: mark every state on some optimal path, and its optimal set
    opt_next: dict = {}
    opt_succ: dict = {}
    seen = set(goals)
    stack = list(goals)
    while stack:
        v = stack.pop()
        for (u, i, cell) in parents.get(v, []):
            opt_next.setdefault(u, set()).add((i, cell[1] * SIZE + cell[0]))
            opt_succ.setdefault(u, []).append((i, cell, v))
            if u not in seen:
                seen.add(u)
                stack.append(u)
    if start not in opt_next:
        row.update(ok=False, path=None, reason="start not on any optimal path")
        return row

    # the canonical path is the optimal path that uses the FEWEST subgoals --
    # Stage 1's optimal plans used 1-5 (mean ~2.2), and a tie-break that
    # prefers many cheap macros would silently make the beam test easier by
    # turning one hard decision into several trivial ones.
    nsub = {g: 0 for g in goals}
    frontier = list(goals)
    while frontier:
        nxt_frontier = []
        for v in frontier:
            for (u, i, cell) in parents.get(v, []):
                if u in seen and nsub.get(u, BIG) > nsub[v] + 1:
                    nsub[u] = nsub[v] + 1
                    nxt_frontier.append(u)
        frontier = nxt_frontier
    path, cur = [], start
    while cur[tidx] != goal:
        opts = sorted(opt_next[cur])
        path.append(dict(positions=[list(c) for c in cur], g=best_g[cur],
                         optimal_next=[[int(r), int(c)] for r, c in opts]))
        step = min((s for s in opt_succ[cur] if nsub.get(s[2], BIG) == nsub[cur] - 1),
                   key=lambda s: (s[0], s[1][1] * SIZE + s[1][0]))
        cur = cur[:step[0]] + (step[1],) + cur[step[0] + 1:]
    row.update(ok=True, path=path, certified_moves=best_g[cur],
               n_decisions=len(path),
               mean_optimal_set=sum(len(p["optimal_next"]) for p in path) / len(path))
    return row


def _one(args):
    inst, max_pops, time_cap = args
    try:
        return optimal_cone(inst, max_pops, time_cap)
    except Exception as exc:                                   # noqa: BLE001
        return dict(idx=inst.get("idx"), env_id=inst["env_id"], ok=False,
                    reason=f"{type(exc).__name__}: {exc}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-pops", type=int, default=600_000)
    p.add_argument("--time-cap", type=float, default=300.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    full = load_bench()
    for i, inst in enumerate(full):
        inst["idx"] = i
    # the SAME two deterministic samples Stage 1 used: the first N and every
    # STRIDE-th, so the beam number is measured on the instances whose optima
    # Stage 1 already certified.
    picks, seen = [], set()
    for src in (full[:a.n], full[::a.stride][:a.n]):
        for inst in src:
            if inst["idx"] not in seen:
                seen.add(inst["idx"])
                picks.append(inst)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(_one, [(i, a.max_pops, a.time_cap) for i in picks]))
    rows.sort(key=lambda r: r["idx"])
    ok = [r for r in rows if r.get("ok")]
    for r in rows:
        print(f"[optset] idx {r['idx']:3d} env {r['env_id']} d*={r.get('d_star')} "
              f"ok={r.get('ok')} decisions={r.get('n_decisions')} "
              f"|opt|={r.get('mean_optimal_set')} stored={r.get('states_stored')} "
              f"[{r.get('seconds')}s]", flush=True)
    summary = dict(n=len(rows), ok=len(ok),
                   mean_decisions=sum(r["n_decisions"] for r in ok) / max(1, len(ok)),
                   mean_optimal_set=sum(r["mean_optimal_set"] for r in ok) / max(1, len(ok)),
                   d_star_matched=sum(1 for r in ok if r["certified_moves"] == r["d_star"]),
                   wall_seconds=round(time.time() - t0, 1))
    print("[optset] summary:", json.dumps(summary))
    out = Path(a.out)
    if not out.is_absolute():
        out = SPR / out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(
        caps=dict(max_pops=a.max_pops, time_cap=a.time_cap), n=a.n,
        stride=a.stride, summary=summary, rows=rows,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        date=time.strftime("%Y-%m-%dT%H:%M:%S")), indent=1) + "\n")
    os.replace(tmp, out)
    print(f"SPR SUBGOAL STAGE2 OPTSET DONE {out}")


if __name__ == "__main__":
    main()
