"""Collect (segment, candidate) decisions from the SEARCH distribution.

Unlike nn.generate (which rolls out the *optimal* trajectory), this follows the
*heuristic-guided* search with epsilon-greedy exploration, so it visits the
off-optimal partial plans an A* frontier actually encounters. Every candidate at
every visited segment is commit-and-solved for its exact cost-to-go (same labels
as nn.generate). Many epsilon rollouts per instance cover a tree of partial plans.

This is off-policy DAgger with the heuristic as the rollout policy: the records
cover the states the search sees, not just the optimal path.

    python -m nn.collect_search --graphs 0-95,1000-1799 --per-graph 6 \
        --rollouts 4 --epsilon 0.3 --out nn/data/search.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from GridEnv import GridEnv, State
from skeleton.astar import AStar, _initial_plan, _segment, _apply
from nn.generate import random_instance, _xy, _fixed_g, _context, parse_graphs


def _record(env_id, state, seg, ctx, cand, ctg, best_ctg, depth):
    bn_ctx, sp_ctx, open_eps = ctx
    return {
        "env_id": env_id,
        "target": _xy(state.target),
        "target_robot": [_xy(state.target_robot.position), state.target_robot.color],
        "helpers": [[_xy(h.position), h.color] for h in state.helpers],
        "seg_start": _xy(seg.start), "seg_end": _xy(seg.end),
        "seg_support": _xy(seg.fix_support), "mover_color": seg.mover.color,
        "ctx_bottlenecks": bn_ctx, "ctx_supports": sp_ctx, "ctx_open_endpoints": open_eps,
        "cand_bottleneck": _xy(cand.subgoal.bottleneck.position),
        "cand_support": _xy(cand.subgoal.support.position),
        "cand_helper": [_xy(cand.subgoal.helper.position), cand.subgoal.helper.color],
        "cand_parent_support": _xy(cand.parent_support),
        "cost_to_go": ctg, "is_optimal": ctg == best_ctg, "depth": depth,
    }


def rollout(env, state, solver, env_id, epsilon, rng, max_candidates=None, max_steps=40):
    """One epsilon-greedy heuristic rollout; record every candidate at each segment."""
    records = []
    plan = _initial_plan(env, state)
    depth = steps = 0
    while not plan.is_complete() and steps < max_steps:
        steps += 1
        parent, child = plan.open_edges()[0]
        seg = _segment(plan, state, parent, child)
        if env.compute_exact_shortest_path_length(seg.start, seg.end, seg.fix_support) is not None:
            plan = solver._expand(env, state, plan)[0]      # forced exact fix, no choice
            continue
        cands = solver.propose(env, seg.end, seg.mover, seg.helpers, seg.support)
        if max_candidates is not None:
            cands = sorted(cands, key=lambda c: solver.score(env, c))[:max_candidates]
        ctx = _context(plan)
        fixed_g = _fixed_g(plan)
        labeled = []
        for cand in cands:
            cp = _apply(env, plan, parent, child, seg, cand)
            if cp is None:
                continue
            done = solver.solve_plan(env, state, cp)
            if done is None:
                continue
            labeled.append((cand, cp, int(done.cost()) - int(fixed_g)))
        if not labeled:
            break
        best = min(c for _, _, c in labeled)
        for cand, _, ctg in labeled:
            records.append(_record(env_id, state, seg, ctx, cand, ctg, best, depth))
        # advance along the HEURISTIC's choice (epsilon-random) -> off-optimal states
        if rng.random() < epsilon:
            chosen = rng.choice(labeled)
        else:
            chosen = min(labeled, key=lambda lc: solver.score(env, lc[0]))
        plan = chosen[1]
        depth += 1
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graphs", default="0-95")
    p.add_argument("--per-graph", type=int, default=6)
    p.add_argument("--rollouts", type=int, default=4)
    p.add_argument("--epsilon", type=float, default=0.3)
    p.add_argument("--max-candidates", type=int, default=14)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--max-iters", type=int, default=1000)
    p.add_argument("--max-frontier", type=int, default=8000)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--out", default="nn/data/search.jsonl")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rng = random.Random(a.seed)
    # mi=1000 gives labels identical to mi=4000 (solve_plan returns optimal or None,
    # never suboptimal); only rare hard completions drop -> ~40x faster.
    solver = AStar(max_iters=a.max_iters, max_frontier=a.max_frontier)
    graphs = parse_graphs(a.graphs)
    if a.nshards > 1:                                  # strided slice balances hard boards
        graphs = graphs[a.shard::a.nshards]
    _, s0 = GridEnv.from_env(graphs[0])
    colors = [s0.target_robot.color] + [h.color for h in s0.helpers]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rec = 0
    t0 = time.time()
    with open(out, "w") as f:
        for gid in graphs:
            env, _ = GridEnv.from_env(gid)
            for _ in range(a.per_graph):
                st = random_instance(env, colors, rng)
                for _ in range(a.rollouts):
                    try:
                        recs = rollout(env, st, solver, gid, a.epsilon, rng,
                                       a.max_candidates, a.max_steps)
                    except Exception:
                        recs = []
                    for r in recs:
                        f.write(json.dumps(r) + "\n")
                    n_rec += len(recs)
            print(f"graph {gid}: {n_rec} records ({time.time() - t0:.0f}s)", flush=True)
    print(f"\ndone: {n_rec} records -> {out}")


if __name__ == "__main__":
    main()
