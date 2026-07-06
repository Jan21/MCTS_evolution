"""Solve ONE Ricochet-Robots puzzle end-to-end with the two trained nets + A*.

The full learned pipeline on a single instance:
  PROPOSE (policy net, NN#1) ranks the valid subgoal candidates for the open segment,
  keep top-k; SCORE (value net, NN#2) predicts each candidate's cost-to-go; A* uses
  g + value as f and expands best-first until the first complete plan pops. Prints the
  solved subgoal DAG and its move cost. With --compare-optimal, also runs the exact
  commit-and-solve A* and reports regret (learned cost - optimal cost).

Board source:
  --board N        solve an instance on an existing environments/env_N.pkl
  --new            generate a brand-new random board (walls + slide graph) first
Instance: a random robot/target placement (fix with --seed) unless the board's own
stored instance is used via --stored.

    # existing board, need both checkpoints:
    python solve.py --board 112 --value <val.ckpt> --policy <pol.ckpt> --compare-optimal
    # brand-new geometry:
    python solve.py --new --seed 7 --value <val.ckpt> --policy <pol.ckpt> --compare-optimal
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import pickle
import random
from pathlib import Path

import torch

from GridEnv import GridEnv
from skeleton.astar import AStar, _initial_plan, _segment, _apply
from nn.generate import random_instance, _context, _fixed_g
from nn.collect_search import _record
from nn.gen_grids import make_board
from train.policy_common import _ix
from train.policy_tf import PolicyTF
from train.looped_pc import LoopedValueNet
from eval.end2end import _policy_logp, _value_cost, _hidx

torch.set_grad_enabled(False)


def nn_astar_plan(env, state, solver, policy, value, env_id, dev, k=5, max_iters=3000):
    """Same search as eval.end2end.nn_astar but returns the completed PartialPlan
    (not just its cost) so the solution can be printed. None if unsolved in budget."""
    cnt = itertools.count()
    frontier = [(0.0, next(cnt), _initial_plan(env, state))]
    iters = 0
    while frontier and iters < max_iters:
        iters += 1
        _, _, plan = heapq.heappop(frontier)
        while not plan.is_complete():                     # apply forced exact fixes
            parent, child = plan.open_edges()[0]
            seg = _segment(plan, state, parent, child)
            if env.compute_exact_shortest_path_length(seg.start, seg.end, seg.fix_support) is not None:
                plan = solver._expand(env, state, plan)[0]
            else:
                break
        if plan.is_complete():
            return plan, iters
        parent, child = plan.open_edges()[0]
        seg = _segment(plan, state, parent, child)
        cands = solver.propose(env, seg.end, seg.mover, seg.helpers, seg.support)
        ctx = _context(plan)
        recs, cps = [], []
        for cand in cands:
            cp = _apply(env, plan, parent, child, seg, cand)
            if cp is None or _hidx(state, _record(env_id, state, seg, ctx, cand, 0, 0, 0)["cand_helper"][0]) is None:
                continue
            recs.append(_record(env_id, state, seg, ctx, cand, 0, 0, 0))
            cps.append(cp)
        if not recs:
            continue
        logp, _ = _policy_logp(policy, recs, dev)
        if logp is None:
            continue
        keys = [(_ix(r["cand_bottleneck"]), _ix(r["cand_support"]), _hidx(state, r["cand_helper"][0])) for r in recs]
        order = sorted(range(len(recs)), key=lambda i: -logp.get(keys[i], -1e9))[:k]
        costs = _value_cost(value, [recs[i] for i in order], dev)
        for i, h in zip(order, costs):
            heapq.heappush(frontier, (float(_fixed_g(cps[i])) + h, next(cnt), cps[i]))
    return None, iters


def describe(plan):
    """Human-readable subgoal DAG: bottlenecks (robot -> cell) + supports (helper -> cell)."""
    lines = []
    for nid, d in plan.g.nodes(data=True):
        if d.get("ntype") == "bottleneck":
            lines.append(f"  bottleneck: robot {d['robot']} -> cell {tuple(d['pos'])}")
        elif d.get("ntype") == "support":
            lines.append(f"  support:    helper {d['robot']} -> cell {tuple(d['pos'])}")
    return "\n".join(lines) if lines else "  (direct: no subgoal needed)"


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--board", type=int, help="existing environments/env_N.pkl id")
    src.add_argument("--new", action="store_true", help="generate a brand-new random board")
    p.add_argument("--value", required=True, help="value-net checkpoint (looped_pc)")
    p.add_argument("--policy", required=True, help="proposal-net checkpoint (policy_tf)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=5, help="proposal top-k kept per decision")
    p.add_argument("--iters", type=int, default=3000, help="A* pop budget")
    p.add_argument("--compare-optimal", action="store_true")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PolicyTF.load_from_checkpoint(a.policy, map_location=dev).to(dev).eval()
    value = LoopedValueNet.load_from_checkpoint(a.value, map_location=dev).to(dev).eval()
    solver = AStar(max_iters=4000, max_frontier=40_000)
    rng = random.Random(a.seed)

    if a.new:
        env_id = 990000 + (a.seed % 10000)
        board = make_board(env_id, rng)
        Path("environments").mkdir(exist_ok=True)
        with open(f"environments/env_{env_id}.pkl", "wb") as f:
            pickle.dump(board, f)
        print(f"generated new board -> environments/env_{env_id}.pkl")
    else:
        env_id = a.board

    env, s0 = GridEnv.from_env(env_id)
    colors = [s0.target_robot.color] + [h.color for h in s0.helpers]
    state = random_instance(env, colors, rng)
    print(f"board {env_id}: target {colors[0]} -> goal {tuple(state.target_robot.position)}; "
          f"{len(state.helpers)} helpers")

    plan, iters = nn_astar_plan(env, state, solver, policy, value, env_id, dev, a.k, a.iters)
    if plan is None:
        print(f"NN A*: no plan within {a.iters} pops")
        return
    cost = int(plan.cost())
    print(f"\nNN A* solved in {iters} pops. plan cost = {cost} moves")
    print(describe(plan))

    if a.compare_optimal:
        opt = solver.solve_plan(env, state, _initial_plan(env, state))
        if opt is None:
            print("\noptimal: solver found none (reference unavailable)")
        else:
            oc = int(opt.cost())
            print(f"\noptimal cost = {oc}  |  regret = {cost - oc}")


if __name__ == "__main__":
    main()
