"""Extract real PartialPlan DAGs for the plan-structure visualization page.

Runs the hand-coded exhaustive search (no neural nets) on a handful of pinned
base-benchmark puzzles, under three vocabularies — original, B1 (transient
stoppers + step-aside "park" repairs) and B2 (supports-by-reference +
generalized parks) — and serializes each first strictly-playable plan's DAG to
JSON for eval/results/plan_structures_data.json. Everything is base-scale
(16x16, 4 robots), CPU, seconds.

Usage:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
  python3 analysis/plan_viz/extract_plans.py eval/results/plan_structures_data.json
"""
import heapq
import json
import os
import sys
import time

os.environ.setdefault("RR_ENV_DIR", "environments")
os.environ.setdefault("RR_GRID", "16")
os.environ.setdefault("RR_ROBOTS", "4")
sys.path.insert(0, ".")

from GridEnv import GridEnv, State, Robot_at            # noqa: E402
from skeleton.astar import AStar, _initial_plan, park_repairs  # noqa: E402
from skeleton import heuristics                          # noqa: E402
from move_planner.state import COLOR_ORDER               # noqa: E402
from eval.realize import strict_moves                    # noqa: E402
from simulate import wall_sets, _board_size              # noqa: E402


def find_playable_plan(env, state, b1=False, b2=False, max_iters=100_000,
                       time_cap=120.0, max_frontier=1_500_000):
    """First strictly-playable complete plan under the given vocabulary."""
    propose = heuristics.propose_b1 if (b1 or b2) else heuristics.propose
    solver = AStar(propose=propose, max_iters=max_iters,
                   max_frontier=max_frontier, by_reference=b2)
    size = _board_size(env.grid_data, None)
    wr, wd = wall_sets(env.grid_data, size)
    start = _initial_plan(env, state)
    frontier = [(start.cost(), 0, start)]
    tie = 1
    max_open = 2 * (len(state.helpers) + 2)
    t0 = time.time()
    it = 0
    while frontier:
        it += 1
        if (it > max_iters or len(frontier) > max_frontier
                or time.time() - t0 > time_cap):
            return None, None
        _, _, cur = heapq.heappop(frontier)
        if cur.is_complete():
            fi = {}
            m = strict_moves(env, state, cur, log=None, fail_info=fi)
            if m is not None:
                return cur, m
            if (b1 or b2) and fi:
                for child in park_repairs(env, state, cur, fi, wr, wd, size,
                                          max_parks=2 if b2 else 1,
                                          pairwise=b2, multi_slide=b2):
                    heapq.heappush(frontier, (child.cost(), tie, child))
                    tie += 1
            continue
        if len(cur.open_edges()) > max_open:
            continue
        for child in solver._expand(env, state, cur):
            heapq.heappush(frontier, (child.cost(), tie, child))
            tie += 1
    return None, None


def serialize(env, state, plan, moves, meta):
    g = plan.g
    nodes = []
    for n, d in g.nodes(data=True):
        pos = d.get("pos")
        robot = d.get("robot")
        node = {
            "id": n,
            "type": d.get("ntype"),
            "pos": None if pos is None else [int(pos[0]), int(pos[1])],
            "robot": getattr(robot, "color", None),
        }
        if d.get("ntype") == "support" and pos is not None:
            node["transient"] = not env._has_adjacent_wall(tuple(pos))
        if d.get("ntype") == "park":
            be = d.get("before_edge")
            node["before_edge"] = None if be is None else list(be)
        nodes.append(node)
    edges = [{
        "from": u, "to": v,
        "status": d.get("status"),
        "cost": d.get("cost"),
        "byref": bool(d.get("byref")),
    } for u, v, d in g.edges(data=True)]
    return {
        **meta,
        "plan_cost": float(plan.cost()),
        "legal_moves": moves,
        "robots": {r.color: [int(r.position[0]), int(r.position[1])]
                   for r in state.all_robots},
        "target_robot": state.target_robot.color,
        "target_cell": [int(state.target[0]), int(state.target[1])],
        "nodes": nodes,
        "edges": edges,
    }


def state_from_instance(inst):
    positions = [tuple(p) for p in inst["positions"]]
    tidx = inst["target_idx"]
    return State(
        target=tuple(inst["target"]),
        target_robot=Robot_at(position=positions[tidx],
                              color=COLOR_ORDER[tidx]),
        helpers=[Robot_at(position=positions[j], color=COLOR_ORDER[j])
                 for j in range(len(positions)) if j != tidx])


def main():
    out_path = sys.argv[1]
    probe_insts = {i["idx"]: i
                   for i in json.load(open(
                       "analysis/artifacts/ceiling_probe_instances.json"))}
    bench = [json.loads(l) for l in open("eval/data/bench450.jsonl")]

    examples = []

    # 1) Original vocabulary: first bench puzzle whose plan needs >= 1 helper
    #    (a subgoal + support chain) — the "before" picture.
    for row_idx, inst in enumerate(bench):
        inst = dict(inst, idx=row_idx)
        env, _ = GridEnv.from_env(inst["env_id"])
        st = state_from_instance(inst)
        plan, moves = find_playable_plan(env, st, time_cap=20)
        if plan is None:
            continue
        n_sub = sum(1 for _, d in plan.g.nodes(data=True)
                    if d.get("ntype") == "subgoal")
        if n_sub >= 2:
            examples.append(serialize(env, st, plan, moves, {
                "stage": "base", "bench_idx": row_idx,
                "env_id": inst["env_id"], "d_star": inst["d_star"],
                "why": "ordinary plan, original language"}))
            break
    else:
        raise SystemExit("no base example found")

    # 2) B1 worked example: idx 333 — impossible in the original language,
    #    solved optimally with a transient (wall-less) support.
    inst = probe_insts[333]
    env, _ = GridEnv.from_env(inst["env_id"])
    st = state_from_instance(inst)
    plan, moves = find_playable_plan(env, st, b1=True)
    assert plan is not None, "idx 333 must be b1-solvable"
    examples.append(serialize(env, st, plan, moves, {
        "stage": "b1", "bench_idx": 333, "env_id": inst["env_id"],
        "d_star": inst["d_star"],
        "why": "transient (wall-less) stopper; impossible before B1"}))

    # 3) B1 park example: a puzzle whose B1 plan needs one step-aside.
    b1_rows = json.load(open("analysis/artifacts/ceiling_probe_results_b1.json"))
    park_idx = next(r["idx"] for r in b1_rows
                    if r["category"] == "REALIZABLE_EXISTS"
                    and r.get("parks", 0) == 1)
    inst = probe_insts[park_idx]
    env, _ = GridEnv.from_env(inst["env_id"])
    st = state_from_instance(inst)
    plan, moves = find_playable_plan(env, st, b1=True)
    assert plan is not None and any(
        d.get("ntype") == "park" for _, d in plan.g.nodes(data=True)), \
        f"idx {park_idx} should yield a park plan"
    examples.append(serialize(env, st, plan, moves, {
        "stage": "b1", "bench_idx": park_idx, "env_id": inst["env_id"],
        "d_star": inst["d_star"],
        "why": "step-aside (park): a robot clears the way, ordered before "
               "one specific slide"}))

    # 4) B2 supports-by-reference: idx 156 — impossible even under B1,
    #    solved move-optimally by reusing an already-placed robot.
    inst = probe_insts[156]
    env, _ = GridEnv.from_env(inst["env_id"])
    st = state_from_instance(inst)
    plan, moves = find_playable_plan(env, st, b2=True)
    assert plan is not None, "idx 156 must be b2-solvable"
    examples.append(serialize(env, st, plan, moves, {
        "stage": "b2", "bench_idx": 156, "env_id": inst["env_id"],
        "d_star": inst["d_star"],
        "why": "supports-by-reference: a proposal names an existing plan "
               "node as its stopper instead of recruiting a fresh robot"}))

    # 5) B2 pairwise parks: idx 76 — needs TWO robots cleared at once.
    inst = probe_insts[76]
    env, _ = GridEnv.from_env(inst["env_id"])
    st = state_from_instance(inst)
    plan, moves = find_playable_plan(env, st, b2=True)
    assert plan is not None and sum(
        1 for _, d in plan.g.nodes(data=True)
        if d.get("ntype") == "park") == 2, "idx 76 should use two parks"
    examples.append(serialize(env, st, plan, moves, {
        "stage": "b2", "bench_idx": 76, "env_id": inst["env_id"],
        "d_star": inst["d_star"],
        "why": "generalized step-aside: two robots must both clear the "
               "same slide"}))

    json.dump({"generated_by":
               "analysis/plan_viz/extract_plans.py (hand-coded search, "
               "no neural networks; strictly-playable plans only)",
               "examples": examples}, open(out_path, "w"), indent=1)
    print(f"wrote {out_path} with {len(examples)} examples:")
    for e in examples:
        n_by_type = {}
        for n in e["nodes"]:
            n_by_type[n["type"]] = n_by_type.get(n["type"], 0) + 1
        print(f"  [{e['stage']}] bench_idx={e['bench_idx']} d*={e['d_star']} "
              f"legal_moves={e['legal_moves']} plan_cost={e['plan_cost']} "
              f"nodes={n_by_type} byref_edges="
              f"{sum(1 for ed in e['edges'] if ed['byref'])}")


if __name__ == "__main__":
    main()
