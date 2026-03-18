"""Dump all partial plans from A* search, separated by type.

Runs A_star_V1 on each environment and saves every partial plan encountered
during search. Plans are organized into folders by type:

    data_gen/plans/
        env_{idx}/
            complete/     — plans with zero open edges (solutions)
            intermediate/ — plans popped from PQ with open edges (not pruned)
            pruned/       — plans skipped because cost >= best_cost

Each plan is saved as a pickle file containing:
    {
        'plan': PartialPlan,
        'cost': float,
        'iteration': int,
        'category': str,          # 'complete', 'intermediate', 'pruned'
        'n_open_edges': int,
        'n_fixed_edges': int,
        'n_nodes': int,
        'proposals': dict,        # per open edge: best subgoal proposal
    }

proposals maps (parent_id, child_id) -> {
    'bn_pos': tuple,
    'sp_pos': tuple,
    'support_robot': Robot_at,
    'moving_robot': Robot_at,
    'score': float,
} or None if the edge is fixable directly (no subgoal needed).

Usage:
    python -m data_gen.dump_plans --n 20
    python -m data_gen.dump_plans --n 128 --output data_gen/plans
"""

from __future__ import annotations

import argparse
import copy
import heapq
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from GridEnv import GridEnv, State, Robot_at
from partial_plan import PartialPlan
from A_star.v1 import _find_sibling_support, _find_committed_supports, _get_proposals

INF = 10_000


def _plan_metadata(plan: PartialPlan, cost: float, iteration: int,
                   category: str, proposals: dict | None = None) -> dict:
    open_edges = plan.open_edges()
    fixed_edges = [(u, v) for u, v, d in plan.g.edges(data=True)
                   if d["status"] == "fixed"]
    return {
        'plan': plan,
        'cost': cost,
        'iteration': iteration,
        'category': category,
        'n_open_edges': len(open_edges),
        'n_fixed_edges': len(fixed_edges),
        'n_nodes': plan.g.number_of_nodes(),
        'proposals': proposals or {},
    }


def _compute_proposals_for_all_edges(grid_env, state, plan):
    """Compute best subgoal proposal for each open edge in the plan.

    Returns dict mapping (parent_id, child_id) -> proposal dict or None.
    """
    proposals = {}
    for parent_id, child_id in plan.open_edges():
        parent_data = plan.g.nodes[parent_id]
        child_data = plan.g.nodes[child_id]
        parent_type = parent_data["ntype"]
        parent_pos = parent_data["pos"]
        child_pos = child_data["pos"]

        seg_support_pos = None
        seg_support_robot = None
        if parent_type == "bottleneck":
            seg_support_pos, seg_support_robot = \
                _find_sibling_support(plan, parent_id)

        if parent_type == "goal":
            moving_robot = state.target_robot
        else:
            moving_robot = parent_data.get("robot", state.target_robot)

        # Check if edge can be fixed directly
        fix_support = seg_support_pos if parent_type == "bottleneck" else None
        exact = grid_env.compute_exact_shortest_path_length(
            child_pos, parent_pos, fix_support)

        if exact is not None:
            proposals[(parent_id, child_id)] = None  # fixable directly
            continue

        # Get subgoal proposals
        seg_helpers = [h for h in state.helpers if h != moving_robot]
        if moving_robot != state.target_robot:
            seg_helpers.append(state.target_robot)

        seg_state = State(
            target=parent_pos,
            target_robot=moving_robot,
            helpers=seg_helpers,
        )

        raw_proposals = _get_proposals(
            grid_env, seg_state, seg_support_robot,
            seg_support_pos, parent_pos)

        if not raw_proposals:
            proposals[(parent_id, child_id)] = None
            continue

        raw_proposals.sort(key=lambda x: x[1])
        best_sg, best_score, _ = raw_proposals[0]

        proposals[(parent_id, child_id)] = {
            'bn_pos': best_sg.bottleneck.position,
            'sp_pos': best_sg.support.position,
            'support_robot': best_sg.support,
            'moving_robot': moving_robot,
            'score': best_score,
        }

    return proposals


def solve_and_dump(env_idx: int, output_dir: Path,
                   max_iterations: int = 10000, max_children: int = 50):
    """Run A* on one environment and dump all partial plans by category."""
    grid_env, state = GridEnv.from_env(env_idx)

    env_dir = output_dir / f"env_{env_idx}"
    for cat in ("complete", "intermediate", "pruned"):
        (env_dir / cat).mkdir(parents=True, exist_ok=True)

    counters = {"complete": 0, "intermediate": 0, "pruned": 0}

    def _save(plan, cost, iteration, category, proposals=None):
        meta = _plan_metadata(plan, cost, iteration, category, proposals)
        idx = counters[category]
        counters[category] += 1
        with open(env_dir / category / f"plan_{idx}.pkl", "wb") as f:
            pickle.dump(meta, f)

    # --- A* search (mirrors v1.py logic, with dumping) ---
    plan = PartialPlan()
    plan.add_node("goal", "goal", pos=state.target)
    plan.add_node("leaf_0", "leaf",
                  pos=state.target_robot.position,
                  robot=state.target_robot)

    best_cost = float('inf')

    # Direct path check
    direct = grid_env.compute_exact_shortest_path_length(
        state.target_robot.position, state.target, None)
    if direct is not None:
        direct_plan = copy.deepcopy(plan)
        direct_plan.add_edge("goal", "leaf_0", status="fixed", cost=direct)
        best_cost = direct
        _save(direct_plan, direct, 0, "complete")

    init_cost = grid_env.compute_relaxed_shortest_path_length(
        state.target_robot.position, state.target, None)
    if init_cost is None:
        init_cost = INF

    if init_cost >= best_cost:
        print(f"env {env_idx}: direct path only — "
              f"{counters['complete']}c/{counters['intermediate']}i/{counters['pruned']}p")
        return counters

    plan.add_edge("goal", "leaf_0", status="open", cost=init_cost)
    _save(plan, plan.cost(), 0, "intermediate")

    tie = 0
    pq = [(plan.cost(), tie, plan, 1)]
    tie += 1
    iteration = 0

    while pq and iteration < max_iterations:
        iteration += 1
        cost, _, current_plan, nc = heapq.heappop(pq)

        if cost >= best_cost:
            _save(current_plan, cost, iteration, "pruned")
            continue

        open_edges = current_plan.open_edges()
        if not open_edges:
            if cost < best_cost:
                best_cost = cost
            _save(current_plan, cost, iteration, "complete")
            continue

        # Save as intermediate (popped, has open edges, not pruned)
        # Compute best proposals for all open edges
        all_proposals = _compute_proposals_for_all_edges(
            grid_env, state, current_plan)
        _save(current_plan, cost, iteration, "intermediate", all_proposals)

        parent_id, child_id = open_edges[0]
        parent_data = current_plan.g.nodes[parent_id]
        child_data = current_plan.g.nodes[child_id]
        parent_type = parent_data["ntype"]
        parent_pos = parent_data["pos"]
        child_pos = child_data["pos"]

        # Determine segment support
        segment_support_pos = None
        segment_support_robot = None
        if parent_type == "bottleneck":
            segment_support_pos, segment_support_robot = \
                _find_sibling_support(current_plan, parent_id)

        # Determine moving robot
        if parent_type == "goal":
            moving_robot = state.target_robot
        else:
            moving_robot = parent_data.get("robot", state.target_robot)

        # Try fixing edge directly
        fix_support = segment_support_pos if parent_type == "bottleneck" else None
        exact = grid_env.compute_exact_shortest_path_length(
            child_pos, parent_pos, fix_support)

        if exact is not None:
            new_plan = copy.deepcopy(current_plan)
            new_plan.g[parent_id][child_id]["status"] = "fixed"
            new_plan.g[parent_id][child_id]["cost"] = exact
            new_cost = new_plan.cost()
            if new_cost < best_cost:
                heapq.heappush(pq, (new_cost, tie, new_plan, nc))
                tie += 1
            continue

        # Propose subgoals
        segment_helpers = [h for h in state.helpers if h != moving_robot]
        if moving_robot != state.target_robot:
            segment_helpers.append(state.target_robot)

        segment_state = State(
            target=parent_pos,
            target_robot=moving_robot,
            helpers=segment_helpers,
        )

        proposals = _get_proposals(
            grid_env, segment_state, segment_support_robot,
            segment_support_pos, parent_pos)

        proposals.sort(key=lambda x: x[1])
        proposals = proposals[:max_children]

        for subgoal, score, edge_parent_support in proposals:
            new_plan = copy.deepcopy(current_plan)
            new_plan.g.remove_edge(parent_id, child_id)

            sg_id = f"sg_{nc}"
            bn_id = f"bn_{nc}"
            sp_id = f"sp_{nc}"
            leaf_id = f"leaf_{nc}"
            nc += 1

            bn_pos = subgoal.bottleneck.position
            sp_pos = subgoal.support.position

            parent_edge_cost = grid_env.compute_exact_shortest_path_length(
                bn_pos, parent_pos, edge_parent_support)
            if parent_edge_cost is None:
                continue

            new_plan.add_node(sg_id, "subgoal",
                              parent_support_pos=edge_parent_support)
            new_plan.add_node(bn_id, "bottleneck",
                              pos=bn_pos, robot=subgoal.bottleneck)
            new_plan.add_node(sp_id, "support",
                              pos=sp_pos, robot=subgoal.support)
            new_plan.add_edge(parent_id, sg_id,
                              status="fixed", cost=parent_edge_cost)
            new_plan.add_edge(sg_id, bn_id, status="fixed", cost=0)
            new_plan.add_edge(sg_id, sp_id, status="fixed", cost=0)

            bn_child_exact = grid_env.compute_exact_shortest_path_length(
                child_pos, bn_pos, sp_pos)
            if bn_child_exact is not None:
                new_plan.add_edge(bn_id, child_id,
                                  status="fixed", cost=bn_child_exact)
            else:
                bn_child_relaxed = grid_env.compute_relaxed_shortest_path_length(
                    child_pos, bn_pos, sp_pos)
                if bn_child_relaxed is None:
                    bn_child_relaxed = INF
                new_plan.add_edge(bn_id, child_id,
                                  status="open", cost=bn_child_relaxed)

            committed = _find_committed_supports(new_plan, subgoal.helper.color)
            committed = [(n, p) for n, p in committed if n != sp_id]

            if not committed:
                new_plan.add_node(leaf_id, "leaf",
                                  pos=subgoal.helper.position,
                                  robot=subgoal.helper)
                sp_leaf_exact = grid_env.compute_exact_shortest_path_length(
                    subgoal.helper.position, sp_pos, None)
                if sp_leaf_exact is not None:
                    new_plan.add_edge(sp_id, leaf_id,
                                      status="fixed", cost=sp_leaf_exact)
                else:
                    sp_leaf_relaxed = grid_env.compute_relaxed_shortest_path_length(
                        subgoal.helper.position, sp_pos, None)
                    if sp_leaf_relaxed is None:
                        sp_leaf_relaxed = INF
                    new_plan.add_edge(sp_id, leaf_id,
                                      status="open", cost=sp_leaf_relaxed)

                new_cost = new_plan.cost()
                if new_cost < best_cost:
                    heapq.heappush(pq, (new_cost, tie, new_plan, nc))
                    tie += 1
            else:
                for commit_nid, commit_pos in committed:
                    commit_child = None
                    for succ in new_plan.g.successors(commit_nid):
                        stype = new_plan.g.nodes[succ].get("ntype")
                        if stype in ("leaf", "support"):
                            commit_child = succ
                            break

                    # Variant A
                    va = copy.deepcopy(new_plan)
                    cost_a = grid_env.compute_exact_shortest_path_length(
                        commit_pos, sp_pos, None)
                    if cost_a is not None:
                        va.add_edge(sp_id, commit_nid,
                                    status="fixed", cost=cost_a)
                    else:
                        cost_a_r = grid_env.compute_relaxed_shortest_path_length(
                            commit_pos, sp_pos, None)
                        if cost_a_r is None:
                            cost_a_r = INF
                        va.add_edge(sp_id, commit_nid,
                                    status="open", cost=cost_a_r)
                    va_cost = va.cost()
                    if va_cost < best_cost:
                        heapq.heappush(pq, (va_cost, tie, va, nc))
                        tie += 1

                    # Variant B
                    if commit_child is not None:
                        vb = copy.deepcopy(new_plan)
                        old_child_pos = vb.g.nodes[commit_child]["pos"]
                        vb.g.remove_edge(commit_nid, commit_child)

                        cost_b1 = grid_env.compute_exact_shortest_path_length(
                            sp_pos, commit_pos, None)
                        if cost_b1 is not None:
                            vb.add_edge(commit_nid, sp_id,
                                        status="fixed", cost=cost_b1)
                        else:
                            cost_b1_r = grid_env.compute_relaxed_shortest_path_length(
                                sp_pos, commit_pos, None)
                            if cost_b1_r is None:
                                cost_b1_r = INF
                            vb.add_edge(commit_nid, sp_id,
                                        status="open", cost=cost_b1_r)

                        cost_b2 = grid_env.compute_exact_shortest_path_length(
                            old_child_pos, sp_pos, None)
                        if cost_b2 is not None:
                            vb.add_edge(sp_id, commit_child,
                                        status="fixed", cost=cost_b2)
                        else:
                            cost_b2_r = grid_env.compute_relaxed_shortest_path_length(
                                old_child_pos, sp_pos, None)
                            if cost_b2_r is None:
                                cost_b2_r = INF
                            vb.add_edge(sp_id, commit_child,
                                        status="open", cost=cost_b2_r)

                        vb_cost = vb.cost()
                        if vb_cost < best_cost:
                            heapq.heappush(pq, (vb_cost, tie, vb, nc))
                            tie += 1

    print(f"env {env_idx}: "
          f"{counters['complete']}c/{counters['intermediate']}i/{counters['pruned']}p")
    return counters


def main():
    parser = argparse.ArgumentParser(
        description="Dump all partial plans from A* search.")
    parser.add_argument('--n', type=int, default=20,
                        help='Number of environments to process')
    parser.add_argument('--output', type=str, default='data_gen/plans',
                        help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output)
    total = {"complete": 0, "intermediate": 0, "pruned": 0}

    for env_idx in range(args.n):
        counters = solve_and_dump(env_idx, output_dir)
        for k in total:
            total[k] += counters[k]

    print(f"\nTotal across {args.n} envs: "
          f"{total['complete']} complete, "
          f"{total['intermediate']} intermediate, "
          f"{total['pruned']} pruned")


if __name__ == '__main__':
    main()
