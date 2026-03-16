"""V3: Randomized Multi-Restart A*.

Runs the A* search multiple times with random tiebreaking among
equally-scored subgoals. Takes the best plan across all restarts.
This explores different parts of the search tree that greedy A*
misses because it always picks the first among equal-score options.
"""
from __future__ import annotations

import copy
import heapq
import random
import time

from GridEnv import GridEnv, State, Robot_at
from partial_plan import PartialPlan
from A_star.base import A_star, SolveResult, PlanEntry, PlanStats
from A_star.v1 import _find_sibling_support, _get_dep_supports, _get_proposals


INF = 10_000


class A_star_V3(A_star):

    def __init__(self, max_iterations=5000, max_children=30,
                 n_restarts=10, seed=42):
        self.max_iterations = max_iterations
        self.max_children = max_children
        self.n_restarts = n_restarts
        self.seed = seed

    def solve(self, grid_env: GridEnv, state: State) -> SolveResult:
        t0 = time.time()

        # Check direct path
        direct = grid_env.compute_exact_shortest_path_length(
            state.target_robot.position, state.target, None)
        if direct is not None:
            plan = PartialPlan()
            plan.add_node("goal", "goal", pos=state.target)
            plan.add_node("leaf_0", "leaf",
                           pos=state.target_robot.position,
                           robot=state.target_robot)
            plan.add_edge("goal", "leaf_0", status="fixed", cost=direct)
            return SolveResult(
                best_plan=plan,
                all_plans=[PlanEntry(
                    plan=plan, cost=direct,
                    stats=PlanStats(wall_time=time.time() - t0,
                                    iteration=0, node_count=2,
                                    rollout_count=0))])

        best_cost = float('inf')
        best_plan = None
        all_plans = []
        rng = random.Random(self.seed)

        for restart in range(self.n_restarts):
            plan, plans_found = self._single_run(
                grid_env, state, t0, best_cost, rng)
            all_plans.extend(plans_found)
            for entry in plans_found:
                if entry.cost < best_cost:
                    best_cost = entry.cost
                    best_plan = entry.plan

        if best_plan is None:
            plan = PartialPlan()
            plan.add_node("goal", "goal", pos=state.target)
            plan.add_node("leaf_0", "leaf",
                           pos=state.target_robot.position,
                           robot=state.target_robot)
            init_cost = grid_env.compute_relaxed_shortest_path_length(
                state.target_robot.position, state.target, None) or INF
            plan.add_edge("goal", "leaf_0", status="open", cost=init_cost)
            best_plan = plan

        return SolveResult(best_plan=best_plan, all_plans=all_plans)

    def _single_run(self, grid_env, state, t0, global_best_cost, rng):
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=state.target)
        plan.add_node("leaf_0", "leaf",
                       pos=state.target_robot.position,
                       robot=state.target_robot)

        init_cost = grid_env.compute_relaxed_shortest_path_length(
            state.target_robot.position, state.target, None)
        if init_cost is None:
            init_cost = INF
        plan.add_edge("goal", "leaf_0", status="open", cost=init_cost)

        tie = 0
        pq = [(plan.cost(), tie, plan, 1)]
        tie += 1

        best_cost = global_best_cost
        found_plans = []
        iteration = 0

        while pq and iteration < self.max_iterations:
            iteration += 1
            cost, _, current_plan, nc = heapq.heappop(pq)

            if cost >= best_cost:
                continue

            open_edges = current_plan.open_edges()
            if not open_edges:
                if cost < best_cost:
                    best_cost = cost
                    found_plans.append(PlanEntry(
                        plan=current_plan, cost=cost,
                        stats=PlanStats(
                            wall_time=time.time() - t0,
                            iteration=iteration,
                            node_count=len(pq),
                            rollout_count=iteration)))
                continue

            parent_id, child_id = open_edges[0]
            parent_data = current_plan.g.nodes[parent_id]
            child_data = current_plan.g.nodes[child_id]
            parent_type = parent_data["ntype"]
            parent_pos = parent_data["pos"]
            child_pos = child_data["pos"]

            segment_support_pos = None
            segment_support_robot = None
            if parent_type == "bottleneck":
                segment_support_pos, segment_support_robot = \
                    _find_sibling_support(current_plan, parent_id)

            if parent_type == "goal":
                moving_robot = state.target_robot
            else:
                moving_robot = parent_data.get("robot", state.target_robot)

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

            segment_state = State(
                target=parent_pos,
                target_robot=moving_robot,
                helpers=state.helpers,
            )

            proposals = _get_proposals(
                grid_env, segment_state, segment_support_robot,
                segment_support_pos, parent_pos)

            # Randomized selection: shuffle among top candidates
            proposals.sort(key=lambda x: x[1])
            # Keep top proposals but shuffle their order
            top_k = proposals[:self.max_children]
            rng.shuffle(top_k)
            proposals = top_k

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
                new_plan.add_node(leaf_id, "leaf",
                                  pos=subgoal.helper.position,
                                  robot=subgoal.helper)

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

        return None, found_plans
