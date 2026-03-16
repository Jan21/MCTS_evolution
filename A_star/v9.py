"""V9: Two-Level Lookahead A*.

Before committing to a subgoal, performs a one-step lookahead:
for each candidate subgoal, estimates what the NEXT level of
subgoals would cost. Uses the lookahead cost as a better heuristic
for ranking proposals.

This helps avoid greedy traps where a locally cheap subgoal leads
to expensive sub-problems, while a slightly more expensive subgoal
leads to easier remaining segments.
"""
from __future__ import annotations

import copy
import heapq
import time

from GridEnv import GridEnv, State, Robot_at
from partial_plan import PartialPlan
from A_star.base import A_star, SolveResult, PlanEntry, PlanStats
from A_star.v1 import _find_sibling_support, _get_dep_supports, _get_proposals


INF = 10_000


class A_star_V9(A_star):

    def __init__(self, max_iterations=5000, max_children=30,
                 lookahead_children=10, time_limit=30):
        self.max_iterations = max_iterations
        self.max_children = max_children
        self.lookahead_children = lookahead_children
        self.time_limit = time_limit

    def solve(self, grid_env: GridEnv, state: State) -> SolveResult:
        t0 = time.time()

        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=state.target)
        plan.add_node("leaf_0", "leaf",
                       pos=state.target_robot.position,
                       robot=state.target_robot)

        direct = grid_env.compute_exact_shortest_path_length(
            state.target_robot.position, state.target, None)
        if direct is not None:
            plan.add_edge("goal", "leaf_0", status="fixed", cost=direct)
            return SolveResult(
                best_plan=plan,
                all_plans=[PlanEntry(
                    plan=plan, cost=direct,
                    stats=PlanStats(wall_time=time.time() - t0,
                                    iteration=0, node_count=2,
                                    rollout_count=0))])

        init_cost = grid_env.compute_relaxed_shortest_path_length(
            state.target_robot.position, state.target, None)
        if init_cost is None:
            init_cost = INF
        plan.add_edge("goal", "leaf_0", status="open", cost=init_cost)

        tie = 0
        pq = [(plan.cost(), tie, plan, 1)]
        tie += 1

        best_cost = float('inf')
        best_plan = None
        all_plans = []
        iteration = 0

        while pq and iteration < self.max_iterations:
            if time.time() - t0 > self.time_limit:
                break
            iteration += 1
            cost, _, current_plan, nc = heapq.heappop(pq)

            if cost >= best_cost:
                continue

            open_edges = current_plan.open_edges()
            if not open_edges:
                if cost < best_cost:
                    best_cost = cost
                    best_plan = current_plan
                    all_plans.append(PlanEntry(
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

            proposals.sort(key=lambda x: x[1])
            proposals = proposals[:self.max_children]

            # Score proposals with lookahead
            scored_proposals = []
            for subgoal, score, edge_parent_support in proposals:
                lookahead_cost = self._lookahead_score(
                    grid_env, state, current_plan, parent_id, child_id,
                    subgoal, score, edge_parent_support, nc)
                scored_proposals.append(
                    (subgoal, lookahead_cost, edge_parent_support))

            scored_proposals.sort(key=lambda x: x[1])

            for subgoal, score, edge_parent_support in scored_proposals:
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

        if best_plan is None:
            best_plan = plan

        return SolveResult(best_plan=best_plan, all_plans=all_plans)

    def _lookahead_score(self, grid_env, state, current_plan,
                         parent_id, child_id, subgoal, base_score,
                         edge_parent_support, nc):
        """Estimate total cost after applying this subgoal + one more level."""
        bn_pos = subgoal.bottleneck.position
        sp_pos = subgoal.support.position
        parent_pos = current_plan.g.nodes[parent_id]["pos"]
        child_pos = current_plan.g.nodes[child_id]["pos"]

        # Cost of parent -> bottleneck
        parent_edge_cost = grid_env.compute_exact_shortest_path_length(
            bn_pos, parent_pos, edge_parent_support)
        if parent_edge_cost is None:
            return INF

        # Cost of bn -> child (with support)
        bn_child_exact = grid_env.compute_exact_shortest_path_length(
            child_pos, bn_pos, sp_pos)
        if bn_child_exact is not None:
            bn_child_cost = bn_child_exact
            bn_child_needs_subgoal = False
        else:
            bn_child_relaxed = grid_env.compute_relaxed_shortest_path_length(
                child_pos, bn_pos, sp_pos)
            bn_child_cost = bn_child_relaxed if bn_child_relaxed else INF
            bn_child_needs_subgoal = True

        # Cost of sp -> helper leaf
        helper_pos = subgoal.helper.position
        sp_leaf_exact = grid_env.compute_exact_shortest_path_length(
            helper_pos, sp_pos, None)
        if sp_leaf_exact is not None:
            sp_leaf_cost = sp_leaf_exact
        else:
            sp_leaf_relaxed = grid_env.compute_relaxed_shortest_path_length(
                helper_pos, sp_pos, None)
            sp_leaf_cost = sp_leaf_relaxed if sp_leaf_relaxed else INF

        immediate_cost = parent_edge_cost + bn_child_cost + sp_leaf_cost

        # Lookahead: if bn->child still open, estimate best subgoal cost
        lookahead_bonus = 0
        if bn_child_needs_subgoal and bn_child_cost < INF:
            moving_robot = subgoal.bottleneck
            segment_state = State(
                target=bn_pos,
                target_robot=moving_robot,
                helpers=state.helpers,
            )
            try:
                sub_proposals = _get_proposals(
                    grid_env, segment_state, subgoal.support,
                    sp_pos, bn_pos)
                if sub_proposals:
                    sub_proposals.sort(key=lambda x: x[1])
                    best_sub = sub_proposals[0]
                    # The lookahead estimates the additional cost
                    # beyond the relaxed estimate we already counted
                    lookahead_bonus = max(0, best_sub[1] - bn_child_cost)
            except Exception:
                pass

        return immediate_cost + lookahead_bonus
