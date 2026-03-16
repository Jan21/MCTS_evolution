"""V12: Greedy Rollout A*.

For each candidate subgoal, performs a full greedy rollout to
completion (using V1's greedy strategy). Ranks proposals by their
actual completed plan cost, not by heuristic estimates.

This avoids the greedy trap: a locally cheap subgoal might lead to
expensive sub-problems, while a slightly more expensive one leads to
a fully-fixable plan. By rolling out each proposal, we see the true
cost before committing.

Only rolls out the top-K proposals to keep computation reasonable.
"""
from __future__ import annotations

import copy
import heapq
import time

from GridEnv import GridEnv, State, Robot_at
from partial_plan import PartialPlan
from A_star.base import A_star, SolveResult, PlanEntry, PlanStats
from A_star.v1 import A_star_V1, _find_sibling_support, _get_dep_supports, _get_proposals


INF = 10_000


class A_star_V12(A_star):

    def __init__(self, max_iterations=5000, max_children=30,
                 rollout_top_k=10, time_limit=30):
        self.max_iterations = max_iterations
        self.max_children = max_children
        self.rollout_top_k = rollout_top_k
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

            # For top-K proposals, do greedy rollout to get actual completed cost
            rollout_scored = []
            for i, (subgoal, score, edge_parent_support) in enumerate(proposals):
                if i < self.rollout_top_k:
                    rollout_cost = self._rollout(
                        grid_env, state, current_plan, parent_id, child_id,
                        subgoal, edge_parent_support, nc + i)
                    rollout_scored.append(
                        (subgoal, rollout_cost, edge_parent_support))
                else:
                    rollout_scored.append(
                        (subgoal, score, edge_parent_support))

            rollout_scored.sort(key=lambda x: x[1])

            for subgoal, score, edge_parent_support in rollout_scored:
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

    def _rollout(self, grid_env, state, current_plan, parent_id, child_id,
                 subgoal, edge_parent_support, nc):
        """Greedy rollout: apply subgoal and then greedily complete the plan."""
        plan = copy.deepcopy(current_plan)
        plan.g.remove_edge(parent_id, child_id)

        parent_pos = plan.g.nodes[parent_id]["pos"]
        child_pos = plan.g.nodes[child_id]["pos"]
        bn_pos = subgoal.bottleneck.position
        sp_pos = subgoal.support.position

        sg_id = f"sg_{nc}"
        bn_id = f"bn_{nc}"
        sp_id = f"sp_{nc}"
        leaf_id = f"leaf_{nc}"
        nc += 1

        parent_edge_cost = grid_env.compute_exact_shortest_path_length(
            bn_pos, parent_pos, edge_parent_support)
        if parent_edge_cost is None:
            return INF

        plan.add_node(sg_id, "subgoal", parent_support_pos=edge_parent_support)
        plan.add_node(bn_id, "bottleneck", pos=bn_pos, robot=subgoal.bottleneck)
        plan.add_node(sp_id, "support", pos=sp_pos, robot=subgoal.support)
        plan.add_node(leaf_id, "leaf", pos=subgoal.helper.position,
                      robot=subgoal.helper)

        plan.add_edge(parent_id, sg_id, status="fixed", cost=parent_edge_cost)
        plan.add_edge(sg_id, bn_id, status="fixed", cost=0)
        plan.add_edge(sg_id, sp_id, status="fixed", cost=0)

        bn_child_exact = grid_env.compute_exact_shortest_path_length(
            child_pos, bn_pos, sp_pos)
        if bn_child_exact is not None:
            plan.add_edge(bn_id, child_id, status="fixed", cost=bn_child_exact)
        else:
            bn_child_relaxed = grid_env.compute_relaxed_shortest_path_length(
                child_pos, bn_pos, sp_pos)
            if bn_child_relaxed is None:
                bn_child_relaxed = INF
            plan.add_edge(bn_id, child_id, status="open", cost=bn_child_relaxed)

        sp_leaf_exact = grid_env.compute_exact_shortest_path_length(
            subgoal.helper.position, sp_pos, None)
        if sp_leaf_exact is not None:
            plan.add_edge(sp_id, leaf_id, status="fixed", cost=sp_leaf_exact)
        else:
            sp_leaf_relaxed = grid_env.compute_relaxed_shortest_path_length(
                subgoal.helper.position, sp_pos, None)
            if sp_leaf_relaxed is None:
                sp_leaf_relaxed = INF
            plan.add_edge(sp_id, leaf_id, status="open", cost=sp_leaf_relaxed)

        # Greedily resolve remaining open edges (max 20 iterations to avoid loops)
        for _ in range(20):
            open_edges = plan.open_edges()
            if not open_edges:
                break

            p_id, c_id = open_edges[0]
            p_data = plan.g.nodes[p_id]
            c_data = plan.g.nodes[c_id]
            p_type = p_data["ntype"]
            p_pos = p_data["pos"]
            c_pos = c_data["pos"]

            seg_sp_pos = None
            seg_sp_robot = None
            if p_type == "bottleneck":
                seg_sp_pos, seg_sp_robot = _find_sibling_support(plan, p_id)

            if p_type == "goal":
                m_robot = state.target_robot
            else:
                m_robot = p_data.get("robot", state.target_robot)

            fix_sp = seg_sp_pos if p_type == "bottleneck" else None
            ex = grid_env.compute_exact_shortest_path_length(c_pos, p_pos, fix_sp)

            if ex is not None:
                plan.g[p_id][c_id]["status"] = "fixed"
                plan.g[p_id][c_id]["cost"] = ex
                continue

            seg_state = State(target=p_pos, target_robot=m_robot,
                             helpers=state.helpers)
            props = _get_proposals(grid_env, seg_state, seg_sp_robot,
                                   seg_sp_pos, p_pos)
            if not props:
                return INF

            props.sort(key=lambda x: x[1])
            sg, _, eps = props[0]

            plan.g.remove_edge(p_id, c_id)
            s_sg = f"sg_{nc}"
            s_bn = f"bn_{nc}"
            s_sp = f"sp_{nc}"
            s_lf = f"leaf_{nc}"
            nc += 1

            bn_p = sg.bottleneck.position
            sp_p = sg.support.position

            pe_cost = grid_env.compute_exact_shortest_path_length(
                bn_p, p_pos, eps)
            if pe_cost is None:
                return INF

            plan.add_node(s_sg, "subgoal", parent_support_pos=eps)
            plan.add_node(s_bn, "bottleneck", pos=bn_p, robot=sg.bottleneck)
            plan.add_node(s_sp, "support", pos=sp_p, robot=sg.support)
            plan.add_node(s_lf, "leaf", pos=sg.helper.position, robot=sg.helper)

            plan.add_edge(p_id, s_sg, status="fixed", cost=pe_cost)
            plan.add_edge(s_sg, s_bn, status="fixed", cost=0)
            plan.add_edge(s_sg, s_sp, status="fixed", cost=0)

            bce = grid_env.compute_exact_shortest_path_length(c_pos, bn_p, sp_p)
            if bce is not None:
                plan.add_edge(s_bn, c_id, status="fixed", cost=bce)
            else:
                bcr = grid_env.compute_relaxed_shortest_path_length(
                    c_pos, bn_p, sp_p)
                plan.add_edge(s_bn, c_id, status="open", cost=bcr or INF)

            sle = grid_env.compute_exact_shortest_path_length(
                sg.helper.position, sp_p, None)
            if sle is not None:
                plan.add_edge(s_sp, s_lf, status="fixed", cost=sle)
            else:
                slr = grid_env.compute_relaxed_shortest_path_length(
                    sg.helper.position, sp_p, None)
                plan.add_edge(s_sp, s_lf, status="open", cost=slr or INF)

        return plan.cost()
