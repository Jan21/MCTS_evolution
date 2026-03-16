"""V8: Diverse Beam Search with structural diversity enforcement.

Like V4 (beam search), but enforces diversity among beam members.
Plans in the beam must differ in their subgoal structure — we measure
diversity by the set of bottleneck positions used. This prevents
the beam from collapsing to slight variations of the same plan.

Also uses ALL open edges expansion: at each step, expands the most
constrained open edge (fewest proposals available).
"""
from __future__ import annotations

import copy
import time

from GridEnv import GridEnv, State, Robot_at
from partial_plan import PartialPlan
from A_star.base import A_star, SolveResult, PlanEntry, PlanStats
from A_star.v1 import _find_sibling_support, _get_dep_supports, _get_proposals


INF = 10_000


def _get_bottleneck_positions(plan):
    """Extract set of bottleneck positions from a plan for diversity."""
    positions = set()
    for node_id, data in plan.g.nodes(data=True):
        if data.get("ntype") == "bottleneck":
            positions.add(data.get("pos"))
    return frozenset(positions)


class A_star_V8(A_star):

    def __init__(self, beam_width=30, max_depth=10, max_children=50,
                 diversity_groups=5, time_limit=30):
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.max_children = max_children
        self.diversity_groups = diversity_groups
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

        beam = [(plan.cost(), plan, 1)]
        best_cost = float('inf')
        best_plan = None
        all_plans = []
        iteration = 0

        for depth in range(self.max_depth):
            if time.time() - t0 > self.time_limit:
                break

            next_beam = []

            for cost, current_plan, nc in beam:
                if cost >= best_cost:
                    continue

                open_edges = current_plan.open_edges()
                if not open_edges:
                    if cost < best_cost:
                        best_cost = cost
                        best_plan = current_plan
                        iteration += 1
                        all_plans.append(PlanEntry(
                            plan=current_plan, cost=cost,
                            stats=PlanStats(
                                wall_time=time.time() - t0,
                                iteration=iteration,
                                node_count=len(beam),
                                rollout_count=iteration)))
                    continue

                # Select most constrained edge (fewest proposals)
                parent_id, child_id = self._select_edge(
                    current_plan, open_edges, grid_env, state)

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
                        next_beam.append((new_cost, new_plan, nc))
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

                for subgoal, score, edge_parent_support in proposals:
                    new_plan = copy.deepcopy(current_plan)
                    new_plan.g.remove_edge(parent_id, child_id)

                    sg_id = f"sg_{nc}"
                    bn_id = f"bn_{nc}"
                    sp_id = f"sp_{nc}"
                    leaf_id = f"leaf_{nc}"
                    nc_new = nc + 1

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
                        next_beam.append((new_cost, new_plan, nc_new))

                    iteration += 1

            if not next_beam:
                break

            # Diverse beam selection: group by bottleneck positions,
            # take top from each group
            next_beam.sort(key=lambda x: x[0])
            beam = self._diverse_select(next_beam)

        if best_plan is None:
            best_plan = plan

        return SolveResult(best_plan=best_plan, all_plans=all_plans)

    def _select_edge(self, plan, open_edges, grid_env, state):
        """Select most constrained open edge."""
        if len(open_edges) == 1:
            return open_edges[0]

        best = open_edges[0]
        best_proposals = float('inf')

        for parent_id, child_id in open_edges:
            parent_data = plan.g.nodes[parent_id]
            parent_type = parent_data["ntype"]
            parent_pos = parent_data["pos"]

            segment_support_pos = None
            segment_support_robot = None
            if parent_type == "bottleneck":
                segment_support_pos, segment_support_robot = \
                    _find_sibling_support(plan, parent_id)

            if parent_type == "goal":
                moving_robot = state.target_robot
            else:
                moving_robot = parent_data.get("robot", state.target_robot)

            segment_state = State(
                target=parent_pos,
                target_robot=moving_robot,
                helpers=state.helpers,
            )

            try:
                proposals = _get_proposals(
                    grid_env, segment_state, segment_support_robot,
                    segment_support_pos, parent_pos)
                n_proposals = len(proposals)
            except Exception:
                n_proposals = 0

            if n_proposals < best_proposals:
                best_proposals = n_proposals
                best = (parent_id, child_id)

        return best

    def _diverse_select(self, candidates):
        """Select diverse beam: group by structure, take best from each group."""
        if len(candidates) <= self.beam_width:
            return candidates

        # Group by bottleneck position sets
        groups = {}
        for item in candidates:
            cost, plan, nc = item
            key = _get_bottleneck_positions(plan)
            if key not in groups:
                groups[key] = []
            groups[key].append(item)

        # Take top from each group, round-robin until beam is full
        selected = []
        group_lists = list(groups.values())
        idx = 0
        while len(selected) < self.beam_width and group_lists:
            for group in group_lists:
                if idx < len(group) and len(selected) < self.beam_width:
                    selected.append(group[idx])
            idx += 1
            group_lists = [g for g in group_lists if idx < len(g)]

        return selected
