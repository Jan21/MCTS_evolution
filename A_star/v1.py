"""V1: Greedy Best-First A* — simplest baseline.

Priority queue ordered by total plan cost. At each step:
  1. Pop the lowest-cost partial plan
  2. Pick the first open edge
  3. Try to fix it directly (exact shortest path exists)
  4. Otherwise, propose subgoal states and create child plans
  5. Track best complete plan for pruning

Handles nodes reachable only via dependent edges by trying
multiple parent-support positions from the dependent edge cache.
"""
from __future__ import annotations

import copy
import heapq
import time

from GridEnv import GridEnv, State, Robot_at
from partial_plan import PartialPlan
from A_star.base import A_star, SolveResult, PlanEntry, PlanStats


INF = 10_000


class A_star_V1(A_star):

    def __init__(self, max_iterations=10000, max_children=50):
        self.max_iterations = max_iterations
        self.max_children = max_children

    def solve(self, grid_env: GridEnv, state: State) -> SolveResult:
        t0 = time.time()

        # Build initial plan: goal -> leaf_target
        plan = PartialPlan()
        plan.add_node("goal", "goal", pos=state.target)
        plan.add_node("leaf_0", "leaf",
                       pos=state.target_robot.position,
                       robot=state.target_robot)

        # Check if target can reach goal directly (independent path)
        # If so, use as initial upper bound but keep searching for
        # potentially cheaper subgoal-based plans via dependent edges.
        best_cost = float('inf')
        best_plan = None
        all_plans = []

        direct = grid_env.compute_exact_shortest_path_length(
            state.target_robot.position, state.target, None)
        if direct is not None:
            direct_plan = copy.deepcopy(plan)
            direct_plan.add_edge("goal", "leaf_0", status="fixed", cost=direct)
            best_cost = direct
            best_plan = direct_plan
            all_plans.append(PlanEntry(
                plan=direct_plan, cost=direct,
                stats=PlanStats(wall_time=time.time() - t0,
                                iteration=0, node_count=2,
                                rollout_count=0)))

        init_cost = grid_env.compute_relaxed_shortest_path_length(
            state.target_robot.position, state.target, None)
        if init_cost is None:
            init_cost = INF

        # Only search for subgoal plans if relaxed estimate suggests
        # there might be something cheaper than the direct path
        if init_cost >= best_cost:
            return SolveResult(best_plan=best_plan, all_plans=all_plans)

        plan.add_edge("goal", "leaf_0", status="open", cost=init_cost)

        # Priority queue: (cost, tiebreaker, plan, node_counter)
        tie = 0
        pq = [(plan.cost(), tie, plan, 1)]
        tie += 1
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

            # Determine segment support (for bottleneck parents)
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

            # Try fixing the edge directly (must match validation's support)
            # Validation uses: None for goal/support parents, sibling_sp for bottleneck
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
            # Swap: remove moving robot from helpers, add target robot if different
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
            proposals = proposals[:self.max_children]

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

                # Edge: parent -> subgoal (fixed)
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

                # bottleneck -> child (target to bottleneck via dep edge)
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

                # support -> helper: chain or leaf
                committed = _find_committed_supports(
                    new_plan, subgoal.helper.color)
                # Exclude the support we just added (sp_id)
                committed = [(n, p) for n, p in committed if n != sp_id]

                if not committed:
                    # No prior commitment: create leaf as before
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
                    # Helper already committed — generate both ordering
                    # variants per committed support
                    for commit_nid, commit_pos in committed:
                        # Find the child of the committed support
                        commit_child = None
                        for succ in new_plan.g.successors(commit_nid):
                            stype = new_plan.g.nodes[succ].get("ntype")
                            if stype in ("leaf", "support"):
                                commit_child = succ
                                break

                        # Variant A: new_support → committed_support → ... → leaf
                        # Helper goes to committed pos first, then to new pos
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

                        # Variant B: committed_support → new_support → leaf
                        # Helper goes to new pos first, then to committed pos
                        if commit_child is not None:
                            vb = copy.deepcopy(new_plan)
                            old_child_pos = vb.g.nodes[commit_child]["pos"]
                            vb.g.remove_edge(commit_nid, commit_child)

                            # committed → new_support
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

                            # new_support → old child (leaf or support)
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

        if best_plan is None:
            best_plan = plan

        return SolveResult(best_plan=best_plan, all_plans=all_plans)


def _find_sibling_support(plan: PartialPlan, bottleneck_id: str):
    """Find the sibling support node of a bottleneck."""
    for sg_parent in plan.g.predecessors(bottleneck_id):
        if plan.g.nodes[sg_parent].get("ntype") == "subgoal":
            for sib in plan.g.successors(sg_parent):
                sib_data = plan.g.nodes[sib]
                if sib_data.get("ntype") == "support":
                    return sib_data["pos"], sib_data["robot"]
    return None, None


def _find_committed_supports(plan: PartialPlan, robot_color: str):
    """Find support nodes in the plan that use a robot of the given color."""
    results = []
    for nid, data in plan.g.nodes(data=True):
        if data.get("ntype") == "support" and data.get("robot") is not None:
            if data["robot"].color == robot_color:
                results.append((nid, data["pos"]))
    return results


def _get_dep_supports(grid_env, pos):
    """Get support positions from dependent edges into pos."""
    supports = set()
    for u, v, d in grid_env.G.in_edges(pos, data=True):
        if 'dependent' in d:
            supports.add(d['dependent'])
    return supports


def _get_proposals(grid_env, segment_state, segment_support_robot,
                   segment_support_pos, parent_pos):
    """Get subgoal proposals with viable parent-edge costs.

    Tries standard propose first, then falls back to dependent-edge
    supports if needed. For each proposal, finds the best parent_support
    that makes the parent->subgoal edge viable.
    """
    # Collect all possible parent_support values
    parent_supports = []
    if segment_support_pos is not None:
        parent_supports.append(segment_support_pos)
    parent_supports.append(None)  # try independent path
    for dep_sp in _get_dep_supports(grid_env, parent_pos):
        if dep_sp not in parent_supports:
            parent_supports.append(dep_sp)

    # Get raw proposals from propose_subgoal_states
    raw_proposals = []
    tried_support_robots = set()

    # Standard approach
    sr_key = (segment_support_robot.position, segment_support_robot.color) \
        if segment_support_robot else None
    tried_support_robots.add(sr_key)
    try:
        subgoals = grid_env.propose_subgoal_states(
            segment_state, support_robot=segment_support_robot)
        raw_proposals.extend(subgoals)
    except Exception:
        pass

    # Fallback: try dep edge supports
    if not raw_proposals:
        for dep_sp in _get_dep_supports(grid_env, parent_pos):
            for helper in segment_state.helpers:
                sr = Robot_at(position=dep_sp, color=helper.color)
                sr_key = (sr.position, sr.color)
                if sr_key in tried_support_robots:
                    continue
                tried_support_robots.add(sr_key)
                try:
                    subgoals = grid_env.propose_subgoal_states(
                        segment_state, support_robot=sr)
                    raw_proposals.extend(subgoals)
                except Exception:
                    continue

    # For each raw proposal, find a viable parent_support
    proposals = []
    for sg, score in raw_proposals:
        bn_pos = sg.bottleneck.position
        for ps in parent_supports:
            cost = grid_env.compute_exact_shortest_path_length(
                bn_pos, parent_pos, ps)
            if cost is not None:
                proposals.append((sg, score + cost, ps))
                break

    return proposals
