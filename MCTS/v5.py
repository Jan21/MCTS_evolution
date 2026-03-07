from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
from omegaconf import OmegaConf

from GridEnv import GridEnv, Robot_at, State
from partial_plan import PartialPlan
from evaluate_plan import evaluate_plan
from MCTS.base import MCTS, SolveResult, PlanEntry, PlanStats


# ---------------------------------------------------------------------------
# Helpers (copied verbatim from v1.py)
# ---------------------------------------------------------------------------

def _counter():
    n = 0
    while True:
        yield n
        n += 1


def _copy_plan(plan: PartialPlan) -> PartialPlan:
    new = PartialPlan()
    new.g = plan.g.copy()
    return new


def _find_sibling_support_pos(plan: PartialPlan, bottleneck_id: str):
    g = plan.g
    for parent in g.predecessors(bottleneck_id):
        if g.nodes[parent].get("ntype") == "subgoal":
            for child in g.successors(parent):
                if g.nodes[child].get("ntype") == "support":
                    return g.nodes[child].get("pos")
    return None


def _find_sibling_support_robot(plan: PartialPlan, bottleneck_id: str):
    g = plan.g
    for parent in g.predecessors(bottleneck_id):
        if g.nodes[parent].get("ntype") == "subgoal":
            for child in g.successors(parent):
                if g.nodes[child].get("ntype") == "support":
                    return g.nodes[child].get("robot")
    return None


def _get_start_pos(g, node_id: str):
    ntype = g.nodes[node_id]["ntype"]
    if ntype == "subgoal":
        for c in g.successors(node_id):
            if g.nodes[c]["ntype"] == "bottleneck":
                return g.nodes[c]["pos"]
    return g.nodes[node_id].get("pos")


def _build_initial_plan(state: State, grid_env: GridEnv) -> PartialPlan:
    plan = PartialPlan()
    plan.add_node("goal", "goal", pos=state.target)
    plan.add_node("leaf_target", "leaf",
                  pos=state.target_robot.position,
                  robot=state.target_robot)
    relaxed_cost = grid_env.compute_relaxed_shortest_path_length(
        state.target_robot.position, state.target)
    plan.add_edge("goal", "leaf_target", status="open", cost=relaxed_cost)
    return plan


def _build_segment_states(plan: PartialPlan, parent_id: str, state: State,
                          grid_env: GridEnv):
    """Return list of (segment_state, support_robot) for propose_subgoal_states.

    Usually returns a single pair, but for goal nodes with only dependent edges,
    returns one pair per possible support position.
    """
    g = plan.g
    parent_data = g.nodes[parent_id]
    parent_type = parent_data["ntype"]

    if parent_type == "goal":
        goal_pos = parent_data["pos"]
        results = []
        # Try independent path (no support needed)
        if (goal_pos, None) in grid_env._bottleneck_support_pairs_cache:
            results.append((State(target=goal_pos,
                                  target_robot=state.target_robot,
                                  helpers=state.helpers), None))
        # Also enumerate dependent support positions
        seen_support = set()
        for _, _, d in grid_env.G.in_edges(goal_pos, data=True):
            if "dependent" in d:
                sp = d["dependent"]
                if sp in seen_support:
                    continue
                seen_support.add(sp)
                for helper in state.helpers:
                    sr = Robot_at(position=sp, color=helper.color)
                    results.append((State(target=goal_pos,
                                         target_robot=state.target_robot,
                                         helpers=state.helpers), sr))
        return results

    if parent_type == "bottleneck":
        sp_robot = _find_sibling_support_robot(plan, parent_id)
        return [(State(target=parent_data["pos"],
                       target_robot=parent_data["robot"],
                       helpers=state.helpers), sp_robot)]

    if parent_type == "support":
        helper = parent_data["robot"]
        remaining = [h for h in state.helpers if h.color != helper.color]
        sp_pos = parent_data["pos"]
        results = []
        if (sp_pos, None) in grid_env._bottleneck_support_pairs_cache:
            results.append((State(target=sp_pos,
                                  target_robot=helper,
                                  helpers=remaining), None))
        # Also enumerate dependent support positions
        seen_support = set()
        for _, _, d in grid_env.G.in_edges(sp_pos, data=True):
            if "dependent" in d:
                dep_sp = d["dependent"]
                if dep_sp in seen_support:
                    continue
                seen_support.add(dep_sp)
                for h in remaining:
                    sr = Robot_at(position=dep_sp, color=h.color)
                    results.append((State(target=sp_pos,
                                         target_robot=helper,
                                         helpers=remaining), sr))
        return results

    return []


def _get_ancestor_positions(plan, node_id):
    """Collect positions of all ancestor nodes in the plan DAG."""
    g = plan.g
    positions = set()
    for anc in nx.ancestors(g, node_id):
        pos = g.nodes[anc].get("pos")
        if pos is not None:
            positions.add(pos)
    pos = g.nodes[node_id].get("pos")
    if pos is not None:
        positions.add(pos)
    return positions


def _get_candidates(plan, parent_id, state, grid_env):
    """Collect all (subgoal, score, parent_support_pos) candidates.

    parent_support_pos is the support position needed for the bn->parent edge
    (comes from the support_robot passed to propose_subgoal_states).
    For parents that don't need dependent edges, parent_support_pos is None.
    Filters out candidates whose bottleneck would create a cycle.
    """
    pairs = _build_segment_states(plan, parent_id, state, grid_env)
    if not pairs:
        return []
    ancestor_pos = _get_ancestor_positions(plan, parent_id)
    all_candidates = []
    seen = set()
    for seg_state, support_robot in pairs:
        cache_key = (seg_state.target,
                     support_robot.position if support_robot else None)
        if cache_key not in grid_env._bottleneck_support_pairs_cache:
            continue
        try:
            candidates = grid_env.propose_subgoal_states(
                seg_state, support_robot)
        except (TypeError, AssertionError):
            continue
        parent_sp = support_robot.position if support_robot else None
        for subgoal, score in candidates:
            if score is not None:
                bn_pos = subgoal.bottleneck.position
                if bn_pos in ancestor_pos:
                    continue
                key = (bn_pos,
                       subgoal.support.position,
                       subgoal.helper.color,
                       parent_sp)
                if key not in seen:
                    seen.add(key)
                    all_candidates.append((subgoal, score, parent_sp))
    return all_candidates


def _apply_subgoal(plan, parent_id, child_id, subgoal, grid_env, cnt,
                   parent_support_pos=None):
    """Apply a subgoal to refine the open edge (parent_id -> child_id).

    parent_support_pos: the support position needed for the bn->parent
        dependent edge. For bottleneck parents this comes from their sibling
        support; for goal/support parents with only dependent in-edges this
        comes from the support_robot used during candidate generation.
        None when the parent is reachable via independent edges.

    Returns new node IDs tuple or None if invalid.
    """
    g = plan.g
    parent_data = g.nodes[parent_id]
    n = next(cnt)

    sg_id = f"sg_{n}"
    bn_id = f"bn_{n}"
    sp_id = f"sp_{n}"
    leaf_id = f"leaf_{subgoal.helper.color}_{n}"

    exact_cost = grid_env.compute_exact_shortest_path_length(
        subgoal.bottleneck.position, parent_data["pos"], parent_support_pos)
    if exact_cost is None:
        return None

    # Remove old open edge
    g.remove_edge(parent_id, child_id)

    # Add nodes (parent_support_pos stored on subgoal for evaluate_plan)
    plan.add_node(sg_id, "subgoal", parent_support_pos=parent_support_pos)
    plan.add_node(bn_id, "bottleneck",
                  pos=subgoal.bottleneck.position,
                  robot=subgoal.bottleneck)
    plan.add_node(sp_id, "support",
                  pos=subgoal.support.position,
                  robot=subgoal.support)
    plan.add_node(leaf_id, "leaf",
                  pos=subgoal.helper.position,
                  robot=subgoal.helper)

    # parent -> subgoal (fixed)
    plan.add_edge(parent_id, sg_id, status="fixed", cost=exact_cost)

    # structural edges
    plan.add_edge(sg_id, bn_id, status="fixed", cost=None)
    plan.add_edge(sg_id, sp_id, status="fixed", cost=None)

    # bottleneck -> old child (open)
    child_pos = _get_start_pos(g, child_id)
    relaxed_bn = grid_env.compute_relaxed_shortest_path_length(
        child_pos, subgoal.bottleneck.position, subgoal.support.position)
    plan.add_edge(bn_id, child_id, status="open",
                  cost=relaxed_bn if relaxed_bn is not None else 999)

    # support -> helper leaf (open)
    relaxed_sp = grid_env.compute_relaxed_shortest_path_length(
        subgoal.helper.position, subgoal.support.position)
    plan.add_edge(sp_id, leaf_id, status="open",
                  cost=relaxed_sp if relaxed_sp is not None else 999)

    return (sg_id, bn_id, sp_id, leaf_id)


def _try_close_edge(plan, parent_id, child_id, grid_env):
    """Try to close an open edge directly (without subgoal insertion).

    For bottleneck parents, also tries the dependent path through the
    sibling support position.

    Returns True if the edge was closed, False otherwise.
    """
    g = plan.g
    parent_data = g.nodes[parent_id]
    child_pos = _get_start_pos(g, child_id)
    parent_pos = parent_data["pos"]

    # Try independent path first
    exact = grid_env.compute_exact_shortest_path_length(
        child_pos, parent_pos, None)
    if exact is not None:
        g.edges[parent_id, child_id]["status"] = "fixed"
        g.edges[parent_id, child_id]["cost"] = exact
        return True

    # For bottleneck parents, try dependent path through sibling support
    if parent_data["ntype"] == "bottleneck":
        sp_pos = _find_sibling_support_pos(plan, parent_id)
        if sp_pos is not None:
            exact = grid_env.compute_exact_shortest_path_length(
                child_pos, parent_pos, sp_pos)
            if exact is not None:
                g.edges[parent_id, child_id]["status"] = "fixed"
                g.edges[parent_id, child_id]["cost"] = exact
                return True

    return False


# ---------------------------------------------------------------------------
# V5 dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OuterNode:
    """Outer MCTS tree node -- holds a (partial) skeleton."""
    plan: PartialPlan
    parent: OuterNode | None
    action: tuple | None
    children: list[OuterNode] = field(default_factory=list)
    visit_count: int = 0
    total_cost: float = 0.0
    # V5: skeleton signature for transposition
    skeleton_sig: frozenset = field(default_factory=frozenset)
    # V5: conflict constraints learned from inner evaluations
    conflict_sets: list[frozenset] = field(default_factory=list)
    # V5: inner search results cache
    inner_results: list[tuple] = field(default_factory=list)  # [(cost_or_None, conflict_set)]

    @property
    def avg_cost(self) -> float:
        if self.visit_count == 0:
            return float("inf")
        return self.total_cost / self.visit_count


def _subgoal_assignment_sig(subgoal, parent_sp) -> tuple:
    """Create a hashable signature for a subgoal assignment."""
    return (subgoal.bottleneck.position,
            subgoal.support.position,
            subgoal.helper.color,
            parent_sp)


def _extract_skeleton_sig(plan: PartialPlan) -> frozenset:
    """Extract a skeleton signature: frozenset of subgoal assignment tuples.

    Each subgoal node contributes (bottleneck_pos, support_pos, helper_color).
    """
    g = plan.g
    sigs = set()
    for nid, data in g.nodes(data=True):
        if data.get("ntype") == "subgoal":
            # Find bottleneck and support children
            bn_pos = None
            sp_pos = None
            helper_color = None
            parent_sp = data.get("parent_support_pos")
            for child in g.successors(nid):
                cdata = g.nodes[child]
                if cdata.get("ntype") == "bottleneck":
                    bn_pos = cdata.get("pos")
                elif cdata.get("ntype") == "support":
                    sp_pos = cdata.get("pos")
                    robot = cdata.get("robot")
                    if robot is not None:
                        helper_color = robot.color
            if bn_pos is not None:
                sigs.add((bn_pos, sp_pos, helper_color, parent_sp))
    return frozenset(sigs)


# ---------------------------------------------------------------------------
# Inner search: realize a skeleton
# ---------------------------------------------------------------------------

def _realize_skeleton(plan: PartialPlan, grid_env: GridEnv,
                      spl_cache: dict | None = None,
                      best_known_cost: float = float("inf"),
                      max_repair_attempts: int = 3) -> tuple:
    """Attempt to realize a skeleton by closing all open edges.

    Returns (plan, cost_or_None, conflict_frozenset).
    - If all edges close: cost is the evaluated plan cost, conflict_set is empty.
    - If some edges fail: cost is None, conflict_set contains the responsible
      subgoal assignments.
    """
    plan = _copy_plan(plan)
    g = plan.g

    # Process open edges in topological order (leaves toward goal)
    # We need reverse topological order of the DAG for leaves-first
    try:
        topo_order = list(nx.topological_sort(g))
    except nx.NetworkXUnfeasible:
        return (plan, None, frozenset())

    # Reverse: leaves first
    topo_order.reverse()

    # Build a mapping from node to the subgoal assignment it belongs to
    node_to_sg_assignment = {}
    for nid, data in g.nodes(data=True):
        if data.get("ntype") == "subgoal":
            bn_pos = None
            sp_pos = None
            helper_color = None
            parent_sp = data.get("parent_support_pos")
            for child in g.successors(nid):
                cdata = g.nodes[child]
                if cdata.get("ntype") == "bottleneck":
                    bn_pos = cdata.get("pos")
                elif cdata.get("ntype") == "support":
                    sp_pos = cdata.get("pos")
                    robot = cdata.get("robot")
                    if robot is not None:
                        helper_color = robot.color
            sg_sig = (bn_pos, sp_pos, helper_color, parent_sp)
            # Map subgoal and its children to this assignment
            node_to_sg_assignment[nid] = sg_sig
            for child in g.successors(nid):
                node_to_sg_assignment[child] = sg_sig

    # Also map leaf nodes that are children of bottleneck/support to their
    # parent's subgoal assignment
    for nid, data in g.nodes(data=True):
        if data.get("ntype") in ("bottleneck", "support"):
            if nid in node_to_sg_assignment:
                for child in g.successors(nid):
                    if child not in node_to_sg_assignment:
                        node_to_sg_assignment[child] = node_to_sg_assignment[nid]

    running_cost = 0.0
    conflict_assignments = set()

    # Process edges in topological order (child first)
    for node_id in topo_order:
        for parent_id in list(g.predecessors(node_id)):
            edge_data = g.edges[parent_id, node_id]
            if edge_data["status"] != "open":
                continue

            # Try to close this edge
            closed = _try_close_edge_cached(
                plan, parent_id, node_id, grid_env, spl_cache,
                max_repair_attempts)

            if closed:
                edge_cost = g.edges[parent_id, node_id].get("cost", 0)
                if edge_cost is not None:
                    running_cost += edge_cost
                # Inner pruning: abort if exceeding best known
                if running_cost >= best_known_cost:
                    return (plan, None, frozenset())
            else:
                # Record conflict: which subgoal assignments are responsible
                for nid in (parent_id, node_id):
                    if nid in node_to_sg_assignment:
                        conflict_assignments.add(node_to_sg_assignment[nid])
                # Abort on first unrealizable edge
                return (plan, None, frozenset(conflict_assignments))

    # All edges closed -- evaluate
    if plan.is_complete():
        cost = evaluate_plan(plan, grid_env)
        return (plan, cost, frozenset())
    else:
        return (plan, None, frozenset())


def _try_close_edge_cached(plan, parent_id, child_id, grid_env,
                           spl_cache: dict | None = None,
                           max_repair_attempts: int = 3) -> bool:
    """Try to close an open edge, using cache for SPL lookups.

    Also attempts limited local repair (trying alternative support positions).
    """
    g = plan.g
    parent_data = g.nodes[parent_id]
    child_pos = _get_start_pos(g, child_id)
    parent_pos = parent_data["pos"]

    # Try independent path first
    exact = _cached_spl(grid_env, child_pos, parent_pos, None, spl_cache)
    if exact is not None:
        g.edges[parent_id, child_id]["status"] = "fixed"
        g.edges[parent_id, child_id]["cost"] = exact
        return True

    # For bottleneck parents, try dependent path through sibling support
    if parent_data["ntype"] == "bottleneck":
        sp_pos = _find_sibling_support_pos(plan, parent_id)
        if sp_pos is not None:
            exact = _cached_spl(grid_env, child_pos, parent_pos, sp_pos,
                                spl_cache)
            if exact is not None:
                g.edges[parent_id, child_id]["status"] = "fixed"
                g.edges[parent_id, child_id]["cost"] = exact
                return True

    return False


def _cached_spl(grid_env, start, end, support_pos, cache: dict | None) -> int | None:
    """Compute exact shortest path length with optional caching."""
    if cache is not None:
        key = (start, end, support_pos)
        if key in cache:
            return cache[key]
        result = grid_env.compute_exact_shortest_path_length(
            start, end, support_pos)
        cache[key] = result
        return result
    return grid_env.compute_exact_shortest_path_length(start, end, support_pos)


# ---------------------------------------------------------------------------
# MCTS_V5 -- Nested Two-Stage MCTS
# ---------------------------------------------------------------------------

class MCTS_V5(MCTS):

    def __init__(self, cfg=None):
        if cfg is None:
            cfg = OmegaConf.load(
                Path(__file__).resolve().parent.parent / "conf" / "v5.yaml")
        self.cfg = cfg

    def solve(self, grid_env: GridEnv, state: State) -> SolveResult:
        self._cnt = _counter()
        self._best_cost = float("inf")
        self._best_plan = None
        self._start_time = time.monotonic()
        self._all_plans = []
        self._iteration = 0
        self._node_count = 0
        self._rollout_count = 0

        # V5: global conflict library
        self._conflict_sets: list[frozenset] = []
        # V5: conflict count per assignment tuple
        self._conflict_counts: dict[frozenset, int] = {}
        # V5: inner SPL cache (shared across all inner evaluations)
        self._spl_cache: dict = {} if self.cfg.inner_spl_cache else None
        # V5: skeleton signature -> best inner result cache
        self._skeleton_cache: dict[frozenset, tuple] = {}

        root_plan = _build_initial_plan(state, grid_env)
        root = OuterNode(plan=root_plan, parent=None, action=None)

        for _ in range(self.cfg.max_iterations):
            self._iteration += 1
            node = self._select(root)

            if node.plan.is_complete():
                cost = self._eval_via_inner(node, grid_env)
                self._backprop(node, cost)
                continue

            child = self._expand(node, grid_env, state)
            if child is None:
                self._backprop(node, self.cfg.penalty_cost)
                continue

            cost = self._rollout(child, grid_env, state)
            self._backprop(child, cost)

        best = (self._best_plan if self._best_plan is not None
                else self._fallback(root, grid_env, state))
        return SolveResult(best_plan=best, all_plans=self._all_plans)

    # -- Selection (LCB for minimization) --

    def _select(self, node: OuterNode) -> OuterNode:
        while node.children:
            if self._can_expand(node):
                return node
            viable = [c for c in node.children
                      if c.plan.cost() <= self._best_cost]
            if not viable:
                return node
            node = min(viable, key=lambda c: self._lcb(c, node))
        return node

    def _lcb(self, child: OuterNode, parent: OuterNode) -> float:
        if child.visit_count == 0:
            return float("-inf")
        exploit = child.avg_cost
        explore = self.cfg.ucb_exploration * math.sqrt(
            math.log(parent.visit_count + 1) / child.visit_count)
        return exploit - explore

    def _can_expand(self, node: OuterNode) -> bool:
        k = math.ceil(
            self.cfg.pw_C * (max(node.visit_count, 1) ** self.cfg.pw_alpha))
        return len(node.children) < k

    # -- Conflict penalty --

    def _conflict_penalty(self, existing_sig: frozenset,
                          new_assignment: tuple) -> float:
        """Compute penalty for adding new_assignment given existing skeleton sig.

        Check if any known conflict set is a subset of existing_sig + new_assignment.
        """
        candidate_sig = existing_sig | {new_assignment}
        penalty = 0.0
        for cs in self._conflict_sets:
            if cs <= candidate_sig:
                count = self._conflict_counts.get(cs, 1)
                penalty += self.cfg.conflict_penalty_weight * count
        return penalty

    # -- Expansion --

    def _expand(self, node: OuterNode, grid_env: GridEnv, state: State):
        plan = node.plan
        open_edges = plan.open_edges()
        if not open_edges:
            return None

        parent_id, child_id = max(
            open_edges, key=lambda e: plan.g.edges[e]["cost"] or 0)

        candidates = _get_candidates(plan, parent_id, state, grid_env)
        if not candidates:
            return None

        # Sort by score + conflict penalty
        def candidate_key(c):
            subgoal, score, parent_sp = c
            assignment = _subgoal_assignment_sig(subgoal, parent_sp)
            penalty = self._conflict_penalty(node.skeleton_sig, assignment)
            return score + penalty

        candidates.sort(key=candidate_key)

        tried = {c.action for c in node.children}
        for subgoal, _score, parent_sp in candidates:
            action_key = (parent_id, child_id,
                          subgoal.bottleneck.position,
                          subgoal.support.position,
                          subgoal.helper.color,
                          parent_sp)
            if action_key in tried:
                continue

            child_plan = _copy_plan(plan)
            result = _apply_subgoal(
                child_plan, parent_id, child_id,
                subgoal, grid_env, self._cnt,
                parent_support_pos=parent_sp)
            if result is None:
                continue

            # Close any directly closeable edges
            for pid, cid in list(child_plan.open_edges()):
                _try_close_edge(child_plan, pid, cid, grid_env)

            # Skeleton pruning
            if self.cfg.skeleton_pruning:
                if child_plan.cost() > self._best_cost:
                    continue

            # Build child skeleton signature
            assignment = _subgoal_assignment_sig(subgoal, parent_sp)
            child_sig = node.skeleton_sig | {assignment}

            child_node = OuterNode(
                plan=child_plan, parent=node, action=action_key,
                skeleton_sig=child_sig)
            self._node_count += 1
            node.children.append(child_node)
            self._check_complete(child_plan, grid_env)
            return child_node

        return None

    # -- Inner evaluation --

    def _eval_via_inner(self, node: OuterNode, grid_env: GridEnv) -> float:
        """Run inner search on a complete skeleton and return cost."""
        plan = node.plan

        # Check skeleton cache
        sig = node.skeleton_sig
        if sig in self._skeleton_cache:
            cached_cost, cached_conflict = self._skeleton_cache[sig]
            if cached_cost is not None:
                return cached_cost
            return self.cfg.penalty_cost

        realized_plan, cost, conflict_set = _realize_skeleton(
            plan, grid_env,
            spl_cache=self._spl_cache,
            best_known_cost=self._best_cost,
            max_repair_attempts=self.cfg.inner_max_repair_attempts)

        # Cache result
        self._skeleton_cache[sig] = (cost, conflict_set)

        # Record inner result on node
        node.inner_results.append((cost, conflict_set))

        if conflict_set:
            self._record_conflict(conflict_set, node)

        if cost is not None:
            if cost < self._best_cost:
                self._best_cost = cost
                self._best_plan = _copy_plan(realized_plan)
            self._all_plans.append(PlanEntry(
                plan=_copy_plan(realized_plan),
                cost=cost,
                stats=PlanStats(
                    wall_time=time.monotonic() - self._start_time,
                    iteration=self._iteration,
                    node_count=self._node_count,
                    rollout_count=self._rollout_count,
                )
            ))
            return cost

        return self.cfg.penalty_cost

    def _record_conflict(self, conflict_set: frozenset, node: OuterNode):
        """Record a conflict set globally and on ancestor nodes."""
        if not conflict_set:
            return
        self._conflict_sets.append(conflict_set)
        self._conflict_counts[conflict_set] = \
            self._conflict_counts.get(conflict_set, 0) + 1
        # Propagate to ancestors
        current = node
        while current is not None:
            current.conflict_sets.append(conflict_set)
            current = current.parent

    # -- Rollout (greedy skeleton completion + inner evaluation) --

    def _rollout(self, node: OuterNode, grid_env: GridEnv, state: State) -> float:
        self._rollout_count += 1
        plan = _copy_plan(node.plan)
        skeleton_sig = set(node.skeleton_sig)

        for _ in range(self.cfg.max_rollout_depth):
            # Close all directly closeable edges first
            for pid, cid in list(plan.open_edges()):
                _try_close_edge(plan, pid, cid, grid_env)

            open_edges = plan.open_edges()
            if not open_edges:
                break

            # Pick highest-cost open edge (greedy)
            parent_id, child_id = max(
                open_edges, key=lambda e: plan.g.edges[e]["cost"] or 0)

            candidates = _get_candidates(plan, parent_id, state, grid_env)
            if not candidates:
                return self.cfg.penalty_cost

            # Sort by score + conflict penalty, pick from top candidates
            def candidate_key(c):
                subgoal, score, parent_sp = c
                assignment = _subgoal_assignment_sig(subgoal, parent_sp)
                candidate_sig = frozenset(skeleton_sig | {assignment})
                penalty = 0.0
                for cs in self._conflict_sets:
                    if cs <= candidate_sig:
                        count = self._conflict_counts.get(cs, 1)
                        penalty += self.cfg.conflict_penalty_weight * count
                return score + penalty

            candidates.sort(key=candidate_key)
            pick_from = max(1, len(candidates) // 2)
            subgoal, _, parent_sp = random.choice(candidates[:pick_from])

            result = _apply_subgoal(
                plan, parent_id, child_id,
                subgoal, grid_env, self._cnt,
                parent_support_pos=parent_sp)
            if result is None:
                continue

            assignment = _subgoal_assignment_sig(subgoal, parent_sp)
            skeleton_sig.add(assignment)

        # Close remaining closeable
        for pid, cid in list(plan.open_edges()):
            _try_close_edge(plan, pid, cid, grid_env)

        if plan.is_complete():
            # Inner evaluation on the completed skeleton
            sig_frozen = frozenset(skeleton_sig)

            # Check skeleton cache
            if sig_frozen in self._skeleton_cache:
                cached_cost, cached_conflict = self._skeleton_cache[sig_frozen]
                if cached_cost is not None:
                    return cached_cost
                return self.cfg.penalty_cost

            realized_plan, cost, conflict_set = _realize_skeleton(
                plan, grid_env,
                spl_cache=self._spl_cache,
                best_known_cost=self._best_cost,
                max_repair_attempts=self.cfg.inner_max_repair_attempts)

            self._skeleton_cache[sig_frozen] = (cost, conflict_set)

            if conflict_set:
                self._record_conflict(conflict_set, node)

            if cost is not None:
                if cost < self._best_cost:
                    self._best_cost = cost
                    self._best_plan = _copy_plan(realized_plan)
                self._all_plans.append(PlanEntry(
                    plan=_copy_plan(realized_plan),
                    cost=cost,
                    stats=PlanStats(
                        wall_time=time.monotonic() - self._start_time,
                        iteration=self._iteration,
                        node_count=self._node_count,
                        rollout_count=self._rollout_count,
                    )
                ))
                return cost

        return plan.cost()

    # -- Backpropagation --

    def _backprop(self, node: OuterNode, cost: float):
        while node is not None:
            node.visit_count += 1
            node.total_cost += cost
            node = node.parent

    # -- Helpers --

    def _eval_complete(self, plan: PartialPlan, grid_env: GridEnv) -> float:
        cost = evaluate_plan(plan, grid_env)
        if cost is not None:
            if cost < self._best_cost:
                self._best_cost = cost
                self._best_plan = _copy_plan(plan)
            self._all_plans.append(PlanEntry(
                plan=_copy_plan(plan),
                cost=cost,
                stats=PlanStats(
                    wall_time=time.monotonic() - self._start_time,
                    iteration=self._iteration,
                    node_count=self._node_count,
                    rollout_count=self._rollout_count,
                )
            ))
            return cost
        return self.cfg.penalty_cost

    def _check_complete(self, plan: PartialPlan, grid_env: GridEnv):
        if plan.is_complete():
            self._eval_complete(plan, grid_env)

    def _fallback(self, root: OuterNode, grid_env: GridEnv, state: State):
        best_node = root
        stack = [root]
        while stack:
            n = stack.pop()
            if n.visit_count > 0 and n.avg_cost < best_node.avg_cost:
                best_node = n
            stack.extend(n.children)

        plan = _copy_plan(best_node.plan)
        for _ in range(self.cfg.max_rollout_depth):
            open_edges = plan.open_edges()
            if not open_edges:
                break
            parent_id, child_id = max(
                open_edges, key=lambda e: plan.g.edges[e]["cost"] or 0)
            if _try_close_edge(plan, parent_id, child_id, grid_env):
                continue
            candidates = _get_candidates(plan, parent_id, state, grid_env)
            if not candidates:
                break
            best_sub, _, best_sp = min(candidates, key=lambda x: x[1])
            result = _apply_subgoal(
                plan, parent_id, child_id,
                best_sub, grid_env, self._cnt,
                parent_support_pos=best_sp)
            if result is None:
                break

        # Close any remaining closeable edges
        for parent_id, child_id in list(plan.open_edges()):
            _try_close_edge(plan, parent_id, child_id, grid_env)

        return plan
