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
# V4 dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SubgoalOption:
    """A subgoal option within a SegmentNode."""
    subgoal: object             # the Subgoal object
    parent_support_pos: object  # parent_support_pos for _apply_subgoal
    footprint: dict             # {robot_color: position}
    visit_count: int = 0
    total_cost: float = 0.0
    score: float = 0.0         # relaxed score from propose_subgoal_states

    @property
    def avg_cost(self) -> float:
        if self.visit_count == 0:
            return float("inf")
        return self.total_cost / self.visit_count

    def option_key(self) -> tuple:
        """Hashable key for this option."""
        sg = self.subgoal
        return (sg.bottleneck.position, sg.support.position,
                sg.helper.color, self.parent_support_pos)


@dataclass
class SegmentNode:
    """Shared segment node in the transposition table."""
    signature: tuple            # (robot_color, start_pos, goal_pos, context_sig)
    options: list[SubgoalOption] = field(default_factory=list)
    all_candidates: list | None = None  # lazy: None until populated
    visit_count: int = 0
    expanded_count: int = 0     # how many options have been expanded so far


@dataclass
class PlanNode:
    """A plan-level MCTS node."""
    plan: PartialPlan
    parent: PlanNode | None
    action: tuple | None
    children: list[PlanNode] = field(default_factory=list)
    visit_count: int = 0
    total_cost: float = 0.0
    # V4: segment assignments: edge_key -> segment signature
    segment_assignments: dict = field(default_factory=dict)
    # V4: footprint summary: {robot_color: position}
    footprint: dict = field(default_factory=dict)
    # V4: selected options per segment for this node's last expansion
    selected_options: dict = field(default_factory=dict)  # edge_key -> SubgoalOption

    @property
    def avg_cost(self) -> float:
        if self.visit_count == 0:
            return float("inf")
        return self.total_cost / self.visit_count


# ---------------------------------------------------------------------------
# V4 helpers
# ---------------------------------------------------------------------------

def _extract_footprint(subgoal) -> dict:
    """Extract footprint {robot_color: position} from a subgoal."""
    fp = {}
    # The support robot must be at its position
    fp[subgoal.support.color] = subgoal.support.position
    # The helper robot is used
    fp[subgoal.helper.color] = subgoal.helper.position
    return fp


def _compute_footprint_clash_penalty(fp1: dict, fp2: dict,
                                      grid_env: GridEnv) -> float:
    """Compute penalty for footprint clashes between two options."""
    penalty = 0.0
    for color in set(fp1) & set(fp2):
        p1, p2 = fp1[color], fp2[color]
        if p1 != p2:
            reloc = grid_env.compute_relaxed_shortest_path_length(p1, p2)
            if reloc is None:
                return float("inf")
            penalty += reloc
    return penalty


def _build_context_sig(plan: PartialPlan, parent_id: str, child_id: str,
                       footprint: dict, grid_env: GridEnv,
                       radius: int) -> frozenset:
    """Build context signature for a segment.

    Collects (robot_color, position) from the plan's footprint for robots
    relevant to this segment (within neighborhood radius of the segment's
    start/goal positions).
    """
    g = plan.g
    parent_pos = g.nodes[parent_id].get("pos")
    child_pos = _get_start_pos(g, child_id)

    if not footprint:
        return frozenset()

    # Use Manhattan distance to determine relevance
    ctx = set()
    for color, pos in footprint.items():
        relevant = False
        if parent_pos is not None:
            dist = abs(pos[0] - parent_pos[0]) + abs(pos[1] - parent_pos[1])
            if dist <= radius:
                relevant = True
        if child_pos is not None and not relevant:
            dist = abs(pos[0] - child_pos[0]) + abs(pos[1] - child_pos[1])
            if dist <= radius:
                relevant = True
        if relevant:
            ctx.add((color, pos))
    return frozenset(ctx)


def _segment_signature(plan: PartialPlan, parent_id: str, child_id: str,
                       state: State, footprint: dict, grid_env: GridEnv,
                       radius: int) -> tuple:
    """Build the full segment signature for transposition lookup."""
    g = plan.g
    parent_data = g.nodes[parent_id]
    parent_pos = parent_data.get("pos")

    # Determine the robot color for this segment
    # The child node's robot, or the target robot if child is the target leaf
    child_data = g.nodes[child_id]
    if child_data.get("robot") is not None:
        robot_color = child_data["robot"].color
    else:
        robot_color = state.target_robot.color

    child_pos = _get_start_pos(g, child_id)

    ctx_sig = _build_context_sig(plan, parent_id, child_id, footprint,
                                  grid_env, radius)

    return (robot_color, child_pos, parent_pos, ctx_sig)


# ---------------------------------------------------------------------------
# MCTS_V4 -- Segment-Node MCTS with Context-Signature Transpositions
# ---------------------------------------------------------------------------

class MCTS_V4(MCTS):

    def __init__(self, cfg=None):
        if cfg is None:
            cfg = OmegaConf.load(
                Path(__file__).resolve().parent.parent / "conf" / "v4.yaml")
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

        # V4: transposition table
        self._transposition: dict[tuple, SegmentNode] = {}
        # V4: pairwise penalty table
        self._pairwise_penalties: dict[tuple, tuple] = {}  # (key_i, key_j) -> (total_penalty, count)

        root_plan = _build_initial_plan(state, grid_env)
        root = PlanNode(plan=root_plan, parent=None, action=None)

        for _ in range(self.cfg.max_iterations):
            self._iteration += 1
            node = self._select(root)

            if node.plan.is_complete():
                cost = self._eval_complete(node.plan, grid_env)
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

    # -- Transposition table --

    def _get_or_create_segment_node(self, sig: tuple) -> SegmentNode:
        """Look up or create a segment node in the transposition table."""
        if sig not in self._transposition:
            self._transposition[sig] = SegmentNode(signature=sig)
        return self._transposition[sig]

    def _populate_segment_candidates(self, seg_node: SegmentNode,
                                      plan: PartialPlan, parent_id: str,
                                      state: State, grid_env: GridEnv):
        """Lazily populate segment node with all candidates."""
        if seg_node.all_candidates is not None:
            return
        candidates = _get_candidates(plan, parent_id, state, grid_env)
        candidates.sort(key=lambda x: x[1])
        seg_node.all_candidates = candidates

    def _expand_segment_options(self, seg_node: SegmentNode,
                                 plan: PartialPlan, parent_id: str,
                                 state: State, grid_env: GridEnv) -> int:
        """Progressive widening: expand options up to the allowed limit.

        Returns the number of options available.
        """
        self._populate_segment_candidates(seg_node, plan, parent_id,
                                           state, grid_env)
        if not seg_node.all_candidates:
            return 0

        # Progressive widening limit
        k = math.ceil(
            self.cfg.pw_C * (max(seg_node.visit_count, 1) ** self.cfg.pw_alpha))

        while seg_node.expanded_count < len(seg_node.all_candidates) and \
              len(seg_node.options) < k:
            subgoal, score, parent_sp = \
                seg_node.all_candidates[seg_node.expanded_count]
            seg_node.expanded_count += 1

            fp = _extract_footprint(subgoal)
            opt = SubgoalOption(
                subgoal=subgoal,
                parent_support_pos=parent_sp,
                footprint=fp,
                score=score,
            )
            seg_node.options.append(opt)

        return len(seg_node.options)

    # -- Selection (LCB for minimization) --

    def _select(self, node: PlanNode) -> PlanNode:
        while node.children:
            if self._can_expand(node):
                return node
            viable = [c for c in node.children
                      if c.plan.cost() <= self._best_cost]
            if not viable:
                return node
            node = min(viable, key=lambda c: self._lcb(c, node))
        return node

    def _lcb(self, child: PlanNode, parent: PlanNode) -> float:
        if child.visit_count == 0:
            return float("-inf")
        exploit = child.avg_cost
        explore = self.cfg.ucb_exploration * math.sqrt(
            math.log(parent.visit_count + 1) / child.visit_count)
        return exploit - explore

    def _can_expand(self, node: PlanNode) -> bool:
        k = math.ceil(
            self.cfg.pw_C * (max(node.visit_count, 1) ** self.cfg.pw_alpha))
        return len(node.children) < k

    # -- Segment-level UCB selection --

    def _select_segment_option(self, seg_node: SegmentNode,
                                current_footprint: dict,
                                grid_env: GridEnv) -> Optional[SubgoalOption]:
        """Select best option from segment node using UCB1 with clash penalty."""
        if not seg_node.options:
            return None

        best_opt = None
        best_score = float("inf")

        for opt in seg_node.options:
            if opt.visit_count == 0:
                return opt  # explore unvisited first

            exploit = opt.avg_cost
            explore = self.cfg.ucb_exploration * math.sqrt(
                math.log(seg_node.visit_count + 1) / opt.visit_count)

            # Add footprint clash penalty
            clash_penalty = _compute_footprint_clash_penalty(
                opt.footprint, current_footprint, grid_env)
            clash_penalty *= self.cfg.footprint_clash_penalty_weight

            # Also add pairwise penalty from table
            pairwise_pen = self._get_expected_pairwise_penalty(
                opt.option_key(), current_footprint)

            ucb = exploit - explore + clash_penalty + pairwise_pen
            if ucb < best_score:
                best_score = ucb
                best_opt = opt

        return best_opt

    def _get_expected_pairwise_penalty(self, opt_key: tuple,
                                        footprint: dict) -> float:
        """Look up accumulated pairwise penalty for this option."""
        # We don't have per-option keys from the footprint easily,
        # so this is a simplified version that returns 0
        # (the main clash penalty is computed directly)
        return 0.0

    # -- Expansion --

    def _expand(self, node: PlanNode, grid_env: GridEnv, state: State):
        plan = node.plan
        open_edges = plan.open_edges()
        if not open_edges:
            return None

        # Plan-level: choose which segment to refine (highest cost open edge)
        parent_id, child_id = max(
            open_edges, key=lambda e: plan.g.edges[e]["cost"] or 0)

        edge_key = (parent_id, child_id)

        # Get or create segment node
        radius = self.cfg.context_neighborhood_radius
        sig = _segment_signature(plan, parent_id, child_id, state,
                                  node.footprint, grid_env, radius)
        seg_node = self._get_or_create_segment_node(sig)

        # Progressive widening on segment node
        n_opts = self._expand_segment_options(
            seg_node, plan, parent_id, state, grid_env)
        if n_opts == 0:
            return None

        # Segment-level: select option via UCB, trying multiple on failure
        tried = {c.action for c in node.children}

        # Build ordered list of options to try: UCB-selected first, then rest
        ucb_opt = self._select_segment_option(
            seg_node, node.footprint, grid_env)
        options_to_try = []
        if ucb_opt is not None:
            options_to_try.append(ucb_opt)
        for alt_opt in seg_node.options:
            if alt_opt is not ucb_opt:
                options_to_try.append(alt_opt)

        opt = None
        action_key = None
        child_plan = None
        result = None
        for candidate_opt in options_to_try:
            ak = (parent_id, child_id) + candidate_opt.option_key()
            if ak in tried:
                continue
            cp = _copy_plan(plan)
            res = _apply_subgoal(
                cp, parent_id, child_id,
                candidate_opt.subgoal, grid_env, self._cnt,
                parent_support_pos=candidate_opt.parent_support_pos)
            if res is not None:
                opt = candidate_opt
                action_key = ak
                child_plan = cp
                result = res
                break

        if opt is None:
            return None

        # Close any directly closeable edges
        for pid, cid in list(child_plan.open_edges()):
            _try_close_edge(child_plan, pid, cid, grid_env)

        # Build child footprint (parent footprint + this option's footprint)
        child_footprint = dict(node.footprint)
        child_footprint.update(opt.footprint)

        # Build child segment assignments
        child_seg_assignments = dict(node.segment_assignments)
        child_seg_assignments[edge_key] = sig

        child_node = PlanNode(
            plan=child_plan, parent=node, action=action_key,
            segment_assignments=child_seg_assignments,
            footprint=child_footprint,
            selected_options={**node.selected_options, edge_key: opt},
        )
        self._node_count += 1
        node.children.append(child_node)
        self._check_complete(child_plan, grid_env)
        return child_node

    # -- Rollout (random greedy completion with segment reuse) --

    def _rollout(self, node: PlanNode, grid_env: GridEnv, state: State):
        self._rollout_count += 1
        plan = _copy_plan(node.plan)
        footprint = dict(node.footprint)
        selected_opts = dict(node.selected_options)  # edge_key -> SubgoalOption

        for _ in range(self.cfg.max_rollout_depth):
            # Close all directly closeable edges first
            for pid, cid in list(plan.open_edges()):
                _try_close_edge(plan, pid, cid, grid_env)

            open_edges = plan.open_edges()
            if not open_edges:
                break

            # Pick random open edge
            parent_id, child_id = random.choice(open_edges)
            edge_key = (parent_id, child_id)

            # Try to use segment node if available
            radius = self.cfg.context_neighborhood_radius
            sig = _segment_signature(plan, parent_id, child_id, state,
                                      footprint, grid_env, radius)
            seg_node = self._get_or_create_segment_node(sig)

            # Populate candidates lazily
            self._populate_segment_candidates(
                seg_node, plan, parent_id, state, grid_env)

            if seg_node.all_candidates:
                # Expand all available options for rollout
                while seg_node.expanded_count < len(seg_node.all_candidates):
                    subgoal, score, parent_sp = \
                        seg_node.all_candidates[seg_node.expanded_count]
                    seg_node.expanded_count += 1
                    fp = _extract_footprint(subgoal)
                    seg_node.options.append(SubgoalOption(
                        subgoal=subgoal, parent_support_pos=parent_sp,
                        footprint=fp, score=score))

            if seg_node.options:
                # Select from segment options: prefer low avg_cost, consider clash
                scored = []
                for opt in seg_node.options:
                    clash = _compute_footprint_clash_penalty(
                        opt.footprint, footprint, grid_env)
                    base = opt.avg_cost if opt.visit_count > 0 else opt.score
                    scored.append((base + clash * self.cfg.footprint_clash_penalty_weight, opt))
                scored.sort(key=lambda x: x[0])
                pick_from = max(1, len(scored) // 2)
                _, opt = random.choice(scored[:pick_from])

                subgoal = opt.subgoal
                parent_sp = opt.parent_support_pos
            else:
                # Fall back to direct candidate generation
                candidates = _get_candidates(plan, parent_id, state, grid_env)
                if not candidates:
                    return self.cfg.penalty_cost
                candidates.sort(key=lambda x: x[1])
                pick_from = max(1, len(candidates) // 2)
                subgoal, _, parent_sp = random.choice(candidates[:pick_from])
                opt = None

            result = _apply_subgoal(
                plan, parent_id, child_id,
                subgoal, grid_env, self._cnt,
                parent_support_pos=parent_sp)
            if result is None:
                continue

            # Update footprint
            if opt is not None:
                footprint.update(opt.footprint)
                selected_opts[edge_key] = opt
            else:
                fp = _extract_footprint(subgoal)
                footprint.update(fp)

        # Close remaining closeable
        for pid, cid in list(plan.open_edges()):
            _try_close_edge(plan, pid, cid, grid_env)

        # Compute total cost including clash penalties
        total_clash_penalty = 0.0
        opt_list = list(selected_opts.values())
        for i in range(len(opt_list)):
            for j in range(i + 1, len(opt_list)):
                pen = _compute_footprint_clash_penalty(
                    opt_list[i].footprint, opt_list[j].footprint, grid_env)
                total_clash_penalty += pen
                # Update pairwise penalty table
                pair_key = (opt_list[i].option_key(), opt_list[j].option_key())
                pair_key_r = (opt_list[j].option_key(), opt_list[i].option_key())
                if pair_key in self._pairwise_penalties:
                    old_total, old_count = self._pairwise_penalties[pair_key]
                    self._pairwise_penalties[pair_key] = (old_total + pen, old_count + 1)
                elif pair_key_r in self._pairwise_penalties:
                    old_total, old_count = self._pairwise_penalties[pair_key_r]
                    self._pairwise_penalties[pair_key_r] = (old_total + pen, old_count + 1)
                else:
                    self._pairwise_penalties[pair_key] = (pen, 1)

        if plan.is_complete():
            self._check_complete(plan, grid_env)
            cost = evaluate_plan(plan, grid_env)
            if cost is not None:
                # Two-level backup: update segment-level stats
                self._backup_segment_stats(selected_opts, cost)
                return cost

        base_cost = plan.cost()

        # Two-level backup: update segment-level stats
        self._backup_segment_stats(selected_opts, base_cost)

        return base_cost

    def _backup_segment_stats(self, selected_opts: dict, cost: float):
        """Update segment-level statistics for each selected option."""
        for edge_key, opt in selected_opts.items():
            opt.visit_count += 1
            opt.total_cost += cost

    # -- Backpropagation --

    def _backprop(self, node: PlanNode, cost: float):
        while node is not None:
            node.visit_count += 1
            node.total_cost += cost
            # Also update segment nodes' visit counts
            for edge_key, sig in node.segment_assignments.items():
                if sig in self._transposition:
                    self._transposition[sig].visit_count += 1
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

    def _fallback(self, root: PlanNode, grid_env: GridEnv, state: State):
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
