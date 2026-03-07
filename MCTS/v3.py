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
# V3 Constraint dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockReq:
    """Robot must be at position to serve as blocker for a segment."""
    robot_color: str
    position: tuple
    segment_id: str
    decision_id: int


@dataclass(frozen=True)
class ProtectReq:
    """Robot must not be moved."""
    robot_color: str
    position: tuple
    segment_id: str
    decision_id: int


@dataclass
class SupportReq:
    """Segment needs one of several (robot_color, position) options."""
    segment_id: str
    choices: list  # list of (robot_color, position) tuples
    chosen: Optional[tuple]  # (robot_color, position) once committed
    decision_id: int

    def committed_key(self):
        """Return a hashable summary if committed."""
        if self.chosen is not None:
            return (self.chosen[0], self.chosen[1], "support")
        return None


@dataclass(frozen=True)
class NoGood:
    """A no-good entry: in a context matching ctx_sig, action is forbidden."""
    ctx_sig: frozenset  # frozenset of (robot_color, position, commitment_type)
    action: tuple       # the forbidden action key


# ---------------------------------------------------------------------------
# V3 Constraint helpers
# ---------------------------------------------------------------------------

def _build_context_signature(commitments):
    """Build a hashable context signature from a list of commitments.

    Returns frozenset of (robot_color, position, commitment_type) tuples.
    """
    sig = set()
    for c in commitments:
        if isinstance(c, BlockReq):
            sig.add((c.robot_color, c.position, "block"))
        elif isinstance(c, ProtectReq):
            sig.add((c.robot_color, c.position, "protect"))
        elif isinstance(c, SupportReq) and c.chosen is not None:
            sig.add((c.chosen[0], c.chosen[1], "support"))
    return frozenset(sig)


def _detect_conflicts(commitments):
    """Check commitments for conflicts.

    Returns (conflict_found, conflict_set) where conflict_set is a list
    of the conflicting commitments, or (False, []) if no conflict.
    """
    # Collect position requirements per robot color
    robot_positions = {}  # color -> list of (position, commitment)
    for c in commitments:
        if isinstance(c, BlockReq):
            robot_positions.setdefault(c.robot_color, []).append(
                (c.position, c))
        elif isinstance(c, ProtectReq):
            robot_positions.setdefault(c.robot_color, []).append(
                (c.position, c))
        elif isinstance(c, SupportReq) and c.chosen is not None:
            robot_positions.setdefault(c.chosen[0], []).append(
                (c.chosen[1], c))

    # Check for position conflicts: same robot at different positions
    for color, entries in robot_positions.items():
        positions_seen = {}
        for pos, commit in entries:
            if pos in positions_seen:
                # Same position is fine; skip
                continue
            for other_pos, other_commit in positions_seen.items():
                if other_pos != pos:
                    return True, [commit, other_commit]
            positions_seen[pos] = commit

    # Check SupportReq exhaustion: all choices eliminated by existing blocks
    blocked = {}  # color -> set of occupied positions
    for c in commitments:
        if isinstance(c, BlockReq):
            blocked.setdefault(c.robot_color, set()).add(c.position)
        elif isinstance(c, ProtectReq):
            blocked.setdefault(c.robot_color, set()).add(c.position)

    for c in commitments:
        if isinstance(c, SupportReq) and c.chosen is None:
            valid_choices = []
            for robot_color, pos in c.choices:
                # A choice is invalid if that robot is already committed elsewhere
                robot_blocks = blocked.get(robot_color, set())
                if pos not in robot_blocks and len(robot_blocks) == 0:
                    valid_choices.append((robot_color, pos))
                elif pos in robot_blocks:
                    # Robot is already at this position as a block -- compatible
                    valid_choices.append((robot_color, pos))
                else:
                    # Robot is committed to a different position
                    pass
            # Actually, simpler check: any choice where the robot isn't
            # required elsewhere at a different position
            feasible = []
            for robot_color, pos in c.choices:
                conflict = False
                for c2 in commitments:
                    if isinstance(c2, (BlockReq, ProtectReq)):
                        if c2.robot_color == robot_color and c2.position != pos:
                            conflict = True
                            break
                    elif isinstance(c2, SupportReq) and c2.chosen is not None:
                        if c2.chosen[0] == robot_color and c2.chosen[1] != pos:
                            conflict = True
                            break
                if not conflict:
                    feasible.append((robot_color, pos))
            if not feasible:
                # All choices exhausted -- find the blocking commitments
                blockers = []
                for robot_color, pos in c.choices:
                    for c2 in commitments:
                        if isinstance(c2, (BlockReq, ProtectReq)):
                            if c2.robot_color == robot_color and c2.position != pos:
                                blockers.append(c2)
                                break
                        elif isinstance(c2, SupportReq) and c2.chosen is not None:
                            if c2.chosen[0] == robot_color and c2.chosen[1] != pos:
                                blockers.append(c2)
                                break
                return True, [c] + blockers[:1]

    return False, []


def _is_nogood_match(nogood_ctx, current_ctx):
    """A no-good matches if current context is a superset of the no-good context."""
    return nogood_ctx.issubset(current_ctx)


def _commitments_for_subgoal(subgoal, segment_id, decision_id, state):
    """Create commitments from applying a subgoal.

    Returns list of commitment objects.
    """
    commits = []
    # BlockReq: support robot must be at its position
    commits.append(BlockReq(
        robot_color=subgoal.support.color,
        position=subgoal.support.position,
        segment_id=segment_id,
        decision_id=decision_id,
    ))

    # SupportReq: helper robot choices (lazy -- all available helpers)
    helper_choices = [(subgoal.helper.color, subgoal.helper.position)]
    # We create a committed SupportReq since the candidate already
    # specifies a concrete helper
    commits.append(SupportReq(
        segment_id=segment_id,
        choices=helper_choices,
        chosen=(subgoal.helper.color, subgoal.helper.position),
        decision_id=decision_id,
    ))

    return commits


# ---------------------------------------------------------------------------
# V3 MCTS tree node
# ---------------------------------------------------------------------------

@dataclass
class MCTSNode:
    plan: PartialPlan
    parent: MCTSNode | None
    action: tuple | None
    children: list[MCTSNode] = field(default_factory=list)
    visit_count: int = 0
    total_cost: float = 0.0
    # V3: constraint store
    commitments: list = field(default_factory=list)
    # V3: decision history -- list of (decision_id, action, commitments_added)
    decision_history: list = field(default_factory=list)
    # V3: dead flag (all children pruned by no-goods)
    dead: bool = False

    @property
    def avg_cost(self) -> float:
        if self.visit_count == 0:
            return float("inf")
        return self.total_cost / self.visit_count


# ---------------------------------------------------------------------------
# MCTS_V3 -- Constraint-Context with Deep No-Good Learning
# ---------------------------------------------------------------------------

class MCTS_V3(MCTS):

    def __init__(self, cfg=None):
        if cfg is None:
            cfg = OmegaConf.load(
                Path(__file__).resolve().parent.parent / "conf" / "v3.yaml")
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
        self._decision_counter = 0
        # V3: global no-good table
        self._nogoods: list[NoGood] = []

        root_plan = _build_initial_plan(state, grid_env)
        root = MCTSNode(plan=root_plan, parent=None, action=None)

        for _ in range(self.cfg.max_iterations):
            self._iteration += 1
            node = self._select(root)

            if node.dead:
                self._backprop(node, self.cfg.penalty_cost)
                continue

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

    # -- No-good helpers --

    def _next_decision_id(self):
        self._decision_counter += 1
        return self._decision_counter

    def _is_action_nogood(self, commitments, action_key):
        """Check if action_key is forbidden given current commitments."""
        ctx_sig = _build_context_signature(commitments)
        for ng in self._nogoods:
            if ng.action == action_key and _is_nogood_match(ng.ctx_sig, ctx_sig):
                return True
        return False

    def _record_nogood(self, commitments, culprit_decision_id, action_key):
        """Record a no-good: context before culprit + forbidden action."""
        # Context = commitments with decision_id < culprit
        prior = [c for c in commitments
                 if self._get_decision_id(c) < culprit_decision_id]
        ctx_sig = _build_context_signature(prior)
        new_ng = NoGood(ctx_sig=ctx_sig, action=action_key)

        # Subsumption check
        if self.cfg.nogood_subsumption:
            surviving = []
            subsumed = False
            for existing in self._nogoods:
                if existing.action != action_key:
                    surviving.append(existing)
                    continue
                # If new context is subset of existing, new subsumes existing
                if new_ng.ctx_sig.issubset(existing.ctx_sig):
                    # new is more general, drop existing
                    continue
                # If existing is subset of new, existing subsumes new
                if existing.ctx_sig.issubset(new_ng.ctx_sig):
                    subsumed = True
                    surviving.append(existing)
                    continue
                surviving.append(existing)
            self._nogoods = surviving
            if not subsumed:
                self._nogoods.append(new_ng)
        else:
            self._nogoods.append(new_ng)

        # Bound no-good table size
        if len(self._nogoods) > self.cfg.max_nogoods:
            self._nogoods = self._nogoods[-self.cfg.max_nogoods:]

    @staticmethod
    def _get_decision_id(commitment):
        """Extract decision_id from a commitment object."""
        if isinstance(commitment, (BlockReq, ProtectReq)):
            return commitment.decision_id
        if isinstance(commitment, SupportReq):
            return commitment.decision_id
        return 0

    # -- Selection (LCB for minimization, with dead-node skipping) --

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.children:
            if node.dead:
                return node
            if self._can_expand(node):
                return node
            alive = [c for c in node.children if not c.dead]
            if not alive:
                node.dead = True
                return node
            viable = [c for c in alive
                      if c.plan.cost() <= self._best_cost]
            if not viable:
                viable = alive
            node = min(viable, key=lambda c: self._lcb(c, node))
        return node

    def _lcb(self, child: MCTSNode, parent: MCTSNode) -> float:
        if child.visit_count == 0:
            return float("-inf")
        exploit = child.avg_cost
        explore = self.cfg.ucb_exploration * math.sqrt(
            math.log(parent.visit_count + 1) / child.visit_count)
        return exploit - explore

    def _can_expand(self, node: MCTSNode) -> bool:
        k = math.ceil(
            self.cfg.pw_C * (max(node.visit_count, 1) ** self.cfg.pw_alpha))
        return len(node.children) < k

    # -- Expansion --

    def _expand(self, node: MCTSNode, grid_env: GridEnv, state: State):
        plan = node.plan
        open_edges = plan.open_edges()
        if not open_edges:
            return None

        parent_id, child_id = max(
            open_edges, key=lambda e: plan.g.edges[e]["cost"] or 0)

        candidates = _get_candidates(plan, parent_id, state, grid_env)
        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1])

        tried = {c.action for c in node.children}
        for subgoal, _score, parent_sp in candidates:
            action_key = (parent_id, child_id,
                          subgoal.bottleneck.position,
                          subgoal.support.position,
                          subgoal.helper.color,
                          parent_sp)
            if action_key in tried:
                continue

            # V3: no-good filtering during expansion
            if self._is_action_nogood(node.commitments, action_key):
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

            # V3: build child commitments and decision history
            decision_id = self._next_decision_id()
            seg_id = f"{parent_id}->{child_id}"
            new_commits = _commitments_for_subgoal(
                subgoal, seg_id, decision_id, state)

            child_commitments = list(node.commitments) + new_commits
            child_history = list(node.decision_history) + [
                (decision_id, action_key, new_commits)]

            # V3: check for conflicts -- record no-goods but still create node
            conflict, conflict_set = _detect_conflicts(child_commitments)
            if conflict and self.cfg.backjump_enabled:
                # Find culprit: most recent decision in conflict set
                conflict_dids = set()
                for c in conflict_set:
                    conflict_dids.add(self._get_decision_id(c))
                culprit_did = max(conflict_dids) if conflict_dids else decision_id
                # Record no-good for future filtering
                culprit_action = action_key
                for did, act, _ in child_history:
                    if did == culprit_did:
                        culprit_action = act
                        break
                self._record_nogood(
                    child_commitments, culprit_did, culprit_action)

            child_node = MCTSNode(
                plan=child_plan, parent=node, action=action_key,
                commitments=child_commitments,
                decision_history=child_history)
            self._node_count += 1
            node.children.append(child_node)
            self._check_complete(child_plan, grid_env)
            return child_node

        # V3: if all candidates filtered, check dead propagation
        self._propagate_dead(node)
        return None

    # -- Dead node propagation --

    def _propagate_dead(self, node: MCTSNode):
        """If all children are dead/filtered, mark node dead and propagate up."""
        if not node.children:
            return  # leaf with no children isn't "dead" per se
        if all(c.dead for c in node.children):
            node.dead = True
            if node.parent is not None:
                self._propagate_dead(node.parent)

    # -- Rollout (with constraint propagation and backjumping) --

    def _rollout(self, node: MCTSNode, grid_env: GridEnv, state: State):
        self._rollout_count += 1
        plan = _copy_plan(node.plan)
        commitments = list(node.commitments)
        decision_history = list(node.decision_history)

        for step in range(self.cfg.max_rollout_depth):
            # Close all directly closeable edges first
            for pid, cid in list(plan.open_edges()):
                _try_close_edge(plan, pid, cid, grid_env)

            open_edges = plan.open_edges()
            if not open_edges:
                break

            # Pick random open edge (like V1 rollout)
            parent_id, child_id = random.choice(open_edges)

            candidates = _get_candidates(plan, parent_id, state, grid_env)
            if not candidates:
                return self.cfg.penalty_cost

            # V3: filter by no-goods during rollout
            filtered = []
            for subgoal, score, parent_sp in candidates:
                action_key = (parent_id, child_id,
                              subgoal.bottleneck.position,
                              subgoal.support.position,
                              subgoal.helper.color,
                              parent_sp)
                if not self._is_action_nogood(commitments, action_key):
                    filtered.append((subgoal, score, parent_sp))

            if not filtered:
                # Fall back to unfiltered candidates if all no-good'd
                filtered = candidates

            # Pick from top half of remaining
            filtered.sort(key=lambda x: x[1])
            pick_from = max(1, len(filtered) // 2)
            subgoal, _, parent_sp = random.choice(filtered[:pick_from])

            action_key = (parent_id, child_id,
                          subgoal.bottleneck.position,
                          subgoal.support.position,
                          subgoal.helper.color,
                          parent_sp)

            result = _apply_subgoal(
                plan, parent_id, child_id,
                subgoal, grid_env, self._cnt,
                parent_support_pos=parent_sp)
            if result is None:
                continue

            # V3: add commitments
            decision_id = self._next_decision_id()
            seg_id = f"{parent_id}->{child_id}"
            new_commits = _commitments_for_subgoal(
                subgoal, seg_id, decision_id, state)
            commitments.extend(new_commits)
            decision_history.append((decision_id, action_key, new_commits))

            # V3: check for conflicts and record no-goods (but don't abort rollout)
            conflict, conflict_set = _detect_conflicts(commitments)
            if conflict and self.cfg.backjump_enabled:
                conflict_dids = set()
                for c in conflict_set:
                    conflict_dids.add(self._get_decision_id(c))
                culprit_did = max(conflict_dids) if conflict_dids else decision_id
                # Record no-good for future use
                culprit_action = action_key
                for did, act, _ in decision_history:
                    if did == culprit_did:
                        culprit_action = act
                        break
                self._record_nogood(commitments, culprit_did, culprit_action)
                # Continue rollout -- the plan may still complete

        # Close remaining closeable
        for pid, cid in list(plan.open_edges()):
            _try_close_edge(plan, pid, cid, grid_env)

        if plan.is_complete():
            self._check_complete(plan, grid_env)
            cost = evaluate_plan(plan, grid_env)
            if cost is not None:
                return cost

        return plan.cost()

    # -- Backpropagation (backjump-aware) --

    def _backprop(self, node: MCTSNode, cost: float):
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

    def _fallback(self, root: MCTSNode, grid_env: GridEnv, state: State):
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
