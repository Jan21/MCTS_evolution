from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
from omegaconf import OmegaConf

from GridEnv import GridEnv, Robot_at, State
from partial_plan import PartialPlan
from evaluate_plan import evaluate_plan
from MCTS.base import MCTS, SolveResult, PlanEntry, PlanStats


# ---------------------------------------------------------------------------
# MCTS tree node (V2: with conflict graph + footprints)
# ---------------------------------------------------------------------------

@dataclass
class MCTSNode:
    plan: PartialPlan
    parent: MCTSNode | None
    action: tuple | None
    children: list[MCTSNode] = field(default_factory=list)
    visit_count: int = 0
    total_cost: float = 0.0
    # V2: {segment_tuple: {robot_color: position}}
    footprints: dict[tuple, dict[str, int]] = field(default_factory=dict)
    # V2: set of frozenset({seg1, seg2}) for helper_reuse conflicts
    conflict_edges: set[frozenset] = field(default_factory=set)

    @property
    def avg_cost(self) -> float:
        if self.visit_count == 0:
            return float("inf")
        return self.total_cost / self.visit_count


# ---------------------------------------------------------------------------
# Helpers
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
    """Return list of (segment_state, support_robot) for propose_subgoal_states."""
    g = plan.g
    parent_data = g.nodes[parent_id]
    parent_type = parent_data["ntype"]

    if parent_type == "goal":
        goal_pos = parent_data["pos"]
        results = []
        if (goal_pos, None) in grid_env._bottleneck_support_pairs_cache:
            results.append((State(target=goal_pos,
                                  target_robot=state.target_robot,
                                  helpers=state.helpers), None))
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
    """Collect all (subgoal, score, parent_support_pos) candidates."""
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

    g.remove_edge(parent_id, child_id)

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

    plan.add_edge(parent_id, sg_id, status="fixed", cost=exact_cost)
    plan.add_edge(sg_id, bn_id, status="fixed", cost=None)
    plan.add_edge(sg_id, sp_id, status="fixed", cost=None)

    child_pos = _get_start_pos(g, child_id)
    relaxed_bn = grid_env.compute_relaxed_shortest_path_length(
        child_pos, subgoal.bottleneck.position, subgoal.support.position)
    plan.add_edge(bn_id, child_id, status="open",
                  cost=relaxed_bn if relaxed_bn is not None else 999)

    relaxed_sp = grid_env.compute_relaxed_shortest_path_length(
        subgoal.helper.position, subgoal.support.position)
    plan.add_edge(sp_id, leaf_id, status="open",
                  cost=relaxed_sp if relaxed_sp is not None else 999)

    return (sg_id, bn_id, sp_id, leaf_id)


def _try_close_edge(plan, parent_id, child_id, grid_env):
    """Try to close an open edge directly (without subgoal insertion)."""
    g = plan.g
    parent_data = g.nodes[parent_id]
    child_pos = _get_start_pos(g, child_id)
    parent_pos = parent_data["pos"]

    exact = grid_env.compute_exact_shortest_path_length(
        child_pos, parent_pos, None)
    if exact is not None:
        g.edges[parent_id, child_id]["status"] = "fixed"
        g.edges[parent_id, child_id]["cost"] = exact
        return True

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
# V2 Conflict graph helpers
# ---------------------------------------------------------------------------

def _extract_footprint(subgoal):
    """Extract {robot_color: position} from a subgoal's support robot."""
    return {subgoal.support.color: subgoal.support.position}


def _detect_helper_reuse(footprints, new_seg, new_fp):
    """Detect helper_reuse conflicts between new_seg's footprint and others.

    Returns set of frozenset({seg1, seg2}) conflict edges.
    """
    conflicts = set()
    for seg, fp in footprints.items():
        if seg == new_seg:
            continue
        for color, pos in new_fp.items():
            if color in fp and fp[color] != pos:
                conflicts.add(frozenset({seg, new_seg}))
    return conflicts


def _get_conflict_components(segments, conflict_edges):
    """Get connected components from conflict edges."""
    g = nx.Graph()
    g.add_nodes_from(segments)
    for edge in conflict_edges:
        pair = list(edge)
        if len(pair) == 2 and pair[0] in segments and pair[1] in segments:
            g.add_edge(pair[0], pair[1])
    return list(nx.connected_components(g))


def _compute_relocation_penalty(footprints, conflict_edges, grid_env):
    """Sum relocation penalties for all helper_reuse conflict edges."""
    penalty = 0.0
    for edge in conflict_edges:
        pair = list(edge)
        if len(pair) != 2:
            continue
        seg1, seg2 = pair
        fp1 = footprints.get(seg1, {})
        fp2 = footprints.get(seg2, {})
        for color in set(fp1) & set(fp2):
            if fp1[color] != fp2[color]:
                reloc = grid_env.compute_relaxed_shortest_path_length(
                    fp1[color], fp2[color])
                if reloc is None:
                    return float("inf")
                penalty += reloc
    return penalty


def _conflict_penalty_for_candidate(subgoal, seg, footprints, grid_env,
                                     unreachable_penalty=100.0):
    """Compute soft penalty for a candidate based on existing footprints."""
    new_fp = _extract_footprint(subgoal)
    penalty = 0.0
    for other_seg, fp in footprints.items():
        if other_seg == seg:
            continue
        for color, pos in new_fp.items():
            if color in fp and fp[color] != pos:
                reloc = grid_env.compute_relaxed_shortest_path_length(
                    pos, fp[color])
                if reloc is None:
                    penalty += unreachable_penalty
                else:
                    penalty += reloc
    return penalty


def _copy_footprints(footprints):
    return {seg: dict(fp) for seg, fp in footprints.items()}


def _copy_conflict_edges(conflict_edges):
    return set(conflict_edges)


# ---------------------------------------------------------------------------
# MCTS_V2 -- V2 Lazy-Coupling with Conflict Graph
# ---------------------------------------------------------------------------

class MCTS_V2(MCTS):

    def __init__(self, cfg=None):
        if cfg is None:
            cfg = OmegaConf.load(
                Path(__file__).resolve().parent.parent / "conf" / "v2.yaml")
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

        root_plan = _build_initial_plan(state, grid_env)
        root = MCTSNode(plan=root_plan, parent=None, action=None)

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

        best = self._best_plan if self._best_plan is not None else self._fallback(root, grid_env, state)
        return SolveResult(best_plan=best, all_plans=self._all_plans)

    # -- Selection (LCB for minimization) --

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.children:
            if self._can_expand(node):
                return node
            viable = [c for c in node.children
                      if c.plan.cost() <= self._best_cost]
            if not viable:
                return node
            node = min(viable, key=lambda c: self._lcb(c, node))
        return node

    def _lcb(self, child: MCTSNode, parent: MCTSNode) -> float:
        if child.visit_count == 0:
            return float("-inf")
        exploit = child.avg_cost
        explore = self.cfg.ucb_exploration * math.sqrt(
            math.log(parent.visit_count + 1) / child.visit_count)
        # V2: exploration bonus for nodes with conflicts (more uncertainty)
        conflict_bonus = 0.0
        if child.conflict_edges:
            open_segs = [tuple(e) for e in child.plan.open_edges()]
            if open_segs:
                components = _get_conflict_components(
                    open_segs, child.conflict_edges)
                max_comp = max((len(c) for c in components), default=1)
                if max_comp > 1:
                    conflict_bonus = self.cfg.conflict_exploration_bonus * math.log(max_comp)
        return exploit - explore - conflict_bonus

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

        # Pick highest-cost open edge to refine
        parent_id, child_id = max(
            open_edges, key=lambda e: plan.g.edges[e]["cost"] or 0)
        seg = (parent_id, child_id)

        candidates = _get_candidates(plan, parent_id, state, grid_env)
        if not candidates:
            return None

        # V2: apply soft conflict penalty to candidate scores
        scored_candidates = []
        for subgoal, score, parent_sp in candidates:
            penalty = _conflict_penalty_for_candidate(
                subgoal, seg, node.footprints, grid_env,
                self.cfg.unreachable_relocation_penalty)
            scored_candidates.append((subgoal, score + penalty, parent_sp))

        scored_candidates.sort(key=lambda x: x[1])

        tried = {c.action for c in node.children}
        for subgoal, _score, parent_sp in scored_candidates:
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

            for pid, cid in list(child_plan.open_edges()):
                _try_close_edge(child_plan, pid, cid, grid_env)

            # V2: build child footprints and detect conflicts
            child_footprints = _copy_footprints(node.footprints)
            new_fp = _extract_footprint(subgoal)
            if seg in child_footprints:
                child_footprints[seg].update(new_fp)
            else:
                child_footprints[seg] = dict(new_fp)

            child_conflicts = _copy_conflict_edges(node.conflict_edges)
            new_conflicts = _detect_helper_reuse(
                child_footprints, seg, child_footprints.get(seg, {}))
            child_conflicts |= new_conflicts

            child_node = MCTSNode(
                plan=child_plan, parent=node, action=action_key,
                footprints=child_footprints,
                conflict_edges=child_conflicts)
            self._node_count += 1
            node.children.append(child_node)
            self._check_complete(child_plan, grid_env)

            # V2: propagate discovered conflicts back to parent
            if new_conflicts:
                node.conflict_edges |= new_conflicts

            return child_node

        return None

    # -- Rollout (random greedy completion with conflict detection) --

    def _rollout(self, node: MCTSNode, grid_env: GridEnv, state: State):
        self._rollout_count += 1
        plan = _copy_plan(node.plan)
        footprints = _copy_footprints(node.footprints)
        conflict_edges = _copy_conflict_edges(node.conflict_edges)

        for _ in range(self.cfg.max_rollout_depth):
            for pid, cid in list(plan.open_edges()):
                _try_close_edge(plan, pid, cid, grid_env)

            open_edges = plan.open_edges()
            if not open_edges:
                break

            parent_id, child_id = random.choice(open_edges)
            seg = (parent_id, child_id)

            candidates = _get_candidates(plan, parent_id, state, grid_env)
            if not candidates:
                return self.cfg.penalty_cost

            # V2: apply conflict penalty during rollout
            scored = []
            for subgoal, score, parent_sp in candidates:
                penalty = _conflict_penalty_for_candidate(
                    subgoal, seg, footprints, grid_env,
                    self.cfg.unreachable_relocation_penalty)
                scored.append((subgoal, score + penalty, parent_sp))

            scored.sort(key=lambda x: x[1])
            pick_from = max(1, len(scored) // 2)
            subgoal, _, parent_sp = random.choice(scored[:pick_from])

            result = _apply_subgoal(
                plan, parent_id, child_id,
                subgoal, grid_env, self._cnt,
                parent_support_pos=parent_sp)
            if result is None:
                continue

            # V2: update footprints and detect conflicts
            new_fp = _extract_footprint(subgoal)
            if seg in footprints:
                footprints[seg].update(new_fp)
            else:
                footprints[seg] = dict(new_fp)

            new_conflicts = _detect_helper_reuse(
                footprints, seg, footprints.get(seg, {}))
            conflict_edges |= new_conflicts

            # Propagate rollout-discovered conflicts back to the MCTS node
            if new_conflicts:
                node.conflict_edges |= new_conflicts

        for pid, cid in list(plan.open_edges()):
            _try_close_edge(plan, pid, cid, grid_env)

        if plan.is_complete():
            self._check_complete(plan, grid_env)
            cost = evaluate_plan(plan, grid_env)
            if cost is not None:
                # V2: add relocation penalties for conflicts
                reloc = _compute_relocation_penalty(
                    footprints, conflict_edges, grid_env)
                return cost + reloc

        # Incomplete or unreachable: use plan cost + relocation penalty
        base = plan.cost()
        reloc = _compute_relocation_penalty(
            footprints, conflict_edges, grid_env)
        if reloc == float("inf"):
            return self.cfg.penalty_cost
        return base + reloc

    # -- Backpropagation --

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

        for parent_id, child_id in list(plan.open_edges()):
            _try_close_edge(plan, parent_id, child_id, grid_env)

        return plan
