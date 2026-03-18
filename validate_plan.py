"""Validation of partial plans for Ricochet Robots.

Runs structural and semantic checks on a PartialPlan DAG:

  1.  The plan skeleton is a valid DAG (cross-branch edges excluded).
  2.  Exactly one goal node with a `pos` attribute; goal is the root (in-degree 0).
  3.  Goal position matches state target.
  4.  Every node has a valid type with its required attributes.
  5.  All positions referenced by nodes are valid grid coordinates.
  6.  Every edge connects a valid (parent_type, child_type) pair.
  7.  Every subgoal has exactly one bottleneck child and one support child.
  8.  Robot continuity: a child's robot must match what the parent expects.
  9.  All leaf nodes have `pos` and `robot` and have no children.
  10. Each leaf's (robot, pos) corresponds to an actual robot in the state.
  11. The plan is complete (no open edges).
  12. Every physical edge has a reachable path on the grid.

Usage:
    from GridEnv import GridEnv
    from validate_plan import validate

    grid_env, state = GridEnv.from_env(0)
    result = validate(plan, grid_env, state)
    print(result)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from GridEnv import GridEnv, Robot_at, State

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODE_TYPES = {"goal", "subgoal", "bottleneck", "support", "leaf"}

REQUIRED_ATTRS: dict[str, list[str]] = {
    "goal":       ["pos"],
    "subgoal":    [],
    "bottleneck": ["pos", "robot"],
    "support":    ["pos", "robot"],
    "leaf":       ["pos", "robot"],
}

# Edges stored as (parent, child) in the code direction (goal → leaf).
# The physical movement goes the other way (child pos → parent pos).
VALID_EDGE_PAIRS: set[tuple[str, str]] = {
    ("goal",       "leaf"),
    ("goal",       "subgoal"),
    ("bottleneck", "leaf"),
    ("bottleneck", "subgoal"),
    ("support",    "leaf"),
    ("support",    "subgoal"),
    ("support",    "support"),
    ("subgoal",    "bottleneck"),
    ("subgoal",    "support"),
    ("bottleneck",    "support"),
}

# Structural edges carry no physical movement cost of their own.
STRUCTURAL_EDGE_PAIRS: set[tuple[str, str]] = {
    ("subgoal", "bottleneck"),
    ("subgoal", "support"),
}

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.passed:
            return "PASSED"
        lines = [f"FAILED ({len(self.errors)} errors):"]
        for e in self.errors:
            lines.append(f"  - {e}")
        return "\n".join(lines)

# ---------------------------------------------------------------------------
# Individual checks – each returns a list of error strings (empty == pass)
# ---------------------------------------------------------------------------

def check_is_dag(g: nx.DiGraph) -> list[str]:
    """The plan skeleton (excluding cross-branch edges) must be acyclic.

    Cross-branch edges (support→support, bottleneck→support) can create
    valid cycles when the same position is reused across plan branches
    at different time steps.  Only the skeleton without these edges must
    be a DAG.
    """
    cross_edge_types = {("support", "support"), ("bottleneck", "support")}
    skeleton = g.copy()
    cross_edges = [
        (u, v) for u, v in skeleton.edges()
        if (skeleton.nodes[u].get("ntype"), skeleton.nodes[v].get("ntype"))
        in cross_edge_types
    ]
    skeleton.remove_edges_from(cross_edges)

    if not nx.is_directed_acyclic_graph(skeleton):
        return ["Plan skeleton contains a cycle (excluding cross-branch edges)"]
    return []


def check_node_types_and_attrs(g: nx.DiGraph) -> list[str]:
    """Every node has a valid ``ntype``, required attributes, and Robot_at instances."""
    errors: list[str] = []
    for nid, data in g.nodes(data=True):
        ntype = data.get("ntype")
        if ntype is None:
            errors.append(f"Node '{nid}': missing 'ntype'")
            continue
        if ntype not in NODE_TYPES:
            errors.append(f"Node '{nid}': unknown type '{ntype}'")
            continue
        for attr in REQUIRED_ATTRS[ntype]:
            if attr not in data:
                errors.append(f"Node '{nid}' (type={ntype}): missing '{attr}'")
        if "robot" in REQUIRED_ATTRS.get(ntype, []) and "robot" in data:
            if not isinstance(data["robot"], Robot_at):
                errors.append(
                    f"Node '{nid}' (type={ntype}): 'robot' must be a Robot_at "
                    f"instance, got {type(data['robot']).__name__}"
                )
    return errors


def check_goal_node(g: nx.DiGraph, state: State) -> list[str]:
    """Exactly one goal node; it is the root; its pos matches state target."""
    errors: list[str] = []
    goals = [
        (nid, d) for nid, d in g.nodes(data=True) if d.get("ntype") == "goal"
    ]

    if len(goals) == 0:
        return ["No goal node found"]
    if len(goals) > 1:
        errors.append(f"Multiple goal nodes: {[n for n, _ in goals]}")

    goal_id, goal_data = goals[0]

    if g.in_degree(goal_id) > 0:
        parents = list(g.predecessors(goal_id))
        errors.append(f"Goal '{goal_id}' is not root; parents={parents}")

    goal_pos = goal_data.get("pos")
    if goal_pos is not None and goal_pos != state.target:
        errors.append(
            f"Goal pos {goal_pos} != state target {state.target}"
        )

    return errors


def check_leaf_nodes(g: nx.DiGraph, state: State) -> list[str]:
    """Leaves have pos+robot, no children, and match actual robots."""
    errors: list[str] = []
    actual = state.all_robots

    for nid, data in g.nodes(data=True):
        if data.get("ntype") != "leaf":
            continue

        if g.out_degree(nid) > 0:
            errors.append(
                f"Leaf '{nid}' has children: {list(g.successors(nid))}"
            )

        robot: Robot_at | None = data.get("robot")
        pos = data.get("pos")
        if robot is None or pos is None:
            continue
        if not isinstance(robot, Robot_at):
            continue

        if pos != robot.position:
            errors.append(
                f"Leaf '{nid}': pos {pos} != robot position "
                f"{robot.position}"
            )

        if not any(
            r.color == robot.color and r.position == pos for r in actual
        ):
            errors.append(
                f"Leaf '{nid}' (robot={robot.color!r}, pos={pos}) "
                f"not found among state robots"
            )

    return errors


def check_positions_on_grid(g: nx.DiGraph, grid_nodes: set) -> list[str]:
    """All node positions must be valid grid coordinates."""
    errors: list[str] = []
    for nid, data in g.nodes(data=True):
        pos = data.get("pos")
        if pos is not None and pos not in grid_nodes:
            errors.append(f"Node '{nid}': pos {pos} is not a valid grid node")
    return errors


def check_edge_types(g: nx.DiGraph) -> list[str]:
    """Every edge must connect a valid (parent_type, child_type) pair."""
    errors: list[str] = []
    for u, v in g.edges():
        ut = g.nodes[u].get("ntype")
        vt = g.nodes[v].get("ntype")
        if (ut, vt) not in VALID_EDGE_PAIRS:
            errors.append(
                f"Edge ('{u}','{v}'): invalid type pair ({ut} -> {vt})"
            )
    return errors


def check_subgoal_children(g: nx.DiGraph) -> list[str]:
    """Every subgoal has exactly one bottleneck child and one support child."""
    errors: list[str] = []
    for nid, data in g.nodes(data=True):
        if data.get("ntype") != "subgoal":
            continue
        child_types = [g.nodes[c].get("ntype") for c in g.successors(nid)]
        bn = child_types.count("bottleneck")
        sp = child_types.count("support")
        if bn != 1:
            errors.append(
                f"Subgoal '{nid}': has {bn} bottleneck children (need 1)"
            )
        if sp != 1:
            errors.append(
                f"Subgoal '{nid}': has {sp} support children (need 1)"
            )
    return errors


def _expected_robot_color(g: nx.DiGraph, parent_id: str, state: State) -> str | None:
    """Return the robot color that a parent node expects its child to carry."""
    parent_data = g.nodes[parent_id]
    parent_type = parent_data.get("ntype")

    if parent_type == "goal":
        return state.target_robot.color
    elif parent_type in ("bottleneck", "support"):
        parent_robot = parent_data.get("robot")
        if isinstance(parent_robot, Robot_at):
            return parent_robot.color
    return None


def check_robot_continuity(g: nx.DiGraph, state: State) -> list[str]:
    """A child's robot must match what its parent expects.

    Checks two cases:
      - Leaf nodes: the leaf's robot must match its parent's robot.
      - Bottleneck nodes inside a subgoal: the bottleneck's robot must match
        what the subgoal's parent expects (looking through the subgoal).
    """
    errors: list[str] = []

    for parent, child in g.edges():
        pt = g.nodes[parent].get("ntype")
        ct = g.nodes[child].get("ntype")

        # Case 1: leaf — robot must match its direct parent.
        if ct == "leaf":
            child_robot = g.nodes[child].get("robot")
            if not isinstance(child_robot, Robot_at):
                continue
            expected = _expected_robot_color(g, parent, state)
            if expected is not None and child_robot.color != expected:
                errors.append(
                    f"Edge ('{parent}','{child}'): leaf robot "
                    f"{child_robot.color!r} != expected {expected!r}"
                )

        # Case 2: subgoal → bottleneck — bottleneck robot must match
        # what the subgoal's parent expects.
        if pt == "subgoal" and ct == "bottleneck":
            bn_robot = g.nodes[child].get("robot")
            if not isinstance(bn_robot, Robot_at):
                continue
            subgoal_parents = list(g.predecessors(parent))
            if not subgoal_parents:
                continue
            expected = _expected_robot_color(g, subgoal_parents[0], state)
            if expected is not None and bn_robot.color != expected:
                errors.append(
                    f"Subgoal '{parent}': bottleneck robot {bn_robot.color!r} "
                    f"!= expected {expected!r} (from parent '{subgoal_parents[0]}')"
                )

    return errors


def check_completeness(g: nx.DiGraph) -> list[str]:
    """A finished plan has no open edges."""
    open_edges = [
        (u, v) for u, v, d in g.edges(data=True) if d.get("status") == "open"
    ]
    if open_edges:
        return [f"Plan has {len(open_edges)} open edge(s): {open_edges}"]
    return []


def check_edge_reachability(g: nx.DiGraph, grid_env: GridEnv) -> list[str]:
    """Every physical edge must have a reachable path on the grid.

    Structural edges (subgoal -> bottleneck, subgoal -> support) are skipped.
    If compute_exact_shortest_path_length returns None, there is no path
    and the edge is invalid.
    """
    errors: list[str] = []

    for u, v in g.edges():
        ut = g.nodes[u].get("ntype")
        vt = g.nodes[v].get("ntype")

        if (ut, vt) in STRUCTURAL_EDGE_PAIRS:
            continue

        start, end, support = _physical_movement(g, parent=u, child=v)
        if start is None or end is None:
            errors.append(
                f"Edge ('{u}','{v}'): cannot determine physical movement"
            )
            continue

        result = grid_env.compute_exact_shortest_path_length(start, end, support)
        if result is None:
            errors.append(
                f"Edge ('{u}','{v}'): no path from {start} to {end}"
                + (f" with support at {support}" if support else "")
            )

    return errors


def _physical_movement(
    g: nx.DiGraph,
    parent: str,
    child: str,
) -> tuple[
    tuple[int, int] | None,
    tuple[int, int] | None,
    tuple[int, int] | None,
]:
    """Determine ``(start_pos, end_pos, support_pos)`` for a physical edge.

    The physical movement goes *from* the child side *to* the parent side.
    """
    child_data = g.nodes[child]
    parent_data = g.nodes[parent]
    child_type = child_data.get("ntype")

    if child_type == "subgoal":
        bn = [
            c for c in g.successors(child)
            if g.nodes[c].get("ntype") == "bottleneck"
        ]
        if not bn:
            return None, None, None
        # Use stored parent_support_pos if available
        support_pos = child_data.get("parent_support_pos")
        if support_pos is None and parent_data.get("ntype") == "bottleneck":
            # Fallback: look up parent's sibling support
            for sg_parent in g.predecessors(parent):
                if g.nodes[sg_parent].get("ntype") == "subgoal":
                    for sib in g.successors(sg_parent):
                        if g.nodes[sib].get("ntype") == "support":
                            support_pos = g.nodes[sib].get("pos")
                            break
                    break
        return (
            g.nodes[bn[0]].get("pos"),
            parent_data.get("pos"),
            support_pos,
        )

    if child_type in ("leaf", "support"):
        support_pos = None
        if parent_data.get("ntype") == "bottleneck":
            # Physical movement to a bottleneck may use a dependent edge
            # through the sibling support position.
            for sg_parent in g.predecessors(parent):
                if g.nodes[sg_parent].get("ntype") == "subgoal":
                    for sib in g.successors(sg_parent):
                        if g.nodes[sib].get("ntype") == "support":
                            support_pos = g.nodes[sib].get("pos")
                            break
                    break
        return child_data.get("pos"), parent_data.get("pos"), support_pos

    return None, None, None

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate(plan, grid_env: GridEnv, state: State) -> ValidationResult:
    """Run every validation check and return a :class:`ValidationResult`.

    Args:
        plan:      A ``PartialPlan`` instance (must expose ``.g``).
        grid_env:  A ``GridEnv`` instance.
        state:     A ``State`` instance (puzzle to solve).
    """
    g = plan.g
    grid_nodes = set(grid_env.G.nodes())
    all_errors: list[str] = []

    checks = [
        ("DAG structure",           check_is_dag(g)),
        ("Node types & attributes", check_node_types_and_attrs(g)),
        ("Goal node",               check_goal_node(g, state)),
        ("Leaf nodes",              check_leaf_nodes(g, state)),
        ("Positions on grid",       check_positions_on_grid(g, grid_nodes)),
        ("Edge types",              check_edge_types(g)),
        ("Subgoal children",        check_subgoal_children(g)),
        ("Robot continuity",        check_robot_continuity(g, state)),
        ("Completeness",            check_completeness(g)),
        ("Edge reachability",       check_edge_reachability(g, grid_env)),
    ]

    for name, errs in checks:
        for e in errs:
            all_errors.append(f"[{name}] {e}")

    return ValidationResult(passed=len(all_errors) == 0, errors=all_errors)
