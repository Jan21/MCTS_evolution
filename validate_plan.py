"""Validation of partial plans for Ricochet Robots.

Runs structural and semantic checks on a PartialPlan DAG:

  1.  The graph is a valid DAG (no cycles).
  2.  Exactly one goal node with a `pos` attribute; goal is the root (in-degree 0).
  3.  Goal position matches state target.
  4.  Every node has a valid type with its required attributes.
  5.  All positions referenced by nodes are valid grid coordinates.
  6.  Every edge connects a valid (parent_type, child_type) pair.
  7.  Every subgoal has exactly one bottleneck child and one support child.
  8.  The bottleneck child's robot matches the robot expected by the subgoal's
      parent (target robot when the parent is the goal node).
  9.  All leaf nodes have `pos` and `robot` and have no children.
  10. Each leaf's (robot, pos) corresponds to an actual robot in the state.
  11. A leaf's robot matches the robot on its parent (bottleneck, support, or
      goal's target robot).
  12. The plan is complete (no open edges).
  13. Every physical (non-structural) edge cost equals the exact shortest path.

Usage:
    from game import Game
    from validate_plan import validate

    game   = Game.from_env(0)
    result = validate(plan, game)
    print(result)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from game import Game
from robot import Robot

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
# Shortest-path stub (TODO: replace with real implementation)
# ---------------------------------------------------------------------------

def compute_exact_shortest_path_length(start_pos, end_pos, support_pos=None):
    """Compute shortest path length between two positions.

    TODO: implement using the grid graph.  For now returns None (skip).
    """
    return None

# ---------------------------------------------------------------------------
# Individual checks – each returns a list of error strings (empty == pass)
# ---------------------------------------------------------------------------

def check_is_dag(g: nx.DiGraph) -> list[str]:
    """The graph must be a directed acyclic graph."""
    if not nx.is_directed_acyclic_graph(g):
        return ["Graph contains a cycle (not a DAG)"]
    return []


def check_node_types_and_attrs(g: nx.DiGraph) -> list[str]:
    """Every node has a valid ``ntype``, required attributes, and Robot instances."""
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
            if not isinstance(data["robot"], Robot):
                errors.append(
                    f"Node '{nid}' (type={ntype}): 'robot' must be a Robot "
                    f"instance, got {type(data['robot']).__name__}"
                )
    return errors


def check_goal_node(g: nx.DiGraph, game: Game) -> list[str]:
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

    # Goal must be the root (no incoming edges).
    if g.in_degree(goal_id) > 0:
        parents = list(g.predecessors(goal_id))
        errors.append(f"Goal '{goal_id}' is not root; parents={parents}")

    # Goal position must match state target.
    goal_pos = goal_data.get("pos")
    if goal_pos is not None and goal_pos != game.state.target:
        errors.append(
            f"Goal pos {goal_pos} != state target {game.state.target}"
        )

    return errors


def check_leaf_nodes(g: nx.DiGraph, game: Game) -> list[str]:
    """Leaves have pos+robot, no children, and match actual robots."""
    errors: list[str] = []
    actual = game.state.all_robots

    for nid, data in g.nodes(data=True):
        if data.get("ntype") != "leaf":
            continue

        # No children.
        if g.out_degree(nid) > 0:
            errors.append(
                f"Leaf '{nid}' has children: {list(g.successors(nid))}"
            )

        robot: Robot | None = data.get("robot")
        pos = data.get("pos")
        if robot is None or pos is None:
            continue  # already caught by check_node_types_and_attrs
        if not isinstance(robot, Robot):
            continue  # already caught by check_node_types_and_attrs

        # Leaf pos must equal the robot's actual current position.
        if pos != (robot.x, robot.y):
            errors.append(
                f"Leaf '{nid}': pos {pos} != robot position "
                f"({robot.x}, {robot.y})"
            )

        # The robot must exist in the state (matched by name and position).
        if not any(
            r.name == robot.name and (r.x, r.y) == pos for r in actual
        ):
            errors.append(
                f"Leaf '{nid}' (robot={robot.name!r}, pos={pos}) "
                f"not found among state robots"
            )

    return errors


def check_positions_on_grid(g: nx.DiGraph, game: Game) -> list[str]:
    """All node positions must be valid grid coordinates."""
    errors: list[str] = []
    for nid, data in g.nodes(data=True):
        pos = data.get("pos")
        if pos is not None and pos not in game.grid_nodes:
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


def check_bottleneck_robot_consistency(g: nx.DiGraph, game: Game) -> list[str]:
    """The bottleneck child's robot must match the parent's expected robot.

    When the subgoal's parent is the goal node the expected robot is the
    state's target robot.  When the parent is a bottleneck or support node
    the expected robot is that node's ``robot`` attribute.
    """
    errors: list[str] = []
    for nid, data in g.nodes(data=True):
        if data.get("ntype") != "subgoal":
            continue

        # Find parent of this subgoal.
        parents = list(g.predecessors(nid))
        if not parents:
            errors.append(f"Subgoal '{nid}' has no parent")
            continue
        parent_id = parents[0]
        parent_data = g.nodes[parent_id]
        parent_type = parent_data.get("ntype")

        # Find bottleneck child.
        bn_children = [
            c for c in g.successors(nid)
            if g.nodes[c].get("ntype") == "bottleneck"
        ]
        if not bn_children:
            continue  # caught by check_subgoal_children
        bn_robot: Robot | None = g.nodes[bn_children[0]].get("robot")
        if not isinstance(bn_robot, Robot):
            continue  # caught by check_node_types_and_attrs

        if parent_type == "goal":
            expected_name = game.state.target_robot.name
        elif parent_type in ("bottleneck", "support"):
            parent_robot = parent_data.get("robot")
            if not isinstance(parent_robot, Robot):
                continue
            expected_name = parent_robot.name
        else:
            continue

        if bn_robot.name != expected_name:
            errors.append(
                f"Subgoal '{nid}': bottleneck robot {bn_robot.name!r} "
                f"!= expected {expected_name!r} (from parent '{parent_id}')"
            )

    return errors


def check_edge_robot_continuity(g: nx.DiGraph, game: Game) -> list[str]:
    """The robot on a leaf must match the robot its parent expects.

    For each physical edge where the child is a leaf:
      - parent is goal       → leaf robot must be the target robot
      - parent is bottleneck → leaf robot must match bottleneck's robot
      - parent is support    → leaf robot must match support's robot
    """
    errors: list[str] = []
    for parent, child in g.edges():
        pt = g.nodes[parent].get("ntype")
        ct = g.nodes[child].get("ntype")
        if ct != "leaf":
            continue

        child_robot = g.nodes[child].get("robot")
        if not isinstance(child_robot, Robot):
            continue  # caught by check_node_types_and_attrs

        if pt == "goal":
            expected_name = game.state.target_robot.name
        elif pt in ("bottleneck", "support"):
            parent_robot = g.nodes[parent].get("robot")
            if not isinstance(parent_robot, Robot):
                continue
            expected_name = parent_robot.name
        else:
            continue

        if child_robot.name != expected_name:
            errors.append(
                f"Edge ('{parent}','{child}'): leaf robot "
                f"{child_robot.name!r} != parent's robot {expected_name!r}"
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


def check_edge_costs(g: nx.DiGraph) -> list[str]:
    """Every physical edge cost must equal the exact shortest path length.

    Structural edges (subgoal -> bottleneck, subgoal -> support) are skipped.

    For an edge whose child is a *subgoal*, the physical movement goes from
    the subgoal's bottleneck position to the parent's position, and the
    subgoal's support position is passed as the blocking ``support_pos``.

    For edges whose child is a *leaf* or *support*, the movement goes from
    the child position to the parent position with no support dependency.
    """
    errors: list[str] = []

    for u, v, edge_data in g.edges(data=True):
        ut = g.nodes[u].get("ntype")
        vt = g.nodes[v].get("ntype")

        # Skip structural edges.
        if (ut, vt) in STRUCTURAL_EDGE_PAIRS:
            continue

        cost = edge_data.get("cost")
        if cost is None:
            continue

        start, end, support = _physical_movement(g, parent=u, child=v)
        if start is None or end is None:
            errors.append(
                f"Edge ('{u}','{v}'): cannot determine physical movement"
            )
            continue

        expected = compute_exact_shortest_path_length(start, end, support)
        if expected is None:
            continue  # SPL not yet implemented, skip cost check
        if expected != cost:
            errors.append(
                f"Edge ('{u}','{v}'): cost={cost} but shortest path={expected}"
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
        # Robot moves from bottleneck pos -> parent pos, with support.
        bn = [
            c for c in g.successors(child)
            if g.nodes[c].get("ntype") == "bottleneck"
        ]
        sp = [
            c for c in g.successors(child)
            if g.nodes[c].get("ntype") == "support"
        ]
        if not bn or not sp:
            return None, None, None
        return (
            g.nodes[bn[0]].get("pos"),
            parent_data.get("pos"),
            g.nodes[sp[0]].get("pos"),
        )

    if child_type in ("leaf", "support"):
        # Direct independent movement: child pos -> parent pos.
        return child_data.get("pos"), parent_data.get("pos"), None

    return None, None, None

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate(plan, game: Game) -> ValidationResult:
    """Run every validation check and return a :class:`ValidationResult`.

    Args:
        plan:  A ``PartialPlan`` instance (must expose ``.g``).
        game:  A ``Game`` instance (board + puzzle state).
    """
    g = plan.g
    all_errors: list[str] = []

    checks = [
        ("DAG structure",                check_is_dag(g)),
        ("Node types & attributes",      check_node_types_and_attrs(g)),
        ("Goal node",                    check_goal_node(g, game)),
        ("Leaf nodes",                   check_leaf_nodes(g, game)),
        ("Positions on grid",            check_positions_on_grid(g, game)),
        ("Edge types",                   check_edge_types(g)),
        ("Subgoal children",             check_subgoal_children(g)),
        ("Bottleneck-robot consistency", check_bottleneck_robot_consistency(g, game)),
        ("Edge robot continuity",        check_edge_robot_continuity(g, game)),
        ("Completeness",                 check_completeness(g)),
        ("Edge costs",                   check_edge_costs(g)),
    ]

    for name, errs in checks:
        for e in errs:
            all_errors.append(f"[{name}] {e}")

    return ValidationResult(passed=len(all_errors) == 0, errors=all_errors)
