"""Evaluate a PartialPlan by summing exact shortest-path lengths over physical edges."""

from __future__ import annotations

from GridEnv import GridEnv
from partial_plan import PartialPlan
from validate_plan import STRUCTURAL_EDGE_PAIRS


def _find_sibling_support_pos(g, bottleneck_id: str):
    """Find the support position that is sibling to a bottleneck node.

    Traverses: bottleneck → parent subgoal → support child → pos.
    """
    for parent in g.predecessors(bottleneck_id):
        if g.nodes[parent].get("ntype") == "subgoal":
            for child in g.successors(parent):
                if g.nodes[child].get("ntype") == "support":
                    return g.nodes[child].get("pos")
    return None


def _edge_movement(g, parent: str, child: str):
    """Determine (start, end, support_pos) for a physical edge.

    Physical movement goes from child side to parent side.

    For parent→subgoal edges, the physical movement is bn→parent.
    The support_pos needed is the one that enables the dependent edge into
    the parent. This is determined by:
      - If parent is a bottleneck: the parent's own sibling support
      - Otherwise (goal or support parent): look at the subgoal's support child
        for the support that enables reaching the parent
    For other edges (parent→leaf, parent→support), support_pos comes from
    the parent's sibling support (only when parent is a bottleneck).
    """
    child_data = g.nodes[child]
    parent_data = g.nodes[parent]
    child_type = child_data.get("ntype")
    parent_type = parent_data.get("ntype")

    if child_type == "subgoal":
        bn = [c for c in g.successors(child)
              if g.nodes[c].get("ntype") == "bottleneck"]
        if not bn:
            return None, None, None
        # Use stored parent_support_pos if available (set during plan construction)
        support_pos = child_data.get("parent_support_pos")
        if support_pos is None and parent_type == "bottleneck":
            support_pos = _find_sibling_support_pos(g, parent)
        return (g.nodes[bn[0]].get("pos"),
                parent_data.get("pos"),
                support_pos)

    if child_type in ("leaf", "support"):
        support_pos = None
        if parent_type == "bottleneck":
            support_pos = _find_sibling_support_pos(g, parent)
        return child_data.get("pos"), parent_data.get("pos"), support_pos

    return None, None, None


def evaluate_plan(plan: PartialPlan, grid_env: GridEnv) -> float | None:
    """Compute total cost: sum of exact shortest path lengths over physical edges.

    Returns None if any physical edge is unreachable.
    """
    g = plan.g
    total = 0

    for parent, child in g.edges():
        parent_type = g.nodes[parent].get("ntype")
        child_type = g.nodes[child].get("ntype")

        if (parent_type, child_type) in STRUCTURAL_EDGE_PAIRS:
            continue

        start, end, support_pos = _edge_movement(g, parent, child)
        if start is None or end is None:
            return None

        cost = grid_env.compute_exact_shortest_path_length(start, end, support_pos)
        if cost is None:
            return None

        total += cost

    return total
