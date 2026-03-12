"""Trace instrumentation for MCTS algorithm visualization.

Records per-iteration events during MCTS search for later playback
in the HTML visualizer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from GridEnv import GridEnv, Robot_at
from partial_plan import PartialPlan


def _serialize_pos(pos):
    if pos is None:
        return None
    return list(pos)


def _serialize_robot(robot):
    if robot is None:
        return None
    return {"position": list(robot.position), "color": robot.color}


def _serialize_plan(plan: PartialPlan) -> dict:
    """Serialize a PartialPlan to a JSON-compatible dict."""
    g = plan.g
    nodes = []
    for nid, data in g.nodes(data=True):
        node = {"id": nid, "ntype": data.get("ntype")}
        if "pos" in data:
            node["pos"] = _serialize_pos(data["pos"])
        if "robot" in data:
            node["robot"] = _serialize_robot(data["robot"])
        if "parent_support_pos" in data:
            node["parent_support_pos"] = _serialize_pos(data["parent_support_pos"])
        nodes.append(node)

    edges = []
    for u, v, data in g.edges(data=True):
        edge = {"source": u, "target": v,
                "status": data.get("status"),
                "cost": data.get("cost")}
        edges.append(edge)

    return {"nodes": nodes, "edges": edges}


def _serialize_grid(grid_env: GridEnv) -> dict:
    """Serialize the grid structure (walls) for visualization.

    Uses grid_data from the env pickle when available (list of 256 cell
    descriptors like 'NW', 'X', 'SE', etc.). Each letter indicates a wall
    on that side of the cell. Index = y*16 + x.

    Falls back to inferring walls from the graph if grid_data is absent.
    """
    grid_data = getattr(grid_env, "grid_data", None)

    if grid_data is not None and len(grid_data) == 256:
        size = 16
        walls = []
        for y in range(size):
            for x in range(size):
                cell = grid_data[y * size + x]
                # Each wall is shared between two cells; only emit once
                # by checking the S and E sides (avoids duplicates with
                # the neighbor's N and W).
                # Emit E (right edge) and S (bottom edge) to avoid
                # duplicates with neighbor's W and N. Border walls
                # are handled by strokeRect in the renderer.
                if "E" in cell and x < size - 1:
                    walls.append({"from": [x, y], "to": [x + 1, y],
                                  "side": "vertical"})
                if "S" in cell and y < size - 1:
                    walls.append({"from": [x, y], "to": [x, y + 1],
                                  "side": "horizontal"})
        return {
            "bounds": {"min_x": 0, "max_x": size - 1,
                       "min_y": 0, "max_y": size - 1},
            "walls": walls,
        }

    # Fallback: infer walls from graph edges
    g = grid_env.G
    all_positions = sorted(g.nodes())
    if not all_positions:
        return {"cells": [], "walls": []}

    xs = [p[0] for p in all_positions]
    ys = [p[1] for p in all_positions]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    walls = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            pos = (x, y)
            if pos not in g:
                continue
            right = (x + 1, y)
            if right in g:
                if not (g.has_edge(pos, right) or g.has_edge(right, pos)):
                    walls.append({"from": list(pos), "to": list(right),
                                  "side": "vertical"})
            down = (x, y + 1)
            if down in g:
                if not (g.has_edge(pos, down) or g.has_edge(down, pos)):
                    walls.append({"from": list(pos), "to": list(down),
                                  "side": "horizontal"})

    return {
        "bounds": {"min_x": min_x, "max_x": max_x,
                   "min_y": min_y, "max_y": max_y},
        "walls": walls,
    }


@dataclass
class TraceEvent:
    """One MCTS iteration's worth of trace data."""
    iteration: int
    phase: str  # "select_expand_rollout", "complete_node", "dead_end"

    # Selection
    selected_node_id: int | None = None
    selection_path: list[int] = field(default_factory=list)
    lcb_scores: dict[int, float] = field(default_factory=dict)

    # Expansion
    expanded_child_id: int | None = None
    target_edge: tuple | None = None  # (parent_id, child_id) in plan
    candidates_count: int = 0
    candidate_positions: list[list] = field(default_factory=list)
    action: dict | None = None  # serialized action
    expansion_closed_edges: list[list] = field(default_factory=list)

    # Rollout
    rollout_steps: list[dict] = field(default_factory=list)
    rollout_cost: float | None = None
    rollout_completed: bool = False

    # Backprop
    backprop_cost: float | None = None

    # State after this iteration
    best_cost: float | None = None
    tree_node_count: int = 0
    complete_plans_found: int = 0

    # Plan snapshot for the expanded child (or selected node)
    plan_snapshot: dict | None = None


class MCTSTracer:
    """Records MCTS execution trace for visualization."""

    def __init__(self):
        self.events: list[TraceEvent] = []
        self.tree_nodes: dict[int, dict] = {}  # node_id -> metadata
        self.tree_edges: list[dict] = []  # parent_id -> child_id
        self._node_id_map: dict[int, int] = {}  # id(MCTSNode) -> int id
        self._next_id = 0
        self.grid_data: dict | None = None
        self.initial_state: dict | None = None
        self.initial_plan: dict | None = None

    def set_grid(self, grid_env: GridEnv, state):
        """Record static grid data once at the start."""
        self.grid_data = _serialize_grid(grid_env)
        self.initial_state = {
            "target": list(state.target),
            "target_robot": _serialize_robot(state.target_robot),
            "helpers": [_serialize_robot(h) for h in state.helpers],
        }

    def get_node_id(self, node) -> int:
        """Get or assign a stable integer ID for an MCTSNode."""
        obj_id = id(node)
        if obj_id not in self._node_id_map:
            self._node_id_map[obj_id] = self._next_id
            self._next_id += 1
        return self._node_id_map[obj_id]

    def register_node(self, node, parent=None):
        """Register a new MCTS tree node."""
        nid = self.get_node_id(node)
        self.tree_nodes[nid] = {
            "id": nid,
            "visit_count": node.visit_count,
            "total_cost": node.total_cost,
            "avg_cost": node.avg_cost,
            "action": self._serialize_action(node.action),
            "is_complete": node.plan.is_complete(),
            "plan_cost": node.plan.cost(),
        }
        if parent is not None:
            pid = self.get_node_id(parent)
            self.tree_edges.append({"source": pid, "target": nid})

    def update_node(self, node):
        """Update visit stats for an existing node."""
        nid = self.get_node_id(node)
        if nid in self.tree_nodes:
            self.tree_nodes[nid]["visit_count"] = node.visit_count
            self.tree_nodes[nid]["total_cost"] = node.total_cost
            self.tree_nodes[nid]["avg_cost"] = node.avg_cost

    def _serialize_action(self, action):
        if action is None:
            return None
        parent_id, child_id, bn_pos, sp_pos, helper_color, parent_sp = action
        return {
            "parent_id": parent_id,
            "child_id": child_id,
            "bn_pos": _serialize_pos(bn_pos),
            "sp_pos": _serialize_pos(sp_pos),
            "helper_color": helper_color,
            "parent_sp": _serialize_pos(parent_sp),
        }

    def record_event(self, event: TraceEvent):
        self.events.append(event)

    def to_json(self) -> dict:
        """Serialize the full trace to a JSON-compatible dict."""
        return {
            "grid": self.grid_data,
            "initial_state": self.initial_state,
            "initial_plan": self.initial_plan,
            "tree_nodes": list(self.tree_nodes.values()),
            "tree_edges": self.tree_edges,
            "events": [self._event_to_dict(e) for e in self.events],
        }

    def _event_to_dict(self, e: TraceEvent) -> dict:
        return {
            "iteration": e.iteration,
            "phase": e.phase,
            "selected_node_id": e.selected_node_id,
            "selection_path": e.selection_path,
            "lcb_scores": {str(k): v for k, v in e.lcb_scores.items()},
            "expanded_child_id": e.expanded_child_id,
            "target_edge": list(e.target_edge) if e.target_edge else None,
            "candidates_count": e.candidates_count,
            "candidate_positions": e.candidate_positions,
            "action": e.action,
            "expansion_closed_edges": e.expansion_closed_edges,
            "rollout_steps": e.rollout_steps,
            "rollout_cost": e.rollout_cost,
            "rollout_completed": e.rollout_completed,
            "backprop_cost": e.backprop_cost,
            "best_cost": e.best_cost if e.best_cost != float("inf") else None,
            "tree_node_count": e.tree_node_count,
            "complete_plans_found": e.complete_plans_found,
            "plan_snapshot": e.plan_snapshot,
        }

    def dump_live(self, path: str | Path = "mcts_live_trace.json"):
        """Write current trace state to a JSON file for live debugging.

        Call this after each iteration to update the file that the
        live-mode visualizer polls.
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_json(), f)

    def save(self, path: str | Path):
        """Write trace to a JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_json(), f)
