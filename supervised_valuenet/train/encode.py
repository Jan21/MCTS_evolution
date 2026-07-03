"""Encode a (decision-state, candidate) record as a grid tensor [C, 16, 16].

The partial plan is represented purely as grid channels (no DAG): board walls,
robot positions, the open segment being decided, the candidate being scored, and
the committed-so-far context. The candidate triple sits in its own channels, so
the value query is explicit and there is no pairing ambiguity.

Encoders are registered by name so the channel scheme is a sweepable knob.
`encode(record, variant)` returns a float32 numpy array; channel count comes from
`CHANNELS[variant]`.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from simulate import wall_sets, slide

GRID = 16
ENV_DIR = Path(__file__).resolve().parent.parent / "environments"


@lru_cache(maxsize=4096)
def walls_for(env_id: int):
    """(walls_right, walls_down) for an env. Caches only the small wall sets so
    thousands of boards fit in memory without re-reading pkls each epoch."""
    with open(ENV_DIR / f"env_{env_id}.pkl", "rb") as f:
        grid_data = pickle.load(f)["grid_data"]
    return wall_sets(grid_data, GRID)


@lru_cache(maxsize=512)
def _dist_maps(env_id: int):
    """(independent_paths, all_paths) distance dicts (heavy; only for grid_dist)."""
    with open(ENV_DIR / f"env_{env_id}.pkl", "rb") as f:
        p = pickle.load(f)
    return p.get("independent_paths", {}), p.get("all_paths", {})


def _mark(ch, cells):
    for c in cells:
        if c is not None:
            x, y = c
            ch[y, x] = 1.0


def _wall_planes(env_id):
    wr, wd = walls_for(env_id)
    right = np.zeros((GRID, GRID), np.float32)
    left = np.zeros((GRID, GRID), np.float32)
    down = np.zeros((GRID, GRID), np.float32)
    up = np.zeros((GRID, GRID), np.float32)
    for (x, y) in wr:
        right[y, x] = 1.0
        if x + 1 < GRID:
            left[y, x + 1] = 1.0
    for (x, y) in wd:
        down[y, x] = 1.0
        if y + 1 < GRID:
            up[y + 1, x] = 1.0
    return [right, left, down, up]


def _grid_v1(record) -> np.ndarray:
    """15-channel scheme: walls(4) + segment(3) + candidate(4) + context(3) + robots(1)."""
    planes = _wall_planes(record["env_id"])

    def plane(cells):
        ch = np.zeros((GRID, GRID), np.float32)
        _mark(ch, cells)
        return ch

    robots = [record["target_robot"][0]] + [h[0] for h in record["helpers"]]
    planes += [
        plane([record["seg_start"]]),            # mover position
        plane([record["seg_end"]]),              # segment goal
        plane([record["seg_support"]]),          # pinned support, if any
        plane([record["cand_bottleneck"]]),      # candidate bottleneck
        plane([record["cand_support"]]),         # candidate support
        plane([record["cand_helper"][0]]),       # candidate helper position
        plane([record["cand_parent_support"]]),  # parent-edge support
        plane(record["ctx_bottlenecks"]),        # committed bottlenecks
        plane(record["ctx_supports"]),           # committed supports
        plane(record["ctx_open_endpoints"]),     # open frontier
        plane(robots),                           # all robot occupancy
    ]
    return np.stack(planes, 0)


# Fixed robot-colour order for per-colour channels (grid_v2).
COLOR_ORDER = ["Red", "Blue", "Green", "Yellow"]


def _grid_v2(record) -> np.ndarray:
    """18 channels: grid_v1 but per-colour robot identity instead of 1 occupancy.

    Adopts the BERT repo's idea of encoding robot *type* per cell. The single
    "all robots" channel is replaced by one occupancy channel per colour, so the
    model can tell which robot sits where.
    """
    planes = _grid_v1(record)[:-1]  # drop the combined-occupancy channel
    robots = [record["target_robot"]] + record["helpers"]
    per_color = []
    for color in COLOR_ORDER:
        ch = np.zeros((GRID, GRID), np.float32)
        _mark(ch, [pos for pos, c in robots if c == color])
        per_color.append(ch)
    return np.concatenate([planes, np.stack(per_color, 0)], 0)


def _grid_xy(record) -> np.ndarray:
    """17 channels: grid_v1 + explicit normalised x and y coordinate planes.

    Gives the model raw coordinates (the BERT repo feeds x,y indices), so its
    positional reasoning has an absolute reference.
    """
    base = _grid_v1(record)
    xs = np.tile(np.linspace(0, 1, GRID, dtype=np.float32), (GRID, 1))      # col
    ys = np.tile(np.linspace(0, 1, GRID, dtype=np.float32)[:, None], (1, GRID))  # row
    return np.concatenate([base, xs[None], ys[None]], 0)


@lru_cache(maxsize=4096)
def _slide_fields(env_id: int):
    """8 board-only channels: for each cell and direction, the (dx, dy) to its
    slide-stop, normalised. This is the game's one-step transition structure --
    learned-on, not a hand-crafted value feature. Cached per board (walls only).
    """
    wr, wd = walls_for(env_id)
    planes = []
    for d in ("up", "down", "left", "right"):
        dxp = np.zeros((GRID, GRID), np.float32)
        dyp = np.zeros((GRID, GRID), np.float32)
        for y in range(GRID):
            for x in range(GRID):
                sx, sy = slide((x, y), d, frozenset(), wr, wd, GRID)
                dxp[y, x] = (sx - x) / GRID
                dyp[y, x] = (sy - y) / GRID
        planes += [dxp, dyp]
    return np.stack(planes, 0)


def _grid_slide(record) -> np.ndarray:
    """23 channels: grid_v1 (walls + positions) + slide-stop displacement fields.

    Gives the transformer the slide DYNAMICS (no distances/costs) so it can learn
    reachability and cost-to-go itself, instead of rediscovering physics.
    """
    return np.concatenate([_grid_v1(record), _slide_fields(record["env_id"])], 0)


_DNORM = 30.0   # distance normaliser; unreachable -> 1.0 (far)


def _dist_field(dmap, anchor, to_anchor):
    """16x16 field of shortest-path length between each cell and `anchor`.

    to_anchor=True -> dist(cell, anchor); else dist(anchor, cell).
    Normalised to ~[0,1]; missing/unreachable -> 1.0.
    """
    field = np.ones((GRID, GRID), np.float32)
    if anchor is None:
        return field
    a = tuple(anchor)
    for y in range(GRID):
        for x in range(GRID):
            key = ((x, y), a) if to_anchor else (a, (x, y))
            d = dmap.get(key)
            if d is not None:
                field[y, x] = min(d / _DNORM, 1.0)
    return field


def _grid_dist(record) -> np.ndarray:
    """18 channels: grid_v1 + 3 heuristic distance fields (bootstrap from
    subgoal_score's components):
      - independent dist (cell -> goal)        [bottleneck->goal term]
      - relaxed dist (mover -> cell)           [target->bottleneck term]
      - relaxed dist (candidate helper -> cell)[helper->support term]
    """
    base = _grid_v1(record)
    indep, allp = _dist_maps(record["env_id"])
    to_goal = _dist_field(indep, record["seg_end"], to_anchor=True)
    from_mover = _dist_field(allp, record["seg_start"], to_anchor=False)
    from_helper = _dist_field(allp, record["cand_helper"][0], to_anchor=False)
    return np.concatenate([base, to_goal[None], from_mover[None], from_helper[None]], 0)


ENCODERS = {
    "grid_v1": _grid_v1,
    "grid_v2": _grid_v2,
    "grid_xy": _grid_xy,
    "grid_dist": _grid_dist,
    "grid_slide": _grid_slide,
}
CHANNELS = {
    "grid_v1": 15,
    "grid_v2": 18,
    "grid_xy": 17,
    "grid_dist": 18,
    "grid_slide": 23,
}


def encode(record, variant="grid_v1") -> np.ndarray:
    return ENCODERS[variant](record)


# --- graph + node-feature builders (were train/gnn.py; kept here so the looped
# transformer has no GNN/torch_geometric dependency) --------------------------

def _graph(env_id):
    """Slide-graph edges + edge type for a board: (edge_index[2,E], etype[E,1]).

    etype = 1.0 for dependent slides (stop only because a blocker robot is
    placed), 0.0 for independent slides (stop at a wall on their own).
    """
    with open(ENV_DIR / f"env_{env_id}.pkl", "rb") as f:
        G = pickle.load(f)["grid_graph"]
    src, dst, etype = [], [], []
    for u, v, d in G.edges(data=True):
        src.append(u[1] * GRID + u[0]); dst.append(v[1] * GRID + v[0])
        etype.append(1.0 if "dependent" in d else 0.0)
    return (torch.tensor([src, dst], dtype=torch.long),
            torch.tensor(etype).unsqueeze(-1))


def _node_features(r):
    """Per-cell 9-channel binary markers for one candidate decision. [256,9]."""
    f = np.zeros((GRID * GRID, 9), np.float32)
    def idx(p): return p[1] * GRID + p[0]
    f[idx(r["seg_start"]), 0] = 1
    f[idx(r["seg_end"]), 1] = 1
    f[idx(r["cand_bottleneck"]), 2] = 1
    f[idx(r["cand_support"]), 3] = 1
    f[idx(r["cand_helper"][0]), 4] = 1
    for h in r["helpers"]:
        f[idx(h[0]), 5] = 1
    # partial-plan context: other open segments + already-committed subgoal cells.
    # cost_to_go is WHOLE-plan completion, so these pending parts shift the value;
    # without them the readout can't see the rest of the plan it must complete.
    for p in r.get("ctx_open_endpoints", []):
        f[idx(p), 6] = 1
    for p in r.get("ctx_bottlenecks", []):
        f[idx(p), 7] = 1
    for p in r.get("ctx_supports", []):
        f[idx(p), 8] = 1
    return torch.from_numpy(f)
