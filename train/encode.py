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

from simulate import wall_sets

GRID = 16
ENV_DIR = Path(__file__).resolve().parent.parent / "environments"


@lru_cache(maxsize=256)
def walls_for(env_id: int):
    """(walls_right, walls_down) for an env, loaded straight from its grid_data."""
    with open(ENV_DIR / f"env_{env_id}.pkl", "rb") as f:
        grid_data = pickle.load(f)["grid_data"]
    return wall_sets(grid_data, GRID)


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


ENCODERS = {
    "grid_v1": _grid_v1,
}
CHANNELS = {
    "grid_v1": 15,
}


def encode(record, variant="grid_v1") -> np.ndarray:
    return ENCODERS[variant](record)
