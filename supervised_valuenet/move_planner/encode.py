"""Featurise a move-based decision record for the shared looped transformer.

A record describes one full board *state* (all robot cells + target) and its
labels. We reuse the exact board machinery from `train/encode.py`:
- `walls_for(env_id)` / `_slide_fields(env_id)` -- board-only planes (cached).
- `_graph(env_id)` -- the slide-graph edges used as the attention mask (`_adj`).
- `COLOR_ORDER` -- the canonical robot slot order shared with the physics + heads.

The subgoal-specific 9-channel `_node_features` is replaced by state channels
(R = number of robots, RR_ROBOTS):

    ch 0..R-1   per-slot robot occupancy (COLOR_ORDER slots)
    ch R        the target robot's cell (which robot must reach the goal)
    ch R+1      the target cell (the goal)
    ch R+2..R+9 slide-stop displacement fields (dx,dy per direction; board-only)

The 5-cell candidate gather is replaced downstream by a global-token value read
and a per-robot-cell policy read; this module also exposes the robot cell indices
and the legal-move mask needed for those heads.
"""
from __future__ import annotations

import numpy as np
import torch

from train.encode import GRID, walls_for, _slide_fields, _graph, COLOR_ORDER  # noqa: F401
from simulate import DIRECTIONS, slide

N = GRID * GRID
NUM_SLOTS = len(COLOR_ORDER)          # RR_ROBOTS (4 at defaults)
NUM_DIRS = len(DIRECTIONS)            # 4
STATE_CHANNELS = NUM_SLOTS + 2        # occupancy(R) + target-robot(1) + target-cell(1)
MOVE_CHANNELS = STATE_CHANNELS + 8    # + slide-displacement fields (R+10; 14 at R=4)


def _ix(p) -> int:
    return int(p[1]) * GRID + int(p[0])


def move_node_features(record, with_slide=True) -> torch.Tensor:
    """Per-cell state features `[N, C]` (C = MOVE_CHANNELS with slide fields, else
    STATE_CHANNELS)."""
    robots = record["robots"]                     # R cells, COLOR_ORDER slots
    tidx = record["target_idx"]
    f = np.zeros((N, STATE_CHANNELS), np.float32)
    for slot, pos in enumerate(robots):
        f[_ix(pos), slot] = 1.0
    f[_ix(robots[tidx]), NUM_SLOTS] = 1.0          # target robot
    f[_ix(record["target"]), NUM_SLOTS + 1] = 1.0  # goal cell
    feats = torch.from_numpy(f)
    if with_slide:
        sf = _slide_fields(record["env_id"])       # [8,G,G], cached per board
        sf = torch.from_numpy(np.asarray(sf)).reshape(8, N).transpose(0, 1)  # [N,8]
        feats = torch.cat([feats, sf], dim=1)
    return feats                                   # [N, C]


def x257(record, with_slide=True) -> torch.Tensor:
    """Pad a zero global/scratchpad row: `[N+1, C]` (row N = global token)."""
    feats = move_node_features(record, with_slide)
    return torch.cat([feats, torch.zeros(1, feats.shape[1])], 0)


def robot_cells(record) -> torch.Tensor:
    """Cell index of each robot slot, for the policy head gather. `[R]` long."""
    return torch.tensor([_ix(p) for p in record["robots"]], dtype=torch.long)


def dest_cells(record) -> torch.Tensor:
    """Slide-destination cell of each `(slot, dir)` move. `[R,4]` long.

    This is the 1-step lookahead the policy head needs: picking the optimal move
    means knowing where each robot would STOP if slid, so the head reads the
    destination cell's embedding, not just the robot's current cell. Illegal
    (no-op) moves map to the robot's own cell and are masked out downstream.
    """
    wr, wd = walls_for(record["env_id"])
    pos = [tuple(p) for p in record["robots"]]
    out = torch.zeros(NUM_SLOTS, NUM_DIRS, dtype=torch.long)
    for i, p in enumerate(pos):
        blk = frozenset(pos[j] for j in range(len(pos)) if j != i)
        for di, d in enumerate(DIRECTIONS):
            out[i, di] = _ix(slide(p, d, blk, wr, wd, GRID))
    return out


def legal_mask(record) -> torch.Tensor:
    """Boolean `[R,4]` (slot x dir); True where the move is legal (non no-op)."""
    m = torch.zeros(NUM_SLOTS, NUM_DIRS, dtype=torch.bool)
    for slot, d in record["legal_moves"]:
        m[slot, d] = True
    return m


def policy_target(record) -> torch.Tensor:
    """Soft policy target `[R,4]`: uniform over the optimal first moves."""
    t = torch.zeros(NUM_SLOTS, NUM_DIRS, dtype=torch.float32)
    best = record["best_moves"]
    if best:
        w = 1.0 / len(best)
        for slot, d in best:
            t[slot, d] = w
    return t
