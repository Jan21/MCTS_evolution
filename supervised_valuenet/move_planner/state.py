"""Full-board Ricochet-Robots state + primitive-move transitions.

The move-based reformulation replaces the subgoal DAG with the raw game:
a **state** is the joint position of every robot, an **action** is
`(robot, direction)`, and applying it slides that one robot until a wall or
another robot stops it (`simulate.slide`, with ALL other robots as blockers --
the true multi-robot physics, not the blocker-clearing relaxation used by the
subgoal solver). The goal is the single target robot on the single target cell.

Everything here is deliberately tiny and hashable so it can be the node type of
a BFS/A* over move space and the key of a transposition/label table.

    positions          : tuple[(x,y), ...] robot cells in COLOR_ORDER slot order
    action             : (robot_slot:int, dir:int)   dir indexes DIRECTIONS
    target_idx         : slot of the robot that must reach the goal cell
"""
from __future__ import annotations

from simulate import DIRECTIONS, slide, wall_sets
from nn.gen_grids import COLORS, GRID

# Canonical robot slot order: gen_grids.COLORS (the palette's first RR_ROBOTS
# names), so the same ordering is used by the physics, the encoder channels and
# the policy head (slot i == COLOR_ORDER[i]).
COLOR_ORDER = COLORS
NUM_ROBOTS = len(COLOR_ORDER)
NUM_DIRS = len(DIRECTIONS)  # 4: up, down, left, right


def board_walls(grid_env, size: int = GRID):
    """(walls_right, walls_down) for a loaded GridEnv (from its raw grid_data)."""
    return wall_sets(grid_env.grid_data, size)


def positions_of(state, color_order=COLOR_ORDER) -> tuple:
    """Canonical robot-position tuple (COLOR_ORDER slots) for a GridEnv.State."""
    by_color = {r.color: tuple(int(v) for v in r.position) for r in state.all_robots}
    return tuple(by_color[c] for c in color_order)


def target_slot(state, color_order=COLOR_ORDER) -> int:
    return color_order.index(state.target_robot.color)


def is_goal(positions: tuple, target_idx: int, target: tuple) -> bool:
    return positions[target_idx] == target


def legal_moves(positions: tuple, wr, wd, size: int = GRID):
    """All non-no-op moves from `positions`.

    Yields `(robot_slot, dir_idx, new_positions)`. A move is a slide of one robot
    with every OTHER robot acting as a blocker; a slide that cannot leave its cell
    (returns the same cell) is a no-op and is dropped.
    """
    out = []
    n = len(positions)
    for i in range(n):
        pos = positions[i]
        blockers = frozenset(positions[j] for j in range(n) if j != i)
        for di, d in enumerate(DIRECTIONS):
            nxt = slide(pos, d, blockers, wr, wd, size)
            if nxt == pos:
                continue  # blocked immediately -> illegal
            new_positions = positions[:i] + (nxt,) + positions[i + 1:]
            out.append((i, di, new_positions))
    return out


def apply_move(positions: tuple, robot_slot: int, dir_idx: int, wr, wd, size: int = GRID):
    """Apply one `(robot_slot, dir_idx)` move; returns new positions (or None no-op)."""
    pos = positions[robot_slot]
    blockers = frozenset(positions[j] for j in range(len(positions)) if j != robot_slot)
    nxt = slide(pos, DIRECTIONS[dir_idx], blockers, wr, wd, size)
    if nxt == pos:
        return None
    return positions[:robot_slot] + (nxt,) + positions[robot_slot + 1:]
