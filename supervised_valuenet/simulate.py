"""Ground-truth slide simulator for Ricochet Robots.

Independent of the precomputed instance graph: every slide is derived directly
from the board wall layout (`grid_data`). Used to verify that proposed subgoals
and complete plans are physically realizable.

Realizability is checked "up to blocker-clearing": a segment is simulated with
only its intended support robot present. Incidental blocking by other robots is
ignored (they are assumed to slide out of the way), per the project's stance
that a plan need not resolve robot-robot interference.
"""
from __future__ import annotations

from collections import deque
from math import isqrt


def _board_size(grid_data, size):
    """Board side length: explicit `size`, else inferred from the wall layout.
    Inference is safer than an env var — a disagreeing global would mis-decode
    boards silently."""
    if size is not None:
        return size
    n = isqrt(len(grid_data))
    assert n * n == len(grid_data), f"grid_data length {len(grid_data)} not a square"
    return n


DIRECTIONS = ("up", "down", "left", "right")


def wall_sets(grid_data, size=None):
    """Return (walls_right, walls_down) as sets of (x, y) = (col, row).

    walls_right(x, y): wall on the right edge of (x, y), between (x,y)/(x+1,y).
    walls_down(x, y):  wall on the bottom edge of (x, y), between (x,y)/(x,y+1).
    """
    size = _board_size(grid_data, size)
    walls_right, walls_down = set(), set()
    for idx, cell in enumerate(grid_data):
        col, row = idx % size, idx // size
        if "E" in cell:
            walls_right.add((col, row))
        if "S" in cell:
            walls_down.add((col, row))
        if "W" in cell and col > 0:
            walls_right.add((col - 1, row))
        if "N" in cell and row > 0:
            walls_down.add((col, row - 1))
    return walls_right, walls_down


def slide(pos, direction, blockers, walls_right, walls_down, size=16):
    """Slide a robot from `pos` in `direction` until a wall or blocker stops it.

    `blockers` is a set of occupied cells (other robots). Returns the stop cell
    (equal to `pos` if it cannot move at all).
    """
    x, y = pos
    d = direction.lower()
    while True:
        can_leave = True
        if d == "up":
            if y == 0 or (x, y - 1) in walls_down:
                can_leave = False
            nxt = (x, y - 1)
        elif d == "down":
            if y == size - 1 or (x, y) in walls_down:
                can_leave = False
            nxt = (x, y + 1)
        elif d == "left":
            if x == 0 or (x - 1, y) in walls_right:
                can_leave = False
            nxt = (x - 1, y)
        elif d == "right":
            if x == size - 1 or (x, y) in walls_right:
                can_leave = False
            nxt = (x + 1, y)
        else:
            raise ValueError(f"bad direction {direction!r}")

        if not can_leave or nxt in blockers:
            return (x, y)
        x, y = nxt


def segment_realizable(grid_env, start, end, support_pos, walls_right, walls_down, size=None):
    """True if a robot can get from `start` to `end` (blocker-clearing + ordering).

    Independent of the plan's cost oracle. For an independent segment, the
    target must reach `end` by pure wall slides. For a supported segment the
    move order is: slide independently to some pre-bottleneck cell `u`, then the
    support robot is placed at `support_pos`, then one slide stops at `end`.
    Candidate `u` cells come from the (geometry-certified) dependent edges into
    `end`; the approach and the final stop are both re-simulated from walls.
    """
    size = _board_size(grid_env.grid_data, size)
    if start == end:
        return True
    if support_pos is None:
        return segment_moves(start, end, None, walls_right, walls_down, size) is not None
    sup = frozenset({support_pos})
    for u, v, d in grid_env.G.in_edges(end, data=True):
        if d.get("dependent") != support_pos:
            continue
        # final supported stop u -> end must be real
        if not any(slide(u, dr, sup, walls_right, walls_down, size) == end
                   for dr in DIRECTIONS):
            continue
        # independent approach start -> u (support not yet placed)
        if u == start or segment_moves(start, u, None, walls_right, walls_down, size) is not None:
            return True
    return False


def verify_plan(plan, grid_env, state, size=None):
    """Check a complete PartialPlan is realizable up to blocker-clearing.

    Each physical segment is re-simulated from the wall layout (not the cost
    oracle), with correct support-placement ordering. Returns (ok, report);
    ok is False if any segment cannot be physically realized.
    """
    from validate_plan import _physical_movement, STRUCTURAL_EDGE_PAIRS

    size = _board_size(grid_env.grid_data, size)
    wr, wd = wall_sets(grid_env.grid_data, size)
    g = plan.g
    report = []
    ok = True
    for u, v in g.edges():
        ut, vt = g.nodes[u].get("ntype"), g.nodes[v].get("ntype")
        if (ut, vt) in STRUCTURAL_EDGE_PAIRS:
            continue
        start, end, support = _physical_movement(g, parent=u, child=v)
        real = (start is not None and end is not None
                and segment_realizable(grid_env, start, end, support, wr, wd, size))
        if not real:
            ok = False
        report.append({
            "edge": (u, v), "start": start, "end": end,
            "support": support, "realizable": real,
            "plan_cost": g.edges[u, v].get("cost"),
        })
    return ok, report


def segment_moves(start, end, support_pos, walls_right, walls_down, size=16):
    """Minimum slides to get a robot from `start` to `end`.

    Only the support robot (if any) is on the board as a stopper. All other
    robots are ignored (blocker-clearing assumption). Returns the move count,
    or None if `end` is unreachable this way.
    """
    blockers = frozenset() if support_pos is None else frozenset({support_pos})
    if start == end:
        return 0
    seen = {start}
    q = deque([(start, 0)])
    while q:
        pos, dist = q.popleft()
        for d in DIRECTIONS:
            nxt = slide(pos, d, blockers, walls_right, walls_down, size)
            if nxt == pos or nxt in seen:
                continue
            if nxt == end:
                return dist + 1
            seen.add(nxt)
            q.append((nxt, dist + 1))
    return None
