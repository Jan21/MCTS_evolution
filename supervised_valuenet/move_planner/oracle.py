"""Exact optimal move-count oracle for the move-based formulation.

Moves are unit cost, so shortest path over joint robot states is optimal. A blind
BFS is hopeless here (a 4-robot component is millions of states and optimal
solutions run to 10+ moves), so we use **A\\* with the classic Ricochet-Robots
admissible heuristic**: the number of slides the target robot alone needs to reach
the goal if a blocker were available at *every* cell (`relaxed_target_dist`). This
is a lower bound on the target robot's moves, hence on the total move count, so
A\\* returns the true optimum and expands very few nodes.

Two entry points:

- `solve(positions, ...)`  -- optimal cost + an optimal path (parent pointers).
- `label_trajectory(...)`  -- roll out one optimal path and, at every state on it,
  compute the exact cost-to-go of each legal successor (a fast child A\\* solve) to
  get the full **optimal-move set** (policy target) alongside the state's
  cost-to-go (value target). One instance yields d*+1 labelled decision states.

Unsolvable / pathological instances (target cannot reach the goal even in the
relaxation, or the search exceeds `max_expansions`) return `None` and are skipped
by the generator, exactly as the subgoal generator drops its failures.
"""
from __future__ import annotations

import heapq
from collections import deque

from simulate import DIRECTIONS, slide

INF = 1 << 30

# unit steps for the relaxed heuristic graph
_STEP = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def _can_step(x, y, d, wr, wd, size):
    """One-cell step with wall/edge check (same physics as simulate.slide). None if blocked."""
    if d == "up":
        if y == 0 or (x, y - 1) in wd:
            return None
        return (x, y - 1)
    if d == "down":
        if y == size - 1 or (x, y) in wd:
            return None
        return (x, y + 1)
    if d == "left":
        if x == 0 or (x - 1, y) in wr:
            return None
        return (x - 1, y)
    if d == "right":
        if x == size - 1 or (x, y) in wr:
            return None
        return (x + 1, y)
    raise ValueError(d)


def relaxed_target_dist(target, wr, wd, size=16):
    """Admissible heuristic field: min slides for the target robot ALONE to reach
    `target`, assuming a blocker is available at every cell (so a slide may stop at
    any cell along its ray). Reverse BFS from the goal over the relaxed move graph:
    a cell `c` reaches `v` in one move if, going some direction from `c`, the ray
    passes through `v` -- i.e. `c` lies straight behind `v` with no wall between.
    """
    dist = {target: 0}
    q = deque([target])
    while q:
        v = q.popleft()
        dv = dist[v]
        # predecessors of v: walk outward from v in each direction; every cell we
        # can reach steps back to is a state that could slide INTO v.
        for d in DIRECTIONS:
            x, y = v
            while True:
                nc = _can_step(x, y, d, wr, wd, size)
                if nc is None:
                    break
                x, y = nc
                if (x, y) not in dist:
                    dist[(x, y)] = dv + 1
                    q.append((x, y))
    return dist


def _successors(positions, wr, wd, size):
    """(robot_slot, dir_idx, new_positions) for every non-no-op move from `positions`."""
    out = []
    n = len(positions)
    for i in range(n):
        pos = positions[i]
        blockers = frozenset(positions[j] for j in range(n) if j != i)
        for di, d in enumerate(DIRECTIONS):
            nxt = slide(pos, d, blockers, wr, wd, size)
            if nxt != pos:
                out.append((i, di, positions[:i] + (nxt,) + positions[i + 1:]))
    return out


def solve(positions, target_idx, target, wr, wd, size=16,
          hdist=None, max_expansions=200_000, want_path=False, cost_cap=INF):
    """A\\* to the goal (target robot on `target`). Returns cost, or (cost, path) if
    `want_path`. `path` is a list of `(positions, action)` from start to goal, the
    goal entry having `action=None`. `cost_cap` prunes any node with `f > cost_cap`
    (used to cheaply test "does a solution of length <= cap exist?"). Returns `None`
    (or `(None, None)`) if unsolvable, capped, or exceeding `max_expansions`."""
    if hdist is None:
        hdist = relaxed_target_dist(target, wr, wd, size)
    h0 = hdist.get(positions[target_idx], INF)
    if h0 >= INF or h0 > cost_cap:
        return (None, None) if want_path else None
    if positions[target_idx] == target:
        return (0, [(positions, None)]) if want_path else 0

    cnt = 0
    pq = [(h0, 0, cnt, positions)]
    g = {positions: 0}
    came = {positions: (None, None)}
    expansions = 0
    while pq:
        f, gc, _, cur = heapq.heappop(pq)
        if gc != g.get(cur, INF):
            continue  # stale
        if cur[target_idx] == target:
            if not want_path:
                return gc
            path = []
            s = cur
            while s is not None:
                ps, act = came[s]
                path.append((s, None if ps is None else act))
                s = ps
            path.reverse()
            return gc, path
        expansions += 1
        if expansions > max_expansions:
            return (None, None) if want_path else None
        ng = gc + 1
        for i, di, child in _successors(cur, wr, wd, size):
            if ng < g.get(child, INF):
                h = hdist.get(child[target_idx], INF)
                if h >= INF or ng + h > cost_cap:
                    continue
                g[child] = ng
                cnt += 1
                came[child] = (cur, (i, di))
                heapq.heappush(pq, (ng + h, ng, cnt, child))
    return (None, None) if want_path else None


def label_trajectory(positions, target_idx, target, wr, wd, size=16,
                     hdist=None, max_expansions=200_000, full_policy=True,
                     full_policy_max_ctg=None):
    """One optimal rollout with per-decision labels.

    Returns `(d_star, records)` or `None`. Each record is a dict:
        positions, cost_to_go, best_moves[(slot,dir)], legal_moves[(slot,dir)], depth
    If `full_policy`, every legal successor is solved so `best_moves` is the FULL
    optimal-move set; otherwise only the move taken on this optimal path is marked
    (cheap behaviour-cloning target). `full_policy_max_ctg` caps the full-set
    computation to states whose cost-to-go is small (the many cheap decisions),
    falling back to the taken move for deep states (where the radius-`here` child
    checks are expensive) -- keeps generation fast while most decisions still get
    the exact optimal-move set.
    """
    if hdist is None:
        hdist = relaxed_target_dist(target, wr, wd, size)
    res = solve(positions, target_idx, target, wr, wd, size, hdist,
                max_expansions, want_path=True)
    if not res or res[0] is None:
        return None
    d_star, path = res

    def is_optimal_child(child, here):
        """True iff the successor lies on an optimal solution, i.e. its cost-to-go
        is exactly `here-1`. A successor always has cost-to-go >= here-1, so a
        cost-capped A\\* of radius here-1 decides this cheaply."""
        if child[target_idx] == target:
            return here == 1
        return solve(child, target_idx, target, wr, wd, size, hdist,
                     max_expansions, cost_cap=here - 1) == here - 1

    records = []
    for depth, (state, _incoming) in enumerate(path):
        here = d_star - depth  # exact cost-to-go of an on-optimal-path state
        if state[target_idx] == target:
            break  # goal: no decision
        # the move actually taken on this path leaves `state` -> it is the NEXT
        # path entry's incoming action.
        taken = path[depth + 1][1] if depth + 1 < len(path) else None
        succ = _successors(state, wr, wd, size)
        legal = [(i, di) for i, di, _ in succ]
        do_full = full_policy and (full_policy_max_ctg is None or here <= full_policy_max_ctg)
        if do_full:
            best = [(i, di) for i, di, child in succ if is_optimal_child(child, here)]
        else:
            best = [taken] if taken is not None else []
        records.append({
            "positions": state, "cost_to_go": here,
            "best_moves": best, "legal_moves": legal, "depth": depth,
            "full": do_full,   # is best_moves the FULL optimal set (else BC single move)
        })
    return d_star, records


# cheap public alias for the eval optimal reference
def optimal_cost(positions, target_idx, target, wr, wd, size=16,
                 hdist=None, max_expansions=200_000):
    return solve(positions, target_idx, target, wr, wd, size, hdist, max_expansions)
