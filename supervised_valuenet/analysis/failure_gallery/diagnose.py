"""Diagnose WHY each structural-failure puzzle is outside the subgoal language.

Input: the forward planner's real move sequences (run_forward.py output).
For every move we work out what stopped the slide; from that, which of the
plan vocabulary's hard constraints the solution violates. The constraints,
per the frozen probe semantics and analysis/b1_design.md:

  V1  A support (stopper) cell must be wall-holdable: only cells where a
      robot can stop by a walls-only slide may be named as supports
      (`GridEnv._has_adjacent_wall`). We recompute the predicate here from
      wall geometry alone: cell c is wall-holdable iff some walls-only slide
      stops on c.
  V2  A placed support never moves again (single static support per helper).
  V3  One robot, one plan role (one leaf per robot): a robot cannot be
      recruited for a second support job, and the target robot (which always
      owns leaf_0) can never be a support at all.
  V4  Segment costs assume blockers slide out of the way for free
      (blocker-clearing), but the vocabulary has no move that actually
      clears an idle robot out of a corridor.

Gates raised per instance (an instance can raise several):

  transient_support  a slide bounces off a robot standing on a cell that is
                     NOT wall-holdable (violates V1)
  vacate             a robot that served as a stopper later moves away, and
                     its old cell is needed (traversed or landed on) by a
                     later move (violates V2)
  relocate           a robot serves as a stopper, then moves, then serves as
                     a stopper again somewhere else (violates V2+V3)
  shared_support     one robot, without moving, stops the slides of two
                     different movers -- two plan roles from a single
                     parking spot (violates V3: one leaf per robot)
  target_support     a slide bounces off the TARGET robot before the target
                     has finished (violates V3: the target owns leaf_0)
  target_clears      another robot's route needs the target robot's start
                     cell, so the target must first step aside -- an extra
                     move with no plan word: exact legs are shortest paths,
                     and a detour stop cannot be a plan node unless it is a
                     supported bottleneck (V3/V4)
  idle_clearing      a robot that never acts as a stopper moves so that a
                     later slide can use the corridor/cell it vacated
                     (violates V4: the "free" blocker-clearing is real work)
  blocked_direct     a walls-only route from the target's start to the goal
                     exists on paper, so the search pins the root segment to
                     that exact walk and never proposes an alternative -- but
                     in the real game a robot blocks the walk (V4). When this
                     fires all other gates are suppressed: unblocking the
                     paper walk is by itself enough to play the plan the
                     language already builds. Verified against the frozen
                     probe: at base scale the gate fires on exactly the
                     NO_REALIZABLE_PLAN rows with complete_tested=1 (the
                     probe built only the walk), and on no others.

Not a gate, deliberately: "the mover crosses the stopper's cell before the
stopper arrives." That timing is expressible: the cost oracle prices paths
with the support as a non-blocker (the b1_design worked example prices the
mover straight through its own support cell), and the realizer's two-phase
split plays the approach before the support is placed. Instances where no
gate fires are reported as the residual family: their obstruction sits in
plan ASSEMBLY order (e.g. a claimed route's stopper must already be part of
the plan when the claim is made -- the fix-3/fix-4 invariants of
SOURCE_OF_TRUTH 5a), which a move trace cannot positively detect.

Caveat, stated once and honestly: gates are read off ONE forward solution.
The probe verdict (no expressible plan exists) is solution-independent; the
gate assignment says which missing words THIS real solution needed, which is
evidence for the family, not a proof that every solution needs that exact
word.
"""
from __future__ import annotations

import json
import os
import pickle
from collections import Counter
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


# ---------------------------------------------------------------------------
# board geometry (walls read straight from the board pickle; no GridEnv)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def board_walls(env_dir: str, env_id: int, grid: int):
    from simulate import wall_sets
    with open(Path(env_dir) / f"env_{env_id}.pkl", "rb") as f:
        grid_data = pickle.load(f)["grid_data"]
    return wall_sets(grid_data, grid)


@lru_cache(maxsize=None)
def wall_holdable_cells(env_dir: str, env_id: int, grid: int):
    """Cells where a walls-only slide can stop = the cells the old vocabulary
    admits as support cells (equivalent of `_has_adjacent_wall`: at least one
    incoming weight-1 edge in the instance graph)."""
    from simulate import slide
    wr, wd = board_walls(env_dir, env_id, grid)
    stops = set()
    for x in range(grid):
        for y in range(grid):
            for d in DIRS:
                s = slide((x, y), d, frozenset(), wr, wd, grid)
                if s != (x, y):
                    stops.add(s)
    return stops


def corridor(a, b):
    """Cells strictly between a and b on a straight slide."""
    (x0, y0), (x1, y1) = a, b
    if x0 == x1:
        step = 1 if y1 > y0 else -1
        return [(x0, y) for y in range(y0 + step, y1, step)]
    step = 1 if x1 > x0 else -1
    return [(x, y0) for x in range(x0 + step, x1, step)]


def paper_walk(start, goal, wr, wd, grid):
    """One shortest walls-only slide path start -> goal (robots ignored), as
    the list of stop cells [start, ..., goal]; None if unreachable. This is
    the walk the plan language pins when it exists (the blocked_direct
    story); used for annotation only."""
    from simulate import slide
    from collections import deque
    if start == goal:
        return [start]
    prev = {start: None}
    q = deque([start])
    while q:
        pos = q.popleft()
        for d in DIRS:
            nxt = slide(pos, d, frozenset(), wr, wd, grid)
            if nxt == pos or nxt in prev:
                continue
            prev[nxt] = pos
            if nxt == goal:
                path = [goal]
                while path[-1] is not None:
                    path.append(prev[path[-1]])
                return path[-2::-1]
            q.append(nxt)
    return None


def walk_blockers(walk, positions, target_idx):
    """Robots (slots) standing on the paper walk's corridors/stops at the
    start position -- the bodies the pinned walk would slide through."""
    cells = set()
    for a, b in zip(walk, walk[1:]):
        cells.update(corridor(a, b))
        cells.add(b)
    return [j for j, p in enumerate(positions)
            if j != target_idx and tuple(p) in cells]


# ---------------------------------------------------------------------------
# per-instance analysis
# ---------------------------------------------------------------------------

def trace_moves(row, env_dir, grid, path_key="path"):
    """Replay and annotate each move: origin, destination, corridor, and what
    stopped the slide (wall vs robot, and which robot on which cell)."""
    from simulate import slide
    wr, wd = board_walls(env_dir, row["env_id"], grid)
    pos = [tuple(p) for p in row["positions"]]
    moves = []
    for t, (slot, d) in enumerate(row[path_key]):
        a = pos[slot]
        blockers = frozenset(p for j, p in enumerate(pos) if j != slot)
        b = slide(pos[slot], d, blockers, wr, wd, grid)
        assert b != a, f"illegal move {t} in idx {row['idx']}"
        dx, dy = DIRS[d]
        beyond = (b[0] + dx, b[1] + dy)
        stopper = None
        for j, p in enumerate(pos):
            if j != slot and p == beyond:
                stopper = j
        # sanity: if no robot beyond, a wall/border must be there
        if stopper is None:
            w = slide(b, d, frozenset(), wr, wd, grid)
            assert w == b, f"move {t}: stopped by nothing?"
        moves.append({
            "t": t, "slot": slot, "dir": d, "from": a, "to": b,
            "corridor": corridor(a, b),
            "stopper_slot": stopper,
            "support_cell": beyond if stopper is not None else None,
        })
        pos[slot] = b
    return moves


def analyze_row(row, env_dir, grid, path_key="path"):
    """Gate flags + the concrete events behind them, for one solved row."""
    moves = trace_moves(row, env_dir, grid, path_key)
    holdable = wall_holdable_cells(env_dir, row["env_id"], grid)
    tidx = row["target_idx"]
    n = len(moves)
    last_move_of = {}
    for m in moves:
        last_move_of[m["slot"]] = m["t"]

    # times each robot serves as stopper: slot -> [(t, cell)]
    stops_at = {}
    for m in moves:
        if m["stopper_slot"] is not None:
            stops_at.setdefault(m["stopper_slot"], []).append(
                (m["t"], m["support_cell"]))

    # cells needed at time t (corridor + landing cell)
    def needed_after(t, cell):
        for m in moves:
            if m["t"] > t and (cell in m["corridor"] or cell == m["to"]):
                return m["t"]
        return None

    events = []
    gates = set()

    # V1: bounce off a robot on a non-wall-holdable cell
    for m in moves:
        if m["stopper_slot"] is not None and m["support_cell"] not in holdable:
            gates.add("transient_support")
            events.append({
                "gate": "transient_support", "t": m["t"],
                "mover": m["slot"], "stopper": m["stopper_slot"],
                "cell": m["support_cell"],
            })

    # V3 (static): one robot, without moving, stops two different movers
    mover_of = {m["t"]: m["slot"] for m in moves}
    for j, servings in stops_at.items():
        for a in range(len(servings) - 1):
            (t1, c1), (t2, c2) = servings[a], servings[a + 1]
            j_moved_between = any(m["slot"] == j and t1 < m["t"] <= t2
                                  for m in moves)
            if not j_moved_between and mover_of[t1] != mover_of[t2]:
                gates.add("shared_support")
                events.append({
                    "gate": "shared_support", "robot": j,
                    "first": (t1, c1), "again": (t2, c2),
                    "movers": (mover_of[t1], mover_of[t2]),
                })

    # V2/V3: stopper later moves
    for j, servings in stops_at.items():
        first_t, first_cell = servings[0]
        later_moves = [m for m in moves if m["slot"] == j and m["t"] > first_t]
        if not later_moves:
            continue
        # relocate: serves as stopper again after moving
        again = [(t, c) for (t, c) in servings
                 if t > later_moves[0]["t"]]
        if again:
            gates.add("relocate")
            events.append({
                "gate": "relocate", "robot": j,
                "first": servings[0], "again": again[0],
            })
        # vacate: the vacated support cell is needed later
        for lm in later_moves:
            # cell it vacates is its position at that time = lm["from"]
            served_here = [t for (t, c) in servings
                           if c == lm["from"] and t < lm["t"]]
            if not served_here:
                continue
            t_need = needed_after(lm["t"], lm["from"])
            if t_need is not None:
                gates.add("vacate")
                events.append({
                    "gate": "vacate", "robot": j, "cell": lm["from"],
                    "served_t": served_here[0], "vacated_t": lm["t"],
                    "needed_t": t_need,
                })
                break

    # V3: bounce off the target robot (which then still has moves to make)
    for m in moves:
        if m["stopper_slot"] == tidx and last_move_of.get(tidx, -1) > m["t"]:
            gates.add("target_support")
            events.append({
                "gate": "target_support", "t": m["t"],
                "mover": m["slot"], "cell": m["support_cell"],
            })

    # V4: idle-robot clearing move (robot never a stopper before or at its
    # new spot; its vacated cell is used by a later slide)
    for m in moves:
        j = m["slot"]
        if j == tidx:
            continue
        ever_stopper = j in stops_at
        if ever_stopper:
            continue
        t_need = needed_after(m["t"], m["from"])
        if t_need is not None:
            gates.add("idle_clearing")
            events.append({
                "gate": "idle_clearing", "robot": j, "cell": m["from"],
                "moved_t": m["t"], "needed_t": t_need,
            })

    # V3/V4: the target steps aside (vacates its start so another robot's
    # route can use it) and later comes back through its own start cell --
    # a detour no plan node can express
    start_cell = tuple(row["positions"][tidx])
    tgt_moves = [m for m in moves if m["slot"] == tidx]
    if tgt_moves:
        t0 = tgt_moves[0]["t"]
        needed = next((m for m in moves
                       if m["t"] > t0 and m["slot"] != tidx
                       and (start_cell in m["corridor"]
                            or start_cell == m["to"])), None)
        recross = any(start_cell in m["corridor"] or start_cell == m["to"]
                      for m in tgt_moves[1:])
        if needed is not None and recross:
            gates.add("target_clears")
            events.append({
                "gate": "target_clears", "cell": start_cell,
                "cleared_t": t0, "needed_t": needed["t"],
                "by": needed["slot"],
            })

    # V4 (paper walk): walls-only route target start -> goal exists, so the
    # search pins the root segment and never decomposes; a real robot blocks
    # the walk. Suppresses every other gate (see module docstring).
    from simulate import segment_moves
    wr, wd = board_walls(env_dir, row["env_id"], grid)
    fantasy = segment_moves(start_cell, tuple(row["target"]), None,
                            wr, wd, grid)
    if fantasy is not None:
        gates = {"blocked_direct"}
        events = [{"gate": "blocked_direct", "paper_moves": fantasy,
                   "real_moves": n}]

    # annotation helper: pass-through-then-park (the worked-example timing;
    # informative, not a gate by itself -- the two-phase realizer already
    # handles arrive-late when the cell is wall-holdable)
    occupied_from = {}   # (cell) -> first time a robot stops there
    for m in moves:
        occupied_from.setdefault(m["to"], m["t"])
    ptp = []
    for m in moves:
        for c in m["corridor"]:
            t2 = occupied_from.get(c)
            if t2 is not None and t2 > m["t"]:
                ptp.append({"t_through": m["t"], "cell": c, "t_parked": t2})
    return {"moves": moves, "gates": sorted(gates), "events": events,
            "pass_through_then_park": ptp}


# ---------------------------------------------------------------------------
# families: name = the joint signature that decides B1 coverage
# ---------------------------------------------------------------------------

# precedence: gates B1 does not cover first (they decide coverage), then the
# park-covered gates, then Layer 1. An instance is labeled by the first of
# its gates in this order.
FAMILY_ORDER = [
    "target_support", "relocate", "shared_support", "target_clears",
    "vacate", "idle_clearing", "blocked_direct",
    "transient_support",
]

UNCOVERED_GATES = {"target_support", "target_clears", "relocate",
                   "shared_support"}
PARK_GATES = {"vacate", "idle_clearing", "blocked_direct"}

FAMILY_LABEL = {
    "transient_support": "Park on a wall-less cell",
    "vacate": "Stopper must step away after its bounce",
    "relocate": "Same robot needed as stopper twice",
    "shared_support": "One parked robot must stop two different sliders",
    "target_support": "The target robot itself must be the stopper",
    "target_clears": "The target robot must first step out of the way",
    "idle_clearing": "A robot in the way must first clear out",
    "blocked_direct": "The only sayable plan walks through a robot",
    "unsolved": "Forward planner also failed (no solution to read)",
    "none": "No single missing word visible (plan-assembly residual)",
}


def family_of(gates):
    """Primary family = the first gate present in FAMILY_ORDER precedence
    (uncovered-by-B1 gates first, then park-covered gates, then Layer 1)."""
    for g in FAMILY_ORDER:
        if g in gates:
            return g
    return "none"


def b1_coverage(gates):
    """Which of the observed gates the B1 extension covers, per
    analysis/b1_design.md and the park augmentation in skeleton/astar.py:
    Layer 1 = transient support cells; parks = one bounded step-aside /
    post-bounce vacate of a placed helper or idle robot. Re-recruiting a
    used helper, target-robot roles, shared static supports and the
    construction-order timing gate are outside B1 as designed."""
    gates = set(gates)
    uncovered = UNCOVERED_GATES & gates
    need_park = PARK_GATES & gates
    if uncovered:
        return "not covered ({} outside B1)".format(", ".join(sorted(uncovered)))
    if need_park and "transient_support" in gates:
        return "covered only with parks (Layer 1 + park augmentation)"
    if need_park:
        return "covered by park augmentation (step-aside/vacate)"
    if "transient_support" in gates:
        return "covered by Layer 1 (transient support cells)"
    return "unclear (no gate observed)"


# ---------------------------------------------------------------------------
# CLI: print the distribution for one scale
# ---------------------------------------------------------------------------

def load_scale(solutions_file, env_dir, grid):
    data = json.load(open(solutions_file))
    out = []
    for row in data["rows"]:
        rec = {"row": row}
        key = None
        if row.get("solved"):
            key = "path"
        elif row.get("fallback_solved"):
            key = "fallback_path"
        if key:
            rec["analysis"] = analyze_row(row, env_dir, grid, key)
            rec["path_key"] = key
        out.append(rec)
    return data["protocol"], out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--solutions", required=True)
    ap.add_argument("--env-dir", required=True)
    ap.add_argument("--grid", type=int, default=16)
    a = ap.parse_args()
    proto, recs = load_scale(a.solutions, a.env_dir, a.grid)
    print(f"n={len(recs)} solved@budget={sum(1 for r in recs if r['row'].get('solved'))} "
          f"solved@fallback={sum(1 for r in recs if r['row'].get('fallback_solved'))}")
    sigs = Counter()
    fams = Counter()
    for r in recs:
        if "analysis" not in r:
            fams[("unsolved", r["row"]["category"])] += 1
            continue
        g = tuple(r["analysis"]["gates"])
        sigs[(g, r["row"]["category"])] += 1
        fams[(family_of(g), r["row"]["category"])] += 1
    print("\ngate signatures x probe category:")
    for (g, c), n in sorted(sigs.items(), key=lambda kv: -kv[1]):
        print(f"  {n:3d}  {c:20s} {g}")
    print("\nprimary family x probe category:")
    for (f, c), n in sorted(fams.items(), key=lambda kv: -kv[1]):
        print(f"  {n:3d}  {c:20s} {f}")
    print("\nper-instance:")
    for r in recs:
        row = r["row"]
        if "analysis" in r:
            an = r["analysis"]
            print(f"  idx={row['idx']:4d} {row['category'][:7]} "
                  f"d*={row['d_star']:2d} moves={len(row[r['path_key']]):2d} "
                  f"fam={family_of(an['gates']):18s} gates={an['gates']} "
                  f"b1={b1_coverage(an['gates'])}")
        else:
            print(f"  idx={row['idx']:4d} {row['category'][:7]} UNSOLVED")


if __name__ == "__main__":
    main()
