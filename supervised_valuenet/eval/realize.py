"""Realize a COMPLETE backward PartialPlan as primitive robot moves.

Two counts, from loose to strict:

- `abstract_moves`: blocker-clearing count. Every physical segment of the plan
  DAG is verified with `simulate.verify_plan` and costed independently with
  only its intended support robot on the board (`simulate.segment_moves`),
  then summed. This mirrors the plan's own cost model (`plan.cost()` should
  match; divergences are logged) but can under-count true game moves because
  all other robots are assumed to be out of the way.

- `strict_moves`: LEGAL joint-game realization. The plan's physical segments
  are topologically ordered (a helper's move to its support cell before the
  mover segment that bounces off that support; each robot's own segments in
  child->parent chain order; a support robot departs only after its bounce is
  consumed) and executed one segment at a time on the FULL joint state with
  `simulate.slide` physics: only the segment's robot moves, its path found by
  BFS over slides with all other robots at their current cells as blockers.
  Returns the total slide count, or None if any segment is unreachable
  (realization failure). A successful strict realization is a legal move
  sequence that ends with the target robot on the target, so its length is
  always >= the move-optimal d*. The default `two_phase=True` keyword adds
  two bounded fallbacks (each only after the atomic schedule fails): a
  failing supported segment is re-executed in the plan's own approach ->
  place-support -> bounce order, and a unit blocked by a robot the plan
  itself moves (e.g. a stopper placement whose cell the future bouncer
  still stands on) is retried once with that robot's own plan moves
  reordered around the blocked unit; see `strict_moves`.

`prefix_playable` applies the same strict machinery to a PARTIAL plan's
already-fixed segments (Lever A's in-search filter); see its docstring.
"""
from __future__ import annotations

import heapq
from collections import deque

from simulate import (DIRECTIONS, _board_size, slide, wall_sets, segment_moves,
                      verify_plan)
from validate_plan import _physical_movement, STRUCTURAL_EDGE_PAIRS


# ---------------------------------------------------------------------------
# abstract (blocker-clearing) realization
# ---------------------------------------------------------------------------

def _abstract_segment_moves(env, start, end, support, wr, wd, size=16):
    """Blocker-clearing move count for one segment.

    Plain `segment_moves` treats the support as a static blocker for the whole
    path, which misses segments whose independent approach must happen BEFORE
    the support is placed. Mirror `simulate.segment_realizable`'s ordering as
    a second route -- approach without the support to a pre-bottleneck cell,
    then one supported slide -- and take the cheaper of the two.
    """
    plain = segment_moves(start, end, support, wr, wd, size)
    if support is None:
        return plain
    best = plain
    sup = frozenset({tuple(support)})
    end_t, start_t = tuple(end), tuple(start)
    for u, _v, d in env.G.in_edges(end_t, data=True):
        if d.get("dependent") != tuple(support):
            continue
        if not any(slide(u, dr, sup, wr, wd, size) == end_t for dr in DIRECTIONS):
            continue
        approach = 0 if u == start_t else segment_moves(start_t, u, None, wr, wd, size)
        if approach is None:
            continue
        if best is None or approach + 1 < best:
            best = approach + 1
    return best


def abstract_moves(env, state, plan, size=None, log=print):
    """(total_moves | None, verify_plan_ok) under the blocker-clearing model."""
    size = _board_size(env.grid_data, size)
    wr, wd = wall_sets(env.grid_data, size)
    ok, report = verify_plan(plan, env, state, size)
    total = 0
    for seg in report:
        if seg["start"] is None or seg["end"] is None:
            return None, ok
        m = _abstract_segment_moves(env, seg["start"], seg["end"],
                                    seg["support"], wr, wd, size)
        if m is None:
            return None, ok
        total += m
    pc = float(plan.cost())
    if abs(pc - total) > 1e-6 and log:
        log(f"[realize] abstract_moves={total} diverges from plan.cost()={pc:.2f}")
    return total, ok


# ---------------------------------------------------------------------------
# strict (joint-game) realization
# ---------------------------------------------------------------------------

def _physical_segments(plan):
    """All non-structural plan edges as movement dicts (start, end, support)."""
    g = plan.g
    segs = []
    for u, v in g.edges():
        ut, vt = g.nodes[u].get("ntype"), g.nodes[v].get("ntype")
        if (ut, vt) in STRUCTURAL_EDGE_PAIRS:
            continue
        start, end, support = _physical_movement(g, parent=u, child=v)
        segs.append({"edge": (u, v), "start": start, "end": end,
                     "support": support})
    return segs


def _mover_source(g, v):
    """Plan node whose robot performs the movement of physical edge (u, v).

    Edges point goal->leaf; the robot travels child->parent. When the child is
    a subgoal, the mover is the subgoal's bottleneck robot; otherwise it is the
    child node's own robot (leaf or support).
    """
    if g.nodes[v].get("ntype") == "subgoal":
        for c in g.successors(v):
            if g.nodes[c].get("ntype") == "bottleneck":
                return c
        return None
    return v


def _support_node_for(g, u, v, support_pos, byref=False):
    """The support NODE whose robot must stand at `support_pos` for edge (u, v).

    Prefers the sibling support of a bottleneck parent (the pinned segment
    support); otherwise any support node at that cell. Returns None when the
    plan assigns no robot to the cell (e.g. a parent_support drawn from the
    goal's dependent-edge supports) -- execution then succeeds only if some
    robot happens to be there.

    `byref=True` (only ever set for plans that contain Lever B2
    supports-by-reference edges, so every stored pre-B2 plan is untouched)
    additionally accepts a referenced mid-chain BOTTLENECK node at the cell:
    the robot passing through it serves as the stopper, and naming the node
    here gives the scheduler the placed-before-bounce / departs-after-bounce
    dependencies for it.
    """
    if support_pos is None:
        return None
    sp = tuple(support_pos)
    if g.nodes[u].get("ntype") == "bottleneck":
        for sg in g.predecessors(u):
            if g.nodes[sg].get("ntype") == "subgoal":
                for sib in g.successors(sg):
                    if (g.nodes[sib].get("ntype") == "support"
                            and tuple(g.nodes[sib]["pos"]) == sp):
                        return sib
                if byref:
                    for sib in g.successors(sg):
                        if (g.edges[sg, sib].get("byref")
                                and tuple(g.nodes[sib]["pos"]) == sp):
                            return sib
    for n, d in g.nodes(data=True):
        if d.get("ntype") == "support" and tuple(d["pos"]) == sp:
            return n
    if byref:
        for n, d in g.nodes(data=True):
            if d.get("ntype") == "bottleneck" and tuple(d["pos"]) == sp:
                return n
    return None


def _topo(deps):
    """Deterministic Kahn topological order (min index first); None on cycle."""
    n = len(deps)
    indeg = [len(d) for d in deps]
    out = [[] for _ in range(n)]
    for i, ds in enumerate(deps):
        for j in ds:
            out[j].append(i)
    heap = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        i = heapq.heappop(heap)
        order.append(i)
        for k in out[i]:
            indeg[k] -= 1
            if indeg[k] == 0:
                heapq.heappush(heap, k)
    return order if len(order) == n else None


def _slide_bfs(start, end, blockers, wr, wd, size=16):
    """Min slides start->end for ONE robot with static `blockers`; None if cut off."""
    if start == end:
        return 0
    seen = {start}
    q = deque([(start, 0)])
    while q:
        cur, dist = q.popleft()
        for d in DIRECTIONS:
            nxt = slide(cur, d, blockers, wr, wd, size)
            if nxt == cur or nxt in seen:
                continue
            if nxt == end:
                return dist + 1
            seen.add(nxt)
            q.append((nxt, dist + 1))
    return None


def _two_phase_ucands(env, end, support, wr, wd, size=16):
    """Certified pre-bounce cells for a supported segment, sorted.

    Mirrors `_abstract_segment_moves`'s second route and
    `simulate.segment_realizable`: candidates are the sources of the
    instance graph's dependent in-edges of `end` whose stopper is the
    segment's support cell, each re-simulated so that one slide from the
    candidate actually stops at `end` with only the support on the board.
    """
    sup = frozenset({tuple(support)})
    end_t = tuple(end)
    out = set()
    for u, _v, d in env.G.in_edges(end_t, data=True):
        if d.get("dependent") != tuple(support):
            continue
        if any(slide(tuple(u), dr, sup, wr, wd, size) == end_t
               for dr in DIRECTIONS):
            out.add(tuple(u))
    return sorted(out)


def _execute_schedule(g, state, segs, idx_by_parent, split, reorder,
                      wr, wd, size, log, check_target=True):
    """One deterministic execution attempt; segments in `split` run two-phase.

    Units are (segment, phase). An unsplit segment is a single atomic unit
    (phase 0, exactly the historical behavior). A split segment is an
    approach unit (phase 0: BFS to the cheapest certified pre-bounce cell,
    executed BEFORE its support is placed) plus a bounce unit (phase 1: BFS
    into `end`, executed AFTER the support placement). `reorder` is a set of
    (before_seg, before_sel, after_seg, after_sel) tuples, each adding one
    extra dependency edge between segment units; sel 0 names a segment's
    phase-0 unit, sel 1 its final unit (see the reorder retry in
    `strict_moves`). With `split` and `reorder` empty the unit graph, Kahn
    order, BFS calls and totals are identical to the historical
    single-phase code path.

    Returns (total, None, None, None) on success, else
    (None, fail_seg, fail_kind, fail_ctx) with fail_kind in {"cyclic",
    "atomic", "approach", "bounce", "target"}. For unit failures fail_ctx is
    {"occupant": color of the robot sitting on the failing segment's end
    cell (or None), "pending": segment indices whose final unit has not yet
    executed}; otherwise fail_ctx is None.
    """
    units = []
    for i in range(len(segs)):
        units.append((i, 0))
        if i in split:
            units.append((i, 1))
    units.sort()
    uix = {u: k for k, u in enumerate(units)}

    def final(i):                      # unit index that completes segment i
        return uix[(i, 1)] if i in split else uix[(i, 0)]

    udeps = [set() for _ in units]     # udeps[k] must all run before unit k
    for i, s in enumerate(segs):
        # (a) chain: the mover must first arrive at its source cell
        if g.nodes[s["src"]].get("ntype") in ("bottleneck", "support"):
            for j in idx_by_parent.get(s["src"], []):
                udeps[uix[(i, 0)]].add(final(j))
        if i in split:
            udeps[uix[(i, 1)]].add(uix[(i, 0)])
        # (b) support placed before the bounce; (c) departs only after it.
        # For a split segment the two-phase order is FORCED: approach leg,
        # then the support placement, then the bounce leg.
        sp = s["support_node"]
        if sp is not None:
            bounce = final(i)
            for j in idx_by_parent.get(sp, []):
                udeps[bounce].add(final(j))
                if i in split:
                    udeps[uix[(j, 0)]].add(uix[(i, 0)])
            for j2, s2 in enumerate(segs):
                if j2 != i and s2["src"] == sp:
                    udeps[uix[(j2, 0)]].add(bounce)

    # Lever B1 park segments (analysis/b1_design.md): a 'park' node moves one
    # robot aside to clear the way for one specific plan segment, so the park
    # must complete before that segment starts. Plans without park nodes take
    # this loop as a no-op (no unit graph change).
    idx_of_edge = {tuple(s2["edge"]): j2 for j2, s2 in enumerate(segs)}
    for i, s in enumerate(segs):
        if g.nodes[s["edge"][0]].get("ntype") != "park":
            continue
        be = g.nodes[s["edge"][0]].get("before_edge")
        j = idx_of_edge.get(tuple(be)) if be is not None else None
        if j is not None and j != i:
            udeps[uix[(j, 0)]].add(final(i))

    def uof(i, sel):                              # unit named by (seg, sel)
        return final(i) if sel == 1 else uix[(i, 0)]

    for bef, bsel, aft, asel in reorder:          # extra reorder edges
        udeps[uof(aft, asel)].add(uof(bef, bsel))

    order = _topo(udeps)
    if order is None:
        return None, None, "cyclic", None

    pos = {r.color: tuple(int(x) for x in r.position) for r in state.all_robots}
    total = 0
    done = set()                       # segments whose final unit has executed

    def ctx(color, end):               # failure context for the retry logic
        occ = next((c for c, p in pos.items() if c != color and p == end),
                   None)
        return {"occupant": occ,
                "pending": [j for j in range(len(segs)) if j not in done],
                # positions at the moment of failure (Lever B1 repair
                # proposals read them; the retry logic ignores the key)
                "pos": dict(pos)}

    for k in order:
        i, phase = units[k]
        s = segs[i]
        color, end = s["color"], tuple(s["end"])
        cur = pos[color]
        blockers = frozenset(p for c, p in pos.items() if c != color)
        if phase == 0:
            if tuple(s["start"]) != cur and log:
                log(f"[realize] strict_moves: segment start {tuple(s['start'])} != "
                    f"current {cur} for {color}; executing from current")
            if i not in split:                    # atomic segment
                m = _slide_bfs(cur, end, blockers, wr, wd, size)
                if m is None:
                    return None, i, "atomic", ctx(color, end)
                total += m
                pos[color] = end
                done.add(i)
                continue
            if cur == end:                        # already done; bounce is free
                continue
            # Historical choice: cheapest reachable pre-bounce cell. Under an
            # active reorder (only ever reached AFTER the default schedule
            # failed) two deterministic preferences are ranked above cost:
            # first keep every unit this segment was reordered ahead of
            # reachable (checked on current positions), then keep this
            # segment's own certified one-slide bounce clear of the robots
            # currently on the board.
            guarded = [(a, asel) for bef, bsel, a, asel in reorder
                       if bef == i and bsel == 0]
            sup = tuple(s["support"]) if s["support"] is not None else None
            best = None                           # (bad_guard, bad_bounce, m, u)
            for u in s["ucands"]:
                m = _slide_bfs(cur, u, blockers, wr, wd, size)
                if m is None:
                    continue
                bad_guard = bad_bounce = False
                if reorder:
                    bad_bounce = not any(
                        slide(u, d, blockers | {sup}, wr, wd, size) == end
                        for d in DIRECTIONS)
                if guarded:
                    hyp = dict(pos)
                    hyp[color] = u
                    for b, bsel in guarded:
                        sb = segs[b]
                        bbl = frozenset(p for c, p in hyp.items()
                                        if c != sb["color"])
                        tgts = (sb["ucands"] if bsel == 0 and b in split
                                else [tuple(sb["end"])])
                        if not any(_slide_bfs(hyp[sb["color"]], tuple(t), bbl,
                                              wr, wd, size) is not None
                                   for t in tgts):
                            bad_guard = True
                            break
                key = (bad_guard, bad_bounce, m, u)
                if best is None or key < best:
                    best = key
            if best is None:
                return None, i, "approach", ctx(color, end)
            total += best[2]
            pos[color] = best[3]
        else:                                     # bounce leg (support placed)
            m = _slide_bfs(cur, end, blockers, wr, wd, size)
            if m is None:
                return None, i, "bounce", ctx(color, end)
            total += m
            pos[color] = end
            done.add(i)

    if check_target and (pos[state.target_robot.color]
                         != tuple(int(x) for x in state.target)):
        return None, None, "target", None
    return total, None, None, None


def _reorder_choice(segs, fail_seg, ctx):
    """Deterministic pick of the ONE reorder retry for a blocked unit.

    The blocked unit's end cell is occupied by `ctx["occupant"]`, a robot
    the plan itself still moves. Returns ("vacate", seg) to run that pending
    segment's approach leg before the blocked unit -- preferring the
    pure-swap signature (a pending segment of the occupant whose own support
    cell IS the blocked cell: the occupant only vacates via the very bounce
    that needs this placement), else ("defer", seg) when the occupant was
    moved ONTO the cell by an already-executed segment (its end is the
    cell), which is re-scheduled after the blocked unit, else ("vacate", .)
    with the pending segment that starts on the cell / any non-zero-length
    one. None when the occupant has no usable segment. Ties break on
    segment index.
    """
    occ = ctx["occupant"]
    end = tuple(segs[fail_seg]["end"])
    pend = [j for j in ctx["pending"]
            if j != fail_seg and segs[j]["color"] == occ]
    swap = [j for j in pend if segs[j]["support"] is not None
            and tuple(segs[j]["support"]) == end]
    if swap:
        return "vacate", min(swap)
    arrived = [j for j in range(len(segs))
               if j not in ctx["pending"] and j != fail_seg
               and segs[j]["color"] == occ and tuple(segs[j]["end"]) == end]
    if arrived:
        return "defer", min(arrived)
    if not pend:
        return None

    def key(j):
        s = segs[j]
        return (tuple(s["start"]) != end,
                tuple(s["start"]) == tuple(s["end"]),
                j)

    return "vacate", min(pend, key=key)


def strict_moves(env, state, plan, size=None, log=print, two_phase=True,
                 fail_info=None):
    # two_phase=True adopted as the default realizer 2026-07-10 after the offline
    # A/B (eval/results/realizer_twophase_ab.json): 0/239 regressions, 44/211 prior
    # failures newly realized. Pass two_phase=False to reproduce the atomic-only
    # historical behavior. The reorder retry below is part of two_phase=True,
    # adopted 2026-07-12 after the offline A/B
    # (eval/results/realizer_reorder_ab.json).
    """Total slides of a legal joint-game execution of `plan`, or None.

    two_phase=False: every segment is atomic and a supported segment runs
    only after its support robot is placed -- the historical behavior,
    unchanged.

    two_phase=True (default): the atomic schedule is tried first, so every
    realization that already succeeds is returned move-for-move identical.
    Only after a failure are two bounded, deterministic retries available:

    - split: if the atomic failure is at a supported segment, that segment is
      re-executed in the plan's own two-phase order (the order
      `_abstract_segment_moves` / `simulate.segment_realizable` already
      cost/verify): the mover's approach leg to a certified pre-bounce cell
      runs BEFORE the support placement, then the support robot is placed,
      then the bounce leg into `end`.

    - reorder: if the failing unit's destination cell is occupied by a robot
      the plan itself moves (typically a support placement whose stopper
      cell the future bouncer still stands on), the schedule is retried with
      extra dependency edges, chosen deterministically by `_reorder_choice`:
      either the occupant's own pending segment vacates the cell first (its
      approach leg is ordered before the blocked unit; if that victim
      segment is supported it is split so only its approach moves early --
      its bounce still waits for its own stopper), or, when the occupant was
      moved onto the cell by an already-executed segment, that arrival is
      re-scheduled after the blocked unit. Other segments bouncing off the
      same cell are kept consistent (they stop against the original
      occupant when they are upstream of the blocked unit, else against the
      replacement robot), and a vacating victim still runs after the
      occupant robot's already-executed segments. At most ONE reorder
      attempt per blocked segment; no other victim or flavor is tried.

    Both retries reuse only the plan's own declared segments in a different
    legal order -- no repair moves are ever invented -- and all legs use the
    same joint-state BFS over `simulate.slide` as atomic segments, their
    move counts simply adding. While a reorder is active, approach legs
    rank certified pre-bounce cells by (keeps reordered-ahead units
    reachable, keeps own one-slide bounce clear, cost) instead of cost
    alone; with no reorder the choice is the historical cheapest cell.
    Retries strictly grow the `split`/`reordered` sets (each bounded by the
    segment count), so the loop terminates; a reordering that induces a
    dependency cycle fails cleanly. (Each retry re-executes the schedule
    from the start, so with a `log` the start-mismatch warnings of earlier
    attempts may repeat.)
    """
    g = plan.g
    size = _board_size(env.grid_data, size)
    wr, wd = wall_sets(env.grid_data, size)
    segs = _physical_segments(plan)
    if not segs:
        return None
    if not _annotate_segments(g, segs, log):
        return None
    return _realize_annotated(env, state, g, segs, wr, wd, size, log,
                              two_phase, fail_info=fail_info)


def _annotate_segments(g, segs, log):
    """Attach src / color / support_node to each segment dict (in place).

    False when some segment has no resolvable mover (the plan cannot be
    realized); mirrors the historical inline loop of `strict_moves`.
    """
    byref = any(d.get("byref") for _, _, d in g.edges(data=True))
    for s in segs:
        src = _mover_source(g, s["edge"][1])
        if src is None or "robot" not in g.nodes[src]:
            if log:
                log(f"[realize] strict_moves: no mover for edge {s['edge']}")
            return False
        s["src"] = src
        s["color"] = g.nodes[src]["robot"].color
        s["support_node"] = _support_node_for(g, s["edge"][0], s["edge"][1],
                                              s["support"], byref=byref)
    return True


def _realize_annotated(env, state, g, segs, wr, wd, size, log, two_phase,
                       check_target=True, fail_info=None):
    """Strict realization loop (atomic schedule + split/reorder retries) over an
    annotated segment list; returns total moves or None. Extracted verbatim from
    `strict_moves` so `prefix_playable` can run it on a segment SUBSET (with
    `check_target=False`, since a prefix need not finish on the target).

    `fail_info`: optional dict; on failure it is filled with the LAST
    attempt's failing-unit description (kind, segment endpoints/color/support,
    and the scheduler's failure context including the position snapshot) so a
    caller — the Lever B1 park-repair proposal step — can build targeted
    repair candidates. Purely observational: passing it changes no behavior."""
    # arrival/placement lookup: plan node -> segments whose parent side is it
    idx_by_parent = {}
    for i, s in enumerate(segs):
        idx_by_parent.setdefault(s["edge"][0], []).append(i)

    split = set()                       # segments executed two-phase
    reorder = set()                     # (bef, bef_sel, aft, aft_sel) deps
    reordered = set()                   # blocked segs already retried once
    while True:
        total, fail_seg, kind, ctx = _execute_schedule(
            g, state, segs, idx_by_parent, split, reorder, wr, wd, size, log,
            check_target=check_target)
        if (fail_info is not None and kind is not None
                and not fail_info.get("fail_seg")):
            # keep the FIRST segment-level failure: later retries can end in
            # an uninformative cyclic/target failure with no segment attached
            fail_info["kind"] = kind
            fail_info["ctx"] = ctx
            fail_info["fail_seg"] = (None if fail_seg is None else {
                "edge": segs[fail_seg]["edge"],
                "start": segs[fail_seg]["start"],
                "end": segs[fail_seg]["end"],
                "color": segs[fail_seg]["color"],
                "support": segs[fail_seg]["support"]})
        if kind is None:
            return total
        if kind == "cyclic" and log:
            log("[realize] strict_moves: cyclic segment ordering")
        if kind == "target" and log:
            log("[realize] strict_moves: target robot not on target after "
                "execution")
        if not two_phase or kind not in ("atomic", "approach", "bounce"):
            return None
        s = segs[fail_seg]
        # retry 1 (split): re-execute a supported segment two-phase
        if kind == "atomic" and s["support"] is not None:
            if "ucands" not in s:
                s["ucands"] = _two_phase_ucands(env, s["end"], s["support"],
                                                wr, wd, size)
            if s["ucands"]:
                split.add(fail_seg)     # retry with this segment two-phased
                continue
        # retry 2 (reorder): the blocked unit's destination is occupied by a
        # robot the plan itself moves -- retry ONCE with the plan's own
        # segments in a different legal order (advance the occupant's
        # departure leg, or defer the occupant's arrival past this unit)
        if fail_seg in reordered or ctx is None or ctx["occupant"] is None:
            return None
        choice = _reorder_choice(segs, fail_seg, ctx)
        if choice is None:
            return None
        flavor, j = choice
        if flavor == "vacate":
            vs = segs[j]
            if vs["support"] is not None:   # approach leg only; bounce waits
                if "ucands" not in vs:
                    vs["ucands"] = _two_phase_ucands(env, vs["end"],
                                                     vs["support"], wr, wd,
                                                     size)
                if not vs["ucands"]:
                    return None
                split.add(j)
            reorder.add((j, 0, fail_seg, 0))
            if kind == "bounce":        # before the whole two-phase pair
                reorder.add((j, 0, fail_seg, 1))
            # (a) the victim's approach still runs after the occupant
            # robot's already-executed segments (duplicate plan identities);
            # (b) any other segment bouncing off the blocked cell either
            # still stops against the ORIGINAL occupant (when it is itself
            # upstream of the blocked unit -- its end is the blocked unit's
            # own support cell -- it runs before the victim departs) or
            # stops against the REPLACEMENT robot (after the blocked unit)
            end = tuple(s["end"])
            for k2 in range(len(segs)):
                if k2 == j or k2 == fail_seg:
                    continue
                if (k2 not in ctx["pending"]
                        and segs[k2]["color"] == segs[j]["color"]):
                    reorder.add((k2, 1, j, 0))
                if (segs[k2]["support"] is not None
                        and tuple(segs[k2]["support"]) == end):
                    if (s["support"] is not None and tuple(s["support"])
                            == tuple(segs[k2]["end"])):
                        reorder.add((k2, 1, j, 0))
                    else:
                        reorder.add((fail_seg, 1, k2, 1))
        else:                               # defer the arrival segment
            reorder.add((fail_seg, 1, j, 0))
        reordered.add(fail_seg)


# ---------------------------------------------------------------------------
# prefix realizability (Lever A: in-search playability filter)
# ---------------------------------------------------------------------------

def _fixed_physical_segments(plan):
    """The already-committed movements: non-structural edges with status 'fixed'
    (same dict shape and edge-iteration order as `_physical_segments`)."""
    g = plan.g
    segs = []
    for u, v, d in g.edges(data=True):
        if d.get("status") != "fixed":
            continue
        ut, vt = g.nodes[u].get("ntype"), g.nodes[v].get("ntype")
        if (ut, vt) in STRUCTURAL_EDGE_PAIRS:
            continue
        start, end, support = _physical_movement(g, parent=u, child=v)
        segs.append({"edge": (u, v), "start": start, "end": end,
                     "support": support})
    return segs


def _determined_indices(g, segs):
    """Indices of the fixed segments whose execution context is already pinned.

    A fixed segment is DETERMINED when everything the strict scheduler would
    order before it is itself a fixed, determined segment:

      - mover delivery: every out-edge of the segment's source node (the
        chain that brings the mover to its start cell) is fixed and its
        segment determined, recursively down to a leaf;
      - support placement: a declared stopper has an assigned support node
        whose own delivery chain is fixed and determined (no node -> not
        checkable -> excluded);
      - support departure: a segment that moves a robot OFF a support cell is
        determined only if every fixed bounce consuming that support is
        determined too (otherwise the prefix would run the departure without
        knowing what must precede it).

    Everything else (segments downstream of an open edge, or whose support is
    not yet placed by the plan) is excluded: its context can still change, so
    nothing about it is checkable. Exclusion cascades by fixpoint.
    """
    idx_by_edge = {s["edge"]: i for i, s in enumerate(segs)}
    n = len(segs)
    ok = [True] * n
    reqs = [set() for _ in range(n)]
    for i, s in enumerate(segs):
        need = [s["src"]]
        if s["support"] is not None:
            if s["support_node"] is None:      # stopper nobody is assigned to
                ok[i] = False
                continue
            need.append(s["support_node"])
        for node in need:
            for u2, v2, d2 in g.out_edges(node, data=True):
                if d2.get("status") == "open":
                    ok[i] = False
                    break
                j = idx_by_edge.get((u2, v2))
                if j is not None:
                    reqs[i].add(j)
            if not ok[i]:
                break
        if not ok[i]:
            continue
        if g.nodes[s["src"]].get("ntype") == "support":
            for j, s2 in enumerate(segs):      # departure waits for bounces
                if j != i and s2["support_node"] == s["src"]:
                    reqs[i].add(j)
    changed = True
    while changed:                             # cascade exclusions
        changed = False
        for i in range(n):
            if ok[i] and any(not ok[j] for j in reqs[i]):
                ok[i] = False
                changed = True
    return [i for i in range(n) if ok[i]]


def prefix_key(plan):
    """Hashable identity of a plan's physical-edge multiset (endpoints, support,
    status). Plans with equal keys get the same `prefix_playable` verdict for
    the same env/state, so callers can memoize on it."""
    g = plan.g
    out = []
    for u, v, d in g.edges(data=True):
        ut, vt = g.nodes[u].get("ntype"), g.nodes[v].get("ntype")
        if (ut, vt) in STRUCTURAL_EDGE_PAIRS:
            continue
        start, end, support = _physical_movement(g, parent=u, child=v)
        src = _mover_source(g, v)
        color = (g.nodes[src]["robot"].color
                 if src is not None and "robot" in g.nodes[src] else None)
        out.append((ut, vt, color,
                    None if start is None else tuple(start),
                    None if end is None else tuple(end),
                    None if support is None else tuple(support),
                    d.get("status")))
    return tuple(sorted(out))


def prefix_playable(env, state, plan, size=None, log=None, two_phase=True):
    """Can the plan's already-committed (fixed) segments still be played?

    For a COMPLETE plan this is exactly `strict_moves(...) is not None`. For a
    PARTIAL plan it strict-realizes only the determined fixed subgraph (see
    `_determined_indices`) with the same two-phase + reorder machinery, all
    robots starting at their instance cells and no final on-target condition.

    One-directional filter, biased permissive: True says nothing about the
    completion (open segments are unchecked); anything not fully pinned down
    by the plan is skipped rather than guessed, and an unanalyzable plan
    passes. False means the committed movements themselves could not be
    executed by the strict realizer. In principle a completion can still
    interleave NEW segments (a freshly recruited helper, an intermediate stop
    on an open chain) before committed ones and rescue such a prefix, so False
    is not an absolute proof of unrealizability; the no-false-pruning A/B in
    `eval/results/prefix_check_ab.json` is the empirical check that this does
    not occur under the deterministic realizer.
    """
    if plan.is_complete():
        return strict_moves(env, state, plan, size=size, log=log,
                            two_phase=two_phase) is not None
    g = plan.g
    size = _board_size(env.grid_data, size)
    wr, wd = wall_sets(env.grid_data, size)
    segs = _fixed_physical_segments(plan)
    if not segs:
        return True
    if not _annotate_segments(g, segs, log):
        return True                            # cannot analyze -> permissive
    keep = _determined_indices(g, segs)
    if not keep:
        return True
    segs = [segs[i] for i in keep]
    return _realize_annotated(env, state, g, segs, wr, wd, size, log,
                              two_phase, check_target=False) is not None
