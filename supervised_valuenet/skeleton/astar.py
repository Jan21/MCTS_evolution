"""A clean A* over partial plans.

The whole algorithm is one idea: search the space of partial plans, ordering
the frontier by plan cost. A plan's cost is g + h automatically: fixed segments
carry their exact cost (g), open segments carry an optimistic relaxed estimate
(h). Pop the cheapest plan; if it has no open segments it is complete and, since
its cost is then all-exact, optimal among everything proposed. Otherwise pick an
open segment and expand it: either pin it to an exact path, or let `propose`
suggest subgoals, each spawning a child plan.

All board intelligence is confined to `heuristics.propose` / `heuristics.score`,
so swapping in a neural network touches nothing here.
"""
from __future__ import annotations

import copy
import heapq
import time

from GridEnv import GridEnv, State, Robot_at
from partial_plan import PartialPlan
from A_star.base import A_star, SolveResult, PlanEntry, PlanStats

from skeleton import heuristics

INF = 10_000


class AStar(A_star):
    def __init__(self, propose=heuristics.propose, score=heuristics.score,
                 max_iters=20_000, beam=None, max_open=None, max_frontier=100_000,
                 by_reference=False):
        self.propose = propose
        self.score = score
        self.max_iters = max_iters
        self.beam = beam       # keep only `beam` cheapest children per expansion
        self.max_open = max_open  # drop plans with more open segments (runaway guard)
        self.max_frontier = max_frontier  # bail if the heap grows past this (memory guard)
        # Lever B2 (supports-by-reference, analysis/b1_extension_notes.md):
        # additionally offer robots the plan has ALREADY placed as candidate
        # helpers, standing at their planned cells, so a proposal may reference
        # an existing plan node as its support instead of recruiting the robot
        # a second time. Default False keeps every existing behavior identical.
        self.by_reference = by_reference

    # -- public API ---------------------------------------------------------

    def solve(self, env: GridEnv, state: State) -> SolveResult:
        """Solve an instance from scratch."""
        return self._search(env, state, _initial_plan(env, state))

    def solve_plan(self, env: GridEnv, state: State, start: PartialPlan):
        """Complete a given partial plan; returns the optimal plan or None."""
        res = self._search(env, state, start)
        return res.best_plan if res.all_plans else None

    def _search(self, env, state, start: PartialPlan) -> SolveResult:
        t0 = time.time()
        frontier = [(start.cost(), 0, start)]
        tie = 1
        # Without backtracking (beam=1) a greedy chain can grow unboundedly;
        # cap open segments so the search always terminates. A real plan needs
        # at most one open pair per helper, plus slack.
        max_open = self.max_open or 2 * (len(state.helpers) + 2)

        # The relaxed estimate on open segments never exceeds their eventual
        # exact decomposition (dependent edges cost 2 but expand to >= 2 moves),
        # so the heuristic is admissible: the first complete plan popped is
        # optimal over everything proposed.
        for it in range(1, self.max_iters + 1):
            if not frontier or len(frontier) > self.max_frontier:
                break
            cost, _, cur = heapq.heappop(frontier)
            if cur.is_complete():
                return SolveResult(best_plan=cur, all_plans=[_entry(cur, t0, it)])
            if len(cur.open_edges()) > max_open:
                continue
            children = self._expand(env, state, cur)  # best-scored first
            if self.beam is not None:
                children = sorted(children, key=lambda p: p.cost())[: self.beam]
            for child in children:
                heapq.heappush(frontier, (child.cost(), tie, child))
                tie += 1

        return SolveResult(best_plan=start, all_plans=[])

    # -- expansion ----------------------------------------------------------

    def _expand(self, env, state, plan) -> list[PartialPlan]:
        """Resolve the first open segment, returning child plans."""
        parent, child = plan.open_edges()[0]
        seg = _segment(plan, state, parent, child)

        # 1. Try to pin the segment to an exact path.
        exact = env.compute_exact_shortest_path_length(
            seg.start, seg.end, seg.fix_support)
        if exact is not None:
            out = plan.copy()
            out.g[parent][child].update(status="fixed", cost=exact)
            return [out]

        # 2. Otherwise expand via proposed subgoals, best-evaluated first.
        # `propose` is heuristic #1 (which subgoals); `score` is heuristic #2
        # (how promising each is). Both are the neural-network swap points.
        if self.by_reference:
            seg.helpers = seg.helpers + _reference_helpers(plan, seg.mover.color)
        cands = self.propose(env, seg.end, seg.mover, seg.helpers, seg.support)
        cands.sort(key=lambda c: self.score(env, c))
        children = [p for p in (_apply(env, plan, parent, child, seg, c,
                                       by_reference=self.by_reference)
                                for c in cands) if p is not None]
        return children


# ---------------------------------------------------------------------------
# Segment description
# ---------------------------------------------------------------------------

class _Seg:
    __slots__ = ("start", "end", "mover", "helpers", "support", "fix_support")


def _segment(plan: PartialPlan, state: State, parent: str, child: str) -> _Seg:
    """Describe the physical move an open edge stands for.

    Edges point goal -> leaf; the robot physically travels child -> parent.
    """
    g = plan.g
    pdata, cdata = g.nodes[parent], g.nodes[child]
    s = _Seg()
    s.start = cdata["pos"]                 # where the moving robot is now
    s.end = pdata["pos"]                   # where it must end up
    s.mover = (state.target_robot if pdata["ntype"] == "goal"
               else pdata.get("robot", state.target_robot))

    # Support pinned for this segment (parent is a bottleneck needing a stopper).
    s.support = None
    if pdata["ntype"] == "bottleneck":
        s.support = _sibling_support(plan, parent)
    s.fix_support = s.support.position if s.support is not None else None

    # Helpers available to this segment: everyone except the mover. Identity is the
    # robot's COLOR: dataclass equality also compares positions, so a mover carried at
    # a planned (moved) position would survive in its own helper list and could be
    # scheduled to bounce off itself.
    s.helpers = [h for h in state.helpers if h.color != s.mover.color]
    if s.mover.color != state.target_robot.color:
        s.helpers.append(state.target_robot)
    return s


def _sibling_support(plan: PartialPlan, bottleneck: str) -> Robot_at | None:
    for sg in plan.g.predecessors(bottleneck):
        if plan.g.nodes[sg].get("ntype") == "subgoal":
            for sib in plan.g.successors(sg):
                if plan.g.nodes[sib].get("ntype") == "support":
                    return plan.g.nodes[sib]["robot"]
            # Lever B2: the subgoal's support may be a REFERENCED node (a
            # mid-chain bottleneck serving as the stopper), marked by the
            # byref edge attribute. Only b2 plans contain such edges.
            for sib in plan.g.successors(sg):
                if (plan.g.edges[sg, sib].get("byref")
                        and plan.g.nodes[sib].get("ntype") == "bottleneck"):
                    return plan.g.nodes[sib]["robot"]
    return None


def _reaches(g, src, dst) -> bool:
    """True when `dst` is reachable from `src` along plan edges (goal->leaf
    direction). Plans are small; a plain DFS suffices."""
    if src == dst:
        return True
    seen = {src}
    stack = [src]
    while stack:
        u = stack.pop()
        for v in g.successors(u):
            if v == dst:
                return True
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return False


def _terminal_support(g, node) -> bool:
    """True when `node` is the LAST support cell its robot stands on: no
    relocation (support-typed) or park predecessor moves it away later."""
    return not any(g.nodes[u].get("ntype") in ("support", "park")
                   for u in g.predecessors(node))


def _reference_helpers(plan: PartialPlan, mover_color) -> list[Robot_at]:
    """Lever B2 phantom helpers: robots the plan already places, offered at
    their PLANNED cells — the terminal support node they are delivered to, or
    a mid-chain bottleneck they pass through — so `propose` can generate
    candidates whose support is an existing plan node (shared supports, the
    target robot serving as a stopper) or a relocation from one. The mover
    itself and park-holding robots are excluded."""
    g = plan.g
    parked = {d["robot"].color for _, d in g.nodes(data=True)
              if d.get("ntype") == "park"}
    out, seen = [], set()
    for n, d in g.nodes(data=True):
        nt = d.get("ntype")
        if nt not in ("support", "bottleneck"):
            continue
        color = d["robot"].color
        if color == mover_color or color in parked:
            continue
        if nt == "support" and not _terminal_support(g, n):
            continue
        key = (color, tuple(d["pos"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(Robot_at(position=tuple(d["pos"]), color=color))
    return out


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

def _initial_plan(env: GridEnv, state: State) -> PartialPlan:
    plan = PartialPlan()
    plan.add_node("goal", "goal", pos=state.target)
    plan.add_node("leaf_0", "leaf", pos=state.target_robot.position,
                  robot=state.target_robot)
    plan.nc = 1
    exact = env.compute_exact_shortest_path_length(
        state.target_robot.position, state.target, None)
    relaxed = env.compute_relaxed_shortest_path_length(
        state.target_robot.position, state.target, None)
    if exact is not None and (relaxed is None or exact <= relaxed):
        plan.add_edge("goal", "leaf_0", status="fixed", cost=exact)
    else:
        plan.add_edge("goal", "leaf_0", status="open",
                      cost=relaxed if relaxed is not None else INF)
    return plan


def _apply(env, plan, parent, child, seg, cand,
           by_reference: bool = False) -> PartialPlan | None:
    """Attach `cand`'s subgoal between `parent` and `child`, return new plan."""
    bn_pos = cand.subgoal.bottleneck.position
    sp_pos = cand.subgoal.support.position

    # Invariant: one physical robot, one leaf identity. A robot that already
    # owns a leaf in this plan has a standing assignment; recruiting it again
    # as a FRESH leaf (at its original cell) schedules the single real robot
    # in two places at once. Compare by COLOR, not dataclass equality, per the
    # helper-identity precedent in _segment.
    # Lever B2 (by_reference=True): such a robot may instead serve BY
    # REFERENCE — the candidate helper stands at the robot's planned cell
    # (see _reference_helpers), and the subgoal wires to the existing plan
    # node instead of recruiting a second leaf. Three shapes: shared support
    # (support cell IS the robot's terminal support cell — zero extra moves),
    # target-as-stopper (support cell is a mid-chain bottleneck of the
    # robot), and relocation (a costed slide from the terminal support cell
    # to a new support cell).
    helper_color = cand.subgoal.helper.color
    ref_node = None
    if any(d.get("ntype") == "leaf" and d["robot"].color == helper_color
           for _, d in plan.g.nodes(data=True)):
        if not by_reference:
            return None
        hp = tuple(cand.subgoal.helper.position)
        for n2, d2 in plan.g.nodes(data=True):
            nt2 = d2.get("ntype")
            if (nt2 in ("support", "bottleneck")
                    and d2["robot"].color == helper_color
                    and tuple(d2["pos"]) == hp
                    and (nt2 == "bottleneck"
                         or _terminal_support(plan.g, n2))):
                ref_node = n2
                break
        if ref_node is None:
            return None            # candidate not at a planned cell: stale
        if (plan.g.nodes[ref_node]["ntype"] == "bottleneck"
                and tuple(sp_pos) != hp):
            return None            # a robot mid-chain can only serve in place
        if _reaches(plan.g, ref_node, parent):
            # The referenced placement depends (via the DAG) on the very
            # segment that would consume it -- a circular timing requirement
            # no schedule can satisfy. Wiring it would create a cycle.
            return None

    # Invariant: a supported (dependent-edge) route cost may only be claimed
    # when the plan assigns a robot to stand on the stopper cell. A
    # parent_support drawn from the board's dependent-edge supports carries no
    # robot; accept it only when a support node already occupies that cell or
    # this candidate's own support is that cell -- otherwise the plan counts
    # as complete with a declared stopper that no plan node ever places.
    # Under B2 a referenced mid-chain bottleneck node also counts as placed.
    ps = cand.parent_support
    if (ps is not None and tuple(ps) != tuple(sp_pos)
            and not any(d.get("ntype") == "support"
                        and tuple(d["pos"]) == tuple(ps)
                        for _, d in plan.g.nodes(data=True))
            and not (by_reference
                     and any(d.get("ntype") == "bottleneck"
                             and tuple(d["pos"]) == tuple(ps)
                             for _, d in plan.g.nodes(data=True)))):
        return None

    parent_cost = env.compute_exact_shortest_path_length(
        bn_pos, seg.end, cand.parent_support)
    if parent_cost is None:
        return None

    out = plan.copy()
    out.g.remove_edge(parent, child)
    n = out.nc
    out.nc += 1
    sg, bn, sp, leaf = f"sg_{n}", f"bn_{n}", f"sp_{n}", f"leaf_{n}"

    out.add_node(sg, "subgoal", parent_support_pos=cand.parent_support)
    out.add_node(bn, "bottleneck", pos=bn_pos, robot=cand.subgoal.bottleneck)
    out.add_edge(parent, sg, status="fixed", cost=parent_cost)
    out.add_edge(sg, bn, status="fixed", cost=0)

    # mover travels (child pos) -> bottleneck, stopping via the segment support.
    _attach_move(env, out, bn, child, start=seg.start, end=bn_pos, support=sp_pos)

    if ref_node is None:
        # Fresh recruitment (the only pre-B2 path, unchanged): a new support
        # node delivered from the helper's own leaf.
        out.add_node(sp, "support", pos=sp_pos, robot=cand.subgoal.support)
        out.add_edge(sg, sp, status="fixed", cost=0)
        out.add_node(leaf, "leaf", pos=cand.subgoal.helper.position,
                     robot=cand.subgoal.helper)
        _attach_move(env, out, sp, leaf,
                     start=cand.subgoal.helper.position, end=sp_pos,
                     support=None)
    elif tuple(sp_pos) == tuple(out.g.nodes[ref_node]["pos"]):
        # Shared support / target-as-stopper: reference the existing node.
        # Zero extra moves; the realizer orders every bounce consuming the
        # node before its robot departs (per-bounce rule).
        out.g.add_edge(sg, ref_node, status="fixed", cost=0, byref=True)
    else:
        # Relocation: the robot slides from its terminal support cell to a
        # new support cell — an ordinary costed plan edge hung off the
        # existing node (the departs-after-bounce rule orders it correctly).
        out.add_node(sp, "support", pos=sp_pos, robot=cand.subgoal.support)
        out.add_edge(sg, sp, status="fixed", cost=0)
        _attach_move(env, out, sp, ref_node,
                     start=tuple(out.g.nodes[ref_node]["pos"]), end=sp_pos,
                     support=None)
    return out


def _attach_move(env, plan, parent, child, start, end, support):
    """Add a parent->child edge: fixed if an exact path exists, else open."""
    exact = env.compute_exact_shortest_path_length(start, end, support)
    if exact is not None:
        plan.add_edge(parent, child, status="fixed", cost=exact)
    else:
        relaxed = env.compute_relaxed_shortest_path_length(start, end, support)
        plan.add_edge(parent, child, status="open",
                      cost=relaxed if relaxed is not None else INF)


# ---------------------------------------------------------------------------
# Lever B1: park ("step aside / vacate") plan augmentation
# ---------------------------------------------------------------------------
#
# A park is a subgoal-shaped commitment "robot X must first stand on cell P",
# scheduled to complete before one specific plan segment runs (so it can clear
# that segment's way). In the DAG it is a `park` node with attrs pos /
# robot / before_edge and one fixed, costed edge to the node where the plan
# leaves the robot (its support node if it was placed — the realizer's
# departs-after-bounce rule then makes it a true post-bounce vacate — else its
# leaf; an idle robot gets a fresh leaf). The park's moves are ordinary plan
# cost: exact walls-only path, no free vacates. See analysis/b1_design.md.


def robot_plan_node(plan, color):
    """Node where the plan leaves robot `color`: its TERMINAL support node
    (the one no relocation or park moves it off — for pre-B2 plans every
    support node is terminal, so this is the historical behavior), else its
    leaf, else None (robot not in the plan)."""
    sup = leaf = None
    for n, d in plan.g.nodes(data=True):
        if (d.get("ntype") == "support" and d["robot"].color == color
                and _terminal_support(plan.g, n)):
            sup = n
        elif d.get("ntype") == "leaf" and d["robot"].color == color:
            leaf = n
    return sup or leaf


def apply_park(env, plan, robot, park_pos, before_edge):
    """Augment a COMPLETE plan with one park: `robot` slides from where the
    plan leaves it to `park_pos`, ordered before `before_edge`'s segment.
    Returns the new plan, or None when the park cannot be expressed (mover
    robots are not parked; the park must have an exact walls-only path)."""
    g = plan.g
    out = plan.copy()
    n = out.nc
    out.nc += 1
    node = robot_plan_node(out, robot.color)
    # A robot that travels through bottlenecks but is NOT finally parked on a
    # support cell (i.e. the target robot / a pure mover) has no single plan
    # node describing where it stands — it cannot be parked. A helper
    # delivered via nested bottlenecks ends at its support node and can.
    if (node is None or g.nodes[node].get("ntype") != "support") and any(
            d.get("ntype") == "bottleneck" and d["robot"].color == robot.color
            for _, d in g.nodes(data=True)):
        return None
    if node is None:                       # idle robot: give it a leaf
        node = f"leaf_{n}"
        out.add_node(node, "leaf", pos=tuple(robot.position), robot=robot)
    start = tuple(out.g.nodes[node]["pos"])
    if tuple(park_pos) == start:
        return None
    exact = env.compute_exact_shortest_path_length(start, tuple(park_pos), None)
    if exact is None:
        return None
    prk = f"prk_{n}"
    out.add_node(prk, "park", pos=tuple(park_pos),
                 robot=Robot_at(position=tuple(park_pos), color=robot.color),
                 before_edge=tuple(before_edge))
    out.add_edge(prk, node, status="fixed", cost=exact)
    return out


def _park_dests(xpos, wr, wd, size, multi_slide=False):
    """Walls-only slide destinations for a parking robot: the <= 4 one-slide
    cells and, with `multi_slide`, the cells one further slide away.
    Deterministic order (direction order, then first-layer order)."""
    from simulate import slide, DIRECTIONS
    layer1 = []
    for d in DIRECTIONS:
        dest = slide(xpos, d, frozenset(), wr, wd, size)
        if dest != xpos and dest not in layer1:
            layer1.append(dest)
    if not multi_slide:
        return layer1
    seen = set(layer1) | {xpos}
    layer2 = []
    for c in layer1:
        for d in DIRECTIONS:
            dest = slide(c, d, frozenset(), wr, wd, size)
            if dest not in seen:
                seen.add(dest)
                layer2.append(dest)
    return layer1 + layer2


def park_repairs(env, state, plan, fail_info, wr, wd, size, max_parks=1,
                 pairwise=False, multi_slide=False):
    """Park-augmented variants of a complete plan whose strict realization
    failed. `fail_info` is the dict filled by `eval.realize.strict_moves`.

    Proposal rule, deterministic: for the failing segment, find every robot
    whose one-robot removal unblocks the mover's slide-BFS at the recorded
    failure positions; for each, try its walls-only slide destinations and
    keep those under which the blocked segment becomes passable. Each
    surviving (robot, cell) yields one `apply_park` child. Plans already
    carrying `max_parks` park nodes return no children (bounded repair).

    Lever B2 generalizations, both default-off so B1 behavior is unchanged:
    `multi_slide` extends a robot's destinations to two-slide cells, tried
    only when none of its one-slide cells clears the segment; `pairwise`
    additionally clears TWO robots at once (each parked, both parks ordered
    before the failing segment), tried only when no single-robot repair
    exists at all. Both need `max_parks` >= 2 to take effect."""
    from eval.realize import _slide_bfs

    n_parks = sum(1 for _, d in plan.g.nodes(data=True)
                  if d.get("ntype") == "park")
    if n_parks >= max_parks:
        return []
    fs = fail_info.get("fail_seg")
    ctx = fail_info.get("ctx") or {}
    pos = ctx.get("pos")
    if not fs or not pos:
        return []
    end = tuple(fs["end"])
    mover_color = fs["color"]
    mover_cur = tuple(pos.get(mover_color, fs["start"]))
    by_color = {r.color: r for r in state.all_robots}
    parked = {d["robot"].color for _, d in plan.g.nodes(data=True)
              if d.get("ntype") == "park"}
    out = []
    movable = [c for c, _ in sorted(pos.items())
               if c != mover_color and c in by_color and c not in parked]
    for x_color in movable:
        xpos = tuple(pos[x_color])
        # would X stepping fully aside unblock the failing slide?
        bl = frozenset(tuple(p) for c, p in pos.items()
                       if c not in (mover_color, x_color))
        if _slide_bfs(mover_cur, end, bl, wr, wd, size) is None:
            continue
        dests1 = _park_dests(xpos, wr, wd, size)
        survivors = [dest for dest in dests1
                     if _slide_bfs(mover_cur, end, bl | {dest},
                                   wr, wd, size) is not None]
        if not survivors and multi_slide:
            survivors = [dest for dest in _park_dests(xpos, wr, wd, size,
                                                      multi_slide=True)
                         if dest not in dests1
                         and _slide_bfs(mover_cur, end, bl | {dest},
                                        wr, wd, size) is not None]
        for dest in survivors:
            child = apply_park(env, plan, by_color[x_color], dest, fs["edge"])
            if child is not None:
                out.append(child)
    if out or not pairwise or n_parks + 2 > max_parks:
        return out
    # No single-robot repair exists: try clearing two robots at once.
    for a in range(len(movable)):
        for b in range(a + 1, len(movable)):
            xc, yc = movable[a], movable[b]
            xpos, ypos = tuple(pos[xc]), tuple(pos[yc])
            bl2 = frozenset(tuple(p) for c, p in pos.items()
                            if c not in (mover_color, xc, yc))
            if _slide_bfs(mover_cur, end, bl2, wr, wd, size) is None:
                continue
            for dx in _park_dests(xpos, wr, wd, size, multi_slide=multi_slide):
                for dy in _park_dests(ypos, wr, wd, size,
                                      multi_slide=multi_slide):
                    if dx == dy:
                        continue
                    if _slide_bfs(mover_cur, end, bl2 | {dx, dy},
                                  wr, wd, size) is None:
                        continue
                    c1 = apply_park(env, plan, by_color[xc], dx, fs["edge"])
                    if c1 is None:
                        continue
                    c2 = apply_park(env, c1, by_color[yc], dy, fs["edge"])
                    if c2 is not None:
                        out.append(c2)
    return out


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def _entry(plan, t0, it):
    return PlanEntry(plan=plan, cost=plan.cost(),
                     stats=PlanStats(wall_time=time.time() - t0, iteration=it,
                                     node_count=plan.g.number_of_nodes(),
                                     rollout_count=it))
