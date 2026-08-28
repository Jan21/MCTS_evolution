"""Stage 3's planner: best-first search over STATE subgoals, ranked by g + h.

The design is the revised Stage 3 of `PLAN_SUBGOAL_DISCOVERY.md`:

  node        the joint robot positions plus `g`, the moves spent so far;
  children    PHYSICS, not a network. For every robot, the exact joint-state
              slide BFS gives the set of cells it can come to rest on with the
              other three frozen, and the true slide count is the edge cost --
              the Stage 2 cost definition, the only one that is an edge weight
              at all. Unreachable cells are simply not children;
  ranking     f = g + c + h(child), with `c` the exact edge cost and `h` an
              estimate of the moves still needed to finish from the child. One
              expansion scores all 1024 candidates and keeps the best k;
  h           the board-only "any-stop" relaxation Stage 1 used (no network
              anywhere -- the control), or the learned `CtgNet`, or the exact
              engine (a diagnostic ceiling, not a planner);
  terminate   the target robot on the target cell. Anytime: the search keeps
              the cheapest solution found and stops early only when that
              solution is PROVED optimal within the pruned graph -- when its
              cost is <= the smallest g + h_adm over the open list, h_adm being
              the admissible relaxation. The rule is identical for every arm
              and can never hide a better solution, because h_adm never
              over-estimates.

Because a macro edge moves one robot while the others stand still, the
concatenation of a path's edges is a legal primitive move sequence by
construction; every reported solution is replayed under the real joint-game
rules anyway before it is counted (`table.py::replay`, the physics of
`eval/replay_validate.py`).

Cells are flat indices `y*n + x` inside the search and the slide is a
precomputed-ray lookup (`build_rays` / `rest_cells_fast`), which is the same
function as `simulate.slide` -- `stage3.py verify` checks that on random states
against `subgoal/space.py::rest_cells` -- but about six times faster, because
the search runs it a few million times per arm.

Searches run in LOCKSTEP: a pool of instances each take one expansion per round
and their queries are scored in one batch, which is what makes the network arm
affordable (4 encoder passes per expansion, batched across the whole pool).

STAGE 4 ADDITIONS (all opt-in; every default reproduces Stage 3 byte for byte):

  bound="tight"   a STRONGER admissible bound than Stage 3's board-only
                  relaxation, used for the optimality stop AND to skip nodes
                  that cannot improve the incumbent.  See `h_tight` below for
                  the proof; it is the only place the robots' positions enter
                  the bound.
  noise=sigma     Gaussian jitter added to the ranking score f of non-goal
                  candidates.  Self-play exploration only; every headline row
                  runs at sigma = 0 and is deterministic.
  collect=True    keep every goal state the search reaches, so `plans()` can
                  hand back every certified plan the search found (Stage 4's
                  self-play labels).
"""
from __future__ import annotations

import heapq
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPR = HERE.parent
REPO = SPR.parent
SV = REPO / "supervised_valuenet"
for _p in (str(SV), str(SPR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simulate import DIRECTIONS                                  # noqa: E402
from subgoal.space import board, relaxed_h                       # noqa: E402
from subgoal.table import SIZE, COLOR_ORDER, replay              # noqa: E402

BIG = 10 ** 6
_RAYS = {}


# ---------------------------------------------------------------------------
# physics, on flat cell indices
# ---------------------------------------------------------------------------

def build_rays(wr, wd, n=SIZE):
    """RAY[cell][dir] = the flat cells a robot passes going that way until a
    wall or the edge, nearest first; PIR is the same as {cell: position}."""
    RAY = [[None] * 4 for _ in range(n * n)]
    PIR = [[None] * 4 for _ in range(n * n)]
    for y in range(n):
        for x in range(n):
            f = y * n + x
            for di, d in enumerate(DIRECTIONS):          # up, down, left, right
                path = []
                cx, cy = x, y
                while True:
                    if d == "up":
                        if cy == 0 or (cx, cy - 1) in wd:
                            break
                        cy -= 1
                    elif d == "down":
                        if cy == n - 1 or (cx, cy) in wd:
                            break
                        cy += 1
                    elif d == "left":
                        if cx == 0 or (cx - 1, cy) in wr:
                            break
                        cx -= 1
                    else:
                        if cx == n - 1 or (cx, cy) in wr:
                            break
                        cx += 1
                    path.append(cy * n + cx)
                RAY[f][di] = path
                PIR[f][di] = {c: j for j, c in enumerate(path)}
    return RAY, PIR


def rays(env_id, n=SIZE):
    if env_id not in _RAYS:
        wr, wd = board(env_id)
        _RAYS[env_id] = build_rays(wr, wd, n)
    return _RAYS[env_id]


def rest_cells_fast(start, blockers, RAY, PIR):
    """{cell: (slides, dir_idx, from_cell)} -- every cell the robot can come to
    rest on with `blockers` (flat cells) frozen. Exact BFS over the real slide
    rule; a slide that does not move the robot is not an edge."""
    out = {start: (0, None, None)}
    q = deque([start])
    while q:
        cur = q.popleft()
        g = out[cur][0] + 1
        rc, pc = RAY[cur], PIR[cur]
        for di in range(4):
            path = rc[di]
            if not path:
                continue
            best = len(path)
            pir = pc[di]
            for b in blockers:
                j = pir.get(b)
                if j is not None and j < best:
                    best = j
            if best == 0:
                continue
            nxt = path[best - 1]
            if nxt in out:
                continue
            out[nxt] = (g, di, cur)
            q.append(nxt)
    del out[start]
    return out


# ---------------------------------------------------------------------------
# Stage 4: a stronger admissible bound, which is the only thing in the search
# that looks at where the ROBOTS are
# ---------------------------------------------------------------------------
#
# Stage 3's bound is board-only: h_free(s) = the "any-stop" relaxation distance
# of the target robot's cell to the target cell (`space.relaxed_h`), a lower
# bound on the number of TARGET-robot moves under any configuration whatsoever,
# because a real slide is one of the relaxed moves.  It never mentions the other
# robots, so even a perfect heuristic could not terminate a search early
# (STAGE3.md section 4).
#
# Split any solution from s into T target-robot moves and O other-robot moves;
# its cost is T + O.
#
#   * if O = 0 the other robots never move, so T >= d_frozen(s), the EXACT
#     number of slides the target robot needs with the others frozen where they
#     stand (one BFS -- the same BFS the expansion of s computes anyway);
#   * if O >= 1 then T >= h_free(s) still, and O >= max(1, r(s)) where r(s) is
#     defined below, so the cost is at least h_free(s) + max(1, r(s)).
#
# so    h_tight(s) = min( d_frozen(s), h_free(s) + max(1, r(s)) )   is admissible
# and it is never below Stage 3's h_free(s).
#
# r(s), the "who can even stop it" term.  A slide stops on the target cell only
# if the cell immediately beyond it, in the direction of travel, is a wall/edge
# or holds another robot.  Let W be the directions in which a wall or the board
# edge blocks movement out of the target cell, and B the cells one step beyond
# the target cell in the remaining directions.
#
#   * W non-empty  -> a wall can stop the robot: r(s) = 0;
#   * some non-target robot already stands on a cell of B: r(s) = 0;
#   * otherwise SOME non-target robot must travel to a cell of B before the
#     target robot can ever come to rest on the target, and those are non-target
#     moves.  r(s) = the smallest any-stop relaxation distance from a non-target
#     robot to the set B -- a lower bound on the moves that takes, by the same
#     relaxation argument.  (Then d_frozen(s) = infinity as well, so the min
#     above costs nothing and h_tight(s) = h_free(s) + r(s) needs no BFS.)
#
# The tight bound is used for two things, both sound and both pre-registered:
# the optimality stop, and skipping a popped node whose g + h_tight already
# equals or exceeds the incumbent -- such a node cannot improve it, so expanding
# it wastes an expansion of the budget.

_SEG = {}


def segments(env_id, n=SIZE):
    """seg[cell] = every cell reachable in ONE any-stop relaxed move (walls
    only, robots ignored).  The relation is symmetric, so a BFS from a set of
    sources gives the relaxed distance TO that set from every cell."""
    key = (env_id, n)
    if key not in _SEG:
        RAY, _PIR = rays(env_id, n)
        _SEG[key] = [[c for di in range(4) for c in RAY[f][di]]
                     for f in range(n * n)]
    return _SEG[key]


def anystop_field(sources, env_id, n=SIZE):
    """[n*n] any-stop relaxation distance from each cell to the nearest source
    (BIG where unreachable). `sources` are flat cell indices."""
    seg = segments(env_id, n)
    dist = [BIG] * (n * n)
    q = deque()
    for c in sources:
        if dist[c] > 0:
            dist[c] = 0
            q.append(c)
    while q:
        c = q.popleft()
        d = dist[c] + 1
        for m in seg[c]:
            if dist[m] > d:
                dist[m] = d
                q.append(m)
    return dist


def blocker_cells(goal, wr, wd, n=SIZE):
    """(B, wall_stop): B = the cells one step beyond `goal` in the directions
    that are NOT blocked by a wall or the edge; wall_stop = True when some
    direction IS blocked, i.e. a wall alone can stop a robot on the goal."""
    gx, gy = goal % n, goal // n
    B, wall_stop = [], False
    for d in DIRECTIONS:                       # up, down, left, right
        if d == "up":
            blocked = gy == 0 or (gx, gy - 1) in wd
            nxt = (gx, gy - 1)
        elif d == "down":
            blocked = gy == n - 1 or (gx, gy) in wd
            nxt = (gx, gy + 1)
        elif d == "left":
            blocked = gx == 0 or (gx - 1, gy) in wr
            nxt = (gx - 1, gy)
        else:
            blocked = gx == n - 1 or (gx, gy) in wr
            nxt = (gx + 1, gy)
        if blocked:
            wall_stop = True
        else:
            B.append(nxt[1] * n + nxt[0])
    return B, wall_stop


# ---------------------------------------------------------------------------
# one search
# ---------------------------------------------------------------------------

class Search:
    def __init__(self, inst, idx, k, max_expansions, size=SIZE,
                 bound="weak", noise=0.0, rng=None, collect=False,
                 refine_cap=48):
        self.inst = inst
        self.idx = idx
        self.k = k
        self.max_expansions = max_expansions
        self.n = size
        self.bound = bound
        self.noise = float(noise)
        self.rng = rng
        self.collect = collect
        self.refine_cap = refine_cap
        self.env_id = int(inst["env_id"])
        self.RAY, self.PIR = rays(self.env_id, size)
        wr, wd = board(self.env_id)
        self.tidx = int(inst["target_idx"])
        gx, gy = int(inst["target"][0]), int(inst["target"][1])
        self.goal = gy * size + gx
        Hd = relaxed_h((gx, gy), wr, wd, size)          # admissible field
        self.H = [Hd.get((c % size, c // size), BIG) for c in range(size * size)]
        start = tuple(int(p[1]) * size + int(p[0]) for p in inst["positions"])
        self.start = start
        self.best_g = {start: 0}
        self.parent = {start: None}
        self.goal_states = set()
        self.bound_refines = 0        # tight-bound BFS calls actually made
        self.bound_skips = 0          # pops skipped because they cannot improve
        if bound == "tight":
            B, wall_stop = blocker_cells(self.goal, wr, wd, size)
            self.Bset = frozenset(B)
            self.wall_stop = wall_stop
            self.HB = None if (wall_stop or not B) else anystop_field(B, self.env_id,
                                                                     size)
        else:
            self.Bset, self.wall_stop, self.HB = frozenset(), True, None
        self._h2c = {}
        h0 = self._h1(start)
        self.open = [(self.H[start[self.tidx]], 0, 0, start)]
        self.adm = [(h0, 0, start)]
        self.expanded = set()
        self.tie = 1
        self.expansions = 0
        self.h_queries = 0            # (state, robot) queries = encoder passes
        self.h_cands = 0              # candidate children scored
        self.rc_cache = {}
        self.best_cost = None
        self.best_state = None
        self.done = False
        self.reason = None
        self.t0 = time.time()
        self.seconds = 0.0
        self._pending = None
        if start[self.tidx] == self.goal:
            self.best_cost, self.best_state = 0, start
            self.goal_states.add(start)
            self._finish("root is already a goal")

    def _finish(self, reason):
        self.done = True
        self.reason = reason
        self.seconds = time.time() - self.t0

    # -- the admissible bound ------------------------------------------------
    def _rblock(self, st):
        """r(st): a lower bound on the number of NON-target moves any solution
        from `st` must make before the target robot can come to rest on the
        target cell. 0 unless a robot has to become the stopper."""
        if self.wall_stop or self.HB is None:
            return 0
        best = BIG
        for i, c in enumerate(st):
            if i == self.tidx:
                continue
            if c in self.Bset:
                return 0
            d = self.HB[c]
            if d < best:
                best = d
        return best

    def _h1(self, st):
        """The O(1) half of the tight bound: h_free + r. Equals Stage 3's
        board-only bound when bound="weak" (r is then always 0)."""
        hf = self.H[st[self.tidx]]
        if hf >= BIG:
            return BIG
        return hf + self._rblock(st)

    def _h2(self, st):
        """The full tight bound, min(d_frozen, h_free + max(1, r)). Costs one
        slide BFS -- the same one the expansion of `st` needs, so it is cached
        in `rc_cache` and paid at most once."""
        v = self._h2c.get(st)
        if v is not None:
            return v
        t = self.tidx
        if st[t] == self.goal:
            self._h2c[st] = 0
            return 0
        hf = self.H[st[t]]
        if hf >= BIG:
            self._h2c[st] = BIG
            return BIG
        r = self._rblock(st)
        if r >= 1:
            v = hf + r                     # d_frozen is infinite in this case
        else:
            blockers = st[:t] + st[t + 1:]
            key = (st[t], blockers)
            cells = self.rc_cache.get(key)
            if cells is None:
                cells = self.rc_cache[key] = rest_cells_fast(
                    st[t], blockers, self.RAY, self.PIR)
                self.bound_refines += 1
            rec = cells.get(self.goal)
            v = min(BIG if rec is None else rec[0], hf + 1)
        self._h2c[st] = v
        return v

    def _lower_bound(self):
        """min over the OPEN list of g + h_adm: a sound lower bound on the cost
        of any solution the pruned graph still hides. Entries for states that
        have been expanded or superseded are dropped lazily; both are safe,
        because their successors carry their own entries.

        With bound="tight" the entries go in carrying the O(1) bound and the
        few at the TOP of the heap are lazily upgraded to the full one, capped
        at `refine_cap` upgrades per call. Capping only weakens the bound; the
        stored value is a lower bound on the true one either way, so the result
        is always sound."""
        adm = self.adm
        refined = 0
        while adm:
            fa, ga, st = adm[0]
            if st in self.expanded or self.best_g.get(st, -1) != ga:
                heapq.heappop(adm)
                continue
            if self.bound != "tight" or refined >= self.refine_cap:
                return fa
            h2 = self._h2(st)
            if ga + h2 <= fa:
                return fa
            heapq.heappop(adm)
            heapq.heappush(adm, (ga + h2, ga, st))
            refined += 1
        return BIG

    def prepare(self):
        """Pop the next node and enumerate its children; return the heuristic
        queries (one per robot with a non-empty child set), or [] when done."""
        while True:
            if self.done:
                return []
            if self.expansions >= self.max_expansions:
                self._finish("expansion budget")
                return []
            if not self.open:
                self._finish("open list exhausted")
                return []
            if self.best_cost is not None and self.best_cost <= self._lower_bound():
                self._finish("proved optimal in the pruned graph")
                return []
            _f, g, _t, st = heapq.heappop(self.open)
            if g > self.best_g.get(st, BIG):
                continue                                  # stale entry
            if st[self.tidx] == self.goal:
                continue                                  # a goal is never expanded
            if self.best_cost is not None:
                if g >= self.best_cost:
                    continue                              # cannot improve
                if self.bound == "tight":
                    # a node whose g + (admissible) bound already reaches the
                    # incumbent cannot improve it, so expanding it would spend
                    # an expansion of the budget on nothing
                    if g + self._h1(st) >= self.best_cost or \
                            g + self._h2(st) >= self.best_cost:
                        self.bound_skips += 1
                        continue
            groups = []
            for i in range(len(st)):
                blockers = st[:i] + st[i + 1:]
                key = (st[i], blockers)
                cells = self.rc_cache.get(key)
                if cells is None:
                    cells = self.rc_cache[key] = rest_cells_fast(
                        st[i], blockers, self.RAY, self.PIR)
                if cells:
                    groups.append((i, cells))
            if not groups:
                continue
            self.expanded.add(st)
            self.expansions += 1
            self._pending = (st, g, groups)
            self.h_queries += len(groups)
            qs = []
            for i, cells in groups:
                cl = list(cells)
                self.h_cands += len(cl)
                qs.append(dict(env_id=self.env_id, positions=st, robot=i,
                               target_idx=self.tidx, target=self.goal,
                               cells=cl, H=self.H, n=self.n))
            return qs

    def commit(self, hvals):
        """hvals: one array per query, h of each candidate child, in query order."""
        st, g, groups = self._pending
        self._pending = None
        cands = []
        tidx, goal = self.tidx, self.goal
        sigma, rng = self.noise, self.rng
        for (i, cells), hs in zip(groups, hvals):
            is_t = (i == tidx)
            for (cell, rec), h in zip(cells.items(), hs):
                ng = g + rec[0]
                if is_t and cell == goal:
                    cands.append((float(ng), ng, i, cell, True))
                elif sigma:
                    cands.append((ng + float(h) + rng.gauss(0.0, sigma),
                                  ng, i, cell, False))
                else:
                    cands.append((ng + float(h), ng, i, cell, False))
        # deterministic order: score, cheaper g, robot slot, cell index
        cands.sort()
        keep = cands[:self.k] + [c for c in cands[self.k:] if c[4]]
        gmap = dict(groups)
        for f, ng, i, cell, is_goal in keep:
            nxt = st[:i] + (cell,) + st[i + 1:]
            if ng >= self.best_g.get(nxt, BIG):
                continue
            self.best_g[nxt] = ng
            self.expanded.discard(nxt)
            cells = gmap[i]
            dirs, cur = [], cell
            while cur != st[i]:
                _c, dd, pv = cells[cur]
                dirs.append(dd)
                cur = pv
            dirs.reverse()
            self.parent[nxt] = (st, i, dirs)
            if is_goal:
                if self.best_cost is None or ng < self.best_cost:
                    self.best_cost, self.best_state = ng, nxt
                if self.collect:
                    self.goal_states.add(nxt)
                continue                              # a goal is never expanded
            heapq.heappush(self.open, (f, ng, self.tie, nxt))
            heapq.heappush(self.adm, (ng + self._h1(nxt), ng, nxt))
            self.tie += 1

    def _path(self, goal_state):
        """(moves, steps) for the best known path to `goal_state`.
        steps = [(parent_state, robot, cell, child_state), ...] root first."""
        seq, steps = [], []
        cur = goal_state
        while self.parent[cur] is not None:
            prev, i, dirs = self.parent[cur]
            seq.extend([[COLOR_ORDER[i], DIRECTIONS[d]] for d in dirs][::-1])
            steps.append((prev, i, cur[i], cur))
            cur = prev
        seq.reverse()
        steps.reverse()
        return seq, steps

    def sequence(self):
        if self.best_state is None:
            return None
        return self._path(self.best_state)[0]

    def plans(self):
        """Every goal state the search reached, with the best path now known to
        it. Stage 4's self-play harvest: each one is replayed under the real
        rules by the caller and only then does it label anything."""
        out = []
        for gs in self.goal_states:
            moves, steps = self._path(gs)
            out.append(dict(cost=self.best_g[gs], moves=moves, steps=steps,
                            proved=(self.reason ==
                                    "proved optimal in the pruned graph"
                                    and gs is self.best_state)))
        return out

    def row(self):
        seq = self.sequence()
        r = dict(idx=self.idx, env_id=self.env_id,
                 d_star=int(self.inst.get("d_star", -1)),
                 expansions=self.expansions, h_queries=self.h_queries,
                 h_candidates=self.h_cands, seconds=round(self.seconds, 3),
                 stop_reason=self.reason, bound_skips=self.bound_skips,
                 bound_refines=self.bound_refines,
                 proved_optimal_in_pruned_graph=(self.reason ==
                                                 "proved optimal in the pruned graph"))
        if seq is None:
            r.update(solved=False, moves_seq=None, realized_strict=None,
                     search_cost=None, replay_ok=None)
            return r
        n, ok, why = replay(self.inst, seq)
        r.update(solved=bool(ok), moves_seq=seq, realized_strict=n,
                 search_cost=self.best_cost, replay_ok=bool(ok),
                 replay_reason=why, length_matches_g=(n == self.best_cost))
        return r


# ---------------------------------------------------------------------------
# heuristics
# ---------------------------------------------------------------------------

class RelaxedHeuristic:
    """The control: h(child) = the "any-stop" relaxation distance of the target
    robot to the target cell in the CHILD state -- board only, admissible, no
    network. This is the priority `subgoal/space.py` used in Stage 1."""

    name = "relaxed"
    learned = False

    def __init__(self):
        self.passes = 0

    def score(self, queries):
        out = []
        for q in queries:
            H = q["H"]
            if q["robot"] == q["target_idx"]:
                out.append([H[c] for c in q["cells"]])
            else:
                out.append([H[q["positions"][q["target_idx"]]]] * len(q["cells"]))
        return out


class NetHeuristic:
    """The learned h: one `CtgNet` encoder pass per (state, robot) scores all
    256 candidate children of that robot at once."""

    name = "net"
    learned = True

    def __init__(self, ckpt, env_ids, env_dir, size=SIZE, device=None, batch=64):
        import torch
        from nn_labeler import encode as nn_encode
        from subgoal.ctgnet import CtgNet, state_features
        self.torch = torch
        self.state_features = state_features
        self.n = size
        self.batch = batch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = CtgNet.load_from_checkpoint(ckpt, map_location=self.device)
        self.net.eval().to(self.device)
        ids = sorted(set(int(e) for e in env_ids))
        self.slot = {e: i for i, e in enumerate(ids)}
        aa, ai = [], []
        for e in ids:
            A_all, A_ind = nn_encode.adjacency(str(env_dir), e, size)
            aa.append(torch.as_tensor(np.asarray(A_all, np.float32)))
            ai.append(torch.as_tensor(np.asarray(A_ind, np.float32)))
        self.A_all = torch.stack(aa).to(self.device)
        self.A_ind = torch.stack(ai).to(self.device)
        self.passes = 0

    def _xy(self, state):
        n = self.n
        return [(c % n, c // n) for c in state]

    def score(self, queries):
        torch = self.torch
        out = [None] * len(queries)
        for s in range(0, len(queries), self.batch):
            chunk = queries[s:s + self.batch]
            n = self.n
            x = np.stack([self.state_features(self._xy(q["positions"]), q["robot"],
                                              q["target_idx"],
                                              (q["target"] % n, q["target"] // n), n)
                          for q in chunk])
            xi = torch.as_tensor([self.slot[q["env_id"]] for q in chunk],
                                 device=self.device)
            xt = torch.as_tensor(x, device=self.device)
            with torch.no_grad():
                logits = self.net(xt, self.A_all[xi], self.A_ind[xi], n)
                v = self.net.ctg_hat(logits).float().cpu().numpy()
            self.passes += len(chunk)
            for j, q in enumerate(chunk):
                out[s + j] = v[j][np.asarray(q["cells"], np.int64)]
        return out


class ExactHeuristic:
    """Diagnostic ceiling, not a planner: h(child) is the TRUE cost-to-go from
    the exact engine, so f = g + c + h is the true total cost of the best
    solution through that child. It answers "is the search or the heuristic at
    fault?" and is far too expensive to be a system (one exact solve per
    candidate child)."""

    name = "exact"
    learned = False

    def __init__(self, sidecar_dir, env_ids, threads=16):
        from subgoal import rustexact
        self.sidecars = rustexact.ensure_sidecars(env_ids, sidecar_dir, threads=threads)
        self.eng = rustexact.ExactCTG(sidecar_dir, threads=threads)
        self.passes = 0

    def score(self, queries):
        items, spans = [], []
        for q in queries:
            st, i, n = q["positions"], q["robot"], q["n"]
            for cell in q["cells"]:
                ch = st[:i] + (cell,) + st[i + 1:]
                items.append((q["env_id"], [(c % n, c // n) for c in ch],
                              q["target_idx"], (q["target"] % n, q["target"] // n)))
            spans.append(len(q["cells"]))
        vals = self.eng.ask(items, self.sidecars)
        self.passes += len(items)
        out, off = [], 0
        for m in spans:
            out.append([BIG if v is None else float(v) for v in vals[off:off + m]])
            off += m
        return out

    def close(self):
        self.eng.close()


# ---------------------------------------------------------------------------
# the lockstep driver
# ---------------------------------------------------------------------------

def run(instances, heuristic, k, max_expansions, concurrency=64, log_every=50,
        bound="weak", noise=0.0, seed=0, collect=False, on_done=None):
    """Run one search per instance; returns the payload rows in bench order.

    `on_done(search)` is called once per finished search before its state is
    dropped -- Stage 4's self-play harvest hooks in there, so the search trees
    never have to be kept alive all at once."""
    import random as _random
    todo = list(enumerate(instances))
    pool, rows = [], [None] * len(instances)
    t0 = time.time()
    finished = 0
    while todo or pool:
        while todo and len(pool) < concurrency:
            i, inst = todo.pop(0)
            pool.append(Search(inst, i, k, max_expansions, bound=bound,
                               noise=noise, collect=collect,
                               rng=_random.Random(seed * 1000003 + i)
                               if noise else None))
        qs, spans = [], []
        for s in pool:
            q = s.prepare()
            spans.append((s, len(q)))
            qs.extend(q)
        if qs:
            vals = heuristic.score(qs)
            off = 0
            for s, m in spans:
                if m:
                    s.commit(vals[off:off + m])
                off += m
        still = []
        for s in pool:
            if s.done:
                if not s.seconds:
                    s.seconds = time.time() - s.t0
                rows[s.idx] = s.row()
                if on_done is not None:
                    on_done(s)
                finished += 1
                if finished % log_every == 0:
                    print(f"[plan] {finished}/{len(instances)} done, "
                          f"{time.time() - t0:.0f}s elapsed", flush=True)
            else:
                still.append(s)
        pool = still
    return rows


def payload(rows, instances, name, k, max_expansions, extra=None):
    """A payload `subgoal/table.py::score` reads without special cases."""
    solved = [r for r in rows if r["solved"]]
    extras = [r["realized_strict"] - r["d_star"] for r in solved]
    opt = sum(1 for e in extras if e == 0)
    agg = dict(n=len(rows), solved=len(solved),
               pct_optimal=(100.0 * opt / len(solved)) if solved else None,
               pct_optimal_of_n=100.0 * opt / len(rows),
               mean_regret=(sum(extras) / len(extras)) if extras else None,
               expansions_mean=float(np.mean([r["expansions"] for r in rows])),
               h_queries_total=int(sum(r["h_queries"] for r in rows)),
               h_candidates_total=int(sum(r["h_candidates"] for r in rows)),
               proved_optimal_in_pruned_graph=sum(
                   1 for r in rows if r["proved_optimal_in_pruned_graph"]),
               budget_limited=sum(1 for r in rows
                                  if r["stop_reason"] == "expansion budget"),
               bound_skips_total=int(sum(r.get("bound_skips", 0) for r in rows)),
               bound_refines_total=int(sum(r.get("bound_refines", 0)
                                           for r in rows)))
    return {"protocol": {"expansions": max_expansions, "k": k,
                         "instances_file": "supervised_valuenet/eval/data/bench450.jsonl",
                         "search": "subgoal/planner.py best-first over state "
                                   "subgoals, physics edges, f = g + c + h",
                         **(extra or {})},
            "systems": {name: {"rows": rows, "aggregate": agg}}}
