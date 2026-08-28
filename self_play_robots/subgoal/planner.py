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
# one search
# ---------------------------------------------------------------------------

class Search:
    def __init__(self, inst, idx, k, max_expansions, size=SIZE):
        self.inst = inst
        self.idx = idx
        self.k = k
        self.max_expansions = max_expansions
        self.n = size
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
        h0 = self.H[start[self.tidx]]
        self.open = [(h0, 0, 0, start)]
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
            self._finish("root is already a goal")

    def _finish(self, reason):
        self.done = True
        self.reason = reason
        self.seconds = time.time() - self.t0

    def _lower_bound(self):
        """min over the OPEN list of g + h_adm: a sound lower bound on the cost
        of any solution the pruned graph still hides. Entries for states that
        have been expanded or superseded are dropped lazily; both are safe,
        because their successors carry their own entries."""
        adm = self.adm
        while adm:
            fa, ga, st = adm[0]
            if st in self.expanded or self.best_g.get(st, -1) != ga:
                heapq.heappop(adm)
                continue
            return fa
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
            if self.best_cost is not None and g >= self.best_cost:
                continue                                  # cannot improve
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
        for (i, cells), hs in zip(groups, hvals):
            is_t = (i == tidx)
            for (cell, rec), h in zip(cells.items(), hs):
                ng = g + rec[0]
                if is_t and cell == goal:
                    cands.append((float(ng), ng, i, cell, True))
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
                continue                              # a goal is never expanded
            heapq.heappush(self.open, (f, ng, self.tie, nxt))
            heapq.heappush(self.adm, (ng + self.H[nxt[tidx]], ng, nxt))
            self.tie += 1

    def sequence(self):
        if self.best_state is None:
            return None
        seq = []
        cur = self.best_state
        while self.parent[cur] is not None:
            prev, i, dirs = self.parent[cur]
            seq.extend([[COLOR_ORDER[i], DIRECTIONS[d]] for d in dirs][::-1])
            cur = prev
        seq.reverse()
        return seq

    def row(self):
        seq = self.sequence()
        r = dict(idx=self.idx, env_id=self.env_id, d_star=int(self.inst["d_star"]),
                 expansions=self.expansions, h_queries=self.h_queries,
                 h_candidates=self.h_cands, seconds=round(self.seconds, 3),
                 stop_reason=self.reason,
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

def run(instances, heuristic, k, max_expansions, concurrency=64, log_every=50):
    """Run one search per instance; returns the payload rows in bench order."""
    todo = list(enumerate(instances))
    pool, rows = [], [None] * len(instances)
    t0 = time.time()
    finished = 0
    while todo or pool:
        while todo and len(pool) < concurrency:
            i, inst = todo.pop(0)
            pool.append(Search(inst, i, k, max_expansions))
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
                   1 for r in rows if r["proved_optimal_in_pruned_graph"]))
    return {"protocol": {"expansions": max_expansions, "k": k,
                         "instances_file": "supervised_valuenet/eval/data/bench450.jsonl",
                         "search": "subgoal/planner.py best-first over state "
                                   "subgoals, physics edges, f = g + c + h",
                         **(extra or {})},
            "systems": {name: {"rows": rows, "aggregate": agg}}}
