"""Stage 1 of PLAN_SUBGOAL_DISCOVERY.md: the ceiling of the STATE subgoal space.

The plan's new subgoal is "robot R comes to rest on cell C" -- 4 x 256 = 1024
candidates per decision at 16x16 with four robots, against the hand-written
proposer's geometric (bottleneck, support, helper) triples. This module
measures what that space can EXPRESS, with no network anywhere, exactly as
`spr/ceiling.py` does for the hand-written language:

  node        the joint robot positions (a full game state) + moves spent
  expand      for every robot, the exact set of cells it can come to rest on,
              found by BFS over `simulate.slide` with EVERY OTHER ROBOT frozen
              at its current cell -- the real joint-game rule, and the same
              physics `eval/realize.py::strict_moves` certifies with. Each
              reachable cell is one child; its edge cost is the true number of
              slides. Unreachable cells are dropped.
  order       f = g + h, h = the "any-stop" relaxation distance of the TARGET
              robot to the target cell (one move may end on any cell of the
              row/column segment, walls only, robots ignored). Every real move
              is a relaxed move, so h is an admissible and consistent lower
              bound on the remaining slides -- the trivial heuristic the plan
              asks for. h is board-only: no network, no d*.
  terminate   the target robot standing on the target cell. With a consistent
              h the first goal state popped is OPTIMAL IN THIS SPACE, so the
              number reported is the space's ceiling, not a sample of it.

Because a macro edge moves ONE robot while the others stand still, the
concatenation of the edges of any path is a legal primitive move sequence by
construction; it is replayed and certified anyway (`certify`, the physics of
`eval/replay_validate.py`).

NOTE on `GridEnv.compute_exact_shortest_path_length`, which the plan suggests
for this: it reads `reachability_matrix`, a BLOCKER-FREE lone-robot distance
built from the plan DAG. It cannot see a helper robot acting as a stopper, so
it both over-costs and mis-reports reachability in a joint state. The BFS here
is the exact joint-state answer and is what the certifier requires.

    PYTHONPATH=supervised_valuenet:self_play_robots python -m subgoal.space \
        --n 20 --out results/subgoal/stage1_state_space.json
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import pickle
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPR = HERE.parent
REPO = SPR.parent
SV = REPO / "supervised_valuenet"
sys.path.insert(0, str(SV))
sys.path.insert(0, str(SPR))

from simulate import DIRECTIONS, slide, wall_sets      # noqa: E402
from subgoal.table import BENCH, ENV_DIR, SIZE, COLOR_ORDER, load_bench  # noqa: E402

_BOARDS: dict[int, tuple] = {}


def board(env_id):
    if env_id not in _BOARDS:
        with open(ENV_DIR / f"env_{env_id}.pkl", "rb") as f:
            grid_data = pickle.load(f)["grid_data"]
        _BOARDS[env_id] = wall_sets(grid_data, SIZE)
    return _BOARDS[env_id]


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------

def rest_cells(start, blockers, wr, wd, size=SIZE):
    """{cell: (slides, direction, from_cell)} -- every cell one robot can come
    to rest on with `blockers` frozen, and the cheapest way in. Exact BFS over
    the real slide rule; a slide that does not move the robot is not an edge."""
    out = {start: (0, None, None)}
    q = deque([start])
    while q:
        cur = q.popleft()
        g = out[cur][0]
        for d in DIRECTIONS:
            nxt = slide(cur, d, blockers, wr, wd, size)
            if nxt == cur or nxt in out:
                continue
            out[nxt] = (g + 1, d, cur)
            q.append(nxt)
    del out[start]
    return out


def relaxed_h(target, wr, wd, size=SIZE):
    """Admissible lower bound: BFS from `target` where one move goes to ANY
    cell of the row/column segment (walls only, robots ignored). Real moves are
    a subset of these, so this never over-estimates."""
    seg = {}
    for y in range(size):
        for x in range(size):
            nbr = []
            for dx, dy, wall in ((1, 0, wr), (-1, 0, None), (0, 1, wd), (0, -1, None)):
                cx, cy = x, y
                while True:
                    if dx == 1:
                        if cx == size - 1 or (cx, cy) in wr:
                            break
                        cx += 1
                    elif dx == -1:
                        if cx == 0 or (cx - 1, cy) in wr:
                            break
                        cx -= 1
                    elif dy == 1:
                        if cy == size - 1 or (cx, cy) in wd:
                            break
                        cy += 1
                    else:
                        if cy == 0 or (cx, cy - 1) in wd:
                            break
                        cy -= 1
                    nbr.append((cx, cy))
            seg[(x, y)] = nbr
    dist = {target: 0}
    q = deque([target])
    while q:
        c = q.popleft()
        for n in seg[c]:
            if n not in dist:
                dist[n] = dist[c] + 1
                q.append(n)
    return dist


# ---------------------------------------------------------------------------
# the search over states
# ---------------------------------------------------------------------------

def search(inst, max_pops=200_000, time_cap=120.0):
    """A* over joint states with (robot, cell) macro edges. Returns the
    ceiling of the state subgoal space on this instance."""
    wr, wd = board(inst["env_id"])
    start = tuple(tuple(p) for p in inst["positions"])
    tidx = inst["target_idx"]
    goal = tuple(inst["target"])
    H = relaxed_h(goal, wr, wd)
    BIG = 10 ** 6

    def h(state):
        return H.get(state[tidx], BIG)

    t0 = time.time()
    rc_cache = {}                          # (cell, blockers) -> rest_cells
    best_g = {start: 0}
    parent = {start: None}                # state -> (prev_state, robot, [dirs])
    heap = [(h(start), 0, 0, start)]
    tie = 1
    pops = expansions = edges = 0
    reached = None
    capped = False
    while heap:
        f, g, _, st = heapq.heappop(heap)
        pops += 1
        if g > best_g.get(st, BIG):
            continue
        if st[tidx] == goal:
            reached = (st, g)
            break
        if pops > max_pops or (time.time() - t0) > time_cap:
            capped = True
            break
        expansions += 1
        for i in range(len(st)):
            blockers = frozenset(c for j, c in enumerate(st) if j != i)
            key = (st[i], blockers)
            cells = rc_cache.get(key)
            if cells is None:
                cells = rc_cache[key] = rest_cells(st[i], blockers, wr, wd)
            edges += len(cells)
            for cell, (cost, _d, _p) in cells.items():
                nxt = st[:i] + (cell,) + st[i + 1:]
                ng = g + cost
                if ng >= best_g.get(nxt, BIG):
                    continue
                best_g[nxt] = ng
                # reconstruct this macro's primitive directions
                dirs, cur = [], cell
                while cur != st[i]:
                    _c, dd, pv = cells[cur]
                    dirs.append(dd)
                    cur = pv
                dirs.reverse()
                parent[nxt] = (st, i, dirs)
                heapq.heappush(heap, (ng + h(nxt), ng, tie, nxt))
                tie += 1

    res = dict(idx=inst.get("idx"), env_id=inst["env_id"], d_star=inst["d_star"],
               pops=pops, expansions=expansions, edges_generated=edges,
               seconds=round(time.time() - t0, 2), capped=capped,
               proved_optimal=(reached is not None and not capped))
    if reached is None:
        res.update(moves=None, n_subgoals=None, subgoals=None, seq=None)
        return res
    st, g = reached
    seq, subgoals = [], []
    cur = st
    while parent[cur] is not None:
        prev, i, dirs = parent[cur]
        subgoals.append([COLOR_ORDER[i], list(cur[i])])
        seq.extend([[COLOR_ORDER[i], d] for d in dirs][::-1])
        cur = prev
    seq.reverse()
    subgoals.reverse()
    res.update(moves=g, n_subgoals=len(subgoals), subgoals=subgoals, seq=seq,
               path_len_matches_g=(len(seq) == g))
    return res


def certify(inst, seq):
    """The physics of eval/replay_validate.py: every move a real full slide,
    target robot on the target at the end. (ok, reason, length)."""
    wr, wd = board(inst["env_id"])
    pos = {COLOR_ORDER[i]: tuple(p) for i, p in enumerate(inst["positions"])}
    for j, (color, d) in enumerate(seq):
        blockers = frozenset(p for c, p in pos.items() if c != color)
        nxt = slide(pos[color], d, blockers, wr, wd, SIZE)
        if nxt == pos[color]:
            return False, f"move {j}: {color} {d} is a no-op", len(seq)
        pos[color] = nxt
    if pos[COLOR_ORDER[inst["target_idx"]]] != tuple(inst["target"]):
        return False, "target robot not on target", len(seq)
    return True, "ok", len(seq)


# ---------------------------------------------------------------------------
# the hand-written language's ceiling, per instance, for the same rows
# ---------------------------------------------------------------------------

def language_ceiling(path):
    """{idx: row} of spr.ceiling output (`best_realizable_moves` per instance)."""
    d = json.loads(Path(path).read_text())
    return {r["idx"]: r for r in d["rows"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bench", default=str(BENCH))
    p.add_argument("--n", type=int, default=20, help="how many instances")
    p.add_argument("--stride", type=int, default=1,
                   help="take every STRIDE-th instance (1 = the first N; a "
                        "stride sample spans the whole difficulty range). "
                        "Deterministic either way -- no instance is chosen for "
                        "its result.")
    p.add_argument("--max-pops", type=int, default=200_000)
    p.add_argument("--time-cap", type=float, default=120.0)
    p.add_argument("--base-ceiling",
                   default=str(SPR / "results/ceiling/g16r4_base.json"))
    p.add_argument("--b2-ceiling",
                   default=str(SPR / "results/ceiling/g16r4_b2.json"))
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    full = load_bench(a.bench)
    for i, inst in enumerate(full):
        inst["idx"] = i
    bench = full[::a.stride][:a.n]
    base = language_ceiling(a.base_ceiling)
    b2 = language_ceiling(a.b2_ceiling)

    rows = []
    t0 = time.time()
    for inst in bench:
        r = search(inst, a.max_pops, a.time_cap)
        if r["seq"]:
            ok, why, n = certify(inst, r["seq"])
            r.update(certified=ok, certify_reason=why, certified_moves=n)
        else:
            r.update(certified=False, certify_reason="no solution", certified_moves=None)
        for tag, src in (("base", base), ("b2", b2)):
            row = src.get(inst["idx"], {})
            r[f"{tag}_moves"] = row.get("best_realizable_moves")
            r[f"{tag}_category"] = row.get("category")
            r[f"{tag}_capped"] = row.get("capped")
        rows.append(r)
        print(f"[stage1] idx {inst['idx']:3d} env {inst['env_id']} d*={inst['d_star']:2d} "
              f"state={r['certified_moves']} (subgoals {r['n_subgoals']}, "
              f"opt={r['proved_optimal']}, cert={r['certified']}) "
              f"base={r['base_moves']} b2={r['b2_moves']} "
              f"[{r['pops']} pops, {r['seconds']}s]", flush=True)

    def summ(key, only=None):
        vals = [(r[key] - r["d_star"]) for r in rows
                if r.get(key) is not None and (only is None or only(r))]
        return dict(n=len(vals),
                    mean_extra=(sum(vals) / len(vals)) if vals else None,
                    n_optimal=sum(1 for v in vals if v == 0))

    summary = {
        "n": len(rows),
        "state_space": summ("certified_moves", lambda r: r.get("certified")),
        "hand_written_base": summ("base_moves"),
        "hand_written_b2": summ("b2_moves"),
        "state_all_proved_optimal": all(r["proved_optimal"] for r in rows),
        "state_all_certified": all(r["certified"] for r in rows),
        "mean_d_star": sum(r["d_star"] for r in rows) / len(rows),
        "mean_subgoals": sum(r["n_subgoals"] for r in rows if r["n_subgoals"] is not None)
                         / max(1, sum(1 for r in rows if r["n_subgoals"] is not None)),
        "wall_seconds": round(time.time() - t0, 1),
    }
    print("\n[stage1] summary:", json.dumps(summary, indent=1))

    if a.out:
        out = Path(a.out)
        if not out.is_absolute():
            out = SPR / out
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            bench=a.bench, n=a.n, stride=a.stride,
            instance_indices=[r["idx"] for r in rows],
            caps=dict(max_pops=a.max_pops, time_cap=a.time_cap),
            base_ceiling=a.base_ceiling, b2_ceiling=a.b2_ceiling,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            date=time.strftime("%Y-%m-%dT%H:%M:%S"),
            summary=summary, rows=rows)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1) + "\n")
        os.replace(tmp, out)
        print(f"SPR SUBGOAL STAGE1 DONE {out}")


if __name__ == "__main__":
    main()
