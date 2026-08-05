"""Lean boards + lazy distance oracle: `GridEnv.from_env` without the O(G^4) wall.

WHY. `nn.gen_grids.make_board` (123-133) generates a board in milliseconds --
`gen_walls` (36) draws the walls, `build_graph` (75) rebuilds the slide graph --
and then spends everything else precomputing two all-pairs distance tables,
`independent_paths` (107) and `all_pairs` (96), each a dict with (G^2)^2 entries
built by `nx.all_pairs_dijkstra_path_length`. `GridEnv.from_env` (GridEnv.py:107)
throws one of those away and *recomputes* it (139-146) at the configured
dependent-edge weight, then `GridEnv.__init__` (46-105) eagerly walks every goal
and every (bottleneck, support) pair, copying the graph once per entry. Measured
cold `from_env`: 8.2 s at 16x16, 60 s at 24x24, 222 s at 32x32; the pkls are
1.6 / 8.8 / 28.6 MB (FINDINGS 60). At 40-96 that is the blocker for
`nn_labeler/descent.py`.

WHAT IS ACTUALLY NEEDED. The descent pipeline never reads a distance table in
bulk. Its only consumers are point queries:

  * `GridEnv.compute_exact_shortest_path_length`   -> reachability_matrix[(s,t)]
  * `GridEnv.compute_relaxed_shortest_path_length` -> relaxed_reachability_matrix
  * `GridEnv.subgoal_score` (269) -- one entry per proposed subgoal
  * `_compute_final_component` (168) touches `reachability_matrix` ONLY when
    `max_final_component_distance` is not None, which this pipeline never sets.

and the encode path (`nn_labeler/encode.py::adjacency`, 135) needs only the slide
graph out of `env_*.pkl`. So the tables can be replaced by an oracle that answers
the same values on demand.

WHAT THIS MODULE PROVIDES.

  * `make_board` / `write_board` -- a board dict with `grid_data`, `grid_graph`,
    `instances` and NO distance tables, produced by the SAME RNG stream as
    `nn.gen_grids.main` (147-150): `gen_walls(rng)` then `random_instance(rng)`,
    with `build_graph` and the two table builders in between consuming nothing.
    Same `(seed, env_id, RR_GRID, RR_WALLS, RR_ROBOTS)` => byte-identical layout.
  * `LazyDistTable` -- a value-identical, read-only stand-in for either table.
    A query (s, t) that misses computes ONE full single-source row (forward from
    s, or backward to t on the reversed graph, whichever the recent miss pattern
    favours) with Dial's algorithm, and memoizes it under an LRU cap. Rows are
    never truncated: a cutoff would turn "far" into `None`, and `None` means
    "no path" everywhere in the planner.
  * `build_env` / `from_env` -- a `GridEnv` whose three eager `__init__` caches
    are the proven-equivalent lazy mirrors of `rust_datagen/pyref/lazy_env.py`
    (`LazyGridEnv`, verified by `pyref/prove_lazy_env.py`) and whose two distance
    tables are `LazyDistTable`. `from_env` is a drop-in for `GridEnv.from_env`.

`from_env` writes no disk cache (the point is that there is nothing expensive to
cache); built envs are kept in a small in-process LRU instead, so descent's
per-board `from_env` call stays free on repeats.

Verified by `nn_labeler/test_leanboard_parity.py`.
"""
from __future__ import annotations

import importlib.util
import os
import pickle
import sys
from collections import OrderedDict
from math import isqrt
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent          # supervised_valuenet/

# nn.gen_grids:31-33 -- kept here so this module never imports gen_grids (which
# freezes GRID / N_INTERIOR / COLORS from RR_* at import time).
PALETTE = ["Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Cyan", "Magenta"]
DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

DEFAULT_DEP_WEIGHT = 2


def default_walls(n: int) -> int:
    """Interior wall segments at stock density (nn/gen_grids.py:30)."""
    return int(round(48 * (n / 16) ** 2))


# ---------------------------------------------------------------------------
# Board generation -- the RNG-consuming half of nn.gen_grids.make_board
# ---------------------------------------------------------------------------

def gen_walls(rng, n: int, n_interior: int) -> list[str]:
    """`nn.gen_grids.gen_walls` (36-72) with GRID/N_INTERIOR passed explicitly.

    The RNG draw sequence is the invariant that matters: `rng.randrange(n)`
    twice then `rng.choice("NSEW")` per attempt, with `continue` (no further
    draws) on an off-board direction or an already-placed side.
    """
    sides = [set() for _ in range(n * n)]

    def idx(x, y):
        return y * n + x

    for x in range(n):
        sides[idx(x, 0)].add("N")
        sides[idx(x, n - 1)].add("S")
    for y in range(n):
        sides[idx(0, y)].add("W")
        sides[idx(n - 1, y)].add("E")

    placed = 0
    while placed < n_interior:
        x, y = rng.randrange(n), rng.randrange(n)
        d = rng.choice("NSEW")
        if d == "E" and x < n - 1:
            a, b = "E", "W"; nx_, ny_ = x + 1, y
        elif d == "W" and x > 0:
            a, b = "W", "E"; nx_, ny_ = x - 1, y
        elif d == "S" and y < n - 1:
            a, b = "S", "N"; nx_, ny_ = x, y + 1
        elif d == "N" and y > 0:
            a, b = "N", "S"; nx_, ny_ = x, y - 1
        else:
            continue
        if a in sides[idx(x, y)]:
            continue
        sides[idx(x, y)].add(a)
        sides[idx(nx_, ny_)].add(b)
        placed += 1

    return ["".join(sorted(s)) for s in sides]


def build_graph(grid_data, n: int):
    """`nn.gen_grids.build_graph` (75-93) with GRID explicit. Consumes no RNG.

    Node and edge INSERTION ORDER is preserved (y-major cells, `DIRS` order):
    it leaks into `propose_subgoal_states` candidate order via set iteration,
    so the mirror has to match line for line. Same body as
    `rust_datagen/pyref/common.py::build_graph_n` (56).
    """
    import networkx as nx
    from simulate import wall_sets, slide

    wr, wd = wall_sets(grid_data, n)
    G = nx.DiGraph()
    G.add_nodes_from((x, y) for y in range(n) for x in range(n))
    for y in range(n):
        for x in range(n):
            c = (x, y)
            for d, (dx, dy) in DIRS.items():
                stop = slide(c, d, frozenset(), wr, wd, n)
                if stop == c:
                    continue
                path, cur = [], c
                while cur != stop:
                    cur = (cur[0] + dx, cur[1] + dy)
                    path.append(cur)
                G.add_edge(c, stop, weight=1)
                for v in path[:-1]:
                    G.add_edge(c, v, weight=2, dependent=(v[0] + dx, v[1] + dy))
    return G


def random_instance(rng, n: int, colors) -> dict:
    """`nn.gen_grids.random_instance` (115-120) with GRID/COLORS explicit."""
    from GridEnv import Robot_at
    pts = rng.sample([(x, y) for y in range(n) for x in range(n)], len(colors) + 1)
    robots = [Robot_at(position=pts[i], color=colors[i]) for i in range(len(colors))]
    return {"target": pts[-1], "target_robot": robots[0],
            "helper_robots": robots[1:]}


def make_board(graph_idx: int, n: int, walls: int | None = None,
               robots: int = 4, seed: int = 0, rng=None) -> dict:
    """`nn.gen_grids.make_board` (123) minus the two all-pairs tables.

    With `rng=None` the RNG is seeded exactly as `nn.gen_grids.main` (147):
    `random.Random(seed * 100000 + graph_idx)`. The draw sequence is
    `gen_walls` then `random_instance` -- `build_graph`, `independent_paths`
    and `all_pairs` consume nothing -- so the walls and the instance are
    byte-identical to what the standard generator would have written.
    """
    import random as _random
    if rng is None:
        rng = _random.Random(seed * 100000 + graph_idx)
    if walls is None:
        walls = default_walls(n)
    grid_data = gen_walls(rng, n, walls)
    G = build_graph(grid_data, n)
    return {
        "graph_idx": graph_idx,
        "grid_data": grid_data,
        "grid_graph": G,
        "instances": [random_instance(rng, n, PALETTE[:robots])],
    }


def write_board(env_dir, graph_idx: int, n: int, walls: int | None = None,
                robots: int = 4, seed: int = 0, overwrite: bool = False) -> Path:
    """Write a lean `env_<idx>.pkl` (atomically). Never clobbers by default.

    The file keeps every key the rest of the pipeline reads -- `grid_data`
    (`eval/realize.py:77`, `simulate.wall_sets`), `grid_graph`
    (`nn_labeler/encode.py:161`), `instances`, `graph_idx` -- and omits only
    `independent_paths` / `all_paths`. NOTE: such a board is NOT loadable by
    `GridEnv.from_env`, whose `reachability_matrix` would be `{}` (GridEnv.py:152);
    use `leanboard.from_env` instead.
    """
    env_dir = Path(env_dir)
    env_dir.mkdir(parents=True, exist_ok=True)
    path = env_dir / f"env_{graph_idx}.pkl"
    if path.exists() and not overwrite:
        return path
    board = make_board(graph_idx, n, walls=walls, robots=robots, seed=seed)
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(board, f)
    os.replace(tmp, path)
    return path


def load_board(env_dir, graph_idx: int) -> dict:
    """Read `env_<idx>.pkl` (lean or full). Full pkls carry the big tables, so
    this is the one place where an existing >=32x32 board still costs seconds."""
    with open(Path(env_dir) / f"env_{graph_idx}.pkl", "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Lazy distance oracle
# ---------------------------------------------------------------------------

def _concat_ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenation of range(s, s+c) for every (s, c). Zero counts allowed."""
    nz = counts > 0
    if not nz.all():
        starts, counts = starts[nz], counts[nz]
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, np.int64)
    out = np.ones(total, np.int64)
    out[0] = starts[0]
    if starts.size > 1:
        ends = np.cumsum(counts)[:-1]
        out[ends] = starts[1:] - (starts[:-1] + counts[:-1]) + 1
    return np.cumsum(out)


class _CSR:
    """Adjacency as (indptr, indices, weights); flat cell index = y*n + x."""
    __slots__ = ("indptr", "indices", "weights", "maxw")

    def __init__(self, indptr, indices, weights):
        self.indptr = indptr
        self.indices = indices
        self.weights = weights
        self.maxw = int(weights.max()) if weights.size else 1


def edge_arrays(G, n: int):
    """(src, dst, weight, is_dependent) over the whole slide graph, once.

    The ONLY Python-level pass over the edges; both tables and both directions
    are derived from these arrays with numpy. Mirrors
    `rust_datagen/pyref/common.py::_adjacency` (159): the same edge set and the
    same `int(d["weight"])`, so the distances below are the ones
    `nn.gen_grids.all_pairs` / `independent_paths` compute.
    """
    src, dst, w, dep = [], [], [], []
    for u, v, d in G.edges(data=True):
        src.append(u[1] * n + u[0])
        dst.append(v[1] * n + v[0])
        w.append(d["weight"])
        dep.append("dependent" in d)
    wa = np.asarray(w if w else [], dtype=float)
    if wa.size:
        if not np.all(wa == np.floor(wa)):
            raise ValueError("non-integer edge weights: bucket relaxation needs "
                             "positive integers (dependent_edge_weight)")
        if wa.min() < 1:
            raise ValueError("edge weight < 1: shortest paths would need a heap, "
                             "not bucket relaxation")
    return (np.asarray(src, np.int64), np.asarray(dst, np.int64),
            wa.astype(np.int32), np.asarray(dep, bool))


def _build_csr(edges, n: int, independent: bool, reverse: bool) -> _CSR:
    src, dst, w, dep = edges
    if independent:
        keep = ~dep
        src, dst, w = src[keep], dst[keep], w[keep]
    if reverse:
        src, dst = dst, src
    order = np.argsort(src, kind="stable")
    src, dst, w = src[order], dst[order], w[order]
    indptr = np.zeros(n * n + 1, np.int64)
    if src.size:
        indptr[1:] = np.bincount(src, minlength=n * n)
    np.cumsum(indptr, out=indptr)
    return _CSR(indptr, dst.astype(np.int32), w.astype(np.int32))


def _dial_row(csr: _CSR, source: int, nn_: int) -> np.ndarray:
    """Single-source shortest paths, exact and untruncated. -1 = unreachable.

    Dial's algorithm: buckets indexed by distance modulo (maxw + 1). Edge
    weights are 1 (independent slide) and `dependent_edge_weight` (2 by
    default), so three buckets suffice and every pop is already final --
    identical results to `nx.all_pairs_dijkstra_path_length`, without a heap.
    """
    dist = np.full(nn_, -1, np.int32)
    indptr, indices, weights = csr.indptr, csr.indices, csr.weights
    nb = csr.maxw + 1
    buckets: list[list[np.ndarray]] = [[] for _ in range(nb)]
    buckets[0].append(np.array([source], np.int64))
    pending = 1
    d = 0
    while pending:
        b = d % nb
        chunk = buckets[b]
        if chunk:
            buckets[b] = []
            cand = chunk[0] if len(chunk) == 1 else np.concatenate(chunk)
            pending -= len(chunk)
            cand = cand[dist[cand] < 0]
            if cand.size:
                cand = np.unique(cand)
                dist[cand] = d
                starts = indptr[cand]
                counts = indptr[cand + 1] - starts
                sel = _concat_ranges(starts, counts)
                if sel.size:
                    nbr = indices[sel]
                    wsel = weights[sel]
                    keep = dist[nbr] < 0
                    nbr, wsel = nbr[keep], wsel[keep]
                    for ww in np.unique(wsel) if wsel.size else ():
                        part = nbr[wsel == ww].astype(np.int64)
                        buckets[(d + int(ww)) % nb].append(part)
                        pending += 1
        d += 1
    return dist


class LazyDistTable:
    """Read-only stand-in for one of the (G^2)^2 distance dicts.

    Supports exactly what `GridEnv` asks of those dicts -- `tbl[(s, t)]` and
    `tbl.get((s, t), default)` (GridEnv.py:276, 384, 393, 401, 410, 212) --
    and never iteration, so value equality is the whole contract (same
    argument as `rust_datagen/pyref/common.py::DistMatrix`, 95).

    A miss computes one FULL single-source row and caches it. Direction is
    chosen adaptively: repeated misses that share a destination (the
    `subgoal_score` pattern -- many bottlenecks, one goal) get a backward row
    on the reversed graph; repeated misses that share a source (the
    `compute_exact_shortest_path_length(start, *, support)` pattern) get a
    forward row. `max_rows` bounds memory at 4 * n^2 bytes per row.
    """

    __slots__ = ("n", "_nn", "_fwd_csr", "_bwd_csr", "_rows", "_max_rows",
                 "_miss_src", "_miss_dst", "queries", "rows_built",
                 "fwd_rows_built", "bwd_rows_built")

    def __init__(self, G, n: int, independent: bool, max_rows: int = 1024,
                 edges=None):
        self.n = n
        self._nn = n * n
        if edges is None:
            edges = edge_arrays(G, n)
        self._fwd_csr = _build_csr(edges, n, independent, reverse=False)
        self._bwd_csr = _build_csr(edges, n, independent, reverse=True)
        self._rows: OrderedDict = OrderedDict()
        self._max_rows = max(2, int(max_rows))
        self._miss_src: dict[int, int] = {}
        self._miss_dst: dict[int, int] = {}
        self.queries = 0
        self.rows_built = 0
        self.fwd_rows_built = 0
        self.bwd_rows_built = 0

    # -- row management ----------------------------------------------------

    def _row(self, key):
        rows = self._rows
        row = rows.get(key)
        if row is not None:
            rows.move_to_end(key)
            return row
        side, idx = key
        csr = self._fwd_csr if side else self._bwd_csr
        row = _dial_row(csr, idx, self._nn)
        rows[key] = row
        self.rows_built += 1
        if side:
            self.fwd_rows_built += 1
        else:
            self.bwd_rows_built += 1
        while len(rows) > self._max_rows:
            rows.popitem(last=False)
        return row

    def _lookup(self, si: int, ti: int):
        rows = self._rows
        row = rows.get((True, si))
        if row is not None:
            rows.move_to_end((True, si))
            v = row[ti]
            return None if v < 0 else int(v)
        row = rows.get((False, ti))
        if row is not None:
            rows.move_to_end((False, ti))
            v = row[si]
            return None if v < 0 else int(v)
        # Miss: build whichever direction the recent miss pattern favours.
        ms = self._miss_src[si] = self._miss_src.get(si, 0) + 1
        md = self._miss_dst[ti] = self._miss_dst.get(ti, 0) + 1
        if ms >= md:
            v = self._row((True, si))[ti]
        else:
            v = self._row((False, ti))[si]
        return None if v < 0 else int(v)

    # -- dict-compatible read API -----------------------------------------

    def __getitem__(self, key):
        s, t = key
        n = self.n
        # A real dict raises KeyError off-board; a bare flat index would wrap
        # to a valid cell and answer silently for the wrong one.
        if not (0 <= s[0] < n and 0 <= s[1] < n and 0 <= t[0] < n and 0 <= t[1] < n):
            raise KeyError(key)
        self.queries += 1
        return self._lookup(s[1] * n + s[0], t[1] * n + t[0])

    def get(self, key, default=None):
        s, t = key
        n = self.n
        if not (0 <= s[0] < n and 0 <= s[1] < n and 0 <= t[0] < n and 0 <= t[1] < n):
            return default
        self.queries += 1
        v = self._lookup(s[1] * n + s[0], t[1] * n + t[0])
        return default if v is None else v

    def __contains__(self, key):
        s, t = key
        n = self.n
        return (0 <= s[0] < n and 0 <= s[1] < n
                and 0 <= t[0] < n and 0 <= t[1] < n)

    def stats(self) -> dict:
        return {"queries": self.queries, "rows_built": self.rows_built,
                "fwd_rows": self.fwd_rows_built, "bwd_rows": self.bwd_rows_built,
                "rows_cached": len(self._rows)}

    # A materialised row is a legitimate bulk answer when a caller really does
    # want a whole source (nothing in the descent path does, today).
    def row_from(self, src) -> np.ndarray:
        return self._row((True, src[1] * self.n + src[0]))

    def row_to(self, dst) -> np.ndarray:
        return self._row((False, dst[1] * self.n + dst[0]))


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------

def _lazy_grid_env_cls():
    """`rust_datagen/pyref/lazy_env.py::LazyGridEnv`, loaded by path.

    That class is the proven lazy mirror of the three eager `GridEnv.__init__`
    caches (`pyref/prove_lazy_env.py` checks it entry-for-entry, including set
    iteration order, which leaks into candidate enumeration order). It lives
    outside a package, so it is loaded from its file rather than imported --
    and cached in `sys.modules` so `isinstance` stays stable.
    """
    name = "nn_labeler._pyref_lazy_env"
    mod = sys.modules.get(name)
    if mod is None:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))       # lazy_env does `from GridEnv import ...`
        path = REPO / "rust_datagen" / "pyref" / "lazy_env.py"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod.LazyGridEnv


def build_env(grid_data, n: int | None = None,
              dependent_edge_weight: float = DEFAULT_DEP_WEIGHT,
              grid_graph=None, max_rows: int = 1024):
    """A `GridEnv` built from the wall layout alone -- no all-pairs precompute.

    Equivalent to `GridEnv.from_env`'s env for `max_final_component_distance=None`:
    same slide graph, same dependent-edge reweighting (GridEnv.py:135-137),
    lazy versions of the three `__init__` caches, and `LazyDistTable` in place
    of the two distance dicts.
    """
    if n is None:
        n = isqrt(len(grid_data))
    assert n * n == len(grid_data), f"grid_data length {len(grid_data)} not a square"
    G = build_graph(grid_data, n) if grid_graph is None else grid_graph
    for _u, _v, d in G.edges(data=True):           # GridEnv.from_env:135-137
        if "dependent" in d:
            d["weight"] = dependent_edge_weight
    edges = edge_arrays(G, n)          # one Python pass, shared by both tables
    ind = LazyDistTable(G, n, independent=True, max_rows=max_rows, edges=edges)
    allp = LazyDistTable(G, n, independent=False, max_rows=max_rows, edges=edges)
    env = _lazy_grid_env_cls()(grid_graph=G, independent_paths=ind, all_paths=allp)
    env.grid_data = list(grid_data)
    env.lean_tables = (ind, allp)
    return env


def env_dir_default() -> Path:
    """`RR_ENV_DIR` if set, else the stock 16x16 dir -- read at CALL time
    (GridEnv.py:11 reads it at import time and freezes it)."""
    return (Path(os.environ["RR_ENV_DIR"]) if "RR_ENV_DIR" in os.environ
            else REPO / "environments")


_ENV_CACHE: OrderedDict = OrderedDict()
_ENV_CACHE_MAX = 2


def from_env(env_index: int, instance_index: int = 0, env_dir=None,
             dependent_edge_weight: float = DEFAULT_DEP_WEIGHT,
             max_final_component_distance=None, max_rows: int = 1024):
    """Drop-in for `GridEnv.from_env` (GridEnv.py:107): returns `(env, State)`.

    Differences, all deliberate:
      * no all-pairs precompute and no `__init__` cache sweep -- both go lazy;
      * no on-disk `cache/env_*_w2_dNone.pkl` (there is nothing costly to
        cache, and the pickle would be larger than the board); a small
        in-process LRU serves descent's repeated per-board calls instead;
      * `max_final_component_distance` must be None -- the capped variant reads
        `reachability_matrix` in bulk inside `_compute_final_component`
        (GridEnv.py:210-214), which is exactly the sweep this module removes.
    """
    if max_final_component_distance is not None:
        raise ValueError("leanboard.from_env supports only "
                         "max_final_component_distance=None")
    env_dir = Path(env_dir) if env_dir is not None else env_dir_default()
    key = (str(env_dir), int(env_index), float(dependent_edge_weight), int(max_rows))
    hit = _ENV_CACHE.get(key)
    if hit is None:
        board = load_board(env_dir, env_index)
        env = build_env(board["grid_data"], grid_graph=board.get("grid_graph"),
                        dependent_edge_weight=dependent_edge_weight,
                        max_rows=max_rows)
        _ENV_CACHE[key] = hit = (env, board["instances"])
        while len(_ENV_CACHE) > _ENV_CACHE_MAX:
            _ENV_CACHE.popitem(last=False)
    else:
        _ENV_CACHE.move_to_end(key)
    env, instances = hit
    from GridEnv import State
    inst = instances[instance_index]
    state = State(target=inst["target"], target_robot=inst["target_robot"],
                  helpers=inst["helper_robots"])
    return env, state


def clear_cache() -> None:
    _ENV_CACHE.clear()


# ---------------------------------------------------------------------------
# CLI: populate a board directory with lean boards
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """Write lean boards, the same layouts `nn.gen_grids` would write.

        python -m nn_labeler.leanboard --n 64 --robots 4 --ids 0-1199 \
            --out environments_g64r4
    """
    import argparse
    p = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    p.add_argument("--n", type=int, required=True, help="grid side")
    p.add_argument("--robots", type=int, default=4)
    p.add_argument("--walls", type=int, default=None,
                   help="interior wall segments (default: stock density)")
    p.add_argument("--ids", required=True, help="board id spec, e.g. 0-1199")
    p.add_argument("--seed", type=int, default=0,
                   help="the --seed nn.gen_grids.main would be run with")
    p.add_argument("--out", required=True)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)

    ids: list[int] = []
    for part in a.ids.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            ids += list(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))

    walls = a.walls if a.walls is not None else default_walls(a.n)
    import time
    t0 = time.time()
    for k, idx in enumerate(ids):
        write_board(a.out, idx, a.n, walls=walls, robots=a.robots,
                    seed=a.seed, overwrite=a.overwrite)
        if k % 100 == 0:
            print(f"env_{idx} ({k + 1}/{len(ids)}) {time.time() - t0:.0f}s",
                  flush=True)
    print(f"LEANBOARD DONE {len(ids)} boards n={a.n} walls={walls} "
          f"robots={a.robots} seed={a.seed} -> {a.out} "
          f"({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
