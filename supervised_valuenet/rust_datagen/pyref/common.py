"""Shared helpers for the pyref cross-engine verification harness.

Everything here is engine-agnostic plumbing used by dump_decisions.py,
replay_python.py and prove_lazy_env.py:

- a size-parameterised mirror of `nn.gen_grids.build_graph` (the module-level
  original reads RR_GRID at import; this one takes `n` explicitly so one
  process can handle any board size),
- fast all-pairs tables (value-identical to `nn.gen_grids.all_pairs` /
  `independent_paths`, verified by prove_lazy_env.py) backed by a compact
  int16 matrix and cached under pyref/cache/,
- `build_env` -- GridEnv construction from grid_data alone (lazy or eager),
- JSON (de)serialisation of State / PartialPlan / Candidate objects, matching
  the replay schemas of DESIGN.md section 3 exactly.

No file in the existing repo is modified; repo modules are imported read-only.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

PYREF_DIR = Path(__file__).resolve().parent
SV_DIR = PYREF_DIR.parent.parent            # supervised_valuenet/
CACHE_DIR = PYREF_DIR / "cache"

for _p in (str(SV_DIR), str(PYREF_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import networkx as nx  # noqa: E402

# Canonical palette (nn/gen_grids.py); robot slot i == PALETTE[i].
PALETTE = ["Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Cyan", "Magenta"]

# Direction order used by the graph builder (nn/gen_grids.py DIRS dict order).
DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def grid_sha(grid_data) -> str:
    return hashlib.sha256("\n".join(grid_data).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Board -> slide graph (mirror of nn.gen_grids.build_graph, explicit size)
# ---------------------------------------------------------------------------

def build_graph_n(grid_data, n) -> nx.DiGraph:
    """Faithful copy of nn.gen_grids.build_graph with GRID passed explicitly.

    Node and edge INSERTION ORDER is preserved exactly (y-major cells,
    directions in DIRS order) -- it leaks into candidate enumeration order
    downstream, so the mirror must match line for line.
    """
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


def reweight_dependent(G: nx.DiGraph, weight) -> None:
    """Set dependent-edge weight (mirror of GridEnv.from_env)."""
    for u, v, d in G.edges(data=True):
        if "dependent" in d:
            d["weight"] = weight


# ---------------------------------------------------------------------------
# All-pairs shortest-path tables
# ---------------------------------------------------------------------------

class DistMatrix:
    """(src, dst) -> int distance | None, backed by an int16 matrix.

    Drop-in for the dict tables GridEnv reads: only __getitem__ / get are ever
    used by GridEnv (lookups, never iteration), so value equality is the whole
    contract. -1 encodes unreachable (None).
    """
    __slots__ = ("n", "m")

    def __init__(self, n: int, m: np.ndarray):
        self.n = n
        self.m = m

    def _i(self, p):
        return p[1] * self.n + p[0]

    def __getitem__(self, key):
        s, t = key
        v = self.m[self._i(s), self._i(t)]
        return None if v < 0 else int(v)

    def get(self, key, default=None):
        s, t = key
        n = self.n
        if not (0 <= s[0] < n and 0 <= s[1] < n and 0 <= t[0] < n and 0 <= t[1] < n):
            return default
        v = self.m[self._i(s), self._i(t)]
        return None if v < 0 else int(v)


# module globals for fork-based workers
_ADJ = None
_NN = 0


def _dijkstra_row(src: int) -> np.ndarray:
    """Single-source Dijkstra over the global adjacency; int distances."""
    INFD = float("inf")
    dist = [INFD] * _NN
    dist[src] = 0
    pq = [(0, src)]
    adj = _ADJ
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    row = np.full(_NN, -1, dtype=np.int16)
    for i, d in enumerate(dist):
        if d != INFD:
            assert d < 32000, "distance overflows int16"
            row[i] = d
    return row


def _dijkstra_chunk(args):
    lo, hi = args
    return lo, np.stack([_dijkstra_row(s) for s in range(lo, hi)])


def _adjacency(G, n, independent: bool):
    """Adjacency as index lists. independent=True drops dependent edges
    (mirror of nn.gen_grids.independent_paths' subgraph)."""
    nn_ = n * n
    adj = [[] for _ in range(nn_)]
    for u, v, d in G.edges(data=True):
        if independent and "dependent" in d:
            continue
        adj[u[1] * n + u[0]].append((v[1] * n + v[0], int(d["weight"])))
    return adj


def _all_pairs_matrix(G, n, independent, workers=1) -> np.ndarray:
    global _ADJ, _NN
    _ADJ = _adjacency(G, n, independent)
    _NN = n * n
    if workers <= 1:
        m = np.stack([_dijkstra_row(s) for s in range(_NN)])
    else:
        import multiprocessing as mp
        chunk = max(1, (_NN + workers * 4 - 1) // (workers * 4))
        spans = [(lo, min(lo + chunk, _NN)) for lo in range(0, _NN, chunk)]
        ctx = mp.get_context("fork")
        with ctx.Pool(min(workers, 16)) as pool:
            parts = pool.map(_dijkstra_chunk, spans)
        m = np.empty((_NN, _NN), dtype=np.int16)
        for lo, block in parts:
            m[lo:lo + block.shape[0]] = block
    _ADJ = None
    return m


def tables_cache_path(grid_data, n, weight) -> Path:
    return CACHE_DIR / "tables" / f"{grid_sha(grid_data)[:20]}_n{n}_w{weight}.npz"


def load_or_build_tables(G, grid_data, n, weight, workers=1, use_cache=True):
    """(independent DistMatrix, all-pairs DistMatrix); disk-cached by content."""
    path = tables_cache_path(grid_data, n, weight)
    if use_cache and path.exists():
        z = np.load(path)
        return DistMatrix(n, z["ind"]), DistMatrix(n, z["allp"])
    ind = _all_pairs_matrix(G, n, independent=True, workers=workers)
    allp = _all_pairs_matrix(G, n, independent=False, workers=workers)
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".npz")
        os.close(fd)
        np.savez_compressed(tmp, ind=ind, allp=allp)
        os.replace(tmp, path)
    return DistMatrix(n, ind), DistMatrix(n, allp)


# ---------------------------------------------------------------------------
# GridEnv construction from grid_data alone
# ---------------------------------------------------------------------------

def build_env(grid_data, n, mode="lazy", weight=2, table_workers=1,
              use_cache=True):
    """A GridEnv equivalent built purely from the wall layout.

    mode="eager"  -> the real GridEnv.__init__ (full cache precompute).
    mode="lazy"   -> lazy_env.LazyGridEnv (identical semantics on demand;
                     equality proven by prove_lazy_env.py).
    Both sides of a dump/replay pair use this same builder, so candidate
    enumeration order (which leaks through set iteration) is reproducible.
    """
    G = build_graph_n(grid_data, n)
    reweight_dependent(G, weight)
    ind, allp = load_or_build_tables(G, grid_data, n, weight,
                                     workers=table_workers, use_cache=use_cache)
    if mode == "eager":
        from GridEnv import GridEnv
        env = GridEnv(grid_graph=G, independent_paths=ind, all_paths=allp,
                      max_final_component_distance=None)
    elif mode == "lazy":
        from lazy_env import LazyGridEnv
        env = LazyGridEnv(grid_graph=G, independent_paths=ind, all_paths=allp)
    else:
        raise ValueError(f"bad env mode {mode!r}")
    env.grid_data = list(grid_data)
    return env


def load_board_grid_data(env_dir, env_id) -> list:
    with open(Path(env_dir) / f"env_{env_id}.pkl", "rb") as f:
        return pickle.load(f)["grid_data"]


def board_obj(env_id, n, grid_data) -> dict:
    return {"env_id": int(env_id), "n": int(n), "grid_data": list(grid_data)}


# ---------------------------------------------------------------------------
# JSON (de)serialisation -- DESIGN.md section 3 shapes
# ---------------------------------------------------------------------------

def xy(p):
    return [int(p[0]), int(p[1])] if p is not None else None


def de_xy(p):
    return None if p is None else (int(p[0]), int(p[1]))


def ser_robot(r):
    return [xy(r.position), r.color]


def de_robot(o):
    from GridEnv import Robot_at
    return Robot_at(position=de_xy(o[0]), color=o[1])


def ser_state(state) -> dict:
    return {"target": xy(state.target),
            "target_robot": ser_robot(state.target_robot),
            "helpers": [ser_robot(h) for h in state.helpers]}


def de_state(o):
    from GridEnv import State
    return State(target=de_xy(o["target"]),
                 target_robot=de_robot(o["target_robot"]),
                 helpers=[de_robot(h) for h in o["helpers"]])


def ser_plan(plan) -> dict:
    """Nodes and edges in insertion order + nc (DESIGN section 3)."""
    nodes = []
    for nid, d in plan.g.nodes(data=True):
        nt = d["ntype"]
        if nt == "goal":
            attrs = {"pos": xy(d["pos"])}
        elif nt == "subgoal":
            attrs = {"parent_support_pos": xy(d.get("parent_support_pos"))}
        else:  # bottleneck / support / leaf
            attrs = {"pos": xy(d["pos"]), "robot": ser_robot(d["robot"])}
        nodes.append([nid, nt, attrs])
    edges = [[u, v, d["status"],
              None if d["cost"] is None else int(d["cost"])]
             for u, v, d in plan.g.edges(data=True)]
    return {"nodes": nodes, "edges": edges, "nc": int(plan.nc)}


def de_plan(o):
    from partial_plan import PartialPlan
    plan = PartialPlan()
    for nid, nt, attrs in o["nodes"]:
        if nt == "goal":
            plan.add_node(nid, nt, pos=de_xy(attrs["pos"]))
        elif nt == "subgoal":
            plan.add_node(nid, nt,
                          parent_support_pos=de_xy(attrs["parent_support_pos"]))
        else:
            plan.add_node(nid, nt, pos=de_xy(attrs["pos"]),
                          robot=de_robot(attrs["robot"]))
    for u, v, status, cost in o["edges"]:
        plan.add_edge(u, v, status=status,
                      cost=None if cost is None else int(cost))
    plan.nc = int(o["nc"])
    return plan


def ser_candidate(c) -> dict:
    return {"bottleneck": xy(c.subgoal.bottleneck.position),
            "support": xy(c.subgoal.support.position),
            "helper": ser_robot(c.subgoal.helper),
            "parent_support": xy(c.parent_support)}


def de_candidate(o, seg):
    """Rebuild a heuristics.Candidate from its dump + the segment context.

    Colors are structural: the bottleneck robot always carries the segment
    mover's color and the support robot the helper's color (GridEnv.
    propose_subgoal_states), so the dump fields are lossless.
    """
    from GridEnv import Robot_at, Subgoal
    from skeleton.heuristics import Candidate
    bn = de_xy(o["bottleneck"])
    sp = de_xy(o["support"])
    helper = de_robot(o["helper"])
    sub = Subgoal(bottleneck=Robot_at(position=bn, color=seg.mover.color),
                  support=Robot_at(position=sp, color=helper.color),
                  goal_pos=seg.end, target_robot=seg.mover, helper=helper)
    return Candidate(subgoal=sub, parent_support=de_xy(o["parent_support"]),
                     score=0.0)


def jdump(obj) -> str:
    return json.dumps(obj)
