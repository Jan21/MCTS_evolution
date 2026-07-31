"""Size-parametric featurisation for the backward value net.

Faithful port of the frozen 16x16 path, with the grid side `n` promoted from a
module-level constant to a function argument:

  * `node_features`  <- `train/encode.py::_node_features` (lines 236-256) plus the
                        zero global row of `train/looped_pc.py::_x257` (56-57)
  * `key_indices`    <- `train/looped_pc.py::DenseDataset.__getitem__` `ix`/`key`
                        (lines 82-88)
  * `adjacency`      <- `train/looped_pc.py::_adj` (38-53) over
                        `train/encode.py::_graph` (220-233)

Deliberately NO imports from `train.*` / `nn.*`: those modules read RR_GRID and
RR_ENV_DIR at import time (`train/encode.py:26`, `GridEnv.py:11`), which freezes
one board size per process -- exactly what this package exists to avoid.

Pure functions, no module-level environment reads. Arrays returned by the cached
builders (`adjacency`, `sin2d_pe`) are SHARED between callers: treat them as
read-only (this mirrors the frozen code, whose `@lru_cache`d `_adj` hands out the
same tensors to every sample).
"""
from __future__ import annotations

import os
import pickle
from functools import lru_cache

import numpy as np

# 9 binary marker channels, in the exact order of train/encode.py:236-256.
BASE_CHANNELS = 9
# x, y, distance-to-nearest-vertical-edge, distance-to-nearest-horizontal-edge.
COORD_CHANNELS = 4


def channels(coord_channels: bool = False) -> int:
    """Feature width produced by `node_features` for the given options."""
    return BASE_CHANNELS + (COORD_CHANNELS if coord_channels else 0)


def flat_index(p, n: int) -> int:
    """(x, y) -> row of the [n*n+1, C] token matrix. train/encode.py:239."""
    return p[1] * n + p[0]


def node_features(rec: dict, n: int, coord_channels: bool = False) -> np.ndarray:
    """Per-cell binary markers for one candidate decision. [(n*n)+1, C] float32.

    Channels 0-8 are the frozen scheme of `train/encode.py::_node_features`
    (236-256), verbatim:

        0 seg_start            5 every helper robot position (shared channel)
        1 seg_end              6 ctx_open_endpoints
        2 cand_bottleneck      7 ctx_bottlenecks
        3 cand_support         8 ctx_supports
        4 cand_helper position

    Row n*n is the GLOBAL/scratchpad token and is all zeros -- the concatenation
    `train/looped_pc.py::_x257` (56-57) does, folded in here so the caller never
    has to know the row exists.

    `coord_channels=True` appends 4 size-normalised geometry channels per cell
    (zeros on the global row); this is new (PLANS.md S0.1 option (b)) and is off
    by default so the default output is bit-identical to the frozen pipeline.

    Port note: the frozen `_node_features` indexes `seg_start`, `seg_end`,
    `cand_bottleneck`, `cand_support` and `cand_helper[0]` unconditionally, so a
    None in any of them raises there. Those fields are never null in the corpora
    (verified on scaling/data/g16r6/backward.jsonl; only `seg_support` and
    `cand_parent_support`, which this scheme does not use, are ever null). We
    skip Nones instead of raising -- a strict superset of the original behaviour,
    reached only on inputs the original could not process at all.
    """
    size = n * n
    f = np.zeros((size + 1, channels(coord_channels)), np.float32)

    def mark(p, c):
        if p is not None:
            f[p[1] * n + p[0], c] = 1.0

    mark(rec["seg_start"], 0)
    mark(rec["seg_end"], 1)
    mark(rec["cand_bottleneck"], 2)
    mark(rec["cand_support"], 3)
    mark(rec["cand_helper"][0], 4)
    for h in rec["helpers"]:
        mark(h[0], 5)
    # partial-plan context: other open segments + already-committed subgoal cells.
    # cost_to_go is WHOLE-plan completion, so these pending parts shift the value;
    # without them the readout can't see the rest of the plan it must complete.
    for p in rec.get("ctx_open_endpoints", []):
        mark(p, 6)
    for p in rec.get("ctx_bottlenecks", []):
        mark(p, 7)
    for p in rec.get("ctx_supports", []):
        mark(p, 8)

    if coord_channels:
        f[:size, BASE_CHANNELS:] = _coord_planes(n)
    return f


@lru_cache(maxsize=64)
def _coord_planes(n: int) -> np.ndarray:
    """[n*n, 4]: x, y, edge-distance in x, edge-distance in y -- all in [0, 1]."""
    denom = float(max(n - 1, 1))
    x = np.arange(n, dtype=np.float32)
    xs = np.tile(x, n)                      # flat index = y*n + x
    ys = np.repeat(x, n)
    out = np.stack([xs / denom, ys / denom,
                    np.minimum(xs, (n - 1) - xs) / denom,
                    np.minimum(ys, (n - 1) - ys) / denom], 1)
    return np.ascontiguousarray(out, dtype=np.float32)


def key_indices(rec: dict, n: int) -> list[int]:
    """The 5 gathered readout cells as flat token rows.

    Order is the frozen one (train/looped_pc.py:86-88): cand_bottleneck,
    cand_support, cand_helper position, seg_start, seg_end.

    Port note: the frozen `ix(p) = p[1] * GRID + p[0]` has no None branch -- it
    raises on a missing cell. None here maps to the global row `n*n` (a real,
    always-present token) so records with an absent cell degrade to "read the
    global summary" instead of crashing. Not exercised by any existing corpus.
    """
    def ix(p):
        return n * n if p is None else p[1] * n + p[0]

    return [ix(rec["cand_bottleneck"]), ix(rec["cand_support"]),
            ix(rec["cand_helper"][0]), ix(rec["seg_start"]),
            ix(rec["seg_end"])]


def adjacency(env_dir, env_id: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense binary A_all / A_ind in [(n*n)+1, (n*n)+1] float32.

    Port of `train/looped_pc.py::_adj` (38-53) over the slide graph read by
    `train/encode.py::_graph` (220-233):

      * one entry per slide-graph edge, `A[dst, src] = 1` (message src -> dst);
      * A_ind keeps only INDEPENDENT slides -- edges whose data dict has NO
        "dependent" key (etype 0.0 in `_graph`); A_ind is therefore a subset of
        A_all;
      * self-loops on every row including the global row n*n;
      * row/column n*n are otherwise zero, so the global token is reachable only
        through the unmasked global attention head.

    Despite the name these are attention MASKS, not row-normalised operators
    (the `_adj` docstring says "row-normalised", but the code at 44-52 writes
    plain 1.0s -- the returned matrices are binary; masking happens downstream in
    `LoopedLayer._head`, 124-125). Cached per (env_dir, env_id, n); the returned
    arrays are shared -- do not mutate.
    """
    return _adjacency(os.path.abspath(os.fspath(env_dir)), int(env_id), int(n))


@lru_cache(maxsize=4096)
def _adjacency(env_dir: str, env_id: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    # lru_cache only needs hashable ARGUMENTS; returning ndarrays is fine.
    with open(os.path.join(env_dir, f"env_{env_id}.pkl"), "rb") as fh:
        graph = pickle.load(fh)["grid_graph"]
    src, dst, etype = [], [], []
    for u, v, d in graph.edges(data=True):
        src.append(u[1] * n + u[0])
        dst.append(v[1] * n + v[0])
        etype.append(1.0 if "dependent" in d else 0.0)

    size = n * n
    A_all = np.zeros((size + 1, size + 1), np.float32)
    A_ind = np.zeros((size + 1, size + 1), np.float32)
    s = np.asarray(src, dtype=np.int64)
    d = np.asarray(dst, dtype=np.int64)
    et = np.asarray(etype, dtype=np.float32)
    A_all[d, s] = 1.0                       # message j(src)->i(dst): A[i,j]
    ind = et == 0
    A_ind[d[ind], s[ind]] = 1.0
    for i in range(size + 1):               # self-loops incl. the global token
        A_all[i, i] = 1.0
        A_ind[i, i] = 1.0
    return A_all, A_ind


def sin2d_pe(n: int, d_model: int) -> np.ndarray:
    """2-D sin-cos absolute positional encoding. [(n*n)+1, d_model] float32.

    ViT/MAE style: d_model splits in half, one half encoding x and one encoding
    y, each half a standard 1-D sincos table with 10000^(2i/half) frequencies
    over the coordinate values 0..n-1. Layout per cell is [x-half | y-half], and
    each half is [sin(all freqs) | cos(all freqs)]. The global row n*n is zero.

    This replaces the frozen net's learned `self.pos`
    (train/looped_pc.py:146), the ONLY grid-size-dependent weight in the model
    (PLANS.md fact 1). Cached per (n, d_model); shared -- do not mutate.
    """
    return _sin2d_pe(int(n), int(d_model))


@lru_cache(maxsize=64)
def _sin2d_pe(n: int, d_model: int) -> np.ndarray:
    if d_model % 4 != 0:
        raise ValueError(f"d_model must be divisible by 4 (half per axis, "
                         f"sin/cos per half); got {d_model}")
    half = d_model // 2
    coords = np.arange(n, dtype=np.float64)
    # omega_i = 1 / 10000^(2i/half)
    omega = 1.0 / (10000.0 ** (np.arange(half // 2, dtype=np.float64) / (half / 2.0)))
    ang = coords[:, None] * omega[None, :]                       # [n, half/2]
    tab = np.concatenate([np.sin(ang), np.cos(ang)], 1)          # [n, half]

    size = n * n
    pe = np.zeros((size + 1, d_model), np.float32)
    xs = np.tile(np.arange(n), n)                                # flat = y*n + x
    ys = np.repeat(np.arange(n), n)
    pe[:size, :half] = tab[xs]
    pe[:size, half:] = tab[ys]
    return pe
