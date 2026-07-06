"""Shared proposal-net helpers, free of any GNN / torch_geometric dependency.

Extracted from the original train/policy.py (which trained a GNN proposal net we no
longer ship). The transformer proposal net (train/policy_tf.py) and the end-to-end
solver (eval/end2end.py) reuse these three:

  _ix       cell (x,y) -> flat 0..255 index
  _features per-decision 7-channel node features (segment + robots + plan context,
            candidate NOT marked -- the proposal net generates it)
  _meta     per-decision structure: valid bottleneck/support sets, optimal target,
            cost_to_go map (drives the autoregressive mask + regret metric)
"""
from __future__ import annotations

import numpy as np
import torch

from train.encode import GRID


def _ix(p):
    return p[1] * GRID + p[0]


def _features(r):
    """7 channels: segment + robots + partial-plan context. NO candidate cells."""
    f = np.zeros((GRID * GRID, 7), np.float32)
    f[_ix(r["seg_start"]), 0] = 1                       # mover now
    f[_ix(r["seg_end"]), 1] = 1                         # goal
    if r.get("seg_support"):
        f[_ix(r["seg_support"]), 2] = 1                 # pinned support (parent is bottleneck)
    f[_ix(r["target_robot"][0]), 3] = 1                 # robots
    for h in r["helpers"]:
        f[_ix(h[0]), 3] = 1
    for p in r.get("ctx_open_endpoints", []):
        f[_ix(p), 4] = 1
    for p in r.get("ctx_bottlenecks", []):
        f[_ix(p), 5] = 1
    for p in r.get("ctx_supports", []):
        f[_ix(p), 6] = 1
    return torch.from_numpy(f)


def _meta(group):
    """Per-decision: valid sets, optimal target, cost_to_go map (for regret)."""
    g0 = group[0]
    helper_cells = [_ix(h[0]) for h in g0["helpers"]]
    hidx = {}
    for i, h in enumerate(g0["helpers"]):
        hidx[tuple(h[0])] = i
    cands = []
    for r in group:
        hi = hidx.get(tuple(r["cand_helper"][0]))
        if hi is None:
            continue
        cands.append((_ix(r["cand_bottleneck"]), _ix(r["cand_support"]),
                      hi, int(r["cost_to_go"]), bool(r["is_optimal"])))
    if not cands:
        return None
    valid_bn = sorted({c[0] for c in cands})
    sup_by_bn = {}
    for bn, sp, hi, ctg, opt in cands:
        sup_by_bn.setdefault(bn, set()).add(sp)
    sup_by_bn = {k: sorted(v) for k, v in sup_by_bn.items()}
    best = min(c[3] for c in cands)
    opts = [c for c in cands if c[4]] or [c for c in cands if c[3] == best]
    o = min(opts, key=lambda c: c[3])
    return dict(rec=g0, env_id=g0["env_id"],
                seg_start=_ix(g0["seg_start"]), seg_end=_ix(g0["seg_end"]),
                helper_cells=helper_cells, valid_bn=valid_bn, sup_by_bn=sup_by_bn,
                tgt_bn=o[0], tgt_sup=o[1], tgt_helper=o[2],
                cands=[(c[0], c[1], c[2], c[3]) for c in cands],   # (bn, sup, helper, ctg)
                ctg_map={(c[0], c[1], c[2]): c[3] for c in cands}, best=best)
