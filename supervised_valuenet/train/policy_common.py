"""Shared proposal-net helpers, free of any GNN / torch_geometric dependency.

Extracted from the original train/policy.py (which trained a GNN proposal net we no
longer ship). The transformer proposal net (train/policy_tf.py) and the end-to-end
solver (eval/end2end.py) reuse these three:

  _ix       cell (x,y) -> flat 0..GRID*GRID-1 index
  _features per-decision 7-channel node features (segment + robots + plan context,
            candidate NOT marked -- the proposal net generates it)
  _meta     per-decision structure: valid bottleneck/support sets, optimal target,
            cost_to_go map (drives the autoregressive mask + regret metric)
"""
from __future__ import annotations

import os

import numpy as np
import torch

from train.encode import GRID

# Lever B2 record filter (FINDINGS 40). `_meta` historically keyed a
# candidate's helper by its robot's START cell, so every BY-REFERENCE record --
# helper standing at a cell the plan itself will create -- was silently dropped
# from the proposal net's training signal (the value net, on raw cell indices,
# saw them). Setting RR_BYREF_RECORDS=1 (or passing byref=True) resolves the
# helper by ROBOT COLOUR instead, so those records train. Default OFF: every
# banked policy checkpoint was trained with the filter in place, and turning it
# on changes the per-decision candidate set (valid_bn / sup_by_bn / the optimal
# target), hence the training signal.
BYREF_RECORDS = os.environ.get("RR_BYREF_RECORDS", "") == "1"


def _ix(p):
    return p[1] * GRID + p[0]


def _features(r):
    """7 channels: segment + robots + partial-plan context. NO candidate cells.

    DESIGN NOTE, NOT IMPLEMENTED -- naming the referenced cell to the policy
    net (Lever B2). With RR_BYREF_RECORDS on, a by-reference candidate is
    scored through its robot's identity slot, whose embedding sits at the
    robot's START cell; nothing in these 7 channels tells the net WHICH planned
    cell the helper is being referenced at, so two by-reference candidates that
    differ only in the referenced cell are indistinguishable to the proposal
    head (the value net does see it, via the raw `cand_helper` cell index in
    its key). The fix is an 8th channel marking the planned cells offered by
    `skeleton/astar.py::_reference_helpers` at this decision (and, for a per-
    candidate variant, a 9th marking the one cell this candidate references).
    That changes `_features`' output width, hence `train/policy_tf.py::_x257`'s
    input width (its `torch.zeros(1, 7)` global row hardcodes it too), hence the first Linear of every banked policy checkpoint --
    `PolicyTF.load_from_checkpoint` would fail with a size mismatch on every
    existing net. It is therefore deliberately NOT done here: adding it is a
    from-scratch retrain of the whole policy family, to be decided after the
    zero-shot supply A/B (jobs/patterns/byref_ab.slurm) says whether the
    identity-only featurization is already enough. Migration sketch if adopted:
    bump the channel count in one place (a module constant), retrain every
    policy net cold, and re-bank -- no value-net or checkpoint-format change is
    involved.
    """
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


def _meta(group, byref=None):
    """Per-decision: valid sets, optimal target, cost_to_go map (for regret).

    `byref` (None = module default `BYREF_RECORDS`, itself from
    RR_BYREF_RECORDS) controls the by-reference record filter: with it False a
    candidate whose helper is not at a robot START cell is skipped (historical
    behaviour, and what every banked policy net was trained under); with it
    True the helper is resolved by ROBOT COLOUR, so by-reference candidates
    enter `cands`, `valid_bn`, `sup_by_bn` and the optimal-target selection.
    Colour resolution is a fallback AFTER the position match, so with byref
    True the accepted set is a strict superset of the byref-False set.

    The helper slot is an identity index into `g0["helpers"]`, i.e. the policy
    net scores a by-reference candidate at its robot's START-cell embedding;
    the referenced cell is not named to the policy (see `_features`).
    """
    if byref is None:
        byref = BYREF_RECORDS
    g0 = group[0]
    helper_cells = [_ix(h[0]) for h in g0["helpers"]]
    hidx = {}
    cidx = {}
    for i, h in enumerate(g0["helpers"]):
        hidx[tuple(h[0])] = i
        cidx[h[1]] = i
    cands = []
    for r in group:
        hi = hidx.get(tuple(r["cand_helper"][0]))
        if hi is None and byref:
            hi = cidx.get(r["cand_helper"][1])
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
