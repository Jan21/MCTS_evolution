"""Size-FREE fork of the backward value net (`train/looped_pc.py`).

The reference `LoopedValueNet` is grid-size-locked by exactly one tensor -- the
learned absolute positional embedding `self.pos: [G*G+1, d_model]` (looped_pc.py:146)
-- plus the module-level `GRID`/`ROBOTS` constants that `train.encode` freezes from
`RR_GRID` at import time (train/encode.py:26-27). Everything else (encoder Linear,
the weight-tied looped block, the 5-cell readout head) is already size-agnostic.

This fork removes both locks:

  * no `self.pos` Parameter and no grid/robots hyper-parameters -- the position
    signal is computed from `n` at call time (`pe="sin2d"`), carried in the input
    channels (`pe="coord"`, in_channels=13), or omitted (`pe="none"`);
  * nothing is imported from `train.*`, so importing this module never freezes a
    grid size. `LoopedLayer` is copied in verbatim instead.

One model instance therefore runs on any board side `n`; batches must still be
size-homogeneous because the attention masks are dense.

MEMORY / CONSTRAINTS (unchanged from the reference): the two structure matrices are
dense `[R_total, n*n+1, n*n+1]` float tensors and the attention scores are
`[R_total, heads, n*n+1, n*n+1]`, so a step costs O(R_total * (n^2+1)^2) -- 16x16 is
cheap, 32x32 is ~16x that per record. Keep `batch_size * max_per_group` small at the
large rungs, exactly as the reference trainer does.

Everything downstream of the encoder matches the reference so any measured delta is
the position signal: 9-channel node features, gathered 5-cell readout, HL-Gauss
classification value, ranking loss, and the `val_top1_optimal` / `val_regret` /
`val_mae` metric names.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# HL-Gauss smoothing width and the classification-loss weight are frozen at the
# reference net's defaults (train/looped_pc.py:138-139, `sigma=1.0`,
# `class_weight=1.0`); this fork's constructor signature is fixed by the chunk
# spec and exposes no knob for them.
SIGMA = 1.0
CLASS_WEIGHT = 1.0


# --- copied from train/looped_pc.py:106-134; kept self-contained to avoid
# import-time GRID freezing (importing train.looped_pc pulls in train.encode,
# which reads RR_GRID at module import and pins one grid size per process).
class LoopedLayer(nn.Module):
    """One weight-tied block: global + all-edge + indep-edge attention heads."""
    def __init__(self, d, heads=4, use_global=True):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.fams = ("g", "a", "i") if use_global else ("a", "i")   # global head over-smooths
        self.qkv = nn.ModuleDict({k: nn.Linear(d, 3 * d) for k in self.fams})
        self.proj = nn.Linear(len(self.fams) * d, d)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def _head(self, X, which, A):
        B, K, d = X.shape
        q, k, v = self.qkv[which](X).chunk(3, -1)
        q = q.view(B, K, self.h, self.dk).transpose(1, 2)      # [B,h,K,dk]
        k = k.view(B, K, self.h, self.dk).transpose(1, 2)
        v = v.view(B, K, self.h, self.dk).transpose(1, 2)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dk)        # [B,h,K,K]
        if A is not None:                                       # EDGE-MASKED attention:
            s = s.masked_fill((A == 0)[:, None], float("-inf"))  # attend only to neighbors
        o = (F.softmax(s, -1) @ v).transpose(1, 2).reshape(B, K, d)
        return o

    def forward(self, X, A_all, A_ind):
        amap = {"g": None, "a": A_all, "i": A_ind}
        a = torch.cat([self._head(X, f, amap[f]) for f in self.fams], -1)
        X = self.norm1(X + self.proj(a))
        X = self.norm2(X + self.mlp(X))
        return X
# --- end verbatim copy ---


PE_MODES = ("sin2d", "coord", "none")


class SizeFreeValueNet(pl.LightningModule):
    """LoopedValueNet without the grid lock: no `pos` Parameter, no grid hparams.

    `pe` selects the position signal:
      * "sin2d" -- 2-D sinusoidal absolute PE built from (x, y) at call time by
        `nn_labeler.encode.sin2d_pe(n, d_model)` (imported lazily inside
        `_sin2d_pe`, cached per (n, device, dtype)). Injectable for tests: set
        `model.pe_fn = fn` where `fn(n, d_model) -> [(n*n)+1, d_model]`.
      * "coord" -- no additive PE; position rides in the node features, which are
        then 13-channel (`in_channels=13`, `coord_channels=True` in the collate).
      * "none" -- no position signal at all (the ablation control: the slide-graph
        already enters through the attention masks).

    Nothing else about the net depends on `n`, so one instance runs at any board
    size; see the module docstring for the O(R*(n^2+1)^2) memory constraint.
    """

    def __init__(self, d_model=192, recurrence=12, heads=4, num_classes=96,
                 lr=3e-4, weight_decay=1e-4, pe="sin2d", in_channels=9):
        super().__init__()
        if pe not in PE_MODES:
            raise ValueError(f"pe must be one of {PE_MODES}, got {pe!r}")
        # No grid/robots hparams on purpose: a checkpoint of this net is valid at
        # every board size, so there is nothing size-shaped to rebuild.
        self.save_hyperparameters()
        self.enc = nn.Linear(in_channels, d_model)
        self.layer = LoopedLayer(d_model, heads, use_global=True)
        self.head = nn.Sequential(nn.Linear(5 * d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, num_classes))
        self.register_buffer("bins", torch.arange(num_classes, dtype=torch.float32))
        self._val = []
        # plain dict, NOT a buffer: caching the PE as a buffer would put a
        # size-shaped tensor back into the checkpoint -- the very lock we removed.
        self._pe_cache = {}
        self.pe_fn = None          # test/ablation injection point; see class doc

    # -- position signal -------------------------------------------------------

    def _sin2d_pe(self, n, device, dtype):
        key = (int(n), device, dtype)
        t = self._pe_cache.get(key)
        if t is None:
            fn = self.pe_fn
            if fn is None:
                # lazy import: model.py must stay importable (and smoke-testable)
                # without nn_labeler/encode.py present.
                from nn_labeler import encode as _encode
                fn = _encode.sin2d_pe
            arr = np.asarray(fn(int(n), self.hparams.d_model))
            want = (int(n) * int(n) + 1, self.hparams.d_model)
            if arr.shape != want:
                raise ValueError(f"sin2d_pe(n={n}) returned {arr.shape}, want {want}")
            t = torch.as_tensor(arr, dtype=dtype, device=device)
            self._pe_cache[key] = t
        return t

    # -- forward ---------------------------------------------------------------

    def _encode(self, x, A_all, A_ind, n):
        n = int(n)
        if x.shape[1] != n * n + 1:
            raise ValueError(f"x has {x.shape[1]} tokens, want n*n+1={n * n + 1} for n={n}")
        if x.shape[-1] != self.hparams.in_channels:
            raise ValueError(f"x has {x.shape[-1]} channels, net built for "
                             f"in_channels={self.hparams.in_channels}")
        X = self.enc(x)
        if self.hparams.pe == "sin2d":
            X = X + self._sin2d_pe(n, X.device, X.dtype)
        for _ in range(self.hparams.recurrence):
            X = self.layer(X, A_all, A_ind)
        return X

    def forward(self, x, A_all, A_ind, n, key):
        """[R,N+1,C] x, [R,N+1,N+1] masks, board side `n`, [R,5] gather index
        -> [R,num_classes] logits.

        Deviation from the reference (looped_pc.py:160-164), which took the whole
        batch dict: the position signal needs `n` explicitly, so the tensors are
        passed positionally; `key` (the 5-cell readout gather) is appended last.
        """
        X = self._encode(x, A_all, A_ind, n)                    # [R,N+1,d]
        idx = torch.arange(X.shape[0], device=X.device)[:, None]
        gathered = X[idx, key].reshape(X.shape[0], -1)          # [R,5d]
        return self.head(gathered)                              # [R,num_classes]

    def _logits(self, b):
        return self(b["x"], b["A_all"], b["A_ind"], b["n"], b["key"])

    # -- value / losses (ported verbatim from looped_pc.py:166-183) -------------

    def _value(self, logits):
        return (F.softmax(logits, -1) * self.bins).sum(-1)

    def _hl_gauss(self, ctg):
        dd = (self.bins[None, :] - ctg.clamp(0, self.hparams.num_classes - 1)[:, None]) \
            / SIGMA
        w = torch.exp(-0.5 * dd * dd)
        return w / w.sum(-1, keepdim=True)

    def _rank(self, cost, group, opt):
        ls = []
        for g in group.unique():
            m = group == g
            o = opt[m].float()
            if o.sum() == 0:
                continue
            ls.append(-(o / o.sum() * F.log_softmax(-cost[m], 0)).sum())
        return torch.stack(ls).mean() if ls else cost.sum() * 0

    # -- steps -----------------------------------------------------------------

    def training_step(self, b, _):
        logits = self._logits(b)
        val = self._value(logits)
        loss = self._rank(val, b["group"], b["opt"])
        loss = loss + CLASS_WEIGHT * \
            -(self._hl_gauss(b["ctg"]) * F.log_softmax(logits, -1)).sum(-1).mean()
        bs = int(b["group"].max()) + 1
        self.log("train_loss", loss, prog_bar=True, batch_size=bs)
        if b.get("clamped"):
            # cost_to_go above num_classes-1 is silently squashed; count it so a
            # too-small bin range shows up in the log instead of as quiet bias.
            self.log("train_clamped", float(b["clamped"]), on_step=False,
                     on_epoch=True, reduce_fx="sum", batch_size=bs)
        return loss

    def validation_step(self, b, batch_idx):
        val = self._value(self._logits(b))
        cfgs = b.get("configs") or []
        for i in range(val.shape[0]):
            g = int(b["group"][i])
            # group ids are batch-LOCAL here (the sampler hands the collate groups,
            # not dataset indices, unlike DenseDataset.__getitem__ at
            # looped_pc.py:90), so batch_idx namespaces them across the epoch.
            self._val.append((batch_idx, g, bool(b["opt"][i]), float(val[i]),
                              float(b["ctg"][i]), cfgs[g] if g < len(cfgs) else "?"))

    def on_validation_epoch_end(self):
        if not self._val:
            return
        gr = {}
        for bi, g, o, p, c, cfg in self._val:
            gr.setdefault((bi, g), []).append((p, o, c, cfg))
        top1 = sum(min(v)[1] for v in gr.values()) / len(gr)
        regret = sum(min(v)[2] - min(c for _, _, c, _ in v) for v in gr.values()) / len(gr)
        mae = sum(abs(p - c) for _, _, _, p, c, _ in self._val) / len(self._val)
        self.log_dict({"val_top1_optimal": top1, "val_regret": regret, "val_mae": mae},
                      prog_bar=True)
        # mixed-corpus training is the point of this fork: whole-pool regret hides
        # a rung that is being carried by the others.
        per_cfg = {}
        for v in gr.values():
            per_cfg.setdefault(v[0][3], []).append(
                min(v)[2] - min(c for _, _, c, _ in v))
        self.log_dict({f"val_regret/{k}": sum(x) / len(x) for k, x in per_cfg.items()})
        self._val.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay)


# -- batching -----------------------------------------------------------------

def collate_groups(batch_groups, featurize_fn, adjacency_fn, key_fn,
                   coord_channels=False, num_classes=96):
    """Flatten `batch_size` decision GROUPS into one size-homogeneous batch.

    Batching semantics match the reference (looped_pc.py:72-103): a batch is
    `batch_size` groups, each contributing <= `max_per_group` records (the
    truncation happens in `dataset.GroupDataset`), and every record becomes one row
    of every tensor. Memory is O(R_total * (n^2+1)^2) from the dense masks -- the
    same wall the reference has.

    All record->tensor functions are dependency-INJECTED so this stays testable
    without the encoder module (and so ablations can swap featurizers):
      featurize_fn(rec, n, coord_channels=bool) -> [(n*n)+1, C]
      adjacency_fn(env_dir, env_id, n)          -> (A_all, A_ind), each [(n*n)+1]^2
      key_fn(rec, n)                            -> 5 cell indices
    Deviation from the spec'd 4-arg form: `key_fn` is a parameter too (the key
    indices come from the encoder module, so they need the same injection point);
    callers pass all three by keyword via functools.partial.

    Records must carry `_n`, `_config`, `_env_dir` (stamped by
    `dataset.load_corpus`), `env_id`, `cost_to_go` and `is_optimal`.

    Returns dict(x, A_all, A_ind, key, ctg, opt, group, n, configs, clamped).
    """
    if not batch_groups:
        raise ValueError("empty batch")
    ns = {int(r["_n"]) for g in batch_groups for r in g}
    if len(ns) != 1:
        raise ValueError(f"batch mixes board sizes {sorted(ns)}; the batch sampler "
                         "must bucket by _n (dense masks cannot be padded cheaply)")
    n = ns.pop()
    hi = num_classes - 1

    xs, keys, ctg, opt, grp, configs = [], [], [], [], [], []
    adj_uniq, adj_index, adj_of_rec = {}, [], []
    clamped = 0
    for gi, g in enumerate(batch_groups):
        if not g:
            raise ValueError(f"group {gi} is empty")
        configs.append(g[0].get("_config", "?"))
        for r in g:
            f = np.asarray(featurize_fn(r, n, coord_channels=coord_channels),
                           dtype=np.float32)
            if f.shape[0] != n * n + 1:
                raise ValueError(f"featurize_fn gave {f.shape}, want [{n * n + 1}, C]")
            xs.append(torch.from_numpy(np.ascontiguousarray(f)))
            k = list(key_fn(r, n))
            if len(k) != 5:
                raise ValueError(f"key_fn gave {len(k)} indices, want 5")
            keys.append([int(i) for i in k])
            c = int(round(float(r["cost_to_go"])))
            cc = max(0, min(c, hi))
            clamped += int(cc != c)
            ctg.append(float(cc))
            opt.append(bool(r["is_optimal"]))
            grp.append(gi)
            # dedupe: one pkl read / graph build per (board, size) per batch,
            # then expanded back to one row per record below.
            akey = (str(r.get("_env_dir")), int(r["env_id"]), n)
            if akey not in adj_uniq:
                A_all, A_ind = adjacency_fn(r.get("_env_dir"), r["env_id"], n)
                adj_uniq[akey] = len(adj_index)
                adj_index.append((
                    torch.as_tensor(np.asarray(A_all, dtype=np.float32)),
                    torch.as_tensor(np.asarray(A_ind, dtype=np.float32))))
            adj_of_rec.append(adj_uniq[akey])

    sel = torch.tensor(adj_of_rec, dtype=torch.long)
    A_all = torch.stack([a for a, _ in adj_index]).index_select(0, sel)
    A_ind = torch.stack([b for _, b in adj_index]).index_select(0, sel)
    return dict(
        x=torch.stack(xs),                                   # [R, N+1, C]
        A_all=A_all, A_ind=A_ind,                            # [R, N+1, N+1]
        key=torch.tensor(keys, dtype=torch.long),            # [R, 5]
        ctg=torch.tensor(ctg, dtype=torch.float32),          # [R]
        opt=torch.tensor(opt, dtype=torch.bool),             # [R]
        group=torch.tensor(grp, dtype=torch.long),           # [R] -> group 0..B-1
        n=int(n), configs=configs, clamped=int(clamped))
