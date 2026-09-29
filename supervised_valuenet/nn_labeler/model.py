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
import os as _os
try:
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel, SDPBackend as _SDPB
    _USE_SDPA = _os.environ.get("RR_SDPA", "1") != "0"
    _USE_AXIAL = _os.environ.get("RR_AXIAL", "0") == "1"   # exact row+column attention for the masked families
    _AXIAL_MIN_N = int(_os.environ.get("RR_AXIAL_MIN_N", "48"))
except Exception:                       # old torch: keep the dense path
    _USE_SDPA = False
    _USE_AXIAL = False
    _AXIAL_MIN_N = 48


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
        if _USE_AXIAL and A is not None and X.is_cuda:
            n = int(round((K - 1) ** 0.5))
            if n * n + 1 == K and n >= _AXIAL_MIN_N:   # below 48 the dense fused kernel is faster
                return axial_head_fast(self.qkv[which], X, A, n, self.h, self.dk, _axial_masks_cached(A, n))
        if _USE_SDPA and X.is_cuda:
            # fused kernel: same masked softmax, no [B,h,K,K] fp32 tensor.
            # Boolean mask is 1 byte/entry and broadcasts over heads. A row
            # with no neighbours yields NaN in the dense path and is mapped
            # to zero here, which downstream code already tolerates.
            m = None if A is None else (A != 0)[:, None]
            with _sdpa_kernel([_SDPB.EFFICIENT_ATTENTION, _SDPB.FLASH_ATTENTION,
                               _SDPB.MATH]):
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
            o = torch.nan_to_num(o, nan=0.0)
            return o.transpose(1, 2).reshape(B, K, d)
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
        # collapse detector: a net on the constant-value plateau has ~zero
        # within-group prediction spread (battery 1, job 4606952 -- 7/8 runs).
        def _std(xs):
            m = sum(xs) / len(xs)
            return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
        spread = sum(_std([p for p, _, _, _ in v]) for v in gr.values()) / len(gr)
        self.log_dict({"val_top1_optimal": top1, "val_regret": regret, "val_mae": mae,
                       "val_group_spread": spread}, prog_bar=True)
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


# ---------------------------------------------------------------------------
# sparse neighbour attention (2026-09-10)
# ---------------------------------------------------------------------------

def adj_to_index(A, max_deg=None):
    """Dense 0/1 adjacency [B,K,K] -> (idx [B,K,D], mask [B,K,D]).

    D is the largest out-degree in the batch. Padded slots point at column 0 and
    are masked. Built once per batch; the per-layer attention then never touches
    a K x K tensor again."""
    import torch
    B, K, _ = A.shape
    deg = A.sum(-1).long()                      # [B,K]
    D = int(deg.max().item()) if max_deg is None else max_deg
    D = max(D, 1)
    order = torch.argsort(A, dim=-1, descending=True, stable=True)   # [B,K,K]
    idx = order[..., :D]                                             # [B,K,D]
    ar = torch.arange(D, device=A.device).view(1, 1, D)
    mask = ar < deg.unsqueeze(-1)                                    # [B,K,D]
    idx = torch.where(mask, idx, torch.zeros_like(idx))
    return idx, mask


def sparse_head(qkv_lin, X, idx, mask, heads, dk):
    """One attention family over a neighbour index instead of a dense mask."""
    import math, torch
    import torch.nn.functional as F
    B, K, d = X.shape
    D = idx.shape[-1]
    q, k, v = qkv_lin(X).chunk(3, -1)
    q = q.view(B, K, heads, dk)                                   # [B,K,h,dk]
    k = k.view(B, K, heads, dk)
    v = v.view(B, K, heads, dk)
    # advanced indexing, never a K x K intermediate of any kind
    b = torch.arange(B, device=X.device).view(B, 1, 1)
    kn = k[b, idx]                                                # [B,K,D,h,dk]
    vn = v[b, idx]
    s = (q.unsqueeze(2) * kn).sum(-1) / math.sqrt(dk)             # [B,K,D,h]
    s = s.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    w = F.softmax(s, dim=2)
    w = torch.nan_to_num(w, nan=0.0)          # a node with no neighbours
    o = (w.unsqueeze(-1) * vn).sum(2)                             # [B,K,h,dk]
    return o.reshape(B, K, heads * dk)


def axial_head(qkv_lin, X, A, n, heads, dk, masks=None):
    """Attention over the same neighbour set as the masked-dense head, computed
    as a row pass and a column pass and merged EXACTLY.

    Every edge in these boards lies within a row or a column (measured: 0 of
    140,892 off-axis at 64x64), so the two passes together visit exactly the
    dense head's neighbour set. They cannot simply be averaged, because softmax
    over a union is not a mixture of softmaxes over the parts; the two are
    merged by the standard log-sum-exp rule instead, which is exact.

    A cell lies in its own row AND its own column, so the diagonal is dropped
    from the column pass to avoid counting it twice.
    """
    import math, torch
    import torch.nn.functional as F
    B, K, d = X.shape
    N = n * n
    q, k, v = qkv_lin(X).chunk(3, -1)

    def pass_(qq, kk, vv, m):
        """-> (o, mx, z): the unnormalised output, the score max, and the
        exp-sum, so two passes can be merged without renormalising twice."""
        s = (qq.unsqueeze(-3) * kk.unsqueeze(-4)).sum(-1) / math.sqrt(dk)
        s = s.masked_fill(~m.unsqueeze(-1), float("-inf"))       # [.., L, L, h]
        mx = s.max(dim=-2, keepdim=True).values                  # [.., L, 1, h]
        e = torch.exp(s - torch.nan_to_num(mx, neginf=0.0))
        e = torch.where(torch.isfinite(s), e, torch.zeros_like(e))
        z = e.sum(dim=-2)                                        # [.., L, h]
        o = (e.unsqueeze(-1) * vv.unsqueeze(-4)).sum(-3)         # [.., L, h, dk]
        return o, mx.squeeze(-2), z

    def split(t):
        return t[:, :N].view(B, n, n, heads, dk)

    qb, kb, vb = split(q), split(k), split(v)
    if masks is None:
        masks = axial_masks(A, n)
    mrow, mcol = masks
    orow, mxr, zr = pass_(qb, kb, vb, mrow)
    qc, kc, vc = (t.transpose(1, 2) for t in (qb, kb, vb))
    ocol, mxc, zc = pass_(qc, kc, vc, mcol)
    ocol, mxc, zc = ocol.transpose(1, 2), mxc.transpose(1, 2), zc.transpose(1, 2)

    # exact merge of two softmaxes over disjoint sets
    M = torch.maximum(mxr, mxc)
    M = torch.nan_to_num(M, neginf=0.0)
    ar = torch.exp(torch.nan_to_num(mxr, neginf=-1e30) - M).unsqueeze(-1)
    ac = torch.exp(torch.nan_to_num(mxc, neginf=-1e30) - M).unsqueeze(-1)
    num = orow * ar + ocol * ac
    den = (zr * ar.squeeze(-1) + zc * ac.squeeze(-1)).clamp(min=1e-30)
    o = (num / den.unsqueeze(-1)).reshape(B, N, heads * dk)
    o = torch.nan_to_num(o, nan=0.0)

    # the global scratchpad token is genuinely all-to-all; keep it dense
    gq = q[:, N:].view(B, 1, heads, dk)
    gk = k.view(B, K, heads, dk)
    gv = v.view(B, K, heads, dk)
    gm = A[:, N:, :].bool()
    gs = (gq.unsqueeze(2) * gk.unsqueeze(1)).sum(-1) / math.sqrt(dk)
    gs = gs.masked_fill(~gm.unsqueeze(-1), float("-inf"))
    gw = torch.nan_to_num(F.softmax(gs, dim=2), nan=0.0)
    go = (gw.unsqueeze(-1) * gv.unsqueeze(1)).sum(2).reshape(B, 1, heads * dk)
    return torch.cat([o, go], 1)


def axial_masks(A, n):
    """(row_mask [B,n,x,x], col_mask [B,n,y,y]) from the dense adjacency, built
    ONCE per board rather than once per attention call.

    The column mask drops the diagonal, since a cell lies in its own row and its
    own column and must not be attended to twice.
    """
    import torch
    B = A.shape[0]
    N = n * n
    Ab = A[:, :N, :N].view(B, n, n, n, n)
    # rows: y1 == y2, gathered without a python loop over y
    ar = torch.arange(n, device=A.device)
    mrow = Ab[:, ar, :, ar, :]                     # [n, B, x1, x2]
    mrow = mrow.permute(1, 0, 2, 3).bool()         # [B, y, x1, x2]
    mcol = Ab[:, :, ar, :, ar]                     # [n, B, y1, y2]
    mcol = mcol.permute(1, 0, 2, 3).bool()         # [B, x, y1, y2]
    mcol = mcol & ~torch.eye(n, dtype=torch.bool, device=A.device)
    return mrow, mcol


# ---------------------------------------------------------------------------
# flex_attention path (2026-09-10)
# ---------------------------------------------------------------------------

_AXIAL_CACHE = {}


def _axial_masks_cached(A, n):
    """Row/column masks per board, built once. Keyed by the adjacency tensor's
    storage (the search caches A on the device per board) and kept alive
    alongside it."""
    import torch
    if torch.is_grad_enabled():
        # training: every step brings a fresh copy of A to the device; caching
        # would pin those copies (1.4 GB each at 96x96) until the card is full
        return axial_masks(A, n)
    key = (A.data_ptr(), A.shape[0], n, str(A.device))
    hit = _AXIAL_CACHE.get(key)
    if hit is None or hit[0] is not A:
        if len(_AXIAL_CACHE) > 16:
            _AXIAL_CACHE.clear()
        hit = _AXIAL_CACHE[key] = (A, axial_masks(A, n))
    return hit[1]


def axial_head_fast(qkv_lin, X, A, n, heads, dk, masks):
    """`axial_head` with the scores as batched matrix products instead of a
    broadcast product (which materialised [B,n,n,n,h,dk]). Same neighbour set,
    same exact log-sum-exp merge of the row and column passes, same dense
    treatment of the global token."""
    import math, torch
    import torch.nn.functional as F
    B, K, d = X.shape
    N = n * n
    q, k, v = qkv_lin(X).chunk(3, -1)
    scale = 1.0 / math.sqrt(dk)

    def pass_(qq, kk, vv, m):
        # qq, kk, vv: [B, L, n, h, dk] (L lines of n cells); m: [B, L, n, n] bool
        s = torch.einsum("blqhd,blkhd->blhqk", qq, kk) * scale         # [B, L, h, q, k]
        s = s.masked_fill(~m.unsqueeze(2), float("-inf"))
        mx = s.amax(dim=-1, keepdim=True)                              # [B, L, h, q, 1]
        e = torch.exp(s - torch.nan_to_num(mx, neginf=0.0))
        e = torch.where(torch.isfinite(s), e, torch.zeros_like(e))
        z = e.sum(dim=-1)                                              # [B, L, h, q]
        o = torch.einsum("blhqk,blkhd->blqhd", e, vv)                  # [B, L, q, h, dk]
        return o, mx.squeeze(-1), z

    def split(t):
        return t[:, :N].view(B, n, n, heads, dk)                       # [B, y, x, h, dk]

    qb, kb, vb = split(q), split(k), split(v)
    mrow, mcol = masks
    orow, mxr, zr = pass_(qb, kb, vb, mrow)                            # rows: lines = y
    qc, kc, vc = (t.transpose(1, 2) for t in (qb, kb, vb))             # columns: lines = x
    ocol, mxc, zc = pass_(qc, kc, vc, mcol)
    ocol = ocol.transpose(1, 2)                                        # [B, y, x, h, dk]
    mxc = mxc.permute(0, 3, 1, 2); zc = zc.permute(0, 3, 1, 2)         # [B, x, h, y] -> [B, y, x, h]
    # bring row stats to [B, y, x, h] as well
    mxr = mxr.permute(0, 1, 3, 2); zr = zr.permute(0, 1, 3, 2)         # [B, y, h, x] -> [B, y, x, h]
    M = torch.nan_to_num(torch.maximum(mxr, mxc), neginf=0.0)
    ar = torch.exp(torch.nan_to_num(mxr, neginf=-1e30) - M)
    ac = torch.exp(torch.nan_to_num(mxc, neginf=-1e30) - M)
    num = orow * ar.unsqueeze(-1) + ocol * ac.unsqueeze(-1)
    den = (zr * ar + zc * ac).clamp(min=1e-30)
    o = (num / den.unsqueeze(-1)).reshape(B, N, heads * dk)
    o = torch.nan_to_num(o, nan=0.0)
    # the global scratchpad token: dense over everything it is allowed to see
    gq = q[:, N:].view(B, 1, heads, dk)
    gk = k.view(B, K, heads, dk); gv = v.view(B, K, heads, dk)
    gm = A[:, N:, :].bool()
    gs = torch.einsum("bqhd,bkhd->bhqk", gq, gk) * scale               # [B, h, 1, K]
    gs = gs.masked_fill(~gm.unsqueeze(1), float("-inf"))
    gw = torch.nan_to_num(F.softmax(gs, dim=-1), nan=0.0)
    go = torch.einsum("bhqk,bkhd->bqhd", gw, gv).reshape(B, 1, heads * dk)
    return torch.cat([o, go], 1)


def segment_bounds(A, n):
    """From the dense adjacency, the four wall-bounded reach limits per cell.

    Returns int32 [4, n*n]: xlo, xhi, ylo, yhi, inclusive. Built once per board.
    This is the LAST time the dense adjacency is needed; everything downstream
    uses these 4n^2 numbers.
    """
    import torch
    N = n * n
    Ab = A[0, :N, :N].view(n, n, n, n)          # [y1, x1, y2, x2]
    ar = torch.arange(n, device=A.device)
    row = Ab[ar, :, ar, :]                      # [y, x1, x2]
    col = Ab[:, ar, :, ar]                      # [x, y1, y2]
    big = torch.full_like(row, n, dtype=torch.long)
    idx = ar.view(1, 1, n).expand_as(row)
    xlo = torch.where(row > 0, idx, big).amin(-1)                     # [y, x1]
    xhi = torch.where(row > 0, idx, torch.full_like(big, -1)).amax(-1)
    ylo = torch.where(col > 0, idx, big).amin(-1)                     # [x, y1]
    yhi = torch.where(col > 0, idx, torch.full_like(big, -1)).amax(-1)
    # to [cell] order, cell = y*n + x
    xlo = xlo.reshape(N); xhi = xhi.reshape(N)
    ylo = ylo.transpose(0, 1).reshape(N); yhi = yhi.transpose(0, 1).reshape(N)
    return torch.stack([xlo, xhi, ylo, yhi]).to(torch.int32)


def make_flex_mask(bounds, n, N, K, device):
    """A mask_mod closing over `bounds`; no K x K tensor is ever formed."""
    xlo, xhi, ylo, yhi = bounds[0], bounds[1], bounds[2], bounds[3]

    def mask_mod(b, h, q, kv):
        import torch
        q_board = q < N
        kv_board = kv < N
        qc = torch.clamp(q, max=N - 1)
        qx, qy = qc % n, qc // n
        kx, ky = torch.clamp(kv, max=N - 1) % n, torch.clamp(kv, max=N - 1) // n
        same_row = (ky == qy) & (kx >= xlo[qc]) & (kx <= xhi[qc])
        same_col = (kx == qx) & (ky >= ylo[qc]) & (ky <= yhi[qc])
        board = q_board & kv_board & (same_row | same_col)
        # in the edge-masked families the trailing scratchpad token is
        # ISOLATED: a self-loop only (measured: A[N,:] and A[:,N] each have
        # exactly one nonzero). The all-to-all family has A=None and never
        # comes through this mask.
        glob = (~q_board) & (~kv_board)
        return board | glob

    return mask_mod
