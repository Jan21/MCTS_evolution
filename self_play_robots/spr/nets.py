"""Size-free policy net for the subgoal action space (+ the value net, imported).

PROBLEM.md section 6.2 (owner recommendation): rebuild policy+value on the
labeler's pe=none recipe. The value side already exists and is imported in
place -- `nn_labeler.model.SizeFreeValueNet` (FINDINGS 53/61: pe=none, 9
channels, HL-Gauss + ranking loss, valid at any board side). This module adds
the missing half:

  SizeFreePolicyNet  == train/policy_tf.py::PolicyTF minus the one grid-locked
                        tensor (`self.pos`, policy_tf.py:116). Same 7-channel
                        decision featurization (train/policy_common.py::_features,
                        ported here with the grid side `n` as an argument), the
                        same weight-tied edge-masked encoder (LoopedLayer, shared
                        with the value net), the same 3-step autoregressive
                        pointer decode bottleneck -> support -> helper
                        (policy_tf.py:130-138), the same cost-weighted soft
                        targets and regret@k validation.

Attribute/method names are kept IDENTICAL to PolicyTF (`_encode`, `seg`,
`q_bn`, `q_sup`, `q_help`) so `eval/end2end.py::_policy_logp` -- and therefore
the arena's `eval/compare.py::_nn_astar_backward` -- drives this net unchanged
when the process is pinned to one config (see spr/bench.py). Nothing here
imports `train.*` (import-time RR_GRID freeze); the featurizers take `n`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from nn_labeler.model import LoopedLayer, SizeFreeValueNet  # noqa: F401 (re-export)

POLICY_CHANNELS = 7


# ---------------------------------------------------------------------------
# featurization (train/policy_common.py::_features / _meta, size-parametric)
# ---------------------------------------------------------------------------

def flat(p, n):
    return p[1] * n + p[0]


def policy_features(r: dict, n: int) -> np.ndarray:
    """[n*n+1, 7] float32 -- 7 decision channels + the zero global row.

    Channels (policy_common.py:62-75 verbatim): 0 mover now (seg_start),
    1 goal (seg_end), 2 pinned support (seg_support), 3 every robot (target +
    helpers, shared), 4 ctx_open_endpoints, 5 ctx_bottlenecks, 6 ctx_supports.
    The candidate is NOT marked -- the policy generates it.
    """
    f = np.zeros((n * n + 1, POLICY_CHANNELS), np.float32)
    f[flat(r["seg_start"], n), 0] = 1
    f[flat(r["seg_end"], n), 1] = 1
    if r.get("seg_support"):
        f[flat(r["seg_support"], n), 2] = 1
    f[flat(r["target_robot"][0], n), 3] = 1
    for h in r["helpers"]:
        f[flat(h[0], n), 3] = 1
    for p in r.get("ctx_open_endpoints", []):
        f[flat(p, n), 4] = 1
    for p in r.get("ctx_bottlenecks", []):
        f[flat(p, n), 5] = 1
    for p in r.get("ctx_supports", []):
        f[flat(p, n), 6] = 1
    return f


def policy_meta(group: list[dict], n: int, byref: bool = False):
    """Per-decision structure: valid bottleneck/support sets, the optimal
    target, the (bn, sup, helper_slot) -> cost_to_go map. Port of
    `train/policy_common.py::_meta` with `n` explicit. Returns None when no
    candidate's helper resolves to a slot (byref off: helper must stand at a
    robot start cell; byref on: fall back to robot colour)."""
    g0 = group[0]
    helper_cells = [flat(h[0], n) for h in g0["helpers"]]
    hidx, cidx = {}, {}
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
        cands.append((flat(r["cand_bottleneck"], n), flat(r["cand_support"], n),
                      hi, int(r.get("cost_to_go", 0)), bool(r.get("is_optimal", False))))
    if not cands:
        return None
    valid_bn = sorted({c[0] for c in cands})
    sup_by_bn: dict[int, set] = {}
    for bn, sp, _hi, _ctg, _opt in cands:
        sup_by_bn.setdefault(bn, set()).add(sp)
    sup_by_bn = {k: sorted(v) for k, v in sup_by_bn.items()}
    best = min(c[3] for c in cands)
    opts = [c for c in cands if c[4]] or [c for c in cands if c[3] == best]
    o = min(opts, key=lambda c: c[3])
    return dict(rec=g0, env_id=g0["env_id"], n=n,
                seg_start=flat(g0["seg_start"], n), seg_end=flat(g0["seg_end"], n),
                helper_cells=helper_cells, valid_bn=valid_bn, sup_by_bn=sup_by_bn,
                tgt_bn=o[0], tgt_sup=o[1], tgt_helper=o[2],
                cands=[(c[0], c[1], c[2], c[3]) for c in cands],
                ctg_map={(c[0], c[1], c[2]): c[3] for c in cands}, best=best)


def _mlp(din, d):
    return nn.Sequential(nn.Linear(din, d), nn.GELU(), nn.Linear(d, d))


def heads_logp_map(net, hi, m):
    """{(bn, sup, helper_slot): AR log-prob} from any net exposing PolicyTF's
    heads (`seg`, `q_bn`, `q_sup`, `q_help`) -- SizeFreePolicyNet or the
    per-size PolicyTF; identical arithmetic to eval/end2end._policy_logp."""
    dev = hi.device
    g = net.seg(torch.cat([hi[m["seg_start"]], hi[m["seg_end"]], hi.mean(0)]))
    vbn = m["valid_bn"]
    bn_lp = F.log_softmax(hi[torch.tensor(vbn, device=dev)] @ net.q_bn(g), 0)
    out = {}
    for bi, bn in enumerate(vbn):
        sl = m["sup_by_bn"][bn]
        sup_lp = F.log_softmax(hi[torch.tensor(sl, device=dev)]
                               @ net.q_sup(torch.cat([g, hi[bn]])), 0)
        for si, sp in enumerate(sl):
            hl = F.log_softmax(hi[torch.tensor(m["helper_cells"], device=dev)]
                               @ net.q_help(torch.cat([g, hi[bn], hi[sp]])), 0)
            for (b2, s2, hi2) in m["ctg_map"]:
                if b2 == bn and s2 == sp:
                    out[(bn, sp, hi2)] = float(bn_lp[bi] + sup_lp[si] + hl[hi2])
    return out


# ---------------------------------------------------------------------------
# the net
# ---------------------------------------------------------------------------

class SizeFreePolicyNet(pl.LightningModule):
    """PolicyTF without `self.pos`; `pe` in {"none", "sin2d"} (none = the
    labeler's production choice, FINDINGS 53)."""

    def __init__(self, d_model=192, recurrence=12, heads=4, temp=1.0, lr=3e-4,
                 weight_decay=1e-4, pe="none"):
        super().__init__()
        self.save_hyperparameters()
        self.enc = nn.Linear(POLICY_CHANNELS, d_model)
        self.layer = LoopedLayer(d_model, heads, use_global=True)
        self.seg = _mlp(3 * d_model, d_model)
        self.q_bn = _mlp(d_model, d_model)
        self.q_sup = _mlp(2 * d_model, d_model)
        self.q_help = _mlp(3 * d_model, d_model)
        self._val = []
        self._pe_cache = {}

    # -- encoder (signature identical to PolicyTF._encode) ---------------------
    def _encode(self, x, A_all, A_ind):
        X = self.enc(x)
        if self.hparams.pe == "sin2d":
            n = int(round((x.shape[1] - 1) ** 0.5))
            key = (n, X.device, X.dtype)
            pe = self._pe_cache.get(key)
            if pe is None:
                from nn_labeler import encode as _enc
                pe = torch.as_tensor(_enc.sin2d_pe(n, self.hparams.d_model),
                                     dtype=X.dtype, device=X.device)
                self._pe_cache[key] = pe
            X = X + pe
        for _ in range(self.hparams.recurrence):
            X = self.layer(X, A_all, A_ind)
        return X                                             # [B, T, d]

    # -- heads ------------------------------------------------------------------
    def _heads(self, hi, m, bn_cond, sup_cond):
        g = self.seg(torch.cat([hi[m["seg_start"]], hi[m["seg_end"]], hi.mean(0)]))
        vbn = torch.tensor(m["valid_bn"], device=hi.device)
        bn_logits = hi[vbn] @ self.q_bn(g)
        vsup = torch.tensor(m["sup_by_bn"][bn_cond], device=hi.device)
        sup_logits = hi[vsup] @ self.q_sup(torch.cat([g, hi[bn_cond]]))
        hc = torch.tensor(m["helper_cells"], device=hi.device)
        help_logits = hi[hc] @ self.q_help(torch.cat([g, hi[bn_cond], hi[sup_cond]]))
        return bn_logits, sup_logits, help_logits

    def logp_map(self, hi, m):
        """{(bn, sup, helper_slot): AR log-prob} over the decision's candidates
        (the inference path; identical arithmetic to eval/end2end._policy_logp)."""
        return heads_logp_map(self, hi, m)

    def _soft_targets(self, m, dev):
        cands = m["cands"]
        w = F.softmax(-torch.tensor([c[3] for c in cands], dtype=torch.float, device=dev)
                      / self.hparams.temp, 0)
        vbn, tbn, tsp = m["valid_bn"], m["tgt_bn"], m["tgt_sup"]
        sup_list = m["sup_by_bn"][tbn]
        bn_t = torch.zeros(len(vbn), device=dev)
        sup_t = torch.zeros(len(sup_list), device=dev)
        help_t = torch.zeros(len(m["helper_cells"]), device=dev)
        for (bn, sp, hi_, _), wc in zip(cands, w):
            bn_t[vbn.index(bn)] += wc
            if bn == tbn:
                sup_t[sup_list.index(sp)] += wc
                if sp == tsp:
                    help_t[hi_] += wc
        return bn_t, sup_t / sup_t.sum(), help_t / help_t.sum()

    # -- steps ------------------------------------------------------------------
    def training_step(self, batch, _):
        x, A_all, A_ind, metas = batch["x"], batch["A_all"], batch["A_ind"], batch["metas"]
        h = self._encode(x, A_all, A_ind)
        loss = 0.0
        for i, m in enumerate(metas):
            hi = h[i]
            bnl, supl, hl = self._heads(hi, m, m["tgt_bn"], m["tgt_sup"])
            bn_t, sup_t, help_t = self._soft_targets(m, h.device)
            loss = loss - (bn_t * F.log_softmax(bnl, 0)).sum()
            loss = loss - (sup_t * F.log_softmax(supl, 0)).sum()
            loss = loss - (help_t * F.log_softmax(hl, 0)).sum()
        loss = loss / max(len(metas), 1)
        self.log("train_loss", loss, prog_bar=True, batch_size=len(metas))
        return loss

    def validation_step(self, batch, _):
        x, A_all, A_ind, metas = batch["x"], batch["A_all"], batch["A_ind"], batch["metas"]
        h = self._encode(x, A_all, A_ind)
        for i, m in enumerate(metas):
            lp = self.logp_map(h[i], m)
            scored = sorted(((lp[k], c) for k, c in m["ctg_map"].items()), reverse=True)
            self._val.append(({k: min(c for _, c in scored[:k]) - m["best"]
                               for k in (1, 3, 5)}, m.get("_config", "?")))

    def on_validation_epoch_end(self):
        if not self._val:
            return
        n = len(self._val)
        out = {f"val_regret@{k}": sum(v[k] for v, _ in self._val) / n for k in (1, 3, 5)}
        out["val_recall@1"] = sum(1 for v, _ in self._val if v[1] == 0) / n
        out["val_recall@5"] = sum(1 for v, _ in self._val if v[5] == 0) / n
        out["val_regret"] = out["val_regret@1"]
        per = {}
        for v, cfg in self._val:
            per.setdefault(cfg, []).append(v[1])
        for cfg, xs in per.items():
            out[f"val_regret/{cfg}"] = sum(xs) / len(xs)
        self.log_dict(out, prog_bar=True)
        self._val.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay)


# ---------------------------------------------------------------------------
# batching for training (size-bucketed; one board size per batch)
# ---------------------------------------------------------------------------

class PolicyGroupDataset(torch.utils.data.Dataset):
    """One item = one decision meta (records must be stamped with _n/_config/
    _env_dir by nn_labeler.dataset.load_corpus). Exposes `.sizes` for
    nn_labeler.dataset.SizeBucketBatchSampler."""

    def __init__(self, groups, byref=False):
        self.metas = []
        for g in groups:
            m = policy_meta(g, int(g[0]["_n"]), byref=byref)
            if m is None:
                continue
            m["_config"] = g[0].get("_config", "?")
            m["_env_dir"] = g[0].get("_env_dir")
            self.metas.append(m)
        self.sizes = [m["n"] for m in self.metas]
        self.groups = [[m["rec"]] for m in self.metas]   # sampler fallback

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, i):
        return self.metas[i]


def collate_policy(metas):
    from nn_labeler import encode
    ns = {m["n"] for m in metas}
    if len(ns) != 1:
        raise ValueError(f"batch mixes sizes {sorted(ns)}")
    n = ns.pop()
    xs, aa, ai = [], [], []
    cache = {}
    for m in metas:
        xs.append(torch.from_numpy(policy_features(m["rec"], n)))
        key = (m["_env_dir"], m["env_id"])
        if key not in cache:
            A_all, A_ind = encode.adjacency(m["_env_dir"], m["env_id"], n)
            cache[key] = (torch.as_tensor(np.asarray(A_all, np.float32)),
                          torch.as_tensor(np.asarray(A_ind, np.float32)))
        aa.append(cache[key][0])
        ai.append(cache[key][1])
    return dict(x=torch.stack(xs), A_all=torch.stack(aa), A_ind=torch.stack(ai),
                metas=metas, n=n)


# ---------------------------------------------------------------------------
# adapters so eval/end2end.py's `_value_cost` drives the size-free value net
# ---------------------------------------------------------------------------

class ValueAdapter:
    """`_value_cost(value, recs, dev)` calls `value(b)` with a dict
    b = {x, A_all, A_ind, key} and then `value._value(logits)`; wrap the
    positional-signature SizeFreeValueNet accordingly (n from x's token count)."""

    def __init__(self, net: SizeFreeValueNet):
        self.net = net

    def __call__(self, b):
        n = int(round((b["x"].shape[1] - 1) ** 0.5))
        return self.net(b["x"], b["A_all"], b["A_ind"], n, b["key"])

    def _value(self, logits):
        return self.net._value(logits)

    def to(self, dev):
        self.net.to(dev)
        return self

    def eval(self):
        self.net.eval()
        return self


# ---------------------------------------------------------------------------
# per-size family (train/policy_tf.py PolicyTF, train/looped_pc.py LoopedValueNet)
# behind the same two call signatures the spr searches use. Importing train.*
# freezes RR_GRID for the process -- the caller pins one config (as spr.bench does).
# ---------------------------------------------------------------------------

class PerSizePolicy:
    """PolicyTF with `logp_map`; `_encode`/`seg`/`q_*` are the net's own."""

    def __init__(self, net):
        self.net = net
        self.seg, self.q_bn, self.q_sup, self.q_help = net.seg, net.q_bn, net.q_sup, net.q_help

    def _encode(self, x, A_all, A_ind):
        return self.net._encode(x, A_all, A_ind)

    def logp_map(self, hi, m):
        return heads_logp_map(self.net, hi, m)

    def to(self, dev):
        self.net.to(dev)
        return self

    def eval(self):
        self.net.eval()
        return self


class PerSizeValue:
    """LoopedValueNet behind SizeFreeValueNet's positional signature."""

    def __init__(self, net):
        self.net = net

    def __call__(self, x, A_all, A_ind, n, key):
        return self.net(dict(x=x, A_all=A_all, A_ind=A_ind, key=key))

    def _value(self, logits):
        return self.net._value(logits)

    def to(self, dev):
        self.net.to(dev)
        return self

    def eval(self):
        self.net.eval()
        return self


def load_policy(path, device="cpu", arch="sizefree"):
    if arch == "persize":
        from train.policy_tf import PolicyTF
        return PerSizePolicy(PolicyTF.load_from_checkpoint(path, map_location=device).to(device).eval())
    return SizeFreePolicyNet.load_from_checkpoint(path, map_location=device).to(device).eval()


def load_value(path, device="cpu", arch="sizefree"):
    if arch == "persize":
        from train.looped_pc import LoopedValueNet
        return PerSizeValue(LoopedValueNet.load_from_checkpoint(path, map_location=device).to(device).eval())
    return SizeFreeValueNet.load_from_checkpoint(path, map_location=device).to(device).eval()
