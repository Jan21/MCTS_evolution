"""Proposal net on the looped-transformer encoder (A + hard mask, the 0.356 winner).

Autoregressive structure-masked subgoal generation (bottleneck -> support -> helper,
cost_to_go-weighted soft targets, regret@k eval) on the edge-MASKED looped transformer
encoder (global head + two neighbor-masked graph heads on A_all/A_ind, weight-tied
loop) -- the same encoder as the value net. Per-decision encoding (board+segment,
candidate NOT marked -- it's generated); decode is a 3-step pointer head.

    python -m train.policy_tf --data nn/data/combined.jsonl --epochs 50
"""
from __future__ import annotations

import argparse
import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from train.encode import GRID, _graph
from train.policy_common import _meta, _features, _ix
from nn.benchmark import load, by_split, group_by_decision

N = GRID * GRID


@lru_cache(maxsize=4096)
def _adj(env_id):
    """Binary A_all / A_ind [257,257] + self-loops, used as attention MASK."""
    ei, et = _graph(env_id)
    ei = ei.numpy(); et = et.view(-1).numpy()
    A_all = np.zeros((N + 1, N + 1), np.float32)
    A_ind = np.zeros((N + 1, N + 1), np.float32)
    A_all[ei[1], ei[0]] = 1.0
    ind = et == 0
    A_ind[ei[1][ind], ei[0][ind]] = 1.0
    for i in range(N + 1):
        A_all[i, i] = 1.0
        A_ind[i, i] = 1.0
    return torch.from_numpy(A_all), torch.from_numpy(A_ind)


def _x257(r):
    return torch.cat([_features(r), torch.zeros(1, 7)], 0)   # [257,7], row 256 = global


class PolicyTFDataset(Dataset):
    def __init__(self, groups):
        self.metas = [m for m in (_meta(g) for g in groups) if m]

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, i):
        m = self.metas[i]
        A_all, A_ind = _adj(m["env_id"])
        return _x257(m["rec"]), A_all, A_ind, m


def collate(items):
    return (torch.stack([it[0] for it in items]),
            torch.stack([it[1] for it in items]),
            torch.stack([it[2] for it in items]),
            [it[3] for it in items])


def _mlp(din, d):
    return nn.Sequential(nn.Linear(din, d), nn.GELU(), nn.Linear(d, d))


class MaskedLayer(nn.Module):
    """global head + edge-masked A_all / A_ind heads (the 0.356 value-net encoder)."""
    def __init__(self, d, heads=4):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.ModuleDict({k: nn.Linear(d, 3 * d) for k in ("g", "a", "i")})
        self.proj = nn.Linear(3 * d, d)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def _head(self, X, which, A):
        B, K, d = X.shape
        q, k, v = self.qkv[which](X).chunk(3, -1)
        q = q.view(B, K, self.h, self.dk).transpose(1, 2)
        k = k.view(B, K, self.h, self.dk).transpose(1, 2)
        v = v.view(B, K, self.h, self.dk).transpose(1, 2)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
        if A is not None:
            s = s.masked_fill((A == 0)[:, None], float("-inf"))
        o = (F.softmax(s, -1) @ v).transpose(1, 2).reshape(B, K, d)
        return o

    def forward(self, X, A_all, A_ind):
        a = torch.cat([self._head(X, "g", None),
                       self._head(X, "a", A_all),
                       self._head(X, "i", A_ind)], -1)
        X = self.norm1(X + self.proj(a))
        X = self.norm2(X + self.mlp(X))
        return X


class PolicyTF(pl.LightningModule):
    def __init__(self, d_model=192, recurrence=12, heads=4, temp=1.0, lr=3e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.enc = nn.Linear(7, d_model)
        self.pos = nn.Parameter(torch.randn(N + 1, d_model) * 0.02)
        self.layer = MaskedLayer(d_model, heads)
        self.seg = _mlp(3 * d_model, d_model)
        self.q_bn = _mlp(d_model, d_model)
        self.q_sup = _mlp(2 * d_model, d_model)
        self.q_help = _mlp(3 * d_model, d_model)
        self._val = []

    def _encode(self, x, A_all, A_ind):
        X = self.enc(x) + self.pos
        for _ in range(self.hparams.recurrence):
            X = self.layer(X, A_all, A_ind)
        return X                                            # [B,257,d]

    def _heads(self, hi, m, bn_cond, sup_cond):
        g = self.seg(torch.cat([hi[m["seg_start"]], hi[m["seg_end"]], hi.mean(0)]))
        vbn = torch.tensor(m["valid_bn"], device=hi.device)
        bn_logits = hi[vbn] @ self.q_bn(g)
        vsup = torch.tensor(m["sup_by_bn"][bn_cond], device=hi.device)
        sup_logits = hi[vsup] @ self.q_sup(torch.cat([g, hi[bn_cond]]))
        hc = torch.tensor(m["helper_cells"], device=hi.device)
        help_logits = hi[hc] @ self.q_help(torch.cat([g, hi[bn_cond], hi[sup_cond]]))
        return bn_logits, sup_logits, help_logits

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

    def training_step(self, batch, _):
        x, A_all, A_ind, metas = batch
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
        x, A_all, A_ind, metas = batch
        h = self._encode(x, A_all, A_ind)
        for i, m in enumerate(metas):
            hi = h[i]
            g = self.seg(torch.cat([hi[m["seg_start"]], hi[m["seg_end"]], hi.mean(0)]))
            vbn = m["valid_bn"]
            bn_lp = F.log_softmax(hi[torch.tensor(vbn, device=h.device)] @ self.q_bn(g), 0)
            sup_lp, help_lp = {}, {}
            for bn in vbn:
                sl = m["sup_by_bn"][bn]
                sup_lp[bn] = F.log_softmax(
                    hi[torch.tensor(sl, device=h.device)] @ self.q_sup(torch.cat([g, hi[bn]])), 0)
                for sp in sl:
                    help_lp[(bn, sp)] = F.log_softmax(
                        hi[torch.tensor(m["helper_cells"], device=h.device)]
                        @ self.q_help(torch.cat([g, hi[bn], hi[sp]])), 0)
            scored = []
            for (bn, sp, hi_), ctg in m["ctg_map"].items():
                lp = (bn_lp[vbn.index(bn)] + sup_lp[bn][m["sup_by_bn"][bn].index(sp)]
                      + help_lp[(bn, sp)][hi_])
                scored.append((float(lp), ctg))
            scored.sort(reverse=True)
            self._val.append({k: min(c for _, c in scored[:k]) - m["best"] for k in (1, 3, 5)})

    def on_validation_epoch_end(self):
        if not self._val:
            return
        n = len(self._val)
        out = {f"val_regret@{k}": sum(v[k] for v in self._val) / n for k in (1, 3, 5)}
        out["val_recall@1"] = sum(1 for v in self._val if v[1] == 0) / n
        out["val_recall@5"] = sum(1 for v in self._val if v[5] == 0) / n
        out["val_regret"] = out["val_regret@1"]
        self.log_dict(out, prog_bar=True)
        self._val.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--recurrence", type=int, default=12)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    a = p.parse_args()

    recs = load(a.data)
    tr = PolicyTFDataset(group_by_decision(by_split(recs, "train")))
    va = PolicyTFDataset(group_by_decision(by_split(recs, "val")))
    print(f"train decisions={len(tr)} val decisions={len(va)}", flush=True)
    dl = dict(batch_size=a.batch_size, collate_fn=collate, num_workers=a.num_workers)
    model = PolicyTF(d_model=a.d_model, recurrence=a.recurrence, heads=a.heads, temp=a.temp)
    ckpt = pl.callbacks.ModelCheckpoint(monitor="val_regret", mode="min")
    pl.Trainer(max_epochs=a.epochs, accelerator="auto", devices=1,
               callbacks=[ckpt], log_every_n_steps=50).fit(
        model, DataLoader(tr, shuffle=True, **dl), DataLoader(va, **dl))


if __name__ == "__main__":
    main()
