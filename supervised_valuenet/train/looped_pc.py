"""Looped-transformer value net -- drop-in replacement for the GNN encoder.

Per the looped-transformer graph papers (inspiration/): one weight-tied block,
applied `recurrence` times, with three attention-head families summed + residual:
  - GLOBAL head   I   * softmax(QK^T) * X W      (every cell attends to every cell)
  - all-edge head A_all * softmax(QK^T) * X W     (relaxed reachability)  ~ conv_all
  - indep head    A_ind * softmax(QK^T) * X W     (exact reachability)    ~ conv_ind
where the structure matrix LEFT-multiplies the attention output (A @ (softmax @ V)).
Row N=G*G is a GLOBAL/scratchpad token (not in the graph). Everything downstream --
9-ch node features, gathered 5-cell readout, HL-Gauss classification value, ranking
loss, regret/MAE metrics match the GNN baseline reported in docs, so any delta is the encoder.

    python -m train.looped --data .../big_aug.jsonl --epochs 50
"""
from __future__ import annotations

import argparse
import math
import random
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from train.encode import GRID, ROBOTS, _graph, _node_features
from nn.benchmark import load, by_split, group_by_decision

N = GRID * GRID            # cells (256 at the default 16x16)
G = N                      # global/scratchpad row index (row N)


@lru_cache(maxsize=4096)
def _adj(env_id):
    """Row-normalised dense A_all / A_ind in [N+1,N+1] with self-loops; row/col N
    (global token) left at zero so it's reached only by the global-attention head."""
    ei, et = _graph(env_id)
    ei = ei.numpy(); et = et.view(-1).numpy()
    A_all = np.zeros((N + 1, N + 1), np.float32)
    A_ind = np.zeros((N + 1, N + 1), np.float32)
    s, d = ei[0], ei[1]
    A_all[d, s] = 1.0                      # message j(src)->i(dst): A[i,j]
    ind = et == 0
    A_ind[d[ind], s[ind]] = 1.0
    for i in range(N + 1):                 # self-loops on every row incl. global token
        A_all[i, i] = 1.0
        A_ind[i, i] = 1.0
    return torch.from_numpy(A_all), torch.from_numpy(A_ind)   # binary; used as attn MASK


def _x257(r):
    return torch.cat([_node_features(r), torch.zeros(1, 9)], 0)   # [N+1,9], last row = global


class DenseDataset(Dataset):
    """Mirrors gnn.GraphDataset: per-decision groups, curriculum frac, max_per_group."""
    def __init__(self, groups, max_per_group=32, sample=True, frac=1.0):
        self.groups = sorted(groups, key=lambda g: min(r["cost_to_go"] for r in g))
        self.max_per_group, self.sample, self.frac = max_per_group, sample, frac

    def set_frac(self, f):
        self.frac = max(0.05, min(1.0, f))

    def __len__(self):
        return max(1, int(len(self.groups) * self.frac))

    def __getitem__(self, i):
        g = self.groups[i]
        if self.max_per_group and len(g) > self.max_per_group:
            opt = [r for r in g if r["is_optimal"]]
            rest = [r for r in g if not r["is_optimal"]]
            k = max(0, self.max_per_group - len(opt))
            rest = random.sample(rest, min(k, len(rest))) if self.sample else rest[:k]
            g = opt + rest
        A_all, A_ind = _adj(g[0]["env_id"])

        def ix(p):
            return p[1] * GRID + p[0]
        items = []
        for r in g:
            key = torch.tensor([ix(r["cand_bottleneck"]), ix(r["cand_support"]),
                                ix(r["cand_helper"][0]), ix(r["seg_start"]),
                                ix(r["seg_end"])], dtype=torch.long)
            items.append((_x257(r), A_all, A_ind, key,
                          float(r["cost_to_go"]), bool(r["is_optimal"]), i))
        return items


def collate(batch):
    flat = [it for sub in batch for it in sub]
    x = torch.stack([f[0] for f in flat])              # [M,N+1,9]
    A_all = torch.stack([f[1] for f in flat])          # [M,N+1,N+1]
    A_ind = torch.stack([f[2] for f in flat])
    key = torch.stack([f[3] for f in flat])            # [M,5]
    ctg = torch.tensor([f[4] for f in flat])
    opt = torch.tensor([f[5] for f in flat], dtype=torch.bool)
    grp = torch.tensor([f[6] for f in flat], dtype=torch.long)
    return dict(x=x, A_all=A_all, A_ind=A_ind, key=key, ctg=ctg, opt=opt, group=grp)


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


class LoopedValueNet(pl.LightningModule):
    def __init__(self, d_model=192, recurrence=12, heads=4, use_global=True, num_classes=50,
                 sigma=1.0, class_weight=1.0, lr=3e-4, weight_decay=1e-4,
                 grid=GRID, robots=ROBOTS):
        super().__init__()
        # grid/robots ride along in hparams so load_from_checkpoint rebuilds the
        # right sizes regardless of the loading process's RR_* env vars.
        self.save_hyperparameters()
        self.enc = nn.Linear(9, d_model)
        self.pos = nn.Parameter(torch.randn(grid * grid + 1, d_model) * 0.02)  # learned PE + global token
        self.layer = LoopedLayer(d_model, heads, use_global)
        self.head = nn.Sequential(nn.Linear(5 * d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, num_classes))
        self.register_buffer("bins", torch.arange(num_classes, dtype=torch.float32))
        self._val = []

    def _encode(self, b):
        X = self.enc(b["x"]) + self.pos
        for _ in range(self.hparams.recurrence):
            X = self.layer(X, b["A_all"], b["A_ind"])
        return X

    def forward(self, b):
        X = self._encode(b)                                     # [M,N+1,d]
        idx = torch.arange(X.shape[0], device=X.device)[:, None]
        gathered = X[idx, b["key"]].reshape(X.shape[0], -1)     # [M,5d]
        return self.head(gathered)                              # [M,num_classes]

    def _value(self, logits):
        return (F.softmax(logits, -1) * self.bins).sum(-1)

    def _hl_gauss(self, ctg):
        dd = (self.bins[None, :] - ctg.clamp(0, self.hparams.num_classes - 1)[:, None]) \
            / self.hparams.sigma
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

    def training_step(self, b, _):
        logits = self(b)
        val = self._value(logits)
        loss = self._rank(val, b["group"], b["opt"])
        loss = loss + self.hparams.class_weight * \
            -(self._hl_gauss(b["ctg"]) * F.log_softmax(logits, -1)).sum(-1).mean()
        self.log("train_loss", loss, prog_bar=True, batch_size=int(b["group"].max()) + 1)
        return loss

    def validation_step(self, b, _):
        val = self._value(self(b))
        for i in range(val.shape[0]):
            self._val.append((int(b["group"][i]), bool(b["opt"][i]),
                              float(val[i]), float(b["ctg"][i])))

    def on_validation_epoch_end(self):
        if not self._val:
            return
        gr = {}
        for g, o, p, c in self._val:
            gr.setdefault(g, []).append((p, o, c))
        top1 = sum(min(v)[1] for v in gr.values()) / len(gr)
        regret = sum(min(v)[2] - min(c for _, _, c in v) for v in gr.values()) / len(gr)
        mae = sum(abs(p - c) for _, _, p, c in self._val) / len(self._val)
        self.log_dict({"val_top1_optimal": top1, "val_regret": regret, "val_mae": mae},
                      prog_bar=True)
        self._val.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay)


class Curriculum(pl.Callback):
    def __init__(self, warmup, ds):
        self.warmup, self.ds = warmup, ds

    def on_train_epoch_start(self, trainer, _):
        f = 1.0 if self.warmup <= 0 else min(1.0, 0.3 + 0.7 * trainer.current_epoch / self.warmup)
        self.ds.set_frac(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--recurrence", type=int, default=12)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-per-group", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--no-global", action="store_true", help="drop the global (over-smoothing) head")
    a = p.parse_args()

    recs = load(a.data)
    tr = DenseDataset(group_by_decision(by_split(recs, "train")), max_per_group=a.max_per_group)
    va = DenseDataset(group_by_decision(by_split(recs, "val")), max_per_group=a.max_per_group,
                      sample=False)
    print(f"train groups={len(tr.groups)} val groups={len(va.groups)}", flush=True)
    dl = dict(batch_size=a.batch_size, collate_fn=collate, num_workers=a.num_workers)
    model = LoopedValueNet(d_model=a.d_model, recurrence=a.recurrence, heads=a.heads,
                           use_global=not a.no_global)
    ckpt = pl.callbacks.ModelCheckpoint(monitor="val_regret", mode="min")
    pl.Trainer(max_epochs=a.epochs, accelerator="auto", devices=1,
               callbacks=[ckpt, Curriculum(a.warmup, tr)], log_every_n_steps=50).fit(
        model, DataLoader(tr, shuffle=True, **dl), DataLoader(va, **dl))


if __name__ == "__main__":
    main()
