"""Depth-recurrent transformer value network (Lightning module).

Each of the 256 grid cells is a token. A single transformer block is applied K
times with shared weights (Universal-Transformer style), so depth = number of
propagation rounds, which is what shortest-path / reachability reasoning needs.
The head is a classifier over cost-to-go bins; the loss is soft cross-entropy
against the HL-Gauss target. A scalar cost is read off as the expected bin and
fed to the value/MAE/top1-optimal metrics.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

GRID = 16


class DepthRecurrentTransformer(pl.LightningModule):
    def __init__(self, in_channels, num_classes=32, d_model=128, nhead=4,
                 dim_ff=256, depth=8, dropout=0.0, lr=3e-4, weight_decay=1e-4,
                 pool="mean"):
        super().__init__()
        self.save_hyperparameters()

        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos = nn.Parameter(torch.zeros(GRID * GRID, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.step_emb = nn.Parameter(torch.zeros(depth, d_model))
        nn.init.normal_(self.step_emb, std=0.02)
        self.block = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self._val = []

    def forward(self, x):                       # x: [B, C, 16, 16]
        b = x.shape[0]
        tok = x.flatten(2).transpose(1, 2)       # [B, 256, C]
        h = self.input_proj(tok) + self.pos      # [B, 256, d]
        for k in range(self.hparams.depth):      # weight-shared recurrence
            h = self.block(h + self.step_emb[k])
        h = self.norm(h)
        pooled = h.mean(1) if self.hparams.pool == "mean" else h[:, 0]
        return self.head(pooled)                 # [B, num_classes]

    # -- loss / steps -------------------------------------------------------

    def _soft_ce(self, logits, target):
        return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()

    def _scalar(self, logits):
        p = F.softmax(logits, dim=-1)
        idx = torch.arange(p.shape[-1], device=p.device, dtype=p.dtype)
        return (p * idx).sum(-1)                  # expected bin

    def training_step(self, batch, _):
        logits = self(batch["x"])
        loss = self._soft_ce(logits, batch["y"])
        self.log("train_loss", loss, prog_bar=True, batch_size=batch["x"].shape[0])
        return loss

    def _eval_step(self, batch, prefix):
        logits = self(batch["x"])
        loss = self._soft_ce(logits, batch["y"])
        self.log(f"{prefix}_loss", loss, prog_bar=True, batch_size=batch["x"].shape[0])
        pred = self._scalar(logits)
        for i in range(pred.shape[0]):
            self._val.append((int(batch["group"][i]), bool(batch["is_optimal"][i]),
                              float(pred[i]), float(batch["cost_to_go"][i])))

    def _eval_epoch_end(self, prefix):
        if not self._val:
            return
        nc = self.hparams.num_classes
        mae = sum(abs(min(p, nc - 1) - min(c, nc - 1)) for _, _, p, c in self._val) / len(self._val)
        acc = sum(round(min(p, nc - 1.0)) == min(round(c), nc - 1)
                  for _, _, p, c in self._val) / len(self._val)
        # top1_optimal: per decision group, is the cheapest-predicted candidate optimal?
        groups = {}
        for g, opt, p, c in self._val:
            groups.setdefault(g, []).append((p, opt))
        top1 = sum(min(v)[1] for v in groups.values()) / len(groups)
        self.log_dict({f"{prefix}_mae": mae, f"{prefix}_acc": acc,
                       f"{prefix}_top1_optimal": top1}, prog_bar=True)
        self._val.clear()

    def validation_step(self, batch, _):
        self._eval_step(batch, "val")

    def on_validation_epoch_end(self):
        self._eval_epoch_end("val")

    def test_step(self, batch, _):
        self._eval_step(batch, "test")

    def on_test_epoch_end(self):
        self._eval_epoch_end("test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs if self.trainer else 50)
        return {"optimizer": opt, "lr_scheduler": sched}
