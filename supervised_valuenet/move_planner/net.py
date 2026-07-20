"""Two-headed move net: one shared encoder, a value head and a policy head.

Flips the subgoal value net (`train/looped_pc.py`) into the move formulation the
supervisor asked for:

    state  --[shared looped transformer encoder]-->  H  --+--> VALUE  : cost-to-go
                                                          +--> POLICY : which
                                                               (robot, direction)

- Encoder: reused VERBATIM from `looped_pc` -- the weight-tied `LoopedLayer`
  (global + A_all + A_ind edge-masked heads), the per-board adjacency masks
  `_adj(env_id)`, a learned positional embedding, `recurrence` loops. Only the
  input `Linear` widens to the move state channels.
- VALUE head: reads the global/scratchpad token plus the goal-cell and
  target-robot-cell embeddings -> HL-Gauss classifier over cost-to-go bins
  (identical calibration trick + `_value` expected-bin readout as looped_pc).
- POLICY head: gathers the four robot-cell embeddings -> 4 direction logits each
  -> a masked distribution over the (robot x direction) action set. Trained to the
  optimal-move set; illegal (no-op) actions are masked to -inf.

Loss = HL-Gauss value CE + `policy_weight` * masked policy cross-entropy.

    python -m move_planner.net --data move_planner/data/moves.jsonl --epochs 50
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from train.looped_pc import LoopedLayer, _adj
from move_planner.encode import (
    x257, robot_cells, dest_cells, legal_mask, policy_target, _ix,
    MOVE_CHANNELS, NUM_SLOTS, NUM_DIRS, GRID,
)
from nn.benchmark import load, by_split


# -- dataset ------------------------------------------------------------------

class MoveDataset(Dataset):
    """One item per labelled board state. No decision-grouping / ranking needed."""

    def __init__(self, records, with_slide=True):
        self.records = records
        self.with_slide = with_slide

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        A_all, A_ind = _adj(r["env_id"])
        rc = robot_cells(r)
        val_cells = torch.tensor([_ix(r["target"]), int(rc[r["target_idx"]])],
                                 dtype=torch.long)  # goal cell, target-robot cell
        return dict(
            x=x257(r, self.with_slide), A_all=A_all, A_ind=A_ind,
            robot_cells=rc, dest_cells=dest_cells(r), val_cells=val_cells,
            legal=legal_mask(r), ptar=policy_target(r),
            ctg=float(r["cost_to_go"]), full=bool(r.get("full", True)),
        )


def collate(batch):
    return dict(
        x=torch.stack([b["x"] for b in batch]),                # [B,N+1,C]
        A_all=torch.stack([b["A_all"] for b in batch]),        # [B,N+1,N+1]
        A_ind=torch.stack([b["A_ind"] for b in batch]),
        robot_cells=torch.stack([b["robot_cells"] for b in batch]),  # [B,R]
        dest_cells=torch.stack([b["dest_cells"] for b in batch]),    # [B,R,4]
        val_cells=torch.stack([b["val_cells"] for b in batch]),      # [B,2]
        legal=torch.stack([b["legal"] for b in batch]),        # [B,R,4]
        ptar=torch.stack([b["ptar"] for b in batch]),          # [B,R,4]
        ctg=torch.tensor([b["ctg"] for b in batch]),           # [B]
        full=torch.tensor([b["full"] for b in batch], dtype=torch.bool),  # [B]
    )


# -- model --------------------------------------------------------------------

class MoveNet(pl.LightningModule):
    # grid/robots default to the process config (RR_GRID/RR_ROBOTS) at construction
    # and are stored in hparams, so load_from_checkpoint rebuilds the exact
    # architecture from the checkpoint alone, whatever the current env vars say.
    def __init__(self, in_channels=MOVE_CHANNELS, d_model=192, recurrence=12, heads=4,
                 use_global=True, num_classes=64, sigma=1.0, policy_weight=1.0,
                 lr=3e-4, weight_decay=1e-4, grid=GRID, robots=NUM_SLOTS):
        super().__init__()
        self.save_hyperparameters()
        self._n = grid * grid          # cells; row _n is the global token
        self.enc = nn.Linear(in_channels, d_model)
        self.pos = nn.Parameter(torch.randn(self._n + 1, d_model) * 0.02)
        self.layer = LoopedLayer(d_model, heads, use_global)
        self.value_head = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, num_classes))
        # policy reads [robot-cell embedding, slide-destination embedding] -> 1 logit
        # per (robot, dir): the destination gives the 1-step lookahead needed to pick
        # the optimal move (a static per-robot readout cannot, and does not learn).
        self.policy_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.register_buffer("bins", torch.arange(num_classes, dtype=torch.float32))
        self._val = []

    # encoder (identical to looped_pc._encode)
    def _encode(self, b):
        X = self.enc(b["x"]) + self.pos
        for _ in range(self.hparams.recurrence):
            X = self.layer(X, b["A_all"], b["A_ind"])
        return X

    def forward(self, b):
        X = self._encode(b)                                    # [B,N+1,d]
        B = X.shape[0]
        rows = torch.arange(B, device=X.device)[:, None]
        glob = X[:, self._n]                                   # [B,d] scratchpad token
        vc = X[rows, b["val_cells"]].reshape(B, -1)            # [B,2d] goal + target robot
        value_logits = self.value_head(torch.cat([glob, vc], -1))   # [B,num_classes]
        robots = X[rows, b["robot_cells"]]                     # [B,R,d]
        dest = X[rows.unsqueeze(-1), b["dest_cells"]]          # [B,R,4,d] destination embeds
        rob = robots.unsqueeze(2).expand(-1, -1, NUM_DIRS, -1)  # [B,R,4,d]
        policy_logits = self.policy_head(torch.cat([rob, dest], -1)).squeeze(-1)  # [B,R,4]
        return value_logits, policy_logits

    def _value(self, logits):
        return (F.softmax(logits, -1) * self.bins).sum(-1)

    def _hl_gauss(self, ctg):
        dd = (self.bins[None, :] - ctg.clamp(0, self.hparams.num_classes - 1)[:, None]) \
            / self.hparams.sigma
        w = torch.exp(-0.5 * dd * dd)
        return w / w.sum(-1, keepdim=True)

    def _masked_policy_loss(self, policy_logits, legal, ptar):
        """Cross-entropy over legal actions vs the soft optimal-move target."""
        B = policy_logits.shape[0]
        logits = policy_logits.reshape(B, -1)     # [B, R*4] actions
        mask = legal.reshape(B, -1)
        target = ptar.reshape(B, -1)
        logits = logits.masked_fill(~mask, float("-inf"))
        logp = F.log_softmax(logits, -1)
        valid = target.sum(-1) > 0                             # skip goal/no-target rows
        if valid.sum() == 0:
            return policy_logits.sum() * 0.0
        ce = -(target * logp.nan_to_num(neginf=0.0)).sum(-1)
        return ce[valid].mean()

    def training_step(self, b, _):
        value_logits, policy_logits = self(b)
        v_loss = -(self._hl_gauss(b["ctg"]) * F.log_softmax(value_logits, -1)).sum(-1).mean()
        p_loss = self._masked_policy_loss(policy_logits, b["legal"], b["ptar"])
        loss = v_loss + self.hparams.policy_weight * p_loss
        self.log_dict({"train_loss": loss, "train_v": v_loss, "train_p": p_loss},
                      prog_bar=True, batch_size=b["ctg"].shape[0])
        return loss

    def validation_step(self, b, _):
        value_logits, policy_logits = self(b)
        val = self._value(value_logits)
        B = policy_logits.shape[0]
        logits = policy_logits.reshape(B, -1).masked_fill(
            ~b["legal"].reshape(B, -1), float("-inf"))
        pred = logits.argmax(-1)                               # chosen action
        best = b["ptar"].reshape(B, -1) > 0                    # optimal action set
        full = b["full"]
        for i in range(B):
            # only score policy top-1 on records with the COMPLETE optimal set
            scored = bool(full[i]) and bool(best[i].any())
            self._val.append((float(val[i]), float(b["ctg"][i]),
                              bool(best[i, pred[i]]) if scored else None))

    def on_validation_epoch_end(self):
        if not self._val:
            return
        mae = sum(abs(p - c) for p, c, _ in self._val) / len(self._val)
        pol = [ok for _, _, ok in self._val if ok is not None]
        top1 = sum(pol) / len(pol) if pol else float("nan")
        self.log_dict({"val_mae": mae, "val_policy_top1": top1}, prog_bar=True)
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
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--num-classes", type=int, default=64)
    p.add_argument("--policy-weight", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=0, help="early-stop on val_mae (0=off)")
    p.add_argument("--no-slide", action="store_true", help="drop slide-displacement channels")
    p.add_argument("--no-global", action="store_true")
    a = p.parse_args()

    recs = load(a.data)
    with_slide = not a.no_slide
    tr = MoveDataset(by_split(recs, "train"), with_slide)
    va = MoveDataset(by_split(recs, "val"), with_slide)
    print(f"train states={len(tr)} val states={len(va)}", flush=True)
    in_ch = MOVE_CHANNELS if with_slide else (MOVE_CHANNELS - 8)
    dl = dict(batch_size=a.batch_size, collate_fn=collate, num_workers=a.num_workers)
    model = MoveNet(in_channels=in_ch, d_model=a.d_model, recurrence=a.recurrence,
                    heads=a.heads, use_global=not a.no_global, num_classes=a.num_classes,
                    policy_weight=a.policy_weight)
    # value MAE converges in ~1 epoch; policy top-1 is the slower, harder metric, so
    # checkpoint/early-stop on it (value stays low throughout).
    ckpt = pl.callbacks.ModelCheckpoint(monitor="val_policy_top1", mode="max", save_last=True)
    callbacks = [ckpt]
    if a.patience > 0:
        callbacks.append(pl.callbacks.EarlyStopping(monitor="val_policy_top1", mode="max",
                                                    patience=a.patience))
    pl.Trainer(max_epochs=a.epochs, accelerator="auto", devices=1,
               callbacks=callbacks, log_every_n_steps=50).fit(
        model, DataLoader(tr, shuffle=True, **dl), DataLoader(va, **dl))


if __name__ == "__main__":
    main()
