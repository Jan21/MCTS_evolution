"""Train the size-free backward value net on one or more corpora at once.

Mirrors `train/looped_pc.py::main` (checkpoint on min val_regret, save_last +
explicit RR_RESUME resume) but takes `--data CONFIG=PATH` repeatedly, so a single
run can mix board sizes and robot counts -- the Plan-C bet from
`nn_labeler/PLANS.md` (one net, trained once on cheap rungs, applied zero-shot up
the ladder). Every record is stamped with its config's `n`/`env_dir` at load, and
batches are bucketed by `n`, so mixing costs nothing but bookkeeping.

Unlike `scaling/train.py` this entrypoint sets no RR_* variables and imports no
`train.*` module: `nn_labeler.encode` takes `n`/`env_dir` per record, so one
process handles every config at once.

    PYTHONPATH=. python -m nn_labeler.train \
        --data g16r6=scaling/data/g16r6/backward.jsonl \
        --data g24r4=scaling/data/g24r4/backward.jsonl \
        --pe sin2d --epochs 20 --batch-size 8

Checkpoints and metrics land in `--out-dir` (default `nn_labeler/runs/<configs>_<pe>`).
The last line on success is `NNLAB TRAIN DONE <out-dir>`.
"""
from __future__ import annotations

import argparse
import functools
import glob
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from nn_labeler import dataset, encode
from nn_labeler.model import SizeFreeValueNet, collate_groups
from scaling.configs import REPO, get


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO / p)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", action="append", required=True, metavar="CONFIG=PATH",
                   help="repeatable: a scaling config name and its corpus, e.g. "
                        "g16r6=scaling/data/g16r6/backward.jsonl (paths are "
                        "repo-root-relative unless absolute)")
    p.add_argument("--pe", choices=["sin2d", "coord", "none"], default="sin2d",
                   help="position signal; coord => 13 input channels")
    p.add_argument("--num-classes", type=int, default=96,
                   help="HL-Gauss bins; sized for 32x32 costs from day one")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8, help="decision GROUPS per batch")
    p.add_argument("--max-per-group", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--recurrence", type=int, default=12)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=4,
                   help="curriculum warmup EPOCHS spent on the easiest 30%% of "
                        "decision groups before the full fit (two static "
                        "phases; the reference's per-epoch frac ramp is not "
                        "implementable with this Lightning version's length "
                        "caching, FINDINGS 52). 0 disables. Battery 1: without "
                        "a curriculum, 7/8 cold trainings collapsed to the "
                        "constant-value plateau")
    p.add_argument("--torch-seed", type=int, default=None)
    p.add_argument("--limit-records", type=int, default=None,
                   help="per corpus; for smoke runs")
    p.add_argument("--out-dir", default=None,
                   help="default nn_labeler/runs/<configs>_<pe>")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    a = p.parse_args()

    specs = []
    for item in a.data:
        if "=" not in item:
            raise SystemExit(f"--data wants CONFIG=PATH, got {item!r}")
        name, path = item.split("=", 1)
        specs.append((get(name), _abs(path)))

    if a.torch_seed is not None:
        pl.seed_everything(a.torch_seed, workers=True)
        print(f"[nnlab] seed_everything({a.torch_seed})", flush=True)
    seed = a.torch_seed if a.torch_seed is not None else 0

    # {config: {split: [board ids]}} -- each config's own splits, so merged
    # corpora never leak boards across splits (env_id 0-1199 is reused by every
    # non-legacy config; the config namespace is what disambiguates them).
    ranges = {cfg.name: {s: cfg.ids(s) for s in ("train", "val", "test")}
              for cfg, _ in specs}

    train_groups, val_groups = [], []
    for cfg, path in specs:
        recs = dataset.load_corpus(str(path), cfg.name, cfg.grid, cfg.env_dir_abs,
                                   limit=a.limit_records)
        tr = dataset.by_split(recs, "train", ranges)
        va = dataset.by_split(recs, "val", ranges)
        # grouped per corpus, never on the merged pool: the decision key is
        # (env_id, target, ...) and env_id aliases across configs.
        gtr, gva = dataset.group_by_decision(tr), dataset.group_by_decision(va)
        over = sum(1 for r in recs
                   if int(round(float(r["cost_to_go"]))) > a.num_classes - 1)
        print(f"[nnlab] {cfg.name} n={cfg.grid} r={cfg.robots} records={len(recs)} "
              f"train={len(tr)}r/{len(gtr)}g val={len(va)}r/{len(gva)}g "
              f"ctg>{a.num_classes - 1}={over} src={path}", flush=True)
        train_groups += gtr
        val_groups += gva
    print(f"[nnlab] pooled: train {len(train_groups)} groups, "
          f"val {len(val_groups)} groups", flush=True)
    if not train_groups or not val_groups:
        raise SystemExit("empty train or val split -- check --data / --limit-records")

    tr_ds = dataset.GroupDataset(train_groups, a.max_per_group, True)
    va_ds = dataset.GroupDataset(val_groups, a.max_per_group, False)
    coll = functools.partial(collate_groups, featurize_fn=encode.node_features,
                             adjacency_fn=encode.adjacency, key_fn=encode.key_indices,
                             coord_channels=(a.pe == "coord"),
                             num_classes=a.num_classes)
    dl_tr = DataLoader(tr_ds, collate_fn=coll, num_workers=a.num_workers,
                       batch_sampler=dataset.SizeBucketBatchSampler(
                           tr_ds, a.batch_size, True, seed))
    dl_va = DataLoader(va_ds, collate_fn=coll, num_workers=a.num_workers,
                       batch_sampler=dataset.SizeBucketBatchSampler(
                           va_ds, a.batch_size, False, seed))

    model = SizeFreeValueNet(d_model=a.d_model, recurrence=a.recurrence,
                             heads=a.heads, num_classes=a.num_classes, lr=a.lr,
                             weight_decay=a.weight_decay, pe=a.pe,
                             in_channels=13 if a.pe == "coord" else 9)

    name = "_".join([cfg.name for cfg, _ in specs] + [a.pe])
    out_dir = _abs(a.out_dir) if a.out_dir else (REPO / "nn_labeler" / "runs" / name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save_last=True keeps a resumable trainer state next to the best-weights ckpt
    # (two 16 h walltime kills in FINDINGS 43 each cost a full retrain without it).
    ckpt = pl.callbacks.ModelCheckpoint(monitor="val_regret", mode="min",
                                        save_top_k=1, save_last=True)

    class CollapseStop(pl.Callback):
        """Stop when the net falls into the constant-value mode and stays there.

        The signature is a near-zero spread of predictions WITHIN a candidate
        group: the net answers the same number for every candidate, so ranking
        is decided by candidate order alone. Seen cold in battery 1 (7/8 runs)
        and MID-TRAINING in the first production run (job 4609800: spread
        2.12 -> 0.00 at epoch 2, then 24 dead epochs; FINDINGS 56). The best
        checkpoint is already saved by `ckpt`, so stopping loses nothing and
        frees the GPU.
        """

        def __init__(self, floor=0.05, patience=3):
            self.floor, self.patience, self.hits = floor, patience, 0

        def on_validation_epoch_end(self, trainer, _pl_module):
            if trainer.sanity_checking:
                return
            v = trainer.callback_metrics.get("val_group_spread")
            if v is None:
                return
            self.hits = self.hits + 1 if float(v) < self.floor else 0
            if self.hits >= self.patience:
                print(f"[nnlab] COLLAPSE STOP: val_group_spread < {self.floor} "
                      f"for {self.patience} consecutive epochs at epoch "
                      f"{trainer.current_epoch}; best checkpoint retained",
                      flush=True)
                trainer.should_stop = True
    # Resume is EXPLICIT (RR_RESUME=1), never automatic: silently resuming a stale
    # last.ckpt after a data or recipe change is the silent-success class
    # FINDINGS 38 documents.
    resume = None
    if os.environ.get("RR_RESUME") == "1":
        lasts = sorted(glob.glob(str(out_dir / "lightning_logs" / "version_*" /
                                     "checkpoints" / "last.ckpt")),
                       key=os.path.getmtime)
        resume = lasts[-1] if lasts else None
        print(f"[resume] RR_RESUME=1 -> {resume or 'no last.ckpt, fresh start'}",
              flush=True)

    accel = {"auto": "auto", "cpu": "cpu", "cuda": "gpu"}[a.device]
    # Curriculum = two STATIC fit phases (see dataset.py docstring for why not
    # a per-epoch frac ramp). Phase A: easiest 30% of groups (GroupDataset is
    # min-ctg sorted) for --warmup epochs, no checkpointing — its only job is
    # to steer cold training away from the constant-value plateau (battery 1,
    # FINDINGS 50). Phase B: the full corpus, monitored checkpointing. A
    # resume (RR_RESUME=1 with a last.ckpt present) skips phase A: the
    # last.ckpt is already past it.
    warm_epochs = min(a.warmup, max(a.epochs - 1, 0))
    if warm_epochs > 0 and resume is None:
        cut = max(1, int(len(tr_ds.groups) * 0.3))
        easy_ds = dataset.GroupDataset(
            [g for g in tr_ds.groups[:cut]], a.max_per_group, True)
        dl_easy = DataLoader(easy_ds, collate_fn=coll,
                             num_workers=a.num_workers,
                             batch_sampler=dataset.SizeBucketBatchSampler(
                                 easy_ds, a.batch_size, True, seed))
        print(f"[nnlab] warmup fit: {warm_epochs} epochs on the easiest "
              f"{cut}/{len(tr_ds.groups)} groups", flush=True)
        warm = pl.Trainer(max_epochs=warm_epochs, accelerator=accel, devices=1,
                          default_root_dir=str(out_dir), log_every_n_steps=50,
                          enable_checkpointing=False)
        warm.fit(model, dl_easy, dl_va)
        if "val_regret" not in warm.callback_metrics:
            raise SystemExit("NNLAB TRAIN FAILED: warmup validation never ran")

    max_ep = max(a.epochs - warm_epochs, 1)
    trainer = pl.Trainer(max_epochs=max_ep, accelerator=accel, devices=1,
                         default_root_dir=str(out_dir),
                         callbacks=[ckpt, CollapseStop()],
                         log_every_n_steps=50)
    trainer.fit(model, dl_tr, dl_va, ckpt_path=resume)
    # A resume whose checkpoint already sits at max_epochs runs no epoch and so
    # logs no metrics -- that is "already finished", not a failure. Only a run
    # that was SUPPOSED to train and produced no validation is broken
    # (FINDINGS 52's silent-no-validation class).
    already_done = resume is not None and trainer.current_epoch >= max_ep
    if "val_regret" not in trainer.callback_metrics and not already_done:
        raise SystemExit("NNLAB TRAIN FAILED: validation never ran "
                         "(no val_regret in callback metrics)")
    if already_done:
        print(f"[nnlab] resume: checkpoint already at max_epochs={max_ep}; "
              f"no training needed", flush=True)

    metrics = {k: round(float(v), 4) for k, v in trainer.callback_metrics.items()
               if torch.is_tensor(v) or isinstance(v, (int, float))}
    print(f"[nnlab] final metrics: {metrics}", flush=True)
    print(f"[nnlab] best {ckpt.best_model_path or '(none)'} "
          f"val_regret={float(ckpt.best_model_score) if ckpt.best_model_score is not None else float('nan'):.4f}",
          flush=True)
    print(f"NNLAB TRAIN DONE {out_dir}", flush=True)


if __name__ == "__main__":
    main()
