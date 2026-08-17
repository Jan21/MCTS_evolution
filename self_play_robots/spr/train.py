"""Train the size-free policy / value nets on one or more corpora (mixed sizes).

One entry point for both nets so the self-play loop has a single, warm-start
capable, collapse-guarded trainer (PROBLEM.md section 4.4: warm start,
CollapseStop, ModelCheckpoint(min val_regret), save_last + explicit resume, all
wired from day one):

    PYTHONPATH=supervised_valuenet:self_play_robots python -m spr.train \
        --system value --data g16r4=nn/data/combined.jsonl \
        --data g24r4=scaling/data/g24r4/backward.jsonl \
        --init self_play_robots/assets/labeler_prod_v1_s11.ckpt \
        --epochs 12 --batch-size 4 --max-per-group 16 --out-dir runs/m1/value_warm

    ... --system policy --epochs 25 --batch-size 8 --out-dir runs/m1/policy

Value: `nn_labeler.model.SizeFreeValueNet` + `nn_labeler.dataset` +
`nn_labeler.encode` (imported, unchanged); recipe = nn_labeler/train.py
(two-phase curriculum warmup, CollapseStop 3 x spread<0.05, ckpt on min
val_regret) plus `--init` (warm start from any SizeFreeValueNet ckpt; the
loop's iteration k -> k+1 handoff) and `--splits` overrides for corpora on
non-standard board ids (FINDINGS 79c). Policy: `spr.nets.SizeFreePolicyNet`,
same data plumbing, ckpt on min val_regret (= regret@1).

Board-id hygiene: records are split by their OWN config's board ranges
(train/val/test disjoint; bench boards live in the test range and are never
trained on). Corpora produced by self-play on fresh lean boards pass
`--splits CFG:train=2000-2699,val=2700-2899` (ids are config-namespaced).

Last line on success: `SPR TRAIN DONE <system> <out-dir>`.
"""
from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from spr import REPO, SV


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (REPO / q if (REPO / q).exists() else SV / q)


class CollapseStop(pl.Callback):
    """nn_labeler/train.py's guard (kept verbatim in behaviour): stop when
    val_group_spread < floor for `patience` consecutive epochs; the best
    checkpoint is already banked by ModelCheckpoint."""

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
            print(f"[spr.train] COLLAPSE STOP: val_group_spread < {self.floor} for "
                  f"{self.patience} epochs at epoch {trainer.current_epoch}; best "
                  f"checkpoint retained", flush=True)
            trainer.should_stop = True


def _combined_env_dir(a, cfg):
    """Symlink dir holding every env_<id>.pkl the corpora reference (config
    pool + self-play board dirs) so the per-size stack's single RR_ENV_DIR
    resolves all of them. None when no record leaves the config's pool."""
    dirs = {}
    for item in a.data:
        _, path = item.split("=", 1)
        with open(_abs(path)) as fh:
            for i, line in enumerate(fh):
                if a.limit_records is not None and i >= a.limit_records:
                    break
                r = json.loads(line)
                if r.get("boards_dir"):
                    dirs[int(r["env_id"])] = os.path.abspath(r["boards_dir"])
    if not dirs:
        return None
    out_dir = Path(a.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    comb = out_dir / "envs"
    comb.mkdir(parents=True, exist_ok=True)
    src_pool = Path(cfg.env_dir_abs)
    for f in src_pool.glob("env_*.pkl"):
        dst = comb / f.name
        if not dst.exists():
            dst.symlink_to(f)
    for eid, d in dirs.items():
        dst = comb / f"env_{eid}.pkl"
        if not dst.exists():
            dst.symlink_to(Path(d) / f"env_{eid}.pkl")
    print(f"[spr.train] persize: combined env dir {comb} ({len(dirs)} self-play boards)",
          flush=True)
    return comb


def parse_splits(specs):
    """--splits CFG:train=a-b,val=c-d[,test=e-f]  ->  {cfg: {split: [ids]}}"""
    import re
    from scaling.configs import parse_ids
    out = {}
    for s in specs or []:
        cfg, rest = s.split(":", 1)
        d = {}
        # id specs contain commas themselves, so split only where a 'name=' follows
        for m in re.finditer(r"(train|val|test)=([0-9,\-]+?)(?=(,(?:train|val|test)=)|$)", rest):
            d[m.group(1)] = parse_ids(m.group(2))
        if not d or any(not v for v in d.values()):
            raise SystemExit(f"--splits {s!r}: could not parse any non-empty split")
        out[cfg] = d
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--system", choices=["policy", "value"], required=True)
    p.add_argument("--data", action="append", required=True, metavar="CONFIG=PATH")
    p.add_argument("--splits", action="append", default=None,
                   metavar="CFG:train=a-b,val=c-d",
                   help="override a config's board-id splits (self-play corpora "
                        "on fresh boards)")
    p.add_argument("--init", default=None, help="warm start: checkpoint to load")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8, help="decision GROUPS per batch")
    p.add_argument("--max-per-group", type=int, default=32)
    p.add_argument("--lr", type=float, default=None,
                   help="default: 3e-4 cold, 1e-4 warm (FINDINGS 44/50 rescue lr)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--recurrence", type=int, default=12)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--pe", choices=["none", "sin2d"], default="none")
    p.add_argument("--num-classes", type=int, default=96, help="value bins")
    p.add_argument("--warmup", type=int, default=4,
                   help="value: curriculum warmup epochs on the easiest 30%% "
                        "(cold starts only; 0 disables)")
    p.add_argument("--temp", type=float, default=1.0, help="policy soft-target temp")
    p.add_argument("--torch-seed", type=int, default=None)
    p.add_argument("--limit-records", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--byref", action="store_true", help="policy: keep by-reference records")
    p.add_argument("--arch", choices=["sizefree", "persize"], default="sizefree",
                   help="persize: train/policy_tf.py PolicyTF or train/looped_pc.py "
                        "LoopedValueNet (learned n^2 positional table; ONE config per "
                        "process; --init warm-starts them too)")
    a = p.parse_args(argv)

    from scaling.configs import get, apply_env
    from nn_labeler import dataset, encode
    if a.arch == "persize":
        # one config per process: pin RR_* BEFORE any train.* import; if the
        # corpora reference self-play board dirs, serve every board through
        # ONE symlink dir (train.encode reads env_<id>.pkl from RR_ENV_DIR only)
        cfgs = {item.split("=", 1)[0] for item in a.data}
        if len(cfgs) != 1:
            raise SystemExit("--arch persize needs exactly one config in --data")
        cfg0 = get(cfgs.pop())
        apply_env(cfg0)
        _persize_env_dir = _combined_env_dir(a, cfg0)
        if _persize_env_dir is not None:
            os.environ["RR_ENV_DIR"] = str(_persize_env_dir)
    from nn_labeler.model import SizeFreeValueNet, collate_groups
    from spr.nets import SizeFreePolicyNet, PolicyGroupDataset, collate_policy

    if a.torch_seed is not None:
        pl.seed_everything(a.torch_seed, workers=True)
    seed = a.torch_seed if a.torch_seed is not None else 0
    lr = a.lr if a.lr is not None else (1e-4 if a.init else 3e-4)

    specs = []
    for item in a.data:
        name, path = item.split("=", 1)
        specs.append((get(name), _abs(path)))
    overrides = parse_splits(a.splits)
    unknown = set(overrides) - {c.name for c, _ in specs}
    if unknown:
        raise SystemExit(f"--splits names configs not in --data: {sorted(unknown)}")
    ranges = {}
    for cfg, _ in specs:
        r = {s: cfg.ids(s) for s in ("train", "val", "test")}
        if cfg.name in overrides:
            r.update(overrides[cfg.name])
        ranges[cfg.name] = r

    train_groups, val_groups, corpus_info = [], [], []
    for cfg, path in specs:
        recs = dataset.load_corpus(str(path), cfg.name, cfg.grid, cfg.env_dir_abs,
                                   limit=a.limit_records)
        # self-play records live on their own (fresh, lean) board dirs
        for r in recs:
            if r.get("boards_dir"):
                r["_env_dir"] = os.path.abspath(r["boards_dir"])
        tr = dataset.by_split(recs, "train", ranges)
        va = dataset.by_split(recs, "val", ranges)
        gtr, gva = dataset.group_by_decision(tr), dataset.group_by_decision(va)
        over = sum(1 for r in recs if int(round(float(r["cost_to_go"]))) > a.num_classes - 1)
        print(f"[spr.train] {cfg.name} n={cfg.grid} records={len(recs)} "
              f"train={len(tr)}r/{len(gtr)}g val={len(va)}r/{len(gva)}g "
              f"ctg>{a.num_classes - 1}={over} src={path}", flush=True)
        corpus_info.append(dict(config=cfg.name, path=str(path), records=len(recs),
                                train_groups=len(gtr), val_groups=len(gva)))
        train_groups += gtr
        val_groups += gva
    if not train_groups or not val_groups:
        raise SystemExit("empty train or val split -- check --data/--splits")

    out_dir = Path(a.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "DATA.json").write_text(json.dumps(dict(
        system=a.system, arch=a.arch, corpora=corpus_info, init=a.init, lr=lr, epochs=a.epochs,
        batch_size=a.batch_size, max_per_group=a.max_per_group, pe=a.pe,
        num_classes=a.num_classes, torch_seed=a.torch_seed, splits=a.splits,
        started=time.strftime("%Y-%m-%dT%H:%M:%S"),
        slurm_job_id=os.environ.get("SLURM_JOB_ID")), indent=1))

    if a.arch == "persize":
        # the supervised per-size trainers' datasets/collates, unchanged; a
        # single board size per process, so no size bucketing is needed
        import torch.utils.data as tud
        if a.system == "value":
            from train.looped_pc import LoopedValueNet, DenseDataset, collate as v_collate
            tr_ds = DenseDataset(train_groups, a.max_per_group, True)
            va_ds = DenseDataset(val_groups, a.max_per_group, False)
            coll = v_collate
            if a.init:
                model = LoopedValueNet.load_from_checkpoint(a.init, map_location="cpu")
                model.hparams.lr = lr
                model.hparams.weight_decay = a.weight_decay
                print(f"[spr.train] warm start from {a.init} (lr={lr})", flush=True)
            else:
                model = LoopedValueNet(d_model=a.d_model, recurrence=a.recurrence, heads=a.heads,
                                       num_classes=a.num_classes if a.num_classes != 96 else 50,
                                       lr=lr, weight_decay=a.weight_decay)
            callbacks_extra = [CollapseStop()]   # inert: LoopedValueNet logs no spread
        else:
            from train.policy_tf import PolicyTF, PolicyTFDataset, collate as p_collate
            tr_ds = PolicyTFDataset(train_groups)
            va_ds = PolicyTFDataset(val_groups)
            coll = p_collate
            if a.init:
                model = PolicyTF.load_from_checkpoint(a.init, map_location="cpu")
                model.hparams.lr = lr
                model.hparams.weight_decay = a.weight_decay
                print(f"[spr.train] warm start from {a.init} (lr={lr})", flush=True)
            else:
                model = PolicyTF(d_model=a.d_model, recurrence=a.recurrence, heads=a.heads,
                                 temp=a.temp, lr=lr, weight_decay=a.weight_decay)
            callbacks_extra = []
        monitor = "val_regret"
        print(f"[spr.train] persize {a.system}: train {len(tr_ds)} val {len(va_ds)}", flush=True)
        dl_tr = DataLoader(tr_ds, batch_size=a.batch_size, shuffle=True, collate_fn=coll,
                           num_workers=a.num_workers)
        dl_va = DataLoader(va_ds, batch_size=a.batch_size, shuffle=False, collate_fn=coll,
                           num_workers=a.num_workers)
    elif a.system == "value":
        tr_ds = dataset.GroupDataset(train_groups, a.max_per_group, True)
        va_ds = dataset.GroupDataset(val_groups, a.max_per_group, False)
        coll = functools.partial(collate_groups, featurize_fn=encode.node_features,
                                 adjacency_fn=encode.adjacency, key_fn=encode.key_indices,
                                 coord_channels=False, num_classes=a.num_classes)
        if a.init:
            model = SizeFreeValueNet.load_from_checkpoint(a.init, map_location="cpu")
            if model.hparams.num_classes != a.num_classes:
                raise SystemExit(f"--init has num_classes={model.hparams.num_classes}, "
                                 f"want {a.num_classes}")
            model.hparams.lr = lr
            model.hparams.weight_decay = a.weight_decay
            print(f"[spr.train] warm start from {a.init} (lr={lr})", flush=True)
        else:
            model = SizeFreeValueNet(d_model=a.d_model, recurrence=a.recurrence,
                                     heads=a.heads, num_classes=a.num_classes, lr=lr,
                                     weight_decay=a.weight_decay, pe=a.pe, in_channels=9)
        monitor = "val_regret"
        callbacks_extra = [CollapseStop()]
    else:
        tr_ds = PolicyGroupDataset(train_groups, byref=a.byref)
        va_ds = PolicyGroupDataset(val_groups, byref=a.byref)
        coll = collate_policy
        if a.init:
            model = SizeFreePolicyNet.load_from_checkpoint(a.init, map_location="cpu")
            model.hparams.lr = lr
            model.hparams.weight_decay = a.weight_decay
            print(f"[spr.train] warm start from {a.init} (lr={lr})", flush=True)
        else:
            model = SizeFreePolicyNet(d_model=a.d_model, recurrence=a.recurrence,
                                      heads=a.heads, temp=a.temp, lr=lr,
                                      weight_decay=a.weight_decay, pe=a.pe)
        monitor = "val_regret"
        callbacks_extra = []
    if a.arch != "persize":
        print(f"[spr.train] {a.system}: train {len(tr_ds)} val {len(va_ds)} "
              f"(groups/decisions)", flush=True)
        dl_tr = DataLoader(tr_ds, collate_fn=coll, num_workers=a.num_workers,
                           batch_sampler=dataset.SizeBucketBatchSampler(tr_ds, a.batch_size, True, seed))
        dl_va = DataLoader(va_ds, collate_fn=coll, num_workers=a.num_workers,
                           batch_sampler=dataset.SizeBucketBatchSampler(va_ds, a.batch_size, False, seed))

    ckpt = pl.callbacks.ModelCheckpoint(monitor=monitor, mode="min", save_top_k=1,
                                        save_last=True)
    resume = None
    if os.environ.get("RR_RESUME") == "1":
        lasts = sorted(glob.glob(str(out_dir / "lightning_logs" / "version_*" /
                                     "checkpoints" / "last.ckpt")), key=os.path.getmtime)
        resume = lasts[-1] if lasts else None
        print(f"[resume] RR_RESUME=1 -> {resume or 'no last.ckpt, fresh start'}", flush=True)
    accel = {"auto": "auto", "cpu": "cpu", "cuda": "gpu"}[a.device]

    warm_epochs = 0
    if a.system == "value" and not a.init and resume is None and a.arch != "persize":
        warm_epochs = min(a.warmup, max(a.epochs - 1, 0))
    if warm_epochs > 0:
        cut = max(1, int(len(tr_ds.groups) * 0.3))
        easy = dataset.GroupDataset(list(tr_ds.groups[:cut]), a.max_per_group, True)
        dl_easy = DataLoader(easy, collate_fn=coll, num_workers=a.num_workers,
                             batch_sampler=dataset.SizeBucketBatchSampler(easy, a.batch_size, True, seed))
        print(f"[spr.train] warmup fit: {warm_epochs} epochs on easiest {cut} groups", flush=True)
        warm = pl.Trainer(max_epochs=warm_epochs, accelerator=accel, devices=1,
                          default_root_dir=str(out_dir), log_every_n_steps=50,
                          enable_checkpointing=False)
        warm.fit(model, dl_easy, dl_va)
    max_ep = max(a.epochs - warm_epochs, 1)
    trainer = pl.Trainer(max_epochs=max_ep, accelerator=accel, devices=1,
                         default_root_dir=str(out_dir),
                         callbacks=[ckpt, *callbacks_extra], log_every_n_steps=50)
    trainer.fit(model, dl_tr, dl_va, ckpt_path=resume)
    already_done = resume is not None and trainer.current_epoch >= max_ep
    if "val_regret" not in trainer.callback_metrics and not already_done:
        raise SystemExit("SPR TRAIN FAILED: validation never ran")
    metrics = {k: round(float(v), 4) for k, v in trainer.callback_metrics.items()
               if torch.is_tensor(v) or isinstance(v, (int, float))}
    best = ckpt.best_model_path or None
    best_score = float(ckpt.best_model_score) if ckpt.best_model_score is not None else None
    (out_dir / "RESULT.json").write_text(json.dumps(dict(
        system=a.system, best_ckpt=best, best_val_regret=best_score, final_metrics=metrics,
        epochs_run=trainer.current_epoch, finished=time.strftime("%Y-%m-%dT%H:%M:%S")), indent=1))
    print(f"[spr.train] final metrics: {metrics}", flush=True)
    print(f"[spr.train] best {best} val_regret={best_score}", flush=True)
    print(f"SPR TRAIN DONE {a.system} {out_dir}", flush=True)


if __name__ == "__main__":
    main()
