"""Train the forward MoveNet on self-play records (warm start + val + checkpointing).

A thin wrapper: the model, the dataset, the collate and the loss are
`move_planner.net.{MoveNet, MoveDataset, collate}` VERBATIM, and the warm-start
recipe (load the previous ckpt, retarget `hparams.lr`/`hparams.policy_weight`,
fine-tune) is `move_planner_v2.train_iterate.train_on_records`'s -- which
`--trainer v2` calls literally. What this module adds is the unattended-loop
hygiene PROBLEM.md 4.4 demands and `train_on_records` does not have:

  * a VALIDATION split taken BY BOARD ID (self-play boards are >= 5000; the
    default holds out every `--val-every`-th board id), so val states never
    share a board with train states;
  * `ModelCheckpoint` on the val metric + `save_last` -- an unattended epoch
    that goes bad can never destroy the best epoch;
  * an explicit `--torch-seed`;
  * an optional exact ANCHOR corpus (`move_planner/data/moves*.jsonl` at
    g16r4, `scaling/data/<cfg>/forward.jsonl` elsewhere -- the supervised
    forward records) mixed in, so the loop cannot drift off the supervised
    distribution -- with a HARD DROP of any record on a test/bench board.

`--config` selects the board configuration (`scaling.configs`). It does two
things and nothing else: it exports the config's RR_GRID/RR_ROBOTS/RR_WALLS
BEFORE the first forward-stack import (the forward stack reads them at import
time), and it takes the anchor corpus's train/val/test board ranges from
`cfg.board_ranges` instead of `nn.benchmark.SPLITS`. For the default `g16r4`
those ranges ARE `nn.benchmark.SPLITS` verbatim (0-95,1000-1799 / 1800-2399 /
112-127,2400-2999), so g16r4 behaviour is bit-identical; at g24r4 they are
0-699 / 700-899 / 900-1049, which is exactly the rebind `scaling.train`
performs when it trains the per-size supervised MoveNet -- so the anchor keeps
its own supervised splits and the pinned bench boards (900-1049) are dropped.

Monitor: `val_policy_top1` (max) when the val split contains records with the
COMPLETE optimal-move set (`full=true`, i.e. anchor records) -- the metric
`move_planner/net.py::main` checkpoints on; otherwise `val_mae` (min), because
self-play records are all `full=false` and `val_policy_top1` is then NaN.

    PYTHONPATH=supervised_valuenet:self_play_robots python -m spr.fwd.train \
        --records runs/spr/fwd/selfplay/g16r4_iter1/records.jsonl \
        --init move_planner/checkpoints/candidate_scored.ckpt \
        --anchor move_planner/data/moves_scored.jsonl --anchor-limit 200000 \
        --epochs 6 --lr 1e-4 --out-dir runs/spr/fwd/train/g16r4_iter1

    ... --config g24r4 --anchor scaling/data/g24r4/forward.jsonl \
        --init scaling/runs/g24r4/forward/.../epoch=7-step=76360.ckpt

Last line: `SPR FWD TRAIN DONE <out-dir>` (and the chosen ckpt on the line above).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path


def load_jsonl(path, limit=None, rng=None):
    recs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    if limit and len(recs) > limit:
        recs = (rng or random.Random(0)).sample(recs, limit)
    return recs


def link_env_dir(records, link_dir, default_dir):
    """One board directory for a mixed corpus, built from symlinks.

    The forward twin of `spr/train.py::_combined_env_dir` (the subgoal arm's fix
    for the same problem): `move_planner.encode` / `train.encode` read every
    board from the single process-wide `$RR_ENV_DIR`, resolved AT IMPORT, but a
    self-play buffer mixes fresh per-iteration boards (each record carries its
    own `boards_dir`) with anchor records on the pinned pool. Copying boards
    would be wasteful and writing fresh boards into
    `supervised_valuenet/environments/` is not allowed, so an `envs/` symlink
    farm under the out-dir is materialized and returned for RR_ENV_DIR.

    Difference from `_combined_env_dir`: it symlinks only the boards the records
    actually reference (the pinned pool holds thousands of pkls and the forward
    anchor corpora touch a few hundred), and it refuses two different sources for
    one env_id instead of silently keeping the first. Idempotent.
    """
    link_dir = Path(link_dir)
    link_dir.mkdir(parents=True, exist_ok=True)
    seen = {}
    for r in records:
        eid = int(r["env_id"])
        src = Path(r.get("boards_dir") or default_dir) / f"env_{eid}.pkl"
        prev = seen.get(eid)
        if prev is not None:
            if prev != src:
                raise SystemExit(f"env_{eid}.pkl claimed by two sources: {prev} "
                                 f"and {src}")
            continue
        seen[eid] = src
        if not src.is_file():
            raise SystemExit(f"missing board {src} (record env_id={eid})")
        dst = link_dir / f"env_{eid}.pkl"
        if dst.is_symlink() or dst.exists():
            if Path(os.path.realpath(dst)) != Path(os.path.realpath(src)):
                raise SystemExit(f"{dst} already points elsewhere")
            continue
        os.symlink(src, dst)
    return link_dir, len(seen)


def split_by_board(recs, val_ids=None, val_every=10):
    """(train, val) by env_id. `val_ids` (a set) wins; else every val_every-th
    DISTINCT board id, deterministically, so a board is wholly in one split."""
    if val_ids is None:
        ids = sorted({r["env_id"] for r in recs})
        val_ids = {b for i, b in enumerate(ids) if i % max(1, val_every) == 0}
    tr = [r for r in recs if r["env_id"] not in val_ids]
    va = [r for r in recs if r["env_id"] in val_ids]
    return tr, va, sorted(val_ids)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="g16r4",
                   help="board config (scaling.configs): RR_* env + the anchor "
                        "corpus's train/val/test board ranges")
    p.add_argument("--records", action="append", required=True,
                   help="self-play JSONL (repeatable = replay buffer window)")
    p.add_argument("--anchor", action="append", default=None,
                   help="supervised forward corpus to mix in (repeatable)")
    p.add_argument("--anchor-limit", type=int, default=None,
                   help="subsample each anchor corpus to N records (seeded)")
    p.add_argument("--init", default=None,
                   help="warm start from this MoveNet ckpt (omit = fresh random net)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=None,
                   help="default 1e-4 warm / 3e-4 cold (move_planner_v2 recipe)")
    p.add_argument("--policy-weight", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-every", type=int, default=10,
                   help="hold out every Nth distinct board id for validation")
    p.add_argument("--val-boards", default=None,
                   help="explicit val board ids, e.g. 5100-5119 (overrides --val-every)")
    p.add_argument("--torch-seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="cuda / cpu / auto")
    p.add_argument("--trainer", choices=["val", "v2"], default="val",
                   help="val: + val split, ModelCheckpoint, save_last (default); "
                        "v2: move_planner_v2.train_iterate.train_on_records verbatim")
    p.add_argument("--limit-records", type=int, default=None, help="smoke runs")
    a = p.parse_args(argv)

    from spr import REPO, SV
    sys.path.insert(0, str(SV))
    from scaling.configs import get as get_config, env as config_env
    bcfg = get_config(a.config)
    out_dir = Path(a.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # BEFORE the first forward-stack import: RR_GRID/RR_ROBOTS/RR_WALLS select the
    # board geometry (legacy g16r4 == all UNSET) and `train.encode.ENV_DIR` is
    # resolved at import time, so a mixed buffer needs ONE board dir (below).
    if not bcfg.legacy:
        os.environ.update(config_env(bcfg))
    orig_env_dir = Path(os.environ.get("RR_ENV_DIR") or bcfg.env_dir_abs)
    link_dir = out_dir / "envs"
    link_dir.mkdir(parents=True, exist_ok=True)
    os.environ["RR_ENV_DIR"] = str(link_dir)

    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    from move_planner.net import MoveNet, MoveDataset, collate
    from move_planner.encode import NUM_SLOTS
    from move_planner_v2.config import Config
    from move_planner_v2.train_iterate import build_fresh_ckpt, train_on_records

    # the config's board ranges; for g16r4 identical to `nn.benchmark.SPLITS`,
    # elsewhere the rebind `scaling.train` uses to train the supervised MoveNet
    SPLITS = {s_: bcfg.ids(s_) for s_ in ("train", "val", "test")}

    pl.seed_everything(a.torch_seed, workers=True)
    rng = random.Random(a.torch_seed)
    dev = a.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    lr = a.lr if a.lr is not None else (1e-4 if a.init else 3e-4)

    def _abs(q):
        return q if Path(q).is_absolute() else str(SV / q)

    sp, info = [], []
    for path in a.records:
        r = load_jsonl(_abs(path), a.limit_records, rng)
        info.append({"kind": "selfplay", "path": _abs(path), "records": len(r)})
        sp += r
    anchor = []
    banned = set(SPLITS["test"])                 # the pinned bench boards live here
    n_banned = 0
    for path in (a.anchor or []):
        r = load_jsonl(_abs(path), a.anchor_limit, rng)
        keep = [x for x in r if x["env_id"] not in banned]
        n_banned += len(r) - len(keep)
        info.append({"kind": "anchor", "path": _abs(path), "records": len(keep)})
        anchor += keep
    if n_banned:
        print(f"[spr.fwd.train] dropped {n_banned} anchor records on test/bench boards",
              flush=True)
    if not sp and not anchor:
        raise SystemExit("no records loaded")
    _, n_boards = link_env_dir(sp + anchor, link_dir, orig_env_dir)
    print(f"[spr.fwd.train] board farm: {n_boards} env pkls symlinked into "
          f"{link_dir} (RR_ENV_DIR)", flush=True)

    val_ids = None
    if a.val_boards:
        from scaling.configs import parse_ids
        val_ids = set(parse_ids(a.val_boards))
    sp_tr, sp_va, sp_val_ids = split_by_board(sp, val_ids, a.val_every) if sp else ([], [], [])
    an_tr, an_va = [], []
    if anchor:                       # anchor records keep their OWN standard splits
        a_tr_ids, a_va_ids = set(SPLITS["train"]), set(SPLITS["val"])
        an_tr = [r for r in anchor if r["env_id"] in a_tr_ids]
        an_va = [r for r in anchor if r["env_id"] in a_va_ids]
    train_recs, val_recs = sp_tr + an_tr, sp_va + an_va
    print(f"[spr.fwd.train] selfplay {len(sp)} ({len(sp_tr)}tr/{len(sp_va)}va over "
          f"{len(sp_val_ids)} val boards), anchor {len(anchor)} "
          f"({len(an_tr)}tr/{len(an_va)}va)  ->  train {len(train_recs)} "
          f"val {len(val_recs)}", flush=True)
    if not train_recs:
        raise SystemExit("empty train split -- check --val-every/--val-boards")

    (out_dir / "DATA.json").write_text(json.dumps(dict(
        config=bcfg.name, grid=bcfg.grid, robots=bcfg.robots,
        corpora=info, init=a.init, lr=lr, epochs=a.epochs, batch_size=a.batch_size,
        policy_weight=a.policy_weight, torch_seed=a.torch_seed, trainer=a.trainer,
        val_boards=sp_val_ids[:50], n_train=len(train_recs), n_val=len(val_recs),
        anchor_dropped_test_boards=n_banned, device=dev,
        started=time.strftime("%Y-%m-%dT%H:%M:%S"),
        slurm_job_id=os.environ.get("SLURM_JOB_ID")), indent=1) + "\n")

    cfg = Config(device=dev, epochs=a.epochs, batch_size=a.batch_size, lr=lr,
                 policy_weight=a.policy_weight, num_workers=a.num_workers,
                 seed=a.torch_seed)
    init = _abs(a.init) if a.init else build_fresh_ckpt(cfg, str(out_dir / "init.ckpt"))
    t0 = time.time()

    if a.trainer == "v2":                        # the upstream loop's trainer, verbatim
        best = train_on_records(init, train_recs, cfg, str(out_dir / "model.ckpt"))
        metrics, monitor, score = {}, None, None
    else:
        model = MoveNet.load_from_checkpoint(init, map_location=dev)
        model.hparams.lr = lr
        model.hparams.policy_weight = a.policy_weight
        robots = model.hparams.get("robots", NUM_SLOTS)
        with_slide = model.hparams.in_channels == robots + 2 + 8
        scorable = sum(1 for r in val_recs if r.get("full") and r.get("best_moves"))
        monitor, mode = (("val_policy_top1", "max") if scorable
                         else ("val_mae", "min"))
        print(f"[spr.fwd.train] warm={bool(a.init)} lr={lr} with_slide={with_slide} "
              f"monitor={monitor} ({scorable} scorable val records)", flush=True)
        dl = dict(batch_size=a.batch_size, collate_fn=collate,
                  num_workers=a.num_workers)
        dl_tr = DataLoader(MoveDataset(train_recs, with_slide), shuffle=True, **dl)
        dl_va = DataLoader(MoveDataset(val_recs or train_recs[:max(1, a.batch_size)],
                                       with_slide), **dl)
        ckpt = pl.callbacks.ModelCheckpoint(monitor=monitor, mode=mode, save_top_k=1,
                                            save_last=True)
        resume = None
        if os.environ.get("RR_RESUME") == "1":
            lasts = sorted(glob.glob(str(out_dir / "lightning_logs" / "version_*" /
                                         "checkpoints" / "last.ckpt")),
                           key=os.path.getmtime)
            resume = lasts[-1] if lasts else None
            print(f"[resume] RR_RESUME=1 -> {resume or 'no last.ckpt, fresh start'}",
                  flush=True)
        trainer = pl.Trainer(max_epochs=a.epochs,
                             accelerator={"cpu": "cpu", "cuda": "gpu"}.get(dev, "auto"),
                             devices=1, default_root_dir=str(out_dir),
                             callbacks=[ckpt], log_every_n_steps=50)
        trainer.fit(model, dl_tr, dl_va, ckpt_path=resume)
        metrics = {k: round(float(v), 4) for k, v in trainer.callback_metrics.items()
                   if torch.is_tensor(v) or isinstance(v, (int, float))}
        best = ckpt.best_model_path or None
        score = float(ckpt.best_model_score) if ckpt.best_model_score is not None else None
        if not best:                             # no monitored epoch -> keep last
            trainer.save_checkpoint(str(out_dir / "model.ckpt"))
            best = str(out_dir / "model.ckpt")

    (out_dir / "RESULT.json").write_text(json.dumps(dict(
        best_ckpt=best, monitor=monitor, best_score=score, final_metrics=metrics,
        seconds=round(time.time() - t0, 1),
        finished=time.strftime("%Y-%m-%dT%H:%M:%S")), indent=1) + "\n")
    (out_dir / "BEST.txt").write_text(str(best) + "\n")
    print(f"[spr.fwd.train] final metrics: {metrics}", flush=True)
    print(f"[spr.fwd.train] best {best} {monitor}={score}", flush=True)
    print(f"SPR FWD TRAIN DONE {out_dir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
