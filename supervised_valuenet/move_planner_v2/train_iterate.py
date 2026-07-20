"""Outer Expert-Iteration loop for the self-play move planner (plain ExIt).

The single entry point. Each iteration:

1. GENERATE  -- `selfplay.generate_iteration` runs the current net's own A* expert
   over forward-walk start states and emits training records: on a solved path,
   value = plain remaining length (`T-j`) and policy = the committed move; unsolved
   instances are dropped. No HER, no Bellman fallback, no oracle.
2. TRAIN     -- load `MoveNet` from the *latest* ckpt (a fresh random net when
   `--from-scratch`), retarget `lr`/`policy_weight`, and fit on the replay buffer
   with the exact same `MoveDataset`/`collate`/loss as the supervised baseline.
3. EVAL      -- `evaluate.benchmark` reports regret vs the exact oracle on val.

There is no difficulty knob to advance: net-solvability is the implicit curriculum,
so the solvable frontier widens on its own as the net improves. Start either from a
random net (`--from-scratch`, pure self-play) or warm from
`move_planner/checkpoints/best.ckpt`. No oracle labels are used for training -- the
oracle appears only inside `benchmark` to measure regret.

    # from scratch (pure self-play, no oracle, no warm start)
    PYTHONPATH=. python -m move_planner_v2.train_iterate \
        --from-scratch --n-iters 18 --instances-per-iter 8000 --device auto
"""
from __future__ import annotations

import argparse
import os
import random
from collections import deque

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from move_planner.net import MoveNet, MoveDataset, collate
from move_planner.encode import MOVE_CHANNELS, NUM_SLOTS
from move_planner.evaluate import Guide, benchmark
from move_planner_v2.config import Config
from move_planner_v2.selfplay import generate_iteration


# -- fresh net (from-scratch / pure self-play) ---------------------------------

def build_fresh_ckpt(cfg: Config, path: str) -> str:
    """Save a randomly-initialised MoveNet as a checkpoint so the loop can `Guide`/
    `load_from_checkpoint` it exactly like a warm-start ckpt. Architecture matches the
    supervised net (MOVE_CHANNELS inputs incl. slide-displacement planes)."""
    model = MoveNet(in_channels=MOVE_CHANNELS, d_model=192, recurrence=12, heads=4,
                    num_classes=64, lr=cfg.lr, policy_weight=cfg.policy_weight)
    torch.save({"state_dict": model.state_dict(),
                "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": pl.__version__}, path)
    return path


# -- training ------------------------------------------------------------------

def train_on_records(base_ckpt: str, records: list[dict], cfg: Config,
                     out_ckpt: str) -> str:
    """Warm-start `MoveNet` from `base_ckpt`, fine-tune on `records`, save `out_ckpt`.

    `configure_optimizers` and `training_step` both read `self.hparams`, so overriding
    `hparams.lr` / `hparams.policy_weight` here retargets the optimizer *and* the loss
    weighting to the self-play fine-tuning schedule without touching `net.py`.
    """
    model = MoveNet.load_from_checkpoint(base_ckpt, map_location=cfg.device)
    model.hparams.lr = cfg.lr
    model.hparams.policy_weight = cfg.policy_weight
    # in_channels == robots+2+8 iff the ckpt was trained with the slide planes
    # (robots from the ckpt's own hparams; legacy ckpts lack it -> process default).
    robots = model.hparams.get("robots", NUM_SLOTS)
    ds = MoveDataset(records, with_slide=(model.hparams.in_channels == robots + 2 + 8))
    dl = DataLoader(ds, shuffle=True, batch_size=cfg.batch_size,
                    collate_fn=collate, num_workers=cfg.num_workers)
    trainer = pl.Trainer(max_epochs=cfg.epochs, accelerator="auto", devices=1,
                         log_every_n_steps=50, enable_checkpointing=False)
    trainer.fit(model, dl)
    trainer.save_checkpoint(out_ckpt)
    return out_ckpt


# -- eval (regret vs oracle) ---------------------------------------------------

def eval_regret(ckpt: str, cfg: Config, split: str = "val") -> None:
    """Benchmark `ckpt`'s NN A* / greedy planners vs the exact oracle on a split.

    Prints solved / mean_regret / %optimal for NN A*, greedy(pol), greedy(val). The
    oracle (`oracle.optimal_cost`, invoked inside `benchmark`) is used only here.
    """
    guide = Guide(ckpt, cfg.device)
    ids = cfg.test_ids() if split == "test" else cfg.val_ids()
    boards = ids[:cfg.eval_boards]   # small fixed probe, not a full 200-board sweep
    print(f"\n=== eval_regret [{split}] ckpt={ckpt}  boards={len(boards)} "
          f"(per_board={cfg.eval_per_board}, iters={cfg.eval_astar_iters}) ===", flush=True)
    benchmark(guide, boards, per_board=cfg.eval_per_board, seed=cfg.seed,
              k=5, astar_iters=cfg.eval_astar_iters)


# -- outer loop ----------------------------------------------------------------

def iterate(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    rng = random.Random(cfg.seed)
    buffer: deque[list[dict]] = deque(maxlen=cfg.replay_iters)

    if cfg.from_scratch:
        ckpt = build_fresh_ckpt(cfg, os.path.join(cfg.out_dir, "init.ckpt"))
        print(f"[from-scratch] fresh random net -> {ckpt}", flush=True)
    else:
        ckpt = cfg.base_ckpt
        eval_regret(cfg.base_ckpt, cfg, "val")   # warm-start baseline

    for it in range(cfg.n_iters):
        print(f"\n########## iteration {it}  (ckpt={ckpt}) ##########", flush=True)
        guide = Guide(ckpt, cfg.device)
        recs, stats = generate_iteration(guide, cfg, rng)
        print(f"[gen] {stats}", flush=True)

        buffer.append(recs)
        train_recs = [r for chunk in buffer for r in chunk]
        print(f"[train] {len(train_recs)} records ({len(buffer)} replay iters)", flush=True)

        out_ckpt = os.path.join(cfg.out_dir, f"iter{it}.ckpt")
        new_ckpt = train_on_records(ckpt, train_recs, cfg, out_ckpt)
        if (it + 1) % cfg.eval_every == 0 or it == cfg.n_iters - 1:
            eval_regret(new_ckpt, cfg, "val")
        ckpt = new_ckpt   # fine-tune forward from this generation's net

    eval_regret(ckpt, cfg, "test")


# -- CLI -----------------------------------------------------------------------

def main() -> None:
    d = Config()
    p = argparse.ArgumentParser(description="self-play (plain Expert-Iteration) loop")
    p.add_argument("--base-ckpt", default=d.base_ckpt)
    p.add_argument("--from-scratch", action="store_true",
                   help="fresh random net, pure self-play (ignore base_ckpt)")
    p.add_argument("--out-dir", default=d.out_dir)
    p.add_argument("--device", default=d.device, help="cuda / cpu / auto")
    p.add_argument("--n-iters", type=int, default=d.n_iters)
    p.add_argument("--instances-per-iter", type=int, default=d.instances_per_iter)
    p.add_argument("--walk-k-max", type=int, default=d.walk_k_max)
    p.add_argument("--walk-target-bias", type=float, default=d.walk_target_bias)
    p.add_argument("--rand-mix-prob", type=float, default=d.rand_mix_prob)
    p.add_argument("--k-top", type=int, default=d.k_top)
    p.add_argument("--astar-iters", type=int, default=d.astar_iters)
    p.add_argument("--replay-iters", type=int, default=d.replay_iters)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--policy-weight", type=float, default=d.policy_weight)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--eval-every", type=int, default=d.eval_every)
    p.add_argument("--seed", type=int, default=d.seed)
    a = p.parse_args()

    cfg = Config(
        base_ckpt=a.base_ckpt, from_scratch=a.from_scratch, out_dir=a.out_dir, device=a.device,
        n_iters=a.n_iters, instances_per_iter=a.instances_per_iter,
        walk_k_max=a.walk_k_max, walk_target_bias=a.walk_target_bias, rand_mix_prob=a.rand_mix_prob,
        k_top=a.k_top, astar_iters=a.astar_iters, replay_iters=a.replay_iters,
        epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, policy_weight=a.policy_weight,
        num_workers=a.num_workers, eval_every=a.eval_every, seed=a.seed,
    )
    iterate(cfg)


if __name__ == "__main__":
    main()
