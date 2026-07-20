"""Outer Expert-Iteration loop for the candidate-scored subgoal-planner self-play.

The single entry point. Each iteration:

1. GENERATE -- `selfplay.generate_iteration`: the current nets' own traced A* solves
   sampled instances; each decision on a winning chain is candidate-scored (siblings
   commit-and-completed by a small-budget run of the same search) into full
   combined.jsonl-schema groups. Unsolved instances are dropped. No oracle.
2. TRAIN    -- fine-tune the VALUE net (`train.looped_pc.LoopedValueNet` +
   `DenseDataset`/`collate`, imported verbatim) then the POLICY net
   (`train.policy_tf.PolicyTF` + its dataset) on a replay window of the last
   `replay_iters` generations. Trainers are instantiated directly (no CLI shell-out);
   `default_root_dir=out_dir` keeps lightning_logs out of the repo root.
3. EVAL     -- a small end2end-style probe on held-out probe-val boards at the EVAL
   budget (k=5, 1200 iters). The exact solver (`skeleton.astar.AStar.solve_plan`)
   appears ONLY here, as the regret reference -- never as a training label.

    # warm (fine-tune existing supervised ckpts)
    PYTHONPATH=. python -m subgoal_selfplay.train_iterate \
        --policy <policy.ckpt> --value <value.ckpt> --out-dir subgoal_selfplay/runs_warm

    # from scratch (random-init nets, pure self-play)
    PYTHONPATH=. python -m subgoal_selfplay.train_iterate \
        --from-scratch --out-dir subgoal_selfplay/runs_scratch
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import deque

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from GridEnv import GridEnv
from skeleton.astar import AStar, _initial_plan
from nn.benchmark import group_by_decision
from train.looped_pc import LoopedValueNet, DenseDataset, collate as value_collate
from train.policy_tf import PolicyTF, PolicyTFDataset, collate as policy_collate
from subgoal_selfplay.config import Config
from subgoal_selfplay.selfplay import generate_iteration, nn_astar_from
from subgoal_selfplay.start_states import random_reachable_instance

# selfplay imports eval.end2end, which disables autograd globally; re-assert here so
# import order can never leave training silently gradient-free.
torch.set_grad_enabled(True)


# -- nets ------------------------------------------------------------------------

def build_nets(cfg: Config) -> tuple[PolicyTF, LoopedValueNet]:
    """Warm: load both supervised checkpoints and retarget lr. From-scratch: fresh
    random nets with the supervised default architecture."""
    if cfg.from_scratch:
        policy = PolicyTF(lr=cfg.lr)
        value = LoopedValueNet(lr=cfg.lr)
        print("[from-scratch] fresh random PolicyTF + LoopedValueNet", flush=True)
    else:
        policy = PolicyTF.load_from_checkpoint(cfg.policy_ckpt, map_location="cpu")
        value = LoopedValueNet.load_from_checkpoint(cfg.value_ckpt, map_location="cpu")
        policy.hparams.lr = cfg.lr
        value.hparams.lr = cfg.lr
        print(f"[warm] policy={cfg.policy_ckpt} value={cfg.value_ckpt} lr={cfg.lr}", flush=True)
    return policy, value


# -- training ----------------------------------------------------------------------

def _trainer(cfg: Config) -> pl.Trainer:
    return pl.Trainer(max_epochs=cfg.epochs,
                      accelerator="gpu" if cfg.device.startswith("cuda") else "cpu",
                      devices=1, log_every_n_steps=20,
                      enable_checkpointing=False,       # ckpt saved explicitly below
                      default_root_dir=cfg.out_dir,     # lightning_logs under out_dir
                      num_sanity_val_steps=0)


def train_value(value: LoopedValueNet, records: list[dict], cfg: Config, out_ckpt: str):
    """Fine-tune the value net on the replay records with the supervised dataset/loss."""
    value.train()   # generation leaves the net in eval mode
    ds = DenseDataset(group_by_decision(records), max_per_group=cfg.max_per_group)
    dl = DataLoader(ds, shuffle=True, batch_size=cfg.value_batch_size,
                    collate_fn=value_collate, num_workers=cfg.num_workers)
    trainer = _trainer(cfg)
    trainer.fit(value, dl)
    trainer.save_checkpoint(out_ckpt)
    return value


def train_policy(policy: PolicyTF, records: list[dict], cfg: Config, out_ckpt: str):
    """Fine-tune the policy net on the replay records with the supervised dataset/loss."""
    policy.train()   # generation leaves the net in eval mode
    ds = PolicyTFDataset(group_by_decision(records))
    dl = DataLoader(ds, shuffle=True, batch_size=cfg.policy_batch_size,
                    collate_fn=policy_collate, num_workers=cfg.num_workers)
    trainer = _trainer(cfg)
    trainer.fit(policy, dl)
    trainer.save_checkpoint(out_ckpt)
    return policy


# -- probe eval (regret vs exact reference) -----------------------------------------

def probe_eval(cfg: Config, policy: PolicyTF, value: LoopedValueNet, tag: str) -> dict:
    """Small end2end-style benchmark on probe-val boards at the EVAL budget.

    ORACLE REFERENCE (eval-only): `AStar.solve_plan` computes the optimal plan cost
    as the regret reference. It is never used for training labels anywhere in this
    package. Instances are re-derived from a fixed seed, so every iteration probes
    the same set.
    """
    dev = cfg.device
    policy = policy.to(dev).eval()
    value = value.to(dev).eval()
    solver = AStar(max_iters=cfg.solver_max_iters, max_frontier=cfg.solver_max_frontier)
    from eval.realize import strict_moves

    rng = random.Random(cfg.seed + 99991)            # fixed probe instance stream
    boards = cfg.val_ids()[:cfg.eval_boards]
    n = solved = opt_fail = ref_fail = strict_solved = 0
    regrets, strict_regrets = [], []
    with torch.no_grad():
        for gid in boards:
            env, s0 = GridEnv.from_env(gid)
            colors = [s0.target_robot.color] + [h.color for h in s0.helpers]
            for _ in range(cfg.eval_per_board):
                st = random_reachable_instance(env, colors, rng, cfg.sample_max_try)
                if st is None:
                    continue
                opt = solver.solve_plan(env, st, _initial_plan(env, st))  # ORACLE (eval-only)
                if opt is None:
                    opt_fail += 1
                    continue
                n += 1
                # strict reference: the exact solver's plan realized to legal moves
                # (eval-only oracle use; a trend gauge, never a headline number)
                ref_stx = strict_moves(env, st, opt, log=None)
                if ref_stx is None:
                    ref_fail += 1
                plan = nn_astar_from(env, st, solver, policy, value, gid, dev,
                                     k=cfg.eval_k, max_iters=cfg.eval_astar_iters)
                if plan is not None:
                    solved += 1
                    regrets.append(int(plan.cost()) - int(opt.cost()))
                    stx = strict_moves(env, st, plan, log=None)
                    if stx is not None:
                        strict_solved += 1
                        if ref_stx is not None:
                            strict_regrets.append(stx - ref_stx)
    out = {"n": n, "solved": solved,
           "solve_rate": round(solved / n, 3) if n else 0.0,
           "mean_regret": round(sum(regrets) / len(regrets), 3) if regrets else float("nan"),
           "optimal": sum(1 for r in regrets if r == 0), "opt_fail": opt_fail,
           "strict_solved": strict_solved,
           "strict_solve_rate": round(strict_solved / n, 3) if n else 0.0,
           "mean_strict_regret": round(sum(strict_regrets) / len(strict_regrets), 3) if strict_regrets else None,
           "ref_fail": ref_fail}
    print(f"[probe {tag}] {out}", flush=True)
    return out


# -- outer loop ----------------------------------------------------------------------

def iterate(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    policy, value = build_nets(cfg)
    buffer: deque[list[dict]] = deque(maxlen=cfg.replay_iters)

    if not cfg.from_scratch:
        probe_eval(cfg, policy, value, "warm-baseline")

    for it in range(cfg.n_iters):
        print(f"\n########## iteration {it} ##########", flush=True)
        recs, stats = generate_iteration(cfg, policy, value, rng)
        print(f"[gen {it}] {stats}", flush=True)
        with open(os.path.join(cfg.out_dir, f"iter{it}_records.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

        buffer.append(recs)
        train_recs = [r for chunk in buffer for r in chunk]
        print(f"[train {it}] {len(train_recs)} records ({len(buffer)} replay iters)", flush=True)
        if train_recs:
            train_value(value, train_recs, cfg,
                        os.path.join(cfg.out_dir, f"iter{it}_value.ckpt"))
            train_policy(policy, train_recs, cfg,
                         os.path.join(cfg.out_dir, f"iter{it}_policy.ckpt"))

        probe = {}
        if (it + 1) % cfg.eval_every == 0 or it == cfg.n_iters - 1:
            probe = probe_eval(cfg, policy, value, f"iter{it}")
        with open(os.path.join(cfg.out_dir, f"iter{it}_stats.json"), "w") as f:
            json.dump({"iteration": it, "gen": stats, "probe": probe}, f, indent=1)
        print(f"[summary {it}] solve_rate={stats['solve_rate']} "
              f"gen_strict={stats['strict_pass_rate']} "
              f"records={stats['n_records']} groups={stats['n_groups']} "
              f"mean_group={stats['mean_group_size']} "
              f"probe_solved={probe.get('solved', '-')}/{probe.get('n', '-')} "
              f"probe_regret={probe.get('mean_regret', '-')} "
              f"probe_strict={probe.get('strict_solve_rate', '-')} "
              f"probe_strict_regret={probe.get('mean_strict_regret', '-')}", flush=True)


# -- CLI ------------------------------------------------------------------------------

def main() -> None:
    d = Config(from_scratch=True)   # defaults source only (skips warm-ckpt validation)
    p = argparse.ArgumentParser(description="candidate-scored ExIt self-play (subgoal planner)")
    p.add_argument("--policy", default=None, help="warm-start PolicyTF ckpt")
    p.add_argument("--value", default=None, help="warm-start LoopedValueNet ckpt")
    p.add_argument("--from-scratch", action="store_true",
                   help="fresh random nets, pure self-play (ignore --policy/--value)")
    p.add_argument("--out-dir", default=d.out_dir)
    p.add_argument("--device", default=d.device, help="cuda / cpu / auto")
    p.add_argument("--n-iters", type=int, default=d.n_iters)
    p.add_argument("--instances-per-iter", type=int, default=d.instances_per_iter)
    p.add_argument("--max-decisions", type=int, default=d.max_decisions_per_instance)
    p.add_argument("--k-top", type=int, default=d.k_top)
    p.add_argument("--astar-iters", type=int, default=d.astar_iters)
    p.add_argument("--sibling-iters", type=int, default=d.sibling_iters)
    p.add_argument("--epsilon", type=float, default=d.epsilon)
    p.add_argument("--strict-filter", action="store_true",
                   help="drop winners whose plan fails strict legal realization (ablation arm)")
    p.add_argument("--gen-realize-check", action="store_true",
                   help="generation search discards unplayable complete plans and keeps "
                        "searching (primary training-pressure arm)")
    p.add_argument("--prefix-check", action="store_true",
                   help="Lever A: generation search prunes child plans whose already-"
                        "fixed segments fail strict prefix realization (and implies the "
                        "complete-pop check of --gen-realize-check), so every kept "
                        "winner is strictly playable by construction")
    p.add_argument("--walk-relabel-prob", type=float, default=d.walk_relabel_prob)
    p.add_argument("--walk-k-max", type=int, default=d.walk_k_max)
    p.add_argument("--replay-iters", type=int, default=d.replay_iters)
    p.add_argument("--lr", type=float, default=None, help="default: 1e-4 warm / 3e-4 scratch")
    p.add_argument("--epochs", type=int, default=None, help="default: 4 warm / 6 scratch")
    p.add_argument("--value-batch-size", type=int, default=d.value_batch_size)
    p.add_argument("--policy-batch-size", type=int, default=d.policy_batch_size)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--eval-boards", type=int, default=d.eval_boards)
    p.add_argument("--eval-per-board", type=int, default=d.eval_per_board)
    p.add_argument("--eval-k", type=int, default=d.eval_k)
    p.add_argument("--eval-astar-iters", type=int, default=d.eval_astar_iters)
    p.add_argument("--eval-every", type=int, default=d.eval_every)
    p.add_argument("--seed", type=int, default=d.seed)
    a = p.parse_args()

    cfg = Config(
        policy_ckpt=a.policy, value_ckpt=a.value, from_scratch=a.from_scratch,
        out_dir=a.out_dir, device=a.device,
        n_iters=a.n_iters, instances_per_iter=a.instances_per_iter,
        max_decisions_per_instance=a.max_decisions,
        k_top=a.k_top, astar_iters=a.astar_iters, sibling_iters=a.sibling_iters,
        epsilon=a.epsilon, strict_filter=a.strict_filter,
        gen_realize_check=a.gen_realize_check, prefix_check=a.prefix_check,
        walk_relabel_prob=a.walk_relabel_prob, walk_k_max=a.walk_k_max,
        replay_iters=a.replay_iters, lr=a.lr, epochs=a.epochs,
        value_batch_size=a.value_batch_size, policy_batch_size=a.policy_batch_size,
        num_workers=a.num_workers,
        eval_boards=a.eval_boards, eval_per_board=a.eval_per_board, eval_k=a.eval_k,
        eval_astar_iters=a.eval_astar_iters, eval_every=a.eval_every, seed=a.seed,
    )
    iterate(cfg)


if __name__ == "__main__":
    main()
