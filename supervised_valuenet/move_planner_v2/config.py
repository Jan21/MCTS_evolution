"""Hyperparameters for the self-play (plain Expert-Iteration) move planner.

A single `Config` dataclass carries every knob and is threaded through `start_states`,
`selfplay`, and `train_iterate`. The only logic here is `train_ids`/`val_ids`/`test_ids`,
which return the env_ids in each split whose per-board pkl exists (the encoder reads the
slide graph from `environments/env_{id}.pkl`). See DESIGN.md.

Defaults are the simplified FROM-SCRATCH recipe: one forward-walk generator, plain ExIt
(value = remaining path length, policy = committed move, unsolved dropped), and a
supervised-grade optimisation budget (a random-init net needs it).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from train.encode import ENV_DIR


@dataclass
class Config:
    # --- warm start / output / device ---
    base_ckpt: str = "move_planner/checkpoints/best.ckpt"
    from_scratch: bool = False   # fresh random MoveNet, pure self-play (ignore base_ckpt)
    out_dir: str = "move_planner_v2/runs"
    device: str = "cuda"         # or "cpu" / "auto"

    # --- board splits (inclusive env_id ranges, pkl-backed) ---
    train_boards: tuple[int, int] = (1000, 1599)
    val_boards: tuple[int, int] = (1800, 1999)
    test_boards: tuple[int, int] = (2400, 2599)

    # --- expert-iteration outer loop ---
    n_iters: int = 18
    instances_per_iter: int = 8000

    # --- generator: target-biased forward-walk + a fraction of raw eval-distribution ---
    walk_k_max: int = 16          # k ~ U(1, walk_k_max); the oracle-free difficulty dial
    walk_target_bias: float = 0.6 # bias the forward walk toward moving the target (depth)
    rand_mix_prob: float = 0.25   # fraction of raw random eval-distribution instances (deep tail)

    # --- expert search budget (strictly larger than the eval budget) ---
    k_top: int = 8
    astar_iters: int = 4000

    # --- value target ---
    value_clamp: int = 63        # cost-to-go bin clamp (matches num_classes-1)

    # --- replay buffer ---
    replay_iters: int = 5

    # --- optimisation (supervised-grade: the net is random-init, not warm) ---
    epochs: int = 6
    batch_size: int = 256
    lr: float = 3e-4
    policy_weight: float = 1.0
    num_workers: int = 8

    # --- eval (small in-loop regret probe, not a full 200-board sweep) ---
    eval_boards: int = 25
    eval_per_board: int = 2
    eval_astar_iters: int = 1500
    eval_every: int = 1

    # --- misc ---
    seed: int = 0

    def __post_init__(self) -> None:
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _existing_ids(bounds: tuple[int, int]) -> list[int]:
        lo, hi = bounds
        return [i for i in range(lo, hi + 1) if os.path.exists(ENV_DIR / f"env_{i}.pkl")]

    def train_ids(self) -> list[int]:
        return self._existing_ids(self.train_boards)

    def val_ids(self) -> list[int]:
        return self._existing_ids(self.val_boards)

    def test_ids(self) -> list[int]:
        return self._existing_ids(self.test_boards)
