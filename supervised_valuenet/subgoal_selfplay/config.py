"""Hyperparameters for candidate-scored Expert-Iteration self-play (subgoal planner).

A single `Config` dataclass carries every knob and is threaded through `start_states`,
`selfplay`, and `train_iterate`. The only logic here is the split-id helpers, which
return the env_ids in each split whose per-board pkl exists (the encoders read the
slide graph from `environments/env_{id}.pkl`). See DESIGN.md.

Two modes share the code path: WARM (fine-tune existing policy/value checkpoints;
`policy_ckpt` + `value_ckpt` are required) and FROM-SCRATCH (`from_scratch=True`,
fresh random `PolicyTF` + `LoopedValueNet`, pure self-play). `lr`/`epochs` default
per mode (fine-tune-grade warm, supervised-grade scratch) unless overridden.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from train.encode import ENV_DIR


@dataclass
class Config:
    # --- warm start / output / device ---
    policy_ckpt: str | None = None   # REQUIRED in warm mode (PolicyTF ckpt)
    value_ckpt: str | None = None    # REQUIRED in warm mode (LoopedValueNet ckpt)
    from_scratch: bool = False       # fresh random nets, pure self-play (ignore ckpts)
    out_dir: str = "subgoal_selfplay/runs"
    device: str = "cuda"             # or "cpu" / "auto"

    # --- board splits (inclusive env_id ranges, pkl-backed) ---
    train_boards: tuple[int, int] = (1000, 1799)
    val_boards: tuple[int, int] = (1800, 2399)   # probe-val

    # --- expert-iteration outer loop ---
    n_iters: int = 6
    instances_per_iter: int = 2000
    max_decisions_per_instance: int = 8   # cap on sibling-scored decisions per instance

    # --- start states ---
    sample_max_try: int = 50          # attempts per instance before giving up
    walk_relabel_prob: float = 0.0    # forward-walk-relabel dial, default OFF
    walk_k_max: int = 8               # walk length k ~ U(1, walk_k_max) when ON

    # --- expert search budget (strictly larger than the eval budget) ---
    k_top: int = 8                    # policy top-k per expansion (eval uses 5)
    astar_iters: int = 1500           # expert pop budget (eval uses 1200)
    sibling_iters: int = 300          # small budget for sibling commit-and-complete
    epsilon: float = 0.15             # prob of injecting random beyond-top-k candidates
    epsilon_extra_max: int = 2        # 1..this many extras when epsilon fires (GEN only)
    strict_filter: bool = False       # OFF by default: the plain loop trains on whatever
                                      # the search found (faithful to move_planner_v2). ON
                                      # drops winners failing strict legal realization —
                                      # the cheap ablation arm.
    gen_realize_check: bool = False   # OFF by default. ON = during data generation the
                                      # search discards completed plans that fail strict
                                      # legal realization and keeps searching (the primary
                                      # training-pressure arm; never used in eval).
    prefix_check: bool = False        # OFF by default (Lever A). ON = during generation the
                                      # search prunes child plans whose already-FIXED
                                      # segments fail strict prefix realization before they
                                      # enter the frontier (eval.realize.prefix_playable),
                                      # in both the traced expert and sibling completions,
                                      # AND the complete-pop strict check of
                                      # gen_realize_check is enabled -- so every kept
                                      # winner is strictly playable by construction.

    # --- proposal machinery (skeleton.AStar used for propose + forced exact fixes) ---
    solver_max_iters: int = 4000
    solver_max_frontier: int = 40_000

    # --- replay buffer ---
    replay_iters: int = 5

    # --- optimisation (None -> resolved per mode in __post_init__) ---
    lr: float | None = None           # 1e-4 warm / 3e-4 scratch
    epochs: int | None = None         # 4 warm / 6 scratch
    value_batch_size: int = 8         # decision-groups per batch (train.looped_pc default)
    policy_batch_size: int = 16       # decisions per batch (train.policy_tf default)
    max_per_group: int = 32           # candidate cap per group (train.looped_pc default)
    num_workers: int = 8

    # --- eval (small in-loop regret probe, not the full benchmark) ---
    eval_boards: int = 25
    eval_per_board: int = 2
    eval_k: int = 5
    eval_astar_iters: int = 1200
    eval_every: int = 1

    # --- misc ---
    seed: int = 0

    def __post_init__(self) -> None:
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.lr is None:
            self.lr = 3e-4 if self.from_scratch else 1e-4
        if self.epochs is None:
            self.epochs = 6 if self.from_scratch else 4
        if not self.from_scratch and not (self.policy_ckpt and self.value_ckpt):
            raise ValueError("warm mode requires --policy and --value checkpoints "
                             "(or pass --from-scratch)")

    @staticmethod
    def _existing_ids(bounds: tuple[int, int]) -> list[int]:
        lo, hi = bounds
        return [i for i in range(lo, hi + 1) if os.path.exists(ENV_DIR / f"env_{i}.pkl")]

    def train_ids(self) -> list[int]:
        return self._existing_ids(self.train_boards)

    def val_ids(self) -> list[int]:
        return self._existing_ids(self.val_boards)
