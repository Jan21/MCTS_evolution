#!/bin/bash
set -u
SCRATCH=/tmp/claude-1010/-mnt-raid-data-Hyner-Petr-MCTS-evolution/6bc085eb-eb79-4a20-97d5-e670e4d4c89c/scratchpad
cd /mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet
export PYTHONPATH=.
echo "[sp2] waiting for forward control retrain"
until grep -q "FWD CONTROL DONE" $SCRATCH/fwd_control_chain.log 2>/dev/null; do sleep 600; done
echo "[sp2] GPU free -> backward self-play with per-step playability check"
CUDA_VISIBLE_DEVICES=1 python3 -m subgoal_selfplay.train_iterate \
  --policy checkpoints_backward/policy_v2.ckpt --value checkpoints_backward/value_v2.ckpt \
  --prefix-check --gen-realize-check \
  --out-dir subgoal_selfplay/runs_warm_prefix > $SCRATCH/sp_warm_prefix.log 2>&1
echo "[sp2] self-play arm exited ($?)"
