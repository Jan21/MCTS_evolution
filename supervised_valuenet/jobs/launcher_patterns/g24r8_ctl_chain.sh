#!/bin/bash
set -u
SCRATCH=/tmp/claude-1010/-mnt-raid-data-Hyner-Petr-MCTS-evolution/6bc085eb-eb79-4a20-97d5-e670e4d4c89c/scratchpad
cd /mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet
export PYTHONPATH=.
G=$($SCRATCH/claim_gpu.sh g24r8_ctl)
echo "[ctl24] claimed GPU $G"
RR_GRID=24 RR_ROBOTS=8 RR_WALLS=108 RR_ENV_DIR=/mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet/environments_g24r8 \
CUDA_VISIBLE_DEVICES=$G python3 $SCRATCH/train_fwd_lowlr_g24r8.py > $SCRATCH/train_g24r8_fwd_ctl.log 2>&1
echo "[ctl24] done ($?)"
rm -rf /tmp/claude-1010/gpu_claims/gpu$G
echo "[ctl24] G24R8 CONTROL DONE"
