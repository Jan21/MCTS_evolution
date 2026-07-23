#!/bin/bash
set -u
SCRATCH=/tmp/claude-1010/-mnt-raid-data-Hyner-Petr-MCTS-evolution/6bc085eb-eb79-4a20-97d5-e670e4d4c89c/scratchpad
cd /mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet
export PYTHONPATH=.
G=$($SCRATCH/claim_gpu.sh g32r4_fwd)
echo "[ctl32] claimed GPU $G"
RR_GRID=32 RR_ROBOTS=4 RR_WALLS=192 RR_ENV_DIR=/mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet/environments_g32r4 \
CUDA_VISIBLE_DEVICES=$G python3 $SCRATCH/train_fwd_lowlr_g32r4.py > $SCRATCH/train_g32r4_fwd3.log 2>&1
echo "[ctl32] done ($?)"
rm -rf /tmp/claude-1010/gpu_claims/gpu$G
echo "[ctl32] G32R4 FWD DONE"
