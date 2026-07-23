#!/bin/bash
set -u
SCRATCH=/tmp/claude-1010/-mnt-raid-data-Hyner-Petr-MCTS-evolution/6bc085eb-eb79-4a20-97d5-e670e4d4c89c/scratchpad
cd /mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet
export PYTHONPATH=.
G=$($SCRATCH/claim_gpu.sh g16r8_ctl)
echo "[ctl8] claimed GPU $G"
RR_GRID=16 RR_ROBOTS=8 RR_WALLS=48 RR_ENV_DIR=/mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet/environments_g16r8 \
CUDA_VISIBLE_DEVICES=$G python3 $SCRATCH/train_fwd_lowlr_g16r8.py > $SCRATCH/train_g16r8_fwd_ctl.log 2>&1
echo "[ctl8] control retrain done ($?)"
rm -rf /tmp/claude-1010/gpu_claims/gpu$G
echo "[ctl8] G16R8 CONTROL DONE"
