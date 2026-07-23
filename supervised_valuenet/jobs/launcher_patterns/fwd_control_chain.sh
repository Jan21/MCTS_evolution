#!/bin/bash
set -u
SCRATCH=/tmp/claude-1010/-mnt-raid-data-Hyner-Petr-MCTS-evolution/6bc085eb-eb79-4a20-97d5-e670e4d4c89c/scratchpad
cd /mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet
export PYTHONPATH=.
echo "[ctl] waiting for g24r4 chain"
until grep -q "G24R4 CHAIN DONE" $SCRATCH/g24r4_chain.log 2>/dev/null; do sleep 600; done
echo "[ctl] g24r4 done -> g16r6 forward control retrain (lr 1e-4)"
RR_GRID=16 RR_ROBOTS=6 RR_WALLS=48 RR_ENV_DIR=/mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet/environments_g16r6 \
CUDA_VISIBLE_DEVICES=1 python3 $SCRATCH/train_fwd_lowlr.py > $SCRATCH/train_g16r6_fwd_ctl.log 2>&1
echo "[ctl] control retrain done (exit $?)"
echo "[ctl] FWD CONTROL DONE"
