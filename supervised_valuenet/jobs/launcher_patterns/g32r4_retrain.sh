#!/bin/bash
set -u
SCRATCH=/tmp/claude-1010/-mnt-raid-data-Hyner-Petr-MCTS-evolution/6bc085eb-eb79-4a20-97d5-e670e4d4c89c/scratchpad
cd /mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet
export PYTHONPATH=.
G=$($SCRATCH/claim_gpu.sh g32r4_retrain)
echo "[g32r4-rt] claimed GPU $G; backward value at batch 2 / group 8"
CUDA_VISIBLE_DEVICES=$G python3 -m scaling.train --config g32r4 --system backward-value -- \
  --data scaling/data/g32r4/backward.jsonl --epochs 25 --warmup 0 \
  --batch-size 2 --max-per-group 8 > $SCRATCH/train_g32r4_bval2.log 2>&1
echo "[g32r4-rt] bval done ($?)"
CUDA_VISIBLE_DEVICES=$G python3 -m scaling.train --config g32r4 --system forward -- \
  --data scaling/data/g32r4/forward.jsonl --epochs 8 --batch-size 8 \
  --num-workers 8 > $SCRATCH/train_g32r4_fwd2.log 2>&1
echo "[g32r4-rt] fwd done ($?)"
rm -rf /tmp/claude-1010/gpu_claims/gpu$G
echo "[g32r4-rt] RETRAIN DONE"
