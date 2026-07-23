#!/bin/bash
# g16r6 row completion chain, single GPU discipline:
# wait for forward training + backward relabel -> backward value -> backward policy
# -> bench (CPU) -> compare (CPU). GPU1 assumed ours throughout.
set -u
SCRATCH=/tmp/claude-1010/-mnt-raid-data-Hyner-Petr-MCTS-evolution/6bc085eb-eb79-4a20-97d5-e670e4d4c89c/scratchpad
cd /mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet
export PYTHONPATH=.

echo "[chain] waiting for g16r6 forward training + backward relabel"
while pgrep -f "scaling.train --config g16r6 --system forward" >/dev/null; do sleep 300; done
until grep -q "g16r6.*relabeled" $SCRATCH/scaling_relabel.log 2>/dev/null; do sleep 300; done
echo "[chain] prerequisites done -> backward value net (seeded like the good run)"

CUDA_VISIBLE_DEVICES=1 python3 -m scaling.train --config g16r6 --system backward-value -- \
  --data scaling/data/g16r6/backward.jsonl --epochs 30 --warmup 0 \
  --batch-size 8 --max-per-group 32 > $SCRATCH/train_g16r6_bval.log 2>&1
echo "[chain] backward value done (exit $?) -> backward policy"

CUDA_VISIBLE_DEVICES=1 python3 -m scaling.train --config g16r6 --system backward-policy -- \
  --data scaling/data/g16r6/backward.jsonl --epochs 30 > $SCRATCH/train_g16r6_bpol.log 2>&1
echo "[chain] backward policy done (exit $?) -> bench (CPU)"

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" python3 -m scaling.bench --config g16r6 \
  --per-board 3 --seed 1 > $SCRATCH/bench_g16r6.log 2>&1
echo "[chain] bench done (exit $?)"
echo "[chain] G16R6 CHAIN DONE — compare step launches on notification"
