#!/bin/bash
# g24r4 row chain: waits for the g16r6 chain (single-GPU discipline), then
# forward net -> backward value -> backward policy -> bench (CPU).
set -u
SCRATCH=/tmp/claude-1010/-mnt-raid-data-Hyner-Petr-MCTS-evolution/6bc085eb-eb79-4a20-97d5-e670e4d4c89c/scratchpad
cd /mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet
export PYTHONPATH=.

echo "[chain24] waiting for g16r6 chain"
until grep -q "G16R6 CHAIN DONE" $SCRATCH/g16r6_chain.log 2>/dev/null; do sleep 600; done
echo "[chain24] g16r6 done -> g24r4 forward net"

CUDA_VISIBLE_DEVICES=1 python3 -m scaling.train --config g24r4 --system forward -- \
  --data scaling/data/g24r4/forward.jsonl --epochs 15 --batch-size 64 \
  --num-workers 8 > $SCRATCH/train_g24r4_fwd.log 2>&1
echo "[chain24] forward done (exit $?) -> backward value"

CUDA_VISIBLE_DEVICES=1 python3 -m scaling.train --config g24r4 --system backward-value -- \
  --data scaling/data/g24r4/backward.jsonl --epochs 30 --warmup 0 \
  --batch-size 4 --max-per-group 16 > $SCRATCH/train_g24r4_bval.log 2>&1
echo "[chain24] backward value done (exit $?) -> backward policy"

CUDA_VISIBLE_DEVICES=1 python3 -m scaling.train --config g24r4 --system backward-policy -- \
  --data scaling/data/g24r4/backward.jsonl --epochs 30 --batch-size 8 > $SCRATCH/train_g24r4_bpol.log 2>&1
echo "[chain24] backward policy done (exit $?) -> bench (CPU)"

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" python3 -m scaling.bench --config g24r4 \
  --per-board 3 --seed 1 > $SCRATCH/bench_g24r4.log 2>&1
echo "[chain24] bench done (exit $?)"
echo "[chain24] G24R4 CHAIN DONE"
