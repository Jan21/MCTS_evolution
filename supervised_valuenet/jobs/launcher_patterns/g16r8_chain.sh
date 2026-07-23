#!/bin/bash
# g16r8 GPU chain: waits for (a) the forward-control rerun to free GPU1 and
# (b) g16r8 data; then 3 trainings -> bench -> compare. Keeps GPU1 busy.
set -u
SCRATCH=/tmp/claude-1010/-mnt-raid-data-Hyner-Petr-MCTS-evolution/6bc085eb-eb79-4a20-97d5-e670e4d4c89c/scratchpad
cd /mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet
export PYTHONPATH=.

echo "[g16r8] waiting for GPU (control rerun) + data"
while pgrep -f "train_fwd_lowlr" >/dev/null; do sleep 300; done
until grep -q "DATAGEN DONE" $SCRATCH/datagen_g16r8.log 2>/dev/null; do sleep 300; done
echo "[g16r8] prerequisites done -> backward value"

CUDA_VISIBLE_DEVICES=1 python3 -m scaling.train --config g16r8 --system backward-value -- \
  --data scaling/data/g16r8/backward.jsonl --epochs 30 --warmup 0 \
  --batch-size 8 --max-per-group 32 > $SCRATCH/train_g16r8_bval.log 2>&1
echo "[g16r8] backward value done (exit $?) -> backward policy"

CUDA_VISIBLE_DEVICES=1 python3 -m scaling.train --config g16r8 --system backward-policy -- \
  --data scaling/data/g16r8/backward.jsonl --epochs 30 > $SCRATCH/train_g16r8_bpol.log 2>&1
echo "[g16r8] backward policy done (exit $?) -> forward"

CUDA_VISIBLE_DEVICES=1 python3 -m scaling.train --config g16r8 --system forward -- \
  --data scaling/data/g16r8/forward.jsonl --epochs 8 --batch-size 128 \
  --num-workers 8 > $SCRATCH/train_g16r8_fwd.log 2>&1
echo "[g16r8] forward done (exit $?) -> bench"

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" python3 -m scaling.bench --config g16r8 \
  --per-board 3 --seed 1 > $SCRATCH/bench_g16r8.log 2>&1
echo "[g16r8] bench done (exit $?)"
echo "[g16r8] G16R8 CHAIN DONE — run compare next"
