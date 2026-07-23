#!/bin/bash
set -u
SCRATCH=/tmp/claude-1010/-mnt-raid-data-Hyner-Petr-MCTS-evolution/6bc085eb-eb79-4a20-97d5-e670e4d4c89c/scratchpad
cd /mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet
export PYTHONPATH=.
echo "[g32r4] waiting for data"
until grep -q "DATAGEN DONE" $SCRATCH/datagen_g32r4.log 2>/dev/null; do sleep 300; done
G=$($SCRATCH/claim_gpu.sh g32r4)
echo "[g32r4] claimed GPU $G"
CUDA_VISIBLE_DEVICES=$G python3 -m scaling.train --config g32r4 --system backward-value -- --data scaling/data/g32r4/backward.jsonl --epochs 30 --warmup 0 --batch-size 4 --max-per-group 16 > $SCRATCH/train_g32r4_bval.log 2>&1
echo "[g32r4] bval done ($?)"
CUDA_VISIBLE_DEVICES=$G python3 -m scaling.train --config g32r4 --system backward-policy -- --data scaling/data/g32r4/backward.jsonl --epochs 30 --batch-size 8 > $SCRATCH/train_g32r4_bpol.log 2>&1
echo "[g32r4] bpol done ($?)"
CUDA_VISIBLE_DEVICES=$G python3 -m scaling.train --config g32r4 --system forward -- --data scaling/data/g32r4/forward.jsonl --epochs 8 --batch-size 64 --num-workers 8 > $SCRATCH/train_g32r4_fwd.log 2>&1
echo "[g32r4] fwd done ($?)"
rm -rf /tmp/claude-1010/gpu_claims/gpu$G
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" python3 -m scaling.bench --config g32r4 --per-board 3 --seed 1 > $SCRATCH/bench_g32r4.log 2>&1
echo "[g32r4] bench done ($?)"
echo "[g32r4] CHAIN DONE"
