#!/bin/bash
set -u
SCRATCH=/tmp/claude-1010/-mnt-raid-data-Hyner-Petr-MCTS-evolution/6bc085eb-eb79-4a20-97d5-e670e4d4c89c/scratchpad
cd /mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet
export PYTHONPATH=.
echo "[g24r8] waiting for data"
until grep -q "DATAGEN DONE" $SCRATCH/datagen_g24r8.log 2>/dev/null; do sleep 300; done
G=$($SCRATCH/claim_gpu.sh g24r8)
echo "[g24r8] claimed GPU $G"
CUDA_VISIBLE_DEVICES=$G python3 -m scaling.train --config g24r8 --system backward-value -- --data scaling/data/g24r8/backward.jsonl --epochs 30 --warmup 0 --batch-size 4 --max-per-group 16 > $SCRATCH/train_g24r8_bval.log 2>&1
echo "[g24r8] bval done ($?)"
CUDA_VISIBLE_DEVICES=$G python3 -m scaling.train --config g24r8 --system backward-policy -- --data scaling/data/g24r8/backward.jsonl --epochs 30 --batch-size 8 > $SCRATCH/train_g24r8_bpol.log 2>&1
echo "[g24r8] bpol done ($?)"
CUDA_VISIBLE_DEVICES=$G python3 -m scaling.train --config g24r8 --system forward -- --data scaling/data/g24r8/forward.jsonl --epochs 8 --batch-size 64 --num-workers 8 > $SCRATCH/train_g24r8_fwd.log 2>&1
echo "[g24r8] fwd done ($?)"
rm -rf /tmp/claude-1010/gpu_claims/gpu$G
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" python3 -m scaling.bench --config g24r8 --per-board 3 --seed 1 > $SCRATCH/bench_g24r8.log 2>&1
echo "[g24r8] bench done ($?)"
echo "[g24r8] CHAIN DONE"
