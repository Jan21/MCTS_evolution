#!/bin/bash
# M1 -- size-free rebuild (PROBLEM.md section 6.2 / 8): train ONE pe=none
# policy or value net on the exact g16r4 (legacy nn/data/combined.jsonl) +
# g24r4 corpora, mixed and size-bucketed.
#
#   sbatch -J spr-m1-policy   self_play_robots/jobs/m1_train.slurm policy     21
#   sbatch -J spr-m1-valwarm  self_play_robots/jobs/m1_train.slurm value_warm 21
#   sbatch -J spr-m1-valcold  self_play_robots/jobs/m1_train.slurm value_cold 21
# arg1 = policy | value_warm | value_cold ; arg2 = torch seed (default 21)
# Recipes: policy cold 25 ep bs 8 (policy_tf.py's recipe, mixed sizes);
# value_warm = init from the labeler of record prod_v1_s11 (FINDINGS 61), lr
# 1e-4, 12 ep, bs 4 x mpg 8 (24x24 memory envelope, FINDINGS 63); value_cold =
# 20 ep, 4 warmup epochs curriculum, lr 3e-4, same batch (the control).
# CollapseStop + ModelCheckpoint(min val_regret) + save_last are wired in
# spr.train. Idempotent: DONE sentinel; RR_RESUME=1 from last.ckpt on resubmit.
# Cost basis: g24r4 pair train+bench 0.54 nh (PROBLEM.md 4.8); expect 2-4 h per
# arm on 1 GPU = 0.25-0.5 nh each.
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
ARM="$1"; SEED="${2:-21}"
DIR="$REPO/runs/spr/m1/${ARM}_s${SEED}"
mkdir -p "$DIR"
echo "SPR M1 TRAIN $ARM seed=$SEED start $(date +%F' '%T) job=$SLURM_JOB_ID"
nvidia-smi -L
if [ -f "$DIR/DONE" ]; then echo "SPR M1 TRAIN SKIP $ARM (DONE)"; exit 0; fi
DATA="--data g16r4=nn/data/combined.jsonl --data g24r4=scaling/data/g24r4/backward.jsonl"
resume=0; ls "$DIR"/lightning_logs/version_*/checkpoints/last.ckpt >/dev/null 2>&1 && resume=1
case "$ARM" in
  policy)     ARGS="--system policy --epochs 25 --batch-size 8" ;;
  value_warm) ARGS="--system value --init $REPO/self_play_robots/assets/labeler_prod_v1_s11.ckpt --epochs 12 --batch-size 4 --max-per-group 8 --lr 1e-4" ;;
  value_cold) ARGS="--system value --epochs 20 --warmup 4 --batch-size 4 --max-per-group 8 --lr 3e-4" ;;
  *) echo "FATAL: arm must be policy|value_warm|value_cold"; exit 1 ;;
esac
t0=$SECONDS
RR_RESUME=$resume python -m spr.train $DATA $ARGS --torch-seed "$SEED" --num-workers 4 \
    --device cuda --out-dir "$DIR"
rc=$?
if [ $rc -ne 0 ]; then echo "SPR M1 TRAIN FAILED $ARM rc=$rc after $((SECONDS-t0))s (resubmit to resume)"; exit $rc; fi
ls "$DIR"/lightning_logs/version_*/checkpoints/epoch=*.ckpt >/dev/null 2>&1 || { echo "SPR M1 TRAIN FAILED $ARM: no epoch ckpt"; exit 1; }
touch "$DIR/DONE"
echo "SPR M1 TRAIN DONE $ARM in $((SECONDS-t0))s $(date +%F' '%T)"
