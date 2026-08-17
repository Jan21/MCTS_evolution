#!/bin/bash
# M1 -- size-free rebuild (PROBLEM.md 6.2 / 8; owner 2026-08-17: "do both
# [size-free and per-size] at 2 sizes and compare"): train pe=none nets on an
# exact corpus set DS in {mixed, g16, g24}:
#   mixed = g16r4 legacy nn/data/combined.jsonl + g24r4 scaling/data/g24r4/backward.jsonl
#   g16   = g16r4 only     g24 = g24r4 only      (single-size nets, benched at BOTH
#           sizes -> the zero-shot cross-size transfer the size-free bet promises)
# and the arms listed (default: policy value_warm; add value_cold for the
# cold-start control). The per-size comparison arm needs no training: the
# supervised per-size pairs ARE the references (eval.compare rows).
#
#   sbatch -J spr-m1-mixed self_play_robots/jobs/m1_train.slurm mixed 21 "policy value_warm value_cold"
#   sbatch -J spr-m1-g16   self_play_robots/jobs/m1_train.slurm g16   21
#   sbatch -J spr-m1-g24   self_play_robots/jobs/m1_train.slurm g24   21
# Recipes: policy cold 25 ep bs 8 (policy_tf.py's); value_warm = init from the
# labeler of record prod_v1_s11 (FINDINGS 61), lr 1e-4, 12 ep, bs 4 x mpg 8
# (24x24 memory envelope, FINDINGS 63); value_cold = 20 ep, 4 warmup epochs,
# lr 3e-4. CollapseStop + ModelCheckpoint(min val_regret) + save_last are wired
# in spr.train. Idempotent per arm (DONE sentinel; RR_RESUME=1 from last.ckpt).
# Cost basis: g24r4 pair train+bench 0.54 nh (PROBLEM.md 4.8); expect 1-3 h per
# arm on 1 GPU; whole job <= 8 h = <= 1 nh.
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
DS="$1"; SEED="${2:-21}"; ARMS="${3:-policy value_warm}"
case "$DS" in
  mixed) DATA="--data g16r4=nn/data/combined.jsonl --data g24r4=scaling/data/g24r4/backward.jsonl" ;;
  g16)   DATA="--data g16r4=nn/data/combined.jsonl" ;;
  g24)   DATA="--data g24r4=scaling/data/g24r4/backward.jsonl" ;;
  *) echo "FATAL: DS must be mixed|g16|g24"; exit 1 ;;
esac
echo "SPR M1 TRAIN ds=$DS seed=$SEED arms=$ARMS start $(date +%F' '%T) job=$SLURM_JOB_ID"
nvidia-smi -L
rc_all=0
for ARM in $ARMS; do
  DIR="$REPO/runs/spr/m1/${DS}_${ARM}_s${SEED}"
  mkdir -p "$DIR"
  if [ -f "$DIR/DONE" ]; then echo "SPR M1 TRAIN SKIP $DS $ARM (DONE)"; continue; fi
  case "$ARM" in
    policy)     ARGS="--system policy --epochs 25 --batch-size 8" ;;
    value_warm) ARGS="--system value --init $REPO/self_play_robots/assets/labeler_prod_v1_s11.ckpt --epochs 12 --batch-size 4 --max-per-group 8 --lr 1e-4" ;;
    value_cold) ARGS="--system value --epochs 20 --warmup 4 --batch-size 4 --max-per-group 8 --lr 3e-4" ;;
    *) echo "FATAL: arm $ARM"; rc_all=1; continue ;;
  esac
  resume=0; ls "$DIR"/lightning_logs/version_*/checkpoints/last.ckpt >/dev/null 2>&1 && resume=1
  t0=$SECONDS
  RR_RESUME=$resume python -m spr.train $DATA $ARGS --torch-seed "$SEED" --num-workers 4 \
      --device cuda --out-dir "$DIR"
  rc=$?
  if [ $rc -ne 0 ]; then echo "SPR M1 TRAIN FAILED $DS $ARM rc=$rc after $((SECONDS-t0))s (resubmit to resume)"; rc_all=1; continue; fi
  ls "$DIR"/lightning_logs/version_*/checkpoints/epoch=*.ckpt >/dev/null 2>&1 || { echo "SPR M1 TRAIN FAILED $DS $ARM: no epoch ckpt"; rc_all=1; continue; }
  touch "$DIR/DONE"
  echo "SPR M1 TRAIN DONE $DS $ARM in $((SECONDS-t0))s $(date +%F' '%T)"
done
echo "SPR M1 TRAIN JOB DONE ds=$DS rc=$rc_all"
exit $rc_all
