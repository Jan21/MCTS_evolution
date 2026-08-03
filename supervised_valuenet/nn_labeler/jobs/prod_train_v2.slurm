#!/bin/bash
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH -t 16:00:00
#SBATCH -J rr-nnlab-prod2
#SBATCH -o /scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/nnlab/prod_train_v2_%j.out
#
# NN-labeler production net, attempt 2. Attempt 1 (job 4609800) trained two
# healthy epochs (val_regret 0.267, spread 2.12) and then fell into the
# constant-value mode at epoch 2 and stayed there for 24 epochs (FINDINGS 56).
# Its best checkpoint is healthy and banked (nn_labeler/banked/prod_v1_s11.ckpt)
# -- this run tries to do better, with the study's own documented rescue for
# unstable training (a lower learning rate, PRIMER 6 / FINDINGS 43) plus the
# new CollapseStop callback so a dead run ends in minutes instead of hours.
#
# Also completes the v1 audit: attempt 1's walltime killed it before 32x32.
#
#   sbatch supervised_valuenet/nn_labeler/jobs/prod_train_v2.slurm <seed> [lr]
set -u
SEED=${1:?seed required}
LR=${2:-1e-4}
module purge
ml Python/3.11.5-GCCcore-13.2.0
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
cd "$REPO/supervised_valuenet"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
nvidia-smi -L

AUDIT_SET="--data g8r4=scaling/data/g8r4/backward.rust.jsonl \
 --data g12r4=scaling/data/g12r4/backward.rust.jsonl \
 --data g16r6=scaling/data/g16r6/backward.jsonl \
 --data g24r4=scaling/data/g24r4/backward.jsonl \
 --data g32r4=scaling/data/g32r4/backward.jsonl"

# 0. finish attempt 1's audit (the 32x32 leg it never reached)
if [ ! -f nn_labeler/results/audit_prod_v1_s11_full.json ]; then
    python -m nn_labeler.audit --ckpt nn_labeler/banked/prod_v1_s11.ckpt \
        --split test --device cuda $AUDIT_SET \
        --out nn_labeler/results/audit_prod_v1_s11_full.json \
        || echo "PROD2 V1 AUDIT FAILED"
fi

# 1. attempt 2
DIR="nn_labeler/runs/prod2_mix8to16_none_s${SEED}_lr${LR}"
resume=""
ls "$DIR"/lightning_logs/version_*/checkpoints/last.ckpt >/dev/null 2>&1 && resume=1
RR_RESUME=${resume:-0} python -m nn_labeler.train \
    --data g8r4=scaling/data/g8r4/backward.rust.jsonl \
    --data g9r4=scaling/data/g9r4/backward.rust.jsonl \
    --data g10r4=scaling/data/g10r4/backward.rust.jsonl \
    --data g11r4=scaling/data/g11r4/backward.rust.jsonl \
    --data g12r4=scaling/data/g12r4/backward.rust.jsonl \
    --data g13r4=scaling/data/g13r4/backward.rust.jsonl \
    --data g14r4=scaling/data/g14r4/backward.rust.jsonl \
    --data g15r4=scaling/data/g15r4/backward.rust.jsonl \
    --data g16r6=scaling/data/g16r6/backward.jsonl \
    --data g16r8=scaling/data/g16r8/backward.jsonl \
    --pe none --num-classes 96 --epochs 30 --batch-size 8 --max-per-group 32 \
    --warmup 4 --lr "$LR" --torch-seed "$SEED" --device cuda --out-dir "$DIR" \
    || { echo "PROD2 TRAIN FAILED seed $SEED lr $LR"; exit 1; }

BEST=$(ls -t "$DIR"/lightning_logs/version_*/checkpoints/epoch=*.ckpt | head -1)
[ -n "$BEST" ] || { echo "PROD2 NO CKPT"; exit 1; }
python -m nn_labeler.audit --ckpt "$BEST" --split test --device cuda $AUDIT_SET \
    --out "nn_labeler/results/audit_prod2_s${SEED}_lr${LR}.json" \
    || echo "PROD2 AUDIT FAILED"
echo "NNLAB PROD2 DONE seed $SEED lr $LR ckpt $BEST"
