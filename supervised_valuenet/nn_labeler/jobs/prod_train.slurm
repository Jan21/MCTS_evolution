#!/bin/bash
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH -t 16:00:00
#SBATCH -J rr-nnlab-prod
#SBATCH -o /scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/nnlab/prod_train_%j.out
#
# NN-labeler PRODUCTION net: pe=none size-free value net on every fresh exact
# base-vocab corpus <= 16x16 (grids 8-15 at r4, plus 16x16 at r6 and r8 for
# grid-16 coverage and robot diversity; the legacy 16x16 corpus is excluded --
# FINDINGS 53 measured it as a distribution artifact). Two-phase curriculum,
# 96 bins, 30 epochs, resume-enabled (RR_RESUME on resubmit).
#
#   sbatch supervised_valuenet/nn_labeler/jobs/prod_train.slurm <seed>
set -u
SEED=${1:?seed required}
module purge
ml Python/3.11.5-GCCcore-13.2.0
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
cd "$REPO/supervised_valuenet"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
nvidia-smi -L

DIR="nn_labeler/runs/prod_mix8to16_none_s${SEED}"
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
    --warmup 4 --torch-seed "$SEED" --device cuda --out-dir "$DIR" \
    || { echo "PROD TRAIN FAILED seed $SEED"; exit 1; }

BEST=$(ls -t "$DIR"/lightning_logs/version_*/checkpoints/epoch=*.ckpt | head -1)
python -m nn_labeler.audit --ckpt "$BEST" --split test --device cuda \
    --data g8r4=scaling/data/g8r4/backward.rust.jsonl \
    --data g12r4=scaling/data/g12r4/backward.rust.jsonl \
    --data g16r6=scaling/data/g16r6/backward.jsonl \
    --data g24r4=scaling/data/g24r4/backward.jsonl \
    --data g32r4=scaling/data/g32r4/backward.jsonl \
    --out "nn_labeler/results/audit_prod_s${SEED}.json" \
    || echo "PROD AUDIT FAILED seed $SEED"
echo "NNLAB PROD DONE seed $SEED"
