#!/bin/bash
# Reference row for the M5 far-size audit: the labeler of record (prod_v1_s11) on the
# SAME exact sets with the SAME instrument as spr.audit's value column (nn_labeler.audit,
# test split) at 56/64, where only the descent-gate argmin agreement was on file.
#   sbatch -J spr-labeler-audit self_play_robots/jobs/labeler_audit_far.slurm
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/supervised_valuenet"
export PYTHONPATH="$REPO/supervised_valuenet" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
OUT="$REPO/self_play_robots/results/audit/labeler_prod_v1_s11_g56g64.json"
echo "SPR LABELER AUDIT start $(date +%F' '%T) job=$SLURM_JOB_ID"
python -m nn_labeler.audit --ckpt "$REPO/self_play_robots/assets/labeler_prod_v1_s11.ckpt" --split test --device cuda --batch-size 2 \
  --data g56r4=scaling/data/g56r4/backward_audit.rust.jsonl --data g64r4=scaling/data/g64r4/backward_audit.rust.jsonl \
  --out "$OUT" || { echo "SPR LABELER AUDIT FAILED"; exit 1; }
echo "SPR LABELER AUDIT DONE $(date +%F' '%T)"
