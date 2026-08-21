#!/bin/bash
# Exact optima for the 200 unseen-exam instances (variants/exam_dstar.py):
# move_planner.oracle.solve at the pinned caps + a 5x-cap second pass.
# CPU work on the GPU allocation (house rule: qgpu-only); 16 cores, 8 workers.
# Projected: <= 2 h * 0.125 = <= 0.25 nh (realistically ~0.1 nh).
#   sbatch -J spr-var-dstar self_play_robots/jobs/variant_exam_dstar.slurm
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/supervised_valuenet"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
export RR_GRID=24 RR_ROBOTS=4 RR_WALLS=108
export RR_ENV_DIR="$REPO/self_play_robots/results/variants/exam/boards_g24r4"
echo "SPR DSTAR start $(date +%F' '%T) job=$SLURM_JOB_ID"
python -m variants.exam_dstar --workers 8 --pass2-mult 5 \
  || { echo "SPR DSTAR FAILED"; exit 1; }
echo "SPR DSTAR DONE $(date +%F' '%T)"
