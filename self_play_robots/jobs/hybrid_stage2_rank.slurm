#!/bin/bash
#SBATCH --job-name=hyb-s2-rank
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/hyb-s2-rank-%j.out
# PLAN_HYBRID_ACTION_SPACE.md Stage 2, cheapest form: rank the old planner's
# abstract plan cost against the recorded arms on the SAME held-out corpus,
# with no planner search at all. CPU work only.
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" \
       PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
echo "job ${SLURM_JOB_ID} on $(hostname), $(date -Is)"
git -C "$REPO" log -1 --pretty="code at %h %s"
python -m subgoal.rank_struct \
  --data results/subgoal/stage3/val_b1800-1849.npz --vocab b2 \
  --out results/subgoal/stage3/rank_struct_val.json
