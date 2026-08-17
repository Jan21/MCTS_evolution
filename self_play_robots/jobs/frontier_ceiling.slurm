#!/bin/bash
# Solve ceilings of the base subgoal language on the FRONTIER (beyond-oracle)
# sets -- no d*, so only the solve ceiling and best-plan moves are meaningful.
#   sbatch -J spr-frontceil self_play_robots/jobs/frontier_ceiling.slurm
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
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
for cfg in g24r4 g32r4 g24r8; do
  out=results/ceiling/${cfg}_frontier_base.json; [ -s "$out" ] && continue
  python -m spr.ceiling --config $cfg --instances scaling/data/$cfg/bench.unsolved.jsonl --vocab base --workers 16 --time-cap 180 --out "$out"
done
echo "SPR FRONTIER CEILING DONE $(date +%F' '%T)"
