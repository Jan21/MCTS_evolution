#!/bin/bash
#SBATCH --job-name=hyb-s2-rank
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/hyb-s2-rank-%j.out
# PLAN_HYBRID_ACTION_SPACE.md Stage 2, cheapest form. CPU only; 8 shards split
# by STATE index so a pooled decision never spans a shard.
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" \
       PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
OUT=results/subgoal/stage3/rank_struct
mkdir -p "$OUT"
echo "job ${SLURM_JOB_ID} on $(hostname), $(date -Is)"
git -C "$REPO" log -1 --pretty="code at %h %s"
for j in 0 1 2 3 4 5 6 7; do
  python -m subgoal.rank_struct --data results/subgoal/stage3/val_b1800-1849.npz \
    --vocab b2 --shards 8 --shard "$j" --out "$OUT/shard_${j}.json" \
    > "$OUT/shard_${j}.log" 2>&1 &
done
wait
echo "shards done $(date -Is)"
python -m subgoal.rank_struct merge $OUT/shard_*.json "$OUT/merged.json"
