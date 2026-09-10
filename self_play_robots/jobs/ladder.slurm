#!/bin/bash
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/ladder-%x-%j.out
set -uo pipefail
ml purge; ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
O=results/subgoal/ladder; mkdir -p $O
echo "config=$CFG top_m=$TOPM lean $(date -Is)"
python -m subgoal.hybrid --config "$CFG" --lean \
  --instances "scaling/data/$CFG/ladder.jsonl" --h relaxed \
  --b0 400 --b-sub 200 --top-m "$TOPM" --time-cap 20 --frontier 40000 \
  --limit 100 --workers 12 \
  --out "$O/${CFG}_m${TOPM}.json"
