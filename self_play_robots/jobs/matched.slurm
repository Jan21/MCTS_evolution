#!/bin/bash
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:58:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/matched-%x-%j.out
set -uo pipefail
ml purge; ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
O=results/subgoal/ladder_matched; mkdir -p $O
# MATCHED-TIME CONTROL: the network-free sub-goal search given the same wall
# clock per instance the trained model used at this size, on the SAME instances.
# The iteration cap is lifted so time is the only binding limit.
echo "config=$CFG matched time-cap=$TC s per instance, n=$LIM, $(date -Is)"
python -m subgoal.hybrid --config "$CFG" --lean \
  --instances "scaling/data/$CFG/ladder.jsonl" --h relaxed \
  --b0 2000000 --b-sub 200 --top-m 0 --time-cap "$TC" --frontier 150000 \
  --limit "$LIM" --workers 8 \
  --out "$O/${CFG}_tc${TC}.json"
