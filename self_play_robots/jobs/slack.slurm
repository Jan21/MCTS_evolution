#!/bin/bash
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/slack-%x-%j.out
set -uo pipefail
ml purge; ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
O=results/subgoal/repair; mkdir -p $O
echo "TRACK 2 (repair-for-quality): slack=$SLK park_cap=$CAP park_depth=$DEP $(date -Is)"
python -m subgoal.hybrid --config g24r4 --lean \
  --instances scaling/data/g24r4/bench.solved.jsonl --h relaxed \
  --b0 1200 --b-sub 200 --top-m 0 --park-cap "$CAP" --park-depth "$DEP" \
  --slack "$SLK" --time-cap 40 --workers 16 \
  --out "$O/g24r4_slack${SLK}_cap${CAP}_dep${DEP}.json"
