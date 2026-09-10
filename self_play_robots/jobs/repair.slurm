#!/bin/bash
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/repair-%x-%j.out
set -uo pipefail
ml purge; ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
O=results/subgoal/repair; mkdir -p $O
echo "TRACK 1: repair generalization. cap=$CAP depth=$DEP prep=$TOPM $(date -Is)"
# pure sub-goal search (TOPM=0) unless stated: this is the paper's system, and
# the only thing changing is how far a blocking robot may step aside.
python -m subgoal.hybrid --config g24r4 \
  --instances scaling/data/g24r4/bench.solved.jsonl --h relaxed \
  --b0 400 --b-sub 200 --top-m "$TOPM" --park-cap "$CAP" --park-depth "$DEP" \
  --time-cap 30 --workers 16 \
  --out "$O/g24r4_cap${CAP}_dep${DEP}_m${TOPM}.json"
