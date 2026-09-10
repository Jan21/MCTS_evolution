#!/bin/bash
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:58:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/split-%x-%j.out
set -uo pipefail
ml purge; ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
B=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/selfplay/mix_b2mix_iter3
VAL=$B/value/lightning_logs/version_0/checkpoints/epoch=0-step=1573.ckpt
# the same mixed-size records and the same splits that trained the current pair
python -m spr.split_value --variant "$VARIANT" \
  --data g24r4=$B/buffer_g24r4.jsonl --data g32r4=$B/buffer_g32r4.jsonl \
  --splits "$(grep ^g24r4 $B/buffer_g24r4.jsonl.splits.txt)" \
  --splits "$(grep ^g32r4 $B/buffer_g32r4.jsonl.splits.txt)" \
  --steps "${STEPS:-2500}" --eval-groups 400 --baseline "$VAL" \
  --out results/subgoal/split/${VARIANT}.json
