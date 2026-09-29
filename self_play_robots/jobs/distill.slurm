#!/bin/bash
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:58:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/distill-%x-%j.out
set -uo pipefail
ml purge; ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
B=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/selfplay/mix_b2mix_iter3
POL=$B/policy/lightning_logs/version_0/checkpoints/epoch=2-step=2361.ckpt
G64=$REPO/runs/spr/selfplay/g64r4_b2_iter1
python -m spr.distill_policy --variant "$VARIANT" --teacher "$POL" \
  --data g24r4=$B/buffer_g24r4.jsonl --data g32r4=$B/buffer_g32r4.jsonl --data g64r4=$G64/buffer.jsonl \
  --splits "$(grep ^g24r4 $B/buffer_g24r4.jsonl.splits.txt)" \
  --splits "$(grep ^g32r4 $B/buffer_g32r4.jsonl.splits.txt)" \
  --splits "$(cat $G64/buffer.jsonl.splits.txt)" \
  --extra-train g24r8=$B/buffer_g24r8.jsonl --splits "$(grep ^g24r8 $B/buffer_g24r8.jsonl.splits.txt)" \
  --steps "${STEPS:-25000}" --eval-groups 400 ${EXTRA:-} \
  --out results/subgoal/distill/policy_${VARIANT}${TAG:-}.json
