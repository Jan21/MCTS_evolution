#!/bin/bash
# PLAN_SUBGOAL_DISCOVERY.md Stage 3, leg 2: train the cost-to-go heuristic.
#
#   sbatch -J spr-sg3-train self_play_robots/jobs/subgoal_stage3_train.slurm
#
# Idempotent: an existing best.txt is skipped. Recurrence 4 is the
# pre-registered depth (STAGE2.md section 4: 12 weight-tied steps collapse this
# net family onto the constant-value plateau).
#
# Cost basis: Stage 2's equivalent net (1.02 M params, 11 200 groups, batch 16,
# fp32) trained 30 epochs in 12.5 min on one A100 (job 4863454). Stage 3 has
# ~9 600 groups and a slightly wider readout, so 40 epochs fit the same order.
# Walltime 55 min -> at most 0.115 node-hours.
#SBATCH -A open-37-42
#SBATCH -p qgpu_exp
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" \
       PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
D="$REPO/self_play_robots/results/subgoal/stage3"
REC="${SG3_REC:-4}"
EPOCHS="${SG3_EPOCHS:-40}"
TAG="${SG3_TAG:-rec$REC}"
RUN="$REPO/runs/spr/subgoal/stage3_$TAG"
mkdir -p "$RUN"
echo "SPR SG3-TRAIN start $(date +%F' '%T) job=${SLURM_JOB_ID:-none} rec=$REC epochs=$EPOCHS"
nvidia-smi -L
if [ -s "$RUN/best.txt" ]; then
  echo "SPR SG3-TRAIN skip (exists)"
else
  t0=$SECONDS
  python -m subgoal.stage3 train \
      --train "$D/train_b1000-1699.npz" --val "$D/val_b1800-1849.npz" \
      --run "$RUN" --epochs "$EPOCHS" --batch 16 --workers 8 \
      --max-minutes 42 --recurrence "$REC" --pe none \
      || echo "SPR SG3-TRAIN FAILED"
  echo "SPR SG3-TRAIN took $((SECONDS-t0))s"
fi
echo "SPR SG3-TRAIN checkpoint $(cat "$RUN/best.txt" 2>/dev/null)"
echo "SPR SG3-TRAIN DONE $(date +%F' '%T)"
