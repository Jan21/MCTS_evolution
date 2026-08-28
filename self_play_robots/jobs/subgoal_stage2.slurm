#!/bin/bash
# PLAN_SUBGOAL_DISCOVERY.md Stage 2: can a network learn goal-conditioned cost?
#
#   sbatch -J spr-sg-stage2 self_play_robots/jobs/subgoal_stage2.slurm
#
# One leg, four idempotent steps (an existing non-empty output is skipped):
#   1 optset  the optimal-preserving subgoal SETS of the 40 Stage 1 instances
#             (CPU, 16 cores -- the GPU allocation's own cores, so no separate job)
#   2 train   the goal-conditioned cost net (subgoal/costnet.py) on states from
#             TRAIN-split boards 1000-1699 only
#   3 eval    ranking metrics + the two learning-free baselines on two held-out
#             board sets: the 450 bench450 ROOT states (boards 2400-2549, the
#             exact inputs a Stage 3 planner sees) and random states on boards
#             1900-1999
#   4 beam    would a top-k beam over each ranking keep an optimal subgoal alive
#
# The .npz corpora are produced beforehand on the login node (seconds of CPU):
#   python -m subgoal.stage2 gen --boards 1000-1699 --per-board 4 --walk 3 --seed 11 --out ...
# Cost basis: fp32, 1.02 M params, 11 200 train groups (= 2.9 M labelled
# candidates) at batch 16 -> 700 steps/epoch = 59 s/epoch measured (job 4863328);
# 30 epochs therefore train
# in ~30 min on one A100; with optset (~5 min) and eval/beam (~3 min) the leg
# fits the 1 h qgpu_exp cap = 0.115 node-hours worst case, ~0.09 expected.
# (qgpu itself is unusable today: 63 of 72 GPU nodes are in maintenance and
# sinfo/squeue put a 2 h qgpu job's start at 2026-08-30; qgpu_exp still runs
# on the two idle nodes immediately.)
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
D="$REPO/self_play_robots/results/subgoal/stage2"
# Arm knobs. The first submission ran the pre-registered production recipe
# (TAG=cost, REC=12, PE=none) and COLLAPSED onto the constant-value plateau
# (job 4863358: val_spearman ~ 0 for 30 epochs, reachability head stuck at the
# base rate). Job 4863393 isolated the cause -- 4 epochs at recurrence 4 reach
# rho 0.58 with either positional signal, 12 steps reach 0.00/0.06 -- so the
# second submission is TAG=rec4, REC=4, same everything else.
TAG="${S2_TAG:-cost}"
REC="${S2_REC:-12}"
PE="${S2_PE:-none}"
EPOCHS="${S2_EPOCHS:-30}"
RUN="$REPO/runs/spr/subgoal/stage2_$TAG"
mkdir -p "$D" "$RUN"
echo "SPR SG-STAGE2 start $(date +%F' '%T) job=${SLURM_JOB_ID:-none} tag=$TAG rec=$REC pe=$PE epochs=$EPOCHS"
nvidia-smi -L

# --- 1. optimal-preserving subgoal sets (CPU) -------------------------------
if [ -s "$D/optsets.json" ]; then
  echo "SPR SG-STAGE2 skip optset (exists)"
else
  t0=$SECONDS
  python -m subgoal.optset --n 20 --stride 23 --workers 16 \
      --max-pops 600000 --time-cap 300 --out "$D/optsets.json" \
      || echo "SPR SG-STAGE2 FAILED optset"
  echo "SPR SG-STAGE2 optset took $((SECONDS-t0))s"
fi

# --- 2. train ---------------------------------------------------------------
if [ -s "$RUN/best.txt" ]; then
  echo "SPR SG-STAGE2 skip train (exists)"
else
  t0=$SECONDS
  python -m subgoal.stage2 train \
      --train "$D/train_b1000-1699.npz" --val "$D/val_b1800-1849.npz" \
      --run "$RUN" --epochs "$EPOCHS" --batch 16 --workers 8 --max-minutes 35 \
      --recurrence "$REC" --pe "$PE" \
      || echo "SPR SG-STAGE2 FAILED train"
  echo "SPR SG-STAGE2 train took $((SECONDS-t0))s"
fi
CK=$(cat "$RUN/best.txt" 2>/dev/null)
echo "SPR SG-STAGE2 checkpoint $CK"
[ -s "$CK" ] || { echo "SPR SG-STAGE2 DONE (no checkpoint)"; exit 1; }

# --- 3. held-out evaluation --------------------------------------------------
ev () {   # <tag> <npz>
  out="$D/eval_${TAG}_$1.json"
  if [ -s "$out" ]; then echo "SPR SG-STAGE2 skip eval $1 (exists)"; return; fi
  t0=$SECONDS
  python -m subgoal.stage2 eval --ckpt "$CK" --data "$2" --batch 8 --out "$out" \
      || echo "SPR SG-STAGE2 FAILED eval $1"
  echo "SPR SG-STAGE2 eval $1 took $((SECONDS-t0))s"
}
ev bench450_roots "$D/test_bench450_roots.npz"
ev b1900-1999     "$D/test_b1900-1999.npz"

# --- 4. beam survival --------------------------------------------------------
if [ -s "$D/beam_$TAG.json" ]; then
  echo "SPR SG-STAGE2 skip beam (exists)"
else
  t0=$SECONDS
  python -m subgoal.stage2 beam --ckpt "$CK" --optsets "$D/optsets.json" \
      --out "$D/beam_$TAG.json" || echo "SPR SG-STAGE2 FAILED beam"
  echo "SPR SG-STAGE2 beam took $((SECONDS-t0))s"
fi

echo "SPR SG-STAGE2 DONE $(date +%F' '%T)"
