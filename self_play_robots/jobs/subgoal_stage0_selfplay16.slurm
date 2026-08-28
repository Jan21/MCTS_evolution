#!/bin/bash
# PLAN_SUBGOAL_DISCOVERY.md Stage 0, third row: the self-play line has NEVER
# been benched on the pinned 16x16 exam (every payload under results/selfplay/
# and results/variants/ is g24r4 / g24r8 / g32r4). The nets are size-free, so
# this job runs them unchanged at g16r4 on bench450 under the arena protocol
# the backward supervised row uses (1200 expansions, k=5, B2 vocabulary,
# anytime, no prefix-check -- prefix_key cannot sort B2 plans).
#
#   sbatch -J spr-sg-stage0 self_play_robots/jobs/subgoal_stage0_selfplay16.slurm
#
# Two arms: v14_stack (the planner-of-record's training arm, status.json M5)
# and mix_b2mix_iter3 (the last main-line curriculum iteration).
# Idempotent: an existing non-empty output is skipped.
# Cost basis: g16r4 B2 arena A* is ~5.4 s/instance; 450 instances 8-wide on one
# GPU ~= 6 min per arm -> ~15 min = 0.03 node-hours (1 h cap = 0.125 nh).
#SBATCH -A open-37-42
#SBATCH -p qgpu_exp
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
OUTD="$REPO/self_play_robots/results/subgoal"
mkdir -p "$OUTD"
V14R="$REPO/runs/spr/variants/v14_stack"
MIXR="$REPO/runs/spr/selfplay/mix_b2mix_iter3"
echo "SPR SG-STAGE0 start $(date +%F' '%T) job=$SLURM_JOB_ID"
nvidia-smi -L

bench () {   # <tag> <policy> <value>
  out="$OUTD/$1_g16r4_bench450_astar.json"
  if [ -s "$out" ]; then echo "SPR SG-STAGE0 skip $1 (exists)"; return; fi
  t0=$SECONDS
  python -m spr.arena bench --arm "sg0_$1_g16r4" --config g16r4 \
      --instances eval/data/bench450.jsonl --policy "$2" --value "$3" \
      --driver spr --flags "--search arena_astar --vocab b2 --anytime" \
      --width 8 --threads 2 --device cuda --out "$out" \
      || echo "SPR SG-STAGE0 FAILED $1"
  echo "SPR SG-STAGE0 $1 took $((SECONDS-t0))s"
}

bench v14_stack \
  "$V14R/policy/lightning_logs/version_0/checkpoints/epoch=4-step=1035.ckpt" \
  "$V14R/value/lightning_logs/version_0/checkpoints/epoch=1-step=826.ckpt"
bench mix_b2mix_iter3 \
  "$MIXR/policy/lightning_logs/version_0/checkpoints/epoch=2-step=2361.ckpt" \
  "$MIXR/value/lightning_logs/version_0/checkpoints/epoch=0-step=1573.ckpt"

echo "SPR SG-STAGE0 DONE $(date +%F' '%T)"
