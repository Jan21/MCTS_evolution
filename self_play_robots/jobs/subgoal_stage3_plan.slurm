#!/bin/bash
# PLAN_SUBGOAL_DISCOVERY.md Stage 3, leg 3: run the planner on all 450 pinned
# 16x16 four-robot benchmark instances.
#
#   sbatch -J spr-sg3-plan self_play_robots/jobs/subgoal_stage3_plan.slurm
#
# Idempotent per configuration: a configuration whose payload already exists is
# skipped, so a walltime kill is resumed by resubmitting the same script. The
# configurations run in order of importance -- the gate row and its control
# first, the beam-width study after -- so a partial run still answers the gate.
#
# Cost basis: the relaxed control measured 4 s per 20 instances at k=5 on one
# core (login-node smoke), so ~2 min for 450; the network arm spends 4 encoder
# passes per expansion (<= 4800 per instance, <= 2.16 M for the whole bench) at
# batch 64 on one A100. Walltime 55 min -> at most 0.115 node-hours.
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
       PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
D="$REPO/self_play_robots/results/subgoal/stage3"
TAG="${SG3_TAG:-rec4}"
RUN="$REPO/runs/spr/subgoal/stage3_$TAG"
CK="${SG3_CKPT:-$(cat "$RUN/best.txt" 2>/dev/null)}"
EXP="${SG3_EXP:-1200}"
CONC="${SG3_CONC:-64}"
PLAN_LIST="${SG3_LIST:-relaxed:5 net:5 net:20 relaxed:20 net:10 relaxed:10 net:50 relaxed:50 net:200 relaxed:200}"
mkdir -p "$D"
echo "SPR SG3-PLAN start $(date +%F' '%T) job=${SLURM_JOB_ID:-none} ckpt=$CK exp=$EXP"
nvidia-smi -L
for spec in $PLAN_LIST; do
  arm="${spec%%:*}"; k="${spec##*:}"
  out="$D/plan_${arm}_k${k}_e${EXP}.json"
  if [ -s "$out" ]; then echo "SPR SG3-PLAN skip $arm k=$k (exists)"; continue; fi
  if [ "$arm" = "net" ] && [ ! -s "$CK" ]; then
    echo "SPR SG3-PLAN skip $arm k=$k (no checkpoint)"; continue
  fi
  t0=$SECONDS
  python -m subgoal.stage3 plan --arm "$arm" ${CK:+--ckpt "$CK"} \
      --k "$k" --expansions "$EXP" --concurrency "$CONC" --net-batch 64 \
      --name "${arm}_k${k}_e${EXP}" --out "$out" \
      || echo "SPR SG3-PLAN FAILED $arm k=$k"
  echo "SPR SG3-PLAN $arm k=$k took $((SECONDS-t0))s"
done
echo "SPR SG3-PLAN DONE $(date +%F' '%T)"
