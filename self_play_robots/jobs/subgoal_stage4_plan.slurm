#!/bin/bash
# PLAN_SUBGOAL_DISCOVERY.md Stage 4: run the planner on an instance set.
#
#   SG4_LIST="net:50:tight net:50:weak" SG4_TAG=partA \
#     sbatch -J spr-sg4-plan self_play_robots/jobs/subgoal_stage4_plan.slurm
#
# One space-separated spec per row, arm:k:bound. Idempotent per row.
#
# Cost basis: Stage 3's widest network row was 924 s for all 450 instances at
# concurrency 64; three rows fit a 55 min walltime -> at most 0.115 node-hours.
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
D="$REPO/self_play_robots/results/subgoal/stage4"
CK="${SG4_CKPT:-$(cat "$REPO/runs/spr/subgoal/stage3_rec4/best.txt")}"
INST="${SG4_INST:-$REPO/supervised_valuenet/eval/data/bench450.jsonl}"
EXP="${SG4_EXP:-1200}"
CONC="${SG4_CONC:-128}"
TAG="${SG4_TAG:-partA}"
mkdir -p "$D"
echo "SPR SG4-PLAN start $(date +%F' '%T) job=${SLURM_JOB_ID:-none} tag=$TAG ckpt=$CK"
nvidia-smi -L
for spec in ${SG4_LIST:?set SG4_LIST}; do
  IFS=: read -r arm k bound <<<"$spec"
  out="$D/${TAG}_${arm}_k${k}_${bound}_e${EXP}.json"
  if [ -s "$out" ]; then echo "SPR SG4-PLAN skip $spec (exists)"; continue; fi
  t0=$SECONDS
  python -m subgoal.stage4 plan --arm "$arm" ${CK:+--ckpt "$CK"} \
      --instances "$INST" --k "$k" --expansions "$EXP" --bound "$bound" \
      --concurrency "$CONC" --net-batch "$CONC" \
      --name "${arm}_k${k}_${bound}" --out "$out" \
      || echo "SPR SG4-PLAN FAILED $spec"
  echo "SPR SG4-PLAN $spec took $((SECONDS-t0))s"
done
echo "SPR SG4-PLAN DONE $(date +%F' '%T)"
