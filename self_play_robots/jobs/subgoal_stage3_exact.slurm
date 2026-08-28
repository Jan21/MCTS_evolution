#!/bin/bash
# PLAN_SUBGOAL_DISCOVERY.md Stage 3, diagnostic leg: the same search with the
# TRUE cost-to-go as h. Not a system -- it costs one exact engine solve per
# candidate child -- but it separates the two ways Stage 3 can fail: if the
# search finds the optimum when h is exact, the search is sound and only the
# heuristic is at fault.
#
#   sbatch -J spr-sg3-exact self_play_robots/jobs/subgoal_stage3_exact.slurm
#
# Cost basis: 89 ms of one core per exact solve; with an exact h every optimal
# child carries the minimal f, so the search terminates in a handful of
# expansions (cap 200). ~450 x ~5 x ~148 = ~330k solves = ~29 000 core-seconds
# = ~19 min on 16 cores. Walltime 55 min -> at most 0.115 node-hours.
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
       PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
D="$REPO/self_play_robots/results/subgoal/stage3"
EXP="${SG3_EXP:-200}"
K="${SG3_K:-5}"
out="$D/plan_exact_k${K}_e${EXP}.json"
mkdir -p "$D"
echo "SPR SG3-EXACT start $(date +%F' '%T) job=${SLURM_JOB_ID:-none}"
if [ -s "$out" ]; then
  echo "SPR SG3-EXACT skip (exists)"
else
  t0=$SECONDS
  python -m subgoal.stage3 plan --arm exact --k "$K" --expansions "$EXP" \
      --concurrency 64 --threads 16 --sidecars "$D/boards" \
      --name "exact_k${K}_e${EXP}" --out "$out" \
      || echo "SPR SG3-EXACT FAILED"
  echo "SPR SG3-EXACT took $((SECONDS-t0))s"
fi
echo "SPR SG3-EXACT DONE $(date +%F' '%T)"
