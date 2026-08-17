#!/bin/bash
# M0 -- arena parity (PROBLEM.md section 8): re-run the recorded supervised
# backward rows through THIS folder's harness (spr.arena -> eval.compare
# chunked 8-wide, merged, replay-certified) and check per-row parity against
# the recorded comparison JSONs.
#
#   sbatch -J spr-m0 self_play_robots/jobs/m0_arena_parity.slurm [arm ...]
# default arms: g16r4_b1s21_b2flags g24r4_exact_prefix
# Cost basis: base450 ~3 min 8-wide (seed_headline_pair.slurm), g24r4 232
# instances at ~8-30 s each 8-wide -> ~10-15 min. One GPU node-share for
# <= 1 h = <= 0.125 nh. Idempotent/resumable (skip-if-exists per chunk).
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
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1
mkdir -p "$REPO/runs/spr"
ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || ARMS=(g16r4_b1s21_b2flags g24r4_exact_prefix)
echo "SPR M0 start $(date +%F' '%T) job=$SLURM_JOB_ID arms=${ARMS[*]}"
rc_all=0
for arm in "${ARMS[@]}"; do
  t0=$SECONDS
  python -m spr.arena bench --arm "$arm" --width 8 --threads 2 || { echo "SPR M0 BENCH FAILED $arm"; rc_all=1; continue; }
  echo "SPR M0 bench $arm took $((SECONDS-t0))s"
  python -m spr.arena parity --arm "$arm" || rc_all=1
done
echo "SPR M0 DONE rc=$rc_all $(date +%F' '%T)"
exit $rc_all
