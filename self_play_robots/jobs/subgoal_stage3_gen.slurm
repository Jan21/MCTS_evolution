#!/bin/bash
# PLAN_SUBGOAL_DISCOVERY.md Stage 3, leg 1: exact cost-to-go labels for the
# children of sampled states, from the project's own engine.
#
#   sbatch -J spr-sg3-gen self_play_robots/jobs/subgoal_stage3_gen.slurm
#
# Idempotent: an existing non-empty .npz is skipped. Pure CPU work -- the
# allocation is GPU-only, so it runs on a GPU job's own cores (2 GPUs = 32
# cores) and the GPUs idle.
#
# Cost basis: measured 89 ms of one core per exact 16x16 four-robot solve
# (rust_datagen forward_instance, full_policy=false, cap 400k). 2400 train
# parents + 300 val parents x 4 robots x ~36 reachable children ~= 390k solves
# = ~34 700 core-seconds = ~18 min on 32 cores. Walltime 55 min (the qgpu_exp
# cap) -> at most 0.25 x 0.92 = 0.23 node-hours, ~0.09 expected.
#SBATCH -A open-37-42
#SBATCH -p qgpu_exp
#SBATCH --gpus 2
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
mkdir -p "$D"
THREADS=${SG3_THREADS:-32}
echo "SPR SG3-GEN start $(date +%F' '%T) job=${SLURM_JOB_ID:-none} threads=$THREADS"

# validation corpus first (cheap), so a walltime kill still leaves it behind
if [ -s "$D/val_b1800-1849.npz" ]; then
  echo "SPR SG3-GEN skip val (exists)"
else
  t0=$SECONDS
  python -m subgoal.stage3 gen --boards 1800-1849 --per-board 3 --parents 2 \
      --walk 3 --seed 12 --threads "$THREADS" \
      --sidecars "$D/boards" --work "$D/work" --out "$D/val_b1800-1849.npz" \
      || echo "SPR SG3-GEN FAILED val"
  echo "SPR SG3-GEN val took $((SECONDS-t0))s"
fi

if [ -s "$D/train_b1000-1699.npz" ]; then
  echo "SPR SG3-GEN skip train (exists)"
else
  t0=$SECONDS
  python -m subgoal.stage3 gen --boards 1000-1699 --per-board 2 --parents 2 \
      --walk 3 --seed 11 --limit 2400 --threads "$THREADS" \
      --sidecars "$D/boards" --work "$D/work" --out "$D/train_b1000-1699.npz" \
      || echo "SPR SG3-GEN FAILED train"
  echo "SPR SG3-GEN train took $((SECONDS-t0))s"
fi

echo "SPR SG3-GEN DONE $(date +%F' '%T)"
