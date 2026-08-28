#!/bin/bash
# PLAN_SUBGOAL_DISCOVERY.md Stage 4: evaluate checkpoints on all 450 pinned
# benchmark instances at the headline configuration.
#
#   SG4_K=<width> SG4_EVAL="b_r1:/path/best.ckpt c_r0:/path/scratch_init.ckpt" \
#     sbatch -J spr-sg4-eval self_play_robots/jobs/subgoal_stage4_eval.slurm
#
# One "tag:checkpoint" per row, idempotent per row.
# Cost: one GPU for at most 55 min = 0.115 node-hours.
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
K="${SG4_K:?set SG4_K}"
EXP="${SG4_EXP:-1200}"
CONC="${SG4_CONC:-128}"
mkdir -p "$D"
echo "SPR SG4-EVAL start $(date +%F' '%T) job=${SLURM_JOB_ID:-none} k=$K"
nvidia-smi -L
for spec in ${SG4_EVAL:?set SG4_EVAL}; do
  tag="${spec%%:*}"; ck="${spec#*:}"
  out="$D/round_${tag}.json"
  if [ -s "$out" ]; then echo "SPR SG4-EVAL skip $tag (exists)"; continue; fi
  [ -s "$ck" ] || { echo "SPR SG4-EVAL FAILED $tag: no checkpoint $ck"; continue; }
  t0=$SECONDS
  python -m subgoal.stage4 plan --arm net --ckpt "$ck" --k "$K" \
      --expansions "$EXP" --bound tight --concurrency "$CONC" \
      --net-batch "$CONC" --name "$tag" --out "$out" \
      || echo "SPR SG4-EVAL FAILED $tag"
  echo "SPR SG4-EVAL $tag took $((SECONDS-t0))s"
done
echo "SPR SG4-EVAL DONE $(date +%F' '%T)"
