#!/bin/bash
# PLAN_SUBGOAL_DISCOVERY.md Stage 4, leg A1: build the held-out dev instance set
# and choose the beam width on it.
#
#   sbatch -J spr-sg4-dev self_play_robots/jobs/subgoal_stage4_dev.slurm
#
# Idempotent: dev200.jsonl and each per-width payload are skipped when they
# already exist, so a walltime kill is resumed by resubmitting the same script.
#
# Cost basis: Stage 3 ran all 450 bench instances at k=200 in 924 s on one A100;
# 200 dev instances at six widths is about 6 x 410 s = 41 min at concurrency 64,
# less at 128. Walltime 55 min -> at most 0.115 node-hours.
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
CONC="${SG4_CONC:-128}"
mkdir -p "$D"
echo "SPR SG4-DEV start $(date +%F' '%T) job=${SLURM_JOB_ID:-none} ckpt=$CK"
nvidia-smi -L
if [ ! -s "$D/dev200.jsonl" ]; then
  python -m subgoal.stage4 devgen --out "$D/dev200.jsonl" || echo "SPR SG4-DEV GEN FAILED"
fi
for k in 5 10 20 50 100 1024; do
  out="$D/dev_net_k${k}_e1200.json"
  if [ -s "$out" ]; then echo "SPR SG4-DEV skip k=$k (exists)"; continue; fi
  t0=$SECONDS
  python -m subgoal.stage4 plan --arm net --ckpt "$CK" \
      --instances "$D/dev200.jsonl" --k "$k" --expansions 1200 --bound tight \
      --concurrency "$CONC" --net-batch "$CONC" \
      --name "dev_net_k${k}" --out "$out" || echo "SPR SG4-DEV FAILED k=$k"
  echo "SPR SG4-DEV k=$k took $((SECONDS-t0))s"
done
python -m subgoal.stage4 beam --dir "$D" --glob 'dev_net_k*_e1200.json' \
    --instances "$D/dev200.jsonl" --json "$D/dev_beam.json" \
    --pick "$D/beam_choice.json"
echo "SPR SG4-DEV DONE $(date +%F' '%T)"
