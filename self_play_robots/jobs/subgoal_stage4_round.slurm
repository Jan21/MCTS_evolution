#!/bin/bash
# PLAN_SUBGOAL_DISCOVERY.md Stage 4, Parts B and C: ONE self-play round.
#
#   SG4_ARM=b SG4_ROUND=1 sbatch -J spr-sg4-b1 \
#       self_play_robots/jobs/subgoal_stage4_round.slurm
#
# arm b = warm-started from the Stage 3 supervised checkpoint;
# arm c = from random initialisation, no exact-engine data at any point.
#
# Four steps, each skipped when its output already exists, so a walltime kill is
# resumed by resubmitting the same script with the same SG4_ARM / SG4_ROUND:
#   1  self-play generation on the train boards (certified labels)
#   2  self-play generation on the validation boards (checkpoint selection)
#   3  train h, warm-started, on a rolling buffer of the last SG4_BUFFER rounds
#   4  evaluate on all 450 pinned benchmark instances at the headline config
#
# Cost basis: measured in leg A1. Walltime 55 min -> at most 0.115 node-hours
# per round; six rounds (three per arm) -> at most 0.69 node-hours.
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
ARM="${SG4_ARM:?set SG4_ARM=b|c}"
R="${SG4_ROUND:?set SG4_ROUND}"
K="${SG4_K:?set SG4_K to the beam width leg A1 chose}"
NROOT="${SG4_NROOT:-1500}"
NVAL="${SG4_NVAL:-250}"
BUFFER="${SG4_BUFFER:-2}"
EPOCHS="${SG4_EPOCHS:-30}"
CONC="${SG4_CONC:-128}"
PREV=$((R-1))
if [ "$R" = "1" ]; then
  if [ "$ARM" = "b" ]; then CK="$(cat "$REPO/runs/spr/subgoal/stage3_rec4/best.txt")"
  else CK="$D/scratch_init.ckpt"; fi
else
  CK="$(cat "$REPO/runs/spr/subgoal/stage4_${ARM}_r${PREV}/best.txt")"
fi
RUN="$REPO/runs/spr/subgoal/stage4_${ARM}_r${R}"
mkdir -p "$D" "$RUN"
echo "SPR SG4-ROUND start $(date +%F' '%T) job=${SLURM_JOB_ID:-none} arm=$ARM round=$R k=$K ckpt=$CK"
nvidia-smi -L
[ -s "$CK" ] || { echo "SPR SG4-ROUND FAILED: no checkpoint $CK"; exit 1; }

TR="$D/sp_${ARM}_r${R}_train.npz"
VA="$D/sp_${ARM}_r${R}_val.npz"
if [ ! -s "$TR" ]; then
  t0=$SECONDS
  python -m subgoal.stage4 selfplay --arm net --ckpt "$CK" \
      --boards 1000-1799 --per-board 4 --n "$NROOT" --k "$K" \
      --seed "$((7000 + 100 * R + $(printf '%d' "'$ARM")))" \
      --concurrency "$CONC" --net-batch "$CONC" --out "$TR" \
      || { echo "SPR SG4-ROUND FAILED gen-train"; exit 1; }
  echo "SPR SG4-ROUND gen-train took $((SECONDS-t0))s"
fi
if [ ! -s "$VA" ]; then
  t0=$SECONDS
  python -m subgoal.stage4 selfplay --arm net --ckpt "$CK" \
      --boards 1850-1899 --per-board 6 --n "$NVAL" --k "$K" \
      --seed "$((9000 + 100 * R + $(printf '%d' "'$ARM")))" \
      --concurrency "$CONC" --net-batch "$CONC" --out "$VA" \
      || { echo "SPR SG4-ROUND FAILED gen-val"; exit 1; }
  echo "SPR SG4-ROUND gen-val took $((SECONDS-t0))s"
fi

BUF=""
for i in $(seq "$R" -1 1); do
  [ $((R - i)) -lt "$BUFFER" ] || break
  f="$D/sp_${ARM}_r${i}_train.npz"
  [ -s "$f" ] && BUF="$BUF $f"
done
if [ ! -s "$RUN/best.txt" ]; then
  t0=$SECONDS
  python -m subgoal.stage4 train --train $BUF --val "$VA" --run "$RUN" \
      --init "$CK" --epochs "$EPOCHS" --batch 16 --workers 8 \
      --min-labels 2 --monitor val_mae_all --mode min --max-minutes 20 \
      || { echo "SPR SG4-ROUND FAILED train"; exit 1; }
  echo "SPR SG4-ROUND train took $((SECONDS-t0))s (buffer:$BUF)"
fi

OUT="$D/round_${ARM}_r${R}.json"
if [ ! -s "$OUT" ]; then
  t0=$SECONDS
  python -m subgoal.stage4 plan --arm net --ckpt "$(cat "$RUN/best.txt")" \
      --k "$K" --expansions 1200 --bound tight --concurrency "$CONC" \
      --net-batch "$CONC" --name "${ARM}_r${R}" --out "$OUT" \
      || { echo "SPR SG4-ROUND FAILED bench"; exit 1; }
  echo "SPR SG4-ROUND bench took $((SECONDS-t0))s"
fi
echo "SPR SG4-ROUND DONE arm=$ARM round=$R $(date +%F' '%T)"
