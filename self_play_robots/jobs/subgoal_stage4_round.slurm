#!/bin/bash
# PLAN_SUBGOAL_DISCOVERY.md Stage 4, Parts B and C: ONE self-play round
# (generation + training). The bench450 evaluation is a separate leg
# (subgoal_stage4_eval.slurm) so neither leg can outrun a 55-minute walltime.
#
#   SG4_ARM=b SG4_ROUND=1 SG4_K=<the width leg A1 chose> \
#     sbatch -J spr-sg4-b1 self_play_robots/jobs/subgoal_stage4_round.slurm
#
# arm b = warm-started from the Stage 3 supervised checkpoint;
# arm c = from random initialisation, no exact-engine data at any point.
#
# Every step is skipped when its output exists, and the generation is SHARDED,
# so a walltime kill loses at most one shard and is resumed by resubmitting the
# same script with the same SG4_ARM / SG4_ROUND.
#
# Cost basis: one GPU for at most 55 min = 0.115 node-hours per round; six
# rounds = at most 0.69 node-hours.
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
NROOT="${SG4_NROOT:-1200}"
NVAL="${SG4_NVAL:-240}"
SHARDS="${SG4_SHARDS:-4}"
PROBE="${SG4_PROBE:-2}"
PEXP="${SG4_PEXP:-200}"
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
SEED=$((7000 + 100 * R + $(printf '%d' "'$ARM")))

for sh in $(seq 0 $((SHARDS-1))); do
  f="$D/sp_${ARM}_r${R}_train_s${sh}.npz"
  [ -s "$f" ] && { echo "SPR SG4-ROUND skip train shard $sh"; continue; }
  t0=$SECONDS
  python -m subgoal.stage4 selfplay --arm net --ckpt "$CK" \
      --boards 1000-1799 --per-board 4 --n "$NROOT" --k "$K" \
      --probe "$PROBE" --probe-expansions "$PEXP" \
      --shards "$SHARDS" --shard "$sh" --seed "$SEED" \
      --concurrency "$CONC" --net-batch "$CONC" --out "$f" \
      || { echo "SPR SG4-ROUND FAILED gen-train shard $sh"; exit 1; }
  echo "SPR SG4-ROUND gen-train shard $sh took $((SECONDS-t0))s"
done
VA="$D/sp_${ARM}_r${R}_val.npz"
if [ ! -s "$VA" ]; then
  t0=$SECONDS
  python -m subgoal.stage4 selfplay --arm net --ckpt "$CK" \
      --boards 1850-1899 --per-board 6 --n "$NVAL" --k "$K" \
      --probe "$PROBE" --probe-expansions "$PEXP" \
      --seed "$((9000 + 100 * R + $(printf '%d' "'$ARM")))" \
      --concurrency "$CONC" --net-batch "$CONC" --out "$VA" \
      || { echo "SPR SG4-ROUND FAILED gen-val"; exit 1; }
  echo "SPR SG4-ROUND gen-val took $((SECONDS-t0))s"
fi

BUF=""
for i in $(seq "$R" -1 1); do
  [ $((R - i)) -lt "$BUFFER" ] || break
  for f in "$D"/sp_${ARM}_r${i}_train_s*.npz; do [ -s "$f" ] && BUF="$BUF $f"; done
done
if [ ! -s "$RUN/best.txt" ]; then
  t0=$SECONDS
  python -m subgoal.stage4 train --train $BUF --val "$VA" --run "$RUN" \
      --init "$CK" --epochs "$EPOCHS" --batch 16 --workers 8 \
      --min-labels 2 --monitor val_mae_all --mode min --max-minutes 20 \
      || { echo "SPR SG4-ROUND FAILED train"; exit 1; }
  echo "SPR SG4-ROUND train took $((SECONDS-t0))s (buffer:$BUF)"
fi
echo "SPR SG4-ROUND DONE arm=$ARM round=$R checkpoint $(cat "$RUN/best.txt") $(date +%F' '%T)"
