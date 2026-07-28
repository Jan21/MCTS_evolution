#!/bin/bash
# Seed study (publishability objection 2.1): every result in this study is
# single-seed, on both sides, which is the largest remaining statistical
# caveat -- no reported difference smaller than the seed spread is defensible,
# and the spread is unmeasured. This retrains the BACKWARD pair at one
# configuration under varied torch seeds and re-runs its benchmark rows, so
# the spread can be reported as min/median/max.
#
#   sbatch -J rr-seed-g16r6-s21 jobs/patterns/seed_study.slurm g16r6 21 cap20000
# arg1 = config, arg2 = torch seed, arg3 = label set ("" for the default).
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/seed/%x-%j.out
set -uo pipefail
ml purge; ml Python/3.11.5-GCCcore-13.2.0
CFG="$1"; SEED="$2"; LABELSET="${3:-}"
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution; SV="$REPO/supervised_valuenet"
source /scratch/project/open-37-42/petrhyner/venv/bin/activate; cd "$SV"
export PYTHONPATH=. PYTHONUNBUFFERED=1
if [ -n "$LABELSET" ]; then DATA="scaling/data/$CFG/backward_b2.${LABELSET}.rust.jsonl"; SUF="-$LABELSET"
else DATA="scaling/data/$CFG/backward_b2.rust.jsonl"; SUF=""; fi
[ -s "$DATA" ] || { echo "FATAL: missing $DATA"; exit 1; }
case "$CFG" in
  g16r6|g16r8|g24r8) LOWLR="--lr 1e-4" ;;
  *) LOWLR="" ;;
esac
RD="scaling/runs/$CFG/backward-value-b2${SUF}-seed${SEED}"
echo "[seed $CFG/$SEED] data=$DATA"
python3 -m scaling.train --config "$CFG" --system backward-value \
    --run-dir "$RD" $LOWLR --torch-seed "$SEED" \
    -- --data "$DATA" --epochs 30 --warmup 0
RC=$?
[ "$RC" -eq 0 ] || { echo "SEED $CFG $SEED FAILED rc=$RC"; exit 1; }
echo "SEED $CFG $SEED DONE"
