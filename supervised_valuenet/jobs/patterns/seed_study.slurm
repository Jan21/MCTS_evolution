#!/bin/bash
# Seed study (publishability objection 2.1): every result in this study is
# single-seed, on both sides, which is the largest remaining statistical
# caveat -- no reported difference smaller than the seed spread is defensible,
# and the spread is unmeasured.
#
# REWRITTEN 2026-07-28 (successor session). The previous version was
# unusable as a seed probe of the production system: it trained ONLY the value
# net, COLD, while the production recipe (b2_retrain_one.slurm) trains the
# pair -- policy cold + value WARM-STARTED from the banked predecessor
# ("cold value retrains are proven seed-unstable", per that script). A spread
# measured on a cold-value recipe characterises a recipe the study does not
# use, and conflates the documented cold-value instability with honest seed
# noise. This version repeats b2_retrain_one.slurm's recipe EXACTLY, varying
# only --torch-seed on both trainings (production: policy unseeded, value
# seed 11).
#
#   sbatch -J rr-seed-g16r6-s21 jobs/patterns/seed_study.slurm g16r6 21 cap20000
# arg1 = config (g16r6 only for now -- warm-start table below), arg2 = torch
# seed, arg3 = label set ("" for the cap-5000 default; use cap20000).
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/seed/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

CFG="$1"; SEED="$2"; LABELSET="${3:-}"
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/supervised_valuenet"
export PYTHONPATH=. PYTHONUNBUFFERED=1

if [ -n "$LABELSET" ]; then
  DATA="scaling/data/$CFG/backward_b2.${LABELSET}.rust.jsonl"; SUF="-$LABELSET"
else
  DATA="scaling/data/$CFG/backward_b2.rust.jsonl"; SUF=""
fi
[ -s "$DATA" ] || { echo "FATAL: missing $DATA"; exit 1; }

# Mirror of b2_retrain_one.slurm's per-config knobs. Only g16r6 is wired up;
# add other configs by copying their case arm from that script VERBATIM.
case "$CFG" in
  g16r6)
    WARM="scaling/runs/g16r6/backward-value/lightning_logs/version_1/checkpoints/epoch=29-step=29700.ckpt"
    LOWLR="--lr 1e-4"; PEPOCHS=30; VEPOCHS=30; PBATCH=""; VBATCH="" ;;
  *) echo "FATAL: seed study wired for g16r6 only (see header)"; exit 1 ;;
esac

PDIR="scaling/runs/$CFG/backward-policy-b2${SUF}-seed${SEED}"
VDIR="scaling/runs/$CFG/backward-value-b2${SUF}-seed${SEED}"
echo "[seed $CFG/$SEED] data=$DATA warm=$WARM"

echo "[seed $CFG/$SEED] proposal net, cold, torch-seed $SEED"
python3 -m scaling.train --config "$CFG" --system backward-policy \
    --run-dir "$PDIR" $LOWLR --torch-seed "$SEED" \
    -- --data "$DATA" --epochs $PEPOCHS $PBATCH
PRC=$?
echo "[seed $CFG/$SEED] policy done ($PRC)"

echo "[seed $CFG/$SEED] value net, warm-started, torch-seed $SEED"
python3 -m scaling.train --config "$CFG" --system backward-value \
    --run-dir "$VDIR" $LOWLR \
    --warm-start "$WARM" --torch-seed "$SEED" \
    -- --data "$DATA" --epochs $VEPOCHS --warmup 0 $VBATCH
VRC=$?
echo "[seed $CFG/$SEED] value done ($VRC)"

# Completion marker gated on both stages (house rule, FINDINGS 38).
if [ "$PRC" -ne 0 ] || [ "$VRC" -ne 0 ]; then
  echo "SEED $CFG $SEED FAILED policy_rc=$PRC value_rc=$VRC"
  exit 1
fi
echo "SEED $CFG $SEED DONE"
