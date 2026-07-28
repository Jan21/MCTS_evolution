#!/bin/bash
# One config's B2 backward retrain: proposal net COLD first (stable trainer),
# then value net WARM-STARTED from the banked predecessor (cold value
# retrains are proven seed-unstable). Usage:
#   sbatch -J rr-b2rt-g16r6 jobs/patterns/b2_retrain_one.slurm g16r6
# arg1 = config (g16r4 | g16r6 | g16r8 | g24r8 | g32r4).
# Data: scaling/data/<cfg>/backward_b2.rust.jsonl (from b2_labels_one).
# Runs land in scaling/runs/<cfg>/backward-{policy,value}-b2/ (vocabulary-
# named, per house rule). Low-lr (1e-4) applies at >=6 robots per the
# launcher-pattern precedent. Banking of best ckpts happens AFTER manual
# inspection of val metrics, not here.
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
# WALLTIME 16 h, not 24: the cooling reservation (MOTD, daily 10:00-18:00 CEST)
# blocks any job requesting more than 16 hours from starting while it is in
# effect. Every lane here is chunked/idempotent, so a 16 h cap costs at most
# one chunk and a resubmit.
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/b2retrain/%x-%j.out

set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

CFG="$1"
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/supervised_valuenet"
export PYTHONPATH=. PYTHONUNBUFFERED=1
# arg2 optionally overrides the label set (e.g. cap20000) and suffixes the run
# dirs, so a config can be retrained on two label sets without either run
# overwriting the other. FINDINGS 32/33: the production cap-5000 labels carry
# ~5% by-reference candidates against ~14% at the original budget, and the
# first Track 1 row showed the cap-5000 retrain LOSING 24.6 points at the
# g16r6 frontier -- so which label set is used is a live experimental variable,
# not a detail.
LABELSET="${2:-}"
if [ -n "$LABELSET" ]; then
  DATA="scaling/data/$CFG/backward_b2.${LABELSET}.rust.jsonl"
  SUFFIX="-$LABELSET"
else
  DATA="scaling/data/$CFG/backward_b2.rust.jsonl"
  SUFFIX=""
fi
[ -s "$DATA" ] || { echo "missing $DATA"; exit 1; }
echo "[b2rt $CFG] label set: $DATA"

# Per-config knobs. Value warm-start sources are the banked predecessors
# (prompt.md Track 0 step 4); batch caps per the 40 GB A100 tables; low-lr
# 1e-4 at >=6 robots; g32r4 value nets use 96 bins.
LOWLR=""; VBATCH=""; PBATCH=""; VBINS=""
PEPOCHS=30; VEPOCHS=30
case "$CFG" in
  g16r4)
    WARM="checkpoints_backward/value_b1.ckpt"
    PEPOCHS=25; VEPOCHS=20 ;;                       # the b1_retrain_chain recipe
  g16r6)
    WARM="scaling/runs/g16r6/backward-value/lightning_logs/version_1/checkpoints/epoch=29-step=29700.ckpt"
    LOWLR="--lr 1e-4" ;;
  g16r8)
    WARM="scaling/runs/g16r8/backward-value/lightning_logs/version_0/checkpoints/epoch=27-step=25676.ckpt"
    LOWLR="--lr 1e-4" ;;
  g24r8)
    WARM="scaling/runs/g24r8/backward-value/lightning_logs/version_0/checkpoints/epoch=10-step=21681.ckpt"
    LOWLR="--lr 1e-4"
    VBATCH="--batch-size 4 --max-per-group 16"; PBATCH="--batch-size 8" ;;
  g32r4)
    WARM="scaling/runs/g32r4/backward-value/lightning_logs/version_1/checkpoints/epoch=15-step=52528.ckpt"
    # NO --num-classes 96. The handoff recipe asked for 96 bins here, but every
    # available warm-start is 50-class, so the two are incompatible: the
    # 2026-07-27 run died with "size mismatch for head.4.bias: 50 vs 96" and
    # left this config with a policy net and no value net. Measured from the
    # actual labels, 50 bins truncate 64 of 176,593 records (0.036%, max cost
    # 71), while going cold to get 96 bins would forfeit the warm-start that
    # guards against the documented cold-value seed instability. Keeping the
    # warm-start and accepting 0.036% truncation is the better trade.
    VBINS=""
    VBATCH="--batch-size 2 --max-per-group 8"; PBATCH="--batch-size 16" ;;
  *) echo "unknown config $CFG"; exit 1 ;;
esac

echo "[b2rt $CFG] proposal net, cold"
python3 -m scaling.train --config "$CFG" --system backward-policy \
    --run-dir "scaling/runs/$CFG/backward-policy-b2$SUFFIX" $LOWLR \
    -- --data "$DATA" --epochs $PEPOCHS $PBATCH
PRC=$?
echo "[b2rt $CFG] policy done ($PRC)"

echo "[b2rt $CFG] value net, warm-started from $WARM"
python3 -m scaling.train --config "$CFG" --system backward-value \
    --run-dir "scaling/runs/$CFG/backward-value-b2$SUFFIX" $LOWLR $VBINS \
    --warm-start "$WARM" --torch-seed 11 \
    -- --data "$DATA" --epochs $VEPOCHS --warmup 0 $VBATCH
VRC=$?
echo "[b2rt $CFG] value done ($VRC)"
if [ "$PRC" -ne 0 ] || [ "$VRC" -ne 0 ]; then
  # Do NOT print the grep-able DONE marker over a failed stage. The 2026-07-27
  # g32r4 run exited 1 on the value net and still announced DONE, so the
  # failure was only noticed when a downstream lane could not find the net.
  echo "B2RETRAIN $CFG FAILED policy_rc=$PRC value_rc=$VRC"
  exit 1
fi
echo "B2RETRAIN $CFG DONE"
