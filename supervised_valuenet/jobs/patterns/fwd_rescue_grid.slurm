#!/bin/bash
# A FAIR RESCUE FOR THE FORWARD PLANNER -- training grid (design memo:
# scratchpad ideas/impl_seeds_forward_plan.md; devils-advocate memo point 2:
# "forward got one lr applied by pattern after observed collapse, one seed").
#
# One array task = one (torch seed, learning rate) cell of a 3 x 3 grid at
# one config, trained with the forward control's recipe of record and NOTHING
# else changed:
#   g16r8: jobs/launcher_patterns/train_fwd_lowlr_g16r8.py = 8 epochs, bs 128,
#          num-workers 8, lr 1e-4 (the rescue), seed 7   -> lightning_logs/version_44
#   g24r4: jobs/launcher_patterns/g24r4_chain.sh = stock lr 3e-4, bs 64,
#          num-workers 8, "--epochs 15" requested but the run of record
#          (scaling/runs/g24r4/forward/version_0) stopped after epoch 7 (best =
#          last = epoch 7, ~14 h) -- so 8 epochs mirrors what was actually
#          trained. UNSURE whether that run was killed or crashed; flagged in
#          the plan.
# Grid: seeds {21, 37, 53} x lr {rescue, 2 neighbours}:
#   g16r8: {5e-5, 1e-4 (rescue), 2e-4}   (stock 3e-4 collapsed at n=1, FINDINGS 6)
#   g24r4: {1e-4, 3e-4 (stock, stable), 6e-4}
# Selection is by the pipeline's OWN validation metric (val_policy_top1 on the
# config's val boards 700-899, the ModelCheckpoint monitor) -- never by test --
# in fwd_rescue_select_bench.slurm.
#
#   sbatch --array=0-8 -J rr-fwdgrid-g16r8 jobs/patterns/fwd_rescue_grid.slurm g16r8
#   # resume link (walltime insurance; skips cells whose DONE exists):
#   sbatch --array=0-8 --dependency=afterany:<jobid> -J rr-fwdgrid-g16r8-r jobs/patterns/fwd_rescue_grid.slurm g16r8
# arg1 = g16r8 | g24r4.  Task id t -> seed = SEEDS[t/3], lr = LRS[t%3].
#
# Run dirs: scaling/runs/<cfg>/forward-rescue/s<seed>-lr<lr>/  (lightning_logs,
# BEST.json, DONE). Idempotent on DONE; resumes from last.ckpt via RR_RESUME=1
# (fwd_train_launcher.py wraps Trainer.fit; move_planner/net.py itself is
# unchanged and already writes last.ckpt).
#
# Cost basis: version_44 (g16r8, 8 ep, bs128) took ~6.0 h on the origin GPU
# (dir mtime 16:10 -> last.ckpt 22:10, 2026-07-16); g24r4 8 epochs ~14 h
# (07:03 -> 21:11, 2026-07-13). 1 GPU-hour = 0.125 nh:
#   g16r8: 9 x ~6 h = 54 GPU-h = 6.75 nh (walltime 10 h; resume link rarely fires)
#   g24r4: 9 x ~14 h = 126 GPU-h = 15.75 nh (walltime 16 h; resume link EXPECTED
#          for some cells if the Karolina A100 is slower than the origin GPU)
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/fwdgrid/%x-%A_%a.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

CFG="$1"
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
SV="$REPO/supervised_valuenet"
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$SV"
export PYTHONPATH=. PYTHONUNBUFFERED=1
mkdir -p "$REPO/runs/fwdgrid"
nvidia-smi -L

SEEDS=(21 37 53)
case "$CFG" in
  g16r8) LRS=(5e-5 1e-4 2e-4); DATA=scaling/data/g16r8/forward.jsonl
         TARGS="--epochs 8 --batch-size 128 --num-workers 8" ;;
  g24r4) LRS=(1e-4 3e-4 6e-4); DATA=scaling/data/g24r4/forward.jsonl
         TARGS="--epochs 8 --batch-size 64 --num-workers 8" ;;
  *) echo "FATAL: cfg must be g16r8 or g24r4"; exit 1 ;;
esac
T="${SLURM_ARRAY_TASK_ID:-${2:-}}"
[ -n "$T" ] || { echo "FATAL: need SLURM_ARRAY_TASK_ID (or arg2) in 0..8"; exit 1; }
SEED="${SEEDS[$((T / 3))]}"; LR="${LRS[$((T % 3))]}"
[ -s "$DATA" ] || { echo "FATAL: missing $DATA"; exit 1; }

DIR="scaling/runs/$CFG/forward-rescue/s${SEED}-lr${LR}"
echo "FWDGRID $CFG task=$T seed=$SEED lr=$LR dir=$DIR $(date +%F' '%T)"
if [ -f "$DIR/DONE" ]; then echo "FWDGRID SKIP $CFG s$SEED lr$LR (DONE)"; exit 0; fi
mkdir -p "$DIR"
STAMP=$(sha256sum "$DATA" | cut -c1-16)
if [ -f "$DIR/DATA.txt" ]; then
  grep -qx "sha256_16=$STAMP" "$DIR/DATA.txt" || { echo "FWDGRID FAILED: corpus changed under partial run"; exit 1; }
else
  printf 'data=%s\nsha256_16=%s\nrecords=%s\nseed=%s\nlr=%s\nargs=%s\nstarted=%s\n' \
    "$DATA" "$STAMP" "$(wc -l < "$DATA")" "$SEED" "$LR" "$TARGS" "$(date -Is)" > "$DIR/DATA.txt"
fi
RESUME=0
ls "$DIR"/lightning_logs/version_*/checkpoints/last.ckpt >/dev/null 2>&1 && RESUME=1
t0=$SECONDS
RR_RESUME=$RESUME python3 jobs/patterns/fwd_train_launcher.py --config "$CFG" --seed "$SEED" --lr "$LR" \
    --run-dir "$DIR" -- --data "$DATA" $TARGS
RC=$?
if [ "$RC" -ne 0 ]; then
  echo "FWDGRID $CFG s$SEED lr$LR LINK-INCOMPLETE rc=$RC after $((SECONDS-t0))s (resume link continues)"; exit 1
fi
[ -s "$DIR/BEST.json" ] || { echo "FWDGRID FAILED: no BEST.json written"; exit 1; }
ls "$DIR"/lightning_logs/version_*/checkpoints/epoch=*.ckpt >/dev/null 2>&1 \
  || { echo "FWDGRID FAILED: exit 0 but no epoch=*.ckpt"; exit 1; }
touch "$DIR/DONE"
echo "FWDGRID $CFG s$SEED lr$LR DONE in $((SECONDS-t0))s: $(cat "$DIR/BEST.json" | tr -d '\n')"
