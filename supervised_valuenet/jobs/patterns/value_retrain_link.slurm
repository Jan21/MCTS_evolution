#!/bin/bash
# One link of a resumable value-net retrain chain (FINDINGS 43 follow-up).
# Chain links share a completion MARKER file; the first link trains fresh,
# later links (submitted afterany) resume from the newest last.ckpt --
# save_last=True landed in train/looped_pc.py on 2026-07-29 -- or skip
# cleanly when the marker already exists. The marker is written only on
# training rc 0 (house rule: a completion marker is gated on the thing it
# claims).
#
#   sbatch -J rr-b2rt-<cfg>-c20k-val jobs/patterns/value_retrain_link.slurm <cfg> <fresh|resume>
# Per-config knobs mirror b2_retrain_one.slurm VERBATIM.
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/b2retrain/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

CFG="$1"; MODE="$2"
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/supervised_valuenet"
export PYTHONPATH=. PYTHONUNBUFFERED=1
MARK="$REPO/runs/b2retrain/MARK.$CFG.value.cap20000.done"
DATA="scaling/data/$CFG/backward_b2.cap20000.rust.jsonl"
[ -s "$DATA" ] || { echo "FATAL: missing $DATA"; exit 1; }

if [ -e "$MARK" ]; then
  echo "VALCHAIN $CFG cap20000 SKIP (already complete: $MARK)"; exit 0
fi
[ "$MODE" = resume ] && export RR_RESUME=1

case "$CFG" in
  g24r8)
    WARM="scaling/runs/g24r8/backward-value/lightning_logs/version_0/checkpoints/epoch=10-step=21681.ckpt"
    LOWLR="--lr 1e-4"; VBATCH="--batch-size 4 --max-per-group 16" ;;
  g32r4)
    WARM="scaling/runs/g32r4/backward-value/lightning_logs/version_1/checkpoints/epoch=15-step=52528.ckpt"
    LOWLR=""; VBATCH="--batch-size 2 --max-per-group 8" ;;
  *) echo "FATAL: unknown config $CFG"; exit 1 ;;
esac

echo "[valchain $CFG/$MODE] data=$DATA warm=$WARM RR_RESUME=${RR_RESUME:-0}"
python3 -m scaling.train --config "$CFG" --system backward-value \
    --run-dir "scaling/runs/$CFG/backward-value-b2-cap20000" $LOWLR \
    --warm-start "$WARM" --torch-seed 11 \
    -- --data "$DATA" --epochs 30 --warmup 0 $VBATCH
RC=$?
if [ "$RC" -eq 0 ]; then
  touch "$MARK"
  echo "VALCHAIN $CFG cap20000 DONE"
else
  echo "VALCHAIN $CFG cap20000 LINK-INCOMPLETE rc=$RC (next link resumes)"
  exit 1
fi
