#!/bin/bash
# One link of a resumable POLICY-net retrain chain -- sibling of
# value_retrain_link.slurm, same marker/skip/resume contract. Exists
# because ~1-in-N Karolina job starts die instantly with an empty log and
# RaisedSignal:15 (observed 4x on 3 different nodes, 2026-07-28/29); an
# afterany retry link absorbs both that flake and walltime kills.
#
#   sbatch -J rr-b2rt-<cfg>-c20k-pol jobs/patterns/policy_retrain_link.slurm <cfg> <fresh|resume>
# Per-config knobs mirror b2_retrain_one.slurm VERBATIM (policy stage:
# cold, unseeded).
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
MARK="$REPO/runs/b2retrain/MARK.$CFG.policy.cap20000.done"
DATA="scaling/data/$CFG/backward_b2.cap20000.rust.jsonl"
[ -s "$DATA" ] || { echo "FATAL: missing $DATA"; exit 1; }

if [ -e "$MARK" ]; then
  echo "POLCHAIN $CFG cap20000 SKIP (already complete: $MARK)"; exit 0
fi
[ "$MODE" = resume ] && export RR_RESUME=1

case "$CFG" in
  g24r8) LOWLR="--lr 1e-4"; PBATCH="--batch-size 8" ;;
  g32r4) LOWLR=""; PBATCH="--batch-size 16" ;;
  *) echo "FATAL: unknown config $CFG"; exit 1 ;;
esac

echo "[polchain $CFG/$MODE] data=$DATA RR_RESUME=${RR_RESUME:-0}"
python3 -m scaling.train --config "$CFG" --system backward-policy \
    --run-dir "scaling/runs/$CFG/backward-policy-b2-cap20000" $LOWLR \
    -- --data "$DATA" --epochs 30 $PBATCH
RC=$?
if [ "$RC" -eq 0 ]; then
  touch "$MARK"
  echo "POLCHAIN $CFG cap20000 DONE"
else
  echo "POLCHAIN $CFG cap20000 LINK-INCOMPLETE rc=$RC (next link resumes)"
  exit 1
fi
