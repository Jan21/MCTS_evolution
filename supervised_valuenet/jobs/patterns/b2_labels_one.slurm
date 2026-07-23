#!/bin/bash
# One config's extended-vocabulary (B2) backward label dataset via the GATED
# Rust engine. Usage:
#   sbatch -J rr-b2lab-g16r6 jobs/patterns/b2_labels_one.slurm g16r6
# arg1 = config (g16r4 | g16r6 | g16r8 | g24r8 | g32r4).
# Output: scaling/data/<cfg>/backward_b2.rust.jsonl (+ rust_work manifests
# recording engine + vocab). House rules: vocabulary-named dataset, never
# mixed with any other vocabulary; do NOT submit before the b2 gate job
# (rr-b2gate) has printed "B2 GATE ALL LEGS ZERO DIFFS".
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/b2labels/%x-%j.out

set -euo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

CFG="$1"
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/supervised_valuenet"
export PYTHONPATH=. OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1

# Budget calibration (measured 2026-07-23 on base boards): a legitimate B2
# rollout completes in ~60-1,700 solver iterations; a WANDERER's
# per-iteration cost explodes with plan size (one wanderer could not exhaust
# a 200k budget in 8 minutes). The budget therefore sits just above the
# legitimate range per config — wanderers count as failed attempts and the
# bridge's wave protocol samples replacements, like the Python SIGALRM did.
case "$CFG" in
  g16r4)        BUDGET=50000  ;;
  g16r6|g16r8)  BUDGET=100000 ;;
  g24r8|g32r4)  BUDGET=200000 ;;
  *)            BUDGET=100000 ;;
esac
OUT="scaling/data/$CFG/backward_b2.rust.jsonl"
python3 -m scaling.rust_bridge --config "$CFG" --system backward --vocab b2 \
    --per-graph 20 --threads 16 --budget-iters "$BUDGET" --force \
    --out "$OUT" --work-tag "backward_b2"
wc -l "$OUT"
echo "B2LABELS $CFG DONE"
