#!/bin/bash
# 2026-07-25: repointed from the retired karolina_bundle to the working
# repo (cold-start audit). HISTORICAL recipe: it uses the PRE-RETRAIN
# nets and the old output filenames. For the Track 1 retrained rows,
# author a fresh script per the repo-root prompt.md (exact filenames
# are pinned by eval/report_data.py).
# Zero-shot extended-language backward rows for one ladder rung:
#   sbatch --gpus 1 -t <T> -J rr-b2r-<cfg> jobs/b2_rows_one.slurm <cfg>
# cfg in {g16r6, g16r8, g24r8, g32r4}. Runs the graded and beyond-oracle
# benches in parallel (backward only, --backward-b2, anytime — same mode as
# the measured B1 rows; budgets 1200/k=5 unchanged). 16x16 rungs use the
# base-trained B1 net pair (proven to generalize across robot count); larger
# boards use their per-config nets (zero-shot on the new candidate types).
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --nodes=1
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/b2rows/%x-%j.out

set -euo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

CFG="$1"
BUNDLE=/scratch/project/open-37-42/petrhyner/MCTS_evolution
SV="$BUNDLE/supervised_valuenet"
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$SV"
export PYTHONPATH=. OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
RUN="$BUNDLE/runs/b2rows"; mkdir -p "$RUN"

case "$CFG" in
  g16r6) export RR_GRID=16 RR_ROBOTS=6 RR_WALLS=48  RR_ENV_DIR="$SV/environments_g16r6"
         BP=checkpoints_backward/policy_b1.ckpt; BV=checkpoints_backward/value_b1.ckpt ;;
  g16r8) export RR_GRID=16 RR_ROBOTS=8 RR_WALLS=48  RR_ENV_DIR="$SV/environments_g16r8"
         BP=checkpoints_backward/policy_b1.ckpt; BV=checkpoints_backward/value_b1.ckpt ;;
  g24r8) export RR_GRID=24 RR_ROBOTS=8 RR_WALLS=108 RR_ENV_DIR="$SV/environments_g24r8"
         BP="scaling/runs/g24r8/backward-policy/lightning_logs/version_0/checkpoints/epoch=29-step=29580.ckpt"
         BV="scaling/runs/g24r8/backward-value/lightning_logs/version_0/checkpoints/epoch=10-step=21681.ckpt" ;;
  g32r4) export RR_GRID=32 RR_ROBOTS=4 RR_WALLS=192 RR_ENV_DIR="$SV/environments_g32r4"
         BP="scaling/runs/g32r4/backward-policy/lightning_logs/version_0/checkpoints/epoch=1-step=1642.ckpt"
         BV="scaling/runs/g32r4/backward-value/lightning_logs/version_1/checkpoints/epoch=15-step=52528.ckpt" ;;
  *) echo "unknown cfg $CFG" >&2; exit 1 ;;
esac

python3 -m eval.compare --instances scaling/data/$CFG/bench.solved.jsonl \
  --expansions 1200 --k 5 --backward-policy "$BP" --backward-value "$BV" \
  --backward-anytime --backward-b2 --forward-ckpts "" --device cpu \
  --out scaling/results/$CFG/comparison_b2.json.tmp --md /dev/null \
  > "$RUN/$CFG.graded.log" 2>&1 && \
  mv scaling/results/$CFG/comparison_b2.json.tmp scaling/results/$CFG/comparison_b2.json &
python3 -m eval.compare --instances scaling/data/$CFG/bench.unsolved.jsonl \
  --expansions 1200 --k 5 --backward-policy "$BP" --backward-value "$BV" \
  --backward-anytime --backward-b2 --forward-ckpts "" --device cpu \
  --out scaling/results/$CFG/comparison_ungraded_b2.json.tmp --md /dev/null \
  > "$RUN/$CFG.ungraded.log" 2>&1 && \
  mv scaling/results/$CFG/comparison_ungraded_b2.json.tmp scaling/results/$CFG/comparison_ungraded_b2.json &
wait
echo "=== $CFG graded ===";   tail -3 "$RUN/$CFG.graded.log"
echo "=== $CFG ungraded ==="; tail -3 "$RUN/$CFG.ungraded.log"
echo "B2ROWS $CFG DONE"
