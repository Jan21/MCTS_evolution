#!/bin/bash
# 2026-07-25: repointed from the retired karolina_bundle to the working
# repo (cold-start audit). HISTORICAL recipe: it uses the PRE-RETRAIN
# nets and the old output filenames. For the Track 1 retrained rows,
# author a fresh script per the repo-root prompt.md (exact filenames
# are pinned by eval/report_data.py).
# One scaling-comparison lane, sharded. Usage:
#   sbatch --gpus 4 -t 36:00:00 -J rr-g32u jobs/lanes_one.slurm g32u 8
#   sbatch --gpus 2 -t 20:00:00 -J rr-g24u jobs/lanes_one.slurm g24u 4
#   sbatch --gpus 2 -t 14:00:00 -J rr-g32g jobs/lanes_one.slurm g32g 4
# arg1 = lane (g32u | g24u | g32g), arg2 = concurrent shards (8 threads each).
# Shards are idempotent (skip when output exists): resubmission resumes.
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --nodes=1
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/lanes/%x-%j.out

set -euo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

LANE="$1"; PAR="$2"; SHARD_GLOB="${3:-}"   # optional: e.g. "g32u.0[0-9]" to run half a lane
BUNDLE=/scratch/project/open-37-42/petrhyner/MCTS_evolution
SV="$BUNDLE/supervised_valuenet"
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$SV"
export PYTHONPATH=. OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1

RUN="$BUNDLE/runs/lanes"
mkdir -p "$RUN/shards" "$RUN/out" "$RUN/logs"

case "$LANE" in
  g32u) SRC=scaling/data/g32r4/bench.unsolved.jsonl ;;
  g24u) SRC=scaling/data/g24r8/bench.unsolved.jsonl ;;
  g32g) SRC=scaling/data/g32r4/bench.solved.jsonl ;;
  *) echo "unknown lane $LANE" >&2; exit 1 ;;
esac
[ -e "$RUN/shards/$LANE.00.jsonl" ] || split -d -a 2 -l 15 "$SRC" "$RUN/shards/$LANE." --additional-suffix=.jsonl

run_one() {
  local shard="$1"
  local base; base=$(basename "$shard" .jsonl)
  local lane="${base%%.*}"
  local out="$RUN/out/$base.json"
  [ -s "$out" ] && { echo "skip $base (done)"; return 0; }
  local BP BV FWD
  case "$lane" in
    g32u|g32g)
      export RR_GRID=32 RR_ROBOTS=4 RR_WALLS=192 RR_ENV_DIR="$SV/environments_g32r4"
      BP="scaling/runs/g32r4/backward-policy/lightning_logs/version_0/checkpoints/epoch=1-step=1642.ckpt"
      BV="scaling/runs/g32r4/backward-value/lightning_logs/version_1/checkpoints/epoch=15-step=52528.ckpt"
      FWD="lightning_logs/version_45/checkpoints/epoch=3-step=158564.ckpt"
      ;;
    g24u)
      export RR_GRID=24 RR_ROBOTS=8 RR_WALLS=108 RR_ENV_DIR="$SV/environments_g24r8"
      BP="scaling/runs/g24r8/backward-policy/lightning_logs/version_0/checkpoints/epoch=29-step=29580.ckpt"
      BV="scaling/runs/g24r8/backward-value/lightning_logs/version_0/checkpoints/epoch=10-step=21681.ckpt"
      FWD="lightning_logs/version_47/checkpoints/epoch=3-step=115144.ckpt"
      ;;
  esac
  echo "start $base $(date +%H:%M:%S)"
  if python3 -m eval.compare --instances "$shard" --expansions 1200 --k 5 \
       --backward-policy "$BP" --backward-value "$BV" --forward-ckpts "$FWD" \
       --backward-prefix-check --device cpu \
       --out "$out.tmp.$$" --md /dev/null > "$RUN/logs/$base.log" 2>&1; then
    mv "$out.tmp.$$" "$out"
    echo "done  $base $(date +%H:%M:%S)"
  else
    echo "FAIL  $base (see logs/$base.log)"; return 1
  fi
}
export -f run_one
export RUN SV

# SHARD_GLOB may hold several space-separated patterns (disjoint sub-jobs)
for pat in ${SHARD_GLOB:-$LANE.*}; do ls "$RUN"/shards/$pat.jsonl; done | sort -u \
  | xargs -P "$PAR" -n 1 -I{} bash -c 'run_one "$@"' _ {}

echo "--- shard pass finished ---"
NSHARDS=$(ls "$RUN"/shards/$LANE.*.jsonl | wc -l)
NDONE=$(ls "$RUN"/out/$LANE.*.json 2>/dev/null | wc -l)
if [ "$NDONE" -ne "$NSHARDS" ]; then
  echo "MERGE SKIPPED for $LANE: $NDONE/$NSHARDS shards done (resubmit for the rest)"
  echo "LANE $LANE PARTIAL"
  exit 0
fi
echo "--- $LANE complete ($NDONE/$NSHARDS), merging ---"
case "$LANE" in
  g32g) python3 eval/merge_compare_shards.py --shards $(ls "$RUN"/out/g32g.*.json | sort) \
          --instances scaling/data/g32r4/bench.solved.jsonl \
          --out scaling/results/g32r4/comparison.json --md scaling/results/g32r4/COMPARISON.md ;;
  g32u) python3 eval/merge_compare_shards.py --shards $(ls "$RUN"/out/g32u.*.json | sort) \
          --instances scaling/data/g32r4/bench.unsolved.jsonl \
          --out scaling/results/g32r4/comparison_ungraded.json --md /dev/null ;;
  g24u) python3 eval/merge_compare_shards.py --shards $(ls "$RUN"/out/g24u.*.json | sort) \
          --instances scaling/data/g24r8/bench.unsolved.jsonl \
          --out scaling/results/g24r8/comparison_ungraded.json --md /dev/null ;;
esac
echo "MERGED $LANE"
echo "LANE $LANE ALL DONE"
