#!/bin/bash
# Sharded rerun of the 24x24/8 beyond-oracle EXTENDED-language backward row
# (the single-process version outran its walltime). Reuses the g24u shard
# files; banks each chunk; last pass merges to comparison_ungraded_b2.json.
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 2
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=rr-b2r-g24u-s
#SBATCH --output=/scratch/project/open-37-42/petrhyner/karolina_bundle/runs/b2rows/%x-%j.out

set -euo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

BUNDLE=/scratch/project/open-37-42/petrhyner/karolina_bundle
SV="$BUNDLE/MCTS_evolution/supervised_valuenet"
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$SV"
export PYTHONPATH=. OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
export RR_GRID=24 RR_ROBOTS=8 RR_WALLS=108 RR_ENV_DIR="$SV/environments_g24r8"
RUN="$BUNDLE/runs/b2rows"; OUT="$RUN/out"; mkdir -p "$OUT"
SHARDS="$BUNDLE/runs/lanes/shards"

BP="scaling/runs/g24r8/backward-policy/lightning_logs/version_0/checkpoints/epoch=29-step=29580.ckpt"
BV="scaling/runs/g24r8/backward-value/lightning_logs/version_0/checkpoints/epoch=10-step=21681.ckpt"

run_one() {
  local shard="$1"; local base; base=$(basename "$shard" .jsonl)
  local out="$OUT/b2.$base.json"
  [ -s "$out" ] && { echo "skip $base"; return 0; }
  echo "start $base $(date +%H:%M:%S)"
  if python3 -m eval.compare --instances "$shard" --expansions 1200 --k 5 \
       --backward-policy "$BP" --backward-value "$BV" \
       --backward-anytime --backward-b2 --forward-ckpts "" --device cpu \
       --out "$out.tmp.$$" --md /dev/null > "$RUN/b2.$base.log" 2>&1; then
    mv "$out.tmp.$$" "$out"; echo "done  $base $(date +%H:%M:%S)"
  else
    echo "FAIL  $base"; return 1
  fi
}
export -f run_one
export OUT RUN BP BV

ls "$SHARDS"/g24u.*.jsonl | xargs -P 4 -n 1 -I{} bash -c 'run_one "$@"' _ {}

N=$(ls "$OUT"/b2.g24u.*.json 2>/dev/null | wc -l)
if [ "$N" -ne 20 ]; then echo "PARTIAL b2-g24u: $N/20"; exit 0; fi
python3 eval/merge_compare_shards.py --shards $(ls "$OUT"/b2.g24u.*.json | sort) \
  --instances scaling/data/g24r8/bench.unsolved.jsonl \
  --out scaling/results/g24r8/comparison_ungraded_b2.json --md /dev/null
echo "MERGED b2-g24u"
echo "B2ROW G24U ALL DONE"
