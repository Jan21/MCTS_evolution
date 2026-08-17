#!/bin/bash
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH -t 6:00:00
#SBATCH -J rr-nnlab-corrupt
#SBATCH -o /scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/nnlab/corrupt_%x_%j.out
#
# The causality arm (FINDINGS 80 design): planners trained on the exact
# g24r4 corpus with CONTROLLED label corruption -- same boards, instances,
# robot count and corpus size as the size-matched control; only argmin
# agreement differs (nn_labeler/corrupt.py, near-tie structural corruption).
# Three arms: d1000 (size control, 100% fidelity), d860 (86%), d822 (82.2%
# -- the g24r8 collapse cell's dose at 4 robots). Same recipe and bench as
# every other g24r4 arm (default torch seed, matching the originals).
#
#   sbatch supervised_valuenet/nn_labeler/jobs/corruption_arm.slurm d822
# (positional arg, not --export -- the env-hold Slurm bug, FINDINGS 72 ops note)
set -u
module purge
ml Python/3.11.5-GCCcore-13.2.0
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
SV="$REPO/supervised_valuenet"
cd "$SV"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
nvidia-smi -L

CFG=g24r4
ARM=${1:?usage: sbatch corruption_arm.slurm d1000|d860|d822}
DATA=nn_labeler/results/corrupt_g24r4_${ARM}.jsonl
[ -f "$DATA" ] || { echo "CORRUPT MISSING DATA $DATA"; exit 1; }
unset RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR

train_one() {                       # train_one <system> <extra args...>
    local system="$1"; shift
    local dir="scaling/runs/${CFG}/${system}-corrupt-${ARM}"
    if [ -f "$dir/DONE" ]; then echo "CORRUPT SKIP train $system"; return 0; fi
    mkdir -p "$dir"
    local resume=""
    ls "$dir"/lightning_logs/version_*/checkpoints/last.ckpt >/dev/null 2>&1 && resume=1
    RR_RESUME=${resume:-0} python -m scaling.train --config "$CFG" \
        --system "$system" --run-dir "$dir" -- --data "$DATA" "$@" \
        || { echo "CORRUPT FAILED train $system"; return 1; }
    ls "$dir"/lightning_logs/version_*/checkpoints/epoch=*.ckpt >/dev/null 2>&1 \
        || { echo "CORRUPT FAILED train $system: no ckpt"; return 1; }
    touch "$dir/DONE"
    echo "CORRUPT DONE train $system"
}

train_one backward-policy --epochs 30 --batch-size 8 || exit 1
train_one backward-value  --epochs 30 --warmup 0 --batch-size 4 --max-per-group 16 || exit 1

BP=$(ls -t scaling/runs/${CFG}/backward-policy-corrupt-${ARM}/lightning_logs/version_*/checkpoints/epoch=*.ckpt | head -1)
BV=$(ls -t scaling/runs/${CFG}/backward-value-corrupt-${ARM}/lightning_logs/version_*/checkpoints/epoch=*.ckpt | head -1)
[ -n "$BP" ] && [ -n "$BV" ] || { echo "CORRUPT NO CKPTS"; exit 1; }

INST=scaling/data/${CFG}/bench.solved.jsonl
OUT=scaling/results/${CFG}/comparison_corrupt_${ARM}.json
if [ ! -s "$OUT" ]; then
    export RR_GRID=24 RR_ROBOTS=4 RR_WALLS=108 RR_ENV_DIR="$SV/environments_${CFG}"
    STAMP=$(printf '%s\n%s\n' "$BP" "$BV" | sha256sum | cut -c1-12)
    RUN="$REPO/runs/nnlab/corrupt.${CFG}.${ARM}.${STAMP}"
    mkdir -p "$RUN"
    [ -e "$RUN/chunk.000.jsonl" ] || \
        split -d -a 3 -l 8 "$INST" "$RUN/chunk." --additional-suffix=.jsonl
    run_chunk() {
        local chunk="$1"; local base; base=$(basename "$chunk" .jsonl)
        local out="$RUN/$base.json"
        [ -s "$out" ] && return 0
        OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" python3 -m eval.compare \
            --instances "$chunk" --expansions 1200 --k 5 \
            --backward-policy "$BP" --backward-value "$BV" \
            --backward-prefix-check --forward-ckpts "" --device cpu \
            --dump-moves --out "$out.tmp.$$" --md /dev/null \
            > "$RUN/$base.log" 2>&1 && mv "$out.tmp.$$" "$out" \
            || { echo "CORRUPT CHUNK FAIL $base"; return 1; }
    }
    export -f run_chunk; export RUN BP BV
    ls "$RUN"/chunk.*.jsonl | xargs -P 8 -n 1 -I{} bash -c 'run_chunk "$@"' _ {}
    NC=$(ls "$RUN"/chunk.*.jsonl | wc -l); ND=$(ls "$RUN"/chunk.*.json 2>/dev/null | wc -l)
    [ "$ND" -eq "$NC" ] || { echo "CORRUPT PARTIAL bench $ND/$NC -- resubmit"; exit 1; }
    python3 eval/merge_compare_shards.py --shards $(ls "$RUN"/chunk.*.json | sort) \
        --instances "$INST" --out "$OUT" --md /dev/null \
        || { echo "CORRUPT MERGE FAILED"; exit 1; }
    python3 -m eval.replay_validate --compare "$OUT" --env-dir "$RR_ENV_DIR" \
        || { mv -f "$OUT" "$OUT.uncertified"; echo "CORRUPT REPLAY FAILED"; exit 1; }
fi
echo "NNLAB CORRUPT DONE $ARM"
