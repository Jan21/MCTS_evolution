#!/bin/bash
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH -t 6:00:00
#SBATCH -J rr-nnlab-exrep
#SBATCH -o /scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/nnlab/exact_replicate_%j.out
#
# Symmetric seed replicate of the EXACT arm at g24r4 (FINDINGS 71's noted
# gap): the twin arm has two seeds straddling the exact arm's single point,
# so the equivalence claim needs the exact arm's own seed spread. Same
# recipe as the original g24r4 runs (TWIN_WIRING section 2), torch seed 21,
# benched with the identical pinned protocol -> comparison_exactseed21.json.
# Idempotent like twin_retrain.slurm (same guards, same chunked bench).
#
#   sbatch supervised_valuenet/nn_labeler/jobs/exact_replicate.slurm
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
DATA=scaling/data/g24r4/backward.jsonl
unset RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR

train_one() {                       # train_one <system> <extra args...>
    local system="$1"; shift
    local dir="scaling/runs/${CFG}/${system}-exact-seed21"
    if [ -f "$dir/DONE" ]; then echo "EXREP SKIP train $system"; return 0; fi
    mkdir -p "$dir"
    local resume=""
    ls "$dir"/lightning_logs/version_*/checkpoints/last.ckpt >/dev/null 2>&1 && resume=1
    RR_RESUME=${resume:-0} python -m scaling.train --config "$CFG" \
        --system "$system" --run-dir "$dir" --torch-seed 21 -- --data "$DATA" "$@" \
        || { echo "EXREP FAILED train $system"; return 1; }
    ls "$dir"/lightning_logs/version_*/checkpoints/epoch=*.ckpt >/dev/null 2>&1 \
        || { echo "EXREP FAILED train $system: no ckpt"; return 1; }
    touch "$dir/DONE"
    echo "EXREP DONE train $system"
}

train_one backward-policy --epochs 30 --batch-size 8 || exit 1
train_one backward-value  --epochs 30 --warmup 0 --batch-size 4 --max-per-group 16 || exit 1

BP=$(ls -t scaling/runs/${CFG}/backward-policy-exact-seed21/lightning_logs/version_*/checkpoints/epoch=*.ckpt | head -1)
BV=$(ls -t scaling/runs/${CFG}/backward-value-exact-seed21/lightning_logs/version_*/checkpoints/epoch=*.ckpt | head -1)
[ -n "$BP" ] && [ -n "$BV" ] || { echo "EXREP NO CKPTS"; exit 1; }

INST=scaling/data/${CFG}/bench.solved.jsonl
OUT=scaling/results/${CFG}/comparison_exactseed21.json
if [ ! -s "$OUT" ]; then
    export RR_GRID=24 RR_ROBOTS=4 RR_WALLS=108 RR_ENV_DIR="$SV/environments_${CFG}"
    STAMP=$(printf '%s\n%s\n' "$BP" "$BV" | sha256sum | cut -c1-12)
    RUN="$REPO/runs/nnlab/exrep.${CFG}.${STAMP}"
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
            || { echo "EXREP CHUNK FAIL $base"; return 1; }
    }
    export -f run_chunk; export RUN BP BV
    ls "$RUN"/chunk.*.jsonl | xargs -P 8 -n 1 -I{} bash -c 'run_chunk "$@"' _ {}
    NC=$(ls "$RUN"/chunk.*.jsonl | wc -l); ND=$(ls "$RUN"/chunk.*.json 2>/dev/null | wc -l)
    [ "$ND" -eq "$NC" ] || { echo "EXREP PARTIAL bench $ND/$NC -- resubmit"; exit 1; }
    python3 eval/merge_compare_shards.py --shards $(ls "$RUN"/chunk.*.json | sort) \
        --instances "$INST" --out "$OUT" --md /dev/null \
        || { echo "EXREP MERGE FAILED"; exit 1; }
    python3 -m eval.replay_validate --compare "$OUT" --env-dir "$RR_ENV_DIR" \
        || { mv -f "$OUT" "$OUT.uncertified"; echo "EXREP REPLAY FAILED"; exit 1; }
fi
echo "NNLAB EXREP DONE"
