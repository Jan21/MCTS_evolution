#!/bin/bash
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH -t 16:00:00
#SBATCH -J rr-nnlab-deploy
#SBATCH -o /scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/nnlab/deployment_%j.out
#
# THE DEPLOYMENT RUN (owner-approved 2026-08-05: "controlled + 1 deployment").
# The controlled suite (FINDINGS 74) held boards+instances fixed and swapped
# labels. This is the other scenario: the NN pipeline generates EVERYTHING --
# fresh lean boards (ids 2000-3049, disjoint from the exact pool 0-1199),
# fresh instances, NN labels -- no exact solver anywhere in data production.
# Planners train on it and face the SAME sha-pinned g32r4 benchmark as every
# other arm. Label fidelity at this config is already characterized (twin
# gate: 89.4% argmin over 35,437 candidates, FINDINGS 74); this run measures
# the full pipeline, not the labels again.
#
# ~4 h generation + ~10 h training + ~3 h bench vs a 16 h cap: expect 2
# submissions; every step skip-if-exists / resumable (twin_retrain patterns).
#
#   sbatch supervised_valuenet/nn_labeler/jobs/deployment.slurm
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

CFG=g32r4
CKPT=nn_labeler/banked/prod_v1_s11.ckpt
[ -f "$CKPT" ] || { echo "DEPLOY MISSING CKPT"; exit 1; }
unset RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR

# ---- 1. fresh lean boards --------------------------------------------------
python -m nn_labeler.leanboard --n 32 --robots 4 --walls 192 \
    --ids 2000-3049 --out environments_g32r4 \
    || { echo "DEPLOY BOARDS FAILED"; exit 1; }
echo "DEPLOY boards ok"

# ---- 2. fresh NN-labeled corpus, chunked -----------------------------------
CORPUS=nn_labeler/results/deploy_g32r4.jsonl
if [ ! -f "$CORPUS" ]; then
    ok=1
    for range in 2000-2349 2350-2699 2700-3049; do
        chunk="nn_labeler/results/deploy_g32r4.${range}.jsonl"
        [ -f "$chunk" ] && { echo "DEPLOY SKIP chunk $range"; continue; }
        t0=$SECONDS
        python -m nn_labeler.descent --config "$CFG" --ckpt "$CKPT" \
            --graphs "$range" --per-graph 10 --boards lean --device cuda \
            --timeout 180 --out "$chunk" \
            || { echo "DEPLOY CHUNK FAILED $range"; ok=0; break; }
        echo "DEPLOY CHUNK $range $(wc -l < "$chunk") records $((SECONDS-t0))s"
    done
    [ "$ok" = 1 ] || exit 1
    cat nn_labeler/results/deploy_g32r4.2000-2349.jsonl \
        nn_labeler/results/deploy_g32r4.2350-2699.jsonl \
        nn_labeler/results/deploy_g32r4.2700-3049.jsonl > "${CORPUS}.tmp" \
        && mv "${CORPUS}.tmp" "$CORPUS"
    echo "DEPLOY CORPUS $(wc -l < "$CORPUS") records (exact corpus: 51353; twin: 39053)"
fi

# ---- 3. train planners (g32r4 recipe verbatim, TWIN_WIRING sec 2) ----------
train_one() {
    local system="$1"; shift
    local dir="scaling/runs/${CFG}/${system}-nndeploy"
    if [ -f "$dir/DONE" ]; then echo "DEPLOY SKIP train $system"; return 0; fi
    mkdir -p "$dir"
    local resume=""
    ls "$dir"/lightning_logs/version_*/checkpoints/last.ckpt >/dev/null 2>&1 && resume=1
    RR_RESUME=${resume:-0} python -m scaling.train --config "$CFG" \
        --system "$system" --run-dir "$dir" \
        --splits train=2000-2699,val=2700-2899,test=2900-3049 \
        -- --data "$CORPUS" "$@" \
        || { echo "DEPLOY FAILED train $system"; return 1; }
    ls "$dir"/lightning_logs/version_*/checkpoints/epoch=*.ckpt >/dev/null 2>&1 \
        || { echo "DEPLOY FAILED train $system: no ckpt"; return 1; }
    touch "$dir/DONE"
    echo "DEPLOY DONE train $system"
}
train_one backward-policy --epochs 30 --batch-size 8 || exit 1
train_one backward-value  --epochs 25 --warmup 0 --batch-size 2 --max-per-group 8 || exit 1

BP=$(ls -t scaling/runs/${CFG}/backward-policy-nndeploy/lightning_logs/version_*/checkpoints/epoch=*.ckpt | head -1)
BV=$(ls -t scaling/runs/${CFG}/backward-value-nndeploy/lightning_logs/version_*/checkpoints/epoch=*.ckpt | head -1)
[ -n "$BP" ] && [ -n "$BV" ] || { echo "DEPLOY NO CKPTS"; exit 1; }

# ---- 4. bench, both sets, twin_retrain machinery ---------------------------
bench_one() {                       # bench_one <graded|frontier>
    local set="$1" INST OUT
    if [ "$set" = graded ]; then
        INST=scaling/data/${CFG}/bench.solved.jsonl
        OUT=scaling/results/${CFG}/comparison_nndeploy.json
    else
        INST=scaling/data/${CFG}/bench.unsolved.jsonl
        OUT=scaling/results/${CFG}/comparison_ungraded_nndeploy.json
    fi
    [ -s "$OUT" ] && { echo "DEPLOY SKIP bench $set"; return 0; }
    export RR_GRID=32 RR_ROBOTS=4 RR_WALLS=192 RR_ENV_DIR="$SV/environments_${CFG}"
    local STAMP; STAMP=$(printf '%s\n%s\n' "$BP" "$BV" | sha256sum | cut -c1-12)
    RUN="$REPO/runs/nnlab/deploy.${CFG}.${set}.${STAMP}"
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
            || { echo "DEPLOY CHUNK FAIL $base"; return 1; }
    }
    export -f run_chunk; export RUN BP BV
    ls "$RUN"/chunk.*.jsonl | xargs -P 8 -n 1 -I{} bash -c 'run_chunk "$@"' _ {}
    local NC ND
    NC=$(ls "$RUN"/chunk.*.jsonl | wc -l); ND=$(ls "$RUN"/chunk.*.json 2>/dev/null | wc -l)
    [ "$ND" -eq "$NC" ] || { echo "DEPLOY PARTIAL bench $set $ND/$NC -- resubmit"; return 1; }
    python3 eval/merge_compare_shards.py --shards $(ls "$RUN"/chunk.*.json | sort) \
        --instances "$INST" --out "$OUT" --md /dev/null \
        || { echo "DEPLOY MERGE FAILED $set"; return 1; }
    python3 -m eval.replay_validate --compare "$OUT" --env-dir "$RR_ENV_DIR" \
        || { mv -f "$OUT" "$OUT.uncertified"; echo "DEPLOY REPLAY FAILED $set"; return 1; }
    echo "DEPLOY DONE bench $set -> $OUT"
}
bench_one graded || exit 1
bench_one frontier || exit 1
echo "NNLAB DEPLOY DONE"
