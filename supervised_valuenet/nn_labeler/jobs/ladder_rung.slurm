#!/bin/bash
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH -t 16:00:00
#SBATCH -J rr-nnlab-rung
#SBATCH -o /scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/nnlab/ladder_rung_%x_%j.out
#
# One NN-labeler ladder rung (FINDINGS 55 path decision): descent-label the
# rung's train+val boards with the frozen production net, then gate the rung
# against its Rust-exact test-split reference (value audit + label-by-label
# descent gate). Chunked and idempotent: labeling runs in 100-board chunks
# with skip-if-exists, so a walltime kill loses at most one chunk; resubmit
# the same command to resume. Merge happens only when all chunks exist.
#
#   sbatch -J rr-nnlab-rung-g17r4 supervised_valuenet/nn_labeler/jobs/ladder_rung.slurm g17r4 <ckpt>
set -u
CFG=${1:?config required}
CKPT=${2:?production checkpoint required}
module purge
ml Python/3.11.5-GCCcore-13.2.0
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
cd "$REPO/supervised_valuenet"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16

DATA="scaling/data/${CFG}"
AUDIT_SET="${DATA}/backward_audit.rust.jsonl"
[ -f "$AUDIT_SET" ] || { echo "RUNG ${CFG} MISSING AUDIT SET"; exit 1; }

# 1. descent-label train+val boards (0-899) in 100-board chunks
for k in 0 1 2 3 4 5 6 7 8; do
    lo=$((k * 100)); hi=$((lo + 99))
    out="${DATA}/backward.nnlab.c${k}.jsonl"
    if [ -f "$out" ]; then echo "RUNG ${CFG} SKIP chunk $k"; continue; fi
    t0=$SECONDS
    python -m nn_labeler.descent --config "$CFG" --ckpt "$CKPT" \
        --graphs "${lo}-${hi}" --per-graph 10 --device cuda --timeout 120 \
        --out "$out" \
        || { echo "RUNG ${CFG} CHUNK $k FAILED"; exit 1; }
    echo "RUNG ${CFG} chunk $k $(wc -l < "$out") records $((SECONDS-t0))s"
done

# merge (chunk files kept for provenance; merged file is the training corpus)
MERGED="${DATA}/backward.nnlab.jsonl"
if [ ! -f "$MERGED" ]; then
    cat "${DATA}"/backward.nnlab.c{0..8}.jsonl > "${MERGED}.tmp" \
        && mv "${MERGED}.tmp" "$MERGED"
fi
echo "RUNG ${CFG} MERGED $(wc -l < "$MERGED") records"

# 2. value audit vs the exact test reference
python -m nn_labeler.audit --ckpt "$CKPT" --split test --device cuda \
    --data "${CFG}=${AUDIT_SET}" \
    --out "nn_labeler/results/audit_ladder_${CFG}.json" \
    || echo "RUNG ${CFG} VALUE AUDIT FAILED"

# 3. descent gate: replay the exact reference's instances, compare labels
DG="nn_labeler/results/descent_gate_${CFG}.jsonl"
if [ ! -f "$DG" ]; then
    python -m nn_labeler.descent --config "$CFG" --ckpt "$CKPT" \
        --instances-from "$AUDIT_SET" --graphs 900-1049 \
        --device cuda --timeout 120 --out "$DG" \
        || { echo "RUNG ${CFG} GATE DESCENT FAILED"; exit 1; }
fi
python -m nn_labeler.audit_descent --exact "$AUDIT_SET" --descent "$DG" \
    --out "nn_labeler/results/dgate_ladder_${CFG}.json" \
    || echo "RUNG ${CFG} GATE AUDIT FAILED"

echo "NNLAB RUNG DONE ${CFG}"
