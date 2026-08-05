#!/bin/bash
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH -t 16:00:00
#SBATCH -J rr-nnlab-intgates
#SBATCH -o /scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/nnlab/ladder_int_gates_%j.out
#
# The per-integer ladder rungs 17-31 (owner kept them 2026-08-05), reshaped by
# FINDINGS 58 + 62 into what actually produces the curve:
#
#   - LEAN boards (FINDINGS 62) instead of make_board's all-pairs precompute --
#     the old ladder_prep.slurm would have needed ~10 walltime-killed
#     resubmissions at rungs 25-31; this generates each rung's gate boards in
#     seconds and is layout-identical by construction (proven, record-identical
#     A/B with the real net).
#   - GATES ONLY (value audit + 600-instance descent gate per rung, the
#     capstone protocol) -- no mass corpus labeling: FINDINGS 58 measured those
#     labels 170x cheaper to produce exactly, and nothing downstream consumes
#     them. The per-integer CURVE is the deliverable, and this job produces all
#     14 points of it. Cost ~0.8 nh vs ~28 nh for full corpus production.
#
# Replaces jobs 4610118 (ladder_prep) + 14 x ladder_rung. Idempotent per rung
# and per step; resubmit the same command to resume.
#
#   sbatch supervised_valuenet/nn_labeler/jobs/ladder_int_gates.slurm
set -u
module purge
ml Python/3.11.5-GCCcore-13.2.0
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
cd "$REPO/supervised_valuenet"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
nvidia-smi -L

./rust_datagen/target/release/datagen selftest >/dev/null \
    || { echo "INTGATES RUST BINARY MISSING"; exit 1; }

# NNLAB_CKPT pins the net explicitly. The CURVE must be one net end-to-end:
# its existing 24/32 anchors (capstone, FINDINGS 59) are v1, so the parallel
# lane runs with NNLAB_CKPT=v1 rather than letting pick_winner switch nets
# mid-curve if the v2 gate lands while this job is running.
CKPT=${NNLAB_CKPT:-$(python nn_labeler/jobs/pick_winner.py)}
echo "INTGATES LABELER OF RECORD $CKPT"
[ -f "$CKPT" ] || { echo "INTGATES MISSING CKPT"; exit 1; }

RUNGS="g17r4 g18r4 g19r4 g20r4 g21r4 g22r4 g23r4 g25r4 g26r4 g27r4 g28r4 g29r4 g30r4 g31r4"
FAILED=0

for cfg in $RUNGS; do
    echo "===== INTGATES $cfg $(date +%T) ====="
    read -r N WALLS ENVDIR <<< "$(python -c "
from scaling.configs import get
c = get('$cfg'); print(c.grid, c.walls, c.env_dir)")"

    # 1. lean gate boards (test range only -- that is all the gates read)
    t0=$SECONDS
    python -m nn_labeler.leanboard --n "$N" --robots 4 --walls "$WALLS" \
        --ids 900-1049 --out "$ENVDIR" \
        || { echo "INTGATES $cfg BOARDS FAILED"; FAILED=1; continue; }
    echo "INTGATES $cfg boards $((SECONDS-t0))s"

    # 2. Rust-exact test reference (idempotent)
    AUDIT_SET="scaling/data/${cfg}/backward_audit.rust.jsonl"
    if [ ! -f "$AUDIT_SET" ]; then
        t0=$SECONDS
        python -m scaling.gen_data --config "$cfg" --system backward \
            --engine rust --per-graph 10 --workers 16 --splits test \
            --out "$AUDIT_SET" > "/tmp/intgates_${cfg}.log" 2>&1 \
            || { echo "INTGATES $cfg RUST LABELS FAILED"; \
                 tail -3 "/tmp/intgates_${cfg}.log"; FAILED=1; continue; }
        echo "INTGATES $cfg exact ref $(wc -l < "$AUDIT_SET") records $((SECONDS-t0))s"
    fi

    # 3. value audit
    if [ ! -f "nn_labeler/results/audit_ladder_${cfg}.json" ]; then
        python -m nn_labeler.audit --ckpt "$CKPT" --split test --device cuda \
            --data "${cfg}=${AUDIT_SET}" \
            --out "nn_labeler/results/audit_ladder_${cfg}.json" \
            || echo "INTGATES $cfg VALUE AUDIT FAILED"
    fi

    # 4. descent gate, capstone protocol (600 replayed test instances)
    DG="nn_labeler/results/descent_gate_${cfg}.jsonl"
    if [ ! -f "$DG" ]; then
        t0=$SECONDS
        python -m nn_labeler.descent --config "$cfg" --ckpt "$CKPT" \
            --instances-from "$AUDIT_SET" --graphs 900-1049 \
            --limit-instances 600 --boards lean --device cuda --timeout 120 \
            --out "$DG" \
            || { echo "INTGATES $cfg GATE DESCENT FAILED"; FAILED=1; continue; }
        echo "INTGATES $cfg gate descent $((SECONDS-t0))s"
    fi
    python -m nn_labeler.audit_descent --exact "$AUDIT_SET" --descent "$DG" \
        --out "nn_labeler/results/dgate_ladder_${cfg}.json" \
        || echo "INTGATES $cfg GATE AUDIT FAILED"
done

if [ "$FAILED" -ne 0 ]; then
    echo "NNLAB INTGATES PARTIAL -- resubmit to resume"
    exit 1
fi
echo "NNLAB INTGATES DONE"
