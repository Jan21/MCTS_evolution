#!/bin/bash
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH -t 16:00:00
#SBATCH -J rr-nnlab-coarse
#SBATCH -o /scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/nnlab/ladder_coarse_%j.out
#
# One COARSE ladder rung (40/48/56/64), enabled by the lean-board path
# (FINDINGS 62): the quality-vs-size curve to its verifiable endpoint. 64 is
# the last size the Rust engine accepts (n<=64 hard assert), so the g64r4 gate
# is the calibration statement that licenses -- or forbids -- trusting labels
# beyond it (FINDINGS 60b).
#
# Per rung: lean boards, Rust-exact test reference, fresh-generation probe
# (throughput + yield at scale, the capstone protocol), 600-instance descent
# gate, value audit. Idempotent per step; resubmit to resume.
#
#   sbatch -J rr-nnlab-coarse-g64r4 supervised_valuenet/nn_labeler/jobs/ladder_coarse.slurm g64r4
set -u
CFG=${1:?config required (g40r4|g48r4|g56r4|g64r4)}
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
    || { echo "COARSE RUST BINARY MISSING"; exit 1; }

CKPT=$(python nn_labeler/jobs/pick_winner.py)
echo "COARSE $CFG LABELER OF RECORD $CKPT"
[ -f "$CKPT" ] || { echo "COARSE MISSING CKPT"; exit 1; }

read -r N WALLS ENVDIR <<< "$(python -c "
from scaling.configs import get
c = get('$CFG'); print(c.grid, c.walls, c.env_dir)")"
# per-instance timeout grows with board area (capstone used 180 at 32)
case "$CFG" in
    g40r4) TMO=240 ;;  g48r4) TMO=300 ;;
    g56r4) TMO=420 ;;  g64r4) TMO=600 ;;
    *) echo "COARSE UNKNOWN CFG $CFG"; exit 1 ;;
esac

# 1. lean boards: fresh-gen probe range + gate range
t0=$SECONDS
python -m nn_labeler.leanboard --n "$N" --robots 4 --walls "$WALLS" \
    --ids 0-19,900-1049 --out "$ENVDIR" \
    || { echo "COARSE $CFG BOARDS FAILED"; exit 1; }
echo "COARSE $CFG boards $((SECONDS-t0))s"

# 2. Rust-exact test reference (the referee; n<=64 only)
AUDIT_SET="scaling/data/${CFG}/backward_audit.rust.jsonl"
if [ ! -f "$AUDIT_SET" ]; then
    t0=$SECONDS
    python -m scaling.gen_data --config "$CFG" --system backward \
        --engine rust --per-graph 10 --workers 16 --splits test \
        --out "$AUDIT_SET" > "/tmp/coarse_${CFG}.log" 2>&1 \
        || { echo "COARSE $CFG RUST LABELS FAILED"; \
             tail -3 "/tmp/coarse_${CFG}.log"; exit 1; }
    echo "COARSE $CFG exact ref $(wc -l < "$AUDIT_SET") records $((SECONDS-t0))s"
fi

# 3. fresh-generation probe (20 boards x 10 -- throughput, yield, drops)
GEN="nn_labeler/results/coarsegen_${CFG}.jsonl"
if [ ! -f "$GEN" ]; then
    t0=$SECONDS
    python -m nn_labeler.descent --config "$CFG" --ckpt "$CKPT" \
        --graphs 0-19 --per-graph 10 --boards lean --device cuda \
        --timeout "$TMO" --out "$GEN" \
        || { echo "COARSE $CFG GEN FAILED"; exit 1; }
    echo "COARSE $CFG gen $(wc -l < "$GEN") records $((SECONDS-t0))s"
fi

# 4. descent gate, capstone protocol
DG="nn_labeler/results/coarsegate_${CFG}.jsonl"
if [ ! -f "$DG" ]; then
    t0=$SECONDS
    python -m nn_labeler.descent --config "$CFG" --ckpt "$CKPT" \
        --instances-from "$AUDIT_SET" --graphs 900-1049 \
        --limit-instances 600 --boards lean --device cuda --timeout "$TMO" \
        --out "$DG" \
        || { echo "COARSE $CFG GATE DESCENT FAILED"; exit 1; }
    echo "COARSE $CFG gate descent $((SECONDS-t0))s"
fi
python -m nn_labeler.audit_descent --exact "$AUDIT_SET" --descent "$DG" \
    --out "nn_labeler/results/coarsegate_${CFG}.json" \
    || { echo "COARSE $CFG GATE AUDIT FAILED"; exit 1; }

# 5. value audit
python -m nn_labeler.audit --ckpt "$CKPT" --split test --device cuda \
    --data "${CFG}=${AUDIT_SET}" \
    --out "nn_labeler/results/audit_coarse_${CFG}.json" \
    || echo "COARSE $CFG VALUE AUDIT FAILED"

echo "NNLAB COARSE DONE $CFG"
