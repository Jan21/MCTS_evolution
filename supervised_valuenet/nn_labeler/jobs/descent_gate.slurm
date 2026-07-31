#!/bin/bash
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH -t 10:00:00
#SBATCH -J rr-nnlab-dgate
#SBATCH -o /scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/nnlab/descent_gate_%j.out
#
# NN-labeler descent quality gate (PLANS.md Plan B/C, debug milestone D4):
# replay the EXACT corpora's test-range instances through the certified-descent
# labeler with a trained size-free checkpoint, then compare labels
# decision-by-decision against the exact solver's (audit_descent). In-domain
# rungs (8, 10), one-step extrapolation (12), and two-step (16, legacy corpus).
#
#   sbatch supervised_valuenet/nn_labeler/jobs/descent_gate.slurm <ckpt> <tag>
#     <ckpt>  SizeFreeValueNet checkpoint path (from the debug battery)
#     <tag>   short label for output filenames, e.g. mix8910_sin2d
set -u
CKPT=${1:?checkpoint path required}
TAG=${2:?tag required}
module purge
ml Python/3.11.5-GCCcore-13.2.0
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
cd "$REPO/supervised_valuenet"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
nvidia-smi -L

gate () {  # gate <config> <exact_jsonl> <graphs_spec>
    local cfg=$1 exact=$2 graphs=$3
    local dout="nn_labeler/results/descent_${TAG}_${cfg}.jsonl"
    if [ ! -f "$dout" ]; then
        python -m nn_labeler.descent --config "$cfg" --ckpt "$CKPT" \
            --instances-from "$exact" --graphs "$graphs" \
            --device cuda --timeout 120 --out "$dout" \
            || { echo "DGATE DESCENT FAILED $cfg"; return 1; }
    else
        echo "DGATE SKIP descent $cfg (exists)"
    fi
    python -m nn_labeler.audit_descent --exact "$exact" --descent "$dout" \
        --out "nn_labeler/results/dgate_${TAG}_${cfg}.json" \
        || echo "DGATE AUDIT FAILED $cfg"
}

gate g8r4  scaling/data/g8r4/backward.rust.jsonl  900-1049
gate g10r4 scaling/data/g10r4/backward.rust.jsonl 900-1049
gate g12r4 scaling/data/g12r4/backward.rust.jsonl 900-1049
gate g16r4 nn/data/combined.jsonl                 112-127,2400-2999

echo "NNLAB DESCENT GATE DONE $TAG"
