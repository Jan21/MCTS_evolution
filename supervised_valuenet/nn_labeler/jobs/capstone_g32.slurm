#!/bin/bash
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH -t 16:00:00
#SBATCH -J rr-nnlab-cap32
#SBATCH -o /scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/nnlab/capstone_g32_%j.out
#
# THE capstone measurement: can the network label 32x32 boards, and what does it
# cost? 32x32 is 3.2x the largest board the labeler net ever trained on, and the
# rung where the exact solver is at its most expensive while still usable -- so
# it is both the hardest test and the last one with full ground truth.
#
# Uses the banked production net (nn_labeler/banked/prod_v1_s11.ckpt) and the
# EXISTING exact base-vocabulary corpus at g32r4 as the referee; no ladder prep
# needed (boards 0-1199 already exist).
#
# Three parts, each chunked/idempotent so a walltime kill loses at most one chunk:
#   A. generation throughput + yield on fresh boards (train range)
#   B. label quality: replay the exact corpus's test instances, compare
#      decision-by-decision against the exact solver's own labels
#   C. the same at 24x24 as the intermediate reference point
#
#   sbatch supervised_valuenet/nn_labeler/jobs/capstone_g32.slurm
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

CKPT=nn_labeler/banked/prod_v1_s11.ckpt
[ -f "$CKPT" ] || { echo "CAP32 MISSING BANKED CKPT"; exit 1; }

# ---- A. generation throughput on fresh boards (20 boards x 10 instances) ----
for cfg in g24r4 g32r4; do
    out="nn_labeler/results/capgen_${cfg}.jsonl"
    if [ -f "$out" ]; then echo "CAP32 SKIP gen $cfg"; else
        t0=$SECONDS
        python -m nn_labeler.descent --config "$cfg" --ckpt "$CKPT" \
            --graphs 0-19 --per-graph 10 --device cuda --timeout 180 \
            --out "$out" \
            || { echo "CAP32 GEN FAILED $cfg"; exit 1; }
        echo "CAP32 GEN $cfg $(wc -l < "$out") records $((SECONDS-t0))s"
    fi
done

# ---- B/C. label quality vs the exact solver, test split ----
gate () {  # gate <config> <exact_corpus>
    local cfg=$1 exact=$2
    local dg="nn_labeler/results/capgate_${cfg}.jsonl"
    if [ ! -f "$dg" ]; then
        t0=$SECONDS
        # --limit-instances bounds the leg: the gate needs a few hundred
        # decisions for a solid read, and an unbounded 32x32 leg could outrun
        # the walltime and lose the whole file (descent writes once, at the end).
        python -m nn_labeler.descent --config "$cfg" --ckpt "$CKPT" \
            --instances-from "$exact" --graphs 900-1049 --limit-instances 600 \
            --device cuda --timeout 180 --out "$dg" \
            || { echo "CAP32 GATE-DESCENT FAILED $cfg"; return 1; }
        echo "CAP32 GATEGEN $cfg $(wc -l < "$dg") records $((SECONDS-t0))s"
    fi
    python -m nn_labeler.audit_descent --exact "$exact" --descent "$dg" \
        --out "nn_labeler/results/capgate_${cfg}.json" \
        || echo "CAP32 GATE-AUDIT FAILED $cfg"
}

gate g24r4 scaling/data/g24r4/backward.jsonl
gate g32r4 scaling/data/g32r4/backward.rust.jsonl

echo "NNLAB CAPSTONE DONE"
