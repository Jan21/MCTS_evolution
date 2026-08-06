#!/bin/bash
#SBATCH -A open-37-42
#SBATCH -p qgpu_free
#SBATCH --gpus 1
#SBATCH -t 2:00:00
#SBATCH -J rr-nnlab-twingate
#SBATCH -o /scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/nnlab/twin_gates_%j.out
#
# Corpus-level label-quality gates for the twin corpora (audit_descent over
# the FULL twin vs its exact corpus). Exists because gate_v2_and_twins.slurm
# runs this audit in the same walltime as the labeling and job 4618940 was
# killed mid-audit; its mop-up path skips the audit when the merged corpus
# already exists. CPU-only work on the cheap qgpu_free lane; idempotent --
# skips missing corpora and existing gates; resubmit any time.
#
#   sbatch supervised_valuenet/nn_labeler/jobs/twin_gates.slurm
set -u
module purge
ml Python/3.11.5-GCCcore-13.2.0
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
cd "$REPO/supervised_valuenet"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=""

source nn_labeler/jobs/twin_map.env

for cfg in g24r4 g24r8 g32r4; do
    twin="nn_labeler/results/twin_${cfg}.jsonl"
    out="nn_labeler/results/twin_gate_${cfg}.json"
    exact=$(eval echo "\$TWIN_CORPUS_${cfg}")
    [ -s "$twin" ] || { echo "TWINGATE SKIP $cfg (no twin corpus yet)"; continue; }
    [ -s "$out" ] && { echo "TWINGATE SKIP $cfg (gate exists)"; continue; }
    t0=$SECONDS
    python -m nn_labeler.audit_descent --exact "$exact" --descent "$twin" \
        --out "$out" \
        || { echo "TWINGATE FAILED $cfg"; continue; }
    echo "TWINGATE DONE $cfg $((SECONDS-t0))s"
done
echo "NNLAB TWINGATE DONE"
