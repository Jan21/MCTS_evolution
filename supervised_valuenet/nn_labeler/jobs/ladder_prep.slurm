#!/bin/bash
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH -t 6:00:00
#SBATCH -J rr-nnlab-lprep
#SBATCH -o /scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/nnlab/ladder_prep_%j.out
#
# NN-labeler ladder prep (CPU work inside a 1-GPU allocation, qgpu-only rule):
# for the 14 new ladder rungs g17..g23,g25..g31 (r4) -- 24/32 pre-exist --
# generate the 1200 boards each (8-way sharded) and Rust-exact-label the TEST
# split (boards 900-1049, per-graph 10, base vocab): the per-rung audit/gate
# reference sets. Idempotent: gen_boards never overwrites; label step skipped
# if the output exists. GPU idles; that is the price of a qgpu-only allocation.
#
#   sbatch supervised_valuenet/nn_labeler/jobs/ladder_prep.slurm
set -u
module purge
ml Python/3.11.5-GCCcore-13.2.0
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
cd "$REPO/supervised_valuenet"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16

./rust_datagen/target/release/datagen selftest >/dev/null \
    || { echo "LPREP RUST BINARY MISSING"; exit 1; }

RUNGS="g17r4 g18r4 g19r4 g20r4 g21r4 g22r4 g23r4 g25r4 g26r4 g27r4 g28r4 g29r4 g30r4 g31r4"

for cfg in $RUNGS; do
    t0=$SECONDS
    for s in 0 1 2 3 4 5 6 7; do
        python -m scaling.gen_boards --config "$cfg" --nshards 8 --shard "$s" \
            > /dev/null 2>&1 &
    done
    wait
    n=$(ls "environments_${cfg}"/env_*.pkl 2>/dev/null | wc -l)
    echo "LPREP BOARDS $cfg $n pkls $((SECONDS-t0))s"
    [ "$n" -eq 1200 ] || { echo "LPREP BOARDS INCOMPLETE $cfg"; exit 1; }

    out="scaling/data/${cfg}/backward_audit.rust.jsonl"
    if [ -f "$out" ]; then echo "LPREP SKIP labels $cfg (exists)"; continue; fi
    t0=$SECONDS
    python -m scaling.gen_data --config "$cfg" --system backward --engine rust \
        --per-graph 10 --workers 16 --splits test --out "$out" \
        > "/tmp/lprep_${cfg}.log" 2>&1 \
        || { echo "LPREP LABELS FAILED $cfg"; tail -3 "/tmp/lprep_${cfg}.log"; exit 1; }
    echo "LPREP LABELS $cfg $(wc -l < "$out") records $((SECONDS-t0))s"
done
echo "NNLAB LADDER PREP DONE"
