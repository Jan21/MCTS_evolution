#!/bin/bash
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH -t 16:00:00
#SBATCH -J rr-nnlab-b2
#SBATCH -o /scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/nnlab/b2_payoff_%j.out
#
# THE PAYOFF EXPERIMENT (motivated by FINDINGS 58).
#
# For BASE vocabulary the Rust exact labeler beats NN descent ~170x, so the NN
# labeler saves nothing there. The regime it exists for is the EXTENDED (B2)
# vocabulary, where the exact campaign projected ~441 node-hours and the
# iteration cap that rescued the cost DESTROYED the labels it existed to make:
# the by-reference candidate share fell 13.5% -> 4.8% at cap 5,000 (FINDINGS
# 32), and the retrain on capped data lost 24.6 points on the frontier (34).
#
# The structural argument, which is stronger than the speed argument: certified
# descent HAS NO ITERATION BUDGET. It is a fixed-depth greedy completion, so
# the wandering-rollout pathology that forced the cap cannot occur, and the
# cap/quality trade-off simply does not exist. If the by-reference share of
# NN-generated B2 labels lands in the uncapped band (~13%) rather than the
# capped band (~5%), the method delivers labels the exact pipeline could not
# afford to produce at all.
#
# Part 1: train a size-free value net on the five EXISTING B2 corpora
#         (sizes 16/24/32, robots 4/6/8) -- vocabulary-pure, never mixed with
#         base labels (house rule).
# Part 2: descent-label B2 at g16r6 and g32r4 with NO cap, and measure the
#         by-reference share (qc_byref) + throughput.
# Part 3: gate the labels against the existing exact B2 corpus.
#
#   sbatch supervised_valuenet/nn_labeler/jobs/b2_payoff.slurm [seed]
set -u
SEED=${1:-11}
module purge
ml Python/3.11.5-GCCcore-13.2.0
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
cd "$REPO/supervised_valuenet"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
nvidia-smi -L

DIR="nn_labeler/runs/b2_mix_none_s${SEED}"
B2=backward_b2.cap20000.rust.jsonl

# ---- 1. train the B2 size-free value net -------------------------------------
if [ ! -f "$DIR/DONE" ]; then
    resume=""
    ls "$DIR"/lightning_logs/version_*/checkpoints/last.ckpt >/dev/null 2>&1 && resume=1
    RR_RESUME=${resume:-0} python -m nn_labeler.train \
        --data g16r4=scaling/data/g16r4/$B2 \
        --data g16r6=scaling/data/g16r6/$B2 \
        --data g16r8=scaling/data/g16r8/$B2 \
        --data g24r8=scaling/data/g24r8/$B2 \
        --data g32r4=scaling/data/g32r4/$B2 \
        --pe none --num-classes 96 --epochs 30 --batch-size 8 --max-per-group 32 \
        --warmup 4 --lr 1e-4 --torch-seed "$SEED" --device cuda --out-dir "$DIR" \
        || { echo "B2 TRAIN FAILED"; exit 1; }
    touch "$DIR/DONE"
fi
CKPT=$(ls -t "$DIR"/lightning_logs/version_*/checkpoints/epoch=*.ckpt | head -1)
[ -n "$CKPT" ] || { echo "B2 NO CKPT"; exit 1; }
echo "B2 CKPT $CKPT"

# ---- 2/3. label B2 with NO cap, measure by-reference share, gate -------------
for cfg in g16r6 g32r4; do
    out="nn_labeler/results/b2lab_${cfg}.jsonl"
    if [ ! -f "$out" ]; then
        t0=$SECONDS
        python -m nn_labeler.descent --config "$cfg" --ckpt "$CKPT" --vocab b2 \
            --graphs 0-49 --per-graph 10 --device cuda --timeout 300 --out "$out" \
            || { echo "B2 LABEL FAILED $cfg"; continue; }
        echo "B2 LABEL $cfg $(wc -l < "$out") records $((SECONDS-t0))s"
    fi
    # the metric the cap destroyed: share of candidates whose helper stands at a
    # PLANNED cell rather than its start cell (expected band 5-20%, base 13%)
    python - "$out" "$cfg" <<'PY'
import json, sys
path, cfg = sys.argv[1], sys.argv[2]
tot = byref = 0
for line in open(path):
    r = json.loads(line)
    starts = {tuple(h[0]) for h in r["helpers"]}
    tot += 1
    byref += tuple(r["cand_helper"][0]) not in starts
print(f"B2 BYREF {cfg}: {byref}/{tot} = {100*byref/max(tot,1):.1f}% "
      f"(capped exact corpora sit at 4.8-12.7%; uncapped exact was 13.5%)")
PY
    python -m nn_labeler.audit_descent \
        --exact "scaling/data/${cfg}/${B2}" --descent "$out" \
        --out "nn_labeler/results/b2gate_${cfg}.json" \
        || echo "B2 GATE FAILED $cfg"
done
echo "NNLAB B2 PAYOFF DONE"
