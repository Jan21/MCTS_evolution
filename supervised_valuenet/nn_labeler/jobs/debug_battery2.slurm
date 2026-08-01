#!/bin/bash
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH -t 14:00:00
#SBATCH -J rr-nnlab-bat2
#SBATCH -o /scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/nnlab/debug_battery2_%j.out
#
# NN-labeler debug battery 2: battery 1 (job 4606952) minus its failure mode.
# Battery 1 found 7/8 cold trainings stuck on the constant-value plateau (three
# architectures, byte-identical best val_regret 1.9154); the port had dropped
# the reference's Curriculum warmup. This battery reruns the informative arms
# WITH the two-phase curriculum (--warmup 4) plus seed-robustness runs
# and pe=none single-size references. Idempotent per-run (DONE markers).
#
#   sbatch supervised_valuenet/nn_labeler/jobs/debug_battery2.slurm
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

COMMON="--num-classes 96 --epochs 20 --batch-size 8 --max-per-group 32 --warmup 4 --device cuda"
D8=g8r4=scaling/data/g8r4/backward.rust.jsonl
D9=g9r4=scaling/data/g9r4/backward.rust.jsonl
D10=g10r4=scaling/data/g10r4/backward.rust.jsonl
D12=g12r4=scaling/data/g12r4/backward.rust.jsonl
D16=g16r4=nn/data/combined.jsonl

run () {  # run <name> <seed> <extra train args...>
    local name=$1 seed=$2; shift 2
    local dir="nn_labeler/runs/$name"
    if [ -f "$dir/DONE" ]; then echo "SKIP $name (DONE)"; else
        local resume=""
        ls "$dir"/lightning_logs/version_*/checkpoints/last.ckpt >/dev/null 2>&1 && resume=1
        RR_RESUME=${resume:-0} python -m nn_labeler.train "$@" $COMMON \
            --torch-seed "$seed" --out-dir "$dir" \
            || { echo "BATTERY2 RUN FAILED $name"; return 1; }
        touch "$dir/DONE"
    fi
    local best
    best=$(ls -t "$dir"/lightning_logs/version_*/checkpoints/epoch=*.ckpt 2>/dev/null | head -1)
    [ -n "$best" ] || { echo "BATTERY2 NO CKPT $name"; return 1; }
    python -m nn_labeler.audit --ckpt "$best" --split test --device cuda \
        --data $D8 --data $D9 --data $D10 --data $D12 --data $D16 \
        --out "nn_labeler/results/audit_${name}.json" \
        || echo "BATTERY2 AUDIT FAILED $name"
}

run c2_mix_none_s11  11 --data $D8 --data $D9 --data $D10 --pe none
run c2_g8_none_s11   11 --data $D8 --pe none
run c2_mix_sin2d_s11 11 --data $D8 --data $D9 --data $D10 --pe sin2d
run c2_g8_sin2d_s11  11 --data $D8 --pe sin2d
run c2_mix_none_s37  37 --data $D8 --data $D9 --data $D10 --pe none
run c2_g8_none_s37   37 --data $D8 --pe none
run c2_ref_g10_none_s11 11 --data $D10 --pe none
run c2_ref_g12_none_s11 11 --data $D12 --pe none

echo "NNLAB DEBUG BATTERY2 DONE"
