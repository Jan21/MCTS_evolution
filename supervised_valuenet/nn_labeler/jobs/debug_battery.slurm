#!/bin/bash
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH -t 14:00:00
#SBATCH -J rr-nnlab-debug
#SBATCH -o /scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/nnlab/debug_battery_%j.out
#
# NN-labeler track, 8x8 debug battery (nn_labeler/PLANS.md, Phase 0 / S0.6).
# 8 short trainings of the size-free value net on the exact-labeled debug rungs
# (g8r4/g9r4/g10r4/g12r4), each followed by a 5-size audit sweep including
# zero-shot 12x12 and legacy 16x16. Chunked + idempotent: each run leaves a DONE
# marker and is skipped on resubmission; an interrupted training resumes from
# last.ckpt (RR_RESUME). Walltime kill loses at most one run's current epoch.
#
#   sbatch supervised_valuenet/nn_labeler/jobs/debug_battery.slurm
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

COMMON="--num-classes 96 --torch-seed 11 --epochs 20 --batch-size 8 --max-per-group 32 --device cuda"
D8=g8r4=scaling/data/g8r4/backward.rust.jsonl
D9=g9r4=scaling/data/g9r4/backward.rust.jsonl
D10=g10r4=scaling/data/g10r4/backward.rust.jsonl
D12=g12r4=scaling/data/g12r4/backward.rust.jsonl
D16=g16r4=nn/data/combined.jsonl

run () {  # run <name> <extra train args...>
    local name=$1; shift
    local dir="nn_labeler/runs/$name"
    if [ -f "$dir/DONE" ]; then echo "SKIP $name (DONE)"; else
        local resume=""
        ls "$dir"/lightning_logs/version_*/checkpoints/last.ckpt >/dev/null 2>&1 && resume=1
        RR_RESUME=${resume:-0} python -m nn_labeler.train "$@" $COMMON --out-dir "$dir" \
            || { echo "BATTERY RUN FAILED $name"; return 1; }
        touch "$dir/DONE"
    fi
    local best
    best=$(ls -t "$dir"/lightning_logs/version_*/checkpoints/epoch=*.ckpt 2>/dev/null | head -1)
    [ -n "$best" ] || { echo "BATTERY NO CKPT $name"; return 1; }
    python -m nn_labeler.audit --ckpt "$best" --split test --device cuda \
        --data $D8 --data $D9 --data $D10 --data $D12 --data $D16 \
        --out "nn_labeler/results/audit_${name}.json" \
        || echo "BATTERY AUDIT FAILED $name"
}

run d1_g8_sin2d      --data $D8 --pe sin2d
run d1_g8_coord      --data $D8 --pe coord
run d1_g8_none       --data $D8 --pe none
run d2_mix8910_sin2d --data $D8 --data $D9 --data $D10 --pe sin2d
run d2_mix8910_coord --data $D8 --data $D9 --data $D10 --pe coord
run d2_mix8910_none  --data $D8 --data $D9 --data $D10 --pe none
run d3_ref_g10_sin2d --data $D10 --pe sin2d
run d3_ref_g12_sin2d --data $D12 --pe sin2d

echo "NNLAB DEBUG BATTERY DONE"
