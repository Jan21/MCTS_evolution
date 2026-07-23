#!/bin/bash
# Learned backward planner with Lever B2 on the pinned base benchmark:
# same nets and flags as the 95.3% B1 row (final450_backward_b1.json) plus
# --backward-b2 (supports-by-reference + generalized parks, zero-shot ranking).
# Backward only (no forward lane needed; the forward rows exist).
#SBATCH --job-name=rr-b2bench
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/karolina_bundle/runs/b2/%x-%j.out

set -euo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

BUNDLE=/scratch/project/open-37-42/petrhyner/karolina_bundle
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$BUNDLE/MCTS_evolution/supervised_valuenet"
export PYTHONPATH=. OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
unset RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR

python3 -m eval.compare --instances eval/data/bench450.jsonl \
    --expansions 1200 --k 5 \
    --backward-policy checkpoints_backward/policy_b1.ckpt \
    --backward-value  checkpoints_backward/value_b1.ckpt \
    --backward-anytime --backward-b2 --forward-ckpts "" --device cpu \
    --out eval/results/final450_backward_b2.json --md /dev/null
echo "B2BENCH DONE"
