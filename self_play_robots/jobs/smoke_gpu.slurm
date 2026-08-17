#!/bin/bash
# GPU smoke test (house rule: qgpu_exp first): tiny end-to-end pass of the M1
# trainers (both systems, mixed sizes, CUDA), the size-free arena driver, the
# MCTS self-play generator (spawned CUDA workers) and the fidelity gauge.
#   sbatch -J spr-smoke self_play_robots/jobs/smoke_gpu.slurm
#SBATCH -A open-37-42
#SBATCH -p qgpu_exp
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
S="$REPO/runs/spr/smoke"; rm -rf "$S"; mkdir -p "$S"
echo "SPR SMOKE start $(date +%F' '%T) job=$SLURM_JOB_ID"; nvidia-smi -L
DATA="--data g16r4=nn/data/combined.jsonl --data g24r4=scaling/data/g24r4/backward.jsonl"
SPL="--splits g16r4:train=0-40,val=41-60 --splits g24r4:train=0-40,val=41-60"
python -m spr.train --system policy $DATA $SPL --limit-records 8000 --epochs 1 --batch-size 8 --device cuda --out-dir "$S/policy" || { echo "SPR SMOKE FAILED policy"; exit 1; }
python -m spr.train --system value $DATA $SPL --limit-records 8000 --epochs 1 --batch-size 4 --max-per-group 8 --init assets/labeler_prod_v1_s11.ckpt --device cuda --out-dir "$S/value" || { echo "SPR SMOKE FAILED value"; exit 1; }
nvidia-smi --query-gpu=memory.used --format=csv
P=$(ls -t "$S"/policy/lightning_logs/version_*/checkpoints/epoch=*.ckpt | head -1)
V=$(ls -t "$S"/value/lightning_logs/version_*/checkpoints/epoch=*.ckpt | head -1)
python -m spr.arena bench --arm smoke_g24 --config g24r4 --instances scaling/data/g24r4/bench.solved.jsonl \
    --policy "$P" --value "$V" --driver spr --flags "--search arena_astar --prefix-check" --limit 16 --width 4 --device cuda \
    --out "$S/bench_g24.json" || { echo "SPR SMOKE FAILED bench"; exit 1; }
python -m spr.arena bench --arm smoke_g24_mcts --config g24r4 --instances scaling/data/g24r4/bench.solved.jsonl \
    --policy "$P" --value "$V" --driver spr --flags "--search mcts --prefix-check --best-at-budget" --limit 8 --width 4 --device cuda \
    --expansions 100 --out "$S/bench_g24_mcts.json" || { echo "SPR SMOKE FAILED bench mcts"; exit 1; }
python -m spr.selfplay --config g24r4 --policy "$P" --value "$V" --boards-dir "$S/boards" --board-ids 9000-9003 \
    --per-board 3 --expansions 100 --stop-after 40 --root-noise 0.25 --complete-siblings --root-all \
    --workers 4 --device cuda --timeout 300 --out "$S/sp/records.jsonl" || { echo "SPR SMOKE FAILED selfplay"; exit 1; }
python -m spr.gauge --config g24r4 --records "$S/sp/records.jsonl" --sample 12 --threads 8 --out "$S/sp/gauge.json" || echo "SPR SMOKE WARN gauge"
echo "SPR SMOKE DONE $(date +%F' '%T)"
