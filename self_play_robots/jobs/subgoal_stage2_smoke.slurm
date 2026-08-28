#!/bin/bash
# Stage 2 GPU smoke: exercise the exact train/eval/beam code paths on a GPU with
# tiny settings before the real leg is submitted. Writes only under /tmp-like
# scratch dirs inside runs/, never into results/.
#   sbatch -J spr-sg-s2smoke self_play_robots/jobs/subgoal_stage2_smoke.slurm
# Cost: 1 GPU on qgpu_exp for a few minutes = <0.03 node-hours.
#SBATCH -A open-37-42
#SBATCH -p qgpu_exp
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" \
       PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
D="$REPO/self_play_robots/results/subgoal/stage2"
S="$REPO/runs/spr/subgoal/stage2_smoke"
rm -rf "$S"; mkdir -p "$S"
echo "SPR SG-S2SMOKE start $(date +%F' '%T) job=${SLURM_JOB_ID:-none}"
nvidia-smi -L

python -m subgoal.optset --n 3 --stride 149 --workers 6 --time-cap 60 \
    --out "$S/optsets.json"

t0=$SECONDS
python -m subgoal.stage2 train --train "$D/train_b1000-1699.npz" \
    --val "$D/val_b1800-1849.npz" --run "$S/run" --epochs 1 --batch 16 \
    --workers 8 --max-minutes 8
echo "SPR SG-S2SMOKE one epoch took $((SECONDS-t0))s"
CK=$(cat "$S/run/best.txt")
python -m subgoal.stage2 eval --ckpt "$CK" --data "$D/test_bench450_roots.npz" \
    --batch 8 --limit 24 --out "$S/eval.json" | tail -25
python -m subgoal.stage2 beam --ckpt "$CK" --optsets "$S/optsets.json" \
    --out "$S/beam.json" | tail -5
echo "SPR SG-S2SMOKE DONE $(date +%F' '%T)"
