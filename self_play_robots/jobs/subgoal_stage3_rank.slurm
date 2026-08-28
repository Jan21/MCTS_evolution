#!/bin/bash
# PLAN_SUBGOAL_DISCOVERY.md Stage 3, analysis leg: how good is the learned h as
# a RANKER, next to the non-learned one, independently of any search? One
# encoder pass per validation (state, robot) group -- seconds of GPU.
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
       PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
D="$REPO/self_play_robots/results/subgoal/stage3"
CK="${SG3_CKPT:-$(cat "$REPO/runs/spr/subgoal/stage3_rec4/best.txt" 2>/dev/null)}"
echo "SPR SG3-RANK start $(date +%F' '%T) job=${SLURM_JOB_ID:-none} ckpt=$CK"
python -m subgoal.stage3 rank --ckpt "$CK" --data "$D/val_b1800-1849.npz" \
    --batch 16 --out "$D/rank_val.json" || echo "SPR SG3-RANK FAILED val"
python -m subgoal.stage3 rank --ckpt "$CK" --data "$D/train_b1000-1699.npz" \
    --batch 16 --out "$D/rank_train.json" || echo "SPR SG3-RANK FAILED train"
echo "SPR SG3-RANK DONE $(date +%F' '%T)"
