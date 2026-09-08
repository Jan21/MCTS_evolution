#!/bin/bash
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/hyb3b-%x-s%a-%A.out
#SBATCH --array=0-3
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
CK=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/subgoal/stage3_rec4/ctg-epoch=11-val_top5=0.9377.ckpt
CKARG=""; [ "$ARM" = "net" ] && CKARG="--ckpt $CK"
echo "arm=$ARM fin_cap=$FIN_CAP fin_iters=$FIN_ITERS shard=${SLURM_ARRAY_TASK_ID} on $(hostname) $(date -Is)"
python -m subgoal.stage3b --arm "$ARM" $CKARG \
  --k 100 --bound tight --fin-cap "$FIN_CAP" --fin-iters "$FIN_ITERS" --concurrency 32 \
  --shards 4 --shard "${SLURM_ARRAY_TASK_ID}" \
  --out "results/subgoal/stage3b/${ARM}_fin${FIN_CAP}_it${FIN_ITERS}_s${SLURM_ARRAY_TASK_ID}.json"
