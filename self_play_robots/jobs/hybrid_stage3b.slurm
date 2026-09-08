#!/bin/bash
#SBATCH --job-name=hyb-3b
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/hyb-3b-%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
CK=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/subgoal/stage3_rec4/ctg-epoch=11-val_top5=0.9377.ckpt
echo "job $SLURM_JOB_ID on $(hostname), $(date -Is), FIN_CAP=${FIN_CAP}"
git -C "$REPO" log -1 --pretty="code at %h %s"
python -m subgoal.stage3b --arm net --ckpt "$CK" \
  --k 100 --bound tight --fin-cap "${FIN_CAP}" --fin-iters 200 --concurrency 64 \
  --out "results/subgoal/stage3b/g16r4_fin${FIN_CAP}.json"
