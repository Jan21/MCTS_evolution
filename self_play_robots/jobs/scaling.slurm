#!/bin/bash
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/scale-%x-s%a-%A.out
#SBATCH --array=0-1
set -uo pipefail
ml purge; ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
CK=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/subgoal/stage4_c_r3/ctg-epoch=2-val_mae_all=1.5468.ckpt
CKARG=""; [ "$ARM" = "net" ] && CKARG="--ckpt $CK"
O=results/subgoal/scaling; mkdir -p $O
echo "config=$CFG arm=$ARM fin=$FIN shard=${SLURM_ARRAY_TASK_ID}/2 $(date -Is)"
python -m subgoal.stage3b --config "$CFG" --bench "scaling/data/$CFG/bench.solved.jsonl" \
  --arm "$ARM" $CKARG --k 100 --bound tight --fin-cap "$FIN" --fin-iters 200 \
  --concurrency 32 --shards 2 --shard "${SLURM_ARRAY_TASK_ID}" \
  --out "$O/${CFG}_${ARM}_fin${FIN}_s${SLURM_ARRAY_TASK_ID}.json"
