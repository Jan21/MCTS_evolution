#!/bin/bash
#SBATCH --job-name=hyb-s1-16
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/hyb-s1-16-%j.out
# PLAN_HYBRID_ACTION_SPACE.md Stage 1 at 16x16. Gate is committed in the plan.
# CPU-only; the GPU is requested because the allocation has no CPU partition.
set -euo pipefail
cd /scratch/project/open-37-42/janhula/mcts_stage5_probe
export OMP_NUM_THREADS=1 PYTHONPATH=supervised_valuenet:self_play_robots
echo "job $SLURM_JOB_ID on $(hostname), $(date -Is)"; git log -1 --pretty="code at %h %s"
python3 -m spr.ceiling_prefix \
  --config g16r4 --instances eval/data/bench450.jsonl --vocab b2 --prefix any \
  --idx-from self_play_robots/results/subgoal/stage1_state_space_first20.json,self_play_robots/results/subgoal/stage1_state_space_stride23.json \
  --workers 16 --time-cap 30 --max-iters 100000 --max-frontier 200000 \
  --out results/ceiling/prefix_g16r4_b2.json
