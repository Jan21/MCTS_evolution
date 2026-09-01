#!/bin/bash
#SBATCH --job-name=hyb-s1-24
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/hyb-s1-24-%j.out
# PLAN_HYBRID_ACTION_SPACE.md Stage 1 at 24x24, on the 29 instances of
# STAGE5_PROBE.md so the row joins the existing table. Tighter per-candidate
# caps than the 16x16 leg because there are ~360 candidates per instance here;
# the baseline arm is recomputed under the SAME caps, so the comparison is
# internal and does not depend on the recorded payload's caps.
set -euo pipefail
cd /scratch/project/open-37-42/janhula/mcts_stage5_probe
export OMP_NUM_THREADS=1 PYTHONPATH=supervised_valuenet:self_play_robots
echo "job $SLURM_JOB_ID on $(hostname), $(date -Is)"; git log -1 --pretty="code at %h %s"
python3 -m spr.ceiling_prefix \
  --config g24r4 --instances scaling/data/g24r4/bench.solved.jsonl --vocab b2 --prefix any \
  --idx-from self_play_robots/results/subgoal/stage5_probe/merged.json \
  --workers 16 --time-cap 20 --max-iters 50000 --max-frontier 150000 \
  --out results/ceiling/prefix_g24r4_b2.json
