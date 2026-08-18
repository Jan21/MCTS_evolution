#!/bin/bash
# B2 (extended language: transient supports + by-reference + parks) ceilings on
# the frontier sets and the g32r4/g24r8 graded sets -- the target for a
# subgoal loop that learns the extended vocabulary by self-play.
#   sbatch -J spr-b2ceil self_play_robots/jobs/b2_ceiling.slurm
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
run() { local out="results/ceiling/$1_$3_b2.json"; [ -s "$out" ] && return 0
  python -m spr.ceiling --config "$1" --instances "$2" --vocab b2 --workers 8 --time-cap 120 --max-frontier 100000 --out "$out"; }
run g24r4 scaling/data/g24r4/bench.unsolved.jsonl frontier
run g32r4 scaling/data/g32r4/bench.solved.jsonl   graded
run g24r8 scaling/data/g24r8/bench.solved.jsonl   graded
run g32r4 scaling/data/g32r4/bench.unsolved.jsonl frontier
run g24r8 scaling/data/g24r8/bench.unsolved.jsonl frontier
echo "SPR B2 CEILING DONE $(date +%F' '%T)"
