#!/bin/bash
# Candidate-generator / plan-language ceiling on the MOVES metric
# (PROBLEM.md section 6.1, offline, no network): exhaustive subgoal search +
# strict realization on the pinned benches, vs d*. Four arms: {g16r4 bench450,
# g24r4 bench.solved} x {base, b2 vocabulary}. CPU-only work inside a 1-GPU
# allocation (qgpu-only project): 16 cores, ~2 h worst case = ~0.25 nh.
#
#   sbatch -J spr-ceiling self_play_robots/jobs/ceiling.slurm
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
mkdir -p "$REPO/runs/spr" results/ceiling
echo "SPR CEILING start $(date +%F' '%T) job=$SLURM_JOB_ID"
run() {  # run <cfg> <instances> <vocab> <timecap>
  local out="results/ceiling/$1_$3.json"
  if [ -s "$out" ]; then echo "SPR CEILING SKIP $out"; return 0; fi
  python -m spr.ceiling --config "$1" --instances "$2" --vocab "$3" \
      --workers 16 --time-cap "$4" --max-iters 200000 --out "$out"
}
run g16r4 eval/data/bench450.jsonl              base 60
run g24r4 scaling/data/g24r4/bench.solved.jsonl base 90
run g16r4 eval/data/bench450.jsonl              b2   60
run g24r4 scaling/data/g24r4/bench.solved.jsonl b2   90
echo "SPR CEILING ALL DONE $(date +%F' '%T)"
