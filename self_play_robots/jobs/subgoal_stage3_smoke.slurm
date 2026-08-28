#!/bin/bash
# Stage 3 timing smoke: how fast is the network arm on one A100? Runs the
# planner with an UNTRAINED checkpoint (the timing does not depend on the
# weights) on 60 instances at the full 1200-expansion budget, so the real plan
# leg can be sized. Results are thrown away; only the wall clock is used.
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
echo "SPR SG3-SMOKE start $(date +%F' '%T) job=${SLURM_JOB_ID:-none}"
nvidia-smi -L
for nb in 64 192; do
  t0=$SECONDS
  python -m subgoal.stage3 plan --arm net --ckpt "$D/untrained_probe.ckpt" \
      --limit 60 --k 5 --expansions 1200 --concurrency 60 --net-batch $nb \
      --name smoke --out "$D/smoke_net_nb$nb.json" 2>&1 | tail -3
  echo "SPR SG3-SMOKE net-batch=$nb 60 instances took $((SECONDS-t0))s"
done
t0=$SECONDS
python -m subgoal.stage3 plan --arm relaxed --limit 60 --k 200 --expansions 1200 \
    --concurrency 60 --name smoke --out "$D/smoke_relaxed_k200.json" 2>&1 | tail -2
echo "SPR SG3-SMOKE relaxed k=200 60 instances took $((SECONDS-t0))s"
echo "SPR SG3-SMOKE DONE $(date +%F' '%T)"
