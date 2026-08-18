#!/bin/bash
# Forward (primitive-move) arm at 24x24: the g24r4 forward MoveNet through
# spr.fwd.arena -- A* (parity vs scaling/results/g24r4/comparison.json's forward
# row: 220/232, regret 0.068, 191 exp, 275 s/puzzle CPU) and MCTS best-at-budget.
#   sbatch -J spr-fwd-g24 self_play_robots/jobs/fwd_g24_arms.slurm
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
CK=$REPO/supervised_valuenet/scaling/runs/g24r4/forward/lightning_logs/version_0/checkpoints/epoch=7-step=76360.ckpt
mkdir -p results/fwd_g24
run() { # run <name> <search> <flags>
  local out=results/fwd_g24/$1.json; [ -s "$out" ] && return 0; local t0=$SECONDS
  python -m spr.fwd.arena bench --name "fwd_g24_$1" --ckpt "$CK" --config g24r4 --instances scaling/data/g24r4/bench.solved.jsonl \
      --search "$2" --flags "$3" --expansions 1200 --k 5 --width 8 --threads 2 --device cuda --out "$out" \
      --parity-ref scaling/results/g24r4/comparison.json --parity-system forward --parity-out "results/fwd_g24/$1.parity.json" \
      || echo "SPR FWD G24 FAILED $1"
  echo "SPR FWD G24 $1 took $((SECONDS-t0))s"
}
run astar astar ""
run mcts_best mcts "--best-at-budget --stop-after 200"
run mcts_first mcts ""
echo "SPR FWD G24 DONE $(date +%F' '%T)"
