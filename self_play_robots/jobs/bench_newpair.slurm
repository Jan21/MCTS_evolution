#!/bin/bash
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:58:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/sgnet-%x-%j.out
# Bench the pair produced by a self-play round (its nets.txt) with the ladder's
# network protocol (jobs/sgnet_smoke.sh). ROUND = results/selfplay/<dir>.
set -uo pipefail
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
NETS="$REPO/self_play_robots/results/selfplay/${ROUND}/nets.txt"
[ -s "$NETS" ] || { echo "BENCH NEWPAIR: no nets.txt for round $ROUND (round unfinished)"; exit 1; }
export POLICY=$(grep '^policy=' "$NETS" | cut -d= -f2-)
export VALUE=$(grep '^value=' "$NETS" | cut -d= -f2-)
echo "BENCH NEWPAIR round=$ROUND cfg=$CFG lim=$LIM off=${OFF:-0} policy=$POLICY value=$VALUE"
export CFG LIM OFF TAG
bash "$REPO/self_play_robots/jobs/sgnet_smoke.sh"
