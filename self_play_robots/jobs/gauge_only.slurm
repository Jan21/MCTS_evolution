#!/bin/bash
# Fidelity gauge of one record file (exact engine on CPU inside a 1-GPU allocation).
#   sbatch -J spr-gauge self_play_robots/jobs/gauge_only.slurm <CFG> <RECORDS> <OUT> <VOCAB> [SAMPLE] [MAXCAND] [TIMECAP]
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1
python -m spr.gauge --config "$1" --records "$2" --out "$3" --vocab "$4" --sample "${5:-80}" --max-candidates "${6:-24}" --time-cap "${7:-2400}" --threads 16
echo "SPR GAUGE JOB DONE $(date +%F' '%T)"
