#!/bin/bash
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 2
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/g24r4/%x-%j.out
set -uo pipefail
ml purge; ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution; SV="$REPO/supervised_valuenet"
source /scratch/project/open-37-42/petrhyner/venv/bin/activate; cd "$SV"
export PYTHONPATH=. OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
export RR_GRID=24 RR_ROBOTS=4 RR_WALLS=108 RR_ENV_DIR="$SV/environments_g24r4"
FWD=scaling/runs/g24r4/forward/lightning_logs/version_0/checkpoints/epoch=7-step=76360.ckpt
BP=scaling/runs/g24r4/backward-policy/lightning_logs/version_0/checkpoints/epoch=1-step=1654.ckpt
BV=scaling/runs/g24r4/backward-value/lightning_logs/version_0/checkpoints/epoch=22-step=38019.ckpt
INST=scaling/data/g24r4/bench.unsolved.jsonl
RUN="$REPO/runs/g24r4"; mkdir -p "$RUN"
# forward + backward(full language) on the frontier set, one file, matched budget
python3 -m eval.compare --instances "$INST" --expansions 1200 --k 5 \
  --backward-policy "$BP" --backward-value "$BV" --backward-anytime --backward-b2 \
  --forward-ckpts "$FWD" --device cpu --dump-moves \
  --out scaling/results/g24r4/comparison_ungraded_b2.json.tmp --md /dev/null && \
  mv scaling/results/g24r4/comparison_ungraded_b2.json.tmp scaling/results/g24r4/comparison_ungraded_b2.json
RC=$?
[ "$RC" -eq 0 ] || { echo "G24R4 FRONTIER FAILED rc=$RC"; exit 1; }
python3 -m eval.replay_validate --compare scaling/results/g24r4/comparison_ungraded_b2.json \
  --env-dir "$SV/environments_g24r4"; CRC=$?
[ "$CRC" -eq 0 ] || { echo "G24R4 FRONTIER REPLAY-FAILED rc=$CRC"; exit 1; }
echo "G24R4 FRONTIER DONE"
