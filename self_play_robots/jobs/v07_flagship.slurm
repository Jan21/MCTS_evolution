#!/bin/bash
# Wave-4 flagship leg: the best verified combination -- v09_strict_value_s8 nets
# + depth-2 root-slides hybrid -- on the FRONTIER exam (unseen + graded d2 rows
# already on disk). Provenance: chosen per the wave-4 gate because v15 (slide
# training) came out flat and depth-2 beat d1 on both landed exams.
#   sbatch -J spr-var-flag self_play_robots/jobs/v07_flagship.slurm
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
cd "$REPO/supervised_valuenet"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
RES="$REPO/self_play_robots/results/variants/v07_hybrid_actions"
N9="$REPO/self_play_robots/results/variants/v09_strict_value_s8/nets.txt"
P9=$(grep '^policy=' "$N9" | cut -d= -f2-); V9=$(grep '^value=' "$N9" | cut -d= -f2-)
FRONT="$REPO/supervised_valuenet/scaling/data/g24r4/bench.unsolved.jsonl"
GED=$(python -c "from scaling.configs import get; print(get('g24r4').env_dir_abs)")
OUT="$RES/bench_frontier_hybrid_d2.json"
echo "SPR FLAG start $(date +%F' '%T) job=$SLURM_JOB_ID"
if [ ! -s "$OUT" ]; then
  RR_ENV_DIR="$GED" RR_GRID=24 RR_ROBOTS=4 RR_WALLS=108 \
  python -m variants.v07_hybrid_actions --instances "$FRONT" --policy "$P9" --value "$V9" \
      --boards pkl --device cuda --prefix-depth 2 --b0 500 --top-m 8 --sub 80 --out "$OUT" \
      || { echo "SPR FLAG FAILED"; exit 1; }
  python -m eval.replay_validate --compare "$OUT" --env-dir "$GED" \
      > "${OUT%.json}.replay.log" 2>&1 || { echo "SPR FLAG REPLAY FAILED"; exit 1; }
fi
for b in bench_frontier_stdmcts bench_frontier_hybrid; do
  [ -s "$RES/$b.json" ] || continue
  python -m spr.gate compare --a "$OUT" --b "$RES/$b.json" > "$RES/gate_frontier_d2_vs_${b##*_}.json" || true
  echo "SPR FLAG gate vs ${b##*_}: $(tr -d '\n' < "$RES/gate_frontier_d2_vs_${b##*_}.json" | cut -c1-200)"
done
echo "SPR FLAG DONE $(date +%F' '%T)"
