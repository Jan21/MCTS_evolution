#!/bin/bash
# Wave-5 item 2 (per-leg): depth-3 slides probe -- diminishing-returns check.
#   sbatch -J spr-var-d3-<exam> self_play_robots/jobs/v07_d3.slurm <unseen|graded>
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
EXAM="$1"
RES="$REPO/self_play_robots/results/variants/v07_hybrid_actions"
N9="$REPO/self_play_robots/results/variants/v09_strict_value_s8/nets.txt"
P=$(grep '^policy=' "$N9" | cut -d= -f2-); V=$(grep '^value=' "$N9" | cut -d= -f2-)
if [ "$EXAM" = unseen ]; then
  INST="$REPO/self_play_robots/results/variants/exam/g24r4_unseen.jsonl"; B=lean
  ED="$REPO/self_play_robots/results/variants/exam/boards_g24r4"
else
  INST="$REPO/supervised_valuenet/scaling/data/g24r4/bench.solved.jsonl"; B=pkl
  ED=$(python -c "from scaling.configs import get; print(get('g24r4').env_dir_abs)")
fi
OUT="$RES/bench_${EXAM}_hybrid_d3.json"
echo "SPR D3 $EXAM start $(date +%F' '%T) job=$SLURM_JOB_ID"
if [ ! -s "$OUT" ]; then
  RR_ENV_DIR="$ED" RR_GRID=24 RR_ROBOTS=4 RR_WALLS=108 \
  python -m variants.v07_hybrid_actions --instances "$INST" --policy "$P" --value "$V" \
      --boards "$B" --device cuda --prefix-depth 3 --b0 460 --top-m 10 --sub 70 --d2-from 4 \
      --out "$OUT" || { echo "SPR D3 FAILED"; exit 1; }
  python -m eval.replay_validate --compare "$OUT" --env-dir "$ED" \
      > "${OUT%.json}.replay.log" 2>&1 || { echo "SPR D3 REPLAY FAILED"; exit 1; }
fi
python -m spr.gate compare --a "$OUT" --b "$RES/bench_${EXAM}_hybrid_d2.json" > "$RES/gate_${EXAM}_d3_vs_d2.json" || true
echo "SPR D3 gate vs d2: $(tr -d '\n' < "$RES/gate_${EXAM}_d3_vs_d2.json" | cut -c1-220)"
echo "SPR D3 DONE $EXAM $(date +%F' '%T)"
