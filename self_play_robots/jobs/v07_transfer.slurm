#!/bin/bash
# Wave-5 item 3 (per-leg): does the action-space ceiling break TRANSFER to
# board sizes / robot counts the hybrid never ran on? One leg = one config x
# one exam: hybrid d2 (single-process driver) + std-MCTS control (chunked
# arena) + paired gate. Nets = mix_b2mix_iter3 (the mature mixed pair;
# provenance in the payload).
#   sbatch -J spr-var-tr-<cfg>-<exam> self_play_robots/jobs/v07_transfer.slurm <g32r4|g24r8> <graded|frontier>
# Sizing: g32r4 graded ~3 h; g24r8 graded ~2 h; g24r8 frontier ~4 h.
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
cd "$REPO/supervised_valuenet"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
CFG="$1"; EXAM="$2"
RES="$REPO/self_play_robots/results/variants/v07_transfer"; mkdir -p "$RES"
NETS="$REPO/self_play_robots/results/selfplay/mix_b2mix_iter3/nets.txt"
P=$(grep '^policy=' "$NETS" | cut -d= -f2-); V=$(grep '^value=' "$NETS" | cut -d= -f2-)
INST="scaling/data/$CFG/bench.solved.jsonl"; [ "$EXAM" = frontier ] && INST="scaling/data/$CFG/bench.unsolved.jsonl"
eval "$(python - "$CFG" <<'PYEOF'
import sys
from scaling.configs import get, env
for k, v in env(get(sys.argv[1])).items():
    print(f"export {k}={v}")
PYEOF
)"
HOUT="$RES/bench_${CFG}_${EXAM}_hybrid_d2.json"
SOUT="$RES/bench_${CFG}_${EXAM}_stdmcts.json"
echo "SPR TRANSFER $CFG $EXAM start $(date +%F' '%T) job=$SLURM_JOB_ID env=$RR_GRID/$RR_ROBOTS/$RR_WALLS"
if [ ! -s "$HOUT" ]; then
  python -m variants.v07_hybrid_actions --instances "$REPO/supervised_valuenet/$INST" \
      --policy "$P" --value "$V" --boards pkl --device cuda \
      --prefix-depth 2 --b0 500 --top-m 8 --sub 80 --out "$HOUT" \
      || { echo "SPR TRANSFER FAILED hybrid"; exit 1; }
  python -m eval.replay_validate --compare "$HOUT" --env-dir "$RR_ENV_DIR" \
      > "${HOUT%.json}.replay.log" 2>&1 || { echo "SPR TRANSFER REPLAY FAILED"; exit 1; }
  echo "SPR TRANSFER hybrid leg done"
fi
if [ ! -s "$SOUT" ]; then
  python -m spr.arena bench --arm "tr_${CFG}_${EXAM}_std" --config "$CFG" --instances "$INST" \
      --policy "$P" --value "$V" --driver spr \
      --flags "--search mcts --best-at-budget --vocab b2 --anytime" \
      --width 8 --threads 2 --device cuda --out "$SOUT" \
      || { echo "SPR TRANSFER FAILED std"; exit 1; }
fi
python -m spr.gate compare --a "$HOUT" --b "$SOUT" > "$RES/gate_${CFG}_${EXAM}_hyb_vs_std.json" || true
echo "SPR TRANSFER gate: $(tr -d '\n' < "$RES/gate_${CFG}_${EXAM}_hyb_vs_std.json" | cut -c1-220)"
echo "SPR TRANSFER DONE $CFG $EXAM $(date +%F' '%T)"
