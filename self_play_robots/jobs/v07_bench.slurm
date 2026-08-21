#!/bin/bash
# v07 root-slides hybrid (wave 3): bench the lab's best nets (v09_strict_value_s8)
# under the hybrid portfolio search vs the SAME nets under standard arena MCTS,
# on the unseen exam and the pinned graded exam. Replay-validates the hybrid
# payloads (composed slide+subgoal move dumps) and writes paired gates.
#   sbatch -J spr-var-v07 self_play_robots/jobs/v07_bench.slurm
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/supervised_valuenet"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
RES="$REPO/self_play_robots/results/variants/v07_hybrid_actions"; mkdir -p "$RES"
NETS="$REPO/self_play_robots/results/variants/v09_strict_value_s8/nets.txt"
P=$(grep '^policy=' "$NETS" | cut -d= -f2-); V=$(grep '^value=' "$NETS" | cut -d= -f2-)
EXAM="$REPO/self_play_robots/results/variants/exam/g24r4_unseen.jsonl"
EXAMB="$REPO/self_play_robots/results/variants/exam/boards_g24r4"
GRADED="$REPO/supervised_valuenet/scaling/data/g24r4/bench.solved.jsonl"
echo "SPR V07 start $(date +%F' '%T) job=$SLURM_JOB_ID nets=$P|$V"
nvidia-smi -L
B0=${B0:-600}; TOPM=${TOPM:-6}; SUB=${SUB:-100}
# ---- hybrid arm (own driver; single process, GPU) -------------------------------
hyb() {  # hyb <tag> <instances> <boards> <envdir-or-- (- = g24r4 config dir)>
  local out="$RES/bench_$1_hybrid.json"
  [ -s "$out" ] && { echo "SKIP hybrid $1"; return 0; }
  local t0=$SECONDS
  local ed="$4"
  [ "$ed" = - ] && ed=$(python -c "from scaling.configs import get; print(get('g24r4').env_dir_abs)")
  # GridEnv/leanboard read RR_* at import/call time -- always pin the g24r4 config
  RR_ENV_DIR="$ed" RR_GRID=24 RR_ROBOTS=4 RR_WALLS=108 \
  python -m variants.v07_hybrid_actions --instances "$2" --policy "$P" --value "$V" \
      --b0 "$B0" --top-m "$TOPM" --sub "$SUB" --boards "$3" --device cuda --out "$out" \
      || { echo "SPR V07 FAILED hybrid $1"; return 1; }
  # replay-certify the composed move dumps
  python -m eval.replay_validate --compare "$out" --env-dir "$ed" \
      > "$RES/replay_$1_hybrid.log" 2>&1 || { echo "SPR V07 REPLAY FAILED $1"; return 1; }
  echo "SPR V07 hybrid $1 took $((SECONDS-t0))s (replay OK)"
}
hyb unseen "$EXAM" lean "$EXAMB" || exit 1
hyb graded "$GRADED" pkl - || exit 1
# ---- control arm: SAME nets, standard arena MCTS best-at-budget ------------------
ctl() {  # ctl <tag> <instances> <extraflags> [arena args]
  local tag="$1" inst="$2" xtra="$3"; shift 3
  local out="$RES/bench_${tag}_stdmcts.json"
  [ -s "$out" ] && { echo "SKIP ctl $tag"; return 0; }
  local t0=$SECONDS
  python -m spr.arena bench --arm "v07ctl_${tag}" --config g24r4 --instances "$inst" \
      --policy "$P" --value "$V" --driver spr \
      --flags "--search mcts --best-at-budget --vocab b2 --anytime $xtra" \
      --width 8 --threads 2 --device cuda "$@" --out "$out" \
      || { echo "SPR V07 FAILED ctl $tag"; return 1; }
  echo "SPR V07 ctl $tag took $((SECONDS-t0))s"
}
ctl unseen "$EXAM" "--boards lean" --env-dir "$EXAMB" || exit 1
ctl graded "$GRADED" "" || exit 1
# ---- paired gates -----------------------------------------------------------------
for t in unseen graded; do
  python -m spr.gate compare --a "$RES/bench_${t}_hybrid.json" --b "$RES/bench_${t}_stdmcts.json" \
      > "$RES/gate_${t}_hybrid_vs_std.json" || true
  echo "SPR V07 gate $t: $(tr -d '\n' < "$RES/gate_${t}_hybrid_vs_std.json" | cut -c1-260)"
done
echo "SPR V07 DONE $(date +%F' '%T)"
