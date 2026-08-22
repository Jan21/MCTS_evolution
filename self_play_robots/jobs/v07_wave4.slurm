#!/bin/bash
# Wave-4 items 3+4 for the hybrid search (one job, phases idempotent):
#   A. depth-2 slides: hybrid d2 (B0=500, top-m 8, sub 80) on unseen + graded
#      (controls + d1 rows already on disk from wave 3).
#   B. robustness: hybrid d1 with the OTHER adopted nets (v14_stack seed-7)
#      + std-mcts control with the same nets, unseen + graded. (The searches
#      are deterministic at bench settings -- root noise off -- so the honest
#      "replicate" is a different-nets run, not a different rng.)
#   C. frontier leg with the wave-3 nets (v09s8): hybrid d1 + std control.
#   sbatch -J spr-var-w4 self_play_robots/jobs/v07_wave4.slurm
# Sizing: unseen d2 ~2 h, graded d2 ~2 h, v14 legs ~3 h, frontier legs ~2.5 h
# -> walltime 8 h, expected ~7 h = 0.9 nh.
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/supervised_valuenet"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
RES="$REPO/self_play_robots/results/variants/v07_hybrid_actions"; mkdir -p "$RES"
N9="$REPO/self_play_robots/results/variants/v09_strict_value_s8/nets.txt"
N14="$REPO/self_play_robots/results/variants/v14_stack/nets.txt"
P9=$(grep '^policy=' "$N9" | cut -d= -f2-);  V9=$(grep '^value=' "$N9" | cut -d= -f2-)
P14=$(grep '^policy=' "$N14" | cut -d= -f2-); V14=$(grep '^value=' "$N14" | cut -d= -f2-)
EXAM="$REPO/self_play_robots/results/variants/exam/g24r4_unseen.jsonl"
EXAMB="$REPO/self_play_robots/results/variants/exam/boards_g24r4"
GRADED="$REPO/supervised_valuenet/scaling/data/g24r4/bench.solved.jsonl"
FRONT="$REPO/supervised_valuenet/scaling/data/g24r4/bench.unsolved.jsonl"
GED=$(python -c "from scaling.configs import get; print(get('g24r4').env_dir_abs)")
echo "SPR W4 start $(date +%F' '%T) job=$SLURM_JOB_ID"
nvidia-smi -L
hyb() {  # hyb <out> <instances> <boards> <envdir> <P> <V> <extra driver flags...>
  local out="$1" inst="$2" b="$3" ed="$4" pp="$5" vv="$6"; shift 6
  [ -s "$out" ] && { echo "SKIP $(basename "$out")"; return 0; }
  local t0=$SECONDS
  RR_ENV_DIR="$ed" RR_GRID=24 RR_ROBOTS=4 RR_WALLS=108 \
  python -m variants.v07_hybrid_actions --instances "$inst" --policy "$pp" --value "$vv" \
      --boards "$b" --device cuda "$@" --out "$out" \
      || { echo "SPR W4 FAILED $(basename "$out")"; return 1; }
  python -m eval.replay_validate --compare "$out" --env-dir "$ed" \
      > "${out%.json}.replay.log" 2>&1 || { echo "SPR W4 REPLAY FAILED $(basename "$out")"; return 1; }
  echo "SPR W4 $(basename "$out") took $((SECONDS-t0))s (replay OK)"
}
ctl() {  # ctl <out> <instances> <P> <V> <extraflags> [arena args]
  local out="$1" inst="$2" pp="$3" vv="$4" xtra="$5"; shift 5
  [ -s "$out" ] && { echo "SKIP $(basename "$out")"; return 0; }
  python -m spr.arena bench --arm "w4_$(basename "$out" .json)" --config g24r4 --instances "$inst" \
      --policy "$pp" --value "$vv" --driver spr \
      --flags "--search mcts --best-at-budget --vocab b2 --anytime $xtra" \
      --width 8 --threads 2 --device cuda "$@" --out "$out" \
      || echo "SPR W4 FAILED ctl $(basename "$out")"
}
# A. depth-2 (v09s8 nets)
hyb "$RES/bench_unseen_hybrid_d2.json" "$EXAM" lean "$EXAMB" "$P9" "$V9" --prefix-depth 2 --b0 500 --top-m 8 --sub 80 || true
hyb "$RES/bench_graded_hybrid_d2.json" "$GRADED" pkl "$GED" "$P9" "$V9" --prefix-depth 2 --b0 500 --top-m 8 --sub 80 || true
# B. robustness: v14 nets, d1 + std control
hyb "$RES/bench_unseen_hybrid_v14nets.json" "$EXAM" lean "$EXAMB" "$P14" "$V14" || true
hyb "$RES/bench_graded_hybrid_v14nets.json" "$GRADED" pkl "$GED" "$P14" "$V14" || true
ctl "$RES/bench_unseen_stdmcts_v14nets.json" "$EXAM" "$P14" "$V14" "--boards lean" --env-dir "$EXAMB"
ctl "$RES/bench_graded_stdmcts_v14nets.json" "$GRADED" "$P14" "$V14" ""
# C. frontier leg (v09s8 nets)
hyb "$RES/bench_frontier_hybrid.json" "$FRONT" pkl "$GED" "$P9" "$V9" || true
ctl "$RES/bench_frontier_stdmcts.json" "$FRONT" "$P9" "$V9" ""
# gates
gate() { [ -s "$2" ] && [ -s "$3" ] || return 0
  python -m spr.gate compare --a "$2" --b "$3" > "$RES/gate_$1.json" || true
  echo "SPR W4 gate $1: $(tr -d '\n' < "$RES/gate_$1.json" | cut -c1-220)"; }
gate unseen_d2_vs_d1     "$RES/bench_unseen_hybrid_d2.json"  "$RES/bench_unseen_hybrid.json"
gate graded_d2_vs_d1     "$RES/bench_graded_hybrid_d2.json"  "$RES/bench_graded_hybrid.json"
gate unseen_d2_vs_std    "$RES/bench_unseen_hybrid_d2.json"  "$RES/bench_unseen_stdmcts.json"
gate graded_d2_vs_std    "$RES/bench_graded_hybrid_d2.json"  "$RES/bench_graded_stdmcts.json"
gate unseen_v14_hyb_vs_std "$RES/bench_unseen_hybrid_v14nets.json" "$RES/bench_unseen_stdmcts_v14nets.json"
gate graded_v14_hyb_vs_std "$RES/bench_graded_hybrid_v14nets.json" "$RES/bench_graded_stdmcts_v14nets.json"
gate frontier_hyb_vs_std "$RES/bench_frontier_hybrid.json"   "$RES/bench_frontier_stdmcts.json"
echo "SPR W4 DONE $(date +%F' '%T)"
