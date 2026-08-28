#!/bin/bash
# The missing control: is the v07 root-slides hybrid a general improvement to
# sub-goal planning, or only a self-play-net effect?  Runs the SAME three arms
# the flagship used, on the pinned g24r4 graded exam, with the ORIGINAL
# solver-taught (supervised, per-size) backward pair recorded in
# scaling/results/g24r4/comparison.json:
#   A. control      standard PUCT sub-goal search, B2 vocabulary, cap 1200
#   B. hybrid d1    600/6/100 = exactly 1200 (the budget-matched arm)
#   C. hybrid d2    500/8/80  = 1140 (the flagship constants)
# The invocations are copied from jobs/v07_bench.slurm (d1 + control) and
# jobs/v07_wave4.slurm (d2); only the checkpoints and `--arch persize` differ.
#   sbatch -J spr-supnets self_play_robots/jobs/v07_supnets.slurm
# Sharding: the hybrid driver writes only at the end, so each hybrid arm runs
# as 4 contiguous shards (skip-if-exists) merged with eval/merge_compare_shards
# -- a walltime kill then costs one shard, not the arm.  All 12 lanes run
# CONCURRENTLY on the one GPU; measured throughput on this workload is
# GPU-serialisation limited, so this is the cheapest arrangement (no idle tail).
# Cost basis (measured, job 4862104): 47 s/instance for a hybrid arm and 35 s
# for the control alone on one GPU -> ~8.3 single-process hours of work; the
# GPU floor is ~4.5-5 h wall = 0.55-0.65 node-hours.  Walltime 6 h (a longer request is deferred past the daily
# 10:00-18:00 cooling maintenance reservation); every phase is idempotent, so a
# walltime kill costs at most the in-flight shards and a resubmit resumes.
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
export REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/supervised_valuenet"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1
RES="$REPO/self_play_robots/results/variants/v07_supervised_nets"; mkdir -p "$RES"
SH="$REPO/runs/spr/supnets_shards"; mkdir -p "$SH"
# the supervised backward pair recorded in scaling/results/g24r4/comparison.json
# (byte-identical to self_play_robots/assets/g24r4_backward_{policy,value}_exact.ckpt)
P="$REPO/supervised_valuenet/scaling/runs/g24r4/backward-policy/lightning_logs/version_0/checkpoints/epoch=1-step=1654.ckpt"
V="$REPO/supervised_valuenet/scaling/runs/g24r4/backward-value/lightning_logs/version_0/checkpoints/epoch=22-step=38019.ckpt"
GRADED="$REPO/supervised_valuenet/scaling/data/g24r4/bench.solved.jsonl"
GED=$(python -c "from scaling.configs import get; print(get('g24r4').env_dir_abs)")
NSH=4
echo "SPR SUPNETS start $(date +%F' '%T) job=$SLURM_JOB_ID shards=$NSH"
echo "  policy=$P"; echo "  value=$V"; echo "  env_dir=$GED"
nvidia-smi -L
# contiguous shards of the pinned exam (order preserved -> merge is exact)
split -d -n l/$NSH --additional-suffix=.jsonl "$GRADED" "$SH/shard."
wc -l "$SH"/shard.*.jsonl

hybshard() {  # hybshard <tag> <shard-idx> <prefix-depth> <b0> <top-m> <sub>
  local tag="$1" i="$2" d="$3" b0="$4" tm="$5" sub="$6"
  local out="$SH/${tag}.$i.json"
  [ -s "$out" ] && { echo "SKIP $(basename "$out")"; return 0; }
  local t0=$SECONDS
  OMP_NUM_THREADS=1 RR_ENV_DIR="$GED" RR_GRID=24 RR_ROBOTS=4 RR_WALLS=108 \
  python -m variants.v07_hybrid_actions --instances "$SH/shard.0$i.jsonl" \
      --policy "$P" --value "$V" --arch persize --boards pkl --device cuda \
      --prefix-depth "$d" --b0 "$b0" --top-m "$tm" --sub "$sub" --out "$out" \
      || { echo "SPR SUPNETS FAILED $tag shard $i"; return 1; }
  echo "SPR SUPNETS $tag shard $i took $((SECONDS-t0))s"
}
merge_hyb() {  # merge_hyb <tag> <out>
  local tag="$1" out="$2"
  [ -s "$out" ] && { echo "SKIP merge $(basename "$out")"; return 0; }
  local shards=(); for i in $(seq 0 $((NSH-1))); do shards+=("$SH/${tag}.$i.json"); done
  for s in "${shards[@]}"; do [ -s "$s" ] || { echo "SPR SUPNETS merge $tag: missing $s"; return 1; }; done
  python eval/merge_compare_shards.py --shards "${shards[@]}" --instances "$GRADED" \
      --out "$out" --md /dev/null || { echo "SPR SUPNETS MERGE FAILED $tag"; return 1; }
  python -m eval.replay_validate --compare "$out" --env-dir "$GED" \
      > "${out%.json}.replay.log" 2>&1 || { echo "SPR SUPNETS REPLAY FAILED $tag"; return 1; }
  echo "SPR SUPNETS merged $tag -> $(basename "$out") (replay OK)"
}
ctl() {  # ctl <out> <width>
  local out="$1" w="$2"
  [ -s "$out" ] && { echo "SKIP $(basename "$out")"; return 0; }
  local t0=$SECONDS
  python -m spr.arena bench --arm "supnets_graded_stdmcts" --config g24r4 --instances "$GRADED" \
      --policy "$P" --value "$V" --driver spr \
      --flags "--search mcts --best-at-budget --vocab b2 --anytime --arch persize" \
      --width "$w" --threads 2 --device cuda --out "$out" \
      || { echo "SPR SUPNETS FAILED ctl"; return 1; }
  echo "SPR SUPNETS ctl took $((SECONDS-t0))s"
}

PIDS=()
for i in $(seq 0 $((NSH-1))); do hybshard hyb_d1 "$i" 1 600 6 100 & PIDS+=($!); done
for i in $(seq 0 $((NSH-1))); do hybshard hyb_d2 "$i" 2 500 8 80  & PIDS+=($!); done
ctl "$RES/bench_graded_stdmcts.json" 4 & PIDS+=($!)
rc=0; for p in "${PIDS[@]}"; do wait "$p" || rc=1; done
echo "SPR SUPNETS lanes finished rc=$rc $(date +%F' '%T)"
merge_hyb hyb_d1 "$RES/bench_graded_hybrid_d1.json"
merge_hyb hyb_d2 "$RES/bench_graded_hybrid_d2.json"

gate() {  # gate <tag> <A> <B>
  [ -s "$2" ] && [ -s "$3" ] || { echo "SPR SUPNETS gate $1: missing input"; return 0; }
  python -m spr.gate compare --a "$2" --b "$3" > "$RES/gate_$1.json" || return 0
  echo "SPR SUPNETS gate $1: $(tr -d '\n' < "$RES/gate_$1.json" | cut -c1-300)"
}
gate graded_d1_vs_std "$RES/bench_graded_hybrid_d1.json" "$RES/bench_graded_stdmcts.json"
gate graded_d2_vs_std "$RES/bench_graded_hybrid_d2.json" "$RES/bench_graded_stdmcts.json"
gate graded_d2_vs_d1  "$RES/bench_graded_hybrid_d2.json" "$RES/bench_graded_hybrid_d1.json"
for f in "$RES"/bench_graded_*.json; do
  python - "$f" <<'PY'
import json, sys, os
d = json.load(open(sys.argv[1]))
for n, s in d["systems"].items():
    if not s.get("rows"): continue
    a = s["aggregate"]
    print(f"AGG {os.path.basename(sys.argv[1])}: {n[:80]} | solved {a['solved']}/{a['n']} "
          f"mean_moves {a['mean_moves']:.3f} mean_regret {a['mean_regret']:.3f} "
          f"pct_optimal {a['pct_optimal']:.2f} mean_exp {a['mean_expansions']:.1f} "
          f"max_exp {max(r['expansions'] for r in s['rows'])}")
PY
done
echo "SPR SUPNETS DONE $(date +%F' '%T)"
