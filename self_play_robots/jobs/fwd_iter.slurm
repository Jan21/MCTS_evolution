#!/bin/bash
# ONE self-play iteration k for the PRIMITIVE-MOVE arm (PROBLEM.md section 5,
# mirroring jobs/selfplay_iter.slurm for the subgoal arm):
#   A generate: fresh lean boards of $CFG (ids >= 5000) -> PUCT MCTS with net k-1
#               -> replayed paths -> ExIt records (value = remaining certified
#               path length, policy = committed move, + mcts_visits)
#   B train:    MoveNet warm-started from net k-1 on the replay window
#               (+ optional exact anchor corpus), val split BY BOARD ID,
#               ModelCheckpoint on the val metric, save_last, fixed torch seed
#   C bench:    arena (1200 expansions / k=5, 8-wide, replay-certified) on the
#               pinned $CFG exam with astar AND mcts best-at-budget
#   D gate:     paired vs iteration k-1 (McNemar + moves sign test) and vs the
#               FROZEN forward baseline row of $CFG
# Idempotent per phase (DONE sentinels); a walltime kill resumes on resubmit.
#
#   sbatch -J spr-fwd-it1 self_play_robots/jobs/fwd_iter.slurm <K> <CKPT_PREV|auto> [PREV_BENCH|auto] [CFG]
#
# CFG (4th positional, or the $CFG env var; default g16r4) selects EVERYTHING
# per-config -- board geometry, exam, anchor corpus, frozen reference row, the
# base checkpoint of iteration 1, and the run/result directories:
#
#   |          | g16r4 (default, unchanged)              | any other cfg (e.g. g24r4)              |
#   |----------|-----------------------------------------|-----------------------------------------|
#   | exam     | eval/data/bench450.jsonl                | scaling/data/$CFG/bench.solved.jsonl    |
#   | anchor   | move_planner/data/moves_scored.jsonl    | scaling/data/$CFG/forward.jsonl         |
#   | ref JSON | eval/results/comparison_forward.json     | scaling/results/$CFG/comparison.json     |
#   | ref row  | move_planner/checkpoints/candidate_scored.ckpt | "forward move planner"            |
#   | base ckpt| move_planner/checkpoints/candidate_scored.ckpt | newest scaling/runs/$CFG/forward/.../epoch=*.ckpt |
#
# `CKPT_PREV=auto` resolves to the base checkpoint at K=1 and to iteration K-1's
# BEST.txt otherwise; `PREV_BENCH=auto` (or empty) pairs each bench against the
# SAME-tag bench of iteration K-1. Board ids are namespaced per config by the
# per-config boards/run/result directories (`<cfg>_iter<k>`), never shared.
#
# Knobs via env: BOARDS (120) PER_BOARD (8) EXP (600) STOP (150) NOISE (0.25)
#   WORKERS (8) WINDOW (3) ANCHOR (per-config path; "" = none)
#   ANCHOR_LIMIT (200000) EPOCHS (6) LR (1e-4) BATCH (256 g16r4 / 64 else)
#   DSTAR (diag|off)
#   EXAM REFJ REF_SYS BASE_CKPT VAL_EVERY (10) TIMEOUT (600)
#   SAMPLER (mix) RAND_MIX (0.25) WALK_K (16)  -- the instance-difficulty dial:
#     walk draws are shallow (measured 2.8 moves at g24r4), eval-distribution
#     draws match the exam (7.5 moves, bench mean d*=7.69), so raise RAND_MIX
#     wherever the exam is deeper than the walk
#   OUT_ROOT (runs/spr/fwd) WIDTH (8) DEV (cuda)
#
# Cost basis. g16r4 (measured F-M2): generation 120 boards x 8 instances at
# <=600 expansions ~1-2 h, training ~1 h, two 450-instance benches ~0.5 h.
# g24r4, MEASURED 2026-08-19 on 1 A100 (smoke jobs 4703311 / 4703367):
#   generation, EXP=800 STOP=200, per instance-ATTEMPT and per worker:
#     eval-distribution draw ("random")  106 s, 67% solved, 9.1 records/solve
#     forward-walk draw      ("walk")     10 s, 100% solved, 3.8 records/solve
#     -> at RAND_MIX=0.5, 58 s/attempt, 83% kept, 5.9 records/kept instance;
#        960 kept instances = ~2.3 h on 8 workers (EXP=400 is 1.7x cheaper but
#        solves only 53% of the eval-distribution draws)
#   training, batch 64: ~12.5 ms per record-epoch (26.8k train + 7.6k val
#     records = 538 s/epoch); ANCHOR_LIMIT=40000 -> ~8 min/epoch
#   benching, 232-instance g24r4 exam, 8-wide: astar ~22 min, mcts ~40 min
#   -> ~4.5-5 h per iteration = 0.6 nh; the 8 h walltime cap bounds it at 1.0 nh.
# The g24r4 exam's mean d* is 7.69 and eval-distribution draws average 8.1
# realized moves while walk draws average 2.8, hence RAND_MIX=0.5 there.
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
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
K="$1"; CKPT_PREV="$2"; PREV_BENCH="${3:-}"; CFG="${4:-${CFG:-g16r4}}"
SV="$REPO/supervised_valuenet"
BOARDS=${BOARDS:-120}; PER_BOARD=${PER_BOARD:-8}; EXP=${EXP:-600}; STOP=${STOP:-150}
NOISE=${NOISE:-0.25}; WORKERS=${WORKERS:-8}; WINDOW=${WINDOW:-3}
EPOCHS=${EPOCHS:-6}; LR=${LR:-1e-4}; VAL_EVERY=${VAL_EVERY:-10}
TIMEOUT=${TIMEOUT:-600}; WALK_K=${WALK_K:-16}; RAND_MIX=${RAND_MIX:-0.25}
OUT_ROOT="${OUT_ROOT:-$REPO/runs/spr/fwd}"; WIDTH=${WIDTH:-8}; DEV=${DEV:-cuda}
SAMPLER=${SAMPLER:-mix}   # mix|random|walk; RAND_MIX = P(eval-distribution draw)

# ---- per-config exam / anchor / frozen reference / base checkpoint -----------
if [ "$CFG" = g16r4 ]; then
  EXAM=${EXAM:-eval/data/bench450.jsonl}
  ANCHOR=${ANCHOR-move_planner/data/moves_scored.jsonl}
  REFJ=${REFJ:-$SV/eval/results/comparison_forward.json}
  REF_SYS=${REF_SYS:-move_planner/checkpoints/candidate_scored.ckpt}
  BASE_CKPT=${BASE_CKPT:-move_planner/checkpoints/candidate_scored.ckpt}
  DSTAR=${DSTAR:-diag}          # the exact forward oracle is cheap at 16x16
  # F-M2 (FINDINGS 6): at 16x16 the move tree never closes, so best-at-budget
  # runs to the cap -- the recorded arm has no stop-after; keep it that way.
  MCTS_FLAGS=${MCTS_FLAGS:---mcts-backup min --best-at-budget}
  BATCH=${BATCH:-256}
else
  EXAM=${EXAM:-scaling/data/$CFG/bench.solved.jsonl}
  ANCHOR=${ANCHOR-scaling/data/$CFG/forward.jsonl}
  REFJ=${REFJ:-$SV/scaling/results/$CFG/comparison.json}
  REF_SYS=${REF_SYS:-forward move planner}
  BASE_CKPT=${BASE_CKPT:-$(ls -t "$SV/scaling/runs/$CFG/forward"/lightning_logs/version_*/checkpoints/epoch=*.ckpt 2>/dev/null | head -1)}
  DSTAR=${DSTAR:-off}           # move_planner.oracle at >=24x24 is minutes/instance
  # exactly the F-M2 g24 arm (jobs/fwd_g24_arms.slurm `mcts_best`)
  MCTS_FLAGS=${MCTS_FLAGS:---best-at-budget --stop-after 200}
  # MEASURED 2026-08-19 (job 4703311): the 24x24 MoveNet at batch 256 OOMs a
  # 40 GB A100 (36.8 GiB allocated at step 0). 64 is move_planner/net.py's own
  # default -- the batch the per-size supervised MoveNet was trained with.
  BATCH=${BATCH:-64}
fi
ANCHOR_LIMIT=${ANCHOR_LIMIT:-200000}

IT="$OUT_ROOT/selfplay/${CFG}_iter${K}"
RES="$REPO/self_play_robots/results/fwd_selfplay/${CFG}_iter${K}"
PREV_RES="$REPO/self_play_robots/results/fwd_selfplay/${CFG}_iter$((K-1))"
mkdir -p "$REPO/runs/spr" "$IT" "$RES" "$OUT_ROOT/boards"
ID0=$((5000 + BOARDS * (K - 1))); ID1=$((ID0 + BOARDS - 1))

if [ "$CKPT_PREV" = auto ]; then     # chain: net (and reference bench) of iteration K-1
  if [ "$K" -le 1 ]; then
    CKPT_PREV="$BASE_CKPT"
  else
    CKPT_PREV=$(cat "$OUT_ROOT/selfplay/${CFG}_iter$((K-1))/train/BEST.txt" 2>/dev/null)
  fi
  [ -n "$CKPT_PREV" ] || { echo "SPR FWD ITER FAILED: auto ckpt unresolved"; exit 1; }
fi
[ "$PREV_BENCH" = auto ] && PREV_BENCH=""     # "auto" = pair each tag with its own
[ -s "$CKPT_PREV" ] || [ -s "$SV/$CKPT_PREV" ] \
  || { echo "SPR FWD ITER FAILED: missing ckpt $CKPT_PREV"; exit 1; }
echo "SPR FWD ITER $CFG k=$K start $(date +%F' '%T) job=${SLURM_JOB_ID:-none} prev=$CKPT_PREV boards=$ID0-$ID1"
echo "SPR FWD ITER $CFG exam=$EXAM anchor=${ANCHOR:-none} ref=$REFJ ref_sys=$REF_SYS"
nvidia-smi -L 2>/dev/null || true
phase() { [ -f "$IT/DONE.$1" ]; }
mark()  { touch "$IT/DONE.$1"; echo "SPR FWD ITER PHASE DONE $1 $(date +%T)"; }

# ---- A generate ---------------------------------------------------------------
if ! phase gen; then
  t0=$SECONDS
  python -m spr.fwd.selfplay --config "$CFG" --ckpt "$CKPT_PREV" \
      --boards-dir "$OUT_ROOT/boards/${CFG}_iter${K}" --board-ids "$ID0-$ID1" \
      --per-board "$PER_BOARD" --seed "$K" --iter "$K" \
      --expansions "$EXP" --k 5 --stop-after "$STOP" --root-noise "$NOISE" --root-all \
      --sampler "$SAMPLER" --rand-mix-prob "$RAND_MIX" --walk-k-max "$WALK_K" \
      --d-star "$DSTAR" \
      --workers "$WORKERS" --threads 2 \
      --device "$DEV" --timeout "$TIMEOUT" --out "$IT/records.jsonl" \
    || { echo "SPR FWD ITER FAILED gen"; exit 1; }
  cp "$IT/records.jsonl.manifest.json" "$RES/generation.manifest.json"
  echo "SPR FWD ITER gen took $((SECONDS-t0))s"; mark gen
fi

# ---- B train (replay window = the last WINDOW iterations' records) -------------
RECARGS=()
for j in $(seq $((K - WINDOW + 1)) "$K"); do
  [ "$j" -ge 1 ] || continue
  f="$OUT_ROOT/selfplay/${CFG}_iter${j}/records.jsonl"
  [ -s "$f" ] && RECARGS+=(--records "$f")
done
[ ${#RECARGS[@]} -gt 0 ] || { echo "SPR FWD ITER FAILED: no records in the window"; exit 1; }
ANCARGS=(); [ -n "$ANCHOR" ] && ANCARGS=(--anchor "$ANCHOR" --anchor-limit "$ANCHOR_LIMIT")
TRAIN_DIR="$IT/train"
if ! phase train; then
  t0=$SECONDS
  RR_RESUME=$([ -d "$TRAIN_DIR/lightning_logs" ] && echo 1 || echo 0) \
  python -m spr.fwd.train --config "$CFG" "${RECARGS[@]}" "${ANCARGS[@]}" --init "$CKPT_PREV" \
      --epochs "$EPOCHS" --lr "$LR" --batch-size "$BATCH" --num-workers 4 \
      --val-every "$VAL_EVERY" --torch-seed $((21 + K)) --device "$DEV" --out-dir "$TRAIN_DIR" \
    || { echo "SPR FWD ITER FAILED train"; exit 1; }
  echo "SPR FWD ITER train took $((SECONDS-t0))s"; mark train
fi
NEW_CKPT=$(cat "$TRAIN_DIR/BEST.txt" 2>/dev/null)
[ -s "$NEW_CKPT" ] || { echo "SPR FWD ITER FAILED: no checkpoint in $TRAIN_DIR"; exit 1; }
echo "SPR FWD ITER net k=$K -> $NEW_CKPT"
printf 'config=%s\niter=%s\nmovenet=%s\ninit=%s\n' "$CFG" "$K" "$NEW_CKPT" "$CKPT_PREV" > "$RES/nets.txt"
cp "$TRAIN_DIR/RESULT.json" "$RES/train.result.json" 2>/dev/null
cp "$TRAIN_DIR/DATA.json" "$RES/train.data.json" 2>/dev/null

# ---- C bench ------------------------------------------------------------------
bench_one() {  # bench_one <tag> <search> <flags>
  local out="$RES/bench_${CFG}_$1.json"
  [ -s "$out" ] && { echo "SPR FWD ITER SKIP bench $1"; return 0; }
  local t0=$SECONDS
  python -m spr.fwd.arena bench --name "fwd_${CFG}_it${K}_$1" --ckpt "$NEW_CKPT" \
      --config "$CFG" --instances "$EXAM" --search "$2" --flags "$3" \
      --expansions 1200 --k 5 --width "$WIDTH" --threads 2 --device "$DEV" \
      --run-root "$OUT_ROOT/runs" --out "$out" \
    || { echo "SPR FWD ITER FAILED bench $1"; return 1; }
  echo "SPR FWD ITER bench $1 took $((SECONDS-t0))s"
}
bench_one astar astar "" || exit 1
bench_one mcts  mcts  "$MCTS_FLAGS" || exit 1

# ---- D gate -------------------------------------------------------------------
for tag in astar mcts; do
  a="$RES/bench_${CFG}_${tag}.json"; [ -s "$a" ] || continue
  python -m spr.fwd.gate compare --a "$a" --b "$REFJ" --b-system "$REF_SYS" \
      --out "$RES/gate_${tag}_vs_supervised.json" >/dev/null \
    || echo "SPR FWD ITER WARN gate $tag vs supervised failed"
  prev="$PREV_BENCH"; [ -n "$prev" ] || prev="$PREV_RES/bench_${CFG}_${tag}.json"
  if [ -s "$prev" ]; then
    python -m spr.fwd.gate compare --a "$a" --b "$prev" \
        --out "$RES/gate_${tag}_vs_prev.json" >/dev/null \
      || echo "SPR FWD ITER WARN gate $tag vs prev failed"
  fi
done
for f in "$RES"/gate_*.json; do
  [ -s "$f" ] && echo "SPR FWD ITER gate $(basename "$f"): $(tr -d '\n' < "$f" | cut -c1-400)"
done
echo "SPR FWD ITER manifest: $(tr -d '\n' < "$IT/records.jsonl.manifest.json" | cut -c1-700)"
echo "SPR FWD ITER DONE $CFG k=$K net=$NEW_CKPT $(date +%F' '%T)"
