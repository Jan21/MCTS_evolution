#!/bin/bash
# ONE self-play iteration k for the PRIMITIVE-MOVE arm (PROBLEM.md section 5,
# mirroring jobs/selfplay_iter.slurm for the subgoal arm):
#   A generate: fresh 16x16 lean boards (ids >= 5000) -> PUCT MCTS with net k-1
#               -> replayed paths -> ExIt records (value = remaining certified
#               path length, policy = committed move, + mcts_visits)
#   B train:    MoveNet warm-started from net k-1 on the replay window
#               (+ optional exact anchor corpus), val split BY BOARD ID,
#               ModelCheckpoint on the val metric, save_last, fixed torch seed
#   C bench:    arena (1200 expansions / k=5, 8-wide, replay-certified) on the
#               pinned g16r4 exam with astar AND mcts best-at-budget
#   D gate:     paired vs iteration k-1 (McNemar + moves sign test) and vs the
#               FROZEN forward baseline row in eval/results/comparison_forward.json
# Idempotent per phase (DONE sentinels); a walltime kill resumes on resubmit.
#
#   sbatch -J spr-fwd-it1 self_play_robots/jobs/fwd_iter.slurm <K> <CKPT_PREV> [PREV_BENCH.json]
# Knobs via env: BOARDS (120) PER_BOARD (8) EXP (600) STOP (150) NOISE (0.25)
#   WORKERS (8) WINDOW (3) ANCHOR (move_planner/data/moves_scored.jsonl; "" = none)
#   ANCHOR_LIMIT (200000) EPOCHS (6) LR (1e-4) BATCH (256) DSTAR (diag|off)
#   OUT_ROOT (runs/spr/fwd) WIDTH (8) DEV (cuda)
#
# Cost basis (to be measured on iteration 1): generation 120 boards x 8 instances
# at <=600 expansions, 8 workers on 1 GPU ~ 1-2 h; training 6 epochs on
# ~10k self-play + 200k anchor records ~ 1 h; two 450-instance benches
# (astar ~1 min, mcts best-at-budget <= ~31 min) ~ 0.5 h  ->  ~3 h on 1 GPU
# = ~0.4 node-hours per iteration; the 12 h walltime cap bounds it at 1.5 nh.
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
K="$1"; CKPT_PREV="$2"; PREV_BENCH="${3:-}"
BOARDS=${BOARDS:-120}; PER_BOARD=${PER_BOARD:-8}; EXP=${EXP:-600}; STOP=${STOP:-150}
NOISE=${NOISE:-0.25}; WORKERS=${WORKERS:-8}; WINDOW=${WINDOW:-3}
ANCHOR=${ANCHOR-move_planner/data/moves_scored.jsonl}; ANCHOR_LIMIT=${ANCHOR_LIMIT:-200000}
EPOCHS=${EPOCHS:-6}; LR=${LR:-1e-4}; BATCH=${BATCH:-256}; DSTAR=${DSTAR:-diag}
OUT_ROOT="${OUT_ROOT:-$REPO/runs/spr/fwd}"; WIDTH=${WIDTH:-8}; DEV=${DEV:-cuda}
IT="$OUT_ROOT/selfplay/g16r4_iter${K}"
mkdir -p "$REPO/runs/spr" "$IT" "$OUT_ROOT/boards"
ID0=$((5000 + BOARDS * (K - 1))); ID1=$((ID0 + BOARDS - 1))
echo "SPR FWD ITER k=$K start $(date +%F' '%T) job=${SLURM_JOB_ID:-none} prev=$CKPT_PREV boards=$ID0-$ID1"
nvidia-smi -L 2>/dev/null || true
phase() { [ -f "$IT/DONE.$1" ]; }
mark()  { touch "$IT/DONE.$1"; echo "SPR FWD ITER PHASE DONE $1 $(date +%T)"; }

# ---- A generate ---------------------------------------------------------------
if ! phase gen; then
  t0=$SECONDS
  python -m spr.fwd.selfplay --config g16r4 --ckpt "$CKPT_PREV" \
      --boards-dir "$OUT_ROOT/boards/g16r4_iter${K}" --board-ids "$ID0-$ID1" \
      --per-board "$PER_BOARD" --seed "$K" --iter "$K" \
      --expansions "$EXP" --k 5 --stop-after "$STOP" --root-noise "$NOISE" --root-all \
      --sampler mix --d-star "$DSTAR" --workers "$WORKERS" --threads 2 \
      --device "$DEV" --timeout 600 --out "$IT/records.jsonl" \
    || { echo "SPR FWD ITER FAILED gen"; exit 1; }
  echo "SPR FWD ITER gen took $((SECONDS-t0))s"; mark gen
fi

# ---- B train (replay window = the last WINDOW iterations' records) -------------
RECARGS=()
for j in $(seq $((K - WINDOW + 1)) "$K"); do
  [ "$j" -ge 1 ] || continue
  f="$OUT_ROOT/selfplay/g16r4_iter${j}/records.jsonl"
  [ -s "$f" ] && RECARGS+=(--records "$f")
done
[ ${#RECARGS[@]} -gt 0 ] || { echo "SPR FWD ITER FAILED: no records in the window"; exit 1; }
ANCARGS=(); [ -n "$ANCHOR" ] && ANCARGS=(--anchor "$ANCHOR" --anchor-limit "$ANCHOR_LIMIT")
TRAIN_DIR="$IT/train"
if ! phase train; then
  t0=$SECONDS
  RR_RESUME=$([ -d "$TRAIN_DIR/lightning_logs" ] && echo 1 || echo 0) \
  python -m spr.fwd.train "${RECARGS[@]}" "${ANCARGS[@]}" --init "$CKPT_PREV" \
      --epochs "$EPOCHS" --lr "$LR" --batch-size "$BATCH" --num-workers 4 \
      --val-every 10 --torch-seed $((21 + K)) --device "$DEV" --out-dir "$TRAIN_DIR" \
    || { echo "SPR FWD ITER FAILED train"; exit 1; }
  echo "SPR FWD ITER train took $((SECONDS-t0))s"; mark train
fi
NEW_CKPT=$(cat "$TRAIN_DIR/BEST.txt" 2>/dev/null)
[ -s "$NEW_CKPT" ] || { echo "SPR FWD ITER FAILED: no checkpoint in $TRAIN_DIR"; exit 1; }
echo "SPR FWD ITER net k=$K -> $NEW_CKPT"

# ---- C bench ------------------------------------------------------------------
bench_one() {  # bench_one <tag> <search> <flags>
  local out="$IT/bench_g16r4_$1.json"
  [ -s "$out" ] && { echo "SPR FWD ITER SKIP bench $1"; return 0; }
  local t0=$SECONDS
  python -m spr.fwd.arena bench --name "fwd_it${K}_$1" --ckpt "$NEW_CKPT" \
      --instances eval/data/bench450.jsonl --search "$2" --flags "$3" \
      --expansions 1200 --k 5 --width "$WIDTH" --threads 2 --device "$DEV" \
      --run-root "$OUT_ROOT/runs" --out "$out" \
    || { echo "SPR FWD ITER FAILED bench $1"; return 1; }
  echo "SPR FWD ITER bench $1 took $((SECONDS-t0))s"
}
bench_one astar astar "" || exit 1
bench_one mcts  mcts  "--mcts-backup min --best-at-budget" || exit 1

# ---- D gate -------------------------------------------------------------------
REFJ="$REPO/supervised_valuenet/eval/results/comparison_forward.json"
REF_SYS="${REF_SYS:-move_planner/checkpoints/candidate_scored.ckpt}"
for tag in astar mcts; do
  a="$IT/bench_g16r4_${tag}.json"; [ -s "$a" ] || continue
  python -m spr.fwd.gate compare --a "$a" --b "$REFJ" --b-system "$REF_SYS" \
      --out "$IT/gate_${tag}_vs_supervised.json" >/dev/null \
    || echo "SPR FWD ITER WARN gate $tag vs supervised failed"
  if [ -n "$PREV_BENCH" ] && [ -s "$PREV_BENCH" ]; then
    python -m spr.fwd.gate compare --a "$a" --b "$PREV_BENCH" \
        --out "$IT/gate_${tag}_vs_prev.json" >/dev/null \
      || echo "SPR FWD ITER WARN gate $tag vs prev failed"
  fi
done
for f in "$IT"/gate_*.json; do
  [ -s "$f" ] && echo "SPR FWD ITER gate $(basename "$f"): $(tr -d '\n' < "$f" | cut -c1-400)"
done
echo "SPR FWD ITER manifest: $(tr -d '\n' < "$IT/records.jsonl.manifest.json" | cut -c1-600)"
echo "SPR FWD ITER DONE k=$K net=$NEW_CKPT $(date +%F' '%T)"
