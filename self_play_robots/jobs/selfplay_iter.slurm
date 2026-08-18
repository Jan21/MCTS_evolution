#!/bin/bash
# ONE self-play iteration k (PROBLEM.md section 5, "Iteration k"):
#   A generate: fresh lean boards -> MCTS with nets k-1 -> certified records
#   B gauge:    depth-0 argmin agreement vs the exact engine (section 4.1)
#   C buffer:   last WINDOW iterations' records (+ optional exact anchor)
#   D/E train:  policy + value warm-started from nets k-1 (CollapseStop etc.)
#   F bench:    arena (1200/k5, prefix-check, replay-certified) on the pinned
#               g24r4 exam with arena_astar AND mcts-best-at-budget
#   G gate:     paired vs iteration k-1 (McNemar) + vs the supervised baseline
# Idempotent per phase (DONE sentinels); a walltime kill resumes on resubmit.
#
#   sbatch -J spr-it1 self_play_robots/jobs/selfplay_iter.slurm g24r4 1 <POLICY.ckpt> <VALUE.ckpt> [PREV_BENCH.json]
# Knobs via env (sbatch --export=... is avoided per house rule; pass via a wrapper
# `env VAR=.. sbatch`? no -- Slurm copies the environment by default): BOARDS (120)
# PER_BOARD (8) EXP (600) STOP (150) WORKERS (8) MINEXP (0)
#   WINDOW (3) ANCHOR (1: add the exact g24r4+g16r4 corpora to training)
#   EP_P (6) EP_V (6) LR (1e-4) NOISE (0.25) SIB (1: complete siblings)
# Cost basis (to be measured on iteration 1): generation ~2-4 h, training
# ~3-4 h, benches ~1 h on 1 GPU  ->  ~1-1.5 nh per iteration.
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
CFG="$1"; K="$2"; PPREV="$3"; VPREV="$4"; PREV_BENCH="${5:-}"
BOARDS=${BOARDS:-120}; PER_BOARD=${PER_BOARD:-8}; EXP=${EXP:-600}; STOP=${STOP:-150}
WORKERS=${WORKERS:-8}; WINDOW=${WINDOW:-3}; ANCHOR=${ANCHOR:-1}; EP_P=${EP_P:-6}; EP_V=${EP_V:-6}
LR=${LR:-1e-4}; NOISE=${NOISE:-0.25}; SIB=${SIB:-1}; MINEXP=${MINEXP:-0}   # MINEXP>0 = hard-instance focus
VOCAB=${VOCAB:-base}          # base | b2 (extended language; FINDINGS §11): separate
                              # run/result dirs, board ids from 8000, b2 arena flags,
                              # policy trained with --byref, NO base-vocab anchors
TAGV=""; ID_BASE=5000; [ "$VOCAB" = b2 ] && { TAGV="_b2"; ID_BASE=8000; ANCHOR=0; }
IT="$REPO/runs/spr/selfplay/${CFG}${TAGV}_iter${K}"; RES="$REPO/self_play_robots/results/selfplay/${CFG}${TAGV}_iter${K}"
mkdir -p "$IT" "$RES" "$REPO/runs/spr/boards"
ID0=$((ID_BASE + BOARDS * (K - 1))); ID1=$((ID0 + BOARDS - 1))
if [ "$PPREV" = auto ]; then    # chain: nets + reference bench of iteration K-1
  PREVRES="$REPO/self_play_robots/results/selfplay/${CFG}${TAGV}_iter$((K-1))"
  PPREV=$(grep '^policy=' "$PREVRES/nets.txt" | cut -d= -f2-); VPREV=$(grep '^value=' "$PREVRES/nets.txt" | cut -d= -f2-)
  [ -n "$PREV_BENCH" ] || PREV_BENCH="$PREVRES/bench_${CFG}_astar.json"
  [ -s "$PPREV" ] && [ -s "$VPREV" ] || { echo "SPR ITER FAILED: auto nets missing in $PREVRES"; exit 1; }
fi
echo "SPR ITER $CFG k=$K start $(date +%F' '%T) job=$SLURM_JOB_ID prev=$PPREV|$VPREV boards=$ID0-$ID1"
nvidia-smi -L
phase() { [ -f "$IT/DONE.$1" ]; }
mark()  { touch "$IT/DONE.$1"; echo "SPR ITER PHASE DONE $1 $(date +%T)"; }

# ---- A generate --------------------------------------------------------------
if ! phase gen; then
  t0=$SECONDS
  SIBFLAG=""; [ "$SIB" = 1 ] && SIBFLAG="--complete-siblings --root-all"
  python -m spr.selfplay --config "$CFG" --policy "$PPREV" --value "$VPREV" \
      --boards-dir "$REPO/runs/spr/boards/${CFG}${TAGV}_iter${K}" --board-ids "$ID0-$ID1" --vocab "$VOCAB" \
      --per-board "$PER_BOARD" --seed "$K" --iter "$K" --expansions "$EXP" --stop-after "$STOP" \
      --root-noise "$NOISE" $SIBFLAG --workers "$WORKERS" --threads 2 --device cuda \
      --timeout 600 --min-expansions "$MINEXP" --out "$IT/records.jsonl" || { echo "SPR ITER FAILED gen"; exit 1; }
  cp "$IT/records.jsonl.manifest.json" "$RES/generation.manifest.json"
  echo "SPR ITER gen took $((SECONDS-t0))s"; mark gen
fi
# ---- B gauge ------------------------------------------------------------------
if ! phase gauge; then
  GC="--sample 200 --time-cap 2400"   # engine budget 100k iters/rollout (spr.gauge default); cap = safety net
  python -m spr.gauge --config "$CFG" --records "$IT/records.jsonl" $GC --threads 16 \
      --vocab "$VOCAB" --out "$RES/gauge.json" || echo "SPR ITER WARN gauge failed (non-fatal)"
  mark gauge
fi
# ---- C buffer -----------------------------------------------------------------
if ! phase buffer; then
  python -m spr.buffer --config "$CFG" --sp-root "$REPO/runs/spr/selfplay" --upto "$K" \
      --window "$WINDOW" --tag "$TAGV" --out "$IT/buffer.jsonl" || { echo "SPR ITER FAILED buffer"; exit 1; }
  mark buffer
fi
SPLITS=$(cat "$IT/buffer.jsonl.splits.txt")
DATA="--data ${CFG}=$IT/buffer.jsonl"
if [ "$ANCHOR" = 1 ]; then
  # exact anchors keep the loop from forgetting the supervised distribution;
  # their standard splits are merged into the config's split spec
  DATA="$DATA --data g24r4=scaling/data/g24r4/backward.jsonl --data g16r4=nn/data/combined.jsonl"
  # SPLITS is "<cfg>:train=A,val=B" over self-play ids; add the standard ranges
  SPLITS=$(python - "$SPLITS" "$CFG" <<'EOF'
import sys
from scaling.configs import get
spec, cfg = sys.argv[1], sys.argv[2]
c = get(cfg)
name, rest = spec.split(":", 1)
parts = dict(kv.split("=", 1) for kv in rest.replace(",val=", "|val=").split("|"))
tr = c.board_ranges["train"] + "," + parts["train"]
va = c.board_ranges["val"] + "," + parts["val"]
print(f"{name}:train={tr},val={va}")
EOF
)
fi
echo "SPR ITER splits: $SPLITS"
# ---- D/E train ----------------------------------------------------------------
train_one() {  # train_one <policy|value> <init> <epochs>
  local sys="$1" init="$2" ep="$3" dir="$IT/$1"
  if [ -f "$dir/DONE" ]; then echo "SPR ITER SKIP train $sys"; return 0; fi
  local resume=0; ls "$dir"/lightning_logs/version_*/checkpoints/last.ckpt >/dev/null 2>&1 && resume=1
  local extra=""
  [ "$sys" = value ] && extra="--batch-size 4 --max-per-group 8"
  [ "$sys" = policy ] && extra="--batch-size 8 --grad-clip 1.0"     # FINDINGS §4 recipe
  [ "$sys" = policy ] && [ "$VOCAB" = b2 ] && extra="$extra --byref"  # keep by-reference records
  local t0=$SECONDS
  RR_RESUME=$resume OMP_NUM_THREADS=16 python -m spr.train --system "$sys" $DATA --splits "$SPLITS" \
      --init "$init" --epochs "$ep" --lr "$LR" $extra --torch-seed $((21 + K)) --num-workers 4 \
      --device cuda --out-dir "$dir" || { echo "SPR ITER FAILED train $sys"; return 1; }
  touch "$dir/DONE"; echo "SPR ITER train $sys took $((SECONDS-t0))s"
}
train_one policy "$PPREV" "$EP_P" || exit 1
train_one value  "$VPREV" "$EP_V" || exit 1
best_ckpt() { ls -t "$1"/lightning_logs/version_*/checkpoints/epoch=*.ckpt 2>/dev/null | head -1; }
BP=$(best_ckpt "$IT/policy"); BV=$(best_ckpt "$IT/value")
[ -n "$BP" ] && [ -n "$BV" ] || { echo "SPR ITER FAILED: no checkpoints"; exit 1; }
printf 'policy=%s\nvalue=%s\n' "$BP" "$BV" > "$IT/nets.txt"; cp "$IT/nets.txt" "$RES/nets.txt"
echo "SPR ITER nets policy=$BP value=$BV"
# ---- F bench ------------------------------------------------------------------
INST=scaling/data/${CFG}/bench.solved.jsonl; FRONT=scaling/data/${CFG}/bench.unsolved.jsonl
[ "$CFG" = g16r4 ] && { INST=eval/data/bench450.jsonl; FRONT=""; }
bench_one() {  # bench_one <tag> <flags> <instances>
  local out="$RES/bench_${CFG}_$1.json"
  [ -s "$out" ] && { echo "SPR ITER SKIP bench $1"; return 0; }
  local t0=$SECONDS
  python -m spr.arena bench --arm "sp_${CFG}_it${K}_$1" --config "$CFG" --instances "$3" \
      --policy "$BP" --value "$BV" --driver spr --flags "$2" --width 8 --threads 2 --device cuda \
      --out "$out" || { echo "SPR ITER FAILED bench $1"; return 1; }
  echo "SPR ITER bench $1 took $((SECONDS-t0))s"
}
VF="--prefix-check"; [ "$VOCAB" = b2 ] && VF="--vocab b2 --anytime"    # arena's B2 convention: anytime + b2 + byref, NO prefix-check (prefix_key cannot sort park/by-ref plans)
bench_one astar "--search arena_astar $VF" "$INST" || exit 1
bench_one mcts  "--search mcts --best-at-budget $VF" "$INST" || exit 1
if [ -n "$FRONT" ]; then   # the frontier set is where the headroom is (FINDINGS §8)
  bench_one frontier_astar "--search arena_astar $VF" "$FRONT" || true
  bench_one frontier_mcts  "--search mcts --best-at-budget $VF" "$FRONT" || true
fi
# ---- G gate -------------------------------------------------------------------
REF=g24r4_exact_prefix; [ "$CFG" = g16r4 ] && REF=g16r4_v2_prefix
python -m spr.gate m1 --new "$RES/bench_${CFG}_astar.json" --ref-arm "$REF" --out "$RES/gate_vs_supervised.json" || true
if [ -n "$PREV_BENCH" ] && [ -s "$PREV_BENCH" ]; then
  python -m spr.gate compare --a "$RES/bench_${CFG}_astar.json" --b "$PREV_BENCH" > "$RES/gate_vs_prev.json" || true
  echo "SPR ITER gate vs prev: $(cat "$RES/gate_vs_prev.json" | tr -d '\n' | cut -c1-400)"
fi
echo "SPR ITER DONE $CFG k=$K $(date +%F' '%T)"
