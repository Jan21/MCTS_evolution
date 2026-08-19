#!/bin/bash
# M5 curriculum -- ONE mixed-size B2 self-play iteration K (PROBLEM.md 8/M5:
# "curriculum to g32r4 (and the g24r8 hard mode)"). Same phases as
# selfplay_iter.slurm (B2 vocabulary throughout, DESIGN.md 4b) but generation,
# buffer, training and benches span CFGS (default "g24r4 g32r4 g24r8"): one
# size-free pair is trained on the union of the per-config buffers (spr.train
# size-bucketed batches, --batch-ref-n 24 shrinks batches above 24x24 so the
# 24x24 memory envelope of FINDINGS 63 holds at 32x32) and benched with the
# arena A* (B2 convention: anytime, no prefix-check) on every config's graded and
# frontier exam. MCTS rows are NOT run in-loop (frontier MCTS costs 1-2 h per
# exam) -- bench the final nets with jobs/bench_pair.slurm.
#
#   sbatch -J spr-mix-it1 -t 08:00:00 self_play_robots/jobs/selfplay_mix_iter.slurm 1 <P.ckpt> <V.ckpt>
#   sbatch -J spr-mix-it2 -t 08:00:00 --dependency=afterany:<jid> self_play_robots/jobs/selfplay_mix_iter.slurm 2 auto auto
# Knobs (env): CFGS BOARDS (30 per config) PER_BOARD (8) EXP (300) STOP (80)
#   WORKERS (6; 32x32 uses WORKERS32=4) WINDOW (3) EP_P/EP_V (6) LR (1e-4) NOISE (0.25)
#   FRONTIER (1: also bench bench.unsolved of every config) ID_BASE (9000)
# Cost basis (measured): gen ~2-2.5 h, train ~0.5 h, benches ~2.7 h -> ~6 h on 1 GPU
# = 0.75 nh per iteration.
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
K="$1"; PPREV="$2"; VPREV="$3"
CFGS=${CFGS:-"g24r4 g32r4 g24r8"}
BOARDS=${BOARDS:-30}; PER_BOARD=${PER_BOARD:-8}; EXP=${EXP:-300}; STOP=${STOP:-80}
WORKERS=${WORKERS:-6}; WORKERS32=${WORKERS32:-4}; WINDOW=${WINDOW:-3}; EP_P=${EP_P:-6}; EP_V=${EP_V:-6}
LR=${LR:-1e-4}; NOISE=${NOISE:-0.25}; FRONTIER=${FRONTIER:-1}; ID_BASE=${ID_BASE:-9000}
TAG="_b2mix"
IT="$REPO/runs/spr/selfplay/mix${TAG}_iter${K}"; RES="$REPO/self_play_robots/results/selfplay/mix${TAG}_iter${K}"
mkdir -p "$IT" "$RES" "$REPO/runs/spr/boards"
ID0=$((ID_BASE + BOARDS * (K - 1))); ID1=$((ID0 + BOARDS - 1))
if [ "$PPREV" = auto ]; then
  PREVRES="$REPO/self_play_robots/results/selfplay/mix${TAG}_iter$((K-1))"
  PPREV=$(grep '^policy=' "$PREVRES/nets.txt" | cut -d= -f2-); VPREV=$(grep '^value=' "$PREVRES/nets.txt" | cut -d= -f2-)
  [ -s "$PPREV" ] && [ -s "$VPREV" ] || { echo "SPR MIX FAILED: auto nets missing in $PREVRES"; exit 1; }
fi
echo "SPR MIX k=$K cfgs=[$CFGS] start $(date +%F' '%T) job=$SLURM_JOB_ID prev=$PPREV|$VPREV boards=$ID0-$ID1"
nvidia-smi -L
phase() { [ -f "$IT/DONE.$1" ]; }
mark()  { touch "$IT/DONE.$1"; echo "SPR MIX PHASE DONE $1 $(date +%T)"; }

# ---- A generate (per config, sequential on the one GPU) -----------------------
for CFG in $CFGS; do
  if ! phase gen_$CFG; then
    t0=$SECONDS
    W=$WORKERS; case "$CFG" in g32*|g4*|g5*|g6*) W=$WORKERS32;; esac
    mkdir -p "$IT/$CFG"
    python -m spr.selfplay --config "$CFG" --policy "$PPREV" --value "$VPREV" \
        --boards-dir "$REPO/runs/spr/boards/${CFG}${TAG}_iter${K}" --board-ids "$ID0-$ID1" --vocab b2 \
        --per-board "$PER_BOARD" --seed $((100 + K)) --iter "$K" --expansions "$EXP" --stop-after "$STOP" \
        --root-noise "$NOISE" --complete-siblings --root-all --workers "$W" --threads 2 --device cuda \
        --timeout 900 --out "$IT/$CFG/records.jsonl" || { echo "SPR MIX FAILED gen $CFG"; exit 1; }
    cp "$IT/$CFG/records.jsonl.manifest.json" "$RES/generation_${CFG}.manifest.json"
    echo "SPR MIX gen $CFG took $((SECONDS-t0))s"; mark gen_$CFG
  fi
done
# ---- C buffer (per config; window over this loop's iterations) ---------------
DATA=""; SPLITS=""
for CFG in $CFGS; do
  # spr.buffer expects <sp-root>/<cfg><tag>_iter<k>/records.jsonl -> point it at a
  # per-config view of this loop's iteration dirs (symlinks, idempotent)
  for k in $(seq 1 "$K"); do
    src="$REPO/runs/spr/selfplay/mix${TAG}_iter${k}/$CFG"; dst="$REPO/runs/spr/selfplay/${CFG}${TAG}_iter${k}"
    [ -d "$src" ] && [ ! -e "$dst" ] && ln -s "$src" "$dst"
  done
  if ! phase buffer_$CFG; then
    python -m spr.buffer --config "$CFG" --sp-root "$REPO/runs/spr/selfplay" --upto "$K" \
        --window "$WINDOW" --tag "$TAG" --out "$IT/buffer_${CFG}.jsonl" || { echo "SPR MIX FAILED buffer $CFG"; exit 1; }
    mark buffer_$CFG
  fi
  DATA="$DATA --data ${CFG}=$IT/buffer_${CFG}.jsonl"
  SPLITS="$SPLITS --splits $(cat "$IT/buffer_${CFG}.jsonl.splits.txt")"
done
echo "SPR MIX data: $DATA"; echo "SPR MIX splits: $SPLITS"
# ---- D/E train ----------------------------------------------------------------
train_one() {  # train_one <policy|value> <init> <epochs>
  local sys="$1" init="$2" ep="$3" dir="$IT/$1"
  if [ -f "$dir/DONE" ]; then echo "SPR MIX SKIP train $sys"; return 0; fi
  local resume=0; ls "$dir"/lightning_logs/version_*/checkpoints/last.ckpt >/dev/null 2>&1 && resume=1
  local extra=""
  [ "$sys" = value ] && extra="--batch-size 4 --max-per-group 8 --batch-ref-n 24"
  [ "$sys" = policy ] && extra="--batch-size 8 --grad-clip 1.0 --byref --batch-ref-n 24"
  local t0=$SECONDS
  RR_RESUME=$resume OMP_NUM_THREADS=16 python -m spr.train --system "$sys" $DATA $SPLITS \
      --init "$init" --epochs "$ep" --lr "$LR" $extra --torch-seed $((121 + K)) --num-workers 4 \
      --device cuda --out-dir "$dir" || { echo "SPR MIX FAILED train $sys"; return 1; }
  touch "$dir/DONE"; echo "SPR MIX train $sys took $((SECONDS-t0))s"
}
train_one policy "$PPREV" "$EP_P" || exit 1
train_one value  "$VPREV" "$EP_V" || exit 1
best_ckpt() { ls -t "$1"/lightning_logs/version_*/checkpoints/epoch=*.ckpt 2>/dev/null | head -1; }
BP=$(best_ckpt "$IT/policy"); BV=$(best_ckpt "$IT/value")
[ -n "$BP" ] && [ -n "$BV" ] || { echo "SPR MIX FAILED: no checkpoints"; exit 1; }
printf 'policy=%s\nvalue=%s\n' "$BP" "$BV" > "$IT/nets.txt"; cp "$IT/nets.txt" "$RES/nets.txt"
echo "SPR MIX nets policy=$BP value=$BV"
# ---- F bench (arena A*, B2 convention) -----------------------------------------
bench_one() {  # bench_one <cfg> <tag> <flags> <instances>
  local out="$RES/bench_$1_$2.json"
  [ -s "$out" ] && { echo "SPR MIX SKIP bench $1 $2"; return 0; }
  local t0=$SECONDS
  python -m spr.arena bench --arm "mix_it${K}_$1_$2" --config "$1" --instances "$4" \
      --policy "$BP" --value "$BV" --driver spr --flags "$3" --width 8 --threads 2 --device cuda \
      --out "$out" || { echo "SPR MIX FAILED bench $1 $2"; return 1; }
  echo "SPR MIX bench $1 $2 took $((SECONDS-t0))s"
}
VF="--vocab b2 --anytime"
for CFG in $CFGS; do bench_one "$CFG" astar "--search arena_astar $VF" "scaling/data/$CFG/bench.solved.jsonl" || true; done
if [ "$FRONTIER" = 1 ]; then
  for CFG in $CFGS; do bench_one "$CFG" frontier_astar "--search arena_astar $VF" "scaling/data/$CFG/bench.unsolved.jsonl" || true; done
fi
# ---- G gate: paired vs the previous mix iteration (or the seed nets' rows) --------
if [ "$K" -gt 1 ]; then
  PREVRES="$REPO/self_play_robots/results/selfplay/mix${TAG}_iter$((K-1))"
  for CFG in $CFGS; do for t in astar frontier_astar; do
    [ -s "$RES/bench_${CFG}_$t.json" ] && [ -s "$PREVRES/bench_${CFG}_$t.json" ] || continue
    python -m spr.gate compare --a "$RES/bench_${CFG}_$t.json" --b "$PREVRES/bench_${CFG}_$t.json" > "$RES/gate_${CFG}_${t}_vs_prev.json" || true
    echo "SPR MIX gate $CFG $t vs prev: $(tr -d '\n' < "$RES/gate_${CFG}_${t}_vs_prev.json" | cut -c1-300)"
  done; done
fi
echo "SPR MIX DONE k=$K $(date +%F' '%T)"
