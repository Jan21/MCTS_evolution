#!/bin/bash
# Variants lab -- ONE matched iteration of variant VID (variants/DESIGN.md):
#   gen (g24r4, B2, frozen seed nets, per-variant fresh boards ids 40000+)
#   -> warm retrain (or cold if COLD=1) -> arena A* benches on graded /
#   frontier / UNSEEN exams -> paired gates vs v00_control (if its files exist).
#   sbatch -J spr-var-<VID> self_play_robots/jobs/variant_iter.slurm <VID> [SEED]
# Knobs come from `python -m variants describe <VID> --format env`; job env can
# still override (VAR=.. sbatch ...). Idempotent (DONE markers under runs/).
# Cost: ~2-3 h on 1 GPU = 0.25-0.4 nh per arm.
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
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
VID="$1"; SEED="${2:-7}"
eval "$(python -m variants describe "$VID" --format env)"
COLD=${COLD:-0}
# frozen seed nets for EVERY arm (mix_b2mix_iter2 -- variants/DESIGN.md)
SEEDNETS="$REPO/self_play_robots/results/selfplay/mix_b2mix_iter2/nets.txt"
P0=$(grep '^policy=' "$SEEDNETS" | cut -d= -f2-); V0=$(grep '^value=' "$SEEDNETS" | cut -d= -f2-)
[ -s "$P0" ] && [ -s "$V0" ] || { echo "SPR VAR FAILED: seed nets missing"; exit 1; }
IT="$REPO/runs/spr/variants/$VID"; RES="$REPO/self_play_robots/results/variants/$VID"
mkdir -p "$IT" "$RES"
EXAM="$REPO/self_play_robots/results/variants/exam/g24r4_unseen.jsonl"
EXAMB="$REPO/self_play_robots/results/variants/exam/boards_g24r4"
echo "SPR VAR $VID start $(date +%F' '%T) job=$SLURM_JOB_ID ids=$GEN_IDS cold=$COLD hooks=$HOOKS"
nvidia-smi -L
phase() { [ -f "$IT/DONE.$1" ]; }
mark()  { touch "$IT/DONE.$1"; echo "SPR VAR PHASE DONE $1 $(date +%T)"; }

# ---- A generate ----------------------------------------------------------------
VFLAG=""; case ",$HOOKS," in *,selfplay,*) VFLAG="--variant $VID";; esac
if ! phase gen; then
  t0=$SECONDS
  SIBFLAG=""; [ "$SIB" = 1 ] && SIBFLAG="--complete-siblings --root-all"
  python -m spr.selfplay --config g24r4 --policy "$P0" --value "$V0" \
      --boards-dir "$REPO/runs/spr/boards/var_${VID}" --board-ids "$GEN_IDS" --vocab b2 \
      --per-board "$PER_BOARD" --seed "$SEED" --iter 1 --expansions "$EXP" --stop-after "$STOP" \
      --root-noise "$NOISE" $SIBFLAG --workers "$WORKERS" --threads 2 --device cuda \
      --timeout 900 --min-expansions "$MINEXP" --emit "$EMIT" --backup "$BACKUP" \
      $VFLAG $SELFPLAY_FLAGS --out "$IT/records.jsonl" || { echo "SPR VAR FAILED gen"; exit 1; }
  cp "$IT/records.jsonl.manifest.json" "$RES/generation.manifest.json"
  echo "SPR VAR gen took $((SECONDS-t0))s"; mark gen
fi
# ---- B train --------------------------------------------------------------------
SPLITS=$(python - "$IT/records.jsonl" <<'EOF'
import json, sys
ids = sorted({json.loads(l)["env_id"] for l in open(sys.argv[1])})
cut = max(1, int(round(len(ids) * 0.85)))
fmt = lambda xs: ",".join(map(str, xs))
print(f"g24r4:train={fmt(ids[:cut])},val={fmt(ids[cut:])}")
EOF
)
TVFLAG=""; case ",$HOOKS," in *,train,*) TVFLAG="--variant $VID";; esac
train_one() {  # train_one <policy|value> <init> <extra>
  local sys="$1" init="$2" extra="$3" dir="$IT/$1"
  if [ -f "$dir/DONE" ]; then echo "SPR VAR SKIP train $sys"; return 0; fi
  local resume=0; ls "$dir"/lightning_logs/version_*/checkpoints/last.ckpt >/dev/null 2>&1 && resume=1
  local initflag=""; [ "$COLD" = 1 ] || initflag="--init $init"
  local t0=$SECONDS
  RR_RESUME=$resume OMP_NUM_THREADS=16 python -m spr.train --system "$sys" \
      --data g24r4="$IT/records.jsonl" --splits "$SPLITS" $initflag \
      --epochs "$EP" --lr "$LR" $extra $TVFLAG $TRAIN_FLAGS --torch-seed $((100 + SEED)) \
      --num-workers 4 --device cuda --out-dir "$dir" || { echo "SPR VAR FAILED train $sys"; return 1; }
  touch "$dir/DONE"; echo "SPR VAR train $sys took $((SECONDS-t0))s"
}
PE_EXTRA="--batch-size 8 --grad-clip 1.0 --byref"
VE_EXTRA="--batch-size 4 --max-per-group 8"
[ "$COLD" = 1 ] && { VE_EXTRA="$VE_EXTRA --warmup 4"; LC=3e-4; } || LC="$LR"
train_one policy "$P0" "$PE_EXTRA" || exit 1
LR="$LC" train_one value "$V0" "$VE_EXTRA" || exit 1
best_ckpt() { ls -t "$1"/lightning_logs/version_*/checkpoints/epoch=*.ckpt 2>/dev/null | head -1; }
BP=$(best_ckpt "$IT/policy"); BV=$(best_ckpt "$IT/value")
[ -n "$BP" ] && [ -n "$BV" ] || { echo "SPR VAR FAILED: no checkpoints"; exit 1; }
printf 'policy=%s\nvalue=%s\n' "$BP" "$BV" > "$RES/nets.txt"
echo "SPR VAR nets policy=$BP value=$BV"
# ---- C bench: graded / frontier / unseen (arena A*, B2 convention) ---------------
BVFLAG=""; case ",$HOOKS," in *,bench,*) BVFLAG="--variant $VID";; esac
bench_one() {  # bench_one <tag> <instances> <extra-arena-args...>
  local tag="$1" inst="$2"; shift 2
  local out="$RES/bench_${tag}_astar.json"
  [ -s "$out" ] && { echo "SPR VAR SKIP bench $tag"; return 0; }
  local t0=$SECONDS
  python -m spr.arena bench --arm "var_${VID}_${tag}" --config g24r4 --instances "$inst" \
      --policy "$BP" --value "$BV" --driver spr \
      --flags "--search arena_astar --vocab b2 --anytime $BVFLAG $BENCH_FLAGS" \
      --width 8 --threads 2 --device cuda "$@" --out "$out" \
      || { echo "SPR VAR FAILED bench $tag"; return 1; }
  echo "SPR VAR bench $tag took $((SECONDS-t0))s"
}
bench_one graded  scaling/data/g24r4/bench.solved.jsonl   || exit 1
bench_one frontier scaling/data/g24r4/bench.unsolved.jsonl || true
bench_one unseen  "$EXAM" --env-dir "$EXAMB" || true
# ---- D gates vs control -----------------------------------------------------------
CTRL="$REPO/self_play_robots/results/variants/v00_control"
if [ "$VID" != v00_control ]; then
  for t in graded frontier unseen; do
    [ -s "$RES/bench_${t}_astar.json" ] && [ -s "$CTRL/bench_${t}_astar.json" ] || continue
    python -m spr.gate compare --a "$RES/bench_${t}_astar.json" --b "$CTRL/bench_${t}_astar.json" \
        > "$RES/gate_${t}_vs_control.json" || true
    echo "SPR VAR gate $t: $(tr -d '\n' < "$RES/gate_${t}_vs_control.json" | cut -c1-260)"
  done
fi
echo "SPR VAR DONE $VID $(date +%F' '%T)"
