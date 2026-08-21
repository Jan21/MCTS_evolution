#!/bin/bash
# Wave-3 chained variant: K matched iterations of ONE variant inside one job
# (net k trains on the window of records from iterations k-2..k; nets chain).
# Benches + gates vs v00_control run after the LAST iteration only.
#   sbatch -J spr-var-<VID>x3 self_play_robots/jobs/variant_chain.slurm <VID> [K=3] [SEED=7]
# Cost: K x (gen ~35-70 min + train ~15 min) + benches ~1.3 h -> <= 7 h.
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
VID="$1"; K="${2:-3}"; SEED="${3:-7}"
eval "$(python -m variants describe "$VID" --format env)"
SEEDNETS="$REPO/self_play_robots/results/selfplay/mix_b2mix_iter2/nets.txt"
P=$(grep '^policy=' "$SEEDNETS" | cut -d= -f2-); V=$(grep '^value=' "$SEEDNETS" | cut -d= -f2-)
IT="$REPO/runs/spr/variants/${VID}_chain"; RES="$REPO/self_play_robots/results/variants/${VID}_chain"
mkdir -p "$IT" "$RES"
EXAM="$REPO/self_play_robots/results/variants/exam/g24r4_unseen.jsonl"
EXAMB="$REPO/self_play_robots/results/variants/exam/boards_g24r4"
VFLAG=""; case ",$HOOKS," in *,selfplay,*) VFLAG="--variant $VID";; esac
TVFLAG=""; case ",$HOOKS," in *,train,*) TVFLAG="--variant $VID";; esac
echo "SPR VARCHAIN $VID K=$K start $(date +%F' '%T) job=$SLURM_JOB_ID"
nvidia-smi -L
IDS0=${GEN_IDS%%-*}
for k in $(seq 1 "$K"); do
  D="$IT/it$k"; mkdir -p "$D"
  LO=$((IDS0 + 5000 + 100 * (k - 1))); HI=$((LO + BOARDS - 1))   # +5000: clear of the one-shot arm's ids
  if [ ! -f "$D/DONE.gen" ]; then
    SIBFLAG=""; [ "$SIB" = 1 ] && SIBFLAG="--complete-siblings --root-all"
    python -m spr.selfplay --config g24r4 --policy "$P" --value "$V" \
        --boards-dir "$REPO/runs/spr/boards/var_${VID}_chain_it$k" --board-ids "$LO-$HI" --vocab b2 \
        --per-board "$PER_BOARD" --seed $((SEED + k)) --iter "$k" --expansions "$EXP" --stop-after "$STOP" \
        --root-noise "$NOISE" $SIBFLAG --workers "$WORKERS" --threads 2 --device cuda \
        --timeout 900 --min-expansions "$MINEXP" --emit "$EMIT" --backup "$BACKUP" \
        $VFLAG $SELFPLAY_FLAGS --out "$D/records.jsonl" || { echo "SPR VARCHAIN FAILED gen it$k"; exit 1; }
    cp "$D/records.jsonl.manifest.json" "$RES/generation_it$k.manifest.json"
    touch "$D/DONE.gen"
  fi
  # window: this + up to 2 previous iterations
  BUF="$D/buffer.jsonl"; : > "$BUF"
  for j in $(seq $((k > 2 ? k - 2 : 1)) "$k"); do cat "$IT/it$j/records.jsonl" >> "$BUF"; done
  SPLITS=$(python - "$BUF" <<'PYEOF'
import json, sys
ids = sorted({json.loads(l)["env_id"] for l in open(sys.argv[1])})
cut = max(1, int(round(len(ids) * 0.85)))
fmt = lambda xs: ",".join(map(str, xs))
print(f"g24r4:train={fmt(ids[:cut])},val={fmt(ids[cut:])}")
PYEOF
)
  for sys_ in policy value; do
    DIR="$D/$sys_"
    [ -f "$DIR/DONE" ] && continue
    extra="--batch-size 8 --grad-clip 1.0 --byref"; [ "$sys_" = value ] && extra="--batch-size 4 --max-per-group 8"
    RR_RESUME=0 OMP_NUM_THREADS=16 python -m spr.train --system "$sys_" \
        --data g24r4="$BUF" --splits "$SPLITS" --init "$([ "$sys_" = policy ] && echo "$P" || echo "$V")" \
        --epochs "$EP" --lr "$LR" $extra $TVFLAG --torch-seed $((100 + SEED + k)) \
        --num-workers 4 --device cuda --out-dir "$DIR" || { echo "SPR VARCHAIN FAILED train $sys_ it$k"; exit 1; }
    touch "$DIR/DONE"
  done
  P=$(ls -t "$D"/policy/lightning_logs/version_*/checkpoints/epoch=*.ckpt | head -1)
  V=$(ls -t "$D"/value/lightning_logs/version_*/checkpoints/epoch=*.ckpt | head -1)
  echo "SPR VARCHAIN it$k nets $P | $V"
done
printf 'policy=%s\nvalue=%s\n' "$P" "$V" > "$RES/nets.txt"
# ---- final benches + gates vs the one-shot control ---------------------------------
bench_one() {
  local tag="$1" inst="$2" xtra="$3"; shift 3
  local out="$RES/bench_${tag}_astar.json"
  [ -s "$out" ] && return 0
  python -m spr.arena bench --arm "varchain_${VID}_${tag}" --config g24r4 --instances "$inst" \
      --policy "$P" --value "$V" --driver spr \
      --flags "--search arena_astar --vocab b2 --anytime $xtra" \
      --width 8 --threads 2 --device cuda "$@" --out "$out" || echo "SPR VARCHAIN FAILED bench $tag"
}
bench_one graded scaling/data/g24r4/bench.solved.jsonl ""
bench_one frontier scaling/data/g24r4/bench.unsolved.jsonl ""
bench_one unseen "$EXAM" "--boards lean" --env-dir "$EXAMB"
CTRL="$REPO/self_play_robots/results/variants/v00_control"
for t in graded frontier unseen; do
  [ -s "$RES/bench_${t}_astar.json" ] && [ -s "$CTRL/bench_${t}_astar.json" ] || continue
  python -m spr.gate compare --a "$RES/bench_${t}_astar.json" --b "$CTRL/bench_${t}_astar.json" \
      > "$RES/gate_${t}_vs_control.json" || true
  echo "SPR VARCHAIN gate $t: $(tr -d '\n' < "$RES/gate_${t}_vs_control.json" | cut -c1-240)"
done
echo "SPR VARCHAIN DONE $VID K=$K $(date +%F' '%T)"
