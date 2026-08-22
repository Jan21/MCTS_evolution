#!/bin/bash
# Adopt-into-main-line experiment (wave 3): ONE mixed-size B2 curriculum
# iteration (g24r4+g32r4+g24r8, as the main loop's selfplay_mix_iter.slurm)
# seeded from mix_b2mix_iter3's nets, but with the lab's recipe: --emit all
# (v04) + strict-moves value targets (v09 train hook). Fresh board ids 9500+
# (the main loop used 9000-9089). Results/provenance stay in the lab:
# results/variants/adopt_mainline/; gates vs mix_b2mix_iter3's own benches
# answer "does the lab recipe improve the main line's next iteration?".
#   sbatch -J spr-var-adopt self_play_robots/jobs/variant_adopt.slurm
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
CFGS=${CFGS:-"g24r4 g32r4 g24r8"}
BOARDS=30; PER_BOARD=8; EXP=300; STOP=80; NOISE=0.25; EP_P=6; EP_V=6; LR=1e-4
BOARDS_R8=${BOARDS_R8:-12}   # emit-all extraction is ~6x slower on 8-robot trees
                             # (~91 s/inst); 12 boards ~ 3.7 h and still yields ~10k
                             # records vs mix-it3's 5.8k path-only leg
SEEDNETS="$REPO/self_play_robots/results/selfplay/mix_b2mix_iter3/nets.txt"
P0=$(grep '^policy=' "$SEEDNETS" | cut -d= -f2-); V0=$(grep '^value=' "$SEEDNETS" | cut -d= -f2-)
IT="$REPO/runs/spr/variants/adopt_mainline"; RES="$REPO/self_play_robots/results/variants/adopt_mainline"
mkdir -p "$IT" "$RES"
ID0=9500; ID1=$((ID0 + BOARDS - 1))
echo "SPR ADOPT start $(date +%F' '%T) job=$SLURM_JOB_ID prev=$P0|$V0 boards=$ID0-$ID1"
nvidia-smi -L
phase() { [ -f "$IT/DONE.$1" ]; }
mark()  { touch "$IT/DONE.$1"; echo "SPR ADOPT PHASE DONE $1 $(date +%T)"; }
DATA=""; SPLITS=""
for CFG in $CFGS; do
  W=6; case "$CFG" in g32*) W=4;; esac
  NB=$BOARDS; [ "$CFG" = g24r8 ] && NB=$BOARDS_R8
  IDB=$((ID0 + NB - 1))
  if ! phase gen_$CFG; then
    mkdir -p "$IT/$CFG"
    python -m spr.selfplay --config "$CFG" --policy "$P0" --value "$V0" \
        --boards-dir "$REPO/runs/spr/boards/adopt_${CFG}" --board-ids "$ID0-$IDB" --vocab b2 \
        --per-board "$PER_BOARD" --seed 950 --iter 4 --expansions "$EXP" --stop-after "$STOP" \
        --root-noise "$NOISE" --complete-siblings --root-all --emit all --workers "$W" --threads 2 \
        --device cuda --timeout 900 --out "$IT/$CFG/records.jsonl" || { echo "SPR ADOPT FAILED gen $CFG"; exit 1; }
    cp "$IT/$CFG/records.jsonl.manifest.json" "$RES/generation_${CFG}.manifest.json"
    mark gen_$CFG
  fi
  DATA="$DATA --data ${CFG}=$IT/$CFG/records.jsonl"
  SP=$(python - "$IT/$CFG/records.jsonl" "$CFG" <<'PYEOF'
import json, sys
ids = sorted({json.loads(l)["env_id"] for l in open(sys.argv[1])})
cut = max(1, int(round(len(ids) * 0.85)))
fmt = lambda xs: ",".join(map(str, xs))
print(f"{sys.argv[2]}:train={fmt(ids[:cut])},val={fmt(ids[cut:])}")
PYEOF
)
  SPLITS="$SPLITS --splits $SP"
done
train_one() {
  local sys="$1" init="$2" ep="$3" extra="$4" dir="$IT/$1"
  [ -f "$dir/DONE" ] && return 0
  RR_RESUME=0 OMP_NUM_THREADS=16 python -m spr.train --system "$sys" $DATA $SPLITS \
      --init "$init" --epochs "$ep" --lr "$LR" $extra --torch-seed 950 --num-workers 4 \
      --device cuda --out-dir "$dir" || { echo "SPR ADOPT FAILED train $sys"; return 1; }
  touch "$dir/DONE"
}
train_one policy "$P0" "$EP_P" "--batch-size 8 --grad-clip 1.0 --byref --batch-ref-n 24" || exit 1
train_one value  "$V0" "$EP_V" "--batch-size 4 --max-per-group 8 --batch-ref-n 24 --variant v09_strict_value" || exit 1
BP=$(ls -t "$IT"/policy/lightning_logs/version_*/checkpoints/epoch=*.ckpt | head -1)
BV=$(ls -t "$IT"/value/lightning_logs/version_*/checkpoints/epoch=*.ckpt | head -1)
printf 'policy=%s\nvalue=%s\n' "$BP" "$BV" > "$RES/nets.txt"
echo "SPR ADOPT nets $BP | $BV"
bench_one() {
  local cfg="$1" tag="$2" inst="$3"
  local out="$RES/bench_${cfg}_${tag}.json"
  [ -s "$out" ] && return 0
  python -m spr.arena bench --arm "adopt_${cfg}_${tag}" --config "$cfg" --instances "$inst" \
      --policy "$BP" --value "$BV" --driver spr \
      --flags "--search arena_astar --vocab b2 --anytime" \
      --width 8 --threads 2 --device cuda --out "$out" || echo "SPR ADOPT FAILED bench $cfg $tag"
}
for CFG in $CFGS; do
  bench_one "$CFG" astar "scaling/data/$CFG/bench.solved.jsonl"
  bench_one "$CFG" frontier_astar "scaling/data/$CFG/bench.unsolved.jsonl"
done
PREV="$REPO/self_play_robots/results/selfplay/mix_b2mix_iter3"
for CFG in $CFGS; do for t in astar frontier_astar; do
  [ -s "$RES/bench_${CFG}_$t.json" ] && [ -s "$PREV/bench_${CFG}_$t.json" ] || continue
  python -m spr.gate compare --a "$RES/bench_${CFG}_$t.json" --b "$PREV/bench_${CFG}_$t.json" \
      > "$RES/gate_${CFG}_${t}_vs_mixit3.json" || true
  echo "SPR ADOPT gate $CFG $t: $(tr -d '\n' < "$RES/gate_${CFG}_${t}_vs_mixit3.json" | cut -c1-220)"
done; done
echo "SPR ADOPT DONE $(date +%F' '%T)"
