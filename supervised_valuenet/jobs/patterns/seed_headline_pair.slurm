#!/bin/bash
# SEED REPLICATES OF THE BACKWARD HEADLINE PAIR (design memo: scratchpad
# ideas/impl_seeds_forward_plan.md; objection: "the headline zero-shot pair is
# a single draw", devils-advocate memo point 3, track1 memo next-step 1).
#
# The headline backward rows are ZERO-SHOT nets (no B2 retraining). Which
# checkpoints those are is read from the protocol block of the headline JSONs
# (scaling/results/<cfg>/comparison_b2.json + comparison_ungraded_b2.json):
#
#   g16r8 (and g16r6, base): the base-trained B1 pair
#       checkpoints_backward/policy_b1.ckpt + value_b1.ckpt
#       recipe = jobs/launcher_patterns/b1_retrain_chain.sh:
#         policy: train.policy_tf --data nn/data/combined_b1.jsonl --epochs 25
#                 (cold, default lr, UNSEEDED in production)
#         value : train.looped_pc  --data nn/data/combined_b1.jsonl --epochs 20
#                 --warmup 0, WARM-STARTED from checkpoints_backward/value_v2.ckpt,
#                 torch seed 11 in production
#       (b2_retrain_one.slurm's g16r4 arm is the same recipe under scaling.train.)
#   g32r4: the per-config BASE-vocabulary pair
#       scaling/runs/g32r4/backward-policy/.../epoch=1-step=1642.ckpt
#       scaling/runs/g32r4/backward-value/lightning_logs/version_1/.../epoch=15-step=52528.ckpt
#       recipe = g32r4_chain.sh (policy 30 ep, bs 8, cold, default lr) +
#                g32r4_retrain.sh (value 25 ep, warmup 0, bs 2, mpg 8, COLD,
#                unseeded) -- verbatim as nn_labeler/jobs/twin_retrain.slurm's
#                g32r4 arm and TWIN_WIRING.md section 2.
#
# This job repeats that recipe EXACTLY for one config and one torch seed
# (--torch-seed on BOTH nets, the seed_study.slurm convention), then benches
# the pair with the headline protocol verbatim (eval.compare, budget 1200,
# k=5, --backward-anytime --backward-b2, CPU, --dump-moves, chunked 8-wide,
# merged, independently replay-certified). Outputs, suffixed by seed so no
# production row can be overwritten:
#
#   g16r8:  scaling/results/g16r8/comparison_b2_seed<S>.json            (graded, 266)
#           scaling/results/g16r8/comparison_ungraded_b2_seed<S>.json   (frontier, 184)
#           eval/results/final450_backward_b2_seed<S>.json              (base 450; ~3 min, free)
#   g32r4:  scaling/results/g32r4/comparison_b2_seed<S>.json            (graded, 175)
#           scaling/results/g32r4/comparison_ungraded_b2_seed<S>.json   (frontier, 275)
#   run dirs: scaling/runs/g16r4/backward-{policy,value}-b1-seed<S>/
#             scaling/runs/g32r4/backward-{policy,value}-base-seed<S>/
#
# Idempotent + resumable (twin_retrain.slurm contract): DONE sentinel per
# training, RR_RESUME=1 from last.ckpt (save_last landed 2026-07-29 in both
# backward trainers), skip-if-exists per bench chunk, chunk dir keyed on
# checkpoint identity, DONE marker gated on certification. A walltime kill
# costs one epoch/one chunk; RESUBMIT THE SAME COMMAND TO CONTINUE (or
# pre-chain a resume link with --dependency=afterany, see
# seed_headline_submit.sh).
#
#   sbatch -J rr-seedhl-g16r8-s21 jobs/patterns/seed_headline_pair.slurm g16r8 21
#   sbatch -J rr-seedhl-g32r4-s21 jobs/patterns/seed_headline_pair.slurm g32r4 21
# arg1 = g16r8 | g32r4, arg2 = torch seed (use 21 37 53 -- the study's seeds).
#
# Measured cost basis (sacct, this project):
#   g16r8: rr-b2rt-g16r4 4h28m trained 25/20 epochs on 376,912 b2 records;
#          combined_b1.jsonl has 108,902 -> expect 1.5-4.5 h; bench: rr-b2r-g16r8
#          took 1h26m as 2 unsharded 8-thread processes -> ~0.5 h 8-wide here.
#          Job ~3-5 h on 1 GPU = 0.4-0.6 nh per seed.
#   g32r4: twin_retrain logs (same corpus size): policy ~3.5 h (12,680 s
#          resumed), value 21,315 s = 5.9 h; bench zero-shot B2 rows took
#          4h32m as 2 unsharded processes (frontier 15,884 lane-s) -> ~1.5-2 h
#          8-wide. Job ~11-13 h on 1 GPU = 1.4-1.6 nh per seed; may need one
#          resume link at the 16 h cap.
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/seedhl/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

CFG="$1"; SEED="$2"
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
SV="$REPO/supervised_valuenet"
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$SV"
export PYTHONPATH=. PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
mkdir -p "$REPO/runs/seedhl"
nvidia-smi -L
unset RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR
echo "SEEDHL $CFG seed=$SEED start $(date +%F' '%T)"

# ---------------------------------------------------------------- recipe ---
case "$CFG" in
  g16r8)
    TCFG=g16r4                                   # legacy config: no SPLITS rebinding (combined_b1 uses legacy ids)
    DATA=nn/data/combined_b1.jsonl
    TAG="b1-seed${SEED}"
    PARGS="--epochs 25";                 PWRAP=""
    # production value_b1 was warm-started from value_v2.ckpt (b1_retrain_chain.sh)
    VARGS="--epochs 20 --warmup 0";      VWRAP="--warm-start checkpoints_backward/value_v2.ckpt"
    [ -s checkpoints_backward/value_v2.ckpt ] || { echo "FATAL: warm-start value_v2.ckpt missing"; exit 1; }
    ;;
  g32r4)
    TCFG=g32r4
    DATA=scaling/data/g32r4/backward.jsonl       # BASE vocabulary (byte-identical to backward.rust.jsonl)
    TAG="base-seed${SEED}"
    PARGS="--epochs 30 --batch-size 8";                       PWRAP=""
    VARGS="--epochs 25 --warmup 0 --batch-size 2 --max-per-group 8"; VWRAP=""   # cold, 50 bins, lr default -- as the record
    ;;
  *) echo "FATAL: cfg must be g16r8 or g32r4"; exit 1 ;;
esac
[ -s "$DATA" ] || { echo "FATAL: missing $DATA"; exit 1; }
STAMP_DATA=$(sha256sum "$DATA" | cut -c1-16)

train_one() {                       # train_one <backward-policy|backward-value> <wrapper flags> -- <trainer args>
  local system="$1"; local wrap="$2"; shift 2
  local dir="scaling/runs/${TCFG}/${system}-${TAG}"
  if [ -f "$dir/DONE" ]; then echo "SEEDHL SKIP train $system (DONE)"; return 0; fi
  mkdir -p "$dir"
  if [ -f "$dir/DATA.txt" ]; then
    grep -qx "sha256_16=$STAMP_DATA" "$dir/DATA.txt" || {
      echo "SEEDHL FAILED train $system: corpus changed under a partial run ($dir/DATA.txt)"; return 1; }
  else
    printf 'data=%s\nsha256_16=%s\nrecords=%s\nseed=%s\nstarted=%s\n' \
      "$DATA" "$STAMP_DATA" "$(wc -l < "$DATA")" "$SEED" "$(date -Is)" > "$dir/DATA.txt"
  fi
  local resume=0
  ls "$dir"/lightning_logs/version_*/checkpoints/last.ckpt >/dev/null 2>&1 && resume=1
  echo "SEEDHL START train $system resume=$resume $(date +%T)  wrap='$wrap' args='$*'"
  local t0=$SECONDS
  RR_RESUME=$resume python3 -m scaling.train --config "$TCFG" --system "$system" \
      --run-dir "$dir" $wrap --torch-seed "$SEED" -- --data "$DATA" "$@"
  local rc=$?
  [ "$rc" -eq 0 ] || { echo "SEEDHL FAILED train $system rc=$rc after $((SECONDS-t0))s"; return 1; }
  ls "$dir"/lightning_logs/version_*/checkpoints/epoch=*.ckpt >/dev/null 2>&1 \
      || { echo "SEEDHL FAILED train $system: exit 0 but no epoch=*.ckpt"; return 1; }
  touch "$dir/DONE"
  echo "SEEDHL DONE train $system in $((SECONDS-t0))s"
}
ckpt_of() { ls -t "$1"/lightning_logs/version_*/checkpoints/epoch=*.ckpt 2>/dev/null | head -1; }

# Policy first (cheap, stable): a broken recipe surfaces in minutes.
train_one backward-policy "$PWRAP" $PARGS || exit 1
train_one backward-value  "$VWRAP" $VARGS || exit 1

PDIR="scaling/runs/${TCFG}/backward-policy-${TAG}"; VDIR="scaling/runs/${TCFG}/backward-value-${TAG}"
BP=$(ckpt_of "$PDIR"); BV=$(ckpt_of "$VDIR")
[ -n "$BP" ] && [ -n "$BV" ] || { echo "SEEDHL FAILED: checkpoints missing"; exit 1; }
# best val_regret of the value draw -- the FINDINGS 44 mode signature; printed
# for the record (comparable only within config+corpus)
python3 - "$BV" <<'EOF' || true
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
for k, v in (ck.get("callbacks") or {}).items():
    if isinstance(v, dict) and "best_model_score" in v:
        s = v["best_model_score"]; s = float(s) if s is not None else None
        print(f"SEEDHL value best_model_score={s} monitor={v.get('monitor')} path={v.get('best_model_path')}")
EOF
echo "SEEDHL nets  policy=$BP  value=$BV"

# ----------------------------------------------------------------- bench ---
# bench_one <label> <instances> <out> <envdir> <grid> <robots> <walls>
bench_one() {
  local label="$1" INST="$2" OUT="$3" ENVDIR="$4" G="$5" R="$6" W="$7"
  [ -s "$INST" ] || { echo "SEEDHL FAILED bench $label: missing $INST"; return 1; }
  if [ -s "$OUT" ] && python3 -c "
import json,sys
d=json.load(open('$OUT'))
rows=[len(s.get('rows') or []) for s in d['systems'].values() if s.get('rows')]
sys.exit(0 if rows and min(rows)>0 else 1)" 2>/dev/null; then
    echo "SEEDHL SKIP bench $label (exists: $OUT)"; return 0
  elif [ -s "$OUT" ]; then
    echo "SEEDHL bench $label: $OUT truncated -- redoing merge"; rm -f "$OUT"
  fi
  if [ "$label" = base450 ]; then unset RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR
  else export RR_GRID=$G RR_ROBOTS=$R RR_WALLS=$W RR_ENV_DIR="$ENVDIR"; fi
  local STAMP; STAMP=$(printf '%s\n%s\n' "$BP" "$BV" | sha256sum | cut -c1-12)
  RUN="$REPO/runs/seedhl/${CFG}.${label}.seed${SEED}.${STAMP}"
  mkdir -p "$RUN"
  printf 'policy=%s\nvalue=%s\ninstances=%s\n' "$BP" "$BV" "$INST" > "$RUN/checkpoints.txt"
  [ -e "$RUN/chunk.000.jsonl" ] || split -d -a 3 -l 8 "$INST" "$RUN/chunk." --additional-suffix=.jsonl
  echo "SEEDHL START bench $label $(date +%T)"
  local t0=$SECONDS
  run_chunk() {
    local chunk="$1"; local base; base=$(basename "$chunk" .jsonl); local out="$RUN/$base.json"
    [ -s "$out" ] && return 0
    if OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" python3 -m eval.compare \
         --instances "$chunk" --expansions 1200 --k 5 \
         --backward-policy "$BP" --backward-value "$BV" \
         --backward-anytime --backward-b2 --forward-ckpts "" --device cpu \
         --dump-moves --out "$out.tmp.$$" --md /dev/null > "$RUN/$base.log" 2>&1; then
      mv "$out.tmp.$$" "$out"
    else echo "SEEDHL CHUNK FAIL $base (see $RUN/$base.log)"; return 1; fi
  }
  export -f run_chunk; export RUN BP BV
  ls "$RUN"/chunk.*.jsonl | xargs -P 8 -n 1 -I{} bash -c 'run_chunk "$@"' _ {}
  local NC ND; NC=$(ls "$RUN"/chunk.*.jsonl | wc -l); ND=$(ls "$RUN"/chunk.*.json 2>/dev/null | wc -l)
  [ "$ND" -eq "$NC" ] || { echo "SEEDHL PARTIAL bench $label $ND/$NC chunks (resubmit to resume)"; return 2; }
  python3 eval/merge_compare_shards.py --shards $(ls "$RUN"/chunk.*.json | sort) \
      --instances "$INST" --out "$OUT" --md /dev/null || { echo "SEEDHL FAILED bench $label: merge"; return 1; }
  local envarg="$ENVDIR"
  python3 -m eval.replay_validate --compare "$OUT" --env-dir "$envarg"
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    mv -f "$OUT" "$OUT.uncertified" 2>/dev/null
    echo "SEEDHL FAILED bench $label REPLAY rc=$rc (quarantined at $OUT.uncertified)"; return 1
  fi
  echo "SEEDHL DONE bench $label in $((SECONDS-t0))s -> $OUT"
}

FAILS=0
case "$CFG" in
  g16r8)
    bench_one graded   scaling/data/g16r8/bench.solved.jsonl   scaling/results/g16r8/comparison_b2_seed${SEED}.json          "$SV/environments_g16r8" 16 8 48 || FAILS=$((FAILS+1))
    bench_one frontier scaling/data/g16r8/bench.unsolved.jsonl scaling/results/g16r8/comparison_ungraded_b2_seed${SEED}.json "$SV/environments_g16r8" 16 8 48 || FAILS=$((FAILS+1))
    bench_one base450  eval/data/bench450.jsonl                eval/results/final450_backward_b2_seed${SEED}.json             "$SV/environments"       16 4 48 || FAILS=$((FAILS+1))
    ;;
  g32r4)
    bench_one graded   scaling/data/g32r4/bench.solved.jsonl   scaling/results/g32r4/comparison_b2_seed${SEED}.json          "$SV/environments_g32r4" 32 4 192 || FAILS=$((FAILS+1))
    bench_one frontier scaling/data/g32r4/bench.unsolved.jsonl scaling/results/g32r4/comparison_ungraded_b2_seed${SEED}.json "$SV/environments_g32r4" 32 4 192 || FAILS=$((FAILS+1))
    ;;
esac
unset RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR
if [ "$FAILS" -ne 0 ]; then
  echo "SEEDHL $CFG seed$SEED PARTIAL ($FAILS lane(s) incomplete) -- resubmit the same command to continue"; exit 1
fi
echo "SEEDHL $CFG seed$SEED DONE"
