#!/bin/bash
# A FAIR RESCUE FOR THE FORWARD PLANNER -- select-by-validation, then bench.
# Companion of fwd_rescue_grid.slurm (submit with --dependency=afterany on the
# grid's resume link). Two steps:
#
# 1. SELECT: over scaling/runs/<cfg>/forward-rescue/s*-lr*/BEST.json, pick the
#    cell with the highest val_policy_top1 -- the forward pipeline's own
#    checkpoint monitor (move_planner/net.py: ModelCheckpoint(monitor=
#    "val_policy_top1", mode="max"); validation = the config's val boards
#    700-899, disjoint from the bench boards 900-1049). Refuses to run unless
#    ALL 9 cells have DONE (a partial grid would bias the selection toward the
#    cells that happened to finish). Writes
#      scaling/runs/<cfg>/forward-rescue/SELECTED.json   (winner + full table)
#    Test rows are never read here: selection is by validation only.
# 2. BENCH the winner with the forward control's protocol verbatim
#    (comparison_forward_control.json: eval.compare --expansions 1200 --k 5
#    --skip-backward --forward-ckpts <ckpt> --device cpu; plus --dump-moves for
#    replay certification), on the pinned graded + frontier pools, chunked
#    8-wide, merged, replay-certified:
#      scaling/results/<cfg>/comparison_forward_rescue.json            (graded)
#      scaling/results/<cfg>/comparison_ungraded_forward_rescue.json   (frontier)
#    The forward rows in those files are directly comparable to the forward
#    rows in comparison_forward_control.json / comparison_ungraded.json (g16r8)
#    and comparison_b2.json / comparison_ungraded_b2.json (g24r4).
#
#   sbatch --dependency=afterany:<grid-resume-jobid> -J rr-fwdsel-g16r8 \
#          jobs/patterns/fwd_rescue_select_bench.slurm g16r8
#   FWD_BENCH_SETS="graded frontier"  (default) -- override to run one set, e.g.
#   FWD_BENCH_SETS=graded for a cheap g24r4 first look.
#
# Cost basis (per-instance seconds stored in the result JSONs, 8-thread lanes):
#   g16r8 forward: graded 5,641 s + frontier 47,215 s = 14.7 lane-h ->
#          8 lanes x 4 threads on --gpus 2 (32 cores): ~3-4 h wall = ~1 nh.
#   g24r4 forward: graded 63,787 s + frontier 534,231 s (41 min/puzzle, budget
#          exhausted on 93%) = 166 lane-h -> ~25-30 h wall on 8x4-thread lanes
#          = 2 links x 16 h x 2 GPUs = up to 8 nh. Graded alone ~3 h = 0.75 nh.
#   g32r4 forward: 895,698 s = 249 lane-h (40 min/puzzle) -> EXCLUDED, see plan.
# Idempotent: chunk skip-if-exists, chunk dir keyed on the selected checkpoint,
# resubmit the same command to resume; DONE marker gated on certification.
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 2
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/fwdgrid/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

CFG="$1"
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
SV="$REPO/supervised_valuenet"
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$SV"
export PYTHONPATH=. PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
mkdir -p "$REPO/runs/fwdgrid"
GRID="scaling/runs/$CFG/forward-rescue"
case "$CFG" in
  g16r8) export RR_GRID=16 RR_ROBOTS=8 RR_WALLS=48;  ENVDIR="$SV/environments_g16r8" ;;
  g24r4) export RR_GRID=24 RR_ROBOTS=4 RR_WALLS=108; ENVDIR="$SV/environments_g24r4" ;;
  *) echo "FATAL: cfg must be g16r8 or g24r4"; exit 1 ;;
esac
export RR_ENV_DIR="$ENVDIR"

# ---------------------------------------------------------------- select ---
SEL="$GRID/SELECTED.json"
python3 - "$GRID" "$SEL" <<'EOF' || exit 1
import glob, json, os, sys
grid, sel = sys.argv[1], sys.argv[2]
cells = sorted(glob.glob(os.path.join(grid, "s*-lr*")))
if len(cells) != 9:
    sys.exit(f"FWDSEL REFUSE: {len(cells)} cell dirs under {grid} (want 9)")
table = []
for c in cells:
    if not os.path.exists(os.path.join(c, "DONE")):
        sys.exit(f"FWDSEL REFUSE: {c} not DONE -- selection over a partial grid would be biased")
    b = json.load(open(os.path.join(c, "BEST.json")))
    if b.get("monitor") != "val_policy_top1" or b.get("best_model_score") is None:
        sys.exit(f"FWDSEL REFUSE: {c}/BEST.json monitor={b.get('monitor')} score={b.get('best_model_score')}")
    ck = b.get("best_model_path") or ""
    if ck and not os.path.isabs(ck):
        ck = os.path.join(c, ck)                 # Lightning may store it relative to the run dir
    if not ck or not os.path.exists(ck):
        sys.exit(f"FWDSEL REFUSE: best_model_path missing for {c}: {b.get('best_model_path')}")
    table.append({"cell": os.path.basename(c), "seed": b["seed"], "lr": b["lr"],
                  "val_policy_top1": b["best_model_score"], "ckpt": ck,
                  "epochs_run": b.get("epochs_run")})
table.sort(key=lambda r: (-r["val_policy_top1"], r["lr"], r["seed"]))   # deterministic tie-break
win = table[0]
# reference: the control of record's validation top-1 is not stored in a JSON;
# FINDINGS 6 quotes 0.878 for the g16r6 rescue -- read version_44's own value from
# its checkpoint for the g16r8 comparison if present.
ref = None
try:
    import torch
    for k, v in (torch.load("lightning_logs/version_44/checkpoints/epoch=6-step=46767.ckpt",
                            map_location="cpu", weights_only=False).get("callbacks") or {}).items():
        if isinstance(v, dict) and "best_model_score" in v:
            ref = {"ckpt": "lightning_logs/version_44/checkpoints/epoch=6-step=46767.ckpt",
                   "val_policy_top1": float(v["best_model_score"])}
except Exception as e:
    ref = {"error": str(e)}
out = {"config": os.path.basename(os.path.dirname(grid)), "selection_metric": "val_policy_top1 (max)",
       "rule": "highest validation top-1 over the 3 seeds x 3 lr grid; ties -> lower lr, lower seed; test never read",
       "winner": win, "table": table, "control_of_record_g16r8": ref}
tmp = sel + ".tmp"; json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, sel)
print("FWDSEL table:")
for r in table: print(f"  {r['cell']:14s} val_top1={r['val_policy_top1']:.4f} epochs={r['epochs_run']} {r['ckpt']}")
print(f"FWDSEL WINNER {win['cell']} val_top1={win['val_policy_top1']:.4f} -> {win['ckpt']}")
EOF
FWD=$(python3 -c "import json;print(json.load(open('$SEL'))['winner']['ckpt'])")
[ -s "$FWD" ] || { echo "FATAL: selected ckpt missing: $FWD"; exit 1; }
export FWD

# ----------------------------------------------------------------- bench ---
bench_one() {                       # bench_one <graded|frontier>
  local set="$1" INST OUT
  if [ "$set" = graded ]; then INST="scaling/data/$CFG/bench.solved.jsonl";   OUT="scaling/results/$CFG/comparison_forward_rescue.json"
  else                          INST="scaling/data/$CFG/bench.unsolved.jsonl"; OUT="scaling/results/$CFG/comparison_ungraded_forward_rescue.json"; fi
  [ -s "$INST" ] || { echo "FWDSEL FAILED bench $set: missing $INST"; return 1; }
  if [ -s "$OUT" ] && python3 -c "
import json,sys
d=json.load(open('$OUT'))
rows=[len(s.get('rows') or []) for s in d['systems'].values() if s.get('rows')]
sys.exit(0 if rows and min(rows)>0 else 1)" 2>/dev/null; then
    echo "FWDSEL SKIP bench $set (exists: $OUT)"; return 0
  elif [ -s "$OUT" ]; then echo "FWDSEL bench $set: $OUT truncated -- redoing merge"; rm -f "$OUT"; fi
  local STAMP; STAMP=$(printf '%s\n' "$FWD" | sha256sum | cut -c1-12)
  RUN="$REPO/runs/fwdgrid/$CFG.$set.$STAMP"; mkdir -p "$RUN"
  printf 'forward=%s\ninstances=%s\n' "$FWD" "$INST" > "$RUN/checkpoints.txt"
  # 4-instance chunks: forward frontier instances run 4-41 min each, so small
  # chunks keep the 8 lanes balanced and a walltime kill cheap.
  [ -e "$RUN/chunk.000.jsonl" ] || split -d -a 3 -l 4 "$INST" "$RUN/chunk." --additional-suffix=.jsonl
  echo "FWDSEL START bench $set $(date +%T) forward=$FWD"
  local t0=$SECONDS
  run_chunk() {
    local chunk="$1"; local base; base=$(basename "$chunk" .jsonl); local out="$RUN/$base.json"
    [ -s "$out" ] && return 0
    if OMP_NUM_THREADS=4 python3 -m eval.compare --instances "$chunk" --expansions 1200 --k 5 \
         --skip-backward --forward-ckpts "$FWD" --device cpu --dump-moves \
         --out "$out.tmp.$$" --md /dev/null > "$RUN/$base.log" 2>&1; then
      mv "$out.tmp.$$" "$out"
    else echo "FWDSEL CHUNK FAIL $base (see $RUN/$base.log)"; return 1; fi
  }
  export -f run_chunk; export RUN
  ls "$RUN"/chunk.*.jsonl | xargs -P 8 -n 1 -I{} bash -c 'run_chunk "$@"' _ {}
  local NC ND; NC=$(ls "$RUN"/chunk.*.jsonl | wc -l); ND=$(ls "$RUN"/chunk.*.json 2>/dev/null | wc -l)
  [ "$ND" -eq "$NC" ] || { echo "FWDSEL PARTIAL bench $set $ND/$NC chunks (resubmit to resume)"; return 2; }
  python3 eval/merge_compare_shards.py --shards $(ls "$RUN"/chunk.*.json | sort) \
      --instances "$INST" --out "$OUT" --md /dev/null || { echo "FWDSEL FAILED bench $set: merge"; return 1; }
  python3 -m eval.replay_validate --compare "$OUT" --env-dir "$ENVDIR"
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    mv -f "$OUT" "$OUT.uncertified" 2>/dev/null
    echo "FWDSEL FAILED bench $set REPLAY rc=$rc (quarantined at $OUT.uncertified)"; return 1
  fi
  echo "FWDSEL DONE bench $set in $((SECONDS-t0))s -> $OUT"
}
FAILS=0
for s in ${FWD_BENCH_SETS:-graded frontier}; do bench_one "$s" || FAILS=$((FAILS+1)); done
if [ "$FAILS" -ne 0 ]; then echo "FWDSEL $CFG PARTIAL ($FAILS lane(s) incomplete) -- resubmit to continue"; exit 1; fi
echo "FWDSEL $CFG DONE"
