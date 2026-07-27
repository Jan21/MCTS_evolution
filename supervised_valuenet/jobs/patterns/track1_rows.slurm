#!/bin/bash
# TRACK 1 -- the definitive backward rows with the B2-RETRAINED networks.
# Reruns ONLY the backward lanes (every forward control row stands), at the
# pinned budgets, writing the exact filenames eval/report_data.py expects:
#
#   base   -> eval/results/final450_backward_b2_retrained.json
#   <cfg>  -> scaling/results/<cfg>/comparison_b2retrained.json           (graded)
#             scaling/results/<cfg>/comparison_ungraded_b2retrained.json  (frontier)
#
# Usage (one lane per invocation, submit after BANKING the config's nets):
#   sbatch -J rr-t1-base        jobs/patterns/track1_rows.slurm base   base450
#   sbatch -J rr-t1-g16r6-grd   jobs/patterns/track1_rows.slurm g16r6  graded
#   sbatch -J rr-t1-g24r8-frn   jobs/patterns/track1_rows.slurm g24r8  frontier
#
# Checkpoints come from scaling/runs/b2_banked.json, written at banking time:
#   {"g16r6": {"policy": "<path>", "value": "<path>"}, ...}
# with the base entry pointing at checkpoints_backward/{policy,value}_b2.ckpt.
# Reading them from a manifest instead of guessing version_N directories is
# deliberate -- resubmissions accumulate version dirs and the handoff's rule is
# to identify runs by hparams+mtime, never by version number.
#
# --dump-moves is ON (it costs no extra BFS since 2026-07-25, and the lane ends
# with an independent replay certification of every solved row).
# --count-slides is deliberately OFF: the counter wraps the hottest function in
# the codebase, so these rows' seconds would stop being comparable to the
# forward control rows. Compute accounting is a SEPARATE pass -- see
# jobs/patterns/track1_accounting.slurm.
#
# Cost, projected from the measured zero-shot B2 lanes (46.0 CPU-hours over all
# nine lanes; g24r8 frontier alone is 34.6): sharded 8-wide on --gpus 2 the
# whole of Track 1 is ~6 h wall, ~1.5 node-hours. Retrained nets should rank
# better and expand less, so that is an upper bound.
#
# Chunks are small and idempotent (skip-if-exists, tmp+mv), so a walltime kill
# loses at most one chunk; resubmit the same command to resume.
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 2
#SBATCH --nodes=1
# WALLTIME 16 h, not 24: the cooling reservation (MOTD, daily 10:00-18:00 CEST)
# blocks any job requesting more than 16 hours from starting while it is in
# effect. Every lane here is chunked/idempotent, so a 16 h cap costs at most
# one chunk and a resubmit.
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/track1/%x-%j.out

set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

CFG="$1"; SET="$2"
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
SV="$REPO/supervised_valuenet"
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$SV"
export PYTHONPATH=. OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
unset RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR

# The lane is named "base" for readability but scaling/runs/b2_banked.json
# is keyed by CONFIG, and the base config is g16r4. Job 4597369 died on
# this mismatch ("no banked entry for base"), so map it explicitly.
MKEY="$CFG"; [ "$CFG" = base ] && MKEY=g16r4
MANIFEST=scaling/runs/b2_banked.json
[ -s "$MANIFEST" ] || { echo "FATAL: $MANIFEST missing -- bank the nets first"; exit 1; }
read -r BP BV < <(python3 -c "
import json,sys
m=json.load(open('$MANIFEST')).get('$MKEY')
if not m: sys.exit('FATAL: no banked entry for $MKEY in $MANIFEST')
print(m['policy'], m['value'])") || exit 1
for c in "$BP" "$BV"; do
  [ -s "$c" ] || { echo "FATAL: banked checkpoint missing: $c"; exit 1; }
done
echo "[track1] $CFG/$SET  policy=$BP  value=$BV"

case "$CFG" in
  base)  INST=eval/data/bench450.jsonl
         OUT=eval/results/final450_backward_b2_retrained.json
         ENVDIR="$SV/environments" ;;
  *)     ENVDIR="$SV/environments_$CFG"
         case "$CFG" in
           g16r6) export RR_GRID=16 RR_ROBOTS=6 RR_WALLS=48 ;;
           g16r8) export RR_GRID=16 RR_ROBOTS=8 RR_WALLS=48 ;;
           g24r4) export RR_GRID=24 RR_ROBOTS=4 RR_WALLS=108 ;;
           g24r8) export RR_GRID=24 RR_ROBOTS=8 RR_WALLS=108 ;;
           g32r4) export RR_GRID=32 RR_ROBOTS=4 RR_WALLS=192 ;;
           *) echo "FATAL: unknown config $CFG"; exit 1 ;;
         esac
         export RR_ENV_DIR="$ENVDIR"
         if [ "$SET" = graded ]; then
           INST="scaling/data/$CFG/bench.solved.jsonl"
           OUT="scaling/results/$CFG/comparison_b2retrained.json"
         else
           INST="scaling/data/$CFG/bench.unsolved.jsonl"
           OUT="scaling/results/$CFG/comparison_ungraded_b2retrained.json"
         fi ;;
esac
[ -s "$INST" ] || { echo "FATAL: instances missing: $INST"; exit 1; }
# A walltime kill during the merge (merge_compare_shards.py writes in place)
# can leave truncated JSON. Accept an existing output only if it parses AND has
# the expected row count, otherwise redo the merge.
if [ -s "$OUT" ] && python3 -c "
import json,sys
d=json.load(open('$OUT'))
rows=[len(s.get('rows') or []) for s in d['systems'].values() if s.get('rows')]
sys.exit(0 if rows and min(rows)>0 else 1)" 2>/dev/null; then
  echo "already merged and parseable: $OUT"; echo "TRACK1 $CFG $SET DONE"; exit 0
elif [ -s "$OUT" ]; then
  echo "existing $OUT is truncated or unparseable -- redoing the merge"
  rm -f "$OUT"
fi

RUN="$REPO/runs/track1/$CFG.$SET"
mkdir -p "$RUN"
# Chunks must stay in benchmark order -- merge_compare_shards.py requires it.
[ -e "$RUN/chunk.000.jsonl" ] || \
  split -d -a 3 -l 8 "$INST" "$RUN/chunk." --additional-suffix=.jsonl

run_chunk() {
  local chunk="$1"; local base; base=$(basename "$chunk" .jsonl)
  local out="$RUN/$base.json"
  [ -s "$out" ] && return 0
  if python3 -m eval.compare --instances "$chunk" --expansions 1200 --k 5 \
       --backward-policy "$BP" --backward-value "$BV" \
       --backward-anytime --backward-b2 --forward-ckpts "" --device cpu \
       --dump-moves --out "$out.tmp.$$" --md /dev/null \
       > "$RUN/$base.log" 2>&1; then
    mv "$out.tmp.$$" "$out"; echo "done $base $(date +%H:%M:%S)"
  else
    echo "FAIL $base (see $RUN/$base.log)"; return 1
  fi
}
export -f run_chunk
export RUN BP BV

ls "$RUN"/chunk.*.jsonl | xargs -P 8 -n 1 -I{} bash -c 'run_chunk "$@"' _ {}

NC=$(ls "$RUN"/chunk.*.jsonl | wc -l)
ND=$(ls "$RUN"/chunk.*.json 2>/dev/null | wc -l)
if [ "$ND" -ne "$NC" ]; then
  echo "MERGE SKIPPED: $ND/$NC chunks done (resubmit to resume)"
  echo "TRACK1 $CFG $SET PARTIAL"; exit 0
fi
python3 eval/merge_compare_shards.py --shards $(ls "$RUN"/chunk.*.json | sort) \
  --instances "$INST" --out "$OUT" --md /dev/null || \
  { echo "FATAL: merge failed"; exit 1; }

echo "== independent replay certification =="
python3 -m eval.replay_validate --compare "$OUT" --env-dir "$ENVDIR"
RC=$?
# The DONE marker is grep-able and is the house rule's completion signal, so it
# must never print over a failed certification. replay_validate exits 1 on any
# replay failure and 2 when it validated nothing at all; both are lane failures.
if [ "$RC" -ne 0 ]; then
  echo "TRACK1 $CFG $SET REPLAY-FAILED rc=$RC"
  exit 1
fi
echo "TRACK1 $CFG $SET DONE"
