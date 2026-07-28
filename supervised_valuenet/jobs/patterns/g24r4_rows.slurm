#!/bin/bash
# g24r4 zero-shot B2 rows -- the one rung missing from the scaling ladder.
# Replaces g24r4_frontier.slurm, which ran eval.compare over the whole
# instance file in ONE process and projected to 51-90 h against a 16 h cap.
# This is the track1_rows.slurm chunk structure (8-wide, idempotent,
# checkpoint-keyed chunk dir, certification-gated DONE) with the ladder's
# established g24r4 protocol: per-config old-vocabulary nets + full B2
# language, forward arm in the same file, matched 1200/k=5 -- the same
# net-provenance convention as the g24r8 and g32r4 zero-shot B2 rows
# (their result JSONs record per-config backward-{policy,value}, not the
# base-B1 pair; the 16x16 rungs' base-B1 convention is the exception, and
# FINDINGS 29 measured the difference to be small and net-favouring).
#
#   sbatch -J rr-g24r4-b2-grd jobs/patterns/g24r4_rows.slurm graded
#   sbatch -J rr-g24r4-b2-frn jobs/patterns/g24r4_rows.slurm frontier
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 2
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/g24r4/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

SET="$1"
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
SV="$REPO/supervised_valuenet"
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$SV"
export PYTHONPATH=. OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
unset RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR
export RR_GRID=24 RR_ROBOTS=4 RR_WALLS=108 RR_ENV_DIR="$SV/environments_g24r4"

FWD=scaling/runs/g24r4/forward/lightning_logs/version_0/checkpoints/epoch=7-step=76360.ckpt
BP=scaling/runs/g24r4/backward-policy/lightning_logs/version_0/checkpoints/epoch=1-step=1654.ckpt
BV=scaling/runs/g24r4/backward-value/lightning_logs/version_0/checkpoints/epoch=22-step=38019.ckpt
for c in "$FWD" "$BP" "$BV"; do
  [ -s "$c" ] || { echo "FATAL: checkpoint missing: $c"; exit 1; }
done

if [ "$SET" = graded ]; then
  INST=scaling/data/g24r4/bench.solved.jsonl
  OUT=scaling/results/g24r4/comparison_b2.json
else
  INST=scaling/data/g24r4/bench.unsolved.jsonl
  OUT=scaling/results/g24r4/comparison_ungraded_b2.json
fi
[ -s "$INST" ] || { echo "FATAL: instances missing: $INST"; exit 1; }

# Resume gate (parse + nonzero rows), as in track1_rows.slurm.
if [ -s "$OUT" ] && python3 -c "
import json,sys
d=json.load(open('$OUT'))
rows=[len(s.get('rows') or []) for s in d['systems'].values() if s.get('rows')]
sys.exit(0 if rows and min(rows)>0 else 1)" 2>/dev/null; then
  echo "already merged and parseable: $OUT"; echo "G24R4 $SET DONE"; exit 0
elif [ -s "$OUT" ]; then
  echo "existing $OUT is truncated or unparseable -- redoing"; rm -f "$OUT"
fi

STAMP=$(printf '%s\n%s\n%s\n' "$BP" "$BV" "$FWD" | sha256sum | cut -c1-12)
RUN="$REPO/runs/track1/g24r4.$SET.$STAMP"
mkdir -p "$RUN"
printf 'policy=%s\nvalue=%s\nforward=%s\n' "$BP" "$BV" "$FWD" > "$RUN/checkpoints.txt"
[ -e "$RUN/chunk.000.jsonl" ] || \
  split -d -a 3 -l 8 "$INST" "$RUN/chunk." --additional-suffix=.jsonl

run_chunk() {
  local chunk="$1"; local base; base=$(basename "$chunk" .jsonl)
  local out="$RUN/$base.json"
  [ -s "$out" ] && return 0
  if python3 -m eval.compare --instances "$chunk" --expansions 1200 --k 5 \
       --backward-policy "$BP" --backward-value "$BV" \
       --backward-anytime --backward-b2 --forward-ckpts "$FWD" --device cpu \
       --dump-moves --out "$out.tmp.$$" --md /dev/null \
       > "$RUN/$base.log" 2>&1; then
    mv "$out.tmp.$$" "$out"; echo "done $base $(date +%H:%M:%S)"
  else
    echo "FAIL $base (see $RUN/$base.log)"; return 1
  fi
}
export -f run_chunk
export RUN BP BV FWD

ls "$RUN"/chunk.*.jsonl | xargs -P 8 -n 1 -I{} bash -c 'run_chunk "$@"' _ {}

NC=$(ls "$RUN"/chunk.*.jsonl | wc -l)
ND=$(ls "$RUN"/chunk.*.json 2>/dev/null | wc -l)
if [ "$ND" -ne "$NC" ]; then
  echo "MERGE SKIPPED: $ND/$NC chunks done (resubmit to resume)"
  echo "G24R4 $SET PARTIAL"; exit 0
fi
python3 eval/merge_compare_shards.py --shards $(ls "$RUN"/chunk.*.json | sort) \
  --instances "$INST" --out "$OUT" --md /dev/null || \
  { echo "FATAL: merge failed"; exit 1; }

echo "== independent replay certification =="
python3 -m eval.replay_validate --compare "$OUT" --env-dir "$SV/environments_g24r4"
RC=$?
if [ "$RC" -ne 0 ]; then
  mv -f "$OUT" "$OUT.uncertified" 2>/dev/null
  echo "G24R4 $SET REPLAY-FAILED rc=$RC (quarantined at $OUT.uncertified)"
  exit 1
fi
echo "G24R4 $SET DONE"
