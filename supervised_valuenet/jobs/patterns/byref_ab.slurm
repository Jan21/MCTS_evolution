#!/bin/bash
# THE BY-REFERENCE SUPPLY A/B (FINDINGS 40 / "Still open" item 1) -- NOT YET
# SUBMITTED. Same nets, same instances, same budget, same protocol; the ONLY
# difference between the two arms is `--backward-byref`, the flag that wires
# Lever B2's by-reference proposal path into the learned driver
# (eval/compare.py::_nn_astar_backward: inject skeleton.astar._reference_helpers,
# apply with by_reference=True, resolve helper featurization by robot identity).
# This separates SUPPLY from RANKING: the nets are the banked cap-20000 g16r6
# pair, trained WITHOUT by-reference records (train/policy_common.py's filter
# was on), so the on-arm is a zero-shot test of whether the extra candidate
# type helps before anyone pays for a retrain.
#
# Usage (four submissions; each is idempotent and resumable):
#   sbatch -J rr-byref-off-grd jobs/patterns/byref_ab.slurm off graded
#   sbatch -J rr-byref-off-frn jobs/patterns/byref_ab.slurm off frontier
#   sbatch -J rr-byref-on-grd  jobs/patterns/byref_ab.slurm on  graded
#   sbatch -J rr-byref-on-frn  jobs/patterns/byref_ab.slurm on  frontier
# The pooled-450 figure is the graded(316) + frontier(134) union, assembled
# from the two files by eval/stats_tests.py exactly as every other pooled row.
#
# OUTPUTS (deliberately NOT any filename eval/report_data.py reads, so an
# exploratory arm can never leak into the report):
#   scaling/results/g16r6/comparison_b2retrained_cap20000_byref_{on,off}.json
#   scaling/results/g16r6/comparison_ungraded_b2retrained_cap20000_byref_{on,off}.json
#
# WHY THE OFF ARM IS RE-RUN INSTEAD OF REUSED. The production rows
# (comparison_{,ungraded_}b2retrained_cap20000.json) were produced by the
# pre-flag driver. Re-running the off arm at THIS revision makes the control
# contemporaneous and doubles as a regression check: the off-arm solve counts
# must equal the production ones (eval/test_byref_wiring.py proves row-level
# identity on a handful of instances; this proves it on all 450). The script
# prints that comparison at the end -- informational, never fatal.
#
# CANDIDATE POOL. By-reference inflates the candidate pool ~1.5x (measured:
# job 4599947 -- 53 by-reference candidates per expansion on top of ~99
# as-shipped applications, in 97% of g16r6 frontier expansions). $BYREF_POOL
# (default 100, i.e. the as-shipped mean) caps the pool at the top-m by
# heuristic score, so the ON arm does no more per-expansion physics than the
# OFF arm and the new candidates must EARN their slots. That is the strict
# substitution test; it also means a loss could be the cap rather than the
# language, so if the on-arm loses, rerun it with BYREF_POOL= (unbounded)
# before concluding anything. The matched-budget protocol itself is untouched
# either way: one expansion is still one policy pass plus one batched value
# pass over <= k children in both arms and in the forward system.
#
# COST, from the measured cap-20000 g16r6 lanes on this exact recipe
# (--gpus 2, billing=2 => 0.25 node-hour per wall hour):
#   graded 316 inst: 07:47 (job 4598162) .. 22:06 (4600164)
#   frontier 134 inst: 23:45 (4598163) .. 45:09 (4600165)
# So one arm over the pooled 450 is ~0.5-1.2 h wall = 0.13-0.30 nh. The ON arm
# does up to 1.5x the candidate work per expansion before the cap and may
# search deeper, so budget 2x: ~0.6 nh. All four lanes: ~0.9 nh expected,
# 2.0 nh if a lane sits to its 8 h walltime. Smoke first if in doubt:
# eval/test_byref_wiring.py runs both checks in ~20 s on a login node.
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 2
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/byref/%x-%j.out

set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

ARM="${1:?usage: byref_ab.slurm <on|off> <graded|frontier>}"
SET="${2:?usage: byref_ab.slurm <on|off> <graded|frontier>}"
case "$ARM" in on|off) ;; *) echo "FATAL: arm must be on|off"; exit 1 ;; esac
case "$SET" in graded|frontier) ;; *) echo "FATAL: set must be graded|frontier"; exit 1 ;; esac

REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
SV="$REPO/supervised_valuenet"
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$SV"
export PYTHONPATH=. OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
unset RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR RR_BYREF RR_BYREF_POOL RR_BYREF_RECORDS

CFG=g16r6
export RR_GRID=16 RR_ROBOTS=6 RR_WALLS=48
ENVDIR="$SV/environments_$CFG"
export RR_ENV_DIR="$ENVDIR"

MANIFEST="${BANK_MANIFEST:-scaling/runs/b2_banked_cap20000.json}"
[ -s "$MANIFEST" ] || { echo "FATAL: $MANIFEST missing"; exit 1; }
read -r BP BV < <(python3 -c "
import json,sys
m=json.load(open('$MANIFEST')).get('$CFG')
if not m: sys.exit('FATAL: no banked entry for $CFG in $MANIFEST')
print(m['policy'], m['value'])") || exit 1
for c in "$BP" "$BV"; do
  [ -s "$c" ] || { echo "FATAL: banked checkpoint missing: $c"; exit 1; }
done

if [ "$SET" = graded ]; then
  INST="scaling/data/$CFG/bench.solved.jsonl"
  OUT="scaling/results/$CFG/comparison_b2retrained_cap20000_byref_$ARM.json"
  PROD="scaling/results/$CFG/comparison_b2retrained_cap20000.json"
else
  INST="scaling/data/$CFG/bench.unsolved.jsonl"
  OUT="scaling/results/$CFG/comparison_ungraded_b2retrained_cap20000_byref_$ARM.json"
  PROD="scaling/results/$CFG/comparison_ungraded_b2retrained_cap20000.json"
fi
[ -s "$INST" ] || { echo "FATAL: instances missing: $INST"; exit 1; }

# The lever. BYREF_POOL="" means unbounded (see the header's caveat).
BYREF_POOL="${BYREF_POOL-100}"
BYREF_ARGS=""
if [ "$ARM" = on ]; then
  BYREF_ARGS="--backward-byref"
  [ -n "$BYREF_POOL" ] && BYREF_ARGS="$BYREF_ARGS --backward-byref-pool $BYREF_POOL"
fi
echo "[byref] $CFG/$SET arm=$ARM pool='${BYREF_POOL:-unbounded}' policy=$BP value=$BV"
echo "[byref] out=$OUT"

if [ -s "$OUT" ] && python3 -c "
import json,sys
d=json.load(open('$OUT'))
rows=[len(s.get('rows') or []) for s in d['systems'].values() if s.get('rows')]
sys.exit(0 if rows and min(rows)>0 else 1)" 2>/dev/null; then
  echo "already merged and parseable: $OUT"; echo "BYREF $ARM $SET DONE"; exit 0
elif [ -s "$OUT" ]; then
  echo "existing $OUT is truncated -- redoing the merge"; rm -f "$OUT"
fi

# Chunk directory keyed on nets AND arm AND pool, so the two arms can never
# reuse each other's chunks (the track1 provenance rule).
STAMP=$(printf '%s\n%s\n%s\n%s\n' "$BP" "$BV" "$ARM" "$BYREF_POOL" | sha256sum | cut -c1-12)
RUN="$REPO/runs/byref/$CFG.$SET.$ARM.$STAMP"
mkdir -p "$RUN"
printf 'policy=%s\nvalue=%s\narm=%s\npool=%s\n' "$BP" "$BV" "$ARM" "$BYREF_POOL" \
  > "$RUN/provenance.txt"
[ -e "$RUN/chunk.000.jsonl" ] || \
  split -d -a 3 -l 8 "$INST" "$RUN/chunk." --additional-suffix=.jsonl

run_chunk() {
  local chunk="$1"; local base; base=$(basename "$chunk" .jsonl)
  local out="$RUN/$base.json"
  [ -s "$out" ] && return 0
  if python3 -m eval.compare --instances "$chunk" --expansions 1200 --k 5 \
       --backward-policy "$BP" --backward-value "$BV" \
       --backward-anytime --backward-b2 $BYREF_ARGS \
       --forward-ckpts "" --device cpu --dump-moves \
       --out "$out.tmp.$$" --md /dev/null > "$RUN/$base.log" 2>&1; then
    mv "$out.tmp.$$" "$out"; echo "done $base $(date +%H:%M:%S)"
  else
    echo "FAIL $base (see $RUN/$base.log)"; return 1
  fi
}
export -f run_chunk
export RUN BP BV BYREF_ARGS

ls "$RUN"/chunk.*.jsonl | xargs -P 8 -n 1 -I{} bash -c 'run_chunk "$@"' _ {}

NC=$(ls "$RUN"/chunk.*.jsonl | wc -l)
ND=$(ls "$RUN"/chunk.*.json 2>/dev/null | wc -l)
if [ "$ND" -ne "$NC" ]; then
  echo "MERGE SKIPPED: $ND/$NC chunks done (resubmit to resume)"
  echo "BYREF $ARM $SET PARTIAL"; exit 0
fi
python3 eval/merge_compare_shards.py --shards $(ls "$RUN"/chunk.*.json | sort) \
  --instances "$INST" --out "$OUT" --md /dev/null || \
  { echo "FATAL: merge failed"; exit 1; }

echo "== independent replay certification =="
python3 -m eval.replay_validate --compare "$OUT" --env-dir "$ENVDIR"
RC=$?
if [ "$RC" -ne 0 ]; then
  mv -f "$OUT" "$OUT.uncertified" 2>/dev/null
  echo "BYREF $ARM $SET REPLAY-FAILED rc=$RC (quarantined at $OUT.uncertified)"
  exit 1
fi

# Arm summary + by-reference supply counters (the on-arm's whole point: did
# by-reference candidates reach the nets, and did the nets ever shortlist one?)
python3 -c "
import json
d=json.load(open('$OUT'))
for nm,s in d['systems'].items():
    a=s.get('aggregate') or {}
    acc=a.get('accounting') or {}
    print(f'[byref] {nm}: solved {a.get(\"solved\")}/{a.get(\"n\")} '
          f'mean_expansions={a.get(\"mean_expansions\")} '
          f'byref_ranked/inst={acc.get(\"byref_cands_ranked\")} '
          f'byref_topk/inst={acc.get(\"byref_cands_topk\")}')
p='$PROD'
try:
    q=json.load(open(p))
    for nm,s in q['systems'].items():
        a=s.get('aggregate') or {}
        print(f'[byref] production control {nm}: solved {a.get(\"solved\")}/{a.get(\"n\")}')
except Exception as e:
    print('[byref] production control unreadable:', e)
"
echo "BYREF $ARM $SET DONE"
