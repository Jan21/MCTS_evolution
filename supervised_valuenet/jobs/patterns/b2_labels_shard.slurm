#!/bin/bash
# B2 label generation, RECALIBRATED and shardable. Replaces b2_labels_one.slurm
# for production use; that script's budgets (50k/100k/200k iterations) made the
# campaign cost ~441 node-hours and it produced nothing (FINDINGS 30).
#
#   sbatch -J rr-b2lab-g16r4-s0 jobs/patterns/b2_labels_shard.slurm g16r4 0 8
#   sbatch -J rr-b2lab-g16r6    jobs/patterns/b2_labels_shard.slurm g16r6 0 1
# arg1 = config, arg2 = shard index, arg3 = shard count.
# Output: scaling/data/<cfg>/backward_b2.rust.shard<S>of<N>.jsonl
#         (or backward_b2.rust.jsonl when N == 1)
# Concatenate shards with jobs/patterns/b2_labels_merge.sh once all land.
#
# CALIBRATION, measured by b2_labels_smoke.slurm on 2026-07-26 (24 spread
# boards per config, per-graph 10):
#   config  old s/graph   new s/graph   full-config cost
#   g16r4       692          36.7          2.7 node-hours
#   g16r6     2,696           1.0          0.05
#   g16r8     2,418           1.5          0.06
#   g24r8     4,579           2.1          0.09
# Every smoked board reached its keeper target. Two calibration facts worth
# keeping:
#   * The cap is NOT monotone in wall-time. cap 2,000 was SLOWER than cap 5,000
#     at base (64.4 vs 36.7 s/graph) because a tighter cap lowers the success
#     rate, so the bridge must send more attempt waves to reach `per-graph`
#     keepers. 5,000 is the measured optimum, not the tightest feasible value.
#   * B2 emits ~18 records per instance, so --per-graph 10 over the full board
#     set still yields ~380k records at base -- 3.5x nn/data/combined_b1.jsonl
#     (108,902) -- while halving the attempt budget.
#
# METHODOLOGICAL NOTE that must travel with any retrained row: a 5,000-iteration
# cap biases the training set toward instances whose rollout succeeds quickly.
# The measured `ok` rate falls from 66.6% (50k budget) to 52.2% (5k) at base.
# This is a change in WHICH instances are labelled, not merely how many.
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
# WALLTIME 16 h, not 24: the cooling reservation (MOTD, daily 10:00-18:00 CEST)
# blocks any job requesting more than 16 hours from starting while it is in
# effect. Base needs ~21.5 h unsharded, hence --nshards 8 (~2.7 h per shard).
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/b2labels/%x-%j.out

set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

CFG="$1"; SHARD="${2:-0}"; NSHARDS="${3:-1}"
# arg4 overrides the iteration cap. 5000 is the throughput optimum, but a
# tighter cap also strips BY-REFERENCE candidates -- the very vocabulary the
# retraining exists to teach. Measured on the same 111 base boards: the
# by-reference share falls from 13.5% at cap 50k to 4.6% at cap 5k.
CAP="${4:-5000}"
PER_GRAPH=10
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
SV="$REPO/supervised_valuenet"
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$SV"
export PYTHONPATH=. OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
unset RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR

ENGINE=rust_datagen/target/release/datagen
"$ENGINE" selftest > /dev/null 2>&1 || { echo "FATAL: engine selftest failed"; exit 1; }

# The cap tag must survive sharding. It previously did not: the sharded branch
# wrote plain `backward_b2.rust.shard<S>of<N>.jsonl` whatever the cap, so a
# cap-20000 base run would have produced 32 shards indistinguishable from
# cap-5000 output, and b2_labels_merge.sh would have concatenated them straight
# over the production corpus. Caught before any file was written (32 jobs
# cancelled, 2026-07-28); the base name now always carries the cap.
if [ "$CAP" -eq 5000 ]; then
  STEM="backward_b2"
else
  STEM="backward_b2.cap${CAP}"
fi
if [ "$NSHARDS" -eq 1 ]; then
  OUT="scaling/data/$CFG/${STEM}.rust.jsonl"; TAG="$STEM"
else
  OUT="scaling/data/$CFG/${STEM}.rust.shard${SHARD}of${NSHARDS}.jsonl"
  TAG="${STEM}_s${SHARD}of${NSHARDS}"
fi
if [ -s "$OUT" ]; then
  echo "already present: $OUT"; echo "B2LABELS $CFG shard $SHARD DONE"; exit 0
fi

echo "[labels] $CFG shard $SHARD/$NSHARDS  cap=$CAP per-graph=$PER_GRAPH"
python3 -m scaling.rust_bridge --config "$CFG" --system backward --vocab b2 \
    --per-graph "$PER_GRAPH" --threads 16 --budget-iters "$CAP" --force \
    --nshards "$NSHARDS" --shard "$SHARD" \
    --out "$OUT.tmp" --work-tag "$TAG"
RC=$?
[ "$RC" -eq 0 ] && [ -s "$OUT.tmp" ] || { echo "FAIL rc=$RC"; exit 1; }
mv "$OUT.tmp" "$OUT"

wc -l "$OUT"
# QC gate 2 (handoff recipe): by-reference share -- candidates whose helper is
# not at any robot's start position. Expected roughly 5-20%; base measured 13%.
python3 - "$OUT" <<'EOF'
import json, sys
# Robot START positions are target_robot[0] plus every helpers[i][0]; there is
# NO `robot_positions` field in the label schema (an earlier version of this
# gate looked for one and would have reported 0.0% on every file). Validated
# against the partial output of job 4591709: 5698/42348 = 13.5%, matching the
# 13% recorded for base in FINDINGS 19.
n = byref = 0
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    r = json.loads(line)
    n += 1
    starts = {tuple(r["target_robot"][0])}
    starts |= {tuple(h[0]) for h in (r.get("helpers") or [])}
    if tuple(r["cand_helper"][0]) not in starts:
        byref += 1
print(f"QC by-reference share: {byref}/{n} = {byref/n*100:.1f}%" if n else "QC: empty")
EOF
echo "B2LABELS $CFG shard $SHARD DONE"
