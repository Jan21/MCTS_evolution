#!/bin/bash
# SMOKE TEST for B2 label generation: measure the STEADY-STATE rate and keeper
# yield at a candidate iteration cap, on a spread sample of boards, before any
# full run is submitted.
#
# Why this exists (FINDINGS 30). The first B2 label campaign was sized from its
# opening graphs -- 58 s/graph over 8 boards -- and the steady-state rate turned
# out to be 692-4,579 s/graph, so the five jobs projected to ~441 node-hours
# against ~879 remaining and produced nothing. The cause is wanderers: 8.8%
# (base) to 16.1% (g16r8) of rollouts exhaust the iteration budget and consume
# 65-91% of ALL iterations, while a legitimate rollout has a median of 174-342
# iterations. An iteration budget does not bound wall-time; a SMALL one does.
#
#   sbatch -J rr-labsmoke-g16r4 jobs/patterns/b2_labels_smoke.slurm g16r4 5000 10 24
# arg1 = config, arg2 = --budget-iters cap, arg3 = --per-graph, arg4 = boards.
#
# Boards are sampled SPREAD across the id range (--nshards 64 --shard 0
# --limit N), not the first N consecutive, because the opening boards are
# exactly the ones that misled the first projection.
#
# Output is thrown away (scaling/data/<cfg>/smoke/), so this never touches a
# real dataset. What matters is the printed rate and yield.
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/labsmoke/%x-%j.out

set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

CFG="${1:-g16r4}"; CAP="${2:-5000}"; PG="${3:-10}"; NB="${4:-24}"
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
SV="$REPO/supervised_valuenet"
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$SV"
export PYTHONPATH=. OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1

ENGINE=rust_datagen/target/release/datagen
"$ENGINE" selftest > /dev/null 2>&1 || { echo "FATAL: engine selftest failed"; exit 1; }

OUTDIR="scaling/data/$CFG/smoke"
mkdir -p "$OUTDIR"
OUT="$OUTDIR/smoke_cap${CAP}_pg${PG}.jsonl"
rm -f "$OUT"

echo "[smoke] cfg=$CFG cap=$CAP per-graph=$PG boards=$NB (spread, shard 0 of 64)"
START=$(date +%s)
python3 -m scaling.rust_bridge --config "$CFG" --system backward --vocab b2 \
    --per-graph "$PG" --threads 16 --budget-iters "$CAP" --force \
    --nshards 64 --shard 0 --limit "$NB" \
    --out "$OUT" --work-tag "smoke_cap${CAP}_pg${PG}"
RC=$?
END=$(date +%s)

echo "=== SMOKE SUMMARY ==="
python3 - "$CFG" "$CAP" "$PG" "$NB" "$OUT" "$((END-START))" <<'EOF'
import collections, json, sys, statistics
cfg, cap, pg, nb, out, secs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6])
work = f"scaling/data/{cfg}/rust_work/smoke_cap{cap}_pg{pg}.results.jsonl"
st = collections.Counter(); iters = collections.defaultdict(list)
board_ok = collections.Counter()
try:
    for line in open(work):
        if not line.strip(): continue
        d = json.loads(line)
        st[d.get("status")] += 1
        iters[d.get("status")].append(d.get("iters", 0))
        if d.get("status") == "ok":
            board_ok[d["id"].split(":")[0]] += 1
except FileNotFoundError:
    print("  (no engine result file)")
recs = sum(1 for _ in open(out)) if __import__("os").path.exists(out) else 0
tot = sum(st.values()) or 1
print(f"  boards={nb}  wall={secs}s  ->  {secs/max(nb,1):.1f} s/graph")
print(f"  records={recs:,}  ({recs/max(nb,1):.0f} per board)")
for s, c in st.most_common():
    v = sorted(iters[s])
    print(f"    {s:18s} {c:6,} ({c/tot*100:5.1f}%)  median iters={statistics.median(v) if v else 0:,.0f}")
full = sum(1 for b, c in board_ok.items() if c >= pg)
print(f"  boards reaching {pg} keepers: {full}/{len(board_ok)}")
for boards, label in ((2112, "g16r4"), (1050, "others")):
    h = boards * secs / max(nb, 1) / 3600
    print(f"  PROJECTION if this rate holds for {boards} boards ({label}): "
          f"{h:.1f} h  =  {h*0.125:.1f} node-hours (1 GPU)")
EOF
echo "LABSMOKE $CFG cap=$CAP pg=$PG DONE rc=$RC"
