#!/bin/bash
# Smoke + shard-equivalence test. 2026-07-23: repointed from the retired
# karolina_bundle to the working repo; qgpu partition per the owner's
# standing order. 2026-07-24: walltime trimmed to 2 h with the owner's
# explicit approval — the run itself takes ~30 min, and on a saturated
# queue a short request backfills days earlier than a 24 h one (the 24 h
# floor exists to protect long chunk jobs, which this is not).
# 1) KAROLINA_PROMPT §3 step 5 acceptance recipe (20 instances, both planners;
#    origin machine scored 19/20 — expect the same).
# 2) The same 20 instances as two 10-instance shards, merged with
#    eval/merge_compare_shards.py; rows (minus wall-clock) must equal the full
#    run's rows, proving sharded lanes are equivalent to unsharded ones.
#SBATCH --job-name=rr-smoke
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 2
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/smoke/%x-%j.out

set -euo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/supervised_valuenet"
export PYTHONPATH=. OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
unset RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR   # base 16x16/4 config

RUN="$REPO/runs/smoke"
mkdir -p "$RUN"
head -20 eval/data/bench450.jsonl >  "$RUN/bench20.jsonl"
head -10 eval/data/bench450.jsonl >  "$RUN/bench20_a.jsonl"
sed -n 11,20p eval/data/bench450.jsonl > "$RUN/bench20_b.jsonl"

BFLAGS="--expansions 1200 --k 5 --backward-policy checkpoints_backward/policy_b1.ckpt \
 --backward-value checkpoints_backward/value_b1.ckpt --backward-anytime --backward-b1 --device cpu"

python3 -m eval.compare --instances "$RUN/bench20.jsonl"   $BFLAGS --out "$RUN/smoke20.json" --md /dev/null \
  > "$RUN/full.log" 2>&1 &
python3 -m eval.compare --instances "$RUN/bench20_a.jsonl" $BFLAGS --out "$RUN/shard_a.json" --md /dev/null \
  > "$RUN/shard_a.log" 2>&1 &
python3 -m eval.compare --instances "$RUN/bench20_b.jsonl" $BFLAGS --out "$RUN/shard_b.json" --md /dev/null \
  > "$RUN/shard_b.log" 2>&1 &
wait

python3 eval/merge_compare_shards.py --shards "$RUN/shard_a.json" "$RUN/shard_b.json" \
  --instances "$RUN/bench20.jsonl" --out "$RUN/merged20.json" --md /dev/null

python3 - "$RUN/smoke20.json" "$RUN/merged20.json" <<'EOF'
import json, sys
full, merged = (json.loads(open(p).read()) for p in sys.argv[1:3])
ok = True
for name, s in full["systems"].items():
    rf = [{k: v for k, v in r.items() if k != "seconds"} for r in s["rows"]]
    rm = [{k: v for k, v in r.items() if k != "seconds"}
          for r in merged["systems"][name]["rows"]]
    if rf != rm:
        ok = False
        diffs = [i for i, (x, y) in enumerate(zip(rf, rm)) if x != y]
        print(f"SHARD MISMATCH in {name!r} at rows {diffs}")
        for i in diffs[:3]:
            print("  full  :", rf[i]); print("  merged:", rm[i])
print("SHARD EQUIVALENCE OK" if ok else "SHARD EQUIVALENCE FAILED")
EOF

echo "SMOKE DONE"
tail -4 "$RUN/full.log"
