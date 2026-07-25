#!/bin/bash
# Extended-budget FORWARD probe on a frontier subsample: answers "is the
# forward planner's frontier collapse a budget-cap artifact?" (the 1200-cap
# rows are budget-saturated; see analysis/publishability.md objection 0.2/1.2).
# Forward-only (existing per-config forward control nets; no backward lane),
# so it does NOT wait for the B2 retraining chain. Usage:
#   sbatch -J rr-fwdprobe-g24r8 jobs/patterns/fwd_budget_probe.slurm g24r8 6000 32
#   sbatch -J rr-fwdprobe-g32r4 jobs/patterns/fwd_budget_probe.slurm g32r4 4800 24
#   sbatch -J rr-fwdprobe-g16r6 jobs/patterns/fwd_budget_probe.slurm g16r6 4800 32
#   sbatch -J rr-fwdprobe-g16r8 jobs/patterns/fwd_budget_probe.slurm g16r8 4800 32
# arg1 = config, arg2 = expansion budget, arg3 = subsample size.
# Subsample: seeded shuffle (seed 1) of bench.unsolved.jsonl, first N —
# deterministic and recorded next to the output. Chunks of 2 instances,
# idempotent (rerun resumes). Output:
#   scaling/results/<cfg>/forward_probe_e<budget>.json
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 2
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/fwdprobe/%x-%j.out

set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

CFG="$1"; BUD="$2"; NINST="$3"
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
SV="$REPO/supervised_valuenet"
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$SV"
export PYTHONPATH=. OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1

case "$CFG" in
  g24r8) export RR_GRID=24 RR_ROBOTS=8 RR_WALLS=108 RR_ENV_DIR="$SV/environments_g24r8"
         FWD="lightning_logs/version_47/checkpoints/epoch=3-step=115144.ckpt" ;;
  g32r4) export RR_GRID=32 RR_ROBOTS=4 RR_WALLS=192 RR_ENV_DIR="$SV/environments_g32r4"
         FWD="lightning_logs/version_45/checkpoints/epoch=3-step=158564.ckpt" ;;
  # The 16x16 rungs are the ones whose forward frontier curves are NOT
  # saturated at the 1200 cap (+6.0 pts over the last 200 expansions at g16r6,
  # +4.3 at g16r8, vs +1.0 / +0.4 on the grid axis -- FINDINGS 20/21). Each
  # uses the SAME forward control checkpoint as its stored frontier row
  # (scaling/results/<cfg>/comparison_ungraded.json protocol), so the probe
  # extends that exact row's budget and nothing else.
  g16r6) export RR_GRID=16 RR_ROBOTS=6 RR_WALLS=48 RR_ENV_DIR="$SV/environments_g16r6"
         FWD="lightning_logs/version_43/checkpoints/epoch=7-step=43224.ckpt" ;;
  g16r8) export RR_GRID=16 RR_ROBOTS=8 RR_WALLS=48 RR_ENV_DIR="$SV/environments_g16r8"
         FWD="lightning_logs/version_44/checkpoints/epoch=6-step=46767.ckpt" ;;
  *) echo "unknown config $CFG"; exit 1 ;;
esac

RUN="$REPO/runs/fwdprobe/$CFG.e$BUD"
mkdir -p "$RUN"
SUB="$RUN/subsample.jsonl"
if [ ! -s "$SUB" ]; then
  python3 - "$CFG" "$NINST" "$SUB" <<'EOF'
import json, random, sys
cfg, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
lines = open(f"scaling/data/{cfg}/bench.unsolved.jsonl").read().splitlines()
idx = list(range(len(lines)))
random.Random(1).shuffle(idx)
keep = sorted(idx[:n])          # benchmark order, required by the merger
with open(out + ".tmp", "w") as f:
    for i in keep:
        f.write(lines[i] + "\n")
import os; os.replace(out + ".tmp", out)
print(f"subsample: {n} of {len(lines)} frontier instances (seed 1)")
EOF
fi
[ -e "$RUN/chunk.00.jsonl" ] || split -d -a 2 -l 2 "$SUB" "$RUN/chunk." --additional-suffix=.jsonl

run_one() {
  local chunk="$1"; local base; base=$(basename "$chunk" .jsonl)
  local out="$RUN/$base.json"
  [ -s "$out" ] && { echo "skip $base (done)"; return 0; }
  echo "start $base $(date +%H:%M:%S)"
  if python3 -m eval.compare --instances "$chunk" --expansions "$BUD" --k 5 \
       --forward-ckpts "$FWD" --device cpu \
       --out "$out.tmp.$$" --md /dev/null > "$RUN/$base.log" 2>&1; then
    mv "$out.tmp.$$" "$out"
    echo "done  $base $(date +%H:%M:%S)"
  else
    echo "FAIL  $base (see $base.log)"; return 1
  fi
}
export -f run_one
export RUN BUD FWD

ls "$RUN"/chunk.*.jsonl | xargs -P 4 -n 1 -I{} bash -c 'run_one "$@"' _ {}

NC=$(ls "$RUN"/chunk.*.jsonl | wc -l)
ND=$(ls "$RUN"/chunk.*.json 2>/dev/null | wc -l)
if [ "$ND" -ne "$NC" ]; then
  echo "MERGE SKIPPED: $ND/$NC chunks done (resubmit to resume)"
  echo "FWDPROBE $CFG PARTIAL"; exit 0
fi
python3 eval/merge_compare_shards.py --shards $(ls "$RUN"/chunk.*.json | sort) \
  --instances "$SUB" --out "scaling/results/$CFG/forward_probe_e$BUD.json" --md /dev/null
echo "FWDPROBE $CFG e$BUD DONE"
