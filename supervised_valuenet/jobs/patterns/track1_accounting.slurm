#!/bin/bash
# TRACK 1 -- compute-accounting pass (publishability objection 1.1).
#
# Runs BOTH systems over the same seeded subsample of a pinned set with
# --count-slides, so backward and forward physics work are measured in the ONE
# unit they share: invocations of simulate.slide. Deliberately a separate job
# from track1_rows.slurm, because the counter wraps the hottest function in the
# codebase -- a counted run's wall-clock is not comparable to an uncounted one,
# so the headline rows must never carry it.
#
#   sbatch -J rr-acct-g16r8-frn jobs/patterns/track1_accounting.slurm g16r8 frontier 32
#   sbatch -J rr-acct-base      jobs/patterns/track1_accounting.slurm base   base450  32
# arg1 = config, arg2 = graded|frontier|base450, arg3 = subsample size.
#
# Subsample: seeded shuffle (seed 1) then sorted back into benchmark order --
# the same recipe fwd_budget_probe.slurm uses, so the two probes are comparable
# and the choice is reproducible rather than "the first N".
#
# Output: runs/accounting/<cfg>.<set>/{backward,forward}.json, folded into
# eval/results/compute_accounting.json by eval/accounting_summary.py, which
# reports MEDIAN and tail per rung -- never the mean alone. FINDINGS 25: over a
# 20-instance base slice one unsolved puzzle carried 89.6% of the backward
# planner's total slide calls, so a mean misrepresents both systems.
#
# Cost scales with the FORWARD arm, which is the expensive one at the frontier
# (measured at budget 1200: 1,166 s/instance at 24x24/8, 2,380 s at 32x32).
# Use n=32 at the 16x16 rungs and n=16 at 24x24/8 and 32x32 unless there is a
# reason to pay more; state the projection before submitting.
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 2
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/accounting/%x-%j.out

set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

CFG="$1"; SET="$2"; NINST="$3"
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
SV="$REPO/supervised_valuenet"
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$SV"
export PYTHONPATH=. OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
unset RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR

case "$CFG" in
  base)  INST=eval/data/bench450.jsonl; ENVDIR="$SV/environments"
         FWD=move_planner/checkpoints/candidate_scored.ckpt ;;   # the BEST forward system
  g16r6) export RR_GRID=16 RR_ROBOTS=6 RR_WALLS=48
         FWD="lightning_logs/version_43/checkpoints/epoch=7-step=43224.ckpt" ;;
  g16r8) export RR_GRID=16 RR_ROBOTS=8 RR_WALLS=48
         FWD="lightning_logs/version_44/checkpoints/epoch=6-step=46767.ckpt" ;;
  g24r8) export RR_GRID=24 RR_ROBOTS=8 RR_WALLS=108
         FWD="lightning_logs/version_47/checkpoints/epoch=3-step=115144.ckpt" ;;
  g32r4) export RR_GRID=32 RR_ROBOTS=4 RR_WALLS=192
         FWD="lightning_logs/version_45/checkpoints/epoch=3-step=158564.ckpt" ;;
  *) echo "FATAL: unknown config $CFG"; exit 1 ;;
esac
if [ "$CFG" != base ]; then
  ENVDIR="$SV/environments_$CFG"; export RR_ENV_DIR="$ENVDIR"
  [ "$SET" = graded ] && INST="scaling/data/$CFG/bench.solved.jsonl" \
                      || INST="scaling/data/$CFG/bench.unsolved.jsonl"
fi

MANIFEST=scaling/runs/b2_banked.json
[ -s "$MANIFEST" ] || { echo "FATAL: $MANIFEST missing -- bank the nets first"; exit 1; }
read -r BP BV < <(python3 -c "
import json,sys
m=json.load(open('$MANIFEST')).get('$CFG')
if not m: sys.exit('FATAL: no banked entry for $CFG')
print(m['policy'], m['value'])") || exit 1

RUN="$REPO/runs/accounting/$CFG.$SET"
mkdir -p "$RUN"
SUB="$RUN/subsample.jsonl"
if [ ! -s "$SUB" ]; then
  python3 - "$INST" "$NINST" "$SUB" <<'EOF'
import json, os, random, sys
src, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
lines = open(src).read().splitlines()
idx = list(range(len(lines)))
random.Random(1).shuffle(idx)
keep = sorted(idx[:n])                     # back into benchmark order
with open(out + ".tmp", "w") as f:
    for i in keep:
        f.write(lines[i] + "\n")
os.replace(out + ".tmp", out)
print(f"subsample: {min(n,len(lines))} of {len(lines)} instances (seed 1)")
EOF
fi

if [ ! -s "$RUN/backward.json" ]; then
  python3 -m eval.compare --instances "$SUB" --expansions 1200 --k 5 \
    --backward-policy "$BP" --backward-value "$BV" \
    --backward-anytime --backward-b2 --forward-ckpts "" --device cpu \
    --count-slides --dump-moves \
    --out "$RUN/backward.json.tmp" --md /dev/null > "$RUN/backward.log" 2>&1 \
    && mv "$RUN/backward.json.tmp" "$RUN/backward.json" \
    || { echo "FAIL backward (see $RUN/backward.log)"; exit 1; }
fi
echo "backward accounting done $(date +%H:%M:%S)"

if [ ! -s "$RUN/forward.json" ]; then
  python3 -m eval.compare --instances "$SUB" --expansions 1200 --k 5 \
    --skip-backward --forward-ckpts "$FWD" --device cpu \
    --count-slides --dump-moves \
    --out "$RUN/forward.json.tmp" --md /dev/null > "$RUN/forward.log" 2>&1 \
    && mv "$RUN/forward.json.tmp" "$RUN/forward.json" \
    || { echo "FAIL forward (see $RUN/forward.log)"; exit 1; }
fi
echo "forward accounting done $(date +%H:%M:%S)"

for s in backward forward; do
  echo "== replay_validate $s =="
  python3 -m eval.replay_validate --compare "$RUN/$s.json" --env-dir "$ENVDIR" \
    2>&1 | tail -1
done
echo "ACCOUNTING $CFG $SET DONE"
