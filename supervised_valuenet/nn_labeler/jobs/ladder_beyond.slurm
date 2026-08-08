#!/bin/bash
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH -t 16:00:00
#SBATCH -J rr-nnlab-beyond
#SBATCH -o /scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/nnlab/ladder_beyond_%j.out
#
# BEYOND THE ENVELOPE (owner-approved 2026-08-05; licensed by FINDINGS 70):
# label boards the exact solver can NEVER check (Rust hard limit n<=64).
# Labels here are certified-valid upper bounds -- physics certification is
# solver-free -- but their gap to optimal is unmeasurable by construction.
# Their only warranty is the flat 17-64 curve (86-93% argmin, FINDINGS
# 65/69/70). Every output is marked UNVERIFIABLE in name and manifest.
#
# This job is the PROBE phase: fresh-generation on 20 boards, certification
# stats, throughput -- the numbers that size any production labeling and the
# downstream-validation phase (train a planner on these labels, measure
# playable-moves solve rate -- the only quality signal that exists up here).
#
#   sbatch -J rr-nnlab-beyond-g80r4 supervised_valuenet/nn_labeler/jobs/ladder_beyond.slurm g80r4
set -u
CFG=${1:?config required (g80r4|g96r4)}
module purge
ml Python/3.11.5-GCCcore-13.2.0
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
cd "$REPO/supervised_valuenet"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
nvidia-smi -L

CKPT=${NNLAB_CKPT:-nn_labeler/banked/prod_v1_s11.ckpt}
echo "BEYOND $CFG LABELER $CKPT (v1 -- the net the 17-64 curve certifies)"
[ -f "$CKPT" ] || { echo "BEYOND MISSING CKPT"; exit 1; }

read -r N WALLS ENVDIR <<< "$(python -c "
from scaling.configs import get
c = get('$CFG'); print(c.grid, c.walls, c.env_dir)")"
case "$CFG" in
    g80r4) TMO=900 ;;  g96r4) TMO=1200 ;;
    *) echo "BEYOND UNKNOWN CFG $CFG"; exit 1 ;;
esac

# 1. lean boards (the only kind that exist at this size)
t0=$SECONDS
python -m nn_labeler.leanboard --n "$N" --robots 4 --walls "$WALLS" \
    --ids 0-19 --out "$ENVDIR" \
    || { echo "BEYOND $CFG BOARDS FAILED"; exit 1; }
echo "BEYOND $CFG boards $((SECONDS-t0))s"

# 2. fresh-generation probe -- UNVERIFIABLE labels, certified upper bounds
OUT="nn_labeler/results/beyond_UNVERIFIABLE_${CFG}.jsonl"
if [ ! -f "$OUT" ]; then
    t0=$SECONDS
    python -m nn_labeler.descent --config "$CFG" --ckpt "$CKPT" \
        --graphs 0-19 --per-graph 10 --boards lean --device cuda \
        --timeout "$TMO" --out "$OUT" \
        || { echo "BEYOND $CFG GEN FAILED"; exit 1; }
    echo "BEYOND $CFG gen $(wc -l < "$OUT") records $((SECONDS-t0))s"
fi
python - "$OUT" "$CFG" <<'PY'
import json, sys
path, cfg = sys.argv[1], sys.argv[2]
n = certified = 0
for line in open(path):
    r = json.loads(line)
    n += 1
    certified += bool(r.get("ctg_certified"))
print(f"BEYOND {cfg} STATS: {n} records, {certified} certified "
      f"({100*certified/max(n,1):.1f}%) -- gap to optimal UNMEASURABLE at n>64")
PY
echo "NNLAB BEYOND DONE $CFG"
