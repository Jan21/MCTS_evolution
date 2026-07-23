#!/bin/bash
# Lever B2 vocabulary gate for rust_datagen (VERIFICATION.md gate-3 pattern):
# pinned decision contexts dumped by the PYTHON reference with the extended
# vocabulary enabled, replayed by the Rust engine, ZERO label differences
# required (pyref/diff_labels.py exits nonzero on any diff). Four legs:
#   1. base 16x16/4, vocab b2   (both mechanisms on the standard board)
#   2. base 16x16/4, vocab b1   (isolates the transient-support path)
#   3. 16x16/8,      vocab b2   (robot axis - by-reference-rich decisions)
#   4. 24x24/8,      vocab b2   (grid axis)
# Python-side b2 rollouts can wander; every instance runs under the dumper's
# SIGALRM cap, so each leg is time-bounded. CPU-only inside a 1-GPU job.
#SBATCH --job-name=rr-b2gate
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/b2gate/%x-%j.out

set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0

REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/supervised_valuenet"
export PYTHONPATH=. OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
unset RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR

RUN="$REPO/runs/b2gate"
ENGINE=rust_datagen/target/release/datagen
mkdir -p "$RUN"
FAIL=0

leg () {  # leg <name> <dump-args...>
  local name="$1"; shift
  local dump="$RUN/dump_$name.jsonl" res="$RUN/replay_$name.jsonl"
  echo "=== LEG $name: dump ==="
  if [ ! -s "$dump" ]; then
    python3 rust_datagen/pyref/dump_decisions.py --task backward --seed 7 \
      --out "$dump.tmp" "$@" && mv "$dump.tmp" "$dump" || { echo "LEG $name DUMP FAILED"; FAIL=1; return; }
  fi
  echo "=== LEG $name: replay + diff ($(wc -l < "$dump") decisions) ==="
  "$ENGINE" replay --work "$dump" --out "$res" --threads 12 --quiet \
    || { echo "LEG $name REPLAY FAILED"; FAIL=1; return; }
  if python3 rust_datagen/pyref/diff_labels.py --dump "$dump" --results "$res"; then
    echo "LEG $name ZERO DIFFS"
  else
    echo "LEG $name HAS DIFFS"; FAIL=1
  fi
}

leg base_b2  --config g16r4 --vocab b2 --instances 8 --n-boards 6 --workers 12 --instance-timeout 300
leg base_b1  --config g16r4 --vocab b1 --instances 6 --n-boards 4 --workers 12 --instance-timeout 300
leg g16r8_b2 --config g16r8 --vocab b2 --instances 6 --n-boards 4 --workers 12 --instance-timeout 300
leg g24r8_b2 --config g24r8 --vocab b2 --instances 4 --n-boards 3 --workers 12 --instance-timeout 600

if [ "$FAIL" -eq 0 ]; then echo "B2 GATE ALL LEGS ZERO DIFFS"; else echo "B2 GATE FAILED"; exit 1; fi
