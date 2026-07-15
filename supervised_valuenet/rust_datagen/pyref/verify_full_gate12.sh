#!/usr/bin/env bash
# Full gate 1-2 verification (DESIGN §9, task A brief item 6).
#
# Regenerates the FULL gate corpus into a temp dir (or $1 if given):
#   - 128 stock boards (dumped as pickled)     tables on the first 50
#   - 100 fresh 16x16, 50 fresh 24x24, 50 fresh 32x32 boards
#     (tables on the first 50 per size)
#   - 31 slide cases per board  -> 328 * 31 = 10168 slides (>= 10k)
# then runs the Rust side and prints exact pass counts.
#
# Usage: pyref/verify_full_gate12.sh [corpus_dir]
# Env:   PY     python with networkx 3.3 (default: ph_main conda env)
#        PROCS  dumper worker processes (default 8, capped at 16 in the dumper)
set -euo pipefail

PY=${PY:-/home/p23131/.conda/envs/ph_main/bin/python3}
PROCS=${PROCS:-8}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CRATE=$(dirname "$SCRIPT_DIR")                    # rust_datagen/
SV=$(dirname "$CRATE")                            # supervised_valuenet/
OUT=${1:-$(mktemp -d /tmp/rust_datagen_gate12.XXXXXX)}
SEED=7

echo "== corpus dir: $OUT"
cd "$SV"
export PYTHONPATH=.

DUMP="rust_datagen/pyref/dump_reference.py"
echo "== dumping 128 stock boards (tables on first 50)"
"$PY" "$DUMP" --stock --tables --tables-limit 50 --slides 31 \
      --seed $SEED --gzip --procs "$PROCS" --out "$OUT" | tail -1
echo "== dumping 100 fresh 16x16 (tables on first 50)"
"$PY" "$DUMP" --fresh --n 16 --count 100 --tables --tables-limit 50 --slides 31 \
      --seed $SEED --gzip --procs "$PROCS" --out "$OUT" | tail -1
echo "== dumping 50 fresh 24x24 (tables on all)"
"$PY" "$DUMP" --fresh --n 24 --count 50 --tables --tables-limit 50 --slides 31 \
      --seed $SEED --gzip --procs "$PROCS" --out "$OUT" | tail -1
echo "== dumping 50 fresh 32x32 (tables on all)"
"$PY" "$DUMP" --fresh --n 32 --count 50 --tables --tables-limit 50 --slides 31 \
      --seed $SEED --gzip --procs "$PROCS" --out "$OUT" | tail -1

echo "== running Rust gate_full"
cd "$CRATE"
export PATH="$HOME/.cargo/bin:$PATH"
RUST_DATAGEN_GATE12_DIR="$OUT" RAYON_NUM_THREADS=16 \
    cargo test --release -j 16 -- --ignored gate_full --nocapture
