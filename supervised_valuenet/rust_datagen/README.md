# rust_datagen

Rust port of the expensive core of training-data generation for the Ricochet
Robots planners: board precompute (slide graph + all-pairs distance tables),
the forward (move-level) exact labeler, and the backward (subgoal-level)
exact labeler. Python keeps orchestration and random sampling; the Rust
engine is a JSONL-in/JSONL-out CLI. Design: `DESIGN.md`. Correctness
evidence: `VERIFICATION.md`. Benchmarks with raw-log traces: `BENCH.md`.

## Build

```bash
# Rust toolchain (one-time; installs to ~/.cargo, no root needed)
curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"

cd supervised_valuenet/rust_datagen
cargo build --release          # binary: target/release/datagen
cargo test --release           # hermetic gates on the committed golden corpus
./target/release/datagen selftest
```

## Run one config end-to-end

The existing shard scripts gained an `--engine rust` switch; sampling stays
in Python (same RNG streams), the engine does the solving:

```bash
cd supervised_valuenet
PYTHONPATH=. python -m scaling.gen_data --config g16r6 --system backward \
    --per-graph 6 --engine rust          # -> scaling/data/g16r6/backward.rust.jsonl
PYTHONPATH=. python -m scaling.gen_data --config g16r6 --system forward \
    --per-graph 10 --score-candidates --engine rust
```

Outputs use `<system>.rust.jsonl` names — existing data is never
overwritten. Work/result files are kept under
`scaling/data/<config>/rust_work/` for audit. The engine defaults to 16
threads (shared box); the bridge accepts `--threads`.

Verification harness (Python reference vs Rust, per-decision label diff):

```bash
make smoke      # fixed-seed matrix: 16/24/32/64 x {4,8} robots x both tasks, ~9 min
make verify     # the full battery legs wired in CI form
make regression # the 12-instance defect corpus (gate 5)
```

## Measured speedups (paired, identical work; BENCH.md traces every number)

| what | Python (1 core) | Rust (1 core) | per-core speedup |
|---|---|---|---|
| backward labeling, g16r4 / g16r6 / g16r8 | 1.9–180 s per batch | 0.03–1.0 s | **53x / 179x / 169x** |
| backward labeling, g24r4 / g32r4 | | | **108x / 80x** |
| forward labeling (5 configs, with candidate scoring) | 100–257 s per batch | 1.5–4.4 s | **58–80x** |
| board precompute (full `from_env` equivalent), 16/24/32 | 8.2 / 60.4 / 222.5 s | 5.7 / 31.4 / 76.2 ms | **1549x / 2184x / 3831x** |
| all-pairs tables alone, 24 / 32 | 4.1 / 12.3 s | 31 / 76 ms | **132x / 161x** |

End-to-end, full g16r6 config (1050 boards) on 32 threads of a loaded box:
backward 141 s + forward 194 s ≈ **5.6 min** (Python pipeline: ~40–60 min on
many cores). 32×32 config: ~36 min extrapolated from 30-board samples
(target < 1 h). Forward regeneration is **byte-identical** to the shipped
`forward.jsonl` on all 1050 boards.

## Correctness status (details in VERIFICATION.md)

- Physics/graphs/tables: exact on 328 boards (2.94M edges), 10,168 slides,
  400 table hashes — zero differences.
- Per-decision label replay: 258,999 labels across sizes 16–64 and 4–8
  robots — zero differences (empirical zero; see VERIFICATION.md).
- Forward pipeline: exact everywhere, including capped searches.
- One documented blocker (not a port bug): end-to-end backward record
  streams inherit an order-sensitivity of the Python reference itself —
  Python's own labels change on 25/150 instances when the same board is
  loaded from pickle vs rebuilt, and its plan-search cost estimate is
  provably non-admissible on rare instances (committed repros:
  `golden/s511_evidence/`). VERIFICATION.md states the measured impact and
  the resolution options; adjudication is a project-owner decision.
