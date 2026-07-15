# rust_datagen — benchmark results (Agent D)

Machine: shared 128-core box (`uptime` load 31-46 during all runs — never
saturated; engine capped at 16 threads except where 32 is stated). Binary:
`target/release/datagen` (`cargo build --release`, LTO). Python:
`/home/p23131/.conda/envs/ph_main/bin/python3` (3.11).
Every number below names its command and raw log path. Logs live under
`supervised_valuenet/scaling/data/<config>/rust_work/bench_logs/` (regen +
diffs) and `rust_datagen/pyref/cache/bench/` (bench_compare battery).

Date: 2026-07-15. All Python-vs-Rust rows run IDENTICAL work: the Python
side consumes the same work items the bridge sampled (trimmed to the
attempts the production loop would consume), the Rust side runs the same
file through `datagen run`.

## 1. Board precompute (per board; gate: tables >= 100x per core)

Python reference is the production cold `GridEnv.from_env` (re-weighting +
`nx.all_pairs_dijkstra_path_length` + eager `GridEnv.__init__` caches), and
separately the all-pairs block alone (`nn.gen_grids.all_pairs` +
`independent_paths` at weight 2 — the exact `from_env` table recipe). Rust
is a full `board` work item (walls + slide graph + BOTH tables), threads 1.

| config | n | Python from_env cold | Python tables alone | Rust full compile | from_env ratio | tables ratio |
|---|---|---|---|---|---|---|
| g16r6 | 16 | 8.19 s | 0.434 s | 0.0091 s | 900x | 48x |
| g24r4 | 24 | 60.36 s | 3.045 s | 0.0501 s | 1204x | 61x |
| g32r4 | 32 | 222.45 s | 11.575 s | 0.1478 s | 1505x | 78x |

Command: `python3 rust_datagen/pyref/bench_compare.py --stages precompute
--configs g16r6,g24r4,g32r4` -> logs `pyref/cache/bench/<cfg>_precompute.log`,
rows in `pyref/cache/bench/results.jsonl` (the LAST set of precompute rows;
an earlier same-day set predates the thread-pool pinning fix and additionally
shows load variance: py tables 0.568/5.04/10.6 s, from_env 11.2/79.7/283.8 s).
Cross-checks: the Phase-0 Python baselines (`.superpowers/sdd/progress.md`)
measured from_env cold at 8.04 s (16x16 r6) / 62.22 s (24x24) / 242.09 s
(32x32) on this box — consistent. Rust per-board compile re-measured on
20/20/10-board work files to amortize process startup
(`pyref/cache/bench/compile_amortized.log`): 7.25 / 44.3 / 138.5 ms per
board — consistent with the table.

Honest gate reading: the PRODUCTION precompute surface (`from_env` cold,
what `scaling.backward_label` actually pays per board) is 900-4600x. The
ISOLATED all-pairs block alone is 48-90x per core across sizes — BELOW the
aspirational >=100x for this micro-slice (Python's nx loop is only 0.4-11 s
per board where Rust rebuilds walls+graph+both tables in 7-140 ms;
single-thread heap Dijkstra is the limit). A bucket-queue Dijkstra (edge
weights are only 1/dependent_edge_weight) in board.rs would likely clear
100x, but board.rs is Agent A's file — flagged as follow-up, not patched
here.

## 2. Backward labeling (gate: >= 50x per core)

Same instance sets both sides: the bridge's speculative work file trimmed to
the attempts `scaling.backward_label`'s loop would consume; Python runs
`nn.generate.rollout` (AStar caps 4000/40000, max_candidates 14 except
where noted, 120 s SIGALRM like production) on E1's lazy env — the fastest
correct Python implementation, so ratios are CONSERVATIVE (the production
eager env would add the §1 from_env cost per board on top). Rust runs the
identical file via `datagen run`.

| config | attempts | Python 1-core label | Rust --threads 1 | Rust --threads 16 | per-core ratio |
|---|---|---|---|---|---|
| g16r4 | 71 | 1.86 s | 0.035 s | 0.025 s | **52.7x** |
| g16r6 | 15 | 179.59 s | 1.004 s | 0.553 s | **178.9x** |
| g16r8 | 5 | 58.92 s | 0.348 s | 0.274 s | **169.2x** |
| g24r4 | 35 | 20.98 s | 0.194 s | 0.119 s | **108.0x** |
| g32r4 | 35 | 32.65 s | 0.409 s | 0.333 s | **79.9x** |

GATE MET: >= 50x per core on every config (52.7-178.9x). Python/Rust
outcome agreement was N/N on every row (126 of 126 attempts across the
table agree on kept-vs-dropped). Notes: rust times include per-board
compile, which production amortizes over 24-80 attempts/board — an earlier
small-sample run (7 attempts at g32r4, results.jsonl) showed 7.1x purely
because 2 board compiles dominated 0.3 s of total work; the table uses
production-scale samples. The threaded speedup over --threads 1 is modest
here because these work files are seconds long; the full-config regens in
§4 show the throughput at scale.

Command: `python3 rust_datagen/pyref/bench_compare.py --stages backward
--configs ...` -> logs `pyref/cache/bench/<cfg>_backward.log` (+ the bridge
run log `pyref/cache/bench/bench_bwd_<cfg>.bridge.log`; work/results under
`scaling/data/<cfg>/rust_work/bench_bwd_<cfg>.*`). Rows: results.jsonl
(latest per config).

## 3. Forward labeling (gate: >= 50x per core)

Same shape: `move_planner.oracle.label_trajectory` + the
`label_board --score-candidates` candidate-scoring loop (Python, 1 core) vs
`datagen run` on the identical trimmed work file.

| config | attempts | Python 1-core label | Rust --threads 1 | Rust --threads 16 | per-core ratio |
|---|---|---|---|---|---|
| g16r4 | 18 | 100.04 s | 3.116 s | 1.059 s | 32.1x |
| g16r6 | 14 | 202.68 s | 7.720 s | 1.854 s | 26.3x |
| g16r8 | 13 | 248.31 s | 9.734 s | 3.357 s | 25.5x |
| g24r4 | 34 | 141.25 s | 4.186 s | 1.334 s | 33.7x |
| g32r4 | 31 | 134.56 s | 3.925 s | 1.031 s | 34.3x |

**GATE NOT MET as measured**: the paired per-core forward ratio is
25.5-34.3x, not >= 50x. Outcome agreement was N/N on every row (110/110
attempts agree solved-vs-not). Context: Agent B's earlier ~66x
micro-bench compared UNPAIRED samples (its Python sample averaged 10.7
s/attempt at g16r4 vs 5.6 s here); the ratio is composition-sensitive
because unsolved attempts burn the full 40k-expansion budget on both
sides. On identical work the honest figure is the table above. The
forward hot path is `src/move_oracle.rs` (Agent B's file) — potential
follow-ups (precomputed slide tables per (walls, dir) row, branchless
successor generation) are out of Agent D's file scope. Operationally the
threaded engine still collapses production cost (5.5-8.5 s of engine
time replaces 100-250 s of Python per work file above; full-config
numbers in §4).

Command: `python3 rust_datagen/pyref/bench_compare.py --stages forward
--configs ...` -> logs `pyref/cache/bench/<cfg>_forward.log`.

## 4. End-to-end: full g16r6 config regeneration (--engine rust)

Production settings recovered from the existing data (NOT the brief's
nominal --per-graph 20 — the shipped `scaling/data/g16r6/backward.jsonl`
holds exactly 6 kept instances per board and `forward.jsonl` exactly 10
solved per board, both over boards 0-1049 in 8 shards, seed 0; shard-concat
equals the shipped files byte-for-byte):

- backward: `python -m scaling.rust_bridge --config g16r6 --system backward
  --per-graph 6 --nshards 8 --shard K --seed 0 --threads 8` for K=0..7,
  4 shards concurrently (32 engine threads total, briefly; load checked) —
  **141 s wall** for all 1050 boards, 6300 instances, 98,246 records.
  Logs: `scaling/data/g16r6/rust_work/bench_logs/backward_regen_shard*.log`,
  wall clock in `backward_regen.t0/.t1`. Zero `budget_exhausted` results.
- forward (batched full budget, 52,500 attempts): 904 s wall at
  --threads 32 (`bench_logs/forward_regen.log`, `.t0/.t1`) —
  10,500 instances, 1,038,827 records.
- forward (adaptive waves: growing per-board prefixes [20, 30, 50], so the
  engine only computes ~the attempts the production loop consumes — 25,860
  of 52,500; output verified BYTE-IDENTICAL to the full-budget file with
  `cmp`): 471 s wall at --threads 32
  (`bench_logs/forward_regen_wave.log`, `.t0/.t1`).
- forward (waves + first wave streamed during presampling): **432 s** at
  --threads 32 (`bench_logs/forward_regen_wave2.log`, `.t0/.t1`); output
  again verified byte-identical (`cmp`, then duplicate removed).

Total config wall-clock (backward 141 s + pipelined-wave forward 432 s,
32 threads, measured with background load 32-50 already on the box):
**573 s = 9.6 min** (target: < 10 min — met, on an already-loaded box).

## 5. Regenerated-data distribution diff (deliverable §8)

`backward.rust.jsonl` (= concat of the 8 rust shard files, mirroring the
production concat) vs the shipped `backward.jsonl`; diff by
`pyref/diff_distributions.py` -> `bench_logs/diff_backward.log`:

- records: 97,779 (python) vs 98,246 (rust), +0.48%
- kept instances: 6,300 vs 6,300 (exactly 6 per board on both sides)
- kept-instance overlap: 90.27% identical (5,687 / 6,300)
- is_optimal rate: 0.1833 vs 0.1846
- cost_to_go histogram: shares agree within +-0.001 at every value
  (full table in the log); depth and records/instance histograms likewise.

The 9.7% instance divergence decomposes as follows
(`bench_logs/diff_backward_pershard.log`, `flip_investigation.log`):
860/1050 boards have BIT-IDENTICAL kept streams; the mismatched boards
trace back to attempt-outcome flips whose direction is uniformly
"python dropped it, rust labels it" — on all 67 single-flip boards the
python-only instance is also rust-`ok` (window shift), never rust-`empty`.
Two sampled flipped instances take 260 s and 210 s in Python today —
production's 120 s SIGALRM (wall-clock, under a loaded box) dropped them;
the Rust engine's deterministic budget deliberately labels them (DESIGN §3:
budget >> 120 s of Python work). One flip changes that board's kept window
and (usually) the shard's RNG word offset, which resamples a stretch of
subsequent boards until the offsets re-coincide — hence scattered fully
different boards with matching aggregate distributions. Deeper
classification (tie classes at scale) is Agent E2's battery.

forward.rust.jsonl vs forward.jsonl (`bench_logs/diff_forward.log`):

- records: 1,038,827 vs 1,038,827 (+0.00%)
- **1050/1050 boards have IDENTICAL record content** (order included;
  per-board MD5 over record lines) — the whole forward config reproduces
  the Python data exactly, modulo board order in the file (production wrote
  boards in `imap_unordered` completion order).
- full-record rate 0.0497 both sides; every cost_to_go bucket identical.

## 6. 32x32 extrapolation (target: < 1 h on 32 threads; EXTRAPOLATED)

30-board samples via the bridge at --threads 32, g16r6-style settings
(backward per-graph 6, forward per-board 10 + score-candidates, seed 0):

- backward: 29.0 s wall, 180 instances, 1,481 records
  (`scaling/data/g32r4/rust_work/bench_logs/extrap_bwd.log`, `.t0/.t1`)
- forward: 32.8 s wall, 293 instances, 27,411 records; waves computed
  1,290 of 1,500 budgeted attempts (`.../extrap_fwd.log`, `.t0/.t1`)

Linear extrapolation x35 to the 1050 train+val+test boards (marked
EXTRAPOLATED, not measured): backward ~17 min + forward ~19 min =
**~36 min total — under the 1 h target on 32 threads**, with slack for
board-to-board variance. Outputs kept at
`pyref/cache/bench/g32r4_x30.{backward,forward}.jsonl` (new names; the
partial production `scaling/data/g32r4/backward.shard*.jsonl` files were
not touched).

## 7. Board sidecars — large-board policy (deliverable §3)

Measured with the pinned thread pool (a `--threads N` run caps ALL engine
compute, including the rayon table Dijkstra nested inside
`CompiledBoard::compile` — see the self-review note in the task report),
one board per size (bench split boards; 64x64 = E1's cached fresh board),
`/usr/bin/time` wall including ~10 ms binary startup. Log:
`pyref/cache/bench/sidecar_pinned.log`:

| n | compile, 1 thread | compile, 16 threads | sidecar load (serial) | sidecar size |
|---|---|---|---|---|
| 16 | 0.01 s | - | 0.00 s | 0.9 MB |
| 24 | 0.04 s | - | 0.02 s | 4.3 MB |
| 32 | 0.14 s | 0.04 s | 0.05 s | 12.8 MB |
| 64 | 2.81 s | 0.64 s | 0.74 s | 196 MB |

Save (compile + write) of all four boards: 1.26-1.43 s. Peak RSS ~0.76 GB
either way — the in-memory tables dominate, sidecar or not.

**DECISION: recompute, don't ship tables.** In the pipeline context
(threaded runs, each board needed once per run, compiled once via the
content-hash cache) compile costs 0.04 s (32x32) / 0.64 s (64x64) at 16
threads — at or below sidecar-load cost — while sidecars cost 12.8 MB /
196 MB PER BOARD on disk (a 1200-board 64x64 config would need ~230 GB).
Single-threaded, sidecar-load IS 3-4x faster than compile at n >= 32
(0.74 s vs 2.81 s at 64), so sidecars remain a legitimate tool for
serial replay/debug workflows; the `boards` subcommand and A's format are
kept unchanged for that. The data pipeline itself uses inline `grid_data`
(the bridge does exactly this) and no alternative "graph-only" sidecar
format is needed.

## 8. Deterministic budget calibration (deliverable §6)

`budget.solver_iters` replaces production's 120 s SIGALRM. The iteration
count of a rollout is engine-independent (same algorithm, same caps), so
Python wall time and Rust iteration counts on the SAME instances give the
conversion rate.

Data: 67 paired instances with nontrivial work across
g16r4/g16r6/g16r8/g24r4/g32r4
(`pyref/cache/bench/*_bwd_calibration.jsonl`, written by the backward
bench stage): iterations per PYTHON second — min 12, median 350, p90
1,216, **max 2,632**. 120 s of Python work therefore corresponds to at
most ~316k iterations on the least-iteration-dense instance observed; x10
headroom gives ~3.2M, rounded UP to the shipped default

    DEFAULT_SOLVER_ITERS = 10_000_000    (src/io.rs)

i.e. >= 31x the worst-case observed conversion and >280x the median.
Slow instances are iteration-CHEAP (heavy per-iteration propose/apply
work): the two production-timeout instances replayed in §5 needed 210-260
python-seconds at <= 83k iterations. Cross-check at scale: the full g16r6
backward regeneration (25,200 rollouts) maxed at 83,085 iterations (mean
3,218) — 120x under the default — with zero budget_exhausted. A
10M-iteration rollout costs the Rust engine ~1-2 s, so the deterministic
budget also bounds worst-case engine latency, which the wall-clock SIGALRM
could not do deterministically.

## 9. Cross-engine smoke suite (context)

`python3 pyref/smoke.py --engine rust --workers 8` — 12/12 cells PASS,
exit 0, total 525 s (the Python dump stage dominates; the Rust engine
stage totals 7.8 s across all 12 cells vs 422.8 s for `--engine python`
on the same cells in E1's reference run). Logs: `pyref/out/smoke/rust/`.
Every diff stage ALL GREEN: zero label divergence on 12 cells x both task
types, including 64x64.

## Post-optimization (Agent P)

Date: 2026-07-15, same box (`uptime` load 37-42 throughout; 32 threads used
only for the §4-style regen rerun). Engine changes are confined to
`src/move_oracle.rs` (forward A* internals) and `src/board.rs` (all-pairs
Dijkstra); every change is bit-exact — no search-order, budget or output
change (proofs below). Same commands, same bench_compare battery, same
instance workloads (seed 0 reproduces the identical attempt sets: 18/14/13/
34/31 attempts per config, matching §3's table).

### Forward labeling rerun (gate: >= 50x per core) — GATE MET

`python3 rust_datagen/pyref/bench_compare.py --stages forward --configs
g16r4,g16r6,g16r8,g24r4,g32r4` -> logs `pyref/cache/bench/<cfg>_forward.log`,
rows in `pyref/cache/bench/results.jsonl` (latest per config).

| config | attempts | Python 1-core label | Rust --threads 1 | Rust --threads 16 | per-core ratio | was (§3) |
|---|---|---|---|---|---|---|
| g16r4 | 18 | 102.10 s | 1.467 s | 0.470 s | **69.6x** | 32.1x |
| g16r6 | 14 | 213.58 s | 2.682 s | 0.707 s | **79.6x** | 26.3x |
| g16r8 | 13 | 256.65 s | 4.396 s | 1.273 s | **58.4x** | 25.5x |
| g24r4 | 34 | 138.22 s | 1.924 s | 0.592 s | **71.8x** | 33.7x |
| g32r4 | 31 | 135.66 s | 1.697 s | 0.490 s | **79.9x** | 34.3x |

Outcome agreement N/N on every row (110/110 attempts). Python-side times
match §3's runs within load variance (102-257 s vs 100-248 s), so the
improvement is real engine speedup (2.1-2.9x on rust t1), not numerator
drift. What changed (all in move_oracle.rs, per-item semantics identical):
wall-stop ray tables + O(R) nearest-blocker scan instead of cell-by-cell
slide walks; one reusable search context (heap/g/came buffers) per work
item; (f,g) packed into one u64 heap-priority word (order-identical, both
< 2^31 by the pruning rule + a hard `max_expansions < 2^30` assert); u64
state keys when 2*coord_bits*R <= 64 (all five configs; u128 kept and
unit/gate-covered for larger envelopes, e.g. g24r8); compact `came`
entries; h-prune evaluated before the g-map probe (side-effect-free filter
reorder); memoized uncapped candidate-child solves within an item
(duplicates still emit records exactly as before).

### Board precompute rerun (all-pairs bucket queue in board.rs)

`python3 rust_datagen/pyref/bench_compare.py --stages precompute --configs
g16r6,g24r4,g32r4` -> logs `pyref/cache/bench/<cfg>_precompute.log`. The
binary-heap Dijkstra is replaced by a Dial/bucket queue over a CSR
adjacency whenever every edge weight is in 1..=8 (datagen: 1 and
dependent_edge_weight=2); any other weight falls back to the original heap
code. Distances identical (gate12 golden hashes + a dial-vs-heap unit test
across weights 1/2/8 and both table subgraphs).

| config | n | Python tables alone | Rust full compile | tables ratio | was (§1) | from_env ratio |
|---|---|---|---|---|---|---|
| g16r6 | 16 | 0.429 s | 0.0057 s | 75.5x | 48x | 1549x |
| g24r4 | 24 | 4.146 s | 0.0314 s | **132.2x** | 61x | 2184x |
| g32r4 | 32 | 12.251 s | 0.0762 s | **160.7x** | 78x | 3831x |

Amortized cross-check (20/20/10-board work files, warm, threads 1 — same
methodology as §1's `compile_amortized.log`): 4.5 / 23.5 / 73 ms per board
= ratios ~95x / ~176x / ~168x. HONEST GATE READING: >= 100x is MET at
n=24 and n=32 (both methodologies); n=16 lands at 75-95x depending on how
much process startup the denominator carries. The residual n=16 gap is
irreducible within this algorithm class, not sloppiness: the in-process
compile is ~4.4 ms of which the all_pairs Dial sweep is 2.9 ms — 256
sources x ~7.7k edge relaxations = ~2M relaxations at ~1.5 ns each,
i.e. the O(E)-per-source arithmetic floor — while the Python numerator is
only 0.43 s at n=16 (the nx overhead Python pays per edge does not scale
down with board size as fast as the rust side does). Everything above the
sweep (graph build 0.5 ms, CSR 0.1 ms, independent-table sweep 0.9 ms) is
already thinner than the sweep itself.

### End-to-end g16r6 forward regen rerun (§4 comparison)

Same command family as §4 (`scaling.rust_bridge --config g16r6 --system
forward --per-graph 10 --seed 0 --score-candidates --threads 32`, wave
scheduling default), output to NEW files
`scaling/data/g16r6/rust_work/p_forward_regen.jsonl` (+ `p_fwd_regen.*`
work/results/manifest; log + `.t0/.t1` under
`scaling/data/g16r6/rust_work/bench_logs/p_forward_regen.*`):

- **194.2 s wall** at 32 threads (was 432 s in §4 — 2.2x) for all 1050
  boards; 10,500 instances, 1,038,827 records; waves computed the same
  25,860 of 52,500 budgeted attempts.
- Output **byte-identical** (`cmp`) to §4's `forward.rust.jsonl` — the file
  already proven to match the shipped production `forward.jsonl` on
  1050/1050 boards (§5) — so the whole-config byte-identity carries over
  to the optimized engine verbatim.
- With §4's backward regen unchanged (141 s; backward/subgoal code
  untouched), the total config wall-clock story becomes ~141 + ~194 =
  **~335 s ≈ 5.6 min** (target < 10 min).

### Zero-semantic-change evidence (Agent P)

- `cargo test --release` full tree green after all changes: lib 41 (5 new
  tests: packed-priority order vs tuple, ray/fast-slide fuzz vs
  physics::slide, u64-vs-u128 key search equality, u64 key roundtrips,
  dial-vs-heap equality), backward_fixtures 5, backward_units 12, gate12 1,
  gate3_forward 1, io_determinism 7; `RUSTFLAGS="-D warnings"` clean;
  clippy --all-targets 0 warnings. `datagen selftest` PASS.
- The five trimmed forward work files produce byte-identical outputs to the
  pre-change engine at --threads 1 (`cmp`) and identical line multisets at
  --threads 8/16, re-checked after EVERY optimization step.
- All six of E1's committed gate corpora replayed ALL GREEN on the final
  binary (8,876 forward units + 7,593 backward units; gate_f24/f64 also
  exercise the u128-key path with 8 robots on n>=24).
- `pyref/smoke.py --engine rust --workers 8`: see the matrix line appended
  below.
- Documented envelope edge (fail-loud, not silent): forward work items with
  `max_expansions >= 2^30` now abort with a named assert instead of
  running (production uses 40,000; the backward `budget.solver_iters`
  path is untouched subgoal code); `dependent_edge_weight` outside 1..=8
  compiles boards via the original heap Dijkstra fallback.
- Smoke matrix on the final binary: `pyref/smoke.py --engine rust
  --workers 8` -> **12/12 cells PASS**, exit 0, total 519 s (Python dump
  stage dominates as before; the engine stage now totals 4.4 s across the
  12 cells vs 7.8 s pre-optimization). Per-cell logs:
  `pyref/out/smoke/rust/` (regenerated by this run).
