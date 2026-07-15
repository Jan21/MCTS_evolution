# pyref measurements (Agent E1)

Machine: shared 128-core box, parallelism capped at 16 workers throughout.
Python 3.11.13 (`/home/p23131/.conda/envs/ph_main/bin/python3`). All wall
times measured on 2026-07-15 with the commands shown; "warm" means the
`pyref/cache/tables/` npz for the boards already exists.

## 1. Python-vs-Python round trip (the E1 acceptance gate)

dump (`dump_decisions.py`) -> replay (`replay_python.py`) -> diff
(`diff_labels.py`), seed 7, caps: backward max_iters=4000
max_frontier=40000 max_candidates=14, forward max_expansions=40000
full_policy_max_ctg=6.

| corpus | boards | instances kept | lines | units compared | diffs |
|---|---|---|---|---|---|
| g16r4 backward | 20 (bench 2400-2419) | 240 | 360 | 3470 candidates | 0 |
| g16r4 forward  | 10 | 50 | 304 | 3682 moves | 0 |
| g24r4 backward | 16 (bench 900-915) | 288 | 412 | 3989 candidates | 0 |
| g24r4 forward  | 10 | 50 | 339 | 4269 moves | 0 |
| n64r4 backward | 3 fresh | 9 | 14 | 134 candidates | 0 |
| n64r4 forward  | 3 fresh | 11 | 68 | 925 moves | 0 |

Extra robustness runs, all ALL-GREEN:

- lazy dump x **eager** replay (g16r4 backward, 360 lines / 3470 candidates);
- **GridEnv.from_env pkl** dump x rebuilt-from-grid_data replay
  (`--env-source pkl`, 92 lines / 900 candidates): the production pkl-loaded
  envs and the grid_data-rebuilt envs produce identical labels — empirically
  de-risking the set-iteration-order concern of DESIGN 5.11 for this corpus.

## 2. Dump / replay rates (workers = as noted)

| corpus | dump rate | dump wall | replay wall |
|---|---|---|---|
| g16r4 backward (240 inst, w8) | 748 lines/min | 29 s | 29 s (w8) |
| g16r4 forward (50 inst, w8) | 1259 lines/min | 15 s | 26 s (w8) |
| g24r4 backward (288 inst, w16) | 414 lines/min | 60 s | 60 s (w16) |
| g24r4 forward (50 inst, w10) | 644 lines/min | 32 s | 18 s (w10) |
| n64r4 backward (9 inst, w3) | 83 lines/min | 10 s | 10 s (w3) |
| n64r4 forward (24 att/board, w3) | 61 lines/min | 68 s | 16 s (w3) |

Backward decision density at 16x16/24x24: ~1.4-1.5 decisions per kept
instance (most segments pin to exact paths and are forced fixes, which is
also why per-line rates look high).

## 3. Lazy-env equality proof (prove_lazy_env.py)

16x16, boards env_2400/2401/2402 (environments/), per board: graph mirror
vs pkl (node+edge lists incl. order and attrs), table values vs
nn.gen_grids reference (2x65536 pairs), final components for all 256 goals
+ all ~730 (bn,sup) keys (set AND iteration order), propose_subgoal_states
for all goals + all pinned keys (ordered), heuristics.propose on 200 random
segments (ordered, incl. parent_support + score), 2000 exact/relaxed probe
triples, and 20 full nn.generate.rollout record streams -- **PROOF OK, zero
mismatches** on all three boards.

24x24 spot proof (env_900 of environments_g24r4, 100 segments, 2000 probes,
8 rollouts): **PROOF OK**.

Env init times from the proof runs:

| size | eager GridEnv init | lazy init |
|---|---|---|
| 16x16 | 7.0-8.1 s | 0.01-0.04 s |
| 24x24 | 51.5 s | 0.03 s |

## 4. 64x64 feasibility (DESIGN section 9 open question) — FEASIBLE with lazy env

Board: fresh nn.gen_grids geometry (RR_GRID=64, 768 interior walls), 4096
nodes, ~129,500 edges, ~12,972 dependent-edge (bn,sup) groups.

| stage | measured |
|---|---|
| gen_walls (board gen) | < 0.1 s |
| build_graph_n | 0.4 s |
| all-pairs tables (2x 4096 sources, 16 workers) | 6.9 s / board (cold); 0.2 s warm npz load (33 MB) |
| lazy env init (copy + wall scan + group scan) | 0.5 s |
| eager `_dependent_edge_cache` (extrapolated: 12,972 keys x 334 ms/key measured over 25 random keys) | **~72 min / board** |
| eager per-goal final components (4096 goals x 0.9 ms) | ~4 s |
| heuristics.propose, cold cache keys | ~410 ms/call (~460 candidates) |
| exact/relaxed table lookup | ~6 us |

So the eager GridEnv build is ~1.2 h **per board** at 64x64 (the DESIGN
section 9 worry, confirmed), while the lazy env brings the whole backward
cell to seconds: the 9-instance backward dump took 10 s wall (3 boards in
parallel), replay 10 s, diff green. **No "python-reference infeasible"
marker is needed at 64x64.**

Caveat measured honestly: at 64x64 most *forward* random instances are not
solvable within max_expansions=40000 (11 kept / 120 attempts; each failed
attempt costs ~1.7 s = the full expansion budget). Backward kept 9/14
attempts. Fresh-board pkls + table npz live under `pyref/cache/` and are
reused across runs.

## 5. Smoke suite stage timings (`smoke.py --engine python`, seed 7)

First full run (cold 32x32/64x64 table caches, per-instance timeout 300 s):

```
smoke matrix -- engine=python, seed=7, total 920s
cell              lines  dump_s engine_s  diff_s  status
--------------------------------------------------------
g16r4.backward       13     1.1      0.8     0.0  PASS
g16r4.forward        75     6.8     14.2     0.0  PASS
g16r8.backward       19   102.0    103.7     0.0  PASS
g16r8.forward        53    20.4     12.5     0.0  PASS
g24r4.backward       10     5.8      4.6     0.0  PASS
g24r4.forward        63    15.1      6.2     0.0  PASS
g24r8.backward       11   220.8    223.7     0.0  PASS
g24r8.forward        39    37.0     20.0     0.0  PASS
g32r4.backward       29    14.9      7.8     0.0  PASS
g32r4.forward       137    35.5     16.8     0.0  PASS
n64r4.backward       14    10.0     10.7     0.0  PASS
n64r4.forward        15    27.0      1.8     0.0  PASS
--------------------------------------------------------
12/12 cells PASS
```

Reading the 920 s total against the <15-min steady-state target: the two
8-robot backward cells dominate (g16r8 206 s, g24r8 445 s across
dump+engine) because with `--engine python` the engine stage REDOES the full
labeling work; with the Rust engine that stage collapses to the dump+diff
side (~8.5 min total dump across all cells, most of it r8 backward). The
shipped default `--instance-timeout 120` (matching production
scaling.backward_label's 120 s SIGALRM; this run used 300 s) additionally
drops the single ~200 s rollout that accounts for most of the g24r8 cost.
The 64x64 cells run in seconds once `pyref/cache/` is warm.

The `--cell-timeout` infeasible path was exercised explicitly (forced 5 s
cap -> cell marked `python-reference infeasible (measured: >5s dump)`, exit
0 as a documented limit, whole process tree killed, no orphan workers).
`--engine stub` fails cleanly at the engine stage only (`ENGINE-FAIL
(rc=3)`, exit 1); `--engine rust` shells to `target/release/datagen replay`
and currently reports `ENGINE-FAIL (rc=2)` against Agent A's placeholder
binary ("datagen: not yet implemented").

## 6. Notes for the Rust side

- Result-line shapes the differ expects: backward
  `{"id", "labels": [{"ctg": int|null, "rejected": bool}, ...]}` (labels in
  candidate order); forward `{"id", "cost_to_go": int|null,
  "optimal_moves": [[slot,dir],...], "legal_moves": [[slot,dir],...]}`.
- Dump lines carry two documented fields beyond the DESIGN section 3
  minimum: `max_candidates` (backward, informational -- the candidate list
  is already truncated) and `full_policy_max_ctg` + `python_labels`
  (forward, consumed by the differ). The Rust work-item parser must
  accept-and-ignore (or echo) these rather than treating them as malformed.
- `datagen replay --work <dump> --out <res> --threads 16` is what
  `smoke.py --engine rust` shells out to.
