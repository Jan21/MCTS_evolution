# Prompt: port the training-data generation to Rust

You are the lead agent, with five parallel implementation agents at your disposal. Your
job: rewrite the *expensive computational core* of this project's training-data
generation in Rust, achieving ≥50× per-core speedup, while proving label-for-label
agreement with the existing Python implementation. Work in
`/mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet/` (branch
`supervised-full-planner`); put the Rust crate in a new `rust_datagen/` directory. Do
not modify any existing Python file except the thin adapter described in §4.

## 1. What the system does (read this, then read the code)

The domain is Ricochet Robots: an N×N grid with walls; R robots; a moved robot slides
in a straight line until it hits a wall or another robot and stops in the cell before
the obstacle (it cannot stop early). One robot is the target; one cell is the goal;
solve in fewest moves. Two planners are trained, and each needs labeled data:

- **Forward (move-level) labels** — `move_planner/generate.py` + `move_planner/oracle.py`:
  for sampled solvable instances, an exact A* over joint robot positions (admissible
  heuristic `relaxed_target_dist`, expansion caps) produces optimal cost-to-go and
  optimal-move labels for every state on an optimal path, and (with `--score-candidates`)
  the exact cost-to-go of every child of those states.
- **Backward (subgoal-level) labels** — `nn/generate.py` (`rollout`) + `nn/collect_search.py`:
  a subgoal is "park helper robot H on support cell S so the target robot slides and
  stops on bottleneck cell B". An exact best-first search over partial plans
  (`skeleton/astar.py`, `skeleton/heuristics.py::propose/score`,
  `GridEnv.py::propose_subgoal_states`, `partial_plan.py`) is used
  commit-and-solve style: at each decision on an optimal trajectory, EVERY candidate
  subgoal is committed and solved to completion to get its exact cost-to-go; records in
  the `nn/data/combined.jsonl` schema (18 fields — read one line of that file and
  `nn/collect_search.py::_record`).
- **Board precompute** — `nn/gen_grids.py`: boards are generated as wall lists
  (`grid_data`: per-cell strings of N/E/S/W wall sides), compiled to a directed slide
  graph (`build_graph`: independent edges = wall stops; dependent edges = stop against
  a robot parked at a named support cell), plus all-pairs shortest-path tables
  (`all_pairs`, `independent_paths`) where dependent edges cost 2. This precompute is a
  major cost at large N (networkx pure-Python Dijkstra).

Measured Python costs (parallelized, 128-core box): 16×16 config ≈ 40–60 min; 24×24 ≈
3–4 h (board precompute ~74 s/board); 8-robot configs ≈ 6 h+ (solver-timeout burns);
32×32 projected 8–20 h. Targets for the port: 16×16 config < 10 min, 32×32 < 1 h
end-to-end on 32 cores; per-core ≥50× on labeling, ≥100× on all-pairs precompute.

## 2. The architecture decision (already made — follow it)

**Rust is a solving/labeling library behind a JSONL CLI; Python keeps orchestration and
random sampling.** Rationale: instance sampling uses Python's `random.Random` streams
whose reproduction in Rust is a bug farm, and sampling is cheap; the expense is solving.
So:

- Input: a JSONL work file. Each line: either a board-precompute request
  (`{"task":"board","grid_data":[...],"n":16}`) or a labeling request
  (`{"task":"backward_rollout"|"forward_instance", board reference, robot positions,
  target, caps, max_candidates, ...}`). Python writes these; design the exact schema
  yourself and document it.
- Output: JSONL — precomputed board tables (a compact binary sidecar, e.g. bincode +
  a JSON manifest, is fine for the big tables) and labeled records that are
  **field-for-field identical in schema** to the existing `combined.jsonl` /
  `moves.jsonl` records.
- Parallelism: rayon across work items inside one process; the CLI takes `--threads`.
- A thin Python adapter (`scaling/rust_bridge.py`, the ONE new Python file) converts
  env pkls / instances to the work format and shells out. Existing shard scripts then
  gain a `--engine rust` switch (adapter-level, no logic changes).

## 3. Correctness contract (the project has been burned here — this is the heart)

This codebase previously suffered silent geometry/label mismatches that invalidated a
full training run, and three proposal-layer bugs that took 47% off the honest solve
rate. The Rust port MUST implement the **current, fixed semantics** and prove it:

1. **Physics/golden gate.** The slide rule, wall parsing, and slide-graph construction
   must match `simulate.py::slide`/`wall_sets` and `nn/gen_grids.py::build_graph`
   edge-for-edge. Gate: for 200 boards across N∈{16,24,32} and the 128 stock boards,
   Rust graph == Python graph exactly (nodes, edges, edge attributes, weights), and
   10,000 random single slides agree exactly.
2. **Table gate.** `all_pairs` / `independent_paths` equal Python's outputs exactly
   (same weighting: dependent edges cost 2) on 50 boards per size.
3. **Label gate (the important one, designed to be tie-break-proof).** Given the SAME
   decision context (board, robot positions, partial-plan context, candidate list),
   every candidate's `cost_to_go` and `is_optimal` must equal the Python oracle's
   EXACTLY, and forward states' cost-to-go/optimal-move sets likewise. Compare
   per-decision, not per-trajectory: trajectory-level record streams may legitimately
   diverge after the first decision because tie-breaking among equally-optimal
   candidates differs — that is acceptable and out of scope. Build the harness so it
   replays PYTHON-produced decision contexts through Rust and diffs labels. Gate:
   ≥10,000 decisions across sizes/robot counts with ZERO label differences.
4. **Cap-boundary tolerance.** Solvers are budgeted (`max_iters`, `max_frontier`,
   expansion caps). Near the cap, expansion order determines solved-vs-unsolved;
   implementations may disagree there. Measure this divergence rate on capped runs; it
   must be <2% of instances and may only be divergence in *solvability at the cap*,
   never in a returned label's value.
5. **Fixed-bug parity.** The Rust proposal/validation layer must include the three
   repairs (see `skeleton/astar.py` around lines 136, 184, 194 and the comments there):
   a mover may never be its own stopper (compare robots by color); one robot may not
   hold two plan roles; a plan claiming a supported route must actually place the
   stopper. Port the CURRENT file, not any historical version. Golden test: the 12
   instance regression set in `analysis/artifacts/` (defect instances listed in
   `eval/results/residual_failures_postfix.json`) — Rust must refuse the same
   defective plans Python now refuses.
6. Everything size- and robot-count-generic from day one (N and R are runtime values;
   no 16s or 4s in the logic).

### 3b. Smoke-test discipline (standing rule for every agent, every milestone)

Before any long run and after every integration milestone, run the **cross-engine
smoke suite**: with one fixed seed, Python samples ~20 instances at EACH of
16×16, 24×24, 32×32, and 64×64 (mixed robot counts, e.g. 4 and 8); both the Python
oracle and the Rust engine label the SAME instances; the harness diffs every label
per-decision and prints a one-screen pass/fail matrix (size × robots × task-type).
Notes:

- "Same seed" means same sampled instances: sampling stays in Python (§2), so seed
  identity is exact by construction. Rust never needs to reproduce Python's
  random-number generator — it must reproduce Python's LABELS for identical inputs.
- 64×64 exists in no config on purpose: it proves size-genericity beyond anything the
  code has seen. Boards for it are generated on the fly by the smoke harness
  (`nn/gen_grids.py` is size-generic; wall count scales with area, ~768 walls).
  Expect the Python side to be slow at 64×64 (all-pairs on 4096 cells — minutes per
  board); keep it to 2–3 boards / 20 instances, and treat the measured Python-vs-Rust
  wall-clock on this smoke as an early benchmark datapoint.
- The smoke suite is the FIRST thing Agent E builds (before the full battery), and the
  lead runs it at every integration point (A landing, B∥C landing, D landing). A red
  smoke blocks all merges. Keep it under ~15 minutes total by capping instance counts,
  not by skipping sizes.

## 4. Work decomposition for your five parallel agents

Phase 0 (you, the lead): read the files named in §1 end-to-end; write
`rust_datagen/DESIGN.md` fixing the work-item schema, crate layout
(`board`, `physics`, `move_oracle`, `subgoal`, `io`, `bin/datagen`), and the shared
type definitions. Everything below parallelizes only after DESIGN.md exists.

- **Agent A — board + physics.** Wall parsing, slide rule, slide-graph builder,
  all-pairs tables (use a proper binary-heap Dijkstra; this is the 100× win). Owns
  gates 1–2 and the golden-test corpus generator (a Python script that dumps
  reference graphs/tables to JSON for the tests).
- **Agent B — forward oracle.** Joint-state A* with the relaxed-distance heuristic
  (`move_planner/move_bfs.py` + `move_planner/oracle.py`), optimal-path labeling and
  candidate scoring (`move_planner/generate.py` semantics incl. `--score-candidates`).
  Owns the forward half of gate 3.
- **Agent C — subgoal machinery.** Partial-plan DAG, candidate enumeration
  (`GridEnv.propose_subgoal_states` + `skeleton/heuristics.py::propose/score`), the
  exact best-first plan search (`skeleton/astar.py` — CURRENT version with all three
  fixes), commit-and-solve labeling (`nn/generate.py::rollout`, `max_candidates` cap,
  per-instance timeout). The biggest chunk; owns the backward half of gate 3 and
  gate 5.
- **Agent D — I/O, CLI, parallelism, packaging.** Work-item schema implementation,
  JSONL streaming, board sidecar format, rayon orchestration, `--threads`, progress
  reporting to stderr, the Python adapter `scaling/rust_bridge.py`, and the
  `--engine rust` switch in the shard scripts (adapter-level only). Owns benchmarks
  and the performance gates of §1.
- **Agent E — differential verification.** FIRST deliverable: the cross-engine smoke
  suite of §3b (it unblocks everyone else's iteration loop). Then the replay harness
  of gate 3 (Python dumps decision contexts; Rust CLI has a `--replay` mode consuming
  them), the cap-boundary divergence measurement (gate 4), a randomized fuzz loop
  (sample instance in Python → label in both → diff), and CI wiring (`cargo test`
  runs gates 1–3 on the committed golden corpus; a `make verify` target runs the full
  battery; `make smoke` runs §3b). Nothing ships until
  Agent E's full battery is green and its report is written to
  `rust_datagen/VERIFICATION.md` with the exact counts.

Integration order: A → (B ∥ C) → D → E full battery. Agents B and C must consume A's
crate, not reimplement physics. Keep every agent's tests in the crate; the golden
corpus lives in `rust_datagen/golden/` (generate once, commit).

## 5. Constraints inherited from the project (non-negotiable)

- Do not touch: `eval/`, `subgoal_selfplay/`, `move_planner*/` logic, `skeleton/`,
  `GridEnv.py`, `simulate.py`, `nn/` logic — the Python oracle stays the reference
  implementation and keeps working unchanged.
- One GPU rule is irrelevant here (all CPU), but be considerate: the box is shared,
  128 cores, often loaded — default `--threads 16`.
- Plain-language docs; define terms at first use; neutral tone; every claimed number
  traced to a test or benchmark output file.
- Determinism: given the same work file and `--threads 1`, output must be
  byte-reproducible run-to-run. Multi-threaded runs may reorder output lines only.
- If any gate cannot be met, STOP and write up why rather than weakening the gate —
  a documented blocker beats a silently divergent labeler. (History: silent label
  divergence cost this project a full GPU training run.)

## 6. Deliverables checklist

- `rust_datagen/` crate: builds with stable Rust, `cargo test` green, no warnings.
- `rust_datagen/DESIGN.md`, `rust_datagen/VERIFICATION.md` (gate counts, divergence
  rates, benchmark table vs Python on identical work files).
- `scaling/rust_bridge.py` + `--engine rust` switch, demonstrated by regenerating one
  full config's data (suggest `g16r6`) and diffing record-count/label distributions
  against the Python-generated `scaling/data/g16r6/backward.jsonl`.
- A one-page `rust_datagen/README.md`: how to build, how to run one config end-to-end,
  measured speedups.
