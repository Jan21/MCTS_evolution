# rust_datagen — design

Rust port of the expensive computational core of training-data generation for the
Ricochet Robots planners. Rust is a **solving/labeling library behind a JSONL CLI**;
Python keeps orchestration and random sampling. The Python code under
`supervised_valuenet/` stays the reference implementation and is never modified
(exceptions: the new adapter `scaling/rust_bridge.py` and an `--engine rust`
switch in `scaling/gen_data.py`, both adapter-level).

Terms used throughout (defined once):

- **Board**: an N×N grid plus a list of wall segments. Stored as `grid_data`, a
  list of N·N strings; the string at index `y*N + x` names which sides of cell
  `(x, y)` carry a wall (`N`, `E`, `S`, `W` characters, sorted).
- **Slide**: a robot moves in a straight line until the cell ahead is blocked by
  a wall, the board edge, or another robot; it stops in the last free cell. It
  cannot stop mid-ray on its own.
- **Slide graph**: directed graph over cells. For every cell and direction with a
  nonempty slide: one **independent edge** cell→stop (weight 1, a pure wall
  stop), and for every intermediate cell `v` strictly between (excluding the
  stop), one **dependent edge** cell→`v` (weight 2) annotated
  `dependent = v + step` — the cell a helper robot must occupy so the slide
  stops at `v` ("support cell").
- **Forward labels**: per-decision records for the move-level planner
  (`moves.jsonl` schema, produced by `move_planner/generate.py`).
- **Backward labels**: per-decision records for the subgoal-level planner
  (`combined.jsonl` schema, 18 fields, produced by `nn/generate.py::rollout`).

## 1. Crate layout

```
rust_datagen/
  Cargo.toml
  DESIGN.md            (this file)
  VERIFICATION.md      (agent E: gate counts, divergence rates, bench table)
  README.md            (build, run one config end-to-end, measured speedups)
  Makefile             (make smoke / make verify / make bench)
  golden/              committed mini-corpus (small, hermetic; cargo test uses it)
  pyref/               Python reference scripts (dumpers, harnesses, smoke, bench)
  src/
    lib.rs
    types.rs           shared types: Cell, Dir, BoardSpec, WorkItem, records
    physics.rs         wall_sets, slide, successors            (Agent A)
    board.rs           slide-graph builder, all-pairs tables,
                       compiled-board cache, sidecar           (Agent A)
    move_oracle.rs     forward joint-state A*, labeling        (Agent B)
    subgoal/
      mod.rs
      grid_env.rs      final components, extended graph,
                       dependent-edge groups, propose/score    (Agent C)
      plan.rs          PartialPlan DAG                         (Agent C)
      astar.rs         best-first plan search + three fixes    (Agent C)
      rollout.rs       commit-and-solve labeling               (Agent C)
    io.rs              JSONL streaming, work dispatch, sidecar (Agent D)
    bin/datagen.rs     CLI                                     (Agent D)
```

Python reference files (read-only ground truth):
`simulate.py` (slide/wall_sets), `nn/gen_grids.py` (build_graph/all_pairs/
independent_paths), `GridEnv.py`, `partial_plan.py`, `skeleton/astar.py` (CURRENT
version with the three fixes at ~lines 136/184/194), `skeleton/heuristics.py`,
`nn/generate.py`, `nn/collect_search.py`, `move_planner/oracle.py`,
`move_planner/state.py`, `move_planner/generate.py`, `scaling/*`.

## 2. Shared types (types.rs)

- `Cell = (u16, u16)` — `(x, y)` = (column, row). `idx = y*n + x`.
- `Dir` — enum `Up=0, Down=1, Left=2, Right=3`, matching Python
  `DIRECTIONS = ("up","down","left","right")`; `dir_idx` in records uses this
  numbering.
- Robots are slot-indexed internally (slot i = i-th color of the canonical
  palette `["Red","Blue","Green","Yellow","Purple","Orange","Cyan","Magenta"]`
  truncated to R). Color strings appear only at the I/O boundary. Robot
  **identity is the color/slot**, never the position.
- Costs and labels are `i64`; "unreachable" is `Option::None` (JSON `null`).
  Subgoal-scoring infinity is `10_000` (Python `INF`), forward-oracle infinity
  `1<<30`. Keep both constants verbatim.

## 3. Work-item schema (input JSONL, one object per line)

Every item has `"task"` and `"id"` (an opaque string the engine echoes back so
the bridge can reassociate results; multi-threaded output may reorder lines).

Board description, either inline or by sidecar reference:

```json
"board": {"env_id": 1000, "n": 16, "grid_data": ["ES","", ...]}
"board": {"env_id": 1000, "sidecar": "boards/env_1000.bin"}
```

The engine compiles boards on first use and caches by (env_id, SHA-256 of
grid_data, dependent_edge_weight) within a run.

### task: "board" — precompute + optional table export

```json
{"task":"board", "id":"b0", "board":{...inline...},
 "dependent_edge_weight": 2,
 "sidecar_out": "boards/env_1000.bin",     // optional: write bincode sidecar
 "emit": "none" | "graph" | "tables"}       // export for gates 1-2
```

Result line (`emit:"graph"` adds `edges`; `emit:"tables"` adds table hashes and,
with `"full_tables": true`, the raw tables):

```json
{"id":"b0", "task":"board", "env_id":1000, "n":16,
 "edges": [[ux,uy,vx,vy,weight, sx,sy|null], ...],   // sorted (u,v) lexicographic
 "all_pairs_sha256": "...", "independent_sha256": "...",
 "n_nodes": 256, "n_edges": 4321}
```

Table hash = SHA-256 over the canonical text `"sx,sy,tx,ty,D\n"` for every pair
in lexicographic (s, t) order, where `D` is the integer distance or `None` —
computed identically by the Python dumper (`pyref/dump_reference.py`) and Rust.

### task: "forward_instance"

```json
{"task":"forward_instance", "id":"f17", "board":{...},
 "robots":[[x,y],...], "target_idx":1, "target":[x,y],
 "max_expansions":40000, "full_policy":true, "full_policy_max_ctg":6,
 "score_candidates":true}
```

Result: `{"id":"f17","status":"solved","d_star":9,"records":[...]}` where
`records` are **exactly** the `moves.jsonl` objects (`env_id, robots, target,
target_idx, cost_to_go, best_moves, legal_moves, depth, full`), or
`{"id":"f17","status":"relaxed_unreachable"|"unsolved"}` (unsolved = cap hit /
no path). Bridge keeps the first `per_board` solved attempts in attempt order —
reproducing the Python driver loop exactly (see §6).

### task: "backward_rollout"

```json
{"task":"backward_rollout", "id":"r3", "board":{...},
 "target":[x,y], "target_robot":[[x,y],"Red"],
 "helpers":[[[x,y],"Blue"], ...],
 "max_candidates":14, "max_iters":4000, "max_frontier":40000,
 "dependent_edge_weight":2,
 "budget": {"solver_iters": 40000000}}
```

Result: `{"id":"r3","status":"ok","records":[...]}` with records exactly in the
18-field `combined.jsonl` schema, or `status:"empty"` (rollout produced no
records — Python's "drop the instance") or `status:"budget_exhausted"`.
`budget.solver_iters` is a **deterministic** replacement for Python's 120 s
SIGALRM: total plan-search iterations spent inside one rollout across all
commit-and-solve calls. Wall-clock timeouts would break byte-reproducibility
(§8); the default budget is calibrated by Agent D so that it corresponds to
well past 120 s of Python work (measured, documented in VERIFICATION.md).
An optional `"timeout_s"` field exists for parity experiments only and is
documented as non-deterministic.

### task: "replay_backward_decision" (gate 3, produced only by pyref dumper)

One decision = one partial plan + the candidate list Python enumerated:

```json
{"task":"replay_backward_decision", "id":"d811", "board":{...},
 "dependent_edge_weight":2, "max_iters":4000, "max_frontier":40000,
 "state": {"target":[x,y], "target_robot":[[x,y],"Red"], "helpers":[...]},
 "plan": {"nodes":[["goal","goal",{"pos":[x,y]}],
                   ["bn_1","bottleneck",{"pos":[x,y],"robot":[[x,y],"Red"]}],
                   ["sp_1","support",{"pos":[x,y],"robot":[[x,y],"Blue"]}],
                   ["sg_1","subgoal",{"parent_support_pos":[x,y]|null}],
                   ["leaf_0","leaf",{"pos":[x,y],"robot":[[x,y],"Red"]}], ...],
          "edges":[["goal","sg_1","fixed",3], ["sg_1","bn_1","fixed",0], ...],
          "nc": 2},
 "open_edge": ["sg_1","leaf_0"],
 "candidates": [{"bottleneck":[x,y], "support":[x,y],
                 "helper":[[x,y],"Blue"], "parent_support":[x,y]|null}, ...],
 "python_labels": [{"ctg": 7, "rejected": false}, ...]}
```

Nodes and edges are listed in Python insertion order; Rust reconstructs the
plan preserving that order (it determines `open_edges()[0]` and iteration
order downstream). Result per candidate: `{"ctg": int|null, "rejected": bool}`
(`rejected` = the `_apply` invariants refused it; `ctg:null` = `solve_plan`
found no completion within caps). The harness diffs Rust vs `python_labels`
and recomputes `is_optimal` from each side's ctg vector.

### task: "replay_forward_state" (gate 3)

```json
{"task":"replay_forward_state", "id":"s12", "board":{...},
 "positions":[[x,y],...], "target_idx":0, "target":[x,y],
 "max_expansions":40000}
```

Result: `{"cost_to_go": int|null, "optimal_moves": [[slot,dir],...],
"legal_moves": [[slot,dir],...]}` where `optimal_moves` is the FULL optimal set
(each legal child solved with `cost_cap = ctg-1`). Diff rules against a Python
record: `cost_to_go` equal; `legal_moves` equal as sets; if the record has
`full=true`, `best_moves` equal as sets; if `full=false`, Python's single
`best_moves` entry must be a member of Rust's `optimal_moves` (the taken move on
Python's optimal path is tie-break-dependent; membership is the tie-break-proof
check).

## 4. Output records (must be field-for-field identical in schema)

Backward (18 fields, order as `nn/generate.py` writes them):
`env_id, target, target_robot, helpers, seg_start, seg_end, seg_support,
mover_color, ctx_bottlenecks, ctx_supports, ctx_open_endpoints,
cand_bottleneck, cand_support, cand_helper, cand_parent_support, cost_to_go,
is_optimal, depth`.

Forward (9 fields): `env_id, robots, target, target_idx, cost_to_go,
best_moves, legal_moves, depth, full`.

JSON details: integers stay integers, booleans booleans, `null` for Python
`None`. Rust emits compact JSON (no spaces); Python emits `", "` separators.
Consumers `json.loads` each line, so this is a non-difference; comparisons are
always on parsed values, never raw bytes (except Rust-vs-Rust determinism, §8).

## 5. Semantics to port exactly (the landmine list)

Each item names the Python ground truth. **When in doubt, do what the Python
line does, not what seems mathematically equivalent.**

1. **Walls** (`simulate.wall_sets`): `E` at (x,y) → walls_right (x,y); `S` →
   walls_down (x,y); `W` mirrors to (x-1,y) only when x>0; `N` mirrors to
   (x,y-1) only when y>0. Border rows/cols carry explicit N/S/E/W chars in
   `grid_data`; the slide rule ALSO clamps at coordinates 0 / n-1 regardless.
2. **Slide** (`simulate.slide`): loop cell by cell; stop when the next cell is
   across a wall/edge or occupied by a blocker. Returns start cell if it cannot
   move at all (callers treat that as a no-op / illegal move).
3. **Graph build** (`nn/gen_grids.build_graph`): iterate cells y-major then x,
   directions in dict order up,down,left,right; skip zero slides; independent
   edge (c→stop, weight 1); dependent edges (c→v, weight 2,
   dependent = v+step) for all intermediate v EXCLUDING the stop. No (c,v)
   pair collides across directions, so a plain map<(u,v),EdgeAttr> is faithful.
   Preserve **edge insertion order** per node (Vec-based adjacency): NetworkX
   iteration order = insertion order, and it leaks into candidate enumeration
   order downstream.
4. **Tables**: `all_pairs` = Dijkstra from every node over `weight` attr where
   dependent edges are first re-weighted to `dependent_edge_weight` (default 2
   — `GridEnv.from_env` recomputes this with the configured weight);
   `independent_paths` = Dijkstra over the subgraph WITHOUT dependent edges
   (weights all 1). Both tables contain an entry for every ordered pair,
   `null` when unreachable. Distances are unique (no tie-break exposure).
5. **`_wall_nodes`** (`GridEnv.__init__`): nodes with ANY incoming edge of
   weight == 1 **after** re-weighting. Do not shortcut to "independent-edge
   heads" — if `dependent_edge_weight == 1` the Python set genuinely includes
   dependent heads. Weight is part of the compiled board key.
6. **Final components** (`GridEnv._compute_final_component`):
   `max_final_component_distance` is always `None` in datagen (assert this in
   the work item; reject otherwise) → final component = ancestors of goal on
   the dependent-edge-free graph. With `blocker_pos`: remove the blocker node,
   then if blocker shares the goal's row, also remove cells strictly past the
   blocker on that row (side away from goal); same for column. Then ancestors.
7. **Extended graph** (`GridEnv.get_extended_graph(support, bottleneck)`):
   only in-edges of `bottleneck` with `dependent == support` are touched.
   With `main_vec = bottleneck - support`: an edge whose direction component
   along the main axis is 0 or has the SAME sign as main_vec is removed; the
   OPPOSITE sign converts to independent (drop `dependent`, weight := 1).
8. **Dependent-edge groups** (`GridEnv._dependent_edge_cache`): key
   (edge head v, support cell), value = list of edges (u,v) in insertion
   order + the extended-graph final component. `compute_exact/relaxed(start,
   end, support)`: if (end, support) has no group → `None`; else
   min over group edges u of table[start,u] + 1 (skipping `null`s); empty →
   `None`. Exact uses the independent table, relaxed uses the all-pairs table.
9. **Bottleneck/support pair collection**
   (`_collect_bottleneck_support_pairs(fc)`): for nodes in fc, in-edges (u,v)
   with u NOT in fc and `dependent` set and the support cell in `_wall_nodes`
   → pair (v, support). Called with `fc | {goal}` (goal-keyed) or
   `fc | {bottleneck}` (pair-keyed). The `(goal, None)` cache entry exists
   only when goal has an independent in-edge, but a cache miss recomputes the
   same thing — semantics identical either way.
10. **subgoal_score**: independent[bn→goal] + relaxed(target→bn via support)
    + relaxed(helper→support), each `None` → 10_000. Sum of three i64s.
11. **propose** (`skeleton/heuristics.propose`): raw = propose_subgoal_states;
    if raw empty, retry once pinned to each dependent support of `goal` × each
    helper. Then parent_support resolution per candidate, in order: pinned
    segment support (if any), `None`, then the goal's dependent supports;
    FIRST one with a non-None exact(bn→goal via ps) wins.
    **Known divergence risk**: the dependent supports come from a Python
    `set`, whose iteration order Rust does not reproduce. When ≥2 supports
    qualify (only reachable when both the pinned support and the independent
    route fail) the chosen `parent_support` — and hence `parent_cost` — can
    differ. Rust uses a documented deterministic order (graph edge insertion
    order, deduplicated). The fuzz battery measures how often this changes a
    LABEL (expected ≈ never); a nonzero rate is reported in VERIFICATION.md
    and escalated rather than papered over. Replay-mode items carry
    `parent_support` explicitly, so gate 3 is immune by construction.
12. **The three fixes** (`skeleton/astar.py`, port the CURRENT file):
    a. `_segment` (~line 136): helpers = state.helpers minus the mover **by
       color**; if the mover is not the target robot, append the target robot
       at the END of the helper list (order matters for enumeration).
    b. `_apply` (~line 184): reject a candidate whose helper color already
       owns a `leaf` node anywhere in the plan (one robot, one leaf identity).
    c. `_apply` (~line 194): a candidate whose `parent_support` is a real cell
       may only be applied when that cell equals the candidate's own support
       cell or an existing `support` node's cell — otherwise the plan would
       claim a supported route with no robot ever placed on the stopper.
    Then `parent_cost = exact(bn→seg.end via parent_support)`, reject if
    `None`. New node ids use the plan's `nc` counter exactly as Python does
    (`sg_{n}`, `bn_{n}`, `sp_{n}`, `leaf_{n}`).
13. **Plan A*** (`skeleton/astar.py::_search`): frontier ordered by
    (plan cost, insertion counter); pop; complete → done (admissible ⇒
    optimal over the proposal space); `len(open_edges) > max_open` (default
    `2*(len(helpers)+2)`) → discard; expand FIRST open edge (edge insertion
    order); children sorted by `score` with a **stable** sort before beam
    truncation (beam unused in datagen, but port it); push all. Terminate
    when `it > max_iters` or frontier empty or `len(frontier) > max_frontier`
    (checked at loop top BEFORE popping). `solve_plan` returns the completed
    plan or None. Iteration budget semantics must match exactly — these caps
    are gate-4 territory.
14. **Initial plan** (`_initial_plan`): goal + leaf_0; edge fixed at exact
    cost when `exact is not None and (relaxed is None or exact <= relaxed)`,
    else open with relaxed (or 10_000 when relaxed is None). `nc = 1`.
15. **Rollout / commit-and-solve** (`nn/generate.rollout`): loop until plan
    complete; take FIRST open edge; if exact path exists → fix it via
    `_expand` (single child) and continue (no record); else enumerate
    candidates, `sorted(key=score)[:max_candidates]` (stable sort; Python
    applies this ONLY when max_candidates is not None — port that literally:
    unsorted full list otherwise); for each candidate `_apply` (rejects → skip
    silently), `solve_plan` (None → skip); `ctg = done.cost() - fixed_g`
    where `fixed_g` = sum of FIXED edge costs of the plan BEFORE this
    decision; record every labeled candidate with `is_optimal = ctg ==
    min(ctg)`; advance to the FIRST labeled candidate (in candidate order)
    whose ctg == best. Context fields (`ctx_*`, `open_endpoints`) are
    extracted in node/edge insertion order. `depth` increments only on
    decisions (not forced fixes).
16. **Forward oracle** (`move_planner/oracle.py`): mirror EXACTLY —
    `relaxed_target_dist` reverse BFS (walk rays outward from each popped
    cell; first-seen wins, FIFO queue); A* priority tuple `(f, g, counter)`
    with the counter incremented per push (so among equal (f,g), FIFO by push
    order); stale-entry skip via `g != g_map[cur]`; goal test on POP;
    `expansions` incremented after the goal test, abort when `expansions >
    max_expansions`; child pruning when `h >= INF` or `f_child > cost_cap`;
    successor order = robot slot-major, then direction order. Wall-adjacent
    detail: `h0 > cost_cap` or `h0 >= INF` → unsolvable before any search;
    `positions[target_idx] == target` → 0 immediately. With this mirroring,
    capped behaviour is bit-identical too, so gate 4 divergence for forward
    should measure 0.
17. **label_trajectory**: records only for on-path states BEFORE the goal
    state; `here = d_star - depth`; `full = full_policy and (cap is None or
    here <= cap)`; full set via `is_optimal_child` = child solve with
    `cost_cap = here-1` compared `== here-1` (goal child: `here == 1`);
    non-full → `best = [taken]`. `score_candidates` (generate.py): for every
    legal move from a recorded state, child record with exact ctg (goal child
    → 0; unsolvable-within-cap child → skipped), `best_moves = []`,
    `legal_moves` of the CHILD state, `depth+1`, `full=false`.
18. **Instance-driver semantics** (bridge, §6): forward per-board RNG
    `random.Random(seed*100003 + env_id ^ 0x9E3779B9)` — careful: Python
    computes `_board_seed(env_id, seed) ^ 0x9E3779B9` = `(seed*100003 +
    env_id) ^ 0x9E3779B9`; sample = `rng.sample(cells_all, R+1)` with
    cells_all in y-major order + `rng.randrange(R)`; attempts cap
    `per_board*5`, relaxed-unreachable consumes an attempt without a solve.
    Backward: ONE shared `random.Random(seed)` across all boards, in board
    order; each attempt consumes one `rng.sample(sorted-nodes? NO —
    `list(env.G.nodes())` in insertion order, y-major)` of R+1 cells;
    attempts cap `per_graph*4`.

## 6. The bridge (`scaling/rust_bridge.py`) and `--engine rust`

The bridge converts env pkls + Python-sampled instances into work items,
shells out to the Rust CLI, and reassembles outputs in Python order.

- **Sampling stays in Python** and is *exact-stream*: for the backward
  labeler's shared RNG, the bridge snapshots `rng.getstate()` per board,
  speculatively draws the full attempt budget, sends all attempts to Rust in
  one batch, decides which attempts Python-with-identical-outcomes would have
  consumed (first `per_graph` successes), then `rng.setstate(snapshot)` and
  re-draws exactly that many attempts before moving to the next board. When
  Rust and Python agree on per-instance success (everything except gate-4
  cap-boundary cases), the kept instance stream is IDENTICAL to Python's.
- Forward sampling is per-board-seeded (no shared stream): presample the
  budget, keep first `per_board` solved, in attempt order.
- `scaling/gen_data.py --engine rust` routes to the bridge with the same
  arguments; `--engine python` (default) is byte-for-byte today's behaviour.
- The bridge writes work files and raw result files under
  `scaling/data/<config>/rust_work/` for auditability.

## 7. CLI (bin/datagen)

```
datagen run    --work items.jsonl --out results.jsonl --threads 16
datagen replay --work decisions.jsonl --out replayed.jsonl --threads 16
datagen boards --work boards.jsonl --out manifests.jsonl --sidecar-dir DIR
datagen selftest                     # runs the committed golden corpus
```

`--threads` defaults to 16 (shared 128-core box; be considerate). rayon pool
sized explicitly; progress lines to stderr every ~5 s (`done/total, rate,
ETA`). Exit nonzero on any malformed work item (fail loud, no silent skips) —
malformed means unparseable/unknown fields; solver failures are result rows.

## 8. Determinism rules

- `--threads 1` → byte-reproducible output for identical work files. Enforced
  by: no wall-clock in any decision path (deterministic budgets), no
  HashMap/HashSet iteration anywhere order can leak into output — use
  Vec/IndexMap/IndexSet or sort explicitly; the compiled board cache keyed by
  content hash; no RNG anywhere in the engine.
- Multi-threaded runs may reorder OUTPUT LINES only (result objects are
  bit-identical per item; the bridge reorders by `id`).
- `cargo test` must not depend on the network or absolute paths.

## 9. Gates → tests (who owns what)

| Gate | Content | Owner | Where |
|------|---------|-------|-------|
| 1 | graph edge-for-edge + 10k slides, 200 boards @ {16,24,32} + 128 stock | A | `cargo test` mini-corpus in `golden/`; full sweep via `make verify` (pyref dumps → temp) |
| 2 | all_pairs + independent tables equal, 50 boards/size | A | canonical-hash compare (§3 board task) |
| 3 | ≥10k decisions replayed, ZERO label diffs (backward per-candidate ctg/rejection; forward value+optimal-set rules of §3) | E (B/C consume) | `pyref/dump_decisions.py` + `datagen replay` + differ |
| 4 | cap-boundary solvability divergence <2% of instances, never a returned label value | E | capped fuzz runs, both engines, classified diff |
| 5 | three fixes + 12-instance regression corpus refuse the same plans | C + E | unit tests per fix; `pyref/dump_regression.py` from `analysis/artifacts/` + `eval/results/residual_failures_postfix.json` |
| perf | per-core ≥50× labeling, ≥100× all-pairs; 16×16 config <10 min, 32×32 <1 h on 32 cores | D | `make bench`, identical work files |

Smoke suite (§3b of the prompt): `make smoke` — fixed seed, ~20 instances per
size at 16/24/32 + 2-3 boards at 64×64, robots 4 and 8, both task types,
per-decision diff, one-screen matrix, <15 min steady-state (64×64 Python
reference boards/caches are built once and cached under `pyref/cache/`,
outside the 15-minute budget). Known risk, to be measured by Agent E: the
eager `GridEnv` cache build is likely infeasible in Python at 64×64; the
harness may construct a lazily-cached GridEnv **subclass inside pyref**
(identical semantics, computed-on-demand) — never by modifying GridEnv.py. If
even that is impractical, the 64×64 backward cell reports Rust-side
invariants only and VERIFICATION.md documents the blocker honestly.

## 10. Integration order

Phase 0 (lead): this file + crate skeleton. Then:

- **Wave 1**: Agent A (physics/board/tables, gates 1-2) ∥ Agent E1 (pyref
  harness scaffolding: dumpers, differ, smoke driver — engine-agnostic,
  validated Python-vs-Python).
- Lead runs A's gates + smoke-with-stub. Red blocks everything.
- **Wave 2**: Agent B (move_oracle) ∥ Agent C (subgoal). Consume A's modules;
  neither touches the other's files; both wire replay entry points to io.rs
  stubs.
- Lead runs smoke at 16×16.
- **Wave 3**: Agent D (CLI, rayon, sidecars, bridge, --engine rust, bench).
- **Wave 4**: Agent E2 full battery + VERIFICATION.md; lead: README, final
  review, deliverable table.

Nothing ships until the full battery is green and VERIFICATION.md carries the
exact counts. If a gate cannot be met, STOP and write up why — a documented
blocker beats a silently divergent labeler.
