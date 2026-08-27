# Part 1 — The setups, the metrics, and the solvers that use no learning

This is the first of three layers of a reference document about the Ricochet Robots
planning study in this repository. Part 1 answers three questions, in order:

1. What is one puzzle, and where do the puzzles come from?
2. Which configurations do we test, and what do we measure on them?
3. How well do the hand-written, non-learned solvers score, how long do they take,
   and where can they not run at all?

Part 2 covers the two learned planner families (move-by-move and subgoals) and the
plan-language extensions. Part 3 covers self-play.

**Reading rules for this part.** Every number below was read out of a file during the
session that produced this document. The final appendix lists each number, its value,
the exact file path, and how it was read. Where a number does not exist, the text says
"not measured" instead of estimating one. Where two files disagree, both are reported
and named.

**Repository root.** `/scratch/project/open-37-42/petrhyner/MCTS_evolution`.
Two source trees matter here:

- `supervised_valuenet/` — the original study. Board generator, exact solvers,
  benchmarks, metrics, comparison harness. Abbreviated **SV** in file paths below.
- `self_play_robots/` — the successor project. Abbreviated **SPR**.

---

## A. The game and the instance format

### A.1 The puzzle

Ricochet Robots is a single-player puzzle on a square grid.

- The grid is `n` by `n` cells. The base game uses `n = 16`.
- Walls sit on cell edges. The four outer edges are always walls. A number of interior
  wall segments is placed at random. A wall segment always appears on both sides at
  once: a wall on the east edge of cell `(x, y)` is the same wall as the one on the
  west edge of cell `(x+1, y)`.
- Several robots stand on distinct cells. Each robot has a colour. The base game uses
  4 robots.
- One robot is the **target robot**. One empty cell is the **goal cell**.

A **move** is a pair `(robot, direction)` with direction in `{up, down, left, right}`.
Applying a move slides that one robot in that direction. The robot does not stop where
you want. It slides until one of three things stops it:

1. a wall on the far edge of the current cell,
2. the outer edge of the board,
3. another robot in the next cell.

The robot stops in the last free cell before the obstacle. A move that would not change
the robot's position is not a legal move.

The puzzle is solved when the target robot stands on the goal cell. The quality measure
is the number of moves. Fewer moves is better.

The interesting consequence of the sliding rule is that robots are used as bumpers. To
stop the target robot on the goal cell, some other robot often has to be parked on the
cell just past the goal first. Solutions therefore contain helper moves that have
nothing to do with the target robot's own path.

**Where this is implemented.** The slide physics is
`SV/simulate.py::slide` (derived directly from the wall layout, no precomputation).
The joint-state game is `SV/move_planner/state.py`: a state is the tuple of all robot
positions in a fixed colour order, an action is `(robot_slot, direction)`, and
`apply_move` slides that robot with all other robots as blockers. `state.py` calls this
"the true multi-robot physics, not the blocker-clearing relaxation used by the subgoal
solver" — that distinction returns in section C.2.

### A.2 What one instance is

The study separates two things that are often merged.

- A **board** is the geometry alone: the grid size and the wall layout. It is stored as
  a Python pickle file `env_<id>.pkl` in a per-configuration directory. Boards have
  integer ids. The id is the only handle used anywhere.
- An **instance** (also called a puzzle) is a board id plus a starting placement: the
  positions of all robots, which robot is the target, and the goal cell.

Instances are stored one per line in JSON Lines files. The schema has five fields and
is the same in every benchmark file in the repository:

| field | type | meaning |
|---|---|---|
| `env_id` | int | the board id, resolved against the configuration's board directory |
| `positions` | list of `[x, y]` | robot cells, in the canonical colour order |
| `target_idx` | int | index into `positions` of the robot that must reach the goal |
| `target` | `[x, y]` | the goal cell |
| `d_star` | int, `null`, or `0` | the exact move optimum, when known — see section C.3 |

A real line, the first of the 16x16 base benchmark:

```
{"d_star":12,"env_id":2400,"positions":[[1,15],[2,12],[11,6],[0,3]],"target":[9,15],"target_idx":0}
```

`d_star` is written by an exact solver at benchmark construction time. It is never
available to any planner at solve time. That rule is stated in the metadata sidecar of
every benchmark file: "reference only, never used at inference".

### A.3 Where boards come from

Boards are generated, not shipped, except for the 128 boards of the original 16x16 set.

The generator is `SV/nn/gen_grids.py::make_board`. Given a board id and a seeded random
number generator it does four things, in this order:

1. `gen_walls(rng)` places the border walls, then `RR_WALLS` interior segments at
   random. Each attempt draws a cell and a side. An attempt that names an off-board
   direction or an already-placed side is discarded without further draws.
2. `build_graph(grid_data)` builds the slide graph from the walls. An edge with weight 1
   is a plain slide that a wall stops. An edge with weight 2 is a slide that needs a
   helper robot parked at a named cell — the graph records that cell as the edge's
   `dependent` attribute.
3. `random_instance(rng)` draws `robots + 1` distinct cells: the robot start cells and
   the goal cell. The first robot is the target robot.
4. `independent_paths` and `all_pairs` precompute two all-pairs shortest-path tables
   over the slide graph, one ignoring helper-dependent edges and one using them.

Wall density is not a free parameter. It is pinned to the density of the base game and
scaled by area: `walls(n) = round(48 * (n/16)^2)`. The base 16x16 board has 48 interior
segments, a 24x24 board has 108, a 32x32 board has 192, a 64x64 board has 768.

Board generation is deterministic per id. `SV/scaling/gen_boards.py` seeds
`random.Random(idx)` per board id, so any shard or restart reproduces the same bytes,
and it never overwrites an existing pickle.

The number of robots and the grid size are process-level settings, read from the
environment variables `RR_GRID`, `RR_ROBOTS`, `RR_WALLS` and `RR_ENV_DIR` at module
import time. One process therefore serves exactly one configuration.

### A.4 Lean boards, and why they exist

Steps 1 to 3 of `make_board` take milliseconds. Step 4 does not. The two all-pairs
tables each hold `(n^2)^2` entries and are built with all-pairs Dijkstra, which costs
about `O(n^4 log n)`. A second table is then rebuilt at load time by
`GridEnv.from_env`.

The measured cost of that step (FINDINGS item 60a, measured on login CPU):

| grid | cold `GridEnv.from_env` | stored pickle |
|---|---|---|
| 16x16 | 8.2 s | 1.6 MB |
| 24x24 | 60 s | 8.8 MB |
| 32x32 | 222 s | 28.6 MB |

Extrapolated by a log-log fit `t ≈ 1.5e-5 * n^4.77` on those three points, a 64x64 board
costs about 1.7 h and a 96x96 board about 11.9 h. A full 1200-board 64x64 set would need
about 230 GB of sidecar tables. That is the size wall.

**Lean boards** remove it. `SV/nn_labeler/leanboard.py` produces a board dictionary with
`grid_data`, `grid_graph` and `instances` and **no distance tables**. It is safe because
of one structural fact, stated in FINDINGS item 62: `make_board`'s two table builders
consume no randomness. The random stream is `gen_walls -> build_graph ->
random_instance` only. A lean board is therefore layout-identical to a standard board by
construction, for the same `(seed, env_id, RR_GRID, RR_WALLS, RR_ROBOTS)`.

The tables are replaced by `LazyDistTable`, a read-only stand-in that answers a distance
query by computing one full single-source row with Dial's algorithm and caching it under
a least-recently-used cap. Rows are never truncated, because a cutoff would turn "far"
into "no path".

Measured lean cost per board, construction plus environment build (FINDINGS item 62c):

| grid | lean cost | eager cost | lean pickle | eager pickle |
|---|---|---|---|---|
| 32x32 | 0.24 s | 222 s (measured) | 1.1 MB | 28.6 MB |
| 64x64 | 1.5 s | ~1.7 h (projected) | 4.9 MB | ~516 MB (projected) |
| 96x96 | 3.8 s | ~11.9 h (projected) | 11.7 MB | ~2.8 GB (projected) |

The honest overhead is that a lazy query costs 1.43 times a dictionary lookup, about
1.14 microseconds more per query, with bit-identical results. Parity is checked
exhaustively by `SV/nn_labeler/test_leanboard_parity.py` over 1.45 M table entries at
16x16 and 24x24 with zero mismatches.

Two consequences matter for later parts. First, the exact Rust solver needs no port to
use lean boards, because `SV/scaling/rust_bridge.py` reads `grid_data` and `instances`
straight out of the pickle. Second, the configurations at 40x40 and above exist only as
lean boards.

---

## B. Every setup we test

### B.1 The naming scheme

A configuration is named `g<grid>r<robots>`. `g24r4` is a 24x24 board with 4 robots.
Wall count is implied by the grid size through the density rule in A.3, so it is not in
the name.

The registry is `SV/scaling/configs.py`. It holds 36 configurations. Every one of them
fixes five things: grid size, robot count, wall count, the board directory, and the
board-id ranges used for training, validation, testing and benchmarking.

### B.2 The board-id splits

There are two split conventions.

**The legacy convention (`g16r4` only).** This configuration inherits the board ids of
the original 16x16 pipeline. Its boards pre-exist and are never regenerated.

| split | board ids |
|---|---|
| train | 0-95, 1000-1799 |
| val | 1800-2399 |
| test | 112-127, 2400-2999 |
| bench | 2400-2549 |

**The standard convention (every other configuration).** 1200 boards per configuration.

| split | board ids |
|---|---|
| train | 0-699 |
| val | 700-899 |
| test | 900-1049 |
| bench | 900-1049 |
| spare | 1050-1199 |

Note that test and bench are the same 150 boards. Note also that the two conventions use
disjoint id blocks for their benchmarks: 2400-2549 for `g16r4`, 900-1049 for everything
else. They must never be described as one range.

A separate rule governs the self-play project. `SPR/variants/__init__.py` states it:
generation uses board ids 40000-49999, examination uses ids 20000-20999, and "Nothing
may ever train on 0-1199 (pinned) or 20000+ (exam)".

### B.3 The full configuration table, with what is on disk

Board counts below are counts of `env_*.pkl` files actually present. Data files are the
label and benchmark files present under `SV/scaling/data/<cfg>/`, with shard files and
partial files excluded.

| cfg | grid | robots | walls | boards on disk | label / benchmark files present |
|---|---|---|---|---|---|
| g8r4 | 8 | 4 | 12 | 1200 | `backward.rust` |
| g9r4 | 9 | 4 | 15 | 1200 | `backward.rust` |
| g10r4 | 10 | 4 | 19 | 1200 | `backward.rust` |
| g11r4 | 11 | 4 | 23 | 1200 | `backward.rust` |
| g12r4 | 12 | 4 | 27 | 1200 | `backward.rust` |
| g13r4 | 13 | 4 | 32 | 1200 | `backward.rust` |
| g14r4 | 14 | 4 | 37 | 1200 | `backward.rust` |
| g15r4 | 15 | 4 | 42 | 1200 | `backward.rust` |
| **g16r4** | 16 | 4 | 48 | 2139 | `backward_b2.rust`, `backward_b2.cap20000.rust`. Benchmark lives elsewhere: `SV/eval/data/bench450.jsonl` |
| **g16r6** | 16 | 6 | 48 | 1200 | `backward`, `backward.rust`, `backward_b2.rust`, `backward_b2.cap20000.rust`, `forward`, `forward.rust`, `bench`, `bench.solved`, `bench.unsolved` |
| **g16r8** | 16 | 8 | 48 | 1200 | `backward`, `backward_b2.rust`, `backward_b2.cap20000.rust`, `forward`, `bench`, `bench.solved`, `bench.unsolved`, `cap_probe_10x_results` |
| g17r4 … g23r4 | 17-23 | 4 | 54-99 | 150 each | `backward_audit.rust` only |
| **g24r4** | 24 | 4 | 108 | 1200 | `backward`, `forward`, `bench`, `bench.solved`, `bench.unsolved` |
| **g24r8** | 24 | 8 | 108 | 1200 | `backward`, `backward.rust`, `backward_b2.rust`, `backward_b2.cap20000.rust`, `forward`, `forward.rust`, `bench`, `bench.solved`, `bench.unsolved` |
| g25r4 … g31r4 | 25-31 | 4 | 117-180 | 150 each | `backward_audit.rust` only |
| **g32r4** | 32 | 4 | 192 | 2250 | `backward`, `backward.rust`, `backward_b2.rust`, `backward_b2.cap20000.rust`, `forward`, `forward.rust`, `bench`, `bench.solved`, `bench.unsolved` |
| g40r4 | 40 | 4 | 300 | 170 | `backward_audit.rust` only |
| g48r4 | 48 | 4 | 432 | 170 | `backward_audit.rust` only |
| g56r4 | 56 | 4 | 588 | 170 | `backward_audit.rust` only |
| g64r4 | 64 | 4 | 768 | 170 | `backward_audit.rust` only |
| g80r4 | 80 | 4 | 1200 | 20 | none |
| g96r4 | 96 | 4 | 1728 | 20 | none |

Reading the file column:

- `backward*.jsonl` — subgoal-decision training labels. One record per candidate subgoal
  at each decision, with its exact cost. `.rust` marks the Rust engine as the producer.
  `_b2` marks the extended plan language. `cap20000` marks a run under a raised
  per-rollout iteration cap.
- `forward*.jsonl` — move-by-move training labels. Per position: the optimal move set
  and the exact cost-to-go.
- `bench.jsonl` — the pinned 450-instance benchmark for that configuration.
- `bench.solved.jsonl` — the subset with a known exact optimum.
- `bench.unsolved.jsonl` — the subset the exact solver failed to grade.
- `backward_audit.rust.jsonl` — an exact-labelled decision corpus on the 150 test
  boards, 10 instances per board. It is a fidelity yardstick, not training data.

Six configurations carry a full pipeline: `g16r4`, `g16r6`, `g16r8`, `g24r4`, `g24r8`,
`g32r4`. These are called the six rungs. Everything else is either a labeler-fidelity
ladder rung or a lean-board size probe.

Two configurations, `g80r4` and `g96r4`, have 20 boards each and no data at all. The
registry marks them UNVERIFIABLE, because they lie beyond the exact solver's size
envelope. See section E.

### B.4 The benchmarks and exams

The word "exam" is the self-play project's term for a fixed instance set used to score a
model. The supervised project calls the same thing a benchmark. They are the same kind
of object.

#### B.4.1 The pinned benchmarks

Every pinned benchmark is 450 instances drawn from 150 boards, 3 instances per board,
with `random.Random(1)` consumed sequentially. Two different scripts produce them, and
the difference between them is important.

- `SV/eval/bench_instances.py` produced `bench450.jsonl` for `g16r4`. Its sampler is
  `move_planner.evaluate.sample_instance`, which **resamples on oracle failure**. It
  tries up to 200 times per draw at a cap of 60,000 expansions and returns only
  instances the exact solver could grade.
- `SV/scaling/bench.py` produced every other benchmark. Its sampler **keeps the
  instance with `d_star = null` on oracle failure**, at a cap of 200,000 expansions and
  a 60 second wall-time limit. It then splits the file into a solved part and an
  unsolved part, and records the failure rate in a metadata sidecar.

The measured results:

| cfg | file | n | graded (d\* known) | ungraded | oracle failure rate | mean d\* on graded | max d\* |
|---|---|---|---|---|---|---|---|
| g16r4 | `SV/eval/data/bench450.jsonl` | 450 | 450 | 0 | 0 by construction — see note | 6.369 | 12 |
| g16r6 | `SV/scaling/data/g16r6/bench.jsonl` | 450 | 316 | 134 | 29.78 % | 5.693 | 10 |
| g16r8 | `SV/scaling/data/g16r8/bench.jsonl` | 450 | 266 | 184 | 40.89 % | 5.188 | 9 |
| g24r4 | `SV/scaling/data/g24r4/bench.jsonl` | 450 | 232 | 218 | 48.44 % | 7.690 | 14 |
| g24r8 | `SV/scaling/data/g24r8/bench.jsonl` | 450 | 161 | 289 | 64.22 % | 5.621 | 9 |
| g32r4 | `SV/scaling/data/g32r4/bench.jsonl` | 450 | 175 | 275 | 61.11 % | 7.823 | 12 |

**Note on the `g16r4` zero.** The base benchmark has an exact optimum on every line
because the sampler discarded and redrew any instance the oracle could not grade. It is
not a measurement of the oracle's failure rate at 16x16 with 4 robots under the same
protocol as the other five rows. No such measurement exists in this repository. The
project's report generator hard-codes the base point of its "oracle death" chart to
`0.0` with the tooltip "(every benchmark puzzle carries an exact optimum)"
(`SV/eval/report_sections_scale.py`, lines 25-27). That statement is true. The number is
a definition, not a measurement, and the two are not on the same protocol.

The graded and ungraded halves get their own names in the result files:

- **graded set** — `bench.solved.jsonl`. Every instance has an integer `d_star`. All
  quality metrics are computable here.
- **frontier set** — `bench.unsolved.jsonl`. Also called the beyond-oracle set. No
  instance has a known optimum. Only solve rate, search cost and wall clock mean
  anything here.

The frontier files use two different placeholder conventions, and this is a real
inconsistency in the repository:

| cfg | placeholder value in `bench.unsolved.jsonl` |
|---|---|
| g16r6 | `0` on all 134 lines |
| g16r8 | `0` on all 184 lines |
| g24r4 | `null` on all 218 lines |
| g24r8 | `0` on all 289 lines |
| g32r4 | `0` on all 275 lines |

The comparison harness detects both forms and suppresses regret and percent-optimal when
it sees them (`SV/eval/compare.py`, lines 492-499 and 756-761). Any consumer that does
not run that check will silently compute regret against zero. One measured example of
exactly that failure appears in section D.4.

#### B.4.2 The fresh unseen exam

The self-play project added a benchmark on boards nothing has ever trained on.

| property | value |
|---|---|
| instance file | `SPR/results/variants/exam/g24r4_unseen.jsonl` |
| optimum sidecar | `SPR/results/variants/exam/g24r4_unseen.dstar.jsonl` |
| board directory | `SPR/results/variants/exam/boards_g24r4/` (lean boards, 50 files, not in git) |
| configuration | g24r4: 24x24, 4 robots, 108 walls |
| board ids | 20000-20049, that is 50 boards |
| instances | 200, exactly 4 per board |
| generator | `SPR/variants/exam.py --config g24r4 --boards 50 --per-board 4 --seed 777` |
| in-line `d_star` | the literal `0` on all 200 lines — a placeholder, never a value |

The optimum lives only in the sidecar. It was produced later by
`SPR/variants/exam_dstar.py` in two passes, using the same exact solver:

| pass | expansion cap | wall-time cap | instances labelled | mean d\* of that pass |
|---|---|---|---|---|
| 1 | 200,000 | 60 s | 103 | 7.553 |
| 2 | 1,000,000 | 300 s | 34 | 12.206 |
| — | — | — | **137 of 200 labelled**, 63 unlabelled | overall mean 8.708, min 1, max 14 |

Pass 1 uses exactly the caps that defined the graded and frontier split of every pinned
benchmark, so the unseen exam divides the same way: **137 graded-unseen and 63
frontier-unseen, inside one physical file**. Any statement about mean regret on the
unseen exam is a statement about those 137 instances only.

#### B.4.3 What each exam is used for

| set | n | grid / robots | d\* known | used for |
|---|---|---|---|---|
| `SV/eval/data/bench450.jsonl` | 450 | 16 / 4 | all 450 | the base head-to-head, and the historical reference every earlier number is quoted against |
| `SV/scaling/data/<cfg>/bench.solved.jsonl` | 316 / 266 / 232 / 161 / 175 | see table | all | quality comparison at scale — solve rate, regret, percent optimal |
| `SV/scaling/data/<cfg>/bench.unsolved.jsonl` | 134 / 184 / 218 / 289 / 275 | see table | none | the beyond-oracle regime. Solve rate, expansions and seconds only |
| `SPR/results/variants/exam/g24r4_unseen.jsonl` | 200 | 24 / 4 | 137 via sidecar | the self-play variants lab, and the only set no model has trained near |
| `SV/scaling/data/<cfg>/backward_audit.rust.jsonl` | 1500 instances each | 17-31, 40, 48, 56, 64 / 4 | per-candidate exact cost | label-fidelity audits, not solve-rate benchmarks |

Two smaller derived sets exist next to the base benchmark and are used only for smoke
tests and pilots: `bench450_first150.jsonl` (150 instances, boards 2400-2449) and
`bench30_pilot.jsonl` (30 instances, boards 2400-2409).

---

## C. Every metric, defined once

The single canonical aggregator is `SV/eval/compare.py::aggregate` (line 490). Every
other driver in either tree — `SPR/spr/bench.py`, `SPR/spr/fwd/bench.py`,
`SV/eval/merge_compare_shards.py` — imports and calls that exact function. So all
headline metrics have one definition and one implementation.

### C.1 Solve rate

| property | value |
|---|---|
| what it measures | the fraction of instances a system solved |
| unit | fraction in `[0, 1]` in JSON, often shown as a percentage in prose |
| direction | higher is better |
| computable on | every set, graded or frontier |
| denominator | **all instances attempted**, not the instances where a plan was found |
| JSON keys | `aggregate.solve_rate`, `aggregate.solved`, `aggregate.n` |
| code | `SV/eval/compare.py:490-513` |

What sets the per-row `solved` flag differs by planner family, and the difference is the
subject of C.2.

- **Subgoal planner** (`SV/eval/compare.py:467`): `row["solved"] = stx is not None`,
  where `stx` is the result of strict realization. A found plan is a weaker, separately
  recorded fact (`plan_found`, aggregated as `plan_found_rate`).
- **Move-by-move planner** (`SV/eval/compare.py:132`): `"solved": cost is not None`.
  The search returns a move sequence that is legal by construction, so there is no
  realization gap.

### C.2 Realized strict moves, and why plans must be replayed

This is the most important definition in Part 1. It is also the one that changed the
project's headline result.

A subgoal planner does not produce moves. It produces a **plan**: a directed acyclic
graph of subgoals. A subgoal says "park helper `H` on support cell `S` so that the
slider stops on bottleneck cell `B`". A plan is *complete* when no segment is left
unresolved. A complete plan is still an abstraction. It may not be playable.

The repository defines two counts, in `SV/eval/realize.py`.

**`abstract_moves`** — the blocker-clearing count. Each physical segment is simulated
and costed **on its own, with only its intended support robot on the board**. Other
robots are assumed to slide out of the way. This mirrors the plan's own cost model. It
can under-count real moves, so a regret computed from it can be negative. It is recorded
only as a diagnostic (`realized_abstract`), and a counter
`n_negative_abstract_regret` exists purely to catch that case.

**`strict_moves`** — the legal joint-game realization, and the metric of record.
Signature at `SV/eval/realize.py:468`:

```python
def strict_moves(env, state, plan, size=None, log=print, two_phase=True,
                 fail_info=None, moves_out=None):
```

What it does, from its own docstring:

> The plan's physical segments are topologically ordered (a helper's move to its support
> cell before the mover segment that bounces off that support; each robot's own segments
> in child->parent chain order; a support robot departs only after its bounce is
> consumed) and executed one segment at a time on the FULL joint state with
> `simulate.slide` physics: only the segment's robot moves, its path found by BFS over
> slides with all other robots at their current cells as blockers. Returns the total
> slide count, or None if any segment is unreachable (realization failure).

There is no boolean `strict` flag. "Strict" is a different function from `abstract`.

| property | value |
|---|---|
| what it measures | the number of primitive robot moves in a legal execution of the plan |
| unit | moves (integer) |
| direction | lower is better. `None` means the plan could not be played |
| computable on | every set — it needs no optimum |
| JSON key | `row.realized_strict`, `aggregate.mean_moves`, `aggregate.mean_realized_strict` |
| code | `SV/eval/realize.py:468-538`, executor at `:559` |

**Why plans must be replayed.** A complete plan can fail six ways, all enumerated in
`_execute_schedule`:

1. `atomic` — the segment's robot cannot reach the segment end when every other robot is
   a blocker.
2. `approach` — no reachable certified cell before a split segment's bounce.
3. `bounce` — the leg after the support is placed is unreachable.
4. `cyclic` — the dependency graph has no topological order.
5. `target` — everything executed, but the target robot is not on the goal cell.
6. no physical segments at all, or no resolvable mover for a segment.

Because a successful strict realization is a legal sequence ending with the target robot
on the goal, its length is always greater than or equal to the move optimum `d*`. That
inequality is what makes regret well defined.

**The measured consequence.** The subgoal planner's historical 99.6 % solve rate counted
plans that could not be played. Forced through strict realization, it fell to 53.1 %
(FINDINGS item 1). Four verified defects later it reached 80 %, and checking playability
inside the search reached 89.1 %. That whole arc exists because of this one metric.

Two related mechanisms sit next to it but are not part of it:

- **`prefix_playable`** (`SV/eval/realize.py:766`) applies the same machinery to the
  already-fixed part of a *partial* plan, so doomed branches are discarded during search
  rather than at the end. It is deliberately permissive: anything it cannot analyse
  passes.
- **Park repairs** (`SV/skeleton/astar.py::park_repairs`) turn a realization failure into
  new, deterministically derived plans that re-enter the search frontier as ordinary
  costed plans. A repaired plan must itself pass `strict_moves` to count as solved.

**Independent certification.** Every arena result file is replayed before it is
accepted. `SV/eval/replay_validate.py` re-plays the dumped move list under full
joint-state physics using only `simulate.slide` and the standard library, and requires
that every move be a legal full slide, that the sequence length equal the reported
`realized_strict`, and that the target robot end on the goal. A failure quarantines the
file as `.json.uncertified` and stops the run.

### C.3 Regret against the exact optimum, and percent optimal

`d*` (written `d_star` in files, and read "d-star") is the exact minimum number of moves
for that instance, computed once at benchmark construction by an exact solver.

**Regret** is the excess over that minimum.

| property | value |
|---|---|
| formula, subgoal planner | `regret = realized_strict − d_star` (`SV/eval/compare.py:468`) |
| formula, move planner | `regret = moves − d_star` (`SV/eval/compare.py:134`) |
| unit | moves |
| direction | lower is better. 0 is optimal |
| computable on | **graded sets only** — it needs a known `d_star` |
| denominator of the mean | **solved rows only**, not all rows |
| JSON key | `row.regret`, `aggregate.mean_regret` |

**Percent optimal** is the share of the solved instances that were solved in exactly
`d*` moves.

| property | value |
|---|---|
| formula | `100 * count(regret == 0) / count(solved)` (`SV/eval/compare.py:507-509`) |
| unit | percent, 0 to 100 |
| direction | higher is better |
| computable on | graded sets only |
| denominator | **solved rows only** |
| JSON key | `aggregate.pct_optimal` |

Both are `null` when the instance file carries no optimum. The harness sets a flag
`d_star_placeholder` when every instance in the file has `d_star` equal to `0` or
`null`, and then nulls `d_star` and `regret` on every row. The in-code comment explains
why:

> Beyond-oracle ("frontier") sets have no known optimum. Their instance files
> historically carry d_star = 0, which silently turns regret into "solution length" and
> pct_optimal into "fraction solved in 0 moves".

**Flag.** Both metrics are defined on a subset only: the graded half of each benchmark,
and the 137 labelled instances of the unseen exam. On the frontier sets they do not
exist at any budget, because no reference optimum exists there at all.

**A second, unrelated quantity is also called "regret".** `SV/nn_labeler/audit.py`
reports a *label-space* regret: the cost-to-go gap between the candidate a value network
ranks first and the truly best candidate, in the same decision. It is measured in the
label's cost units, not in game moves, and it is not comparable to the regret above.
This document calls it **label regret** wherever it appears.

### C.4 Expansions — the search budget unit

Both planner families are best-first searches. They are compared at a fixed budget so
that neither can simply buy a better result. The unit is the **expansion**.

The definition is written into the `protocol` block of every result file
(`SV/eval/compare.py:786-789`):

> one popped search node whose children are generated (= 1 policy pass + 1 batched value
> pass over <= k children, in both systems)

`k` is the proposal width — the number of candidate continuations the search keeps at
each expansion. It is 5 by default everywhere, and 5 in every result file cited in this
document. The standard budget is 1200 expansions.

| property | value |
|---|---|
| what it measures | search work, in units of one network-scored node expansion |
| unit | count |
| direction | lower is better at equal solve rate |
| computable on | every set |
| JSON key | `row.expansions`, `aggregate.mean_expansions` |
| denominator of the mean | **all rows**, including unsolved ones |

**Where the counter increments.** In the subgoal search
(`SV/eval/compare.py::_nn_astar_backward`), the loop is `while frontier and expansions <
max_expansions`, and the increment sits at line 243 with the comment `# this pop
generates children`. Three kinds of work are explicitly free and are counted in separate
accounting fields rather than against the budget:

- forced exact fixes, where a segment turns out to have a known exact path
  (`free_exact_fix_expands`),
- the final pop of a complete plan,
- all physics: strict realization, prefix checks and park repairs
  (`physics_calls_strict_realize`, `physics_calls_prefix_check`,
  `physics_calls_park_repair`).

**Two honest caveats, both documented in the code.**

1. *Over-count.* The increment happens before candidate generation. A pop that produces
   no candidates still costs one expansion while making zero network passes.
2. *Forward derivation.* For the move-by-move planner under `eval.compare`, expansions
   are not incremented but derived from the number of network calls:
   `expansions = max(0, (guide.calls - c0 - 1)) // 2` (`SV/eval/compare.py:128`). The
   wrapper's docstring states the model: one root call, then exactly two calls per
   expansion that has successors. The *cap* actually passed to the search is
   `max_iters`, which increments even on a pop with no successors, so the reported
   expansions can be lower than the iterations actually spent. `SPR/spr/fwd/bench.py`
   keeps both counters per row and cross-checks them (`expansions_from_calls` against
   `expansions_own`).

**The granularity asymmetry, stated by the project itself** (footnote 1 of the generated
comparison markdown, `SV/eval/compare.py:637-643`):

> one backward expansion commits an entire subgoal (a multi-move commitment: mover
> approach plus helper placement), while one forward expansion commits a single
> primitive move. A shared expansion cap therefore grants the backward planner strictly
> more solution-building work per expansion; the matched budget is generous to the
> backward system (the conservative direction for the comparison).

This is why wall-clock time is reported next to expansions and never instead of it.

### C.5 Wall-clock seconds per puzzle

| property | value |
|---|---|
| what it measures | elapsed time of the search on one instance |
| unit | seconds |
| direction | lower is better |
| computable on | every set |
| JSON key | `row.seconds`, `aggregate.mean_seconds` |
| clock | `time.perf_counter` in the benchmark drivers |
| denominator of the mean | all rows |

**What the timer includes and excludes.** The timer starts after the board is loaded and
the state is built, and stops when the search returns
(`SV/eval/compare.py:349` and `:425`). Therefore:

- **Excluded**: network checkpoint loading (done once, outside the loop), board loading
  and its cache, and state construction.
- **Included**: the whole search loop and all network passes.
- **Conditional**: in *anytime* mode, strict realization, park repairs and prefix checks
  run inside the search, so their cost is inside `seconds`. In plain mode the final
  certifying realization runs after the timer stops and is not counted.

Wall clock is machine-dependent. Solve rates, move counts and expansion counts are not.
Result files produced on different machines can be compared on the latter and not on the
former. The repository states this explicitly for the rows measured on this cluster.

A run-level `wall_seconds` also exists (`SPR/spr/arena.py:216`). It is a different
number: it covers subprocess spawning, merging and replay certification for the whole
run, not one puzzle.

### C.6 Argmin agreement — the label-fidelity metric

This metric does not score a planner. It scores a *labeller*: a system that produces
training labels, checked against a system that produces exact ones.

Background: at each decision the subgoal labeller enumerates candidate subgoals and
prices each one by solving to completion. The candidate with the lowest cost is the
**argmin** — the one a trained network is supposed to pick.

| property | value |
|---|---|
| what it measures | how often a candidate labeller's top pick is genuinely optimal |
| unit | fraction in `[0, 1]` |
| direction | higher is better |
| computable on | matched decision corpora only — the same instances labelled twice |
| JSON key | `decisions.argmin_agreement` and `summary.argmin_agreement` |
| code | `SV/nn_labeler/audit_descent.py:227-238` and `:300-303` |

The computation, per matched decision group:

```python
best = min(range(len(drecs)),
           key=lambda i: (int(drecs[i]["cost_to_go"]), i))
bk = cand_key(drecs[best])
argmin_scored += 1
if bk not in emap:
    argmin_unmatched += 1
    hit = False
else:
    hit = bool(emap[bk]["is_optimal"])
argmin_hits += hit
```

Precisely:

- The argmin ranges over the candidate records the tested labeller emitted for that
  decision group, ordered by integer `cost_to_go`.
- Ties are broken by file order — the tuple key `(cost_to_go, i)` makes the
  earliest-listed minimum win.
- A hit requires two things: the chosen candidate must exist on the exact side, and it
  must carry `is_optimal == True` there. Candidates that have no exact counterpart count
  as misses.
- The denominator is the number of matched decision groups.
- On both sides `is_optimal` means `cost_to_go == min(cost_to_go)`, so ties all count as
  optimal.

A caution recorded in the code: deeper decision groups diverge by construction, so
"depth 0 is where the audit is meaningful". Every gate reported below has 100 % depth-0
coverage.

A separate, similarly named metric scores a *checkpoint* rather than a corpus:
`top1_optimal` in `SV/nn_labeler/audit.py` takes the argmin over the network's predicted
costs for the candidates in a group. Do not merge the two.

### C.7 Metric availability at a glance

| metric | graded pinned sets | frontier pinned sets | unseen exam | label corpora |
|---|---|---|---|---|
| solve rate | yes | yes | yes | not applicable |
| realized strict moves | yes | yes | yes | not applicable |
| regret vs d\* | yes | **no — no optimum exists** | yes, on 137 of 200 | not applicable |
| percent optimal | yes | **no** | yes, on 137 of 200 | not applicable |
| expansions | yes | yes | yes | not applicable |
| seconds per puzzle | yes | yes | yes | yes |
| argmin agreement | not applicable | not applicable | not applicable | yes |

---

## D. The non-learned solvers, scored

Four systems in this repository solve or bound puzzles with no neural network anywhere.
They are the ground floor everything else is measured against.

| system | what it is | what it produces | where it is used |
|---|---|---|---|
| D.1 exact move oracle | A\* over joint robot states | the exact optimum `d*` | benchmark labels, forward training labels |
| D.2 exact subgoal labeller | best-first search over partial plans, every candidate priced by a full solve | per-candidate exact costs | backward training labels, audit corpora |
| D.3 hand-written-scorer subgoal search | the production search with both network calls replaced by hand-written functions | plans, scored like a real planner | the no-network control |
| D.4 exhaustive plan-language probe | exhaustive best-first enumeration of the plan language, every complete plan strictly realized | the language's ceiling | measuring what the plan vocabulary can express at all |

Both D.1 and D.2 exist twice: as a Python reference implementation and as a Rust port.
The Rust port is the production engine.

### D.1 The exact move oracle

**What it does.** `SV/move_planner/oracle.py::solve` runs A\* over joint robot states.
Every move costs 1. The heuristic is `relaxed_target_dist`: a reverse breadth-first
search from the goal cell that counts how many slides the target robot alone would need
if a blocker were available at every cell. The docstring states the guarantee:

> This is a lower bound on the target robot's moves, hence on the total move count, so
> A\* returns the true optimum and expands very few nodes.

**What it guarantees.** When it returns a number, that number is the exact minimum. The
heuristic is admissible, edges have unit cost, and the goal test happens on pop. When it
cannot finish it returns `None`. It never returns an approximation.

**Its caps.**

| caller | expansion cap | wall-time cap | source |
|---|---|---|---|
| module default | 200,000 | none | `SV/move_planner/oracle.py` |
| `SV/scaling/bench.py` (all scaled benchmarks) | 200,000 | 60.0 s, via `signal.setitimer` | CLI defaults |
| `move_planner.evaluate.sample_instance` (base benchmark) | 60,000 | none | `SV/move_planner/evaluate.py:172` |
| `move_planner.generate` (forward training labels) | 40,000 | none | CLI default |
| Rust engine, forward tasks | 40,000 | **none by design** | `SV/rust_datagen/src/io.rs` |

The Rust engine refuses a wall-time cap outright. Its error message: `timeout_s is a
non-deterministic parity knob this engine does not implement; use budget.solver_iters`.
Determinism is preferred over a time limit.

**Its size limit.** The Rust engine packs the joint robot state into one integer. Cell
coordinates use `coord_bits(n)` bits each, and that function tops out at 6 bits, which
addresses 0 to 63. The envelope assertion is compiled into the release build
(`SV/rust_datagen/src/move_oracle.rs:94-102`):

```rust
fn assert_envelope(n: u16, r: usize) {
    assert!(n <= 64, "state packing supports n <= 64, got n = {n}");
    assert!(r as u32 * CELL_BITS <= 128, ...);
}
```
with `const CELL_BITS: u32 = 12;` giving a maximum of 10 robots. A second, friendlier
check sits at the input boundary in `io.rs`. **So exact ground truth is obtainable at
grid size 64 and at no size above it.**

**Measured results — the graded share of each benchmark.** This is the oracle's own
score. It is the complement of the failure rates in section B.4.1.

| cfg | grid / robots | instances | graded by the oracle | failure rate | mean d\* | max d\* |
|---|---|---|---|---|---|---|
| g16r4 | 16 / 4 | 450 | 450 | **0 by construction** | 6.369 | 12 |
| g16r6 | 16 / 6 | 450 | 316 | 29.78 % | 5.693 | 10 |
| g16r8 | 16 / 8 | 450 | 266 | 40.89 % | 5.188 | 9 |
| g24r4 | 24 / 4 | 450 | 232 | 48.44 % | 7.690 | 14 |
| g32r4 | 32 / 4 | 450 | 175 | 61.11 % | 7.823 | 12 |
| g24r8 | 24 / 8 | 450 | 161 | **64.22 %** | 5.621 | 9 |

Two facts to carry forward. First, the worst rung is `g24r8` at 64.22 %, not `g32r4`.
The project's own PRIMER lists the curve as "0 % (4 robots) -> 29.8 % (6 robots) ->
40.9 % (8 robots) -> 48.4 % (24x24) -> 61.1 % (32x32)" and omits the 24x24 8-robot point
entirely. FINDINGS item 18b already records that omission as an erratum. Second, the
maximum exact optimum stays at 9 to 14 moves at every size measured. Larger boards
lengthen slides, they do not lengthen solutions.

**Where more budget helps, measured.** All 184 oracle failures at `g16r8` were re-run at
ten times the budget (`SV/scaling/data/g16r8/cap_probe_10x_results.jsonl`, 184 rows):

| outcome | count | share |
|---|---|---|
| solved at 10x budget | 75 | 40.8 % |
| still unsolved | 109 | 59.2 % |

The 75 newly solved instances have mean `d*` 8.23 and maximum 10 — harder than the
graded set's mean of 5.19, as expected.

**Timing.** No per-instance timing file for the oracle survives on disk. The evidence
that exists is indirect and is reported honestly:

- The 60-second wall-time cap means a failure at `g24r8` costs up to 60 s. An analysis
  note in the repository states "Bench cost is dominated by oracle failures (60 s each)"
  (`SV/analysis/pass4_scaling.md:237`).
- The exact *forward labelling* run at 64x64 in the Rust verification battery reported
  a yield of "~11 solvable per 120 attempts" at the 40,000-expansion budget
  (`SV/rust_datagen/VERIFICATION.md:606-608`), that is about 9 %.

**Where it cannot run.**

| limit | value | consequence |
|---|---|---|
| grid size | `n > 64` | hard assertion failure in release builds. No exact answer exists at 80x80 or 96x96 at any budget. |
| robot count | `R > 10` | same assertion. No configuration in the study exceeds 8. |
| expansion budget | 200,000 at benchmark time | 29.8 % to 64.2 % of instances go ungraded, depending on rung. |
| wall time | 60 s at benchmark time | contributes to the same failures. The two causes are not separated in the stored metadata. |

### D.2 The exact subgoal labeller

**What it does.** This is a second, different exact system, and it is important not to
merge it with D.1. It searches over *partial plans*, not over robot positions. For each
decision it enumerates the candidate subgoals, commits each one, and solves the
remaining plan to completion to price it. The candidate with the lowest resulting cost is
marked `is_optimal`.

The Python reference is `SV/scaling/backward_label.py`, wrapping `SV/nn/generate.py`.
The production engine is the Rust port, reached through `SV/scaling/rust_bridge.py`.

**Its caps.** The bridge writes these into every work item:

| cap | value | meaning |
|---|---|---|
| `max_candidates` | 14 | candidates considered per decision |
| `max_iters` | 4,000 | pops of the plan-space search |
| `max_frontier` | 40,000 | heap-size guard, the memory limit |
| `dependent_edge_weight` | 2 | the cost model for helper-dependent slides |
| `budget.solver_iters` | 10,000,000 | total plan-solve iterations per rollout |
| wall time | **none** | the Python path uses a 120 s alarm. The Rust path replaces it with the deterministic iteration budget |

**What it guarantees, and what it does not.** Per-decision labels are exact under those
caps, and the Rust port reproduces them bit for bit. The formal gate replayed 20,649
decision contexts and 258,999 labels against the Python reference with zero differences
(`SV/rust_datagen/VERIFICATION.md`). A later gate for the extended plan language replayed
894 labels across 68 pinned decisions with zero differences.

The honest limit is recorded in `SV/rust_datagen/ADOPTION.md`. The inner plan search's
cost estimate is **not admissible**, and the Python reference's candidate ordering
depends on the iteration order of a Python set. So the *trajectory* a rollout takes can
differ between two runs at equal-score tie points. The measured incidence is 12 records
in 71,200, about 0.017 %. This was adjudicated and accepted, because the reference itself
is under-determined at those points.

**Its yield, measured.** The labeller does not solve a fixed instance list. It samples up
to `per_graph * 4` instances per board and keeps the first `per_graph` that produce
records. The per-board bookkeeping is stored in the manifests, so the keep rate is
directly measurable. This is the labeller's own success rate, and it is the closest thing
the study has to an "exact solver failure rate" at large sizes.

| cfg | grid | boards | attempts | kept | keep rate |
|---|---|---|---|---|---|
| g8r4 | 8 | 1050 | 16,156 | 10,500 | 0.650 |
| g10r4 | 10 | 1050 | 16,259 | 10,500 | 0.646 |
| g15r4 | 15 | 1050 | 16,825 | 10,500 | 0.624 |
| g17r4 | 17 | 150 | 2,437 | 1,500 | 0.616 |
| g20r4 | 20 | 150 | 2,520 | 1,500 | 0.595 |
| g24r4 (20-board probe) | 24 | 20 | 338 | 200 | 0.592 |
| g31r4 | 31 | 150 | 2,577 | 1,500 | 0.582 |
| g32r4 (20-board probe) | 32 | 20 | 379 | 200 | 0.528 |
| g40r4 | 40 | 150 | 2,582 | 1,500 | 0.581 |
| g48r4 | 48 | 150 | 2,662 | 1,500 | 0.564 |
| g56r4 | 56 | 150 | 2,669 | 1,500 | 0.562 |
| g64r4 | 64 | 150 | 2,716 | 1,500 | 0.552 |

The degradation from 8x8 to 64x64 is gentle: 0.650 to 0.552 across an eight-fold
increase in board side. Contrast that with the move oracle's 0 % to 64 % failure curve
over a two-fold increase. The two systems fail at completely different rates, because
they solve completely different problems. Section E returns to this.

Note also what this table is not. Because failures are resampled away, the audit corpora
at 40 to 64 contain 1500 of 1500 instances each. That is a property of the sampler, not
evidence that the labeller succeeds on every instance.

**Timing, measured in job logs.** Each of these runs labelled 150 boards at 10 instances
per board with 16 threads on one GPU node's CPUs.

| cfg | grid | records | wall seconds | seconds per kept instance |
|---|---|---|---|---|
| g40r4 | 40 | 11,821 | 31 | 0.021 |
| g48r4 | 48 | 11,394 | 61 | 0.041 |
| g56r4 | 56 | 11,507 | 111 | 0.074 |
| g64r4 | 64 | 11,321 | 188 | 0.125 |

Board construction for the same runs, using lean boards: 20 s at 40x40, 31 s at 48x48,
45 s at 56x56, 63 s at 64x64 for 170 boards each.

A second timing pair exists at 24x24 and 32x32, quoted in FINDINGS item 58 as 6 s for
200 instances at 24x24 and 15 s for 200 instances at 32x32 on 4 threads, that is 0.030
and 0.075 seconds per instance. **The primary file for those two numbers is not on
disk** — FINDINGS names a session scratchpad that no longer exists. What survives is the
run's raw audit trail (`exact_g24r4_b0_19` and `exact_g32r4_b0_19` under
`SV/scaling/data/<cfg>/rust_work/`), which confirms the run happened at the stated scale
but carries no timing. Those two numbers are therefore reported here as FINDINGS prose,
not as verified measurements.

**Reported speed-ups against the Python reference** (`SV/rust_datagen/README.md`, traced
in `BENCH.md`, caveated in `VERIFICATION.md` section 9): backward labelling 52.7x at
`g16r4`, 178.9x at `g16r6`, 169.2x at `g16r8`, 108.0x at `g24r4` and 79.9x at `g32r4`.
Forward labelling ran 58.4x to 79.9x faster after optimisation. `VERIFICATION.md` adds two
cautions: the `g16r4` margin is load-sensitive and an independent re-run measured 50.4x,
and an earlier "~66x" forward figure was unpaired and must not be quoted.

**Where it cannot run.** The same `n <= 64` envelope as D.1, for the same reason — it
uses the same packed state. At 80x80 and 96x96 nothing can check its output, which is
why `SV/scaling/configs.py` marks those two configurations UNVERIFIABLE in a source
comment.

### D.3 The hand-written-scorer subgoal search — the no-network control

**What it does.** `SV/analysis/heuristic_baseline.py` answers a direct question the
project owner asked: are the learned subgoal results just the hand-written labeller
replayed? It runs the production evaluation loop and replaces exactly the two points
where a network acts:

- the shortlist, ranked by `skeleton.heuristics.score` ascending instead of by policy
  network log-probability,
- the search frontier, ordered by `plan.cost()` — the labeller's own admissible estimate
  — instead of by fixed cost plus value-network cost-to-go.

Everything else is unchanged: the same candidate pool, the same 1200-expansion budget,
the same width `k = 5`, the same anytime realization check, the same park repairs. All
solved rows were replay-certified inside the job.

**Measured results.**

| set | n | solved | solve rate | mean expansions | median s | mean s | total s | mean regret | percent optimal |
|---|---|---|---|---|---|---|---|---|---|
| base 450 (`g16r4`) | 450 | 394 | **87.56 %** | 21.5 | 0.0044 | 0.0372 | 16.7 | 1.756 | 58.38 % |
| g16r6 graded | 316 | 276 | **87.34 %** | 163.8 | 0.0067 | 0.5630 | 177.9 | 2.101 | 56.88 % |
| g16r6 frontier | 134 | 77 | **57.46 %** | 576.8 | 1.0845 | 1.5578 | 208.7 | not defined | not defined |
| g16r6 pooled | 450 | 353 | **78.44 %** | — | — | — | — | not defined | not defined |

The gap to the learned system, from FINDINGS item 49, with bootstrap confidence
intervals: base 450 +8.0 points `[+5.6, +10.7]`, `g16r6` graded +9.5 `[+5.7, +13.7]`,
`g16r6` frontier +23.1 `[+15.4, +31.2]`, and pooled +13.6 `[+10.2, +17.1]`, all with
`p < 0.0001`.

**A data-versus-document flag.** The stored aggregate for the `g16r6` frontier file
reports `mean_regret = 16.19` and `pct_optimal = 0.0`. Those numbers are meaningless.
That file's instances carry `d_star = 0`, so "regret" there is simply the solution
length. The comparison harness detects this case and suppresses both quantities, but the
stored file predates or bypasses that guard, so the raw aggregate contains the wrong
numbers. Read solve rate, expansions and seconds from frontier files. Read nothing else.

**Where it has not been run.** Only three sets exist on disk: base 450, `g16r6` graded
and `g16r6` frontier. There is **no measurement of this control at `g16r8`, `g24r4`,
`g24r8` or `g32r4`**. Its script accepts all of those configurations. The runs were
simply never made. Not measured.

**Why it matters for Part 2.** On the `g16r6` frontier the no-network subgoal search
scores 57.5 %, and the trained move-by-move planner scores 48.5 % on the same set. The
point estimate favours the formulation with no learning at all, though the interval
`[-2.3, +20.1]` with `p = 0.14` means the comparison is underpowered. On the graded sets
it loses badly. The reading FINDINGS records is that both ingredients are needed:
subgoal structure to survive the frontier, learned ranking to win everywhere.

### D.4 The exhaustive plan-language probe

**What it does.** `SPR/spr/ceiling.py` measures what the plan vocabulary can express, with
no network anywhere. For each instance it runs a best-first search over partial plans in
abstract-cost order — the solver's own admissible ordering — and strictly realizes every
complete plan it pops, using the same `strict_moves` certifier the arena uses.

It records four things per instance:

- `first_realizable_moves` — the strict move count of the *first* certified plan popped.
  This is what a search with perfect ranking and a first-solution convention would return.
- `best_realizable_moves` — the smallest strict move count over every certified plan
  popped before the popped abstract cost reaches that minimum.
- the abstract cost of each of those plans,
- a category.

The four categories are:

| category | meaning |
|---|---|
| `REALIZABLE_EXISTS` | at least one complete plan was found and played successfully |
| `NO_REALIZABLE_PLAN` | complete plans exist, none survives physics |
| `NO_COMPLETE_PLAN` | the language cannot even express a complete plan here |
| `INCONCLUSIVE` | a cap tripped before either answer was reached |

**Vocabulary levels.** The probe takes a `--vocab` flag with three values. This is the
plan language, and Part 2 is about how it grew.

| level | what it adds |
|---|---|
| `base` | plain subgoals: a bottleneck the mover stops at, a support cell a helper occupies, the chosen helper |
| `b1` | stoppers may park on cells with no wall behind them, and a plan may schedule a robot to slide aside before a named segment. Failed realizations get deterministic park repairs. |
| `b2` | `b1` plus reuse of a robot the plan already placed — one parked robot stopping two sliders, the target robot itself serving as a stopper, a placed helper moving on to a second post — and generalized park repairs that can clear two robots at once |

No result file produced by this probe uses `b1`. Every run stored under
`SPR/results/ceiling/` is `base` or `b2`. The `b1` level was measured by the older probe
described in D.5, not by this one.

**What it guarantees.** It is exhaustive *under its caps and no further*. The flag
`exhausted` is set only when no cap tripped. There is also a termination bound: once a
realizable plan of strict cost `c` is found, popping stops when the abstract cost reaches
`c + slack`, since nothing cheaper can remain. The docstring states the residual caveat
plainly: `best_realizable_moves` is a **tight upper bound on the true language optimum,
not a proof**, because a plan whose strict count undercuts its abstract cost — an
incidental robot happening to serve as a stopper — could sit beyond the bound.

That caveat was tested. Re-running with `slack` of 4 and 12 moves the mean best-plan
regret by at most 0.093 moves. The bound is tight in practice.

**Its caps.**

| flag | default | what it limits |
|---|---|---|
| `--max-iters` | 200,000 | pops per instance |
| `--max-frontier` | 400,000 | heap size per instance, the memory guard |
| `--time-cap` | 60.0 s | wall clock per instance |
| `--park-cap` | 2 | parks a repair may schedule |
| `--slack` | 0.0 | extra abstract cost to keep exploring past the bound |
| `--workers` | 16 | worker processes |

There is no depth flag. One internal limit is not exposed: a plan with more than
`2 * (helpers + 2)` open edges is skipped, which is 10 at 4 robots.

**Measured results — the graded arms, where an optimum exists.** These are the numbers
that bound what any subgoal planner can ever achieve.

| set | n | vocab | solve ceiling | mean d\* | best plan moves | best regret | reach d\* | first plan moves | first regret |
|---|---|---|---|---|---|---|---|---|---|
| g16r4 bench450 | 450 | base | 408 = **90.67 %** | 6.262 | 7.681 | **+1.419** | 61.76 % | 8.081 | +1.819 |
| g16r4 bench450 | 450 | b2 | 441 = **98.00 %** | 6.324 | 7.220 | **+0.896** | 66.89 % | 7.762 | +1.438 |
| g24r4 graded | 232 | base | 216 = **93.10 %** | 7.593 | 9.315 | **+1.722** | 57.41 % | 9.727 | +2.134 |
| g24r4 graded | 232 | b2 | 228 = **98.28 %** | 7.640 | 8.811 | **+1.171** | 62.72 % | 9.351 | +1.711 |
| g24r8 graded | 161 | base | 148 = **91.93 %** | 5.534 | 7.115 | +1.581 | 58.78 % | 7.385 | +1.851 |
| g24r8 graded | 161 | b2 | 159 = **98.76 %** | 5.610 | 6.943 | +1.333 | 62.89 % | 7.340 | +1.730 |
| g32r4 graded | 175 | base | 156 = **89.14 %** | 7.692 | 9.250 | +1.558 | 57.69 % | 9.442 | +1.750 |
| g32r4 graded | 175 | b2 | 166 = **94.86 %** | 7.729 | 9.006 | +1.277 | 61.45 % | 9.271 | +1.542 |

Failure taxonomy for the base arms, which are all fully exhaustive with zero capped
instances:

| set | realizable | no realizable plan | no complete plan |
|---|---|---|---|
| g16r4 bench450 | 408 | 20 | 22 |
| g24r4 graded | 216 | 7 | 9 |
| g24r8 graded | 148 | 10 | 3 |
| g32r4 graded | 156 | 11 | 8 |

**Measured results — the frontier arms.**

| set | n | vocab | realizable | inconclusive | solve ceiling | note |
|---|---|---|---|---|---|---|
| g24r4 frontier | 218 | base | 103 | 0 | **47.25 %** | fully exhaustive |
| g24r4 frontier | 218 | b2 | 114 | 104 | **≥ 52.29 %** | 115 instances hit a cap |
| g32r4 frontier | 275 | base | 142 | 0 | **51.64 %** | fully exhaustive |
| g32r4 frontier | 275 | b2 | 152 | 123 | **≥ 55.27 %** | 133 hit a cap |
| g24r8 frontier | 289 | b2 | 180 | 109 | **≥ 62.28 %** | 124 hit a cap |

The base-vocabulary frontier run for `g24r8` does not exist on disk, although the job
script requests it. That cell is missing.

Every `b2` ceiling here is a lower bound, not a ceiling, because a third to a half of the
instances hit a cap. The base ceilings are true ceilings, because nothing hit a cap.

**A soundness check that passes everywhere.** The counter `n_best_below_dstar` is 0 on
every graded arm. No certified plan ever produced fewer moves than the exact optimum. The
probe never contradicts the oracle.

**Timing.**

Two different times are reported, and they must not be confused. **Wall / n** is the
whole run's elapsed time divided by the instance count, so it reflects the worker count.
**Per-row** columns are the probe's own per-instance timer, summed across workers.

| set | n | vocab | workers | wall s | wall / n | per-row mean s | per-row median s | per-row max s |
|---|---|---|---|---|---|---|---|---|
| g16r4 bench450 | 450 | base | 16 | 5.8 | 0.013 | 0.038 | 0.000 | 4.71 |
| g16r4 bench450 | 450 | b2 | 16 | 98.6 | 0.219 | 2.474 | 0.010 | 63.77 |
| g24r4 graded | 232 | base | 16 | 9.6 | 0.041 | 0.075 | 0.000 | 3.74 |
| g24r4 graded | 232 | b2 | 16 | 103.0 | 0.444 | 3.382 | 0.010 | 95.59 |
| g24r4 frontier | 218 | base | 16 | 11.0 | 0.050 | 0.234 | 0.030 | 5.61 |
| g24r4 frontier | 218 | b2 | 8 | 480.9 | 2.206 | 16.776 | 22.025 | 43.05 |
| g24r8 graded | 161 | base | 16 | 66.7 | 0.414 | 1.012 | 0.000 | 62.18 |
| g24r8 graded | 161 | b2 | 8 | 24.4 | 0.152 | 0.550 | 0.010 | 17.92 |
| g24r8 frontier | 289 | b2 | 8 | 466.7 | 1.615 | 12.256 | 11.070 | 48.06 |
| g32r4 graded | 175 | base | 16 | 22.0 | 0.126 | 0.036 | 0.000 | 1.98 |
| g32r4 graded | 175 | b2 | 8 | 82.6 | 0.472 | 2.053 | 0.000 | 33.87 |
| g32r4 frontier | 275 | base | 16 | 37.5 | 0.136 | 0.191 | 0.020 | 3.94 |
| g32r4 frontier | 275 | b2 | 8 | 587.1 | 2.135 | 15.713 | 18.900 | 47.91 |

Two readings. The base vocabulary is nearly free: the whole 450-instance base benchmark
finishes in 5.8 seconds on 16 workers, and the median instance takes no measurable time.
The extended `b2` vocabulary is far more expensive on three of the four graded sets —
comparing total per-row seconds, it costs 65.9 times more at `g16r4` (1113.4 s against
16.9 s), 45.1 times more at `g24r4` (784.7 s against 17.4 s) and 57.0 times more at
`g32r4` (359.3 s against 6.3 s). At `g24r8` the comparison inverts (88.5 s for `b2`
against 162.9 s for base), because the base arm there is dominated by three instances:
two on board 995 costing 62.18 s and 58.17 s, and one on board 939 costing 34.7 s. Those
three account for 155.05 s of the arm's 162.92 s total. On the frontier sets the `b2` median instance takes
11 to 22 seconds and a third to a half of all instances trip a cap.

**Where it cannot run.** The probe has no lean-board path, so it cannot be pointed at the
40x40 to 96x96 configurations or at the unseen exam's boards. **There is no ceiling number
for the unseen exam.** Not measured.

### D.5 The older ceiling probe, and where the two disagree

A second, earlier probe exists in the supervised tree
(`SV/analysis/artifacts/ceiling_probe.py`). It was applied only to the instances the
planner failed, not to the whole benchmark, and it drove the base-scale story the
repository's documents tell. Both probe families were checked in this session.

The older chain at 16x16 with 4 robots, all derived from one 49-instance probe of the
planner's failures:

| step | file | result | implied ceiling |
|---|---|---|---|
| base vocabulary | `ceiling_probe_results.json` | 22 no complete plan, 20 no realizable plan, 7 realizable | (450 − 42) / 450 = **90.67 %** |
| after B1 | `ceiling_probe_results_b1.json` | same 49 re-probed: 38 realizable, 7 + 4 = 11 structural | (450 − 11) / 450 = **97.56 %** |
| after B2 | `ceiling_probe_results_b2.json` | the 11 residuals re-probed: 9 realizable, 2 inconclusive | (450 − 2) / 450 = **99.56 %** |

The two inconclusive instances were re-probed on a 1 TB node at a 5 million-plan frontier
and 30-minute caps and stayed inconclusive, at 1202 s and 1800 s
(`ceiling_probe_results_b2_deep.json`).

**Where the two probes agree.** Exactly. The base-vocabulary ceiling is 408 of 450 in
both, with the identical 22 / 20 split of the two structural failure kinds. Two
independent implementations, one number.

**Where they differ.** The extended-vocabulary ceiling at 16x16 with 4 robots:

| source | value | why |
|---|---|---|
| `SV/FINDINGS.md` item 16, from `ceiling_probe_results_b2.json` | **99.6 %** | only the 11 residual instances were re-probed, at large caps, and the other 439 were assumed to stay solvable |
| `SPR/results/ceiling/g16r4_b2.json` | **98.0 %** | the whole 450 re-probed at 60 s / 200,000 iterations / 400,000 frontier. 9 instances hit a cap and count as failures |

This is a caps difference, not a contradiction. Neither number is an upper bound that the
other violates: 98.0 % is a lower bound at tighter caps, and 99.6 % is a lower bound at
looser caps on a smaller re-probed set. The self-play project's own FINDINGS records the
reconciliation in the same terms ("B2 98.0% here vs the 99.6% recorded with 4x caps").
Quote whichever you like, but say which probe and which caps produced it.

The 6-robot chain, verified the same way from
`SV/scaling/results/g16r6/ceiling_probe_*.json`:

| step | probed set | result | implied ceiling |
|---|---|---|---|
| base | 105 planner failures | 48 no complete plan, 23 no realizable plan, 28 realizable, 6 inconclusive | (450 − 71) / 450 = **84.2 %** |
| B1 | 77 re-probed | 63 realizable, 2 no realizable plan, 12 inconclusive | ≥ (450 − 14) / 450 = **96.9 %** |
| B2 | 14 re-probed | 5 realizable, 9 inconclusive, **0 proven impossible** | ≥ (450 − 9) / 450 = **98.0 %** |

Timing for those runs, from the stored per-row `seconds`: base probe mean 7.46 s per
instance with a maximum of 62.5 s. The B1 probe averaged 111.05 s with a maximum of
674.9 s. The B2 probe averaged 398.5 s with a maximum of 655.0 s. The cost of proving impossibility grows
with the vocabulary, sharply.

**Cap sensitivity, checked.** Every "no complete plan" verdict was re-probed at four
times the plan-width cap. At 16x16 with 4 robots, all 22 stayed structural
(`cap_sensitivity_base.json`). At 16x16 with 6 robots, all 48 stayed structural, with one
instance moving between the two structural categories
(`SV/scaling/results/g16r6/cap_sensitivity.json`). **Zero of 70 became solvable.** The
ceiling is a property of the language, not of the search budget.

---

## E. The honest limits of layer 1

Everything Parts 2 and 3 build rests on the ground truth described above. Here is
exactly where that ground truth stops.

### E.1 Exact optimality has a hard size wall at 64

The exact solver packs the joint robot state into a fixed-width integer with 6 bits per
coordinate. That is 0 to 63. `assert!(n <= 64)` is compiled into the release build. There
is no configuration flag, no slower fallback, and no approximation. Above 64 the study
has **no exact answer at any price**.

This is why `SV/scaling/configs.py` describes `g64r4` as "Coarse ladder endpoint, lean
boards; last size with exact ground truth" and `g80r4` and `g96r4` as "UNVERIFIABLE
rung, lean boards; beyond the Rust envelope -- no exact check exists". Those two
configurations have 20 boards each and no data.

The practical answer the project chose is a licence argument, not a proof: show that a
learned labeller matches the exact labeller at every size up to 64, and treat that as the
warranty for using it above 64. The measured curve, from
`SV/nn_labeler/results/{cap,coarse}gate_g*.json`:

| rung | argmin agreement | mean label gap | share gap 0 | negative gaps | depth-0 coverage |
|---|---|---|---|---|---|
| 24x24 | 92.63 % | 0.566 | 85.2 % | 3 | 100 % |
| 32x32 | 91.52 % | 0.553 | 87.7 % | 0 | 100 % |
| 40x40 | 90.74 % | 0.616 | 86.5 % | 2 | 100 % |
| 48x48 | 90.59 % | 0.618 | 87.0 % | 0 | 100 % |
| 56x56 | 89.60 % | 0.655 | 86.6 % | 0 | 100 % |
| 64x64 | 90.06 % | 0.721 | 86.6 % | 3 | 100 % |

The ordering quality is flat from 24 to 64. The absolute calibration drifts: the mean gap
grows from 0.55 to 0.72. Whether that licence holds at 80 and 96 is, by construction,
untestable.

### E.2 Below the size wall, most instances are already ungraded

The wall at 64 is not the binding constraint at the sizes the study actually benchmarks.
The binding constraint is that the exact solver fails on most instances well before then.

| cfg | grid / robots | share of the pinned benchmark with an exact optimum |
|---|---|---|
| g16r4 | 16 / 4 | 100 %, but by resampling, not by measurement |
| g16r6 | 16 / 6 | 70.2 % |
| g16r8 | 16 / 8 | 59.1 % |
| g24r4 | 24 / 4 | 51.6 % |
| g32r4 | 32 / 4 | 38.9 % |
| g24r8 | 24 / 8 | 35.8 % |

At the hardest measured rung, roughly two puzzles in three cannot be graded at all. Ten
times the budget recovers 40.8 % of the failures at `g16r8` and leaves 59.2 %
unreachable.

Three consequences follow, and they shape everything downstream.

1. **Quality metrics exist on a shrinking, biased subset.** Regret and percent optimal
   are computable only on the graded half. That half is, by construction, the *easier*
   half — it is defined as the instances an exact search finished quickly. Any statement
   of the form "system A has lower regret than system B at 24x24 with 8 robots" is a
   statement about the easiest 35.8 % of that benchmark.
2. **Supervised training for the move-by-move planner runs out of teacher.** Its labels
   are exactly what D.1 produces. Where D.1 fails, no forward label exists.
3. **On the frontier sets, solving is self-certifying and nothing else is measurable.**
   A solved instance produces a legal replayed move sequence, which proves solvability.
   But there is no reference to compare its length against. Only solve rate, expansions
   and wall clock are meaningful there.

### E.3 The two exact systems fail at very different rates, and they solve different problems

It is tempting to read "the exact solver dies at 64 % failure" and "the exact labeller
keeps 55 % of its samples at 64x64" as the same fact. They are not.

- The **move oracle** (D.1) computes `d*`: the optimum of the whole puzzle over joint
  robot states. Its state space grows as `(n^2)^R`. It is the thing that dies.
- The **subgoal labeller** (D.2) prices individual candidate subgoals inside a plan
  search. Its keep rate falls only from 0.650 at 8x8 to 0.552 at 64x64.

So the study has exact *subgoal* labels far past the size where it has exact *move*
optima. That asymmetry is what makes the ladder from 17x17 to 64x64 possible at all, and
it is also why a solve-rate comparison at those sizes has no optimum to be scored
against.

### E.4 The plan-language ceiling is a separate, measured wall

Even with perfect ranking and unlimited search, the base subgoal language cannot solve
every puzzle, and cannot reach the optimum on the ones it does solve.

| set | base-language solve ceiling | best-plan mean regret at that ceiling |
|---|---|---|
| g16r4 bench450 | 90.67 % | +1.419 |
| g24r4 graded | 93.10 % | +1.722 |
| g24r8 graded | 91.93 % | +1.581 |
| g32r4 graded | 89.14 % | +1.558 |

These are exhaustive results with zero capped instances, and the cap-sensitivity check
in D.5 shows the structural verdicts do not move at four times the caps. The extended
`b2` vocabulary raises the ceilings to 94.9 % to 98.8 % and lowers the mean regret to
+0.90 to +1.33, but its own probe is partly capped on the frontier sets.

The load-bearing consequence for Part 2: this ceiling is **a property of the vocabulary,
not of training**. No amount of learning moves it. That is the whole reason the plan
language was extended, and it is the bridge into Part 2.

### E.5 The things that were simply not measured

| gap | status |
|---|---|
| exact-oracle failure rate at 16x16 with 4 robots under the scaling protocol | **not measured**; the 0 % on the project's chart is a definition |
| no-network control at `g16r8`, `g24r4`, `g24r8`, `g32r4` | **not measured**; the script supports them, the runs were never made |
| base-vocabulary ceiling on the `g24r8` frontier set | **missing**; the job script requests it, the file does not exist |
| ceiling of any vocabulary on the fresh unseen exam | **not measured**; the probe has no lean-board path |
| any exact ground truth above 64x64 | **impossible**, not merely unmeasured |
| per-instance timing of the exact move oracle | **not on disk**; only aggregate job timings and the 60 s cap survive |
| `b1` vocabulary ceiling runs in the new probe | **none on disk**; every stored run is `base` or `b2` |
| separation of expansion-cap failures from wall-time failures in the benchmark metadata | **not recorded**; only the combined failure count is stored |

---

## F. Places where the data does not support a claim the documents make

Four, all verified this session.

1. **The "oracle death" curve's base point is not a measurement.** `SV/PRIMER.md` section
   6 and the generated report both show 0 % failure at 16x16 with 4 robots as the first
   point of a five-point curve. The other four points come from `bench.jsonl.meta.json`
   files that record real failures. The base point is hard-coded to `0.0` in
   `SV/eval/report_sections_scale.py:25-27`, and the underlying benchmark was built by a
   sampler that discards and redraws unsolvable instances at a *different* expansion cap
   (60,000 rather than 200,000). The five points are not on one protocol.

2. **The same curve omits its worst point.** `SV/PRIMER.md` section 6 lists
   "0 % -> 29.8 % -> 40.9 % -> 48.4 % -> 61.1 %" and does not mention `g24r8` at
   64.22 %, which is the highest failure rate measured anywhere in the study.
   `SV/FINDINGS.md` item 18b already records this as an erratum, so the finding is
   documented but the PRIMER text was not corrected.

3. **The stored `g16r6` frontier aggregate contains a meaningless regret.**
   `SV/scaling/results/g16r6/comparison_ungraded_heuristic_baseline.json` reports
   `mean_regret` of 16.19 and `pct_optimal` of 0.0 on a set whose instances all carry
   `d_star = 0`. The current harness suppresses both quantities in that case. Any reader
   or downstream script that trusts the stored aggregate will report a number that means
   "mean solution length" while calling it regret.

4. **Two ceiling numbers for the same cell circulate.** The extended-vocabulary ceiling at
   16x16 with 4 robots is 99.6 % in `SV/FINDINGS.md` item 16 and 98.0 % in
   `SPR/results/ceiling/g16r4_b2.json`. The difference is caps and probe scope, and
   `SPR/FINDINGS.md` item 3 already states the reconciliation. It is listed here because
   the number appears without its caps in several places.

A fifth item is an inconsistency rather than an unsupported claim: the frontier
benchmark files use `d_star = 0` at four configurations and `d_star = null` at `g24r4`.
Both are handled by the harness. Only one is safe for a naive reader.

---

## G. The three things to carry into Part 2

1. **A plan is not a solution until it is replayed.** Solve rate for the subgoal family
   means "a complete plan was found *and* executed legally on the full board", measured
   by `strict_moves`. That single decision moved the project's headline from 99.6 % to
   53.1 %, and everything after it is the climb back.

2. **The exact solver's reach shrinks fast, and quality metrics shrink with it.** Regret
   and percent optimal exist only where an exact optimum exists: 100 % of the base
   benchmark, but only 51.6 % at 24x24, 38.9 % at 32x32 and 35.8 % at 24x24 with 8
   robots. Beyond that line, solve rate is the only quality signal, and it is
   self-certifying. That is the regime Part 3 lives in.

3. **There is a measured ceiling that no training can move.** With no networks and
   exhaustive search, the base subgoal language solves at most 89 % to 93 % of the graded
   benchmarks, at a mean of 1.4 to 1.7 moves above the optimum. The extension of that
   language, not better learning, is what raised it. That is the subject of Part 2.

---

## Appendix — verified numbers

Every number in this document was read from a file during the session that wrote it.
Paths are relative to `/scratch/project/open-37-42/petrhyner/MCTS_evolution`. `SV` =
`supervised_valuenet`, `SPR` = `self_play_robots`.

### Board and generator facts

| number | value | file | how read |
|---|---|---|---|
| wall density rule | `round(48 * (n/16)^2)` | `SV/scaling/configs.py::default_walls` | read source |
| walls at 16 / 24 / 32 / 64 | 48 / 108 / 192 / 768 | `SV/scaling/configs.py` | evaluated the rule per config |
| cold `from_env` 16 / 24 / 32 | 8.2 / 60 / 222 s | `SV/FINDINGS.md` item 60a | read prose, cross-checked against `SV/nn_labeler/leanboard.py` module docstring |
| eager pickle 16 / 24 / 32 | 1.6 / 8.8 / 28.6 MB | `SV/FINDINGS.md` item 60a | same |
| lean cost 32 / 64 / 96 | 0.24 / 1.5 / 3.8 s | `SV/FINDINGS.md` item 62c | read prose |
| lean pickle 32 / 64 / 96 | 1.1 / 4.9 / 11.7 MB | `SV/FINDINGS.md` item 62c | read prose |
| lean parity | 1.45 M entries, 0 mismatches | `SV/FINDINGS.md` item 62a | read prose |

### Configuration inventory

| number | value | file | how read |
|---|---|---|---|
| configurations in the registry | 36 | `SV/scaling/configs.py` | imported `CONFIGS` and counted |
| boards on disk per config | see the table in B.3 | `SV/environments*/env_*.pkl` | globbed and counted per config |
| data files per config | see the table in B.3 | `SV/scaling/data/<cfg>/*.jsonl` | globbed, shards excluded |
| standard splits | train 0-699, val 700-899, test/bench 900-1049 | `SV/scaling/configs.py::_STD_RANGES` | read source |
| legacy splits | train 0-95,1000-1799; val 1800-2399; test 112-127,2400-2999; bench 2400-2549 | `SV/scaling/configs.py` | read source |
| id-budget rule | generation 40000-49999, exam 20000-20999, never train on 0-1199 or 20000+ | `SPR/variants/__init__.py` | read source comment |

### Benchmarks

| number | value | file | how read |
|---|---|---|---|
| bench450 lines / d\* nulls | 450 / 0 | `SV/eval/data/bench450.jsonl` | parsed all lines |
| bench450 mean / min / max d\* | 6.3689 / 1 / 12 | same | computed over all 450 |
| bench450 protocol | boards 2400-2549, 3 per board, seed 1 | `SV/eval/data/bench450.jsonl.meta.json` | read JSON |
| g16r6 n / ungraded / rate | 450 / 134 / 0.29778 | `SV/scaling/data/g16r6/bench.jsonl` and `.meta.json` | parsed lines and read JSON; the two agree |
| g16r8 n / ungraded / rate | 450 / 184 / 0.40889 | `SV/scaling/data/g16r8/bench.jsonl` and `.meta.json` | same |
| g24r4 n / ungraded / rate | 450 / 218 / 0.48444 | `SV/scaling/data/g24r4/bench.jsonl` and `.meta.json` | same |
| g24r8 n / ungraded / rate | 450 / 289 / 0.64222 | `SV/scaling/data/g24r8/bench.jsonl` and `.meta.json` | same |
| g32r4 n / ungraded / rate | 450 / 275 / 0.61111 | `SV/scaling/data/g32r4/bench.jsonl` and `.meta.json` | same |
| mean d\* on graded, five rungs | 5.693 / 5.188 / 7.690 / 5.621 / 7.823 | `SV/scaling/data/*/bench.jsonl` | computed over non-null rows |
| max d\* on graded, five rungs | 10 / 9 / 14 / 9 / 12 | same | computed |
| solved / unsolved file line counts | 316+134, 266+184, 232+218, 161+289, 175+275 | `SV/scaling/data/*/bench.{solved,unsolved}.jsonl` | counted lines; each pair sums to 450 |
| frontier placeholder values | `0` at g16r6, g16r8, g24r8, g32r4; `null` at g24r4 | `SV/scaling/data/*/bench.unsolved.jsonl` | parsed all lines, counted distinct values |
| oracle caps in every meta | 200,000 expansions, 60.0 s | `SV/scaling/data/*/bench.jsonl.meta.json` | read JSON |
| base sampler cap and retries | 60,000 expansions, `max_try=200`, resamples on failure | `SV/move_planner/evaluate.py:162-176`, `SV/eval/bench_instances.py:33-35` | read source |
| report base point hard-coded | `"value": 0.0` | `SV/eval/report_sections_scale.py:25-27` | read source |

### The unseen exam

| number | value | file | how read |
|---|---|---|---|
| instances / lines in sidecar | 200 / 200 | `SPR/results/variants/exam/g24r4_unseen.jsonl`, `.dstar.jsonl` | counted lines |
| board ids / boards / per board | 20000-20049 / 50 / exactly 4 | `g24r4_unseen.jsonl` | parsed, counted per `env_id` |
| in-line d\* values | `0` on all 200 | same | counted distinct values |
| labelled / unlabelled | 137 / 63 | `g24r4_unseen.dstar.jsonl` | counted non-null `d_star` |
| pass 1 / pass 2 counts | 103 / 34 | same | counted by `pass` field |
| pass 1 / pass 2 mean d\* | 7.553 / 12.206 | same | computed per pass |
| overall mean / min / max d\* | 8.708 / 1 / 14 | same | computed over 137 |
| caps used | (200,000, 60.0 s) and (1,000,000, 300.0 s) | same | counted distinct `(expansions_cap, wall_s)` |
| generator command | `--config g24r4 --boards 50 --per-board 4 --seed 777` | `SPR/variants/exam.py` | read source defaults |

### Metric definitions

| fact | file and line | how read |
|---|---|---|
| `aggregate` is the single definition | `SV/eval/compare.py:490-529` | read source |
| solve-rate denominator is `n` | `SV/eval/compare.py:503` | read source |
| backward `solved` = strict realization succeeded | `SV/eval/compare.py:467` | read source |
| forward `solved` = search returned a cost | `SV/eval/compare.py:132` | read source |
| `strict_moves` signature and semantics | `SV/eval/realize.py:468` and docstring lines 1-31 | read source |
| regret formulas | `SV/eval/compare.py:134` and `:468` | read source |
| `pct_optimal` denominator is solved rows | `SV/eval/compare.py:507-509` | read source |
| placeholder detection | `SV/eval/compare.py:492-499`, `:756-761` | read source |
| expansion definition string | `SV/eval/compare.py:786-789` | read source |
| expansion increment site | `SV/eval/compare.py:243` | read source |
| forward expansions derived from calls | `SV/eval/compare.py:128`, wrapper docstring at `:64-68` | read source |
| granularity asymmetry footnote | `SV/eval/compare.py:637-643` | read source |
| default `k` = 5 and default budget = 1200 expansions | `SV/eval/compare.py:664-665`, `SPR/spr/arena.py:43-44` | read source defaults |
| timer start and stop | `SV/eval/compare.py:349` and `:425` | read source |
| argmin agreement computation | `SV/nn_labeler/audit_descent.py:227-238`, `:300-303` | read source |
| replay certification requirements | `SV/eval/replay_validate.py:8-21` | read source |

### The exact solvers

| number | value | file | how read |
|---|---|---|---|
| Rust size envelope | `n <= 64` | `SV/rust_datagen/src/move_oracle.rs:95` | read the assertion |
| Rust robot envelope | `R <= 10` via `CELL_BITS = 12` | `SV/rust_datagen/src/move_oracle.rs:65, 96-101` | read source |
| coordinate bits cap at 6 | `coord_bits` returns 6 for `n > 32` | `SV/rust_datagen/src/move_oracle.rs:75-83` | read source |
| Python oracle default cap | `max_expansions=200_000` | `SV/move_planner/oracle.py::solve` | read signature |
| Python oracle has no time cap | wall time imposed by the caller | `SV/scaling/bench.py:52` uses `setitimer` | read source |
| backward labeller caps | `max_iters=4000`, `max_frontier=40_000`, `max_candidates=14` | `SV/scaling/backward_label.py:51`, `SV/scaling/rust_bridge.py` | read source |
| deterministic rollout budget | `DEFAULT_SOLVER_ITERS = 10_000_000` | `SV/rust_datagen/src/io.rs:57` | read source |
| engine rejects wall-time caps | error message quoted in D.2 | `SV/rust_datagen/src/io.rs` | read source |
| forward yield at 64x64 | ~11 solvable per 120 attempts at 40,000 expansions | `SV/rust_datagen/VERIFICATION.md:606-608` | read prose |
| Rust label-parity gate | 20,649 contexts, 258,999 labels, 0 diffs | `SV/rust_datagen/VERIFICATION.md` | read verdict table |
| B2 label-parity gate | 4 legs, 68 decisions, 894 labels, 0 diffs | `SV/prompt.md` and `SV/FINDINGS.md` item 19 | read prose |
| tie-divergence incidence | 12 records in 71,200, ~0.017 % | `SV/rust_datagen/ADOPTION.md` | read prose |
| backward speed-ups | 52.7x / 178.9x / 169.2x / 108.0x / 79.9x | `SV/rust_datagen/README.md`, `VERIFICATION.md` section 9 | read tables |
| 10x cap probe | 184 rows, 75 solved, 109 unsolved | `SV/scaling/data/g16r8/cap_probe_10x_results.jsonl` | parsed, counted by `status` |
| 10x probe recovered d\* | mean 8.227, max 10 | same | computed over the 75 |
| labeller keep rates | see the table in D.2 | `SV/scaling/data/*/rust_work/*.manifest.json` | summed `kept_idx` and `attempts_consumed` per board |
| audit-set instance counts | 1500 distinct instances at every rung checked | `SV/scaling/data/g{17,40,64}r4/backward_audit.rust.jsonl` | counted distinct `(env_id, target, target_robot)` |
| audit-set record counts | 13,221 / 11,821 / 11,394 / 11,507 / 11,321 | `SV/scaling/data/g{17,40,48,56,64}r4/backward_audit.rust.jsonl` | counted lines |
| exact labelling wall time 40/48/56/64 | 31 / 61 / 111 / 188 s | `runs/nnlab/ladder_coarse_46204{85,86,87,88}.out` | grepped `COARSE ... exact ref` lines |
| lean board build 40/48/56/64 | 20 / 31 / 45 / 63 s | same logs | grepped `COARSE ... boards` lines |
| exact labelling 24 / 32 timing | 0.030 / 0.075 s per instance | `SV/FINDINGS.md` item 58 | **prose only — the primary timing file is a session scratchpad no longer on disk**; the run's audit trail exists at `SV/scaling/data/g{24,32}r4/rust_work/exact_g{24,32}r4_b0_19.*` |

### The no-network control

| number | value | file | how read |
|---|---|---|---|
| base 450: solved / rate | 394 / 0.87556 | `SV/eval/results/final450_backward_heuristic_baseline.json` | parsed rows, counted `solved` |
| base 450: mean expansions / mean s / total s | 21.50 / 0.0372 / 16.7 | same | computed over all rows |
| base 450: mean regret / percent optimal | 1.7563 / 58.38 % | same | computed over solved rows with a known d\* |
| g16r6 graded: solved / rate | 276 / 0.87342 | `SV/scaling/results/g16r6/comparison_heuristic_baseline.json` | same method |
| g16r6 graded: mean expansions / mean s | 163.84 / 0.5630 | same | computed |
| g16r6 graded: mean regret / percent optimal | 2.1014 / 56.88 % | same | computed |
| g16r6 frontier: solved / rate | 77 / 0.57463 | `SV/scaling/results/g16r6/comparison_ungraded_heuristic_baseline.json` | same method |
| g16r6 frontier: mean expansions / mean s | 576.83 / 1.5578 | same | computed |
| g16r6 frontier stored regret is meaningless | 16.19 and 0.0 % optimal on a `d_star = 0` set | same | computed from the stored rows and cross-checked against the placeholder rule |
| protocol for all three | 1200 expansions, `k = 5` | the same three files, `protocol` block | read JSON |
| confidence intervals and p-values | +8.0, +9.5, +23.1, +13.6 points | `SV/FINDINGS.md` item 49 | read prose; cell-level source named there as `SV/eval/results/stats_tests.json` |

### The ceiling probes

| number | value | file | how read |
|---|---|---|---|
| new probe, all summaries and caps | see the tables in D.4 | `SPR/results/ceiling/*.json` | parsed the `summary` and `caps` blocks of all 17 files |
| per-instance probe timings | see the timing table in D.4 | same files | `wall_seconds` divided by `summary.n`, and per-row `seconds` statistics |
| `n_best_below_dstar` = 0 everywhere | 0 on every graded arm | same files | read the field in each summary |
| slack sensitivity | best regret 1.4191 -> 1.3971 -> 1.3873 at g16r4; 1.7222 -> 1.6852 -> 1.6296 at g24r4 | `SPR/results/ceiling/g{16,24}r4_base{,_slack4,_slack12}.json` | read the summaries |
| old probe, base | 22 / 20 / 7 over 49 instances | `SV/analysis/artifacts/ceiling_probe_results.json` | parsed, counted by category |
| old probe, B1 | 38 / 7 / 4 over the same 49 | `SV/analysis/artifacts/ceiling_probe_results_b1.json` | same |
| old probe, B2 | 9 realizable, 2 inconclusive over 11 | `SV/analysis/artifacts/ceiling_probe_results_b2.json` | same |
| B2 deep re-probe | 2 instances, both inconclusive, 1202 s and 1800 s | `SV/analysis/artifacts/ceiling_probe_results_b2_deep.json` | parsed |
| g16r6 base probe | 48 / 23 / 28 / 6 over 105 | `SV/scaling/results/g16r6/ceiling_probe_old_vocab.json` | parsed, counted by category |
| g16r6 B1 probe | 63 / 2 / 12 over 77 | `SV/scaling/results/g16r6/ceiling_probe_b1.json` | same |
| g16r6 B2 probe | 5 / 9 over 14 | `SV/scaling/results/g16r6/ceiling_probe_b2.json` | same |
| g16r6 probe timings | mean 7.46 / 111.05 / 398.48 s per instance | the three files above | computed from per-row `seconds` |
| cap sensitivity, base | 22 of 22 stay structural | `SV/analysis/artifacts/cap_sensitivity_base.json` | parsed, counted by category |
| cap sensitivity, 6 robots | 48 of 48 stay structural | `SV/scaling/results/g16r6/cap_sensitivity.json` | same |
| `ceiling.py` caps and flags | 200,000 / 400,000 / 60 s / park 2 | `SPR/spr/ceiling.py` argument parser | read source |
| `ceiling.py` loop and categories | quoted in D.4 | `SPR/spr/ceiling.py:91-131` | read source |

### Label-fidelity gates

| number | value | file | how read |
|---|---|---|---|
| argmin agreement 24 / 32 | 0.92633 / 0.91518 | `SV/nn_labeler/results/capgate_g{24,32}r4.json` | read `summary.argmin_agreement` |
| argmin agreement 40 / 48 / 56 / 64 | 0.90741 / 0.90590 / 0.89599 / 0.90058 | `SV/nn_labeler/results/coarsegate_g{40,48,56,64}r4.json` | same |
| mean label gap 24 to 64 | 0.566 / 0.553 / 0.616 / 0.618 / 0.655 / 0.721 | the same six files | read `summary.gap_mean` |
| share of exactly-optimal labels | 85.2 % to 87.7 % | same | read `summary.share_gap_zero` |
| depth-0 coverage | 1.0 at every rung | same | read `summary.depth0_coverage` |
