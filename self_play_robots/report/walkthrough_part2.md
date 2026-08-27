# Part 2 — The two learned planners, and what the plan-language extensions bought

**Scope.** Part 1 defined the puzzle, the benchmark sets, the metrics and the
non-learned solvers. Part 3 covers self-play. This part sits between them. It
describes the two learned planning approaches precisely, reports how each scores
on the setups Part 1 defined, and measures what the two extensions to the plan
language actually delivered.

**Three results from Part 1 are used here and not repeated.** First, a plan
counts as solved only after it is replayed on the board under full physics.
Second, quality metrics (regret, percent optimal) exist only where the exact
optimum exists: the exact solver grades 51.6% of instances at 24×24, 38.9% at
32×32 and 35.8% at 24×24 with 8 robots. Third, the base subgoal language has an
exhaustively measured ceiling that training cannot move.

**Every number in this document was read from a named file during this session.**
Appendix A is the provenance table. Appendix B lists what could not be verified
and where the project's own documents disagree with the data.

---

## 0. Terms and notation

| Term | Meaning |
|---|---|
| **Forward planner** | The move-by-move planner. It chooses one robot slide at a time. Code: `supervised_valuenet/move_planner/`. |
| **Backward planner** | The subgoal planner. It chooses one subgoal at a time. Code: `supervised_valuenet/skeleton/`. |
| **Slide** | One primitive game move. A robot slides until a wall or another robot stops it. |
| **d\*** | The exact move-optimal solution length, produced by the exact solver at benchmark-build time only. |
| **Regret** | Realized moves minus d\*. It is defined only where d\* is known. |
| **Graded set** | The instances of a 450-puzzle pool that the exact solver could solve. Files named `bench.solved.jsonl`. |
| **Frontier set** | The instances of the same pool that the exact solver could **not** solve. Files named `bench.unsolved.jsonl`. No d\* exists there. |
| **Configuration** | `g<grid>r<robots>`. `g24r4` means a 24×24 board with 4 robots. |
| **Expansion (search step)** | One popped search node whose children are generated. In both planners this costs one policy pass plus one batched value pass over at most *k* children. |
| **Budget** | The cap on expansions per puzzle. Every row in this document uses 1200 expansions and *k* = 5. |
| **Realization** | Turning a subgoal plan into a legal move sequence and replaying it. Only the backward planner needs this step. |
| **Base vocabulary** | The original subgoal plan language. |
| **B1** | The first language extension: transient supports plus step-aside parks. |
| **B2** | The second language extension: supports-by-reference plus generalized parks. |

The six configurations and their graded/frontier splits are below. Every pool
holds 450 puzzles. The five scaling rows come from the `bench.jsonl.meta.json`
file of each configuration. The base row has no such field, so I derived it by
reading all 450 instances of `eval/data/bench450.jsonl` and confirming that every
one carries a positive `d_star` (minimum 1, maximum 12).

| Configuration | Board | Robots | Graded | Frontier | Exact-solver failure rate |
|---|---|---|---|---|---|
| g16r4 (base) | 16×16 | 4 | 450 | 0 | 0.0% |
| g16r6 | 16×16 | 6 | 316 | 134 | 29.8% |
| g16r8 | 16×16 | 8 | 266 | 184 | 40.9% |
| g24r4 | 24×24 | 4 | 232 | 218 | 48.4% |
| g24r8 | 24×24 | 8 | 161 | 289 | 64.2% |
| g32r4 | 32×32 | 4 | 175 | 275 | 61.1% |

---

## A. The two action spaces

### A.1 Move-by-move — the forward planner

Source: `supervised_valuenet/move_planner/{state,evaluate,net,encode}.py` and
`supervised_valuenet/eval/compare.py::run_forward`.

**State.** The joint position of every robot, plus the index of the target robot
and the goal cell. Formally a tuple of `(x, y)` cells in a fixed colour order.
The state is Markov. The move history can be forgotten.

**Action.** A pair `(robot_slot, direction)` with four directions. The action set
therefore has size `robots × 4`. A slide that cannot leave its cell is a no-op.
`state.py::legal_moves` drops those, so the real branching factor is smaller.

**Transition.** The chosen robot slides until a wall or another robot stops it.
Every other robot acts as a blocker. This is the true multi-robot physics, not a
relaxation.

**Goal test.** The target robot stands on the goal cell.

**Cost.** Every move costs 1.

**Branching factor (measured this session).** I counted the legal moves at the
start position of each benchmark instance.

| Configuration | Instances sampled | Mean legal moves | Median | Max |
|---|---|---|---|---|
| g16r4 | 60 | 13.17 | 13.0 | 16 |
| g24r4 | 40 | 13.72 | 14.0 | 16 |
| g24r8 | 30 | 26.80 | 27.0 | 30 |

Branching grows with the robot count and not with the board size. The ceiling is
`4 × robots`.

**Solution depth.** The depth equals the number of moves in the solution. On the
graded sets the forward planner's mean solution length runs from 5.27 moves
(g16r8) to 7.57 moves (g24r4). On the frontier sets, where no optimum is known,
its solved instances average 8.5 to 11.9 moves.

**The networks.** One network, `MoveNet`, with a shared encoder and two heads.
The encoder is a weight-tied looped transformer over the board cells plus one
global token. Its input channels are per-robot occupancy planes, a target-robot
plane, a goal plane, and eight slide-displacement planes that encode the board's
one-step dynamics.

- The **value head** predicts the exact optimal cost-to-go from the state. It is
  a classifier over cost bins and the value is read as the expected bin.
- The **policy head** predicts which `(robot, direction)` to play. It reads each
  robot's cell embedding together with the embedding of the cell that robot
  would slide to. Illegal actions are masked.

**The search.** `move_planner/evaluate.py::nn_astar` is a best-first search over
board states. The priority is `f = g + value(child)`, where `g` is the number of
moves so far. At each expansion the policy orders the legal successors and the
search keeps the top *k*. A `best_g` table prevents re-expansion at a worse cost.

**Stopping rule.** The search returns as soon as it pops a goal state. Because
the transition model is the real game, any returned path is playable by
construction. There is no realization step and no realization failure.

**How a step is counted.** `run_forward` wraps the network in a counter.
`nn_astar` makes one root call and then exactly two calls per expansion that has
successors. The recorded expansion count is `(calls - 1) // 2`.

### A.2 Subgoals — the backward planner

Source: `supervised_valuenet/skeleton/{astar,heuristics}.py`,
`supervised_valuenet/partial_plan.py`, `supervised_valuenet/GridEnv.py`,
`supervised_valuenet/eval/realize.py` and
`supervised_valuenet/eval/compare.py::_nn_astar_backward`.

**What one subgoal is.** A subgoal is a triple:

1. a **bottleneck** cell that the moving robot must reach,
2. a **support** cell that a helper robot must occupy, and
3. the **helper** robot chosen to occupy it.

Read as an English sentence: *"park helper H on cell S, so that the moving robot
stops on cell B."* A robot cannot stop mid-slide, so it needs either a wall or
another robot to stop it. The support robot is that stopper.

Each candidate also carries a **parent support**: the cell whose occupant lets
the bottleneck reach the segment's endpoint. This field records which stopper the
plan relies on for the next leg.

**Worked example (benchmark index 0, board 2400).** Read from
`eval/results/plan_structures_data.json`. Red starts at (1,15) and must reach
(9,15). The plan has two subgoals.

- Subgoal 1: bottleneck (9,15) for Red, support (10,15) held by Blue. Red slides
  right and Blue stops it exactly on the goal. Cost of Red's leg: 7 moves.
- Subgoal 2: Blue cannot reach (10,15) on its own. So bottleneck (10,15) for
  Blue, support (11,15) held by Green. Cost of Blue's leg: 2. Cost of Green's
  leg: 3.

Total plan cost 12, which equals both the realized move count and d\* = 12.

**What a partial plan is.** A directed acyclic graph, `partial_plan.PartialPlan`.
Node types are `goal`, `subgoal`, `bottleneck`, `support`, `leaf` and `park`. A
`leaf` is a robot at its current cell. Edges point from goal towards leaves. The
robot physically travels in the opposite direction.

Every edge carries a status and a cost:

- **fixed** — an exact shortest path exists for this leg. The cost is exact.
- **open** — no exact path exists yet. The cost is an optimistic relaxed
  estimate, and the leg still needs a subgoal.

A plan is **complete** when it has no open edges. Its cost is the sum of the edge
costs. Because fixed edges carry exact costs and open edges carry optimistic
estimates, the plan cost is automatically `g + h`.

**The search.** `skeleton/astar.py::AStar._search` pops the cheapest plan. If the
plan is complete, it is returned. Otherwise the search resolves the first open
edge. It first tries to pin that edge to an exact path. If that succeeds, the
edge becomes fixed and no choice was made. If it fails, `heuristics.propose`
generates candidate subgoals and each candidate spawns one child plan through
`_apply`.

`_apply` enforces two invariants that matter later:

- One physical robot owns one leaf. A robot already recruited cannot be
  recruited again at its original cell. Otherwise the plan would schedule one
  robot in two places at once.
- A supported route cost may be claimed only when some plan node actually places
  a robot on the stopper cell.

**Realization.** `eval/realize.py` converts a complete plan into moves. Two
counts exist.

- `abstract_moves` costs each leg on its own, with only its intended support
  robot present. This mirrors the plan's own cost model. It can under-count.
- `strict_moves` performs a legal joint-game execution. The plan's legs are
  ordered topologically. A helper reaches its support cell before the mover
  bounces off it. A support robot departs only after its bounce is consumed.
  Each leg then runs on the full board with every other robot as a blocker.

**Solved means `strict_moves` returned a count.** If the schedule cannot be
played, the instance is not solved, however good the plan looked on paper.

**Branching factor (measured this session).** I counted the candidates that
`propose` offers at the first open segment of each benchmark instance.

| Configuration | Instances with an open root segment | Base vocabulary: mean / median / max | B1 vocabulary: mean / median / max |
|---|---|---|---|
| g16r4 | 52 of 60 | 146.9 / 15.0 / 792 | 257.9 / 30.0 / 1701 |
| g24r4 | 38 of 40 | 215.4 / 25.5 / 1017 | 523.5 / 57.0 / 2421 |
| g24r8 | 27 of 30 | 698.2 / 84.0 / 4557 | 1986.2 / 210.0 / 11613 |

Two things follow. The subgoal branching factor is one to two orders of magnitude
larger than the forward one. It also grows with the board area and with the robot
count, because the candidate set is (bottleneck, support) pairs times helpers.

A second measurement confirms this inside a real search rather than at the root.
`analysis/artifacts/byref_topk_ablation.json` recorded 4,577,425 candidate
applications across 46,351 expansions on the g16r6 frontier set. That is **98.76
candidates per expansion** under the B1 vocabulary.

**Solution depth (measured this session).** I ran the hand-coded exhaustive
search with no networks and counted the subgoals in the first strictly realizable
plan it found.

| Configuration | Instances | Plans realized | Subgoals per plan: mean / median / max | Distribution | Plan DAG nodes (mean) | Realized moves (mean) |
|---|---|---|---|---|---|---|
| g16r4 | 80 | 72 | 0.92 / 1 / 3 | 0 subgoals: 23, 1: 37, 2: 7, 3: 5 | 5.67 | 8.60 |
| g24r4 | 60 | 56 | 0.82 / 1 / 3 | 0 subgoals: 22, 1: 25, 2: 6, 3: 3 | 5.29 | 9.70 |

The depth is one to three subgoals and it does not grow with the board. About
one third of instances need no subgoal at all: the target robot reaches the goal
along an exact path. So the subgoal search is shallow and very wide, and the
move search is narrow and deep. That single sentence explains most of the cost
results in section B.

**The networks.** Two separate networks share the same looped-transformer
encoder family as the forward net.

- The **proposal net** (`train/policy_tf.py::PolicyTF`) scores candidate
  subgoals. It decodes bottleneck, then support, then helper, as a three-step
  pointer head over board cells. It is trained on cost-weighted soft targets, so
  it learns which candidate leads to the cheapest completion.
- The **value net** (`train/looped_pc.py::LoopedValueNet`) predicts the cost to
  complete a partial plan, given the board, the open segment and the candidate.
  It is a classifier over cost bins, read as an expected bin.

At each expansion the driver builds one record per surviving candidate, ranks
them with one proposal-net pass, keeps the top *k*, and scores those *k* with one
batched value-net pass. Children enter the frontier at
`fixed_g(child) + value(child)`.

**How a step is counted.** `_nn_astar_backward` counts an expansion only when a
popped plan reaches the proposal step. Two kinds of pop are free and are not
counted:

- a pop whose first open edge can be pinned to an exact path (no choice is made),
- the final pop of a complete plan.

This is documented in `eval/compare.py` and it is exactly the same accounting
rule as the forward side: one expansion equals one policy pass plus one batched
value pass.

**Stopping rules.** The backward driver supports three modes, and they are not
equivalent. This matters for section B and for section C.

| Mode | Flag | Behaviour |
|---|---|---|
| Plain | none | Return the first complete plan. Realization is checked once, afterwards. If it fails, the instance is unsolved even though budget remained. |
| Prefix-check | `--backward-prefix-check` | Before a child plan enters the frontier, its already-fixed legs are replayed. Children whose prefix cannot be played are discarded. Still stops at the first complete plan. |
| Anytime | `--backward-anytime` | A complete plan that fails realization is discarded and the search continues inside the same budget. The first plan that plays out is returned. |

The forward planner has only one behaviour, and it matches the anytime idea by
construction: its first solution is always playable.

### A.3 Side by side

| Property | Forward (move-by-move) | Backward (subgoals) |
|---|---|---|
| Search node | a board state | a partial plan (a DAG) |
| Action | one robot slide | one subgoal |
| Branching (measured, 24×24, 4 robots) | 13.7 mean | 215 mean, base vocabulary |
| Depth (measured, 24×24, 4 robots) | 7.6 moves mean | 0.8 subgoals mean, max 3 |
| Physics used at search time | exact multi-robot slides | precomputed slide graph and distance tables |
| Value target | exact optimal moves to goal | cost to complete the partial plan |
| Policy target | best `(robot, direction)` | best `(bottleneck, support, helper)` |
| Result is playable? | always | only after `strict_moves` succeeds |
| Extra work not counted as an expansion | none | realization checks, prefix checks, park repairs, exact-fix pops, per-board table build |

---

## B. How each planner scores

### B.0 The protocol behind every table

All rows use the same protocol. Both planners run on the same pinned instance
file, verified by a SHA-256 hash recorded in each result file. The budget is 1200
expansions and the proposal width is *k* = 5. Both planners run on CPU. Quality
is measured in primitive moves against d\*, and the exact solver is used only to
build that label, never at inference time.

Two reading rules apply throughout.

1. **On frontier sets, regret and percent optimal do not exist.** The instance
   files carry a placeholder d\*. Newer result files detect this and suppress the
   derived fields. Older files do not, and they publish a `mean_regret` that is
   in fact the mean solution length. Section B.5 names the affected files.
2. **Wall-clock seconds compare only within one machine.** The study moved to the
   Karolina cluster on 2026-07-20. Files dated before that ran on a different
   machine. Solve counts, move counts and expansion counts are machine
   independent. Seconds are not.

### B.1 16×16 with 4 robots — the base configuration (450 puzzles, all graded)

| System | Solved | Solve rate | Mean regret | % optimal | Mean moves | Mean expansions | Mean s | Median s | p90 s |
|---|---|---|---|---|---|---|---|---|---|
| Forward (`candidate_scored.ckpt`) | 450/450 | 100.0% | 0.067 | 94.2% | 6.436 | 36.0 | 0.986 | 0.38 | 1.97 |
| Backward, base vocabulary, anytime | 401/450 | 89.1% | 2.147 | 50.4% | 8.387 | 7.6 | 1.094 | 0.35 | 3.07 |
| Backward, base vocabulary, prefix-check | 401/450 | 89.1% | 2.145 | 50.4% | 8.384 | 7.1 | 1.070 | 0.36 | 2.99 |
| Backward, B1 | 429/450 | 95.3% | 2.028 | 53.1% | 8.296 | 9.7 | 1.228 | 0.38 | 3.19 |
| Backward, B2 (B1-trained nets) | 430/450 | 95.6% | 2.040 | 53.0% | 8.307 | 9.7 | 0.550 | 0.18 | 1.44 |
| Backward, B2, retrained on cap-5,000 labels | 433/450 | 96.2% | 2.051 | 53.8% | 8.330 | 14.6 | 2.077 | 0.80 | 7.40 |
| Backward, B2, retrained on cap-20,000 labels | 429/450 | 95.3% | 2.261 | 53.6% | 8.538 | 15.3 | 2.347 | 0.98 | 7.47 |

The first four rows and the forward row ran on the origin machine. The last three
ran on Karolina. Their seconds are not comparable across that line.

**Reading.** The forward planner owns this configuration outright. It solves
everything, it is 30 times closer to the optimum, and it does so in about one
second per puzzle. The backward planner needs four to five times fewer search
steps, and that is its only advantage here. The gap is significant:
430/450 against 450/450 gives a difference of −4.44 points, 95% CI
[−6.22, −2.67], McNemar p = 1.9e-06.

### B.2 16×16 with 6 and 8 robots

Graded sets:

| Configuration | System | Solved | Solve rate | Mean regret | % optimal | Mean moves | Mean expansions | Mean s |
|---|---|---|---|---|---|---|---|---|
| g16r6 (316) | Forward control | 314 | 99.4% | 0.096 | 92.4% | 5.777 | 64.2 | 21.649 |
| g16r6 | Backward, base vocabulary, per-config nets | 275 | 87.0% | 2.222 | 50.9% | 7.865 | 72.9 | 11.697 |
| g16r6 | Backward, base vocabulary, base-B1 nets | 282 | 89.2% | 2.681 | 49.6% | 8.312 | 26.6 | 4.417 |
| g16r6 | Backward, B1 | 304 | 96.2% | 2.312 | 52.3% | 8.000 | 27.2 | 5.040 |
| g16r6 | Backward, B2 | 306 | 96.8% | 2.386 | 52.6% | 8.062 | 27.1 | 2.694 |
| g16r8 (266) | Forward control | 261 | 98.1% | 0.107 | 92.0% | 5.272 | 62.4 | 21.206 |
| g16r8 | Forward, best of nine retrainings | 266 | 100.0% | 0.098 | 91.4% | 5.286 | 44.9 | 16.027 |
| g16r8 | Backward, base vocabulary, per-config nets | 230 | 86.5% | 1.965 | 51.3% | 7.130 | 53.8 | 14.713 |
| g16r8 | Backward, base vocabulary, base-B1 nets | 232 | 87.2% | 2.060 | 51.3% | 7.224 | 27.9 | 6.097 |
| g16r8 | Backward, B2 | 262 | 98.5% | 2.550 | 50.4% | 7.740 | 8.6 | 1.750 |

Frontier sets. No optimum exists, so only solve rate, effort and time are
meaningful.

| Configuration | System | Solved | Solve rate | Mean moves | Mean expansions | Mean s | Median s |
|---|---|---|---|---|---|---|---|
| g16r6 (134) | Forward control | 65 | 48.5% | 9.785 | 835.1 | 280.986 | 326.22 |
| g16r6 | Backward, base vocabulary, per-config nets, prefix-check | 70 | 52.2% | 14.914 | 296.1 | 80.407 | 11.30 |
| g16r6 | Backward, base vocabulary, base-B1 nets, anytime | 80 | 59.7% | 16.587 | 153.5 | 17.722 | 0.88 |
| g16r6 | Backward, B1 | 108 | 80.6% | 16.852 | 269.5 | 41.979 | 4.58 |
| g16r6 | Backward, B2 | 108 | 80.6% | 16.824 | 269.1 | 19.553 | 2.25 |
| g16r8 (184) | Forward control | 93 | 50.5% | 8.828 | 818.8 | 256.604 | 334.35 |
| g16r8 | Forward, best of nine retrainings | 101 | 54.9% | 8.812 | 745.4 | 288.593 | 304.62 |
| g16r8 | Backward, base vocabulary, per-config nets, prefix-check | 88 | 47.8% | 13.227 | 264.3 | 87.642 | 1.87 |
| g16r8 | Backward, base vocabulary, base-B1 nets, anytime | 95 | 51.6% | 14.958 | 176.9 | 38.514 | 0.38 |
| g16r8 | Backward, B2 | 163 | 88.6% | 17.810 | 182.8 | 27.041 | 1.84 |

**One caution about the g16r8 forward row in `comparison.json`.** That file
records a forward arm that solves 25/266 = 9.4%. The project's own report
generator marks it as withheld and never renders it as a comparable result. Its
training run collapsed. The control of record is
`comparison_forward_control.json` at 261/266. The best-of-nine retraining is
`comparison_forward_rescue.json` at 266/266. I use the control and the rescue,
never the collapsed row.

### B.3 The big boards — 24×24 and 32×32

Graded sets:

| Configuration | System | Solved | Solve rate | Mean regret | % optimal | Mean moves | Mean expansions | Mean s | Median s |
|---|---|---|---|---|---|---|---|---|---|
| g24r4 (232) | Forward | 220 | 94.8% | 0.068 | 94.1% | 7.568 | 191.0 | 274.945 | 65.94 |
| g24r4 | Backward, base vocabulary, prefix-check | 205 | 88.4% | 4.220 | 38.5% | 11.810 | 26.9 | 7.803 | 2.90 |
| g24r4 | Backward, B2, anytime | 199 | 85.8% | 4.889 | 36.7% | 12.327 | 53.5 | 23.448 | 21.95 |
| g24r8 (161) | Forward | 157 | 97.5% | 0.166 | 85.4% | 5.758 | 151.6 | 183.505 | 59.49 |
| g24r8 | Backward, base vocabulary, prefix-check | 144 | 89.4% | 2.486 | 50.7% | 7.993 | 44.0 | 34.056 | 1.18 |
| g24r8 | Backward, B2, anytime | 148 | 91.9% | 2.345 | 46.6% | 7.892 | 94.9 | 65.897 | 1.21 |
| g32r4 (175) | Forward | 133 | 76.0% | 0.135 | 88.7% | 7.338 | 469.2 | 1378.611 | 655.95 |
| g32r4 | Backward, base vocabulary, prefix-check | 147 | 84.0% | 2.367 | 51.7% | 9.980 | 13.8 | 10.222 | 2.06 |
| g32r4 | Backward, B2, anytime | 154 | 88.0% | 2.844 | 47.4% | 10.519 | 34.5 | 30.606 | 8.80 |

Frontier sets:

| Configuration | System | Solved | Solve rate | Mean moves | Mean expansions | Mean s | Median s | Budget exhausted |
|---|---|---|---|---|---|---|---|---|
| g24r4 (218) | Forward | 15 | 6.9% | 11.867 | 1162.9 | 2450.601 | 2561.93 | 203/218 = 93.1% |
| g24r4 | Backward, B2, anytime | 125 | 57.3% | 22.536 | 110.2 | 49.377 | 54.04 | 0/218 |
| g24r8 (289) | Forward | 44 | 15.2% | 8.545 | 1100.8 | 1166.196 | 1227.57 | 245/289 = 84.8% |
| g24r8 | Backward, base vocabulary, prefix-check | 154 | 53.3% | 15.675 | 260.1 | 200.967 | 4.25 | 47/289 = 16.3% |
| g24r8 | Backward, B2, anytime | 161 | 55.7% | 15.522 | 635.4 | 431.277 | 423.76 | 128/289 = 44.3% |
| g32r4 (275) | Forward | 2 | 0.7% | 11.500 | 1198.5 | 2379.784 | 2377.04 | 273/275 = 99.3% |
| g32r4 | Backward, base vocabulary, prefix-check | 127 | 46.2% | 21.276 | 29.7 | 14.598 | 5.92 | 0/275 |
| g32r4 | Backward, B2, anytime | 196 | 71.3% | 24.587 | 96.1 | 57.761 | 59.37 | 0/275 |

No old-language frontier row was ever produced at g24r4. The forward frontier
cell for that configuration comes from the two-system file
`comparison_ungraded_b2.json`, and the project's report generator records that
wiring note in `eval/report_data.py`.

### B.4 Who wins what

The significance figures below come from `eval/results/stats_tests.json`. The
test is an exact two-sided McNemar test on discordant pairs. The confidence
intervals come from a percentile bootstrap over boards, because puzzles share a
board's wall layout. The pooled row is the union of graded and frontier — the
whole 450-puzzle pool at that configuration — and it needs no selection
argument.

| Configuration | Set | Backward (best language) | Forward | Difference | 95% CI | p |
|---|---|---|---|---|---|---|
| g16r4 | graded | 430/450 | 450/450 | −4.4 | [−6.2, −2.7] | 1.9e-06 |
| g16r6 | graded | 306/316 | 314/316 | −2.5 | [−4.8, −0.6] | 0.039 |
| g16r6 | frontier | 108/134 | 65/134 | **+32.1** | [+22.1, +42.3] | 7.1e-10 |
| g16r6 | pooled | 414/450 | 379/450 | **+7.8** | [+4.2, +11.6] | 1.6e-05 |
| g16r8 | graded | 262/266 | 261/266 | +0.4 | [−1.9, +2.7] | 1.00 |
| g16r8 | frontier | 163/184 | 93/184 | **+38.0** | [+30.2, +45.5] | 4.3e-17 |
| g16r8 | pooled | 425/450 | 354/450 | **+15.8** | [+11.6, +20.0] | 2.3e-15 |
| g24r4 | graded | 199/232 | 220/232 | **−9.1** | [−13.7, −4.4] | 0.00032 |
| g24r4 | frontier | 125/218 | 15/218 | **+50.5** | [+44.0, +57.1] | 4.4e-32 |
| g24r4 | pooled | 324/450 | 235/450 | **+19.8** | [+14.4, +25.1] | 3.9e-14 |
| g24r8 | graded | 148/161 | 157/161 | −5.6 | [−10.5, −1.2] | 0.035 |
| g24r8 | frontier | 161/289 | 44/289 | **+40.5** | [+34.6, +46.4] | 3.1e-30 |
| g24r8 | pooled | 309/450 | 201/450 | **+24.0** | [+19.6, +28.7] | 1.7e-21 |
| g32r4 | graded | 154/175 | 133/175 | **+12.0** | [+3.5, +20.4] | 0.0038 |
| g32r4 | frontier | 196/275 | 2/275 | **+70.6** | [+64.9, +76.1] | 8.0e-59 |
| g32r4 | pooled | 350/450 | 135/450 | **+47.8** | [+42.7, +52.7] | 3.0e-51 |

**Stated plainly:**

- **Solution quality is the forward planner's, everywhere it solves.** Its mean
  regret runs from 0.067 to 0.166 moves and it reaches the optimum on 85% to 94%
  of graded instances. The backward planner's regret runs from 1.97 to 4.89 moves
  and it reaches the optimum on 37% to 54%. No configuration reverses this. This
  gap is not a training artifact. Section C.4 shows that a large part of it is
  the language itself: the base language's own move floor is 1.26 at 16×16 and
  1.62 at 24×24, against a measured planner regret of 2.15 and 4.22.
- **Solve rate on graded sets belongs to the forward planner up to 24×24.** It
  wins the base configuration, g16r6, g24r4 and g24r8. At g16r8 it ties against
  its control of record (261 against 262, p = 1.00) and wins outright with its
  best-of-nine retraining (266/266). It loses only at 32×32, by 12.0 points.
- **Solve rate on frontier sets belongs to the backward planner everywhere it was
  measured with its full language.** The margins are +32.1, +38.0, +50.5, +40.5
  and +70.6 points.
- **On the whole 450-puzzle pool the backward planner wins every configuration
  except the base one**, and the margin grows monotonically along both hardness
  axes: +7.8, +15.8, +19.8, +24.0, +47.8.
- **Search effort favours the backward planner at every configuration.** The
  ratio of mean expansions, forward over backward with its best language, is
  3.7× at g16r4, 2.4× and 3.1× at g16r6, 7.3× and 4.5× at g16r8, 3.6× and 10.6×
  at g24r4, 1.6× and 1.7× at g24r8, and 13.6× and 12.5× at g32r4 (graded and
  frontier in each pair).

**The scale trend, in one sentence.** As boards grow, the forward planner's
search cost grows faster than its budget and it stops solving, while the backward
planner's cost grows slowly, so the subgoal margin widens.

### B.5 Fairness — which comparisons hold and which do not

This section answers the fairness question directly. Four issues exist. Three
favour the forward planner and one favours the backward planner.

**Issue 1 — the stopping rules are not the same, and they are not even the same
across backward arms.** I extracted the exact search flags from the `protocol.command`
field of every result file.

| Result file group | Search variant used |
|---|---|
| Every `comparison.json` / `comparison_ungraded.json` at g16r8, g24r4, g24r8, g32r4, and `comparison_ungraded.json` at g16r6, and `final450_backward_prefix.json` | `--backward-prefix-check` |
| `comparison.json` at g16r6, `final450_backward_anytime.json`, every `*basenets_oldvocab*` file | `--backward-anytime` |
| Every `*_b1.json` and `*_b2.json` file | `--backward-anytime` plus `--backward-b1` or `--backward-b2` |

So the old-language arm and the extended-language arm differ in the stopping rule
at four of six configurations. The forward planner stops at its first goal state
in every row.

Direction of the effect: the anytime rule can only help solve rate, because it
retries after a realization failure inside the same budget. At the base
configuration the two variants give exactly the same solve count, 401/450, so the
confound is empirically null there. At the other configurations it is not
measured. **This matters for section C.5, where the anytime arm still lost.**

**Issue 2 — the shared expansion cap gives the backward planner more work per
step.** `eval/compare.py` states this in the generated protocol note: one
backward expansion commits a whole subgoal, which is worth several primitive
moves, while one forward expansion commits a single move. The matched budget is
therefore generous to the backward system. It is the conservative direction for
the headline comparison, and I record it as such.

**Issue 3 — real work is performed that no expansion counter records.** All of it
is on the backward side.

| Unmetered work | Where it happens |
|---|---|
| Realization checks | one `strict_moves` call per complete plan popped |
| Prefix checks | one replay per candidate child, in the prefix-check arms |
| Park repairs | a blocked-slide BFS over every robot and every step-aside cell |
| Exact-fix pops | a pop that pins an edge without proposing is not counted |
| Per-board tables | `GridEnv.from_env` builds and caches exact distance tables, and the timer starts after it |

The counters in the result files show the size of this. At g24r4 the base
prefix-check arm records 36.24 prefix-check calls per instance against 8.46
policy passes (`comparison_exactseed21.json`, which shares that protocol). The
B2 arm records 1.95 realization calls and 1.09 park-repair calls per instance
against 14.29 policy passes (`comparison_b2.json`).

The project measured this in a common unit. `simulate.slide` is the one physics
primitive both stacks reach. On the same five base-configuration puzzles
(`eval/results/instrumentation_ab/v3_split_{backward5,forward5}.json`):

| System | Physics slides, median | All slides including featurization, median | Expansions, mean |
|---|---|---|---|
| Backward, full language, anytime | 64 (`strict_realize`) | 983 | 3.4 |
| Forward (`best.ckpt`) | 5,776 (`forward_search`) | 40,448 | 337.0 |

The backward search itself performs **zero** slide calls, because it plans over
precomputed graph and distance tables. The physics-only ratio is about 90×
against an expansion ratio of about 99×. So the efficiency advantage survives a
matched physics unit essentially unchanged. It neither widens nor shrinks.

**Issue 4 — wall-clock is comparable only inside one file.** Within any one
result file the forward and backward rows ran in the same job on the same
machine, so their seconds are directly comparable. Across files they often are
not. I verified the size of the machine effect directly. The g24r4 forward rows
in `comparison.json` (2026-07-14, origin) and `comparison_b2.json` (2026-07-29,
Karolina) are **identical in all 232 rows** on solved status, move count and
expansion count. Only the timings differ: median 65.94 s against 87.62 s, a
factor of 1.33. Any cross-file seconds ratio at that configuration carries that
factor.

The same-machine cross-file pairs that do exist are listed here, because section
C uses them.

| Comparison | Same machine? | Old language | Extended language |
|---|---|---|---|
| g16r6 graded, fixed nets | yes, both Karolina | 4.417 s / 26.6 exp | 2.694 s / 27.1 exp |
| g16r6 frontier, fixed nets | yes, both Karolina | 17.722 s / 153.5 exp | 19.553 s / 269.1 exp |
| g16r8 graded, fixed nets | yes, both Karolina | 6.097 s / 27.9 exp | 1.750 s / 8.6 exp |
| g16r8 frontier, fixed nets | yes, both Karolina | 38.514 s / 176.9 exp | 27.041 s / 182.8 exp |
| g32r4 graded | yes, both Karolina | 10.222 s / 13.8 exp | 30.606 s / 34.5 exp |
| g32r4 frontier | yes, both Karolina | 14.598 s / 29.7 exp | 57.761 s / 96.1 exp |
| g24r8 frontier | yes, both Karolina | 200.967 s / 260.1 exp | 431.277 s / 635.4 exp |
| g24r4 graded | **no** (origin against Karolina) | 7.803 s / 26.9 exp | 23.448 s / 53.5 exp |
| g24r8 graded | **no** (origin against Karolina) | 34.056 s / 44.0 exp | 65.897 s / 94.9 exp |
| base 450, B1 against B2 | **no** (origin against Karolina) | 1.228 s / 9.7 exp | 0.550 s / 9.7 exp |

**Files that publish a misleading `mean_regret`.** These frontier files predate
the placeholder sentinel. Their `mean_regret` equals their `mean_moves` exactly,
because the placeholder d\* is 0. The number is a solution length, not a regret.

- `scaling/results/g16r6/comparison_ungraded.json` (14.914)
- `scaling/results/g16r6/comparison_ungraded_b1.json` (16.852)
- `scaling/results/g16r6/comparison_ungraded_b2.json` (16.824)
- `scaling/results/g16r8/comparison_ungraded.json` (13.227)
- `scaling/results/g16r8/comparison_ungraded_b2.json` (17.810)
- `scaling/results/g24r8/comparison_ungraded.json` (15.675)
- `scaling/results/g24r8/comparison_ungraded_b2.json` (15.522)
- `scaling/results/g32r4/comparison_ungraded.json` (21.276)
- `scaling/results/g32r4/comparison_ungraded_b2.json` (24.587)

The newer files at g24r4 and the `basenets_oldvocab` files record
`d_star_placeholder: true` and suppress the field correctly. No table in this
document quotes a frontier regret.

**The cost trend.** Median seconds per puzzle. Every pair below is same-machine.
Where a same-machine extended-language row exists, both backward arms are shown,
because the choice of arm changes the ratio a great deal.

| Configuration and set | Forward, median s | Backward, base vocabulary | Ratio | Backward, extended language | Ratio |
|---|---|---|---|---|---|
| g16r4 (450) | 0.38 | 0.35 (anytime) | 1.1× | 0.38 (B1) | 1.0× |
| g16r6 graded | 5.38 | 0.51 | 10.5× | 0.39 (B1) | 13.8× |
| g16r6 frontier | 326.22 | 11.30 | 28.9× | 4.58 (B1) | 71.2× |
| g16r8 graded | 5.55 | 0.43 | 12.9× | not same machine | — |
| g16r8 frontier | 334.35 | 1.87 | 178.8× | not same machine | — |
| g24r4 graded | 65.94 | 2.90 | 22.7× | not same machine | — |
| g24r4 frontier | 2561.93 | never measured | — | 54.04 (B2) | 47.4× |
| g24r8 graded | 59.49 | 1.18 | 50.4× | not same machine | — |
| g24r8 frontier | 1227.57 | 4.25 | 288.8× | 423.76 (B2) | **2.9×** |
| g32r4 graded | 655.95 | 2.06 | 318.4× | 8.80 (B2) | 74.5× |
| g32r4 frontier | 2377.04 | 5.92 | 401.5× | 59.37 (B2) | 40.0× |

Four of these pairs come from two different result files that ran on the same
machine rather than from one file: g16r4, g16r6 graded, g16r8 graded and g16r8
frontier. The other seven pairs come from a single file each.

The last column carries a warning that section C will develop. At the 24×24
8-robot frontier the extended language reduces the backward planner's wall-clock
advantage from 289× to 2.9×, because it spends 635 expansions where the base
language spends 260.

Two honest qualifications. The base configuration is the one place where the
forward planner is not slower. And the backward planner's cost distribution is
heavy tailed. At the g16r8 frontier its p90 is 384.35 s against a median of
1.87 s, so at the p90 the margin against forward's 392.61 s all but disappears.
Median and tail must always be printed together.

### B.6 What the networks contribute

One control isolates the learned ranking from the formulation. The production
evaluation loop was run with only the two neural scoring points replaced by the
labeler's hand-written scorer, at the same budget, the same candidate pool and
the same play-out check.

| Set | With networks | Hand-written scorer | Difference |
|---|---|---|---|
| base 450 | 430 = 95.6% | 394 = 87.6% | +8.0 points |
| g16r6 graded (316) | 306 = 96.8% | 276 = 87.3% | +9.5 points |
| g16r6 frontier (134) | 108 = 80.6% | 77 = 57.5% | +23.1 points |

I verified all three hand-written rows directly: 394/450 at 21.50 mean
expansions, 276/316 at 163.84, and 77/134 at 576.83. The three network rows are
the B2 rows of section B.1 and B.2. The confidence intervals are quoted from
`supervised_valuenet/FINDINGS.md` §49 and I did not re-derive them.

The reading is that both ingredients are needed. The subgoal formulation alone
already beats the trained forward planner on the g16r6 frontier. The learned
ranking adds most where it matters, on the hardest set.

---

## C. The plan-language extensions

This is the heart of Part 2. Two extensions were added to the subgoal
vocabulary. The project calls them B1 and B2. This section describes what each
one adds, what it costs, and what it measurably delivered.

**The single most important distinction in this section.** Two different
quantities are easy to confuse and must be kept apart.

- **Expressibility.** How many instances the language can describe at all. An
  exhaustive search with no networks measures this. It answers: *does a playable
  plan exist in this vocabulary?*
- **Achievement.** What a trained planner reaches at a fixed budget with that
  vocabulary available. A benchmark run measures this. It answers: *does the
  planner find such a plan in 1200 steps?*

The two extensions moved the first quantity a lot. They moved the second one
much less, and at one configuration they moved it the wrong way.

### C.1 The base vocabulary and the one line that limits it

A support cell is a cell where a helper robot must stand. The helper has to get
there first, and a robot can only stop at a wall or at another robot. The base
vocabulary handles this by admitting only support cells that already have an
adjacent wall. The test is one line in `GridEnv._collect_bottleneck_support_pairs`:

```
if self._has_adjacent_wall(support_pos):
    pairs.add((v, support_pos))
```

Everything else in the stack was already general. The board graph records a
dependent edge for every intermediate stop of every slide, whether the stopper
cell is wall-adjacent or not. The cost oracles read all of those edges. The plan
DAG already expresses support-for-support through its open-edge recursion. Only
the proposal step filtered.

So the base language cannot say: *"put the helper on a cell that only becomes
holdable because a second robot goes there first."*

### C.2 B1 — transient supports and step-aside parks

B1 adds two things. Both stay inside the subgoal formalism. Neither adds a raw
primitive-move escape hatch.

**B1a — transient supports.** `GridEnv.propose_subgoal_states(..., transient=True)`
additionally yields (bottleneck, support) pairs found by the same rule but with
the wall test inverted, so wall-less support cells are admitted. Each `Subgoal`
gains a `transient` flag. The helper's placement leg then has no exact path, so
it stays open and the existing recursion turns it into its own nested subgoal.
`skeleton/heuristics.propose_b1` is the drop-in that enables this.

*Concrete example — benchmark index 333, board 2511, d\* = 4.* Read from
`eval/results/plan_structures_data.json`. Yellow is the target at (6,1) and must
reach (3,1). Blue is at (4,9). Green is at (4,5).

- Root subgoal: bottleneck (3,1) for Yellow, support (4,1) held by Blue. The
  cell (4,1) has **no adjacent wall**. The base vocabulary discards this pair,
  and the exhaustive probe then builds zero complete plans for this puzzle.
- Blue cannot reach (4,1) alone. Nested subgoal: bottleneck (4,1) for Blue,
  support (4,0) held by Green. The cell (4,0) is at the border wall, so it is an
  ordinary base-vocabulary support.
- Plan cost 0 + 2 + 0 + 1 + 1 = 4. The strict realization plays 4 legal moves.
  Both equal d\* = 4.

**B1b — parks (step aside).** A plan may schedule one robot to slide to a parking
cell *before* one named plan leg runs, so it clears that leg's way. In the DAG a
park is a `park` node carrying a position, a robot and a `before_edge`, with one
fixed costed edge back to the node where the plan leaves that robot. The park's
moves are ordinary plan cost. There are no free vacates.

Parks are not proposed by the networks. They are produced from realization
feedback. When a complete plan fails the strict check, `skeleton.astar.park_repairs`
reads the first failing leg and the position snapshot, finds every robot whose
removal would unblock the mover, tries that robot's one-slide destinations, and
keeps the destinations under which the blocked leg becomes passable. Each
survivor re-enters the frontier as an ordinary costed plan.

*Concrete example — benchmark index 28, board 2409, d\* = 7.* Red must reach
(7,3) and the exact path costs 6. Blue at (10,3) blocks it. The plan carries one
park node: Blue slides to (10,0) at cost 1, ordered before the `goal → leaf_0`
leg. Plan cost 7. Realized moves 11, which is above d\* = 7 but legal.

**What B1 costs.**

- *Branching.* Measured this session at the root segment: the candidate pool
  grows from 146.9 to 257.9 at g16r4 (1.76×), from 215.4 to 523.5 at g24r4
  (2.43×), and from 698.2 to 1986.2 at g24r8 (2.85×). The median grows from 15 to
  30, from 25.5 to 57 and from 84 to 210.
- *Realization work.* Parks fire only after a complete plan fails, and each park
  proposal runs a blocked-slide BFS per robot and per destination. The project's
  own slide-count pilot identifies park repair as the physics cost centre: on a
  20-instance slice one unsolved puzzle spent 106,817 slide calls, 96% of them
  inside generalized park repair, which is 89.6% of the whole slice's physics
  work. This is a tail cost, not a mean cost.
- *Ranking.* Parks are never scored by the value net. They re-enter the frontier
  on raw abstract cost.

### C.3 B2 — supports by reference and generalized parks

B2 adds **no new candidate shape to the vocabulary of a single subgoal**. It
relaxes a bookkeeping invariant instead.

**B2a — supports by reference.** The base rule is one robot, one leaf. A robot
that the plan already commits cannot be recruited a second time at its original
cell, because that would schedule one robot in two places. B2 lets such a robot
serve **by reference**: the candidate helper stands at the robot's *planned*
cell, and the new subgoal wires to the existing plan node instead of recruiting a
second leaf. `skeleton.astar._reference_helpers` offers those phantom helpers and
`_apply(by_reference=True)` wires them. Three shapes exist.

| Shape | Meaning | Extra moves |
|---|---|---|
| Shared support | The new support cell **is** the robot's terminal support cell. One parked robot stops two different sliders. | zero |
| Target as stopper | The support cell is a mid-chain bottleneck of the robot. The robot serves as a stopper while passing through. | zero |
| Relocation | The robot slides from its terminal support cell to a new support cell. | an ordinary costed plan edge |

Two guards keep this sound. A robot mid-chain may only serve in place. And a
reference is refused when the referenced placement depends, through the DAG, on
the very leg that would consume it, because no schedule can satisfy that circular
timing.

*Concrete example — benchmark index 156, board 2452, d\* = 8.* Green is the
target at (11,9) and must reach (12,4). The plan has three subgoals.

- Subgoal 1: bottleneck (12,4) for Green, support (12,5) held by Red.
- Subgoal 2: Green first needs to reach (12,0). Bottleneck (12,0) for Green,
  support (13,0) held by Blue. Green's leg costs 3, Blue's leg costs 2.
- Subgoal 3: Red must reach (12,5). It first needs bottleneck (12,0). Its
  stopper is **Blue, already standing at (13,0)** — the same support node
  `sp_2` that subgoal 2 created. The edge `sg_3 → sp_2` carries `byref: true`
  and costs 0. Red's own leg costs 1.

Total plan cost 8, realized in 8 legal moves, equal to d\* = 8. Without the
by-reference rule the plan would have to recruit a second robot for the same
stopper duty, and no such robot was available.

**B2b — generalized parks.** Three relaxations of B1's repair, all default off:
the park cap rises from 1 to 2, destinations extend to two-slide cells when no
one-slide cell clears the leg, and two robots may be cleared at once when no
single-robot repair exists at all.

*Concrete example — benchmark index 76, board 2425, d\* = 6.* Two robots both
block the same slide. The plan carries two park nodes. Plan cost 13, realized in
13 legal moves. One park is not enough for this puzzle, so B1 cannot express it.

**What B2 costs.**

- *Branching.* By-reference candidates do not appear at the root, because no
  robot has been placed yet. My root-segment measurement therefore shows B1 and
  B2 identical, which is the expected result. Inside a real search the inflation
  was measured on the g16r6 frontier set: 4,577,425 candidate applications across
  46,351 expansions without by-reference, and a counterfactual 2,451,324 extra
  by-reference candidates over the same expansions. That is **+52.89 candidates
  per expansion, a 1.54× pool**, and by-reference candidates would be offered in
  **97.0% of expansions** and on 134 of 134 instances.
- *Realization work.* Generalized parks multiply the repair search. Pairwise
  clearing is quadratic in the movable robots, and two-slide destinations
  multiply the destination set.
- *Ranking.* Nothing about B2 changes what the value net was trained on.

### C.4 Effect (i) — expressibility

This is the exhaustive probe, `self_play_robots/spr/ceiling.py`. It performs a
best-first search over partial plans in abstract-cost order with **no networks
anywhere** and strictly realizes every complete plan it pops. It records four
categories per instance.

| Category | Meaning |
|---|---|
| `REALIZABLE_EXISTS` | a playable plan was found |
| `NO_REALIZABLE_PLAN` | complete plans exist, none of them plays out |
| `NO_COMPLETE_PLAN` | the language cannot even decompose this instance |
| `INCONCLUSIVE` | the probe hit its caps without deciding |

It also records `best_realizable_moves`, the cheapest certified plan popped
before the abstract cost reaches it. `mean_gap_best` is the mean of
`best_realizable_moves − d*` over graded instances.

**The measured ceilings.**

| Configuration and set | n | Vocabulary | `n_realizable` | Solve ceiling | `INCONCLUSIVE` | `mean_gap_best` | % reaching d\* |
|---|---|---|---|---|---|---|---|
| g16r4 (450) | 450 | base | 408 | 90.67% | 0 | 1.4191 | 61.76% |
| g16r4 (450) | 450 | B2 | 441 | **≥98.00%** | 9 | 0.8957 | 66.89% |
| g24r4 graded | 232 | base | 216 | 93.10% | 0 | 1.7222 | 57.41% |
| g24r4 graded | 232 | B2 | 228 | **≥98.28%** | 4 | 1.1711 | 62.72% |
| g24r8 graded | 161 | base | 148 | 91.93% | 0 | 1.5811 | 58.78% |
| g24r8 graded | 161 | B2 | 159 | **≥98.76%** | 2 | 1.3333 | 62.89% |
| g32r4 graded | 175 | base | 156 | 89.14% | 0 | 1.5577 | 57.69% |
| g32r4 graded | 175 | B2 | 166 | **≥94.86%** | 9 | 1.2771 | 61.45% |
| g24r4 frontier | 218 | base | 103 | 47.25% | 0 | — | — |
| g24r4 frontier | 218 | B2 | 114 | **≥52.29%** | 104 | — | — |
| g32r4 frontier | 275 | base | 142 | 51.64% | 0 | — | — |
| g32r4 frontier | 275 | B2 | 152 | **≥55.27%** | 123 | — | — |

Three properties of this table must be stated.

1. **The base-vocabulary rows are exact.** `capped = 0` in every one. The search
   ran to exhaustion. Those ceilings are not lower bounds.
2. **The B2 rows are lower bounds.** They hit the caps. That is why they carry
   "≥".
3. **The `mean_gap_best` columns are computed over different instance sets.** The
   base column averages over 408 instances at g16r4 and the B2 column over 441.
   The B2 set includes 37 harder instances that the base language cannot express
   at all. A raw comparison of the two means is not paired.

**The paired comparison, computed this session.** I matched the probe rows by
instance index and restricted to instances that both vocabularies express.

| Configuration and set | Realizable under both | Only under B2 | Only under base | `mean_gap_best` base → B2 | Change | B2 shorter / longer / equal | % optimal base → B2 |
|---|---|---|---|---|---|---|---|
| g16r4 (450) | 404 | 37 | 4 | 1.2550 → 0.7574 | **−0.4975** | 53 / 1 / 350 | 62.38% → 69.55% |
| g24r4 graded | 214 | 14 | 2 | 1.6168 → 1.1028 | **−0.5140** | 19 / 2 / 193 | 57.94% → 62.62% |
| g24r8 graded | 147 | 12 | 1 | 1.4762 → 1.1429 | **−0.3333** | 13 / 0 / 134 | 59.18% → 65.31% |
| g32r4 graded | 152 | 14 | 4 | 1.3421 → 1.0724 | **−0.2697** | 7 / 0 / 145 | 59.21% → 62.50% |
| g24r4 frontier | 79 | 35 | 24 | — | — | — | — |
| g32r4 frontier | 105 | 47 | 37 | — | — | — | — |

The "only under base" column is a cap artifact, not a language regression. B2's
plan set is a strict superset of the base plan set, because `propose_b1` returns
the base pairs plus more and `by_reference=True` only adds wiring options. Those
instances are ones where the larger B2 search exhausted its time cap. This is
visible at g16r4, where the B2 probe marks indices 15, 62, 209 and 214 as
`INCONCLUSIVE` although the base probe proves them realizable.

The frontier rows also used **unequal caps**: 180 s per instance for the base
probe and 120 s for the B2 probe. The base probe therefore had more time and
still expressed fewer instances. The B2 frontier advantage is understated.

**The B1 decomposition at the base configuration.** The older incremental probes
separate B1's two parts. They ran on the 49 instances the best backward planner
failed. I verified the category counts of each file.

| Vocabulary | `NO_COMPLETE_PLAN` | `NO_REALIZABLE_PLAN` | `REALIZABLE_EXISTS` | Base-configuration ceiling |
|---|---|---|---|---|
| base | 22 | 20 | 7 | 401 + 7 = 408/450 = 90.7% |
| + transient supports only | 7 | 14 | 28 | 401 + 28 = 429/450 = 95.3% |
| + parks (full B1) | 7 | 4 | 38 | 401 + 38 = 439/450 = 97.6% |
| + B2 on the 11 residual | 0 | 0 | 9 (2 `INCONCLUSIVE`) | 439 + 9 = 448/450 = 99.6% |

Transient supports carry the larger share. Parks add 10 more instances. B2 adds
the last 9.

**A discrepancy between two measurements of the same quantity, reported in
full.** The project's headline base-configuration B2 ceiling is 99.6% (448/450).
It comes from the chain above: 439 from the B1 probe, plus 9 recovered by a B2
probe that ran only on the 11 residual instances at much larger caps. The newer
whole-bench probe reads **98.0% (441/450)** in a single pass at a 60 s per-instance
cap. Both are honest lower bounds and they do not contradict each other. They are
different experiments. I computed the union of every proven-realizable index
across all four probe files: **448/450 = 99.56%**, with only indices 405 and 427
never proven under any vocabulary. So the project's 99.6% reproduces exactly as a
union over runs, and the 98.0% is what one bounded pass reaches. Both numbers
should be quoted with their protocol attached. `self_play_robots/FINDINGS.md` §3
already flags this ("B2 98.0% here vs the 99.6% recorded with 4× caps").

**A caveat on the quality half of the ceiling, from the project's own audit.**
`best_realizable_moves` is not a proven upper bound on the language's move
optimum. The probe stops when the popped abstract cost reaches the best strict
count, which assumes strict ≥ abstract. That assumption fails when an incidental
robot happens to serve as a stopper. The audit
(`self_play_robots/results/audit_claims_2026-08-18.md` §7b) found certified
planner plans shorter than the probe's "best" on 4 of 215 g24r4 instances and 3
of 401 g16r4 instances, and a full-enumeration re-probe on subsets lowered the
mean gap by about 0.05 moves. So the `mean_gap_best` figures are approximately
0.05 too high. The **solve** ceilings are unaffected. The audit re-verified that
across 21 g16r4 and 27 g24r4 base-vocabulary payloads, no planner ever solved an
instance the base probe calls unrealizable.

**Effect (i), summarized.** The extended language raises what is possible by 5 to
7 points of solve ceiling on graded sets, by about 4 to 5 points on frontier
sets, and it lowers the language's own move floor by 0.27 to 0.51 moves in a
paired comparison. Those are real properties of the vocabulary. No training can
produce them and no training can remove them.

### C.5 Effect (ii) — what the trained planner achieved

Now the second quantity. Same benchmark, same budget, a trained planner.

The clean way to read this is at **fixed networks and a fixed search variant**,
so that only the vocabulary changes. That control exists at two configurations.
The `basenets_oldvocab` files run the base-B1 network pair with the **old**
vocabulary under `--backward-anytime`, which is exactly the setting of the B1 and
B2 rows.

| Configuration and set | Old vocabulary | B1 | B2 | old → B1 | B1 → B2 |
|---|---|---|---|---|---|
| g16r6 graded (316) | 282 | 304 | 306 | **+22** | +2 |
| g16r6 frontier (134) | 80 | 108 | 108 | **+28** | **0** |
| g16r8 graded (266) | 232 | not run | 262 | — | — |
| g16r8 frontier (184) | 95 | not run | 163 | — | — |

At g16r6, where all three arms exist at fixed nets and a fixed search variant,
**essentially the whole gain is B1**. B2 adds 2 solves on the graded set and
nothing at all on the frontier.

The remaining configurations have no fixed-nets old-language arm. Their old row
uses `--backward-prefix-check` while their B2 row uses `--backward-anytime`, so
the vocabulary and the stopping rule change together. Those cells are reported
with that caveat attached.

| Configuration and set | Old vocabulary | B2 | Difference | 95% CI | p | Significant? |
|---|---|---|---|---|---|---|
| g16r4 (450) | 401 | 430 | +6.4 | [+3.8, +9.1] | 1.1e-06 | yes |
| g16r6 graded | 275 | 306 | +9.8 | [+6.1, +13.8] | 3.4e-07 | yes |
| g16r6 frontier | 70 | 108 | +28.4 | [+20.2, +37.0] | 4.1e-10 | yes |
| g16r8 graded | 230 | 262 | +12.0 | [+8.0, +16.4] | 4.1e-09 | yes |
| g16r8 frontier | 88 | 163 | +40.8 | [+33.0, +48.3] | 7.3e-20 | yes |
| **g24r4 graded** | **205** | **199** | **−2.6** | **[−7.3, +2.1]** | **0.377** | **no** |
| g24r8 graded | 144 | 148 | +2.5 | [−2.4, +7.6] | 0.424 | no |
| g24r8 frontier | 154 | 161 | +2.4 | [−3.1, +7.9] | 0.464 | no |
| g32r4 graded | 147 | 154 | +4.0 | [−1.8, +10.0] | 0.248 | no |
| g32r4 frontier | 127 | 196 | +25.1 | [+18.9, +31.4] | 8.2e-13 | yes |

The shape is clear. At 16×16 the extension delivers large, significant gains, and
they are largest at the frontier. At the big boards it delivers one large
frontier gain (g32r4, +25.1) and four cells that are statistically
indistinguishable from zero, one of which points the wrong way.

**Retraining on extended-vocabulary labels.** The natural next step is to train
the networks on labels drawn from the extended vocabulary. That was performed at
the base configuration.

| Base-configuration arm | Solved | Mean regret | % optimal | Mean expansions |
|---|---|---|---|---|
| B2 with B1-trained nets (zero-shot ranking) | 430/450 | 2.040 | 53.0% | 9.7 |
| B2 retrained on the cap-5,000 corpus | 433/450 | 2.051 | 53.8% | 14.6 |
| B2 retrained on the cap-20,000 corpus | 429/450 | 2.261 | 53.6% | 15.3 |

Retraining moved the base configuration by +3 and then by −4 solves, and it
increased the search effort by more than half. `FINDINGS.md` §47 states the
campaign's summary in the same direction: "zero-shot remains the method's best
configuration at every rung measured."

**What the extension costs at run time.** Using only same-machine pairs, so the
seconds are meaningful:

| Configuration and set | Base vocabulary | Extended language | Change in expansions | Change in seconds |
|---|---|---|---|---|
| g16r6 graded (fixed nets) | 26.6 exp / 4.417 s | 27.1 exp / 2.694 s | +2% | **−39%** |
| g16r6 frontier (fixed nets) | 153.5 exp / 17.722 s | 269.1 exp / 19.553 s | +75% | +10% |
| g16r8 graded (fixed nets) | 27.9 exp / 6.097 s | 8.6 exp / 1.750 s | **−69%** | **−71%** |
| g16r8 frontier (fixed nets) | 176.9 exp / 38.514 s | 182.8 exp / 27.041 s | +3% | −30% |
| g32r4 graded | 13.8 exp / 10.222 s | 34.5 exp / 30.606 s | **+150%** | **+199%** |
| g32r4 frontier | 29.7 exp / 14.598 s | 96.1 exp / 57.761 s | **+224%** | **+296%** |
| g24r8 frontier | 260.1 exp / 200.967 s | 635.4 exp / 431.277 s | **+144%** | **+115%** |

The pattern is consistent with the solve-rate pattern. At 16×16 the extension is
free or cheaper, because it finds a playable plan sooner and terminates. At the big
boards it costs two to four times the search steps and two to four times the
time. At g24r4 the two arms are not same-machine, but the machine-independent
number is unambiguous: 26.9 expansions become 53.5, a factor of 1.99.

### C.6 The honest result at 24×24 — confirmed, with the confounds named

The task brief asked me to confirm or refute a specific claim: that the extension
made the supervised planner **worse** at 24×24 under the same checkpoints and the
same budget, while lowering the language's theoretical floor.

**Confirmed on both halves.** Here is the verification.

*Same checkpoints.* `comparison.json` and `comparison_b2.json` at g24r4 both
record the identical backward pair:
`scaling/runs/g24r4/backward-policy/lightning_logs/version_0/checkpoints/epoch=1-step=1654.ckpt`
(mtime 2026-07-14T01:12:29) and
`scaling/runs/g24r4/backward-value/lightning_logs/version_0/checkpoints/epoch=22-step=38019.ckpt`
(mtime 2026-07-14T00:14:18).

*Same instances.* Both files record `instances_sha256 = 0e8a5bada3a0…` for
`scaling/data/g24r4/bench.solved.jsonl`, n = 232.

*Same budget.* Both record `expansions = 1200` and `k = 5`.

*The measured outcome.*

| Metric | Base vocabulary | B2 | Change |
|---|---|---|---|
| Solved | 205/232 = 88.36% | 199/232 = 85.78% | **−6 puzzles, −2.59 points** |
| McNemar | — | — | p = 0.377, CI [−7.3, +2.1] |
| Discordant pairs (computed this session) | — | — | B2 gains 13, loses 19 |
| Mean regret over solved | 4.220 | 4.889 | **+0.669 moves** |
| Percent optimal | 38.5% | 36.7% | **−1.8 points** |
| Mean moves on the 186 both-solved instances | 10.661 | 12.247 | **+1.586 moves** |
| Both-solved, B2 shorter / longer / equal | — | — | 15 / 41 / 130 |
| Mean expansions | 26.9 | 53.5 | **1.99×** |
| Plans found | 208 | 208 | unchanged |
| Plans found that then failed realization | 3 | 9 | **3×** |

Meanwhile, at the same configuration, the language's own floor moved the other
way: the solve ceiling rose from 93.10% to at least 98.28% and the paired mean
move gap fell from 1.6168 to 1.1028.

**So: a richer language, a lower theoretical floor, and a worse trained planner.**

**The confounds, named honestly.** The two arms differ in three ways, not one.

1. The vocabulary changed (base → B1 candidates plus generalized parks).
2. The stopping rule changed (prefix-check → anytime).
3. The prefix filter was disabled. `mean_children_pruned` is 1.42 in the base
   arm and 0.00 in the B2 arm.

Changes 2 and 3 pull in opposite directions. Anytime should **help** solve rate,
because it retries after a realization failure rather than stopping. Losing the
prefix filter should **hurt**, because doomed branches now consume budget. So the
result is not a clean single-variable measurement, and I will not present it as
one.

What is not in doubt is the direction at the level the reader cares about. The
extended-language arm received the retry rule that can only help, received a
strictly larger set of expressible plans, spent twice the search steps, and still
solved six fewer puzzles and produced solutions 1.59 moves longer where both
arms solved. Whatever the prefix filter contributed, the extension was not worth its cost at this
configuration.

The project's own log reaches the same conclusion from a different direction.
`FINDINGS.md` §46 records the same cell as "at this rung the full language adds
nothing on the graded set (−2.6, p=0.38) — consistent with §37/§39's finding that
the language gain is a frontier phenomenon."

### C.7 Why the richer language did not reach the trained planner

Four measured mechanisms explain the gap. All four are recorded in the project's
own audits, and I verified the underlying files.

**Mechanism 1 — the by-reference machinery was never wired into the learned
planner.** `eval/compare.py::_nn_astar_backward` calls `solver.propose(...)`
without injecting `_reference_helpers` and calls `_apply(...)` without
`by_reference=True`. The wiring exists only in `skeleton/astar.py::AStar._expand`,
which the neural loop reaches only on the free exact-fix branch, and that branch
returns above the by-reference block. The instrumented census confirms it: across
46,351 expansions and 4,577,425 candidate applications, the shipped driver
generated, ranked, shortlisted and expanded **exactly zero** by-reference
candidates, while a correctly wired driver would have offered **2,451,324** of
them in 97.0% of expansions on 134 of 134 instances.

Consequence: **every "B2" row measured in the supervised study is B1 plus
generalized park repairs.** The statement in `FINDINGS.md` §16 and in
`eval/compare.py`'s own help text that "the nets rank the new candidate type
zero-shot" is false for every measured row. The project recorded this correction
itself in §40. I re-read both source files and confirm the wiring claim.

**Mechanism 2 — the featurization cannot name a by-reference candidate.**
`eval/end2end.py::_hidx` maps a candidate's helper by its robot's **start**
position, and the driver drops any candidate whose helper is not at a start cell.
A by-reference helper never is, by definition. So even if the candidates were
generated, they could not be encoded.

**Mechanism 3 — the policy net never trained on one.**
`train/policy_common.py` silently skips any record whose `cand_helper` is not at
a start cell. Every by-reference record was filtered out of every corpus before
the policy trainer saw it. Only the value net, which uses raw cell indices, saw
them. The corpora do contain them:
`analysis/artifacts/byref_shares.json` records a by-reference share of 12.7% at
g16r4, 11.0% at g16r6, 10.4% at g16r8, 10.5% at g24r8 and 10.6% at g32r4 for the
cap-20,000 corpora.

**The direct test of mechanisms 1 to 3.** The by-reference step was later wired
into the driver and run as an A/B at g16r6 with the same banked networks and the
same pinned instances.

| Arm | Graded (316) | Frontier (134) | By-reference candidates ranked per instance | Entering the top-k per instance |
|---|---|---|---|---|
| by-reference off | 310 | 105 | 0 | 0 |
| by-reference on | 310 | 108 | 599.1 (graded), 5363.3 (frontier) | 95.4 (graded), 717.5 (frontier) |

The supply is enormous and the effect is +3 frontier solves and 0 graded solves.
`FINDINGS.md` §78 records the pooled McNemar as p = 0.61. The off arm reproduces
the production rows exactly, which validates the A/B. The reading in the log is
the right one: the nets never saw a by-reference record and cannot name the
referenced cell, so this is a lower bound on the step's value. The supply is not
the bottleneck. The training signal is.

**Mechanism 4 — the extended-vocabulary labels price plans that cannot be
played.** This is the deepest of the four. A review of the B2 label pipeline
sampled 16 depth-0 decisions covering 28 exact-optimal B2 completions and found
that `strict_moves` certifies only **13 of 28, or 46%** of them. The remaining
54% are transient-support plans that the abstract language accepts and physics
rejects. The exact B2 labeler also never prices parks, because parks are a
realization-time repair and not part of any labeling vocabulary.

So a network trained on exact B2 labels is trained to prefer candidates that lead
to abstract-cheap plans, and roughly half of the abstract-cheap plans in the
extended language do not play out. That is a direct explanation for two of this
document's observations: why retraining on richer corpora did not help
(section C.5) and why the B2 arm at g24r4 found the same number of plans but had
three times as many of them fail realization (section C.6).

**The one-sentence conclusion of section C.** A richer language raises what is
*possible*. It does not by itself teach a planner to use it, and when its cost
model is a worse predictor of real moves, it can make a trained planner worse.

---

## D. What Part 2 hands to Part 3

Four facts. Part 3 depends on all of them.

**D.1 — The base subgoal language is saturated. Training cannot move it.** The
exhaustive base-vocabulary ceilings are exact, with `capped = 0` in every case:
216/232 at g24r4 graded, 156/175 at g32r4 graded, 148/161 at g24r8 graded,
103/218 at the g24r4 frontier and 142/275 at the g32r4 frontier. The supervised
size-free planner already sits at or within a few solves of every one of them:
147/161 at g24r8 graded, 156/175 at g32r4 graded, 141/275 at the g32r4 frontier
and 99/218 at the g24r4 frontier (verified in
`self_play_robots/results/transfer/*.json`). A self-play loop that searches in
the base vocabulary therefore has almost nothing left to win on solve rate. Any
loop that wants headroom must use the extended vocabulary.

**D.2 — Each vocabulary has a measured quality floor, and both floors are far
above the exact optimum.** In the paired comparison the base language cannot get
below a mean gap of 1.26 moves at 16×16 and 1.62 at 24×24, or above about 62%
optimal. The extended language lowers those to 0.76 and 1.10 and raises the
optimal share to about 63% to 70%. The forward planner's measured regret is 0.067
at 16×16 and 0.068 at 24×24, at 94% optimal. So **no subgoal-space planner,
however well trained, can approach the forward planner on move quality at these
sizes.** The quality claim has to live in the primitive-move arm. Two caveats
travel with these floors: they are approximately 0.05 moves too high because of
the probe's stopping rule, and the anytime-versus-first-solution difference is
worth about 0.4 moves at both sizes under perfect ranking (`mean_gap_first` minus
`mean_gap_best` is 0.40 at g16r4 and 0.41 at g24r4).

**D.3 — The two action spaces have opposite cost profiles, and both are
measured.** The subgoal space is shallow and very wide: 0.8 to 0.9 subgoals per
plan, at 147 to 1986 candidates offered per decision, growing with board area and
robot count. The move space is narrow and deep: at most `4 × robots` actions per
state, and 5 to 12 states deep. In practice the subgoal planner needs 8.6 to 635
expansions per puzzle and the forward planner needs 36 to 1199, and the forward
planner exhausts the 1200-step budget on 84% to 99% of frontier instances at the
big boards while the subgoal planner exhausts it on 0% to 44%. The subgoal search
performs zero physics slides. Its physics cost sits entirely in realization,
prefix checks and park repairs, and it is heavy tailed.

**D.4 — The open question Part 3 inherits: a richer language that trained
planners could not exploit.** The extended language permits at least 98% at the
base configuration and at least 98.3% at 24×24. The best supervised planner
reaches 95.6% and 85.8%. The gap is not expressibility. Four measured mechanisms
block the supervised route: the by-reference candidates were never generated
in the evaluation loop, the featurization cannot name them, the policy trainer
silently filtered every one of them out of every corpus, and roughly half of the
exact extended-vocabulary labels price plans that physics rejects. Wiring the
first three and re-running the A/B produced +3 frontier solves out of 134, at
p = 0.61.

That is the handoff. The supervised route to the extended language is exhausted.
The remaining lever is a training signal that is generated by certified play
rather than by an abstract cost model — which is exactly what a self-play loop
produces, at any board size, without an exact solver.

---

## Appendix A — Verified numbers and their provenance

All paths are relative to `/scratch/project/open-37-42/petrhyner/MCTS_evolution/`.
"Aggregate" means the field was read from
`systems[<name>].aggregate.<field>` of the named JSON. "Protocol" means it was
read from `protocol.<field>`.

### A.1 Configuration metadata

| Number | Value | File | How read |
|---|---|---|---|
| g16r6 oracle failure rate | 0.29778 (134/450) | `supervised_valuenet/scaling/data/g16r6/bench.jsonl.meta.json` | `oracle_failure_rate`, `n_oracle_failed` |
| g16r8 oracle failure rate | 0.40889 (184/450) | `.../g16r8/bench.jsonl.meta.json` | same |
| g24r4 oracle failure rate | 0.48444 (218/450) | `.../g24r4/bench.jsonl.meta.json` | same |
| g24r8 oracle failure rate | 0.64222 (289/450) | `.../g24r8/bench.jsonl.meta.json` | same |
| g32r4 oracle failure rate | 0.61111 (275/450) | `.../g32r4/bench.jsonl.meta.json` | same |
| g16r4 fully graded | 450/450, `d_star` in [1, 12] | `supervised_valuenet/eval/data/bench450.jsonl` | read every line, checked `d_star` not in (None, 0) |
| g24r4 instance hash shared by both arms | `0e8a5bada3a0…` | `scaling/results/g24r4/comparison{,_b2}.json` | `protocol.instances_sha256`, compared |

### A.2 My own measurements this session

Environment: `ml Python/3.11.5-GCCcore-13.2.0 bzip2/1.0.8-GCCcore-13.2.0`,
`/scratch/project/open-37-42/petrhyner/venv`, `PYTHONPATH=supervised_valuenet:self_play_robots`,
run from `supervised_valuenet/`, `OMP_NUM_THREADS=2`, login node, CPU only.

| Number | Value | How produced |
|---|---|---|
| Forward legal moves at start, g16r4 | mean 13.17, median 13.0, max 16 (60 instances) | `move_planner.state.legal_moves` on each instance of `eval/data/bench450.jsonl` |
| Forward legal moves at start, g24r4 | mean 13.72, median 14.0, max 16 (40 instances) | same, `scaling/data/g24r4/bench.solved.jsonl` |
| Forward legal moves at start, g24r8 | mean 26.80, median 27.0, max 30 (30 instances) | same, `scaling/data/g24r8/bench.solved.jsonl` |
| Root candidates, g16r4, base / B1 | 146.9 / 257.9 mean; 15 / 30 median; 792 / 1701 max (52 instances with an open root segment) | `skeleton.astar._initial_plan` + `_segment`, then `heuristics.propose` and `heuristics.propose_b1` |
| Root candidates, g24r4, base / B1 | 215.4 / 523.5 mean; 25.5 / 57 median; 1017 / 2421 max (38) | same |
| Root candidates, g24r8, base / B1 | 698.2 / 1986.2 mean; 84 / 210 median; 4557 / 11613 max (27) | same |
| Subgoals per plan, g16r4 | mean 0.92, median 1, max 3; 0:23, 1:37, 2:7, 3:5 (72 of 80 realized) | exhaustive best-first search over partial plans, first plan passing `eval.realize.strict_moves`, counting `subgoal` nodes |
| Plan DAG nodes / realized moves, g16r4 | 5.67 / 8.60 mean | same run |
| Subgoals per plan, g24r4 | mean 0.82, median 1, max 3; 0:22, 1:25, 2:6, 3:3 (56 of 60 realized) | same |
| Plan DAG nodes / realized moves, g24r4 | 5.29 / 9.70 mean | same run |
| Paired ceiling comparison, all rows of section C.4 | see table | matched `rows[].idx` between the base and B2 ceiling JSONs, restricted to instances realizable under both and with `d_star` not in (None, 0) |
| Union of proven-realizable indices, g16r4 | 448/450 = 99.56%, missing 405 and 427 | union over `g16r4_base.json`, `g16r4_b2.json`, `ceiling_probe_results_b1.json`, `ceiling_probe_results_b2.json` |
| g24r4 discordance, base vs B2 | B2 gains 13, loses 19 | positional row match on `rows[].solved` |
| g24r4 both-solved move lengths | 10.661 vs 12.247 over 186 instances; B2 shorter 15, longer 41, equal 130 | `rows[].realized_strict` on rows solved by both |
| g24r4 forward rows identical across the two files | 232 of 232 | compared `solved`, `moves`, `expansions` per row |
| g24r4 forward median seconds, origin vs Karolina | 65.94 vs 87.62, ratio 1.329 | median of `rows[].seconds` in each file |
| Median / p90 / total seconds and budget saturation, all of section B | see tables | percentiles over `rows[].seconds`; saturation counted as `expansions >= 0.99 × protocol.expansions` |
| Backward search flags per file | see section B.5 | regular-expression scan of `protocol.command` for the five `--backward-*` flags (anytime, prefix-check, b1, b2, byref) |

### A.3 Head-to-head aggregates (all from `systems[...].aggregate`)

| Configuration / set | File | Key values read |
|---|---|---|
| g16r4 forward | `supervised_valuenet/eval/results/comparison_forward.json` | `candidate_scored.ckpt` row: solved 450/450, regret 0.0667, pct_optimal 94.22, moves 6.436, expansions 36.0, seconds 0.986 |
| g16r4 backward, base vocab, anytime | `eval/results/final450_backward_anytime.json` | 401/450, 2.1467, 50.37, 8.387, 7.6, 1.094 |
| g16r4 backward, base vocab, prefix | `eval/results/final450_backward_prefix.json` | 401/450, 2.1446, 50.37, 8.384, 7.13, 1.070 |
| g16r4 backward B1 | `eval/results/final450_backward_b1.json` | 429/450, 2.0280, 53.15, 8.296, 9.7, 1.228 |
| g16r4 backward B2 | `eval/results/final450_backward_b2.json` | 430/450, 2.0395, 53.02, 8.307, 9.7, 0.550 |
| g16r4 backward B2 retrained cap-5,000 | `eval/results/final450_backward_b2_retrained.json` | 433/450, 2.0508, 53.81, 8.330, 14.6, 2.077 |
| g16r4 backward B2 retrained cap-20,000 | `eval/results/final450_backward_b2_retrained_cap20000.json` | 429/450, 2.2611, 53.61, 8.538, 15.3, 2.347 |
| g16r6 graded, old vocab, per-config nets | `scaling/results/g16r6/comparison.json` | 275/316, 2.2218, 50.91, 7.865, 72.9, 11.697; forward first run 276/316 |
| g16r6 graded, old vocab, base-B1 nets | `scaling/results/g16r6/comparison_basenets_oldvocab.json` | 282/316, 2.6809, 49.65, 8.312, 26.6, 4.417 |
| g16r6 graded B1 | `scaling/results/g16r6/comparison_b1.json` | 304/316, 2.3125, 52.30, 8.000, 27.2, 5.040 |
| g16r6 graded B2 | `scaling/results/g16r6/comparison_b2.json` | 306/316, 2.3856, 52.61, 8.062, 27.1, 2.694 |
| g16r6 graded forward control | `scaling/results/g16r6/comparison_forward_control.json` | 314/316, 0.0955, 92.36, 5.777, 64.2, 21.649 |
| g16r6 frontier, old vocab, per-config nets | `scaling/results/g16r6/comparison_ungraded.json` | backward 70/134, moves 14.914, 296.1, 80.407; forward 65/134, 9.785, 835.1, 280.986 |
| g16r6 frontier, old vocab, base-B1 nets | `scaling/results/g16r6/comparison_ungraded_basenets_oldvocab.json` | 80/134, 16.587, 153.5, 17.722 |
| g16r6 frontier B1 | `scaling/results/g16r6/comparison_ungraded_b1.json` | 108/134, 16.852, 269.5, 41.979 |
| g16r6 frontier B2 | `scaling/results/g16r6/comparison_ungraded_b2.json` | 108/134, 16.824, 269.1, 19.553 |
| g16r8 graded, old vocab, per-config nets | `scaling/results/g16r8/comparison.json` | backward 230/266, 1.9652, 51.30, 7.130, 53.8, 14.713; forward 25/266 (withheld) |
| g16r8 graded, old vocab, base-B1 nets | `scaling/results/g16r8/comparison_basenets_oldvocab.json` | 232/266, 2.0603, 51.29, 7.224, 27.9, 6.097 |
| g16r8 graded B2 | `scaling/results/g16r8/comparison_b2.json` | 262/266, 2.5496, 50.38, 7.740, 8.6, 1.750 |
| g16r8 graded forward control | `scaling/results/g16r8/comparison_forward_control.json` | 261/266, 0.1073, 91.95, 5.272, 62.4, 21.206 |
| g16r8 graded forward rescue | `scaling/results/g16r8/comparison_forward_rescue.json` | 266/266, 0.0977, 91.35, 5.286, 44.9, 16.027 |
| g16r8 frontier, old vocab, per-config nets | `scaling/results/g16r8/comparison_ungraded.json` | backward 88/184, 13.227, 264.3, 87.642; forward 93/184, 8.828, 818.8, 256.604 |
| g16r8 frontier, old vocab, base-B1 nets | `scaling/results/g16r8/comparison_ungraded_basenets_oldvocab.json` | 95/184, 14.958, 176.9, 38.514 |
| g16r8 frontier B2 | `scaling/results/g16r8/comparison_ungraded_b2.json` | 163/184, 17.810, 182.8, 27.041 |
| g16r8 frontier forward rescue | `scaling/results/g16r8/comparison_ungraded_forward_rescue.json` | 101/184, 8.812, 745.4, 288.593 |
| g24r4 graded, base vocab | `scaling/results/g24r4/comparison.json` | backward 205/232, 4.2195, 38.54, 11.810, 26.9, 7.803; forward 220/232, 0.0682, 94.09, 7.568, 191.0, 274.945 |
| g24r4 graded B2 | `scaling/results/g24r4/comparison_b2.json` | backward 199/232, 4.8894, 36.68, 12.327, 53.5, 23.448; forward identical to above at 319.639 s |
| g24r4 base vocab, seed-21 nets, Karolina | `scaling/results/g24r4/comparison_exactseed21.json` | 205/232, 4.5171, 40.00, 12.068, 26.5, 27.596; prefix-check calls 36.237/instance |
| g24r4 frontier | `scaling/results/g24r4/comparison_ungraded_b2.json` | backward 125/218, 22.536, 110.2, 49.377; forward 15/218, 11.867, 1162.9, 2450.601 |
| g24r8 graded, base vocab | `scaling/results/g24r8/comparison.json` | backward 144/161, 2.4861, 50.69, 7.993, 44.0, 34.056; forward 157/161, 0.1656, 85.35, 5.758, 151.6, 183.505 |
| g24r8 graded B2 | `scaling/results/g24r8/comparison_b2.json` | 148/161, 2.3446, 46.62, 7.892, 94.9, 65.897 |
| g24r8 frontier, base vocab | `scaling/results/g24r8/comparison_ungraded.json` | backward 154/289, 15.675, 260.1, 200.967; forward 44/289, 8.545, 1100.8, 1166.196 |
| g24r8 frontier B2 | `scaling/results/g24r8/comparison_ungraded_b2.json` | 161/289, 15.522, 635.4, 431.277 |
| g32r4 graded, base vocab | `scaling/results/g32r4/comparison.json` | backward 147/175, 2.3673, 51.70, 9.980, 13.8, 10.222; forward 133/175, 0.1353, 88.72, 7.338, 469.2, 1378.611 |
| g32r4 graded B2 | `scaling/results/g32r4/comparison_b2.json` | 154/175, 2.8442, 47.40, 10.519, 34.5, 30.606 |
| g32r4 frontier, base vocab | `scaling/results/g32r4/comparison_ungraded.json` | backward 127/275, 21.276, 29.7, 14.598; forward 2/275, 11.500, 1198.5, 2379.784 |
| g32r4 frontier B2 | `scaling/results/g32r4/comparison_ungraded_b2.json` | 196/275, 24.587, 96.1, 57.761 |

### A.4 Significance cells

All from `supervised_valuenet/eval/results/stats_tests.json`, field `cells`, read
by matching `(rung, set, a, b)`. Method recorded in `method`: exact two-sided
McNemar on discordant pairs, plus a percentile bootstrap over boards with 10,000
resamples at seed 0. The file holds 120 cells. The values quoted in sections B.4
and C.5 are `solved_a`, `solved_b`, `diff`, `ci95_lo`, `ci95_hi`, `mcnemar_p`.

### A.5 Ceiling probes

All from `self_play_robots/results/ceiling/*.json`, field `summary`.

| File | `n_realizable` / `n` | `solve_ceiling` | `capped` | `mean_gap_best` | `mean_gap_first` | `pct_best_optimal` | `caps.time_cap` |
|---|---|---|---|---|---|---|---|
| `g16r4_base.json` | 408/450 | 0.90667 | 0 | 1.4191 | 1.8186 | 61.76 | 60 s |
| `g16r4_b2.json` | 441/450 | 0.98000 | 12 | 0.8957 | 1.4376 | 66.89 | 60 s |
| `g16r4_base_slack4.json` | 408/450 | 0.90667 | 0 | 1.3971 | 1.8186 | 62.25 | 60 s |
| `g16r4_base_slack12.json` | 408/450 | 0.90667 | 0 | 1.3873 | 1.8186 | 62.50 | 60 s |
| `g24r4_base.json` | 216/232 | 0.93103 | 0 | 1.7222 | 2.1343 | 57.41 | 90 s |
| `g24r4_b2.json` | 228/232 | 0.98276 | 7 | 1.1711 | 1.7105 | 62.72 | 90 s |
| `g24r4_base_slack4.json` | 216/232 | 0.93103 | 0 | 1.6852 | 2.1343 | 57.87 | 90 s |
| `g24r8_base.json` | 148/161 | 0.91925 | 0 | 1.5811 | 1.8514 | 58.78 | 120 s |
| `g24r8_graded_b2.json` | 159/161 | 0.98758 | 2 | 1.3333 | 1.7296 | 62.89 | 120 s |
| `g32r4_base.json` | 156/175 | 0.89143 | 0 | 1.5577 | 1.7500 | 57.69 | 120 s |
| `g32r4_graded_b2.json` | 166/175 | 0.94857 | 11 | 1.2771 | 1.5422 | 61.45 | 120 s |
| `g24r4_frontier_base.json` | 103/218 | 0.47248 | 0 | — | — | — | **180 s** |
| `g24r4_frontier_b2.json` | 114/218 | 0.52294 | 115 | — | — | — | **120 s** |
| `g32r4_frontier_base.json` | 142/275 | 0.51636 | 0 | — | — | — | **180 s** |
| `g32r4_frontier_b2.json` | 152/275 | 0.55273 | 133 | — | — | — | **120 s** |
| `g24r8_frontier_b2.json` | 180/289 | 0.62284 | 124 | — | — | — | 120 s |

Category counts, read from `summary.categories`:

- `g16r4_base.json`: 408 `REALIZABLE_EXISTS`, 22 `NO_COMPLETE_PLAN`, 20 `NO_REALIZABLE_PLAN`
- `g16r4_b2.json`: 441 `REALIZABLE_EXISTS`, 9 `INCONCLUSIVE`
- `g24r4_base.json`: 216 / 9 / 7
- `g24r4_frontier_base.json`: 103 / 99 / 16
- `g32r4_frontier_base.json`: 142 / 122 / 11

The older incremental probes, read as a bare list of rows with a `category` field:

| File | Rows | Categories |
|---|---|---|
| `supervised_valuenet/analysis/artifacts/ceiling_probe_results.json` | 49 | 22 `NO_COMPLETE_PLAN`, 20 `NO_REALIZABLE_PLAN`, 7 `REALIZABLE_EXISTS` |
| `.../ceiling_probe_results_b1_transient_only.json` | 49 | 7 / 14 / 28 |
| `.../ceiling_probe_results_b1.json` | 49 | 7 / 4 / 38 |
| `.../ceiling_probe_results_b2.json` | 11 | 9 `REALIZABLE_EXISTS`, 2 `INCONCLUSIVE` |
| `.../ceiling_probe_results_b2_deep.json` | 2 | 2 `INCONCLUSIVE` |
| `scaling/results/g16r6/ceiling_probe_old_vocab.json` | 105 | 48 / 23 / 28, plus 6 `INCONCLUSIVE` |
| `scaling/results/g16r6/ceiling_probe_b1.json` | 77 | 63 `REALIZABLE_EXISTS`, 12 `INCONCLUSIVE`, 2 `NO_REALIZABLE_PLAN` |
| `scaling/results/g16r6/ceiling_probe_b2.json` | 14 | 5 `REALIZABLE_EXISTS`, 9 `INCONCLUSIVE` |

### A.6 By-reference measurements

| Number | Value | File | How read |
|---|---|---|---|
| Candidate applications per expansion, g16r6 frontier | 4,577,425 / 46,351 = 98.76 | `analysis/artifacts/byref_topk_ablation.json` | `summary.totals.asis_apply_calls / asis_expansion_groups` |
| By-reference candidates generated by the shipped driver | 0 | same | `summary.totals.asis_byref_{proposed,generated,ranked,topk,expanded}` |
| Counterfactual by-reference supply | 2,451,324, i.e. +52.89 per expansion, pool 1.536× | same | `summary.totals.shadow_byref_generated` |
| Expansions that would offer one | 44,967 / 46,351 = 97.0% | same | `summary.totals.shadow_groups_with_byref` |
| Instances with counterfactual supply | 134/134 | same | `summary.instances_with_shadow_byref` |
| By-reference share of the cap-20,000 corpora | 12.7 / 11.0 / 10.4 / 10.5 / 10.6 % | `analysis/artifacts/byref_shares.json` | `share_pct` per config |
| Wired A/B, g16r6 graded | on 310/316, off 310/316 | `scaling/results/g16r6/comparison_b2retrained_cap20000_byref_{on,off}.json` | aggregate `solved` |
| Wired A/B, g16r6 frontier | on 108/134, off 105/134 | `scaling/results/g16r6/comparison_ungraded_b2retrained_cap20000_byref_{on,off}.json` | aggregate `solved` |
| By-reference candidates ranked / entering top-k | graded 599.111 / 95.446, frontier 5363.254 / 717.478 | same four files | `aggregate.accounting.byref_cands_{ranked,topk}` |

### A.7 Physics-work pilot

| Number | Value | File | How read |
|---|---|---|---|
| Backward physics slides, median | 64 (`strict_realize` bucket) | `eval/results/instrumentation_ab/v3_split_backward5.json` | median over `rows[].accounting.slide_calls` |
| Backward all slides, median | 983 (of which `abstract_scoring` 909) | same | same |
| Backward mean expansions | 3.4 | same | mean of `rows[].expansions` |
| Forward physics slides, median | 5,776 (`forward_search`) | `eval/results/instrumentation_ab/v3_split_forward5.json` | same |
| Forward all slides, median | 40,448 (of which `forward_encode` 34,672) | same | same |
| Forward mean expansions | 337.0 | same | same |
| Backward `backward_search` bucket | absent, i.e. zero slides | both files | no such bucket appears in any row |

### A.8 Plan-structure examples

All from `eval/results/plan_structures_data.json`, field `examples`, read as node
and edge lists. Produced by a hand-coded search with no neural networks, and
strictly playable plans only.

| Index | Board | d\* | Plan cost | Legal moves | Subgoals | Parks | `byref` edges | Illustrates |
|---|---|---|---|---|---|---|---|---|
| 0 | 2400 | 12 | 12.0 | 12 | 2 | 0 | 0 | an ordinary plan in the base vocabulary |
| 333 | 2511 | 4 | 4.0 | 4 | 2 | 0 | 0 | a transient (wall-less) stopper — impossible before B1 |
| 28 | 2409 | 7 | 7.0 | 11 | 0 | 1 | 0 | a step-aside park |
| 156 | 2452 | 8 | 8.0 | 8 | 3 | 0 | 1 | a support by reference |
| 76 | 2425 | 6 | 13.0 | 13 | 0 | 2 | 0 | a generalized step-aside clearing two robots |

### A.9 The no-network control

| Number | Value | File | How read |
|---|---|---|---|
| base 450, hand-written scorer | 394/450 = 87.56%, 21.50 mean expansions | `eval/results/final450_backward_heuristic_baseline.json` | aggregate |
| g16r6 graded, hand-written scorer | 276/316 = 87.34%, 163.84 mean expansions | `scaling/results/g16r6/comparison_heuristic_baseline.json` | aggregate |
| g16r6 frontier, hand-written scorer | 77/134 = 57.46%, 576.83 mean expansions | `scaling/results/g16r6/comparison_ungraded_heuristic_baseline.json` | aggregate |

### A.10 Self-play-side numbers used in section D

| Number | Value | File | How read |
|---|---|---|---|
| Size-free supervised planner, g24r4 frontier | 99/218 | `self_play_robots/results/transfer/g24r4_frontier_astar.json` | aggregate `solved` |
| Size-free supervised planner, g32r4 frontier | 141/275 | `.../g32r4_frontier_astar.json` | same |
| Size-free supervised planner, g24r8 graded | 147/161 | `.../g24r8_graded_astar.json` | same |
| Size-free supervised planner, g32r4 graded | 156/175 | `.../g32r4_graded_astar.json` | same |

---

## Appendix B — Disagreements with the project's documents, and what I could not verify

### B.1 Two files disagree, and I name both

**B.1.1 — The base-configuration B2 ceiling is recorded as two different
numbers.**

| Source | Value | Protocol |
|---|---|---|
| `supervised_valuenet/FINDINGS.md` §16 and the Verdict paragraph | **99.6%** (448/450) | 439 from the full-B1 probe, plus 9 of the 11 residual instances re-probed under B2 at much larger caps |
| `self_play_robots/results/ceiling/g16r4_b2.json` | **98.0%** (441/450) | one whole-bench pass at a 60 s per-instance cap, 9 `INCONCLUSIVE` |

Both are honest lower bounds and neither is wrong. They measure the same quantity
with different search budgets. I computed the union of every index proven
realizable across all four probe files and obtained **448/450 = 99.56%**, with
only indices 405 and 427 unproven under any vocabulary. So the 99.6% figure
reproduces exactly, provided it is quoted as a union over runs.
`self_play_robots/FINDINGS.md` §3 already flags the difference. The supervised
log does not.

**B.1.2 — `eval/compare.py` contradicts itself in the same file.** Line 690, the
help text of `--backward-b2`, still reads "by-reference candidates are ranked
zero-shot by the B1 nets". Line 322 of the same file records the correction: "the
historical claim here that 'the nets rank them zero-shot' was false."
`FINDINGS.md` §16 and §17 carry the same false statement in prose, and §40 is the
correction. The instrumented census settles it: zero by-reference candidates were
generated, ranked, shortlisted or expanded in 46,351 expansions. The correction
is right and the help text is stale.

**B.1.3 — Nine result files publish a `mean_regret` that is a solution length.**
They are listed in section B.5. In each of them `mean_regret` equals `mean_moves`
to the last digit, because the frontier instances carry a placeholder `d_star`
of 0. The newer files at g24r4 and the `basenets_oldvocab` files record
`d_star_placeholder: true` and suppress the field. The project logged this as a
live footgun in `FINDINGS.md` §25. It is still live in those nine files.

**B.1.4 — `FINDINGS.md` §37's language table mixes provenances, and the project
corrected it in §39.** The "old" column at g16r6 uses per-config networks while
the B1 and B2 columns use the base-B1 pair, and it mixes `--backward-prefix-check`
rows with `--backward-anytime` rows. I verified both halves of that correction
independently. The clean fixed-nets, fixed-variant series at g16r6 is
282 → 304 → 306 graded and 80 → 108 → 108 frontier, which is what section C.5
uses. §37's qualitative conclusion survives the correction.

### B.2 Claims in the repository's documents that the data does not support

**B.2.1 — "The nets rank the new candidate type zero-shot."** Not supported. See
B.1.2. Every measured B2 row is B1 plus generalized park repairs.

**B.2.2 — "B2 removes nearly all of what remained of the ceiling" read as a
planner result.** The ceiling claim itself is supported. Reading it as a statement
about the trained planner is not. At fixed networks and a fixed search variant,
B1 → B2 is +2 solves on the g16r6 graded set and 0 on the g16r6 frontier. At
g24r4 the full language is 6 solves **behind** the base language.

**B.2.3 — `mean_gap_best` used as a proven language optimum.** The probe's
stopping rule assumes that a strict realization never costs less than the plan's
abstract cost, and that assumption fails when an incidental robot serves as a
stopper. The project's own audit
(`self_play_robots/results/audit_claims_2026-08-18.md` §7b) found certified plans
shorter than the probe's "best" on 4 of 215 g24r4 instances and 3 of 401 g16r4
instances, and estimated the bias at about 0.05 moves. The **solve** ceilings are
unaffected and were cross-checked against 21 g16r4 and 27 g24r4 base-vocabulary
payloads with no counterexample.

### B.3 What I could not verify

1. **The per-configuration physics-work accounting does not exist.**
   `eval/results/compute_accounting.json` carries the full bucket semantics but
   its `records` array is empty. The only slide-count data is the five-instance
   pilot of Appendix A.7 with `move_planner/checkpoints/best.ckpt` as the forward
   arm, which is not the base-configuration headline forward network. So the 90×
   physics ratio is a five-puzzle pilot, not a per-configuration measurement.
2. **The 46% realization rate of exact extended-vocabulary labels.** The figure
   13 of 28 comes from `self_play_robots/results/review_b2_pipeline_2026-08-18.md`
   §0.2. The review states that it computed this on the login node from
   `results/selfplay/g24r4_b2_iter1/gauge.work/engine.results.jsonl`. That file
   exists (3.5 MB). The derived count itself lives in a session scratchpad file
   that is not in the repository. I quote it and I did not re-derive it.
3. **Confidence intervals for the no-network control.** The solve counts
   (394/450, 276/316, 77/134) are verified from files. The intervals
   ([+5.6, +10.7], [+5.7, +13.7], [+15.4, +31.2]) are quoted from
   `FINDINGS.md` §49.
4. **The size of the anytime-versus-prefix-check confound away from the base
   configuration.** At the base configuration both variants give 401/450, so the
   confound is empirically null there. No such control was ever run at g24r4,
   g24r8 or g32r4. That is the single measurement that would turn section C.6's
   result from "the extension did not pay for itself" into a clean
   single-variable statement. It is cheap and it is unrun.
5. **A B1 arm at four of six configurations.** `FINDINGS.md` §37 records a
   read-only audit concluding that B1 rows exist only at the base configuration
   and g16r6, and that no per-configuration B1 network was ever trained. I
   confirmed the absence of the files. I did not repeat the cross-tree audit.
6. **Node-hour costs and job accounting.** Not checked. No Slurm queries were
   made.
7. **The by-reference count of 71 in a self-play iteration.** The
   2026-08-18 audit states that this number cannot be reproduced from the stored
   records, because the records carry no by-reference flag. I did not attempt it.

### B.4 Two statements I want to leave marked as inference, not measurement

- **"The `only under base` instances of the paired ceiling table are cap
  artifacts."** This rests on a code-level argument, not on a direct experiment.
  `propose_b1` returns the base pairs plus the transient pairs, and
  `by_reference=True` only adds wiring options, so the B2 plan set is a strict
  superset of the base plan set. The observation that all such instances are
  marked `INCONCLUSIVE` rather than `NO_COMPLETE_PLAN` is consistent with it.
- **"The g24r4 quality loss under B2 is caused by an abstract cost model that
  predicts real moves less well."** The pieces are measured — the plans found
  are equal in number, three times as many fail realization, both-solved
  solutions are 1.59 moves longer, and 54% of exact B2 labels price plans that
  physics rejects — but no experiment isolates the mechanism at that
  configuration.
