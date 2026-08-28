# Learned subgoal discovery — Stage 4

`PLAN_SUBGOAL_DISCOVERY.md` §4, Stage 4, run 2026-08-28.

**Everything down to the "Results" line was committed before the Stage 4 job it
judges was submitted**, in two commits: the Part A pre-registration (this file
down to §A, plus `subgoal/stage4.py`, the bound in `subgoal/planner.py` and the
three job scripts) before any Part A job ran, and the frozen Part A bar
(`results/subgoal/stage4/partA_bar.json`) before any Part B or Part C job ran.
Nothing below has been edited after a number was read; everything explored
afterwards is labelled exploratory and reported separately.

## Why Stage 4 is re-ordered

Stage 3 passed its gate at 60.7% and then found two things that are worth more
than the training run that produced it (`STAGE3.md` §7):

* **the beam is mis-sized.** An expansion costs exactly four encoder passes at
  *every* `k`, because the candidate cell is read out rather than encoded. So
  the prune buys nothing in network cost and only throws children away. Widening
  it lifted the same checkpoint from 60.7% to 79.8% at identical network cost;
* **the optimality bound is board-only.** 312 of 450 searches end on the
  expansion budget, and even an EXACT `h` cannot terminate early, because the
  stop test proves optimality with a relaxation that never looks at the robots.

Running self-play against the `k = 5` configuration would be measuring a
19-point handicap, so Stage 4 takes both free wins first, re-baselines, and
gates the learning on the re-baselined number.

## Part A — the free wins, and the new bar

### A1. The beam width

**The rule, fixed before the sweep ran:** the headline width is the SMALLEST
`k` of `{5, 10, 20, 50, 100, 1024}` whose optimal percentage on a HELD-OUT dev
set is within **1.0 point** of the best width's on that dev set. `k = 1024` is
no prune at all. The dev set is 200 instances on boards **1900–1999**
(`subgoal/stage4.py::devgen`), generated fresh for Stage 4 and disjoint from
the benchmark boards 2400–2549, from Stage 3's training boards 1000–1699 and
from Stage 3's validation boards 1800–1849. **The width is never chosen on
bench450.**

The dev set carries an exact `d*` from the same engine Stage 3 labelled with.
That is a property of the INSTANCES, used to score a search setting every arm
shares — the network-free control included. No network trains on it, and Part C
(which forbids exact-engine data) inherits the width as a protocol constant, in
the same way it inherits "1200 expansions".

### A2. The optimality bound

Stage 3's bound is `h_free(s)`, the any-stop relaxation distance of the target
robot to the target cell: board only. Stage 4 adds the two terms that look at
where the robots are (`subgoal/planner.py`, `Search._h1` / `Search._h2`). Write
any solution from `s` as `T` target-robot moves plus `O` other-robot moves:

* if `O = 0` the others never move, so `T >= d_frozen(s)`, the EXACT number of
  slides the target robot needs with the others frozen where they stand — one
  BFS, and the same BFS the expansion of `s` needs anyway;
* if `O >= 1` then `T >= h_free(s)` still, and `O >= max(1, r(s))`, where `r(s)`
  is 0 when a wall or the edge can stop a robot on the target cell or when a
  non-target robot already stands on a cell one step beyond it, and otherwise is
  the smallest any-stop relaxation distance from a non-target robot to those
  cells — because some robot must go and be the stopper first.

so **`h_tight(s) = min( d_frozen(s), h_free(s) + max(1, r(s)) )`** is admissible
and never below Stage 3's bound. It is used for two things, both sound:

1. the optimality stop, exactly as before but with the stronger bound;
2. skipping a popped node whose `g + h_tight` already reaches the incumbent —
   such a node cannot improve it, so expanding it would spend an expansion of
   the budget on nothing.

`h_tight` enters the open list as its O(1) half (`_h1`) and the few entries at
the top of the bound heap are lazily upgraded to the full one, capped at 48
upgrades per query; capping only weakens the bound, never breaks it.

**Both checks must pass before any Part A number is used**
(`python -m subgoal.stage4 verify`):

* **admissibility** — on random states of benchmark boards, `h_tight(s)` must
  never exceed the exact engine's true cost-to-go of `s`;
* **no drift** — with `bound="weak"` the search must reproduce Stage 3's stored
  payload rows field for field, so the Stage 3 numbers this stage compares
  against are still the numbers that code produces.

**Measured, and reported whatever it is:** how many of the 312 budget-limited
searches the stronger bound converts into a terminated, proved-optimal search.

### A3. The re-baselined table

The plan's four-row table, recomputed by `subgoal/table.py` from every payload's
own move dumps, at the new headline configuration: **1200 expansions, the beam
width A1 chose, the A2 bound**. The inherited protocol row (1200 expansions,
`k = 5`, board-only bound) is reported beside it so the 60.7% stays visible.

**That new number, frozen into `results/subgoal/stage4/partA_bar.json` and
committed before any self-play job is submitted, is the bar Parts B and C must
beat.**

## Part B — certified self-play

The point of the whole plan: the network learns `h` from its own certified
experience instead of from the exact engine.

* **fresh boards, never a benchmark board.** Roots are random placements plus
  0–2 random macro moves on train-split boards **1000–1799**; the per-round
  validation corpus comes from boards **1850–1899**. bench450's boards
  (2400–2549) and the dev boards (1900–1999) are never generated on.
* **search with exploration noise.** The round's own checkpoint is the `h`;
  candidate scores get Gaussian jitter with **sigma = 0.5** (untuned, fixed
  before the first round). Goal children are never jittered.
* **replay or nothing.** Every finished plan is replayed under the real
  joint-game rules (`subgoal/table.py::replay`) and its length checked against
  the search's own cost. **A plan that fails replay produces no training
  record.**
* **the label is measured, never predicted.** For every state on a certified
  plan, `h(parent, robot, cell)` is the true remaining cost of that certified
  plan from the child. Where several rounds or several plans reach the same
  child, the smallest certified cost wins.
* **probes, so the beam's decision is representable.** One certified plan labels
  only the children it walked through, and the beam has to ORDER a whole
  candidate set. So for every state the first pass reaches, **3** of its other
  reachable children — same robot — become roots of their own searches, and a
  certified plan from such a child is, by the same definition, the label of that
  child. This is the only source of labels for children an optimal plan never
  takes. It consults no oracle.
* **rolling buffer, warm start, collapse guard.** Round `r` trains from round
  `r-1`'s checkpoint on the last **2** rounds of labels, at the pre-registered
  recurrence **4** (at 12 the Stage 2 net sat on the constant-value plateau for
  30 epochs). `subgoal/stage4.py` watches `val_spread_reach`, the spread of the
  net's predictions over the physically reachable children, and stops the run if
  it stays under 0.05 for 5 epochs. Each round's checkpoint is selected by the
  best `val_mae_all` on that round's self-play validation corpus — a corpus with
  no exact-engine label in it.
* **at least three rounds**, with the per-round series on all 450 and its
  denominators.

**Gate (pre-registered):** the self-play planner's percent of the 450 solved
move-optimally must **EXCEED the Part A bar**, at the Part A headline
configuration, every solution replay-certified. If three rounds cannot beat it,
Stage 4 reports the negative plainly: that is a real result about whether
certified self-play can improve a heuristic that exact labels already trained
well.

## Part C — the question the project never answered

One arm, the same number of rounds, the same loop, from **random
initialisation** (`results/subgoal/stage4/scratch_init.ckpt`, seed 2026) and
with **no exact-engine data at any point** — no supervised warm start, no
oracle label, no oracle-labelled validation set. Its round-0 row is the random
network planning on bench450. Its series is reported beside Part B's, and the
question it answers is whether the supervised start matters at convergence.

## Rules for this stage

* the headline metric is unchanged: percent of all 450 solved with a provably
  optimal move count, every solution replay-certified by `subgoal/table.py`;
* no knob is touched after a gate number is read; anything explored afterwards
  is labelled exploratory and the pre-registered result stays the headline;
* never train on, select on, or generate on a benchmark board or a pinned exam
  id;
* Stage 4 stays under 6 node-hours.

---

## Results
