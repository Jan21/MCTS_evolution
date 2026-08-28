# Learned subgoal discovery — Stage 3 (PRE-REGISTRATION)

`PLAN_SUBGOAL_DISCOVERY.md` §4, revised Stage 3, run 2026-08-28. This file is
committed **before any Stage 3 training or planning job is submitted**, so the
gate, the headline configuration and the tie-breaking rule are on record ahead
of any number. Results are appended below afterwards; nothing above the
"Results" line is edited once a result has been read.

## What Stage 3 builds

Stage 2 learned the wrong object. Its target — the cost of *reaching* a subgoal
— is the edge weight of the macro expansion, which physics computes in well
under a millisecond and which the search must compute anyway to build the
child. Stage 2's beam measurement (`STAGE2.md` §6) showed the binding
constraint is the **selection rule**, and that the term worth adding is the
remaining distance to the final goal. So:

* **edges from physics.** Children of a state are `(robot, cell)` pairs from
  the exact joint-state slide BFS with the other three robots frozen — the
  Stage 2 cost definition, the only one that is an edge weight — with their
  true slide counts as costs (`subgoal/planner.py::rest_cells_fast`, verified
  cell-for-cell against Stage 1's `subgoal/space.py::rest_cells`, which is
  built on `simulate.slide`, by `subgoal.stage3 verify`);
* **rank by `g + h`.** `g` is the real cost so far; `h` estimates the moves
  still needed to finish from the child. One expansion scores all 1024
  candidates and keeps the best `k`;
* **learn `h` only**, on exact-engine cost-to-go targets.

## The pre-registered gate

> **The learned-`h` planner must solve MORE than 53.3% of the 450 pinned
> 16×16 four-robot benchmark instances with a provably optimal move count**
> (the backward supervised planner's 240/450, FINDINGS 29), at the arena
> protocol of **1200 expansions and k = 5**, every solution replay-certified.

Failing that, the plan says the search or the heuristic is wrong, self-play
will not rescue it, and Stage 3 reports the negative and stops. The gate
constant lives in `subgoal/stage3.py::GATE_BACKWARD_PCT`.

## The pre-registered configurations

| id | arm | h | k | expansions | role |
|---|---|---|---|---|---|
| **A** | `net` | learned `CtgNet` | **5** | **1200** | **the headline / gate row** |
| **B** | `relaxed` | any-stop relaxation, board only, no network | 5 | 1200 | the control: is the learned `h` worth anything at all? |
| C | `net` / `relaxed` | as above | 10, 20, 50, 200 | 1200 | the beam-width study |
| D | `exact` | the true cost-to-go from the engine | 5 | 1200 | diagnostic ceiling: search or heuristic? |

`k = 200` is effectively no prune: a 16x16 four-robot state has on average
only ~148 physically reachable candidates across all four robots, so a beam of
200 keeps essentially all of them. The network arm spends **4 encoder passes per
expansion at every k**, because the candidate cell is read out rather than
encoded, so every row of the beam study uses *identical* network calls
(4 × expansions per instance) and only the beam changes — which is exactly the
"matched network calls with a wider beam" the revised plan asks for. Row D is a
diagnostic, never a system: it costs one exact solve per candidate child.

Fixed for every row:

* **anytime with a sound optimality stop.** The search keeps the cheapest
  solution found and stops early only when that cost is `<=` the smallest
  `g + h_adm` over the open list, `h_adm` being the admissible relaxation. That
  rule is identical in every arm and can never hide a better solution.
* **deterministic tie-breaking**: candidates are ordered by
  `(f, g, robot slot, cell index)`. No randomness anywhere in the planner.
* a child that puts the target robot on the target cell gets `h = 0` and is
  always retained — that is the search's termination test, not a beam decision.
* every reported solution is replayed under the real joint-game rules
  (`subgoal/table.py::replay`) and counts as optimal only if the replayed
  length equals `d*`.

## The learned `h`

`subgoal/ctgnet.py`. Target:

    h(s, R, C) = the exact number of moves still needed to put the target robot
                 on the target cell, from the CHILD state s' = s with robot R
                 moved to cell C.

Labels come from the project's own exact engine — `move_planner/oracle.py`'s
joint-state A\*, through its gate-verified Rust port `rust_datagen`
(`forward_instance`, `full_policy=false`, expansion cap 400 000). Never a
bound; a state whose solve hits the cap carries no label at all.

Architecture is deliberately Stage 2's, so the comparison is about the target
and not the model: the size-free `LoopedLayer` encoder, weight-tied, d_model
192, 4 heads, edge-masked attention, `pe="none"`, **recurrence 4** (at the
production 12 the Stage 2 net sat on the constant-value plateau for 30 epochs,
`STAGE2.md` §4), the 96-bin HL-Gauss head. Two changes: four input channels
(query robot / other robots / target robot / target cell) instead of two,
because cost-to-go depends on the puzzle; and a five-token readout (query robot
cell, candidate cell, target robot cell, target cell, global scratchpad). The
candidate cell stays out of the encoder, so one pass scores 256 children.
Loss: HL-Gauss cross-entropy on every labelled child plus the reference
listwise ranking loss applied to `c + h` with the exact `c` and the predicted
`h` — the quantity the beam orders by.

Corpus: parents are drawn on **train-split boards 1000–1699 only** (2 random
puzzles per board, 2 parent states per puzzle, each 0–3 random macro moves from
the drawn placement — Stage 2's recipe), validation on boards **1800–1849**.
Zero overlap with the benchmark boards 2400–2549. Checkpoint selected by the
best `val_top5` — the fraction of validation `(state, robot)` groups whose five
best-predicted children contain a child with the truly minimal `c + h` — which
is the beam's own criterion, computed on boards no training state came from.

**No knob is touched after a gate number is read.** Anything explored
afterwards is labelled exploratory and reported separately from the headline.

---

## Results

_(to be filled in by the run)_
