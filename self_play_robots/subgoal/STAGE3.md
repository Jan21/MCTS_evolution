# Learned subgoal discovery — Stage 3

`PLAN_SUBGOAL_DISCOVERY.md` §4, revised Stage 3, run 2026-08-28.

**Everything down to the "Results" line was committed before any Stage 3
training or planning job was submitted** (f807931, dab9d1a, 03c8872, 243d800),
so the gate, the headline configuration, the tie-breaking rule and the
checkpoint criterion are on record ahead of any number, and none of it has been
edited since a result was read. Deviations from it are listed under the
verdict.

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
| E | `net` / `relaxed` | as above | 5, 20 | **300** | the backward planner's own network-call budget: it spends one network pass per expansion, 1200 per instance; this planner spends four, so 300 expansions is the call-matched row |

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
best `val_top5` — the fraction of validation DECISIONS whose five
best-predicted children contain a child with the truly minimal `c + h`, where a
decision pools all four robots' candidates exactly as the beam does. (Amended
2026-08-28 before any Stage 3 training job was submitted: the metric was first
written per `(state, robot)` group, which saturates near 100% for every scorer
because a single robot offers only ~37 children; the pooled version is the
beam's own decision.) Computed on boards no training state came from.

**No knob is touched after a gate number is read.** Anything explored
afterwards is labelled exploratory and reported separately from the headline.

---

## Results

Every number below is recomputed this session from the payloads' own move
dumps: `subgoal/table.py` replays each solution under the real joint-game rules
(`simulate.slide`, a no-op slide illegal — the physics of
`eval/replay_validate.py`) and counts a puzzle optimal only when the replayed
length equals its `d*`. **0 replay failures across all 15 planner payloads**, 0
misaligned rows, 0 length disagreements, 0 solutions shorter than `d*`.

## Verdict

**Gate: PASS.** At the pre-registered protocol — 1200 expansions, k = 5, all 450
instances — the learned-`h` planner solves **60.7% of the 450 move-optimally
(273/450)** against the backward supervised planner's 53.3% (240/450). It is
also the best-quality subgoal planner in the project by a wide margin on its
solves (mean 0.708 extra moves against 1.840 and 1.995).

Two findings matter more than the gate.

1. **The learned `h` is worth about twenty points over the non-learned one, at
   every beam width.** Same search, same physics edges, same budget, only `h`
   changes: 60.7% vs **41.1%** at k = 5, 71.6% vs 59.3% at k = 10, 74.9% vs
   65.3% at 20, 79.6% vs 66.7% at 50, 79.8% vs 67.1% unpruned. This is the
   comparison the stage was told to make, and it is the clearest positive
   result the project has for a learned component: the network is not
   decorating a search that already worked.
2. **k = 5 is the wrong beam for this space, exactly as the plan suspected.**
   The arena's k = 5 was calibrated for a vocabulary of ~50 candidates; this one
   offers 1024, of which ~148 are physically reachable. Widening the beam at
   *identical* network cost (the candidate cell is read out, not encoded, so an
   expansion always costs 4 encoder passes whatever k is) lifts the learned
   planner from 60.7% to **79.8% (359/450)** and drops its mean extra moves from
   0.708 to 0.265.

### Deviations from the pre-registration

One, and it concerns the diagnostic row only. Row **D** (`arm=exact`) was
pre-registered at 1200 expansions and was run at **20**. Reason, measured, in
§4: with an exact `h` the first solution the search finds is already optimal and
further expansions cannot improve it, but the generic early stop proves
optimality with the weak admissible relaxation and cannot see that, so the arm
burns its whole budget breaking ties — 16.4 s per instance, about two hours for
the benchmark (job 4864344, cancelled at 50/450). No other row deviates, and no
knob was touched after a number was read. Everything explored beyond the
pre-registered set is labelled as such.

## 1. The plan's table

`python -m subgoal.stage3 report --row ...`, which is `subgoal/table.py` with a
fourth row. Budget 1200 expansions, k = 5 for every row.

| model | optimal % of 450 | solved / 450 | extra moves (on its solves) |
|---|---|---|---|
| forward (move-by-move, supervised) | 94.2% (424/450) | 450/450 | 0.067 (n=450) |
| backward (subgoal, supervised) | 53.3% (240/450) | 432/450 | 1.840 (n=432) |
| current self-play line (v14_stack nets) | 51.3% (231/450) | 439/450 | 1.995 (n=439) |
| **new: learned subgoal discovery** | **60.7% (273/450)** | 415/450 | **0.708 (n=415)** |

The first three rows reproduce their payload aggregates exactly (Stage 0 gate
PASS, PASS, PASS), so the fourth is measured on the same scale as the others.
At the beam this space actually wants the fourth row is **79.8% (359/450),
404/450 solved, 0.265 extra** — reported as a separate row in §3, never
substituted for the protocol row.

The forward move-by-move planner is still far ahead at 94.2%. Stage 3 does not
change that; it changes the standing of *subgoal* planning, which went from
53.3% to 60.7% at protocol and to 79.8% at the beam width proportionate to the
vocabulary.

## 2. Learned `h` against the non-learned one

### As planners (the comparison the stage was told to make)

Identical search, identical physics edges, identical budget and beam; the only
difference is `h`.

| | learned `CtgNet` `h` | relaxed any-stop `h` (no network) | difference |
|---|---|---|---|
| optimal % of 450, k=5 / 1200 exp | **60.7% (273)** | 41.1% (185) | **+19.6 pts** |
| solved / 450 | 415 | 381 | +34 |
| mean extra moves on its solves | 0.708 | 1.381 | −0.673 |
| optimal % at k=10 | 71.6% (322) | 59.3% (267) | +12.3 |
| optimal % at k=20 | 74.9% (337) | 65.3% (294) | +9.6 |
| optimal % at k=50 | 79.6% (358) | 66.7% (300) | +12.9 |
| optimal % at k=200 (unpruned) | 79.8% (359) | 67.1% (302) | +12.7 |

The gap is largest exactly where the plan predicted the difficulty is — a hard
top-5 prune — and it never closes. Per `d*` at k = 5, the learned `h` is even
with the relaxed one on the trivial instances (`d*` ≤ 3: 50/55 vs 50/55) and
pulls away as the puzzle deepens (`d*` = 6: 49/73 vs 26/73; `d*` = 8: 37/73 vs
20/73; `d*` = 9: 17/62 vs 13/62).

### As rankers, with no search at all

`python -m subgoal.stage3 rank`, on the validation corpus (boards 1800–1849, no
training state came from them). One decision = one state's whole candidate set,
all four robots pooled, exactly as the beam sees it; the true ordering is
`f = c + ctg` with the exact edge cost and the exact cost-to-go.

| scorer | Spearman rho vs the true cost-to-go | **top-1 of f = c + h** | **top-5 of f = c + h** | MAE (moves) |
|---|---|---|---|---|
| **learned `CtgNet` h** | 0.378 | **69.9%** | **93.8%** | **2.01** |
| any-stop relaxation h (no network) | **0.518** | 47.8% | 83.4% | 5.44 |
| h = 0 (rank by the edge cost alone) | — (constant) | 34.3% | 79.9% | 7.86 |

289 decisions, mean 39.3 labelled children per (state, robot) group.

This is the sharpest statement of what the network bought. The relaxed
heuristic is *more monotone* in the true cost-to-go than the network
(rho 0.518 vs 0.378) — but it under-estimates it by 5.4 moves on average,
unevenly. The beam does not rank `h`; it ranks `g + c + h` with an EXACT `c`,
so an `h` that is off by five moves in a way that varies across candidates
destroys the ordering however well it correlates. The learned `h` is
calibrated (MAE 2.01), and that is what turns 47.8% into 69.9% at the top of
the list and 83.4% into 93.8% inside a top-5 beam. Ranking by the edge cost
alone — the Stage 2 design, with no remaining-distance term at all — is worst
at 34.3% / 79.9%, which is the same ordering Stage 2's beam table found.

## 3. The beam-width study

Every row is all 450 instances, replay-certified. **The network arm spends four
encoder passes per expansion at every k**, because the candidate cell is read
out rather than encoded — so within a budget the k rows are matched on network
calls and differ only in the prune. The 300-expansion block is the *backward
planner's own* network-call budget: it spends one network pass per expansion,
1200 per instance, and this planner spends four, so 300 expansions is the
call-matched row. (The exact-`h` diagnostic row is in §4, not here — it is not
a system.)

| h | budget | beam k | optimal % of 450 | solved / 450 | extra moves | mean expansions used | heuristic queries | wall s |
|---|---|---|---|---|---|---|---|---|
| net | 1200 | 5 | 60.7% (273/450) | 415/450 | 0.708 | 581 | 1042562 | 462.5 |
| net | 1200 | 10 | 71.6% (322/450) | 394/450 | 0.312 | 779 | 1399052 | 624.9 |
| net | 1200 | 20 | 74.9% (337/450) | 386/450 | 0.332 | 853 | 1533459 | 699.0 |
| net | 1200 | 50 | 79.6% (358/450) | 402/450 | 0.291 | 891 | 1600604 | 768.0 |
| net | 1200 | 200 | 79.8% (359/450) | 404/450 | 0.265 | 898 | 1613270 | 924.4 |
| relaxed | 1200 | 5 | 41.1% (185/450) | 381/450 | 1.381 | 557 | 997420 | 92.6 |
| relaxed | 1200 | 10 | 59.3% (267/450) | 349/450 | 0.479 | 696 | 1250841 | 119.9 |
| relaxed | 1200 | 20 | 65.3% (294/450) | 345/450 | 0.348 | 687 | 1234515 | 131.8 |
| relaxed | 1200 | 50 | 66.7% (300/450) | 351/450 | 0.365 | 682 | 1225692 | 159.4 |
| relaxed | 1200 | 200 | 67.1% (302/450) | 354/450 | 0.379 | 679 | 1219133 | 270.6 |
| net | 300 | 5 | 54.7% (246/450) | 345/450 | 0.643 | 186 | 333296 | 155.0 |
| net | 300 | 20 | 60.0% (270/450) | 324/450 | 0.441 | 236 | 424630 | 199.5 |
| relaxed | 300 | 5 | 39.1% (176/450) | 308/450 | 1.159 | 195 | 349395 | 38.8 |
| relaxed | 300 | 20 | 52.2% (235/450) | 295/450 | 0.522 | 211 | 378284 | 46.9 |

Three things to read out of it.

* **Widening the beam is free in network calls and worth 19 points to the
  learned planner** (60.7 → 79.8) and 26 to the relaxed one (41.1 → 67.1).
  k = 5 was simply mis-sized for a 1024-candidate space.
* **Solved-count and optimality trade off, and only for the weak heuristic.**
  The relaxed arm *loses* solves as k grows (381 → 345) because a wider beam
  spends the same 1200 expansions on a broader, shallower tree and can no
  longer reach deep solutions. The learned arm does not pay that price above
  k = 10 (394 → 404): a better `h` keeps the extra breadth pointed the right
  way.
* **Even at the backward planner's own network-call budget the learned planner
  wins**: 54.7% at k = 5 and 60.0% at k = 20 with 300 expansions, against 53.3%.

The residual failure is depth, not vocabulary. At k = 50 the learned planner is
perfect on every instance with `d*` ≤ 5 (148/148) and 68/73 at `d*` = 6, then
falls to 46/73 at `d*` = 8 and 28/62 at `d*` = 9; 312 of its 450 searches end on
the expansion budget rather than with an optimality proof.

## 4. Diagnostic: is the search or the heuristic at fault?

With an EXACT `h` every child's `f = g + c + h` is the true cost of the best
solution through that child, so the optimality-preserving children always carry
the minimal `f` and always survive the prune. That makes `arm=exact` a clean
ceiling on the SEARCH: whatever it fails at is the search's fault, not the
heuristic's. It is not a system — one exact engine solve per candidate child,
1 230 492 of them for this row.

| h | beam k | expansions | optimal % of 450 | solved / 450 | extra moves |
|---|---|---|---|---|---|
| exact engine cost-to-go | 5 | 20 | **88.4% (398/450)** | 400/450 | **0.005** |

**The search is sound and the heuristic is the binding constraint.** At the same
k = 5 that gives the learned planner 60.7% and the relaxed one 41.1%, an exact
`h` reaches 88.4% — and of the 400 puzzles it solves at all, **398 are optimal**
(99.5%). Per `d*` it is perfect through `d*` = 6 (221/221) and 67/69 at 7.

The 20-expansion cap is the whole of its remaining shortfall, and it is a cap on
*finding* a solution, never on its quality: 322 of the 450 searches end on the
budget, and the 50 unsolved instances are the deep ones (`d*` = 9: 39/62,
`d*` = 10: 8/21). 88.4% is therefore a LOWER bound on what an exact `h` would
reach at the protocol's 1200 expansions. The cap is 20 rather than 1200 because
the generic early stop proves optimality with the weak admissible relaxation and
cannot see that this `h` is exact, so the arm otherwise burns its whole budget
breaking ties between equally-optimal nodes — measured at 16.4 s per instance,
about two hours for the benchmark (job 4864344, cancelled at 50/450). Twenty
expansions is far above the 1–5 macro moves an optimal solution needs (Stage 1),
and with an exact `h` extra expansions cannot improve a solution that has
already been found.

Read together with §2 and §3: the search finds and certifies the optimum
whenever the ranking puts an optimal child in the beam, the learned `h` puts it
there 93.8% of the time on a held-out decision against the relaxation's 83.4%,
and the gap between 88.4% (exact `h`) and 79.8% (learned `h`, unpruned) is what
a better heuristic is still worth.

## 5. The heuristic itself

Target, corpus and architecture are as pre-registered. The exact engine
(`move_planner/oracle.py` through its gate-verified Rust port) labelled
**421 208** reachable children of 2400 parent states on train-split boards
1000–1699 and **53 012** on validation boards 1800–1849; **335 280 (79.6%)** and
**44 486 (83.9%)** got an exact cost-to-go, the rest hit the 400 000-expansion
cap and carry no label at all. Mean labelled cost-to-go 7.58 / 7.40, max 20 /
15. Board overlap: train↔val **0**, train↔bench450 **0**, val↔bench450 **0**
(699 and 50 distinct boards).

**No collapse this time.** At the pre-registered recurrence 4 the net trained
normally: `val_group_spread` 0.83–1.86 throughout (Stage 2's collapsed runs sat
at exactly 0.00), and the beam metric rose from 0.817 at epoch 0 to its maximum
**0.9377 at epoch 11**, i.e. 93.8% of held-out decisions keep a truly optimal
child inside a top-5 beam.

| epoch | 0 | 5 | **11** | 20 | 30 | 39 |
|---|---|---|---|---|---|---|
| `val_top5` (pooled decision) | 0.817 | 0.924 | **0.938** | 0.900 | 0.900 | 0.855 |
| `val_top1` | 0.488 | 0.671 | **0.699** | 0.637 | 0.581 | 0.609 |
| `val_mae_group` (moves) | 1.86 | 1.78 | 2.01 | 2.11 | 2.45 | 3.07 |
| `val_group_spread` | 1.05 | 0.92 | 1.17 | 1.15 | 1.57 | 1.59 |

After epoch ~25 the net overfits (MAE climbs steadily while the ranking metric
drifts down), so the checkpoint is a best-of-40 selection on a validation set
disjoint from both the training boards and the benchmark boards. That is
legitimate early stopping, but it should be said plainly: the reported planner
uses epoch 11, not the last epoch, and the last epoch would have scored `val_top5`
0.855 rather than 0.938.

## 6. Compute

| what | job | partition | GPUs | elapsed | node-hours |
|---|---|---|---|---|---|
| exact-engine labelling of 474 220 children | 4863820 | qgpu_exp | 2 | 38 m 05 s | 0.159 |
| timing smoke (sizing the plan legs) | 4863872 | qgpu_exp | 1 | 3 m 15 s | 0.007 |
| train the cost-to-go net, 40 epochs | 4863894 | qgpu_exp | 1 | 12 m 52 s | 0.027 |
| plan leg A (gate row + control + relaxed beam study) | 4863895 | qgpu_exp | 1 | 21 m 12 s | 0.044 |
| plan leg B (learned beam study) | 4863896 | qgpu_exp | 1 | 51 m 01 s | 0.106 |
| plan leg C (call-matched 300-expansion rows) | 4863903 | qgpu_exp | 1 | 7 m 52 s | 0.016 |
| exact-`h` diagnostic | 4864942 | qgpu_exp | 1 | 20 m 56 s | 0.044 |
| *wasted:* exact-`h`, first attempt (pipe deadlock) | 4863897 | qgpu_exp | 2 | 34 m 57 s | 0.146 |
| *wasted:* exact-`h`, second attempt (200-expansion cap, too slow) | 4864344 | qgpu_exp | 1 | 23 m 30 s | 0.049 |
| ranking analysis (`rank`) | — | login node, 4 cores | 0 | 36 s | 0 |
| **total** | | | | **4 h 47 m of A100 GPU time** | **0.598** |

Stage 3's budget was 3 node-hours; it used **0.598**, of which 0.195 was wasted
on two failed attempts at the diagnostic leg (jobs 4864209 and 4864210 were
cancelled before starting and cost nothing). The first failure was a real bug:
the streaming driver for the exact engine wrote a whole round of ~9500 queries
before reading any result, filled the engine's stdout pipe and deadlocked both
sides; fixed by draining stdout on a dedicated reader thread
(`subgoal/rustexact.py::ExactCTG`). The second was a mis-sized budget, described
in §4.

## 7. What this says about Stage 4

The plan's Stage 4 replaces the exact-engine labels with certified self-play
labels and asks whether the optimal percentage rises across rounds. On this
stage's evidence that is **worth running, and the interesting version of it is
not the one the plan describes**.

For it:

* the fourth row is real and beats both subgoal baselines, so there is a
  working planner to run self-play *inside*, which is the thing Stage 2 could
  not offer;
* the learned `h` is the component doing the work (§2), and `h` is exactly what
  self-play produces labels for: every certified plan the planner finishes
  gives the true replayed cost-to-go of every state on it, with no oracle. The
  label the loop would generate is the same object the supervised run just
  trained on, which is the cleanest possible setting for a self-play round;
* the supervised checkpoint peaked at epoch 11 of 40 on 335 280 labels and then
  overfitted, so this `h` is data-limited rather than capacity-limited — more
  labels are the obvious lever, and self-play makes them for free.

Against it, and this is the part that should shape the design:

* **the exact-`h` ceiling is 88.4% and the learned `h` already reaches 79.8%.**
  The whole remaining headroom for a better heuristic at this budget is under
  nine points. A self-play loop that improves `h` cannot do better than that
  unless the search changes too;
* **the biggest single win in this stage cost nothing and was not learning at
  all**: sizing the beam to the vocabulary moved the learned planner 19 points
  (60.7 → 79.8) at identical network cost. Before another training loop, the
  arena convention (`k = 5`, inherited from a ~50-candidate vocabulary) should
  be re-set for this space;
* **the failures are depth failures.** 312 of 450 searches at k = 50 end on the
  expansion budget, and the exact-`h` arm loses its 50 unsolved instances to a
  budget on *finding* a solution, not on its quality. More accurate `h` does not
  fix a search that runs out of expansions on `d*` ≥ 9;
* the early stop is weak: it proves optimality with the board-only relaxation,
  which is why even a perfect `h` cannot terminate. A tighter bound would
  convert expansions into solved instances at no learning cost.

**Judgement.** Run Stage 4, but re-order it: first re-size the beam and
strengthen the optimality bound (both free, both measured here to be worth more
than the last training run), then run self-play against *that* planner, and gate
it on beating 79.8% rather than 60.7%. Running self-play against the k = 5
configuration would be measuring a 19-point handicap.

---

## Provenance appendix

Every number above, with the file it came from and how it was read.

### Code (all committed before the runs that used it)

| file | what it fixes |
|---|---|
| `self_play_robots/subgoal/planner.py` | the search, the three heuristics, the lockstep driver, the payload format |
| `self_play_robots/subgoal/ctgnet.py` | the cost-to-go target, the net, the two losses, the pooled beam metric |
| `self_play_robots/subgoal/rustexact.py` | the exact-engine driver: board sidecars, bulk labelling, the streaming diagnostic |
| `self_play_robots/subgoal/stage3.py` | `gen` / `train` / `plan` / `rank` / `beam` / `report` / `verify`; `GATE_BACKWARD_PCT = 53.3` |
| `self_play_robots/subgoal/table.py` | the four-row table, unchanged from Stage 0 |
| `self_play_robots/jobs/subgoal_stage3_{gen,train,plan,exact,rank,smoke}.slurm` | the legs |

The gate, the headline configuration, the tie-breaking rule and the checkpoint
criterion were committed (f807931, dab9d1a, 03c8872, 243d800) **before any
Stage 3 training or planning job was submitted**. Nothing was changed after a
result was read.

### Physics

| claim | how it was checked |
|---|---|
| the search's fast slide is `simulate.slide` | `python -m subgoal.stage3 verify --n 20 --states 10`: 800 (state, robot) groups compared cell-for-cell and cost-for-cost against Stage 1's `subgoal/space.py::rest_cells`, **0 mismatches** |
| the fast physics does not change any result | the 20-instance relaxed smoke run before and after the rewrite is byte-identical on move sequences, expansions and replay lengths (2.5x faster) |
| every reported solution is legal and its length is what is counted | `subgoal/table.py::replay` on every row of every payload: **0 replay failures in 15 payloads** |
| the exact labeller agrees with the reference oracle | the Rust engine's `cost_to_go` on the first 12 bench450 roots equals their `d*` (12, 4, 7, 1, 4, 6, 2, 4, 10, 6, 1, 7) |

### Data

| number | file | how it was read |
|---|---|---|
| the three reference rows | `results/fwd_m0/astar_candidate_scored.json`, `results/m0/g16r4_b1s21_b2flags.json`, `results/subgoal/v14_stack_g16r4_bench450_astar.json` | `python -m subgoal.table`, replaying every move dump; Stage 0 gate PASS on all three |
| the fourth row and the whole beam study | `results/subgoal/stage3/plan_{net,relaxed}_k{5,10,20,50,200}_e{1200,300}.json` | `python -m subgoal.stage3 beam`, `... report --gate`; summary dumped to `results/subgoal/stage3/beam_summary.json` and `table4.json` |
| stop reasons and the per-`d*` breakdown | the same payloads | `subgoal/_stage3_diag.py` |
| training and validation corpora | `results/subgoal/stage3/{train_b1000-1699,val_b1800-1849}.npz` | `python -m subgoal.stage3 gen`; label counts, cost-to-go statistics and the three board-overlap checks read directly off the arrays |
| the training curve | `runs/spr/subgoal/stage3_rec4/lightning_logs/version_0/metrics.csv` | per-epoch `val_top5` / `val_top1` / `val_mae_group` / `val_group_spread` |
| the checkpoint | `runs/spr/subgoal/stage3_rec4/ctg-epoch=11-val_top5=0.9377.ckpt` | selected by `ModelCheckpoint(monitor="val_top5", mode="max")` |
| the ranking comparison | `results/subgoal/stage3/rank_val.json` | `python -m subgoal.stage3 rank --data val_b1800-1849.npz` (36 s of login-node CPU, no GPU) |
| the exact-`h` diagnostic | `results/subgoal/stage3/plan_exact_k5_e20.json` | `python -m subgoal.stage3 plan --arm exact --k 5 --expansions 20` |
| job elapsed times | `sacct -j ... -X` | the compute table |

### Logs

`runs/spr/spr-sg3-gen-4863820.out`, `spr-sg3-smoke-4863872.out`,
`spr-sg3-train-4863894.out`, `spr-sg3-planA-4863895.out`,
`spr-sg3-planB-4863896.out`, `spr-sg3-planC-4863903.out`,
`spr-sg3-exact-4863897.out` (the deadlock), `spr-sg3-exact-4864344.out`
(the mis-sized cap) and `spr-sg3-exact-4864942.out` (the reported diagnostic).
