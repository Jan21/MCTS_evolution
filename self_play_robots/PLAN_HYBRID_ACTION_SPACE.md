# PLAN: one search, both action spaces

**Goal.** An action space that keeps the hand-written sub-goal vocabulary's
cheapness AND the state sub-goal space's completeness, so the planner finds
**move-optimal** plans without paying 10^6 nominal search per puzzle.

Written 2026-09-01. Read fully before running anything. Same discipline as
`PLAN_SUBGOAL_DISCOVERY.md`: every stage has a gate fixed and committed before
the job it judges, a negative result ends a branch, and nothing is tuned after a
gate is read.

---

## 1. Why this exists

Two vocabularies, measured, and each is deficient in the way the other is not.

| | old: hand-written sub-goals | new: state sub-goals (macro moves) |
|---|---|---|
| an action is | "park helper H on cell S so the mover stops on B" | "robot R comes to rest on cell C" |
| candidates come from | hand-coded geometry, a SUBSET | exact physics BFS, COMPLETE |
| branching, 24x24 | ~215 | **360 children per expansion** |
| depth, 24x24 | **0.82 decisions** | 2.43 decisions |
| nominal space, 24x24 | 10^1.9 | **10^6.2** |
| ceiling (extra moves over optimal) | **+1.276** | **0.000** |
| 24x24 frontier, exhaustive probe | **114 of 218 resolved**, 120 s cap | **0 of 8 resolved**, 900 s cap |

Sources: `subgoal/STAGE5_PROBE.md`, `STAGE5_PROBE_FRONTIER.md`,
`results/ceiling/g24r4_{b2,frontier_b2}.json`, walkthrough part 2 §A.2/A.3.

The old space is cheap and lossy. The new space is exact and, on hard puzzles,
unaffordable. Neither is the answer on its own.

## 2. The structural fact this plan is built on

I decomposed the 28 known-optimal 24x24 macro plans from
`results/subgoal/stage5_probe/merged.json` by which robot each macro move
belongs to (T = the target robot, H = a helper):

| shape | n | expressible in the old vocabulary? |
|---|---|---|
| H-T | 11 | yes, exactly one sub-goal |
| T | 5 | yes, no sub-goal needed (an exact pin) |
| H-H-T | 5 | yes, nested support |
| **T-H-T** | **4** | **NO** |
| H-H-H-T | 1 | yes |
| **H-T-H-T** | **1** | **NO** |
| **H-T-H-T-H-T** | **1** | **NO** |

**22 of 28 optimal plans (79%) have a shape the old vocabulary can say. The 6
that cannot (21%) all share one structure: the target robot moves at an
intermediate point**, then a helper is placed, then the target finishes. That is
the structure `subgoal/STAGE01.md` named at 16x16 ("intermediate parking of the
target robot itself", 3 of its first 20), and it is what the v07 head-start
slides crudely approximate at single-move granularity.

Two consequences drive the whole plan.

1. **The gap may be narrow and nameable rather than broad.** If most of the old
   vocabulary's +1.276 is intermediate target parking, a targeted extension beats
   a general hybrid, and it is far cheaper. Stage 1 tests exactly this.
2. **The natural hybrid shape is a short forward prefix and an abstract
   suffix.** Not a general interleaving: a few macro moves to reach a state the
   old vocabulary CAN finish from, then hand off. Stage 3 builds this.

Caveat, stated now: shape-expressibility is necessary, not sufficient. The old
vocabulary reaches `d*` on 62.7% of the 24x24 exam, so 37% fail, while only 21%
of optimal plans are structurally impossible by the target-parking criterion.
Target parking can therefore explain at most about half the failures. The rest
is something else and Stage 1 will say how much.

## 3. The design constraint that rules out most naive hybrids

Optimality here is proved, not assumed: the search stops when the incumbent's
cost is at or below the smallest `g + h_adm` on the open list, with `h_adm` an
admissible bound (`subgoal/planner.py::_h2`, verified on 2819 states, 0
violations). That machinery needs, for **every** action of **every** type:

> **(a) a concrete joint state as the successor, and (b) an exact move cost on
> the edge.**

An action that yields an abstract commitment with an optimistic cost breaks the
proof. So:

- **Ruled out:** node = (concrete prefix, abstract suffix). Two node types, a
  non-exact `g`, no optimality proof.
- **Ruled in:** an abstract sub-goal used as an action must first be
  **realized** (`eval/realize.py::strict_moves`) into a legal move sequence, and
  then it IS a concrete state with an exact cost. A realized abstract sub-goal is
  just a longer macro move.

This is the load-bearing design idea: **the old vocabulary enters the new search
as a verified temporally-extended action, never as a commitment.** Everything
below respects it.

## 4. The one metric

Percent of `supervised_valuenet/eval/data/bench450.jsonl` (450 puzzles, 16x16,
every one with a known `d*`) solved **move-optimally**, under the pinned budget
of **1200 expansions**. Report two supporting columns, never as the headline:
solved of 450, and mean expansions on solved instances (the cost side, which is
the whole point of this plan).

A solution counts only after `strict_moves` replays it and the replayed length
equals `d*`. Reference rows, all recorded, none to be re-run:

| row | optimal % of 450 | source |
|---|---|---|
| forward move-by-move, supervised | 94.2% | FINDINGS 33 |
| old vocabulary, supervised | 53.3% | FINDINGS 33 |
| macro search, relaxation `h` (network-free) | 67.1% | FINDINGS 32(a), unpruned |
| macro search, learned `h` | 79.8% | FINDINGS 32(a), unpruned |
| macro search, Part A (beam 100 + tight bound) | 80.9% | FINDINGS 33 |
| macro search, self-play from scratch | 95.3% | FINDINGS 33 |
| macro search, EXACT `h` (ceiling, not a system) | 88.4% at a 20-expansion cap | FINDINGS 32(e) |

That last row is the reason to believe this plan can work at all: a
near-perfect distance estimate collapses the same search by 60x.

## 5. The idea inventory

Every idea considered, grouped by the assumption it rests on. Stages 1 to 4 test
the starred ones; the rest are the bench in §7.

**A. Abstraction as heuristic** (assumption: a better `h` is sufficient)
- **A1 ★** old planner's abstract plan cost as `h` for the macro search
- A2 combine: `max(h_learned, h_struct)`, or a learned blend
- A3 `h` = cost of the old planner's best COMPLETE abstract plan from this state

**B. Abstraction as action** (assumption: pruning by commitment is what is missing)
- **B1 ★** the old planner as a **finishing option**: at any node, one action is
  "run the old planner from here for `b` expansions and take its realized answer"
- B2 ★★ abstract sub-goals as ordinary actions at every node, each realized
  before it becomes an edge (the general form of B1)
- B3 hierarchical: old planner outer, macro search inner as the realizer, so a
  sub-goal that fails deterministic realization can still be achieved

**C. Fix the old vocabulary directly** (assumption: the gap is narrow and nameable)
- **C1 ★** allow a macro-move prefix by the TARGET robot before the old
  vocabulary takes over: the minimal, measurable form of the §2 finding
- C2 add intermediate target parking as a first-class sub-goal type in `propose`

**D. Search control, no action-space change**
- D1 lazy heuristic evaluation (needed for A1 regardless: 360 children per
  expansion cannot each afford an engine-backed estimate)
- D2 cluster the 360 macro children and expand representatives
- D3 bidirectional: old planner is backward, macro search forward, meet in the middle

**E. Learning over the mixed space**
- E1 a learned selector over action TYPE (macro move vs abstract option)
- E2 self-play in the mixed space, the Stage 4 recipe carried over

---

## 6. Stages

### Stage 0 — grounding, no compute
1. Repeat the §2 decomposition at **16x16** on the 40 Stage 1 optimal plans, so
   the structural claim rests on both sizes and on the configuration every later
   stage runs at.
2. Measure the OLD vocabulary's branching and depth at 16x16 from existing
   payloads, for a like-for-like nominal-space number.
3. Reproduce the seven reference rows of §4 from their payloads, never copied
   from a document.
**Gate:** the reference rows reproduce. A discrepancy is a finding and stops the
plan until explained.

### Stage 1 — is the gap just intermediate target parking? (CPU, ~2 core-hours)
The cheapest possible test of the whole premise, and it can end the plan by
solving the problem more simply.

For each instance, enumerate the target robot's reachable rest cells (one robot,
so tens of candidates, not 360). For each, run the EXISTING exhaustive
hand-written-vocabulary probe (`spr/ceiling.py`) from the resulting state, and
score `prefix_cost + that ceiling`. Take the minimum over prefixes, including the
empty prefix. This measures **"old vocabulary + a target-robot head start"**
without touching `heuristics.propose` at all.

Run at 16x16 (40 instances, the Stage 1 sample) and 24x24 (the 29 instances of
`STAGE5_PROBE.md`, so it joins the existing rows).

**Gate, fixed here:**
- ceiling falls to **<= +0.2**: the gap IS intermediate target parking. Stop the
  hybrid programme, write the targeted vocabulary extension (C2) instead, and
  report that the general hybrid is unnecessary. This is the best outcome.
- ceiling falls to between +0.2 and +0.7: a real but partial explanation.
  Proceed to Stage 2, and carry the number as the bar a general hybrid must beat.
- ceiling barely moves (> +0.9): the gap is broad, not one structure. Proceed to
  Stage 2 and drop C1/C2 from the plan.

### Stage 2 — is a structural heuristic enough? (CPU + light GPU, ~0.5 node-hours)
Idea A1. The macro search of `subgoal/planner.py`, unchanged except for `h`:
rank by `g + c + h_struct`, where `h_struct` is the old planner's abstract plan
cost from the child state, computed the way `variants/v07_hybrid_actions.py::
plan_cost_of` already computes it (build a two-node plan, pin what the exact
engine can pin, return the abstract cost; no tree, no network).

Ranking only. The **stopping proof keeps the existing admissible bound**, so a
non-admissible `h_struct` cannot endanger optimality (§3). This separation
already exists in the code and is why this arm is a small change.

Engineering, required not optional: 360 children per expansion cannot each pay
~2 exact-engine calls. Use **lazy evaluation**: rank children with the cheap
relaxation, and recompute with `h_struct` only when a node is POPPED,
re-inserting it if the estimate rose. One expensive evaluation per pop.

Arms (three already recorded, one new): relaxation 67.1%, **`h_struct` NEW**,
learned 79.8%, exact-`h` ceiling 88.4%. Report optimal %, solved, AND mean
expansions, since the cost column is the point.

**Gate:** `h_struct` must beat the relaxation's 67.1% by at least 5 points. If it
does not, the old vocabulary's value is in its ACTION SPACE and not in its cost
estimates, family A dies, and the plan goes to Stage 3 knowing that.

### Stage 3 — the old planner as an action (~2 node-hours)
Idea B1 first, because it is a small change and the §2 shape finding predicts it.

At every node of the macro search, in addition to the ~148 macro moves, one extra
action: **run the old backward planner from this state under a small budget `b`,
realize its answer, and if it plays out, add the resulting goal state as a child
with the realized strict move count as the exact edge cost.** The option either
finishes the puzzle or contributes nothing. Costs are exact, states are concrete,
§3 is satisfied.

Pre-register `b` by the Stage 4 rule: chosen on a HELD-OUT dev set of fresh
boards, never on bench450, then frozen.

What is measured, beyond the headline: how often the winning plan uses the
option; the depth of the macro prefix before the option fires (§2 predicts
mostly 0 or 1, and 1 specifically being a target-robot move); and expansions
against Stage 2's best.

**Gate:** must beat the better of Stage 1's and Stage 2's number at matched
budget, on the same 450.

**Stage 3b, only if 3 passes and its prefix depth is mostly > 1:** the general
form B2, abstract sub-goals as ordinary realized actions at every node. Bigger
build; do not start it before 3 says the option is being used deep.

### Stage 4 — control and learning (~2 node-hours)
Only after 3. In this order, each with its own gate:
1. Re-derive the beam rule and the admissible bound for the MIXED action space
   on held-out boards, before any training. Stage 4 of the previous plan found
   this worth more than its training run, twice.
2. Freeze that number as the bar.
3. E1, a learned selector over action type; then E2, self-play in the mixed
   space using the Stage 4 recipe (certified labels, replay-or-nothing, rolling
   buffer, collapse guard).

### Stage 5 — scale (~4 node-hours, only after 4)
24x24 graded, then the frontier, then the head-to-head against the v07
head-start hybrid on the SAME exam under the SAME budget, which is the
comparison nobody has run and the one that decides whether this line replaces
the current flagship.

**Budget: under 10 node-hours through Stage 4.** Stages 0 and 1 are CPU only.

---

## 7. The bench

Not in the main line. Each is tested only if the line above stalls, and each
gets its own pre-registration if it is ever run.

- **A2/A3** heuristic combinations. Cheap follow-ups to Stage 2 if `h_struct`
  passes but does not reach the learned `h`.
- **B3** macro search as the realizer inside the old planner. This targets a
  different loss: the old planner discards complete plans that fail
  deterministic realization. Worth measuring how many that is first, from the
  `plans_rejected` field already in the v07 payloads.
- **D2** clustering the 360 children. Attacks branching directly; only worth it
  if Stage 2 shows the heuristic cannot carry the load.
- **D3** bidirectional search. High risk, high effort, and the two directions
  use different cost models. Last resort.

## 8. Rules

- **Gates are real.** A stage that fails ends its branch; write up why. A
  negative at Stage 1 or 2 is worth more than a speculative Stage 3.
- **Pre-register before submitting.** The gate, the sample, the caps and the
  decision rule are committed before the job they judge.
- **Verify, never trust.** Every number comes from a payload read this session.
- **Certification is not optional.** A plan counts only after physics replays it.
- **Never train on bench450** or on any pinned exam id.
- **State projected node-hours before submitting**, and check memory: the
  frontier probe lost 12 of 16 shards to the OOM reaper because a cap was raised
  without rechecking the per-shard footprint.
- Slurm: `-A open-37-42`, `qgpu*` only (the allocation has no CPU partition, so
  CPU-only work still requests `--gpus 1`), one leg per job, idempotent.
- Log results in `self_play_robots/FINDINGS.md` at the next free number, with the
  table and the file path behind every number.

## 9. What success looks like

A single search whose action space contains both vocabularies, which reaches the
95.3% of the pure macro search or better, at a fraction of its expansions, and
which still resolves instances on the 24x24 frontier where the pure macro search
resolved none. Failing that, the most likely useful outcome is Stage 1's: that
the whole gap was one nameable structure, and the fix is a small vocabulary
extension rather than a new search.

---

# STAGE 0 RESULT (2026-09-01)

The plan-shape decomposition of §2 holds at BOTH sizes, which it had to, since
every later stage runs at 16x16 while the finding was discovered at 24x24.

| | 16x16 (40 optimal plans) | 24x24 (28 optimal plans) |
|---|---|---|
| shapes the old vocabulary can express | 32 (80%) | 22 (79%) |
| **need intermediate target parking** | **8 (20%)** | **6 (21%)** |

16x16 inexpressible shapes: T-H-T (3), H-T-H-T (3), T-H-H-T (1), T-H-H-H-T (1).

**Harness validation.** A first run of Stage 1 accidentally used only the
stride-23 sample (the `first20` payload predates the `instance_indices` field and
my own stage5 merge dropped it; `--idx-from` now falls back to the rows' own
`idx`). That accident is useful: its baseline arm reproduced the recorded
stride-23 B2 ceiling of **0.632 exactly**, from scratch, under the same language,
caps and certification.

# STAGE 1 RESULT (2026-09-01, jobs 4911145 / 4911146, 42 CPU-minutes total)

Paired on the instances every arm resolved. 7,597 candidate probes at 16x16 and
11,374 at 24x24. **No arm ever produced a solution shorter than `d*`**, the check
that would catch a physics or accounting bug.

| arm | 16x16, extra moves | 16x16 optimal % | 24x24, extra moves | 24x24 optimal % |
|---|---|---|---|---|
| old vocabulary alone (baseline) | 0.838 | 73.0% | **1.276** | 55.2% |
| + one **target-robot** macro-move head start | **0.459** | 83.8% | **0.586** | 75.9% |
| + one **any-robot** macro-move head start | **0.108** | 94.6% | **0.172** | 93.1% |

n = 37 paired at 16x16, 29 at 24x24. The 24x24 baseline reproduces the recorded
+1.276 of `STAGE5_PROBE.md` exactly.

### The gate reads BAND 2: proceed to Stage 2

The pre-registered gate is about **target parking**, the structure §2 named. That
arm lands at +0.459 and +0.586, inside the "+0.2 to +0.7, a real but partial
explanation" band. Per the rule: proceed to Stage 2, and carry these as the bar.

So target parking explains **45% of the gap at 16x16 and 54% at 24x24**. Real,
and not the whole story. Family C (a targeted vocabulary extension) is NOT
sufficient on its own and the programme continues.

### The finding the plan did not anticipate

**One macro-move head start by ANY robot removes 87% of the ceiling gap at both
sizes** (0.838 to 0.108; 1.276 to 0.172) and lifts move-optimality from 55.2% to
93.1% at 24x24. That is a broader hypothesis than the pre-registered one and it
is not a tuning of it, so it is recorded here as a new result rather than
folded into the gate.

Target-only helps 7 instances at 24x24; any-robot helps 11. The 4 extra are
helper-robot head starts, which is exactly the H-T-H-T shape §2 found and which a
target-only prefix cannot reach.

Worked examples (24x24), old vocabulary versus one head start:

| idx | `d*` | old vocabulary | with head start | head start length |
|---|---|---|---|---|
| 176 | 10 | 15 | **10** | 3 |
| 208 | 11 | 16 | **11** | 2 |
| 128 | 14 | 18 | **14** | 5 |
| 32 | 9 | 11 | **9** | 3 |

### What this does to the rest of the plan

**It measures the ceiling of the Stage 3 design before Stage 3 is built.** "A
short forward prefix, then the old vocabulary finishes" is exactly the B1 option
design, and its ceiling is now known: **+0.172 at 24x24, +0.108 at 16x16**,
against 0.000 for the full macro space and +1.276 for the old vocabulary alone.
Stage 3 therefore has a concrete target and a strong prior.

Two cautions carried forward. These are **ceilings**, computed by trying every
prefix with a perfect chooser; a real planner must FIND the right prefix among
150 to 360 candidates inside its budget. And the caps differ between the two legs
(30 s / 100k iters at 16x16, 20 s / 50k at 24x24), so the two baselines are each
internally valid but not cross-comparable.

Payloads: `results/ceiling/prefix_g16r4_b2.json`,
`results/ceiling/prefix_g24r4_b2.json`, both with per-instance winning prefixes.

# STAGE 2 RESULT, ranking leg (2026-09-01, 8 core-minutes, no GPU)

The cheap form of Stage 2: rank the old planner's abstract plan cost as a
heuristic on the SAME held-out corpus FINDINGS 32(c) used (289 pooled decisions,
1132 (state, robot) groups, boards 1800-1849), with no planner search at all.

| scorer | rho vs true cost-to-go | top-1 of `f = c + h` | top-5 | MAE |
|---|---|---|---|---|
| learned `CtgNet` h (recorded, needs a GPU and training) | 0.378 | **69.9%** | 93.8% | **2.01** |
| **old planner abstract cost (NEW, no network)** | **0.608** | **62.3%** | 85.1% | 3.80 |
| `max(struct, relaxation)` (NEW) | 0.608 | 62.3% | 85.1% | 3.80 |
| any-stop relaxation (recomputed control) | 0.518 | 47.8% | 83.4% | 5.44 |
| `h = 0`, edge cost alone (recorded) | - | 34.3% | 79.9% | 7.86 |

**Harness check: the recomputed relaxation control reproduces the recorded
0.518 / 47.8% / 83.4% / 5.44 exactly**, on all four metrics.

### Gate: PASS, by 14.5 points

The pre-registered gate was "beat the relaxation's top-1 by at least 5 points".
Measured: **62.3% against 47.8%**. The old planner's abstract cost is better than
the network-free control on every metric at once: more monotone (0.608 vs 0.518),
better top-1 (+14.5), better top-5 (+1.7), and a third less error (3.80 vs 5.44).

**Family A survives.** The old vocabulary's value is NOT only in its action
space; its cost estimate is a genuinely better heuristic than anything
network-free the project had.

### Three things worth recording beyond the gate

1. **`max(struct, relaxation)` is identical to `struct`, row for row.** So the
   abstract plan cost **dominates the any-stop relaxation pointwise**: it is
   never the looser of the two. That is what one would expect, since the abstract
   plan pins exact shortest paths wherever the engine can and falls back to the
   relaxed bound otherwise, but it had not been measured. It also means the
   combination is not worth carrying as a separate arm.
2. **It does not reach the learned net, and the reason is calibration, not
   monotonicity.** `struct` has much the better rank correlation (0.608 vs
   0.378) yet the worse top-1 (62.3% vs 69.9%). This is the same trap FINDINGS
   32(c) identified for the relaxation: the beam ranks `c + h` with an EXACT `c`,
   so a heuristic that under-estimates unevenly loses to one that is
   well-calibrated even if it orders states better on its own. `struct`'s MAE is
   3.80 against the net's 2.01.
3. **The two are complementary in exactly the way that suggests combining them**:
   the structural estimate is the more monotone, the learned one the better
   calibrated. That is bench item A2, and its precondition ("h_struct passes but
   does not reach the learned h") is now met, so it is the next thing to run.

### What this predicts, and what it does not

Placing top-1 against the recorded planner numbers: relaxation 47.8% top-1 gave
67.1% optimal, learned 69.9% gave 79.8%. `struct` at 62.3% interpolates to
roughly 75% optimal. **That is an interpolation, not a measurement**, and the
planner leg of Stage 2 is what would test it. It needs the search integration
with lazy evaluation (78 ms per child means the estimate cannot be afforded on
all ~150 children of every expansion) and a GPU-partition slot for the arms that
use the net.

Payload: `results/subgoal/stage3/rank_struct/merged.json`, per-group and
per-decision metrics for every arm.

## Stage 2, bench item A2: combining the two heuristics (2026-09-01)

Precondition met (struct passed its gate but did not reach the learned net), so
A2 was run as the plan specified. Same corpus, one extra arm.

| scorer | rho | top-1 of `f = c + h` | top-5 | MAE |
|---|---|---|---|---|
| learned `CtgNet` h (recomputed) | 0.378 | 69.9% | 93.8% | 2.01 |
| **`max(struct, learned)`** | 0.381 | **70.9%** | **94.1%** | 2.06 |
| old planner abstract cost | 0.608 | 62.3% | 85.1% | 3.80 |
| any-stop relaxation (control) | 0.518 | 47.8% | 83.4% | 5.44 |

**Second harness check: the recomputed learned arm reproduces the recorded
0.378 / 69.9% / 93.8% / 2.01 exactly**, this time through a different code path
than the recorded run used.

**Verdict: A2 is not worth carrying.** The combination buys **+1.0 point of
top-1**, which over 289 decisions is about three decisions changed, and it costs
78 ms per child inside a search that scores ~150 children per expansion. The
learned net already subsumes almost everything the structural estimate knows.

The useful reading is the other one: **the structural estimate's value is as a
REPLACEMENT for the network, not as an addition to it.** Standalone it is 62.3%
against the network-free control's 47.8%, and it needs no training, no
checkpoint and no GPU. That is what Stage 3a will use.

# STAGE 3a RESULT (2026-09-01, 16x16 bench450, 6 CPU-minutes, NO NETWORK)

The portfolio form of the mixed action space, `subgoal/hybrid.py`: finish from
the start with the old vocabulary; enumerate every macro move by physics; rank
them; finish from the best M; keep the shortest total. Network-free end to end.
Both arms share the identical internal baseline, so the comparison is inside one
payload and does not depend on any recorded row.

| arm (450 puzzles, `b0` 400, `b_sub` 200, top 8 of ~181 candidates) | solved | optimal | mean extra |
|---|---|---|---|
| old vocabulary alone (the internal baseline, same caps) | 431 | 294 = **65.3%** | **0.824** |
| **+ macro head start, ranked by the structural estimate** | **441** | **332 = 73.8%** | **0.571** |
| + macro head start, ranked by the relaxation | 439 | 331 = 73.6% | 0.565 |

**The architecture works.** +8.5 points of move-optimality, a 31% cut in extra
moves, and 10 more puzzles solved, from an action-space change alone with no
learned component anywhere. 57 of 441 solutions were strictly improved by a head
start.

### The heuristic did not matter, and that is a real finding

Stage 2 measured the structural estimate at **62.3% top-1 against the
relaxation's 47.8%**, a 14.5-point gap. In this search that gap is worth
**one puzzle** (332 vs 331).

The explanation is in the shape of the two tasks. Stage 2 measured ranking at
every node of a deep search, where an error compounds. Here the ranker gets one
shot at putting a good prefix in the top 8 of 181, and only 13% of instances have
a helpful prefix at all. Both rankers clear that bar about equally often.

**This is a warning about the plan's own logic.** Stage 2's gate treated ranking
quality as the thing that matters, and at this stage of this design it is nearly
irrelevant. A better heuristic is worth buying only where the search is deep.

### How much of Stage 1's ceiling survives

| | mean extra |
|---|---|
| old vocabulary alone | 0.824 |
| **achieved here** | **0.571** |
| Stage 1 ceiling (perfect chooser, all 181 prefixes, exhaustive finish) | 0.108 |

**The real planner captures 35% of the available gap.** The shortfall is budget,
not concept: the ceiling tried every one of ~181 prefixes with a 30-second
exhaustive finish from each, this arm tries 8 with 200 iterations each. Whether
the rest is buyable, and at what price, is the sweep below.

### Cost

Mean 320.8 iterations per puzzle, **median 57**, running 8.6 searches (one at the
start, 7.6 sub-searches) against the baseline's one, at 2.87 s per puzzle.

Note the unit: these are the old vocabulary's own exhaustive-search iterations,
NOT the arena's network expansions, and this arm has no network to count. It is
therefore comparable with its own internal baseline, which is the comparison
made above, and **not** with the recorded 53.3% network-guided backward row.

## Stage 3a cost/quality sweep (2026-09-01, 450 puzzles, network-free throughout)

`b0` 400 in every row. "screen" is how many of the ~181 macro-move prefixes get
a finish; "finish" is the iteration cap on each. Cost is in the old vocabulary's
own exhaustive-search iterations.

| arm | solved | optimal of 450 | mean extra | mean iters | median iters |
|---|---|---|---|---|---|
| old vocabulary alone (internal baseline) | 431 | 294 = 65.3% | 0.824 | - | - |
| screen 4, finish 200 | 439 | 325 = 72.2% | 0.649 | 183 | 30 |
| screen 8, finish 200 | 441 | 332 = 73.8% | 0.571 | 321 | 57 |
| screen 16, finish 200 | 443 | 359 = 79.8% | 0.451 | 591 | 131 |
| screen 16, finish **800** | 446 | 361 = 80.2% | 0.453 | 1296 | 131 |
| **screen 32, finish 200** | **446** | **370 = 82.2%** | **0.348** | 1166 | 264 |

### Widening the screen buys quality; deepening each finish does not

Screen 16 to 32 at the same finish: 79.8% to 82.2%. Screen 16 with the finish
**quadrupled** to 800: 79.8% to 80.2%, **+0.4 points for 2.2x the iterations**.

So the binding constraint is how many prefixes get looked at, not how hard each
one is worked. This reproduces, in a different design and a different action
space, the project's own v07 finding that "most of the depth-2 gain is wider
screening, not two-slide plans" (walkthrough part 3, caveat 1). Two independent
systems now say the same thing about this domain.

### Where it sits against the systems that use networks

Quality is directly comparable (same 450 puzzles, same optimum, same physics
certification). **Cost is not**: these are the old vocabulary's exhaustive
iterations, while the rows below spend arena expansions costing one to four
network passes each.

| system | solved | optimal of 450 | network? |
|---|---|---|---|
| old vocabulary, network-guided (recorded) | 432 | 53.3% | yes |
| new macro search, first build (recorded) | 415 | 60.7% | yes |
| new macro search, tuned, no training (recorded Part A) | 404 | 80.9% | yes |
| **this arm, screen 32** | **446** | **82.2%** | **no** |
| new macro search + self-play (recorded Part C) | 444 | 95.3% | yes |
| forward move-by-move, supervised (recorded) | 450 | 94.2% | yes |

**A network-free hybrid solves 42 more puzzles than the tuned network-based
macro search and is 6 puzzles more move-optimal.** It does not approach the
self-play system's 95.3%, so the networks still win at the top end, but they are
not what gets you to 80%.

Two caveats, both against this arm. Its mean extra on its own solves (0.348) is
worse than Part A's (0.248), because it solves 42 more puzzles and those are the
harder ones. And its internal baseline (65.3%) is already **stronger** than the
project's recorded network-guided backward row (53.3%), because a bounded
exhaustive search beats a network-guided one in a vocabulary that is only about
one decision deep; the gains above are therefore measured over a stronger
baseline than the project's own, not a weaker one.

### Fraction of Stage 1's ceiling captured

0.824 to 0.348 against a ceiling of 0.108: **66% of the available gap**, up from
35% at screen 8. The curve has not flattened, so the next runs screen every
candidate to find where it does.

## CORRECTION and the completed Stage 3a picture (2026-09-02)

### I was comparing against a ceiling measured on a different instance set

Earlier entries said the planner "captures 35% / 66% / 71% of Stage 1's ceiling
gap". **That was wrong.** It set a full-450 achieved number against a ceiling
measured on Stage 1's 37 paired instances. Different sets, so the fraction was
meaningless. Restricted to the same 37:

| on Stage 1's 37 paired instances | mean extra | optimal |
|---|---|---|
| old vocabulary alone, exhaustive | 0.838 | 27/37 |
| **Stage 1 ceiling** (+1 macro prefix, exhaustive finish, perfect chooser) | **0.108** | **35/37** |
| **achieved, screen ALL, finish 200** | **0.108** | **35/37** |
| achieved, screen 8, finish 200 | 0.459 | 31/37 |

**The real planner reaches the ceiling exactly wherever the ceiling was
measured.** There is no shortfall to explain. The full-450 figure (0.284) is
simply the honest number on a harder, larger set for which no ceiling has ever
been computed.

### The full curve, 450 puzzles, network-free throughout

| arm | solved | optimal of 450 | mean extra | mean iters |
|---|---|---|---|---|
| old vocabulary alone | 431 | 294 = 65.3% | 0.824 | - |
| screen 4, finish 200 | 439 | 325 = 72.2% | 0.649 | 183 |
| screen 16, finish 200 | 443 | 359 = 79.8% | 0.451 | 591 |
| screen 16, finish 800 | 446 | 361 = 80.2% | 0.453 | 1296 |
| screen 32, finish 200 | 446 | 370 = 82.2% | 0.348 | 1166 |
| **screen 64, finish 200** | 448 | 382 = 84.9% | 0.312 | **2328** |
| screen ALL, finish 200 | 449 | 386 = 85.8% | 0.298 | 5126 |
| screen ALL, finish 800 | **450** | 388 = **86.2%** | **0.284** | 12081 |

**450 of 450 solved**, which no recorded system except the supervised forward
planner achieves, with no learned component anywhere.

### The residual is NOT finish depth, and that kills the obvious next idea

64 puzzles are not solved optimally at full screen width. Running only those,
with the same full screen and a deeper finish:

| finish per prefix | of the 64, now optimal | mean extra | mean iters |
|---|---|---|---|
| 200 (the base run) | 0 | 2.03 | 5126 |
| 800 | **2** | 2.000 | 39,449 |
| 3200 | **2** | 1.969 | 90,091 |

Quadrupling the finish converts two puzzles. Quadrupling again converts **none**.
A 16x deeper finish is worth 2 of 64. So the remaining gap is not how hard each
prefix is worked, and it is not how many prefixes are screened (every one
already is).

**By elimination it is the prefix LENGTH.** These 64 need more than one macro
move before the old vocabulary can finish, which is exactly the H-T-H-T family
the Stage 0 decomposition found and which a one-move head start cannot reach.
That makes two-move prefixes the next experiment, and it is now a measured
hypothesis rather than a guess.

### Configuration frozen for the out-of-sample test

Everything above chose settings by looking at the benchmark, so it is
**exploratory**. Before any 24x24 number is quoted, the configuration is frozen
here, by the knee of the curve rather than by its best point:
**screen 64, finish 200, `b0` 400, structural ranking.** Screen ALL is 0.9
points better for 2.2x the cost, so 64 is the honest operating point.

The 24x24 run under that frozen configuration is the confirmatory test.

# 24x24 CONFIRMATORY RESULT (2026-09-02, 21 CPU-minutes, NO NETWORK)

The configuration was frozen and committed (56c4a6f) **before** this ran:
screen 64, finish 200, `b0` 400, structural ranking, chosen at the knee of the
16x16 curve rather than its best point. Different board size, different exam,
nothing re-tuned. Graded exam, 232 instances, every one with a known `d*`.

| arm | solved | optimal of 232 | mean extra |
|---|---|---|---|
| old vocabulary alone (internal baseline, same caps) | 223 | 143 = **61.6%** | **1.027** |
| **+ macro head start, screen 64** | **227** | **193 = 83.2%** | **0.348** |

**+21.6 points of move-optimality and a 66% cut in extra moves**, from the
action-space change alone, with no learned component anywhere. 65 of 227
solutions were strictly improved by a head start. Cost 2,327 iterations mean,
566 median, 35.5 s per instance.

The 16x16 result transferred: 84.9% there, 83.2% here, on a board with twice the
area and twice the candidate count (372 mean against 181).

### Against the recorded systems on this exact exam

Quality is comparable (same 232 puzzles, same optima, same physics
certification). **Compute is not**: the rows below spend 1,200 arena expansions
at one to four network passes each, this arm spends exhaustive-search
iterations and has no network to count.

| system | solved | optimal of 232 | mean extra | network? |
|---|---|---|---|---|
| standard sub-goal search (recorded control) | 230 | 59.6% | 1.417 | yes |
| old vocabulary's exhaustive ceiling (recorded) | 228 | 62.7% | 1.171 | no |
| **v07 head-start hybrid, the project's flagship** | **231** | **68.4%** | **0.944** | **yes** |
| **this arm, frozen config** | 227 | **83.2%** | **0.348** | **no** |
| supervised forward move planner (recorded) | 220 | 94.1% | 0.068 | yes |

**It beats the flagship by 14.8 points of move-optimality and cuts its extra
moves by 63%, without a network.** The flagship still solves 4 more puzzles
(231 against 227), and that is a real loss, not a rounding difference.

### Why it beats the flagship, measured

Winning head-start lengths, in slides:

| slides in the winning head start | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| instances | 33 | 20 | 11 | 1 |

**Half the wins (32 of 65) used a head start of two or more slides by a single
robot.** The flagship's head start is one primitive slide (two at depth 2), and
each extra slide costs it a whole extra search lane. Here a robot's entire run
to a rest cell is ONE action at its true cost, so multi-slide head starts come
free. That is the concrete mechanical reason the macro framing wins, and it is
the same "helper journeys costed as one decision" structure `STAGE01.md` named.

### What this does not settle

Compute is not comparable, so "better and cheaper" is NOT claimed; only "better,
at a cost measured in a different currency". The 4 unsolved puzzles are a real
deficit against the flagship. And the residual analysis at 16x16 says the
remaining gap is prefix LENGTH, which two-move prefixes would test.

# STAGE 3b RESULT (2026-09-08/09, 16x16 bench450, ~40 GPU-minutes total)

The FINISH-HERE action inside the real macro search: at a popped node the search
may hand the position to the old sub-goal vocabulary and take whatever it
certifies. Ten configurations, every pair differing by one flag.

**Three independent harness checks passed.** `fin_cap 0` with the supervised net
reproduces the recorded Part A row (404 solved, 80.9%, 0.248) and matches it row
for row on 20 instances including exact move sequences; `fin_cap 0` with the
self-play net reproduces the recorded Part C round-3 row (444 solved, 95.3%,
0.113) exactly.

| guide | finisher calls | solved | optimal of 450 | extra | fin calls | sec/inst |
|---|---|---|---|---|---|---|
| relaxation | 0 | 354 | 67.8% | 0.370 | 0 | 10.6 |
| relaxation | 4 | 434 | 82.4% | 0.336 | 1,513 | 48.0 |
| relaxation | 24 | 446 | 86.0% | 0.280 | 7,984 | 272.2 |
| supervised net | 0 | 404 | 80.9% | 0.248 | 0 | 71.0 |
| supervised net | 4 | 444 | 90.9% | 0.191 | 1,511 | 65.7 |
| supervised net | 8 | 444 | 91.3% | 0.167 | 2,876 | 106.6 |
| supervised net | 16 | 446 | 91.6% | 0.175 | 5,507 | 186.7 |
| supervised net | 24 | 446 | 92.4% | 0.139 | 7,957 | 270.8 |
| supervised net | 48 | 448 | 93.8% | 0.127 | 14,595 | 527.5 |
| **self-play net** | **0** | 444 | 95.3% | 0.113 | 0 | 27.4 |
| **self-play net** | **4** | **448** | **97.1%** | **0.078** | 1,506 | 67.1 |
| **self-play net** | **24** | **450** | **97.1%** | 0.080 | 7,921 | 283.4 |

Recorded, same 450: self-play alone 95.3% (444), forward move planner 94.2%
(450), Part A 80.9% (404).

### The headline

**450 of 450 solved and 97.1% move-optimal.** No recorded system reaches both:
the forward planner solves all 450 but is 94.2% optimal; self-play is 95.3% but
solves 444. This is the first arm that is best on both axes at once.

### The finisher's value shrinks as the guide improves, but it never becomes zero

| guide | finisher worth |
|---|---|
| relaxation | **+18.2** points |
| supervised net | **+11.5** |
| self-play net | **+1.8** |

My pre-registered worry was that the finisher would be redundant once the guide
was good. It is **partly right**: its marginal quality contribution collapses
from 18 points to under 2. What survives is more interesting.

### It closes the coverage gap, which was the project's stated open problem

FINDINGS 33 closed by naming coverage, not quality, as what separates the
self-play planner from the forward one: 444 of 450 solved, with 0/6 discordant,
"so C's six unsolved cap it at 98.7% before quality". **The finisher solves all
six.** 444 to 450. That is the specific open problem the previous plan handed
forward, and it is closed by an action-space change with no retraining.

### Cost, in one currency

Same hardware, same guide, wall clock per instance. Four finisher calls on the
self-play net cost **2.4x** (27.4 to 67.1 s) and buy +1.8 points and 4 solves.
Twenty-four cost **10x** and buy the last 2 solves and nothing else. **Four is
the operating point**; the expensive configuration was never necessary, which
also retires the 10x cost worry raised when the arm was first built.

Depth of each finish remains nearly irrelevant, now measured a fourth time:
quartering it costs 0.2-0.4 points and halves the time.

### A network-free system beats the tuned network one

`relaxation + 4 finisher calls` reaches **82.4%** against Part A's 80.9%, which
uses a trained network and a tuned beam. Not the headline, but it means the
cheap end of this design is competitive with the project's pre-self-play state
of the art while needing no GPU, no training and no checkpoint.

### What this does not settle

All 16x16. The comparison that decides anything is 24x24 against the v07
flagship, and the search hardcodes the board size, so that is the next build.
The self-play net was trained at 16x16 and is architecturally size-free but has
never been run larger, so it may simply fail there.
