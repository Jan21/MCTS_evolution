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
