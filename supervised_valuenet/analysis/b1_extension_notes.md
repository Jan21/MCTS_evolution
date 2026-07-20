# Extension #5 — Lever B1, the temporary-support subgoal (draft for FINDINGS.md)

Written 2026-07-17. Design: `analysis/b1_design.md`. Every number below traces
to a checked-in artifact file. This section is drafted in the FINDINGS.md
register for the main session to merge; nothing here changes any default code
path (the new vocabulary is opt-in, `b1=True` / `--b1`).

## The finding, in one paragraph

The backward planner's expressiveness ceiling was not one wall but two, and
both moved. Extending the plan language with **temporary supports** — a
stopper may stand on a wall-less cell that only another robot (or the plan's
own timing) makes holdable, and a placed robot may be scheduled to **step
aside** before a slide it would otherwise block — lifts the measured ceiling
at the base scale from **90.7% to 97.6%** (structural failures 42/450 →
11/450) and at 6 robots — the scale where the ceiling matters most — from
**84.2% to at least 96.9%** (proven-impossible 71/450 → 2/450, with 12
probes unresolved at their caps). The extension is subgoal-pure (new candidate types inside
`propose`; no raw-move fallback), every maneuver move is ordinary costed plan
edges, and the default path is proven unchanged (per-row identical probe
re-run; 249/249 stored realizations byte-identical).

## What was added (two new things a plan can say)

1. **Transient support** (`GridEnv.propose_subgoal_states(..., transient=True)`,
   `skeleton/heuristics.py:propose_b1`): the old vocabulary only admitted
   support cells with an adjacent wall — cells a helper can park on by
   itself. The one-line gate (`_has_adjacent_wall` in
   `_collect_bottleneck_support_pairs`) was the dominant cause of
   undecomposable puzzles. The new candidate type admits wall-less support
   cells; the helper's placement edge then decomposes recursively into its
   own subgoal (support-for-support), all costed by the existing exact/
   relaxed oracles (the instance graph always had the needed dependent
   edges). "Arrive late" — the target passing through the support cell
   before the stopper lands — is realized by the already-adopted fix-2
   two-phase order; the realizer needed no change for this part.
   `Subgoal` carries a new `transient: bool = False` field so labeling and
   the future proposal-net head can see the type.

2. **Park / vacate** (`skeleton/astar.py:apply_park`, `park_repairs`;
   `eval/realize.py`): a plan may schedule one robot to slide to a parking
   cell **before** a named plan segment runs, clearing its way — the
   "step aside" maneuver. In the DAG it is a `park` node (pos, robot,
   `before_edge`) with one fixed, costed edge from the node where the plan
   leaves the robot: its support node if it was placed (the realizer's
   existing departs-after-bounce rule then makes it a true post-bounce
   vacate) or its leaf (an idle robot stepping aside). Park candidates are
   proposed from realization feedback: when a complete plan fails the strict
   check, `park_repairs` reads the first failing segment and the position
   snapshot, keeps only (robot, cell) pairs under which the blocked slide
   becomes passable, and each survivor re-enters the search as an ordinary
   costed plan (bounded: `--park-cap`, default 1 park per plan).

The exact solver (the hand-coded exhaustive search) gains the vocabulary
automatically because every planner generates candidates through `propose`;
the neural networks only rank candidates and have not been retrained (see
follow-ups).

## Cost honesty

No maneuver is free. Transient-support placements are ordinary plan edges
costed by the graph oracles; a park edge carries the exact walls-only path
cost of the sideways slide. Demonstration on the worked-example puzzle
(benchmark idx 333, env 2511): the B1 plan's `plan.cost()` = 4 = its strict
legal realization = the true optimum d\* = 4
(`analysis/artifacts/ceiling_probe_results_b1.json`, idx 333).

## Zero-regression proof (nothing chosen ⇒ nothing changes)

- **Default probe path, end to end:** the ceiling probe re-run with default
  flags after all B1 code landed reproduces the checked-in baseline
  per-row — identical categories AND identical realizable move counts for
  all 49 instances (`analysis/artifacts/ceiling_probe_default_recheck_post_b1.json`
  vs `analysis/artifacts/ceiling_probe_results.json`).
- **Realizer A/B, stored plans:** all 249 stored plans (the 211 originally
  failing plans + the 38 post-fix residual set) realize identically under
  the pre-B1 and post-B1 realizer — 249/249 equal verdicts and move counts,
  0 mismatches (`eval/results/realizer_b1_ab.json`,
  script `analysis/artifacts/b1_realizer_ab.py`). The realizer edits are
  additive: a position snapshot in the failure context, a first-failure
  record (both purely observational), and a scheduling dependency that only
  exists for plans containing `park` nodes.
- **Vocabulary is a superset:** with B1 ON, a 40-instance random sample of
  currently-solved bench450 puzzles all remain solvable by the exhaustive
  search (40/40 REALIZABLE_EXISTS,
  `analysis/artifacts/b1_solved40_results.json`).
- The NN evaluation path (`eval.compare`) defaults to the old vocabulary;
  bench450's 401 solved are untouched by construction.

## The re-measured ceiling — base scale (16×16, 4 robots)

Probe re-run on the same 49 instances the best backward mode fails
(`analysis/artifacts/ceiling_probe.py --b1`; results
`ceiling_probe_results_b1.json`, transient-only ablation
`ceiling_probe_results_b1_transient_only.json`). All 49 conclusive (max
43 s, max 18,485 expansions; no budget bumps needed at this scale).

| vocabulary | no plan can be written | plans exist, none playable | playable plan exists | ceiling |
|---|---|---|---|---|
| old (static supports) | 22 | 20 | 7 | 408/450 = **90.7%** |
| + transient supports | 7 | 14 | 28 | 429/450 = **95.3%** |
| + parks (full B1) | 7 | 4 | **38** | **439/450 = 97.6%** |

- 31 of the 42 structurally impossible puzzles became expressible-and-playable
  (21 by transient supports alone; 10 more need one park).
- Recovery quality is far better than the old headroom recoveries: 7 of the
  38 plans are move-optimal, median excess 3 moves (old recoveries: mostly
  14–24 moves over optima of 3–11).
- The worked-example puzzle (idx 333) that anchored the ceiling story is
  solved optimally.

**What remains at base scale (11/450 = 2.4%):** 7 puzzles still admit no
complete plan — their dead ends are helper-delivery chains that bottom out
(with 3 helpers there is no robot left to place the next needed stopper, or
the needed cell is reachable by no chain at all); the scoped next extension
for these is **relocation** (re-recruiting an already-placed helper from its
support cell for a second stopper role), designed in `analysis/b1_design.md`
§3-Layer-2 but not implemented — the evidence base (7 puzzles, 1.6%) did not
justify surgery on the recruitment invariants in this pass. 4 puzzles have
plans that all fail scheduling in ways one park cannot fix (e.g. the blocker
is the target robot mid-chain, which no subgoal may park).

## The re-measured ceiling — 6 robots (the scale measurement)

Baseline (old vocabulary): `scaling/results/g16r6/ceiling_probe_old_vocab.json`
— of the 105 backward failures probed, 48 NO_COMPLETE_PLAN + 23
NO_REALIZABLE_PLAN (structural share 71/450 ≈ 16% of the benchmark) + 28
recoverable + 6 INCONCLUSIVE (all at the 60 s cap).

B1 re-probe of the 71 structural + 6 inconclusive instances
(`scaling/results/g16r6/ceiling_probe_b1.json`; each row records its pass:
transient-only at 300 s / 400k frontier, then the 28-instance residue re-run
with parks at 600 s / 1.5M frontier):

| old verdict | → playable | → still impossible | → unresolved (caps hit) |
|---|---|---|---|
| NO_COMPLETE_PLAN (48) | **40** | 0 | 8 |
| NO_REALIZABLE_PLAN (23) | **19** | 2 | 2 |
| INCONCLUSIVE (6) | **4** | 0 | 2 |

- **Proven-impossible drops from 71 to 2.** Counting every unresolved probe
  as impossible (the conservative reading), the 6-robot ceiling moves from
  **84.2% to at least 96.9%**; counting only the proven-impossible, to
  99.6%. 11 of the 63 recoveries needed a park.
- Recovery quality on the graded subset (26 instances with a true optimum):
  **10 of 26 recovered plans are move-optimal**, median excess 3 moves.
- The 12 unresolved probes are frontier-bound, not time-bound (the worst,
  a beyond-oracle instance, filled a 1.5-million-plan frontier in ~6
  minutes); 10 of 12 are beyond-oracle instances whose forward solutions are
  also unknown. Resolving them needs a memory-shaped probe change (e.g.
  depth-bounded iterative deepening), not more wall-clock.

## Cross-validation against the failure-family taxonomy

`analysis/failure_families.md` (FINDINGS §12) predicted, per instance, which
structural failures B1-as-designed should flip (`transient_support`,
`blocked_direct`) and which it should not (the robot-role families:
`shared_support`, `target_support`, `relocate`, `target_clears`). Joining
those per-instance labels (from `analysis/failure_gallery/diagnose.py` over
the forward-solution artifacts) with the B1 probe verdicts on `(tag, idx)`:

**Base scale (42 structural instances, all conclusive):**

| | actually flipped | did not flip |
|---|---|---|
| predicted flip (21) | **20** | 1 |
| predicted no-flip (21, incl. the assembly residual) | 11 | **10** |

**6 robots (61 structural instances with a conclusive B1 verdict; 29 of them
had no forward solution and hence no prediction — all 29 flipped):**

| | actually flipped | did not flip |
|---|---|---|
| predicted flip (20) | **18** | 2 |
| predicted no-flip (13, incl. the two assembly residuals) | **12** | 0* |

\* the thirteenth predicted-no-flip instance (a `target_support`) ended
INCONCLUSIVE at the probe caps and is counted in neither column.

Two findings, both in the taxonomy's own pre-registered directions:

1. **Every prediction-beating flip used zero parks.** All 23 role-family
   instances that flipped (11 base + 12 six-robot) were solved by Layer-1
   transient supports finding a *different plan shape* that routes around
   the role conflict — exactly the caveat the taxonomy stated ("B1 might
   still solve some of these through a different plan shape its new words
   enable"). The price is length: those recoveries are the long ones (base
   examples: idx 328, predicted target-support, 21 moves vs d\*=7; idx 109,
   predicted target-clears, 33 vs 7). So the role families are real as
   descriptions of the *efficient* solutions, but they overstate the hard
   ceiling: measured B1 coverage is **31/42 (74%)** at base and (conservatively)
   **≥ 63/77 (82%)** at 6 robots, against the design-based prediction of
   ~50% and ~61%.
2. **The three conclusive misses in the predicted-flip direction are one
   generator limitation, precisely characterized.** Base idx 76 and 6-robot
   (gra, 115) need TWO robots cleared off the pinned walk — the park
   proposer's unblock test is single-robot, so no candidate is ever
   generated (verified: pairwise removal unblocks both). 6-robot (gra, 251)
   has a single blocker whose two walls-only slide destinations both still
   block — parks only propose one-slide destinations, and this instance
   needs a multi-slide park. Neither is a vocabulary or realizer limit (the
   DAG and scheduler accept multiple parks and arbitrary park cells); both
   are deliberate scope choices in `park_repairs`, and both are cheap to
   generalize.

## Engineering judgment on the uncovered role families (B2 scoping, not implemented)

The measured residue (base: 5 `shared_support` + 3 `target_support` +
2 `relocate` + 1 multi-robot `blocked_direct`; 6 robots conclusive: 2, both
`blocked_direct`-shaped) plus the mechanics of this implementation support
one conclusion: **the role families are bookkeeping extensions of the
existing design, not new vocabulary.** All three reduce to one mechanism —
*supports by reference*: let a proposal name an **existing plan node** as its
support instead of recruiting a fresh robot at its leaf.

- `shared_support`: a second bounce references an already-placed support
  node. The realizer's ordering already generalizes (every bounce consuming
  a support node is ordered before that robot departs — the rule is
  per-bounce, not per-recruitment); the one-leaf invariant is untouched
  because nothing is re-recruited.
- `relocate`: a placement edge hangs off the robot's existing support node
  (its plan position) instead of a fresh leaf — the chain rules that order
  mover deliveries already order it, and the departs-after-bounce rule fires
  because the segment's source IS the old support node.
- `target_support`: the reference is a bottleneck node of the target robot's
  own chain (the target parked mid-chain serves as a stopper); the
  scheduler's source-node dependencies order the bounce before the target's
  next chain hop. Same mechanism, applied to bottleneck nodes.

Risk sits in `_apply`'s invariants (the fix-3/fix-4 checks exist precisely
to police recruitment), so a B2 iteration carries the same A/B burden as this
one. Recommended B2 scope, in order: (i) generalize `park_repairs` (pairwise
clearing + multi-slide park destinations — recovers 3 measured instances for
little risk), (ii) supports-by-reference (one mechanism, three families).
`target_clears` remains a singleton; B1 already flipped its representative
via an alternate shape.

## Follow-ups (not this task's lane)

- **Labels + training:** the proposal/value networks have never seen
  transient or park candidates; label regeneration with the B1 vocabulary
  (Python driver first) and retraining are needed before the NN planner can
  use the new ceiling. The Rust datagen crate does not know the `transient`
  field or park nodes yet.
- **`validate_plan.py`** does not know the `park` node type (the probe does
  not call it; the checker should learn the type before B1 plans enter
  training data).
- **B2 candidates** (see the scoping section above): generalized parks
  (pairwise clearing, multi-slide destinations) and supports-by-reference
  for the role families.
- **Probe memory shape:** the 12 unresolved 6-robot probes are
  frontier-(memory-)bound; a depth-bounded iterative-deepening probe variant
  would make them decidable.
- The in-search prefix filter treats park scheduling permissively (parks are
  not modeled in `_determined_indices`); harmless (permissive-biased) but
  worth tightening alongside retraining.
