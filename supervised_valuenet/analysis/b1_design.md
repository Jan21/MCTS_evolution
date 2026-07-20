# Lever B1 — the temporary-support subgoal: design note

Written 2026-07-17, before implementation (stage 1 of the staged plan). Plain
language; terms as in SOURCE_OF_TRUTH.md. Scope: lift the backward planner's
measured expressiveness ceiling (90.7% at base scale, ~84% at 6 robots per
`scaling/results/g16r6/ceiling_probe_old_vocab.json`) while staying strictly
inside the subgoal formalism. Lever C (raw-move fallback) is out of scope and
not used.

## 1. The one-line gate that causes most of the ceiling

A subgoal is "park a helper on a *support* cell so the slider stops on a
*bottleneck* cell." Candidate (bottleneck, support) pairs are collected in
`GridEnv._collect_bottleneck_support_pairs`, which admits a support cell only
if it passes `_has_adjacent_wall(support_pos)` — i.e. **only cells where a
robot can park by sliding against a wall** may serve as supports.

Everything else in the stack is already general:

- The instance graph (`nn/gen_grids.py:build_graph`) records a dependent edge
  for **every** intermediate stop of every slide, wall-adjacent or not, with
  its stopper cell in `d["dependent"]`.
- The cost oracles (`GridEnv.compute_exact_shortest_path_length` /
  `..._relaxed_...`) consult `_dependent_edge_cache`, which is built from
  **all** dependent edges, unfiltered.
- The plan DAG already expresses support-for-support: a helper-placement edge
  that has no exact unsupported path stays *open* and is recursively expanded
  by `propose` into its own subgoal (skeleton/astar.py `_expand`/`_apply`).
- The strict realizer already executes "approach before the support arrives"
  (the fix-2 two-phase split), and already orders a support robot's departure
  after its bounce is consumed (`eval/realize.py:_execute_schedule`, deps
  b/c).

So the missing vocabulary is precisely: **supports on cells that can only be
held because another robot (or the timing of the plan) makes them holdable.**
That is the "temporary support" of SOURCE_OF_TRUTH §5c: a stopper that may
arrive late, exist only transiently, or be created by a second helper.

## 2. Worked exactness argument (benchmark idx 333, env 2511, d* = 4)

The only 4-move solution: Green up (4,5)→(4,0) [stops at the border wall];
Yellow (target) left (6,1)→(1,1) [passing *through* (4,1)]; Blue up
(4,9)→(4,1) [stopped by Green — (4,1) has **no adjacent wall**]; Yellow right
(1,1)→(3,1) [stopped by Blue; goal].

Under the current vocabulary the pair (bottleneck (3,1), support (4,1)) is
discarded by the wall filter, and the exhaustive probe builds zero complete
plans (`analysis/artifacts/ceiling_probe_results.json`, idx 333). With the
filter lifted for a new candidate type, the existing machinery completes the
whole job:

- Root subgoal: bottleneck (3,1) = the goal, support (4,1), helper Blue.
  Mover edge cost `exact((6,1),(3,1),(4,1))` = 2 (via the graph's dependent
  edge (1,1)→(3,1)).
- Blue's placement edge (4,9)→(4,1) has no exact unsupported path → stays
  open → recursively expands to the nested subgoal bottleneck (4,1), support
  (4,0) (wall-adjacent, static), helper Green; costs 1 + 1.
- Plan cost 0+2+0+1+1 = 4 = d*. **No free moves: every maneuver move is a
  costed plan edge.**
- Strict realization: the atomic schedule fails at Yellow's supported segment
  (Blue would already sit at (4,1), blocking Yellow's leftward approach);
  the existing two-phase split retries it approach-first — Yellow → (1,1)
  *before* the support chain runs — and the Kahn order executes
  Yellow-approach, Green, Blue, Yellow-bounce: 4 legal moves. Hand-traced
  against `_execute_schedule`'s dependency rules; stage 2 verifies by
  running it.

## 3. The extension, layer by layer

### Layer 1 — transient support cells (this change)

**New candidate type.** `GridEnv.propose_subgoal_states(state, support_robot,
transient=True)` additionally yields (bottleneck, support) pairs collected by
the same final-component crossing rule but with the wall-adjacency test
*inverted* (only non-wall cells; the static pairs are unchanged and stay
first-class). Each `Subgoal` gains a `transient: bool = False` field (default
keeps every existing constructor and pickle valid) so downstream consumers —
candidate labeling, the future proposal-net head, the Rust port — can
distinguish the type.

**Search.** `skeleton/heuristics.py:propose(..., b1=False)` threads the flag;
`propose_b1` is the ready-made swap-in for `AStar(propose=...)`. Nothing in
`skeleton/astar.py` changes: transient candidates ride the existing open-edge
recursion, the existing fix-3/fix-4 invariants in `_apply` apply to them
unchanged, and the admissibility argument is unchanged (a weight-2 dependent
edge still expands to ≥ 2 real moves: 1 mover slide + ≥ 1 helper-placement
move).

**Cost honesty.** No new cost rules exist to get wrong: transient supports
are costed by the same exact/relaxed oracles from the same graph edges, and
the helper-delivery moves appear as ordinary plan edges with their own costs.
Stage-2 acceptance includes `plan.cost() == strict_moves(...)` on idx 333.

**Realization.** `eval/realize.py` is untouched in this layer. "Arrive late"
is the already-adopted two-phase split; support-for-support placement is an
ordinary supported segment; departure ordering already exists.

**The exact solver gains the vocabulary automatically** because every planner
(the hand-coded exhaustive probe and the NN-guided search alike) generates
candidates through `propose`; the networks only rank them.

### Layer 2 — vacate / relocate (evidence-gated, in this task if the residue
demands it)

Layer 1 cannot express two things, deliberately deferred until the stage-3
residue says they are needed:

- **Vacate:** moving a support robot *off* its cell after its bounce is
  consumed, to clear a path or a destination cell for a later segment. The
  plan currently never moves a robot except mover-chains and one placement
  per helper (fix 3).
- **Relocate:** re-recruiting an already-placed helper *from its support
  cell* for a second support job (currently blocked by the one-leaf-per-robot
  invariant, which is about recruiting a robot at its *original* cell twice —
  a relocation would instead hang the new placement edge off the robot's
  existing support node, and the realizer's departs-after-bounce dependency
  already orders it correctly).

Both would be new `propose` candidates whose extra moves are ordinary costed
plan edges (no free vacates). If the stage-3 probe shows the remaining
failures do not need them, they are reported as designed-but-not-needed.

### Deliberately out of scope

- Lever C (bounded primitive-move edges) — not authorized; a plateau is
  reported as a plateau.
- Retraining the proposal/value networks on the new candidate type, and label
  regeneration (Python driver first; the Rust datagen crate does not know
  `transient` yet) — the main session's lane; noted as follow-ups.
- Changing pickled `GridEnv` caches: transient pairs are computed lazily at
  runtime and memoized per instance (`getattr`-guarded), so existing on-disk
  caches remain valid and byte-identical.

## 4. Zero-regression argument and its verification

With `b1=False` (the default everywhere) the code path is unchanged — same
pairs, same candidates, same plans, same realizer — so current behavior is
identical *by construction*; stage 3 verifies it anyway:

1. Realizer A/B (pattern of `eval/results/realizer_*_ab.json`): Layer 1 does
   not touch `eval/realize.py`, so stored-plan realizations are byte-identical
   trivially; if Layer 2 lands, the full stored-plan A/B is run and must show
   0 regressions.
2. Probe-level: with `--b1` the exhaustive search may only *add* candidates;
   a previously-found playable plan can only be found at equal-or-lower cost.
   Verified empirically on a solved sample of bench450 (no solved instance
   becomes unsolved under the extended vocabulary).
3. `git diff`-level: the default-path changes are flag-threading only.

## 5. Budget risk

Transient pairs enlarge the proposal set (every slide's intermediate stops
become eligible supports), so exhaustive probes branch harder. The 6-robot
old-vocabulary probe already has 6 INCONCLUSIVE rows, all at the 60 s time
cap. The B1 probe therefore takes its caps from the command line, runs the
scale set with a larger time budget, and reports any row that still cannot be
resolved as INCONCLUSIVE rather than guessing.

## 6. Acceptance per stage

- Stage 2: hand-coded exhaustive search with B1 finds a strictly-playable
  4-move plan for idx 333; plan cost equals legal move count; default-flag
  path produces zero diff on a control instance set.
- Stage 3: ceiling probe re-run on the base 49-failure set → new ceiling;
  bench450 solved-set regression check (401 stay solved).
- Stage 4: probe re-run on the 71 structural 6-robot instances (+ the 6
  inconclusives at a bigger budget) → the scale-lift headline.
- Stage 5: report + `analysis/b1_extension_notes.md` (FINDINGS-style draft).
