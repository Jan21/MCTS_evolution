# Why backward subgoal plans fail strict realization — failure taxonomy and grid-size audit

Diagnosis pass over the 450-instance benchmark (`eval/data/bench450.jsonl`) where the backward
planner found a complete subgoal plan for 450/450 instances but only 239 survived strict legal
realization (`eval/realize.py::strict_moves`). All 211 failures were regenerated and classified
with an instrumented copy of the realizer. All work ran on CPU (no GPU, per machine policy);
regeneration reproduced the stored run exactly, so nothing was lost by the device change.

Scratch artifacts (per-instance records): `diag_all450.json`, `diag_pass2.json`,
`diag_pass3.json`, `failing_plans.pkl`, scripts `diag_failures.py`, `diag_pass2.py`,
`diag_pass3.py` — all in this scratchpad directory. No repo file was modified.

---

## 1. How strict realization works, and where it can fail

`strict_moves(env, state, plan)` (eval/realize.py:182) converts a plan DAG into one legal move
sequence:

1. Collect the plan's physical segments — every non-structural edge, each a movement
   `start -> end` with an optional `support` cell the mover bounces off
   (`_physical_segments`, realize.py:88; movement extraction in
   validate_plan.py:331 `_physical_movement`).
2. Find each segment's mover robot (`_mover_source`, realize.py:102).
3. Build ordering constraints (realize.py:204–217): (a) a robot must arrive at a node before
   departing from it; (b) a support robot must be placed before the segment that bounces off
   it; (c) it may leave only after that bounce is consumed. Order the segments with a
   deterministic, fixed topological sort (`_topo`, realize.py:142 — min-index Kahn; no
   alternatives are ever tried).
4. Execute segments one at a time on the full joint state: the mover's path is a BFS over
   slides with all other robots frozen as blockers (`_slide_bfs`, realize.py:163;
   physics = `simulate.slide`). Each segment is **atomic**: the whole path start->end runs
   under one fixed robot configuration.
5. Finally check the target robot sits on the target.

Failure channels and their behavior:

| channel | where | returns | logged? |
|---|---|---|---|
| plan has no physical segments | realize.py:187–188 | None | no (silent) |
| no mover found for an edge | realize.py:196–200 | None | yes |
| cyclic segment ordering | realize.py:219–223 | None | yes |
| segment BFS-unreachable | realize.py:235–237 | None | **no (silent)** |
| target robot not on target at the end | realize.py:241–245 | None | yes |
| segment start != robot's current cell | realize.py:231–233 | — (warns, executes from current cell) | yes |

Note: the one channel that turns out to matter (BFS-unreachable) is the only failing channel
that produces no log line, so the existing log output cannot be used to build this taxonomy —
hence the instrumented re-run.

Related but distinct: `abstract_moves` (realize.py:65) checks each segment **alone** on the
board (only its own support present) via `simulate.verify_plan` (simulate.py:102) and
`segment_realizable` (simulate.py:74). All 450 plans pass this blocker-clearing check and
`realized_abstract == plan.cost()` for 450/450 — so every failure below is purely a
joint-execution effect, never board geometry.

## 2. Reproduction and determinism

Setup mirrored the stored run (`eval/results/comparison_backward.json`): checkpoints
`checkpoints_backward/policy_v2.ckpt` + `value_v2.ckpt`, k=5, 1200 expansions, no
realize-check, `torch.manual_seed(0)` as in eval/compare.py:413, plans regenerated with
`eval.compare._nn_astar_backward` (compare.py:110). Rows in the stored results file align 1:1
with the instance file (0 env_id/d_star mismatches).

Determinism check (all 450 instances):

| check | result |
|---|---|
| plan regenerated | 450/450 |
| regenerated `plan.cost()` == stored `plan_cost_abstract` | 450/450 |
| regenerated strict outcome == stored `solved` flag | 450/450 |
| stored-failing that unexpectedly passed on re-run | 0 |

CPU inference is bit-identical for this pipeline; every number below is about the exact plans
the benchmark scored.

## 3. Failure taxonomy (211 instances)

**Channel counts: every single failure is a BFS-unreachable segment.**

| channel | count |
|---|---|
| segment BFS-unreachable | **211** |
| no mover / cyclic ordering / target-off-goal / empty plan | 0 |

By difficulty (matches the stored 95/67/34/20% strict-solve rates):

| d\* bin | 1–3 | 4–6 | 7–9 | 10+ |
|---|---|---|---|---|
| failures (all BFS-unreachable) | 3 | 54 | 134 | 20 |
| bin size | 55 | 166 | 204 | 25 |

**Decomposition of the unreachable segment** (what exactly blocks it, measured at the moment
of failure):

| sub-class | n | share | meaning |
|---|---|---|---|
| support absent | 103 | 49% | the segment declares a support cell the mover must bounce off, and **no robot is standing there** when the segment runs. 101/103 would be locally reachable with the stopper in place. |
| needs interleaving ("other") | 78 | 37% | no robot-free path exists from the mover's current cell (support present in 73, none declared in 5); 55/78 are unreachable even in the plan's own frame with the support statically present — the segment is only playable if the mover's approach happens **before** the support is placed (see §5). |
| pure robot blockage | 30 | 14% | a wall-legal path exists but robots stand in it. |

Cross-cutting features:

- Failing segment position in the execution order: mean fraction 0.37 (median 0.38); 99
  failures in the first third, 90 middle, 22 last third; 27 fail on the very first segment,
  20 on the last (the hop into the goal). Deep plans usually die early-to-mid execution.
- Failing plans are structurally deeper: modal segment counts 3/5/7 (68/65/41 instances)
  vs 1/3 for passing plans (103/123).
- End cell occupied by another robot: 24 instances.
- Mover is the target robot: 83 instances.
- Start-mismatch warnings (the drift channel that only warns): >=1 warning in 90/211 failing
  instances vs 2/239 passing — drift and failure are strongly associated, but the mismatch at
  the failing segment itself is concentrated in the support-absent class (72/103) and absent
  from pure blockage (0/30).
- Blocker identity for the 30 pure-blockage instances (single critical blockers, or the
  minimal blocking pair): bystander robots the plan never touches, 18; robots the plan is
  going to move later (still on their start cell), 15; the **target robot itself** blocks in
  14 of the 30. Across all 211 failures with identifiable critical blockers, plan-parked
  robots also appear (11 at rest mid-plan, 3 on already-consumed support cells, 2 on
  still-needed ones), and the target robot is a critical blocker in 25 instances.

## 4. The dominant mechanism: plans that use a robot as its own support

For 92 of the 103 support-absent failures, the robot standing on the declared support cell at
failure time is **the mover itself** — and pass 3 shows this is a *static plan property*, not
an execution accident: the failing segment belongs to a subgoal whose support robot is the
same robot as its bottleneck mover in exactly those 92 plans.

Minimal example (bench instance idx 1, env 2400, d\*=4). Red must reach (13,0); the plan says:

- Red slides to the goal (13,0) bouncing off a support at (14,0) — robot "Green".
- To put Green on (14,0), subgoal `sg_2` is proposed: Green (bottleneck) reaches (14,0) by
  bouncing off a support at (15,0) — **also Green**, whose leaf is Green's own start (15,0).
  The support placement edge costs 0 because "the robot is already there".

As played this is impossible: Green cannot bounce off itself. The abstract checks accept it
because they never ask *who* provides a stopper — `segment_realizable` (simulate.py:74–99)
verifies "approach to a pre-bounce cell, then a robot at (15,0), then one slide", and the cost
model gives the self-placement 0 moves. The abstract cost model therefore actively *rewards*
this pattern (free supports), which is presumably why the nets learned to propose it — and why
65/450 abstract plan costs undercut d\*.

**Where it comes from**: skeleton/astar.py:133–135 builds a segment's helper list as

```python
s.helpers = [h for h in state.helpers if h != s.mover]
if s.mover != state.target_robot:
    s.helpers.append(state.target_robot)
```

`Robot_at` is a dataclass, so `!=` compares (position, color). The mover for a nested segment
carries its *planned* position (e.g. Green at (14,0)) while `state.helpers` holds Green at its
*initial* cell (15,0) — the inequality is true and the mover survives in its own helper list,
so `propose` (skeleton/heuristics.py:37) can select it as its own support. The same
position-sensitive comparison on line 134 can add the target robot as its own helper.
Filtering by color instead of dataclass equality closes the leak.

Prevalence (pass 3, static count over plans):

| plans containing >=1 self-support subgoal | count |
|---|---|
| among the 211 failing plans | **101 (48%)** |
| among the 239 passing plans | 2 (0.8%) |

A static, physics-free plan check ("support robot == bottleneck robot in some subgoal")
predicts strict failure almost perfectly on this benchmark.

## 5. Counterfactual realizers: what a smarter realizer could and could not save

Each failing plan was re-executed under five alternative realizers (same physics, same legal
moves; per-instance results in `diag_all450.json` / `diag_pass2.json` / `diag_pass3.json`):

| realizer variant | fixes | mean regret of fixed | notes |
|---|---|---|---|
| A. flexible order (work-list: run any dependency-ready, currently reachable segment) | **0/211** | — | ordering freedom alone rescues nothing |
| B. A + hold rule (never move a robot off a still-needed support cell while alternatives exist) | 0/211 | — | same |
| C. A + single-robot detour (relocate one robot <=3 slides when stalled, incl. recruiting a stopper) | 86/211 | 5.33 | detour moves are expensive and blind |
| D. **two-phase execution** (split supported segments into approach -> place support -> bounce, mirroring the plan's own semantics) | **52/211** | **1.90** | zero extra moves; fixes 46/78 of the "needs interleaving" class |
| E. two-phase + single-robot detour/recruit | 106/211 | 5.50 | ceiling of one-robot realization repairs |

Readings:

1. **Segment order is not the problem.** No failure is a cyclic-ordering or
   wrong-interleaving-of-whole-segments case (channels: 0; counterfactuals A/B: 0 fixes).
2. **Atomic segment execution is a real realizer gap (~25%).** `strict_moves` runs each
   segment under one frozen robot configuration and forces the support to be placed *before*
   the segment (deps (b), realize.py:210–214). The plan's own model —
   `segment_realizable` (simulate.py:88–98) and `_abstract_segment_moves`'s second route
   (realize.py:44–62) — explicitly allows approach *before* support placement. 52 failing
   plans (46/78 of the "needs interleaving" class) are perfectly playable, at competitive
   cost (regret 1.90 vs 1.20 on currently solved instances), once the realizer honors the
   plan's own two-phase semantics. These are realizer false-negatives, not bad plans.
3. **Detours cannot substitute for planning.** Letting the realizer move one robot aside (or
   recruit a stopper) adds ~4+ moves of regret per solve and still fails on half of the
   support-absent class: landing a robot on one exact cell is itself a plan-worthy problem in
   this game (that is the planner's job). The self-support plans are mostly unrepairable at
   realization time — after the mover vacates its own support cell, nobody scheduled can
   refill it (recruit succeeds in only 26/103).

## 6. Lever assessment (addendum §3 L4) and recommended residual fixes

The addendum's L4 hypothesis — "if most failures are *a robot not in the plan sits on the
path*, add a move-the-blocking-robot-away proposal type" — does **not** match the data. Pure
robot blockage is the smallest class (30/211 = 14%, of which bystanders are 18), and 15 of
those 30 blockers are robots the plan was going to move anyway. The dominant classes are
plans that assume a stopper that can never exist (self-support, 44% of all failures) and a
realizer too coarse for the plan's own execution semantics (25%).

Recommended residual sequence after L0 (anytime realization-checking, still the first thing
to benchmark — it needs no code and bounds everything else):

1. **Close the self-support leak in proposals** (planning-side, ~2 lines, no formalism
   change): compare by color in skeleton/astar.py:133–135. This removes the mechanism behind
   48% of failing plans (and 0.8% of passing ones), stops the search from wasting anytime
   budget on a plan family that is almost never playable, and stops self-play (strict_filter
   gate at subgoal_selfplay/selfplay.py:233–239) from training toward it. A cheap complement:
   reject/penalize zero-cost self-placements in `propose` or add the static self-support test
   to plan validation (validate_plan.py currently allows it).
2. **Make the strict realizer two-phase** (realizer-side, no new moves, no oracle): execute
   supported segments as approach -> place support -> bounce, exactly as
   `_abstract_segment_moves` already costs them. Worth ~52 instances (~25% of failures) at
   regret 1.90 on this benchmark, and it removes a systematic bias against deep plans in the
   headline metric. Since the output is still a verified legal move sequence, it stays fair
   under the comparison protocol.
3. Only then consider value-net features (e.g. a blocked-segment or self-support flag) for
   whatever residue remains; realizer detours are better left out of the headline system —
   they cost ~5 moves of regret per rescue, though they could serve as a last-chance
   fallback when solve rate matters more than regret.

Item 1 changes which plans the search produces (needs new benchmark + probably retraining to
realign the nets); item 2 changes only realization and can be A/B-tested on the stored plans
immediately.

## 7. Grid-size generalization audit (backward plan -> moves path)

Everything on the realization path is hardcoded to 16x16 by default parameters; nothing reads
`RR_GRID`. Inventory of every `size=16` on the path:

| function | size default | internal calls (size threaded correctly) |
|---|---|---|
| simulate.wall_sets | simulate.py:20 | uses `size` to decode `grid_data` at :28 |
| simulate.slide | simulate.py:40 | boundary checks :55, :63 |
| simulate.segment_realizable | simulate.py:74 | :87, :93, :97 |
| simulate.verify_plan | simulate.py:102 | wall_sets :111, segment_realizable :121 |
| simulate.segment_moves | simulate.py:132 | slide :147 |
| eval.realize._abstract_segment_moves | realize.py:37 | :46, :55, :57 |
| eval.realize.abstract_moves | realize.py:65 | wall_sets :67, verify_plan :68, :73–74 |
| eval.realize._slide_bfs | realize.py:163 | slide :172 |
| eval.realize.strict_moves | realize.py:182 | wall_sets :185, _slide_bfs :235 |

Callers that omit `size` (and therefore run at 16 regardless of the board):

- eval/compare.py:209 (anytime realize-check), :226 (abstract_moves), :227 and :229
  (strict_moves);
- subgoal_selfplay/selfplay.py:237 (strict_filter gate).

Internal threading is complete — every inner call passes `size` down — so the only *live*
16s are the entry-point defaults plus these callers. For contrast, the forward path is
already config-aware: move_planner/state.py:30–32 passes `GRID` (from nn/gen_grids.py:28,
an `RR_GRID` reader) into `wall_sets`; train/encode.py:26 likewise. The physics modules are
the only stage still locked to 16, matching addendum §4.

**Board size exposure**: `GridEnv` has no `size` attribute, but `from_env` attaches the raw
board (`grid_env.grid_data`, GridEnv.py:146), a flat list of length size^2 (verified: 256 for
env 2400, present in the cache pickles too). So `size = isqrt(len(env.grid_data))` is always
available exactly where realization starts.

**Minimal patch plan (not applied).** Change the four env-carrying entry points to
`size=None` and infer:

1. `simulate.wall_sets(grid_data, size=None)` -> `size = isqrt(len(grid_data))` (assert
   perfect square);
2. `simulate.verify_plan(..., size=None)` and `simulate.segment_realizable(..., size=None)`
   -> infer from `grid_env.grid_data`;
3. `eval.realize.strict_moves(..., size=None)` and `abstract_moves(..., size=None)` -> infer
   from `env.grid_data`.

Low-level helpers (`slide`, `segment_moves`, `_slide_bfs`, `_abstract_segment_moves`) keep
explicit `size` parameters — they already receive it from the entry points. **No caller
changes needed**: the compare.py and selfplay.py call sites become correct automatically.
Roughly 15 lines total.

**Infer-from-env vs threading RR_GRID**: inferring from the board is the safer choice here.
`RR_ENV_DIR` selects the boards while `RR_GRID` is a second, independent import-time knob
(scaling/README.md "Configuration mechanism"); if they ever disagree — e.g. an `eval.compare`
command missing the `RR_*` prefix the README says must be written into its command line —
`wall_sets` mis-decodes the board **silently** (idx%16 on a 24-wide board raises nothing) and
every strict count is quietly wrong. A size derived from `len(grid_data)` cannot disagree
with the loaded board, works in mixed-size processes, needs zero configuration for the legacy
pipeline, and keeps the physics layer free of import-time environment coupling. `RR_GRID`
remains the right mechanism where fixed tensor shapes are needed at import time (encoders,
nets); the patch does not touch it.

## 8. Sanity check of passing instances

Three stored-solved instances (idx 2 env 2400 d\*=7; idx 3 env 2401 d\*=1; idx 4 env 2401
d\*=4) were realized with a path-recording variant of `strict_moves`, then re-verified
move-by-move: every single move was replayed with `simulate.slide` on the full joint state
(each move must actually displace its robot and stop exactly where claimed), ending with the
target robot on the target. All three replay clean, and the move totals equal the stored
`realized_strict` (7, 1, 4). The realizer's success cases are genuinely legal move sequences.

## 9. Caveats

- Counterfactual realizers A–E are greedy work-list searches, not exhaustive over execution
  orders; their fix counts are lower bounds (the 0/211 for pure reordering is a strong
  signal, not a proof).
- The detour variants cap relocation at 3 slides of one robot per stall; a stronger repair
  search would fix more instances but at even higher regret and further from the plan's
  intent.
- L0 (anytime) was not benchmarked here (out of scope); its interaction with the self-support
  fix — fewer doomed plans in the proposal stream means the anytime budget goes further — is
  an argument for doing both, not for choosing between them.
