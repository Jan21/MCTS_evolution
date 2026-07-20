# Source of truth — Ricochet Robots planner study

**Status:** canonical. Last verified 2026-07-13. Where this document and any other
file in the repo disagree, this document is correct; the other file is stale (see
§6, "Document map"). Every number here is traced to the file it comes from. The one
piece of new analysis in this document — the ceiling breakdown in §5 — was produced
by re-running the code, and the script and its output are checked in under
`analysis/artifacts/ceiling_probe.py` and `analysis/artifacts/ceiling_probe_results.json`.

Terms are defined on first use. No result is asserted without a source.

---

## 1. What this project is really about

The surface topic is **Ricochet Robots**: a puzzle on a walled square grid with a few
coloured robots. When you send a robot in one of the four directions it **slides in a
straight line until it hits a wall or another robot**, then stops in the cell just
before the obstacle — it cannot stop early. One robot is the *target*; one cell is the
*goal*. You win when the target robot lands on the goal, and you want to do it in as few
moves as possible. The slide rule is implemented once, in `simulate.py:slide`.

The real subject is not solving the puzzle — it is **how to train a solver**, and which
way of training scales. The project compares two independent choices, giving four
systems (§2). The measuring stick is a fixed set of 450 puzzles at the base size (a
16×16 grid, 4 robots), and the honest question underneath everything is:

> As puzzles get bigger and harder, which way of building a solver keeps working — and
> in particular, does a solver that plans over *subgoals* eventually beat one that
> searches move by move?

A hard rule runs through the whole project: **at solve time the planner must be pure
neural network + search — no exact solver in the loop.** An exact solver ("the oracle")
is used only to make training labels and to compute a reference optimum for scoring.
This matters because that oracle stops being computable as puzzles grow (§4).

---

## 2. The four systems (two structures × two ways to train)

Two independent choices:

- **Structure** — *how* the solver searches.
  - **Forward / move-based:** search primitive moves from the start toward the goal.
    State = the positions of all robots; an action = (which robot, which direction).
    Implemented in `move_planner/`.
  - **Backward / subgoal-based:** search over *subgoals* from the goal back toward the
    robots' start cells. A **subgoal** means "park a helper robot on a *support* cell so
    the target robot slides and stops on a *bottleneck* cell." One subgoal is worth
    several moves. Implemented in `skeleton/astar.py` + `GridEnv.py` + the nets in
    `train/`.
- **Training** — *where the learning signal comes from.*
  - **Supervised:** copy an exact optimal solver (the oracle).
  - **Self-play:** learn only from the solver's own solved games; no oracle. This is
    "expert iteration" — the solver's own budgeted search is the teacher for its next,
    faster version.

Crossing the two gives the table this project is built around. **Self-play is the point
in both rows, not only the backward one** — see §4.

|                | Supervised (copy the oracle)                    | **Self-play (no oracle)**                          |
|----------------|--------------------------------------------------|----------------------------------------------------|
| **Forward**    | `move_planner/` — the **"naive" baseline**       | `move_planner_v2/` — mature, works well            |
| **Backward**   | `train/` nets on subgoal-oracle labels           | `subgoal_selfplay/` — early, few iterations        |

Both structures share the same neural-network *encoder* — a "looped transformer" that
passes messages along the board's slide graph (`train/looped_pc.py:LoopedLayer`, reused
by `move_planner/net.py`). They do **not** share weights.

One naming note to prevent confusion:

- The repo is called `MCTS_evolution`, but **MCTS (Monte-Carlo Tree Search) is not
  used.** The self-play loops use a simpler "your own search is the teacher" scheme
  (expert iteration); MCTS was considered and set aside because at ≤16 possible moves a
  value-guided A\* teacher collapses to the same thing (`move_planner/README.md §12`).
- `train/policy_tf.py` — the `tf` means **transformer**, not TensorFlow. The whole
  codebase is PyTorch + PyTorch-Lightning; there is no TensorFlow anywhere.

---

## 3. What is measured today

All numbers below are on the **same 450 puzzles** (16×16, 4 robots; boards 2400–2549,
three puzzles per board), pinned by `eval/bench_instances.py` and stored in
`eval/data/bench450.jsonl` (sha256 `1b384dfd…`). Every system runs best-first search
capped at **1200 expansions** with the **top 5** candidates kept per step. An
"expansion" is one popped search node whose children are generated.

### 3a. The one rule that keeps the comparison honest: strict moves, never abstract cost

The backward planner produces a *subgoal plan*, not a move sequence. To compare it
fairly with the forward planner it must be **realized** — turned into an actual legal
move sequence under full physics (all robots present as blockers). That conversion is
`eval/realize.py:strict_moves`, and a backward puzzle counts as **solved only if strict
realization succeeds.**

The backward planner also reports an **abstract cost** (`plan.cost()`), computed under an
optimistic model where every robot except the one intended stopper is assumed to slide
out of the way. **Abstract cost must never be compared to the true optimum** — it can be
*smaller* than the real optimum (34 of 425 found plans have "negative regret," i.e. they
are impossible as played). Only strict, legally-executed move counts are comparable.
This distinction is the reason earlier "99.6% solved" claims were misleading: they
counted plans that were *found*, not plans that could be *played* (§6).

### 3b. Canonical results (the honest, strict-moves numbers)

Source: `COMPARISON.md` + `eval/results/final450_backward_{plain,anytime}.json` +
`eval/results/comparison_forward.json`.

| System | Solve rate | Mean regret (extra moves) | % solved optimally | Mean expansions | Sec/puzzle |
|---|---|---|---|---|---|
| Backward subgoal, plain | 356/450 = **79.1%** | 1.624 | 55.9 | **5.3** | 0.97 |
| Backward subgoal, "anytime" (keep searching past unplayable plans) | 401/450 = **89.1%** | 2.147 | 50.4 | 7.6 | 1.09 |
| Forward move, supervised — the "naive" baseline (`best.ckpt`) | 381/450 = 84.7% | 0.354 | 72.7 | 423.5 | 11.4 |
| Forward move, supervised, candidate-scored (`candidate_scored.ckpt`) | 450/450 = **100%** | **0.067** | 94.2 | 36.0 | 0.99 |
| Forward move, **self-play, warm start** (`v2/runs_warm/iter5`) | 439/450 = 97.6% | 0.055 | 95.0 | 120.8 | 3.28 |
| Forward move, **self-play, from scratch** (`v2/runs_scratch_v5/iter15`) | 394/450 = 87.6% | 0.213 | 82.5 | 304.0 | 8.26 |

Reading of the base scale (16×16, 4 robots):

1. **Forward wins on quality.** The best forward model solves every puzzle nearly
   optimally. The best backward mode solves ~89% and is further from optimal.
2. **Backward wins on search effort by 1–2 orders of magnitude** — ~3–8 expansions
   versus 36–424 — because one subgoal commits several moves at once.
3. **The clearest "smarter than naive" evidence is at a small budget.** At a cap of just
   10 expansions the backward-anytime planner solves 78.7% while the naive forward net
   solves 2.4% and even the candidate-scored forward net solves 34.7%
   (`eval/results/budget_curves.json`). Backward front-loads its solving.

Offline (per-decision) network quality, for reference: the forward value net predicts
moves-to-go with mean error 0.263 and picks the optimal move 96.0% of the time; the
backward value net predicts subgoal-cost-to-go with mean error 2.17 and ranks the best
candidate first 81.0% of the time (`COMPARISON.md` appendix; the two error numbers are
in different units and are not comparable).

---

## 4. Self-play in both formulations — and why it is the decisive axis at scale

Self-play matters in **both** rows of the §2 table, for one concrete reason: **the exact
oracle that supervises the forward baseline stops being computable as puzzles grow.** On
the 6-robot scaling benchmark the oracle already fails on **134/450 = 29.8%** of
instances even with generous limits (`scaling/data/g16r6/bench.jsonl.meta.json`). Once
the oracle dies, supervised training is off the table and **self-play is the only
training method left** — so the real head-to-head at scale is *forward self-play vs
backward self-play*, not the abstract "forward vs backward."

State of each self-play arm:

- **Forward self-play (`move_planner_v2/`) is mature and works.** From random weights,
  no oracle, it reaches **87.6%** — beating the supervised "naive" baseline (84.7%) on
  both solve rate and regret. Warm-started it reaches 97.6%.
- **Backward self-play (`subgoal_selfplay/`) is early.** Only 2–3 iterations exist on
  disk. It also has a distribution problem: on its own training data only ~**40%** of the
  winning plans it trains on are strictly playable (`analysis/pass1_selfplay.md §6`), so
  the loop can reinforce plans that will not play out. **This is the least-finished, most
  important cell in the table**, and it is directly downstream of the ceiling problem in
  §5: the same in-search realizability check that would raise the ceiling would also make
  self-play train only on playable plans.

---

## 5. The ~90% ceiling on the backward planner — what it is, and how to break it

This is the section the rest of the repo most needs corrected. The short version:

> The backward planner's "~90% ceiling" is **real and structural, not a training or
> tuning artifact.** On this benchmark **42 of 450 puzzles (9.3%) have no strictly-
> playable subgoal plan at all** under the current plan vocabulary and realizer, so the
> true ceiling is ~**90.7%**. Only ~1.6% of the gap is the network missing plans that do
> exist. To lift the ceiling you must **extend what a plan can express**; better training
> or search alone cannot get past ~90.7%.

### 5a. How the ceiling number was verified

The figure everyone cites ("about 90%") was previously supported by a failure taxonomy
(`eval/results/residual_failures_postfix.json`) that was built at an **intermediate**
stage of the repair history — before two of the four fixes landed — and never re-run.
Reconstructing the real history from the result files (`eval/results/*.json`):

| Stage | What changed | Plain solve (first-150 slice) |
|---|---|---|
| original | — | 53.1% |
| postfix | fix: a robot could be scheduled to bounce off *itself* (`skeleton/astar.py:136`, compare by colour) | 63.3% |
| postfix2 | fix: the realizer now executes a supported segment as approach → place support → bounce | 74.7% |
| postfix3/4 | fix: one robot can't fill two plan roles (`astar.py:184`); a declared stopper must actually be placed (`astar.py:194`) | 80.0% |

The last two fixes tightened *validation*, which had a subtle effect: puzzles that used
to "find a bad plan" now "find no plan at all." The count of puzzles where the search
constructs any complete plan dropped from 450/450 to **425/450**
(`final450_backward_*.json`). So the final ceiling is dominated by *the planner failing
to build a valid plan*, which the old taxonomy did not measure.

To measure it correctly, each of the **49** puzzles the anytime planner fails on was
re-solved with the **hand-coded** subgoal search (no neural network, so guidance is not
the bottleneck) run **exhaustively** — it keeps searching past unplayable plans until it
either finds a strictly-playable one or proves none exists. The exact failing puzzles
were selected by row index (the result rows align 1:1 with `bench450.jsonl`; note there
are 3 puzzles per board, so selecting by board id is wrong — an easy mistake that
inverts the answer). Script and output: `analysis/artifacts/ceiling_probe.py`,
`analysis/artifacts/ceiling_probe_results.json`. Every one of the 49 resolved
conclusively (each in under 3.3 s).

### 5b. What the ceiling actually is

The 49 anytime failures split into three groups:

| Group | Count | What it means | Is it a ceiling? |
|---|---|---|---|
| **No complete plan exists** | **22** | Even exhaustive search builds no complete subgoal plan. The vocabulary cannot express any decomposition of this puzzle. | **Yes** |
| **No playable plan exists** | **20** | Complete plans exist, but *none* of them survive strict realization. Classic downward-refinement failure. | **Yes** |
| **A playable plan exists, the network missed it** | **7** | A strictly-playable plan exists; the top-5 / 1200-budget search never generated it. | No — guidance headroom |

So **42/450 (9.3%) are a true structural ceiling** → best possible solve rate with the
current vocabulary + realizer is ~**90.7%** (408/450). The anytime planner already
reaches 89.1%, so perfect guidance would add only ~1.6 points — and 6 of those 7
recoverable puzzles need a very long plan (recovered move counts 14–24 against optima of
3–11; only one is optimal). The headroom from better search is small *and* low quality.

Two things worth stating plainly, because they contradict looser phrasings elsewhere:

- The ceiling **hits easy puzzles too**, not just deep ones: a 4-move puzzle (`idx 333`)
  has no expressible plan, and a 3-move puzzle (`idx 418`) has only unplayable ones. This
  is a property of the plan *language*, not of difficulty.
- This is the **downward-refinement property** from the classical planning literature
  (Bacchus & Yang 1994, cited in `analysis/pass5_research.md`): an abstraction is only
  useful if its abstract solutions can always be refined into concrete ones. Here they
  cannot, ~9% of the time. The strict-solve rate *is* the "refinement probability."

### 5c. How to break it

Ordered by leverage. Only the last group actually lifts the ceiling; the first is the
highest-value change overall because it fixes correctness and self-play at once.

**Lever A — make "playable" a first-class part of search and training (highest value,
does not lift the ceiling).**
Check each subgoal edge's strict realizability *as it enters the search*, not only at the
end (the standard learned-subgoal-search architecture, e.g. kSubS). Effects:
(i) "plan found" comes to mean "playable plan found," so the planner stops committing to
doomed plans; (ii) it reclaims guidance headroom — the 7 puzzles where a playable plan
exists but the search missed it — by not spending budget on doomed branches, moving
anytime toward the ~90.7% ceiling; (iii) crucially, it makes **self-play train only on
playable plans**,
which is the fix for the ~40%-playable training-distribution problem in §4. This is the
single change that most improves the *system*, even though it leaves the ceiling where it
is.

**Lever B — extend what a plan can express (this is what lifts the ceiling).** Two
sub-levers matching the two structural groups:

- **B1 — a "temporary support" subgoal (stays inside the subgoal formalism).** Today a
  helper placed as a stopper is assumed to stay put forever (`GridEnv.py:Subgoal` has a
  single static `support`). Add a subgoal type where a stopper robot is scheduled to
  **vacate or relocate after its bounce is consumed** — i.e. "move a robot aside and
  bring it back." This is exactly the maneuver the 22 "no complete plan" puzzles need,
  and it helps the 20 "no playable plan" puzzles by giving the planner a legal way to
  free a cell that is currently blocked by a robot it needs later. This is the
  recommended primary fix because it keeps the method purely subgoal-based (consistent
  with the project's methodology constraint).

- **B2 — make the abstraction sound ("angelic" reachability).** The 20 "no playable plan"
  puzzles arise because the optimistic blocker-clearing cost model claims routes that
  cannot be played. Replacing those reachability claims with ones that are refinable by
  construction means the search never proposes an unplayable plan in the first place —
  it either finds a real plan (once B1 widens the vocabulary) or reports honestly that
  none exists. B2 and Lever A are the same idea applied at the cost model and at the
  search filter respectively.

**Lever C — completeness fallback (guaranteed, but relaxes the pure-subgoal method).**
If B1/B2 plateau, allow a **bounded number of primitive-move edges inside the subgoal
plan** (the published hybrid-search completeness restorer, Kujanpää et al., noted in
`analysis/pass5_research.md`). This guarantees the backward planner can solve anything the
forward planner can, because any step the subgoal language cannot express falls back to a
few raw moves. The cost is that the plan is no longer purely subgoals — flag this against
the "stay inside the subgoal formalism" constraint before adopting it. Present it as the
safety net, not the first move.

**Lever D — guidance polish (lowest priority).** Widen the kept-candidate count beyond 5
or blend in the admissible hand-coded proposer to recover the 7 headroom puzzles without
Lever A. Small, mostly high-regret gains.

### 5d. Expected effect and the one thing to measure

- Lever A → anytime solve moves toward the current ceiling (~90.7%; the recoverable
  headroom is only ~1.6 points) and self-play stops training on unplayable plans.
  Recommended first regardless of anything else.
- Lever B (B1 primary, B2 supporting) → the ceiling itself moves above 90.7%; how far is
  an empirical question. The single verification step is: implement B1, then re-run
  `eval/compare.py` on `bench450.jsonl` and re-run the probe in
  `analysis/artifacts/ceiling_probe.py` to measure the new structural ceiling.
- **A clean negative is a legitimate result.** If extending the vocabulary cannot lift
  the ceiling economically, that is itself the strongest evidence that at this scale the
  forward / self-play formulation is the better vehicle — which is exactly the comparison
  the project exists to make.

---

## 6. Document map — which files are current, which are stale

| File | Status | Note |
|---|---|---|
| `SOURCE_OF_TRUTH.md` (this file) | **canonical** | Verified 2026-07-13. |
| `COMPARISON.md` | current for numbers; **one claim corrected here** | Its results tables are right. Its "~90% ceiling = one puzzle in ten has no legal plan in the vocabulary" is *directionally* right (verified 90.7%) but its *mechanism* is stale — see §5, the real split is 22 "no plan can be built" vs 20 "no plan can be played," not a single vocabulary bucket. |
| `eval/results/residual_failures_postfix.json` | **stale mechanism** | Built at the 74.7% stage, before two fixes; its class counts do not describe the final state. Superseded by §5 and `analysis/artifacts/ceiling_probe_results.json`. |
| `RESULTS.md` | current | Forward-planner depth; consistent with §3. |
| `addendum.md`, `todo.md`, `analysis/pass1–5_*.md` | current (planning/analysis) | The two-phase goal, scaling plan, and literature review. §5's fix menu maps onto their L0–L5 levers. |
| `handoff.md` (repo root) and `supervised_valuenet/handoff.md` | **stale** | Both describe the *forward* self-play as "the goal" and list long-dead job PIDs. The current goal is the forward-vs-backward comparison and the scaling study. Banner added. |
| `README.md`, `README-move.md` | current (orientation) | Component-level how-to. |

Numbers this document corrects relative to loose phrasings elsewhere:

- The ceiling's cause is two distinct structural mechanisms (22 + 20), not one; and it is
  ~90.7%, measured, not estimated.
- "Plan found" is 425/450 in the final state, not 450/450 — the planner genuinely fails
  to build a plan for 25 puzzles.
- Any "99.6% / 100%" backward figure refers to plans *found*, in abstract units, not to
  puzzles *solved* by legal moves.

---

## 7. What is not yet answered

- **The scaling crossover has not been measured.** The whole point — does backward
  overtake forward as puzzles grow — has data prepared (the 6-robot and 24×24 boards
  exist and nets are trained) but **no head-to-head has been run**;
  `scaling/results/` is effectively empty.
- **The bigger-grid axis is blocked in practice.** The realizer's entry points now infer
  board size (`eval/realize.py:74`), but the scaling harness still hard-skips the backward
  planner at any non-16 grid (`scaling/run_config.py:106`, `skip_back = g != 16`). Until
  that gate is threaded and flipped, backward cannot be scored on the 24×24 boards whose
  data already exists.
- **Backward self-play** (§4) is the least-developed arm and depends on Lever A (§5c) to
  become a fair contender.

---

## Appendix — reproducing the ceiling analysis (§5)

```
cd supervised_valuenet
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python3 analysis/artifacts/ceiling_probe.py \
    analysis/artifacts/ceiling_probe_instances.json /tmp/out.json
```

`ceiling_probe_instances.json` is the 49 anytime-failing puzzles selected by row index
from `final450_backward_anytime.json`. The script runs the hand-coded subgoal A\*
(`skeleton/astar.py`) as an exhaustive anytime search and labels each puzzle
`NO_COMPLETE_PLAN` / `NO_REALIZABLE_PLAN` / `REALIZABLE_EXISTS`. Runs on CPU in well under
a minute total; no GPU, no network, no training.
