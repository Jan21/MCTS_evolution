# Addendum — subgoal self-play: what to measure, which levers to pull, and the scaling plan

Context for continuing the subgoal-planner self-play work (`subgoal_selfplay/`). Nothing here
demands abandoning the current run; exploration is fine. But every experiment should be scored
against the metric in §1, and the levers in §3 are ordered by expected payoff per unit of work.

**The two-phase goal.**
- **Phase A (parity):** on the current 16×16 / 4-robot benchmark, the backward subgoal planner
  should match the forward move planner *in the honest metric* (defined below).
- **Phase B (scaling):** show the backward planner's advantage *grows* as the environment gets
  more complex, along both axes — bigger grid (more states) and more robots (more actions).

Phase A is a prerequisite for Phase B: the backward planner's current failure mode gets *worse*
with problem depth (numbers in §2), so scaling up before fixing it would make the comparison
worse, not better.

---

## 1. The metric that decides everything: strict executability

A backward plan is a chain of subgoals, not moves. To compare against the forward planner it
must be converted into an actual legal move sequence — every robot sliding under full physics,
all other robots present as blockers. That conversion is `eval/realize.py::strict_moves`
("strict realization"). The agreed comparison protocol (`COMPARISON.md`, `eval/compare.py`)
defines:

- **solved** (backward) = strict realization succeeds within the search budget;
- **cost** = the strict move count; **regret** = strict moves − d\* (d\* = exact optimum,
  oracle-computed once per benchmark instance, never used at inference);
- abstract plan cost (`plan.cost()`, the planner's internal count that assumes blocking robots
  out of the way) must **never** be compared to d\* — on the current benchmark 65 of 450
  abstract plans are *shorter than the true optimum*, i.e. impossible as played.

### Where the backward planner actually stands (bench450, matched budget: 1200 expansions, k=5)

| criterion | result |
|---|---|
| finds a complete subgoal plan | 450/450, at a mean of **2.9 expansions** (budget: 1200), 0.34 s/inst |
| plan survives strict realization ("solved") | **239/450 = 53.1%**, regret 1.201 |
| forward planner, same protocol (candidate-scored ckpt) | 100%, regret 0.067, 36 expansions |

Strict-solve rate by puzzle difficulty (d\* = optimal move count):

| d\* | 1–3 | 4–6 | 7–9 | 10+ |
|---|---|---|---|---|
| backward strict-solve | 95% (52/55) | 67% (112/166) | 34% (70/204) | 20% (5/25) |

Read this table twice. The search is essentially never the problem (2.9 of 1200 expansions
used). The **only** gap to parity is that deep plans stop being playable — segments get blocked
by robots the plan abstraction assumed away. Fixing executability *is* Phase A.

---

## 2. What the current self-play loop optimizes — and its blind spot

The running warm-start loop (`subgoal_selfplay/train_iterate.py`) works: generation solve rate
96.6%, probe regret improved 1.48 → 1.18 after one round. But three facts limit what it can
prove as configured:

1. **`strict_filter` is OFF** (`config.py:50`; the gate is at `selfplay.py:233`). Winning plans
   are turned into training labels whether or not they can be played. The loop can therefore
   reinforce exactly the plan patterns that fail realization.
2. **All costs are abstract.** `win_cost = plan.cost()` (`selfplay.py:239`) and the probe's
   regret reference (`train_iterate.py:131`) are both in the relaxed blocker-clearing count.
   The probe number (1.18) is a fine *trend* indicator but is not the honest metric and cannot
   be quoted against the forward planner.
3. **Executability is measured nowhere in the loop.** No stat says what fraction of winning
   plans is strictly playable, so progress on the metric that decides the comparison is
   invisible.

Improving the abstract objective may coincidentally improve strict executability — or not. As
is, the loop cannot tell.

---

## 3. Levers, in order

### L0 — run the already-built "anytime" benchmark first (zero code, ~minutes)

`eval/compare.py` already implements the biggest candidate lever, and it has **never been
benchmarked**: `--backward-anytime` (flag at `compare.py:404`, logic at `compare.py:142–147`)
makes the search discard a completed plan that fails strict realization and *keep searching*
within the same expansion budget. Given the planner uses 2.9 of 1200 expansions, there is
~400× slack to try alternative plans. The realization check is pure game physics (the same
slide rules the forward planner applies at every expansion) — no oracle, no solver knowledge;
it is fair under the protocol.

```bash
CUDA_VISIBLE_DEVICES=<empty-gpu> PYTHONPATH=. python -m eval.compare \
  --instances eval/data/bench450.jsonl --expansions 1200 --k 5 \
  --backward-policy checkpoints_backward/policy_v2.ckpt \
  --backward-value checkpoints_backward/value_v2.ckpt \
  --backward-anytime --forward-ckpts "" --device cuda \
  --out eval/results/comparison_backward_anytime.json \
  --md eval/results/COMPARISON_anytime.md          # do NOT clobber root COMPARISON.md
```

This single number tells us how much of the 53% → parity gap the *current* nets can already
close when simply forbidden from returning unplayable plans. Everything else calibrates
against it.

### L1 — make the loop see the metric (~30 lines)

Add to each generation round and to the probe: fraction of winning plans passing
`strict_moves`, mean strict cost, and a strict-regret probe (realize the probe plans; compare
to d\* or at least to the strict cost of the exact solver's plan). Until the loop logs these,
no self-play result can claim progress toward Phase A.

### L2 — point the training pressure at executability

Cheapest version: set `strict_filter=True` — the flag already exists and drops unplayable
winners before labeling. Expect training data volume to shrink (only ~53% of winners pass
today; measure the real fraction on training boards) and the kept distribution to skew toward
plans that execute — which is the point. Watch data volume per round; if it starves, raise
`instances_per_iter`.

Stronger version: label with strict costs — `win_cost = strict_moves(win_plan)` instead of
`plan.cost()`, and realize each sibling completion the same way (requires
`selfplay.py::nn_astar_from` to return the completed plan, not just its cost — small change).
Two cautions: (i) at inference `f = fixed_g + value` mixes an abstract `g` with a strict-trained
value head — ranking quality is what matters, but watch for regressions; (ii) the warm-start
value net was trained on abstract costs, so expect a small scale shift (strict ≥ abstract;
means 7.15 vs ~7.7 on the current bench). Sequence: `strict_filter` alone first (unit-safe),
strict-cost labels only if the filter plateaus below parity.

### L3 — anytime inside generation (the two halves reinforce each other)

If L0 confirms anytime realization-checking as the eval planner, generation should match it:
put the same realize-check into `selfplay.py::nn_astar_traced` (reject unplayable complete
plans, keep searching, return the first playable one). Compared with `strict_filter` (which
throws the instance away), this keeps the instance and finds a playable winner for it — more
training data, from exactly the distribution the eval planner produces. Training then sees
only playable plans, and the value/policy nets gradually learn which plan shapes fail — the
learned analog of the check itself.

### L4 — only if L0–L3 plateau below parity: fix the abstraction gap at the source

Diagnose the residual strict failures (realize's log distinguishes: unreachable segment under
BFS, no mover, cyclic ordering, target off-goal). If most failures are "a robot not in the
plan sits on the path", the proposal vocabulary is the limit: `skeleton/heuristics.py::propose`
offers bottleneck/support/helper candidates but nothing that says "move that robot out of the
way". Options, increasing effort: a blocker-clearing candidate type; a value-net input feature
marking blocked segments; realizer-side single-robot detours. Evidence first — this is the
only lever that touches the plan formalism.

### Keep the current run as the control arm

The running strict-filter-OFF loop is a useful ablation: "self-play on the abstract objective"
vs "self-play on the strict objective" is exactly the comparison that shows the objective
mattered. Name new run dirs distinctly (`runs_warm_strict/`, …) and keep `iterN_records.jsonl`.

---

## 4. Phase B — the scaling comparison

### Why scaling should favor the backward planner (now with measured curves)

Forward search effort explodes with solution depth. Mean expansions used by the *best* forward
model (candidate-scored) on bench450, by difficulty: d\* 1–3 → **3**; 4–6 → **11**; 7–9 →
**50**; 10+ → **159**. Roughly ×3–4 per difficulty band; at a fixed 1200-expansion budget the
forward planner runs out of search as mean d\* grows. The backward planner completes plans at
~3 expansions *regardless of depth* — subgoal plans grow far slower than move counts.
Meanwhile its executability decays with depth (95→20%, §1), which is why Phase A comes first:
after the fix, deeper boards should hurt forward (budget) much more than backward
(compression).

The other scaling asymmetry is **training cost**. The forward planner's best recipe
(candidate-scoring) needs the exact move oracle for every candidate child; the oracle itself
degrades with scale — on the g16r6 smoke bench (6 robots), **3 of 6 instances already exceeded
the oracle's caps** (200k expansions / 60 s). The backward self-play loop needs no oracle at
all. At scale this becomes: forward's strongest training gets expensive-to-impossible exactly
where backward self-play keeps working. Report labeling CPU-time and oracle-failure rate per
configuration alongside quality — they are first-class results, not footnotes.

### The two axes measure different things

- **Robot axis (g16r6, g16r8)** — action space. Forward branching grows (4 moves × R robots)
  and the joint-state oracle strains (already visible at r6). But more robots also means more
  potential blockers, i.e. more executability pressure on backward plans. Both systems are
  stressed at their weak point: genuinely informative, outcome not predetermined.
- **Grid axis (g24r4, g32r4)** — state space. Deeper solutions; forward's budget problem grows
  fastest here; backward's subgoal compression should shine — this is the likelier crossover
  axis.

### Readiness

- **Robot axis is runnable now.** g16r6 boards + forward/backward training data already exist
  (`scaling/data/g16r6/`); `realize.py` is robot-count-agnostic at 16×16. Missing: train both
  systems (`scaling.train`), build the full bench (`scaling.bench`), run `eval.compare` under
  the config env (`scaling.run_config --config g16r6` prints the exact sequence).
- **Grid axis is blocked on one thing:** `eval/realize.py` and `simulate.py` default
  `size=16` everywhere and never read `RR_GRID` (readers today: `scaling/*`,
  `move_planner/net.py`, `train/encode.py`, `nn/gen_grids.py` — not the physics modules).
  Fix: derive `size` from the board itself (`len(env.grid_data) == size²`) or thread `RR_GRID`
  through `wall_sets / slide / segment_moves / verify_plan / strict_moves / abstract_moves`
  and their callers (`eval/compare.py`, `subgoal_selfplay/selfplay.py`). Until then, no
  backward score exists at G≠16 (g24r4 training data is already generated and waiting).

### Protocol per configuration (fixed in advance, to keep the result honest)

Same instance file for every system; 1200 expansions, k=5; strict moves only; solve rate over
all instances; regret only over instances whose d\* the oracle produced; report mean
expansions, wall-time, labeling cost, oracle-failure rate. Fair pairings:

- oracle-heavy training: backward supervised vs forward candidate-scored;
- oracle-free training: backward self-play vs forward self-play (from-scratch);
- naive forward (trajectory-only supervised) stays in the table as a reference row, not the bar.

---

## 5. Success bars

- **Phase A (bench450):** strict solve ≥ 97.5% with strict regret ≤ 0.35 — inside the forward
  band (naive 84.7%/0.354; candidate-scored 100%/0.067) — while keeping mean expansions at
  least an order of magnitude below forward's. Stretch: ≥ 99% / ≤ 0.1.
- **Phase B:** at ≥ 1 scaled configuration, backward (strict) beats the best forward system at
  matched budget — or matches it with ≥ 10× fewer expansions and materially cheaper training
  labels. A clean negative (no crossover found on either axis) is also a reportable outcome.

---

## 6. Hygiene

- Shared machine: pin an *empty* GPU (`nvidia-smi`, then `CUDA_VISIBLE_DEVICES=<idx>`); GPU 1
  and 2 currently carry long-running training jobs.
- Never point `RR_ENV_DIR` across configurations; one boards dir per config.
- `eval/compare.py` rewrites its `--md` target: pass an explicit path for exploratory runs so
  the canonical `COMPARISON.md` only changes when the run is meant to update it.
- Quote no abstract-cost number against d\*, ever (§1).
