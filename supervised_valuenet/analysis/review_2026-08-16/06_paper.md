# Memo — the paper(s)

## 1. Two papers

A (planner) asks *which formulation scales*; B (labeler) asks *how good synthetic
supervision must be*. B's result is domain-general and would be buried inside A. B is
nearer submittable; A is the thesis and has the bigger gaps.

- **Main (A):** *Executable Abstraction: Measuring and Lifting the Expressiveness
  Ceiling of Subgoal Planning* (AIJ/JAIR).
- **Companion (B):** *Good Enough Answer Keys: A Size-Free Neural Labeler and the
  Fidelity Threshold for Replacing an Exact Solver.*

**Abstract (A).** We compare two learned planners on Ricochet Robots — move-level
forward search and backward search over a subgoal vocabulary — at identical budgets,
counting a puzzle solved only if its plan plays out under full physics, which cuts the
subgoal planner's historical 99.6% to 53.1%. Bug fixes plus playability checking inside
the search recover 89.1% at ~7 steps, and a network-free probe shows the remainder is a
property of the plan language, not of training: for 9.3% of puzzles no playable plan
exists in the vocabulary. Controlled vocabulary surgery lifts that ceiling
90.7 → 97.6 → 99.6%, and we quantify the exact solver's mortality on two hardness axes
(0→41% failures in robot count, 48→61% in grid), bounding what supervised move-level
training can be given. On the whole pinned pool the subgoal planner wins every rung
above base scale by +7.8 to +47.8 points (p<0.0001, board-clustered bootstrap), while
forward keeps the quality crown wherever the oracle can still grade. We report the
counterweights: efficiency in a matched physics unit, budget curves showing the 16×16
collapse is partly a cap artifact, and a negative result — retraining on
extended-vocabulary labels is bistable and usually hurts.

## 2. Figures (6) — all six already have data

1. **Schematic** (no data): slide physics and the two formulations.
2. **Oracle mortality:** two curves (robots, grid) plus the 10×-budget probe.
3. **Headline table:** per rung, graded/frontier/pooled, McNemar p and clustered CIs —
   `eval/results/stats_tests.json`.
4. **Ceiling arc:** 53.1 → 89.1 → 90.7 → 97.6 → 99.6, plus failure families —
   `analysis/artifacts/ceiling_probe_*.json`.
5. **Two currencies:** search steps vs median `slide` calls (64 vs 5,776), beside
   solve-vs-budget curves — `budget_curves_by_rung.json`, `compute_accounting.json`.
6. **Retraining bistability:** val_regret vs frontier solve rate, one point per retrain
   — `analysis/artifacts/{seed_spread,valnet_modes}.json`.

(B's own two: the flat 17→64 fidelity ladder; the three-cell dose-response.)

## 3. Missing, ranked

1. **kSubS-style forward-generative subgoal baseline** (large). Otherwise "subgoals win"
   can't be separated from "backward regression plus a hand-built vocabulary wins".
2. **Forward self-play at scale** (medium). The at-scale opponent is oracle-supervised
   only, so frontier collapse confounds formulation failure with teacher death.
3. **Wire by-reference (B2) into the learned proposal path** and policy training
   (medium). Today it is a ceiling result plus an admitted integration gap.
4. **Variance and tuning symmetry** (medium): forward is single-seed at scale and got lr
   rescues while backward is bistable. Seeds, lr sweep, k-sweep — or concede them.
5. **Second domain, or a case-study reframe** (large / free). Lunar Lockout or Atomix is
   the cheap real option; else own "engineered hierarchy vs generic flat".
6. **B's own gaps** (medium): the 89% and 82% cells are single-seed, and fidelity
   co-varies with corpus size and robot count — a few seeds, one decoupling cell.
7. **Related work for B** (small): noisy-label thresholds, verifier-filtered synthetic
   data, distillation, model collapse, size generalisation. A's base is strong already.
