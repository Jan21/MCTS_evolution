# Where and why the backward planner fails

**(1) Taxonomy (g24r8 frontier set, n=289, 128 backward failures, `failure_examples.json`):**
- **never_completed 122/128 (95%)** — search exhausts its 1200-expansion budget without ever assembling a complete abstract plan. Median failed run burns ~790s, close to the cap.
- **all_rejected 6/128 (5%)** — an abstract plan completes but every candidate fails move-by-move playout.
- Only **5/128 (4%)** of backward failures are rescued by the forward move-level planner — these are genuinely hard for both, not backward-specific blind spots.

Cross-cutting stratification (`frontier_strata.py/json`, oracle-independent proxy): failures concentrate in **"goal OPEN"** puzzles (no wall backing the goal cell, so a helper robot must be parked first) more than "goal walled" ones — e.g. g24r8: goal-OPEN solved 49.0% vs goal-walled 73.4%; g24r4 similarly (59.9% vs 49.0%, smaller gap). Travel distance stratifies less sharply and non-monotonically.

**(2) Biggest failure kind and fix.** The dominant mode is budget exhaustion (never_completed), and it's worst on goal-OPEN puzzles that require parking-robot subgoals. FINDINGS 44/45/47 rule out "just retrain harder": value-net retraining is bistable (good vs bad optimization basin, visible in val_regret pre-benchmark), and at every 8-robot rung tested (g16r8, g24r8) **zero draws landed in the good basin across 3+ seeds** — this is a rung-level optimization pathology in the current warm-start recipe, not a data-volume problem. So retraining is currently a dead end at 8-robot scale; the zero-shot value net stays the best configuration. The more promising lever is the **plan language/search itself** for goal-OPEN cases specifically, since that's where the search stalls out.

**(3) Next-step ideas:**
- **Raise/vary the expansion budget** on a small never_completed sample to see if failures are truly budget-bound vs. value-quality-bound (diagnostic, then maybe a real fix). Payoff: clarifies root cause. Cost: small.
- **New value-net warm-start/optimization recipe** (cold start, different lr schedule, different warm source) targeted at 8-robot rungs, since 3+ seeds under the current recipe never reach the good basin. Payoff: could recover ~13-20 frontier points seen lost in retraining. Cost: medium.
- **Richer plan-language primitives for helper-robot parking**, since goal-OPEN puzzles are the harder stratum and require exactly this subgoal type. Payoff: could shrink the majority failure class. Cost: medium-large (design + relabel).
- **Instrument never_completed runs** (search-tree logging: plateau vs. fanout explosion) on a handful of failures to distinguish "search too slow" from "search converging to wrong plans." Payoff: sharpens which of the above fixes to prioritize. Cost: small.
