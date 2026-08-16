# Efficiency claim review

**(1) Fair unit?** Not by default — FINDINGS §25 explicitly flags that raw
"expansions" mixes a cheap forward move-step with a costlier backward subgoal
decision. They built a matched unit (`slide` calls, the shared physics
primitive) and found the advantage survives but the multiplier moves: ~99x on
expansions vs ~90x physics-only on the 5-puzzle pilot; an earlier 630x claim
was retracted after finding it double-counted featurization slides. The
matched-unit check is base-scale only (n=5-20), and the mean is dominated by
a heavy tail (one unsolved puzzle: 106,817 slides, mostly park-repair,
exceeding forward's costliest solve) — §25 insists on median+tail reporting,
not mean, for exactly this reason.

Wall-clock is reported, but only at one rung (32×32, same-machine KPI tile):
31s vs 23min gradable, 58s vs 40min frontier — agrees with and even exceeds
the step-ratio. But §27 flags disagreement at base scale: wall-clock actually
*favors forward* (0.99 vs 1.23s/puzzle) despite backward's step advantage —
"the time advantage is real only at scale." So wall-clock is not a uniform
confirmation; it's one same-machine anecdote at the top of the scale plus one
contradicting data point at the bottom.

**(2) Budget curve:** Not a uniform backward win. `report_sections_budget.py`
+ report.html show forward *catches up and overtakes* backward on gradable
sets by ~400-800 steps at 16×16 (all robot counts) and 24×24 — differences
go negative (e.g. base: -4.4pp at 1200). Backward stays ahead at every budget
only on: (a) frontier/beyond-oracle sets at all scales, and (b) gradable sets
at 32×32, where forward never closes the gap (still -12pp at cap). So the
efficiency-plus-quality story is scale/hardness-dependent, not universal.

**(3) Next steps**
1. Extend matched-unit accounting (already built, §25) from the 5-20 puzzle
   base-scale pilot to every rung/budget in the sweep. Payoff: turns the
   fairness answer from an anecdote into the headline number. Cost: medium.
2. Extend the same-machine wall-clock KPI tile from 32×32-only to all six
   rungs. Payoff: resolves whether the base-scale inversion is a one-off or a
   real small-scale pattern. Cost: small (infra exists).
3. Report solve-rate-vs-wall-clock (or vs-physics-work) anytime curves
   instead of vs-step-count. Payoff: sidesteps the unit-fairness debate
   entirely with one plot. Cost: medium.
4. Always report median+tail (never mean) compute per cell, with park-repair
   cost isolated — the current headline risks an adversarial "the mean says
   backward is worse" rebuttal. Cost: small (reporting-only).
5. Fit a per-step-type cost model (slide-calls per subgoal step vs per move
   step, by rung) and re-derive the whole budget table in physics units for
   zero extra search — mirrors the report's existing zero-compute
   reconstruction trick. Payoff: single defensible "Nx" number replacing the
   step-count headline everywhere. Cost: medium-large.
