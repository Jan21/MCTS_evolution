# Paper-B leverage ranking of follow-up experiments (2026-08-17)

Opus agent, this session; grounded in analysis/review_2026-08-16/ memos and
PAPER_PLAN.md. This brief drove the 2026-08-17 launch decision.

## The framing that decided everything

Paper B's title claim ("the Fidelity Threshold") is causal, but
PAPER_PLAN.md:115 forbids the sentence — three cells, mostly single-seed,
with **fidelity, corpus size and robot count moving together** (§73/§74).
The gap between title and permitted claim is a CONFOUNDING problem, not a
seed problem. `03_nnlabeler.md:13-15` ranks it first; `06_paper.md:58`
asks for "a few seeds, one decoupling cell".

## Ranking (full arguments in the session log)

1. **Controlled label-corruption arm at g24r4 + size-matched control** —
   the only design that isolates fidelity causally: hold boards, instances,
   robot count, corpus size fixed; move only argmin agreement. Corrupt
   structurally (argmin → runner-up, near-tie groups), anchor at the g24r8
   collapse dose (82.2%) before interpolating. Negative outcome equally
   valuable: if 82% at 4 robots does NOT collapse, the g24r8 result is
   about robot count — learn it before reviewer 2 does.
   → **LAUNCHED: jobs 4678370 (d1000 size ctl) / 4678371 (d860) /
   4678372 (d822), ~1.9 nh total** (cheaper than the brief's 10-nh estimate
   because g24r4 retrains measure 0.54 nh).
2. **Seeds, selectively** — g32r4 twin is the fragile cell (effect < known
   seed spread). → **LAUNCHED: 4678373 (1.63 nh)**. Others deferred.
3. **B2 rescue** — Paper A's item; re-opens an abandoned claim. Not funded
   here (design archived as 03_b2_rescue_design.md).
4. **80/96 downstream validation** — worst risk/reward: no interpretable
   null, 1,350-record training set, and the feasibility study found the
   planner is not size-free (01_8096_planner_feasibility.md). No compute;
   the architectural asymmetry is the citable finding.
5. **Nothing** — viable per the review (Paper B "nearer submittable",
   gaps ranked 6th of 7) but requires retitling away from "threshold".

## Pre-registered reading rules for the corruption arms

- d1000 (size control) vs the exact g24r4 row: any gap is the
  corpus-shrinkage effect alone. Expect ≈ equivalence (twin at 91% matched
  exact with MORE damage than pure subsampling).
- d822 collapse (solve-rate drop of the g24r8 order) → fidelity is causal;
  the paper may claim the threshold.
- d822 ≈ exact → the g24r8 collapse is a robot-count effect; the paper's
  claim retreats to "fidelity band, 4-robot configs robust to 82%".
- d860 sits between → monotone dose-response on a single uncounfounded axis.
- All three arms share one subsample (corrupt.py, seed 11) and the standard
  pinned g24r4 bench; compare against BOTH exact seeds' rows (88.4/88.4
  solve) and the twin band (91.4/87.9).
