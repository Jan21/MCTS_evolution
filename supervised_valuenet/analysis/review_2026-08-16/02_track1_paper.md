# Track 1 as a paper — reviewer memo

## 1. Strongest, cleanest result

The pooled six-rung ladder: both planners, same pinned 450-puzzle pools, same
1,200-step budget, every claimed solve replayed on the real physics, paired
McNemar plus board-clustered bootstrap. It needs no oracle, no selection
argument, and no trust in either system's bookkeeping.

Abstract sentence: *"Under an identical search budget on identical puzzle pools,
a planner searching over subgoals loses to a move-level planner at 16x16 with 4
robots (95.6% vs 100%) but overtakes it on every harder configuration, the
margin growing monotonically along both hardness axes to +47.8 points at 32x32
(77.8% vs 30.0%), all p<0.0001 and every solve certified by replay."*

## 2. Under-supported claims

- **Single seed, asymmetrically.** One draw per arm; forward's base cell is
  best-of-4 against one backward run. Where seeds were measured the spread was
  28 points and bimodal. Fine for 48-point margins, fatal for the small ones
  (16x16/8r graded "parity", 32x32 graded +12).
- **Solve rate carries the paper; quality is quietly lost.** Backward runs
  ~2.0-2.1 moves above optimum vs forward's 0.067, and beyond the oracle quality
  is unmeasured exactly where backward wins.
- **"Matched budget" is a contested unit.** One search step is not equal work,
  and the extended-budget probes use n=24-32 subsamples, not the scored sets. At
  16x16/6r, 4x budget buys forward +21.9 points; only 32x32 is budget-robust.
- **The "full language" chapter.** By-reference steps were dead code in the eval
  driver, so every B2 row is B1 plus park repairs and the 99.6% ceiling is
  unreachable — extension 2 has no causal result.
- **Retraining bistability** rests on 4 seeds at one rung: a phenomenon, not a
  mechanism.
- **Missing corner:** no 32x32 x 8 robots, where both axes are maximal.

## 3. Next steps

1. **Seed both headline arms 3x at two rungs (16x16/8r, 32x32/4r)** because the
   ladder is single-draw and measured seed spread exceeds several reported
   margins. Payoff: headline becomes defensible. Cost: medium.
2. **Add the 32x32 x 8-robot rung** (labels now feasible via the NN labeler)
   because the claim is "margin grows along both axes" and that cell is missing.
   Payoff: strongest single figure. Cost: small-medium.
3. **Wire by-reference into the proposal path, featurization and policy
   training** because the shadow probe says 97% of expansions would offer one
   and 4 ceiling points sit unclaimed. Payoff: a real ablation instead of a
   ceiling plus apology. Cost: medium.
4. **Make an iso-wall-clock curve the primary fairness figure and extend budget
   probes to full frontier sets** because the matched-step unit is the first
   referee objection and subsampled probes showed it matters. Payoff: closes the
   fairness attack. Cost: medium-large.
5. **Add an external learned-subgoal baseline (kSubS-style)** because otherwise
   the paper shows "our subgoals beat move-level search", not "subgoals beat
   move-level search". Payoff: scopes the novelty claim. Cost: large; fine as
   declared future work.
