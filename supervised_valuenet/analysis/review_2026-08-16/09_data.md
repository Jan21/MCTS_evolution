# Training data and labels — memo

**(1) How labels are made, and known weaknesses.** A hand-written exact
backward/forward solver (now Rust-ported, ~50-180x) self-labels boards by
searching under an iteration budget (`--budget-iters`, default was 50,000,
recalibrated to 5,000 for affordability). Three concrete weaknesses are
documented:
- **Cap depletion (F32/33):** cutting the budget 50k→5k didn't just shrink
  the corpus, it selectively strips "by-reference" candidates (reusing an
  already-placed robot) — the exact type the B2 vocabulary exists to teach.
  Measured 13.5%→4.8% by-reference share at cap-matched settings; going
  5k→20k recovers most of it (4.8%→11.0%), at 9-12x generation cost.
- **Training bistability (F44/45), the bigger issue:** warm-started value-net
  retraining on identical data/recipe is *bimodal* by seed — good mode
  (val_regret ~0.74-0.80, frontier ~78%) vs collapse mode (val_regret
  ~2.1-2.5, frontier ~34-56%), a near-3x separation visible in val loss
  before any benchmarking. The corpus doesn't cause recovery directly — it
  gates *access* to the good basin (2/4 seeds reach it on rich corpus at
  g16r6 vs 0/3 on depleted; 0/3 at g16r8 regardless). At 8-robot rungs no
  good basin has ever been observed, so retraining is currently a net
  negative there (-26 to -13 points vs zero-shot).
- Exact labeler itself has a real non-admissibility/order-sensitivity bug on
  rare instances (rust_datagen README), independent of budget.

**(2) Does quantity/quality matter — is there a scaling curve?** Yes for
composition (cap 5k→20k recovers by-reference share, dose-response-like),
but the *causal* story is confounded by seed/mode: F44 shows the "corpus
effect" (F36's +22.4 pts) is observationally identical to a lucky seed draw
at 50% base rate. No clean quantity-only scaling curve exists yet — mode
gating dominates over raw label count in the retraining results measured so
far. Separately, NN-labeler battery work (F53/54) shows *label quality
itself* extrapolates gracefully with training-corpus board-size mix (flat
regret to 3.2x train size for the labeler net, not the planner).

**(3) Next-step ideas, data-focused:**
1. **Best-of-k value-seed selection as standard practice** (F44 already
   proposes this) — train k=3 seeds, bank lowest val_regret, benchmark once.
   Payoff: turns retraining from a coin flip into a reliable use of existing
   data. Cost: small (~2-3x value-training compute per config, no new data).
2. **Diagnose why 8-robot rungs have no good basin** — try cold value init,
   different warm-start source, lr/schedule sweep at g16r8/g24r8 before
   concluding retraining is dead there. Payoff: could unlock the two rungs
   currently regressing vs zero-shot. Cost: medium (several training-recipe
   variants x 2 rungs).
3. **Generate cap-20000 (or higher) corpora at all rungs, not just the four
   already done** — direct extension of F33's measured recovery. Payoff:
   more by-reference examples, likely narrows the composition gap uniformly.
   Cost: medium-large (~50 nh at base scale per F32/33; cheap at small rungs).
4. **Curriculum ordering for retraining** (already validated for the
   NN-labeler port in F50/52 — dropping curriculum caused 7/8 collapsed
   runs) — check whether the *production* B2 retrain recipe also lacks a
   curriculum warmup; if so, add one. Payoff: could be a direct fix for the
   bistability problem itself, not just a selection workaround. Cost: small
   (recipe change + a handful of reruns to test).
5. **Fix the exact-labeler's non-admissible/order-sensitivity bug** before
   trusting label quality further at scale. Payoff: removes a known
   ground-truth contamination source ahead of any new large-scale data push.
   Cost: small-medium (bug is already isolated with repros in
   `rust_datagen/golden/s511_evidence/`; needs an adjudication + fix).
