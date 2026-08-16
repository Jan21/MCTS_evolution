# Self-improvement loops — review memo

**(1) Iterated or one-shot?** Both exist, doing different things. `subgoal_selfplay/`
(train_iterate.py) IS a genuine iterated loop — generate→label→retrain repeated 6
rounds, warm or from-scratch, with a replay buffer (last 5 rounds) and a progress-check
probe each round. It's real and it worked (FINDINGS 7): probe regret 0.724→0.222 over 6
rounds, matching supervised solve rate at 18% fewer search steps. But it only ever ran
at BASE scale (train_boards 1000-1799, small boards) — never on the 32×32+ boards where
the exact solver can't supply ground truth. The twin-planner work (FINDINGS 44/45/50/
66/67/71/74) is NOT iterative: NN labeler → one corpus → one planner retrain → compare
to exact-trained planner, single round, no feedback of the new planner's own solves back
into new labels. So: iteration exists, but only where it's least needed (base, where
exact labels are cheap); the frontier where iteration matters (beyond exact) is
one-shot only.

**(2) A clean self-improvement experiment beyond the exact solver.** Run
`subgoal_selfplay`'s actual loop (not the twin one-shot) with `train_boards` set to
32×32+, seeded from the banked v1/v2 NN labeler or from a twin-trained planner, with the
existing progress-check probe repointed at NN-labeler-certified references (never exact,
since it's absent there) plus the frontier-solve/strict-playability stats already
logged per iteration. Risks: (a) **drift** — sibling costs are supplied by the
networks' own search, so at low-fidelity sizes (FINDINGS 74: 82% argmin gates a 21-pt
solve-rate collapse) errors can compound round over round with no exact anchor to catch
it; (b) **collapse** — the value net is documented bistable (FINDINGS 44/50, 7/8 cold
trainings collapsed) and a mid-training collapse at scale (FINDINGS 57) already
happened once — CollapseStop exists but only guards single-run training, not a whole
self-play arm silently drifting into a bad mode across rounds; (c) **unverifiable
labels** — beyond the solver's n≤64 envelope (FINDINGS 59) there is no ground truth at
all to progress-check against, only self-consistency, so "improvement" could be the
buffer overfitting to the search's own biases (epsilon-wildcards mitigate but don't
solve this).

**(3) Next-step ideas.**
- Run the existing `subgoal_selfplay` loop unmodified but with boards resized to
  32×32/g32r4, seeded warm from the banked twin planner; log fidelity/drift per round
  against the twin gate metric. Payoff: first real iterated-frontier data point.
  Cost: medium (reuses code, ~few GPU-hours/round × 6).
- Add a CollapseStop-style guard at the loop level (abort/rollback a self-play round if
  probe regret regresses past a threshold vs the buffer's running best). Payoff: makes
  scaling the loop safe to leave unattended. Cost: small (extends existing detector).
- Feed twin-corpus labels as the round-0 seed instead of exact/warm checkpoints, then
  let self-play iterate purely on its own solves at 32×32+ — tests whether iteration
  can climb past the 82-91% argmin ceiling FINDINGS 74 measured in one-shot. Payoff:
  large if it works (breaks the fidelity ceiling); cost: medium-large (multiple rounds,
  each needing a fresh progress-check design since exact is absent).
- Cross-check iterated self-play plans against the certified-descent gate (§66's
  argmin/gap machinery) each round instead of only the probe's regret, to catch drift
  before it compounds. Payoff: cheap early-warning system. Cost: small.
- Deliberately run a from-scratch (no warm start) self-play arm at 32×32 as a collapse
  stress test, since from-scratch trains at full strength (3e-4, 6 passes) and has no
  supervised anchor at all. Payoff: characterizes worst-case drift/collapse risk before
  trusting any iterated frontier result. Cost: medium.
