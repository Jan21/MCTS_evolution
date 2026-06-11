# A* vs MCTS for subgoal partial-plan search

*2026-06-11*

## Question

The project began as MCTS (`MCTS_evolution`) and pivoted to A*. Is A* actually
the better backbone for this problem, both now (hardcoded heuristics) and later
(two neural networks: one proposing subgoals, one evaluating them)?

## Conclusion

A* is the better fit. The problem's structure points to best-first search with a
heuristic, not tree search with rollouts. MCTS would add machinery that is blind
to the cost structure this problem hands us for free.

## Why A* fits the structure

- **Single-agent, deterministic, fully observable.** No opponent, no
  stochasticity. MCTS's core strengths (averaging over uncertainty, balancing
  exploration vs exploitation under noise) address problems this domain does not
  have.
- **Additive cost that decomposes over segments.** `f = g + h` is exact and
  natural here: fixed segments contribute their real move count (`g`), open
  segments contribute a relaxed estimate (`h`). A* is built for this. MCTS
  maximizes a reward-to-go estimated by rollouts and ignores the additive bound
  we already have in closed form.
- **Admissible, near-perfect heuristic.** The relaxed shortest-path length is
  always <= the exact one, so A* is optimal. The earlier research finding (all 12
  variants converge to the same plan because the heuristic predicts true cost so
  accurately) is itself the tell: when the heuristic is this good, you want the
  method that exploits it most directly. That is A*, not random rollouts.
- **Bounded branching.** `propose_subgoal_states` returns a small candidate set
  per open segment, so every child can be expanded. MCTS's advantage of sampling
  when the action set is too large to enumerate does not apply.
- **Shallow solutions.** Plans complete in fewer than ten expansions; there is no
  deep horizon that needs rollout-based value estimates.

## The neural-network phase (the real question)

The two-network goal looks AlphaZero-shaped, and AlphaZero uses MCTS. But
AlphaZero-MCTS earns its keep in adversarial, high-variance games where
visit-count statistics drive policy improvement. This is single-agent cost
minimization, which is the home turf of **learned heuristic search** (neural A*,
learned beam search):

- Network #2 (score) becomes the learned heuristic `h` (cost-to-go).
- Network #1 (propose) becomes the learned policy that orders and prunes
  branches.
- Search-improved plans, plus the move counts they achieve, are the training
  targets for the value network. The bootstrap loop closes **without** MCTS.

The `skeleton/` harness already spans the full spectrum needed: full A*
(optimal), `beam=k`, and `beam=1` (pure greedy policy rollout). That covers the
one regime where MCTS-style robustness matters, a noisy value network early in
training, via beam width and, if needed, weighted A* or stochastic proposal
sampling. This is cheaper and simpler than UCT plus rollouts.

## Caveat

If the learned `score` ends up non-admissible and badly miscalibrated early in
training, greedy A* can be misled where MCTS averaging would be steadier. The
remedy is beam width or weighted A* inside the same harness, not switching
paradigms.

## Takeaway

Use A* (more precisely: best-first search with a learned heuristic) as the
backbone. Keep the two heuristics as clean swap points (`skeleton/heuristics.py`:
`propose`, `score`). Reach for MCTS-flavored ideas (exploration, sampling) only
as optional robustness knobs layered onto the A* harness, never as the default
engine.
