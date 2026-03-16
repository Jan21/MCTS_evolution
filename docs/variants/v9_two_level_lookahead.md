# V9: Two-Level Lookahead A*

## Algorithm Description
Before committing to a subgoal, estimates the cost of the sub-problems it creates:
1. For each candidate subgoal, compute:
   - Immediate costs: parent→bn, bn→child, sp→helper (exact where possible)
   - Lookahead: if bn→child is still open, what's the best subgoal for resolving it?
   - Score = immediate_cost + max(0, best_sub_proposal_score - relaxed_estimate)
2. Rank proposals by this lookahead-augmented score
3. Otherwise standard A* with priority queue

## Heuristic Formula
`f(plan) = plan.cost()` for PQ ordering.

Proposal ranking: `lookahead_score = immediate_cost + lookahead_bonus`
where `lookahead_bonus = max(0, next_level_best_score - current_relaxed_estimate)`.

## Best Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| max_iterations | 5000 | PQ pop limit |
| max_children | 30 | Max proposals per expansion |
| lookahead_children | 10 | Proposals to evaluate at next level |
| time_limit | 30 | Wall-clock timeout (seconds) |

## Per-Environment Behaviour
Identical to V1 across all 20 environments.

## Comparison to Other Variants
Same cost as V1. The lookahead doesn't change proposal rankings because the relaxed SPL
estimates are already accurate enough that `lookahead_bonus ≈ 0` for the top proposals.

## Intuition
Lookahead helps when the immediate heuristic is misleading — when a locally cheap option
leads to expensive sub-problems. Here, the `subgoal_score` heuristic already accounts for
all three cost components, and the relaxed SPL estimates closely match the true costs.
The lookahead confirms what the heuristic already knew.
