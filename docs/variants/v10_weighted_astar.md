# V10: Weighted A* with Inflation Factor

## Algorithm Description
Uses `f(n) = fixed_cost + w * open_cost` where `w` varies across passes:
- w > 1: inflates open-edge estimates → more greedy, prefers plans with more fixed edges
- w = 1: standard A* (same as V1)
- w < 1: deflates open-edge estimates → explores plans with high open-edge cost

Runs multiple passes with different w values, keeping the best plan overall.

## Heuristic Formula
`f(plan) = sum(fixed_edge_costs) + w * sum(open_edge_costs)`

Priority queue ordering changes with w, but actual cost comparison uses true `plan.cost()`.

## Best Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| max_iterations | 5000 | Per-weight PQ iteration limit |
| max_children | 50 | Max proposals per expansion |
| weights | [0.5, 0.8, 1.0, 1.5, 2.0, 3.0] | Inflation factors to try |
| time_limit | 30 | Wall-clock timeout (seconds) |

## Per-Environment Behaviour
Identical to V1 across all 20 environments. All weight values lead to the same final plan.

## Comparison to Other Variants
Same cost as V1. Different weights change PQ ordering but the same proposals are still
optimal at each expansion.

## Intuition
Weighted A* helps when the heuristic is informative but slightly inaccurate, and different
weightings can steer search toward regions the base heuristic undervalues. Here, the
heuristic is essentially exact for the top proposals, so weighting doesn't change which
proposal "wins" at any decision point.
