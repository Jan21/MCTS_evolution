# V8: Diverse Beam Search

## Algorithm Description
Extends V4 (beam search) with two innovations:
1. **Structural diversity**: Beam members are grouped by their set of bottleneck positions.
   Selection picks the best from each group round-robin, preventing the beam from collapsing
   to variations of the same plan.
2. **Most-constrained edge selection**: Expands the open edge with the fewest viable proposals,
   resolving the hardest sub-problem first (fail-fast strategy).

## Heuristic Formula
`f(plan) = plan.cost()` for beam ranking. Diversity enforced by grouping plans by their
`frozenset(bottleneck_positions)`.

## Best Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| beam_width | 30 | Total beam size |
| max_depth | 10 | Maximum depth |
| max_children | 50 | Max proposals per expansion |
| diversity_groups | 5 | Target number of distinct groups |
| time_limit | 30 | Wall-clock timeout (seconds) |

## Per-Environment Behaviour
Identical to V1 across all 20 environments. The diversity mechanism rarely activates because
most environments have only 1-3 viable bottleneck position sets.

## Comparison to Other Variants
Same cost as V1. The most-constrained edge selection doesn't help because edge expansion
order doesn't affect which proposals are available at each step.

## Intuition
Diversity enforcement helps when the search landscape has many distinct high-quality regions.
Here, the landscape has essentially one basin of attraction — all "diverse" plans converge to
the same cost because the same bottleneck positions are optimal regardless of exploration order.
