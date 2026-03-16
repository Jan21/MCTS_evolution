# V2: Multi-Heuristic A* with Edge Selection

## Algorithm Description
Extends V1 with two modifications:
1. **Depth penalty**: Priority = `cost + α * n_open_edges`, penalizing plans with more unresolved segments
2. **Edge selection strategies**: Instead of always expanding the first open edge, choose by:
   - `first`: default (same as V1)
   - `cheapest`: pick the open edge with lowest cost estimate
   - `most_constrained`: pick the edge whose parent has fewest incoming edges (proxy for fewer proposals)

## Heuristic Formula
`f(plan) = plan.cost() + depth_penalty * count(open_edges)`

## Best Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| max_iterations | 10000 | PQ pop limit |
| max_children | 30 | Max proposals per expansion |
| depth_penalty | 0.5 | Weight on open edge count |
| edge_selection | "cheapest" | How to pick which edge to expand |

Tuned 10 configurations of (depth_penalty, edge_selection, max_children). All produced avg=8.90.

## Per-Environment Behaviour
Identical to V1 across all 20 environments.

## Comparison to Other Variants
No difference from V1. The depth penalty doesn't change PQ ordering enough to explore different subgoals, and edge selection doesn't matter because environments typically have only 1-2 open edges at any time.

## Intuition
The search space is too narrow for heuristic variations to matter. With ~5-20 proposals per expansion and greedy always finding optimal plans in ~6 iterations, there's no room for different heuristics to lead to different exploration paths.
