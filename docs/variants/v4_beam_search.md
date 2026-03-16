# V4: Beam Search A*

## Algorithm Description
Level-by-level expansion keeping only the top-K plans at each depth:
1. Start with initial plan in beam
2. Expand all plans at current depth
3. Score all resulting children
4. Keep only the K best (by total cost)
5. Repeat until a complete plan is found or max_depth reached

## Heuristic Formula
`f(plan) = plan.cost()` — used for beam selection.

## Best Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| beam_width | 20 | Max plans kept per depth level |
| max_depth | 10 | Maximum expansion depth |
| max_children | 20 | Max proposals per expansion |

## Per-Environment Behaviour
Identical to V1 across all 20 environments.

## Comparison to Other Variants
No improvement. The beam width of 20 is more than sufficient since the optimal plan is always among the top-cost partial plans at each depth level.

## Intuition
Beam search helps when the branching factor is high and memory is a concern. Here, the effective branching factor is low (few viable proposals per expansion), so the beam never needs to discard the optimal partial plan. The algorithm essentially degenerates to regular A* because K > effective branch factor.
