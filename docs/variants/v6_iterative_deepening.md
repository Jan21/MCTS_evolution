# V6: Iterative Deepening A* (IDA*)

## Algorithm Description
Sets a cost threshold and runs DFS pruning any plan exceeding it:
1. Initialize threshold = initial plan cost estimate (relaxed)
2. Run DFS, skipping any plan with cost > threshold
3. If no complete plan found, increase threshold to the minimum pruned cost
4. Repeat until solution found or max_passes exceeded

Combines DFS's memory efficiency with A*'s optimality guarantee.

## Heuristic Formula
Threshold-based pruning: skip plan if `plan.cost() > threshold`.
Threshold increases monotonically to the next tightest bound.

## Best Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| max_iterations_per_pass | 50000 | DFS iterations per threshold |
| max_children | 50 | Max proposals per expansion |
| max_passes | 20 | Max threshold increases |
| time_limit | 30 | Wall-clock timeout (seconds) |

## Per-Environment Behaviour
Identical to V1 across all 20 environments. Typically finds solution in the first pass because the initial threshold (relaxed cost) is close to the actual optimal cost.

## Comparison to Other Variants
Same results as V1. Uses less memory than V1 (DFS stack vs PQ) but more iterations due to threshold restarts.

## Intuition
IDA* is most beneficial when there's a large gap between heuristic estimate and true cost. Here, the relaxed SPL heuristic is tight (close to exact), so the first threshold already admits the optimal solution without needing multiple passes.
