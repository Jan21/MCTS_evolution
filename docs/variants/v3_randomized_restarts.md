# V3: Randomized Multi-Restart A*

## Algorithm Description
Runs A* search `n_restarts` times with randomized proposal ordering. Within each restart:
1. Get proposals sorted by score
2. Keep top-K proposals
3. **Shuffle** the top-K randomly before expansion
4. Run standard A* with priority queue
5. Track best plan across all restarts

Uses seeded RNG for reproducibility.

## Heuristic Formula
Same as V1: `f(plan) = plan.cost()`

Randomization only affects which proposals are expanded first within each iteration, not the PQ ordering.

## Best Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| max_iterations | 5000 | Per-restart iteration limit |
| max_children | 30 | Max proposals per expansion |
| n_restarts | 10 | Number of random restarts |
| seed | 42 | Random seed |

## Per-Environment Behaviour
Identical to V1 across all 20 environments.

## Comparison to Other Variants
No improvement over V1. The PQ mechanism means that even with shuffled insertion order, the best-cost plan is always popped first. Shuffling only delays when a particular proposal is inserted, but doesn't change which complete plan the PQ converges to.

## Intuition
Random restarts help when there are many equally-good options that lead to different outcomes. Here, the PQ's cost-based ordering dominates: regardless of insertion order, the lowest-cost partial plan is always expanded first, which consistently leads to the same optimal subgoal at each step.
