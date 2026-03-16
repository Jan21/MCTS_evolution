# V1: Greedy Best-First A*

## Algorithm Description
Priority queue (min-heap) ordered by total plan cost. At each step:
1. Pop the lowest-cost partial plan
2. Pick the first open edge
3. Try to fix it directly (if exact shortest path exists via independent edges)
4. Otherwise, call `propose_subgoal_states()` to get candidate subgoals
5. For each candidate, create a new partial plan with the subgoal applied
6. If cost < best known, push to priority queue
7. Prune any plan whose cost >= best complete plan found so far

Handles nodes reachable only via dependent edges by trying multiple parent-support positions from the dependent edge cache.

## Heuristic Formula
`f(plan) = plan.cost() = sum of all edge costs (exact for fixed, relaxed for open)`

Open edges use `compute_relaxed_shortest_path_length` (includes dependent edges), which serves as an optimistic/admissible heuristic.

## Best Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| max_iterations | 10000 | PQ pop limit |
| max_children | 50 | Max proposals per expansion |

These are generous — the algorithm typically finds solutions in ~6 iterations.

## Per-Environment Behaviour
Solves all 20 environments successfully. Average cost: **8.90**. Most environments need 1 subgoal (cost = parent→bn + bn→child + sp→helper). Environments 0, 13, 18 are hardest (cost 13-15), requiring deeper subgoal chains.

## Comparison to Other Variants
All variants V1-V12 (except V7) produce **identical results** (avg=8.90, 20/20 solved). V7 (anti-greedy) is worse at 9.30 because forcing non-optimal proposals hurts envs 10 and 11.

## Intuition
The `subgoal_score` heuristic (sum of 3 relaxed SPLs) is extremely accurate for these environments — it correctly ranks proposals such that the best-scored proposal always leads to the globally optimal plan. No search strategy can improve upon greedy because the heuristic ranking matches the true optimal ordering at every decision point.
