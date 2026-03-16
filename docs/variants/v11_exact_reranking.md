# V11: Exact-Cost Reranking A*

## Algorithm Description
Reranks proposals using exact (independent-path-only) costs instead of the default
relaxed (includes dependent edges) scoring:
1. Get proposals from `_get_proposals` (uses `subgoal_score` with relaxed SPL)
2. Rerank each proposal by computing exact costs for all 3 edges + open-edge penalty
3. `exact_score = parent→bn + bn→child + sp→helper + penalty * n_open_edges_created`
4. Sort by exact_score instead of original score
5. Otherwise standard A* with priority queue

The open-edge penalty discourages proposals that create unresolvable segments.

## Heuristic Formula
Proposal ranking: `exact_score = sum(exact_or_relaxed_costs) + open_edge_penalty * n_open`

Tuned `open_edge_penalty` ∈ {-3, -1, 0, 0.5, 1, 2, 3, 5, 10}. All values produce identical results.

## Best Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| max_iterations | 10000 | PQ pop limit |
| max_children | 50 | Max proposals per expansion |
| open_edge_penalty | 3.0 | Cost added per new open edge |
| time_limit | 30 | Wall-clock timeout (seconds) |

## Per-Environment Behaviour
Identical to V1 across all 20 environments.

## Comparison to Other Variants
Same cost as V1 regardless of penalty value. The exact costs preserve the same ranking as
relaxed costs for the top proposals in all 20 environments.

## Intuition
Exact reranking would help if relaxed estimates are misleading — if a proposal looks good
with dependent edges included but poor without them. For these environments, the relaxed
and exact rankings agree on the top proposal, so reranking has no effect.
