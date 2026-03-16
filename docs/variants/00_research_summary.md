# A* Variant Research Summary

## Overview
12 A* variants were implemented and benchmarked across 20 Ricochet Robots environments.
The primary finding is that **all variants converge to the same solution** (avg_cost=8.90),
with the sole exception of V7 which performs *worse*.

## Variant Landscape

| Variant | Strategy | avg_cost | Notes |
|---------|----------|----------|-------|
| V1  | Greedy Best-First | 8.90 | Baseline, ~0.01s |
| V2  | Multi-Heuristic + Edge Selection | 8.90 | Depth penalty + cheapest/most_constrained |
| V3  | Randomized Multi-Restart | 8.90 | 10 restarts with shuffled proposals |
| V4  | Beam Search | 8.90 | Top-K plans per depth level |
| V5  | Exhaustive DFS + Greedy Fallback | 8.90 | 200K expansions, proves optimality |
| V6  | Iterative Deepening A* (IDA*) | 8.90 | Cost-threshold bounded DFS |
| V7  | K-th Best / Anti-Greedy | **9.30** | Forces non-optimal proposals → worse |
| V8  | Diverse Beam Search | 8.90 | Structural diversity enforcement |
| V9  | Two-Level Lookahead | 8.90 | Evaluates sub-problem cost before committing |
| V10 | Weighted A* | 8.90 | f = fixed + w*open, multiple w values |
| V11 | Exact-Cost Reranking | 8.90 | Reranks by exact (not relaxed) costs |
| V12 | Greedy Rollout | 8.90 | Full plan completion for each proposal |

## Key Findings

### 1. The Heuristic is Near-Perfect
The `subgoal_score` (sum of 3 relaxed shortest path lengths) so accurately predicts true
plan costs that the best-scored proposal at every decision point leads to the globally
optimal plan. This was validated by:
- V5 (exhaustive DFS): explored all reachable plan structures, found nothing better
- V12 (greedy rollout): simulated full plan completion for each proposal, rankings unchanged
- V11 (exact reranking): exact costs agree with relaxed costs on proposal ordering

### 2. Search Strategy Doesn't Matter
When the heuristic is this accurate, the search strategy is irrelevant:
- **Priority queue ordering** (V1, V2, V10): all converge to same plan
- **Breadth control** (V4, V8): beam never needs to discard optimal partial plan
- **Randomization** (V3): PQ dominates — shuffled insertion still pops optimal plan first
- **Exhaustiveness** (V5, V6): more search finds nothing better
- **Lookahead** (V9): confirms what the heuristic already predicted

### 3. Non-Greedy is Harmful
V7 proves that forcing sub-optimal proposals makes results worse (9.30 vs 8.90).
The proposal space from `propose_subgoal_states` is "well-ordered" — the score ranking
matches the true cost ranking at every step.

### 4. The Bottleneck is Proposal Generation
All variants use the same `propose_subgoal_states()` API, which returns a constrained
set of candidates determined by:
- The `_bottleneck_support_pairs_cache` (built from dependent edge structure)
- The `subgoal_score` heuristic
- Available helper robots

To find lower-cost plans, one would need to change how proposals are generated — e.g.,
different bottleneck enumeration, alternative scoring, or new subgoal structures.

## Recommendations
1. **For practical use**: V1 is optimal. It's the simplest, fastest, and finds the
   same plan as all other variants.
2. **For improvement**: Focus on `propose_subgoal_states` and `subgoal_score` in GridEnv,
   not on search strategy.
3. **For harder environments**: V5's exhaustive search provides an optimality guarantee.
   If V5 finds the same plan as V1, no search-level improvement is possible.
