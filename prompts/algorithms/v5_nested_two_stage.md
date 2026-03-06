# V5: Nested (Two-Stage) MCTS: Skeleton First, Realization Second

## Overview

Separates the search into two levels. The **outer MCTS** searches over **plan skeletons** — high-level subgoal/support assignments that define the structure of the plan DAG without committing to exact paths. The **inner search** takes a skeleton and attempts to **realize** it: compute exact shortest paths for all segments, detect conflicts, and return the realized cost plus any coupling information. Conflicts discovered by the inner search feed back as constraints on the outer search, guiding it toward skeletons that are actually realizable. This clean separation avoids mixing strategic decisions (which subgoals to use) with tactical ones (can these segments coexist).

## State Representation

### Outer MCTS nodes (skeleton level)

A skeleton is a `PartialPlan` where:
- All subgoal nodes are assigned (bottleneck positions, support positions, robot assignments)
- All edges exist in the DAG
- But edge statuses may still be "open" — the exact costs are not yet computed
- The skeleton captures the **topology and robot assignments** of the plan

Each outer node holds:
- **Skeleton plan**: a `PartialPlan` with all subgoals chosen but edges potentially open
- **Skeleton signature**: hashable representation for transposition (frozenset of subgoal assignments)
- **Inner search results cache**: list of `(realized_cost, conflict_set)` from previous inner evaluations
- **Conflict constraints**: set of conflict sets reported by inner search — used to prune or penalize future skeleton extensions

### Inner search state (realization level)

Given a skeleton, the inner search works on:
- The fixed skeleton DAG (topology and robot assignments are given)
- For each physical edge, the task: compute `grid_env.compute_exact_shortest_path_length(start, end, support_pos)` and verify it's not None
- A **commitment tracker**: as edges are fixed, record which robots are at which positions

The inner search is simpler than the outer — it's essentially a constraint satisfaction / verification pass, optionally with limited local search to resolve conflicts.

## Action Space

### Outer MCTS actions

At each outer node, the action is to **extend the skeleton** by choosing a subgoal for one open segment:

1. Select an open segment from the skeleton (e.g., the one with highest relaxed cost, or by UCB over segments)
2. Call `grid_env.propose_subgoal_states()` to get candidate subgoals
3. Filter candidates against known conflict constraints:
   - If a candidate would create a (robot, position) assignment that appears in a recorded conflict set, penalize or skip it
4. Apply the chosen subgoal to the skeleton (add subgoal/bottleneck/support nodes, set up edges)

Progressive widening on subgoal choices, same schedule as V1.

**Split-vs-joint decisions** (optional extension): at each outer node, the search can also choose to **not decompose** a segment — leave it as a single open edge to be handled jointly with its neighbors. This is an explicit OR choice that V5's structure naturally accommodates.

### Inner search actions

Given a complete skeleton (all subgoals assigned):
1. For each physical edge in topological order (leaves to goal):
   a. Compute `grid_env.compute_exact_shortest_path_length(start_pos, end_pos, support_pos)`
   b. If reachable: set edge status to "fixed", record cost
   c. If unreachable: record this edge + its dependencies as a **conflict set** and abort

2. **Local repair** (optional): if an edge is unreachable, try small modifications:
   - Swap the support robot for an alternative helper (if the skeleton allows flexibility)
   - Adjust support position within the same bottleneck-support pair options
   - If repair succeeds, continue; otherwise report the conflict

## Evaluation / Rollout

### Outer rollout (skeleton completion)

When the outer MCTS needs to evaluate a partial skeleton:
1. **Greedy skeleton completion**: for each remaining open segment, pick the subgoal with the best `grid_env.subgoal_score()`. This produces a complete skeleton quickly.
2. **Inner evaluation**: pass the complete skeleton to the inner search
3. Return the inner search result as the rollout value

### Inner evaluation

Given a complete skeleton:
1. Attempt to fix all edges (compute exact shortest paths)
2. If all edges are fixable:
   - Run `validate(plan, grid_env, state)` as a sanity check
   - Return `evaluate_plan(plan, grid_env)` as the cost
3. If some edges are unreachable:
   - Extract the **conflict set**: the minimal set of (subgoal assignment, edge) pairs responsible for the failure
   - Return `(infinity, conflict_set)` — the cost is infinite, but the conflict set is informative

### Inner search as MCTS (advanced variant)

For complex instances, the inner "search" can itself be a small MCTS:
- State: partially-realized skeleton (some edges fixed, some open)
- Action: choose order of edge realization + minor adjustments (support robot swaps)
- Value: number of successfully fixed edges + total cost of fixed edges
- This is only needed when realization order matters (fixing edge A first constrains edge B)

For simpler instances, a single deterministic pass (topological order) suffices.

## Backup & Selection

### Outer MCTS backup
- **Selection**: UCB1 over skeleton extensions. Conflict constraints modify the UCB score — subgoals that would trigger a known conflict get a penalty proportional to the number of times that conflict has been observed.
- **Backup**: the value backed up is the inner search cost (or infinity for unrealizable skeletons). Running average as usual.
- **Conflict feedback**: when the inner search reports a conflict set, the outer MCTS records it at the node where the conflicting subgoal was chosen. Future visits to that node see the conflict and can avoid or penalize the same choice.

### Transposition tables (both levels)
- **Outer**: skeleton signatures enable reuse across different derivation orders
- **Inner**: segment-level caching — if "robot A from P1 to P2 with support at P3" was computed before, reuse the exact SPL result. This is straightforward since `grid_env.compute_exact_shortest_path_length` is deterministic.

### Pruning
- **Outer**: if a partial skeleton's relaxed cost already exceeds the best realized cost, prune
- **Inner**: if partial realization cost exceeds best known, abort early

## Coupling / Conflict Handling

V5's two-level structure provides a natural separation of concerns:

### Outer level: strategic conflict avoidance
- Conflict sets from the inner search are recorded as constraints on skeleton construction
- A conflict set `{(subgoal_A, edge_X), (subgoal_B, edge_Y)}` means: these two subgoal assignments are jointly infeasible
- Future outer MCTS rollouts avoid skeletons containing both subgoal_A and subgoal_B (or at least penalize them heavily)
- This is similar to V3's no-goods but operates at the skeleton level rather than the decision-step level

### Inner level: tactical conflict detection
- The inner search discovers conflicts by actually trying to compute exact shortest paths
- Conflict detection is precise — based on `compute_exact_shortest_path_length` returning None
- Conflicts are reported upward with minimal conflict sets for maximum learning

### Conflict generalization
- A conflict set `{sg_A, sg_B}` can be **generalized**: if sg_A and sg_B conflict because they both need helper H, then any skeleton containing two subgoals that both need H at different positions will likely conflict
- The outer MCTS can maintain a **conflict pattern library**: `(helper_robot, position_set) -> conflict` that applies across specific subgoal instances

## Key Differences from Other Versions

- **vs V1**: two-level search separates strategic (subgoal choice) from tactical (path realization) decisions. V1 mixes them in a single search.
- **vs V2**: V2's conflict graph is within a single plan; V5's conflicts flow between two search levels with clear directionality (inner → outer).
- **vs V3**: V3 interleaves constraint reasoning with refinement in one search. V5 cleanly separates them — the outer search is constraint-free except for learned conflict sets, and the inner search is a focused realization pass.
- **vs V4**: V4 shares segment statistics across plans via a transposition table. V5 shares information via conflict sets flowing from inner to outer. V4's reuse is statistical (average costs); V5's is structural (which combinations are infeasible).

## Complexity & Tradeoffs

**Strengths:**
- Clean separation of concerns — each level can be optimized independently
- Inner search provides precise conflict information (based on exact computations, not heuristics)
- Conflict sets enable strong pruning of the outer skeleton space
- Inner caching of exact SPL results is simple and highly effective (deterministic, context-free)
- Natural place to add a split-vs-joint decision at the outer level

**Weaknesses:**
- Most computationally expensive per rollout — each outer rollout requires a full inner evaluation
- Skeleton completion + inner evaluation latency may limit the number of outer MCTS iterations within a time budget
- The two-level structure adds implementation complexity (two MCTS loops, inter-level communication)
- Conflict sets may be too specific (a conflict between two particular subgoals doesn't generalize to other subgoals without explicit generalization logic)
- Inner local repair adds complexity and may not always find fixes even when they exist

**Best suited for:** instances with heavy coupling where strategic subgoal choice is critical — i.e., many subgoals are available but only certain combinations are jointly realizable. The two-level structure excels when the space of skeletons is large but the inner realization pass is fast (most edges are easily fixable). Also a good fit when the search budget is generous enough to afford the per-rollout inner evaluation cost.
