# MCTS1 Implementation & Debugging Log

## Overview

Implemented MCTS V1 "Contextual Plan-Refinement" algorithm for Ricochet Robots puzzle solver. The algorithm searches for low-cost partial plans (DAGs) that decompose the problem into subgoals.

## Initial Implementation

### Files Created/Modified
- **`MCTS.py`** — Main implementation: `MCTS` ABC, `MCTSNode`, `MCTS1` class with full MCTS loop
- **`conf/mcts1.yaml`** — Hydra/OmegaConf config with hyperparameters
- **`benchmark.py`** — Updated to use `MCTS1`
- **`GridEnv.py`** — Fixed `subgoal_score` None handling
- **`evaluate_plan.py`** — Fixed `_edge_movement` support_pos logic
- **`validate_plan.py`** — Fixed `_physical_movement` support_pos logic

### Architecture
- `MCTSNode` dataclass: holds `plan`, `parent`, `action`, `children`, `visit_count`, `total_cost`
- Selection: LCB (Lower Confidence Bound) for cost minimization
- Expansion: progressive widening (`k = ceil(pw_C * N^pw_alpha)`)
- Rollout: random greedy completion
- Backpropagation: walk up parent chain, accumulate costs

### Key Helpers
- `_build_initial_plan(state, grid_env)` — Creates root plan with goal + leaf_target
- `_build_segment_states(plan, parent_id, state, grid_env)` — Builds `(State, support_robot)` pairs for `propose_subgoal_states`
- `_get_candidates(plan, parent_id, state, grid_env)` — Collects `(subgoal, score, parent_support_pos)` tuples
- `_apply_subgoal(plan, parent_id, child_id, subgoal, grid_env, cnt, parent_support_pos)` — Core DAG mutation
- `_try_close_edge(plan, parent_id, child_id, grid_env)` — Close an open edge directly without subgoal insertion

---

## Bug 1: `GridEnv.subgoal_score` TypeError

**Symptom**: `TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'`

**Cause**: `compute_relaxed_shortest_path_length` returns `None` for unreachable positions, but `subgoal_score` tried to add `None + int`.

**Fix**: Added `INF = 10_000` fallback for None values in `GridEnv.subgoal_score`.

---

## Bug 2: Goal with only dependent edges

**Symptom**: `AssertionError` in `propose_subgoal_states` — `(goal, None)` not in `_bottleneck_support_pairs_cache`.

**Cause**: Some goal positions (e.g., `(6,15)` in env_0) have NO independent in-edges, only dependent ones. The cache only has entries like `((6,15), (5,15))` or `((6,15), (7,15))`.

**Fix**: In `_build_segment_states`, when `(goal, None)` is not in cache, enumerate all dependent edge support positions from `grid_env.G.in_edges(goal)`.

---

## Bug 3: `parent_support_pos` not threaded through

**Symptom**: `evaluate_plan` and `validate_plan` computed wrong costs for `parent→subgoal` edges.

**Cause**: The support position for the bn→parent dependent edge comes from the `support_robot` passed to `propose_subgoal_states`, NOT from the subgoal's own support child. This distinction wasn't tracked.

**Fix**:
- `_get_candidates` returns 3-tuples `(subgoal, score, parent_sp)`
- `_apply_subgoal` stores `parent_support_pos` as an attribute on the subgoal node
- `evaluate_plan._edge_movement` and `validate_plan._physical_movement` read `parent_support_pos` from the subgoal node

---

## Bug 4: Circular bottleneck positions

**Symptom**: Infinite decomposition chains — `propose_subgoal_states` returns the goal position itself as a bottleneck.

**Fix**: Filter candidates whose bottleneck position appears in `_get_ancestor_positions(plan, parent_id)`.

---

## Bug 5 (Critical): `_try_close_edge` missing dependent paths for bottleneck parents

**Symptom**: Plans never completed — `bn → leaf_target` edges couldn't close even though the robot COULD reach the bottleneck via a dependent edge.

**Root Cause**: `_try_close_edge` always passed `support_pos=None` to `compute_exact_shortest_path_length`. For bottleneck parents, the physical path from the child (leaf) to the parent (bottleneck) may require a dependent edge through the **sibling support** position. Example:

```
env_0: Target at (12,9), bn=(10,15), sibling sp=(11,15)
exact((12,9) -> (10,15), sp=None) = None     # no independent path!
exact((12,9) -> (10,15), sp=(11,15)) = 10     # works via dependent edge
```

**Fix**: Updated `_try_close_edge` to also check `compute_exact_shortest_path_length(child_pos, parent_pos, sibling_support_pos)` when the parent is a bottleneck.

**Cascading fix**: Also updated `validate_plan._physical_movement` and `evaluate_plan._edge_movement` to return the sibling support position for leaf/support children of bottleneck parents (was returning `None`).

---

## Bug 6 (Critical): `_build_segment_states` short-circuiting on independent edges

**Symptom**: env_1 never found a valid plan. All candidates from `propose_subgoal_states` either had unreachable bottlenecks or were rejected by `_apply_subgoal`.

**Root Cause**: `_build_segment_states` for goal parents checked `if (goal, None) in cache: return [(state, None)]` — short-circuiting. If the goal had ANY independent in-edge, it would ONLY try independent paths. But for env_1:

```
env_1: Goal=(11,0), target at (13,5)
Independent in-edges to (11,0): (11,1), (11,2), (11,3), (11,4) — all weight=1
But target can't reach ANY of these independently!

Dependent in-edges: (13,0)->(11,0) dep=(10,0), etc.
Target CAN reach (13,0) independently at cost 1.
```

The `propose_subgoal_states(state, None)` returned candidates with bottleneck positions like `(10,0)` and `(13,0)`, but `_apply_subgoal` rejected them because `exact(bn, goal, None)` was `None` (no independent path from bn to goal).

With `support_robot` set (e.g., sp=(10,0)): `exact(bn, goal, sp=(10,0))` returns valid costs, and the plan completes.

**Fix**: Changed `_build_segment_states` to ALWAYS enumerate dependent support positions, even when independent in-edges exist. The function now builds a list starting with `(state, None)` (if cache has it) AND adds all dependent support position variants.

Same fix applied for `support` parent type.

---

## Final Result

```
=== Benchmark: MCTS1 ===
  Environments: 10  (solved: 10, failed: 0)
  Total cost:   89
  Avg cost:     8.90
  Min cost:     4
  Max cost:     16

  Per environment:
    env_0: cost=12
    env_1: cost=4
    env_2: cost=10
    env_3: cost=8
    env_4: cost=11
    env_5: cost=8
    env_6: cost=5
    env_7: cost=16
    env_8: cost=6
    env_9: cost=9
```

## Key Architectural Insights

### Physical Movement & Support Positions
- Physical movement in the plan DAG goes child→parent (leaf toward goal)
- When a parent is a `bottleneck`, the child→parent edge may need a dependent path through the **sibling support** position
- This applies to `_try_close_edge`, `validate_plan._physical_movement`, and `evaluate_plan._edge_movement`

### `_bottleneck_support_pairs_cache` Semantics
- `cache[(goal, sp)]` = set of `(bn_pos, inner_sp)` pairs
- These are positions where dependent edges cross INTO the final component
- `bn_pos` is already INSIDE the final component; `inner_sp` enables the dependent crossing edge
- `precomputed_final_components` uses `nx.ancestors` on the FULL graph (including dependent edges)
- `reachability_matrix` only stores independent shortest paths

### Always Enumerate Dependent Edges
- Even when a goal/support has independent in-edges, dependent paths may be the ONLY viable route for the target robot
- The independent-only candidates may have unreachable bottleneck positions
- Must enumerate both independent AND dependent support positions

### `parent_support_pos` vs Subgoal's Own Support
- `parent_support_pos`: the support needed for the bn→parent dependent edge (comes from the context of how this subgoal was generated)
- Subgoal's support: the support needed for deeper decomposition (child→bottleneck edge)
- These are DIFFERENT positions serving different purposes
- `parent_support_pos` is stored on the subgoal node for `evaluate_plan` and `validate_plan`

## Config (`conf/mcts1.yaml`)

```yaml
max_iterations: 5000
ucb_exploration: 1.41
pw_C: 2.0
pw_alpha: 0.5
max_rollout_depth: 30
penalty_cost: 1000.0
```
