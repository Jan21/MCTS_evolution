# V1: Contextual Plan-Refinement Graph MCTS

## Overview

The simplest of the five variants. Each MCTS node represents a full partial-plan DAG. Actions refine the plan by selecting an open segment and proposing a subgoal for it. A transposition table deduplicates identical partial plans reached via different action sequences. Coupling between segments is handled implicitly through the plan cost — no explicit conflict detection or constraint store.

## State Representation

Each MCTS node holds:

- **Partial plan**: a `PartialPlan` instance (the DAG with goal/subgoal/bottleneck/support/leaf nodes and open/fixed edges)
- **Transposition key**: a hashable signature of the partial plan (e.g., frozenset of node IDs with their attributes + edge statuses) used to look up whether this plan state has been visited before

The root node contains the initial plan: a single `"goal"` node with one open edge to a `"leaf_target"` node (the target robot at its starting position).

## Action Space

An action is a **(segment, subgoal)** pair:

1. **Segment selection**: pick one open edge `(parent, child)` from `plan.open_edges()`
2. **Subgoal proposal**: call `grid_env.propose_subgoal_states(state_for_segment, support_robot)` where `state_for_segment` is constructed from the segment's endpoints and available helpers. This returns a list of `(Subgoal, score)` tuples.

Applying an action:
- Removes the open edge `(parent, child)`
- Adds a new subgoal node (e.g., `"sg_N"`) with:
  - A bottleneck child (`"bn_N"`, pos=subgoal.bottleneck.position, robot=subgoal.bottleneck)
  - A support child (`"sp_N"`, pos=subgoal.support.position, robot=subgoal.support)
- The edge from `parent` to `"bn_N"` becomes **fixed** with cost from `grid_env.compute_exact_shortest_path_length(bn_pos, parent_pos, support_pos)`
- Structural edges `"sg_N" -> "bn_N"` and `"sg_N" -> "sp_N"` carry no cost
- New open edges are created from `"bn_N"` and `"sp_N"` to appropriate leaf/subgoal nodes

**Progressive widening**: since `propose_subgoal_states` can return many candidates, limit the number of children expanded per segment using a widening schedule (e.g., `k = ceil(C * N^alpha)` where N is visit count, C and alpha are parameters).

## Evaluation / Rollout

**Heuristic evaluation** of a partial plan (used as rollout value):

- `plan.cost()` — sum of all edge costs (fixed edges use exact SPL, open edges use relaxed SPL from `grid_env.compute_relaxed_shortest_path_length`)
- Lower cost = better plan

**Rollout policy** (greedy completion):
1. While `plan.open_edges()` is non-empty:
   - Pick the open edge with highest relaxed cost (most uncertain)
   - Call `propose_subgoal_states` and pick the subgoal with the best (lowest) `grid_env.subgoal_score()`
   - Apply it to the plan
2. If the completed plan passes `validate(plan, grid_env, state)`, return `evaluate_plan(plan, grid_env)` as the rollout value
3. If validation fails or `evaluate_plan` returns None, return a large penalty value

## Backup & Selection

- **Selection**: UCB1 — at each node, pick the child action maximizing `Q(a) + C_explore * sqrt(ln(N_parent) / N_child)`. Since we minimize cost, `Q(a)` should be the negative of average cost (or equivalently, use LCB for minimization).
- **Backup**: after a rollout, propagate the cost value up the tree. Each node stores (visit_count, total_cost). Backup = running average.
- **Transposition table**: before creating a new node, check if the resulting plan signature already exists. If so, reuse the existing node's statistics. This turns the tree into a DAG.
- **Pruning**: maintain `best_complete_cost` (best cost of any fully validated plan found so far). During selection, skip any node whose `plan.cost()` already exceeds `best_complete_cost`.

## Coupling / Conflict Handling

**Implicit only.** Coupling between segments (e.g., two segments requiring the same helper robot) is not explicitly detected. Instead:

- If a subgoal assigns a helper that is already used elsewhere in the plan, the resulting plan will either fail validation (robot continuity check) or have high cost due to unreachable edges.
- The high cost / validation failure feeds back through MCTS statistics, causing UCB to avoid those subgoal choices in future iterations.

This is the main limitation of V1 — it discovers coupling only through trial and error, without learning structured conflict information.

## Key Differences from Other Versions

- **vs V2**: no explicit conflict graph or merging — coupling is handled purely by cost feedback
- **vs V3**: no constraint store, no no-goods, no backjumping — just standard MCTS backup
- **vs V4**: no segment-level transposition table — transpositions are at the full plan level
- **vs V5**: single-level search, no separation of skeleton and realization

## Complexity & Tradeoffs

**Strengths:**
- Simple to implement — straightforward MCTS with plan states
- Transposition table provides some reuse across derivation orders
- Progressive widening controls branching factor

**Weaknesses:**
- No structured conflict learning — wastes rollouts discovering the same coupling issues repeatedly
- Full-plan transpositions are expensive to hash and may have low hit rates
- Greedy rollout may systematically miss good plans that require non-greedy subgoal choices

**Best suited for:** instances where segments are mostly independent (few helper conflicts), or as a baseline to measure the value of the more sophisticated mechanisms in V2-V5.
