# V4: Segment-Node MCTS with Context-Signature Transpositions

## Overview

Introduces a two-level node structure: **plan nodes** (full partial plans, as in V1) point to shared **segment nodes** via a transposition table. A segment node represents the subproblem "move robot R to position P" keyed by a context signature (which other robots are frozen/reserved nearby). Segment nodes accumulate MCTS statistics across many different plans, enabling systematic reuse of "how to get robot A to position P" knowledge. Coupling is handled by penalizing segment-option pairs whose footprints clash in a given plan context.

## State Representation

### Plan nodes
Each plan node holds:
- **Partial plan**: a `PartialPlan` instance
- **Segment assignments**: for each open segment, a reference to its segment node in the transposition table
- **Footprint summary**: aggregated set of `(robot, position)` commitments across all segments in this plan

### Segment nodes (transposition table entries)
Keyed by a **segment signature**:
- `(robot_color: str, start_pos: tuple, goal_pos: tuple, context_sig: frozenset)`
- `context_sig` = frozenset of `(robot_color, position)` for all robots that are frozen/reserved in the vicinity (helpers committed by other segments that constrain this segment's options)

Each segment node holds:
- **Subgoal options**: list of subgoals proposed via `grid_env.propose_subgoal_states()`, each with:
  - The `Subgoal` object
  - A **footprint**: set of `(robot, position)` pairs required by this subgoal
  - MCTS statistics: `(visit_count, total_cost)`
- **Visit count** for the segment node itself (for UCB at segment level)

### Transposition table
A dict: `segment_signature -> segment_node`. When a plan needs to refine a segment, it looks up the signature in the table. If found, it reuses the existing node (and its accumulated statistics). If not, it creates a new one.

## Action Space

Two-stage, same structure as V1, but the second stage operates on the shared segment node:

1. **Plan level**: choose which open segment to refine (from `plan.open_edges()`)
2. **Segment level**: at the segment node, choose which subgoal option to apply (UCB + progressive widening over the segment node's subgoal options)

The segment node's subgoal options are populated lazily:
- First visit: call `grid_env.propose_subgoal_states()` and store all candidates
- Progressive widening: on each visit, expand at most `k = ceil(C * N^alpha)` new candidates (sorted by `grid_env.subgoal_score()`)

### Context signature construction
When a plan node looks up a segment in the transposition table:
1. Collect all `(robot, position)` commitments from other segments in the current plan
2. Filter to robots within a relevant neighborhood of this segment's start/goal positions (using grid adjacency or relaxed reachability)
3. Freeze this as `context_sig`

This ensures that the segment node is reused when the surrounding robot configuration is equivalent, but creates separate nodes when context differs materially.

## Evaluation / Rollout

### Rollout at plan level
1. For each open segment, select a subgoal option from its segment node (using the segment node's UCB statistics as a prior, then greedy tie-breaking)
2. Combine all selected options into a complete plan
3. Check for **footprint clashes**: if two selected options require the same robot at different positions, add a penalty
4. Validate with `validate(plan, grid_env, state)` and evaluate with `evaluate_plan(plan, grid_env)`

### Penalty for footprint clashes
For each pair of segment options `(opt_i, opt_j)` whose footprints overlap on robot R at positions P1 != P2:
- Penalty = `grid_env.compute_relaxed_shortest_path_length(P1, P2)` (cost to relocate R)
- If relocation is impossible (returns None), penalty = infinity

### Using segment statistics as priors
When a segment node is visited for the first time in a new plan context:
- Import the visit counts and average costs from the segment node as **pseudo-counts** (weighted by a decay factor to account for context differences)
- This gives immediate guidance even in new plans, based on experience from other plans

## Backup & Selection

### Two-level backup
After a rollout with total cost C:
1. **Segment level**: for each segment node whose option was selected, update `(visit_count, total_cost)` for that option
2. **Plan level**: update the plan node's statistics with the full plan cost (including footprint clash penalties)

### Selection
- **Plan level**: UCB1 to choose which segment to refine next
- **Segment level**: UCB1 over the segment node's subgoal options, using the segment node's accumulated statistics

### Pairwise penalty table
Maintain a table of `(segment_option_i, segment_option_j) -> average_penalty` based on observed footprint clashes. During selection, add the expected pairwise penalty to the UCB estimate for each option, conditioned on the options already selected for other segments in the current plan.

## Coupling / Conflict Handling

V4 handles coupling through **footprint-based penalties** rather than explicit conflict graphs or constraint stores:

1. **Footprint extraction**: each subgoal option records which robots it needs and where
2. **Clash detection**: when combining options across segments in a plan, check footprints for overlaps
3. **Penalty application**: clashing options get penalized proportionally to relocation cost
4. **Pairwise learning**: the penalty table accumulates statistics about which option pairs tend to clash, enabling better selection in future rollouts

This is lighter-weight than V3's constraint reasoning but provides cross-plan learning that V2 lacks.

### Protection commitments (optional)
When a segment option requires a robot to remain stationary (the robot serves as a blocker), record a `protect(robot, position)` in its footprint. Other options that would move that robot get an additional penalty.

## Key Differences from Other Versions

- **vs V1**: segment-level transposition table provides systematic reuse; pairwise penalty table provides structured coupling information
- **vs V2**: reuse is across plans (via shared segment nodes), not just within a single plan's conflict graph
- **vs V3**: no backjumping or no-goods — coupling is handled via soft penalties rather than hard constraint pruning. Simpler but less precise.
- **vs V5**: single-level search with shared segment nodes, rather than a nested skeleton/realization separation

## Complexity & Tradeoffs

**Strengths:**
- Strong reuse: "how to move robot A to position P" is learned once and shared across all plans that need it
- Context signatures provide controlled generalization — similar contexts share statistics, different contexts get separate nodes
- Pairwise penalty table enables cross-plan learning about which segment combinations are problematic
- Moderate implementation complexity (simpler than V3 or V5)

**Weaknesses:**
- Context signature design is critical — too coarse and the transposition conflates different situations; too fine and reuse drops
- Footprint-based penalties are approximate — they don't capture all forms of coupling (e.g., ordering constraints)
- The transposition table can grow large for instances with many robots and positions
- Pseudo-count initialization requires tuning the decay factor

**Best suited for:** instances where the same robot-to-position subproblems recur across different plans (many helpers, multiple viable subgoal decompositions). The reuse payoff is highest when the search explores many alternative plans that share common segment subproblems.
