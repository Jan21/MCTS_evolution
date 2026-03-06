# V2: Lazy-Coupling MCTS with Conflict Graph + Progressive Merging

## Overview

Extends V1 by adding an explicit **conflict graph** over segments. Segments are assumed independent by default (lazy decomposition). During rollouts, coupling between segments is detected — e.g., two segments needing the same helper, or one segment moving a robot another segment needs stationary. Detected couplings are recorded as conflict edges. Connected components in the conflict graph define groups of segments that must be considered jointly. The search can either refine a single segment or **merge** a conflicted component and refine it as a unit.

## State Representation

Each MCTS node holds:

- **Partial plan**: a `PartialPlan` instance
- **Conflict graph**: an undirected graph where nodes are open segment IDs (edges from `plan.open_edges()`) and edges represent detected couplings. Each conflict edge is annotated with the coupling type:
  - `helper_reuse`: two segments require the same helper robot at different positions
  - `mover_blocker`: one segment moves a robot that another segment needs as a stationary blocker
- **Component partition**: derived from connected components of the conflict graph. Each component is a set of segment IDs that must be refined together.
- **Segment footprints**: for each segment, a dict recording which robots are committed and at which positions (populated as subgoals are chosen)

## Action Space

At each node, two types of actions are available:

### A. Single-segment refinement (default)
Same as V1: pick an open segment, call `grid_env.propose_subgoal_states()`, apply a subgoal. Only available for segments whose conflict-graph component has size 1 (no known couplings).

### B. Joint refinement of a merged component
When a conflict-graph component has size > 1, the search can choose to **refine the component jointly**:
- Select a component `C = {seg_1, seg_2, ...}`
- Pick one segment within C to refine next
- When proposing subgoals, filter out candidates that would worsen conflicts within C (e.g., a subgoal that reuses a helper already committed by another segment in C)
- Optionally add a **relocation penalty** to the plan cost: if helper H is needed at position P1 by seg_1 and P2 by seg_2, add `grid_env.compute_relaxed_shortest_path_length(P1, P2)` as a connector cost

The choice between (A) and (B) is itself an OR decision at the MCTS node, explored by UCB.

### Progressive widening
Applied to both the segment choice and subgoal choice, same as V1.

## Evaluation / Rollout

Same greedy rollout as V1, but with two additions:

1. **Coupling detection during rollout**: after each subgoal application, scan the footprints of all segments for conflicts:
   - For each helper robot used in a subgoal, check if any other segment's footprint already uses that helper at a different position → add `helper_reuse` conflict edge
   - For each robot moved by a segment, check if any other segment's footprint requires that robot stationary → add `mover_blocker` conflict edge

2. **Penalty integration**: when computing rollout value:
   - Start with `evaluate_plan(plan, grid_env)` if the plan is complete and valid
   - Add relocation penalties for each conflict edge (relaxed SPL between the two required positions)
   - If a conflict is truly infeasible (no relocation possible), return infinity

## Backup & Selection

- **Selection**: UCB1 as in V1, but with an additional bias term for nodes with known conflicts. Nodes in large conflict components get a small exploration bonus (they have more uncertainty).
- **Backup**: same running-average backup of costs, but **conflict edges persist** at the node level. When a rollout discovers a new conflict, the conflict edge is added to the node's conflict graph and stays there for all future visits.
- **Pruning**: same best-complete-cost pruning as V1. Additionally, if a component's combined relaxed cost + relocation penalties already exceeds the best known solution, prune it.

## Coupling / Conflict Handling

This is the core differentiator of V2. The conflict graph provides:

1. **Structured memory**: unlike V1 where coupling is only captured implicitly in MCTS statistics, V2 explicitly records which segments conflict and why
2. **Progressive merging**: segments start independent (component size 1) and are merged only when conflicts are discovered. This avoids the exponential cost of considering all segment interactions upfront.
3. **Conflict-driven search focus**: once a conflict is detected, future rollouts through that node are aware of it and can either avoid it (by choosing different subgoals) or handle it (by joint refinement)

### Conflict detection specifics

After applying a subgoal to segment `(parent, child)`:
- Extract the footprint: `{robot: position for robot in [subgoal.bottleneck, subgoal.support]}`
- For each other segment `(p2, c2)` with a footprint:
  - If any robot appears in both footprints at different positions → `helper_reuse` edge
  - If the subgoal moves the target robot and another segment requires it stationary at its current position → `mover_blocker` edge

### Relocation cost

When two segments conflict via `helper_reuse` (helper H needed at P1 and P2):
- Relocation cost = `grid_env.compute_relaxed_shortest_path_length(P1, P2)`
- This cost is added to the plan evaluation as a penalty
- The plan DAG may need a "relocation edge" connecting the two subgoal nodes that share the helper

## Key Differences from Other Versions

- **vs V1**: adds explicit conflict graph and merging — converts implicit cost feedback into structured coupling information
- **vs V3**: conflict edges are simple pairwise annotations, not a full constraint store with no-goods and backjumping
- **vs V4**: conflicts are local to each plan node, not shared across plans via segment transpositions
- **vs V5**: single-level search with inline conflict handling, not separated into skeleton + realization

## Complexity & Tradeoffs

**Strengths:**
- Optimistic by default — only pays the cost of joint reasoning when coupling actually occurs
- Conflict graph is lightweight and easy to maintain
- Progressive merging naturally adapts to instance difficulty (easy instances = small components, hard instances = larger merged components)

**Weaknesses:**
- Conflict detection is reactive — the first few rollouts through a coupling may be wasted before the conflict is discovered
- Relocation penalties are heuristic — the actual cost of accommodating two conflicting segments may be higher or lower than the relaxed SPL estimate
- Merging is monotonic (components only grow) — if a conflict was spurious (caused by a specific subgoal choice that won't be repeated), the merge is permanent within that node

**Best suited for:** instances with moderate coupling — some segments interact but most are independent. The progressive merging adapts well to this middle ground.
