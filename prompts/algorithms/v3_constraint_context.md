# V3: Constraint-Context MCTS with Deep No-Good Learning

## Overview

The most sophisticated conflict-handling variant. Each MCTS node maintains a **constraint store** built from explicit commitment objects (BlockReq, ProtectReq, SupportReq). When a conflict is detected during rollout, the algorithm performs **backjumping** to the deepest decision that introduced the conflicting commitment, and records a **context-specific no-good** that prevents the same bad combination from being tried again. This avoids the V1 problem of penalizing top-level subgoal choices when the actual fault lies in a deep refinement decision.

## State Representation

Each MCTS node holds:

- **Partial plan**: a `PartialPlan` instance
- **Constraint store** (`context`): a list of commitment objects, each tied to the decision (action) that created it:
  - `BlockReq(robot: Robot_at, position: tuple, segment_id: str, decision_id: int)` — robot must be at position to serve as a blocker for the given segment
  - `ProtectReq(robot: Robot_at, segment_id: str, decision_id: int)` — robot must not be moved (added lazily, only when a mover-blocker conflict is detected)
  - `SupportReq(segment_id: str, choices: list[tuple[Robot_at, tuple]], chosen: Optional[tuple], decision_id: int)` — segment needs one of several (robot, position) options; `chosen` is None until committed
- **No-good table**: a set of `NoGood` entries, each being a tuple of (context_signature, forbidden_action) pairs. A no-good says: "in a context matching this signature, do not take this action."
- **Decision history**: ordered list of (decision_id, action, commitments_added) for backjumping

### Context Signature

A context signature is a hashable summary of the constraint store, used for no-good matching:
- `frozenset of (robot.color, robot.position, commitment_type) for each commitment`
- Approximate matching: a no-good applies if the current context is a superset of the no-good's context signature (more commitments = more constrained, so the no-good still applies)

## Action Space

Same two-stage action as V1/V2: pick an open segment, then pick a subgoal via `grid_env.propose_subgoal_states()`.

But with two critical filters:

### No-good filtering
Before expanding a child action at a node:
1. Compute the current context signature
2. Check the no-good table: if `(current_context_signature, candidate_action)` matches any no-good, skip this action
3. This prunes the action space dynamically based on learned conflict information

### Disjunctive support commitments
When a subgoal is proposed, instead of immediately committing to a specific helper robot, create a `SupportReq` with multiple `choices`. The commitment to a specific (robot, position) is deferred until:
- Only one choice remains feasible (forced by other constraints)
- The segment is about to become fixed (must commit)
- A rollout needs a concrete assignment

This lazy commitment reduces spurious conflicts — two segments that *could* use different helpers won't conflict until they're forced to share one.

## Evaluation / Rollout

### Rollout with constraint propagation

1. While `plan.open_edges()` is non-empty:
   a. Pick an open segment (highest relaxed cost)
   b. Generate subgoal candidates via `grid_env.propose_subgoal_states()`
   c. Filter by no-goods (skip any action matching a recorded no-good for the current context)
   d. Among remaining candidates, pick the one with best `grid_env.subgoal_score()`
   e. Apply the subgoal to the plan
   f. **Add commitments** to the constraint store:
      - `BlockReq` for the support robot at its designated position
      - `SupportReq` with all feasible (helper, position) choices
   g. **Check for conflicts** against existing commitments:
      - Same robot required at two different positions → conflict
      - Robot needed stationary but scheduled to move → conflict
      - All choices in a `SupportReq` eliminated → conflict

2. **On conflict**:
   a. Identify the **conflict set**: the minimal set of commitments that are mutually inconsistent
   b. Find the **culprit decision**: the most recent decision in the history whose commitments are in the conflict set
   c. Record a **no-good**: `(context_at_culprit_decision, culprit_action)` — meaning "given these prior commitments, this action leads to a dead end"
   d. **Backjump**: instead of continuing the rollout, return to the culprit decision and try a different action
   e. If no alternative actions exist at the culprit, backjump further up

3. If rollout completes without conflict:
   - Validate with `validate(plan, grid_env, state)`
   - Evaluate with `evaluate_plan(plan, grid_env)`
   - Return cost (or penalty on failure)

### Conflict set extraction

Given a conflict (e.g., robot R needed at P1 by segment S1 and P2 by segment S2):
- Conflict set = `{BlockReq(R, P1, S1, d1), BlockReq(R, P2, S2, d2)}`
- Culprit = the commitment with the larger `decision_id` (the later decision)
- Context = all commitments with `decision_id < culprit.decision_id`

## Backup & Selection

- **Selection**: UCB1 with no-good pruning. At each node, available actions = all children minus those filtered by no-goods for the current context. If all actions are filtered, the node is marked as **dead** (no feasible refinement).
- **Backup**:
  - Cost values propagate up as in V1
  - No-goods propagate **downward** — a no-good recorded at depth d applies to all future visits through that subtree
  - Dead nodes propagate upward — if all children of a node are dead, the node itself is dead
- **Backjumping integration**: when a rollout triggers backjumping, the cost attributed to the rollout is assigned only to the path from root to the culprit decision (not the full path to the leaf). This prevents blaming innocent ancestors.

## Coupling / Conflict Handling

V3's conflict handling has three layers:

### Layer 1: Preventive (lazy commitments)
- `SupportReq` with multiple choices delays commitment
- Conflicts that would arise from premature assignment are avoided entirely
- Commitment is only forced when necessary

### Layer 2: Detective (constraint checking)
- After each decision, check all pairs of commitments for consistency
- Specific checks:
  - **Position conflict**: same robot, different required positions
  - **Mobility conflict**: robot needed stationary by one segment, but moved by another
  - **Resource exhaustion**: a `SupportReq` with no remaining valid choices

### Layer 3: Corrective (no-goods + backjumping)
- When a conflict is detected, blame is assigned to the **specific deep decision** that caused it
- A no-good prevents the same mistake from being repeated in the same context
- Backjumping skips over innocent decisions, focusing search effort on the actual choice point

### No-good generalization
To maximize reuse of learned no-goods:
- A no-good `(ctx, action)` applies whenever the current context is a **superset** of `ctx`
- This means: if action A was bad given commitments {C1, C2}, it's also bad given {C1, C2, C3, ...}
- Subsumption: if no-good N1's context is a subset of N2's context for the same action, N1 subsumes N2 (N2 can be removed)

## Key Differences from Other Versions

- **vs V1**: adds full constraint reasoning — no-goods, backjumping, lazy commitments. Much more targeted conflict response.
- **vs V2**: V2 records pairwise conflicts between segments; V3 records no-goods about specific *decisions* in specific *contexts*. V3 is finer-grained — it can distinguish "subgoal X is bad when Y was already chosen" from "subgoal X is always bad."
- **vs V4**: V3's no-goods are local to each plan's decision history; V4's transposition table shares segment-level information across plans. These are complementary approaches.
- **vs V5**: V3 interleaves conflict detection with refinement in a single search; V5 separates skeleton choice from realization and passes conflicts between levels.

## Complexity & Tradeoffs

**Strengths:**
- Precise blame assignment — only the actual culprit decision is penalized, not the entire subgoal template
- No-good learning prevents repeating the same mistakes across rollouts
- Lazy commitments reduce the frequency of conflicts in the first place
- Backjumping avoids wasting rollout budget on doomed continuations

**Weaknesses:**
- Most complex to implement — requires maintaining decision history, constraint store, no-good table, and backjumping logic
- No-good table can grow large; needs periodic garbage collection or bounded size
- Context signatures must be carefully designed — too specific and no-goods rarely match; too general and they over-prune
- Backjumping within an MCTS rollout is non-standard and interacts subtly with the backup rule

**Best suited for:** instances with deep, structured coupling — where the same conflict pattern recurs across many rollouts and the conflict is caused by a specific deep decision, not by the top-level subgoal choice. Also valuable when the search budget is large enough to amortize the overhead of no-good learning.
