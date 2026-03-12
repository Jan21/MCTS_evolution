# MCTS Debugger Visualizer — Design Doc

## Goal

Redesign the MCTS visualizer (`mcts_visualizer.html`) with a pseudocode debugger as the primary view. The debugger drives the right-side panes contextually — as you step through pseudocode lines, the tree, DAG, and grid panes show information relevant to that specific line.

## Layout

```
+--------------------+------------------+
|                    |   MCTS Tree      |
|   PSEUDOCODE       |   (D3, zoomable) |
|   DEBUGGER         +------------------+
|   (~40% width)     |   Plan DAG       |
|                    |   (D3, zoomable) |
|   [code area]      +------------------+
|                    |   Grid Board     |
|   [watch panel]    |   (canvas)       |
+--------------------+------------------+
|   Timeline / Controls (iteration +    |
|   sub-step navigation)                |
+---------------------------------------+
```

- Left column (40%): Pseudocode debugger with collapsible watch panel below
- Right column (60%): 3 panes stacked vertically (tree, DAG, grid), each ~33% height
- Bottom: Timeline bar with dual-level navigation

## Pseudocode Debugger

~35 lines of medium-detail pseudocode covering the MCTS loop. Lines grouped into 4 blocks: SELECT, EXPAND, ROLLOUT, BACKPROP. Current line highlighted with yellow background + gutter marker. Active phase block gets subtle background tint.

```
 1  for iter in 1..max_iterations:
 2
 3    # === SELECTION ===
 4    node = root
 5    while node.children:
 6      if can_expand(node):        # progressive widening
 7        break
 8      viable = [c for c in children
 9                 if c.cost < best_cost]
10      node = argmin(viable, LCB)  # exploit - explore
11
12    # === EXPANSION ===
13    if node.plan.is_complete():
14      cost = evaluate(plan)
15      goto BACKPROP
16    edge = max(open_edges, key=cost)
17    candidates = get_candidates(edge)
18    for subgoal in sorted(candidates):
19      if subgoal not in tried:
20        child = apply_subgoal(plan, subgoal)
21        close_optimal_edges(child)
22        break
23    if no child: goto BACKPROP (dead end)
24
25    # === ROLLOUT ===
26    plan = copy(child.plan)
27    for step in 1..max_depth:
28      close_all_closeable_edges()
29      if no open edges: break
30      edge = random(open_edges)
31      subgoal = random_top_half(candidates)
32      apply_subgoal(plan, subgoal)
33    cost = evaluate(plan) or plan.cost()
34
35    # === BACKPROP ===
36    while node is not None:
37      node.visits += 1
38      node.total_cost += cost
39      node = node.parent
```

### Watch Panel

Collapsible table below the pseudocode. Shows key-value pairs relevant to the current sub-step's phase:

- **SELECT**: `node`, `viable` list, LCB scores per viable child
- **EXPAND**: `edge` (target open edge), `candidates_count`, `subgoal` (bn/sp/helper), `child` node id
- **ROLLOUT**: `step`, `edge`, `subgoal`, `closed_edges`, `cost`
- **BACKPROP**: `cost`, `node` (walking up), `visits`, `avg_cost`

## Context Linking (Pseudocode Line → Right Panes)

### During SELECTION (lines 4-10)

| Pane | Shows |
|------|-------|
| MCTS Tree | Selection path highlighted gold. LCB scores as labels on viable children. Current node pulsing. |
| Plan DAG | Plan of the currently selected node. |
| Grid | Positions from the selected node's plan. |

### During EXPANSION (lines 12-23)

| Pane | Shows |
|------|-------|
| MCTS Tree | Selected node highlighted. New child appears with "new" indicator on apply_subgoal. |
| Plan DAG | Before/after: target open edge highlighted red → new nodes (sg, bn, sp, leaf) appear with green glow. Closed edges flash. |
| Grid | Target edge positions. On subgoal apply: bottleneck (square), support (triangle), helper leaf (circle) with robot colors. |

### During ROLLOUT (lines 25-33)

| Pane | Shows |
|------|-------|
| MCTS Tree | Child node highlighted, dimmed (rollout is simulated). |
| Plan DAG | Rollout plan evolving step-by-step. Open edges dashed red, closed edges solid teal. |
| Grid | Rollout positions accumulate. New positions flash per step. |

### During BACKPROP (lines 35-39)

| Pane | Shows |
|------|-------|
| MCTS Tree | Backprop path highlighted teal, animating child → root. Node sizes/colors update. |
| Plan DAG | Frozen (final state from expansion/rollout). |
| Grid | Frozen. |

### Special Cases

- `complete_node` phase: SELECT → BACKPROP, skip EXPAND/ROLLOUT. DAG shows complete plan.
- `dead_end` phase: SELECT → EXPAND (no candidates) → BACKPROP with penalty. DAG shows stuck plan.

## Data Flow

### No tracer changes needed

Existing `MCTSTracer` fields map directly:
- `selection_path` + `lcb_scores` → SELECT sub-steps
- `target_edge` + `candidates_count` + `action` → EXPAND sub-steps
- `rollout_steps` + `rollout_cost` + `rollout_completed` → ROLLOUT sub-steps
- `backprop_cost` + `plan_snapshot` → BACKPROP sub-steps
- `phase` ("select_expand_rollout", "complete_node", "dead_end") → which lines activate

### Sub-step generation (client-side JS)

Each trace event is expanded on load into an ordered list of sub-steps:

```js
event → [
  {line: 4, phase: "select", vars: {node: 0}},
  {line: 10, phase: "select", vars: {node: 3, lcb: {...}}},
  {line: 16, phase: "expand", vars: {edge: [...]}},
  {line: 20, phase: "expand", vars: {subgoal: {...}}},
  {line: 36, phase: "backprop", vars: {cost: 12.5, node: 5}},
  {line: 37, phase: "backprop", vars: {node: 3}},
  {line: 37, phase: "backprop", vars: {node: 0}},
]
```

Each sub-step also carries a `context` object for the right panes:
```js
{
  treeHighlight: {selectionPath: [0,3], selectedNode: 3},
  dagPlan: <plan_snapshot>,
  dagHighlight: {newNodes: [...], targetEdge: [...]},
  gridHighlight: {positions: [...], type: "subgoal"},
}
```

## Timeline Controls

```
| < iter  > iter | < step  > step |  > play  | [===slider====] 12/47   |
|                                  |          | iter 5, step 3/7        |
| [*  *     *  **    *] bookmarks |  speed v  |                         |
```

- **Iteration arrows**: Jump to first sub-step of prev/next iteration
- **Step arrows**: Walk through sub-steps within current iteration
- **Play**: Auto-advances sub-steps, brief pause between iterations
- **Slider**: Scrubs iterations (snaps to first sub-step)
- **Label**: "iter 5, step 3/7"
- **Bookmarks**: Green = new best, orange = first complete

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Left/Right | Previous/next sub-step |
| Shift+Left/Right | Previous/next iteration |
| Space | Play/pause |
| 1-4 | Jump to SELECT/EXPAND/ROLLOUT/BACKPROP of current iteration |

### Click Interactions

- **Click pseudocode line**: Jump to that line's sub-step in current iteration (if executed)
- **Click MCTS tree node**: Jump to the iteration where that node was expanded
- Lines not executed in current iteration are dimmed

## File Structure

- `mcts_visualizer.html` — rewritten with new layout, all JS inline (single-file template)
- `visualize_mcts.py` — unchanged (injects trace JSON into `__TRACE_DATA_PLACEHOLDER__`)
- `mcts_tracer.py` — unchanged

## Tech Stack

Same as current: D3.js (tree + DAG), Canvas (grid), vanilla JS. No new dependencies.
