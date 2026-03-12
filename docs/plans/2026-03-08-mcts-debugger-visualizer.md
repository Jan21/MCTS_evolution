# MCTS Debugger Visualizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite `mcts_visualizer.html` as a debugger-first MCTS visualizer where pseudocode line-stepping drives contextual updates in the tree, DAG, and grid panes.

**Architecture:** Single-file HTML template (injected with trace JSON via `visualize_mcts.py`). Left column = pseudocode debugger + watch panel. Right column = 3 stacked visualization panes. A sub-step system expands each trace event into granular pseudocode line activations. No backend changes needed — `mcts_tracer.py` and `visualize_mcts.py` stay unchanged.

**Tech Stack:** D3.js v7 (CDN), Canvas API, vanilla JS. Single HTML file with inline CSS/JS.

---

## Task 1: HTML Shell & CSS Layout

**Files:**
- Create: `mcts_visualizer_v2.html` (new file, will replace `mcts_visualizer.html` later)

**Step 1: Write the HTML structure**

Create the full HTML shell with the new layout. This is the skeleton — no JS yet.

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MCTS Debugger Visualizer</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
/* Full CSS here — see details below */
</style>
</head>
<body>
  <div id="top-bar">
    <span class="title">MCTS Debugger</span>
    <span class="stat">Iteration: <b id="stat-iter">0</b></span>
    <span class="stat">Best Cost: <b id="stat-best">-</b></span>
    <span class="stat">Tree Nodes: <b id="stat-nodes">1</b></span>
    <span class="stat">Plans Found: <b id="stat-plans">0</b></span>
  </div>

  <div id="main">
    <!-- Left: Debugger -->
    <div id="debugger-col">
      <div class="pane" id="pane-code">
        <div class="pane-header"><span>Pseudocode</span><span id="code-phase" style="font-weight:normal;color:#888;"></span></div>
        <div class="pane-body" id="code-pane"></div>
      </div>
      <div class="pane" id="pane-watch">
        <div class="pane-header clickable" id="watch-header"><span>Watch</span><span id="watch-toggle">▼</span></div>
        <div class="pane-body" id="watch-pane"></div>
      </div>
    </div>

    <!-- Right: Visualizations -->
    <div id="viz-col">
      <div class="pane" id="pane-tree">
        <div class="pane-header"><span>MCTS Search Tree</span><span id="tree-info" style="font-weight:normal;color:#888;"></span></div>
        <div class="pane-body" id="tree-pane"></div>
      </div>
      <div class="pane" id="pane-dag">
        <div class="pane-header"><span>Partial Plan DAG</span><span id="dag-info" style="font-weight:normal;color:#888;"></span></div>
        <div class="pane-body" id="dag-pane"></div>
      </div>
      <div class="pane" id="pane-grid">
        <div class="pane-header"><span>Grid Board</span><span id="grid-info" style="font-weight:normal;color:#888;"></span></div>
        <div class="pane-body" id="grid-pane"><canvas id="grid-canvas"></canvas></div>
      </div>
    </div>
  </div>

  <div id="timeline">
    <button id="btn-prev-iter" title="Previous iteration (Shift+←)">⇤</button>
    <button id="btn-prev" title="Previous step (←)">◄</button>
    <button id="btn-play" title="Play/Pause (Space)">▶</button>
    <button id="btn-next" title="Next step (→)">►</button>
    <button id="btn-next-iter" title="Next iteration (Shift+→)">⇥</button>
    <input type="range" id="slider" min="0" max="0" value="0">
    <span class="iter-label" id="slider-label">iter 0, step 0/0</span>
    <div id="bookmark-bar"></div>
  </div>
</body>
</html>
```

CSS layout rules (key parts):

```css
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }

#top-bar { background: #16213e; padding: 8px 16px; display: flex; align-items: center; gap: 16px; border-bottom: 1px solid #333; flex-shrink: 0; }
#top-bar .title { font-size: 14px; font-weight: 600; color: #a8d8ea; }
#top-bar .stat { font-size: 12px; color: #888; }
#top-bar .stat b { color: #e0e0e0; }

#main { flex: 1; display: flex; gap: 2px; background: #333; min-height: 0; }
#debugger-col { width: 40%; display: flex; flex-direction: column; gap: 2px; min-height: 0; }
#viz-col { width: 60%; display: flex; flex-direction: column; gap: 2px; min-height: 0; }

#pane-code { flex: 1; min-height: 0; }
#pane-watch { flex: 0 0 auto; max-height: 40%; transition: max-height 0.2s; }
#pane-watch.collapsed { max-height: 28px; }
#pane-watch.collapsed .pane-body { display: none; }

#pane-tree, #pane-dag, #pane-grid { flex: 1; min-height: 0; }

.pane { background: #1a1a2e; display: flex; flex-direction: column; overflow: hidden; min-height: 0; }
.pane-header { background: #16213e; padding: 6px 12px; font-size: 12px; font-weight: 600; color: #a8d8ea; border-bottom: 1px solid #333; flex-shrink: 0; display: flex; justify-content: space-between; align-items: center; }
.pane-header.clickable { cursor: pointer; }
.pane-body { flex: 1; overflow: auto; position: relative; min-height: 0; }

/* Pseudocode styling */
.code-line { display: flex; font-family: 'Consolas', 'Fira Code', monospace; font-size: 12px; line-height: 1.7; padding: 0 8px; cursor: pointer; }
.code-line:hover { background: rgba(255,255,255,0.05); }
.code-line .gutter { width: 28px; color: #555; text-align: right; padding-right: 8px; user-select: none; flex-shrink: 0; }
.code-line .code { white-space: pre; }
.code-line.active { background: rgba(255, 215, 0, 0.15); }
.code-line.active .gutter { color: #ffd700; }
.code-line.active .gutter::before { content: "►"; position: absolute; left: 4px; color: #ffd700; }
.code-line.phase-bg { background: rgba(168, 216, 234, 0.05); }
.code-line.dimmed { opacity: 0.35; }
.code-comment { color: #6a9955; }
.code-keyword { color: #c586c0; }
.code-function { color: #dcdcaa; }
.code-number { color: #b5cea8; }

/* Watch panel */
#watch-pane { padding: 8px 12px; font-size: 12px; }
#watch-pane table { width: 100%; border-collapse: collapse; }
#watch-pane td, #watch-pane th { padding: 2px 6px; text-align: left; border-bottom: 1px solid #333; }
#watch-pane th { color: #888; font-weight: normal; width: 40%; }

/* Timeline */
#timeline { background: #16213e; padding: 8px 16px; border-top: 1px solid #333; flex-shrink: 0; display: flex; align-items: center; gap: 8px; }
#timeline button { background: #0f3460; color: #e0e0e0; border: 1px solid #555; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }
#timeline button:hover { background: #1a5276; }
#timeline button.active { background: #2980b9; }
#timeline input[type=range] { flex: 1; accent-color: #a8d8ea; }
#timeline .iter-label { font-size: 12px; min-width: 120px; }
#bookmark-bar { display: flex; gap: 4px; margin-left: 8px; }
.bookmark { width: 8px; height: 8px; border-radius: 50%; cursor: pointer; border: 1px solid #555; }
.bookmark.new-best { background: #2ecc71; }
.bookmark.first-complete { background: #f39c12; }

/* Reuse existing tree/dag/grid styles from v1 */
#tree-pane svg { width: 100%; height: 100%; }
.tree-node circle { cursor: pointer; stroke-width: 2; }
.tree-node text { font-size: 10px; fill: #ccc; }
.tree-link { fill: none; stroke: #555; stroke-width: 1.5; }
.tree-link.selection-path { stroke: #ffd700; stroke-width: 3; }
.tree-link.backprop-path { stroke: #4ecdc4; stroke-width: 2.5; stroke-dasharray: 4,2; }
.tree-node.selected circle { stroke: #ffd700; stroke-width: 3; }

#dag-pane svg { width: 100%; height: 100%; }
.dag-node { cursor: pointer; }
.dag-node text { font-size: 10px; fill: #ccc; }
.dag-edge { fill: none; stroke-width: 1.5; }
.dag-edge.fixed { stroke: #4ecdc4; }
.dag-edge.open { stroke: #e74c3c; stroke-dasharray: 5,3; }
.dag-edge.structural { stroke: #555; stroke-width: 1; }
.dag-edge-label { font-size: 9px; fill: #888; }
.dag-node.new-node { filter: drop-shadow(0 0 4px #2ecc71); }
.dag-edge.target-edge { stroke: #ffd700; stroke-width: 3; }

#grid-canvas { display: block; }

.cost-low { color: #2ecc71; }
.cost-mid { color: #f39c12; }
.cost-high { color: #e74c3c; }
```

**Step 2: Verify the HTML renders**

Open `mcts_visualizer_v2.html` in a browser. Confirm the layout shows:
- Top stats bar
- Left column with pseudocode + watch panes
- Right column with 3 stacked panes
- Bottom timeline with 5 buttons + slider

Expected: Empty panes with correct layout proportions. No JS errors.

**Step 3: Commit**

```bash
git add mcts_visualizer_v2.html
git commit -m "feat: add v2 visualizer HTML shell with debugger layout"
```

---

## Task 2: Pseudocode Rendering & Sub-step Engine

**Files:**
- Modify: `mcts_visualizer_v2.html` (add `<script>` section)

**Step 1: Add the pseudocode data and rendering**

Add inside `<script>`:

```javascript
// ============================================================
// PSEUDOCODE DEFINITION
// ============================================================
// Each line: [lineNum, indentLevel, text, phase, lineId]
// lineId is used to map sub-steps to lines
const PSEUDOCODE = [
  [1, 0, "for iter in 1..max_iterations:", null, "loop"],
  [2, 0, "", null, "blank"],
  [3, 1, "# === SELECTION ===", "select", "select_header"],
  [4, 1, "node = root", "select", "select_start"],
  [5, 1, "while node.children:", "select", "select_while"],
  [6, 2, "if can_expand(node):", "select", "select_pw_check"],
  [7, 3, "break", "select", "select_pw_break"],
  [8, 2, "viable = [c for c in children", "select", "select_viable"],
  [9, 2, "         if c.cost < best_cost]", "select", "select_viable2"],
  [10, 2, "node = argmin(viable, LCB)", "select", "select_lcb"],
  [11, 0, "", null, "blank2"],
  [12, 1, "# === EXPANSION ===", "expand", "expand_header"],
  [13, 1, "if node.plan.is_complete():", "expand", "expand_complete_check"],
  [14, 2, "cost = evaluate(plan)", "expand", "expand_complete_eval"],
  [15, 2, "goto BACKPROP", "expand", "expand_complete_goto"],
  [16, 1, "edge = max(open_edges, key=cost)", "expand", "expand_pick_edge"],
  [17, 1, "candidates = get_candidates(edge)", "expand", "expand_candidates"],
  [18, 1, "for subgoal in sorted(candidates):", "expand", "expand_loop"],
  [19, 2, "if subgoal not in tried:", "expand", "expand_tried"],
  [20, 2, "child = apply_subgoal(plan, subgoal)", "expand", "expand_apply"],
  [21, 2, "close_optimal_edges(child)", "expand", "expand_close"],
  [22, 2, "break", "expand", "expand_break"],
  [23, 1, "if no child: goto BACKPROP (dead end)", "expand", "expand_dead_end"],
  [24, 0, "", null, "blank3"],
  [25, 1, "# === ROLLOUT ===", "rollout", "rollout_header"],
  [26, 1, "plan = copy(child.plan)", "rollout", "rollout_copy"],
  [27, 1, "for step in 1..max_depth:", "rollout", "rollout_loop"],
  [28, 2, "close_all_closeable_edges()", "rollout", "rollout_close"],
  [29, 2, "if no open edges: break", "rollout", "rollout_done"],
  [30, 2, "edge = random(open_edges)", "rollout", "rollout_pick"],
  [31, 2, "subgoal = random_top_half(candidates)", "rollout", "rollout_subgoal"],
  [32, 2, "apply_subgoal(plan, subgoal)", "rollout", "rollout_apply"],
  [33, 1, "cost = evaluate(plan) or plan.cost()", "rollout", "rollout_eval"],
  [34, 0, "", null, "blank4"],
  [35, 1, "# === BACKPROP ===", "backprop", "backprop_header"],
  [36, 1, "while node is not None:", "backprop", "backprop_loop"],
  [37, 2, "node.visits += 1", "backprop", "backprop_visit"],
  [38, 2, "node.total_cost += cost", "backprop", "backprop_cost"],
  [39, 2, "node = node.parent", "backprop", "backprop_up"],
];

function renderPseudocode(activeLineId, activePhase, executedLineIds) {
  const pane = document.getElementById("code-pane");
  let html = "";
  PSEUDOCODE.forEach(([num, indent, text, phase, lineId]) => {
    const isActive = lineId === activeLineId;
    const isPhase = phase === activePhase;
    const isExecuted = executedLineIds.has(lineId);
    const isDimmed = activePhase && phase && phase !== activePhase && !isExecuted;

    let cls = "code-line";
    if (isActive) cls += " active";
    else if (isPhase) cls += " phase-bg";
    if (isDimmed) cls += " dimmed";

    // Syntax highlight
    let highlighted = text
      .replace(/^(# .*)$/, '<span class="code-comment">$1</span>')
      .replace(/\b(for|in|if|while|break|goto|not|or)\b/g, '<span class="code-keyword">$1</span>')
      .replace(/\b(can_expand|argmin|max|evaluate|get_candidates|sorted|apply_subgoal|close_optimal_edges|close_all_closeable_edges|random|random_top_half|copy)\b/g, '<span class="code-function">$1</span>');

    const indentStr = "  ".repeat(indent);
    html += `<div class="${cls}" data-line-id="${lineId}">`;
    html += `<span class="gutter">${num}</span>`;
    html += `<span class="code">${indentStr}${highlighted}</span>`;
    html += `</div>`;
  });
  pane.innerHTML = html;

  // Click handler: jump to sub-step for this line
  pane.querySelectorAll(".code-line").forEach(el => {
    el.addEventListener("click", () => {
      const lid = el.dataset.lineId;
      jumpToLineInCurrentIteration(lid);
    });
  });

  // Scroll active line into view
  const activeLine = pane.querySelector(".code-line.active");
  if (activeLine) activeLine.scrollIntoView({ block: "center", behavior: "smooth" });

  // Phase label
  document.getElementById("code-phase").textContent = activePhase ? activePhase.toUpperCase() : "";
}
```

**Step 2: Add the sub-step engine**

This converts each trace event into an array of sub-steps. Add after the pseudocode code:

```javascript
// ============================================================
// SUB-STEP ENGINE
// ============================================================
// Expand each trace event into ordered sub-steps
// Each sub-step: { lineId, phase, vars, context }

function buildSubSteps(ev, evIndex) {
  const steps = [];
  const phase = ev.phase;

  // --- SELECTION ---
  // Always starts with select
  steps.push({
    lineId: "select_start", phase: "select",
    vars: { node: ev.selection_path?.[0] ?? 0 },
    context: { treeHighlight: { selectionPath: [ev.selection_path?.[0] ?? 0] } }
  });

  if (ev.selection_path && ev.selection_path.length > 1) {
    // Walk through selection path
    for (let i = 1; i < ev.selection_path.length; i++) {
      const pathSoFar = ev.selection_path.slice(0, i + 1);
      const nodeId = ev.selection_path[i];

      // LCB evaluation step
      if (Object.keys(ev.lcb_scores || {}).length > 0) {
        steps.push({
          lineId: "select_lcb", phase: "select",
          vars: {
            node: nodeId,
            viable_count: Object.keys(ev.lcb_scores).length,
            lcb_scores: ev.lcb_scores,
          },
          context: { treeHighlight: { selectionPath: pathSoFar, lcbScores: ev.lcb_scores } }
        });
      } else {
        steps.push({
          lineId: "select_while", phase: "select",
          vars: { node: nodeId },
          context: { treeHighlight: { selectionPath: pathSoFar } }
        });
      }
    }
  }

  // If can_expand returned true (node has children but we stopped)
  if (ev.selection_path && ev.selection_path.length > 1 && phase === "select_expand_rollout") {
    // We may have stopped due to progressive widening
    // Only add PW check if the selected node has children (indicated by selection stopping early)
  }

  // --- PHASE BRANCHING ---
  if (phase === "complete_node") {
    // Complete node: evaluate and backprop
    steps.push({
      lineId: "expand_complete_check", phase: "expand",
      vars: { is_complete: true },
      context: { treeHighlight: { selectedNode: ev.selected_node_id } }
    });
    steps.push({
      lineId: "expand_complete_eval", phase: "expand",
      vars: { cost: ev.backprop_cost },
      context: { treeHighlight: { selectedNode: ev.selected_node_id }, dagPlan: ev.plan_snapshot }
    });
    steps.push({
      lineId: "expand_complete_goto", phase: "expand",
      vars: {},
      context: { treeHighlight: { selectedNode: ev.selected_node_id } }
    });

  } else if (phase === "dead_end") {
    // Dead end: no candidates
    steps.push({
      lineId: "expand_pick_edge", phase: "expand",
      vars: { edge: ev.target_edge },
      context: {
        treeHighlight: { selectedNode: ev.selected_node_id },
        dagPlan: ev.plan_snapshot,
        dagHighlight: { targetEdge: ev.target_edge },
      }
    });
    steps.push({
      lineId: "expand_candidates", phase: "expand",
      vars: { candidates_count: 0 },
      context: { dagPlan: ev.plan_snapshot, dagHighlight: { targetEdge: ev.target_edge } }
    });
    steps.push({
      lineId: "expand_dead_end", phase: "expand",
      vars: { cost: ev.backprop_cost },
      context: { dagPlan: ev.plan_snapshot }
    });

  } else if (phase === "select_expand_rollout") {
    // Full cycle: expand + rollout

    // EXPANSION
    if (ev.target_edge) {
      steps.push({
        lineId: "expand_pick_edge", phase: "expand",
        vars: { edge: ev.target_edge },
        context: {
          treeHighlight: { selectedNode: ev.selected_node_id },
          dagPlan: ev.plan_snapshot,
          dagHighlight: { targetEdge: ev.target_edge },
        }
      });
    }
    steps.push({
      lineId: "expand_candidates", phase: "expand",
      vars: { candidates_count: ev.candidates_count },
      context: {
        dagPlan: ev.plan_snapshot,
        dagHighlight: { targetEdge: ev.target_edge },
      }
    });
    if (ev.action) {
      steps.push({
        lineId: "expand_apply", phase: "expand",
        vars: {
          bn_pos: ev.action.bn_pos,
          sp_pos: ev.action.sp_pos,
          helper_color: ev.action.helper_color,
        },
        context: {
          treeHighlight: { selectedNode: ev.selected_node_id, expandedChild: ev.expanded_child_id },
          dagPlan: ev.plan_snapshot,
          dagHighlight: { targetEdge: ev.target_edge, action: ev.action },
          gridHighlight: { positions: [ev.action.bn_pos, ev.action.sp_pos], type: "subgoal" },
        }
      });
      steps.push({
        lineId: "expand_close", phase: "expand",
        vars: {},
        context: { dagPlan: ev.plan_snapshot }
      });
    }

    // ROLLOUT
    if (ev.rollout_steps && ev.rollout_steps.length > 0) {
      steps.push({
        lineId: "rollout_copy", phase: "rollout",
        vars: {},
        context: {
          treeHighlight: { expandedChild: ev.expanded_child_id },
          dagPlan: ev.plan_snapshot,
        }
      });

      ev.rollout_steps.forEach((rs, i) => {
        if (rs.type === "close_only") {
          steps.push({
            lineId: "rollout_close", phase: "rollout",
            vars: { step: rs.step, closed: rs.closed },
            context: { treeHighlight: { expandedChild: ev.expanded_child_id } }
          });
          steps.push({
            lineId: "rollout_done", phase: "rollout",
            vars: { step: rs.step },
            context: { treeHighlight: { expandedChild: ev.expanded_child_id } }
          });
        } else if (rs.type === "no_candidates") {
          steps.push({
            lineId: "rollout_pick", phase: "rollout",
            vars: { step: rs.step, edge: rs.edge },
            context: { treeHighlight: { expandedChild: ev.expanded_child_id } }
          });
        } else if (rs.type === "subgoal") {
          steps.push({
            lineId: "rollout_subgoal", phase: "rollout",
            vars: {
              step: rs.step,
              bn_pos: rs.bn_pos,
              sp_pos: rs.sp_pos,
              helper_color: rs.helper_color,
              candidates_count: rs.candidates_count,
              applied: rs.applied,
            },
            context: {
              treeHighlight: { expandedChild: ev.expanded_child_id },
              gridHighlight: { positions: [rs.bn_pos, rs.sp_pos], type: "rollout" },
            }
          });
        }
      });

      steps.push({
        lineId: "rollout_eval", phase: "rollout",
        vars: { cost: ev.rollout_cost, completed: ev.rollout_completed },
        context: { treeHighlight: { expandedChild: ev.expanded_child_id } }
      });
    }
  }

  // --- BACKPROP (all phases end with backprop) ---
  steps.push({
    lineId: "backprop_loop", phase: "backprop",
    vars: { cost: ev.backprop_cost },
    context: {
      treeHighlight: { backpropPath: ev.selection_path, backpropChild: ev.expanded_child_id },
      dagPlan: ev.plan_snapshot,
    }
  });

  // Walk back up the selection path for individual backprop steps
  const backpropPath = [...(ev.selection_path || [])].reverse();
  if (ev.expanded_child_id !== null && ev.expanded_child_id !== undefined) {
    backpropPath.unshift(ev.expanded_child_id);
  }
  backpropPath.forEach((nodeId, i) => {
    steps.push({
      lineId: i < backpropPath.length - 1 ? "backprop_up" : "backprop_visit",
      phase: "backprop",
      vars: { node: nodeId, cost: ev.backprop_cost },
      context: {
        treeHighlight: {
          backpropPath: backpropPath.slice(0, i + 1),
          backpropNode: nodeId,
        },
      }
    });
  });

  return steps;
}

// Pre-compute all sub-steps on load
const allSubSteps = [];  // flat array: [{eventIndex, stepIndex, ...substep}]
const iterationBoundaries = [];  // index into allSubSteps where each iteration starts
events.forEach((ev, i) => {
  iterationBoundaries.push(allSubSteps.length);
  const steps = buildSubSteps(ev, i);
  steps.forEach((s, j) => {
    allSubSteps.push({ ...s, eventIndex: i, stepIndex: j, totalSteps: steps.length });
  });
});
```

**Step 3: Verify sub-steps generate correctly**

Add a temporary `console.log(allSubSteps.length, iterationBoundaries.length)` and check in browser console. For a typical trace with ~50 events, expect 200-500 sub-steps.

**Step 4: Commit**

```bash
git add mcts_visualizer_v2.html
git commit -m "feat: add pseudocode definition and sub-step engine"
```

---

## Task 3: Navigation State & Timeline Controls

**Files:**
- Modify: `mcts_visualizer_v2.html`

**Step 1: Add navigation state and control functions**

```javascript
// ============================================================
// NAVIGATION STATE
// ============================================================
let currentSubStep = 0;  // index into allSubSteps
let playing = false;
let playTimer = null;

function getCurrentEvent() {
  const ss = allSubSteps[currentSubStep];
  return ss ? events[ss.eventIndex] : null;
}
function getCurrentIterationIndex() {
  const ss = allSubSteps[currentSubStep];
  return ss ? ss.eventIndex : 0;
}

function setSubStep(idx) {
  currentSubStep = Math.max(0, Math.min(idx, allSubSteps.length - 1));
  const ss = allSubSteps[currentSubStep];
  slider.value = ss.eventIndex;
  const iterSteps = iterationBoundaries[ss.eventIndex + 1]
    ? iterationBoundaries[ss.eventIndex + 1] - iterationBoundaries[ss.eventIndex]
    : allSubSteps.length - iterationBoundaries[ss.eventIndex];
  const stepInIter = currentSubStep - iterationBoundaries[ss.eventIndex];
  sliderLabel.textContent = `iter ${ss.eventIndex + 1}/${events.length}, step ${stepInIter + 1}/${iterSteps}`;
  updateAll();
}

function nextSubStep() { stopPlay(); setSubStep(currentSubStep + 1); }
function prevSubStep() { stopPlay(); setSubStep(currentSubStep - 1); }
function nextIteration() {
  stopPlay();
  const curIter = getCurrentIterationIndex();
  if (curIter + 1 < iterationBoundaries.length) {
    setSubStep(iterationBoundaries[curIter + 1]);
  }
}
function prevIteration() {
  stopPlay();
  const curIter = getCurrentIterationIndex();
  if (curIter > 0) {
    setSubStep(iterationBoundaries[curIter - 1]);
  } else {
    setSubStep(0);
  }
}

function jumpToLineInCurrentIteration(lineId) {
  const curIter = getCurrentIterationIndex();
  const start = iterationBoundaries[curIter];
  const end = iterationBoundaries[curIter + 1] || allSubSteps.length;
  for (let i = start; i < end; i++) {
    if (allSubSteps[i].lineId === lineId) {
      setSubStep(i);
      return;
    }
  }
}

function startPlay() {
  playing = true;
  document.getElementById("btn-play").textContent = "⏸";
  document.getElementById("btn-play").classList.add("active");
  playTimer = setInterval(() => {
    if (currentSubStep >= allSubSteps.length - 1) { stopPlay(); return; }
    setSubStep(currentSubStep + 1);
  }, 300);
}
function stopPlay() {
  playing = false;
  document.getElementById("btn-play").textContent = "▶";
  document.getElementById("btn-play").classList.remove("active");
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
}
```

**Step 2: Wire up controls and keyboard shortcuts**

```javascript
// ============================================================
// TIMELINE WIRING
// ============================================================
const slider = document.getElementById("slider");
const sliderLabel = document.getElementById("slider-label");
slider.max = Math.max(0, events.length - 1);

slider.addEventListener("input", () => {
  const iterIdx = parseInt(slider.value);
  setSubStep(iterationBoundaries[iterIdx] || 0);
});

document.getElementById("btn-prev").addEventListener("click", prevSubStep);
document.getElementById("btn-next").addEventListener("click", nextSubStep);
document.getElementById("btn-prev-iter").addEventListener("click", prevIteration);
document.getElementById("btn-next-iter").addEventListener("click", nextIteration);
document.getElementById("btn-play").addEventListener("click", () => {
  if (playing) stopPlay(); else startPlay();
});

// Watch panel toggle
document.getElementById("watch-header").addEventListener("click", () => {
  document.getElementById("pane-watch").classList.toggle("collapsed");
  const toggle = document.getElementById("watch-toggle");
  toggle.textContent = document.getElementById("pane-watch").classList.contains("collapsed") ? "▶" : "▼";
});

// Keyboard
document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft" && e.shiftKey) prevIteration();
  else if (e.key === "ArrowRight" && e.shiftKey) nextIteration();
  else if (e.key === "ArrowLeft") prevSubStep();
  else if (e.key === "ArrowRight") nextSubStep();
  else if (e.key === " ") { e.preventDefault(); if (playing) stopPlay(); else startPlay(); }
  else if (e.key === "1") jumpToPhase("select");
  else if (e.key === "2") jumpToPhase("expand");
  else if (e.key === "3") jumpToPhase("rollout");
  else if (e.key === "4") jumpToPhase("backprop");
});

function jumpToPhase(phase) {
  const curIter = getCurrentIterationIndex();
  const start = iterationBoundaries[curIter];
  const end = iterationBoundaries[curIter + 1] || allSubSteps.length;
  for (let i = start; i < end; i++) {
    if (allSubSteps[i].phase === phase) {
      setSubStep(i);
      return;
    }
  }
}

// Bookmarks
const bookmarks = [];
{
  let firstComplete = false;
  let bestCost = Infinity;
  events.forEach((e, i) => {
    if (e.complete_plans_found > 0 && !firstComplete) {
      bookmarks.push({ index: i, type: "first-complete" });
      firstComplete = true;
    }
    if (e.best_cost !== null && e.best_cost < bestCost) {
      bestCost = e.best_cost;
      bookmarks.push({ index: i, type: "new-best" });
    }
  });
  const bar = document.getElementById("bookmark-bar");
  bookmarks.forEach(b => {
    const dot = document.createElement("div");
    dot.className = `bookmark ${b.type}`;
    dot.title = `${b.type} at iteration ${b.index + 1}`;
    dot.addEventListener("click", () => { stopPlay(); setSubStep(iterationBoundaries[b.index] || 0); });
    bar.appendChild(dot);
  });
}
```

**Step 3: Verify navigation**

Open in browser. Arrow keys should step through sub-steps. Shift+arrows should jump iterations. Slider should scrub iterations. Label should show "iter X/Y, step A/B".

**Step 4: Commit**

```bash
git add mcts_visualizer_v2.html
git commit -m "feat: add dual-level navigation and timeline controls"
```

---

## Task 4: Watch Panel Rendering

**Files:**
- Modify: `mcts_visualizer_v2.html`

**Step 1: Implement the watch panel renderer**

```javascript
// ============================================================
// WATCH PANEL
// ============================================================
function renderWatch() {
  const pane = document.getElementById("watch-pane");
  const ss = allSubSteps[currentSubStep];
  if (!ss) { pane.innerHTML = ""; return; }

  const vars = ss.vars || {};
  const ev = events[ss.eventIndex];
  let html = "<table>";

  // Always show iteration + phase
  html += row("iteration", ev.iteration);
  html += row("phase", ss.phase);
  html += row("best_cost", fmtCost(ev.best_cost));

  // Phase-specific variables
  Object.entries(vars).forEach(([key, val]) => {
    html += row(key, formatVar(key, val));
  });

  html += "</table>";
  pane.innerHTML = html;
}

function row(key, val) {
  return `<tr><th>${key}</th><td>${val}</td></tr>`;
}

function formatVar(key, val) {
  if (val === null || val === undefined) return "-";
  if (key === "cost" || key === "rollout_cost" || key === "backprop_cost") return `<span class="${costClass(val)}">${fmtCost(val)}</span>`;
  if (key === "edge" && Array.isArray(val)) return `${val[0]} → ${val[1]}`;
  if (key === "bn_pos" || key === "sp_pos") return fmtPos(val);
  if (key === "lcb_scores" && typeof val === "object") {
    const entries = Object.entries(val).sort((a, b) => a[1] - b[1]);
    return entries.map(([nid, lcb]) => `${nid}:${lcb === -Infinity ? "-∞" : Number(lcb).toFixed(2)}`).join(", ");
  }
  if (key === "closed" && Array.isArray(val)) return val.map(e => e.join("→")).join(", ");
  if (typeof val === "boolean") return val ? "✓" : "✗";
  if (typeof val === "number") return Number.isFinite(val) ? val.toString() : "∞";
  if (Array.isArray(val)) return JSON.stringify(val);
  return String(val);
}

function fmtCost(c) {
  if (c === null || c === undefined) return "-";
  if (c === Infinity || c > 999) return "∞";
  return Number(c).toFixed(1);
}
function costClass(c) {
  if (c === null || c === undefined || c === Infinity || c > 500) return "cost-high";
  if (c < 10) return "cost-low";
  if (c < 30) return "cost-mid";
  return "cost-high";
}
function fmtPos(p) {
  if (!p) return "?";
  return `(${p[0]},${p[1]})`;
}
```

**Step 2: Verify watch panel**

Step through iterations. Watch panel should show different variables per phase (LCB scores during select, edge/candidates during expand, etc.).

**Step 3: Commit**

```bash
git add mcts_visualizer_v2.html
git commit -m "feat: add watch panel with phase-specific variables"
```

---

## Task 5: MCTS Tree Pane (Context-Aware)

**Files:**
- Modify: `mcts_visualizer_v2.html`

**Step 1: Port and enhance the tree renderer**

Port the existing `initTree`, `buildTreeHierarchy`, `costColor`, and `renderTree` functions from the current `mcts_visualizer.html`. Modify `renderTree` to use the sub-step context instead of directly reading the event:

Key changes from v1:
- Read highlight info from `allSubSteps[currentSubStep].context.treeHighlight` instead of directly from event
- Add backprop path highlighting (teal dashed) when phase is "backprop"
- Add "new node" pulse when `context.treeHighlight.expandedChild` is set
- Tree node click: find the event where `expanded_child_id === clickedNodeId` and jump to it

The tree rendering code is ~100 lines — copy from `mcts_visualizer.html` lines 264-406, then adapt the highlight logic to read from `ss.context.treeHighlight` (where `ss = allSubSteps[currentSubStep]`).

Critical adaptation in `renderTree`:
```javascript
// Replace direct event reads with context-aware reads:
const ss = allSubSteps[currentSubStep];
const ctx = ss?.context?.treeHighlight || {};
const selectionPath = new Set((ctx.selectionPath || []).map(String));
const expandedChild = ctx.expandedChild;
const backpropPath = new Set((ctx.backpropPath || []).map(String));
const backpropNode = ctx.backpropNode;
```

Node click handler change:
```javascript
allNodes.on("click", (e, d) => {
  // Find the iteration where this node was expanded
  const targetIter = events.findIndex(ev => ev.expanded_child_id === d.data.id);
  if (targetIter >= 0) {
    stopPlay();
    setSubStep(iterationBoundaries[targetIter] || 0);
  }
});
```

**Step 2: Verify tree rendering**

Generate a trace: `python visualize_mcts.py --env 0 --iterations 50`
Inject into v2 template. Verify:
- Selection path highlights gold during SELECT phase
- New child pulses during EXPAND
- Backprop path shows teal during BACKPROP
- Clicking a node jumps to its expansion iteration

**Step 3: Commit**

```bash
git add mcts_visualizer_v2.html
git commit -m "feat: add context-aware MCTS tree pane"
```

---

## Task 6: Plan DAG Pane (Context-Aware)

**Files:**
- Modify: `mcts_visualizer_v2.html`

**Step 1: Port and enhance the DAG renderer**

Port `initDAG`, `NTYPE_SHAPES`, and `renderPlanDAG` from v1 (lines 529-687). Adapt to use sub-step context:

Key changes:
- Read plan from `ss.context.dagPlan || ev.plan_snapshot`
- Add `dagHighlight` support: when `context.dagHighlight.targetEdge` is set, highlight that edge with gold stroke class `target-edge`
- When `context.dagHighlight.action` is set, add `new-node` class (green glow via CSS `drop-shadow`) to the nodes that match the action's bn_pos/sp_pos

Critical adaptation:
```javascript
function renderPlanDAG() {
  // ...existing layout code...
  const ss = allSubSteps[currentSubStep];
  const dagHL = ss?.context?.dagHighlight || {};
  const targetEdgeKey = dagHL.targetEdge ? `${dagHL.targetEdge[0]}-${dagHL.targetEdge[1]}` : null;

  // When rendering edges, add target-edge class:
  plan.edges.forEach(e => {
    // ...existing edge rendering...
    const edgeKey = `${e.source}-${e.target}`;
    if (edgeKey === targetEdgeKey) {
      line.classed("target-edge", true);
    }
  });

  // When rendering nodes, add new-node class for action positions:
  if (dagHL.action) {
    plan.nodes.forEach(n => {
      if (n.pos && dagHL.action.bn_pos && n.pos[0] === dagHL.action.bn_pos[0] && n.pos[1] === dagHL.action.bn_pos[1] && n.ntype === "bottleneck") {
        g.classed("new-node", true);
      }
      // Similar for support
    });
  }
}
```

**Step 2: Verify DAG rendering**

Step to an EXPAND phase sub-step. Verify:
- Target edge highlighted gold
- New subgoal nodes glow green after apply_subgoal

**Step 3: Commit**

```bash
git add mcts_visualizer_v2.html
git commit -m "feat: add context-aware plan DAG pane"
```

---

## Task 7: Grid Board Pane (Context-Aware)

**Files:**
- Modify: `mcts_visualizer_v2.html`

**Step 1: Port and enhance the grid renderer**

Port `initGrid`, `renderGrid` from v1 (lines 692-860). Add context-aware highlighting:

Key addition — after drawing plan nodes, check `ss.context.gridHighlight`:
```javascript
const ss = allSubSteps[currentSubStep];
const gridHL = ss?.context?.gridHighlight || {};
if (gridHL.positions) {
  gridHL.positions.forEach(pos => {
    if (!pos) return;
    const [hx, hy] = pos;
    ctx.strokeStyle = gridHL.type === "subgoal" ? "#ffd700" : "#a8d8ea";
    ctx.lineWidth = 3;
    ctx.strokeRect(
      ox + (hx - min_x) * cellSize + 2,
      oy + (hy - min_y) * cellSize + 2,
      cellSize - 4, cellSize - 4
    );
  });
}
```

**Step 2: Verify grid highlighting**

During EXPAND, grid should highlight bottleneck/support positions with gold borders. During ROLLOUT, positions should highlight with blue borders.

**Step 3: Commit**

```bash
git add mcts_visualizer_v2.html
git commit -m "feat: add context-aware grid board pane"
```

---

## Task 8: Master Update Loop & Init

**Files:**
- Modify: `mcts_visualizer_v2.html`

**Step 1: Wire everything together**

```javascript
// ============================================================
// UPDATE ALL
// ============================================================
function updateAll() {
  const ss = allSubSteps[currentSubStep];
  if (!ss) return;
  const ev = events[ss.eventIndex];

  // Stats bar
  document.getElementById("stat-iter").textContent = ev.iteration;
  document.getElementById("stat-best").textContent = fmtCost(ev.best_cost);
  document.getElementById("stat-nodes").textContent = ev.tree_node_count;
  document.getElementById("stat-plans").textContent = ev.complete_plans_found;

  // Build set of executed lineIds for this iteration
  const curIter = ss.eventIndex;
  const start = iterationBoundaries[curIter];
  const end = iterationBoundaries[curIter + 1] || allSubSteps.length;
  const executedLineIds = new Set();
  for (let i = start; i < end; i++) {
    executedLineIds.add(allSubSteps[i].lineId);
  }

  renderPseudocode(ss.lineId, ss.phase, executedLineIds);
  renderWatch();
  renderTree();
  renderPlanDAG();
  renderGrid();
}

// ============================================================
// INIT
// ============================================================
function init() {
  initTree();
  initDAG();
  initGrid();

  if (allSubSteps.length > 0) {
    setSubStep(0);
  }
}

window.addEventListener("resize", () => {
  renderTree();
  renderPlanDAG();
  renderGrid();
});

init();
```

**Step 2: Add the trace data placeholder**

Make sure the script starts with:
```javascript
const TRACE = __TRACE_DATA_PLACEHOLDER__;
const events = TRACE.events || [];
const treeNodesData = {};
const treeEdgesData = [];
const grid = TRACE.grid || {};
const initialState = TRACE.initial_state || {};
const initialPlan = TRACE.initial_plan || {};

(TRACE.tree_nodes || []).forEach(n => { treeNodesData[n.id] = {...n}; });
(TRACE.tree_edges || []).forEach(e => { treeEdgesData.push(e); });

const ROBOT_COLORS = {
  "red": "#e74c3c", "blue": "#3498db", "green": "#2ecc71", "yellow": "#f1c40f",
  "R": "#e74c3c", "B": "#3498db", "G": "#2ecc71", "Y": "#f1c40f",
};
function robotColor(color) {
  return ROBOT_COLORS[color] || ROBOT_COLORS[color?.toLowerCase()] || "#9b59b6";
}
```

**Step 3: End-to-end test**

```bash
python visualize_mcts.py --env 0 --iterations 50 --output mcts_trace_v2.html
```

But first update `visualize_mcts.py` to point to the v2 template:
```python
HTML_TEMPLATE_PATH = Path(__file__).resolve().parent / "mcts_visualizer_v2.html"
```

Open `mcts_trace_v2.html` in browser. Verify all 4 panes update as you step through. Arrow keys work. Play works. Bookmarks work. Clicking tree nodes jumps to iterations. Clicking pseudocode lines jumps to sub-steps.

**Step 4: Commit**

```bash
git add mcts_visualizer_v2.html visualize_mcts.py
git commit -m "feat: wire up master update loop and end-to-end test"
```

---

## Task 9: Replace v1 & Clean Up

**Files:**
- Delete: `mcts_visualizer.html` (old v1)
- Rename: `mcts_visualizer_v2.html` → `mcts_visualizer.html`
- Modify: `visualize_mcts.py` (restore template path)

**Step 1: Replace files**

```bash
mv mcts_visualizer.html mcts_visualizer_v1_backup.html
mv mcts_visualizer_v2.html mcts_visualizer.html
```

**Step 2: Restore template path in visualize_mcts.py**

Change `HTML_TEMPLATE_PATH` back to:
```python
HTML_TEMPLATE_PATH = Path(__file__).resolve().parent / "mcts_visualizer.html"
```

**Step 3: Final end-to-end test**

```bash
python visualize_mcts.py --env 38 --output mcts_trace.html
```

Open and verify everything works.

**Step 4: Remove backup**

```bash
rm mcts_visualizer_v1_backup.html
```

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: replace v1 visualizer with debugger-focused v2"
```

---

## Summary

| Task | Description | Est. Lines |
|------|-------------|-----------|
| 1 | HTML shell & CSS layout | ~120 |
| 2 | Pseudocode definition & sub-step engine | ~200 |
| 3 | Navigation state & timeline controls | ~100 |
| 4 | Watch panel rendering | ~60 |
| 5 | MCTS tree pane (context-aware) | ~120 |
| 6 | Plan DAG pane (context-aware) | ~120 |
| 7 | Grid board pane (context-aware) | ~100 |
| 8 | Master update loop & init | ~50 |
| 9 | Replace v1 & clean up | ~5 |

Total: ~875 lines of HTML/CSS/JS in a single file.
