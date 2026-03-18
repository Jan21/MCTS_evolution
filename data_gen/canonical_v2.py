"""Canonical representation V2 — partial plan encoded into grid features.

Converts (GridEnv, State, PartialPlan, open_edge) into:
  - encoder input:  (256, FEATURE_DIM) per-cell feature matrix
  - decoder target:  3 tokens [bn_pos, sp_pos, robot_cell_or_TARGET]

Symmetries eliminated:
  1. Color permutation  — robots identified as target/helper by role, not color
  2. Branch ordering    — plan encoded spatially in grid cells, no sequence
  3. Temporal ordering  — explicitly encoded via before/after/independent

Robot colors are preserved in saved metadata for visualization but are NOT
part of the feature vector.
"""

from __future__ import annotations

import copy
import numpy as np
import networkx as nx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID_SIZE = 16
NUM_CELLS = GRID_SIZE * GRID_SIZE  # 256

# Output vocabulary for the 3rd token (robot identification)
#   0–255 : helper identified by its current cell position
#   256   : target robot
TOKEN_TARGET = 256
VOCAB_SIZE = 257

# Feature vector layout (per cell)
#   0    : x coordinate (0–15)
#   1    : y coordinate (0–15)
#   2–5  : wall N, S, W, E
#   6    : is_goal
#   7    : is_target_robot  (the robot that must move for the open edge)
#   8    : is_helper         (any other robot)
#   9    : temporal_before   (plan node closer to leaves than open edge)
#   10   : temporal_after    (plan node closer to goal than open edge)
#   11   : target_start      (child node of the open edge)
#   12   : target_end        (parent node of the open edge)
#   13   : is_bottleneck     (plan node is a bottleneck)
#   14   : is_support_or_leaf (plan node is a support or leaf)
#   15   : came_from_x       (source position x, -1 for leaves)
#   16   : came_from_y       (source position y, -1 for leaves)
#   17   : goes_to_x         (destination position x, -1 if terminal)
#   18   : goes_to_y         (destination position y, -1 if terminal)
FEATURE_DIM = 19


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------

def pos_to_index(x: int, y: int) -> int:
    """(x, y) -> flat cell index (row-major: y * 16 + x)."""
    return y * GRID_SIZE + x


def index_to_pos(idx: int) -> tuple[int, int]:
    """Flat cell index -> (x, y)."""
    return idx % GRID_SIZE, idx // GRID_SIZE


# ---------------------------------------------------------------------------
# Temporal classification
# ---------------------------------------------------------------------------

def classify_temporal(plan, open_edge: tuple[str, str]) -> dict[str, str]:
    """Classify every plan node as 'before', 'after', or 'independent'.

    Rules:
      - Trace path from open edge parent up to goal; all nodes on this
        path are 'after'. At each subgoal on this path, its support
        child is also 'after'.
      - Descendants below the open edge child are 'before'.
      - Independent nodes at the same DAG depth as the open edge parent
        are 'after'.
      - Everything else is 'independent'.
    """
    g = plan.g
    parent_id, child_id = open_edge

    classification = {}

    # --- 1. Find 'after' nodes: path from parent to goal + support branches ---
    after_set = set()

    # Walk from parent_id up to goal (following predecessors)
    path_to_goal = []
    current = parent_id
    while current is not None:
        path_to_goal.append(current)
        after_set.add(current)
        preds = list(g.predecessors(current))
        current = preds[0] if preds else None

    # At each subgoal on this path, mark its support child as 'after' too
    for nid in path_to_goal:
        if g.nodes[nid]['ntype'] == 'subgoal':
            for _, child in g.out_edges(nid):
                child_type = g.nodes[child]['ntype']
                if child_type == 'support':
                    after_set.add(child)

    # Also walk down from support nodes that are 'after' to collect
    # their subtrees as 'after' (support chains, cross-branch edges)
    after_support_queue = [n for n in after_set
                           if g.nodes[n]['ntype'] == 'support']
    visited_support = set(after_support_queue)
    while after_support_queue:
        node = after_support_queue.pop()
        for _, succ in g.out_edges(node):
            if succ not in after_set:
                succ_type = g.nodes[succ]['ntype']
                if succ_type in ('support', 'subgoal'):
                    after_set.add(succ)
                    if succ not in visited_support:
                        visited_support.add(succ)
                        after_support_queue.append(succ)
                    # If subgoal, add its children too
                    if succ_type == 'subgoal':
                        for _, sg_child in g.out_edges(succ):
                            after_set.add(sg_child)

    # --- 2. Find 'before' nodes: descendants of child_id ---
    before_set = set()

    def _collect_descendants(node):
        before_set.add(node)
        for _, succ in g.out_edges(node):
            if succ not in before_set and succ not in after_set:
                _collect_descendants(succ)

    _collect_descendants(child_id)

    # --- 3. Independent nodes at same depth as parent -> 'after' ---
    # Compute depth of parent_id (distance from goal to parent)
    parent_depth = len(path_to_goal) - 1  # goal is at depth 0

    # BFS from goal to compute depths
    goal_node = path_to_goal[-1]
    depths = {goal_node: 0}
    queue = [goal_node]
    while queue:
        node = queue.pop(0)
        for _, succ in g.out_edges(node):
            if succ not in depths:
                depths[succ] = depths[node] + 1
                queue.append(succ)

    for nid in g.nodes():
        if nid in after_set or nid in before_set:
            continue
        node_depth = depths.get(nid, -1)
        if node_depth == parent_depth:
            after_set.add(nid)

    # --- 4. Assign classifications ---
    for nid in g.nodes():
        if nid in after_set:
            classification[nid] = 'after'
        elif nid in before_set:
            classification[nid] = 'before'
        else:
            classification[nid] = 'independent'

    return classification


# ---------------------------------------------------------------------------
# Came-from / goes-to for each node
# ---------------------------------------------------------------------------

def compute_edge_encoding(plan) -> dict[str, dict]:
    """For each node, compute came_from and goes_to positions.

    came_from: the position of the child node in the movement chain
               (where the robot was before arriving here). -1,-1 for leaves.
    goes_to:   the position of the parent node in the movement chain
               (where the robot is heading). -1,-1 if terminal.
    """
    g = plan.g
    result = {}

    for nid, data in g.nodes(data=True):
        ntype = data['ntype']

        if ntype == 'leaf':
            # Leaf: came_from = nowhere, goes_to = parent position
            goes_to = (-1.0, -1.0)
            for pred in g.predecessors(nid):
                pred_data = g.nodes[pred]
                if pred_data['ntype'] in ('bottleneck', 'support'):
                    goes_to = (float(pred_data['pos'][0]),
                               float(pred_data['pos'][1]))
                    break
            result[nid] = {
                'came_from': (-1.0, -1.0),
                'goes_to': goes_to,
            }

        elif ntype == 'bottleneck':
            # came_from: child position (leaf or deeper bottleneck)
            came_from = (-1.0, -1.0)
            for _, child in g.out_edges(nid):
                child_data = g.nodes[child]
                if child_data['ntype'] == 'leaf':
                    came_from = (float(child_data['pos'][0]),
                                 float(child_data['pos'][1]))
                elif child_data['ntype'] == 'subgoal':
                    # Find the bottleneck child of this subgoal
                    for _, sg_child in g.out_edges(child):
                        if g.nodes[sg_child]['ntype'] == 'bottleneck':
                            came_from = (float(g.nodes[sg_child]['pos'][0]),
                                         float(g.nodes[sg_child]['pos'][1]))
                            break
                break

            # goes_to: parent position (goal or higher bottleneck via subgoal)
            goes_to = (-1.0, -1.0)
            for pred in g.predecessors(nid):
                pred_data = g.nodes[pred]
                if pred_data['ntype'] == 'subgoal':
                    for sg_pred in g.predecessors(pred):
                        sg_pred_data = g.nodes[sg_pred]
                        if sg_pred_data['ntype'] in ('goal', 'bottleneck',
                                                     'support'):
                            goes_to = (float(sg_pred_data['pos'][0]),
                                       float(sg_pred_data['pos'][1]))
                            break
                    break

            result[nid] = {
                'came_from': came_from,
                'goes_to': goes_to,
            }

        elif ntype == 'support':
            # came_from: child position (leaf or deeper support)
            came_from = (-1.0, -1.0)
            for _, child in g.out_edges(nid):
                child_data = g.nodes[child]
                if child_data['ntype'] == 'leaf':
                    came_from = (float(child_data['pos'][0]),
                                 float(child_data['pos'][1]))
                elif child_data['ntype'] == 'support':
                    came_from = (float(child_data['pos'][0]),
                                 float(child_data['pos'][1]))
                elif child_data['ntype'] == 'subgoal':
                    for _, sg_child in g.out_edges(child):
                        if g.nodes[sg_child]['ntype'] == 'support':
                            came_from = (float(g.nodes[sg_child]['pos'][0]),
                                         float(g.nodes[sg_child]['pos'][1]))
                            break
                break  # take first child

            # goes_to: parent position
            goes_to = (-1.0, -1.0)
            for pred in g.predecessors(nid):
                pred_data = g.nodes[pred]
                if pred_data['ntype'] == 'support':
                    goes_to = (float(pred_data['pos'][0]),
                               float(pred_data['pos'][1]))
                elif pred_data['ntype'] == 'subgoal':
                    for sg_pred in g.predecessors(pred):
                        sg_pred_data = g.nodes[sg_pred]
                        if sg_pred_data['ntype'] in ('goal', 'bottleneck',
                                                     'support'):
                            goes_to = (float(sg_pred_data['pos'][0]),
                                       float(sg_pred_data['pos'][1]))
                            break
                break

            result[nid] = {
                'came_from': came_from,
                'goes_to': goes_to,
            }

        elif ntype == 'goal':
            result[nid] = {
                'came_from': (-1.0, -1.0),
                'goes_to': (-1.0, -1.0),
            }

        elif ntype == 'subgoal':
            result[nid] = {
                'came_from': (-1.0, -1.0),
                'goes_to': (-1.0, -1.0),
            }

    return result


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(grid_env, state, plan, open_edge: tuple[str, str]
                     ) -> np.ndarray:
    """Build per-cell feature matrix (256, FEATURE_DIM).

    Encodes board, robot roles, and partial plan state into grid cells.
    Robot colors are NOT encoded — only target/helper distinction.
    """
    g = plan.g
    features = np.zeros((NUM_CELLS, FEATURE_DIM), dtype=np.float32)
    grid_data = grid_env.grid_data

    parent_id, child_id = open_edge

    # --- Board features (dims 0–6) ---
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            idx = y * GRID_SIZE + x
            cell = grid_data[idx] if grid_data and idx < len(grid_data) else 'X'
            features[idx, 0] = x
            features[idx, 1] = y
            features[idx, 2] = float('N' in cell or y == 0)
            features[idx, 3] = float('S' in cell or y == GRID_SIZE - 1)
            features[idx, 4] = float('W' in cell or x == 0)
            features[idx, 5] = float('E' in cell or x == GRID_SIZE - 1)

    # Goal
    gx, gy = state.target
    features[pos_to_index(gx, gy), 6] = 1.0

    # --- Robot identification (dims 7–8) ---
    # Determine the "target robot" for this open edge (by color/role)
    parent_data = g.nodes[parent_id]
    parent_type = parent_data['ntype']

    if parent_type == 'goal':
        moving_robot = state.target_robot
    else:
        moving_robot = parent_data.get('robot', state.target_robot)

    # All robots at their INITIAL board positions (from state)
    all_robots = [state.target_robot] + list(state.helpers)

    # Find the moving robot's initial position (not the plan node position)
    moving_initial_pos = moving_robot.position  # fallback
    for robot in all_robots:
        if robot.color == moving_robot.color:
            moving_initial_pos = robot.position
            break

    # Mark target robot at its initial board position
    tx, ty = moving_initial_pos
    features[pos_to_index(int(tx), int(ty)), 7] = 1.0

    # Mark helper positions (all robots that are not the moving robot)
    seen = set()
    for robot in all_robots:
        pos_key = (robot.position, robot.color)
        if pos_key in seen:
            continue
        seen.add(pos_key)
        if robot.color != moving_robot.color:
            hx, hy = robot.position
            features[pos_to_index(int(hx), int(hy)), 8] = 1.0

    # --- Temporal classification (dims 9–10) ---
    temporal = classify_temporal(plan, open_edge)

    # --- Open edge markers (dims 11–12) ---
    child_data = g.nodes[child_id]

    # target_start = child node position
    if 'pos' in child_data:
        cx, cy = child_data['pos']
        features[pos_to_index(int(cx), int(cy)), 11] = 1.0

    # target_end = parent node position
    if 'pos' in parent_data:
        px, py = parent_data['pos']
        features[pos_to_index(int(px), int(py)), 12] = 1.0

    # --- Plan node features (dims 13–18) ---
    edge_encoding = compute_edge_encoding(plan)

    for nid, data in g.nodes(data=True):
        ntype = data['ntype']
        if 'pos' not in data:
            continue  # subgoal nodes have no position

        pos = data['pos']
        idx = pos_to_index(int(pos[0]), int(pos[1]))

        # Temporal (dims 9–10)
        # When multiple nodes share a cell, 'after' takes priority
        # (committed plan structure above the open edge)
        t = temporal.get(nid, 'independent')
        if t == 'before':
            if features[idx, 10] == 0:  # don't overwrite 'after'
                features[idx, 9] = 1.0
        elif t == 'after':
            features[idx, 9] = 0.0  # clear any 'before' from overlap
            features[idx, 10] = 1.0

        # Node type (dims 13–14)
        if ntype == 'bottleneck':
            features[idx, 13] = 1.0
        elif ntype in ('support', 'leaf'):
            features[idx, 14] = 1.0

        # Edge encoding (dims 15–18)
        enc = edge_encoding.get(nid)
        if enc:
            features[idx, 15] = enc['came_from'][0]
            features[idx, 16] = enc['came_from'][1]
            features[idx, 17] = enc['goes_to'][0]
            features[idx, 18] = enc['goes_to'][1]

    return features


# ---------------------------------------------------------------------------
# Target extraction
# ---------------------------------------------------------------------------

def extract_target(plan, state, subgoal_id: str, open_edge: tuple[str, str]
                   ) -> tuple[int, int, int]:
    """Extract the 3 target tokens for a subgoal being added.

    Returns (bn_cell_index, sp_cell_index, robot_token) where robot_token
    is either TOKEN_TARGET or the cell index of the helper robot's
    current (initial) position.
    """
    g = plan.g
    parent_id, _ = open_edge

    # Find bottleneck and support children of the subgoal
    bn_data = sp_data = None
    for _, child in g.out_edges(subgoal_id):
        child_data = g.nodes[child]
        if child_data['ntype'] == 'bottleneck':
            bn_data = child_data
        elif child_data['ntype'] == 'support':
            sp_data = child_data

    if bn_data is None or sp_data is None:
        raise ValueError(f"Subgoal {subgoal_id} missing bottleneck or support")

    bn_idx = pos_to_index(int(bn_data['pos'][0]), int(bn_data['pos'][1]))
    sp_idx = pos_to_index(int(sp_data['pos'][0]), int(sp_data['pos'][1]))

    # Determine robot token
    parent_data = g.nodes[parent_id]
    parent_type = parent_data['ntype']
    if parent_type == 'goal':
        moving_robot = state.target_robot
    else:
        moving_robot = parent_data.get('robot', state.target_robot)

    support_robot = sp_data['robot']
    if support_robot.color == moving_robot.color:
        robot_token = TOKEN_TARGET
    else:
        # Identify helper by its initial (leaf) cell position
        # Find the leaf node for this robot in the complete plan
        robot_token = _find_robot_initial_pos_index(g, support_robot)

    return bn_idx, sp_idx, robot_token


def _find_robot_initial_pos_index(g, robot) -> int:
    """Find the cell index of a robot's initial position (leaf node)."""
    for nid, data in g.nodes(data=True):
        if (data['ntype'] == 'leaf' and
                data.get('robot') is not None and
                data['robot'].color == robot.color):
            pos = data['pos']
            return pos_to_index(int(pos[0]), int(pos[1]))
    # Fallback: use the robot's position attribute directly
    return pos_to_index(int(robot.position[0]), int(robot.position[1]))


# ---------------------------------------------------------------------------
# Training data generation from complete plans
# ---------------------------------------------------------------------------

def generate_training_examples(grid_env, state, complete_plan
                               ) -> list[dict]:
    """Generate training examples by peeling back subgoals from a complete plan.

    For a plan with K subgoals, generates K examples. Each example is created
    by removing one subgoal (and restoring the open edge it resolved),
    then using the removed subgoal as the supervision target.

    Returns list of dicts with keys:
        'features': np.ndarray (256, FEATURE_DIM)
        'target':   tuple (bn_idx, sp_idx, robot_token)
        'open_edge': tuple (parent_id, child_id)
        'subgoal_id': str
        'robot_colors': dict with robot color info for visualization
    """
    g = complete_plan.g

    # Find all subgoal nodes
    subgoal_ids = [nid for nid, data in g.nodes(data=True)
                   if data['ntype'] == 'subgoal']

    # Collect robot color metadata for visualization
    robot_colors = {}
    for robot in [state.target_robot] + list(state.helpers):
        idx = pos_to_index(int(robot.position[0]), int(robot.position[1]))
        robot_colors[idx] = robot.color

    examples = []

    for sg_id in subgoal_ids:
        # Find the subgoal's parent and its bottleneck/support children
        sg_parent = None
        for pred in g.predecessors(sg_id):
            sg_parent = pred
            break

        bn_id = sp_id = None
        for _, child in g.out_edges(sg_id):
            ntype = g.nodes[child]['ntype']
            if ntype == 'bottleneck':
                bn_id = child
            elif ntype == 'support':
                sp_id = child

        if sg_parent is None or bn_id is None or sp_id is None:
            continue

        # Find the leaf/terminal descendant of the bottleneck branch
        bn_leaf = _find_deepest_leaf_in_branch(g, bn_id)
        if bn_leaf is None:
            continue

        # Create a partial plan by removing this subgoal
        partial = _remove_subgoal(complete_plan, sg_id, sg_parent,
                                  bn_id, sp_id, bn_leaf)

        open_edge = (sg_parent, bn_leaf)

        # Extract features from the partial plan
        features = extract_features(grid_env, state, partial, open_edge)

        # Extract target from the complete plan
        target = extract_target(complete_plan, state, sg_id, open_edge)

        examples.append({
            'features': features,
            'target': target,
            'open_edge': open_edge,
            'subgoal_id': sg_id,
            'robot_colors': robot_colors,
        })

    return examples


def _find_deepest_leaf_in_branch(g, bottleneck_id: str) -> str | None:
    """Find the deepest leaf reachable from a bottleneck via the bn chain."""
    current = bottleneck_id
    while True:
        found_next = False
        for _, child in g.out_edges(current):
            child_type = g.nodes[child]['ntype']
            if child_type == 'leaf':
                return child
            elif child_type == 'subgoal':
                for _, sg_child in g.out_edges(child):
                    if g.nodes[sg_child]['ntype'] == 'bottleneck':
                        current = sg_child
                        found_next = True
                        break
                if found_next:
                    break
        if not found_next:
            return None


def _remove_subgoal(complete_plan, sg_id, sg_parent,
                    bn_id, sp_id, bn_leaf):
    """Create a partial plan by removing a subgoal and restoring open edge.

    Removes: sg_id, bn_id, sp_id, and the entire support subtree.
    Also removes intermediate subgoals/bottlenecks in the bn chain.
    Reconnects: sg_parent -> bn_leaf with status='open'.
    """
    from partial_plan import PartialPlan

    g = complete_plan.g
    partial = PartialPlan()

    # Nodes that must be kept (ancestors of the subgoal + the reconnect leaf)
    keep_nodes = {bn_leaf}
    current = sg_parent
    while current is not None:
        keep_nodes.add(current)
        preds = list(g.predecessors(current))
        current = preds[0] if preds else None

    # Collect nodes to remove
    remove_nodes = {sg_id, bn_id, sp_id}
    _collect_support_subtree(g, sp_id, remove_nodes)
    _collect_bn_chain(g, bn_id, bn_leaf, remove_nodes)

    # Never remove protected nodes (cross-branch edges can reach ancestors)
    remove_nodes -= keep_nodes

    # Copy all nodes except removed ones
    for nid, data in g.nodes(data=True):
        if nid not in remove_nodes:
            partial.g.add_node(nid, **data)

    # Copy all edges except those involving removed nodes
    for u, v, data in g.edges(data=True):
        if u not in remove_nodes and v not in remove_nodes:
            partial.g.add_edge(u, v, **data)

    # Add the open edge
    partial.g.add_edge(sg_parent, bn_leaf, status='open', cost=None)

    return partial


def _collect_support_subtree(g, node_id, remove_set):
    """Collect all nodes in the support subtree for removal."""
    for _, child in g.out_edges(node_id):
        if child not in remove_set:
            remove_set.add(child)
            _collect_support_subtree(g, child, remove_set)


def _collect_bn_chain(g, bn_id, bn_leaf, remove_set):
    """Collect intermediate nodes in bottleneck chain between bn_id and bn_leaf."""
    current = bn_id
    while True:
        found = False
        for _, child in g.out_edges(current):
            child_type = g.nodes[child]['ntype']
            if child_type == 'leaf' and child == bn_leaf:
                return
            elif child_type == 'subgoal':
                for _, sg_child in g.out_edges(child):
                    if g.nodes[sg_child]['ntype'] == 'bottleneck':
                        if _is_ancestor_of(g, sg_child, bn_leaf):
                            remove_set.add(child)
                            remove_set.add(sg_child)
                            # Remove support sibling and its subtree
                            for _, sib in g.out_edges(child):
                                if g.nodes[sib]['ntype'] == 'support':
                                    remove_set.add(sib)
                                    _collect_support_subtree(g, sib, remove_set)
                            current = sg_child
                            found = True
                            break
                if found:
                    break
        if not found:
            return


def _is_ancestor_of(g, ancestor, descendant) -> bool:
    """Check if descendant is reachable from ancestor."""
    if ancestor == descendant:
        return True
    for _, child in g.out_edges(ancestor):
        if _is_ancestor_of(g, child, descendant):
            return True
    return False
