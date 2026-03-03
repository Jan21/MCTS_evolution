import pickle
from collections import deque

with open('env/env_0.pkl', 'rb') as f:
    _env = pickle.load(f)
G = _env['grid_graph']


def _compute_final_component(goal):
    """BFS on reverse of G using only weight=1 edges from goal."""
    visited = {goal}
    queue = deque([goal])
    while queue:
        node = queue.popleft()
        for u, _, d in G.in_edges(node, data=True):
            if d['weight'] == 1 and u not in visited:
                visited.add(u)
                queue.append(u)
    return visited


def _has_adjacent_wall(pos):
    """True if pos has at least one incoming weight=1 edge."""
    return any(d['weight'] == 1 for _, _, d in G.in_edges(pos, data=True))


def _collect_bottleneck_support_pairs(final_component):
    """Collect (bottleneck, support_pos) pairs crossing into the final component."""
    pairs = set()
    for u, v, d in G.edges(data=True):
        if d.get('weight') == 100 and 'dependent' in d:
            if v in final_component and u not in final_component:
                support_pos = d['dependent']
                if _has_adjacent_wall(support_pos):
                    pairs.add((v, support_pos))
    return pairs


def _mock_score(state, support, bottleneck):
    return 0.0


def propose_subgoal_states(state, support):
    """Generate candidate subgoal states for an open segment.

    Returns list of ((bottleneck_pos, support), score).
    """
    goal = state['target']
    final_component = _compute_final_component(goal)

    if support is not None:
        support_pos, robot = support
        if not _has_adjacent_wall(support_pos):
            return []
        results = []
        for (bottleneck, s_pos) in _collect_bottleneck_support_pairs(final_component):
            if s_pos == support_pos:
                score = _mock_score(state, support, bottleneck)
                results.append(((bottleneck, support), score))
        return results
    else:
        pairs = _collect_bottleneck_support_pairs(final_component)
        results = []
        for (bottleneck, support_pos) in pairs:
            for robot in state['helper_robots']:
                proposed_support = (support_pos, robot)
                score = _mock_score(state, proposed_support, bottleneck)
                results.append(((bottleneck, proposed_support), score))
        return results
