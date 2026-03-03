import pickle
from collections import deque
from dataclasses import dataclass


with open('env/env_0.pkl', 'rb') as f:
    _env = pickle.load(f)
G = _env['grid_graph']


@dataclass
class Robot_at:
    position: tuple[float, float]
    color: str

@dataclass
class State:
    goal: tuple[float, float]
    target_robot: Robot_at
    helpers: list[Robot_at]

@dataclass
class Subgoal:
    bottleneck: Robot_at
    support: Robot_at

# TODO asi bych to udelal tak, ze bych si nejdrive vyextrahoval graph, 
# ktery ma v sobe jen ty independent edges a pak bych pouzil nx.ancestors(G, source) a tim ziskas ten final component
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

# TODO na později, udělat více sofistikovaný check
def _has_adjacent_wall(pos):
    """True if pos has at least one incoming weight=1 edge."""
    return any(d['weight'] == 1 for _, _, d in G.in_edges(pos, data=True))


def _collect_bottleneck_support_pairs(final_component):
    """Collect (bottleneck, support_pos) pairs crossing into the final component."""
    pairs = set()
    # TODO na později, tady by se to možná trochu dalo urychlit tim, že se bude iterovat
    # přes incomming edges do každěho prvku v final_component
    for u, v, d in G.edges(data=True): 
        if d.get('weight') == 100 and 'dependent' in d:
            if v in final_component and u not in final_component:
                support_pos = d['dependent']
                if _has_adjacent_wall(support_pos):
                    pairs.add((v, support_pos))
    return pairs


def _mock_score(state, support, bottleneck):
    return 0.0

# TODO asi bych použil jednoduchou datovou strukturu, která bude reprezentovat pozici robota. Bude říkát kde je a jaký robot to je

def propose_subgoal_states(state, support):
    """Generate candidate subgoal states for an open segment.

    Returns list of ((bottleneck_pos, support), score).
    """
    goal = state['target']
    # TODO, toto nebude fungovat pro ten case, kdyz je tam ten support
    # to jedine co se liší u toho casu, kdy je tady ten support je ten final component, zbytek je stejný pro oba casy.
    # tzn. to co děláš v tom if else bude stejné pro oba casy, jakmile budeš mít ten final_component
    # ten final component ziskas tak, ze hrany ktere byly dependent a byly podminene pozici toho supportu, 
    # tak ty uděláš nepodmíněné a pak si vytáhneš všechny předky toho cile v tomto novém grafu.
    # asi bych ten graf dal jako argument do te funkce
    final_component = _compute_final_component(goal)

    if support is not None:
        support_pos, robot = support
        if not _has_adjacent_wall(support_pos):
            return []
        results = []
        for (bottleneck, s_pos) in _collect_bottleneck_support_pairs(final_component):
            if s_pos == support_pos:
                score = _mock_score(state, support, bottleneck)
                # TODO tady by se mela ulozit instance Subgoal + score
                results.append(((bottleneck, support), score))
        return results
    else:
        pairs = _collect_bottleneck_support_pairs(final_component)
        results = []
        for (bottleneck, support_pos) in pairs:
            for robot in state['helper_robots']:
                proposed_support = (support_pos, robot)
                score = _mock_score(state, proposed_support, bottleneck)
                # TODO tady by se mela ulozit instance Subgoal + score
                results.append(((bottleneck, proposed_support), score))
        return results
