from dataclasses import dataclass
import pickle
from pathlib import Path

import networkx as nx

ENV_DIR = Path(__file__).resolve().parent / "environments"

@dataclass
class Robot_at:
    position: tuple[float, float]
    color: str

@dataclass
class State:
    """A single puzzle instance: which robot must reach which cell."""

    target: tuple[int, int]
    target_robot: Robot_at
    helpers: list[Robot_at]

    @property
    def all_robots(self) -> list[Robot_at]:
        return [self.target_robot] + self.helpers

@dataclass
class Subgoal:
    bottleneck: Robot_at
    support: Robot_at
    goal_pos: tuple[int, int]
    target_robot: Robot_at
    helper: Robot_at


class GridEnv:
    def __init__(self, grid_graph: nx.DiGraph, independent_paths: dict, all_paths: dict):
        self.G = grid_graph
        self.reachability_matrix = independent_paths
        self.relaxed_reachability_matrix = all_paths

        self.precomputed_final_components = {}
        for goal in self.G.nodes():
            self.precomputed_final_components[goal] = set(nx.ancestors(self.G, goal))

    @classmethod
    def from_env(cls, env_index: int, instance_index: int = 0, env_dir: Path | str = ENV_DIR):
        path = Path(env_dir) / f"env_{env_index}.pkl"
        with open(path, "rb") as f:
            env = pickle.load(f)

        inst = env["instances"][instance_index]
        state = State(target=inst["target"], target_robot=inst["target_robot"], helpers=inst["helper_robots"])
        grid_env = cls(
            grid_graph=env["grid_graph"],
            independent_paths=env.get("independent_paths", {}),
            all_paths=env.get("all_paths", {}),
        )
        return grid_env, state

    # TODO na později, udělat více sofistikovaný check
    def _has_adjacent_wall(self, pos):
        """True if pos has at least one incoming weight=1 edge."""
        return any(d['weight'] == 1 for _, _, d in self.G.in_edges(pos, data=True))

    def _collect_bottleneck_support_pairs(self, final_component):
        """Collect (bottleneck, support_pos) pairs crossing into the final component."""
        pairs = set()
        # TODO na později, tady by se to možná trochu dalo urychlit tim, že se bude iterovat
        # přes incomming edges do každěho prvku v final_component
        for u, v, d in self.G.edges(data=True):
            if d.get('weight') == 100 and 'dependent' in d:
                if v in final_component and u not in final_component:
                    support_pos = d['dependent']
                    if self._has_adjacent_wall(support_pos):
                        pairs.add((v, support_pos))
        return pairs

    def subgoal_score(self, subgoal: Subgoal):
        bottleneck_pos = subgoal.bottleneck.position
        support_pos = subgoal.support.position
        goal_pos = subgoal.goal_pos
        target_pos = subgoal.target_robot.position
        helper_pos = subgoal.helper.position

        bottleneck_to_goal_score = self.reachability_matrix[(bottleneck_pos, goal_pos)]
        target_to_bottleneck_score = self.compute_relaxed_shortest_path_length(target_pos, bottleneck_pos, support_pos)
        helper_to_support_score = self.compute_relaxed_shortest_path_length(helper_pos, support_pos)
        return bottleneck_to_goal_score + target_to_bottleneck_score + helper_to_support_score
        

    def propose_subgoal_states(self, state: State, support_robot: Robot_at):
        """Generate candidate subgoal states for an open segment.

        Returns list of (Subgoal, score).
        """
        goal = state['goal']
        target_robot = state['target_robot']
        target_color = target_robot['color']

        if support_robot is not None:
            support_pos, robot = support_robot['position'], support_robot['color']
            extended_G = self.get_extended_graph(support_pos)
            final_component = set(nx.ancestors(extended_G, goal))
        else:
            final_component = self.precomputed_final_components[goal]
        pairs = self._collect_bottleneck_support_pairs(final_component)
        results = []
        for (bottleneck_pos, support_pos) in pairs:
            for helper_robot in state['helpers']:
                helper_color = helper_robot['color']
                new_support_robot = Robot_at(position=support_pos, color=helper_color)
                bottleneck_robot = Robot_at(position=bottleneck_pos, color=target_color)
                subgoal = Subgoal(
                    bottleneck=bottleneck_robot, 
                    support=new_support_robot, 
                    goal_pos=goal, 
                    target_robot=target_robot,
                    helper=helper_robot)
                score = self.subgoal_score(subgoal)
                results.append((subgoal, score))
        return results

    def get_extended_graph(self, support_pos, bottleneck_pos, remove_dependent: bool = True):
        new_G = self.G.copy()
        # Compute the main direction vector from support_pos to bottleneck_pos
        main_vec = (bottleneck_pos[0] - support_pos[0], bottleneck_pos[1] - support_pos[1])

        def direction(u, v):
            return (v[0] - u[0], v[1] - u[1])

        edges_to_remove_local = []
        edges_to_make_independent = []

        for u, v, data in new_G.in_edges(bottleneck_pos, data=True):
            edge_vec = direction(u, v)
            if data.get('dependent') == support_pos:
                if main_vec[0] != 0:
                    if edge_vec[0] == 0:
                        edges_to_remove_local.append((u, v))
                    elif edge_vec[0] * main_vec[0] > 0:
                        edges_to_remove_local.append((u, v))
                    elif edge_vec[0] * main_vec[0] < 0:
                        edges_to_make_independent.append((u, v))
                elif main_vec[1] != 0:
                    if edge_vec[1] == 0:
                        edges_to_remove_local.append((u, v))
                    elif edge_vec[1] * main_vec[1] > 0:
                        edges_to_remove_local.append((u, v))
                    elif edge_vec[1] * main_vec[1] < 0:
                        edges_to_make_independent.append((u, v))

        new_G.remove_edges_from(edges_to_remove_local)
        for u, v in edges_to_make_independent:
            new_G[u][v].pop('dependent')
            new_G[u][v]['weight'] = 1
        if remove_dependent:    
            edges_to_remove = [(u, v) for u, v, data in new_G.edges(data=True) if 'dependent' in data]
            new_G.remove_edges_from(edges_to_remove)
        return new_G

    def compute_exact_shortest_path_length(self, start, end, support_pos=None):
        if support_pos is None:
            return self.reachability_matrix[(start, end)]
        else:
            extended_G = self.get_extended_graph(support_pos, end)
            try:
                return nx.shortest_path_length(extended_G, start, end)
            except nx.NetworkXNoPath:
                return None

    def compute_relaxed_shortest_path_length(self, start, end, support_pos=None):
        if support_pos is None:
            return self.relaxed_reachability_matrix[(start, end)]
        else:
            extended_G = self.get_extended_graph(support_pos, end, remove_dependent=False)
            try:
                return nx.shortest_path_length(extended_G, start, end, weight='weight')
            except nx.NetworkXNoPath:
                return float('inf')

