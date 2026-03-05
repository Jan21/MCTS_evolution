from collections import defaultdict
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

        self._wall_nodes = frozenset(
            node for node in self.G.nodes()
            if any(d['weight'] == 1 for _, _, d in self.G.in_edges(node, data=True))
        )

        grouped = defaultdict(list)
        for u, v, d in self.G.edges(data=True):
            if 'dependent' in d:
                key = (v, d['dependent'])
                grouped[key].append((u, v, d))

        self._dependent_edge_cache = {}
        for (bottleneck_pos, support_pos), edges in grouped.items():
            ext_G = self.get_extended_graph(support_pos, bottleneck_pos)
            ext_G_independent = self.remove_dependent_edges(ext_G)
            final_component = set(nx.ancestors(ext_G_independent, bottleneck_pos))
            del ext_G_independent
            del ext_G
            self._dependent_edge_cache[(bottleneck_pos, support_pos)] = {
                'edges': edges,
                'final_component': final_component,
            }

        self._bottleneck_support_pairs_cache = {}
        for goal in self.G.nodes():
            has_independent_in_edge = any(
                'dependent' not in d for _, _, d in self.G.in_edges(goal, data=True)
            )
            if has_independent_in_edge:
                self._bottleneck_support_pairs_cache[(goal, None)] = (
                    self._collect_bottleneck_support_pairs(self.precomputed_final_components[goal])
                )
        for (bottleneck_pos, support_pos), entry in self._dependent_edge_cache.items():
            self._bottleneck_support_pairs_cache[(bottleneck_pos, support_pos)] = (
                self._collect_bottleneck_support_pairs(entry['final_component'])
            )

    @classmethod
    def from_env(cls, env_index: int, instance_index: int = 0, env_dir: Path | str = ENV_DIR):
        path = Path(env_dir) / f"env_{env_index}.pkl"
        with open(path, "rb") as f:
            env = pickle.load(f)
            GridEnv._process_env_pickle(env)

        inst = env["instances"][instance_index]
        state = State(target=inst["target"], target_robot=inst["target_robot"], helpers=inst["helper_robots"])
        grid_env = cls(
            grid_graph=env["grid_graph"],
            independent_paths=env.get("independent_paths", {}),
            all_paths=env.get("all_paths", {}),
        )
        grid_env._grid_data = env.get("grid_data")
        return grid_env, state

    def visualize(self, state: "State", subgoals: list | None = None):
        from visualization.render import open_in_browser
        open_in_browser(self, state, subgoals)

    def _has_adjacent_wall(self, pos):
        """True if pos has at least one incoming weight=1 edge."""
        return pos in self._wall_nodes

    def remove_dependent_edges(self, G: nx.DiGraph):
            edges_to_remove = [(u, v) for u, v, data in G.edges(data=True) if 'dependent' in data]
            G.remove_edges_from(edges_to_remove)
            return G

    def _collect_bottleneck_support_pairs(self, final_component):
        """Collect (bottleneck, support_pos) pairs crossing into the final component."""
        pairs = set()
        for node in final_component:
            for u, v, d in self.G.in_edges(node, data=True):
                if u not in final_component and d.get('weight') == 100 and 'dependent' in d:
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
        

    def propose_subgoal_states(self, state: State, support_robot: Robot_at = None):
        """Generate candidate subgoal states for an open segment.

        Returns list of (Subgoal, score).
        """
        goal = state.target
        target_robot = state.target_robot
        target_color = target_robot.color

        if support_robot is not None:
            support_pos, robot = support_robot.position, support_robot.color
            cache_key = (goal, support_pos)
        else:
            support_pos = None
            cache_key = (goal, None)

        cached_pairs = self._bottleneck_support_pairs_cache.get(cache_key)
        if cached_pairs is not None:
            pairs = cached_pairs
        else:
            assert False, "No cached pairs found"
            extended_G = self.get_extended_graph(support_pos, bottleneck_pos=goal)
            final_component = set(nx.ancestors(extended_G, goal))
            pairs = self._collect_bottleneck_support_pairs(final_component)
        results = []
        for (bottleneck_pos, support_pos) in pairs:
            for helper_robot in state.helpers:
                helper_color = helper_robot.color
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

    def get_extended_graph(self, support_pos, bottleneck_pos):
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
        return new_G

    def compute_exact_shortest_path_length(self, start, end, support_pos=None):
        if support_pos is None:
            return self.reachability_matrix[(start, end)]
        else:
            cache_key = (end, support_pos)
            dep_entry =self._dependent_edge_cache.get(cache_key)
            if dep_entry is None:
                return None
            dep_edges_to_end = dep_entry['edges']
            lengths = []
            for u, v, d in dep_edges_to_end:
                if self.reachability_matrix[(start, u)] is not None:
                    lengths.append(self.reachability_matrix[(start, u)] + 1)
            if len(lengths) == 0:
                return None
            return min(lengths)

    def compute_relaxed_shortest_path_length(self, start, end, support_pos=None):
        if support_pos is None:
            return self.relaxed_reachability_matrix[(start, end)]
        else:
            cache_key = (end, support_pos)
            dep_entry =self._dependent_edge_cache.get(cache_key)
            if dep_entry is None:
                return None
            dep_edges_to_end = dep_entry['edges']
            lengths = []
            for u, v, d in dep_edges_to_end:
                if self.relaxed_reachability_matrix[(start, u)] is not None:
                    lengths.append(self.relaxed_reachability_matrix[(start, u)] + 1)
            if len(lengths) == 0:
                return None
            return min(lengths)

    def _process_env_pickle(pkl):
        processed_instances = []
        for instance in pkl['instances']:
            processed_helpers = []
            for helper in instance['helper_robots']:
                processed_helpers.append(Robot_at(helper['position'], helper['color']))

            processed_target_robot = Robot_at(instance['target_robot']['position'],
                                              instance['target_robot']['color'])

            processed_instances.append(
                {'helper_robots': processed_helpers,
                 'target_robot': processed_target_robot,
                 'target': instance['target']}
            )

        pkl['instances'] = processed_instances

        G = pkl['grid_graph']

        independent_G = nx.DiGraph()
        independent_G.add_nodes_from(G.nodes())
        independent_G.add_edges_from(
            (u, v, d) for u, v, d in G.edges(data=True) if 'dependent' not in d
        )

        def compute_matrix(graph):
            matrix = defaultdict(lambda: None)
            for source, lengths in nx.all_pairs_dijkstra_path_length(graph, weight='weight'):
                for target, length in lengths.items():
                    matrix[(source, target)] = length
            return matrix

        pkl['independent_paths'] = compute_matrix(independent_G)
        pkl['all_paths'] = compute_matrix(G)
            
