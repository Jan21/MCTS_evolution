from collections import defaultdict
from dataclasses import dataclass
import pickle
from pathlib import Path

import networkx as nx

ENV_DIR = Path(__file__).resolve().parent / "environments"
CACHE_DIR = ENV_DIR / "cache"

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
    def __init__(self, grid_graph: nx.DiGraph, independent_paths: dict, all_paths: dict,
                 max_final_component_distance=None):
        self.G = grid_graph
        self.reachability_matrix = independent_paths
        self.relaxed_reachability_matrix = all_paths
        self.max_final_component_distance = max_final_component_distance

        # Final component = cells that can reach the goal WITHOUT any helper,
        # so it must be computed on the dependent-edge-free graph. Computing it
        # on self.G (with dependent edges) makes nearly every cell an ancestor.
        self.precomputed_final_components = {}
        independent_G = self.remove_dependent_edges(self.G.copy())
        for goal in self.G.nodes():
            self.precomputed_final_components[goal] = self._compute_final_component(
                independent_G, goal)

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
            final_component = self._compute_final_component(
                ext_G_independent, bottleneck_pos,
                is_extended_graph=True, blocker_pos=support_pos)
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
                # Include the goal itself: a bottleneck may stop *directly* at the
                # goal via a support (the goal's own dependent in-edges), not only
                # at cells inside its final component.
                self._bottleneck_support_pairs_cache[(goal, None)] = (
                    self._collect_bottleneck_support_pairs(
                        self.precomputed_final_components[goal] | {goal})
                )
        for (bottleneck_pos, support_pos), entry in self._dependent_edge_cache.items():
            self._bottleneck_support_pairs_cache[(bottleneck_pos, support_pos)] = (
                self._collect_bottleneck_support_pairs(
                    entry['final_component'] | {bottleneck_pos})
            )

    @classmethod
    def from_env(cls, env_index: int, instance_index: int = 0,
                 env_dir: Path | str = ENV_DIR,
                 dependent_edge_weight: float = 2,
                 max_final_component_distance=None):
        env_dir = Path(env_dir)
        cache_dir = env_dir / "cache"
        cache_path = cache_dir / (
            f"env_{env_index}_w{dependent_edge_weight}"
            f"_d{max_final_component_distance}.pkl")

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            return cached["grid_env"], cached["state"]

        path = env_dir / f"env_{env_index}.pkl"
        with open(path, "rb") as f:
            env = pickle.load(f)

        g = env["grid_graph"]

        # Set dependent edge weights to the configured value
        for u, v, d in g.edges(data=True):
            if "dependent" in d:
                d["weight"] = dependent_edge_weight

        # Recompute all_paths with the configured weight
        all_paths = {}
        nodes = sorted(g.nodes())
        lengths = dict(nx.all_pairs_dijkstra_path_length(g, weight="weight"))
        for src in nodes:
            src_lengths = lengths.get(src, {})
            for dst in nodes:
                all_paths[(src, dst)] = src_lengths.get(dst, None)

        inst = env["instances"][instance_index]
        state = State(target=inst["target"], target_robot=inst["target_robot"], helpers=inst["helper_robots"])
        grid_env = cls(
            grid_graph=g,
            independent_paths=env.get("independent_paths", {}),
            all_paths=all_paths,
            max_final_component_distance=max_final_component_distance,
        )
        grid_env.grid_data = env.get("grid_data")

        # Cache for future loads
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"grid_env": grid_env, "state": state}, f)

        return grid_env, state

    def _compute_final_component(self, G: nx.DiGraph, goal,
                                 is_extended_graph=False, blocker_pos=None):
        """Cells from which the goal is reachable without a helper.

        Args:
            G: graph to take ancestors on (already dependent-edge-free).
            goal: target position.
            is_extended_graph: True when G is an extended graph, so distances
                must be measured on G itself rather than the reachability matrix.
            blocker_pos: a support-robot position that acts as a wall. The node
                and every cell beyond it in the same row/column are dropped.

        When self.max_final_component_distance is set, only cells within that
        many moves of the goal are kept (caps the number of proposed subgoals).
        """
        if blocker_pos is not None and blocker_pos in G.nodes():
            G = G.copy()
            G.remove_node(blocker_pos)
            bx, by = blocker_pos
            gx, gy = goal
            nodes_to_remove = []
            # Same row as goal: drop cells past the blocker horizontally.
            if by == gy:
                if bx > gx:
                    nodes_to_remove = [n for n in G.nodes() if n[0] > bx and n[1] == by]
                elif bx < gx:
                    nodes_to_remove = [n for n in G.nodes() if n[0] < bx and n[1] == by]
            # Same column as goal: drop cells past the blocker vertically.
            if bx == gx:
                if by > gy:
                    nodes_to_remove = [n for n in G.nodes() if n[1] > by and n[0] == bx]
                elif by < gy:
                    nodes_to_remove = [n for n in G.nodes() if n[1] < by and n[0] == bx]
            G.remove_nodes_from(nodes_to_remove)

        ancestors = set(nx.ancestors(G, goal))
        if self.max_final_component_distance is None:
            return ancestors
        if is_extended_graph:
            distances = nx.single_source_shortest_path_length(
                G.reverse(), goal, cutoff=self.max_final_component_distance)
            return ancestors & distances.keys()
        return {
            node for node in ancestors
            if self.reachability_matrix.get((node, goal)) is not None
            and self.reachability_matrix[(node, goal)] <= self.max_final_component_distance
        }

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
                if u not in final_component and 'dependent' in d:
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

        INF = 10_000
        if bottleneck_to_goal_score is None:
            bottleneck_to_goal_score = INF
        if target_to_bottleneck_score is None:
            target_to_bottleneck_score = INF
        if helper_to_support_score is None:
            helper_to_support_score = INF

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
            if support_pos is None:
                final_component = self.precomputed_final_components[goal]
            else:
                extended_G = self.get_extended_graph(support_pos, bottleneck_pos=goal)
                extended_G_independent = self.remove_dependent_edges(extended_G)
                final_component = self._compute_final_component(
                    extended_G_independent, goal,
                    is_extended_graph=True, blocker_pos=support_pos)
                del extended_G
                del extended_G_independent
            pairs = self._collect_bottleneck_support_pairs(final_component | {goal})
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

