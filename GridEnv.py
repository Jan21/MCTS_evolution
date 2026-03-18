from collections import defaultdict
from dataclasses import dataclass
import pickle
from pathlib import Path

import networkx as nx
import ricochet_solver_py

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
    def __init__(self, grid_graph: nx.DiGraph, independent_paths: dict, all_paths: dict, grid_data=None, grid_size=(16, 16)):
        self.G = grid_graph
        self.reachability_matrix = independent_paths
        self.relaxed_reachability_matrix = all_paths
        self.grid_data = grid_data
        self.grid_size = grid_size

        self.precomputed_final_components = {}
        independent_G = self.remove_dependent_edges(self.G.copy())
        for goal in self.G.nodes():
            self.precomputed_final_components[goal] = set(nx.ancestors(independent_G, goal))

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

        inst = env["instances"][instance_index]

        parsed_helpers = [Robot_at(r['position'], r['color']) for r in inst['helper_robots']]
        parsed_target_robot = Robot_at(inst['target_robot']['position'], inst['target_robot']['color'])

        state = State(target=inst["target"], target_robot=parsed_target_robot, helpers=parsed_helpers)
        grid_env = cls(
            grid_graph=env["grid_graph"],
            independent_paths=env.get("independent_paths", {}),
            all_paths=env.get("all_paths", {}),
            grid_data=env.get("grid_data", None)
        )
        return grid_env, state

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
                # score = self.subgoal_score(subgoal)
                score = 1
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

    def solve(self, state: "State") -> dict:
        """Solve the puzzle using the Rust BFS solver.

        Returns dict with "moves", "move_count", "start_positions", "end_positions".
        """
        size = self.grid_size[0]
        grid_data = self.grid_data

        walls_right_set = set()
        walls_down_set = set()
        for idx, cell in enumerate(grid_data):
            col = idx % size
            row = idx // size
            if 'E' in cell:
                walls_right_set.add((col, row))
            if 'S' in cell:
                walls_down_set.add((col, row))
            if 'W' in cell and col > 0:
                walls_right_set.add((col - 1, row))
            if 'N' in cell and row > 0:
                walls_down_set.add((col, row - 1))
        walls_right = list(walls_right_set)
        walls_down = list(walls_down_set)

        # Build robot_positions as [red, blue, green, yellow]
        color_order = {"red": 0, "blue": 1, "green": 2, "yellow": 3}
        robot_positions = [None] * 4
        for robot in state.all_robots:
            idx = color_order[robot.color.lower()]
            robot_positions[idx] = robot.position
        # Fill missing robots with an out-of-the-way position
        for i in range(4):
            if robot_positions[i] is None:
                robot_positions[i] = (size - 1, size - 1)

        target_robot = state.target_robot.color.lower()
        target_pos = state.target

        return ricochet_solver_py.solve_with_walls(
            walls_right=walls_right,
            walls_down=walls_down,
            board_size=size,
            target_pos=target_pos,
            target_robot=target_robot,
            robot_positions=robot_positions,
        )

    def _get_wall_sets(self):
        """Return (walls_right, walls_down) as sets of (x, y) tuples."""
        size = self.grid_size[0]
        walls_right = set()
        walls_down = set()
        for idx, cell in enumerate(self.grid_data):
            col = idx % size
            row = idx // size
            if 'E' in cell:
                walls_right.add((col, row))
            if 'S' in cell:
                walls_down.add((col, row))
            if 'W' in cell and col > 0:
                walls_right.add((col - 1, row))
            if 'N' in cell and row > 0:
                walls_down.add((col, row - 1))
        return walls_right, walls_down

    def simulate_slide(self, positions: dict, robot_color: str, direction: str):
        """Simulate a robot sliding in direction. Returns (new_pos, blocker).

        positions: {color: (x, y), ...}
        blocker: color string of the robot that stopped it, or "wall".
        """
        walls_right, walls_down = self._get_wall_sets()
        size = self.grid_size[0]
        x, y = positions[robot_color]

        occupied = {}
        for color, pos in positions.items():
            if color != robot_color:
                occupied[pos] = color

        d = direction.lower()
        # deltas and wall-check per direction
        # walls_right(x,y) = wall on right edge of (x,y), between (x,y) and (x+1,y)
        # walls_down(x,y) = wall on bottom edge of (x,y), between (x,y) and (x,y+1)
        while True:
            can_leave = True
            if d == 'up':
                if y == 0:
                    can_leave = False
                elif (x, y - 1) in walls_down:  # wall between (x,y-1) and (x,y)
                    can_leave = False
                nx_, ny_ = x, y - 1
            elif d == 'down':
                if y == size - 1:
                    can_leave = False
                elif (x, y) in walls_down:  # wall between (x,y) and (x,y+1)
                    can_leave = False
                nx_, ny_ = x, y + 1
            elif d == 'left':
                if x == 0:
                    can_leave = False
                elif (x - 1, y) in walls_right:  # wall between (x-1,y) and (x,y)
                    can_leave = False
                nx_, ny_ = x - 1, y
            elif d == 'right':
                if x == size - 1:
                    can_leave = False
                elif (x, y) in walls_right:  # wall between (x,y) and (x+1,y)
                    can_leave = False
                nx_, ny_ = x + 1, y

            if not can_leave:
                return (x, y), "wall"
            if (nx_, ny_) in occupied:
                return (x, y), occupied[(nx_, ny_)]
            x, y = nx_, ny_

