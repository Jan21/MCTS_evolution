"""Game API for Ricochet Robots environments.

Provides structured classes for loading and accessing environment data:
    Robot: A robot with name and position.
    State: An immutable game instance (target + robots).
    Game: A loaded environment with grid graph, path matrices, and states.
"""

import pickle
from dataclasses import dataclass

from utils import (
    ensure_robot_unpickling,
    list_environment_files,
    compute_independent_paths,
    compute_all_paths,
)


@dataclass(frozen=True)
class Robot:
    """A robot on the grid board."""
    name: str
    pos: tuple  # (x, y)


@dataclass(frozen=True)
class State:
    """An immutable game instance: target position + robot configuration."""
    target: tuple           # (x, y) goal position
    target_robot: Robot
    helper_robots: tuple    # tuple of Robot


class Game:
    """A loaded Ricochet Robots environment.

    Attributes:
        grid_graph: nx.DiGraph with 256 nodes and weighted edges.
        grid_data: List of wall codes (256 entries).
        grid_nodes: Frozenset of (x, y) positions on the grid.
        independent_paths: Dict mapping (src, dst) -> int | None.
        all_paths: Dict mapping (src, dst) -> float.
        states: List of State objects (one per instance).
        graph_idx: Environment index from the pickle.
    """

    def __init__(self, grid_graph, grid_data, independent_paths, all_paths,
                 states, graph_idx=None):
        self.grid_graph = grid_graph
        self.grid_data = grid_data
        self.grid_nodes = frozenset(grid_graph.nodes())
        self.independent_paths = independent_paths
        self.all_paths = all_paths
        self.states = states
        self.graph_idx = graph_idx

    @classmethod
    def from_pickle(cls, pkl_path):
        """Load a Game from an environment pickle file.

        Converts the sibling project's Robot objects into our own Robot
        dataclass and creates State objects for each instance.

        Args:
            pkl_path: Path to env_X.pkl file.

        Returns:
            Game instance.

        Raises:
            FileNotFoundError: If pickle file doesn't exist.
            KeyError: If required keys are missing.
        """
        ensure_robot_unpickling()
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        required_keys = {'graph_idx', 'grid_data', 'grid_graph', 'instances'}
        missing = required_keys - set(data.keys())
        if missing:
            raise KeyError(f"Environment pickle missing keys: {missing}")

        grid_graph = data['grid_graph']

        # Parse instances into State objects
        states = []
        for inst in data['instances']:
            target_robot = _convert_robot(inst['target_robot'])
            helper_robots = tuple(
                _convert_robot(r) for r in inst['helper_robots']
            )
            states.append(State(
                target=inst['target'],
                target_robot=target_robot,
                helper_robots=helper_robots,
            ))

        # Get or compute path matrices
        independent_paths = data.get('independent_paths')
        if independent_paths is None:
            independent_paths = compute_independent_paths(grid_graph)

        all_paths = data.get('all_paths')
        if all_paths is None:
            all_paths = compute_all_paths(grid_graph)

        return cls(
            grid_graph=grid_graph,
            grid_data=data['grid_data'],
            independent_paths=independent_paths,
            all_paths=all_paths,
            states=states,
            graph_idx=data['graph_idx'],
        )

    @classmethod
    def load_many(cls, env_dir=None, num=None):
        """Load multiple Game objects from environment pickle files.

        Args:
            env_dir: Directory containing env_*.pkl files.
            num: If provided, load only the first num files.

        Returns:
            List of Game instances, sorted by environment index.
        """
        paths = list_environment_files(env_dir, num=num)
        return [cls.from_pickle(p) for p in paths]


def _convert_robot(sibling_robot):
    """Convert a sibling project Robot object to our Robot dataclass."""
    return Robot(
        name=sibling_robot.name,
        pos=(sibling_robot.x, sibling_robot.y),
    )
