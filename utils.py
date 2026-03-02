"""Utility functions for MCTS_evolution benchmark framework.

Provides environment loading, path matrix computation, subgoal extraction,
and game-mechanic utilities for Ricochet Robots.
"""

import sys
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx

# Constants matching the sibling project's config.yaml
INDEPENDENT_WEIGHT = 1
DEPENDENT_WEIGHT = 100
BOARD_SIZE = 16

# Sibling project path (needed for Robot class unpickling)
SIBLING_PROJECT = Path(__file__).resolve().parent.parent / "ricochet_robots_simple"


def ensure_robot_unpickling():
    """Add sibling project to sys.path so Robot class can be deserialized.

    The environment pickles contain Robot objects serialized from
    ricochet_robots_simple. We need the Robot class importable.
    """
    path_str = str(SIBLING_PROJECT)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def load_environment(pkl_path):
    """Load an environment pickle file.

    Args:
        pkl_path: Path to env_X.pkl file.

    Returns:
        Dict with keys: graph_idx, grid_data, grid_graph, instances,
        and optionally independent_paths, all_paths.

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

    return data


def list_environment_files(env_dir=None, num=None):
    """List environment pickle files sorted by index.

    Args:
        env_dir: Directory containing env_*.pkl files.
                 Defaults to ./environments/
        num: If provided, return only the first num files.

    Returns:
        List of Path objects, sorted by env index.
    """
    if env_dir is None:
        env_dir = Path(__file__).parent / "environments"
    env_dir = Path(env_dir)
    files = sorted(
        env_dir.glob("env_*.pkl"),
        key=lambda p: int(p.stem.split("_")[1])
    )
    if num is not None:
        files = files[:num]
    return files


def get_independent_paths(env_data):
    """Get or compute independent paths matrix.

    Uses precomputed matrix from pickle if available, otherwise
    computes from scratch using only weight=1 edges (BFS).

    Args:
        env_data: Dict from load_environment().

    Returns:
        Dict mapping (source, target) -> distance (int) or None.
        None means no path exists via independent edges.
    """
    if 'independent_paths' in env_data:
        return env_data['independent_paths']
    return compute_independent_paths(env_data['grid_graph'])


def get_all_paths(env_data):
    """Get or compute weighted all-edges paths matrix.

    Uses precomputed matrix from pickle if available, otherwise
    computes from scratch using Dijkstra with actual edge weights.

    Args:
        env_data: Dict from load_environment().

    Returns:
        Dict mapping (source, target) -> weighted distance (float).
        float('inf') means unreachable.
    """
    if 'all_paths' in env_data:
        return env_data['all_paths']
    return compute_all_paths(env_data['grid_graph'])


def compute_independent_paths(grid_graph):
    """Compute shortest paths using only weight=1 (independent) edges.

    Args:
        grid_graph: nx.DiGraph with weight edge attributes.

    Returns:
        Dict mapping (source, target) -> distance (int) or None.
    """
    nodes = list(grid_graph.nodes())
    w1_edges = [
        (u, v) for u, v, d in grid_graph.edges(data=True)
        if d['weight'] == INDEPENDENT_WEIGHT
    ]
    w1_graph = nx.DiGraph()
    w1_graph.add_nodes_from(nodes)
    w1_graph.add_edges_from(w1_edges)

    independent_paths = {}
    for source in nodes:
        if w1_graph.out_degree(source) == 0 and source not in {
            v for _, v in w1_edges
        }:
            for target in nodes:
                independent_paths[(source, target)] = (
                    0 if source == target else None
                )
            continue
        lengths = nx.single_source_shortest_path_length(w1_graph, source)
        for target in nodes:
            independent_paths[(source, target)] = lengths.get(target, None)
    return independent_paths


def compute_all_paths(grid_graph):
    """Compute shortest weighted paths using all edges (Dijkstra).

    Args:
        grid_graph: nx.DiGraph with weight edge attributes.

    Returns:
        Dict mapping (source, target) -> weighted distance (float).
    """
    nodes = list(grid_graph.nodes())
    all_paths = {}
    for source in nodes:
        lengths = nx.single_source_dijkstra_path_length(
            grid_graph, source, weight='weight'
        )
        for target in nodes:
            all_paths[(source, target)] = lengths.get(target, float('inf'))
    return all_paths


def get_subgoals(grid_graph, target_position):
    """Extract subgoals for a target position.

    A subgoal is an entry point where the target robot can enter the
    final component (positions reachable from goal via weight=1 edges)
    using a helper robot to block.

    Args:
        grid_graph: nx.DiGraph with weight and dependent edge attributes.
        target_position: (x, y) goal position.

    Returns:
        Dict: {subgoal_pos: {helper_pos: [target_bot_positions...]}}
    """
    final_component = get_final_component(grid_graph, target_position)

    subgoals = defaultdict(lambda: defaultdict(list))
    for source, target, data in grid_graph.edges(data=True):
        if data['weight'] == DEPENDENT_WEIGHT:
            subgoal_pos = target
            helper_pos = data['dependent']
            target_bot_pos = source
            if (target_bot_pos not in final_component
                    and subgoal_pos in final_component):
                subgoals[subgoal_pos][helper_pos].append(target_bot_pos)

    return {k: dict(v) for k, v in subgoals.items()}


def get_final_component(grid_graph, target_position):
    """Get the final component: nodes that can reach target via weight=1 only.

    Args:
        grid_graph: nx.DiGraph with weight edge attributes.
        target_position: (x, y) goal position.

    Returns:
        Set of (x, y) positions in the final component.
    """
    w1_edges = [
        (u, v) for u, v, d in grid_graph.edges(data=True)
        if d['weight'] == INDEPENDENT_WEIGHT
    ]
    w1_graph = nx.DiGraph()
    w1_graph.add_edges_from(w1_edges)

    final_component = set()
    if target_position in w1_graph:
        final_component = set(nx.ancestors(w1_graph, target_position))
        final_component.add(target_position)
    return final_component
