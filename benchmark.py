"""Benchmark framework for MCTS plan evaluation on Ricochet Robots.

Main interface:
    solve(environment_pkl, algorithm) -> list of dicts with "plan" and "stats"
    validate_plan(plan, env_data) -> bool
    run_benchmark(algorithm, num_instances) -> dict with evaluation results
"""

import time
from abc import ABC, abstractmethod

import networkx as nx

from partial_plan import PartialPlan
from utils import (
    load_environment,
    list_environment_files,
    get_independent_paths,
    get_subgoals,
)

N_SMALL = 10


# ========== Validation ==========

def validate_plan(plan, env_data):
    """Validate a PartialPlan against an environment.

    A plan is valid if and only if:
    1. The plan is complete (no open edges).
    2. The DAG structure is well-formed (correct node types, required attrs).
    3. Every physical edge (non-structural) corresponds to a path
       traversable via independent (weight=1) edges only.
    4. The cost of each physical edge matches the actual independent
       shortest path length.

    Structural edges (subgoal -> bottleneck/support) are skipped.

    Args:
        plan: PartialPlan instance.
        env_data: Dict from load_environment().

    Returns:
        True if valid, False otherwise.
    """
    # 1. Completeness
    if not plan.validate_plan():
        return False

    # 2. Structural integrity
    if not _validate_structure(plan):
        return False

    # 3. & 4. Physical edge validation
    independent_paths = get_independent_paths(env_data)

    for parent, child, edge_data in plan.g.edges(data=True):
        if plan.is_structural_edge(parent, child):
            continue

        positions = plan.resolve_segment_positions(parent, child)
        if positions is None:
            return False

        source_pos, dest_pos = positions

        # Check independent path exists
        ind_dist = independent_paths.get((source_pos, dest_pos))
        if ind_dist is None:
            return False

        # Check cost matches for fixed edges with a cost
        if edge_data.get('cost') is not None:
            if edge_data['cost'] != ind_dist:
                return False

    return True


def _validate_structure(plan):
    """Validate the DAG structure of a plan.

    Checks:
        - Exactly one goal node with pos attribute.
        - All leaf nodes have pos and robot attributes.
        - All bottleneck/support nodes have pos attribute.
        - The graph is a valid DAG (no cycles).
    """
    # Must have at least one node
    if plan.g.number_of_nodes() == 0:
        return False

    # Exactly one goal node
    goal_nodes = plan.nodes_by_type("goal")
    if len(goal_nodes) != 1:
        return False
    if 'pos' not in goal_nodes[0][1]:
        return False

    # All leaves must have pos and robot
    for _, attrs in plan.nodes_by_type("leaf"):
        if 'pos' not in attrs or 'robot' not in attrs:
            return False

    # All bottleneck/support must have pos
    for ntype in ("bottleneck", "support"):
        for _, attrs in plan.nodes_by_type(ntype):
            if 'pos' not in attrs:
                return False

    # Must be a DAG
    if not nx.is_directed_acyclic_graph(plan.g):
        return False

    return True


# ========== Algorithm Base Class ==========

class Algorithm(ABC):
    """Base class for plan-search algorithms.

    Subclasses implement solve_instance() to produce a plan for a single
    game instance.
    """

    def __init__(self, name=None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def solve_instance(self, env_data, instance):
        """Solve a single instance.

        Args:
            env_data: Dict from load_environment().
            instance: Dict with helper_robots, target_robot, target.

        Returns:
            Dict with:
                "plan": PartialPlan instance
                "stats": Dict with "cost", "wallclock_time", "auc"
        """


class MockAlgorithm(Algorithm):
    """Mock algorithm that creates simple valid plans for testing.

    Strategy:
    1. If target robot can reach goal via independent edges directly,
       creates a 2-node plan (goal <- leaf).
    2. If subgoals exist and a valid decomposition is found, creates
       a 6-node plan (goal <- sg <- bn/sp <- leaves).
    3. Otherwise returns an empty plan with infinite cost.
    """

    def solve_instance(self, env_data, instance):
        start_time = time.time()

        grid_graph = env_data['grid_graph']
        independent_paths = get_independent_paths(env_data)
        target = instance['target']
        target_robot = instance['target_robot']
        target_pos = (target_robot.x, target_robot.y)

        plan = PartialPlan()

        # Strategy 1: Direct independent path
        direct_cost = independent_paths.get((target_pos, target))
        if direct_cost is not None:
            plan.add_node("goal", "goal", pos=target)
            plan.add_node("leaf0", "leaf", pos=target_pos,
                          robot=target_robot.name)
            plan.add_edge("goal", "leaf0", status="fixed",
                          cost=direct_cost)
            return _make_result(plan, start_time)

        # Strategy 2: Single subgoal decomposition
        plan = self._try_subgoal_plan(
            grid_graph, independent_paths, instance
        )
        if plan is not None:
            return _make_result(plan, start_time)

        # Strategy 3: Fallback — empty plan
        return _make_result(PartialPlan(), start_time)

    def _try_subgoal_plan(self, grid_graph, independent_paths, instance):
        """Try to build a plan using the first viable subgoal.

        Subgoal structure:
            sg_pos: position where target enters the final component
                    (destination of the dependent edge, inside FC)
            helper_pos: where the helper robot must be to create the block
            target_bot_pos: where the target robot must be before the
                    dependent edge fires (source of the dependent edge)

        Plan nodes:
            goal(pos=target) -> sg1(entry_pos=sg_pos)
                sg1 -> bn1(pos=target_bot_pos)  [structural]
                sg1 -> sp1(pos=helper_pos)      [structural]
                bn1 -> leaf_target(pos=target_current)
                sp1 -> leaf_helper(pos=helper_current)

        Physical edges validated:
            goal->sg1:         entry_pos -> goal  (inside FC, independent)
            bn1->leaf_target:  target_current -> target_bot_pos (independent)
            sp1->leaf_helper:  helper_current -> helper_pos (independent)
        """
        target = instance['target']
        target_robot = instance['target_robot']
        target_pos = (target_robot.x, target_robot.y)

        subgoals = get_subgoals(grid_graph, target)
        if not subgoals:
            return None

        for sg_pos, helper_dict in subgoals.items():
            # Cost: entry_pos (sg_pos) to goal (independent, inside FC)
            entry_to_goal_cost = independent_paths.get((sg_pos, target))
            if entry_to_goal_cost is None:
                continue

            for helper_pos, target_bot_positions in helper_dict.items():
                for target_bot_pos in target_bot_positions:
                    # Cost: target robot current pos to target_bot_pos
                    target_to_bn = independent_paths.get(
                        (target_pos, target_bot_pos)
                    )
                    if target_to_bn is None:
                        continue

                    # Cost: best helper to support position
                    best_helper, best_helper_cost = self._find_best_helper(
                        independent_paths, instance['helper_robots'],
                        helper_pos
                    )
                    if best_helper is None:
                        continue

                    # Build the 6-node plan
                    plan = PartialPlan()
                    plan.add_node("goal", "goal", pos=target)
                    plan.add_node("sg1", "subgoal",
                                  entry_pos=sg_pos)
                    plan.add_node("bn1", "bottleneck",
                                  pos=target_bot_pos,
                                  robot=target_robot.name)
                    plan.add_node("sp1", "support",
                                  pos=helper_pos,
                                  robot=best_helper.name)
                    plan.add_node("leaf_target", "leaf",
                                  pos=target_pos,
                                  robot=target_robot.name)
                    plan.add_node("leaf_helper", "leaf",
                                  pos=(best_helper.x, best_helper.y),
                                  robot=best_helper.name)

                    # Physical edges
                    plan.add_edge("goal", "sg1", status="fixed",
                                  cost=entry_to_goal_cost)
                    plan.add_edge("bn1", "leaf_target", status="fixed",
                                  cost=target_to_bn)
                    plan.add_edge("sp1", "leaf_helper", status="fixed",
                                  cost=best_helper_cost)

                    # Structural edges
                    plan.add_edge("sg1", "bn1", status="fixed")
                    plan.add_edge("sg1", "sp1", status="fixed")

                    return plan

        return None

    @staticmethod
    def _find_best_helper(independent_paths, helper_robots, helper_pos):
        """Find the helper robot with shortest independent path to helper_pos."""
        best_robot = None
        best_cost = None
        for robot in helper_robots:
            h_pos = (robot.x, robot.y)
            cost = independent_paths.get((h_pos, helper_pos))
            if cost is not None and (best_cost is None or cost < best_cost):
                best_cost = cost
                best_robot = robot
        return best_robot, best_cost


def _make_result(plan, start_time):
    """Build a result dict from a plan and start time."""
    elapsed = time.time() - start_time
    cost = plan.cost() if plan.g.number_of_nodes() > 0 else float('inf')
    return {
        "plan": plan,
        "stats": {
            "cost": cost,
            "wallclock_time": elapsed,
            "auc": cost * elapsed,
        }
    }


# ========== Benchmark Runner ==========

def solve(environment_pkl, algorithm=None):
    """Solve all instances in an environment pickle.

    Args:
        environment_pkl: Path to env_X.pkl file.
        algorithm: Algorithm instance. Defaults to MockAlgorithm().

    Returns:
        List of dicts, each with "plan" and "stats" keys.
    """
    if algorithm is None:
        algorithm = MockAlgorithm()
    env_data = load_environment(environment_pkl)
    results = []
    for instance in env_data['instances']:
        result = algorithm.solve_instance(env_data, instance)
        results.append(result)
    return results


def run_benchmark(algorithm, num_instances=128, env_dir=None,
                  validate_first=True, n_small=N_SMALL):
    """Run benchmark across multiple environment files.

    Workflow:
    1. If validate_first: run on first n_small environments, validate
       all plans. Abort with error details if any fail.
    2. Run on num_instances environments, collect stats.
    3. Return aggregated results.

    Args:
        algorithm: Algorithm instance.
        num_instances: Number of environment files to evaluate.
        env_dir: Directory containing env_*.pkl files.
        validate_first: Whether to validate on small subset first.
        n_small: Size of validation subset.

    Returns:
        Dict with average_cost, total_auc, num_solved, num_total, results.
    """
    env_files = list_environment_files(env_dir, num=num_instances)

    # Phase 1: Validation on small subset
    if validate_first:
        _run_validation_phase(algorithm, env_files[:n_small])

    # Phase 2: Full evaluation
    all_results = []
    total_cost = 0.0
    total_auc = 0.0
    count = 0

    for pkl_path in env_files:
        env_results = solve(pkl_path, algorithm=algorithm)
        for result in env_results:
            all_results.append(result)
            stats = result['stats']
            if stats['cost'] != float('inf'):
                total_cost += stats['cost']
                total_auc += stats['auc']
                count += 1

    avg_cost = total_cost / count if count > 0 else float('inf')

    return {
        "average_cost": avg_cost,
        "total_auc": total_auc,
        "num_solved": count,
        "num_total": len(all_results),
        "results": all_results,
    }


def _run_validation_phase(algorithm, env_files):
    """Run validation on a subset of environments. Raises on failure."""
    for pkl_path in env_files:
        env_data = load_environment(pkl_path)
        for instance in env_data['instances']:
            result = algorithm.solve_instance(env_data, instance)
            plan = result['plan']
            if plan.g.number_of_nodes() == 0:
                continue
            if not validate_plan(plan, env_data):
                raise ValueError(
                    f"Validation failed for {pkl_path.name}: "
                    f"open_edges={len(plan.open_edges())}, "
                    f"cost={plan.cost()}"
                )
