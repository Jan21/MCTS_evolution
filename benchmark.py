"""Benchmark framework for MCTS plan evaluation on Ricochet Robots.

Main interface:
    solve(game, algorithm) -> list of dicts with "plan" and "stats"
    validate_plan(plan, game, state) -> bool
    run_benchmark(algorithm, num_instances) -> dict with evaluation results
    compute_auc(trace, total_time) -> float
"""

import time
from abc import ABC, abstractmethod

import networkx as nx

from partial_plan import PartialPlan
from game import Game
from utils import get_subgoals, list_environment_files, compute_shortest_path

VALID_NODE_TYPES = {"goal", "subgoal", "bottleneck", "support", "leaf"}

N_SMALL = 10


# ========== Validation ==========

def validate_plan(plan, game, state):
    """Validate a PartialPlan against a game environment and state.

    A plan is valid if and only if:
    1. The plan is complete (no open edges).
    2. The DAG structure is well-formed (correct node types, required attrs).
    3. The goal matches the state's target position.
    4. All leaf robot positions match actual robot positions in the state.
    5. Every physical edge (non-structural) corresponds to a traversable path:
       - For bottleneck->leaf edges: path using independent edges AND
         dependent edges activated by the sibling support position.
       - For all other physical edges: path using independent edges only.
    6. The cost of each physical edge matches the actual shortest path length.

    Structural edges (subgoal -> bottleneck/support) are skipped.

    Args:
        plan: PartialPlan instance.
        game: Game instance.
        state: State instance.

    Returns:
        True if valid, False otherwise.
    """
    # 1. Completeness
    if not plan.validate_plan():
        return False

    # 2. & 3. & 4. Structural integrity + instance checks
    if not _validate_structure(plan, game, state):
        return False

    # 5. & 6. Physical edge validation
    for parent, child, edge_data in plan.g.edges(data=True):
        if plan.is_structural_edge(parent, child):
            continue

        positions = plan.resolve_segment_positions(parent, child)
        if positions is None:
            return False

        source_pos, dest_pos = positions

        parent_type = plan.g.nodes[parent].get('ntype')
        child_type = plan.g.nodes[child].get('ntype')

        if parent_type == 'bottleneck' and child_type == 'leaf':
            # Context-aware: use support position to activate dependent edges
            support_pos = _get_sibling_support_pos(plan, parent)
            if support_pos is None:
                return False
            expected_dist = compute_shortest_path(
                game.grid_graph, source_pos, dest_pos,
                support_positions={support_pos}
            )
        else:
            # All other physical edges: precomputed independent paths
            expected_dist = game.independent_paths.get((source_pos, dest_pos))

        if expected_dist is None:
            return False

        # Check cost matches for fixed edges with a cost
        if edge_data.get('cost') is not None:
            if edge_data['cost'] != expected_dist:
                return False

    return True


def _validate_structure(plan, game, state):
    """Validate the DAG structure of a plan.

    Checks:
        - All nodes have a known ntype.
        - Exactly one goal node with pos attribute; goal is the root (no parents).
        - Goal position matches state.target.
        - All leaf nodes have pos and robot; leaves have no children.
        - Each leaf's (robot, pos) matches an actual robot in the state.
        - All bottleneck/support nodes have pos and robot.
        - All positions are valid grid nodes.
        - Every subgoal has exactly one bottleneck and one support child.
        - Every edge is a valid type pair (only 6 allowed patterns).
        - All nodes are reachable from the goal.
        - The graph is a valid DAG (no cycles).
    """
    g = plan.g

    # Must have at least one node
    if g.number_of_nodes() == 0:
        return False

    grid_nodes = game.grid_nodes

    # All nodes must have a valid ntype
    for nid, attrs in g.nodes(data=True):
        if attrs.get('ntype') not in VALID_NODE_TYPES:
            return False

    # Exactly one goal node with pos; goal must be the DAG root
    goal_nodes = plan.nodes_by_type("goal")
    if len(goal_nodes) != 1:
        return False
    goal_nid, goal_attrs = goal_nodes[0]
    if 'pos' not in goal_attrs:
        return False
    if goal_attrs['pos'] not in grid_nodes:
        return False
    if len(list(g.predecessors(goal_nid))) > 0:
        return False

    # Goal position must match the state's target
    if goal_attrs['pos'] != state.target:
        return False

    # Build lookup of valid robots: {(name, pos)} from the state
    valid_robots = {(state.target_robot.name, state.target_robot.pos)}
    for hr in state.helper_robots:
        valid_robots.add((hr.name, hr.pos))

    # All leaves must have pos and robot; leaves must have no children
    # Each leaf must correspond to an actual robot position
    for nid, attrs in plan.nodes_by_type("leaf"):
        if 'pos' not in attrs or 'robot' not in attrs:
            return False
        if attrs['pos'] not in grid_nodes:
            return False
        if len(list(g.successors(nid))) > 0:
            return False
        if (attrs['robot'], attrs['pos']) not in valid_robots:
            return False

    # All bottleneck/support must have pos and robot
    for ntype in ("bottleneck", "support"):
        for nid, attrs in plan.nodes_by_type(ntype):
            if 'pos' not in attrs or 'robot' not in attrs:
                return False
            if attrs['pos'] not in grid_nodes:
                return False

    # Subgoal entry_pos must be a valid grid node
    # Each subgoal must have exactly one bottleneck and one support child
    for nid, attrs in plan.nodes_by_type("subgoal"):
        if 'entry_pos' not in attrs:
            return False
        if attrs['entry_pos'] not in grid_nodes:
            return False
        child_types = sorted(
            g.nodes[c].get('ntype') for c in g.successors(nid)
        )
        if child_types != ['bottleneck', 'support']:
            return False

    # Every edge must be a valid parent_type -> child_type pair
    valid_edge_types = {
        ('goal', 'leaf'),           # physical: direct path
        ('goal', 'subgoal'),        # physical: decompose via subgoal
        ('subgoal', 'bottleneck'),  # structural: subgoal groups bottleneck
        ('subgoal', 'support'),     # structural: subgoal groups support
        ('bottleneck', 'leaf'),     # physical: robot reaches bottleneck
        ('support', 'leaf'),        # physical: robot reaches support
    }
    for parent, child in g.edges():
        parent_type = g.nodes[parent].get('ntype')
        child_type = g.nodes[child].get('ntype')
        if (parent_type, child_type) not in valid_edge_types:
            return False

    # Must be a DAG
    if not nx.is_directed_acyclic_graph(g):
        return False

    # All nodes must be reachable from goal
    reachable = nx.descendants(g, goal_nid) | {goal_nid}
    if reachable != set(g.nodes()):
        return False

    return True


def _get_sibling_support_pos(plan, bottleneck_nid):
    """Find the support position that is sibling to a bottleneck node.

    Walks up from the bottleneck to its parent subgoal, then finds the
    support child of that subgoal.

    Args:
        plan: PartialPlan instance.
        bottleneck_nid: Node ID of the bottleneck node.

    Returns:
        (x, y) position of the sibling support node, or None if
        the expected structure is not found.
    """
    parents = plan.parents(bottleneck_nid)
    if len(parents) != 1:
        return None

    subgoal_nid = parents[0]
    if plan.g.nodes[subgoal_nid].get('ntype') != 'subgoal':
        return None

    for sibling_nid in plan.children(subgoal_nid):
        sibling_attrs = plan.g.nodes[sibling_nid]
        if sibling_attrs.get('ntype') == 'support':
            return sibling_attrs.get('pos')

    return None


# ========== Algorithm Base Class ==========

class Algorithm(ABC):
    """Base class for plan-search algorithms.

    Subclasses implement solve_instance() to produce a plan for a single
    game instance.
    """

    def __init__(self, name=None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def solve_instance(self, game, state):
        """Solve a single instance.

        Args:
            game: Game instance.
            state: State instance.

        Returns:
            Dict with:
                "plan": PartialPlan instance (best plan found)
                "stats": Dict with "cost", "wallclock_time", "trace", "auc"
                    trace: list of (time, cost) tuples for anytime AUC
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

    def solve_instance(self, game, state):
        start_time = time.time()

        independent_paths = game.independent_paths
        target = state.target
        target_robot = state.target_robot
        target_pos = target_robot.pos

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
            game.grid_graph, independent_paths, state
        )
        if plan is not None:
            return _make_result(plan, start_time)

        # Strategy 3: Fallback — empty plan
        return _make_result(PartialPlan(), start_time)

    def _try_subgoal_plan(self, grid_graph, independent_paths, state):
        """Try to build a plan using the first viable subgoal."""
        target = state.target
        target_robot = state.target_robot
        target_pos = target_robot.pos

        subgoals = get_subgoals(grid_graph, target)
        if not subgoals:
            return None

        for sg_pos, helper_dict in subgoals.items():
            entry_to_goal_cost = independent_paths.get((sg_pos, target))
            if entry_to_goal_cost is None:
                continue

            for helper_pos, target_bot_positions in helper_dict.items():
                for target_bot_pos in target_bot_positions:
                    target_to_bn = independent_paths.get(
                        (target_pos, target_bot_pos)
                    )
                    if target_to_bn is None:
                        continue

                    best_helper, best_helper_cost = self._find_best_helper(
                        independent_paths, state.helper_robots,
                        helper_pos
                    )
                    if best_helper is None:
                        continue

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
                                  pos=best_helper.pos,
                                  robot=best_helper.name)

                    plan.add_edge("goal", "sg1", status="fixed",
                                  cost=entry_to_goal_cost)
                    plan.add_edge("bn1", "leaf_target", status="fixed",
                                  cost=target_to_bn)
                    plan.add_edge("sp1", "leaf_helper", status="fixed",
                                  cost=best_helper_cost)
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
            cost = independent_paths.get((robot.pos, helper_pos))
            if cost is not None and (best_cost is None or cost < best_cost):
                best_cost = cost
                best_robot = robot
        return best_robot, best_cost


def compute_auc(trace, total_time):
    """Compute area under the cost-vs-time step function.

    For anytime algorithms, the trace records (time, cost) at each
    improvement. The AUC integrates the step function from the first
    solution time to total_time.

    Args:
        trace: List of (wallclock_time, cost) tuples, sorted by time,
               each representing a new best solution found.
        total_time: Total wallclock time of the algorithm run.

    Returns:
        AUC value (float). Returns float('inf') if trace is empty.
    """
    if not trace:
        return float('inf')
    auc = 0.0
    for i in range(len(trace)):
        t_start = trace[i][0]
        t_end = trace[i + 1][0] if i + 1 < len(trace) else total_time
        auc += trace[i][1] * (t_end - t_start)
    return auc


def _make_result(plan, start_time):
    """Build a result dict from a plan and start time."""
    elapsed = time.time() - start_time
    cost = plan.cost() if plan.g.number_of_nodes() > 0 else float('inf')
    trace = [(elapsed, cost)] if cost != float('inf') else []
    return {
        "plan": plan,
        "stats": {
            "cost": cost,
            "wallclock_time": elapsed,
            "trace": trace,
            "auc": compute_auc(trace, elapsed),
        }
    }


# ========== Benchmark Runner ==========

def solve(game, algorithm=None):
    """Solve all instances in a game.

    Args:
        game: Game instance.
        algorithm: Algorithm instance. Defaults to MockAlgorithm().

    Returns:
        List of dicts, each with "plan" and "stats" keys.
    """
    if algorithm is None:
        algorithm = MockAlgorithm()
    results = []
    for state in game.states:
        result = algorithm.solve_instance(game, state)
        results.append(result)
    return results


def run_benchmark(algorithm, num_instances=128, env_dir=None,
                  validate_first=True, n_small=N_SMALL):
    """Run benchmark across multiple environment files.

    Workflow:
    1. If validate_first: run on first n_small environments, validate
       all plans. Abort with error details if any fail. Results are kept.
    2. Run on remaining environments, collect stats.
    3. Return aggregated results from both phases.

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

    all_results = []

    # Phase 1: Validation on small subset (results are reused)
    if validate_first:
        validated = _run_validation_phase(algorithm, env_files[:n_small])
        all_results.extend(validated)
        remaining_files = env_files[n_small:]
    else:
        # TODO probably should crash or something
        remaining_files = env_files

    # Phase 2: Remaining environments
    for pkl_path in remaining_files:
        game = Game.from_pickle(pkl_path)
        env_results = solve(game, algorithm=algorithm)
        all_results.extend(env_results)

    # Aggregate
    total_cost = 0.0
    total_auc = 0.0
    count = 0
    for result in all_results:
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
    """Run and validate on a subset of environments.

    Returns the results so they can be reused in the full evaluation.
    Raises ValueError if any non-empty plan fails validation.
    """
    results = []
    for pkl_path in env_files:
        game = Game.from_pickle(pkl_path)
        for state in game.states:
            result = algorithm.solve_instance(game, state)
            results.append(result)
            plan = result['plan']
            if plan.g.number_of_nodes() == 0:
                continue
            if not validate_plan(plan, game, state):
                raise ValueError(
                    f"Validation failed for env_{game.graph_idx}: "
                    f"open_edges={len(plan.open_edges())}, "
                    f"cost={plan.cost()}"
                )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Ricochet Robots benchmark")
    parser.add_argument("-n", "--num-instances", type=int, default=128,
                        help="Number of environment files to evaluate (default: 128)")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip validation phase")
    args = parser.parse_args()

    algo = MockAlgorithm()
    print(f"Running benchmark with {algo.name} on {args.num_instances} environments...")
    results = run_benchmark(algo, num_instances=args.num_instances,
                            validate_first=not args.no_validate)

    print(f"\nResults:")
    print(f"  Solved: {results['num_solved']} / {results['num_total']}")
    print(f"  Average cost: {results['average_cost']:.2f}")
    print(f"  Total AUC: {results['total_auc']:.4f}")
