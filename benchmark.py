"""Benchmark pipeline for Ricochet Robots partial plan algorithms.

Usage:
    from benchmark import MockAlgorithm, run

    algo = MockAlgorithm()
    results = run(algo, env_indices=range(10), n_val=3, n_eval=10)
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod

from GridEnv import GridEnv, State, Robot_at
from partial_plan import PartialPlan
from validate_plan import validate, STRUCTURAL_EDGE_PAIRS


# ── Algorithm interface ──────────────────────────────────────────────────

class Algorithm(ABC):
    @abstractmethod
    def solve(self, grid_env: GridEnv, state: State) -> PartialPlan:
        """Return a PartialPlan for the given environment and puzzle state."""
        ...


# ── Metric computation ──────────────────────────────────────────────────

def _find_sibling_support_pos(g, bottleneck_id: str):
    """Find the support position that is sibling to a bottleneck node.

    Traverses: bottleneck → parent subgoal → support child → pos.
    """
    for parent in g.predecessors(bottleneck_id):
        if g.nodes[parent].get("ntype") == "subgoal":
            for child in g.successors(parent):
                if g.nodes[child].get("ntype") == "support":
                    return g.nodes[child].get("pos")
    return None


def _edge_movement(g, parent: str, child: str):
    """Determine (start, end, support_pos) for a physical edge.

    Physical movement goes from child side to parent side.
    """
    child_data = g.nodes[child]
    parent_data = g.nodes[parent]
    child_type = child_data.get("ntype")
    parent_type = parent_data.get("ntype")

    if child_type == "subgoal":
        bn = [c for c in g.successors(child)
              if g.nodes[c].get("ntype") == "bottleneck"]
        sp = [c for c in g.successors(child)
              if g.nodes[c].get("ntype") == "support"]
        if not bn or not sp:
            return None, None, None
        return (
            g.nodes[bn[0]].get("pos"),
            parent_data.get("pos"),
            g.nodes[sp[0]].get("pos"),
        )

    if child_type in ("leaf", "support"):
        support_pos = None
        if parent_type == "bottleneck":
            support_pos = _find_sibling_support_pos(g, parent)
        return child_data.get("pos"), parent_data.get("pos"), support_pos

    return None, None, None


def evaluate_plan(plan: PartialPlan, grid_env: GridEnv) -> float | None:
    """Compute total cost: sum of exact shortest path lengths over physical edges.

    Returns None if any physical edge is unreachable.
    """
    g = plan.g
    total = 0

    for parent, child in g.edges():
        parent_type = g.nodes[parent].get("ntype")
        child_type = g.nodes[child].get("ntype")

        if (parent_type, child_type) in STRUCTURAL_EDGE_PAIRS:
            continue

        start, end, support_pos = _edge_movement(g, parent, child)
        if start is None or end is None:
            return None

        cost = grid_env.compute_exact_shortest_path_length(start, end, support_pos)
        if cost is None:
            return None

        total += cost

    return total


# ── Pipeline ─────────────────────────────────────────────────────────────

def solve(algorithm: Algorithm, grid_env: GridEnv, state: State):
    """Run algorithm, validate, and evaluate a single instance.

    Returns (plan, metric) where metric is None if validation fails
    or an edge is unreachable.
    """
    plan = algorithm.solve(grid_env, state)

    result = validate(plan, grid_env, state)
    if not result.passed:
        return plan, None

    metric = evaluate_plan(plan, grid_env)
    return plan, metric


def run(
    algorithm: Algorithm,
    env_indices: list[int],
    n_val: int = 10,
    n_eval: int = 100,
) -> dict[int, tuple[PartialPlan, float | None]]:
    """Run the benchmark pipeline.

    Args:
        algorithm:   Algorithm instance to benchmark.
        env_indices: Pool of environment indices to draw from.
        n_val:       Number of envs for validation pre-check (drawn first).
        n_eval:      Total number of envs to evaluate (includes the n_val envs).

    Returns:
        {env_index: (plan, metric)}  —  metric is None on failure.
    """
    val_indices = env_indices[:n_val]
    eval_indices = env_indices[:n_eval]
    results: dict[int, tuple[PartialPlan, float | None]] = {}

    # Phase 1: validation pre-check
    for env_idx in val_indices:
        grid_env, state = GridEnv.from_env(env_idx)
        plan = algorithm.solve(grid_env, state)
        val_result = validate(plan, grid_env, state)
        if not val_result.passed:
            print(f"[validation] env_{env_idx}: FAILED")
            for err in val_result.errors:
                print(f"  {err}")
            return {}
        metric = evaluate_plan(plan, grid_env)
        results[env_idx] = (plan, metric)

    # Phase 2: evaluation only (skip already-processed envs)
    for env_idx in eval_indices:
        if env_idx in results:
            continue
        grid_env, state = GridEnv.from_env(env_idx)
        plan = algorithm.solve(grid_env, state)
        metric = evaluate_plan(plan, grid_env)
        results[env_idx] = (plan, metric)

    return results


# ── Mock algorithm ───────────────────────────────────────────────────────

class MockAlgorithm(Algorithm):
    """Trivial algorithm: direct goal → leaf plan.

    Works when the target robot can independently reach the goal.
    Otherwise validation will fail.
    """

    def solve(self, grid_env: GridEnv, state: State) -> PartialPlan:
        plan = PartialPlan()

        plan.add_node("goal", "goal", pos=state.target)
        plan.add_node("leaf_target", "leaf",
                      pos=state.target_robot.position,
                      robot=state.target_robot)

        cost = grid_env.compute_exact_shortest_path_length(
            state.target_robot.position, state.target,
        )
        plan.add_edge("goal", "leaf_target", status="fixed", cost=cost)
        return plan


# ── Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark partial plan algorithms.")
    parser.add_argument("--n_val", type=int, default=3,
                        help="Number of envs for validation pre-check.")
    parser.add_argument("--n_eval", type=int, default=10,
                        help="Total number of envs to evaluate.")
    args = parser.parse_args()

    algo = MockAlgorithm()
    results = run(algo, env_indices=list(range(args.n_eval)), n_val=args.n_val, n_eval=args.n_eval)

    if not results:
        print("Benchmark aborted: validation failed.")
    else:
        metrics = [m for _, (_, m) in sorted(results.items()) if m is not None]
        failed = len(results) - len(metrics)

        print(f"\n=== Benchmark: {type(algo).__name__} ===")
        print(f"  Environments: {len(results)}  (solved: {len(metrics)}, failed: {failed})")
        if metrics:
            print(f"  Total cost:   {sum(metrics)}")
            print(f"  Avg cost:     {sum(metrics) / len(metrics):.2f}")
            print(f"  Min cost:     {min(metrics)}")
            print(f"  Max cost:     {max(metrics)}")

        print("\n  Per environment:")
        for env_idx, (_, metric) in sorted(results.items()):
            status = f"cost={metric}" if metric is not None else "FAILED"
            print(f"    env_{env_idx}: {status}")
