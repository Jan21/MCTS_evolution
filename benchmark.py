"""Benchmark pipeline for Ricochet Robots partial plan algorithms.

Usage:
    from benchmark import MockAlgorithm, Benchmark

    bench = Benchmark(env_indices=range(10), n_val=3, n_eval=10)
    results = bench.run(MockAlgorithm())
"""

from __future__ import annotations

import argparse
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

from GridEnv import GridEnv, State, Robot_at
from partial_plan import PartialPlan
from validate_plan import validate
from evaluate_plan import evaluate_plan


# ── Algorithm interface ──────────────────────────────────────────────────

class Algorithm(ABC):
    @abstractmethod
    def solve(self, grid_env: GridEnv, state: State) -> PartialPlan:
        """Return a PartialPlan for the given environment and puzzle state."""
        ...


# ── Benchmark ────────────────────────────────────────────────────────────

class Benchmark:
    """Loads environments once at init, then runs algorithms against them."""

    CACHE_DIR = Path("environments/cache")

    def __init__(
        self,
        env_indices: list[int],
        n_val: int = 10,
        n_eval: int = 100,
        use_cache: bool = True,
    ):
        self.n_val = n_val
        self.n_eval = n_eval
        self.val_indices = env_indices[:n_val]
        self.eval_indices = env_indices[:n_eval]
        self.use_cache = use_cache

        if use_cache:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Pre-load all environments
        self.envs: dict[int, tuple[GridEnv, State]] = {}
        for idx in self.eval_indices:
            self.envs[idx] = self._load_env(idx)

    def _load_env(self, idx: int) -> tuple[GridEnv, State]:
        cache_path = self.CACHE_DIR / f"env_{idx}.pkl"

        if self.use_cache and cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        grid_env, state = GridEnv.from_env(idx)

        if self.use_cache:
            with open(cache_path, "wb") as f:
                pickle.dump((grid_env, state), f)

        return grid_env, state

    def solve(self, algorithm: Algorithm, grid_env: GridEnv, state: State):
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
        self,
        algorithm: Algorithm,
    ) -> dict[int, tuple[PartialPlan, float | None]]:
        """Run the benchmark pipeline.

        Returns:
            {env_index: (plan, metric)}  —  metric is None on failure.
        """
        results: dict[int, tuple[PartialPlan, float | None]] = {}

        # Phase 1: validation pre-check
        for env_idx in self.val_indices:
            grid_env, state = self.envs[env_idx]
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
        for env_idx in self.eval_indices:
            if env_idx in results:
                continue
            grid_env, state = self.envs[env_idx]
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
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable environment caching.")
    args = parser.parse_args()

    algo = MockAlgorithm()
    bench = Benchmark(
        env_indices=list(range(args.n_eval)),
        n_val=args.n_val, n_eval=args.n_eval,
        use_cache=not args.no_cache,
    )
    results = bench.run(algo)

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
