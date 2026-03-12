"""Benchmark pipeline for Ricochet Robots partial plan algorithms.

Usage:
    from MCTS import MCTS_V1, MCTS_V2
    from benchmark import Benchmark

    bench = Benchmark(env_indices=range(10), n_val=3, n_eval=10)
    results = bench.run(MCTS_V1())
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from GridEnv import GridEnv, State
from validate_plan import validate
from evaluate_plan import evaluate_plan
from MCTS import MCTS, SolveResult, MCTS_V1, MCTS_V2, MCTS_V3, MCTS_V4, MCTS_V5
from visualize_results import (
    format_comparison_table, format_discovery_stats,
    build_json_data, save_json, generate_html,
)


# -- Benchmark ----------------------------------------------------------------

class Benchmark:
    """Loads environments once at init, then runs algorithms against them."""

    CACHE_DIR = Path("environments/cache")

    def __init__(
        self,
        env_indices: list[int],
        n_val: int = 10,
        n_eval: int = 100,
        use_cache: bool = True,
        dependent_edge_weight: float = 2,
    ):
        self.n_val = n_val
        self.n_eval = n_eval
        self.val_indices = env_indices[:n_val]
        self.eval_indices = env_indices[:n_eval]
        self.use_cache = use_cache
        self.dependent_edge_weight = dependent_edge_weight

        # Pre-load all environments (caching handled by GridEnv.from_env)
        self.envs: dict[int, tuple[GridEnv, State]] = {}
        for idx in self.eval_indices:
            self.envs[idx] = GridEnv.from_env(
                idx, dependent_edge_weight=self.dependent_edge_weight)

    def solve(self, algorithm: MCTS, grid_env: GridEnv, state: State):
        """Run algorithm, validate, and evaluate a single instance.

        Returns (result, metric) where result is a SolveResult and metric is
        None if validation fails or an edge is unreachable.
        """
        result = algorithm.solve(grid_env, state)
        best_plan = result.best_plan

        val_result = validate(best_plan, grid_env, state)
        if not val_result.passed:
            return result, None

        metric = evaluate_plan(best_plan, grid_env)
        return result, metric

    def run(
        self,
        algorithm: MCTS,
    ) -> dict[int, tuple[SolveResult, float | None]]:
        """Run the benchmark pipeline.

        Returns:
            {env_index: (solve_result, metric)}  --  metric is None on failure.
        """
        algo_name = type(algorithm).__name__
        results: dict[int, tuple[SolveResult, float | None]] = {}

        # Phase 1: validation pre-check
        for env_idx in self.val_indices:
            grid_env, state = self.envs[env_idx]
            result = algorithm.solve(grid_env, state)
            best_plan = result.best_plan
            val_result = validate(best_plan, grid_env, state)
            if not val_result.passed:
                print(f"[validation] {algo_name} env_{env_idx}: FAILED")
                for err in val_result.errors:
                    print(f"  {err}")
                return {}
            metric = evaluate_plan(best_plan, grid_env)
            results[env_idx] = (result, metric)

        # Phase 2: evaluation only (skip already-processed envs)
        for env_idx in self.eval_indices:
            if env_idx in results:
                continue
            grid_env, state = self.envs[env_idx]
            result = algorithm.solve(grid_env, state)
            best_plan = result.best_plan
            metric = evaluate_plan(best_plan, grid_env)
            results[env_idx] = (result, metric)

        return results


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark partial plan algorithms.")
    parser.add_argument("--n_val", type=int, default=3,
                        help="Number of envs for validation pre-check.")
    parser.add_argument("--n_eval", type=int, default=20,
                        help="Total number of envs to evaluate.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable environment caching.")
    parser.add_argument("--algorithms", type=str, default="V1,V2,V3,V4,V5",
                        help="Comma-separated list of algorithm versions to "
                             "run (default: V1,V2,V3,V4,V5).")
    args = parser.parse_args()

    # Available algorithms
    algorithms = [
        ("V1", MCTS_V1()),
        ("V2", MCTS_V2()),
        ("V3", MCTS_V3()),
        ("V4", MCTS_V4()),
        ("V5", MCTS_V5()),
    ]

    # Filter by --algorithms flag
    requested = {s.strip() for s in args.algorithms.split(",")}
    algorithms = [(name, algo) for name, algo in algorithms if name in requested]

    if not algorithms:
        print(f"No matching algorithms for: {args.algorithms}")
        print("Available: V1, V2, V3, V4, V5")
        raise SystemExit(1)

    bench = Benchmark(
        env_indices=list(range(args.n_eval)),
        n_val=args.n_val, n_eval=args.n_eval,
        use_cache=not args.no_cache,
    )

    algo_names: list[str] = []
    all_results: dict[str, dict[int, tuple[SolveResult, float | None]]] = {}

    for name, algo in algorithms:
        print(f"\nRunning {name} ({type(algo).__name__})...")
        results = bench.run(algo)
        if not results:
            print(f"Benchmark aborted for {name}: validation failed.")
            continue
        algo_names.append(name)
        all_results[name] = results

    if not algo_names:
        print("\nNo algorithms completed successfully.")
        raise SystemExit(1)

    # Collect all env indices that appear in any result set
    all_env_indices = sorted(
        set().union(*(r.keys() for r in all_results.values())))

    print(f"\n=== Benchmark Results ===\n")
    format_comparison_table(algo_names, all_results, all_env_indices)

    print(f"\n=== Plan Discovery Stats ===")
    format_discovery_stats(algo_names, all_results)

    # Export results
    json_data = build_json_data(algo_names, all_results)
    save_json(json_data)
    generate_html(json_data)
