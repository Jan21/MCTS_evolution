"""Hyperparameter search for V7 (K-th Best / Anti-Greedy A*).

Tests different max_rank and max_children values to find any
configuration that beats V1's avg_cost=8.90 on any environment.
"""
from benchmark import Benchmark
from A_star.v7 import A_star_V7
from A_star.v1 import A_star_V1

bench = Benchmark(env_indices=list(range(20)), n_val=1, n_eval=20)

# Get V1 baseline per-env
v1 = A_star_V1()
v1_results = bench.run(v1)
v1_costs = {}
for idx, (result, metric) in v1_results.items():
    v1_costs[idx] = metric

configs = [
    {"max_rank": 1, "max_children": 50},
    {"max_rank": 2, "max_children": 50},
    {"max_rank": 3, "max_children": 50},
    {"max_rank": 5, "max_children": 50},
    {"max_rank": 10, "max_children": 50},
    {"max_rank": 3, "max_children": 100},
    {"max_rank": 5, "max_children": 100},
]

for cfg in configs:
    algo = A_star_V7(**cfg, time_limit=30)
    results = bench.run(algo)
    if not results:
        print(f"  Config {cfg}: FAILED")
        continue

    costs = {}
    for idx, (result, metric) in results.items():
        costs[idx] = metric

    avg = sum(c for c in costs.values() if c is not None) / len(costs)
    improvements = []
    for idx in sorted(costs.keys()):
        if costs[idx] is not None and v1_costs.get(idx) is not None:
            if costs[idx] < v1_costs[idx]:
                improvements.append(f"env_{idx}: {v1_costs[idx]}→{costs[idx]}")

    print(f"  Config {cfg}: avg={avg:.2f}"
          f"  {'IMPROVEMENTS: ' + ', '.join(improvements) if improvements else 'No improvements'}")
