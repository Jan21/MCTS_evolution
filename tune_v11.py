"""Hyperparameter search for V11 (Exact-Cost Reranking).

Tests different open_edge_penalty values.
"""
from benchmark import Benchmark
from A_star.v11 import A_star_V11
from A_star.v1 import A_star_V1

bench = Benchmark(env_indices=list(range(20)), n_val=1, n_eval=20)

# Get V1 baseline
v1 = A_star_V1()
v1_results = bench.run(v1)
v1_costs = {}
for idx, (result, metric) in v1_results.items():
    v1_costs[idx] = metric

penalties = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, -1.0, -3.0]

for penalty in penalties:
    algo = A_star_V11(open_edge_penalty=penalty, time_limit=30)
    results = bench.run(algo)
    if not results:
        print(f"  Penalty={penalty}: FAILED")
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

    print(f"  Penalty={penalty:6.1f}: avg={avg:.2f}"
          f"  {'IMPROVEMENTS: ' + ', '.join(improvements) if improvements else 'No improvements'}")
