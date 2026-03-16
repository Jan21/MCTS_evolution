"""Hyperparameter search for V2."""
from GridEnv import GridEnv, State
from A_star.v2 import A_star_V2
from validate_plan import validate
from evaluate_plan import evaluate_plan

envs = {}
for i in range(20):
    envs[i] = GridEnv.from_env(i)

configs = [
    {"depth_penalty": 0.0, "edge_selection": "first"},
    {"depth_penalty": 0.0, "edge_selection": "cheapest"},
    {"depth_penalty": 0.0, "edge_selection": "most_constrained"},
    {"depth_penalty": 0.5, "edge_selection": "first"},
    {"depth_penalty": 0.5, "edge_selection": "cheapest"},
    {"depth_penalty": 0.5, "edge_selection": "most_constrained"},
    {"depth_penalty": 1.0, "edge_selection": "cheapest"},
    {"depth_penalty": 2.0, "edge_selection": "cheapest"},
    {"depth_penalty": 0.5, "edge_selection": "cheapest", "max_children": 10},
    {"depth_penalty": 0.5, "edge_selection": "cheapest", "max_children": 100},
]

for cfg in configs:
    mc = cfg.pop("max_children", 30)
    algo = A_star_V2(max_children=mc, **cfg)
    costs = []
    solved = 0
    for i in range(20):
        grid_env, state = envs[i]
        result = algo.solve(grid_env, state)
        vr = validate(result.best_plan, grid_env, state)
        if vr.passed:
            metric = evaluate_plan(result.best_plan, grid_env)
            if metric is not None:
                costs.append(metric)
                solved += 1
    avg = sum(costs) / len(costs) if costs else float('inf')
    cfg["max_children"] = mc
    print(f"dp={cfg['depth_penalty']:.1f} edge={cfg['edge_selection']:18s} "
          f"mc={mc:3d} | avg={avg:6.2f} solved={solved}/20")
