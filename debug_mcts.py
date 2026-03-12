"""Run MCTS with live trace dumping for IDE debugging.

1. Start the live server:  python live_debug_server.py
2. Open http://localhost:8075 in browser
3. Set breakpoints in MCTS/v1.py (e.g., line 411: the for-loop)
4. Debug this file in VS Code (F5 or right-click -> Debug Python File)
5. Each time you hit Continue (F5), the browser updates with new data
"""

from omegaconf import OmegaConf
from GridEnv import GridEnv
from MCTS.v1 import MCTS_V1

cfg = OmegaConf.load("conf/v1.yaml")
cfg.max_iterations = 50  # keep it small for debugging

env_index = 38
grid_env, state = GridEnv.from_env(
    env_index,
    dependent_edge_weight=cfg.get("dependent_edge_weight", 2))

algo = MCTS_V1(cfg=cfg, trace=True)
result = algo.solve(grid_env, state,
    debug_hook=lambda algo, it: algo.tracer.dump_live())

print(f"Done: {len(result.all_plans)} plans found")
