"""Run MCTS with tracing and generate an HTML visualizer.

Usage:
    python visualize_mcts.py [--env 0] [--iterations 500] [--output mcts_trace.html]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from GridEnv import GridEnv
from MCTS.v1 import MCTS_V1


HTML_TEMPLATE_PATH = Path(__file__).resolve().parent / "mcts_visualizer.html"
DEFAULT_CFG_PATH = Path(__file__).resolve().parent / "conf" / "v1.yaml"


def generate_trace(env_index: int = 0, iterations: int | None = None) -> dict:
    """Run MCTS V1 on a single environment with tracing enabled."""
    cfg = OmegaConf.load(DEFAULT_CFG_PATH)
    grid_env, state = GridEnv.from_env(
        env_index,
        dependent_edge_weight=cfg.get("dependent_edge_weight", 2))
    if iterations is not None:
        cfg.max_iterations = iterations

    algo = MCTS_V1(cfg=cfg, trace=True)
    result = algo.solve(grid_env, state)

    print(f"Env {env_index}: best cost = "
          f"{result.all_plans[-1].cost if result.all_plans else 'N/A'}, "
          f"{len(algo.tracer.events)} iterations traced, "
          f"{len(algo.tracer.tree_nodes)} tree nodes")

    return algo.tracer.to_json()


def build_html(trace_data: dict, template_path: Path = HTML_TEMPLATE_PATH) -> str:
    """Inject trace data into the HTML template."""
    template = template_path.read_text()
    json_str = json.dumps(trace_data)
    return template.replace("__TRACE_DATA_PLACEHOLDER__", json_str)


def main():
    parser = argparse.ArgumentParser(description="Generate MCTS visualization")
    parser.add_argument("--env", type=int, default=38, help="Environment index")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override max_iterations (default: from config)")
    parser.add_argument("--output", type=str, default="mcts_trace.html",
                        help="Output HTML file")
    args = parser.parse_args()

    trace = generate_trace(args.env, args.iterations)
    html = build_html(trace)

    out_path = Path(args.output)
    out_path.write_text(html)
    print(f"Wrote visualization to {out_path}")


if __name__ == "__main__":
    main()
