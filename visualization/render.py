import json
import webbrowser
from pathlib import Path

_TEMPLATE = Path(__file__).parent / "template.html"


def build_html(grid_env, state, subgoals=None):
    G = grid_env.G
    nodes = list(G.nodes())
    rows = [n[0] for n in nodes]
    cols = [n[1] for n in nodes]
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    robots = [
        {
            "position": list(state.target_robot.position),
            "color": state.target_robot.color.lower(),
            "role": "target",
        }
    ] + [
        {
            "position": list(h.position),
            "color": h.color.lower(),
            "role": "helper",
        }
        for h in state.helpers
    ]

    subgoal_data = []
    if subgoals:
        for sg, score in subgoals:
            subgoal_data.append(
                {
                    "bottleneck": list(sg.bottleneck.position),
                    "bottleneck_color": sg.bottleneck.color.lower(),
                    "support": list(sg.support.position),
                    "support_color": sg.support.color.lower(),
                    "goal": list(sg.goal_pos),
                    "score": score,
                }
            )

    grid_data = getattr(grid_env, "_grid_data", None)

    data = {
        "grid_data": grid_data,
        "robots": robots,
        "target": list(state.target),
        "subgoals": subgoal_data,
        "bounds": {
            "min_row": min_row,
            "max_row": max_row,
            "min_col": min_col,
            "max_col": max_col,
        },
    }

    template = _TEMPLATE.read_text()
    return template.replace("__DATA__", json.dumps(data))


def open_in_browser(grid_env, state, subgoals=None):
    html = build_html(grid_env, state, subgoals)
    path = Path.home() / "gridenv_vis.html"
    path.write_text(html)
    webbrowser.open(path.as_uri())
