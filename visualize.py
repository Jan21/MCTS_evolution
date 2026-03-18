import GridEnv
from GridEnv import State
import json
import copy
from dataclasses import asdict, is_dataclass

class VisualizationScene:
    def __init__(self, grid_env: GridEnv.GridEnv, **elements):
        self.grid_env = grid_env
        self.elements = elements  # e.g. target=(11,5), robots=[...], highlight=[...]

    def to_dict(self) -> dict:
        d = {
            "grid_size": self.grid_env.grid_size,
            "walls": [
                (w[0], w[1], f"{self.grid_env.grid_data[w[1] * self.grid_env.grid_size[0] + w[0]]}")
                for w in self.grid_env._wall_nodes
            ],
        }
        for key, val in self.elements.items():
            d[key] = self._serialize(val)
        return d

    def _serialize(self, val):
        if is_dataclass(val):
            return asdict(val)
        if isinstance(val, list):
            return [self._serialize(v) for v in val]
        if isinstance(val, dict):
            return {k: self._serialize(v) for k, v in val.items()}
        return val


class Visualizer:
    DEFAULT_SETTINGS = {
        "grid_lines_width": 1,
        "wall_width": 2,
        "grid_square_color": "white",
        "wall_color": "black",
        "coordinates_shown": True,
        "target_color": "red",
        "robot_radius": 0.3,
        "final_component_color": "lightblue",
    }

    def __init__(self, settings: dict = None):
        self.scenes: list[VisualizationScene] = []
        self.settings = {**self.DEFAULT_SETTINGS, **(settings or {})}

    def add_scene(self, grid_env, state: State = None, **elements) -> "Visualizer":
        """Add a scene. If state is provided, auto-solves and attaches solution for the Solver tab."""
        if state is not None:
            result = grid_env.solve(state)
            color_order = ["red", "blue", "green", "yellow"]
            start_positions = {
                color: list(result["start_positions"][i])
                for i, color in enumerate(color_order)
            }
            elements.setdefault("target", state.target)
            elements.setdefault("target_robot", state.target_robot)
            elements.setdefault("helper_robots", state.helpers)
            elements["solution"] = {
                "moves": result["moves"],
                "start_positions": start_positions,
                "target_robot": state.target_robot.color.lower(),
                "target_pos": list(state.target),
            }
        self.scenes.append(VisualizationScene(grid_env, **elements))
        return self

    def to_dict(self) -> dict:
        result = {"settings": self.settings}
        for i, scene in enumerate(self.scenes):
            result[i] = scene.to_dict()
        return result

    def save(self, path="visualization_data.json"):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
