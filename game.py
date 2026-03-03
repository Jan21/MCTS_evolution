"""Ricochet Robots game board and puzzle state."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from robot import Robot

ENV_DIR = Path(__file__).resolve().parent / "environments"


@dataclass
class State:
    """A single puzzle instance: which robot must reach which cell."""

    target: tuple[int, int]
    target_robot: Robot
    helpers: list[Robot]

    @property
    def all_robots(self) -> list[Robot]:
        return [self.target_robot] + self.helpers


class Game:
    """A Ricochet Robots board loaded from an environment file.

    Attributes:
        grid_graph:        The full directed graph (independent + dependent edges).
        grid_nodes:        Set of valid (x, y) positions on the board.
        state:             The puzzle instance to solve.
        independent_paths: Precomputed shortest-path lengths using only
                           independent (weight==1) edges.
        all_paths:         Precomputed shortest-path lengths using all edges.
    """

    def __init__(
        self,
        grid_graph: nx.DiGraph,
        state: State,
        independent_paths: dict | None = None,
        all_paths: dict | None = None,
    ):
        self.grid_graph = grid_graph
        self.grid_nodes: set[tuple[int, int]] = set(grid_graph.nodes)
        self.state = state
        self.independent_paths = independent_paths or {}
        self.all_paths = all_paths or {}

    @classmethod
    def from_env(
        cls,
        env_index: int,
        instance_index: int = 0,
        env_dir: Path | str = ENV_DIR,
    ) -> Game:
        """Load ``env_{env_index}.pkl`` and build a Game."""
        path = Path(env_dir) / f"env_{env_index}.pkl"
        with open(path, "rb") as f:
            env = pickle.load(f)

        inst = env["instances"][instance_index]
        state = State(
            target=inst["target"],
            target_robot=inst["target_robot"],
            helpers=inst["helper_robots"],
        )

        return cls(
            grid_graph=env["grid_graph"],
            state=state,
            independent_paths=env.get("independent_paths"),
            all_paths=env.get("all_paths"),
        )
