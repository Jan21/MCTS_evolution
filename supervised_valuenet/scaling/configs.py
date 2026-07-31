"""Registry of board configurations for the grid-size / robot-count scaling study.

Each config pins one (grid, robots, walls, board directory, board-id splits)
combination. Exactly one config applies per process: the RR_* variables from
`env(cfg)` must be in the environment BEFORE any repo module is imported,
because module-level constants read them at import time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def default_walls(grid: int) -> int:
    """Interior wall segments at stock density (48 @ 16x16), scaled by area."""
    return round(48 * (grid / 16) ** 2)


def parse_ids(spec: str) -> list[int]:
    """"0-9,20" -> [0..9, 20] (same format as nn.generate.parse_graphs)."""
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def compress_ids(ids) -> str:
    """Inverse of parse_ids: ids -> compact sorted "a-b,c" spec."""
    ids = sorted(ids)
    parts, i = [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        parts.append(str(ids[i]) if i == j else f"{ids[i]}-{ids[j]}")
        i = j + 1
    return ",".join(parts)


@dataclass(frozen=True)
class BoardConfig:
    name: str
    grid: int
    robots: int
    walls: int
    env_dir: str        # board pkl directory, relative to the repo root
    board_ranges: dict  # {"train"|"val"|"test"|"bench": "a-b,c-d" id spec}
    all_boards: str     # every board id the config owns, as an id spec
    legacy: bool = False  # True = the already-running 16x16 pipeline; read-only
    notes: str = ""

    @property
    def env_dir_abs(self) -> Path:
        return REPO / self.env_dir

    def ids(self, split: str) -> list[int]:
        return parse_ids(self.board_ranges[split])

    def all_ids(self) -> list[int]:
        return parse_ids(self.all_boards)


def env(cfg: BoardConfig) -> dict:
    """The process-level RR_* variables that select this config."""
    return {
        "RR_GRID": str(cfg.grid),
        "RR_ROBOTS": str(cfg.robots),
        "RR_WALLS": str(cfg.walls),
        # absolute so it survives subprocess cwd changes and run-dir chdirs
        "RR_ENV_DIR": str(cfg.env_dir_abs),
    }


def apply_env(cfg: BoardConfig) -> None:
    """Set the config's env vars; call BEFORE importing any repo module."""
    os.environ.update(env(cfg))


def shell_env(cfg: BoardConfig) -> str:
    """`env(cfg)` as a shell command prefix."""
    return " ".join(f"{k}={v}" for k, v in env(cfg).items())


# Non-legacy configs share one layout: 1200 boards, train 0-699, val 700-899,
# test/bench 900-1049 (1050-1199 spare).
_STD_RANGES = {"train": "0-699", "val": "700-899",
               "test": "900-1049", "bench": "900-1049"}


def _std(name: str, grid: int, robots: int, notes: str = "") -> BoardConfig:
    return BoardConfig(name=name, grid=grid, robots=robots,
                       walls=default_walls(grid), env_dir=f"environments_{name}",
                       board_ranges=dict(_STD_RANGES), all_boards="0-1199",
                       notes=notes)


CONFIGS = {c.name: c for c in [
    BoardConfig(
        name="g16r4", grid=16, robots=4, walls=48, env_dir="environments",
        board_ranges={"train": "0-95,1000-1799", "val": "1800-2399",
                      "test": "112-127,2400-2999", "bench": "2400-2549"},
        all_boards="0-127,1000-2999", legacy=True,
        notes="Baseline: boards/splits of the already-running 16x16 pipeline "
              "(nn.benchmark.SPLITS; bench = eval/data/bench450.jsonl boards). "
              "Its boards and data are pre-existing and never regenerated."),
    _std("g16r6", 16, 6, "Robot axis: 6 robots on stock 16x16 geometry."),
    _std("g16r8", 16, 8, "Robot axis: 8 robots on stock 16x16 geometry."),
    _std("g24r4", 24, 4, "Grid axis: 24x24 at stock wall density (108 walls)."),
    _std("g24r8", 24, 8, "Grid+robot axis: 24x24 with 8 robots."),
    _std("g32r4", 32, 4, "Stretch: 32x32 at stock wall density (192 walls)."),
    _std("g8r4", 8, 4, "NN-labeler debug rung (2026-07-31): 8x8 pipeline shakeout."),
    _std("g9r4", 9, 4, "NN-labeler debug rung (2026-07-31): odd-size step-up audit."),
    _std("g10r4", 10, 4, "NN-labeler debug rung (2026-07-31): mixed-size training."),
    _std("g11r4", 11, 4, "NN-labeler interpolation control (2026-07-31): held out of training."),
    _std("g12r4", 12, 4, "NN-labeler debug rung (2026-07-31): extrapolation audit."),
    _std("g13r4", 13, 4, "NN-labeler interpolation control (2026-07-31): held out of training."),
    _std("g14r4", 14, 4, "NN-labeler training-mix rung (2026-07-31)."),
    _std("g15r4", 15, 4, "NN-labeler interpolation control (2026-07-31): held out of training."),
]}


def get(name: str) -> BoardConfig:
    try:
        return CONFIGS[name]
    except KeyError:
        raise SystemExit(f"unknown config {name!r}; available: "
                         f"{', '.join(CONFIGS)}")
