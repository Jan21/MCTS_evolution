"""MCTS version template — copy this file to create a new version.

1. Copy to MCTS/vN.py
2. Rename MCTS_VX to MCTS_VN
3. Create conf/mctsN.yaml (or reuse existing config)
4. Add `from MCTS.vN import MCTS_VN` to MCTS/__init__.py
5. Implement solve()
"""
from __future__ import annotations

import time
from pathlib import Path

from omegaconf import OmegaConf

from GridEnv import GridEnv, State
from partial_plan import PartialPlan
from MCTS.base import MCTS, SolveResult, PlanEntry, PlanStats


class MCTS_VX(MCTS):

    def __init__(self, cfg=None):
        if cfg is None:
            cfg = OmegaConf.load(
                Path(__file__).resolve().parent.parent / "conf" / "mcts1.yaml")
        self.cfg = cfg

    def solve(self, grid_env: GridEnv, state: State) -> SolveResult:
        raise NotImplementedError("Copy this stub and implement solve()")
