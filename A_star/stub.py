"""A* version template — copy this file to create a new version.

1. Copy to A_star/vN.py
2. Rename A_star_VX to A_star_VN
3. Create conf/a_starN.yaml (or reuse existing config)
4. Add `from A_star.vN import A_star_VN` to A_star/__init__.py
5. Implement solve()
"""
from __future__ import annotations

import time
from pathlib import Path

from omegaconf import OmegaConf

from GridEnv import GridEnv, State
from partial_plan import PartialPlan
from A_star.base import A_star, SolveResult, PlanEntry, PlanStats


class A_star_VX(A_star):

    def __init__(self, cfg=None):
        if cfg is None:
            cfg = OmegaConf.load(
                Path(__file__).resolve().parent.parent / "conf" / "a_star1.yaml")
        self.cfg = cfg

    def solve(self, grid_env: GridEnv, state: State) -> SolveResult:
        raise NotImplementedError("Copy this stub and implement solve()")
