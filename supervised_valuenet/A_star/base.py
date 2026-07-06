from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from GridEnv import GridEnv, State
from partial_plan import PartialPlan


@dataclass
class PlanStats:
    wall_time: float        # seconds since solve() start
    iteration: int          # A* iteration when found
    node_count: int         # tree nodes at time of discovery
    rollout_count: int      # rollouts completed so far


@dataclass
class PlanEntry:
    plan: PartialPlan
    cost: float
    stats: PlanStats


@dataclass
class SolveResult:
    best_plan: PartialPlan
    all_plans: list[PlanEntry] = field(default_factory=list)


class A_star(ABC):
    @abstractmethod
    def solve(self, grid_env: GridEnv, state: State) -> SolveResult:
        """Run A* and return a SolveResult with best plan and all found plans."""
        ...
