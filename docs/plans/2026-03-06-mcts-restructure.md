# MCTS Restructure: Folder, Multi-Version Benchmark, Rich Return Type

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move MCTS versions into an `MCTS/` package with separate files, change `solve()` to return all complete plans with statistics, and update benchmark to compare multiple algorithms side-by-side.

**Architecture:** `MCTS/` becomes a Python package. Each version is a self-contained file (`v1.py`, `v2.py`) with its own helpers, duplicated intentionally for independence. `MCTS/base.py` has the abstract base class and the `SolveResult` dataclass. `benchmark.py` loops over a list of algorithms and prints a comparison table.

**Tech Stack:** Python, networkx, omegaconf (existing)

---

### Task 1: Create MCTS package with base module

**Files:**
- Create: `MCTS/__init__.py`
- Create: `MCTS/base.py`

**Step 1: Create `MCTS/base.py`**

Contains:
- `MCTS` abstract base class with new `solve()` signature returning `SolveResult`
- `SolveResult` dataclass: `best_plan`, `all_plans` (list of `PlanEntry`)
- `PlanEntry` dataclass: `plan`, `cost`, `stats` (dict with `wall_time`, `iteration`, `node_count`, `rollout_count`)

```python
# MCTS/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from GridEnv import GridEnv, State
from partial_plan import PartialPlan

@dataclass
class PlanStats:
    wall_time: float        # seconds since solve() start
    iteration: int          # MCTS iteration when found
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

class MCTS(ABC):
    @abstractmethod
    def solve(self, grid_env: GridEnv, state: State) -> SolveResult:
        ...
```

**Step 2: Create `MCTS/__init__.py`**

Re-exports base class and will later re-export version classes:

```python
from MCTS.base import MCTS, SolveResult, PlanEntry, PlanStats
```

**Step 3: Verify import works**

Run: `python -c "from MCTS import MCTS, SolveResult, PlanEntry, PlanStats; print('OK')"`

---

### Task 2: Move V1 (MCTS1.py) into MCTS/v1.py

**Files:**
- Read: `MCTS1.py` (current V1 implementation)
- Create: `MCTS/v1.py`
- Modify: `MCTS/__init__.py` (add re-export)

**Step 1: Copy MCTS1.py content to MCTS/v1.py**

Changes needed:
- Import `MCTS, SolveResult, PlanEntry, PlanStats` from `MCTS.base` instead of defining locally
- Rename class to `MCTS_V1`
- Update `solve()` to return `SolveResult` with all complete plans + stats
- Config path: `Path(__file__).resolve().parent.parent / "conf" / "mcts1.yaml"`
- Add tracking: `self._all_plans`, `self._start_time`, `self._iteration`, `self._node_count`, `self._rollout_count`
- In `_eval_complete`, when a valid complete plan is found, append a `PlanEntry` to `self._all_plans`
- In `solve()` return, build `SolveResult(best_plan=..., all_plans=self._all_plans)`

**Step 2: Update `MCTS/__init__.py`**

```python
from MCTS.base import MCTS, SolveResult, PlanEntry, PlanStats
from MCTS.v1 import MCTS_V1
```

**Step 3: Verify**

Run: `python -c "from MCTS import MCTS_V1; print('OK')"`

---

### Task 3: Move V2 (current MCTS.py) into MCTS/v2.py

**Files:**
- Read: `MCTS.py` (current V2 implementation)
- Create: `MCTS/v2.py`
- Modify: `MCTS/__init__.py` (add re-export)

**Step 1: Copy MCTS.py V2 content to MCTS/v2.py**

Same changes as Task 2:
- Import from `MCTS.base`
- Rename class to `MCTS_V2`
- Update `solve()` return type to `SolveResult`
- Add stats tracking (wall_time, iteration, node_count, rollout_count)
- Config path: `Path(__file__).resolve().parent.parent / "conf" / "mcts1.yaml"` (or mcts2.yaml if created)

**Step 2: Update `MCTS/__init__.py`**

```python
from MCTS.base import MCTS, SolveResult, PlanEntry, PlanStats
from MCTS.v1 import MCTS_V1
from MCTS.v2 import MCTS_V2
```

**Step 3: Verify**

Run: `python -c "from MCTS import MCTS_V2; print('OK')"`

---

### Task 4: Create stub file for future versions

**Files:**
- Create: `MCTS/stub.py`

Contains a minimal skeleton that future versions can copy:

```python
# MCTS/stub.py — copy this file to create a new MCTS version
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
                Path(__file__).resolve().parent.parent / "conf" / "mcts_vx.yaml")
        self.cfg = cfg

    def solve(self, grid_env: GridEnv, state: State) -> SolveResult:
        raise NotImplementedError("Copy this stub and implement solve()")
```

---

### Task 5: Update benchmark.py for multi-algorithm comparison

**Files:**
- Modify: `benchmark.py`

**Step 1: Update imports**

```python
from MCTS import MCTS, SolveResult, MCTS_V1, MCTS_V2
```

**Step 2: Update `Benchmark.run()` and `Benchmark.solve()`**

`solve()` now calls `algorithm.solve()` which returns `SolveResult`. Extract `best_plan` for validation/evaluation. Return `SolveResult` alongside metric.

`run()` signature: takes a list of `(name, algorithm)` pairs. Returns `{name: {env_idx: (SolveResult, metric)}}`.

**Step 3: Update `__main__` block**

- Define algorithms list: `[("V1", MCTS_V1()), ("V2", MCTS_V2())]`
- Run benchmark for each algorithm
- Print comparison table with per-env costs side by side
- Print summary stats per algorithm (avg, min, max)
- Print plan discovery stats (how many plans found, best plan iteration)

**Step 4: Verify**

Run: `python benchmark.py --n_val 3 --n_eval 10`
Expected: comparison table for V1 and V2

---

### Task 6: Clean up old files

**Files:**
- Delete: `MCTS.py` (replaced by `MCTS/v2.py`)
- Delete: `MCTS1.py` (replaced by `MCTS/v1.py`)

**Step 1: Remove old files**

Only after benchmark passes with the new structure.

**Step 2: Final verification**

Run: `python benchmark.py --n_val 3 --n_eval 10`
Expected: same results as before, no import errors.
