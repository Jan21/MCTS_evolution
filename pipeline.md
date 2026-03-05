# Benchmark Pipeline

## Overview

The benchmark evaluates how well an algorithm solves Ricochet Robots puzzles.
It loads grid environments, asks an algorithm to produce a `PartialPlan` for each,
validates the plan structure, and computes a cost metric.

See `pipeline.svg` for a visual diagram.

---

## Entry Point

```
python benchmark.py --n_val 10 --n_eval 100
```

- `n_val` — number of environments to validate (sanity check).
- `n_eval` — total number of environments to evaluate (includes the `n_val` envs).

---

## Step-by-Step

### 1. Load Environment

Each environment is a pickle file (`environments/env_{i}.pkl`) containing:
- A 16x16 grid graph with weighted edges (weight=1 independent, weight=100 dependent)
- Precomputed reachability matrices (exact and relaxed)
- A list of puzzle instances (different robot placements on the same grid)

`GridEnv.from_env(env_idx)` returns:
- `grid_env` — the loaded grid with pathfinding methods
- `state` — the puzzle instance (target position, target robot, helper robots)

### 2. Algorithm Produces a PartialPlan

The `Algorithm` ABC has one method:

```python
def solve(self, grid_env: GridEnv, state: State) -> PartialPlan
```

The algorithm receives the full environment and puzzle state, and returns a
`PartialPlan` — a DAG with these node types:

| Node Type   | Attributes     | Role                                   |
|-------------|----------------|----------------------------------------|
| `goal`      | `pos`          | Target cell (root, in-degree 0)        |
| `subgoal`   | —              | Groups a bottleneck + support pair      |
| `bottleneck`| `pos`, `robot` | Cell the target robot must pass through |
| `support`   | `pos`, `robot` | Cell a helper robot must occupy         |
| `leaf`      | `pos`, `robot` | Starting position of a robot (no children) |

Edges are either **structural** (subgoal->bottleneck, subgoal->support — no physical
movement) or **physical** (all others — represent actual robot movement).

### 3. Validation (Phase 1 only)

Runs on the first `n_val` environments. Uses `validate(plan, grid_env, state)` from
`validate_plan.py`, which performs 10 checks:

1. **DAG structure** — no cycles
2. **Node types & attributes** — valid types, required attrs present, `Robot_at` instances
3. **Goal node** — exactly one, is root, pos matches `state.target`
4. **Leaf nodes** — have pos+robot, no children, match actual robots in state
5. **Positions on grid** — all node positions are valid grid coordinates
6. **Edge types** — only valid (parent_type, child_type) pairs
7. **Subgoal children** — each subgoal has exactly 1 bottleneck + 1 support child
8. **Robot continuity** — child robot color matches what parent expects
9. **Completeness** — no open edges remain
10. **Edge reachability** — every physical edge has a valid path on the grid

If **any** validation fails, the entire benchmark aborts. This is a sanity check —
if the algorithm can't produce valid plans on a small sample, there's no point
evaluating it on the full set.

### 4. Evaluation (Metric Computation)

Runs on all `n_eval` environments. The metric is the **sum of physical edge costs**.

For each physical edge `(parent -> child)` in the plan, we compute
`compute_exact_shortest_path_length(start, end, support_pos)`.

The `(start, end, support_pos)` depends on the edge type:

#### Case A: child is a `subgoal`

The subgoal groups a bottleneck and a support. The physical movement is from
the bottleneck position to the parent position, using the support as a dependency.

```
start       = bottleneck child's pos
end         = parent's pos (goal or another bottleneck)
support_pos = support child's pos
```

Example: `goal(6,15) -> sg1 -> bn1(5,15) + sp1(6,14)`
The physical edge `goal -> sg1` costs `shortest_path(bn1.pos, goal.pos, sp1.pos)`.

#### Case B: child is `leaf`/`support`, parent is `bottleneck`

The robot moves from its current position to the bottleneck. Since the bottleneck
is reached via a dependent edge, we need the sibling support position.

```
start       = child's pos
end         = bottleneck's pos
support_pos = sibling support's pos (found via: bottleneck -> parent subgoal -> support child)
```

Example: `bn1(5,15) -> leaf_y(12,9)`
The physical edge costs `shortest_path(leaf_y.pos, bn1.pos, sp1.pos)`.

#### Case C: child is `leaf`/`support`, parent is NOT `bottleneck`

Simple independent movement — no support needed.

```
start       = child's pos
end         = parent's pos
support_pos = None
```

Example: `sp1(6,14) -> leaf_r(3,3)`
The physical edge costs `shortest_path(leaf_r.pos, sp1.pos, None)`.

#### Final metric

```
metric = sum of all physical edge costs
```

If any edge returns `None` (unreachable), the entire metric is `None`.

### 5. Results

The `run()` function returns:

```python
{env_index: (plan, metric)}
```

The `__main__` block prints a summary:
- Total environments, solved count, failed count
- Aggregate stats: total/avg/min/max cost
- Per-environment breakdown

---

## How to Add a New Algorithm

```python
from benchmark import Algorithm, run
from GridEnv import GridEnv, State
from partial_plan import PartialPlan

class MyAlgorithm(Algorithm):
    def solve(self, grid_env: GridEnv, state: State) -> PartialPlan:
        plan = PartialPlan()
        # ... build plan ...
        return plan

results = run(MyAlgorithm(), env_indices=list(range(128)), n_val=10, n_eval=128)
```

---

## File Dependencies

```
benchmark.py
  imports: GridEnv.py      (GridEnv, State, Robot_at)
           partial_plan.py  (PartialPlan)
           validate_plan.py (validate, STRUCTURAL_EDGE_PAIRS)

environments/
  env_0.pkl .. env_127.pkl  (loaded by GridEnv.from_env)
```
