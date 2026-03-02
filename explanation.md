# Implementation Explanation

## 1. The Game: Ricochet Robots

A 16x16 grid board with walls on some cell edges. There are 4 robots: one **target robot** (Yellow) that must reach a designated **target cell**, and 3 **helper robots** (Red, Blue, Green) that serve as movable obstacles. Robots slide in a cardinal direction until they hit a wall or another robot -- they cannot stop mid-slide.

## 2. Game API (`game.py`)

The `game.py` module is the API layer for loading and accessing environment data. It defines three classes:

### Robot (frozen dataclass)
```python
Robot(name="Yellow", pos=(12, 9))
```
Our own lightweight representation, converted from the sibling project's Robot objects at pickle load time. Immutable. Attributes: `name` (str), `pos` ((x, y) tuple).

### State (frozen dataclass)
```python
State(target=(6, 15), target_robot=Robot(...), helper_robots=(Robot(...), ...))
```
An immutable game instance: the target position the target robot must reach, plus the starting positions of all robots. `helper_robots` is a tuple (not list) for immutability. One `State` per instance in the pickle.

### Game
```python
game = Game.from_pickle("environments/env_0.pkl")
games = Game.load_many(num=10)
```
A loaded environment. Attributes:
- `grid_graph`: NetworkX DiGraph (256 nodes, ~3900 edges)
- `grid_data`: Wall codes per cell
- `grid_nodes`: Frozenset of all (x, y) positions for fast membership checks
- `independent_paths`: Dict `(src, dst) -> int | None`
- `all_paths`: Dict `(src, dst) -> float`
- `states`: List of `State` objects
- `graph_idx`: Environment index

`from_pickle` handles the sys.path hack for deserializing the sibling project's Robot objects, converts them to our `Robot` dataclass, and computes path matrices if they're missing from the pickle.

## 3. The Grid Graph

Nodes are `(x, y)` tuples. Edges encode all possible single-slide moves:

**Independent edges** (`weight=1`): Moves that always work. The robot slides and stops because it hits a wall. Example: `(0,0) -> (0,5)` means a robot at `(0,0)` slides south and a wall stops it at `(0,5)`.

**Dependent edges** (`weight=100`): Moves that require another robot to be blocking. They carry an extra `dependent` attribute. Example: `(0,0) -> (0,1)` with `dependent=(0,2)` means a robot at `(0,0)` can slide to `(0,1)` **only if** another robot is already at `(0,2)` to block it. Without that blocker, the sliding robot would pass right through `(0,1)`.

From any cell, there are typically 2-4 independent edges (one per direction, stopping at walls) and many dependent edges (one for each possible intermediate stopping point if a helper were placed).

## 4. Path Computation

### Precomputed All-Pairs Matrices

Two precomputed matrices (256x256 = 65,536 entries each), stored on the `Game` object:

**`game.independent_paths[(src, dst)]`**: Shortest path length using only `weight=1` edges, counted in **hops** (number of slides). Returns `None` if unreachable without helpers. Example: `((0,0), (4,0)) = 1` (one direct slide), `((0,0), (1,0)) = None` (can't reach without a helper).

**`game.all_paths[(src, dst)]`**: Shortest weighted path using all edges (Dijkstra). Since dependent edges cost 100, the value reveals how many dependent moves are involved: a value of `102` means 1 dependent edge (100) + 2 independent hops (2). Returns `float('inf')` if completely unreachable.

### Dynamic Single-Pair Path Computation

**`compute_shortest_path(grid_graph, start, end, support_positions=None)`** (`utils.py`): Computes the shortest path (in hops) from `start` to `end` using BFS on a filtered subgraph. The subgraph includes:
- All independent (weight=1) edges -- always available
- Dependent (weight=100) edges whose `dependent` attribute is in `support_positions` -- activated, counted as 1 hop

Returns `int` (hop count) or `None` (unreachable).

When `support_positions` is None or empty, this is equivalent to `independent_paths[(start, end)]`. When a support position is provided, it models the scenario where a helper robot is already at that position, activating dependent edges that rely on that blocker.

Example: In env_1, the path from `(13,5)` to `(12,0)` is unreachable via independent edges alone (`independent_paths` returns `None`). But with a support robot at `(10,0)`, `compute_shortest_path(g, (13,5), (12,0), support_positions={(10,0)})` returns `6` -- the target robot can use dependent edges activated by the helper at `(10,0)`.

This function is used by validation for `bottleneck -> leaf` edges (where the support robot is known to be in position) and will be used by the MCTS when building plans.

## 5. Final Component and Subgoals

### Final Component

The **final component** for a target position is the set of all grid cells from which a robot can reach the target using **only independent edges**. Computed as `nx.ancestors(w1_graph, target) | {target}` on the subgraph of weight=1 edges. Once the target robot is anywhere in the final component, it can reach the goal without any help.

For some targets, the final component is just the target cell itself (meaning every path to the goal requires at least one dependent move). For others, it can include 10-20+ cells.

### Subgoals

A **subgoal** represents one way to get the target robot into the final component using a helper. The structure returned by `get_subgoals(grid_graph, target)` is:

```
{subgoal_pos: {helper_pos: [target_bot_positions...]}}
```

Reading this: if a helper robot is placed at `helper_pos`, then the target robot can slide from any of the `target_bot_positions` along a dependent edge and land at `subgoal_pos`, which is inside the final component. From there, the target reaches the goal independently.

Concretely, for a subgoal entry:
- `subgoal_pos` = where the target enters the final component (the **destination** of the dependent edge)
- `helper_pos` = where the helper must stand to create the block (the `dependent` attribute of that edge)
- `target_bot_pos` = where the target starts the slide (the **source** of the dependent edge, outside the final component)

## 6. The Partial Plan DAG

A `PartialPlan` is a directed acyclic graph representing a strategy to move robots into position. The DAG has edges pointing **from goal toward leaves** (parent -> child), which is the **decomposition direction**: the goal decomposes into subproblems.

### Node types

| Type | Key attributes | Meaning |
|------|---------------|---------|
| `goal` | `pos` | The target cell the target robot must reach |
| `subgoal` | `entry_pos` | Entry point into the final component via a dependent edge |
| `bottleneck` | `pos`, `robot` | Position the target robot must reach before the dependent edge fires (outside FC) |
| `support` | `pos`, `robot` | Position a helper robot must occupy to create the block |
| `leaf` | `pos`, `robot` | A robot's current position (where it starts) |

### Edge types

**Physical edges**: Represent actual robot movement. They connect nodes that have grid positions and carry a `cost` (the number of hops). Valid physical edge patterns: `goal -> subgoal`, `goal -> leaf`, `bottleneck -> leaf`, `support -> leaf`.

**Structural edges**: Pure grouping. A `subgoal -> bottleneck` and `subgoal -> support` edge says "this subgoal requires both this bottleneck and this support to be achieved." They carry `cost=None` and are not validated for physical traversability.

### Valid edge-type pairs (6 total)

| Edge pattern | Type | Meaning |
|-------------|------|---------|
| `goal -> leaf` | physical | Target robot reaches goal directly (no subgoal needed) |
| `goal -> subgoal` | physical | Path from subgoal entry to goal (inside final component) |
| `subgoal -> bottleneck` | structural | Subgoal groups its bottleneck |
| `subgoal -> support` | structural | Subgoal groups its support |
| `bottleneck -> leaf` | physical | Target robot reaches bottleneck from current position |
| `support -> leaf` | physical | Helper robot reaches support from current position |

No other edge-type combinations are valid. Every subgoal must have exactly one bottleneck child and one support child.

### Edge status

- `"fixed"`: The segment is resolved. Its cost is the **exact** shortest path length.
- `"open"`: The segment is unresolved. Its cost is a **relaxed** estimate (using all edges, including dependent ones with cost 1 each). An open edge means the plan is incomplete.

### A complete plan has no open edges.

### How physical edges map to grid positions

The DAG edges point goal-ward (parent -> child), but robots physically move **child -> parent** (from their current position toward the goal). `resolve_segment_positions(parent, child)` returns `(source_pos, dest_pos)` in the **physical movement direction**:

| DAG edge (parent -> child) | source_pos (robot starts here) | dest_pos (robot ends here) |
|---|---|---|
| `goal -> leaf` (direct) | `leaf.pos` | `goal.pos` |
| `goal -> subgoal` | `subgoal.entry_pos` | `goal.pos` |
| `bottleneck -> leaf` | `leaf.pos` | `bottleneck.pos` |
| `support -> leaf` | `leaf.pos` | `support.pos` |
| `subgoal -> bottleneck/support` | N/A (structural, returns `None`) | N/A |

### Concrete example: 6-node plan

```
goal(pos=(6,15))  --physical-->  sg1(entry_pos=(6,15))
                                    |--structural-->  bn1(pos=(5,15), robot=Yellow)
                                    |                    \--physical-->  leaf_target(pos=(12,9), robot=Yellow)
                                    \--structural-->  sp1(pos=(5,15), robot=Green)
                                                         \--physical-->  leaf_helper(pos=(1,8), robot=Green)
```

Physical segments validated:
1. `goal -> sg1`: Can Yellow slide from `(6,15)` to `(6,15)`? (entry_pos to goal, inside FC -- cost 0, trivially yes). Checked via `independent_paths`.
2. `bn1 -> leaf_target`: Can Yellow slide from `(12,9)` to `(5,15)`? Checked via `compute_shortest_path` **with support position** `(5,15)` (Green's support pos activates dependent edges).
3. `sp1 -> leaf_helper`: Can Green slide from `(1,8)` to `(5,15)`? Checked via `independent_paths` (helper moves independently).

The **dependent edge** itself (Yellow at `(5,15)` sliding to `(6,15)` because Green blocks at some position) is **implicit** in the subgoal node. It is not directly validated because it is by construction valid (it comes from `get_subgoals`).

### Plan cost

Sum of `cost` on all physical edges (structural edges have `cost=None` and are excluded). This counts the total number of hops across all segments.

### Why the plan "may not be realizable"

The plan validates each segment considering its immediate context (the support position for bottleneck edges), but does **not** check cross-segment interactions. For example, the helper might need to be at position P, but reaching P requires sliding through a cell that the target robot occupies. We explicitly do not care about this -- the plan is an optimistic decomposition, and realizability is a separate concern.

## 7. Validation

`validate_plan(plan, game, state)` checks three categories:

### Structural checks (`_validate_structure`)
- All nodes have a known type (`goal`, `subgoal`, `bottleneck`, `support`, `leaf`).
- Exactly one `goal` node; it is the DAG root (no predecessors).
- **Goal position matches `state.target`** -- the plan solves the right problem.
- All `leaf` nodes are actual leaves (no successors).
- **Each leaf's `(robot, pos)` matches an actual robot in the state** -- leaves reference real robot starting positions.
- Required attributes are present: `pos` on goal/bottleneck/support/leaf, `entry_pos` on subgoal, `robot` on bottleneck/support/leaf.
- All positions are valid grid nodes (exist in the 16x16 grid).
- Every subgoal has exactly one bottleneck child and one support child.
- Every edge is a valid type pair (only the 6 allowed patterns listed above).
- The graph is a DAG (no cycles).
- All nodes are reachable from the goal (no disconnected components).

### Physical edge checks
For every non-structural edge:
1. `resolve_segment_positions` must return a valid `(source_pos, dest_pos)`.
2. A traversable path must exist between those positions:
   - **`bottleneck -> leaf` edges**: Uses `compute_shortest_path` with the sibling support position. The support robot's position activates dependent edges, which can shorten the path or make a previously-unreachable bottleneck reachable. The sibling support is found by walking up from the bottleneck to its parent subgoal, then finding the support child (`_get_sibling_support_pos`).
   - **All other physical edges** (`goal -> leaf`, `goal -> subgoal`, `support -> leaf`): Uses the precomputed `independent_paths` matrix (O(1) lookup, independent edges only).
3. If the edge has a cost, it must match the actual shortest path length exactly.

### What validation does NOT check
- Whether the plan is realizable across segments (robots might block each other). This is by design -- we check each segment considering its immediate context (support position for bottleneck edges) but not cross-segment interactions.

## 8. Algorithm Interface

```python
class Algorithm(ABC):
    def solve_instance(self, game, state) -> dict:
        # Returns {"plan": PartialPlan, "stats": {...}}
```

Arguments:
- `game`: `Game` instance (grid graph, path matrices)
- `state`: `State` instance (target position, robot positions)

The `stats` dict contains:
- `cost`: Final plan cost (sum of physical edge costs), or `float('inf')` if unsolved.
- `wallclock_time`: Total seconds elapsed.
- `trace`: List of `(time, cost)` tuples recording each improvement found. For anytime algorithms (like MCTS), this tracks the best cost over time. For `MockAlgorithm`, this is a single entry.
- `auc`: Area under the cost-vs-time curve, computed from the trace.

### MockAlgorithm

A non-anytime reference implementation with three strategies tried in order:

1. **Direct path**: If `independent_paths[(target_pos, goal)]` is not `None`, build a 2-node plan (`goal <- leaf`). The target can already reach the goal without any help.
2. **Single subgoal**: Iterate over subgoals from `get_subgoals`. For each, check if (a) the entry-to-goal path is independent, (b) the target can independently reach the bottleneck position, and (c) some helper can independently reach the support position. Build a 6-node plan with the first viable combination.
3. **Fallback**: Return an empty plan with `cost=inf`.

Note: MockAlgorithm only uses `independent_paths` for bottleneck reachability (strategy 2b). A real MCTS would use `compute_shortest_path` with the support position to find more solutions where the bottleneck is only reachable with the support already in place.

## 9. AUC (Area Under Curve)

`compute_auc(trace, total_time)` integrates a step function:

```
cost
  |
15|xxxx
12|    xxxx
10|        xxxxxxxxxxxx
  |_________________________ time
  0   0.5  1.2          5.0
```

Each `(time, cost)` entry in the trace marks when the algorithm found a better solution. The area of each rectangle is `cost * (next_time - current_time)`. The last rectangle extends to `total_time`.

For the example above: `AUC = 15*0.4 + 12*0.7 + 10*3.8 = 52.4`

Lower AUC = better anytime performance (found good solutions faster).

If `trace` is empty (no solution found), `AUC = float('inf')`.

For `MockAlgorithm`, the trace has one entry at the end of computation, so `AUC ≈ 0` (no holding time). This is expected -- MockAlgorithm is not anytime.

## 10. Benchmark Runner

`run_benchmark(algorithm, num_instances=128)` operates in two phases:

**Phase 1 -- Validation** (first `n_small=10` environments): Loads each as a `Game`, runs the algorithm on every `State`, validates every non-empty plan via `validate_plan(plan, game, state)`. Raises `ValueError` on the first invalid plan. Results are kept.

**Phase 2 -- Evaluation** (remaining environments): Loads and solves `env_files[n_small:]` only. Results from phase 1 are reused (no double-solving).

Returns:
```python
{
    "average_cost": float,   # mean cost across solved instances
    "total_auc": float,      # sum of per-instance AUC
    "num_solved": int,       # instances where cost != inf
    "num_total": int,        # total instances across all environments
    "results": [...]         # list of per-instance result dicts
}
```

## 11. File Layout

| File | Purpose |
|------|---------|
| `game.py` | `Robot`, `State`, `Game` classes -- API layer for loading environments |
| `partial_plan.py` | `PartialPlan` DAG class with node/edge operations, position resolution, cost computation |
| `benchmark.py` | `Algorithm` ABC, `MockAlgorithm`, `validate_plan`, `compute_auc`, `solve`, `run_benchmark` |
| `utils.py` | `compute_shortest_path`, `compute_independent_paths`, `compute_all_paths`, subgoal extraction, final component, file listing |
| `tests.py` | 72 unit tests covering all modules |
| `environments/` | 128 pickle files (`env_0.pkl` -- `env_127.pkl`) |

## 12. Extending with a Real MCTS Algorithm

```python
class MyMCTS(Algorithm):
    def solve_instance(self, game, state):
        start_time = time.time()
        trace = []
        best_plan = None
        best_cost = float('inf')

        # ... MCTS search loop ...
        # On each improvement:
        #   trace.append((time.time() - start_time, new_cost))
        #   best_plan = new_plan
        #   best_cost = new_cost

        elapsed = time.time() - start_time
        return {
            "plan": best_plan or PartialPlan(),
            "stats": {
                "cost": best_cost,
                "wallclock_time": elapsed,
                "trace": trace,
                "auc": compute_auc(trace, elapsed),
            }
        }
```

The MCTS will build partial plans by choosing open segments and proposing subgoals for them. Each subgoal creates new segments (some fixed, some open). The search continues until a complete plan is found (no open edges), tracking improvements in the trace for AUC computation.

Key difference from MockAlgorithm: when computing costs for `bottleneck -> leaf` segments, the MCTS should use `compute_shortest_path(grid_graph, leaf_pos, bottleneck_pos, support_positions={support_pos})` instead of `independent_paths`. This accounts for the support robot's blocking effect, which can enable paths that are unreachable via independent edges alone, or shorten existing ones.
