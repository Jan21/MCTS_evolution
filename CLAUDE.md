# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCTS-based planner for the Ricochet Robots puzzle game. Robots slide on a 16x16 grid until hitting walls or other robots. The goal is to find a plan (sequence of subgoal decompositions) that moves a target robot to a goal cell using helper robots as obstacles. Plans are DAGs, not move sequences — they decompose the problem into subgoals without simulating actual robot movement.

## Running

```bash
# Run benchmark with default settings (3 validation, 10 evaluation envs)
python benchmark.py

# Custom benchmark run
python benchmark.py --n_val 10 --n_eval 128 --no-cache

# Run validation examples
python validate_example.py

# Run pipeline example
python validation_pipeline_example.py
```

No package manager or build system — just plain Python files. Dependencies: `networkx`, `pickle` (stdlib).

## Architecture

The system has two layers: a **grid environment** that encodes the puzzle physics, and a **plan layer** that builds/validates/evaluates decomposition DAGs.

### Grid Layer (`GridEnv.py`)

`GridEnv` wraps a NetworkX DiGraph representing the 16x16 board. Edge types:
- `weight=1`: independent edges (robot can traverse without help)
- `weight=100` + `dependent=pos`: dependent edges (require a helper robot at `pos`)

Key precomputed data (loaded from `environments/env_*.pkl`):
- `reachability_matrix`: exact shortest paths ignoring dependent edges
- `relaxed_reachability_matrix`: shortest paths using dependent edges (penalty=1 per dependent edge)
- `precomputed_final_components`: cells reachable from goal without help
- `_bottleneck_support_pairs_cache`: valid (bottleneck, support) entry points into final components
- `_dependent_edge_cache`: extended graph analysis per (bottleneck, support) pair

Core dataclasses: `Robot_at(position, color)`, `State(target, target_robot, helpers)`, `Subgoal(bottleneck, support, goal_pos, target_robot, helper)`.

### Plan Layer

**`PartialPlan`** (`partial_plan.py`): A NetworkX DAG with typed nodes:
- `goal` (root, pos) → `subgoal` (groups bottleneck+support) → `bottleneck` (pos, robot) + `support` (pos, robot) → `leaf` (pos, robot, no children)
- Edges are `"open"` (relaxed cost) or `"fixed"` (exact cost). A complete plan has no open edges.
- Structural edges (`subgoal→bottleneck`, `subgoal→support`) carry no physical movement cost.

**`validate_plan.py`**: 10 checks (DAG structure, node types, goal/leaf consistency, grid positions, edge types, subgoal children, robot continuity, completeness, edge reachability). Returns `ValidationResult(passed, errors)`.

**`evaluate_plan.py`**: Sums exact shortest path lengths over physical edges. `support_pos` is passed only when the destination is a bottleneck node (reached via dependent edge). Returns `None` if any edge is unreachable.

**`benchmark.py`**: Two-phase pipeline. Phase 1 validates on `n_val` envs (aborts on failure). Phase 2 evaluates cost on `n_eval` envs. Implement `Algorithm.solve(grid_env, state) -> PartialPlan` to add new algorithms.


## Key Conventions

- Node IDs in plans are strings (e.g., `"goal"`, `"sg1"`, `"bn1"`, `"leaf_target"`)
- `robot` attributes on nodes must be `Robot_at` instances, not strings
- Physical movement direction in the DAG goes child→parent (leaf positions toward goal)
- 128 pre-generated environments in `environments/env_0.pkl` through `env_127.pkl`
- Cached GridEnv objects go in `environments/cache/`
