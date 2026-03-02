# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCTS_evolution implements Monte Carlo Tree Search algorithms for the **Ricochet Robots** puzzle game. The core idea: use AND-OR tree search over partial plans represented as DAGs, where each plan decomposes the goal into subgoals (bottleneck + support positions) connected by segments.

## Commands

```bash
python -m pytest tests.py -v          # Run all tests
python tests.py                        # Run tests via unittest
python -c "from benchmark import run_benchmark, MockAlgorithm; print(run_benchmark(MockAlgorithm(), num_instances=10))"
```

## Architecture

```
partial_plan.py     # PartialPlan DAG class (NetworkX DiGraph)
benchmark.py        # Algorithm ABC, MockAlgorithm, validate_plan(), solve(), run_benchmark()
utils.py            # Environment loading, path matrices, subgoal extraction
tests.py            # 35 tests covering all modules
environments/       # 128 pickle files (env_0.pkl–env_127.pkl, ~1.7MB each)
prompts/            # Design specs and problem description
```

## Key Concepts

- **Partial Plan DAG**: Root is goal, leaves are current robot positions, intermediate nodes are subgoals grouping bottleneck + support pairs.
- **Edge types**: Physical edges (carry cost, validated) and structural edges (subgoal→bn/sp, cost=None, grouping only).
- **Validation**: Every physical segment must correspond to a path using ONLY independent (weight=1) edges. Dependent edges are implicit in subgoal nodes.
- **Plan cost**: Sum of costs on physical edges only.

### PartialPlan node types
- `goal` (pos)
- `subgoal` (entry_pos — where target enters final component via the dependent edge)
- `bottleneck` (pos — where target must reach BEFORE the dependent edge, robot)
- `support` (pos — where helper must be positioned, robot)
- `leaf` (pos — current robot position, robot)

### Physical edge position resolution
| Edge type | source_pos | dest_pos |
|-----------|-----------|----------|
| goal → subgoal | subgoal.entry_pos | goal.pos |
| bottleneck → leaf | leaf.pos | bottleneck.pos |
| support → leaf | leaf.pos | support.pos |
| goal → leaf (direct) | leaf.pos | goal.pos |

## Algorithm Extension

Subclass `benchmark.Algorithm` and implement `solve_instance(env_data, instance)`:
```python
class MyMCTS(Algorithm):
    def solve_instance(self, env_data, instance):
        # Build a PartialPlan, return {"plan": plan, "stats": {...}}
```

## Environment Pickle Structure

Each `env_X.pkl` dict: `graph_idx`, `grid_data`, `grid_graph` (nx.DiGraph, 256 nodes), `instances` (list with `helper_robots`, `target_robot`, `target`), `independent_paths`, `all_paths`.

Grid graph edges: weight=1 (independent, wall stops) or weight=100 (dependent, requires robot blocking at `dependent` position).

## Dependencies

- Python 3.11+, `networkx`
- Sibling project `ricochet_robots_simple` needed only for `Robot` class unpickling (added to sys.path automatically by `utils.py`)
