# Astar_evolution

A research workbench for solving **Ricochet Robots** puzzles by **partial-plan A\* search**. Twelve A\* variants are implemented against a shared subgoal-proposal API, then benchmarked head-to-head across a library of 128 precomputed environments.

> **TL;DR:** Instead of searching the raw move space, the planner searches over *plan structures* (DAGs of subgoals). A subgoal-scoring heuristic ranks candidate subgoals so well that all twelve search strategies converge to the same optimal plan. The interesting variable turned out to be proposal generation, not search.

---

## The Problem

In Ricochet Robots, robots slide in a cardinal direction until they hit a wall or another robot; they cannot stop mid-slide. To route the target robot to its goal cell you often need helper robots parked at exact positions to act as stoppers. Searching the raw state space (every robot times every direction) blows up fast, and most of that search is wasted on physically valid but strategically useless moves.

## The Solution

This project reformulates the puzzle as **subgoal planning over a graph of the instance**:

- The board is compiled into a directed graph. An edge that needs no helper is *independent*; an edge that needs a helper parked somewhere carries a `dependent` attribute naming the support cell.
- A **Partial Plan** is a DAG whose root is the goal and whose leaves are robot start positions. Intermediate nodes are **subgoal states** (a bottleneck cell plus its support cell). Edges are movement segments, each either *fixed* (exact cost known) or *open* (only a relaxed estimate known).
- A\* expands open segments by proposing subgoal states, scoring them with a heuristic, and pruning any partial plan costlier than the best complete plan found so far.

The core API lives on `GridEnv`: `propose_subgoal_states`, `subgoal_score`, `compute_exact_shortest_path_length`, `compute_relaxed_shortest_path_length`. Every variant is built on exactly these four methods, so they differ only in *how they search*, never in *what they can see*.

---

## Repository Layout

```
Astar_evolution/
├── GridEnv.py              # Instance graph, subgoal proposal + scoring, path-length oracles
├── partial_plan.py         # PartialPlan DAG (nodes, edges, cost)
├── A_star/
│   ├── base.py             # A_star ABC, SolveResult / PlanEntry / PlanStats
│   ├── v1.py … v12.py      # 12 search-strategy variants
│   └── stub.py             # Minimal template for a new variant
├── validate_plan.py        # Structural validation (no dependent edge traversed illegally)
├── evaluate_plan.py        # Plan cost = sum of exact shortest-path lengths over physical edges
├── benchmark.py            # Loads envs once, runs a variant, validates + evaluates
├── visualize_results.py    # Comparison tables + JSON + HTML report
├── environments/           # 128 precomputed env_*.pkl instances (+ cache/)
├── docs/variants/          # Per-variant design notes + 00_research_summary.md
├── prompts/                # Problem description + algorithm-generation prompts
└── benchmark_results.{json,html}
```

## Requirements

- Python 3.10+ (uses `tuple[int, int]` / `X | None` syntax)
- [`networkx`](https://networkx.org/) — instance graph and shortest paths
- [`omegaconf`](https://omegaconf.readthedocs.io/) — config loading

```bash
pip install networkx omegaconf
```

---

## Quick Start

Run the benchmark on the default variant (V1) over the first 20 environments:

```bash
python benchmark.py --n_eval 20 --n_val 3 --algorithms V1
```

This prints a per-environment cost table and plan-discovery stats, then writes `benchmark_results.json` and `benchmark_results.html`.

Solve a single instance from Python:

```python
from GridEnv import GridEnv
from A_star import A_star_V1
from validate_plan import validate
from evaluate_plan import evaluate_plan

grid_env, state = GridEnv.from_env(env_index=7)   # loads environments/env_7.pkl (cached)
result = A_star_V1().solve(grid_env, state)

best = result.best_plan
print("valid:", validate(best, grid_env, state).passed)
print("cost: ", evaluate_plan(best, grid_env))
print("nodes:", list(best.g.nodes(data=True)))
```

Compare several variants at once:

```python
from benchmark import Benchmark
from A_star import A_star_V1, A_star_V5

bench = Benchmark(env_indices=list(range(20)), n_val=3, n_eval=20)
for algo in (A_star_V1(), A_star_V5()):
    results = bench.run(algo)   # {env_index: (SolveResult, cost_or_None)}
```

---

## The Twelve Variants

All variants share the proposal API and converge to the same average cost (8.90 on the 20-env research set), with one deliberate exception. See `docs/variants/` for per-variant notes.

| Variant | Strategy | Idea |
|---------|----------|------|
| V1  | Greedy Best-First | Baseline. Priority queue by total plan cost. Fastest. |
| V2  | Multi-Heuristic + Edge Selection | Depth penalty, cheapest / most-constrained edge picking. |
| V3  | Randomized Multi-Restart | 10 restarts with shuffled proposal order. |
| V4  | Beam Search | Keep top-K plans per depth level. |
| V5  | Exhaustive DFS + Greedy Fallback | Up to 200K expansions; proves optimality. |
| V6  | Iterative Deepening A\* (IDA\*) | Cost-threshold bounded DFS. |
| V7  | K-th Best / Anti-Greedy | Forces non-optimal proposals. The control that gets **worse** (9.30). |
| V8  | Diverse Beam Search | Beam with structural-diversity enforcement. |
| V9  | Two-Level Lookahead | Evaluate sub-problem cost before committing. |
| V10 | Weighted A\* | `f = fixed + w * open`, several `w`. |
| V11 | Exact-Cost Reranking | Rerank proposals by exact (not relaxed) costs. |
| V12 | Greedy Rollout | Complete the whole plan per proposal, then compare. |

### Headline Finding

The `subgoal_score` heuristic (sum of three relaxed shortest-path lengths) predicts true plan cost so accurately that the best-scored proposal at every decision point already lies on the globally optimal plan. Exhaustive search (V5), full rollout (V12), and exact reranking (V11) find nothing better; only forcing *sub*-optimal proposals (V7) degrades the result. The conclusion: when the heuristic is this good, search strategy is irrelevant, and the real leverage is in **proposal generation** (`propose_subgoal_states` / `subgoal_score`). Full write-up in `docs/variants/00_research_summary.md`.

---

## Autonomous Research Harness

The twelve variants were not hand-authored one by one. They are the output of an **autonomous research loop**: an agent driven by `prompts/AUTORESEARCH_PROMPT.txt` that, for each variant, copies `A_star/stub.py`, implements a search strategy, smoke-tests it (3 envs), runs a full evaluation (20 envs), tunes one or two hyperparameters for at most five runs, and writes up findings in `docs/variants/<name>.md` before moving on (a ~15-minute budget per variant). The deliverable is the **landscape**, not a single winner: `A_star/v*.py` plus `docs/variants/*.md` describing what works, what does not, and why.

The harness operates under a fixed contract. These files are **immutable** during a research run; variants may only read from them:

```
GridEnv.py   validate_plan.py   evaluate_plan.py
benchmark.py   visualize_results.py
A_star/base.py   A_star/stub.py   environments/
```

Every variant lives in its own `A_star/vN.py` (class `A_star_VN`, subclassing `A_star`), must return a `SolveResult` whose plan passes `validate_plan.py`, and must **never post-process the finished plan** (no local-search cleanup after the A\* search ends). Only the raw search result counts.

> Historical note: the project began as MCTS (see the `MCTS V1-V5` commit and the now-stale `prompts/visualize_alg.md`, which still references a removed `MCTS/v1.py`) before pivoting to the partial-plan A\* formulation described here.

## How a Solve Works

```
                 GridEnv.from_env(idx)
                          │
                  ┌───────┴────────┐
                  │   GridEnv      │  instance graph G, dependent-edge cache,
                  │                │  bottleneck/support pairs, path oracles
                  └───────┬────────┘
                          │  state (target, target_robot, helpers)
                          ▼
      A_star_Vx.solve(grid_env, state)
                          │
        ┌─────────────────┴──────────────────┐
        │  loop over open segments:          │
        │   1. try exact fix  ───────────────┼──► fixed segment
        │   2. else propose_subgoal_states   │
        │   3. score + prune vs best cost    │
        │   4. push child PartialPlans       │
        └─────────────────┬──────────────────┘
                          ▼
                 SolveResult(best_plan, all_plans)
                          │
              ┌───────────┴────────────┐
              ▼                        ▼
       validate_plan            evaluate_plan
   (structurally legal?)     (sum exact segment costs)
```

`Benchmark.run` does this in two phases: a validation pre-check on `n_val` environments (aborts the whole run if any plan is structurally invalid), then evaluation-only over the remaining `n_eval`.

---

## Key Concepts

| Term | Meaning |
|------|---------|
| **Instance graph** | Directed graph of the board. Plain edges are independent; `dependent` edges name the support cell a helper must occupy. |
| **Final component** | Cells from which the robot can reach the goal with no helper (ancestors of the goal ignoring dependent edges). |
| **Bottleneck** | A cell inside the final component reachable from outside it only through a dependent edge. |
| **Support** | The helper position that enables a given bottleneck's dependent edge. |
| **Subgoal state** | A `(bottleneck, support)` pair. |
| **Open / Fixed segment** | A plan edge whose cost is a relaxed estimate (open) or an exact dependent-free path (fixed). |
| **Complete plan** | A partial plan with zero open segments; its cost is the sum of segment costs. |

Exact vs relaxed: the **exact** shortest path may not traverse any `dependent` edge (infinite if none exists); the **relaxed** path may, charging a penalty of 1 per dependent edge. A plan is valid as long as every segment corresponds to a dependent-free path; physical robot collisions are intentionally *not* checked.

---

## Adding a New Variant

1. Copy `A_star/stub.py` to `A_star/v13.py` and subclass `A_star`, implementing `solve(self, grid_env, state) -> SolveResult`.
2. Build plans with `partial_plan.PartialPlan`; expand open edges only through the `GridEnv` API methods.
3. Register it in `A_star/__init__.py` and add it to the list in `benchmark.py`.
4. Drop a design note in `docs/variants/`.

The research summary is blunt about where gains live: changing the search loop will not beat V1. To find cheaper plans you must change how proposals are generated, scored, or which subgoal structures are even considered.

---

## Limitations

- **Plans are not checked for physical realizability.** Two robots may be required in conflicting positions; validation only confirms no segment illegally crosses a dependent edge. This is by design (see `prompts/problem description.md`).
- **Environments are precomputed.** The 128 `env_*.pkl` files are fixed inputs; there is no board generator in this repo.
- **One research-scale result set.** The "all variants converge to 8.90" finding is over the first 20 environments with `dependent_edge_weight=2`; it is a property of this instance distribution, not a universal theorem.
- **Caches are weight-keyed.** Changing `dependent_edge_weight` invalidates `environments/cache/`; the cache regenerates on next load.

## Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `ModuleNotFoundError: networkx` / `omegaconf` | `pip install networkx omegaconf`. |
| `[validation] V… env_N: FAILED` then empty results | A variant produced a structurally invalid plan; the benchmark aborts that variant. Inspect the printed errors and the offending `solve`. |
| Stale or wrong costs after tuning weights | Delete `environments/cache/`; it is keyed by `dependent_edge_weight`. |
| `assert False, "No cached pairs found"` in `propose_subgoal_states` | The `(goal, support)` pair was not precomputed for this instance; usually means the segment state was constructed with an unexpected support. |

---

## License

No license file is currently present. This is a CIIRC research collaboration; check with the project owners before reuse or redistribution.
