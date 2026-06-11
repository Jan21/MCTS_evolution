# skeleton — clean A* over partial plans

A small, deliberately simple re-implementation of the partial-plan solver,
built to be the harness a neural network plugs into. The original `A_star/`
variants are left untouched; this is a parallel, cleaner take.

## The idea in one paragraph

Search the space of **partial plans**, ordering the frontier by plan cost. A
plan's cost is `g + h` for free: fixed segments carry their exact move count
(`g`), open segments carry an optimistic relaxed estimate (`h`). Pop the
cheapest plan; if it has no open segments it is complete and optimal (its cost
is then all-exact, and every other frontier entry costs at least as much).
Otherwise take an open segment and expand it: pin it to an exact path if one
exists, else ask `propose` for candidate subgoals, each spawning a child plan.
The relaxed estimate never exceeds the true decomposition, so the heuristic is
admissible and the first complete plan popped is optimal over everything
proposed.

## The two swap points (this is the whole point)

Both live in `heuristics.py` and have stable signatures:

| Hook | Meaning | NN role |
|------|---------|---------|
| `propose(env, goal, mover, helpers, support)` | which subgoals to consider for getting `mover` to `goal` | policy net (#1) |
| `score(env, candidate)` | how promising a candidate is (ranks expansions) | value net (#2) |

`astar.py` never touches the board directly for intelligence — it only calls
these two. Replace them with networks of the same signature and nothing in the
search changes.

## Search modes

`AStar(beam=None)` — full A*, optimal within the proposal set.
`AStar(beam=k)`   — keep only the `k` cheapest children per expansion.
`AStar(beam=1)`   — pure greedy rollout (the policy, no backtracking).

A `max_open` runaway guard ensures even `beam=1` always terminates.

## Results (envs 0–19, verified realizable by `simulate.py`)

| mode | solved | realizable | avg cost |
|------|--------|-----------|----------|
| full A* | 20/20 | 20/20 | 7.40 |
| beam=3  | 20/20 | 20/20 | 7.40 |
| beam=1  | 20/20 | 20/20 | 7.65 |

Matches or beats the exhaustive `A_star/v5` (e.g. env_4: 9 vs 10), and every
returned plan is confirmed physically realizable (up to blocker-clearing).

## Run

```bash
python -m skeleton.run --n 20            # full A*
python -m skeleton.run --n 20 --beam 1   # greedy policy rollout
```

Each env is solved, structurally validated (`validate_plan`), and simulated
for true realizability (`simulate.verify_plan`).

## Files

- `heuristics.py` — `propose` + `score` (the NN swap points)
- `astar.py` — the search (`AStar`)
- `run.py` — benchmark + realizability gate
