"""Self-play record generation for the move planner (plain Expert Iteration).

The current net's own budgeted A* (`evaluate.nn_astar`) is the *expert*. On a generated
instance (forward-walk puzzle, `start_states.sample_start_state`) it either solves it or
not:

- SOLVED  -> emit one record per decision state on the found path: value = the plain
  remaining path length `T - j` (a Monte-Carlo return; the loose-early / self-tightening
  target that keeps the search admissible-high and finds solutions), policy = the
  committed move (one-hot); plus a goal-anchor record (`cost_to_go=0`).
- UNSOLVED -> DROP the instance (exactly as the supervised generator discards
  unsolvable instances). The solvable frontier then ratchets outward on its own.

No oracle labels are ever emitted (oracle is eval-only). Records use the exact JSONL
schema the supervised generator emits, so `net.MoveDataset` / `collate` are reused
verbatim:

    {env_id, robots[[x,y]xR COLOR_ORDER], target[x,y], target_idx, cost_to_go,
     best_moves[[slot,dir]], legal_moves[[slot,dir]], depth, full:False}
"""
from __future__ import annotations

from collections import Counter

from move_planner.state import legal_moves
from move_planner.evaluate import Guide, nn_astar
from move_planner.state import apply_move
from train.encode import walls_for
from nn.gen_grids import GRID
from move_planner_v2.start_states import sample_start_state


def _record(env_id, s, target, tidx, ctg, best_moves, wr, wd, depth):
    """One JSONL record in the supervised schema (`full=False` for all self-play)."""
    legal = [[int(slot), int(d)] for slot, d, _ in legal_moves(s, wr, wd, GRID)]
    return {
        "env_id": int(env_id),
        "robots": [[int(p[0]), int(p[1])] for p in s],
        "target": [int(target[0]), int(target[1])],
        "target_idx": int(tidx),
        "cost_to_go": float(min(max(ctg, 0), 63)),
        "best_moves": [[int(a), int(b)] for a, b in best_moves],
        "legal_moves": legal,
        "depth": int(depth),
        "full": False,
    }


def path_to_states(start: tuple, path: list, wr, wd, size: int = GRID) -> list:
    """Replay an A* plan -> visited states `[start, ..., goal]` (len == len(path)+1)."""
    states = [start]
    for slot, d in path:
        states.append(apply_move(states[-1], slot, d, wr, wd, size))
    return states


def exit_records(env_id: int, states: list, path: list, tidx: int, target: tuple,
                 wr, wd) -> list:
    """Solved path -> per-decision (value = remaining length, policy = committed move)
    records + one goal-anchor (cost_to_go=0)."""
    T = len(path)
    recs = []
    for j, s_j in enumerate(states[:-1]):          # decision states (skip the goal state)
        recs.append(_record(env_id, s_j, target, tidx, T - j, [tuple(path[j])], wr, wd, j))
    recs.append(_record(env_id, states[-1], target, tidx, 0, [], wr, wd, T))  # goal anchor
    return recs


def play_instance(guide: Guide, env_id: int, start: tuple, tidx: int, target: tuple,
                  wr, wd, cfg) -> tuple:
    """Run the expert on one instance -> `(records, solved)`. Solved -> ExIt records;
    unsolved -> `([], False)` (dropped)."""
    cost, path = nn_astar(guide, env_id, start, tidx, target, wr, wd,
                          k=cfg.k_top, max_iters=cfg.astar_iters)
    if path is None:
        return [], False
    states = path_to_states(start, path, wr, wd)
    return exit_records(env_id, states, path, tidx, target, wr, wd), True


def generate_iteration(guide: Guide, cfg, rng) -> tuple:
    """Sample `cfg.instances_per_iter` forward-walk puzzles, solve each with the expert,
    collect ExIt records -> `(records, stats)`.

    `stats = {solve_rate, n_records, mean_plan_len, n_instances, ctg_hist}` -- ctg_hist
    is the falsification test for the distribution fix (deep, cost-to-go>=5 states must
    now appear, matching the supervised ~38% tail)."""
    train_ids = cfg.train_ids()
    records, plan_lens = [], []
    n_instances = n_solved = 0
    for _ in range(cfg.instances_per_iter):
        env_id = rng.choice(train_ids)
        wr, wd = walls_for(env_id)
        inst = sample_start_state(env_id, cfg, rng)
        if inst is None:
            continue
        start, tidx, target = inst
        recs, solved = play_instance(guide, env_id, start, tidx, target, wr, wd, cfg)
        n_instances += 1
        if solved:
            records.extend(recs)
            n_solved += 1
            plan_lens.append(sum(1 for r in recs if r["best_moves"]))
    ctg_hist = Counter(int(round(r["cost_to_go"])) for r in records)
    stats = {
        "solve_rate": (n_solved / n_instances) if n_instances else 0.0,
        "n_records": len(records),
        "mean_plan_len": (sum(plan_lens) / len(plan_lens)) if plan_lens else 0.0,
        "n_instances": n_instances,
        "ctg_ge5_frac": round(sum(v for c, v in ctg_hist.items() if c >= 5)
                              / max(1, len(records)), 3),
    }
    return records, stats
