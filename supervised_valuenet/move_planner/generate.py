"""Generate move-based value+policy training records.

For each board we generate fresh geometry (`gen_grids` walls + slide graph, saved
as `env_{id}.pkl` so the encoder can read it back), sample random solvable
instances (random robot cells + a random target robot/cell), solve each optimally
with the heuristic A\\* oracle (`nn.move_bfs`), and emit one JSONL record per
decision on the optimal path:

    {env_id, robots[[x,y]xR in COLOR_ORDER slots], target[x,y], target_idx,
     cost_to_go, best_moves[[slot,dir]], legal_moves[[slot,dir]], depth, full}

`cost_to_go` is the EXACT optimal move count from that state (value target).
`best_moves` is the optimal-move set (policy target); `full` says whether it is the
complete set (small cost-to-go) or a single behaviour-cloning move (deep state).

Generation is embarrassingly parallel over boards -- one worker per board.

    python -m move_planner.generate --graphs 1000-1799,1800-2099,2400-2699 \
        --per-board 30 --workers 32 --out move_planner/data/moves.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool

from nn.gen_grids import gen_walls, build_graph, GRID
from simulate import wall_sets
from move_planner import oracle as move_bfs
from move_planner.state import apply_move, is_goal, legal_moves, NUM_ROBOTS
from train.encode import ENV_DIR
import random
import pickle


def parse_graphs(spec: str):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _board_seed(env_id, seed):
    return seed * 100003 + env_id


def gen_and_save_board(env_id, seed):
    """Build + persist a fresh board (lean pkl: grid_data + grid_graph) in ENV_DIR.

    NEVER overwrites: an existing pkl (e.g. a rich board from the backward
    pipeline) is kept and its grid_data reused, so the records emitted here always
    match the walls the encoder will read back from that pkl."""
    path = ENV_DIR / f"env_{env_id}.pkl"
    if path.exists():
        print(f"[gen_and_save_board] {path} exists -- keeping it (skip write)", flush=True)
        with open(path, "rb") as f:
            return pickle.load(f)["grid_data"]
    rng = random.Random(_board_seed(env_id, seed))
    grid_data = gen_walls(rng)
    G = build_graph(grid_data)
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"graph_idx": env_id, "grid_data": grid_data, "grid_graph": G}, f)
    return grid_data


def _rec(env_id, positions, target, tidx, ctg, best_moves, legal_moves, depth, full):
    return {
        "env_id": env_id,
        "robots": [[int(p[0]), int(p[1])] for p in positions],
        "target": [int(target[0]), int(target[1])],
        "target_idx": tidx,
        "cost_to_go": int(ctg),
        "best_moves": [[int(s), int(d)] for s, d in best_moves],
        "legal_moves": [[int(s), int(d)] for s, d in legal_moves],
        "depth": int(depth),
        "full": bool(full),
    }


def label_board(env_id, per_board, seed, max_expansions, full_policy_max_ctg,
                score_candidates=False):
    """Label random instances. If `score_candidates`, additionally emit, for every
    decision state, one value-only record per candidate MOVE = its resulting state with
    that state's EXACT optimal cost-to-go (mirrors the original's "score the candidates
    via A*", and gives the value net direct off-path coverage)."""
    grid_data = gen_and_save_board(env_id, seed)
    wr, wd = wall_sets(grid_data, GRID)
    rng = random.Random(_board_seed(env_id, seed) ^ 0x9E3779B9)
    cells_all = [(x, y) for y in range(GRID) for x in range(GRID)]
    out = []
    solved = attempts = 0
    max_attempts = per_board * 5
    while solved < per_board and attempts < max_attempts:
        attempts += 1
        cells = rng.sample(cells_all, NUM_ROBOTS + 1)
        positions = tuple(cells[:NUM_ROBOTS])
        target = cells[NUM_ROBOTS]
        tidx = rng.randrange(NUM_ROBOTS)
        hdist = move_bfs.relaxed_target_dist(target, wr, wd, GRID)
        if hdist.get(positions[tidx], move_bfs.INF) >= move_bfs.INF:
            continue  # target cell unreachable even in the relaxation -> unsolvable
        res = move_bfs.label_trajectory(positions, tidx, target, wr, wd, GRID, hdist,
                                        max_expansions=max_expansions, full_policy=True,
                                        full_policy_max_ctg=full_policy_max_ctg)
        if res is None:
            continue
        _d, recs = res
        solved += 1
        for r in recs:
            out.append(_rec(env_id, r["positions"], target, tidx, r["cost_to_go"],
                            r["best_moves"], r["legal_moves"], r["depth"], r["full"]))
            if not score_candidates:
                continue
            # score every candidate move by its resulting state's EXACT cost-to-go
            for slot, di in r["legal_moves"]:
                child = apply_move(r["positions"], slot, di, wr, wd, GRID)
                if child is None:
                    continue
                if is_goal(child, tidx, target):
                    c_ctg = 0
                else:
                    c_ctg = move_bfs.solve(child, tidx, target, wr, wd, GRID, hdist,
                                           max_expansions=max_expansions)
                    if c_ctg is None:
                        continue  # child unsolvable within budget -> skip
                c_legal = [(s, d) for s, d, _ in legal_moves(child, wr, wd, GRID)]
                out.append(_rec(env_id, child, target, tidx, c_ctg, [], c_legal,
                                r["depth"] + 1, False))   # value-only (policy skips it)
    return env_id, solved, attempts, out


def _worker(arg):
    env_id, cfg = arg
    return label_board(env_id, **cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graphs", default="1000-1199")
    p.add_argument("--per-board", type=int, default=30)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-expansions", type=int, default=40_000)
    p.add_argument("--full-policy-max-ctg", type=int, default=6)
    p.add_argument("--score-candidates", action="store_true",
                   help="also emit each candidate move's resulting state + its exact cost-to-go (value coverage)")
    p.add_argument("--out", default="move_planner/data/moves.jsonl")
    a = p.parse_args()

    ids = parse_graphs(a.graphs)
    cfg = dict(per_board=a.per_board, seed=a.seed, max_expansions=a.max_expansions,
               full_policy_max_ctg=a.full_policy_max_ctg, score_candidates=a.score_candidates)
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    n_rec = n_solved = done = 0
    t0 = time.time()
    with open(a.out, "w") as f, Pool(a.workers) as pool:
        for env_id, solved, attempts, recs in pool.imap_unordered(
                _worker, [(i, cfg) for i in ids]):
            for r in recs:
                f.write(json.dumps(r) + "\n")
            n_rec += len(recs)
            n_solved += solved
            done += 1
            if done % 25 == 0 or done == len(ids):
                dt = time.time() - t0
                print(f"[{done}/{len(ids)}] boards  solved_inst={n_solved} "
                      f"records={n_rec}  {dt:.0f}s  ({n_rec/max(dt,1e-9):.0f} rec/s)",
                      flush=True)
    print(f"done: {n_rec} records from {n_solved} solved instances over {len(ids)} "
          f"boards -> {a.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
