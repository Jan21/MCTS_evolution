"""Independent replay certification of `eval.compare --dump-moves` results.

For every per-instance row that carries a realized primitive-move sequence
(backward rows: `moves`, a [color, direction] list; forward rows: `moves_seq`),
replay the sequence under FULL joint-state physics from the instance's start
positions -- every robot a blocker, walls decoded from the instance board's
`grid_data` -- and require that

  1. every move is a legal full slide (the robot actually leaves its cell and
     stops exactly where `simulate.slide` says it must),
  2. the sequence length equals the row's claimed move count
     (`realized_strict` for backward rows, `moves` for forward rows), and
  3. the target robot ends on the goal cell.

Deliberately independent of the planning/realization stack: imports are the
`simulate` physics layer (`slide`, `wall_sets`) plus stdlib ONLY -- no
realizer, no plan classes, no GridEnv. Boards are read straight from
`env_<env_id>.pkl` (stdlib pickle; the pkl's stored graph object materializes
transitively, but nothing from it is used -- only the raw `grid_data` wall
strings). Rows are matched to instances by position: `eval.compare` emits
exactly one row per instance, in instance-file order.

    PYTHONPATH=. python -m eval.replay_validate \
        --compare eval/results/comparison.json \
        [--instances eval/data/bench450.jsonl] [--env-dir environments]

`--instances` defaults to the compare file's `protocol.instances_file`;
`--env-dir` defaults to $RR_ENV_DIR, else `environments/` beside this repo.
Exit status: 0 all replayed rows pass; 1 any failure; 2 nothing to validate.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from math import isqrt
from pathlib import Path

from simulate import slide, wall_sets

# Robot slot -> color naming, mirrored from nn/gen_grids.py PALETTE (slot i of
# an instance's `positions` is PALETTE[i]); hardcoded so this validator never
# imports the planner stack.
PALETTE = ["Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Cyan",
           "Magenta"]
DIRECTIONS = ("up", "down", "left", "right")


def _load_board(env_dir, env_id, cache={}):
    """(walls_right, walls_down, size) for a board, from its pkl's grid_data."""
    if env_id not in cache:
        with open(Path(env_dir) / f"env_{env_id}.pkl", "rb") as f:
            grid_data = pickle.load(f)["grid_data"]
        size = isqrt(len(grid_data))
        assert size * size == len(grid_data), \
            f"env {env_id}: grid_data length {len(grid_data)} not a square"
        wr, wd = wall_sets(grid_data, size)
        cache[env_id] = (wr, wd, size)
    return cache[env_id]


def replay_row(inst, seq, claimed, env_dir):
    """Replay `seq` on `inst`'s board; returns (ok, reason)."""
    wr, wd, size = _load_board(env_dir, inst["env_id"])
    positions = [tuple(p) for p in inst["positions"]]
    colors = PALETTE[:len(positions)]
    pos = dict(zip(colors, positions))
    if len(set(positions)) != len(positions):
        return False, "instance has overlapping start positions"
    for j, mv in enumerate(seq):
        if (not isinstance(mv, (list, tuple)) or len(mv) != 2
                or mv[0] not in pos or mv[1] not in DIRECTIONS):
            return False, f"move {j}: malformed entry {mv!r}"
        color, direction = mv
        cur = pos[color]
        blockers = frozenset(p for c, p in pos.items() if c != color)
        nxt = slide(cur, direction, blockers, wr, wd, size)
        if nxt == cur:
            return False, (f"move {j}: {color} {direction} from {cur} is a "
                           "no-op (illegal move)")
        pos[color] = nxt
    if claimed is not None and len(seq) != claimed:
        return False, f"length {len(seq)} != claimed move count {claimed}"
    goal = tuple(inst["target"])
    tcolor = colors[inst["target_idx"]]
    if pos[tcolor] != goal:
        return False, (f"target robot {tcolor} ends at {pos[tcolor]}, "
                       f"not goal {goal}")
    return True, f"{len(seq)} moves"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compare", required=True,
                   help="comparison JSON produced by eval.compare --dump-moves")
    p.add_argument("--instances", default=None,
                   help="instance JSONL (default: protocol.instances_file)")
    p.add_argument("--env-dir", default=None,
                   help="board pkl directory (default: $RR_ENV_DIR, else "
                        "environments/ beside this repo)")
    a = p.parse_args()

    payload = json.loads(Path(a.compare).read_text())
    inst_path = a.instances or payload["protocol"]["instances_file"]
    instances = [json.loads(line) for line in
                 Path(inst_path).read_text().splitlines() if line.strip()]
    env_dir = a.env_dir or os.environ.get(
        "RR_ENV_DIR", Path(__file__).resolve().parent.parent / "environments")

    n_pass = n_fail = n_skip = 0
    for name, system in payload["systems"].items():
        rows = system.get("rows")
        if not rows:
            continue
        if len(rows) != len(instances):
            print(f"FAIL [{name}]: {len(rows)} rows != "
                  f"{len(instances)} instances")
            n_fail += 1
            continue
        for i, row in enumerate(rows):
            if isinstance(row.get("moves"), list):        # backward dump
                seq, claimed = row["moves"], row.get("realized_strict")
            elif isinstance(row.get("moves_seq"), list):  # forward dump
                seq, claimed = row["moves_seq"], row.get("moves")
            else:
                n_skip += 1
                continue
            ok, why = replay_row(instances[i], seq, claimed, env_dir)
            tag = "PASS" if ok else "FAIL"
            print(f"{tag} [{name}] row {i} env={instances[i]['env_id']}: {why}")
            n_pass += ok
            n_fail += not ok
    print(f"replay_validate: {n_pass} passed, {n_fail} failed, "
          f"{n_skip} rows without move dumps skipped")
    if n_fail:
        sys.exit(1)
    if not n_pass:
        print("replay_validate: NOTHING VALIDATED (was --dump-moves set?)")
        sys.exit(2)


if __name__ == "__main__":
    main()
