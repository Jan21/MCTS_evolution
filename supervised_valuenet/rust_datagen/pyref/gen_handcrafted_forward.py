"""Handcrafted forward replay cases (task B review, fixture gaps to close).

Emits `replay_forward_state` work items covering branches that random
sampling can never or rarely reach, then computes the Python reference
labels with `replay_python.py`'s own logic and writes a differ-ready dump:

  - start-on-target (`positions[target_idx] == target`): structurally
    impossible under the R+1-distinct-cells samplers, public-API branch;
    expected cost_to_go 0, empty optimal set.
  - cost_to_go == 1 / == 2 states: every non-goal child's capped solve runs
    with cost_cap = 0 / 1, exercising the `h0 > cost_cap` pre-check on the
    child solves of the optimal-set computation.
  - relaxed-unreachable target (sealed pocket on a synthetic board):
    cost_to_go null.
  - tight max_expansions (500) unsolved states: capped-None parity.
  - 64x64 fresh-board cases (start-on-target + short-distance states):
    Python-parity at the u64/u128 packing boundary sizes.

    python rust_datagen/pyref/gen_handcrafted_forward.py \
        --out rust_datagen/pyref/out/handcrafted/forward.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

PYREF_DIR = Path(__file__).resolve().parent
SV_DIR = PYREF_DIR.parent.parent
for _p in (str(SV_DIR), str(PYREF_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CHILD_FLAG = "RR_PYREF_CHILD_HANDF"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=99991)
    return p.parse_args(argv)


def main():
    a = parse_args()
    if os.environ.get(CHILD_FLAG) != "1":
        env = {**os.environ,
               "RR_GRID": "16", "RR_ROBOTS": "4", "RR_WALLS": "48",
               "RR_ENV_DIR": str(SV_DIR / "environments"),
               CHILD_FLAG: "1", "PYTHONPATH": str(SV_DIR)}
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                             *sys.argv[1:]], env=env, cwd=str(SV_DIR)).returncode
        sys.exit(rc)
    _child(a)


def _sealed_board(n=16):
    """Stock-density-free synthetic board with a fully sealed pocket at
    (7,7): walls on all four sides make the cell relaxed-unreachable."""
    grid = [""] * (n * n)
    for x in range(n):
        grid[0 * n + x] += "N"
        grid[(n - 1) * n + x] += "S"
    for y in range(n):
        grid[y * n + 0] += "W"
        grid[y * n + (n - 1)] += "E"
    px, py = 7, 7
    grid[py * n + px] += "NESW"
    grid = ["".join(sorted(set(g))) for g in grid]
    return grid


def _child(a):
    import common
    from simulate import wall_sets
    from move_planner import oracle

    rng = random.Random(a.seed)
    items = []

    def add(tag, board, n, positions, tidx, target, me=40_000):
        items.append({
            "task": "replay_forward_state",
            "id": f"hand:{tag}",
            "board": board,
            "positions": [list(map(int, p)) for p in positions],
            "target_idx": int(tidx),
            "target": [int(target[0]), int(target[1])],
            "max_expansions": me,
        })

    # ---- board A: stock env_0 --------------------------------------------
    grid16 = common.load_board_grid_data(SV_DIR / "environments", 0)
    b16 = common.board_obj(0, 16, grid16)
    wr, wd = wall_sets(grid16, 16)

    # start-on-target, several shapes (incl. another robot adjacent)
    add("sot_basic", b16, 16, [(3, 3), (5, 5), (9, 9), (12, 2)], 0, (3, 3))
    add("sot_idx3", b16, 16, [(3, 3), (5, 5), (9, 9), (12, 2)], 3, (12, 2))
    add("sot_adjacent", b16, 16, [(3, 3), (3, 4), (9, 9), (12, 2)], 0, (3, 3))

    # ctg 1 / 2 states: search random states, verify via the oracle
    found = {1: 0, 2: 0}
    cells = [(x, y) for y in range(16) for x in range(16)]
    while min(found.values()) < 4:
        pts = rng.sample(cells, 5)
        positions, target, tidx = tuple(pts[:4]), pts[4], rng.randrange(4)
        hdist = oracle.relaxed_target_dist(target, wr, wd, 16)
        if hdist.get(positions[tidx], oracle.INF) >= oracle.INF:
            continue
        d = oracle.solve(positions, tidx, target, wr, wd, 16, hdist, 40_000)
        if d in (1, 2) and found[d] < 4:
            add(f"ctg{d}_{found[d]}", b16, 16, positions, tidx, target)
            found[d] += 1

    # tight-cap unsolved states (max_expansions 500)
    n_unsolved = 0
    while n_unsolved < 4:
        pts = rng.sample(cells, 5)
        positions, target, tidx = tuple(pts[:4]), pts[4], rng.randrange(4)
        hdist = oracle.relaxed_target_dist(target, wr, wd, 16)
        if hdist.get(positions[tidx], oracle.INF) >= oracle.INF:
            continue
        if oracle.solve(positions, tidx, target, wr, wd, 16, hdist, 500) is None:
            add(f"cap500_{n_unsolved}", b16, 16, positions, tidx, target, me=500)
            n_unsolved += 1

    # ---- board B: synthetic sealed pocket --------------------------------
    sealed = _sealed_board(16)
    bs = {"env_id": 900000, "n": 16, "grid_data": sealed}
    # target in the pocket -> relaxed-unreachable for any robot outside
    add("sealed_target", bs, 16, [(1, 1), (14, 1), (1, 14), (14, 14)], 0, (7, 7))
    add("sealed_target_idx2", bs, 16, [(2, 2), (13, 1), (5, 10), (14, 13)], 2, (7, 7))
    # robot IN the pocket, target elsewhere: h0 finite? (pocket sealed both
    # ways in the relaxation walk) -> also unreachable
    add("sealed_robot", bs, 16, [(7, 7), (14, 1), (1, 14), (14, 14)], 0, (1, 1))
    # start-on-target inside the pocket: the ctg==0 branch precedes h0 checks
    add("sealed_sot", bs, 16, [(7, 7), (14, 1), (1, 14), (14, 14)], 0, (7, 7))

    # ---- board C: 64x64 fresh (E1's cached board 0) -----------------------
    p64 = PYREF_DIR / "cache" / "boards_n64_w768_s0" / "env_0.pkl"
    with open(p64, "rb") as f:
        grid64 = pickle.load(f)["grid_data"]
    b64 = {"env_id": 0, "n": 64, "grid_data": list(grid64)}
    wr64, wd64 = wall_sets(grid64, 64)
    add("n64_sot", b64, 64, [(10, 10), (20, 20), (30, 30), (40, 40)], 0, (10, 10))
    # short-distance 64x64 states (found by probing slides from a robot)
    cells64 = [(x, y) for y in range(64) for x in range(64)]
    n64 = 0
    while n64 < 4:
        pts = rng.sample(cells64, 5)
        positions, tidx = tuple(pts[:4]), rng.randrange(4)
        start = positions[tidx]
        blockers = [p for i, p in enumerate(positions) if i != tidx]
        from simulate import slide
        tgt = slide(start, rng.choice(["up", "down", "left", "right"]),
                    set(blockers), wr64, wd64, 64)
        if tgt == start:
            continue
        hd = oracle.relaxed_target_dist(tuple(tgt), wr64, wd64, 64)
        d = oracle.solve(positions, tidx, tuple(tgt), wr64, wd64, 64, hd, 40_000)
        if d is not None and d <= 3:
            add(f"n64_short_{n64}", b64, 64, positions, tidx, tuple(tgt))
            n64 += 1

    # ---- write items, compute python reference, merge ---------------------
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bare = out.with_suffix(".items.jsonl")
    with open(bare, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    pyres = out.with_suffix(".pyres.jsonl")
    rc = subprocess.run([sys.executable, str(PYREF_DIR / "replay_python.py"),
                         "--dump", str(bare), "--out", str(pyres),
                         "--workers", "4"]).returncode
    if rc != 0:
        print("[hand] python reference replay FAILED")
        sys.exit(rc)
    ref = {}
    with open(pyres) as f:
        for raw in f:
            r = json.loads(raw)
            ref[r["id"]] = r
    with open(out, "w") as f:
        for it in items:
            r = ref[it["id"]]
            it = dict(it)
            it["full_policy_max_ctg"] = 6
            it["python_labels"] = {
                "cost_to_go": r["cost_to_go"],
                "best_moves": r["optimal_moves"],
                "legal_moves": r["legal_moves"],
                "full": True,
                "depth": 0,
            }
            f.write(json.dumps(it) + "\n")
    print(f"[hand] {len(items)} handcrafted cases -> {out}")
    counts = {}
    for it in items:
        counts[it["id"].split(":")[1].split("_")[0]] = \
            counts.get(it["id"].split(":")[1].split("_")[0], 0) + 1
    print(f"[hand] case families: {counts}")


if __name__ == "__main__":
    main()
