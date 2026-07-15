"""Fixture dumper for gate 3, forward half (move_planner oracle parity).

Dumps, for boards from environments{,_g16r6,_g16r8,_g24r4,_g32r4}:
  (a) relaxed_target_dist fields (full per-cell array, idx = y*n+x),
  (b) solve cases incl. cost_cap and tight max_expansions boundary cases,
  (c) full label_trajectory outputs (records verbatim),
  (d) label_board-equivalent instance record streams WITH score_candidates
      (the exact interleaved stream generate.py's inner loop emits for a
      GIVEN instance -- the per-instance loop is replicated here verbatim;
      no move_planner file is modified or monkeypatched).

Output: rust_datagen/golden/forward/forward_<config>.json.gz (committed).

Run (parent mode spawns one subprocess per config with the RR_* variables
set BEFORE any repo import, mirroring scaling/configs.py):

    cd supervised_valuenet
    /home/p23131/.conda/envs/ph_main/bin/python3 \
        rust_datagen/pyref/gen_fixtures_forward.py
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

PYREF_DIR = Path(__file__).resolve().parent
RUST_DIR = PYREF_DIR.parent                  # rust_datagen/
SV_DIR = RUST_DIR.parent                     # supervised_valuenet/
OUT_DIR = RUST_DIR / "golden" / "forward"

# name, env_dir (relative to supervised_valuenet/), grid, robots, walls, board ids
CONFIGS = [
    ("g16r4", "environments",       16, 4,  48, [0, 3, 1000, 2400]),
    ("g16r6", "environments_g16r6", 16, 6,  48, [0, 300, 700, 900]),
    ("g16r8", "environments_g16r8", 16, 8,  48, [0, 300, 700, 900]),
    ("g24r4", "environments_g24r4", 24, 4, 108, [0, 300, 700, 900]),
    ("g32r4", "environments_g32r4", 32, 4, 192, [0, 300, 700, 900]),
]

MAX_EXP = 40_000  # generate.py --max-expansions default
FULL_POLICY_MAX_CTG = 6  # generate.py default


# ---------------------------------------------------------------------------
# helpers (child mode only; repo imports happen inside run_config)
# ---------------------------------------------------------------------------

def load_grid_data(env_dir: Path, env_id: int) -> list:
    with open(env_dir / f"env_{env_id}.pkl", "rb") as f:
        return pickle.load(f)["grid_data"]


def seal_cell(grid_data, n, cx, cy):
    """Copy of grid_data with cell (cx,cy) walled off on all four sides
    (mirrored onto the neighbours). Makes (cx,cy) relaxed-unreachable."""
    g = list(grid_data)

    def add(x, y, ch):
        s = set(g[y * n + x])
        s.add(ch)
        g[y * n + x] = "".join(sorted(s))

    add(cx, cy, "N"); add(cx, cy, "E"); add(cx, cy, "S"); add(cx, cy, "W")
    add(cx, cy - 1, "S"); add(cx + 1, cy, "W")
    add(cx, cy + 1, "N"); add(cx - 1, cy, "E")
    return g


def L(cells):
    return [[int(a), int(b)] for a, b in cells]


def moves_l(moves):
    return [[int(s), int(d)] for s, d in moves]


def _rec(env_id, positions, target, tidx, ctg, best_moves, legal_moves, depth, full):
    """Verbatim copy of move_planner/generate.py::_rec (importing generate.py
    would drag in train.encode; the function is the 9-field record schema)."""
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


# ---------------------------------------------------------------------------
# child mode: dump one config
# ---------------------------------------------------------------------------

def run_config(name: str) -> None:
    sys.path.insert(0, str(SV_DIR))
    from simulate import wall_sets  # noqa: E402
    from move_planner import oracle  # noqa: E402
    from move_planner.state import apply_move, is_goal, legal_moves  # noqa: E402

    idx = [c[0] for c in CONFIGS].index(name)
    _, env_dir, n, R, _walls, board_ids = CONFIGS[idx]
    rng = random.Random(0x5EED0 + idx)

    boards = [{"env_id": i, "grid_data": load_grid_data(SV_DIR / env_dir, i)}
              for i in board_ids]
    n_real = len(boards)
    if name == "g16r4":
        boards.append({"env_id": 990000,
                       "grid_data": seal_cell(boards[0]["grid_data"], n, 5, 5),
                       "synthetic": "board 0 with cell (5,5) sealed on all sides"})
    walls = [wall_sets(b["grid_data"], n) for b in boards]
    cells_all = [(x, y) for y in range(n) for x in range(n)]

    def sample_instance(exclude=()):
        pool = [c for c in cells_all if c not in exclude]
        cells = rng.sample(pool, R + 1)
        return tuple(cells[:R]), cells[R], rng.randrange(R)

    def hdist_flat(dist):
        return [dist.get((x, y)) for y in range(n) for x in range(n)]

    def min_expansions(bi, pos, ti, tgt, hd, cost_cap=oracle.INF, hi=MAX_EXP):
        """Minimal max_expansions at which solve succeeds (None if never <= hi).
        Valid because the search is identical until the cap aborts it, so
        success is monotone in max_expansions."""
        wr, wd = walls[bi]

        def ok(k):
            return oracle.solve(pos, ti, tgt, wr, wd, n, hd,
                                max_expansions=k, cost_cap=cost_cap) is not None

        if not ok(hi):
            return None
        lo, h = 1, hi
        while lo < h:
            mid = (lo + h) // 2
            if ok(mid):
                h = mid
            else:
                lo = mid + 1
        return lo

    # -- (a) relaxed_target_dist ------------------------------------------
    hdist_cases = []
    hd_boards = [0, 1, 2] if name == "g16r4" else [0, 1, 2, 3]
    for bi in hd_boards:
        tgt = rng.choice(cells_all)
        wr, wd = walls[bi]
        hdist_cases.append({"board": bi, "target": [tgt[0], tgt[1]],
                            "dist": hdist_flat(
                                oracle.relaxed_target_dist(tgt, wr, wd, n))})
    if name == "g16r4":  # sealed board: target (5,5) reachable from nowhere
        bi = len(boards) - 1
        wr, wd = walls[bi]
        hdist_cases.append({"board": bi, "target": [5, 5],
                            "dist": hdist_flat(
                                oracle.relaxed_target_dist((5, 5), wr, wd, n))})

    # -- (b) solve cases ----------------------------------------------------
    solve_cases = []
    plain = []  # (bi, pos, ti, tgt, hd, d_star) with d_star >= 2

    def add_solve_case(bi, pos, ti, tgt, hd, max_exp, cost_cap, d_star):
        wr, wd = walls[bi]
        kwargs = dict(max_expansions=max_exp)
        if cost_cap is not None:
            kwargs["cost_cap"] = cost_cap
        cost = oracle.solve(pos, ti, tgt, wr, wd, n, hd, **kwargs)
        solve_cases.append({"board": bi, "positions": L(pos), "target_idx": ti,
                            "target": [tgt[0], tgt[1]],
                            "max_expansions": max_exp, "cost_cap": cost_cap,
                            "cost": cost, "d_star": d_star})
        return cost

    for bi in range(n_real):
        for _ in range(6):
            pos, tgt, ti = sample_instance()
            wr, wd = walls[bi]
            hd = oracle.relaxed_target_dist(tgt, wr, wd, n)
            cost = add_solve_case(bi, pos, ti, tgt, hd, MAX_EXP, None, None)
            solve_cases[-1]["d_star"] = cost  # uncapped -> cost IS d_star
            if cost is not None and cost >= 2:
                plain.append((bi, pos, ti, tgt, hd, cost))
    while len(plain) < 6:  # top up (rare: too many trivial/unsolved samples)
        pos, tgt, ti = sample_instance()
        wr, wd = walls[0]
        hd = oracle.relaxed_target_dist(tgt, wr, wd, n)
        d = oracle.solve(pos, ti, tgt, wr, wd, n, hd, max_expansions=MAX_EXP)
        if d is not None and d >= 2:
            plain.append((0, pos, ti, tgt, hd, d))

    for bi, pos, ti, tgt, hd, d in plain[:2]:           # 8 cost_cap cases
        for cap in (d, d - 1, d // 2, d + 3):
            add_solve_case(bi, pos, ti, tgt, hd, MAX_EXP, cap, d)
    for bi, pos, ti, tgt, hd, d in plain[2:5]:          # 6 expansion-boundary
        k = min_expansions(bi, pos, ti, tgt, hd)
        assert k is not None and k >= 2, (name, k)
        for me in (k, k - 1):
            c = add_solve_case(bi, pos, ti, tgt, hd, me, None, d)
            solve_cases[-1]["min_expansions"] = k
            assert (c == d) if me == k else (c is None), (name, me, k, c)
    bi, pos, ti, tgt, hd, d = plain[5]                  # 2 combined cap+boundary
    kc = min_expansions(bi, pos, ti, tgt, hd, cost_cap=d)
    assert kc is not None and kc >= 2, (name, kc)
    for me in (kc, kc - 1):
        c = add_solve_case(bi, pos, ti, tgt, hd, me, d, d)
        solve_cases[-1]["min_expansions"] = kc
        assert (c == d) if me == kc else (c is None), (name, me, kc, c)

    # -- (c) label_trajectory ------------------------------------------------
    def dump_trajectory(bi, pos, ti, tgt, max_exp, full_policy, cap):
        wr, wd = walls[bi]
        hd = oracle.relaxed_target_dist(tgt, wr, wd, n)
        res = oracle.label_trajectory(pos, ti, tgt, wr, wd, n, hd,
                                      max_expansions=max_exp,
                                      full_policy=full_policy,
                                      full_policy_max_ctg=cap)
        case = {"board": bi, "positions": L(pos), "target_idx": ti,
                "target": [tgt[0], tgt[1]], "max_expansions": max_exp,
                "full_policy": full_policy, "full_policy_max_ctg": cap,
                "result": None}
        if res is not None:
            d_star, recs = res
            case["result"] = {"d_star": d_star, "records": [
                {"positions": L(r["positions"]), "cost_to_go": r["cost_to_go"],
                 "best_moves": moves_l(r["best_moves"]),
                 "legal_moves": moves_l(r["legal_moves"]),
                 "depth": r["depth"], "full": r["full"]} for r in recs]}
        trajectory_cases.append(case)
        return res

    trajectory_cases = []
    made = 0
    while made < 5:  # 5 standard cases (full policy, ctg cap 6)
        bi = made % n_real
        pos, tgt, ti = sample_instance()
        if dump_trajectory(bi, pos, ti, tgt, MAX_EXP, True,
                           FULL_POLICY_MAX_CTG) is None:
            trajectory_cases.pop()  # keep only solvable ones in this block
            continue
        made += 1
    for full_policy, cap in ((True, None), (False, None)):  # 2 variants
        while True:
            pos, tgt, ti = sample_instance()
            if dump_trajectory(0, pos, ti, tgt, MAX_EXP, full_policy, cap):
                break
            trajectory_cases.pop()
    while True:  # 1 unsolvable-at-cap case (result null)
        bi, pos, ti, tgt, hd, d = plain[rng.randrange(len(plain))]
        if d >= 2:
            res = dump_trajectory(bi, pos, ti, tgt, 1, True, FULL_POLICY_MAX_CTG)
            assert res is None
            break

    # -- (d) label_board-equivalent instances (score_candidates) -------------
    def label_instance_py(bi, env_id, pos, ti, tgt, max_exp, cap,
                          score_candidates):
        """Verbatim replication of generate.py::label_board's per-instance
        loop (the body of `while solved < per_board`, given the sample)."""
        wr, wd = walls[bi]
        hdist = oracle.relaxed_target_dist(tgt, wr, wd, n)
        if hdist.get(pos[ti], oracle.INF) >= oracle.INF:
            return {"status": "relaxed_unreachable"}
        res = oracle.label_trajectory(pos, ti, tgt, wr, wd, n, hdist,
                                      max_expansions=max_exp, full_policy=True,
                                      full_policy_max_ctg=cap)
        if res is None:
            return {"status": "unsolved"}
        d, recs = res
        out = []
        for r in recs:
            out.append(_rec(env_id, r["positions"], tgt, ti, r["cost_to_go"],
                            r["best_moves"], r["legal_moves"], r["depth"],
                            r["full"]))
            if not score_candidates:
                continue
            for slot, di in r["legal_moves"]:
                child = apply_move(r["positions"], slot, di, wr, wd, n)
                if child is None:
                    continue
                if is_goal(child, ti, tgt):
                    c_ctg = 0
                else:
                    c_ctg = oracle.solve(child, ti, tgt, wr, wd, n, hdist,
                                         max_expansions=max_exp)
                    if c_ctg is None:
                        continue  # child unsolvable within budget -> skip
                c_legal = [(s, d2) for s, d2, _ in legal_moves(child, wr, wd, n)]
                out.append(_rec(env_id, child, tgt, ti, c_ctg, [], c_legal,
                                r["depth"] + 1, False))
        return {"status": "solved", "d_star": d, "records": out}

    instance_cases = []

    def add_instance_case(bi, pos, ti, tgt, max_exp, want_status):
        env_id = boards[bi]["env_id"]
        got = label_instance_py(bi, env_id, pos, ti, tgt, max_exp,
                                FULL_POLICY_MAX_CTG, True)
        if got["status"] != want_status:
            return False
        instance_cases.append({"board": bi, "env_id": env_id,
                               "positions": L(pos), "target_idx": ti,
                               "target": [tgt[0], tgt[1]],
                               "max_expansions": max_exp, "full_policy": True,
                               "full_policy_max_ctg": FULL_POLICY_MAX_CTG,
                               "score_candidates": True, **got})
        return True

    made = 0
    while made < 3:  # 3 solved instances
        bi = made % n_real
        pos, tgt, ti = sample_instance()
        made += add_instance_case(bi, pos, ti, tgt, MAX_EXP, "solved")
    while True:  # 1 unsolved (cap hit) instance
        bi, pos, ti, tgt, hd, d = plain[rng.randrange(len(plain))]
        if d >= 2 and add_instance_case(bi, pos, ti, tgt, 1, "unsolved"):
            break
    if name == "g16r4":  # 1 relaxed_unreachable on the sealed board
        bi = len(boards) - 1
        pos, _, ti = sample_instance(exclude=((5, 5),))
        assert add_instance_case(bi, pos, ti, (5, 5), MAX_EXP,
                                 "relaxed_unreachable")

    # -- write ---------------------------------------------------------------
    doc = {"config": name, "n": n, "robots": R, "boards": boards,
           "hdist_cases": hdist_cases, "solve_cases": solve_cases,
           "trajectory_cases": trajectory_cases,
           "instance_cases": instance_cases}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"forward_{name}.json.gz"
    with open(out_path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            gz.write(json.dumps(doc).encode())
    print(f"[{name}] boards={len(boards)} hdist={len(hdist_cases)} "
          f"solve={len(solve_cases)} traj={len(trajectory_cases)} "
          f"instances={len(instance_cases)} -> {out_path}", flush=True)


# ---------------------------------------------------------------------------
# parent mode: one subprocess per config with RR_* set before any import
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="child mode: dump one config")
    a = ap.parse_args()
    if a.config:
        run_config(a.config)
        return

    procs = []
    for name, env_dir, grid, robots, walls, _ids in CONFIGS:
        env = dict(os.environ)
        env.update({
            "RR_GRID": str(grid), "RR_ROBOTS": str(robots),
            "RR_WALLS": str(walls),
            "RR_ENV_DIR": str(SV_DIR / env_dir),
            "PYTHONPATH": str(SV_DIR),
        })
        procs.append((name, subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--config", name],
            env=env, cwd=str(SV_DIR))))
    fails = [name for name, p in procs if p.wait() != 0]
    if fails:
        sys.exit(f"FAILED configs: {fails}")
    print("all configs dumped OK")


if __name__ == "__main__":
    main()
