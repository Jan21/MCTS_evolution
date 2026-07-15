"""Paired forward labeling, Python vs Rust, on IDENTICAL instances (gate 4).

Samples forward instances exactly like `move_planner.generate.label_board`
(per-board RNG, y-major cells, R+1 sample + randrange target), labels each
attempt on BOTH sides -- Python `oracle.label_trajectory` (records) or the
relaxed-unreachable pre-check, Rust a `forward_instance` work item -- and
compares:

  - instance status (solved / unsolved / relaxed_unreachable): the forward
    oracle is bit-mirrored, so even at tightened `--max-expansions` the
    expected divergence is ZERO;
  - full record streams of solved instances, compared as parsed JSON values
    field-for-field in order.

    python rust_datagen/pyref/paired_forward.py --config g16r4 \
        --instances 250 --n-boards 25 --max-expansions 2000 --seed 777 \
        --workers 16 --out-prefix rust_datagen/pyref/out/gate4/f16_me2000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

PYREF_DIR = Path(__file__).resolve().parent
SV_DIR = PYREF_DIR.parent.parent
for _p in (str(SV_DIR), str(PYREF_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CHILD_FLAG = "RR_PYREF_CHILD_PFWD"
DEFAULT_ENGINE = SV_DIR / "rust_datagen" / "target" / "release" / "datagen"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--instances", type=int, default=100,
                   help="target solved instances across boards (a floor)")
    p.add_argument("--n-boards", type=int, default=20)
    p.add_argument("--boards", default=None)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--workers", type=int, default=8, help="<=16")
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--max-expansions", type=int, default=40_000)
    p.add_argument("--full-policy-max-ctg", type=int, default=6)
    p.add_argument("--score-candidates", action="store_true")
    p.add_argument("--engine-bin", default=str(DEFAULT_ENGINE))
    p.add_argument("--threads", type=int, default=8)
    return p.parse_args(argv)


def main():
    a = parse_args()
    if os.environ.get(CHILD_FLAG) != "1":
        from scaling import configs
        cfg = configs.get(a.config)
        env = {**os.environ, **configs.env(cfg), CHILD_FLAG: "1",
               "PYTHONPATH": str(SV_DIR)}
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                             *sys.argv[1:]], env=env, cwd=str(SV_DIR)).returncode
        sys.exit(rc)
    _child(a)


def _board_worker(job):
    (gid, grid_data, n, robots, a_dict) = job
    a = argparse.Namespace(**a_dict)
    import common
    from simulate import wall_sets
    from move_planner import oracle

    wr, wd = wall_sets(grid_data, n)
    rng = random.Random((a.seed * 100003 + gid) ^ 0x9E3779B9)
    cells_all = [(x, y) for y in range(n) for x in range(n)]
    lines, items = [], []
    solved = attempts = 0
    while solved < a.per_board and attempts < a.per_board * 5:
        attempts += 1
        cells = rng.sample(cells_all, robots + 1)
        positions = tuple(cells[:robots])
        target = cells[robots]
        tidx = rng.randrange(robots)
        iid = f"e{gid}:a{attempts - 1}"
        t0 = time.time()
        hdist = oracle.relaxed_target_dist(target, wr, wd, n)
        recs = None
        if hdist.get(positions[tidx], oracle.INF) >= oracle.INF:
            status = "relaxed_unreachable"
        else:
            res = oracle.label_trajectory(
                positions, tidx, target, wr, wd, n, hdist,
                max_expansions=a.max_expansions, full_policy=True,
                full_policy_max_ctg=a.full_policy_max_ctg)
            if res is None:
                status = "unsolved"
            else:
                status = "solved"
                d_star, recs0 = res

                def _rec(positions, ctg, best, legal, depth, full):
                    return {
                        "env_id": gid,
                        "robots": [common.xy(p) for p in positions],
                        "target": common.xy(target),
                        "target_idx": int(tidx),
                        "cost_to_go": int(ctg),
                        "best_moves": [[int(s), int(d)] for s, d in best],
                        "legal_moves": [[int(s), int(d)] for s, d in legal],
                        "depth": int(depth),
                        "full": bool(full),
                    }

                recs = []
                for r in recs0:
                    recs.append(_rec(r["positions"], r["cost_to_go"],
                                     r["best_moves"], r["legal_moves"],
                                     r["depth"], r["full"]))
                    if not a.score_candidates:
                        continue
                    # mirror move_planner.generate.label_board's
                    # score-candidates loop exactly
                    from move_planner.state import apply_move, is_goal, legal_moves
                    for slot, di in r["legal_moves"]:
                        child = apply_move(r["positions"], slot, di, wr, wd, n)
                        if child is None:
                            continue
                        if is_goal(child, tidx, target):
                            c_ctg = 0
                        else:
                            c_ctg = oracle.solve(child, tidx, target, wr, wd,
                                                 n, hdist,
                                                 max_expansions=a.max_expansions)
                            if c_ctg is None:
                                continue
                        c_legal = [(s, d) for s, d, _ in
                                   legal_moves(child, wr, wd, n)]
                        recs.append(_rec(child, c_ctg, [], c_legal,
                                         r["depth"] + 1, False))
                solved += 1
        lines.append({"iid": iid, "env_id": gid, "status": status,
                      "wall_s": round(time.time() - t0, 3),
                      "records": recs})
        items.append({
            "task": "forward_instance", "id": iid,
            "board": common.board_obj(gid, n, grid_data),
            "robots": [common.xy(p) for p in positions],
            "target_idx": int(tidx), "target": common.xy(target),
            "max_expansions": a.max_expansions,
            "full_policy": True,
            "full_policy_max_ctg": a.full_policy_max_ctg,
            "score_candidates": bool(a.score_candidates),
        })
    return gid, lines, items


def _child(a):
    import math
    import common
    from scaling import configs
    cfg = configs.get(a.config)
    n, robots = cfg.grid, cfg.robots
    ids = (configs.parse_ids(a.boards) if a.boards
           else cfg.ids("bench")[: a.n_boards])
    boards = [(gid, common.load_board_grid_data(cfg.env_dir_abs, gid))
              for gid in ids]
    a.per_board = math.ceil(a.instances / len(boards))

    a_dict = vars(a)
    jobs = [(gid, gd, n, robots, a_dict) for gid, gd in boards]
    workers = max(1, min(a.workers, 16, len(jobs)))
    t0 = time.time()
    if workers == 1:
        results = [_board_worker(j) for j in jobs]
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=mp.get_context("fork")) as ex:
            results = list(ex.map(_board_worker, jobs))

    prefix = Path(a.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    n_att = counts = 0
    from collections import Counter
    counts = Counter()
    with open(f"{prefix}.py.jsonl", "w") as pf, \
         open(f"{prefix}.work.jsonl", "w") as wf:
        for gid, lines, items in results:
            for ln, it in zip(lines, items):
                pf.write(json.dumps(ln) + "\n")
                wf.write(json.dumps(it) + "\n")
                n_att += 1
                counts[ln["status"]] += 1
    print(f"[pfwd] {cfg.name} me={a.max_expansions}: {n_att} attempts "
          f"({dict(counts)}) in {time.time() - t0:.1f}s", file=sys.stderr)

    rust_path = f"{prefix}.rust.jsonl"
    rc = subprocess.run([a.engine_bin, "run", "--work", f"{prefix}.work.jsonl",
                         "--out", rust_path, "--threads", str(a.threads),
                         "--quiet"]).returncode
    if rc != 0:
        sys.exit(rc)

    # ---- compare ----------------------------------------------------------
    rust = {}
    with open(rust_path) as f:
        for raw in f:
            r = json.loads(raw)
            rust[r["id"]] = r
    status_map = {"solved": "solved", "unsolved": "unsolved",
                  "relaxed_unreachable": "relaxed_unreachable"}
    n_status_diff = n_rec_diff = n_inst = n_rec = 0
    examples = []
    with open(f"{prefix}.py.jsonl") as f:
        for raw in f:
            ln = json.loads(raw)
            r = rust.get(ln["iid"])
            n_inst += 1
            if r is None or status_map[ln["status"]] != r.get("status"):
                n_status_diff += 1
                if len(examples) < 5:
                    examples.append(f"{ln['iid']}: py={ln['status']} "
                                    f"rust={None if r is None else r.get('status')}")
                continue
            if ln["status"] == "solved":
                pr, rr = ln["records"], r["records"]
                n_rec += len(pr)
                if pr != rr:
                    n_rec_diff += 1
                    if len(examples) < 5:
                        k = next(i for i, (x, y) in enumerate(zip(pr, rr))
                                 if x != y) if len(pr) == len(rr) else -1
                        examples.append(f"{ln['iid']}: record diff at {k} "
                                        f"(lens {len(pr)}/{len(rr)})")
    verdict = "ALL GREEN" if n_status_diff == 0 and n_rec_diff == 0 else "FAIL"
    print(f"[pfwd] {cfg.name} me={a.max_expansions}: {n_inst} attempts, "
          f"status diffs={n_status_diff}, solved record-stream diffs="
          f"{n_rec_diff} ({n_rec} records) -> {verdict}")
    for ex in examples:
        print(f"  {ex}")
    with open(f"{prefix}.report.json", "w") as f:
        json.dump({"config": cfg.name, "max_expansions": a.max_expansions,
                   "attempts": n_inst, "status_diffs": n_status_diff,
                   "record_stream_diffs": n_rec_diff, "records": n_rec,
                   "examples": examples}, f, indent=1)
    sys.exit(0 if verdict == "ALL GREEN" else 1)


if __name__ == "__main__":
    main()
