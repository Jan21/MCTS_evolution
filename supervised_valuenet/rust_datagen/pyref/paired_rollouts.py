"""Paired end-to-end backward rollouts, Python vs Rust, on IDENTICAL instances.

This is the DESIGN.md section 5.11 measurement driver (task C review, section
3, hard mandate): the per-decision replay of gate 3 feeds Rust the candidate
list Python enumerated -- including Python's `parent_support` -- so it can
never see the one documented divergence risk, which lives in candidate
GENERATION (`heuristics.propose` resolves `parent_support` from a Python set
whose iteration order Rust does not reproduce). Only full rollouts run
independently on both engines can expose it.

For each sampled instance this script:
  1. runs the REAL `nn.generate.rollout` (imported, not copied) on E1's lazy
     env -- the Python reference record stream;
  2. emits one `backward_rollout` work item with the identical instance.
The Rust engine then labels the same work file, and `compare_rollouts.py`
aligns the two record streams decision by decision.

Instance sampling is per-board seeded (`random.Random(seed*100003 + env_id)`,
same recipe as dump_decisions.py) so the corpus is deterministic under any
--workers. Kept semantics are production's: an instance is kept iff the
rollout emitted records. Python rollouts run under a generous SIGALRM cap
(default 900 s >> production's 120 s); timed-out instances are excluded from
label comparison and reported separately (they are gate-4 material, not label
divergence).

    python rust_datagen/pyref/paired_rollouts.py --config g16r6 \
        --instances 450 --n-boards 45 --seed 4242 --workers 16 \
        --out-prefix rust_datagen/pyref/out/s511/g16r6
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

PYREF_DIR = Path(__file__).resolve().parent
SV_DIR = PYREF_DIR.parent.parent
for _p in (str(SV_DIR), str(PYREF_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CHILD_FLAG = "RR_PYREF_CHILD_PAIRED"
DEFAULT_ENGINE = SV_DIR / "rust_datagen" / "target" / "release" / "datagen"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--instances", type=int, default=200,
                   help="target kept instances across boards (a floor)")
    p.add_argument("--n-boards", type=int, default=20)
    p.add_argument("--boards", default=None, help="explicit board id spec")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--workers", type=int, default=8, help="<=16")
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--max-candidates", type=int, default=14)
    p.add_argument("--max-iters", type=int, default=4000)
    p.add_argument("--max-frontier", type=int, default=40_000)
    p.add_argument("--dependent-edge-weight", type=int, default=2)
    p.add_argument("--instance-timeout", type=int, default=900)
    p.add_argument("--engine-bin", default=str(DEFAULT_ENGINE))
    p.add_argument("--threads", type=int, default=8, help="engine threads")
    p.add_argument("--budget-iters", type=int, default=None)
    p.add_argument("--skip-engine", action="store_true",
                   help="write python side + work file only")
    return p.parse_args(argv)


def main():
    a = parse_args()
    if os.environ.get(CHILD_FLAG) != "1":
        from scaling import configs                       # env-free import
        cfg = configs.get(a.config)
        env = {**os.environ, **configs.env(cfg), CHILD_FLAG: "1",
               "PYTHONPATH": str(SV_DIR)}
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                             *sys.argv[1:]], env=env, cwd=str(SV_DIR)).returncode
        sys.exit(rc)
    _child(a)


# ===========================================================================
# child (RR_* set)
# ===========================================================================

class _Timeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _Timeout()


def _board_worker(job):
    (gid, grid_data, n, robots, a_dict) = job
    a = argparse.Namespace(**a_dict)
    import common
    from skeleton.astar import AStar
    from nn.generate import random_instance, rollout

    env = common.build_env(grid_data, n, mode="lazy",
                           weight=a.dependent_edge_weight, table_workers=1)
    colors = common.PALETTE[:robots]
    rng = random.Random(a.seed * 100003 + gid)
    solver = AStar(max_iters=a.max_iters, max_frontier=a.max_frontier)
    signal.signal(signal.SIGALRM, _raise_timeout)

    lines, items = [], []
    kept = attempts = 0
    while kept < a.per_board and attempts < a.per_board * 4:
        attempts += 1
        st = random_instance(env, colors, rng)
        iid = f"e{gid}:i{attempts - 1}"
        t0 = time.time()
        signal.alarm(a.instance_timeout)
        status, recs = "ok", []
        try:
            recs = rollout(env, st, solver, gid, a.max_candidates)
            if not recs:
                status = "empty"
        except _Timeout:
            status, recs = "timeout", []
        except Exception as e:                             # noqa: BLE001
            status, recs = f"error:{type(e).__name__}", []
        finally:
            signal.alarm(0)
        wall = time.time() - t0
        if status == "ok":
            kept += 1
        lines.append({"iid": iid, "env_id": gid, "status": status,
                      "wall_s": round(wall, 3), "n_records": len(recs),
                      "instance": {
                          "target": common.xy(st.target),
                          "target_robot": [common.xy(st.target_robot.position),
                                           st.target_robot.color],
                          "helpers": [[common.xy(h.position), h.color]
                                      for h in st.helpers]},
                      "records": recs})
        items.append({
            "task": "backward_rollout",
            "id": iid,
            "board": common.board_obj(gid, n, grid_data),
            "target": common.xy(st.target),
            "target_robot": [common.xy(st.target_robot.position),
                             st.target_robot.color],
            "helpers": [[common.xy(h.position), h.color] for h in st.helpers],
            "max_candidates": a.max_candidates,
            "max_iters": a.max_iters,
            "max_frontier": a.max_frontier,
            "dependent_edge_weight": a.dependent_edge_weight,
            **({"budget": {"solver_iters": a.budget_iters}}
               if a.budget_iters is not None else {}),
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

    # warm table caches serially with the full pool (no nested pools)
    for gid, grid_data in boards:
        path = common.tables_cache_path(grid_data, n, a.dependent_edge_weight)
        if path.exists():
            continue
        t0 = time.time()
        G = common.build_graph_n(grid_data, n)
        common.reweight_dependent(G, a.dependent_edge_weight)
        common.load_or_build_tables(G, grid_data, n, a.dependent_edge_weight,
                                    workers=min(a.workers, 16))
        print(f"[tables] env_{gid} n={n}: built in {time.time() - t0:.1f}s",
              file=sys.stderr)

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
    py_path = Path(f"{prefix}.py.jsonl")
    work_path = Path(f"{prefix}.work.jsonl")
    n_att = n_ok = n_to = n_empty = n_err = 0
    with open(py_path, "w") as pf, open(work_path, "w") as wf:
        for gid, lines, items in results:
            for ln, it in zip(lines, items):
                pf.write(json.dumps(ln) + "\n")
                wf.write(json.dumps(it) + "\n")
                n_att += 1
                s = ln["status"]
                n_ok += s == "ok"
                n_to += s == "timeout"
                n_empty += s == "empty"
                n_err += s.startswith("error")
    print(f"[paired] {cfg.name}: {n_att} attempts on {len(boards)} boards "
          f"(ok={n_ok} empty={n_empty} timeout={n_to} error={n_err}) "
          f"in {time.time() - t0:.1f}s -> {py_path}", file=sys.stderr)

    manifest = {"config": cfg.name, "seed": a.seed, "boards": ids,
                "per_board": a.per_board, "max_candidates": a.max_candidates,
                "max_iters": a.max_iters, "max_frontier": a.max_frontier,
                "instance_timeout": a.instance_timeout,
                "attempts": n_att, "ok": n_ok, "empty": n_empty,
                "timeout": n_to, "error": n_err}
    with open(f"{prefix}.manifest.json", "w") as mf:
        json.dump(manifest, mf, indent=1)

    if a.skip_engine:
        return
    rust_path = Path(f"{prefix}.rust.jsonl")
    t1 = time.time()
    rc = subprocess.run([a.engine_bin, "run", "--work", str(work_path),
                         "--out", str(rust_path), "--threads", str(a.threads),
                         "--quiet"]).returncode
    if rc != 0:
        print(f"[paired] ENGINE FAILED rc={rc}", file=sys.stderr)
        sys.exit(rc)
    print(f"[paired] engine done in {time.time() - t1:.1f}s -> {rust_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
