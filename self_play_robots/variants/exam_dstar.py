"""Exact optima (d*) for the unseen exam -- absolute grounding (owner
2026-08-21: "perfect play needs N moves on these puzzles; system X uses +Y").

    PYTHONPATH=supervised_valuenet:self_play_robots RR_GRID=24 RR_ROBOTS=4 \
    RR_WALLS=108 RR_ENV_DIR=<exam boards> python -m variants.exam_dstar \
        [--workers 8] [--pass2-mult 5]

Instrument: `move_planner.oracle.solve` with the PINNED bench caps (200k
expansions / 60 s) -- the exact same oracle and caps that defined every
config's graded/frontier split (scaling/data/*/bench.jsonl.meta.json), so
"graded-unseen" means precisely what "graded" means elsewhere. Instances the
caps defeat get a second pass at `--pass2-mult`x caps (an optimum found under
ANY cap is exact; the cap only bounds the search). Rows that still fail keep
d_star=null (frontier-unseen).

Sidecar: `results/variants/exam/g24r4_unseen.dstar.jsonl`, one row per exam
line, order-aligned and env_id/target-checked:
  {"i": line#, "env_id": ..., "d_star": int|null, "pass": 1|2|null,
   "expansions_cap": ..., "wall_s": ...}
`load_dstar()` returns the aligned list for consumers (report engine, gates).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import time
from pathlib import Path

from variants import REPO

EXAM = REPO / "self_play_robots" / "results" / "variants" / "exam" / "g24r4_unseen.jsonl"
OUT = EXAM.with_suffix("").with_suffix("")  # strip .jsonl
SIDECAR = EXAM.parent / "g24r4_unseen.dstar.jsonl"
CAP_EXP, CAP_S = 200_000, 60.0             # the pinned bench oracle caps


def load_dstar(path=SIDECAR):
    if not Path(path).is_file():
        return None
    return [json.loads(l) for l in open(path)]


class _Timeout(Exception):
    pass


def _label(args):
    i, inst, mult = args
    from move_planner.oracle import solve
    from train.encode import walls_for
    wr, wd = walls_for(inst["env_id"])
    positions = tuple(tuple(p) for p in inst["positions"])
    cap_e, cap_s = int(CAP_EXP * mult), CAP_S * mult
    def alarm(signum, frame):
        raise _Timeout()
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(int(cap_s) + 5)
    t0 = time.time()
    try:
        d = solve(positions, inst["target_idx"], tuple(inst["target"]),
                  wr, wd, size=24, max_expansions=cap_e)
    except _Timeout:
        d = None
    finally:
        signal.alarm(0)
    return i, inst["env_id"], (None if d is None else int(d)), round(time.time() - t0, 1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--pass2-mult", type=float, default=5.0)
    a = p.parse_args(argv)
    instances = [json.loads(l) for l in open(EXAM)]
    rows = {i: {"i": i, "env_id": inst["env_id"], "d_star": None, "pass": None,
                "expansions_cap": CAP_EXP, "wall_s": CAP_S}
            for i, inst in enumerate(instances)}
    ctx = mp.get_context("fork")
    for pass_no, mult in ((1, 1.0), (2, a.pass2_mult)):
        todo = [(i, instances[i], mult) for i, r in rows.items() if r["d_star"] is None]
        if not todo or (pass_no == 2 and a.pass2_mult <= 1):
            continue
        print(f"[dstar] pass {pass_no}: {len(todo)} instances at "
              f"{int(CAP_EXP * mult)} exp / {CAP_S * mult:.0f} s", flush=True)
        t0 = time.time()
        with ctx.Pool(a.workers) as pool:
            for k, (i, env_id, d, dt) in enumerate(
                    pool.imap_unordered(_label, todo, chunksize=1)):
                if d is not None:
                    rows[i].update(d_star=d, pass_=None)
                    rows[i]["pass"] = pass_no
                    rows[i]["expansions_cap"] = int(CAP_EXP * mult)
                    rows[i]["wall_s"] = CAP_S * mult
                if (k + 1) % 25 == 0:
                    done = sum(1 for r in rows.values() if r["d_star"] is not None)
                    print(f"[dstar]   {k + 1}/{len(todo)} ({done} labeled, "
                          f"{time.time() - t0:.0f}s)", flush=True)
    for r in rows.values():
        r.pop("pass_", None)
    labeled = [r for r in rows.values() if r["d_star"] is not None]
    tmp = SIDECAR.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for i in range(len(instances)):
            f.write(json.dumps(rows[i]) + "\n")
    os.replace(tmp, SIDECAR)
    ds = [r["d_star"] for r in labeled]
    print(f"[dstar] labeled {len(labeled)}/{len(instances)}; mean d* "
          f"{sum(ds) / len(ds):.2f}, max {max(ds)}", flush=True)
    print(f"EXAM DSTAR DONE {SIDECAR}")


if __name__ == "__main__":
    main()
