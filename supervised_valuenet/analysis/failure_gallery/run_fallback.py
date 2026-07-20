"""Targeted 4x-budget retry for selected unsolved rows of a solutions file.

The inline fallback in run_forward.py is disabled for the 6-robot run (an
unsolved instance costs ~40 CPU-minutes at 4x budget, and only gallery
candidates need the retry). This script re-solves chosen unsolved rows at the
bigger budget and updates the solutions file in place (fallback_* keys), so
build_gallery.py can report them.

Same RR_* env-var contract as run_forward.py. Select rows with
--only tag:idx[,tag:idx...]; default = every unsolved row (use deliberately).

    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" RR_GRID=16 RR_ROBOTS=6 \
    RR_WALLS=48 RR_ENV_DIR=.../environments_g16r6 PYTHONPATH=. \
    python3 analysis/failure_gallery/run_fallback.py \
        --solutions analysis/failure_gallery/forward_solutions_g16r6.json \
        --ckpt lightning_logs/version_43/checkpoints/epoch=7-step=43224.ckpt \
        --only gra:85,bey:12
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solutions", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--only", default=None,
                    help="comma list tag:idx; default all unsolved rows")
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    import torch
    torch.manual_seed(0)
    from move_planner.evaluate import nn_astar, Guide, walls_for
    from nn.gen_grids import GRID
    from eval.compare import _CountingGuide
    from run_forward import replay  # same-dir import

    data = json.load(open(a.solutions))
    budget = data["protocol"]["budget"] * a.factor
    sel = None
    if a.only:
        sel = {(t, int(i)) for t, i in
               (x.split(":") for x in a.only.split(","))}
    guide = _CountingGuide(Guide(a.ckpt, a.device))
    dirnames = ("up", "down", "left", "right")
    n_done = 0
    for row in data["rows"]:
        if row.get("solved") or row.get("fallback_solved"):
            continue
        key = (row.get("tag", ""), row["idx"])
        if sel is not None and key not in sel:
            continue
        wr, wd = walls_for(row["env_id"])
        positions = tuple(tuple(p) for p in row["positions"])
        target = tuple(row["target"])
        c0 = guide.calls
        t0 = time.perf_counter()
        cost, path = nn_astar(guide, row["env_id"], positions,
                              row["target_idx"], target, wr, wd,
                              k=row["k"], max_iters=budget)
        dt = time.perf_counter() - t0
        exp = max(0, guide.calls - c0 - 1) // 2
        if path is not None:
            path = [(s, dirnames[d]) for s, d in path]
            replay(path, positions, row["target_idx"], target, wr, wd, GRID)
        row.update(fallback_budget=budget, fallback_solved=cost is not None,
                   fallback_moves=cost, fallback_path=path,
                   fallback_expansions=exp, fallback_seconds=round(dt, 3))
        n_done += 1
        print(f"  idx={row['idx']}/{row.get('tag','')} -> "
              f"{cost if cost is not None else 'UNSOLVED'} "
              f"({exp} exp, {dt:.0f}s)", flush=True)
    data["protocol"]["fallback_budget"] = budget
    Path(a.solutions).write_text(json.dumps(data, indent=1) + "\n")
    print(f"[run_fallback] retried {n_done} rows at budget {budget}; "
          f"updated {a.solutions}")


if __name__ == "__main__":
    main()
