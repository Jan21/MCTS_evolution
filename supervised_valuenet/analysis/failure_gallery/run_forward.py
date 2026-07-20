"""Solve the backward planner's structural-failure instances with the forward planner.

The ceiling probes proved that for these puzzles no strictly-playable subgoal
plan exists under the old plan vocabulary (categories NO_COMPLETE_PLAN /
NO_REALIZABLE_PLAN). This script asks the forward move planner (pure NN + A*,
no oracle) to actually solve each of them, producing the real move sequences
the failure-gallery diagnosis is built on.

Environment config comes from the usual RR_* env vars, set BEFORE launching
(the modules read them at import). Base scale needs none; 6-robot needs
RR_GRID=16 RR_ROBOTS=6 RR_WALLS=48 RR_ENV_DIR=.../environments_g16r6.

    # base (16x16, 4 robots)
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
    python3 analysis/failure_gallery/run_forward.py \
        --instances analysis/artifacts/ceiling_probe_instances.json \
        --probe analysis/artifacts/ceiling_probe_results.json \
        --ckpt move_planner/checkpoints/candidate_scored.ckpt \
        --out analysis/failure_gallery/forward_solutions_base.json

Every returned path is replayed move-by-move under simulate.slide before it is
written; an illegal or non-goal-reaching path is a hard error. Instances the
planner cannot solve at the standard budget are recorded honestly
(solved=false) and retried once at --fallback-budget (default 4x), stored
under separate fallback_* keys so the standard-budget result stays untouched.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

STRUCTURAL = ("NO_COMPLETE_PLAN", "NO_REALIZABLE_PLAN")


def replay(path, positions, target_idx, target, wr, wd, grid):
    """Replay (slot, dir) moves under simulate.slide. Returns the list of
    board states (positions tuples) after each move; raises on any illegal
    move or if the target robot does not finish on the target cell."""
    from simulate import slide

    pos = [tuple(p) for p in positions]
    states = [tuple(pos)]
    for i, (slot, d) in enumerate(path):
        blockers = frozenset(p for j, p in enumerate(pos) if j != slot)
        new = slide(pos[slot], d, blockers, wr, wd, grid)
        if new == pos[slot]:
            raise ValueError(f"move {i} ({slot},{d}) is illegal (no motion)")
        pos[slot] = new
        states.append(tuple(pos))
    if pos[target_idx] != tuple(target):
        raise ValueError("target robot does not finish on the target cell")
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--probe", required=True,
                    help="ceiling probe results json (categories, by idx)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--budget", type=int, default=1200)
    ap.add_argument("--fallback-budget", type=int, default=4800)
    ap.add_argument("--stride", type=int, default=1,
                    help="process every stride-th structural instance ...")
    ap.add_argument("--offset", type=int, default=0,
                    help="... starting at this offset (for parallel shards; "
                         "merge shard outputs with merge_shards.py)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    import torch
    torch.manual_seed(0)

    from move_planner.evaluate import nn_astar, Guide, walls_for
    from move_planner.state import NUM_ROBOTS
    from nn.gen_grids import GRID
    from eval.compare import _CountingGuide

    instances = json.load(open(a.instances))
    probe = json.load(open(a.probe))
    # idx alone is NOT unique at 6 robots (graded and beyond-oracle rows share
    # idx values); the join key is (tag, idx), present in both files.
    cat = {(r["tag"], r["idx"]): r["category"] for r in probe}
    assert len(cat) == len(probe), "duplicate (tag, idx) in probe results"
    work = [i for i in instances if cat.get((i["tag"], i["idx"])) in STRUCTURAL]
    n_total = len(work)
    work = work[a.offset::a.stride]
    print(f"[run_forward] {len(work)} of {n_total} structural instances "
          f"(of {len(instances)} probed; shard {a.offset}/{a.stride}); "
          f"grid={GRID} robots={NUM_ROBOTS} k={a.k} budget={a.budget}")
    assert work and len(work[0]["positions"]) == NUM_ROBOTS, (
        f"instance has {len(work[0]['positions'])} robots but the module "
        f"config says {NUM_ROBOTS}; set RR_* env vars before launching")

    guide = _CountingGuide(Guide(a.ckpt, a.device))
    dirnames = ("up", "down", "left", "right")
    rows = []
    for n, inst in enumerate(work):
        env_id = inst["env_id"]
        wr, wd = walls_for(env_id)
        positions = tuple(tuple(p) for p in inst["positions"])
        target = tuple(inst["target"])
        tidx = inst["target_idx"]

        def solve(budget):
            c0 = guide.calls
            t0 = time.perf_counter()
            cost, path = nn_astar(guide, env_id, positions, tidx, target,
                                  wr, wd, k=a.k, max_iters=budget)
            dt = time.perf_counter() - t0
            exp = max(0, guide.calls - c0 - 1) // 2
            if path is not None:
                path = [(s, dirnames[d]) for s, d in path]
                assert cost == len(path)
                replay(path, positions, tidx, target, wr, wd, GRID)  # hard check
            return cost, path, exp, dt

        cost, path, exp, dt = solve(a.budget)
        row = {
            "idx": inst["idx"], "tag": inst["tag"], "env_id": env_id,
            "d_star": inst["d_star"],
            "mode": inst.get("mode", "graded"),
            "category": cat[(inst["tag"], inst["idx"])],
            "positions": [list(p) for p in positions], "target": list(target),
            "target_idx": tidx,
            "budget": a.budget, "k": a.k,
            "solved": cost is not None, "moves": cost,
            "path": path, "expansions": exp, "seconds": round(dt, 3),
        }
        if cost is None and a.fallback_budget > a.budget:
            fcost, fpath, fexp, fdt = solve(a.fallback_budget)
            row.update(fallback_budget=a.fallback_budget,
                       fallback_solved=fcost is not None, fallback_moves=fcost,
                       fallback_path=fpath, fallback_expansions=fexp,
                       fallback_seconds=round(fdt, 3))
        rows.append(row)
        verdict = f"{cost} moves" if cost is not None else "UNSOLVED"
        print(f"  [{n+1}/{len(work)}] idx={inst['idx']}/{inst['tag']} "
              f"env={env_id} cat={row['category'][:7]} -> {verdict} "
              f"({exp} exp, {dt:.1f}s)", flush=True)

    solved = sum(r["solved"] for r in rows)
    out = {
        "protocol": {
            "ckpt": a.ckpt, "k": a.k, "budget": a.budget,
            "fallback_budget": a.fallback_budget,
            "grid": GRID, "robots": NUM_ROBOTS,
            "instances_file": a.instances, "probe_file": a.probe,
            "structural_categories": list(STRUCTURAL),
            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": ("every stored path replay-validated under simulate.slide; "
                     "d_star is a placeholder (0) for mode=beyond_oracle rows"),
        },
        "rows": rows,
    }
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"[run_forward] solved {solved}/{len(rows)} at budget {a.budget}; "
          f"wrote {a.out}")


if __name__ == "__main__":
    main()
