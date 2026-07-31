"""Frontier results stratified by an ORACLE-INDEPENDENT hardness proxy.

Third mitigation for `analysis/publishability.md` objection 0.3: the frontier
sets are the puzzles a move-level exhaustive search failed to grade, so
membership is defined by an oracle's failure and is adversarial to move-level
planners by construction. A referee's fair question is whether the backward
planner's frontier margin is an artifact of that selection or holds across the
set's own internal difficulty range.

The proxy must not consult the oracle, so it is computed from the instance and
its board alone:

  * **goal backing** — whether the goal cell has a wall on at least one of its
    four sides. In this domain a robot slides until something stops it, so a
    goal with no wall behind it can only be reached by parking a helper robot
    first. This is the domain's classic hardness signal and it is exactly what
    the backward planner's vocabulary is about, which makes it the sharpest
    place to look for a selection artifact.
  * **target travel** — Manhattan distance from the target robot's start cell
    to the goal, split at the set's own terciles.

Both are functions of (board, start positions, goal) only; neither uses d*,
solver status, or any result. Strata are therefore a partition of the frontier
set that carries none of the selection that defined it.

    PYTHONPATH=. python -m analysis.frontier_strata
        [--out analysis/artifacts/frontier_strata.json]

Analysis only: reads the archived result JSONs and the board pickles.
"""
from __future__ import annotations

import argparse
import json
import pickle
from math import isqrt
from pathlib import Path

from simulate import wall_sets
from eval.stats_tests import mcnemar_exact

# (rung, forward file/system, backward file/system) — the full-language
# backward row against its forward control, on each rung's frontier set.
RUNGS = ["g16r6", "g16r8", "g24r4", "g24r8", "g32r4"]
ENV_DIR = {"g16r4": "environments"}


def board(cfg, env_id, cache={}):
    key = (cfg, env_id)
    if key not in cache:
        d = ENV_DIR.get(cfg, f"environments_{cfg}")
        with open(Path(d) / f"env_{env_id}.pkl", "rb") as f:
            grid = pickle.load(f)["grid_data"]
        size = isqrt(len(grid))
        cache[key] = wall_sets(grid, size) + (size,)
    return cache[key]


def goal_backed(cfg, inst):
    """True if the goal cell has a wall on any of its four sides."""
    wr, wd, size = board(cfg, inst["env_id"])
    x, y = tuple(inst["target"])
    return (y == 0 or (x, y - 1) in wd            # wall above
            or y == size - 1 or (x, y) in wd      # wall below
            or x == 0 or (x - 1, y) in wr         # wall left
            or x == size - 1 or (x, y) in wr)     # wall right


def travel(inst):
    sx, sy = tuple(inst["positions"][inst["target_idx"]])
    gx, gy = tuple(inst["target"])
    return abs(sx - gx) + abs(sy - gy)


def solved_of(path, want):
    if not Path(path).exists():
        return None
    d = json.loads(Path(path).read_text())
    hits = [s for s in d["systems"].values()
            if s.get("kind") == want and s.get("rows")]
    if len(hits) != 1:
        return None
    return [bool(r.get("solved")) for r in hits[0]["rows"]]


def terciles(values):
    s = sorted(values)
    n = len(s)
    return s[n // 3], s[2 * n // 3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="analysis/artifacts/frontier_strata.json")
    a = p.parse_args()

    out = []
    hdr = (f"{'rung':7s} {'stratum':28s} {'n':>4s} {'bwd B2':>8s} "
           f"{'fwd':>8s} {'diff':>8s} {'p':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for cfg in RUNGS:
        inst_path = Path(f"scaling/data/{cfg}/bench.unsolved.jsonl")
        if not inst_path.exists():
            continue
        insts = [json.loads(l) for l in inst_path.read_text().splitlines()
                 if l.strip()]
        bwd = solved_of(f"scaling/results/{cfg}/comparison_ungraded_b2.json",
                        "backward")
        # g24r4 never had an old-language frontier file; its forward frontier
        # rows live in the same two-system comparison_ungraded_b2.json
        # (2026-07-29 chunked lane) -- mirror of eval/report_data.py.
        fwd_file = (f"scaling/results/{cfg}/comparison_ungraded_b2.json"
                    if cfg == "g24r4"
                    else f"scaling/results/{cfg}/comparison_ungraded.json")
        fwd = solved_of(fwd_file, "forward")
        if not bwd or not fwd or len(bwd) != len(insts) != len(fwd):
            print(f"{cfg:7s} SKIPPED (rows missing or misaligned)")
            continue
        dists = [travel(i) for i in insts]
        t1, t2 = terciles(dists)
        strata = {}
        for i, inst in enumerate(insts):
            band = ("travel low" if dists[i] <= t1
                    else "travel mid" if dists[i] <= t2 else "travel high")
            back = "goal walled" if goal_backed(cfg, inst) else "goal OPEN"
            for name in (back, band, f"{back} / {band}"):
                strata.setdefault(name, []).append(i)
        for name in sorted(strata):
            idx = strata[name]
            n = len(idx)
            sa = sum(bwd[i] for i in idx)
            sb = sum(fwd[i] for i in idx)
            b_only = sum(1 for i in idx if bwd[i] and not fwd[i])
            c_only = sum(1 for i in idx if fwd[i] and not bwd[i])
            pv = mcnemar_exact(b_only, c_only)
            print(f"{cfg:7s} {name:28s} {n:4d} {sa/n*100:7.1f}% "
                  f"{sb/n*100:7.1f}% {(sa-sb)/n*100:+7.1f}% {pv:8.4f}"
                  f"{'*' if pv < 0.05 else ''}")
            out.append({"rung": cfg, "stratum": name, "n": n,
                        "backward_b2_solved": sa, "forward_solved": sb,
                        "backward_rate": sa / n, "forward_rate": sb / n,
                        "diff": (sa - sb) / n, "mcnemar_p": pv,
                        "travel_terciles": [t1, t2]})
        print()

    payload = {
        "proxy": ("oracle-independent: goal-cell wall backing (a goal with no "
                  "adjacent wall needs a parked helper to stop on) and "
                  "Manhattan travel of the target robot, split at each "
                  "frontier set's own terciles. Neither uses d*, solver "
                  "status or any result."),
        "systems": ("backward = comparison_ungraded_b2.json (full language, "
                    "zero-shot ranking); forward = the forward control in "
                    "comparison_ungraded.json"),
        "strata": out,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
