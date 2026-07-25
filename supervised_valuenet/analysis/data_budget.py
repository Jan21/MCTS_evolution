"""Per-configuration training-data budget, both systems, in one table.

Closes `analysis/publishability.md` objection 0.9 ("per-config label sets not
shown comparable — backward 51k–116k records across configs; different
currencies per system"). A referee cannot judge whether a head-to-head is fair
without knowing how much supervision each side received, and the two systems
count in different units: the backward planner learns from *decisions* (one
record per candidate at one search node), the forward planner from *moves*
(one record per state on an optimal trajectory). The table therefore reports
both the raw record counts and the boards/puzzles they were drawn from, and
states the unit rather than pretending one number compares.

Counting is by line, streamed, so the multi-hundred-MB label files are never
held in memory. Sizes are the on-disk bytes of the file actually used for
training (shards are ignored — they are the same records split for
parallel generation).

    PYTHONPATH=. python -m analysis.data_budget
        [--out analysis/artifacts/data_budget.json]

Analysis only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scaling.configs import CONFIGS

# (system, unit, candidate paths in preference order). The first file that
# exists is the one reported; the rest are recorded as alternates so a later
# regeneration is visible rather than silently replacing the number.
BASE_FILES = {
    "backward (old vocabulary)": ("decisions", ["nn/data/combined.jsonl"]),
    "backward (B1 vocabulary)": ("decisions", ["nn/data/combined_b1.jsonl"]),
    "backward (B2 vocabulary)": ("decisions", ["nn/data/combined_b2.jsonl"]),
    "forward": ("moves", ["move_planner/data/moves_combined.jsonl",
                          "move_planner/data/moves.jsonl"]),
}
SCALING_FILES = {
    "backward (old vocabulary)": ("decisions", ["backward.jsonl"]),
    "backward (B2 vocabulary)": ("decisions", ["backward_b2.rust.jsonl"]),
    "forward": ("moves", ["forward.jsonl"]),
}


def count_lines(path):
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def entry(label, unit, paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return {"system": label, "unit": unit, "file": str(p),
                    "records": count_lines(p), "bytes": p.stat().st_size}
    return {"system": label, "unit": unit, "file": paths[0],
            "records": None, "bytes": None, "status": "not generated"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="analysis/artifacts/data_budget.json")
    a = p.parse_args()

    rows = []
    for key, cfg in CONFIGS.items():
        base = key == "g16r4"
        train_boards = len(cfg.ids("train"))
        spec = BASE_FILES if base else SCALING_FILES
        for label, (unit, paths) in spec.items():
            full = paths if base else [f"scaling/data/{key}/{p}" for p in paths]
            e = entry(label, unit, full)
            e.update(config=key, train_boards=train_boards,
                     bench_instances=None)
            bench = Path(f"scaling/data/{key}/bench.jsonl") if not base \
                else Path("eval/data/bench450.jsonl")
            if bench.exists():
                e["bench_instances"] = count_lines(bench)
            rows.append(e)

    hdr = (f"{'config':7s} {'system':26s} {'unit':10s} {'records':>11s} "
           f"{'MB':>8s} {'train boards':>13s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        rec = "—" if r["records"] is None else f"{r['records']:,}"
        mb = "—" if r["bytes"] is None else f"{r['bytes']/1e6:.1f}"
        print(f"{r['config']:7s} {r['system']:26s} {r['unit']:10s} "
              f"{rec:>11s} {mb:>8s} {r['train_boards']:>13d}")

    payload = {
        "note": ("record counts are NOT comparable across systems: a backward "
                 "record is one candidate at one search decision, a forward "
                 "record is one state on an optimal move trajectory. Compare "
                 "within a system across configurations, and report the unit "
                 "whenever a count is quoted."),
        "rows": rows,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
