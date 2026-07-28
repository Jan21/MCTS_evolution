"""Seed-robustness summary for the g16r6 cap-20000 retrain (FINDINGS 39).

Reads the production g16r6 cap-20000 Track 1 rows plus the per-seed rows
produced by jobs/patterns/seed_study.slurm + seed_followup.sh (seeds
21/37/53; production seed: policy unseeded, value --torch-seed 11), and
writes analysis/artifacts/seed_spread.json with per-set solved counts,
min/median/max, and the spread. Skips seeds whose files are absent, so it
can run while lanes are still landing.

    PYTHONPATH=. python3 -m analysis.seed_spread
"""
import json
import os
import statistics

OUT = "analysis/artifacts/seed_spread.json"
BASE = "scaling/results/g16r6"
SETS = {
    "graded": "comparison_b2retrained_cap20000{}.json",
    "frontier": "comparison_ungraded_b2retrained_cap20000{}.json",
}
ARMS = {"production": "", "seed21": "_seed21", "seed37": "_seed37",
        "seed53": "_seed53"}


def solved_count(path):
    d = json.load(open(path))
    systems = [s for s in d["systems"].values()
               if s.get("kind") == "backward" and s.get("rows")]
    assert len(systems) == 1, (path, len(systems))
    rows = systems[0]["rows"]
    n_solved = sum(bool(r.get("solved")) for r in rows)
    agg = systems[0].get("aggregate") or {}
    if "solved" in agg:
        assert agg["solved"] == n_solved, (path, agg["solved"], n_solved)
    return n_solved, len(rows)


def main():
    out = {"sets": {}}
    for set_name, tpl in SETS.items():
        arms = {}
        for arm, suf in ARMS.items():
            path = os.path.join(BASE, tpl.format(suf))
            if not os.path.exists(path):
                continue
            s, n = solved_count(path)
            arms[arm] = {"file": path, "solved": s, "n": n}
        if not arms:
            continue
        counts = [v["solved"] for v in arms.values()]
        out["sets"][set_name] = {
            "arms": arms,
            "n_arms": len(arms),
            "min": min(counts),
            "median": statistics.median(counts),
            "max": max(counts),
            "spread": max(counts) - min(counts),
        }
        print(f"{set_name}: " + ", ".join(
            f"{k}={v['solved']}/{v['n']}" for k, v in arms.items())
            + f"  spread={out['sets'][set_name]['spread']}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(OUT + ".tmp", OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
