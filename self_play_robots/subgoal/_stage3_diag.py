"""Ad-hoc Stage 3 diagnostics for the write-up: stop reasons and difficulty
breakdown of one planner payload. Not part of the pipeline."""
import json, sys, collections
from pathlib import Path
sys.path[:0] = ["supervised_valuenet", "self_play_robots"]

for path in sys.argv[1:]:
    pay = json.loads(Path(path).read_text())
    sysd = list(pay["systems"].values())[0]
    rows = sysd["rows"]
    stop = collections.Counter(r["stop_reason"] for r in rows)
    by_d = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        by_d[r["d_star"]][1] += 1
        if r["solved"] and r["realized_strict"] == r["d_star"]:
            by_d[r["d_star"]][0] += 1
    print(f"== {Path(path).name}  k={pay['protocol']['k']} "
          f"exp={pay['protocol']['expansions']}")
    print("   stop reasons:", dict(stop))
    print("   optimal by d*:", {d: f"{v[0]}/{v[1]}" for d, v in sorted(by_d.items())})
    solved = [r for r in rows if r["solved"]]
    print(f"   solved {len(solved)}/{len(rows)}, optimal "
          f"{sum(1 for r in solved if r['realized_strict'] == r['d_star'])}, "
          f"mean expansions {sysd['aggregate']['expansions_mean']:.0f}, "
          f"encoder passes {sysd['aggregate']['h_queries_total']}, "
          f"candidates scored {sysd['aggregate']['h_candidates_total']}")
