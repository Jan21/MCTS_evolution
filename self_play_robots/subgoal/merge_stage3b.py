"""Merge sharded stage3b payloads. Shards are contiguous instance ranges, so the
rows simply concatenate; the aggregate is recomputed by the project's own
`payload` from the merged rows rather than averaged across shards."""
import json, sys, glob
from pathlib import Path
sys.path.insert(0, ".")
from subgoal.table import BENCH, load_bench
import subgoal.planner as P

paths = sorted(glob.glob(sys.argv[1]))
out = sys.argv[2]
bench = load_bench(BENCH)
rows, extra, name = [], None, None
for q in paths:
    d = json.load(open(q))
    k = list(d["systems"])[0]
    sysd = d["systems"][k]
    name = name or k
    extra = extra or {kk: vv for kk, vv in sysd.items() if kk not in ("rows", "aggregate")}
    rows.extend(sysd["rows"])
rows.sort(key=lambda r: r["idx"])
assert len(rows) == len(bench), f"{len(rows)} rows for {len(bench)} instances"
pay = P.payload(rows, bench, name, extra.get("k", 100), extra.get("expansions", 1200),
                extra=dict(merged_from=[Path(q).name for q in paths]))
agg = pay["systems"][list(pay["systems"])[0]]["aggregate"]
fc = sum(r.get("fin_calls", 0) or 0 for r in rows)
fh = sum(r.get("fin_hits", 0) or 0 for r in rows)
fa = sum(1 for r in rows if r.get("answer_from_finisher"))
print(f"[merged] solved {agg['solved']}/{len(bench)}, optimal {agg['pct_optimal_of_n']:.1f}% "
      f"of {len(bench)}, mean extra {agg['mean_regret']}")
print(f"[merged] finisher: {fc} calls, {fh} improved the incumbent, "
      f"{fa} instances answered BY the finisher")
Path(out).write_text(json.dumps(pay, indent=1) + "\n")
print("MERGED", out)
