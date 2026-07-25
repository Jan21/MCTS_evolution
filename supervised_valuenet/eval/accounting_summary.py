"""Fold the per-rung accounting runs into eval/results/compute_accounting.json.

Reads `runs/accounting/<cfg>.<set>/{backward,forward}.json` (produced by
`jobs/patterns/track1_accounting.slurm`, which runs both systems over the same
seeded subsample with `--count-slides`) and emits one record per
(rung, set, system).

**Median and tail, never the mean alone.** FINDINGS §25: over a 20-instance
base slice, one UNSOLVED backward puzzle spent 106,817 slide calls — 89.6% of
the whole slice's physics work — against a median of 74.5. A mean would
describe neither system. Each record therefore carries median, mean, p90, max,
the share of total work in the single most expensive instance, and the split
by phase, so a reader can see the distribution rather than a summary of it.

    PYTHONPATH=. python -m eval.accounting_summary
        [--runs runs/accounting] [--out eval/results/compute_accounting.json]

Analysis only.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO_RUNS = Path(__file__).resolve().parent.parent.parent / "runs" / "accounting"


def _pct(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[i]


def summarise(path, system_kind):
    payload = json.loads(Path(path).read_text())
    hits = [s for s in payload["systems"].values()
            if s.get("kind") == system_kind and s.get("rows")]
    if len(hits) != 1:
        return None
    rows = hits[0]["rows"]
    totals, buckets = [], {}
    for r in rows:
        sc = r.get("accounting", {}).get("slide_calls") or {}
        totals.append(sum(sc.values()))
        for b, v in sc.items():
            buckets[b] = buckets.get(b, 0) + v
    if not totals:
        return None
    s = sorted(totals)
    grand = sum(totals)
    scalar = {}
    for key, val in (rows[0].get("accounting") or {}).items():
        if key == "slide_calls":
            continue
        vals = [r["accounting"].get(key) for r in rows
                if isinstance(r["accounting"].get(key), (int, float))]
        if vals:
            scalar[key] = {"mean": statistics.mean(vals),
                           "median": statistics.median(vals)}
    return {
        "n": len(rows),
        "solved": sum(1 for r in rows if r.get("solved")),
        "mean_expansions": statistics.mean(
            [r["expansions"] for r in rows if r.get("expansions") is not None]),
        "slide_calls": {
            "median": statistics.median(totals),
            "mean": statistics.mean(totals),
            "p90": _pct(s, 0.90),
            "max": s[-1],
            "max_share_of_total": (s[-1] / grand) if grand else None,
            "by_phase_mean": {b: v / len(rows) for b, v in sorted(buckets.items())},
        },
        "other_counters": scalar,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=str(REPO_RUNS))
    p.add_argument("--out", default="eval/results/compute_accounting.json")
    a = p.parse_args()

    records = []
    root = Path(a.runs)
    for d in sorted(root.glob("*.*")) if root.exists() else []:
        if not d.is_dir():
            continue
        cfg, _, sset = d.name.partition(".")
        for fname, kind in (("backward.json", "backward"),
                            ("forward.json", "forward")):
            f = d / fname
            if not f.exists():
                continue
            got = summarise(f, kind)
            if got:
                got.update(rung=cfg, set=sset, system=kind, source=str(f))
                records.append(got)

    payload = {
        "unit": ("invocations of simulate.slide -- the one physics primitive "
                 "both planners bottom out in (forward successor generation; "
                 "backward realization / prefix-check / park-repair BFS). "
                 "Measured with eval.compare --count-slides over a seeded "
                 "subsample, both systems on the SAME instances."),
        "reporting_rule": ("median and tail, never the mean alone: the "
                           "backward distribution is heavy-tailed because a "
                           "single unsolved puzzle can dominate a slice "
                           "(FINDINGS 25)."),
        "caveat": ("counted runs are not wall-clock comparable to uncounted "
                   "ones; the headline rows in track1_rows.slurm carry no "
                   "counter for exactly that reason."),
        "records": records,
    }
    Path(a.out).write_text(json.dumps(payload, indent=1))

    if not records:
        print(f"no accounting runs under {root}; wrote empty {a.out}")
        return
    hdr = (f"{'rung':7s} {'set':9s} {'system':9s} {'n':>4s} {'exp':>8s} "
           f"{'slides med':>11s} {'mean':>11s} {'p90':>11s} {'max':>11s} "
           f"{'max share':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for r in records:
        sc = r["slide_calls"]
        print(f"{r['rung']:7s} {r['set']:9s} {r['system']:9s} {r['n']:4d} "
              f"{r['mean_expansions']:8.1f} {sc['median']:11,.0f} "
              f"{sc['mean']:11,.0f} {sc['p90']:11,.0f} {sc['max']:11,.0f} "
              f"{sc['max_share_of_total']*100:8.1f}%")
    print(f"\nwrote {a.out}  ({len(records)} records)")


if __name__ == "__main__":
    main()
