"""Forward vs backward solve rate across the whole budget range, every rung.

The head-to-head tables report one budget point (1,200 expansions). A paper
needs the trend, and it is recoverable at ZERO compute: both searches are
deterministic and the expansion budget only truncates, so an instance solved
after `e` expansions is solved at every budget >= e. Solve rate at budget B is
therefore #(solved and expansions <= B) / n, read straight off the archived
per-instance rows (`eval/budget_curves_from_rows.py` builds those curves;
this module turns them into the paired forward-vs-backward table).

Budgets above the 1,200 cap come from the extended-budget probe jobs, which
are real runs on a seeded frontier subsample rather than reconstructions, and
are reported separately and labelled as such -- they are NOT on the same
instance set as the full-set columns.

    PYTHONPATH=. python -m eval.budget_table [--out eval/results/budget_table.json]
        [--markdown]

Analysis only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.report_data import RUNGS

CURVES = Path("eval/results/budget_curves_by_rung.json")
PROBES = Path("eval/results/budget_probe_summary.json")

# Which file/system carries each arm, per rung and set. Backward = the full
# plan language (what the study now calls simply "backward").
BASE = {"graded": {"backward": ("eval/results/final450_backward_b2.json", "backward"),
                   "forward": ("eval/results/comparison_forward.json", "forward")}}
SCALING = {
    "graded": {"backward": ("comparison_b2.json", "backward"),
               "forward": ("comparison_forward_control.json", "forward")},
    "frontier": {"backward": ("comparison_ungraded_b2.json", "backward"),
                 "forward": ("comparison_ungraded.json", "forward")},
}
# g24r4/g32r4/g24r8 keep their forward control inside comparison.json
FORWARD_FALLBACK = "comparison.json"

NAMES = {"g16r4": "16×16 · 4 robots (base)", "g16r6": "16×16 · 6 robots",
         "g16r8": "16×16 · 8 robots", "g24r4": "24×24 · 4 robots",
         "g24r8": "24×24 · 8 robots", "g32r4": "32×32 · 4 robots"}


def _curve(curves, relpath, kind):
    """(solve_rate list, n) for the one system of `kind` in `relpath`."""
    entry = curves["sources"].get(relpath)
    if not entry:
        return None
    hits = [(name, c) for name, c in entry.items()
            if (kind == "backward") == ("backward" in name.lower())]
    if not hits:
        return None
    # a file may hold several forward checkpoints; take the best final rate,
    # mirroring how the report picks the base forward cell
    name, c = max(hits, key=lambda nc: nc[1]["solve_rate_1200"])
    return c["solve_rate"], c["n"], name


def build():
    curves = json.loads(CURVES.read_text())
    budgets = curves["budgets"]
    rows = []
    for rung in RUNGS:
        key = rung["key"]
        sets = (BASE if rung.get("base") else SCALING)
        for set_name, arms in sets.items():
            rec = {"rung": key, "rung_label": NAMES.get(key, key),
                   "set": set_name, "budgets": budgets}
            ok = True
            for arm, (fname, kind) in arms.items():
                rel = fname if rung.get("base") else f"scaling/results/{key}/{fname}"
                got = _curve(curves, rel, kind)
                if got is None and arm == "forward" and not rung.get("base"):
                    rel = f"scaling/results/{key}/{FORWARD_FALLBACK}"
                    got = _curve(curves, rel, kind)
                if got is None:
                    ok = False
                    break
                rec[arm] = {"solve_rate": got[0], "n": got[1],
                            "system": got[2], "source": rel}
            if ok:
                rec["diff"] = [(b - f) for b, f in
                               zip(rec["backward"]["solve_rate"],
                                   rec["forward"]["solve_rate"])]
                rows.append(rec)
    probes = json.loads(PROBES.read_text())["probes"] if PROBES.exists() else []
    return {"method": ("solve rate at budget B = #(solved and expansions <= B)/n, "
                       "reconstructed from archived per-instance expansion counts; "
                       "valid because both searches are deterministic and the "
                       "budget only truncates. Backward = full plan language."),
            "extended_note": ("rows above 1,200 come from the extended-budget "
                              "probe jobs: REAL runs on a seeded frontier "
                              "subsample, not reconstructions, and not on the "
                              "same instance set as the full-set columns"),
            "budgets": budgets, "rows": rows, "extended": probes}


def markdown(payload):
    b = payload["budgets"]
    show = [5, 25, 100, 200, 400, 800, 1200]
    idx = [b.index(x) for x in show]
    out = []
    for r in payload["rows"]:
        out.append(f"\n### {r['rung_label']} — {r['set']} "
                   f"(n={r['backward']['n']})\n")
        out.append("| budget | " + " | ".join(str(b[i]) for i in idx) + " |")
        out.append("|---" * (len(idx) + 1) + "|")
        for arm in ("backward", "forward"):
            cells = " | ".join(f"{r[arm]['solve_rate'][i]*100:.1f}%" for i in idx)
            out.append(f"| {arm} | {cells} |")
        cells = " | ".join(f"{r['diff'][i]*100:+.1f}" for i in idx)
        out.append(f"| **difference** | {cells} |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="eval/results/budget_table.json")
    p.add_argument("--markdown", action="store_true")
    a = p.parse_args()
    payload = build()
    Path(a.out).write_text(json.dumps(payload, indent=1))
    if a.markdown:
        print(markdown(payload))
    else:
        b = payload["budgets"]
        show = [5, 25, 100, 400, 1200]
        idx = [b.index(x) for x in show]
        hdr = f"{'rung':22s} {'set':9s} {'arm':9s} " + " ".join(
            f"{b[i]:>7d}" for i in idx)
        print(hdr); print("-" * len(hdr))
        for r in payload["rows"]:
            for arm in ("backward", "forward"):
                print(f"{r['rung_label']:22s} {r['set']:9s} {arm:9s} " +
                      " ".join(f"{r[arm]['solve_rate'][i]*100:6.1f}%" for i in idx))
            print(f"{'':22s} {'':9s} {'diff':9s} " +
                  " ".join(f"{r['diff'][i]*100:+6.1f} " for i in idx))
        print(f"\nwrote {a.out}  ({len(payload['rows'])} rung/set pairs, "
              f"{len(payload['extended'])} extended-budget probes)")


if __name__ == "__main__":
    main()
