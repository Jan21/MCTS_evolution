"""Reconstruct solve-rate-vs-budget curves from archived per-instance rows.

Both search families are deterministic best-first searches whose budget only
truncates: the search at budget B is a prefix of the search at budget 1200.
Therefore an instance recorded as solved with `expansions` = e is solved at
every budget >= e, and an unsolved instance is unsolved at every budget
<= 1200 — so the full curve for budgets <= 1200 is derivable from the stored
rows at zero compute:

    solve_rate(B) = #(solved and expansions <= B) / n

This is the "budget curves, not single points" reporting the matched-budget
literature demands ("What Matters in Hierarchical Search" §4.3; see
analysis/publishability.md objection 0.2). Curves ABOVE 1200 cannot be
derived this way — that is what the forward extended-budget probe jobs are
for.

Auto-discovers every comparison JSON with per-instance rows (so newly landed
result files, e.g. the b2-retrained rows, join on the next run):

    PYTHONPATH=. python3 eval/budget_curves_from_rows.py
    -> eval/results/budget_curves_by_rung.json
"""
from __future__ import annotations

import glob
import json
import os

BUDGETS = [5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1200]

SOURCES = sorted(
    glob.glob("eval/results/final450_*.json")
    + glob.glob("eval/results/comparison_forward.json")
    + glob.glob("eval/results/comparison_backward.json")
    + glob.glob("scaling/results/*/comparison*.json")
)


def curve(rows):
    n = len(rows)
    solved_at = sorted(r["expansions"] for r in rows
                       if r.get("solved") and isinstance(r.get("expansions"), (int, float)))
    rates = []
    for b in BUDGETS:
        k = 0
        for e in solved_at:
            if e <= b:
                k += 1
            else:
                break
        rates.append(k / n if n else None)
    return rates


def main():
    out = {"budgets": BUDGETS,
           "method": "reconstructed from archived per-instance rows; "
                     "solve_rate(B) = #(solved & expansions<=B)/n; valid "
                     "because both searches are deterministic and the budget "
                     "only truncates (prefix property)",
           "sources": {}}
    for path in SOURCES:
        try:
            d = json.load(open(path))
        except Exception:
            continue
        systems = d.get("systems")
        if not isinstance(systems, dict):
            continue
        entry = {}
        for name, s in systems.items():
            rows = s.get("rows")
            if not rows or "expansions" not in rows[0]:
                continue
            rates = curve(rows)
            # self-check: the reconstructed rate at 1200 must equal the
            # stored aggregate (fail loud on mismatch)
            agg = s.get("aggregate", {}).get("solve_rate")
            if agg is not None and rates[-1] is not None \
                    and abs(agg - rates[-1]) > 1e-9:
                raise SystemExit(f"MISMATCH {path} / {name}: "
                                 f"reconstructed {rates[-1]} vs aggregate {agg}")
            entry[name] = {"n": len(rows),
                           "solve_rate": [None if r is None else round(r, 6)
                                          for r in rates],
                           "solve_rate_1200": agg}
        if entry:
            out["sources"][path] = entry
    dst = "eval/results/budget_curves_by_rung.json"
    tmp = dst + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, dst)
    n_sys = sum(len(v) for v in out["sources"].values())
    print(f"wrote {dst}: {len(out['sources'])} source files, "
          f"{n_sys} system curves, all 1200-point checks passed")


if __name__ == "__main__":
    main()
