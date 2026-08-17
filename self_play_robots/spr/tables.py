"""Markdown tables from comparison payloads (for FINDINGS entries).

    python -m spr.tables results/m1/*.json [--ref supervised_valuenet/eval/results/final450_backward_prefix.json ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from spr.arena import summarize


def _f(x, spec=".2f"):
    return "--" if x is None else format(x, spec)


def row(path, label=None):
    d = json.loads(Path(path).read_text())
    s = summarize(d)
    n = s["n"]
    return (f"| {label or Path(path).stem} | {s['solved']}/{n} ({100 * s['solve_rate']:.1f}%) | "
            f"{_f(s['mean_moves'])} | {_f(s['mean_regret'])} | {_f(s['pct_optimal'], '.1f')} | "
            f"{_f(s['mean_expansions'], '.1f')} | {_f(s['mean_seconds'], '.1f')} |")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--ref", action="append", default=[])
    a = p.parse_args(argv)
    print("| system | solved | mean moves | mean regret | % optimal | mean expansions | s/inst |")
    print("|---|---|---|---|---|---|---|")
    for r in a.ref:
        print(row(r, "REF " + Path(r).stem))
    for pth in a.paths:
        try:
            print(row(pth))
        except Exception as e:  # noqa: BLE001
            print(f"| {Path(pth).stem} | ERR {e} | | | | | |")


if __name__ == "__main__":
    main()
