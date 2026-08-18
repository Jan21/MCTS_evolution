"""Iteration trend table for a loop (markdown, for FINDINGS).

    python -m spr.trend --config g24r4 [--tag _b2] [--upto K]
Reads results/selfplay/<cfg><tag>_iter<k>/{generation.manifest.json,gauge.json,
bench_<cfg>_{astar,mcts,frontier_astar,frontier_mcts}.json} and, for k=0 of the
B2 loop, results/selfplay/<cfg>_b2_iter0/*_bench_{solved,unsolved}_{astar,mcts}.json.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from spr import RESULTS
from spr.arena import summarize


def _load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return None


def _bench(p):
    d = _load(p)
    if not d or "systems" not in d:
        return None
    s = summarize(d)
    reg = "--" if s["mean_regret"] is None else f"{s['mean_regret']:.2f}"
    return f"{s['solved']}/{s['n']}, {s['mean_moves']:.2f} mv, {reg}, {s['mean_expansions']:.0f} exp"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--upto", type=int, default=20)
    a = p.parse_args(argv)
    cfg = a.config
    print("| iter | instances / certified / records | gauge argmin | A* graded | MCTS graded | A* frontier | MCTS frontier |")
    print("|---|---|---|---|---|---|---|")
    for k in range(0, a.upto + 1):
        d = RESULTS / "selfplay" / f"{cfg}{a.tag}_iter{k}"
        if not d.is_dir():
            continue
        m = _load(d / "generation.manifest.json") or {}
        gen = (f"{m.get('instances')} / {m.get('solved')} / {m.get('records')}" if m else "—")
        g = _load(d / "gauge.json")
        gauge = f"{g['audit_summary']['argmin_agreement']:.3f}" if g and g.get("audit_summary") else "—"
        if k == 0:
            files = {n: (glob.glob(str(d / f"*bench_solved_{n}.json")) or [None])[0] for n in ("astar", "mcts")}
            ffiles = {n: (glob.glob(str(d / f"*bench_unsolved_{n}.json")) or [None])[0] for n in ("astar", "mcts")}
            row = [_bench(files["astar"]) if files["astar"] else None, _bench(files["mcts"]) if files["mcts"] else None,
                   _bench(ffiles["astar"]) if ffiles["astar"] else None, _bench(ffiles["mcts"]) if ffiles["mcts"] else None]
        else:
            row = [_bench(d / f"bench_{cfg}_astar.json"), _bench(d / f"bench_{cfg}_mcts.json"),
                   _bench(d / f"bench_{cfg}_frontier_astar.json"), _bench(d / f"bench_{cfg}_frontier_mcts.json")]
        print(f"| {k} | {gen} | {gauge} | " + " | ".join(x or "—" for x in row) + " |")


if __name__ == "__main__":
    main()
