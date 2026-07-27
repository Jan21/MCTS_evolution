"""Matched summary of the extended-budget forward probes.

The probe jobs (`jobs/patterns/fwd_budget_probe.slurm`) rerun the FORWARD
planner on a seeded frontier subsample at 4-5x the standard 1,200-expansion
cap. A probe file on its own only says what forward reaches at the larger
budget; the question of publishability objections 0.2/1.2 needs three numbers
on the SAME instances:

    forward @1200   (from the rung's stored comparison_ungraded.json)
    forward @budget (the probe)
    backward @1200  (from comparison_ungraded_b2.json)

This module recovers the subsample's positions in the full frontier file by
matching (env_id, positions, target, target_idx) -- not by row order, since the
subsample is a seeded shuffle re-sorted into benchmark order -- and reports the
climb, the paired differences and exact McNemar p-values. FINDINGS 28 and 31
are generated from its output.

    PYTHONPATH=. python -m eval.budget_probe_summary
        [--out eval/results/budget_probe_summary.json]

Analysis only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.stats_tests import mcnemar_exact

# (config, probe budget) -- one entry per submitted probe.
PROBES = [("g16r6", 4800), ("g16r8", 4800), ("g24r8", 6000), ("g32r4", 4800)]
RUNS = Path("/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/fwdprobe")


def _key(inst):
    return (inst["env_id"], tuple(map(tuple, inst["positions"])),
            tuple(inst["target"]), inst["target_idx"])


def _rows(path, kind):
    if not Path(path).exists():
        return None
    payload = json.loads(Path(path).read_text())
    hits = [s for s in payload["systems"].values()
            if s.get("kind") == kind and s.get("rows")]
    return hits[0]["rows"] if len(hits) == 1 else None


def _jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()
            if l.strip()]


def summarise(cfg, budget):
    probe_file = Path(f"scaling/results/{cfg}/forward_probe_e{budget}.json")
    sub_file = RUNS / f"{cfg}.e{budget}" / "subsample.jsonl"
    full_file = Path(f"scaling/data/{cfg}/bench.unsolved.jsonl")
    if not (probe_file.exists() and sub_file.exists() and full_file.exists()):
        return {"rung": cfg, "budget": budget, "status": "not measured"}

    full = _jsonl(full_file)
    sub = _jsonl(sub_file)
    pos = {_key(i): n for n, i in enumerate(full)}
    try:
        idx = [pos[_key(i)] for i in sub]
    except KeyError:
        return {"rung": cfg, "budget": budget,
                "status": "subsample does not match the frontier file"}

    f_probe = _rows(probe_file, "forward")
    f_std = _rows(f"scaling/results/{cfg}/comparison_ungraded.json", "forward")
    b_std = _rows(f"scaling/results/{cfg}/comparison_ungraded_b2.json",
                  "backward")
    if not (f_probe and f_std and b_std) or len(f_probe) != len(idx):
        return {"rung": cfg, "budget": budget, "status": "rows unavailable"}

    n = len(idx)
    fwd_lo = [bool(f_std[i]["solved"]) for i in idx]
    fwd_hi = [bool(r["solved"]) for r in f_probe]
    bwd = [bool(b_std[i]["solved"]) for i in idx]

    def paired(a, b):
        bo = sum(1 for x, y in zip(a, b) if x and not y)
        co = sum(1 for x, y in zip(a, b) if y and not x)
        return {"diff": (sum(a) - sum(b)) / n, "a_only": bo, "b_only": co,
                "mcnemar_p": mcnemar_exact(bo, co)}

    agg = json.loads(probe_file.read_text())
    mean_exp = None
    for s in agg["systems"].values():
        if s.get("kind") == "forward" and s.get("aggregate"):
            mean_exp = s["aggregate"].get("mean_expansions")
    return {
        "rung": cfg, "budget": budget, "status": "ok", "n": n,
        "forward_1200": {"solved": sum(fwd_lo), "rate": sum(fwd_lo) / n},
        "forward_probe": {"solved": sum(fwd_hi), "rate": sum(fwd_hi) / n,
                          "mean_expansions": mean_exp},
        "backward_1200": {"solved": sum(bwd), "rate": sum(bwd) / n},
        "forward_climb": (sum(fwd_hi) - sum(fwd_lo)) / n,
        "backward_vs_forward_1200": paired(bwd, fwd_lo),
        "backward_vs_forward_probe": paired(bwd, fwd_hi),
        "sources": {"probe": str(probe_file), "subsample": str(sub_file),
                    "forward_1200": f"scaling/results/{cfg}/comparison_ungraded.json",
                    "backward_1200": f"scaling/results/{cfg}/comparison_ungraded_b2.json"},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="eval/results/budget_probe_summary.json")
    a = p.parse_args()
    recs = [summarise(c, b) for c, b in PROBES]

    hdr = (f"{'rung':7s} {'bud':>5s} {'n':>4s} {'fwd@1200':>9s} {'fwd@probe':>10s} "
           f"{'climb':>7s} {'bwd@1200':>9s} {'bwd-fwd@1200':>14s} "
           f"{'bwd-fwd@probe':>15s}")
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        if r.get("status") != "ok":
            print(f"{r['rung']:7s} {r['budget']:5d} -- {r.get('status')}")
            continue
        lo, hi = r["backward_vs_forward_1200"], r["backward_vs_forward_probe"]
        print(f"{r['rung']:7s} {r['budget']:5d} {r['n']:4d} "
              f"{r['forward_1200']['rate']*100:8.1f}% "
              f"{r['forward_probe']['rate']*100:9.1f}% "
              f"{r['forward_climb']*100:+6.1f} "
              f"{r['backward_1200']['rate']*100:8.1f}% "
              f"{lo['diff']*100:+7.1f} p={lo['mcnemar_p']:.4f} "
              f"{hi['diff']*100:+7.1f} p={hi['mcnemar_p']:.4f}")
    Path(a.out).write_text(json.dumps(
        {"note": ("all three numbers per rung are on the SAME seeded "
                  "subsample; the probe gives FORWARD a 4-5x budget while "
                  "backward stays at 1,200, so 'bwd-fwd@probe' is a "
                  "robustness check against an over-budgeted opponent, not a "
                  "like-for-like row"),
         "probes": recs}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
