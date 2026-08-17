"""Milestone gates (PROBLEM.md section 8) computed from result files.

    python -m spr.gate m1 --new results/m1/<tag>_g16r4.json --ref-arm g16r4_v2_prefix
    python -m spr.gate compare --a A.json --b B.json      # paired per-instance A/B

M1 gate: solve rate within 3.5 pts AND %optimal within 6.6 pts of the per-size
supervised base-vocabulary pair (the measured seed bars, section 4.7). Paired
comparisons report the McNemar exact test on the per-instance solved vectors
(the promotion instrument of section 4.7 / FINDINGS 78) plus mean-moves on the
both-solved subset.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from spr import RESULTS, SV
from spr.arena import summarize, _backward_system

# per-size supervised references (base vocabulary, prefix-check): the M1 opponents
REFS = {
    "g16r4_v2_prefix": ("eval/results/final450_backward_prefix.json",
                        "g16r4 v2 pair, prefix-check, base vocab (FINDINGS 3)"),
    "g16r4_b1s21_b2flags": ("eval/results/final450_backward_b2_seed21.json",
                            "g16r4 B1-seed21 pair under B2 flags (FINDINGS 80)"),
    "g24r4_exact_prefix": ("scaling/results/g24r4/comparison.json",
                           "g24r4 exact pair, prefix-check (FINDINGS 67/74)"),
    "g24r4_exactseed21_prefix": ("scaling/results/g24r4/comparison_exactseed21.json",
                                 "g24r4 exact pair seed 21, prefix-check (FINDINGS 71)"),
}
SOLVE_BAR = 3.5      # points
OPT_BAR = 6.6        # points


def _load(p):
    return json.loads(Path(p).read_text())


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(a_payload, b_payload):
    _, sa = _backward_system(a_payload)
    _, sb = _backward_system(b_payload)
    ra, rb = sa["rows"], sb["rows"]
    if len(ra) != len(rb):
        raise SystemExit("row count mismatch")
    if a_payload["protocol"].get("instances_sha256") != b_payload["protocol"].get("instances_sha256"):
        raise SystemExit("instance sha mismatch -- not the same exam")
    b = sum(1 for x, y in zip(ra, rb) if x["solved"] and not y["solved"])   # A only
    c = sum(1 for x, y in zip(ra, rb) if y["solved"] and not x["solved"])   # B only
    both = [(x, y) for x, y in zip(ra, rb) if x["solved"] and y["solved"]]
    ma = sum(x["realized_strict"] for x, _ in both) / len(both) if both else None
    mb = sum(y["realized_strict"] for _, y in both) / len(both) if both else None
    wins = sum(1 for x, y in both if x["realized_strict"] < y["realized_strict"])
    losses = sum(1 for x, y in both if x["realized_strict"] > y["realized_strict"])
    return {"n": len(ra), "solved_a": sum(r["solved"] for r in ra),
            "solved_b": sum(r["solved"] for r in rb), "a_only": b, "b_only": c,
            "mcnemar_p": mcnemar_exact(b, c), "both_solved": len(both),
            "mean_moves_a_both": ma, "mean_moves_b_both": mb,
            "moves_wins_a": wins, "moves_wins_b": losses,
            "sign_p_moves": mcnemar_exact(wins, losses)}


def gate_m1(new_path, ref_arm):
    new = _load(new_path)
    ref_rel, desc = REFS[ref_arm]
    ref = _load(SV / ref_rel)
    sn, sr = summarize(new), summarize(ref)
    d_solve = 100.0 * (sn["solve_rate"] - sr["solve_rate"])
    if sn["pct_optimal"] is None or sr["pct_optimal"] is None:
        raise SystemExit("M1 gate needs a graded set (pct_optimal is None on one side)")
    d_opt = sn["pct_optimal"] - sr["pct_optimal"]
    ok = d_solve >= -SOLVE_BAR and d_opt >= -OPT_BAR
    out = {"new": sn, "ref": sr, "ref_desc": desc, "delta_solve_pts": d_solve,
           "delta_opt_pts": d_opt, "bars": {"solve": SOLVE_BAR, "opt": OPT_BAR},
           "pass": ok}
    try:
        out["paired"] = paired(new, ref)
    except SystemExit as e:
        out["paired_error"] = str(e)
    print(json.dumps(out, indent=1))
    print(f"M1 GATE {'PASS' if ok else 'FAIL'} vs {ref_arm}: solve {d_solve:+.1f} pts "
          f"(bar -{SOLVE_BAR}), optimal {d_opt:+.1f} pts (bar -{OPT_BAR})")
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("m1")
    g.add_argument("--new", required=True)
    g.add_argument("--ref-arm", required=True, choices=sorted(REFS))
    g.add_argument("--out", default=None)
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    a = p.parse_args(argv)
    if a.cmd == "m1":
        out = gate_m1(a.new, a.ref_arm)
        if a.out:
            Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    else:
        print(json.dumps(paired(_load(a.a), _load(a.b)), indent=1))


if __name__ == "__main__":
    main()
