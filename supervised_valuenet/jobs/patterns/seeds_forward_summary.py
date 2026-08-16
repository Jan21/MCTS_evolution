"""Read-out for the two experiments (seed replicates of the backward headline
pair; forward rescue): per-arm solved counts next to the production rows, so
the paper's claim can be checked at a glance. Read-only; skips files that
have not landed. Login-node safe.

    PYTHONPATH=. python3 jobs/patterns/seeds_forward_summary.py
"""
import json
import os
import statistics

SEEDS = (21, 37, 53)


def solved(path, kind):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    sy = [s for s in d["systems"].values() if s.get("kind") == kind and s.get("rows")]
    if not sy:
        return None
    rows = sy[0]["rows"]
    return sum(bool(r.get("solved")) for r in rows), len(rows)


def line(label, cells):
    got = [(k, v) for k, v in cells if v is not None]
    txt = "  ".join(f"{k}={v[0]}/{v[1]}" for k, v in got)
    counts = [v[0] for _, v in got if _.startswith("seed")]
    spread = f"  seed min/med/max={min(counts)}/{statistics.median(counts)}/{max(counts)}" if len(counts) >= 2 else ""
    print(f"{label:34s} {txt}{spread}")


print("== (A) backward headline pair, seed replicates (zero-shot nets, B2 language) ==")
for cfg, sets in (("g16r8", (("graded", "comparison_b2{}.json"), ("frontier", "comparison_ungraded_b2{}.json"))),
                  ("g32r4", (("graded", "comparison_b2{}.json"), ("frontier", "comparison_ungraded_b2{}.json")))):
    for name, tpl in sets:
        base = f"scaling/results/{cfg}/"
        cells = [("prod", solved(base + tpl.format(""), "backward"))]
        cells += [(f"seed{s}", solved(base + tpl.format(f"_seed{s}"), "backward")) for s in SEEDS]
        fwd_file = {"g16r8": {"graded": "comparison_forward_control.json", "frontier": "comparison_ungraded.json"},
                    "g32r4": {"graded": "comparison.json", "frontier": "comparison_ungraded.json"}}[cfg][name]
        cells.append(("fwd_control", solved(base + fwd_file, "forward")))
        line(f"{cfg} {name}", cells)
cells = [("prod", solved("eval/results/final450_backward_b2.json", "backward"))]
cells += [(f"seed{s}", solved(f"eval/results/final450_backward_b2_seed{s}.json", "backward")) for s in SEEDS]
line("base450", cells)

print("\n== (B) forward rescue (selected by val_policy_top1, never by test) ==")
for cfg in ("g16r8", "g24r4"):
    sel = f"scaling/runs/{cfg}/forward-rescue/SELECTED.json"
    if os.path.exists(sel):
        s = json.load(open(sel))
        print(f"{cfg}: winner {s['winner']['cell']} val_top1={s['winner']['val_policy_top1']:.4f}; "
              + "  ".join(f"{r['cell']}:{r['val_policy_top1']:.3f}" for r in s["table"]))
    else:
        cells = sorted(f for f in (os.listdir(f"scaling/runs/{cfg}/forward-rescue")
                                   if os.path.isdir(f"scaling/runs/{cfg}/forward-rescue") else []))
        done = [c for c in cells if os.path.exists(f"scaling/runs/{cfg}/forward-rescue/{c}/DONE")]
        print(f"{cfg}: no SELECTED.json yet ({len(done)}/9 grid cells DONE)")
    base = f"scaling/results/{cfg}/"
    ctl_g = "comparison_forward_control.json" if cfg == "g16r8" else "comparison_b2.json"
    ctl_f = "comparison_ungraded.json" if cfg == "g16r8" else "comparison_ungraded_b2.json"
    line(f"{cfg} graded  ", [("fwd_control", solved(base + ctl_g, "forward")),
                             ("fwd_rescue", solved(base + "comparison_forward_rescue.json", "forward")),
                             ("bwd_headline", solved(base + "comparison_b2.json", "backward"))])
    line(f"{cfg} frontier", [("fwd_control", solved(base + ctl_f, "forward")),
                             ("fwd_rescue", solved(base + "comparison_ungraded_forward_rescue.json", "forward")),
                             ("bwd_headline", solved(base + "comparison_ungraded_b2.json", "backward"))])
