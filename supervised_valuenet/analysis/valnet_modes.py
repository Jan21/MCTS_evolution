"""Best val_regret of every B2 value-net retrain, as a mode diagnostic.

FINDINGS 44: warm-started value-net retraining is bistable. Good-mode runs
sit near val_regret 0.74-0.80 (g16r6) and collapse-mode runs near 2.1-2.5,
and the mode maps one-to-one onto frontier benchmark recovery vs collapse
across every measured run. This walks every backward-value-b2* run dir and
records min val_regret, so the report and any banking decision can read the
mode from a file instead of prose.

Comparability note: val_regret is comparable ONLY within a config+corpus
(same val split). Cross-config numbers are context, not comparisons (§33b).

    PYTHONPATH=. python3 -m analysis.valnet_modes
"""
import csv
import glob
import json
import os

OUT = "analysis/artifacts/valnet_modes.json"


def main():
    out = {}
    for mpath in sorted(glob.glob(
            "scaling/runs/*/backward-value-b2*/lightning_logs/"
            "version_*/metrics.csv")):
        parts = mpath.split("/")
        cfg, rundir, version = parts[2], parts[3], parts[5]
        vals = [float(r["val_regret"]) for r in csv.DictReader(open(mpath))
                if r.get("val_regret")]
        if not vals:
            continue
        key = f"{cfg}/{rundir}"
        entry = out.get(key)
        best = round(min(vals), 4)
        if entry is None or best < entry["best_val_regret"]:
            out[key] = {"config": cfg, "run_dir": rundir, "version": version,
                        "best_val_regret": best, "epochs_logged": len(vals)}
        print(f"{key} [{version}]: best={best} ({len(vals)} epochs)")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(OUT + ".tmp", OUT)
    print(f"wrote {OUT} ({len(out)} runs)")


if __name__ == "__main__":
    main()
