"""Inspect the B2 retrain runs and propose the banking manifest.

The handoff's banking rules, made checkable:

  * **Never identify a run by version number.** Resubmissions accumulate
    `version_N` directories, so this reports every candidate with its hparams
    and checkpoint mtime and lets you choose.
  * **The single checkpoint under a run's `checkpoints/` IS its best**, because
    `train/looped_pc.py` builds `ModelCheckpoint(monitor="val_regret",
    mode="min")`. Its `best_model_score` is stored inside the checkpoint, so
    the trajectory does not need a CSV logger (there is none — the runs emit
    only tfevents, and tensorboard is not installed).
  * **`val_regret` must beat or approach the config's old-vocabulary run.**
    The incumbent score is read from the predecessor checkpoint the retrain
    warm-started from, so the comparison is like-for-like.
  * **Value retrains are seed-unstable.** The known bad signature is a value
    net whose best score never improves on its warm-start. That is flagged,
    not silently banked.

    PYTHONPATH=. python -m scaling.bank_b2                 # report only
    PYTHONPATH=. python -m scaling.bank_b2 --write         # + write manifest

`--write` emits `scaling/runs/b2_banked.json`, which
`jobs/patterns/track1_rows.slurm` and `track1_accounting.slurm` read. It does
NOT copy the base checkpoints into `checkpoints_backward/` — that is the one
irreversible step and stays manual (see the printed instruction).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

CONFIGS = ["g16r4", "g16r6", "g16r8", "g24r8", "g32r4"]

# Old-vocabulary predecessors, i.e. the incumbent each retrain must beat or
# approach. These are exactly the --warm-start sources in b2_retrain_one.slurm.
INCUMBENT_VALUE = {
    "g16r4": "checkpoints_backward/value_b1.ckpt",
    "g16r6": "scaling/runs/g16r6/backward-value/lightning_logs/version_1/checkpoints/epoch=29-step=29700.ckpt",
    "g16r8": "scaling/runs/g16r8/backward-value/lightning_logs/version_0/checkpoints/epoch=27-step=25676.ckpt",
    "g24r8": "scaling/runs/g24r8/backward-value/lightning_logs/version_0/checkpoints/epoch=10-step=21681.ckpt",
    "g32r4": "scaling/runs/g32r4/backward-value/lightning_logs/version_1/checkpoints/epoch=15-step=52528.ckpt",
}


def probe(path):
    """(best_score, monitor, epoch, step, hparams, mtime) for a Lightning ckpt."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:                        # corrupt / partial write
        return {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    best = monitor = None
    for value in (ck.get("callbacks") or {}).values():
        if isinstance(value, dict) and "best_model_score" in value:
            best = value["best_model_score"]
            monitor = value.get("monitor")
            break
    if hasattr(best, "item"):
        best = best.item()
    return {
        "path": str(path),
        "best_score": best,
        "monitor": monitor,
        "epoch": ck.get("epoch"),
        "step": ck.get("global_step"),
        "hparams": {k: v for k, v in (ck.get("hyper_parameters") or {}).items()
                    if isinstance(v, (int, float, str, bool))},
        "mtime": time.strftime("%Y-%m-%dT%H:%M:%S",
                               time.localtime(Path(path).stat().st_mtime)),
    }


def candidates(cfg, system):
    """Every checkpoint under scaling/runs/<cfg>/backward-<system>-b2/."""
    root = Path("scaling/runs") / cfg / f"backward-{system}-b2" / "lightning_logs"
    return sorted(root.glob("version_*/checkpoints/*.ckpt")) if root.exists() else []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true",
                   help="write scaling/runs/b2_banked.json from the best "
                        "candidate per config (still does not copy the base "
                        "checkpoints into checkpoints_backward/)")
    p.add_argument("--out", default="scaling/runs/b2_banked.json")
    a = p.parse_args()

    manifest, problems = {}, []
    for cfg in CONFIGS:
        print(f"\n===== {cfg} =====")
        inc = INCUMBENT_VALUE.get(cfg)
        inc_score = None
        if inc and Path(inc).exists():
            inc_score = probe(inc).get("best_score")
            print(f"  incumbent value (old vocabulary): val_regret="
                  f"{inc_score if inc_score is None else round(inc_score, 4)}"
                  f"  [{inc}]")
        else:
            print(f"  incumbent value: MISSING ({inc})")

        chosen = {}
        for system in ("policy", "value"):
            cands = candidates(cfg, system)
            if not cands:
                print(f"  {system}: no runs yet")
                continue
            print(f"  {system}: {len(cands)} candidate checkpoint(s)")
            infos = [probe(c) for c in cands]
            for info in infos:
                if "error" in info:
                    print(f"     ERROR {info['path']}: {info['error']}")
                    continue
                print(f"     mtime={info['mtime']}  epoch={info['epoch']}"
                      f"  {info['monitor']}={info['best_score']}")
                print(f"       {info['path']}")
            ok = [i for i in infos if "error" not in i
                  and i.get("best_score") is not None]
            if not ok:
                continue
            best = min(ok, key=lambda i: i["best_score"])   # val_regret: lower better
            chosen[system] = best
            if len(ok) > 1:
                print(f"     -> picking mtime={best['mtime']} "
                      f"({best['monitor']}={best['best_score']:.4f}); "
                      "confirm this is the run you meant")

            if system == "value" and inc_score is not None:
                delta = best["best_score"] - inc_score
                verdict = ("IMPROVES" if delta < 0 else
                           "matches" if delta < 0.05 else "WORSE")
                print(f"     value vs incumbent: {best['best_score']:.4f} vs "
                      f"{inc_score:.4f}  ({delta:+.4f}) -> {verdict}")
                if delta >= 0.05:
                    problems.append(
                        f"{cfg}: value retrain WORSE than its warm-start "
                        f"({best['best_score']:.4f} vs {inc_score:.4f}). The "
                        "known seed-instability signature -- do NOT bank; "
                        "rerun with --torch-seed varied.")

        if {"policy", "value"} <= set(chosen):
            manifest[cfg] = {"policy": chosen["policy"]["path"],
                             "value": chosen["value"]["path"]}

    print("\n===== summary =====")
    for cfg in CONFIGS:
        print(f"  {cfg}: {'READY' if cfg in manifest else 'incomplete'}")
    for warn in problems:
        print(f"  !! {warn}")

    if a.write and manifest:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(manifest, indent=1))
        print(f"\nwrote {a.out} ({len(manifest)} configs)")
        if "g16r4" in manifest:
            print("\nBase still needs its checkpoints copied by hand (the Track 1\n"
                  "base command consumes these exact paths):\n"
                  f"  cp {manifest['g16r4']['policy']} "
                  "checkpoints_backward/policy_b2.ckpt\n"
                  f"  cp {manifest['g16r4']['value']} "
                  "checkpoints_backward/value_b2.ckpt\n"
                  "then re-point the g16r4 entry of the manifest at those copies.")
    elif a.write:
        print("\nnothing to write -- no config has both nets yet")


if __name__ == "__main__":
    main()
