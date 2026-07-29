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
  * **The incumbent's `val_regret` is context, NOT a gate.** It is read from
    the predecessor checkpoint the retrain warm-started from, but the two
    numbers are computed on DIFFERENT validation splits — the incumbent's on
    old-vocabulary labels, the retrain's on B2 labels that contain
    by-reference candidates the old split has none of. A higher number can
    mean a harder validation set rather than a worse network, so it must not
    decide banking. (An earlier version of this script gated on it and
    reported two healthy runs as failures.) The decisive test is the
    benchmark: does the retrained pair produce better Track 1 rows?
  * **Candidates are only comparable when they trained on the same labels.**
    `val_regret` values from runs trained (and therefore validated) on
    different label sets are scores on different exams: the run with the
    easier validation split (the cap-5000 set has fewer hard by-reference
    candidates than the cap-20000 set) wins `min()` regardless of true
    quality. The checkpoints' `hyper_parameters` do NOT record the data path
    (verified: `train/looped_pc.py` saves only architecture/optimiser
    settings), so provenance falls back to an mtime heuristic against
    `scaling/data/<cfg>/backward_b2.cap20000.rust.jsonl`, and the report
    states which basis was used for each candidate. When several candidates
    are not known to share provenance, this script refuses to auto-pick: it
    prints them all, asks for a hand decision, and leaves the config OUT of
    the written manifest. `--force` does not override that refusal — there
    is no defensible automatic pick to force.
  * **Value retrains are seed-unstable.** The known bad signature is a value
    net whose best score never improves on its warm-start (best epoch 0).
    A config carrying that flag is EXCLUDED from the written manifest, not
    just warned about; `--force` banks it anyway, for a human who has looked
    at the run and decided.
  * **A retrained value net far above its incumbent draws a warning, not a
    gate.** More than 50% higher `val_regret` than the incumbent is printed
    for human attention, with the explicit caveat that the two numbers come
    from different validation splits and the gap is therefore NOT evidence
    the net is bad.

    PYTHONPATH=. python -m scaling.bank_b2                 # report only
    PYTHONPATH=. python -m scaling.bank_b2 --write         # + write manifest
    PYTHONPATH=. python -m scaling.bank_b2 --write --force # bank past the
                                                           # instability flag

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


def candidates(cfg, system, label_set=None):
    """Every checkpoint under scaling/runs/<cfg>/backward-<system>-b2*/.

    The trailing glob matters: `b2_retrain_one.slurm` suffixes the run
    directory with the label set when one is given (`backward-value-b2-cap20000`),
    so globbing only the bare `-b2` directory made the alternate-corpus nets --
    the entire point of the provenance machinery -- invisible to the banker.
    """
    root = Path("scaling/runs") / cfg
    if not root.exists():
        return []
    # --label-set names the corpus explicitly, which is strictly better than
    # inferring provenance from mtimes: "" (default) is the bare -b2 dir,
    # "cap20000" is -b2-cap20000. With it set there is exactly one candidate
    # lineage, so the refusal path never has to fire.
    # last.ckpt is trainer RESUME state (save_last=True since 2026-07-29),
    # not a selected checkpoint -- never a banking candidate.
    if label_set is not None:
        suffix = f"-{label_set}" if label_set else ""
        run = root / f"backward-{system}-b2{suffix}"
        return sorted(p for p in (run / "lightning_logs").glob(
            "version_*/checkpoints/*.ckpt")
            if p.name != "last.ckpt") if run.exists() else []
    out = []
    for run in sorted(root.glob(f"backward-{system}-b2*")):
        out.extend(sorted(p for p in (run / "lightning_logs").glob(
            "version_*/checkpoints/*.ckpt") if p.name != "last.ckpt"))
    return out


def provenance(cfg, info):
    """Best-effort training-data provenance for one candidate: (label, basis).

    The Lightning checkpoints do NOT record the training data path —
    `train/looped_pc.py`'s save_hyperparameters() captures only
    architecture/optimiser settings (verified on a live checkpoint) — so
    unless a data-like hparam key ever appears, this falls back to mtime
    ordering against the cap-20000 label file. The run-directory name cannot
    help either: candidates() pins it, so every candidate for a
    (config, system) pair shares it by construction. The basis string is
    printed with each candidate so the report never presents the mtime
    heuristic as a recorded fact.
    """
    for key, val in (info.get("hparams") or {}).items():
        if "data" in key.lower() and isinstance(val, str):
            return val, "recorded in checkpoint hparams"
    cap20 = Path("scaling/data") / cfg / "backward_b2.cap20000.rust.jsonl"
    if not cap20.exists():
        return "cap5000", "only one B2 label set exists for this config"
    if Path(info["path"]).stat().st_mtime < cap20.stat().st_mtime:
        return ("cap5000",
                "mtime heuristic: checkpoint predates the cap-20000 label file")
    return ("UNKNOWN",
            "mtime heuristic: checkpoint postdates the cap-20000 label file, "
            "so it could have trained on either label set")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true",
                   help="write scaling/runs/b2_banked.json from the best "
                        "candidate per config (still does not copy the base "
                        "checkpoints into checkpoints_backward/)")
    p.add_argument("--label-set", default=None,
                   help='bank only checkpoints trained on this corpus: "" for '
                        'the default backward_b2.rust.jsonl run dirs, or e.g. '
                        '"cap20000" for backward-<sys>-b2-cap20000. Makes '
                        'provenance explicit instead of inferred from mtimes, '
                        'so the mixed-provenance refusal never has to fire.')
    p.add_argument("--out", default="scaling/runs/b2_banked.json")
    p.add_argument("--force", action="store_true",
                   help="bank a config even when its value retrain carries "
                        "the epoch-0 instability flag (for a human who has "
                        "looked at the run and decided). Does NOT override "
                        "the provenance refusal: among candidates not known "
                        "to share training data there is no defensible "
                        "automatic pick to force.")
    a = p.parse_args()

    manifest = {}      # cfg -> banked paths
    excluded = {}      # cfg -> why it is kept OUT of the manifest
    forced = []        # cfgs banked only because --force was given
    advisories = []    # human-attention warnings; never affect banking
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

        chosen, blockers = {}, []
        for system in ("policy", "value"):
            cands = candidates(cfg, system, a.label_set)
            if not cands:
                print(f"  {system}: no runs yet")
                continue
            print(f"  {system}: {len(cands)} candidate checkpoint(s)")
            infos = [probe(c) for c in cands]
            for info in infos:
                if "error" in info:
                    print(f"     ERROR {info['path']}: {info['error']}")
                    continue
                info["provenance"], info["prov_basis"] = provenance(cfg, info)
                print(f"     mtime={info['mtime']}  epoch={info['epoch']}"
                      f"  {info['monitor']}={info['best_score']}")
                print(f"       {info['path']}")
                print(f"       trained on: {info['provenance']} "
                      f"({info['prov_basis']})")
            ok = [i for i in infos if "error" not in i
                  and i.get("best_score") is not None]
            if not ok:
                continue
            if len(ok) == 1:
                best = ok[0]
            else:
                # min(val_regret) is only meaningful across candidates that
                # took the SAME exam. Runs trained on different label sets
                # validate on different splits, and the one with the easier
                # split (the cap-5000 set has fewer hard by-reference
                # candidates) wins min() regardless of true quality — the
                # same class of mistake as the retracted incumbent gate
                # documented above. So auto-pick only when every candidate
                # is positively known to share provenance.
                provs = {i["provenance"] for i in ok}
                if len(provs) == 1 and "UNKNOWN" not in provs:
                    best = min(ok, key=lambda i: i["best_score"])  # lower better
                    print(f"     -> picking mtime={best['mtime']} "
                          f"({best['monitor']}={best['best_score']:.4f}); "
                          "confirm this is the run you meant")
                else:
                    print(f"     -> NOT auto-picking: {len(ok)} candidates "
                          "without a shared training-data provenance "
                          f"({', '.join(sorted(provs))}). Their val_regret "
                          "values are scores on different validation splits "
                          "and cannot be compared by min().")
                    print("        Choose by hand: confirm each run's DATA "
                          "path in its Slurm log under runs/b2retrain/, then "
                          "edit scaling/runs/b2_banked.json yourself. "
                          "--force does not override this refusal.")
                    blockers.append(
                        f"{system}: {len(ok)} candidates without shared "
                        "training-data provenance — hand pick required")
                    continue
            chosen[system] = best

            if system == "value" and inc_score is not None:
                # NOT a like-for-like comparison, and it must not be used as a
                # gate. The incumbent's val_regret was computed on an
                # OLD-VOCABULARY validation split; the retrain's is computed on
                # the B2 split, which contains by-reference candidates the old
                # split has none of. A higher number can mean "harder
                # validation set" rather than "worse network". Reported as
                # context only; the decisive test is the benchmark itself.
                delta = best["best_score"] - inc_score
                print(f"     value val_regret {best['best_score']:.4f} "
                      f"(B2 split) vs incumbent {inc_score:.4f} "
                      f"(old-vocabulary split), delta {delta:+.4f} — "
                      "NOT comparable, context only")
                if inc_score > 0 and best["best_score"] > 1.5 * inc_score:
                    # Absolute-quality flag: a run that improved once and
                    # then collapsed still has a "best" score, so the
                    # epoch-0 check alone cannot catch it. A warning, NOT a
                    # gate — the cross-split caveat above is real.
                    msg = (f"{cfg}: retrained value net's best val_regret "
                           f"{best['best_score']:.4f} is more than 50% above "
                           f"its incumbent's {inc_score:.4f}. This flag asks "
                           "for HUMAN ATTENTION ONLY — it is NOT evidence "
                           "the net is bad, because the two numbers come "
                           "from different validation splits (B2 vs "
                           "old-vocabulary) and the gap can mean a harder "
                           "validation set rather than a worse network. Not "
                           "a gate: look at this config's benchmark rows "
                           "before drawing any conclusion.")
                    print(f"     !! {msg}")
                    advisories.append(msg)

            if system == "value" and best.get("epoch") == 0:
                # A genuine instability signature that IS valid on its own:
                # the best epoch is the very first one, i.e. the run never
                # improved on its warm-start and every later epoch was worse.
                # Unlike the incumbent comparison this needs no cross-split
                # reading, so it excludes the config from the manifest
                # rather than merely warning; --force is the human override.
                if a.force:
                    print("     !! best epoch 0 — seed-instability "
                          "signature; banking anyway because --force was "
                          "given")
                    forced.append(cfg)
                else:
                    blockers.append(
                        "value retrain's best epoch is 0 — it never "
                        "improved after warm-start. That is the "
                        "seed-instability signature; rerun with "
                        "--torch-seed varied, or bank anyway with --force.")

        if blockers:
            excluded[cfg] = "; ".join(blockers)
            print(f"  EXCLUDED from manifest: {excluded[cfg]}")
        elif {"policy", "value"} <= set(chosen):
            manifest[cfg] = {"policy": chosen["policy"]["path"],
                             "value": chosen["value"]["path"]}

    print("\n===== summary =====")
    for cfg in CONFIGS:
        if cfg in manifest:
            note = ("  (instability flag overridden by --force)"
                    if cfg in forced else "")
            print(f"  {cfg}: READY{note}")
        elif cfg in excluded:
            print(f"  {cfg}: EXCLUDED — {excluded[cfg]}")
        else:
            print(f"  {cfg}: incomplete")
    for warn in advisories:
        print(f"  !! {warn}")

    if a.write and manifest:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(manifest, indent=1))
        print(f"\nwrote {a.out} ({len(manifest)} configs)")
        if excluded:
            print(f"left OUT of the manifest: {', '.join(excluded)} "
                  "(reasons in the summary above)")
        if "g16r4" in manifest:
            print("\nBase still needs its checkpoints copied by hand (the Track 1\n"
                  "base command consumes these exact paths):\n"
                  f"  cp {manifest['g16r4']['policy']} "
                  "checkpoints_backward/policy_b2.ckpt\n"
                  f"  cp {manifest['g16r4']['value']} "
                  "checkpoints_backward/value_b2.ckpt\n"
                  "then re-point the g16r4 entry of the manifest at those copies.")
    elif a.write:
        print("\nnothing to write -- no config is bankable "
              "(see summary for exclusions and gaps)")


if __name__ == "__main__":
    main()
