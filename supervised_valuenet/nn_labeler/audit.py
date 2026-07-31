"""Audit a size-free value checkpoint against exact-labeled corpora (S0.5).

This is the honesty instrument of the NN-labeler track (nn_labeler/PLANS.md):
given a checkpoint and one or more exact-solver-labeled corpora, it scores the
net's cost-to-go predictions per decision group and reports, per config:

- n_groups / n_records          coverage of the audited split
- top1_optimal                  argmin-prediction picks an optimal candidate
- regret                        ctg(chosen) - ctg(best), averaged over groups
- mae                           |pred - exact ctg| over records
- bias                          mean (pred - exact ctg)  (sign of drift)
- regret_by_depth               regret keyed by decision depth (drift locator)

The audited configs need NOT appear in the checkpoint's training set -- that is
the point: zero-shot audits at sizes the net never saw are how each ladder rung
is gated before bulk labeling.

    PYTHONPATH=. python -m nn_labeler.audit \
        --ckpt nn_labeler/runs/g8r4_sin2d/lightning_logs/version_0/checkpoints/epoch=*.ckpt \
        --data g12r4=scaling/data/g12r4/backward.rust.jsonl \
        --split test --out nn_labeler/results/audit_g8sin2d_on_g12.json

Output JSON is written atomically (tmp + rename). The last line on success is
`NNLAB AUDIT DONE <out>`.
"""
from __future__ import annotations

import argparse
import functools
import glob
import json
import os
from pathlib import Path

import torch

from nn_labeler import dataset, encode
from nn_labeler.model import SizeFreeValueNet, collate_groups
from scaling.configs import REPO, get


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO / p)


@torch.no_grad()
def audit_groups(model, groups, batch_size, device):
    """Score decision groups; returns (per-group rows, per-record abs/signed errs)."""
    coord = model.hparams.pe == "coord"
    coll = functools.partial(collate_groups, featurize_fn=encode.node_features,
                             adjacency_fn=encode.adjacency,
                             key_fn=encode.key_indices, coord_channels=coord,
                             num_classes=model.hparams.num_classes)
    ds = dataset.GroupDataset(groups, max_per_group=None, sample=False)
    sampler = dataset.SizeBucketBatchSampler(ds, batch_size, False, None)
    rows, abs_err, sgn_err = [], [], []
    for idxs in sampler:
        b = coll([ds[i] for i in idxs])
        val = model._value(model(b["x"].to(device), b["A_all"].to(device),
                                 b["A_ind"].to(device), b["n"],
                                 b["key"].to(device))).cpu()
        for g in b["group"].unique():
            m = b["group"] == g
            preds, ctgs = val[m], b["ctg"][m].float()
            opts = b["opt"][m]
            chosen = int(preds.argmin())
            depth = ds[idxs[int(g)]][0].get("depth", -1)
            rows.append({
                "top1_optimal": bool(opts[chosen]),
                "regret": float(ctgs[chosen] - ctgs.min()),
                "depth": int(depth),
            })
            abs_err += (preds - ctgs).abs().tolist()
            sgn_err += (preds - ctgs).tolist()
    return rows, abs_err, sgn_err


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True,
                   help="checkpoint path; globs allowed (must match exactly one)")
    p.add_argument("--data", action="append", required=True, metavar="CONFIG=PATH")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--limit-records", type=int, default=None, help="per corpus")
    p.add_argument("--batch-size", type=int, default=16, help="groups per batch")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    hits = sorted(glob.glob(str(_abs(a.ckpt))))
    if len(hits) != 1:
        raise SystemExit(f"--ckpt must resolve to exactly one file, got {len(hits)}: "
                         f"{hits[:3]}")
    ckpt = hits[0]
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if a.device == "auto" else a.device
    model = SizeFreeValueNet.load_from_checkpoint(ckpt, map_location=device)
    model.eval().to(device)

    report = {"ckpt": ckpt, "split": a.split, "configs": {},
              "model_hparams": {k: v for k, v in dict(model.hparams).items()
                                if isinstance(v, (int, float, str, bool))}}
    for item in a.data:
        name, path = item.split("=", 1)
        cfg = get(name)
        recs = dataset.load_corpus(str(_abs(path)), cfg.name, cfg.grid,
                                   cfg.env_dir_abs, limit=a.limit_records)
        ranges = {cfg.name: {a.split: cfg.ids(a.split)}}
        recs = dataset.by_split(recs, a.split, ranges)
        groups = dataset.group_by_decision(recs)
        if not groups:
            report["configs"][name] = {"n_groups": 0, "note": "empty split"}
            print(f"[audit] {name}: EMPTY {a.split} split", flush=True)
            continue
        rows, abs_err, sgn_err = audit_groups(model, groups, a.batch_size, device)
        by_depth = {}
        for r in rows:
            by_depth.setdefault(r["depth"], []).append(r["regret"])
        entry = {
            "n_groups": len(rows), "n_records": len(abs_err),
            "top1_optimal": sum(r["top1_optimal"] for r in rows) / len(rows),
            "regret": sum(r["regret"] for r in rows) / len(rows),
            "mae": sum(abs_err) / len(abs_err),
            "bias": sum(sgn_err) / len(sgn_err),
            "regret_by_depth": {str(d): round(sum(v) / len(v), 4)
                                for d, v in sorted(by_depth.items())},
        }
        report["configs"][name] = entry
        print(f"[audit] {name} n={cfg.grid} {a.split}: groups={entry['n_groups']} "
              f"top1={entry['top1_optimal']:.3f} regret={entry['regret']:.3f} "
              f"mae={entry['mae']:.3f} bias={entry['bias']:+.3f}", flush=True)

    out = _abs(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=1, sort_keys=True))
    os.replace(tmp, out)
    print(f"NNLAB AUDIT DONE {out}", flush=True)


if __name__ == "__main__":
    main()
