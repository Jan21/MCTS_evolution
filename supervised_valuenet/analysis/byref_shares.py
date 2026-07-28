"""By-reference share of every B2 label corpus, as a machine-readable artifact.

FINDINGS 34/36 quote per-corpus by-reference shares (the dose in the
label-budget dose-response) from ad-hoc qc_byref.py runs recorded only in
prose. This walks every merged corpus file and writes
analysis/artifacts/byref_shares.json so the report can cite a source.

Definition mirrors scaling/qc_byref.py exactly: a candidate is by-reference
when its helper stands on a cell that is no robot's start position.

    PYTHONPATH=. python3 -m analysis.byref_shares
"""
import glob
import json
import os
import re

OUT = "analysis/artifacts/byref_shares.json"


def share(path):
    n = byref = 0
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            starts = {tuple(r["target_robot"][0])}
            starts |= {tuple(h[0]) for h in (r.get("helpers") or [])}
            if tuple(r["cand_helper"][0]) not in starts:
                byref += 1
    return n, byref


def main():
    out = {}
    for path in sorted(glob.glob("scaling/data/*/backward_b2*.rust.jsonl")):
        base = os.path.basename(path)
        if "shard" in base:
            continue
        cfg = path.split("/")[2]
        m = re.search(r"cap(\d+)", base)
        cap = f"cap{m.group(1)}" if m else "cap5000"
        n, b = share(path)
        out[f"{cfg}.{cap}"] = {
            "file": path, "records": n, "by_reference": b,
            "share_pct": round(b / n * 100, 1) if n else None,
        }
        print(f"{cfg}.{cap}: {b}/{n} = {out[f'{cfg}.{cap}']['share_pct']}%")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(OUT + ".tmp", OUT)
    print(f"wrote {OUT} ({len(out)} corpora)")


if __name__ == "__main__":
    main()
