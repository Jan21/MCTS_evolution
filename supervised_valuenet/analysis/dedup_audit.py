"""Wall-layout near-duplicate audit across train / val / bench splits.

Closes the audit half of `analysis/publishability.md` objection 0.7 ("train/test
hygiene under-documented"). The board ID ranges are already known disjoint; what
was never checked is whether a *bench* board's wall layout is a near-copy of a
*train* board's — which would leak the test set even with disjoint IDs, because
the networks see wall geometry, not board IDs.

Method. A board's layout signature is the set of its interior wall segments,
`{("R", cell), ("D", cell)}`, decoded by `simulate.wall_sets` from the board
pickle's `grid_data` — the same decoding both planners use. Two boards are
compared by Jaccard similarity of those sets:

    J(a, b) = |a & b| / |a | b|      (1.0 == byte-identical wall layout)

For every split pair the audit reports the number of exact duplicates
(J == 1.0), the number of pairs above `--threshold`, and the maximum J observed
with the board IDs that attain it. Border walls are excluded (every board has
them, so including them would inflate every J).

    PYTHONPATH=. python -m analysis.dedup_audit [--threshold 0.9]
        [--out analysis/artifacts/dedup_audit.json]

Analysis only: reads board pickles, runs no planner and no network.
"""
from __future__ import annotations

import argparse
import json
import pickle
from math import isqrt
from pathlib import Path

from simulate import wall_sets
from scaling.configs import CONFIGS

# Base (legacy 16x16/4) boards live in environments/; every other config has
# its own directory. Bench boards for the base config are bench450's 2400-2549.
ENV_DIR = {"g16r4": "environments"}


def env_dir(cfg_key):
    return ENV_DIR.get(cfg_key, f"environments_{cfg_key}")


def signature(path):
    """Interior wall-segment set for one board, or None if unreadable."""
    try:
        with open(path, "rb") as f:
            grid_data = pickle.load(f)["grid_data"]
    except (OSError, KeyError, pickle.UnpicklingError):
        return None
    size = isqrt(len(grid_data))
    if size * size != len(grid_data):
        return None
    wr, wd = wall_sets(grid_data, size)
    # drop the board border: right walls on the last column, down walls on the
    # last row are present on every board and carry no discriminating signal.
    sig = {("R", c) for c in wr if c[0] != size - 1}
    sig |= {("D", c) for c in wd if c[1] != size - 1}
    return frozenset(sig)


def load_split(cfg_key, ids):
    d = env_dir(cfg_key)
    out = {}
    for i in ids:
        sig = signature(Path(d) / f"env_{i}.pkl")
        if sig is not None:
            out[i] = sig
    return out


def compare(a, b, threshold):
    """(n_exact, n_above, max_j, argmax pair) between two id->signature maps."""
    n_exact = n_above = 0
    best = (0.0, None, None)
    for ia, sa in a.items():
        for ib, sb in b.items():
            if ia == ib and a is b:
                continue
            inter = len(sa & sb)
            union = len(sa | sb)
            j = inter / union if union else 1.0
            if j == 1.0:
                n_exact += 1
            if j >= threshold:
                n_above += 1
            if j > best[0]:
                best = (j, ia, ib)
    return n_exact, n_above, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--out", default="analysis/artifacts/dedup_audit.json")
    a = p.parse_args()

    results = []
    for key, cfg in CONFIGS.items():
        try:
            splits = {name: cfg.ids(name)
                      for name in ("train", "val", "bench")}
        except KeyError:
            continue
        sigs = {}
        for name, ids in splits.items():
            sigs[name] = load_split(key, ids)
            if not sigs[name]:
                break
        if not all(sigs.get(n) for n in ("train", "val", "bench")):
            results.append({"config": key, "skipped": "boards not on disk",
                            "env_dir": env_dir(key)})
            print(f"{key:7s} SKIPPED (boards not present in {env_dir(key)})")
            continue

        # ID-disjointness is the claim the paper already makes; re-verify it.
        overlap = {f"{x}&{y}": sorted(set(splits[x]) & set(splits[y]))[:5]
                   for x, y in (("train", "bench"), ("train", "val"),
                                ("val", "bench"))}
        entry = {"config": key, "env_dir": env_dir(key),
                 "n": {n: len(s) for n, s in sigs.items()},
                 "id_overlap": {k: v for k, v in overlap.items() if v},
                 "threshold": a.threshold, "pairs": {}}
        for x, y in (("train", "bench"), ("val", "bench"), ("train", "val")):
            n_exact, n_above, best = compare(sigs[x], sigs[y], a.threshold)
            entry["pairs"][f"{x}_vs_{y}"] = {
                "n_pairs": len(sigs[x]) * len(sigs[y]),
                "n_exact_duplicate": n_exact,
                f"n_jaccard_ge_{a.threshold}": n_above,
                "max_jaccard": round(best[0], 4),
                "max_jaccard_boards": [best[1], best[2]],
            }
            print(f"{key:7s} {x:5s} vs {y:6s} "
                  f"pairs={len(sigs[x])*len(sigs[y]):>7d} "
                  f"exact={n_exact:>3d} >=thr={n_above:>4d} "
                  f"max_J={best[0]:.4f} (env_{best[1]} vs env_{best[2]})")
        results.append(entry)

    payload = {
        "method": ("Jaccard similarity of interior wall-segment sets "
                   "{('R'|'D', cell)} decoded by simulate.wall_sets; board "
                   "border excluded; splits from scaling.configs board_ranges"),
        "threshold": a.threshold,
        "configs": results,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
