"""Paired significance tests and board-clustered CIs for every head-to-head cell.

Closes `analysis/publishability.md` objection 0.1 ("no CIs or tests anywhere;
the headline graded win is a one-puzzle margin") and supplies the pooled
graded+frontier union of objection 0.3.

Every comparison is PAIRED: both systems ran the same pinned instance file at
the same budget, so each instance contributes a (solved_A, solved_B) pair.
Two statistics per cell:

  * **Exact McNemar** — a two-sided binomial test on the discordant pairs only
    (b = A solves & B fails, c = B solves & A fails; H0: b ~ Binom(b+c, 1/2)).
    Exact rather than chi-square because several cells have b+c < 25.
  * **Board-clustered bootstrap CI** for the paired difference in solve rate.
    Benchmarks put up to 3 puzzles on the same board (bench450: exactly 3 on
    each of 150 boards), and puzzles on one board share its wall layout, so
    instances are NOT independent. The bootstrap therefore resamples BOARDS
    with replacement and takes every puzzle of a drawn board. Percentile
    interval, seeded, so reruns reproduce.

Pairing is only permitted between result files that record the same
`protocol.instances_sha256` — the guard against silently comparing systems
that ran different instance sets. Rows are matched to instances positionally
(`eval.compare` writes exactly one row per instance, in file order).

    PYTHONPATH=. python -m eval.stats_tests [--boot 10000] [--seed 0]
        [--out eval/results/stats_tests.json]

Writes a JSON of cells and prints a readable table. Analysis only — reads the
archived result JSONs, runs no planner.
"""
from __future__ import annotations

import argparse
import json
import random
from math import comb
from pathlib import Path

from eval.report_data import RUNGS

SCALING = Path("scaling/results")
SCALING_DATA = Path("scaling/data")
BASE_RESULTS = Path("eval/results")

# Base-scale (g16r4) cell: bench450, one instance file, several result files.
BASE_SYSTEMS = {
    "bwd_old": ("final450_backward_prefix.json", "backward"),
    "bwd_b1": ("final450_backward_b1.json", "backward"),
    "bwd_b2": ("final450_backward_b2.json", "backward"),
    "bwd_retrained": ("final450_backward_b2_retrained.json", "backward"),
    # "forward:best" mirrors eval/report_data.py: the base forward cell is the
    # BEST forward system in the file (candidate_scored.ckpt, 450/450), not the
    # first one listed (best.ckpt, 381/450). Comparing against anything weaker
    # would flatter the backward planner.
    "fwd": ("comparison_forward.json", "forward:best"),
}

# The missing 2x2 cell (objection 1.3), measured 2026-07-26 by
# jobs/patterns/basenets_oldvocab.slurm. Present only at the 16x16 rungs,
# where the published rows confounded language with net provenance.
EXTRA_SLOTS = {
    "graded": {"bwd_basenets_old":
               ("comparison_basenets_oldvocab.json", "backward"),
               },
    "frontier": {"bwd_basenets_old":
                 ("comparison_ungraded_basenets_oldvocab.json", "backward"),
                 },
}

# Which pairs to test, in reporting order. (A, B) reads "A vs B".
PAIRS = [
    # language effect at FIXED nets: both sides are the base-B1 net pair
    ("bwd_b2", "bwd_basenets_old"),
    # net-provenance effect at FIXED (old) language: base-B1 vs per-config nets
    ("bwd_basenets_old", "bwd_old"),
    ("bwd_b2", "fwd"),            # the headline full-language head-to-head
    ("bwd_retrained", "fwd"),     # Track 1's definitive row (when it lands)
    ("bwd_old", "fwd"),           # old-language like-for-like
    ("bwd_b2", "bwd_old"),        # the language effect at fixed nets
    ("bwd_retrained", "bwd_b2"),  # the retraining effect at fixed language
    # the label-corpus effect at fixed nets, recipe and lineage (FINDINGS 36)
    ("bwd_retrained_cap20k", "bwd_retrained"),
    ("bwd_retrained_cap20k", "bwd_b2"),
    # THE headline row once the corpora are regenerated: the properly retrained
    # backward planner against the forward control.
    ("bwd_retrained_cap20k", "fwd"),
]

LABELS = {
    "bwd_old": "backward (old language)",
    "bwd_b1": "backward (B1)",
    "bwd_b2": "backward (B2, zero-shot ranking)",
    "bwd_retrained": "backward (B2, retrained on cap-5,000 labels)",
    "bwd_retrained_cap20k": "backward (B2, retrained on cap-20,000 labels)",
    "bwd_basenets_old": "backward (base-B1 nets, OLD vocabulary)",
    "fwd": "forward control",
}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value on discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    # P(X <= min(b,c)) + P(X >= max(b,c)) under Binom(n, 1/2), = 2 * the tail
    # when b != c (symmetric), clipped at 1.
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def cluster_bootstrap_ci(pairs, boards, n_boot, seed, alpha=0.05):
    """Percentile CI for mean(a) - mean(b), resampling BOARDS with replacement.

    `pairs` is a list of (solved_A, solved_B) bools, `boards` the parallel list
    of board ids. Returns (lo, hi) or (None, None) when there is nothing to
    resample.
    """
    by_board = {}
    for (sa, sb), bd in zip(pairs, boards):
        by_board.setdefault(bd, []).append((sa, sb))
    keys = list(by_board)
    if not keys:
        return None, None
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        na = nb = n = 0
        for _ in range(len(keys)):
            for sa, sb in by_board[keys[rng.randrange(len(keys))]]:
                na += sa
                nb += sb
                n += 1
        diffs.append((na - nb) / n)
    diffs.sort()
    lo = diffs[int(alpha / 2 * n_boot)]
    hi = diffs[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _match_system(payload, want):
    """Resolve a system key: exact name, '<kind>:best', else unique 'kind'."""
    systems = payload.get("systems", {})
    if want in systems:
        return systems[want]
    if want.endswith(":best"):
        kind = want[:-len(":best")]
        hits = [s for s in systems.values()
                if s.get("kind") == kind and s.get("rows")]
        if not hits:
            return None
        return max(hits, key=lambda s: s["aggregate"]["solve_rate"])
    hits = [s for s in systems.values() if s.get("kind") == want]
    if len(hits) == 1:
        return hits[0]
    return None


def load_cell(path, want):
    """(solved list, instances_sha256, instances_file) or None if unavailable."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    system = _match_system(payload, want)
    if not system or not system.get("rows"):
        return None
    proto = payload.get("protocol", {})
    return ([bool(r.get("solved")) for r in system["rows"]],
            proto.get("instances_sha256"), proto.get("instances_file"))


def board_ids(inst_path):
    ids = []
    for line in Path(inst_path).read_text().splitlines():
        if line.strip():
            ids.append(json.loads(line)["env_id"])
    return ids


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def collect_sets():
    """Yield (rung_key, set_name, instances_file, {sys_key: (path, want)})."""
    for rung in RUNGS:
        if rung.get("base"):
            yield (rung["key"], "graded", "eval/data/bench450.jsonl",
                   {k: (BASE_RESULTS / f, w)
                    for k, (f, w) in BASE_SYSTEMS.items()})
            continue
        cfg = rung["key"]
        future = rung.get("future", {})
        for set_name, inst in (("graded", "bench.solved.jsonl"),
                               ("frontier", "bench.unsolved.jsonl")):
            slots = dict(rung.get(set_name) or {})
            slots.update(EXTRA_SLOTS.get(set_name, {}))
            # Filenames are pinned HERE, deliberately not read from
            # report_data's `future` dict. That dict is a DISPLAY policy -- it
            # was repointed at the cap-20000 files so the report would stop
            # publishing corpus-deficient rows -- and inheriting it silently
            # made `bwd_retrained` (labelled "cap-5,000") load cap-20000 data,
            # collapsing the corpus-effect comparison to a self-comparison
            # (310 vs 310, diff 0.0) and dropping 8 cells. Statistics must
            # keep every arm addressable regardless of what the report shows.
            for slot, fname in (
                    ("bwd_retrained",
                     "comparison_b2retrained.json" if set_name == "graded"
                     else "comparison_ungraded_b2retrained.json"),
                    ("bwd_retrained_cap20k",
                     "comparison_b2retrained_cap20000.json"
                     if set_name == "graded"
                     else "comparison_ungraded_b2retrained_cap20000.json")):
                slots[slot] = (fname, "backward")
            if not slots:
                continue
            yield (cfg, set_name, str(SCALING_DATA / cfg / inst),
                   {k: (SCALING / cfg / f, w) for k, (f, w) in slots.items()})


def _emit(cells, cfg, set_name, boards, loaded, n_boot, seed):
    """Append one cell per testable pair for a (rung, set)."""
    for a, b in PAIRS:
        if a not in loaded or b not in loaded:
            continue
        sa, sha_a, file_a = loaded[a]
        sb, sha_b, file_b = loaded[b]
        if sha_a and sha_b and sha_a != sha_b:
            cells.append({"rung": cfg, "set": set_name, "a": a, "b": b,
                          "skipped": "instances_sha256 mismatch",
                          "file_a": file_a, "file_b": file_b})
            continue
        if not (len(sa) == len(sb) == len(boards)):
            cells.append({"rung": cfg, "set": set_name, "a": a, "b": b,
                          "skipped": f"row-count mismatch "
                                     f"{len(sa)}/{len(sb)}/{len(boards)}",
                          "file_a": file_a, "file_b": file_b})
            continue
        n = len(sa)
        nb_only = sum(1 for x, y in zip(sa, sb) if x and not y)
        nc_only = sum(1 for x, y in zip(sa, sb) if y and not x)
        lo, hi = cluster_bootstrap_ci(list(zip(sa, sb)), boards, n_boot, seed)
        p = mcnemar_exact(nb_only, nc_only)
        cells.append({
            "rung": cfg, "set": set_name, "a": a, "b": b,
            "label_a": LABELS.get(a, a), "label_b": LABELS.get(b, b),
            "n": n, "n_boards": len(set(boards)),
            "solved_a": sum(sa), "solved_b": sum(sb),
            "rate_a": sum(sa) / n, "rate_b": sum(sb) / n,
            "diff": (sum(sa) - sum(sb)) / n,
            "discordant_a_only": nb_only, "discordant_b_only": nc_only,
            "mcnemar_p": p, "ci95_lo": lo, "ci95_hi": hi,
            "significant_at_05": p < 0.05,
            "file_a": file_a, "file_b": file_b,
        })


def run(n_boot, seed, out_path):
    cells = []
    per_rung = {}                       # cfg -> {set_name: (boards, loaded)}
    for cfg, set_name, inst_file, slots in collect_sets():
        if not Path(inst_file).exists():
            continue
        boards = board_ids(inst_file)
        loaded = {}
        for key, (path, want) in slots.items():
            got = load_cell(path, want)
            if got:
                loaded[key] = (got[0], got[1], str(path))
        _emit(cells, cfg, set_name, boards, loaded, n_boot, seed)
        per_rung.setdefault(cfg, {})[set_name] = (boards, loaded)

    # ---- pooled graded+frontier union (publishability objection 0.3) -------
    # The frontier sets are selected BY FAILURE of a move-level exhaustive
    # search, which is adversarial to move-level planners by construction. The
    # union of the two halves is the whole pinned 450-puzzle pool at that rung
    # and carries no such selection: it is the selection-free headline.
    for cfg, sets in per_rung.items():
        if not {"graded", "frontier"} <= set(sets):
            continue
        gb, gl = sets["graded"]
        fb, fl = sets["frontier"]
        pooled = {}
        for key in set(gl) & set(fl):
            pooled[key] = (gl[key][0] + fl[key][0], None,
                           f"{gl[key][2]} + {fl[key][2]}")
        _emit(cells, cfg, "pooled", gb + fb, pooled, n_boot, seed)

    payload = {
        "method": {
            "test": "exact two-sided McNemar on discordant pairs",
            "ci": f"percentile bootstrap, {n_boot} resamples of BOARDS "
                  f"(clustered; puzzles share a board's wall layout), seed {seed}",
            "pairing": "positional row match; refused unless both files record "
                       "the same protocol.instances_sha256",
        },
        "n_boot": n_boot, "seed": seed,
        "cells": cells,
    }
    payload["method"]["pooled"] = (
        "the 'pooled' set is the graded+frontier union — the whole pinned "
        "450-puzzle pool at that rung, free of the frontier sets' "
        "selection-by-move-level-failure")
    Path(out_path).write_text(json.dumps(payload, indent=1))
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="eval/results/stats_tests.json")
    a = p.parse_args()
    payload = run(a.boot, a.seed, a.out)

    hdr = (f"{'rung':7s} {'set':9s} {'A vs B':44s} {'n':>4s} {'A':>6s} "
           f"{'B':>6s} {'diff':>7s} {'95% CI':>17s} {'p':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for c in payload["cells"]:
        if c.get("skipped"):
            print(f"{c['rung']:7s} {c['set']:9s} "
                  f"{c['a']+' vs '+c['b']:44s} SKIPPED: {c['skipped']}")
            continue
        star = "*" if c["significant_at_05"] else " "
        print(f"{c['rung']:7s} {c['set']:9s} "
              f"{c['label_a']+' vs '+c['label_b']:44s} {c['n']:4d} "
              f"{c['rate_a']*100:5.1f}% {c['rate_b']*100:5.1f}% "
              f"{c['diff']*100:+6.1f}% "
              f"[{c['ci95_lo']*100:+5.1f},{c['ci95_hi']*100:+5.1f}] "
              f"{c['mcnemar_p']:8.4f}{star}")
    print(f"\nwrote {a.out}  ({len(payload['cells'])} cells)  "
          f"* = significant at 0.05")


if __name__ == "__main__":
    main()
