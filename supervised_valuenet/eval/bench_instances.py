"""Materialize the exact instance set used by the move-planner benchmark.

Replicates `move_planner.evaluate.benchmark`'s sampling loop: a single
`random.Random(seed)` consumed sequentially over the boards, `per_board`
draws per board via `sample_instance` (imported from `move_planner.evaluate`,
not copied, so the RNG consumption is bit-identical). Each drawn instance is
written as one JSONL line together with `d_star`, the exact move-optimal cost
that `sample_instance` computes -- the same reference the forward benchmark
uses. A sidecar `<out>.meta.json` records the sampling protocol and the
sha256 of the canonical instance list.

    PYTHONPATH=. python -m eval.bench_instances --boards 2400-2549 \
        --per-board 3 --seed 1 --out eval/data/bench450.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import random

from move_planner.evaluate import sample_instance, parse_boards, walls_for


def materialize(boards, per_board, seed):
    """Verbatim re-run of the benchmark's sampling loop; returns instance dicts."""
    rng = random.Random(seed)
    out = []
    for env_id in boards:
        wr, wd = walls_for(env_id)
        for _ in range(per_board):
            inst = sample_instance(wr, wd, rng)
            if inst is None:
                continue
            positions, tidx, target, d = inst
            out.append({
                "env_id": int(env_id),
                "positions": [[int(x), int(y)] for (x, y) in positions],
                "target_idx": int(tidx),
                "target": [int(target[0]), int(target[1])],
                "d_star": int(d),
            })
    return out


def canonical_body(instances) -> str:
    """Canonical JSONL body (sorted keys, compact separators): the sha256 basis."""
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n"
                   for r in instances)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--boards", default="2400-2549")
    p.add_argument("--per-board", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="eval/data/bench450.jsonl")
    a = p.parse_args()

    boards = parse_boards(a.boards)
    t0 = time.time()
    instances = materialize(boards, a.per_board, a.seed)
    body = canonical_body(instances)
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    meta = {
        "protocol": {
            "sampler": "move_planner.evaluate.benchmark sampling loop "
                       "(one random.Random(seed) consumed sequentially over "
                       "boards, per_board draws per board via sample_instance)",
            "boards": a.boards,
            "per_board": a.per_board,
            "seed": a.seed,
            "d_star": "exact move-optimal cost returned by sample_instance "
                      "(move_planner.oracle.solve); reference only, never used "
                      "at inference",
        },
        "n_instances": len(instances),
        "sha256": sha,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path = Path(str(out) + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {len(instances)} instances over {len(boards)} boards "
          f"to {out} ({time.time()-t0:.0f}s)")
    print(f"sha256={sha}")
    print(f"meta -> {meta_path}")


if __name__ == "__main__":
    main()
