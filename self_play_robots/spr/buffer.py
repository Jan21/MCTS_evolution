"""Replay buffer assembly for one training round of the loop (PROBLEM.md 5).

    python -m spr.buffer --config g24r4 --sp-root runs/spr/selfplay --upto 3 --window 3 \
        --val-frac 0.15 --out runs/spr/selfplay/g24r4_iter3/buffer.jsonl

Concatenates the certified record files of iterations (upto-window+1 .. upto)
(`<sp-root>/<cfg>_iter<k>/records.jsonl`), keeps board provenance (records
carry `env_id` in the iteration's fresh id range and `boards_dir`), and
writes `<out>.splits.txt` holding the `--splits` argument for spr.train: per
iteration the LAST `val-frac` of its board ids are validation, the rest
training (board-disjoint by construction; the pinned pool 0-1199 is never
present). Also writes `<out>.manifest.json` with per-iteration counts.

Because every self-play board directory differs, the trainer resolves
adjacency per record through `_env_dir`; the buffer therefore stamps each
record's `boards_dir` into a sidecar map the trainer reads via
`--env-dir-map <out>.envdirs.json` (env_id -> dir).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from spr import REPO


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--sp-root", default="runs/spr/selfplay")
    p.add_argument("--upto", type=int, required=True)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    root = Path(a.sp_root)
    if not root.is_absolute():
        root = REPO / root
    out = Path(a.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    iters = [k for k in range(max(1, a.upto - a.window + 1), a.upto + 1)]
    train_ids, val_ids, envdirs, per_iter = [], [], {}, []
    n_total = 0
    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for k in iters:
            rp = root / f"{a.config}_iter{k}" / "records.jsonl"
            if not rp.is_file():
                print(f"[buffer] iteration {k}: missing {rp} -- skipped")
                continue
            ids = set()
            n = 0
            for line in open(rp):
                r = json.loads(line)
                ids.add(int(r["env_id"]))
                if r.get("boards_dir"):
                    envdirs[str(r["env_id"])] = r["boards_dir"]
                f.write(line if line.endswith("\n") else line + "\n")
                n += 1
            ids = sorted(ids)
            cut = max(1, int(round(len(ids) * (1 - a.val_frac))))
            train_ids += ids[:cut]
            val_ids += ids[cut:]
            per_iter.append({"iter": k, "records": n, "boards": len(ids),
                             "train_boards": cut, "val_boards": len(ids) - cut})
            n_total += n
    os.replace(tmp, out)
    from scaling.configs import compress_ids
    splits = f"{a.config}:train={compress_ids(train_ids)},val={compress_ids(val_ids)}"
    (out.parent / (out.name + ".splits.txt")).write_text(splits + "\n")
    (out.parent / (out.name + ".envdirs.json")).write_text(json.dumps(envdirs, indent=0))
    (out.parent / (out.name + ".manifest.json")).write_text(json.dumps(
        {"config": a.config, "iters": iters, "window": a.window, "records": n_total,
         "per_iter": per_iter, "splits": splits}, indent=1))
    print(f"[buffer] {n_total} records from iters {iters} -> {out}")
    print(f"[buffer] splits: {splits}")
    print(f"SPR BUFFER DONE {out}")


if __name__ == "__main__":
    main()
