"""Merge run_forward.py shard outputs into one solutions file.

    python3 analysis/failure_gallery/merge_shards.py OUT SHARD1 SHARD2 ...

Rows are concatenated and sorted by (tag, idx); the protocol comes from the
first shard (they must agree on ckpt/k/budget) with a shards note added.
Duplicate (tag, idx) rows are an error.
"""
import json
import sys
from pathlib import Path


def main():
    out, *shards = sys.argv[1:]
    rows, proto = [], None
    for s in shards:
        d = json.load(open(s))
        if proto is None:
            proto = d["protocol"]
        else:
            for k in ("ckpt", "k", "budget", "grid", "robots"):
                assert d["protocol"][k] == proto[k], (s, k)
        rows += d["rows"]
    keys = [(r.get("tag", ""), r["idx"]) for r in rows]
    assert len(set(keys)) == len(keys), "duplicate (tag, idx) across shards"
    rows.sort(key=lambda r: (r.get("tag", ""), r["idx"]))
    proto["shards"] = [str(s) for s in shards]
    Path(out).write_text(json.dumps({"protocol": proto, "rows": rows},
                                    indent=1) + "\n")
    print(f"[merge_shards] {len(rows)} rows from {len(shards)} shards -> {out}")


if __name__ == "__main__":
    main()
