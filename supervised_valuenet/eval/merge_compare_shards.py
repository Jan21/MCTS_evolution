"""Merge per-shard `eval.compare` result files into one canonical result.

Why this exists: on Karolina a whole CPU node is charged per hour regardless of
use, so a long single-process evaluation lane is split into contiguous shards
of its instance file and the shards run in parallel on one node. Each shard is
a plain `eval.compare` run on its slice; this script concatenates the
per-instance rows back in benchmark order and recomputes the aggregates with
`eval.compare.aggregate` itself, so the merged file is schema- and
number-identical to what one unsharded run would have produced (timing fields
are per-instance wall clock, unaffected by sharding; equivalence is verified in
the smoke job by diffing a full run against a merged 2-shard run).

Usage:
  PYTHONPATH=. python3 eval/merge_compare_shards.py \
      --shards out1.json out2.json ... \
      --instances scaling/data/g32r4/bench.solved.jsonl \
      --out scaling/results/g32r4/comparison.json \
      --md scaling/results/g32r4/COMPARISON.md

Shards must be given in benchmark order (shard 1 = first slice, ...). The
script refuses to merge if protocols disagree (budget, k, checkpoints, device)
or if the concatenated row count does not match the full instance file.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.compare import aggregate, load_instances, write_markdown  # noqa: E402

PROTOCOL_MUST_MATCH = ("expansions", "k", "device", "checkpoints")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--instances", required=True,
                   help="the FULL (unsharded) instance file")
    p.add_argument("--out", required=True)
    p.add_argument("--md", default="/dev/null")
    a = p.parse_args()

    payloads = [json.loads(Path(s).read_text()) for s in a.shards]
    proto0 = payloads[0]["protocol"]
    for s, pl in zip(a.shards[1:], payloads[1:]):
        for key in PROTOCOL_MUST_MATCH:
            if pl["protocol"].get(key) != proto0.get(key):
                sys.exit(f"protocol mismatch on {key!r} in {s}")

    names = list(payloads[0]["systems"].keys())
    for s, pl in zip(a.shards[1:], payloads[1:]):
        if list(pl["systems"].keys()) != names:
            sys.exit(f"system set mismatch in {s}")

    instances, sha, meta = load_instances(a.instances)

    systems, order = {}, []
    for name in names:
        kind = payloads[0]["systems"][name]["kind"]
        if kind == "pending":
            systems[name] = {"kind": "pending"}
            order.append(name)
            continue
        rows = []
        for pl in payloads:
            rows.extend(pl["systems"][name]["rows"])
        if len(rows) != len(instances):
            sys.exit(f"{name}: {len(rows)} rows != {len(instances)} instances")
        agg = (aggregate(rows, "realized_strict") if kind == "backward"
               else aggregate(rows))
        systems[name] = {"kind": kind, "aggregate": agg, "rows": rows}
        order.append(name)

    protocol = dict(proto0)
    protocol.update({
        "instances_file": a.instances,
        "instances_sha256": sha,
        "n_instances": len(instances),
        "instances_meta": meta,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results_file": a.out,
        "command": ("merged from %d shard runs: " % len(payloads)
                    + " | ".join(pl["protocol"].get("command", "?")
                                 for pl in payloads)),
    })

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"protocol": protocol, "systems": systems}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(a.md, protocol, systems, order)
    print(f"[merge] wrote {out} ({len(instances)} instances, "
          f"{len(payloads)} shards)")
    for nm in order:
        s = systems[nm]
        if "aggregate" in s:
            ag = s["aggregate"]
            print(f"  {nm}: solved {ag['solved']}/{ag['n']}")


if __name__ == "__main__":
    main()
