"""Distribution diff between a Python-generated jsonl and its Rust regen.

Backward (`combined.jsonl` schema): record count, cost_to_go histogram,
is_optimal rate, records/instance, depth histogram, kept-instance overlap
rate (instance = (env_id, target, target_robot, helpers)).

Forward (`moves.jsonl` schema): record count, cost_to_go histogram, depth
histogram, full-record rate, records/board, plus a per-board content hash
comparison (the forward pipeline has no documented divergence class, so
whole boards are expected to match byte-for-byte modulo board order).

    python rust_datagen/pyref/diff_distributions.py --system backward \
        --python scaling/data/g16r6/backward.jsonl \
        --rust scaling/data/g16r6/backward.rust.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict


def stats_backward(path):
    n = 0
    ctg = Counter()
    depth = Counter()
    opt = 0
    inst = defaultdict(set)          # env -> instance keys
    per_inst = Counter()             # instance key -> records
    order = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            n += 1
            ctg[r["cost_to_go"]] += 1
            depth[r["depth"]] += 1
            opt += r["is_optimal"]
            key = (r["env_id"], json.dumps(r["target"]),
                   json.dumps(r["target_robot"]), json.dumps(r["helpers"]))
            if key not in per_inst:
                order.append(key)
            inst[r["env_id"]].add(key)
            per_inst[key] += 1
    return {"n": n, "ctg": ctg, "depth": depth, "opt": opt,
            "inst_keys": set(per_inst), "inst_order": order,
            "recs_per_inst": Counter(per_inst.values())}


def stats_forward(path):
    n = 0
    ctg = Counter()
    depth = Counter()
    full = 0
    per_board = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            n += 1
            ctg[r["cost_to_go"]] += 1
            depth[r["depth"]] += 1
            full += r["full"]
            per_board[r["env_id"]].append(line.strip())
    hashes = {gid: hashlib.md5("\n".join(lines).encode()).hexdigest()
              for gid, lines in per_board.items()}
    return {"n": n, "ctg": ctg, "depth": depth, "full": full,
            "boards": len(per_board),
            "recs_per_board": Counter(len(v) for v in per_board.values()),
            "board_hashes": hashes}


def hist_row(c: Counter, buckets):
    tot = sum(c.values())
    return " ".join(f"{b}:{sum(v for k, v in c.items() if lo <= k <= hi) / tot:.3f}"
                    for b, (lo, hi) in buckets)


def show_hist(name, ca, cb, la, lb):
    keys = sorted(set(ca) | set(cb))
    ta, tb = sum(ca.values()), sum(cb.values())
    print(f"  {name:>12}  {la:>10} {lb:>10}   (share)")
    for k in keys:
        print(f"  {k!s:>12}  {ca.get(k, 0):>10} {cb.get(k, 0):>10}   "
              f"{ca.get(k, 0) / ta:.4f} vs {cb.get(k, 0) / tb:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--system", choices=["backward", "forward"], required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--rust", required=True)
    a = p.parse_args()

    if a.system == "backward":
        sp, sr = stats_backward(a.python), stats_backward(a.rust)
        print(f"records: python {sp['n']}  rust {sr['n']}  "
              f"({(sr['n'] - sp['n']) / sp['n']:+.2%})")
        print(f"instances: python {len(sp['inst_keys'])}  "
              f"rust {len(sr['inst_keys'])}")
        shared = sp["inst_keys"] & sr["inst_keys"]
        print(f"instance overlap: {len(shared)} shared "
              f"({len(shared) / len(sp['inst_keys']):.2%} of python, "
              f"{len(shared) / len(sr['inst_keys']):.2%} of rust)")
        same_order = sp["inst_order"] == sr["inst_order"]
        print(f"kept-instance stream identical (order incl.): {same_order}")
        print(f"is_optimal rate: python {sp['opt'] / sp['n']:.4f}  "
              f"rust {sr['opt'] / sr['n']:.4f}")
        show_hist("cost_to_go", sp["ctg"], sr["ctg"], "python", "rust")
        show_hist("depth", sp["depth"], sr["depth"], "python", "rust")
        show_hist("recs/inst", sp["recs_per_inst"], sr["recs_per_inst"],
                  "python", "rust")
    else:
        sp, sr = stats_forward(a.python), stats_forward(a.rust)
        print(f"records: python {sp['n']}  rust {sr['n']}  "
              f"({(sr['n'] - sp['n']) / sp['n']:+.2%})")
        print(f"boards: python {sp['boards']}  rust {sr['boards']}")
        same = sum(1 for g, h in sp["board_hashes"].items()
                   if sr["board_hashes"].get(g) == h)
        print(f"boards with IDENTICAL record content (order incl.): "
              f"{same}/{sp['boards']}")
        print(f"full-record rate: python {sp['full'] / sp['n']:.4f}  "
              f"rust {sr['full'] / sr['n']:.4f}")
        show_hist("cost_to_go", sp["ctg"], sr["ctg"], "python", "rust")
        show_hist("depth", sp["depth"], sr["depth"], "python", "rust")


if __name__ == "__main__":
    main()
