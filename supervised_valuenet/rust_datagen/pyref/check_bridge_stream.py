"""Exact-stream check for the backward bridge (DESIGN.md section 6).

The subtlest piece of `scaling.rust_bridge` is the shared-RNG speculation:
per board it snapshots `rng.getstate()`, over-draws the FULL attempt budget,
then rewinds and re-draws exactly `attempts_consumed` samples so the stream
entering the next board is what `scaling.backward_label`'s sequential loop
would have produced. This script proves that logic on 3 real boards:

1. runs the bridge (backward, 3 boards) against the Rust engine;
2. replays a PURE-PYTHON REFERENCE LOOP -- the literal backward_label
   control flow (draw one sample per attempt, keep on success, stop at
   per_graph kept or budget attempts) -- on a fresh `random.Random(seed)`,
   using the engine's per-attempt ok/not-ok outcomes as the success oracle
   (identical outcomes by construction, which is exactly the premise under
   which the streams must be IDENTICAL);
3. asserts, per board: every reference draw equals the bridge's speculative
   draw at the same attempt index (work file), the kept attempt indices
   equal the bridge manifest, and the output jsonl is exactly the
   concatenation of the kept attempts' records;
4. as an outcomes-agree spot check, compares the kept instance stream
   against the production Python data (`scaling/data/<config>/backward.*`),
   classifying any mismatch as outcome disagreement vs stream bug.

    python rust_datagen/pyref/check_bridge_stream.py            # g16r6 defaults
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import subprocess
import sys
from pathlib import Path

PYREF_DIR = Path(__file__).resolve().parent
SV_DIR = PYREF_DIR.parent.parent
sys.path.insert(0, str(SV_DIR))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="g16r6")
    p.add_argument("--nshards", type=int, default=8)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--per-graph", type=int, default=6)
    p.add_argument("--boards", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--production", default=None,
                   help="python-generated jsonl for the outcomes-agree spot "
                        "check (default: scaling/data/<config>/backward."
                        "shard{K}of{N}.jsonl if present)")
    a = p.parse_args()

    out_dir = PYREF_DIR / "out" / "bridge_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{a.config}.backward.jsonl"
    tag = f"stream_check_{a.config}"
    if out.exists():
        out.unlink()

    # 1. run the bridge
    cmd = [sys.executable, "-m", "scaling.rust_bridge",
           "--config", a.config, "--system", "backward",
           "--per-graph", str(a.per_graph),
           "--nshards", str(a.nshards), "--shard", str(a.shard),
           "--limit", str(a.boards), "--seed", str(a.seed),
           "--threads", str(a.threads), "--out", str(out),
           "--work-tag", tag]
    print("[check] running bridge:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=str(SV_DIR),
                        env={**__import__("os").environ,
                             "PYTHONPATH": str(SV_DIR)}).returncode
    if rc != 0:
        raise SystemExit(f"bridge failed rc={rc}")

    from scaling import configs
    cfg = configs.get(a.config)
    work_dir = SV_DIR / "scaling" / "data" / cfg.name / "rust_work"
    work = [json.loads(l) for l in open(work_dir / f"{tag}.work.jsonl")]
    results = {r["id"]: r for r in
               (json.loads(l) for l in open(work_dir / f"{tag}.results.jsonl"))}
    manifest = json.load(open(work_dir / f"{tag}.manifest.json"))
    out_recs = [json.loads(l) for l in open(out)]

    by_attempt = {}          # (gid, i) -> work item
    for it in work:
        gid, att = it["id"][1:].split(":a")
        by_attempt[(int(gid), int(att))] = it

    # 2-3. pure-Python reference loop, engine outcomes as the oracle
    rng = random.Random(a.seed)
    budget = a.per_graph * 4
    boards = manifest["boards"]
    colors = manifest["colors"]
    R = len(colors)
    n_draws_checked = 0
    ref_records = []
    for entry in boards:
        gid = entry["env_id"]
        with open(cfg.env_dir_abs / f"env_{gid}.pkl", "rb") as f:
            cells = list(pickle.load(f)["grid_graph"].nodes())
        kept = []
        attempts = 0
        while len(kept) < a.per_graph and attempts < budget:
            pts = rng.sample(cells, R + 1)
            i = attempts
            attempts += 1
            it = by_attempt[(gid, i)]
            drawn = {"target": [int(pts[-1][0]), int(pts[-1][1])],
                     "target_robot": [[int(pts[0][0]), int(pts[0][1])], colors[0]],
                     "helpers": [[[int(pts[h + 1][0]), int(pts[h + 1][1])],
                                  colors[h + 1]] for h in range(R - 1)]}
            for k, v in drawn.items():
                assert it[k] == v, (
                    f"STREAM BUG: board {gid} attempt {i} field {k}: "
                    f"reference drew {v}, bridge work item has {it[k]}")
            n_draws_checked += 1
            r = results[it["id"]]
            if r.get("status") == "ok" and r.get("records"):
                kept.append(i)
                ref_records.extend(r["records"])
        assert kept == entry["kept_idx"], (
            f"STREAM BUG: board {gid} kept {kept}, manifest {entry['kept_idx']}")
        assert attempts == entry["attempts_consumed"], (
            f"STREAM BUG: board {gid} consumed {attempts}, "
            f"manifest {entry['attempts_consumed']}")
    assert ref_records == out_recs, (
        f"OUTPUT BUG: reference concatenation has {len(ref_records)} records, "
        f"bridge out file has {len(out_recs)}")
    print(f"[check] reference loop PASS: {len(boards)} boards, "
          f"{n_draws_checked} draws verified against the speculative stream, "
          f"kept sets + output ({len(out_recs)} records) identical")

    # 4. outcomes-agree spot check vs production Python data
    prod = a.production
    if prod is None:
        cand = (SV_DIR / "scaling" / "data" / cfg.name /
                f"backward.shard{a.shard}of{a.nshards}.jsonl")
        prod = cand if cand.exists() else None
    if prod is None:
        print("[check] no production file for the spot check; skipped")
        return
    def inst_stream(records, gids):
        seen, order = set(), []
        for r in records:
            if r["env_id"] not in gids:
                continue
            key = (r["env_id"], json.dumps(r["target"]),
                   json.dumps(r["target_robot"]), json.dumps(r["helpers"]))
            if key not in seen:
                seen.add(key)
                order.append(key)
        return order
    gids = {e["env_id"] for e in boards}
    prod_stream = inst_stream((json.loads(l) for l in open(prod)), gids)
    rust_stream = inst_stream(out_recs, gids)
    if prod_stream == rust_stream:
        print(f"[check] production spot check PASS: kept instance stream "
              f"identical to {prod} ({len(rust_stream)} instances)")
    else:
        onlyp = [k for k in prod_stream if k not in rust_stream]
        onlyr = [k for k in rust_stream if k not in prod_stream]
        print(f"[check] kept streams differ vs production: "
              f"{len(onlyp)} python-only / {len(onlyr)} rust-only instances "
              f"-- outcome disagreement (ok/empty divergence), NOT a stream "
              f"bug (the reference-loop assertions above passed)")
        for k in (onlyp + onlyr)[:5]:
            print("   ", k)
    print("[check] ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
