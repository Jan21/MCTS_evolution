"""Fidelity gauge (PROBLEM.md 4.1): self-play labels vs exact optima, depth 0.

The supervised track's calibrated instrument: argmin agreement of a corpus's
depth-0 decision labels against exact labels of the SAME decisions (FINDINGS
74: ~91% -> downstream-equivalent, ~89% -> solve rate kept / optimality lost,
~82% -> collapse). Board side <= 64 only (the exact engine's envelope).

    PYTHONPATH=supervised_valuenet:self_play_robots python -m spr.gauge \
        --config g24r4 --records runs/spr/selfplay/g24r4_iter1/records.jsonl \
        --sample 200 --out results/selfplay/g24r4_iter1/gauge.json

Steps: sample instances from the record file's depth-0 groups; label them
exactly with the Rust engine (`rust_datagen` `backward_rollout` items through
`scaling.rust_bridge.EngineProc`, ALL candidates, 4000-iteration solve
budget -- the exact labeler's own settings); write both subsets; run
`nn_labeler.audit_descent` (the instrument, unchanged) and lift its
`argmin_agreement` + gap statistics into the output JSON. Last line:
`SPR GAUGE DONE <out> argmin=<x>`.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

from spr import REPO, SV


def _key(r):
    return (r["env_id"], tuple(r["target"]), tuple(r["target_robot"][0]),
            r["target_robot"][1], tuple((tuple(h[0]), h[1]) for h in r["helpers"]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--records", required=True, help="self-play (or any) corpus jsonl")
    p.add_argument("--boards-dir", default=None,
                   help="board pkl dir (default: the records' boards_dir field, else "
                        "the config's env dir)")
    p.add_argument("--sample", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max-iters", type=int, default=4000)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    from scaling.configs import get
    from scaling.rust_bridge import EngineProc, DEFAULT_ENGINE
    cfg = get(a.config)
    if cfg.grid > 64:
        raise SystemExit("exact engine envelope is n <= 64")
    rec_path = Path(a.records)
    if not rec_path.is_absolute():
        rec_path = REPO / rec_path
    out = Path(a.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / (out.stem + ".work")
    work.mkdir(exist_ok=True)

    # depth-0 groups by instance
    by_inst = {}
    dir_of = {}                     # env_id -> board dir (per record, multi-iteration safe)
    n_rec = 0
    for line in open(rec_path):
        r = json.loads(line)
        n_rec += 1
        if r.get("boards_dir"):
            dir_of[int(r["env_id"])] = r["boards_dir"]
        if r.get("depth") == 0:
            by_inst.setdefault(_key(r), []).append(r)
    default_dir = (Path(a.boards_dir).resolve() if a.boards_dir else cfg.env_dir_abs)
    boards_dir = default_dir
    keys = sorted(by_inst, key=repr)
    rng = random.Random(a.seed)
    rng.shuffle(keys)
    keys = keys[:a.sample]
    print(f"[gauge] {n_rec} records, {len(by_inst)} depth-0 instances, sampling {len(keys)}; "
          f"boards={boards_dir}", flush=True)

    # exact labels via the Rust engine
    sub_sp = work / "selfplay_subset.jsonl"
    sub_ex = work / "exact_subset.jsonl"
    with open(sub_sp, "w") as f:
        for k in keys:
            for r in by_inst[k]:
                f.write(json.dumps(r) + "\n")
    boards = {}
    items = []
    for i, k in enumerate(keys):
        env_id = k[0]
        if env_id not in boards:
            bd = Path(dir_of.get(int(env_id), default_dir))
            with open(bd / f"env_{env_id}.pkl", "rb") as fh:
                boards[env_id] = list(pickle.load(fh)["grid_data"])
        items.append({
            "task": "backward_rollout", "id": f"g{i}",
            "board": {"env_id": int(env_id), "n": int(cfg.grid), "grid_data": boards[env_id]},
            "target": list(k[1]), "target_robot": [list(k[2]), k[3]],
            "helpers": [[list(h), c] for h, c in k[4]],
            "max_candidates": 64, "max_iters": a.max_iters, "max_frontier": 40000,
            "dependent_edge_weight": 2,
        })
    t0 = time.time()
    ep = EngineProc(DEFAULT_ENGINE, a.threads, work / "engine.work.jsonl",
                    work / "engine.results.jsonl")
    n_ok = n_ex = 0
    try:
        ep.send(items)
        results = ep.collect([it["id"] for it in items])
    finally:
        ep.close()
    with open(sub_ex, "w") as f:
        for r in results:
            if r.get("status") == "ok" and r.get("records"):
                n_ok += 1
                for rec in r["records"]:
                    if rec.get("depth") == 0:
                        f.write(json.dumps(rec) + "\n")
                        n_ex += 1
    print(f"[gauge] exact engine: {n_ok}/{len(items)} instances labeled, {n_ex} depth-0 "
          f"records ({time.time() - t0:.0f}s)", flush=True)

    audit_out = work / "audit.json"
    env = dict(os.environ, PYTHONPATH=f"{SV}:{SV.parent / 'self_play_robots'}")
    rc = subprocess.call([sys.executable, "-m", "nn_labeler.audit_descent",
                          "--exact", str(sub_ex), "--descent", str(sub_sp),
                          "--out", str(audit_out)], cwd=SV, env=env)
    if rc != 0:
        raise SystemExit("[gauge] audit_descent failed")
    audit = json.loads(audit_out.read_text())
    summ = audit.get("summary", audit)
    payload = {"config": cfg.name, "records": str(rec_path), "boards_dir": str(boards_dir),
               "sample": len(keys), "exact_labeled": n_ok, "seed": a.seed,
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "audit_summary": summ, "audit_file": str(audit_out)}
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n")
    os.replace(tmp, out)
    arg = None
    for k2 in ("argmin_agreement", "argmin_agreement_depth0"):
        if isinstance(summ, dict) and k2 in summ:
            arg = summ[k2]
            break
    print(f"SPR GAUGE DONE {out} argmin={arg}", flush=True)


if __name__ == "__main__":
    main()
