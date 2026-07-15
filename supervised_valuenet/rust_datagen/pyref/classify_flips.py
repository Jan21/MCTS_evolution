"""Attempt-level classification of the g16r6 backward regen instance flips.

Task D's regeneration of the g16r6 backward data reproduced 90.27% of the
production kept-instance stream; the flips were attributed to production's
120 s wall-clock timeout, but three of five sampled flips replayed fast
(0.0 / 0.7 / 17.7 s), which a timeout cannot explain. This tool settles the
question at the attempt level, for ALL boards, not a sample:

  stage reconstruct
      Re-simulates the production `scaling.backward_label` loop per shard --
      one shared `random.Random(seed)` in board order, `rng.sample(cells,
      R+1)` per attempt -- and aligns the draws against the SHIPPED
      `backward.jsonl` kept stream. Every draw is thereby classified as
      python-kept (it is the next kept instance of that board) or
      python-dropped (production produced no records for it: rollout empty,
      timeout, or the labeler's bare `except Exception`). The reconstruction
      is exact: it asserts every kept instance is matched in order within
      the attempt budget. A sampled draw that matched a LATER kept instance
      out of order would break the alignment loudly, not silently.
      Writes <prefix>.attempts.jsonl + <prefix>.work.jsonl (one
      backward_rollout item per python attempt -- rust judges the very draws
      python judged, immune to the flip-cascade resampling that makes
      whole-file diffs ambiguous).

  stage classify (after `datagen run` on the work file)
      Joins attempts with rust results:
        python-kept  vs rust ok      -> record streams compared field-for-field
                                        (mismatches exported in attempts mode
                                        for compare_rollouts.py -- the tie
                                        classes are adjudicated there);
        python-kept  vs rust empty   -> FLIP, label divergence: BLOCKER class;
        python-dropped vs rust empty -> agreement (rollout genuinely empty);
        python-dropped vs rust ok    -> FLIP: python replay (stage replay)
                                        decides the class.
      Writes <prefix>.flips.jsonl.

  stage replay
      Replays every python-dropped/rust-ok flip in today's Python
      (`nn.generate.rollout`, lazy env, SIGALRM cap): wall > 120 s ->
      production-timeout class; wall <= 120 s with records equal to rust's
      -> a production-run wall-clock/exception artifact (reported by replay
      time; today's engines AGREE); records nonempty but different from
      rust's -> exported for tie-class adjudication; records EMPTY ->
      label divergence: BLOCKER class. Writes <prefix>.classification.json.

    python rust_datagen/pyref/classify_flips.py --stage reconstruct \
        --out-prefix rust_datagen/pyref/out/flips/g16r6
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

PYREF_DIR = Path(__file__).resolve().parent
SV_DIR = PYREF_DIR.parent.parent
for _p in (str(SV_DIR), str(PYREF_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CHILD_FLAG = "RR_PYREF_CHILD_FLIPS"
DEFAULT_ENGINE = SV_DIR / "rust_datagen" / "target" / "release" / "datagen"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="g16r6")
    p.add_argument("--python-file", default=None,
                   help="shipped combined.jsonl (default scaling/data/<cfg>/backward.jsonl)")
    p.add_argument("--stage", choices=["reconstruct", "classify", "replay"],
                   required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--nshards", type=int, default=8)
    p.add_argument("--per-graph", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-candidates", type=int, default=14)
    p.add_argument("--replay-cap", type=int, default=150,
                   help="python replay wall cap (s); > cap classifies as "
                        "past-production-timeout")
    p.add_argument("--workers", type=int, default=12, help="replay stage, <=16")
    return p.parse_args(argv)


def main():
    a = parse_args()
    if os.environ.get(CHILD_FLAG) != "1":
        from scaling import configs
        cfg = configs.get(a.config)
        env = {**os.environ, **configs.env(cfg), CHILD_FLAG: "1",
               "PYTHONPATH": str(SV_DIR)}
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                             *sys.argv[1:]], env=env, cwd=str(SV_DIR)).returncode
        sys.exit(rc)
    _child(a)


# ---------------------------------------------------------------------------


def _inst_key_from_pts(pts, colors):
    return json.dumps({
        "target": [int(pts[-1][0]), int(pts[-1][1])],
        "target_robot": [[int(pts[0][0]), int(pts[0][1])], colors[0]],
        "helpers": [[[int(pts[i + 1][0]), int(pts[i + 1][1])], colors[i + 1]]
                    for i in range(len(colors) - 1)]})


def _inst_key_from_rec(rec):
    return json.dumps({"target": rec["target"],
                       "target_robot": rec["target_robot"],
                       "helpers": rec["helpers"]})


def _load_python_streams(path):
    """env_id -> ordered kept instance keys; and key -> records."""
    streams = defaultdict(list)
    records = defaultdict(list)
    with open(path) as f:
        for raw in f:
            rec = json.loads(raw)
            k = (rec["env_id"], _inst_key_from_rec(rec))
            if k not in records:
                streams[rec["env_id"]].append(_inst_key_from_rec(rec))
            records[k].append(rec)
    return streams, records


def _reconstruct(a, cfg):
    from scaling import configs
    ids_all = sorted({i for s in "train,val,test".split(",")
                      for i in cfg.ids(s)})
    py_path = a.python_file or str(SV_DIR / "scaling" / "data" / cfg.name
                                   / "backward.jsonl")
    streams, _records = _load_python_streams(py_path)

    prefix = Path(a.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    att_f = open(f"{prefix}.attempts.jsonl", "w")
    work_f = open(f"{prefix}.work.jsonl", "w")

    n_att = n_kept = n_drop = 0
    boards_bad = []
    for shard in range(a.nshards):
        ids = ids_all[shard::a.nshards]
        rng = random.Random(a.seed)
        # colors exactly as backward_label derives them (first board's s0)
        with open(cfg.env_dir_abs / f"env_{ids[0]}.pkl", "rb") as f:
            first = pickle.load(f)
        inst0 = first["instances"][0]
        colors = ([inst0["target_robot"].color]
                  + [h.color for h in inst0["helper_robots"]])
        R = len(colors)
        for gid in ids:
            if gid == ids[0]:
                d = first
            else:
                with open(cfg.env_dir_abs / f"env_{gid}.pkl", "rb") as f:
                    d = pickle.load(f)
            cells = list(d["grid_graph"].nodes())
            board = {"env_id": int(gid), "n": int(cfg.grid),
                     "grid_data": list(d["grid_data"])}
            expected = list(streams.get(gid, []))
            kept = attempts = 0
            pos = 0
            while kept < a.per_graph and attempts < a.per_graph * 4:
                attempts += 1
                pts = rng.sample(cells, R + 1)
                key = _inst_key_from_pts(pts, colors)
                if pos < len(expected) and key == expected[pos]:
                    outcome, pos, kept = "kept", pos + 1, kept + 1
                    n_kept += 1
                else:
                    outcome = "dropped"
                    n_drop += 1
                n_att += 1
                iid = f"b{gid}:a{attempts - 1}"
                inst = json.loads(key)
                att_f.write(json.dumps({
                    "shard": shard, "env_id": gid, "iid": iid,
                    "outcome_py": outcome, "instance": inst}) + "\n")
                work_f.write(json.dumps({
                    "task": "backward_rollout", "id": iid, "board": board,
                    "target": inst["target"],
                    "target_robot": inst["target_robot"],
                    "helpers": inst["helpers"],
                    "max_candidates": a.max_candidates,
                    "max_iters": 4000, "max_frontier": 40000,
                    "dependent_edge_weight": 2}) + "\n")
            if pos != len(expected):
                boards_bad.append((gid, pos, len(expected)))
    att_f.close()
    work_f.close()
    print(f"[reconstruct] {n_att} attempts ({n_kept} kept, {n_drop} dropped) "
          f"across {a.nshards} shards")
    if boards_bad:
        print(f"[reconstruct] ALIGNMENT FAILURE on {len(boards_bad)} boards: "
              f"{boards_bad[:10]}")
        sys.exit(1)
    print("[reconstruct] alignment OK: every shipped kept instance matched "
          "in order on every board")


def _classify(a, cfg):
    prefix = Path(a.out_prefix)
    py_path = a.python_file or str(SV_DIR / "scaling" / "data" / cfg.name
                                   / "backward.jsonl")
    _streams, records = _load_python_streams(py_path)
    rust = {}
    with open(f"{prefix}.results.jsonl") as f:
        for raw in f:
            r = json.loads(raw)
            rust[r["id"]] = r

    counts = defaultdict(int)
    flips = []
    mism_py = open(f"{prefix}.recmismatch.py.jsonl", "w")
    mism_rs = open(f"{prefix}.recmismatch.rust.jsonl", "w")
    with open(f"{prefix}.attempts.jsonl") as f:
        for raw in f:
            att = json.loads(raw)
            r = rust.get(att["iid"])
            if r is None:
                counts["missing_rust_result"] += 1
                continue
            r_ok = r.get("status") == "ok" and r.get("records")
            if att["outcome_py"] == "kept":
                if not r_ok:
                    counts["FLIP_py_kept_rust_" + r.get("status", "?")] += 1
                    flips.append({**att, "flip": "py_kept_rust_not",
                                  "rust_status": r.get("status")})
                    continue
                pk = (att["env_id"], json.dumps(att["instance"]))
                py_recs = records[pk]
                if py_recs == r["records"]:
                    counts["kept_records_identical"] += 1
                else:
                    counts["kept_records_mismatch"] += 1
                    mism_py.write(json.dumps({
                        "iid": att["iid"], "env_id": att["env_id"],
                        "status": "ok", "instance": att["instance"],
                        "records": py_recs}) + "\n")
                    mism_rs.write(json.dumps(
                        {"id": att["iid"], "status": "ok",
                         "records": r["records"]}) + "\n")
            else:
                if r_ok:
                    counts["FLIP_py_dropped_rust_ok"] += 1
                    flips.append({**att, "flip": "py_dropped_rust_ok",
                                  "rust_records": r["records"]})
                else:
                    counts["agree_dropped_" + r.get("status", "?")] += 1
    mism_py.close()
    mism_rs.close()
    with open(f"{prefix}.flips.jsonl", "w") as f:
        for fl in flips:
            f.write(json.dumps(fl) + "\n")
    per_board = defaultdict(int)
    for fl in flips:
        per_board[fl["env_id"]] += 1
    hist = defaultdict(int)
    for _gid, k in per_board.items():
        hist[k] += 1
    print(f"[classify] {json.dumps(dict(counts), indent=1)}")
    print(f"[classify] flip boards: {len(per_board)}; flips-per-board "
          f"histogram: {dict(sorted(hist.items()))}")
    print(f"[classify] flips -> {prefix}.flips.jsonl; record mismatches -> "
          f"{prefix}.recmismatch.*.jsonl (adjudicate via compare_rollouts)")


# -- replay stage -------------------------------------------------------------

class _Timeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _Timeout()


def _replay_one(job):
    fl, n, cap, mc = job
    import common
    from skeleton.astar import AStar
    from nn.generate import rollout
    from GridEnv import State, Robot_at

    grid_data = _replay_one.grid_cache.get(fl["env_id"])
    env = common.build_env(grid_data, n, mode="lazy", weight=2,
                           table_workers=1)
    inst = fl["instance"]
    st = State(target=tuple(inst["target"]),
               target_robot=Robot_at(position=tuple(inst["target_robot"][0]),
                                     color=inst["target_robot"][1]),
               helpers=[Robot_at(position=tuple(p), color=c)
                        for p, c in inst["helpers"]])
    solver = AStar(max_iters=4000, max_frontier=40_000)
    signal.signal(signal.SIGALRM, _raise_timeout)
    t0 = time.time()
    signal.alarm(cap)
    try:
        recs = rollout(env, st, solver, fl["env_id"], mc)
        status = "ok" if recs else "empty"
    except _Timeout:
        status, recs = "over_cap", []
    except Exception as e:                                   # noqa: BLE001
        status, recs = f"error:{type(e).__name__}", []
    finally:
        signal.alarm(0)
    wall = time.time() - t0
    return fl["iid"], fl["env_id"], status, wall, recs


class _GridCache:
    def __init__(self, env_dir):
        self.env_dir = env_dir
        self.cache = {}

    def get(self, gid):
        if gid not in self.cache:
            with open(self.env_dir / f"env_{gid}.pkl", "rb") as f:
                self.cache[gid] = pickle.load(f)["grid_data"]
        return self.cache[gid]


def _replay(a, cfg):
    prefix = Path(a.out_prefix)
    flips = [json.loads(l) for l in open(f"{prefix}.flips.jsonl")]
    todo = [fl for fl in flips if fl["flip"] == "py_dropped_rust_ok"]
    _replay_one.grid_cache = _GridCache(cfg.env_dir_abs)

    jobs = [(fl, cfg.grid, a.replay_cap, a.max_candidates) for fl in todo]
    workers = max(1, min(a.workers, 16))
    if workers == 1 or not jobs:
        results = [_replay_one(j) for j in jobs]
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=mp.get_context("fork")) as ex:
            results = list(ex.map(_replay_one, jobs))

    by_iid = {fl["iid"]: fl for fl in todo}
    classes = defaultdict(list)
    mism_py = open(f"{prefix}.flipmismatch.py.jsonl", "w")
    mism_rs = open(f"{prefix}.flipmismatch.rust.jsonl", "w")
    for iid, gid, status, wall, recs in results:
        fl = by_iid[iid]
        entry = {"iid": iid, "env_id": gid, "replay_status": status,
                 "replay_wall_s": round(wall, 1)}
        if status == "over_cap" or wall > 120:
            classes["production_timeout_gt120s"].append(entry)
        elif status == "ok":
            if recs == fl["rust_records"]:
                classes["fast_drop_engines_agree"].append(entry)
            else:
                classes["fast_drop_records_differ"].append(entry)
                mism_py.write(json.dumps({
                    "iid": iid, "env_id": gid, "status": "ok",
                    "instance": fl["instance"], "records": recs}) + "\n")
                mism_rs.write(json.dumps(
                    {"id": iid, "status": "ok",
                     "records": fl["rust_records"]}) + "\n")
        elif status == "empty":
            classes["BLOCKER_py_empty_rust_ok"].append(entry)
        else:
            classes["replay_" + status].append(entry)
    mism_py.close()
    mism_rs.close()

    kept_flips = [fl for fl in flips if fl["flip"] == "py_kept_rust_not"]
    report = {
        "config": cfg.name,
        "replay_cap_s": a.replay_cap,
        "n_flips_py_dropped_rust_ok": len(todo),
        "n_flips_py_kept_rust_not": len(kept_flips),
        "classes": {k: {"n": len(v), "entries": v}
                    for k, v in sorted(classes.items())},
    }
    with open(f"{prefix}.classification.json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: len(v) for k, v in sorted(classes.items())},
                     indent=1))
    print(f"[replay] full detail -> {prefix}.classification.json")


def _child(a):
    from scaling import configs
    cfg = configs.get(a.config)
    if a.stage == "reconstruct":
        _reconstruct(a, cfg)
    elif a.stage == "classify":
        _classify(a, cfg)
    else:
        _replay(a, cfg)


if __name__ == "__main__":
    main()
