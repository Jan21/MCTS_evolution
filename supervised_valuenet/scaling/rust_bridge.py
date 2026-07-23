"""Rust-engine adapter for `scaling.gen_data` (DESIGN.md section 6).

Python keeps orchestration and ALL random sampling; the Rust engine
(`rust_datagen/target/release/datagen`) does the solving/labeling. The kept
instance stream is EXACT-STREAM faithful to the Python labelers:

- backward (`scaling.backward_label` semantics): one shared
  `random.Random(seed)` across the shard's boards in board order. Per board
  the bridge snapshots `rng.getstate()`, speculatively draws the FULL attempt
  budget (`per_graph * 4` attempts, each one `rng.sample(cells, R+1)` over
  `list(grid_graph.nodes())` of the REAL env pkl), streams the attempts to
  ONE long-running engine process, and keeps the first `per_graph` attempts
  whose result is `status == "ok"` with nonempty records (production
  semantics: an instance is kept exactly when the rollout emitted records).
  It then rewinds (`rng.setstate`) and re-draws exactly
  `attempts_consumed` samples -- the index of the per_graph-th success + 1,
  or the full budget -- so the stream state entering the NEXT board is
  identical to Python's. When both engines agree on per-instance success
  (everything except cap-boundary cases), the kept instance stream is
  IDENTICAL to `scaling.backward_label`'s.
- forward (`move_planner.generate` semantics): per-board
  `random.Random((seed*100003 + env_id) ^ 0x9E3779B9)`; the budget
  (`per_board * 5` attempts: `rng.sample(cells_all, R+1)` with `cells_all`
  y-major, plus `rng.randrange(R)`) is presampled per board, everything runs
  as one batch, and the first `per_board` attempts with `status == "solved"`
  are kept in attempt order. No shared stream, no realignment needed.

Work items, raw engine results and a per-board manifest are kept under
`scaling/data/<config>/rust_work/` for audit.

    python -m scaling.rust_bridge --config g16r6 --system backward \
        --per-graph 6 --nshards 8 --shard 0 --seed 0 --threads 16
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHILD_FLAG = "RR_RUST_BRIDGE_CHILD"
DEFAULT_ENGINE = REPO / "rust_datagen" / "target" / "release" / "datagen"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--system", choices=["forward", "backward"], required=True)
    p.add_argument("--per-graph", type=int, default=20,
                   help="instances kept per board (forward: per-board)")
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N boards of this shard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="default: scaling/data/<config>/<system>.rust.jsonl "
                        "(never overwrites an existing file; see --force)")
    p.add_argument("--threads", type=int, default=16,
                   help="engine threads (shared box; keep <= 16 unless briefly benchmarking)")
    p.add_argument("--score-candidates", action="store_true", help="forward only")
    p.add_argument("--max-candidates", type=int, default=14, help="backward only")
    p.add_argument("--vocab", default="base", choices=["base", "b1", "b2"],
                   help="backward only: plan-language vocabulary (house rule: "
                        "never mix vocabularies in one dataset; name outputs "
                        "by vocabulary, e.g. backward_b2.rust.jsonl)")
    p.add_argument("--budget-iters", type=int, default=None,
                   help="backward deterministic rollout budget "
                        "(default: engine default, calibrated >> 120 s of Python work)")
    p.add_argument("--engine-bin", default=str(DEFAULT_ENGINE))
    p.add_argument("--force", action="store_true",
                   help="allow overwriting an existing --out file")
    p.add_argument("--work-tag", default=None,
                   help="basename tag for rust_work files (default: out stem)")
    return p.parse_args(argv)


def main():
    a = parse_args()
    if os.environ.get(CHILD_FLAG) != "1":
        # One config per process: repo modules read RR_* at import, and the
        # env pkls unpickle classes from repo modules. Re-exec with the
        # config's environment (idempotent when gen_data already set it).
        from scaling import configs
        cfg = configs.get(a.config)
        env = {**os.environ, **configs.env(cfg), CHILD_FLAG: "1"}
        env["PYTHONPATH"] = str(REPO) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        rc = subprocess.run([sys.executable, "-m", "scaling.rust_bridge",
                             *sys.argv[1:]], env=env, cwd=str(REPO)).returncode
        sys.exit(rc)
    _child(a)


# ===========================================================================
# child (RR_* set)
# ===========================================================================

def _resolve(a):
    from scaling import configs
    cfg = configs.get(a.config)
    if not 0 <= a.shard < a.nshards:
        raise SystemExit("--shard must be in [0, --nshards)")
    ids = sorted({i for s in a.splits.split(",") for i in cfg.ids(s)})
    ids = ids[a.shard::a.nshards]
    if a.limit is not None:
        ids = ids[:a.limit]
    missing = [i for i in ids if not (cfg.env_dir_abs / f"env_{i}.pkl").exists()]
    if missing:
        raise SystemExit(f"{len(missing)}/{len(ids)} board pkls missing from "
                         f"{cfg.env_dir_abs} (first: {missing[:5]})")

    out = (Path(a.out) if a.out else
           REPO / "scaling" / "data" / cfg.name / f"{a.system}.rust.jsonl")
    if a.out is None and a.nshards > 1:
        out = out.with_name(f"{out.stem}.shard{a.shard}of{a.nshards}.jsonl")
    if out.exists() and not a.force:
        raise SystemExit(f"refusing to overwrite existing {out} "
                         f"(pass --force or choose another --out)")
    out.parent.mkdir(parents=True, exist_ok=True)

    engine = Path(a.engine_bin)
    if not engine.exists():
        raise SystemExit(f"engine binary not found: {engine}\n"
                         f"build it: cargo build --release "
                         f"(in {REPO / 'rust_datagen'})")

    work_dir = REPO / "scaling" / "data" / cfg.name / "rust_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    tag = a.work_tag or out.stem
    return cfg, ids, out, engine, work_dir, tag


def _load_board(cfg, gid):
    with open(cfg.env_dir_abs / f"env_{gid}.pkl", "rb") as f:
        d = pickle.load(f)
    return d


def _board_obj(cfg, gid, grid_data):
    return {"env_id": int(gid), "n": int(cfg.grid),
            "grid_data": list(grid_data)}


def _xy(p):
    return [int(p[0]), int(p[1])]


# -- backward ---------------------------------------------------------------

class EngineProc:
    """One long-running `datagen run --work - --out -` co-process.

    A reader thread collects result lines into a dict by id (teeing the raw
    line to `results_path`); `collect` blocks until the given ids are all
    present. Work lines are teed to `work_path`.
    """

    def __init__(self, engine, threads, work_path, results_path):
        self.proc = subprocess.Popen(
            [str(engine), "run", "--work", "-", "--out", "-",
             "--threads", str(threads), "--quiet"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.results = {}
        self.cond = threading.Condition()
        self.work_f = open(work_path, "w")
        self.res_f = open(results_path, "w")
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.proc.stdout:
            self.res_f.write(line)
            r = json.loads(line)
            with self.cond:
                self.results[r["id"]] = r
                self.cond.notify_all()
        with self.cond:
            self.cond.notify_all()

    def send(self, items):
        for it in items:
            line = json.dumps(it)
            self.work_f.write(line + "\n")
            self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def collect(self, ids):
        with self.cond:
            while not all(i in self.results for i in ids):
                if self.proc.poll() is not None and not all(
                        i in self.results for i in ids):
                    raise RuntimeError(
                        f"engine exited rc={self.proc.returncode} before "
                        f"returning all results")
                self.cond.wait(timeout=1.0)
            return [self.results.pop(i) for i in ids]

    def close(self):
        self.proc.stdin.close()
        self.reader.join()
        rc = self.proc.wait()
        self.work_f.close()
        self.res_f.close()
        if rc != 0:
            raise RuntimeError(f"engine exited rc={rc}")


def _backward(a, cfg, ids, out, engine, work_dir, tag):
    budget = a.per_graph * 4
    rng = random.Random(a.seed)

    # colors: scaling.backward_label reads them from
    # GridEnv.from_env(graphs[0]) -> instances[0]; same pkl fields here,
    # without triggering from_env's eager table/cache build.
    first = _load_board(cfg, ids[0])
    inst0 = first["instances"][0]
    colors = [inst0["target_robot"].color] + [h.color for h in inst0["helper_robots"]]
    R = len(colors)

    ep = EngineProc(engine, a.threads,
                    work_dir / f"{tag}.work.jsonl",
                    work_dir / f"{tag}.results.jsonl")
    manifest = []
    n_rec = n_inst = 0
    t0 = time.time()
    try:
        with open(out, "w") as f:
            for gid in ids:
                d = first if gid == ids[0] else _load_board(cfg, gid)
                grid_data = d["grid_data"]
                cells = list(d["grid_graph"].nodes())
                board = _board_obj(cfg, gid, grid_data)

                snap = rng.getstate()
                items = []
                for i in range(budget):
                    pts = rng.sample(cells, R + 1)
                    items.append({
                        "task": "backward_rollout",
                        "id": f"b{gid}:a{i}",
                        "board": board,
                        "target": _xy(pts[-1]),
                        "target_robot": [_xy(pts[0]), colors[0]],
                        "helpers": [[_xy(pts[h + 1]), colors[h + 1]]
                                    for h in range(R - 1)],
                        "max_candidates": a.max_candidates,
                        "max_iters": 4000,
                        "max_frontier": 40000,
                        "dependent_edge_weight": 2,
                        **({"vocab": a.vocab} if a.vocab != "base" else {}),
                        **({"budget": {"solver_iters": a.budget_iters}}
                           if a.budget_iters is not None else {}),
                    })
                ep.send(items)
                results = ep.collect([it["id"] for it in items])

                kept_idx = []
                consumed = budget
                for i, r in enumerate(results):
                    if r.get("status") == "ok" and r.get("records"):
                        kept_idx.append(i)
                        if len(kept_idx) == a.per_graph:
                            consumed = i + 1
                            break
                # realign the shared stream exactly as Python consumed it
                rng.setstate(snap)
                for _ in range(consumed):
                    rng.sample(cells, R + 1)

                for i in kept_idx:
                    for rec in results[i]["records"]:
                        f.write(json.dumps(rec) + "\n")
                        n_rec += 1
                n_inst += len(kept_idx)
                manifest.append({"env_id": gid, "kept_idx": kept_idx,
                                 "attempts_consumed": consumed,
                                 "budget": budget})
                print(f"graph {gid}: {len(kept_idx)} instances, {n_rec} records "
                      f"so far ({time.time() - t0:.0f}s)", flush=True)
    finally:
        ep.close()
    with open(work_dir / f"{tag}.manifest.json", "w") as mf:
        json.dump({"config": cfg.name, "system": "backward", "seed": a.seed,
                   "engine": "rust", "vocab": a.vocab,
                   "per_graph": a.per_graph, "nshards": a.nshards,
                   "shard": a.shard, "colors": colors,
                   "boards": manifest}, mf, indent=1)
    print(f"done: {n_inst} instances, {n_rec} records -> {out} "
          f"({time.time() - t0:.1f}s)")


# -- forward ----------------------------------------------------------------

def _forward(a, cfg, ids, out, engine, work_dir, tag):
    budget = a.per_graph * 5
    R = cfg.robots
    n = cfg.grid
    cells_all = [(x, y) for y in range(n) for x in range(n)]

    # Adaptive waves: the production loop stops each board after per_board
    # solves (~2x per_board attempts consumed on average), so sending the
    # full budget wastes most of the work. Waves send growing prefixes; the
    # kept selection below only ever needs the consumed prefix.
    caps = sorted({min(budget, 2 * a.per_graph),
                   min(budget, 3 * a.per_graph), budget})
    ep = EngineProc(engine, a.threads,
                    work_dir / f"{tag}.work.jsonl",
                    work_dir / f"{tag}.results.jsonl")
    kept = {}                        # gid -> (kept_idx, {iid: result})
    n_sent = 0
    try:
        # Presample every board's full attempt budget (per-board RNG:
        # outcomes are independent, so unsent attempts never affect the
        # sampled stream), streaming each board's first wave to the engine
        # as it is sampled so sampling overlaps compute.
        t0 = time.time()
        items = {}                   # gid -> [work item] (budget long)
        pending = {}                 # gid -> attempts sent
        for gid in ids:
            d = _load_board(cfg, gid)
            board = _board_obj(cfg, gid, d["grid_data"])
            rng = random.Random((a.seed * 100003 + gid) ^ 0x9E3779B9)
            lst = []
            for i in range(budget):
                cells = rng.sample(cells_all, R + 1)
                tidx = rng.randrange(R)
                lst.append({
                    "task": "forward_instance",
                    "id": f"f{gid}:a{i}",
                    "board": board,
                    "robots": [_xy(p) for p in cells[:R]],
                    "target_idx": tidx,
                    "target": _xy(cells[R]),
                    "max_expansions": 40000,
                    "full_policy": True,
                    "full_policy_max_ctg": 6,
                    "score_candidates": bool(a.score_candidates),
                })
            items[gid] = lst
            ep.send(lst[:caps[0]])
            n_sent += min(caps[0], budget)
            pending[gid] = caps[0]
        print(f"[bridge] presampled {len(ids)} boards x {budget} attempts, "
              f"first wave ({caps[0]}/board) streamed while sampling "
              f"({time.time() - t0:.1f}s)", flush=True)

        for cap in caps:
            wave = []
            for gid in list(pending):
                lo = pending[gid]
                if lo >= cap:
                    continue
                wave += items[gid][lo:cap]
                pending[gid] = cap
            if wave:
                n_sent += len(wave)
                ep.send(wave)
            for gid in list(pending):
                sent = pending[gid]
                res = ep.collect([it["id"] for it in items[gid][:sent]])
                kept_idx = []
                for i, r in enumerate(res):
                    if r.get("status") == "solved":
                        kept_idx.append(i)
                        if len(kept_idx) == a.per_graph:
                            break
                if len(kept_idx) == a.per_graph or sent == budget:
                    kept[gid] = (kept_idx,
                                 {items[gid][i]["id"]: res[i] for i in kept_idx})
                    del pending[gid]
                else:                # needs the next wave; re-stash results
                    with ep.cond:
                        for it, r in zip(items[gid][:sent], res):
                            ep.results[it["id"]] = r
    finally:
        ep.close()
    print(f"[bridge] engine done: {n_sent} of {len(ids) * budget} budgeted "
          f"attempts computed (waves at {caps})", flush=True)

    manifest = []
    n_rec = n_inst = 0
    with open(out, "w") as f:
        for gid in ids:
            kept_idx, res = kept[gid]
            for i in kept_idx:
                for rec in res[items[gid][i]["id"]]["records"]:
                    f.write(json.dumps(rec) + "\n")
                    n_rec += 1
            n_inst += len(kept_idx)
            manifest.append({"env_id": gid, "kept_idx": kept_idx})
    with open(work_dir / f"{tag}.manifest.json", "w") as mf:
        json.dump({"config": cfg.name, "system": "forward", "seed": a.seed,
                   "per_board": a.per_graph, "nshards": a.nshards,
                   "shard": a.shard, "boards": manifest}, mf, indent=1)
    print(f"done: {n_inst} instances, {n_rec} records -> {out} "
          f"({time.time() - t0:.1f}s)")


def _child(a):
    cfg, ids, out, engine, work_dir, tag = _resolve(a)
    print(f"[bridge] {cfg.name}/{a.system}: {len(ids)} boards, "
          f"engine {engine}, threads {a.threads} -> {out}", flush=True)
    if a.system == "backward":
        _backward(a, cfg, ids, out, engine, work_dir, tag)
    else:
        _forward(a, cfg, ids, out, engine, work_dir, tag)


if __name__ == "__main__":
    main()
