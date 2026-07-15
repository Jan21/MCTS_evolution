"""Cross-engine benchmarks: identical work through Python and Rust (Agent D).

Measures, per config (g16r4 g16r6 g16r8 g24r4 g32r4 by default):

- precompute: `GridEnv.from_env` cold (the production per-board precompute)
  vs `datagen` board compile on the same boards, plus the all-pairs table
  block alone (`nn.gen_grids.all_pairs` + `independent_paths` re-weighted at
  2, the exact `from_env` recipe) — the >=100x per-core gate;
- backward labeling: the same instance sets (taken from a rust_bridge work
  file, trimmed to the attempts the production loop would consume) through
  `nn.generate.rollout` (Python, single core, lazy env — the FASTEST correct
  Python implementation, i.e. conservative for the ratio) and through
  `datagen run` at --threads 1 / 16 (and 32 with --with-32) — the >=50x
  per-core gate. Per-instance Python wall time is paired with the engine's
  reported solve iterations for the deterministic-budget calibration;
- forward labeling: same shape via `move_planner.oracle` with the
  label_board candidate-scoring loop vs `datagen run` — the >=50x gate.

Every measurement is appended as a JSON row to
`pyref/cache/bench/results.jsonl`; per-stage logs live next to it. RR_*
configs are honored by re-spawning the child stages with the config's env
(one config per process), exactly like the other pyref tools.

    python rust_datagen/pyref/bench_compare.py --configs g16r6 --stages backward
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PYREF_DIR = Path(__file__).resolve().parent
RD_DIR = PYREF_DIR.parent
SV_DIR = RD_DIR.parent
BENCH_DIR = PYREF_DIR / "cache" / "bench"
ENGINE = RD_DIR / "target" / "release" / "datagen"
PY = sys.executable

sys.path.insert(0, str(SV_DIR))
sys.path.insert(0, str(PYREF_DIR))

# per-config workload sizes (python single-core cost bounds these)
BOARDS = {"precompute": {"16": 2, "24": 1, "32": 1},
          "backward": 2, "forward": 2}
# backward sample sizes: large enough that per-board compile amortizes as in
# production (which labels 24-80 attempts per board, so compile is <5% of a
# board's work — tiny samples otherwise understate the rust side)
PER_GRAPH_B = {"g16r4": 20, "g16r6": 6, "g16r8": 2, "g24r4": 10, "g32r4": 10}
PER_BOARD_F = {"g16r4": 5, "g16r6": 5, "g16r8": 4, "g24r4": 5, "g32r4": 4}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--configs", default="g16r4,g16r6,g16r8,g24r4,g32r4")
    p.add_argument("--stages", default="precompute,backward,forward")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--with-32", action="store_true",
                   help="also time --threads 32 (check uptime first!)")
    p.add_argument("--seed", type=int, default=0)
    # child-mode plumbing
    p.add_argument("--stage", default=None, help=argparse.SUPPRESS)
    p.add_argument("--config", default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def emit(row):
    row = {"ts": round(time.time(), 1), **row}
    print("RESULT " + json.dumps(row), flush=True)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    with open(BENCH_DIR / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def time_engine(work, out, threads):
    t0 = time.time()
    rc = subprocess.run([str(ENGINE), "run", "--work", str(work),
                         "--out", str(out), "--threads", str(threads),
                         "--quiet"]).returncode
    dt = time.time() - t0
    if rc != 0:
        raise SystemExit(f"engine rc={rc} on {work}")
    return dt


# ===========================================================================
# child stages (RR_* env set by the parent)
# ===========================================================================

class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def stage_precompute(a, cfg_name):
    from scaling import configs
    cfg = configs.get(cfg_name)
    k = BOARDS["precompute"][str(cfg.grid)]
    ids = cfg.ids("bench")[:k]
    import pickle
    import shutil

    # Python: GridEnv.from_env cold (fresh env dir copy, no cache)
    tmp = BENCH_DIR / f"tmp_envs_{cfg_name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for gid in ids:
        shutil.copy(cfg.env_dir_abs / f"env_{gid}.pkl", tmp / f"env_{gid}.pkl")
    from GridEnv import GridEnv
    t0 = time.time()
    for gid in ids:
        GridEnv.from_env(gid, env_dir=tmp)
    t_from_env = (time.time() - t0) / k

    # Python: the all-pairs table block alone (from_env recipe: dependent
    # edges re-weighted to 2, then all_pairs + independent_paths)
    from nn.gen_grids import all_pairs, independent_paths
    boards = []
    for gid in ids:
        with open(cfg.env_dir_abs / f"env_{gid}.pkl", "rb") as f:
            boards.append((gid, pickle.load(f)))
    t0 = time.time()
    for gid, d in boards:
        G = d["grid_graph"]
        for u, v, dd in G.edges(data=True):
            if "dependent" in dd:
                dd["weight"] = 2
        all_pairs(G, "weight")
        independent_paths(G)
    t_tables = (time.time() - t0) / k

    # Rust: board compile (graph + walls + BOTH tables), single thread
    work = BENCH_DIR / f"{cfg_name}_boards.work.jsonl"
    with open(work, "w") as f:
        for gid, d in boards:
            f.write(json.dumps({
                "task": "board", "id": f"pb{gid}",
                "board": {"env_id": gid, "n": cfg.grid,
                          "grid_data": list(d["grid_data"])},
                "dependent_edge_weight": 2, "emit": "none"}) + "\n")
    # warm run to exclude binary/page-cache startup, then timed
    time_engine(work, BENCH_DIR / f"{cfg_name}_boards.out.jsonl", 1)
    t_rust = time_engine(work, BENCH_DIR / f"{cfg_name}_boards.out.jsonl", 1) / k

    shutil.rmtree(tmp)
    emit({"stage": "precompute", "config": cfg_name, "n": cfg.grid,
          "boards": k,
          "py_from_env_cold_s_per_board": round(t_from_env, 3),
          "py_tables_block_s_per_board": round(t_tables, 3),
          "rust_compile_s_per_board": round(t_rust, 4),
          "ratio_from_env_vs_rust": round(t_from_env / t_rust, 1),
          "ratio_tables_vs_rust": round(t_tables / t_rust, 1)})


def _load_work(tag, cfg_name):
    from scaling import configs
    cfg = configs.get(cfg_name)
    wdir = SV_DIR / "scaling" / "data" / cfg.name / "rust_work"
    work = [json.loads(l) for l in open(wdir / f"{tag}.work.jsonl")]
    results = {r["id"]: r for r in
               (json.loads(l) for l in open(wdir / f"{tag}.results.jsonl"))}
    manifest = json.load(open(wdir / f"{tag}.manifest.json"))
    return cfg, work, results, manifest


def _trim(work, manifest, budget_key):
    """Keep only the attempts the sequential Python loop would consume."""
    consumed = {}
    for e in manifest["boards"]:
        if "attempts_consumed" in e:
            consumed[e["env_id"]] = e["attempts_consumed"]
        else:
            k = e["kept_idx"]
            per = manifest[budget_key]
            budget = per * 5
            consumed[e["env_id"]] = (k[-1] + 1) if len(k) == per else budget
    out = []
    for it in work:
        gid, att = it["id"][1:].split(":a")
        if int(att) < consumed[int(gid)]:
            out.append(it)
    return out


def stage_backward(a, cfg_name):
    import common
    from GridEnv import Robot_at, State
    from nn.generate import rollout
    from skeleton.astar import AStar

    cfg, work, results, manifest = _load_work(f"bench_bwd_{cfg_name}", cfg_name)
    items = _trim(work, manifest, "per_graph")
    trimmed = BENCH_DIR / f"{cfg_name}_bwd_trimmed.work.jsonl"
    with open(trimmed, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    # Rust timings on the trimmed set
    rust = {}
    for th in ([1, a.threads] + ([32] if a.with_32 else [])):
        rust[th] = time_engine(trimmed, BENCH_DIR / f"{cfg_name}_bwd.out.jsonl", th)

    # Python single core, lazy env (env build reported separately)
    solver = AStar(max_iters=4000, max_frontier=40_000)
    signal.signal(signal.SIGALRM, _alarm)
    by_board = {}
    for it in items:
        by_board.setdefault(it["board"]["env_id"], []).append(it)
    t_env = t_label = 0.0
    pairs = []          # (t_py per instance, rust iters) for calibration
    n_ok = n_to = agree = 0
    for gid, its in by_board.items():
        gd = its[0]["board"]["grid_data"]
        t0 = time.time()
        env = common.build_env(gd, cfg.grid, mode="lazy", weight=2,
                               table_workers=16)
        t_env += time.time() - t0
        for it in its:
            st = State(
                target=tuple(it["target"]),
                target_robot=Robot_at(position=tuple(it["target_robot"][0]),
                                      color=it["target_robot"][1]),
                helpers=[Robot_at(position=tuple(p), color=c)
                         for p, c in it["helpers"]])
            t0 = time.time()
            signal.alarm(120)   # production backward_label guard
            try:
                recs = rollout(env, st, solver, gid, it["max_candidates"])
            except _Timeout:
                recs = []
                n_to += 1
            except Exception:
                recs = []
            finally:
                signal.alarm(0)
            dt = time.time() - t0
            t_label += dt
            r = results[it["id"]]
            pairs.append({"id": it["id"], "t_py": round(dt, 4),
                          "iters": r.get("iters"),
                          "py_kept": bool(recs), "rust_status": r["status"]})
            n_ok += bool(recs)
            agree += bool(recs) == (r["status"] == "ok")
    with open(BENCH_DIR / f"{cfg_name}_bwd_calibration.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    emit({"stage": "backward", "config": cfg_name, "boards": len(by_board),
          "attempts": len(items), "py_kept": n_ok, "py_timeouts": n_to,
          "outcome_agreement": f"{agree}/{len(items)}",
          "py_label_s": round(t_label, 2), "py_env_s": round(t_env, 2),
          **{f"rust_t{th}_s": round(t, 3) for th, t in rust.items()},
          "per_core_ratio": round(t_label / rust[1], 1)})


def stage_forward(a, cfg_name):
    from simulate import wall_sets
    from move_planner import oracle
    from move_planner.state import apply_move, is_goal, legal_moves

    cfg, work, results, manifest = _load_work(f"bench_fwd_{cfg_name}", cfg_name)
    items = _trim(work, manifest, "per_board")
    trimmed = BENCH_DIR / f"{cfg_name}_fwd_trimmed.work.jsonl"
    with open(trimmed, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    rust = {}
    for th in ([1, a.threads] + ([32] if a.with_32 else [])):
        rust[th] = time_engine(trimmed, BENCH_DIR / f"{cfg_name}_fwd.out.jsonl", th)

    # Python single core: label_board's per-attempt work on the same items
    n = cfg.grid
    by_board = {}
    for it in items:
        by_board.setdefault(it["board"]["env_id"], []).append(it)
    t_label = 0.0
    n_solved = agree = n_rec = 0
    for gid, its in by_board.items():
        gd = its[0]["board"]["grid_data"]
        wr, wd = wall_sets(gd, n)
        for it in its:
            positions = tuple(tuple(p) for p in it["robots"])
            target = tuple(it["target"])
            tidx = it["target_idx"]
            t0 = time.time()
            solved = False
            recs_out = 0
            hdist = oracle.relaxed_target_dist(target, wr, wd, n)
            if hdist.get(positions[tidx], oracle.INF) < oracle.INF:
                res = oracle.label_trajectory(
                    positions, tidx, target, wr, wd, n, hdist,
                    max_expansions=it["max_expansions"], full_policy=True,
                    full_policy_max_ctg=it["full_policy_max_ctg"])
                if res is not None:
                    solved = True
                    _d, recs = res
                    for r in recs:
                        recs_out += 1
                        if not it["score_candidates"]:
                            continue
                        for slot, di in r["legal_moves"]:
                            child = apply_move(r["positions"], slot, di, wr, wd, n)
                            if child is None:
                                continue
                            if is_goal(child, tidx, target):
                                c_ctg = 0
                            else:
                                c_ctg = oracle.solve(
                                    child, tidx, target, wr, wd, n, hdist,
                                    max_expansions=it["max_expansions"])
                                if c_ctg is None:
                                    continue
                            legal_moves(child, wr, wd, n)
                            recs_out += 1
            t_label += time.time() - t0
            n_solved += solved
            n_rec += recs_out
            agree += solved == (results[it["id"]]["status"] == "solved")
    emit({"stage": "forward", "config": cfg_name, "boards": len(by_board),
          "attempts": len(items), "py_solved": n_solved, "py_records": n_rec,
          "outcome_agreement": f"{agree}/{len(items)}",
          "py_label_s": round(t_label, 2),
          **{f"rust_t{th}_s": round(t, 3) for th, t in rust.items()},
          "per_core_ratio": round(t_label / rust[1], 1)})


# ===========================================================================
# parent
# ===========================================================================

def run_bridge(cfg_name, system, boards, per, seed, threads, tag, extra=()):
    out = BENCH_DIR / f"{tag}.out.jsonl"
    if out.exists():
        out.unlink()
    cmd = [PY, "-m", "scaling.rust_bridge", "--config", cfg_name,
           "--system", system, "--per-graph", str(per),
           "--limit", str(boards), "--seed", str(seed),
           "--threads", str(threads), "--out", str(out),
           "--work-tag", tag, *extra]
    log = BENCH_DIR / f"{tag}.bridge.log"
    with open(log, "w") as lf:
        rc = subprocess.run(cmd, cwd=str(SV_DIR), stdout=lf, stderr=lf,
                            env={**os.environ, "PYTHONPATH": str(SV_DIR)}
                            ).returncode
    if rc != 0:
        raise SystemExit(f"bridge failed rc={rc}; see {log}")


def main():
    a = parse_args()
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    if a.stage:                       # child
        {"precompute": stage_precompute,
         "backward": stage_backward,
         "forward": stage_forward}[a.stage](a, a.config)
        return

    if not ENGINE.exists():
        raise SystemExit(f"build the engine first: cargo build --release "
                         f"(missing {ENGINE})")
    from scaling import configs
    print(f"[bench] load check: {os.popen('uptime').read().strip()}")
    for cfg_name in a.configs.split(","):
        cfg = configs.get(cfg_name)
        for stage in a.stages.split(","):
            if stage == "backward":
                run_bridge(cfg_name, "backward", BOARDS["backward"],
                           PER_GRAPH_B[cfg_name], a.seed, a.threads,
                           f"bench_bwd_{cfg_name}")
            elif stage == "forward":
                run_bridge(cfg_name, "forward", BOARDS["forward"],
                           PER_BOARD_F[cfg_name], a.seed, a.threads,
                           f"bench_fwd_{cfg_name}", ("--score-candidates",))
            env = {**os.environ, **configs.env(cfg),
                   "PYTHONPATH": str(SV_DIR)}
            log = BENCH_DIR / f"{cfg_name}_{stage}.log"
            print(f"[bench] {cfg_name}/{stage} -> {log}", flush=True)
            with open(log, "w") as lf:
                child = [PY, str(Path(__file__).resolve()),
                         "--stage", stage, "--config", cfg_name,
                         "--threads", str(a.threads), "--seed", str(a.seed)]
                if a.with_32:
                    child.append("--with-32")
                p = subprocess.Popen(child, cwd=str(SV_DIR), env=env,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True)
                for line in p.stdout:
                    lf.write(line)
                    if line.startswith("RESULT "):
                        print("  " + line.strip()[7:], flush=True)
                if p.wait() != 0:
                    raise SystemExit(f"{cfg_name}/{stage} failed; see {log}")
    print(f"[bench] done; rows in {BENCH_DIR / 'results.jsonl'}")


if __name__ == "__main__":
    main()
