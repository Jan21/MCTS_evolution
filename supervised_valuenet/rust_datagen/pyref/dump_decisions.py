"""Dump gate-3 replay corpora (DESIGN.md section 3) from the Python labelers.

Backward: runs the nn.generate.rollout loop (copied verbatim below and
instrumented -- nn/generate.py itself is never modified; `_initial_plan`,
`_segment`, `_apply`, `solve_plan`, `heuristics.propose` are the real imported
functions) and, at every DECISION (non-forced step), writes one
`replay_backward_decision` line: full plan serialisation, state, open edge,
the candidate list exactly as Python labels it (after the stable-sort
max_candidates truncation) and `python_labels` per candidate
({"ctg": int|null, "rejected": bool}).

Forward: runs `move_planner.oracle.label_trajectory` with the
move_planner.generate sampling recipe and writes one `replay_forward_state`
line per record, carrying the Python record under `python_labels` for the
differ.

Config selection: RR_* module constants are read at import, one config per
process -- so this script RE-EXECS ITSELF in a subprocess with the config's
env (scaling.configs.env) before importing any repo module. `--n/--robots
--fresh-boards` covers sizes with no config (64x64): boards are generated via
nn.gen_grids (RR_GRID set) and cached under pyref/cache/.

Solver caps are explicit CLI args recorded in every dump line: backward
max_iters=4000 / max_frontier=40000 / max_candidates=14 (the
scaling.backward_label settings), forward max_expansions=40000 /
full_policy_max_ctg=6 (move_planner.generate defaults). NOTE for the Rust
work-item parser: `max_candidates` (backward) and `full_policy_max_ctg` +
`python_labels` (forward) are documented pyref-dump fields; accept-and-echo.

Instance sampling is per-board seeded (random.Random(seed*100003 + env_id);
forward additionally ^ 0x9E3779B9 exactly like move_planner.generate), so the
dump is deterministic under any --workers.

    python rust_datagen/pyref/dump_decisions.py --config g16r4 --task backward \
        --instances 40 --seed 7 --out /tmp/back16.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

PYREF_DIR = Path(__file__).resolve().parent
SV_DIR = PYREF_DIR.parent.parent
for _p in (str(SV_DIR), str(PYREF_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CHILD_FLAG = "RR_PYREF_CHILD"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=None,
                   help="scaling config name (g16r4,g16r6,g16r8,g24r4,g24r8,g32r4)")
    p.add_argument("--n", type=int, default=None,
                   help="grid size for config-less mode (e.g. 64)")
    p.add_argument("--robots", type=int, default=4)
    p.add_argument("--walls", type=int, default=None,
                   help="interior walls (default: stock density scaled by area)")
    p.add_argument("--fresh-boards", type=int, default=3,
                   help="config-less mode: boards to generate via nn.gen_grids")
    p.add_argument("--board-seed", type=int, default=0)
    p.add_argument("--task", choices=["backward", "forward"], required=True)
    p.add_argument("--instances", type=int, default=20,
                   help="total kept instances across boards")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True)
    p.add_argument("--boards", default=None,
                   help="explicit board id spec (default: first --n-boards of the bench split)")
    p.add_argument("--n-boards", type=int, default=5)
    p.add_argument("--workers", type=int, default=8, help="board workers (<=16)")
    p.add_argument("--env-mode", choices=["lazy", "eager"], default="lazy")
    p.add_argument("--env-source", choices=["rebuild", "pkl"], default="rebuild",
                   help="backward env: rebuild from grid_data (default) or "
                        "GridEnv.from_env pkl cache (A/B experiment only)")
    # backward caps (scaling.backward_label settings)
    p.add_argument("--max-candidates", type=int, default=14)
    p.add_argument("--max-iters", type=int, default=4000)
    p.add_argument("--max-frontier", type=int, default=40_000)
    p.add_argument("--vocab", default="base", choices=["base", "b1", "b2"],
                   help="plan-language vocabulary (backward task): base, "
                        "b1 (+transient supports), b2 (+supports-by-reference)")
    p.add_argument("--dependent-edge-weight", type=int, default=2)
    p.add_argument("--instance-timeout", type=int, default=300,
                   help="per-instance rollout wall cap (s); timed-out instances dropped")
    # forward caps (move_planner.generate defaults)
    p.add_argument("--max-expansions", type=int, default=40_000)
    p.add_argument("--full-policy-max-ctg", type=int, default=6)
    return p.parse_args(argv)


def _default_walls(n):
    return round(48 * (n / 16) ** 2)


def _rr_env(a):
    """The RR_* variables this run needs (must be set before repo imports)."""
    if a.config:
        from scaling import configs                    # env-free import
        cfg = configs.get(a.config)
        return configs.env(cfg)
    if a.n is None:
        raise SystemExit("need --config or --n")
    walls = a.walls if a.walls is not None else _default_walls(a.n)
    bdir = PYREF_DIR / "cache" / f"boards_n{a.n}_w{walls}_s{a.board_seed}"
    return {"RR_GRID": str(a.n), "RR_ROBOTS": str(a.robots),
            "RR_WALLS": str(walls), "RR_ENV_DIR": str(bdir)}


def main():
    a = parse_args()
    if os.environ.get(CHILD_FLAG) != "1":
        env = {**os.environ, **_rr_env(a), CHILD_FLAG: "1",
               "PYTHONPATH": str(SV_DIR)}
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                             *sys.argv[1:]], env=env, cwd=str(SV_DIR)).returncode
        sys.exit(rc)
    _child(a)


# ===========================================================================
# child (RR_* set): repo imports are safe from here on
# ===========================================================================

class _Timeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _Timeout()


def _resolve_boards(a):
    """[(env_id, grid_data)], the tag, n, robots."""
    import common
    if a.config:
        from scaling import configs
        cfg = configs.get(a.config)
        n, robots, tag = cfg.grid, cfg.robots, cfg.name
        if a.boards:
            ids = configs.parse_ids(a.boards)
        else:
            ids = cfg.ids("bench")[: a.n_boards]
        boards = [(gid, common.load_board_grid_data(cfg.env_dir_abs, gid))
                  for gid in ids]
        return boards, tag, n, robots, cfg.env_dir_abs
    n, robots = a.n, a.robots
    walls = a.walls if a.walls is not None else _default_walls(n)
    tag = f"n{n}r{robots}"
    bdir = Path(os.environ["RR_ENV_DIR"])
    bdir.mkdir(parents=True, exist_ok=True)
    boards = []
    for idx in range(a.fresh_boards):
        path = bdir / f"env_{idx}.pkl"
        if path.exists():
            import pickle
            with open(path, "rb") as f:
                grid_data = pickle.load(f)["grid_data"]
        else:
            from nn.gen_grids import gen_walls          # reads RR_GRID/RR_WALLS
            t0 = time.time()
            rng = random.Random(a.board_seed * 100000 + idx)
            grid_data = gen_walls(rng)
            import pickle
            with open(path, "wb") as f:
                pickle.dump({"graph_idx": idx, "grid_data": grid_data}, f)
            print(f"[boards] generated env_{idx} (n={n}) in "
                  f"{time.time() - t0:.1f}s -> {path}", file=sys.stderr)
        boards.append((idx, grid_data))
    return boards, tag, n, robots, bdir


def _prepare_tables(boards, n, weight, workers):
    """Warm the table cache serially, each board using the full worker pool
    (board workers then just load the cache -- no nested pools)."""
    import common
    for gid, grid_data in boards:
        path = common.tables_cache_path(grid_data, n, weight)
        if path.exists():
            continue
        t0 = time.time()
        G = common.build_graph_n(grid_data, n)
        common.reweight_dependent(G, weight)
        common.load_or_build_tables(G, grid_data, n, weight, workers=workers)
        print(f"[tables] env_{gid} n={n}: built in {time.time() - t0:.1f}s",
              file=sys.stderr)


# -- backward ---------------------------------------------------------------

def _rollout_dump(env, state, solver, env_id, n, grid_data, tag, inst_idx, a):
    """nn.generate.rollout, copied verbatim and instrumented: one
    replay_backward_decision line per non-forced decision. Mirrors the
    labeler's Lever B2 wiring: reference helpers join the segment before
    proposing when the solver was built with by_reference=True."""
    import common
    from skeleton.astar import _initial_plan, _segment, _apply, _reference_helpers
    from nn.generate import _fixed_g

    max_candidates = a.max_candidates
    lines = []
    plan = _initial_plan(env, state)
    depth = 0
    while not plan.is_complete():
        parent, child = plan.open_edges()[0]
        seg = _segment(plan, state, parent, child)

        # No decision when the segment pins to an exact path; just fix it.
        if env.compute_exact_shortest_path_length(
                seg.start, seg.end, seg.fix_support) is not None:
            plan = solver._expand(env, state, plan)[0]
            continue

        if solver.by_reference:
            seg.helpers = seg.helpers + _reference_helpers(plan, seg.mover.color)
        cands = solver.propose(env, seg.end, seg.mover, seg.helpers, seg.support)
        if max_candidates is not None:
            cands = sorted(cands, key=lambda c: solver.score(env, c))[:max_candidates]

        fixed_g = _fixed_g(plan)
        plan_ser = common.ser_plan(plan)            # BEFORE any apply
        labels, labeled = [], []
        for cand in cands:
            child_plan = _apply(env, plan, parent, child, seg, cand,
                                by_reference=solver.by_reference)
            if child_plan is None:
                labels.append({"ctg": None, "rejected": True})
                continue
            done = solver.solve_plan(env, state, child_plan)
            if done is None:
                labels.append({"ctg": None, "rejected": False})
                continue
            ctg = int(done.cost()) - int(fixed_g)
            labels.append({"ctg": ctg, "rejected": False})
            labeled.append((cand, child_plan, ctg))

        if cands:
            lines.append({
                "task": "replay_backward_decision",
                "id": f"{tag}:e{env_id}:i{inst_idx}:d{depth}",
                "board": common.board_obj(env_id, n, grid_data),
                "dependent_edge_weight": a.dependent_edge_weight,
                "max_iters": a.max_iters,
                "max_frontier": a.max_frontier,
                "max_candidates": a.max_candidates,
                "vocab": getattr(a, "vocab", "base"),
                "state": common.ser_state(state),
                "plan": plan_ser,
                "open_edge": [parent, child],
                "candidates": [common.ser_candidate(c) for c in cands],
                "python_labels": labels,
            })
        if not labeled:
            break
        best_ctg = min(c for _, _, c in labeled)
        # advance along the optimal trajectory (first labeled at best ctg)
        plan = next(cp for cand, cp, c in labeled if c == best_ctg)
        depth += 1
    return lines


def _backward_board(job):
    (gid, grid_data, n, robots, a_dict, tag) = job
    a = argparse.Namespace(**a_dict)
    import common
    from nn.generate import make_solver, random_instance

    t0 = time.time()
    if a.env_source == "pkl":
        from GridEnv import GridEnv
        env, _ = GridEnv.from_env(gid)
    else:
        env = common.build_env(grid_data, n, mode=a.env_mode,
                               weight=a.dependent_edge_weight, table_workers=1)
    t_env = time.time() - t0
    colors = common.PALETTE[:robots]
    rng = random.Random(a.seed * 100003 + gid)
    solver = make_solver(getattr(a, "vocab", "base"),
                         max_iters=a.max_iters, max_frontier=a.max_frontier)
    signal.signal(signal.SIGALRM, _raise_timeout)

    lines = []
    kept = attempts = timeouts = errors = 0
    t1 = time.time()
    while kept < a.per_board and attempts < a.per_board * 4:
        attempts += 1
        st = random_instance(env, colors, rng)
        signal.alarm(a.instance_timeout)
        try:
            inst = _rollout_dump(env, st, solver, gid, n, grid_data, tag,
                                 attempts - 1, a)
        except _Timeout:
            timeouts += 1
            inst = []
        except Exception:
            errors += 1
            inst = []
        finally:
            signal.alarm(0)
        if not inst:
            continue
        lines += inst
        kept += 1
    return (gid, kept, attempts, timeouts, errors, lines,
            {"env_s": t_env, "rollout_s": time.time() - t1})


# -- forward ----------------------------------------------------------------

def _forward_board(job):
    (gid, grid_data, n, robots, a_dict, tag) = job
    a = argparse.Namespace(**a_dict)
    import common
    from simulate import wall_sets
    from move_planner import oracle

    t0 = time.time()
    wr, wd = wall_sets(grid_data, n)
    # sampling recipe of move_planner.generate.label_board
    rng = random.Random((a.seed * 100003 + gid) ^ 0x9E3779B9)
    cells_all = [(x, y) for y in range(n) for x in range(n)]
    lines = []
    solved = attempts = 0
    max_attempts = a.per_board * 5
    while solved < a.per_board and attempts < max_attempts:
        attempts += 1
        cells = rng.sample(cells_all, robots + 1)
        positions = tuple(cells[:robots])
        target = cells[robots]
        tidx = rng.randrange(robots)
        hdist = oracle.relaxed_target_dist(target, wr, wd, n)
        if hdist.get(positions[tidx], oracle.INF) >= oracle.INF:
            continue  # unsolvable even in the relaxation
        res = oracle.label_trajectory(
            positions, tidx, target, wr, wd, n, hdist,
            max_expansions=a.max_expansions, full_policy=True,
            full_policy_max_ctg=a.full_policy_max_ctg)
        if res is None:
            continue
        _d, recs = res
        inst_idx = attempts - 1
        for r in recs:
            lines.append({
                "task": "replay_forward_state",
                "id": f"{tag}:e{gid}:i{inst_idx}:s{r['depth']}",
                "board": common.board_obj(gid, n, grid_data),
                "positions": [common.xy(p) for p in r["positions"]],
                "target_idx": int(tidx),
                "target": common.xy(target),
                "max_expansions": a.max_expansions,
                "full_policy_max_ctg": a.full_policy_max_ctg,
                "python_labels": {
                    "cost_to_go": int(r["cost_to_go"]),
                    "best_moves": [[int(s), int(d)] for s, d in r["best_moves"]],
                    "legal_moves": [[int(s), int(d)] for s, d in r["legal_moves"]],
                    "full": bool(r["full"]),
                    "depth": int(r["depth"]),
                },
            })
        solved += 1
    return (gid, solved, attempts, 0, 0, lines,
            {"env_s": 0.0, "rollout_s": time.time() - t0})


# -- driver -----------------------------------------------------------------

def _child(a):
    import math
    boards, tag, n, robots, _bdir = _resolve_boards(a)
    a.per_board = math.ceil(a.instances / len(boards))
    workers = max(1, min(a.workers, 16, len(boards)))
    if a.task == "backward" and a.env_source == "rebuild":
        _prepare_tables(boards, n, a.dependent_edge_weight, workers=16)

    a_dict = vars(a)
    jobs = [(gid, gd, n, robots, a_dict, tag) for gid, gd in boards]
    fn = _backward_board if a.task == "backward" else _forward_board

    t0 = time.time()
    if workers == 1:
        results = [fn(j) for j in jobs]
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=mp.get_context("fork")) as ex:
            results = list(ex.map(fn, jobs))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_lines = n_kept = n_att = n_to = n_err = 0
    with open(out, "w") as f:
        for gid, kept, attempts, timeouts, errors, lines, tm in results:
            for ln in lines:
                f.write(json.dumps(ln) + "\n")
            n_lines += len(lines)
            n_kept += kept
            n_att += attempts
            n_to += timeouts
            n_err += errors
            print(f"[{a.task}] env_{gid}: kept={kept} attempts={attempts} "
                  f"timeouts={timeouts} errors={errors} lines={len(lines)} "
                  f"(env {tm['env_s']:.1f}s, rollouts {tm['rollout_s']:.1f}s)",
                  file=sys.stderr)
    dt = time.time() - t0
    rate = n_lines / (dt / 60) if dt > 0 else 0.0
    print(f"[{a.task}] {tag}: {n_lines} lines from {n_kept} instances "
          f"({n_att} attempts, {n_to} timeouts, {n_err} errors) in {dt:.1f}s "
          f"[{rate:.1f} lines/min, workers={workers}] -> {out}",
          file=sys.stderr)
    print(json.dumps({"tag": tag, "task": a.task, "lines": n_lines,
                      "instances": n_kept, "attempts": n_att,
                      "timeouts": n_to, "errors": n_err,
                      "wall_s": round(dt, 1)}))


if __name__ == "__main__":
    main()
