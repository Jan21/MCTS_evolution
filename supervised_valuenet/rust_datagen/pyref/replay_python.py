"""Python reference REPLAYER for gate-3 dumps (DESIGN.md section 3).

Reads replay work items produced by dump_decisions.py, reconstructs everything
from the dump alone (board -> GridEnv from grid_data; PartialPlan -> fresh
nx.DiGraph, nodes/edges in dump order; Robot_at rebuilt from [pos, color]) and
recomputes the labels:

- replay_backward_decision: rerun `_apply` + `solve_plan` per candidate ->
  one result line {"id", "labels": [{"ctg": int|null, "rejected": bool}, ...]}
  (the shape the Rust engine emits for `datagen replay`).
- replay_forward_state: recompute value + FULL optimal-move set ->
  {"id", "cost_to_go": int|null, "optimal_moves": [[slot,dir],...],
   "legal_moves": [[slot,dir],...]}.

Nothing is read from env pkls or RR_* config: if the labels diff against the
dump's python_labels, the dump schema is incomplete -- that is exactly what
this validates. It doubles as the debugging reference for Rust divergences.

    python rust_datagen/pyref/replay_python.py --dump d.jsonl --out r.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PYREF_DIR = Path(__file__).resolve().parent
SV_DIR = PYREF_DIR.parent.parent
for _p in (str(SV_DIR), str(PYREF_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common  # noqa: E402


def replay_backward_line(line, env):
    from skeleton.astar import AStar, _segment, _apply
    from nn.generate import _fixed_g

    state = common.de_state(line["state"])
    plan = common.de_plan(line["plan"])
    parent, child = line["open_edge"]

    oe = plan.open_edges()
    if not oe or list(oe[0]) != [parent, child]:
        return {"id": line["id"],
                "error": f"open_edge mismatch: dump={line['open_edge']} "
                         f"reconstructed={oe[0] if oe else None}"}

    solver = AStar(max_iters=line["max_iters"],
                   max_frontier=line["max_frontier"])
    seg = _segment(plan, state, parent, child)
    fixed_g = _fixed_g(plan)

    labels = []
    for c in line["candidates"]:
        cand = common.de_candidate(c, seg)
        child_plan = _apply(env, plan, parent, child, seg, cand)
        if child_plan is None:
            labels.append({"ctg": None, "rejected": True})
            continue
        done = solver.solve_plan(env, state, child_plan)
        if done is None:
            labels.append({"ctg": None, "rejected": False})
        else:
            labels.append({"ctg": int(done.cost()) - int(fixed_g),
                           "rejected": False})
    return {"id": line["id"], "labels": labels}


def replay_forward_line(line, wr, wd, hdist_cache):
    from move_planner import oracle

    n = line["board"]["n"]
    positions = tuple(tuple(p) for p in line["positions"])
    target = tuple(line["target"])
    tidx = int(line["target_idx"])
    me = int(line["max_expansions"])

    hdist = hdist_cache.get(target)
    if hdist is None:
        hdist = oracle.relaxed_target_dist(target, wr, wd, n)
        hdist_cache[target] = hdist

    ctg = oracle.solve(positions, tidx, target, wr, wd, n, hdist, me)
    succ = oracle._successors(positions, wr, wd, n)
    legal = [[i, di] for i, di, _ in succ]
    optimal = []
    if ctg is not None:
        for i, di, child in succ:
            if child[tidx] == target:
                ok = (ctg == 1)
            else:
                ok = oracle.solve(child, tidx, target, wr, wd, n, hdist, me,
                                  cost_cap=ctg - 1) == ctg - 1
            if ok:
                optimal.append([i, di])
    return {"id": line["id"], "cost_to_go": ctg,
            "optimal_moves": optimal, "legal_moves": legal}


def _replay_group(job):
    """One board's lines, replayed in one worker. Returns [(idx, result)]."""
    board, entries, env_mode, weight = job
    n = board["n"]
    grid_data = board["grid_data"]
    out = []
    env = None
    wr = wd = None
    hdist_cache = {}
    for idx, raw in entries:
        line = json.loads(raw)
        if line["task"] == "replay_backward_decision":
            if env is None:
                env = common.build_env(grid_data, n, mode=env_mode,
                                       weight=weight, table_workers=1)
            out.append((idx, replay_backward_line(line, env)))
        elif line["task"] == "replay_forward_state":
            if wr is None:
                from simulate import wall_sets
                wr, wd = wall_sets(grid_data, n)
            out.append((idx, replay_forward_line(line, wr, wd, hdist_cache)))
        else:
            out.append((idx, {"id": line.get("id"),
                              "error": f"unknown task {line['task']!r}"}))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dump", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--env-mode", choices=["lazy", "eager"], default="lazy")
    p.add_argument("--workers", type=int, default=8, help="<=16")
    a = p.parse_args()

    t0 = time.time()
    groups = {}   # (sha, weight) -> [board, [(idx, raw)...], mode, weight]
    n_backward = n_forward = 0
    with open(a.dump) as f:
        for idx, raw in enumerate(f):
            raw = raw.strip()
            if not raw:
                continue
            line = json.loads(raw)
            board = line["board"]
            weight = int(line.get("dependent_edge_weight", 2))
            key = (common.grid_sha(board["grid_data"]), weight, line["task"])
            g = groups.get(key)
            if g is None:
                g = [board, [], a.env_mode, weight]
                groups[key] = g
            g[1].append((idx, raw))
            if line["task"] == "replay_backward_decision":
                n_backward += 1
            else:
                n_forward += 1

    # Warm table caches serially with the full pool (no nested pools).
    for (sha, weight, task), (board, entries, _m, _w) in groups.items():
        if task != "replay_backward_decision":
            continue
        path = common.tables_cache_path(board["grid_data"], board["n"], weight)
        if path.exists():
            continue
        t = time.time()
        G = common.build_graph_n(board["grid_data"], board["n"])
        common.reweight_dependent(G, weight)
        common.load_or_build_tables(G, board["grid_data"], board["n"], weight,
                                    workers=16)
        print(f"[tables] n={board['n']} env_{board['env_id']}: built in "
              f"{time.time() - t:.1f}s", file=sys.stderr)

    jobs = [tuple(g) for g in groups.values()]
    workers = max(1, min(a.workers, 16, len(jobs)))
    if workers == 1:
        parts = [_replay_group(j) for j in jobs]
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=mp.get_context("fork")) as ex:
            parts = list(ex.map(_replay_group, jobs))

    results = [r for part in parts for r in part]
    results.sort(key=lambda t: t[0])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_err = 0
    with open(out, "w") as f:
        for _idx, res in results:
            if "error" in res:
                n_err += 1
            f.write(json.dumps(res) + "\n")
    dt = time.time() - t0
    print(f"[replay] {len(results)} lines ({n_backward} backward, "
          f"{n_forward} forward, {n_err} errors) in {dt:.1f}s "
          f"[workers={workers}] -> {out}", file=sys.stderr)
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
