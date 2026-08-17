"""Candidate-generator (plan-language) ceiling on the MOVES metric.

PROBLEM.md section 6.1: "the candidate generator bounds the reachable policy: if
optimal play requires a subgoal the generator never proposes, no amount of
search finds it. Measure the generator's ceiling early." The supervised track's
ceiling probe (`analysis/artifacts/ceiling_probe.py`) answered the SOLVE-RATE
question (does any playable subgoal plan exist?) on the failing instances
only. This module answers the question that matters for the self-play metric,
on the WHOLE pinned bench: how many primitive moves does the best plan the
language can express cost, versus the exact optimum d*?

Per instance, an exhaustive best-first search over partial plans in abstract
plan-cost order (the solver's own admissible ordering; NO network anywhere)
strictly realizes every complete plan it pops (`eval.realize.strict_moves`,
the arena's own certified count) and records

  first_realizable_moves  strict moves of the FIRST realizable plan popped
                          (the supervised probe's definition),
  best_realizable_moves   min strict moves over every realizable plan popped
                          before the popped abstract cost reaches that
                          minimum -- the language optimum under the (usual)
                          strict >= abstract relation; a plan whose strict
                          count undercuts its abstract cost (an incidental
                          robot serving as a stopper) can in principle sit
                          beyond the bound, so this is a tight upper bound on
                          the true language optimum, not a proof,
  first_abstract / best_abstract   the abstract plan cost of those plans,

plus category (REALIZABLE_EXISTS / NO_REALIZABLE_PLAN / NO_COMPLETE_PLAN /
INCONCLUSIVE at the caps), work counters and d*. Base vocabulary by default;
`--vocab b2` reproduces the extended-language probe (propose_b1 +
by-reference + park repairs, cap 2), the setting the g16r4 baseline pair is
benched under.

    PYTHONPATH=supervised_valuenet:self_play_robots python -m spr.ceiling \
        --config g24r4 --instances scaling/data/g24r4/bench.solved.jsonl \
        --vocab base --workers 16 --time-cap 90 --out results/ceiling/g24r4_base.json

Writes atomically; last line `SPR CEILING DONE <out>`.
"""
from __future__ import annotations

import argparse
import heapq
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

from spr import RESULTS, SV

# set per process by _init_worker / main (one config per process)
_G = {}


def _plan_key(p):
    nodes = tuple(sorted(
        (d.get("ntype"), tuple(d["pos"]) if d.get("pos") is not None else None,
         getattr(d.get("robot"), "color", None)) for _, d in p.g.nodes(data=True)))
    edges = tuple(sorted(
        (p.g.nodes[u].get("ntype"), p.g.nodes[v].get("ntype"), e.get("status"),
         e.get("cost")) for u, v, e in p.g.edges(data=True)))
    return (nodes, edges)


def probe(env, state, solver, vocab, max_iters, max_frontier, time_cap,
          park_cap=2, max_open=None, slack=0.0):
    from skeleton.astar import _initial_plan, park_repairs
    from eval.realize import strict_moves
    from simulate import wall_sets, _board_size

    b1 = vocab in ("b1", "b2")
    b2 = vocab == "b2"
    size = _board_size(env.grid_data, None)
    wr, wd = wall_sets(env.grid_data, size)
    start = _initial_plan(env, state)
    frontier = [(start.cost(), 0, start)]
    tie = 1
    max_open = max_open or 2 * (len(state.helpers) + 2)
    seen = set()
    t0 = time.time()
    it = 0
    abstract_found = False
    complete_tested = 0
    first_real = first_abs = best_real = best_abs = None
    capped = False
    bound_hit = False
    while frontier:
        it += 1
        if it > max_iters or len(frontier) > max_frontier or \
                (time.time() - t0) > time_cap:
            capped = True
            break
        cost, _, cur = heapq.heappop(frontier)
        if best_real is not None and cost >= best_real + slack:
            bound_hit = True          # nothing cheaper (abstract, + slack) remains
            break
        k = _plan_key(cur)
        if k in seen:
            continue
        seen.add(k)
        if cur.is_complete():
            abstract_found = True
            complete_tested += 1
            fi = {} if b1 else None
            m = strict_moves(env, state, cur, log=None, fail_info=fi)
            if m is not None:
                if first_real is None:
                    first_real, first_abs = m, float(cost)
                if best_real is None or m < best_real:
                    best_real, best_abs = m, float(cost)
            elif b1 and fi:
                for child in park_repairs(env, state, cur, fi, wr, wd, size,
                                          max_parks=park_cap, pairwise=b2,
                                          multi_slide=b2):
                    heapq.heappush(frontier, (child.cost(), tie, child))
                    tie += 1
            continue
        if len(cur.open_edges()) > max_open:
            continue
        for child in solver._expand(env, state, cur):
            heapq.heappush(frontier, (child.cost(), tie, child))
            tie += 1
    exhausted = not capped                 # frontier emptied or bound met
    cat = ("REALIZABLE_EXISTS" if best_real is not None else
           "INCONCLUSIVE" if capped else
           "NO_REALIZABLE_PLAN" if abstract_found else "NO_COMPLETE_PLAN")
    return dict(category=cat, abstract_found=abstract_found,
                complete_tested=complete_tested,
                first_realizable_moves=first_real, first_abstract=first_abs,
                best_realizable_moves=best_real, best_abstract=best_abs,
                best_is_bounded=(best_real is not None and bound_hit),
                exhausted=exhausted, capped=capped, iters=it,
                seconds=round(time.time() - t0, 2))


def _init_worker(config, vocab, caps):
    """Runs in each worker AFTER the fork: RR_* are already in os.environ
    (set by main before any repo import), so repo imports freeze the right
    grid here."""
    sys.path.insert(0, str(SV))
    from GridEnv import GridEnv
    from skeleton.astar import AStar
    from skeleton import heuristics
    _G["GridEnv"] = GridEnv
    _G["solver"] = AStar(
        propose=heuristics.propose_b1 if vocab in ("b1", "b2") else heuristics.propose,
        max_iters=caps["max_iters"], max_frontier=caps["max_frontier"],
        beam=None, by_reference=(vocab == "b2"))
    _G["vocab"] = vocab
    _G["caps"] = caps
    _G["envs"] = {}


def _work(item):
    idx, inst = item
    from GridEnv import State, Robot_at
    from move_planner.state import COLOR_ORDER
    env = _G["envs"].get(inst["env_id"])
    if env is None:
        env, _ = _G["GridEnv"].from_env(inst["env_id"])
        _G["envs"].clear()
        _G["envs"][inst["env_id"]] = env
    positions = [tuple(p) for p in inst["positions"]]
    tidx = inst["target_idx"]
    st = State(target=tuple(inst["target"]),
               target_robot=Robot_at(position=positions[tidx], color=COLOR_ORDER[tidx]),
               helpers=[Robot_at(position=positions[j], color=COLOR_ORDER[j])
                        for j in range(len(positions)) if j != tidx])
    caps = _G["caps"]
    try:
        res = probe(env, st, _G["solver"], _G["vocab"], caps["max_iters"],
                    caps["max_frontier"], caps["time_cap"], caps["park_cap"],
                    slack=caps.get("slack", 0.0))
    except Exception as e:  # never lose the shard to one bad instance
        res = dict(category="ERROR", error=repr(e))
    res.update(idx=idx, env_id=inst["env_id"], d_star=inst.get("d_star"))
    return res


def summarize(rows):
    n = len(rows)
    real = [r for r in rows if r.get("best_realizable_moves") is not None]
    graded = [r for r in real if r.get("d_star") not in (None, 0)]
    cats = Counter(r["category"] for r in rows)
    out = {
        "n": n, "categories": dict(cats),
        "solve_ceiling": len(real) / n if n else None,
        "n_realizable": len(real), "n_graded_realizable": len(graded),
        "capped": sum(1 for r in rows if r.get("capped")),
        "best_bounded": sum(1 for r in real if r.get("best_is_bounded")),
    }
    if graded:
        gap_best = [r["best_realizable_moves"] - r["d_star"] for r in graded]
        gap_first = [r["first_realizable_moves"] - r["d_star"] for r in graded]
        out.update({
            "mean_d_star": sum(r["d_star"] for r in graded) / len(graded),
            "mean_best_moves": sum(r["best_realizable_moves"] for r in graded) / len(graded),
            "mean_first_moves": sum(r["first_realizable_moves"] for r in graded) / len(graded),
            "mean_gap_best": sum(gap_best) / len(graded),
            "mean_gap_first": sum(gap_first) / len(graded),
            "pct_best_optimal": 100.0 * sum(1 for g in gap_best if g <= 0) / len(graded),
            "pct_first_optimal": 100.0 * sum(1 for g in gap_first if g <= 0) / len(graded),
            "gap_best_hist": dict(sorted(Counter(gap_best).items())),
            "n_best_below_dstar": sum(1 for g in gap_best if g < 0),
            "n_abstract_below_dstar": sum(
                1 for r in graded if r["best_abstract"] is not None
                and r["best_abstract"] < r["d_star"]),
        })
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--instances", required=True, help="SV-relative or absolute")
    p.add_argument("--vocab", choices=["base", "b1", "b2"], default="base")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--time-cap", type=float, default=60.0)
    p.add_argument("--max-iters", type=int, default=200_000)
    p.add_argument("--max-frontier", type=int, default=400_000)
    p.add_argument("--park-cap", type=int, default=2)
    p.add_argument("--slack", type=float, default=0.0,
                   help="keep popping until abstract cost >= best strict + slack "
                        "(strict can undercut abstract when an incidental robot "
                        "serves as a stopper; slack tightens the ceiling)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    # one config per process: RR_* before any repo import (children inherit)
    sys.path.insert(0, str(SV))
    from scaling.configs import get, env as cfg_env
    cfg = get(a.config)
    for k in ("RR_GRID", "RR_ROBOTS", "RR_WALLS", "RR_ENV_DIR"):
        os.environ.pop(k, None)
    if not cfg.legacy:
        os.environ.update(cfg_env(cfg))

    inst_path = Path(a.instances)
    if not inst_path.is_absolute():
        inst_path = SV / inst_path
    insts = [json.loads(l) for l in inst_path.read_text().splitlines() if l.strip()]
    if a.limit:
        insts = insts[:a.limit]
    items = list(enumerate(insts))
    items.sort(key=lambda t: t[1]["env_id"])       # env locality per worker
    caps = dict(max_iters=a.max_iters, max_frontier=a.max_frontier,
                time_cap=a.time_cap, park_cap=a.park_cap, slack=a.slack)
    print(f"[ceiling] {a.config} vocab={a.vocab} n={len(items)} workers={a.workers} "
          f"caps={caps} instances={inst_path}", flush=True)
    t0 = time.time()
    rows = []
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers, initializer=_init_worker,
                  initargs=(a.config, a.vocab, caps)) as pool:
        for i, res in enumerate(pool.imap_unordered(_work, items, chunksize=1)):
            rows.append(res)
            if (i + 1) % 25 == 0 or i + 1 == len(items):
                done = sum(1 for r in rows if r.get("best_realizable_moves") is not None)
                print(f"[ceiling] {i + 1}/{len(items)} realizable={done} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: r["idx"])
    summ = summarize(rows)
    out = Path(a.out)
    if not out.is_absolute():
        out = (RESULTS.parent / out) if str(out).startswith("results/") else (RESULTS / out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": a.config, "vocab": a.vocab, "instances": str(inst_path),
               "caps": caps, "workers": a.workers,
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "wall_seconds": round(time.time() - t0, 1),
               "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "summary": summ, "rows": rows}
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n")
    os.replace(tmp, out)
    print("[ceiling] summary:", json.dumps(summ, indent=1), flush=True)
    print(f"SPR CEILING DONE {out}", flush=True)


if __name__ == "__main__":
    main()
