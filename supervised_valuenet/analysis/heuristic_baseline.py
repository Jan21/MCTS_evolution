"""The no-network control: the hand-written scorer at the eval budget.

Owner question (2026-07-31): are the learned rows just the hand-written
solver's score? This measures it. The search loop is a faithful copy of
`eval.compare._nn_astar_backward` under the production protocol (anytime
realization-checking, B1 vocabulary + B2 generalized park repairs, the same
1,200-expansion budget, the same top-k shortlist width) with EXACTLY two
substitutions — the two places the neural networks act:

  shortlist   top-k candidates by the hand-written `heuristics.score`
              (ascending) instead of policy-net log-probability;
  frontier    plans ordered by `plan.cost()` — the labeler's admissible
              g + relaxed-open-segment estimate — instead of
              fixed_g + value-net cost-to-go.

Deliberately UNCHANGED: the `_hidx` candidate filter (the pool of
candidates is identical to the learned rows', so the ONLY difference is
ranking), the free exact-fix loop, park repairs, and the realization
check. Solved rows dump their move sequences so `eval.replay_validate`
can certify them independently.

    PYTHONPATH=. python3 -m analysis.heuristic_baseline \
        --config g16r6 --instances scaling/data/g16r6/bench.solved.jsonl \
        --expansions 1200 --k 5 --out <file.json> [--limit N]

--config base sets no RR_* env (16x16/4r defaults); scaled configs set the
grid env vars BEFORE the engine imports, mirroring track1_rows.slurm.
"""
import argparse
import hashlib
import heapq
import itertools
import json
import os
import time

CONFIGS = {
    "base": None,
    "g16r6": (16, 6, 48), "g16r8": (16, 8, 48), "g24r4": (24, 4, 108),
    "g24r8": (24, 8, 108), "g32r4": (32, 4, 192),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="base", choices=sorted(CONFIGS))
    p.add_argument("--instances", required=True)
    p.add_argument("--expansions", type=int, default=1200)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    if CONFIGS[a.config] is not None:
        g, r, w = CONFIGS[a.config]
        os.environ.update(RR_GRID=str(g), RR_ROBOTS=str(r), RR_WALLS=str(w),
                          RR_ENV_DIR=f"environments_{a.config}")

    # engine imports AFTER env vars, as in eval.compare's process model
    from GridEnv import GridEnv, State, Robot_at
    from skeleton.astar import (AStar, park_repairs, _initial_plan, _segment,
                                _apply)
    from skeleton import heuristics
    from simulate import wall_sets
    from move_planner.state import COLOR_ORDER
    from eval.realize import strict_moves
    from nn.generate import _context, _fixed_g
    from nn.collect_search import _record
    from eval.end2end import _hidx

    solver = AStar(propose=heuristics.propose_b1,
                   max_iters=4000, max_frontier=40_000, by_reference=True)

    def search(env, st, env_id, realize_check, park_hook):
        cnt = itertools.count()
        frontier = [(0.0, next(cnt), _initial_plan(env, st))]
        expansions = rejected = 0
        first_failed = None
        while frontier and expansions < a.expansions:
            _, _, plan = heapq.heappop(frontier)
            while not plan.is_complete():             # free forced exact fixes
                parent, child = plan.open_edges()[0]
                seg = _segment(plan, st, parent, child)
                if env.compute_exact_shortest_path_length(
                        seg.start, seg.end, seg.fix_support) is not None:
                    plan = solver._expand(env, st, plan)[0]
                else:
                    break
            if plan.is_complete():
                if realize_check(plan):
                    return plan, expansions, rejected
                rejected += 1
                for rp in park_hook(plan):
                    heapq.heappush(frontier,
                                   (float(rp.cost()), next(cnt), rp))
                if first_failed is None:
                    first_failed = plan
                continue
            expansions += 1
            parent, child = plan.open_edges()[0]
            seg = _segment(plan, st, parent, child)
            cands = solver.propose(env, seg.end, seg.mover, seg.helpers,
                                   seg.support)
            kept = []
            for cand in cands:
                cp = _apply(env, plan, parent, child, seg, cand)
                if cp is None or _hidx(st, _record(
                        env_id, st, seg, _context(plan), cand, 0, 0,
                        0)["cand_helper"][0]) is None:
                    continue
                kept.append((cand, cp))
            if not kept:
                continue
            # THE substitution: hand-written score, not policy log-prob
            kept.sort(key=lambda t: solver.score(env, t[0]))
            for cand, cp in kept[:a.k]:
                # THE second substitution: admissible plan cost, not value net
                heapq.heappush(frontier, (float(cp.cost()), next(cnt), cp))
        return first_failed, expansions, rejected

    insts = [json.loads(l) for l in open(a.instances) if l.strip()]
    if a.limit:
        insts = insts[:a.limit]
    sha = hashlib.sha256(open(a.instances, "rb").read()).hexdigest()

    rows = []
    cur_env_id, env = None, None
    t_all = time.perf_counter()
    for inst in insts:
        env_id = inst["env_id"]
        if env_id != cur_env_id:
            env, _ = GridEnv.from_env(env_id)
            cur_env_id = env_id
        positions = [tuple(p) for p in inst["positions"]]
        tidx = inst["target_idx"]
        st = State(target=tuple(inst["target"]),
                   target_robot=Robot_at(position=positions[tidx],
                                         color=COLOR_ORDER[tidx]),
                   helpers=[Robot_at(position=positions[j],
                                     color=COLOR_ORDER[j])
                            for j in range(len(positions)) if j != tidx])
        moves_cache, fail_cache = {}, {}

        def realize_check(p):
            fi, mv = {}, []
            m = strict_moves(env, st, p, log=None, fail_info=fi,
                             moves_out=mv)
            if m is not None:
                moves_cache[id(p)] = (m, mv)
            else:
                fail_cache[id(p)] = fi
            return m is not None

        _wr, _wd = wall_sets(env.grid_data)
        _size = int(round(len(env.grid_data) ** 0.5))

        def park_hook(p):
            fi = fail_cache.get(id(p))
            if not fi:
                return []
            try:
                return park_repairs(env, st, p, fi, _wr, _wd, _size,
                                    max_parks=2, pairwise=True,
                                    multi_slide=True)
            except Exception:
                return []

        t0 = time.perf_counter()
        plan, expansions, rejected = search(env, st, env_id,
                                            realize_check, park_hook)
        dt = time.perf_counter() - t0
        solved = plan is not None and id(plan) in moves_cache
        row = {"env_id": env_id, "d_star": inst.get("d_star"),
               "plan_found": plan is not None, "solved": bool(solved),
               "expansions": expansions, "plans_rejected": rejected,
               "seconds": dt,
               "plan_cost_abstract": (float(plan.cost())
                                      if plan is not None else None),
               "realized_strict": (moves_cache[id(plan)][0]
                                   if solved else None)}
        if solved:
            row["moves"] = moves_cache[id(plan)][1]
        rows.append(row)
        print(f"[{len(rows)}/{len(insts)}] env={env_id} "
              f"solved={solved} exp={expansions} rej={rejected} "
              f"{dt:.1f}s", flush=True)

    n = len(rows)
    solved_n = sum(r["solved"] for r in rows)
    payload = {
        "systems": {"backward subgoal planner (heuristic-scored, "
                    "anytime realization-checked) [B1 vocabulary + "
                    "generalized parks, NO neural networks]": {
            "kind": "backward", "rows": rows,
            "aggregate": {
                "n": n, "solved": solved_n,
                "solve_rate": solved_n / n if n else 0.0,
                "mean_expansions": sum(r["expansions"] for r in rows) / n,
                "mean_seconds": sum(r["seconds"] for r in rows) / n,
            }}},
        "protocol": {
            "script": "analysis/heuristic_baseline.py",
            "config": a.config, "instances_file": a.instances,
            "instances_sha256": sha, "n_instances": n,
            "expansions": a.expansions, "k": a.k,
            "dump_moves": True,
            "note": ("hand-written-scorer control: identical candidate "
                     "pool, budget, k, anytime check and park repairs as "
                     "the learned rows; only the two network scoring "
                     "points replaced (heuristics.score shortlist; "
                     "plan.cost() frontier)"),
            "wall_seconds": time.perf_counter() - t_all,
        }}
    with open(a.out + ".tmp", "w") as fh:
        json.dump(payload, fh)
    os.replace(a.out + ".tmp", a.out)
    print(f"HEURISTIC_BASELINE {a.config} solved {solved_n}/{n} "
          f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
