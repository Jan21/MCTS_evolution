"""Arena driver for the size-free nets (and, later, the self-play searches).

`eval/compare.py::run_backward` hard-codes the per-size net classes
(`PolicyTF`, `LoopedValueNet`), so it cannot load `spr.nets` checkpoints. This
driver keeps EVERYTHING else identical to the arena: it imports and calls
`eval.compare._nn_astar_backward` (the arena's search loop, unchanged),
`eval.realize.strict_moves` / `prefix_playable` (the arena's certification and
Lever-A prefix filter), and `eval.compare.aggregate`, and it emits the same
per-instance row schema and payload layout ({protocol, systems}) so
`eval/merge_compare_shards.py`, `eval/replay_validate.py` and the report
tooling consume its output unchanged. One config per process (RR_* set by the
caller / spr.arena) because the arena's featurizers read RR_GRID at import.

    RR_GRID=24 ... python -m spr.bench --instances chunk.jsonl \
        --policy P.ckpt --value V.ckpt --search arena_astar --prefix-check \
        --expansions 1200 --k 5 --dump-moves --out chunk.json

`--search`:
  arena_astar   eval.compare._nn_astar_backward with the size-free nets behind
                the arena's own `_policy_logp` / `_value_cost` (adapters in
                spr.nets) -- the M1 gate path;
  spr_astar     spr.search.astar (own loop; f-mode / anytime-best options);
  mcts          spr.search.mcts (PUCT over subgoal decisions).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--instances", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--value", required=True)
    p.add_argument("--search", default="arena_astar",
                   choices=["arena_astar", "spr_astar", "mcts", "greedy"])
    p.add_argument("--expansions", type=int, default=1200)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--prefix-check", action="store_true")
    p.add_argument("--anytime", action="store_true",
                   help="arena anytime mode: keep searching past complete "
                        "plans that fail strict realization")
    p.add_argument("--best-at-budget", action="store_true",
                   help="spr searches: return the best certified plan found "
                        "within the budget instead of the first")
    p.add_argument("--f-mode", choices=["child", "parent"], default="child",
                   help="spr_astar: f = fixed_g(child)+ctg (arena) or "
                        "fixed_g(parent)+ctg (label-consistent)")
    p.add_argument("--mcts-c", type=float, default=1.5)
    p.add_argument("--mcts-backup", choices=["min", "mean"], default="min")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dump-moves", action="store_true")
    p.add_argument("--boards", choices=["pkl", "lean"], default="pkl")
    p.add_argument("--out", required=True)
    p.add_argument("--md", default="/dev/null")
    a = p.parse_args(argv)

    import torch
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    from GridEnv import GridEnv, State, Robot_at
    from skeleton.astar import AStar
    from skeleton import heuristics
    from move_planner.state import COLOR_ORDER
    from eval.compare import _nn_astar_backward, aggregate, load_instances, write_markdown
    from eval.realize import abstract_moves, strict_moves, prefix_playable, prefix_key
    from nn_labeler import leanboard
    from spr.nets import load_policy, load_value, ValueAdapter

    dev = a.device
    policy = load_policy(a.policy, dev)
    value_net = load_value(a.value, dev)
    value = ValueAdapter(value_net)
    solver = AStar(propose=heuristics.propose, max_iters=4000, max_frontier=40_000)
    load_env = leanboard.from_env if a.boards == "lean" else GridEnv.from_env

    instances, sha, meta = load_instances(a.instances)
    placeholder = bool(instances) and all(i.get("d_star") in (0, None) for i in instances)

    search_impl = None
    if a.search != "arena_astar":
        from spr import search as spr_search
        search_impl = spr_search

    rows = []
    cur_env_id, env = None, None
    for i, inst in enumerate(instances):
        env_id = inst["env_id"]
        if env_id != cur_env_id:
            env, _ = load_env(env_id)
            cur_env_id = env_id
        positions = [tuple(p) for p in inst["positions"]]
        tidx = inst["target_idx"]
        st = State(target=tuple(inst["target"]),
                   target_robot=Robot_at(position=positions[tidx], color=COLOR_ORDER[tidx]),
                   helpers=[Robot_at(position=positions[j], color=COLOR_ORDER[j])
                            for j in range(len(positions)) if j != tidx])
        d_star = inst["d_star"]
        acct = {"nn_policy_calls": 0, "nn_value_calls": 0, "free_exact_fix_expands": 0,
                "rejected_plan_pops": 0, "physics_calls_prefix_check": 0,
                "physics_calls_strict_realize": 0, "physics_calls_park_repair": 0,
                "park_plans_pushed": 0}
        moves_cache, check_cache = {}, {}
        t0 = time.perf_counter()
        realize_check = None
        if a.anytime:
            def realize_check(pl, _env=env, _st=st):
                mv = [] if a.dump_moves else None
                acct["physics_calls_strict_realize"] += 1
                m = strict_moves(_env, _st, pl, log=None, moves_out=mv)
                check_cache[id(pl)] = m
                if a.dump_moves and m is not None:
                    moves_cache[id(pl)] = mv
                return m is not None
        prefix_filter = None
        if a.prefix_check:
            pfx = {}
            def prefix_filter(pl, _env=env, _st=st):
                key = prefix_key(pl)
                hit = pfx.get(key)
                if hit is None:
                    acct["physics_calls_prefix_check"] += 1
                    hit = pfx[key] = prefix_playable(_env, _st, pl)
                return hit
        extra = {}
        if a.search == "arena_astar":
            plan, expansions, rejected, pruned = _nn_astar_backward(
                env, st, solver, policy, value, env_id, dev, a.k, a.expansions,
                realize_check=realize_check, prefix_filter=prefix_filter,
                park_hook=None, acct=acct)
        else:
            res = search_impl.run(
                a.search, env, st, solver, policy, value_net, env_id, dev,
                k=a.k, max_expansions=a.expansions, prefix_filter=prefix_filter,
                best_at_budget=a.best_at_budget, f_mode=a.f_mode,
                c_puct=a.mcts_c, backup=a.mcts_backup, acct=acct,
                dump_moves=a.dump_moves)
            plan, expansions, rejected, pruned = (res.plan, res.expansions,
                                                 res.rejected, res.pruned)
            if res.moves is not None:
                moves_cache[id(plan)] = res.moves
                check_cache[id(plan)] = res.strict
            extra = res.extra
        acct["rejected_plan_pops"] = rejected
        dt = time.perf_counter() - t0
        row = {"env_id": env_id, "d_star": None if placeholder else d_star,
               "plan_found": plan is not None, "plan_cost_abstract": None,
               "realized_abstract": None, "realized_strict": None, "solved": False,
               "regret": None, "expansions": expansions, "seconds": dt,
               "plans_rejected": rejected, "children_pruned": pruned}
        if plan is not None:
            row["plan_cost_abstract"] = float(plan.cost())
            ab, _ok = abstract_moves(env, st, plan, log=None)
            if id(plan) in check_cache:
                stx = check_cache[id(plan)]
            else:
                mv = [] if a.dump_moves else None
                acct["physics_calls_strict_realize"] += 1
                stx = strict_moves(env, st, plan, log=None, moves_out=mv)
                if a.dump_moves and stx is not None:
                    moves_cache[id(plan)] = mv
            row["realized_abstract"] = ab
            row["realized_strict"] = stx
            row["solved"] = stx is not None
            row["regret"] = None if stx is None or placeholder else stx - d_star
        if a.dump_moves and row["solved"]:
            row["moves"] = moves_cache[id(plan)]
        row["accounting"] = acct
        if extra:
            row["search"] = extra
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  [spr.bench] {i + 1}/{len(instances)}", flush=True)

    name = f"spr {a.search} size-free planner"
    if a.anytime:
        name += " (anytime)"
    if a.prefix_check:
        name += " (prefix-check)"
    if a.best_at_budget:
        name += " (best-at-budget)"
    if a.search == "spr_astar":
        name += f" [f={a.f_mode}]"
    if a.search == "mcts":
        name += f" [c={a.mcts_c} backup={a.mcts_backup}]"
    ckpt_info = {c: time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(c)))
                 for c in (a.policy, a.value)}
    protocol = {
        "expansions": a.expansions, "k": a.k, "instances_file": a.instances,
        "instances_sha256": sha, "n_instances": len(instances), "instances_meta": meta,
        "checkpoints": ckpt_info, "device": dev, "d_star_placeholder": placeholder,
        "count_slides": False, "dump_moves": a.dump_moves, "byref": False,
        "byref_pool": None, "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results_file": a.out, "search": a.search,
        "search_options": {"prefix_check": a.prefix_check, "anytime": a.anytime,
                           "best_at_budget": a.best_at_budget, "f_mode": a.f_mode,
                           "mcts_c": a.mcts_c, "mcts_backup": a.mcts_backup},
        "command": " ".join(["python -m spr.bench"] + sys.argv[1:]),
        "expansion_definition": "one popped/expanded search node whose children are "
                                "generated (= 1 policy pass + 1 batched value pass "
                                "over <= k children)",
    }
    systems = {name: {"kind": "backward", "aggregate": aggregate(rows, "realized_strict"),
                      "rows": rows}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"protocol": protocol, "systems": systems}, indent=2) + "\n")
    write_markdown(a.md, protocol, systems, [name])
    ag = systems[name]["aggregate"]
    print(f"[spr.bench] {name}: solved {ag['solved']}/{ag['n']} regret={ag['mean_regret']} "
          f"opt%={ag['pct_optimal']} moves={ag['mean_moves']} exp={ag['mean_expansions']}",
          flush=True)


if __name__ == "__main__":
    main()
