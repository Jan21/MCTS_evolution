"""Hybrid action space, first instantiation (wave 3): ROOT PRIMITIVE SLIDES at
inference -- a portfolio search that may spend part of the arena budget on
"move one robot first, then plan subgoals from there".

Why: FINDINGS 3 (main log) proved the moves headline is unreachable inside any
pure-subgoal language (ceiling +0.9/+1.2 mean regret vs the forward planner's
0.04-0.07), and the lab's three-way unseen table (variants/FINDINGS.md 5) put
the gap at 3/43 both-solved move wins against forward. The only route is the
action space. Root slides are the cheapest sound extension: a depth-1 primitive
move needs NO State->PartialPlan re-projection (the moved state is just a fresh
instance with the same target), certification stays exact (the sub-plan is
certified against the moved state; total strict = 1 + sub-strict; the dumped
move sequence [slide] + sub-moves replay-validates against the ORIGINAL state).

Search: there is no separate search function. The algorithm is written inline
in `bench_main`, the only entry point, run as
`python -m variants.v07_hybrid_actions --instances ... --out ...` (no
subcommand). Steps, in order:
  1. standard PUCT mcts on the original state, budget --b0 (default 600);
  2. enumerate all legal one-robot slides (move_planner.state.legal_moves) and
     build each moved state's initial plan (forced fixes free, as at any root).
     At --prefix-depth 2 or 3, the --d2-from (default 4) cheapest states of the
     previous depth each get one more legal slide; repeated positions are
     dropped. Depth-1, depth-2 and depth-3 prefixes then compete in ONE
     candidate list, ranked by abstract initial-plan cost plus prefix length;
  3. keep the --top-m (default 6) cheapest candidates and run standard mcts on
     each, budget --sub each (default 100), best-at-budget. A candidate whose
     rank cost is 3 or more above the best certified total so far is skipped
     and spends nothing (a pruning rule with no derivation, chosen by hand);
  4. a prefix result competes as (prefix length + strict). Return the best
     certified result overall.
Budget: the lanes can spend at most B0 + TOP_M*SUB, and --expansions (default
1200, the arena convention) is a hard cap on the sum. The defaults give exactly
1200. The landed configurations do NOT: the flagship depth-2 rows use
500/8/80 = 1140 and the depth-3 probe uses 460/10/70 = 1160, both BELOW the
1200 a standard-search control may use. Actual expansions spent are summed and
reported per row. This is deliberately a PORTFOLIO (no shared tree): sound,
simple, and it measures the ACTION-SPACE question -- "is one clever slide worth
more than 600 subgoal expansions?" -- without new estimators.

Bench-only: generation/training are untouched; run with the lab's best nets
(v09_strict_value_s8) vs the SAME nets under standard mcts. Every landed
flagship row is depth 2 at 500/8/80, not the 600/6/100 defaults below.
"""
from __future__ import annotations

from variants import Variant

B0, TOP_M, SUB = 600, 6, 100

VARIANT = Variant(
    vid="v07_hybrid_actions",
    axis="action-space",
    title="Root primitive slides (portfolio search, inference)",
    hypothesis="Spending half the arena budget on 'slide one robot, then plan' "
               "closes part of the 3/43 both-solved moves gap to the forward "
               "planner on unseen boards at unchanged solve rate.",
    mechanism="Standalone bench driver: standard mcts(600) on the original "
              "state + mcts(100) on the top-6 slide states (ranked by initial-"
              "plan cost); best certified result wins; slide totals pay +1 "
              "strict move; composed move dumps replay-validate.",
    expected_failure="One slide rarely changes the reachable plan set at 24x24 "
                     "(4 robots, sparse interactions) -> ties broken toward "
                     "the original search, moves unchanged, budget wasted on "
                     "sub-searches (visible as lower solve rate).",
    status="wave3",
)


def bench_main(argv=None):
    import argparse, json, os, time
    from pathlib import Path

    p = argparse.ArgumentParser(description="v07 root-slides bench driver")
    p.add_argument("--instances", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--value", required=True)
    p.add_argument("--expansions", type=int, default=1200)
    p.add_argument("--b0", type=int, default=B0)
    p.add_argument("--top-m", type=int, default=TOP_M)
    p.add_argument("--sub", type=int, default=SUB)
    p.add_argument("--prefix-depth", type=int, choices=[1, 2, 3], default=1,
                   help="2/3: also consider two-/three-slide prefixes (expanded "
                        "from the best --d2-from states of the previous depth); "
                        "prefixes pay their length in strict moves")
    p.add_argument("--d2-from", type=int, default=4)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--boards", choices=["pkl", "lean"], default="pkl")
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    import torch
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    from GridEnv import GridEnv, State, Robot_at
    from move_planner.state import COLOR_ORDER, legal_moves
    from simulate import wall_sets, DIRECTIONS, _board_size
    from skeleton.astar import _initial_plan
    from eval.compare import aggregate, load_instances
    from nn_labeler import leanboard
    from nn.generate import make_solver
    from spr.nets import load_policy, load_value
    from spr.search import Evaluator, mcts, forced_fixes
    import random

    policy = load_policy(a.policy, a.device)
    value_net = load_value(a.value, a.device)
    ev = Evaluator(policy, value_net, a.device, byref=True)
    solver = make_solver("b2")
    load_env = leanboard.from_env if a.boards == "lean" else GridEnv.from_env
    instances, sha, meta = load_instances(a.instances)
    if a.limit:
        instances = instances[:a.limit]
    placeholder = bool(instances) and all(i.get("d_star") in (0, None) for i in instances)

    rows, cur, env, wr, wd, n = [], None, None, None, None, None
    for i, inst in enumerate(instances):
        if inst["env_id"] != cur:
            env, _ = load_env(inst["env_id"])
            cur = inst["env_id"]
            n = _board_size(env.grid_data, None)
            wr, wd = wall_sets(env.grid_data, n)
        positions = tuple(tuple(p) for p in inst["positions"])
        tidx = inst["target_idx"]
        def mk_state(pos):
            return State(target=tuple(inst["target"]),
                         target_robot=Robot_at(position=pos[tidx], color=COLOR_ORDER[tidx]),
                         helpers=[Robot_at(position=pos[j], color=COLOR_ORDER[j])
                                  for j in range(len(pos)) if j != tidx])
        st = mk_state(positions)
        ev.acct = {"nn_policy_calls": 0, "nn_value_calls": 0}
        acct = {"nn_policy_calls": 0, "nn_value_calls": 0, "free_exact_fix_expands": 0,
                "rejected_plan_pops": 0, "physics_calls_prefix_check": 0,
                "physics_calls_strict_realize": 0, "physics_calls_park_repair": 0,
                "park_plans_pushed": 0}
        t0 = time.perf_counter()
        spent = 0
        # 1. original-state search
        r0 = mcts(env, st, solver, ev, cur, n, a.k, a.b0, None, True, 1.5, "min",
                  acct, True, parks=True, rng=random.Random(i))
        spent += r0.expansions
        best = None                      # (total_strict, moves, tag)
        if r0.strict is not None:
            best = (r0.strict, list(r0.moves or []), "subgoal", r0.plan)
        # 2. rank slide PREFIXES by moved-state initial-plan cost (+ prefix length)
        def plan_cost_of(st2):
            try:
                pl0 = forced_fixes(env, st2, solver, _initial_plan(env, st2))
                return float(pl0.cost())
            except Exception:
                return None
        cands = []                      # (rank_cost, prefix_moves, state)
        d1 = []
        for slot, di, newpos in legal_moves(positions, wr, wd, n):
            pos2 = tuple(tuple(q) for q in newpos)
            st2 = mk_state(pos2)
            c0 = plan_cost_of(st2)
            if c0 is None:
                continue
            d1.append((c0, pos2, [(slot, di)], st2))
            cands.append((c0 + 1, [(slot, di)], st2))
        prev = sorted(d1, key=lambda c: c[0])
        seen = {positions}
        for depth in range(2, a.prefix_depth + 1):
            nxt = []
            for c0, pos2, pre, _st2 in prev[:a.d2_from]:
                for slot, di, newpos in legal_moves(pos2, wr, wd, n):
                    pos3 = tuple(tuple(q) for q in newpos)
                    if pos3 in seen:
                        continue
                    seen.add(pos3)
                    st3 = mk_state(pos3)
                    c1 = plan_cost_of(st3)
                    if c1 is None:
                        continue
                    cands.append((c1 + depth, pre + [(slot, di)], st3))
                    nxt.append((c1, pos3, pre + [(slot, di)], st3))
            prev = sorted(nxt, key=lambda c: c[0])
        cands.sort(key=lambda c: c[0])
        # 3. sub-searches on the top-M prefix states
        for cost0, pre, st2 in cands[:a.top_m]:
            budget = min(a.sub, max(0, a.expansions - spent))
            if budget <= 0:
                break
            if best is not None and cost0 >= best[0] + 3:
                continue                 # hopeless margin only (abstract vs strict units differ)
            r2 = mcts(env, st2, solver, ev, cur, n, a.k, budget, None, True, 1.5,
                      "min", acct, True, parks=True, rng=random.Random(1000 + i))
            spent += r2.expansions
            plen = len(pre)
            if r2.strict is not None and (best is None or plen + r2.strict < best[0]):
                mv = [[COLOR_ORDER[sl], DIRECTIONS[dd]] for sl, dd in pre] + list(r2.moves or [])
                tag = f"slide{plen}" if plen > 1 else "slide"
                tag += ":" + "+".join(f"{COLOR_ORDER[sl]}:{DIRECTIONS[dd]}" for sl, dd in pre)
                best = (plen + r2.strict, mv, tag, r2.plan)
        dt = time.perf_counter() - t0
        acct["nn_policy_calls"] = ev.acct["nn_policy_calls"]
        acct["nn_value_calls"] = ev.acct["nn_value_calls"]
        row = {"env_id": cur, "d_star": None if placeholder else inst["d_star"],
               "plan_found": best is not None or r0.plan is not None,
               "plan_cost_abstract": None if best is None else float(best[3].cost()),
               "realized_abstract": None,
               "realized_strict": None if best is None else best[0],
               "solved": best is not None,
               "regret": (None if best is None or placeholder
                          else best[0] - inst["d_star"]),
               "expansions": spent, "seconds": dt,
               "plans_rejected": r0.rejected, "children_pruned": r0.pruned,
               "accounting": acct,
               "search": {"winner": None if best is None else best[2],
                          "b0": a.b0, "top_m": a.top_m, "sub": a.sub,
                          "prefix_depth": a.prefix_depth}}
        if best is not None:
            row["moves"] = best[1]
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  [v07] {i + 1}/{len(instances)} "
                  f"(slide wins so far: {sum(1 for r in rows if ((r.get('search') or {}).get('winner') or '').startswith('slide'))})",
                  flush=True)

    name = (f"v07 root-slides hybrid d{a.prefix_depth} [B2] "
            f"(b0={a.b0} top_m={a.top_m} sub={a.sub}) policy={Path(a.policy).name}")
    payload = {"protocol": {"expansions": a.expansions, "k": a.k,
                            "instances_file": str(a.instances), "instances_sha256": sha,
                            "n_instances": len(instances), "instances_meta": meta,
                            "checkpoints": {a.policy: "", a.value: ""},
                            "device": a.device, "d_star_placeholder": placeholder,
                            "dump_moves": True, "byref": True,
                            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "expansion_definition":
                                "portfolio: standard subgoal expansions summed over "
                                "the original-state search and <=top_m slide-state "
                                "sub-searches; slide totals pay +1 strict move"},
               "systems": {name: {"kind": "backward", "rows": rows,
                                  "aggregate": aggregate(rows, "realized_strict")}}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out)
    agg = payload["systems"][name]["aggregate"]
    sw = sum(1 for r in rows if ((r.get("search") or {}).get("winner") or "").startswith("slide") and r["solved"])
    print(f"[v07] solved {agg['solved']}/{agg['n']} mv {agg['mean_moves']:.2f} "
          f"exp {agg['mean_expansions']:.0f} | slide-winners {sw}")
    print(f"V07 BENCH DONE {out}")


if __name__ == "__main__":
    import sys
    bench_main(sys.argv[1:])
