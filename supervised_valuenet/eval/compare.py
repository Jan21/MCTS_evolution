"""Matched-budget head-to-head: backward subgoal planner vs forward move planner.

Both systems search with the same per-instance expansion budget and the same
top-k proposal width, over the SAME instance file (`eval.bench_instances`).
One expansion = one popped node whose children are generated; in both systems
this costs exactly one policy pass plus one batched value pass over <= k
children, so NN passes per expansion are matched. Quality is measured in
primitive moves against the per-instance move-optimal reference d* stored in
the instance file (oracle used only to make that label, never at inference).

Forward systems run `move_planner.evaluate.nn_astar` unmodified. The backward
system runs a local copy of `eval.end2end.nn_astar` that (i) counts expansions
as pops that reach the propose step (free exact-fix pops and the final
complete-plan pop are not expansions) and (ii) returns the complete plan
object, which `eval.realize` converts to primitive moves (abstract and strict
counts; "solved" = strict realization success).

    PYTHONPATH=. python -m eval.compare --instances eval/data/bench450.jsonl \
        --expansions 1200 --k 5 \
        --backward-policy <policy.ckpt> --backward-value <value.ckpt> \
        --forward-ckpts move_planner/checkpoints/best.ckpt,move_planner/checkpoints/candidate_scored.ckpt \
        --out eval/results/comparison.json --md COMPARISON.md

Use `--skip-backward` to run only the forward systems.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import heapq
import itertools
import json
import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# instances
# ---------------------------------------------------------------------------

def load_instances(path):
    body = Path(path).read_bytes()
    sha = hashlib.sha256(body).hexdigest()
    instances = [json.loads(line) for line in body.decode("utf-8").splitlines()
                 if line.strip()]
    meta_path = Path(str(path) + ".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    return instances, sha, meta


# ---------------------------------------------------------------------------
# forward systems (move planner)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _nullctx():
    """No-op stand-in so the counted and uncounted paths share one `with`."""
    yield


class _CountingGuide:
    """Wraps a move_planner Guide; counts eval_states calls (NN passes).

    `nn_astar` makes 1 root call, then exactly 2 calls (1 policy + 1 batched
    value) per expansion that has successors, so expansions ~= (calls - 1) // 2.

    `encode_bucket` (a context-manager factory; set only under --count-slides,
    None otherwise) re-attributes the slide calls made INSIDE eval_states to
    their own slide-counter bucket. Everything eval_states slides for is NN
    input featurization -- `dest_cells`' one-step lookahead per (robot, dir)
    plus the once-per-board `_slide_fields` fill -- not search physics; at 4
    robots and k=5 that is ~96 of the ~112 slides per forward expansion. The
    backward planner's counterpart featurization reads precomputed
    graph/distance tables and never calls slide, so leaving these in
    `forward_search` would let a cross-system ratio pass featurization off as
    physics work. Attribution only: the wrapped call itself is unchanged, and
    with `encode_bucket=None` the call path is exactly the pre-split one.
    """

    def __init__(self, guide, encode_bucket=None):
        self._guide = guide
        self._encode_bucket = encode_bucket
        self.calls = 0

    def eval_states(self, *args, **kwargs):
        self.calls += 1
        if self._encode_bucket is None:
            return self._guide.eval_states(*args, **kwargs)
        with self._encode_bucket():
            return self._guide.eval_states(*args, **kwargs)


def run_forward(ckpt, instances, k, budget, device, log=print,
                dump_moves=False, count_slides=False, placeholder_d_star=False):
    from move_planner.evaluate import nn_astar, Guide, walls_for
    from move_planner.state import COLOR_ORDER
    from simulate import DIRECTIONS
    from eval import slide_counter

    if count_slides:                    # after the imports above, so their
        slide_counter.install()         # captured `slide` bindings get rebound
    guide = _CountingGuide(
        Guide(ckpt, device),
        encode_bucket=((lambda: slide_counter.bucket("forward_encode"))
                       if count_slides else None))
    walls = {}
    rows = []
    for i, inst in enumerate(instances):
        env_id = inst["env_id"]
        if env_id not in walls:
            walls[env_id] = walls_for(env_id)
        wr, wd = walls[env_id]
        positions = tuple(tuple(p) for p in inst["positions"])
        target = tuple(inst["target"])
        d_star = inst["d_star"]
        c0 = guide.calls
        s0 = slide_counter.counts() if count_slides else None
        t0 = time.perf_counter()
        with slide_counter.bucket("forward_search") if count_slides \
                else _nullctx():
            cost, _path = nn_astar(guide, env_id, positions,
                                   inst["target_idx"], target, wr, wd,
                                   k=k, max_iters=budget)
        dt = time.perf_counter() - t0
        expansions = max(0, (guide.calls - c0 - 1)) // 2
        row = {
            "env_id": env_id,
            "d_star": None if placeholder_d_star else d_star,
            "solved": cost is not None,
            "moves": cost,
            "regret": (None if cost is None or placeholder_d_star
                       else cost - d_star),
            "expansions": expansions,
            "seconds": dt,
        }
        if dump_moves and cost is not None:
            # "moves" is already the count here, so the sequence gets its own key
            row["moves_seq"] = [[COLOR_ORDER[s], DIRECTIONS[d]]
                                for s, d in _path]
        row["accounting"] = {"nn_calls": guide.calls - c0}
        if count_slides:
            row["accounting"]["slide_calls"] = slide_counter.delta(s0)
        rows.append(row)
        if log and (i + 1) % 50 == 0:
            log(f"  [forward {Path(ckpt).name}] {i+1}/{len(instances)}")
    return rows


# ---------------------------------------------------------------------------
# backward system (subgoal planner)
# ---------------------------------------------------------------------------

def _nn_astar_backward(env, state, solver, policy, value, env_id, dev,
                       k, max_expansions, realize_check=None,
                       prefix_filter=None, park_hook=None, acct=None):
    """Local copy of eval.end2end.nn_astar with two changes: expansions are
    counted as pops that reach the propose step (capped at `max_expansions`;
    free exact-fix loops and the final complete-plan pop are free), and the
    COMPLETE PLAN object is returned instead of its cost.
    With `realize_check` (anytime mode): a popped complete plan that fails the
    check is discarded and the search continues within the same expansion
    budget; the first passing plan is returned. If the budget exhausts, the
    first failed complete plan (if any) is returned so plan_found stays
    truthful.
    With `prefix_filter` (Lever A, opt-in): every child plan produced by an
    expansion is tested for prefix realizability BEFORE being pushed; children
    whose already-fixed segments cannot be played are dropped, so the budget is
    not spent completing doomed branches. Orthogonal to `realize_check` (which
    still governs what happens when a complete plan POPS).
    `acct` (optional dict): compute-accounting counters incremented in place
    (nn_policy_calls, nn_value_calls, free_exact_fix_expands); observational.
    Returns (plan | None, expansions_used, n_rejected, n_pruned)."""
    from skeleton.astar import _initial_plan, _segment, _apply
    from nn.generate import _context, _fixed_g
    from nn.collect_search import _record
    from train.policy_common import _ix
    from eval.end2end import _policy_logp, _value_cost, _hidx

    cnt = itertools.count()
    frontier = [(0.0, next(cnt), _initial_plan(env, state))]
    expansions = 0
    rejected = 0
    pruned = 0
    first_failed = None
    while frontier and expansions < max_expansions:
        _, _, plan = heapq.heappop(frontier)
        while not plan.is_complete():                     # free forced exact fixes
            parent, child = plan.open_edges()[0]
            seg = _segment(plan, state, parent, child)
            if env.compute_exact_shortest_path_length(seg.start, seg.end,
                                                      seg.fix_support) is not None:
                if acct is not None:
                    acct["free_exact_fix_expands"] += 1
                plan = solver._expand(env, state, plan)[0]
            else:
                break
        if plan.is_complete():
            if realize_check is None or realize_check(plan):
                return plan, expansions, rejected, pruned  # first (executable) complete
            rejected += 1                                 # anytime: discard, keep searching
            if park_hook is not None:
                # B1 parks: deterministic physics repairs of the failed plan
                # re-enter the frontier as ordinary costed plans
                for rp in park_hook(plan):
                    heapq.heappush(frontier, (float(rp.cost()), next(cnt), rp))
            if first_failed is None:
                first_failed = plan
            continue
        expansions += 1                                   # this pop generates children
        parent, child = plan.open_edges()[0]
        seg = _segment(plan, state, parent, child)
        cands = solver.propose(env, seg.end, seg.mover, seg.helpers, seg.support)
        ctx = _context(plan)
        recs, cps = [], []
        for cand in cands:
            cp = _apply(env, plan, parent, child, seg, cand)
            if cp is None or _hidx(state, _record(env_id, state, seg, ctx, cand,
                                                  0, 0, 0)["cand_helper"][0]) is None:
                continue
            recs.append(_record(env_id, state, seg, ctx, cand, 0, 0, 0))
            cps.append(cp)
        if not recs:
            continue
        if acct is not None:
            acct["nn_policy_calls"] += 1
        logp, _ = _policy_logp(policy, recs, dev)
        if logp is None:
            continue
        keys = [(_ix(r["cand_bottleneck"]), _ix(r["cand_support"]),
                 _hidx(state, r["cand_helper"][0])) for r in recs]
        order = sorted(range(len(recs)), key=lambda i: -logp.get(keys[i], -1e9))[:k]
        if acct is not None:
            acct["nn_value_calls"] += 1
        costs = _value_cost(value, [recs[i] for i in order], dev)
        for i, h in zip(order, costs):
            if prefix_filter is not None and not prefix_filter(cps[i]):
                pruned += 1                               # doomed prefix: never push
                continue
            heapq.heappush(frontier, (float(_fixed_g(cps[i])) + h, next(cnt), cps[i]))
    return first_failed, expansions, rejected, pruned


def run_backward(policy_ckpt, value_ckpt, instances, k, budget, device,
                 log=print, anytime=False, prefix_check=False, b1=False,
                 b2=False, dump_moves=False, count_slides=False,
                 placeholder_d_star=False):
    from GridEnv import GridEnv, State, Robot_at
    from skeleton.astar import AStar, park_repairs
    from skeleton import heuristics
    from simulate import wall_sets
    from train.policy_tf import PolicyTF
    from train.looped_pc import LoopedValueNet
    from move_planner.state import COLOR_ORDER
    from eval.realize import (abstract_moves, strict_moves, prefix_playable,
                              prefix_key)
    from eval import slide_counter

    if count_slides:                    # after the imports above, so their
        slide_counter.install()         # captured `slide` bindings get rebound
    ctx = slide_counter.bucket if count_slides else (lambda _name: _nullctx())

    policy = PolicyTF.load_from_checkpoint(policy_ckpt, map_location=device).to(device).eval()
    value = LoopedValueNet.load_from_checkpoint(value_ckpt, map_location=device).to(device).eval()
    # B1 mode: candidates come from the extended vocabulary (wall-less transient
    # stoppers); failed complete plans additionally get deterministic park repairs
    # (below). Requires nets trained on B1-vocabulary labels for sensible ranking.
    # B2 mode (implies the B1 vocabulary): generalized park repairs (pairwise
    # clearing, multi-slide destinations, park cap 2). NOTE (FINDINGS 40): the
    # AStar by_reference flag set below affects only solver._expand, which the
    # NN loop reaches solely on the free-exact-fix branch -- _nn_astar_backward
    # itself never injects _reference_helpers and never passes
    # by_reference=True to _apply, and _hidx drops any candidate whose helper
    # is not at a robot start. So by-reference candidates are structurally
    # absent from every learned row this driver produces; the earlier claim
    # here that "the nets rank them zero-shot" was false. The correctly wired
    # by-reference path lives in nn/generate.py and the ceiling probe.
    if b2:
        b1 = True
    solver = AStar(propose=heuristics.propose_b1 if b1 else heuristics.propose,
                   max_iters=4000, max_frontier=40_000, by_reference=b2)

    cur_env_id, env = None, None
    rows = []
    for i, inst in enumerate(instances):
        env_id = inst["env_id"]
        if env_id != cur_env_id:
            env, _ = GridEnv.from_env(env_id)             # lazy per-board build/cache
            cur_env_id = env_id
        positions = [tuple(p) for p in inst["positions"]]
        tidx = inst["target_idx"]
        st = State(
            target=tuple(inst["target"]),
            target_robot=Robot_at(position=positions[tidx], color=COLOR_ORDER[tidx]),
            helpers=[Robot_at(position=positions[j], color=COLOR_ORDER[j])
                     for j in range(len(positions)) if j != tidx])
        d_star = inst["d_star"]
        s0 = slide_counter.counts() if count_slides else None
        t0 = time.perf_counter()
        # compute-accounting counters (observational; emitted per row). The
        # physics_calls_* counters count ENTRY-POINT calls into the
        # realize/park layer from their three compare.py call sites; the
        # per-slide unit that IS commensurable with the forward planner's
        # physics is the separate `slide_calls` map, filled under
        # --count-slides by eval/slide_counter.py.
        acct = {"nn_policy_calls": 0, "nn_value_calls": 0,
                "free_exact_fix_expands": 0, "rejected_plan_pops": 0,
                "physics_calls_prefix_check": 0,
                "physics_calls_strict_realize": 0,
                "physics_calls_park_repair": 0,
                "park_plans_pushed": 0}
        moves_cache = {}                              # id(plan) -> [color, dir] list
        check_cache = {}
        fail_cache = {}
        realize_check = None
        if anytime:
            def realize_check(p, _env=env, _st=st, _cache=check_cache,
                              _fails=fail_cache, _acct=acct,
                              _mvs=moves_cache):
                fi = {}
                mv = [] if dump_moves else None
                _acct["physics_calls_strict_realize"] += 1
                with ctx("strict_realize"):
                    m = strict_moves(_env, _st, p, log=None, fail_info=fi,
                                     moves_out=mv)
                _cache[id(p)] = m
                if dump_moves and m is not None:
                    _mvs[id(p)] = mv
                if m is None:
                    _fails[id(p)] = fi
                return m is not None
        park_hook = None
        if b1 and anytime:
            _wr, _wd = wall_sets(env.grid_data)
            _size = len(env.grid_data) ** 0.5
            def park_hook(p, _env=env, _st=st, _fails=fail_cache,
                          _wr=_wr, _wd=_wd, _acct=acct):
                fi = _fails.get(id(p))
                if not fi:
                    return []
                _acct["physics_calls_park_repair"] += 1
                try:
                    with ctx("park_repair"):
                        rps = park_repairs(_env, _st, p, fi, _wr, _wd,
                                           int(round(_size)),
                                           max_parks=2 if b2 else 1,
                                           pairwise=b2, multi_slide=b2)
                except Exception:
                    return []
                _acct["park_plans_pushed"] += len(rps)
                return rps
        prefix_filter = None
        if prefix_check:
            pfx_cache = {}                                # per-instance memo
            def prefix_filter(p, _env=env, _st=st, _cache=pfx_cache,
                              _acct=acct):
                key = prefix_key(p)
                hit = _cache.get(key)
                if hit is None:
                    _acct["physics_calls_prefix_check"] += 1
                    with ctx("prefix_check"):
                        hit = _cache[key] = prefix_playable(_env, _st, p)
                return hit
        with ctx("backward_search"):
            plan, expansions, rejected, pruned = _nn_astar_backward(
                env, st, solver, policy, value, env_id, device, k, budget,
                realize_check=realize_check, prefix_filter=prefix_filter,
                park_hook=park_hook, acct=acct)
        acct["rejected_plan_pops"] = rejected
        dt = time.perf_counter() - t0
        row = {
            "env_id": env_id,
            "d_star": None if placeholder_d_star else d_star,
            "plan_found": plan is not None,
            "plan_cost_abstract": None, "realized_abstract": None,
            "realized_strict": None, "solved": False, "regret": None,
            "expansions": expansions, "seconds": dt,
            "plans_rejected": rejected,
            "children_pruned": pruned,
        }
        if plan is not None:
            row["plan_cost_abstract"] = float(plan.cost())
            # `abstract_scoring` bucket: abstract_moves re-costs each segment
            # of the FOUND plan under the blocker-clearing model to produce
            # the diagnostic `realized_abstract` figure. That is scoring of a
            # result, not search (no frontier is touched) and not realization
            # (no legal joint-state execution is attempted), so it gets its
            # own bucket rather than inflating backward_search or
            # strict_realize -- previously these slides ran under no bucket
            # at all and were silently missing from `slide_calls`.
            with ctx("abstract_scoring"):
                ab, _verified = abstract_moves(env, st, plan, log=log)
            if anytime:
                stx = check_cache.get(id(plan))
                if id(plan) not in check_cache:           # budget-exhausted fallback plan
                    mv = [] if dump_moves else None
                    acct["physics_calls_strict_realize"] += 1
                    with ctx("strict_realize"):
                        stx = strict_moves(env, st, plan, log=None,
                                           moves_out=mv)
                    if dump_moves and stx is not None:
                        moves_cache[id(plan)] = mv
            else:
                mv = [] if dump_moves else None
                acct["physics_calls_strict_realize"] += 1
                with ctx("strict_realize"):
                    stx = strict_moves(env, st, plan, log=log, moves_out=mv)
                if dump_moves and stx is not None:
                    moves_cache[id(plan)] = mv
            row["realized_abstract"] = ab
            row["realized_strict"] = stx
            row["solved"] = stx is not None               # strict realization success
            row["regret"] = (None if stx is None or placeholder_d_star
                             else stx - d_star)
        if dump_moves and row["solved"]:
            row["moves"] = moves_cache[id(plan)]
        if count_slides:
            acct["slide_calls"] = slide_counter.delta(s0)
        row["accounting"] = acct
        rows.append(row)
        if log and (i + 1) % 25 == 0:
            log(f"  [backward] {i+1}/{len(instances)}")
    return rows


# ---------------------------------------------------------------------------
# aggregation + reports
# ---------------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(rows, moves_key="moves"):
    n = len(rows)
    solved = [r for r in rows if r.get("solved")]
    # Beyond-oracle ("frontier") sets have no known optimum. Their instance
    # files historically carry d_star = 0, which silently turns regret into
    # "solution length" and pct_optimal into "fraction solved in 0 moves" --
    # publishability objection 0.6. When the driver detects that placeholder it
    # nulls d_star per row, and every derived quantity is suppressed here
    # rather than published as a number that means something else.
    placeholder = bool(rows) and rows[0].get("d_star") is None
    agg = {
        "n": n,
        "solved": len(solved),
        "solve_rate": len(solved) / n if n else None,
        "d_star_placeholder": placeholder,
        "mean_regret": None if placeholder else _mean([r["regret"]
                                                       for r in solved]),
        "pct_optimal": None if placeholder or not solved else
                       (100.0 * sum(1 for r in solved if r["regret"] == 0)
                        / len(solved)),
        "mean_moves": _mean([r[moves_key] for r in solved]),
        "mean_expansions": _mean([r.get("expansions") for r in rows]),
        "mean_seconds": _mean([r["seconds"] for r in rows]),
    }
    if rows and "plan_found" in rows[0]:                  # backward extras
        found = [r for r in rows if r["plan_found"]]
        agg["plan_found"] = len(found)
        agg["plan_found_rate"] = len(found) / n if n else None
        agg["mean_plan_cost_abstract"] = _mean([r["plan_cost_abstract"] for r in found])
        agg["mean_realized_abstract"] = _mean([r["realized_abstract"] for r in found])
        agg["mean_realized_strict"] = _mean([r["realized_strict"] for r in solved])
        agg["n_negative_abstract_regret"] = None if placeholder else sum(
            1 for r in found
            if r["realized_abstract"] is not None
            and r["realized_abstract"] - r["d_star"] < 0)
        if "children_pruned" in rows[0]:
            agg["mean_children_pruned"] = _mean(
                [r["children_pruned"] for r in rows])
    if rows and "accounting" in rows[0]:                  # per-counter means
        acc = {}
        for key in rows[0]["accounting"]:
            if key == "slide_calls":                     # nested bucket -> mean
                buckets = sorted({b for r in rows
                                  for b in r["accounting"].get(key, {})})
                acc[key] = {b: _mean([r["accounting"].get(key, {}).get(b, 0)
                                      for r in rows]) for b in buckets}
                acc["slide_calls_total"] = _mean(
                    [sum(r["accounting"].get(key, {}).values()) for r in rows])
            else:
                acc[key] = _mean([r["accounting"].get(key) for r in rows])
        agg["accounting"] = acc
    return agg


def _fmt(x, spec=".3f"):
    if x is None:
        return "--"
    if isinstance(x, float):
        return format(x, spec)
    return str(x)


def write_markdown(path, protocol, systems, order):
    """`systems`: name -> {'aggregate':..., 'kind': 'forward'|'backward'|'pending'}."""
    n = protocol.get("n_instances")
    lines = []
    lines.append("# Matched-budget comparison: backward subgoal planner vs "
                 "forward move planner")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- **Instances**: the identical set of {n} instances for every "
                 f"system, materialized once by `eval/bench_instances.py` "
                 f"(`{protocol['instances_file']}`, sha256 "
                 f"`{protocol['instances_sha256'][:16]}...`). Each instance "
                 "stores the exact move-optimal cost d\\* (oracle used only for "
                 "this reference label, never at inference).")
    lines.append(f"- **Matched budget**: every system runs best-first search "
                 f"capped at **{protocol['expansions']} expansions** per "
                 f"instance with **top-k = {protocol['k']}** proposals per "
                 "expansion (primary operating point). One expansion = one "
                 "popped node whose children are generated; in both systems "
                 "this costs exactly one policy pass plus one batched value "
                 "pass over <= k children, so NN passes per expansion are "
                 "equivalent. Wall-clock time is reported as a secondary "
                 "metric.")
    lines.append("- **Scoring**: primitive moves. Forward plans are already "
                 "primitive move sequences. Backward plans are subgoal DAGs "
                 "realized into moves by `eval/realize.py`; the backward "
                 "\"solved\" criterion is STRICT realization success (segments "
                 "executed in dependency order under full joint-state slide "
                 "physics). Regret = achieved moves - d\\*, over solved "
                 "instances only.")
    lines.append(f"- **Run**: `{protocol.get('command', '')}` "
                 f"({protocol['date']}). This report is regenerated by every "
                 "`eval.compare` run; the table reflects the instance file "
                 "and budget above.")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(f"(n = {n} instances; solve rate uses n as denominator)")
    lines.append("")
    lines.append("| system | solve rate | mean regret (moves) | % optimal | "
                 "mean plan length (moves) | mean expansions | mean s/inst |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in order:
        info = systems[name]
        if info.get("kind") == "pending":
            lines.append(f"| {name} | -- | -- | -- | -- | -- | -- |")
            continue
        a = info["aggregate"]
        sr = f"{a['solved']}/{a['n']} ({100.0*a['solve_rate']:.1f}%)" if a["n"] else "--"
        lines.append(
            f"| {name} | {sr} | {_fmt(a['mean_regret'])} | "
            f"{_fmt(a['pct_optimal'], '.1f')} | {_fmt(a['mean_moves'], '.2f')} | "
            f"{_fmt(a['mean_expansions'], '.1f')} | {_fmt(a['mean_seconds'])} |")
    lines.append("")

    back = [(nm, s) for nm, s in systems.items()
            if s.get("kind") == "backward" and "aggregate" in s]
    lines.append("## Appendix: backward planner detail")
    lines.append("")
    if back:
        for nm, s in back:
            a = s["aggregate"]
            lines.append(f"- **{nm}**: plan-found rate "
                         f"{a['plan_found']}/{a['n']} vs strict-solve rate "
                         f"{a['solved']}/{a['n']}. Mean abstract plan cost "
                         f"{_fmt(a['mean_plan_cost_abstract'], '.2f')}; mean "
                         f"abstract realized moves "
                         f"{_fmt(a['mean_realized_abstract'], '.2f')} (plans "
                         f"found); mean strict realized moves "
                         f"{_fmt(a['mean_realized_strict'], '.2f')} (strictly "
                         f"solved). Instances with negative abstract regret: "
                         f"{a['n_negative_abstract_regret']}.")
    else:
        lines.append("- Backward system not run in this report (pending "
                     "checkpoints); rows above are placeholders.")
    lines.append("- The abstract count sums each plan segment simulated with "
                 "only its intended support robot on the board "
                 "(blocker-clearing model). It can under-count true game "
                 "moves, so regret computed from it can be negative; the "
                 "strict count is the fair, legal-execution figure and is "
                 "what the main table reports.")
    lines.append("")
    lines.append("## Footnotes")
    lines.append("")
    lines.append("1. **Expansion granularity**: one backward expansion commits "
                 "an entire subgoal (a multi-move commitment: mover approach "
                 "plus helper placement), while one forward expansion commits "
                 "a single primitive move. A shared expansion cap therefore "
                 "grants the backward planner strictly more solution-building "
                 "work per expansion; the matched budget is generous to the "
                 "backward system (the conservative direction for the "
                 "comparison).")
    lines.append("2. **Environment model**: the backward planner's environment "
                 "includes precomputed exact shortest-path distance tables per "
                 "board (used for exact-fix checks and edge costs at "
                 "inference). The forward planner uses only the wall geometry "
                 "at inference.")
    lines.append("3. Checkpoints, budgets, timestamps and the full "
                 "per-instance rows are recorded in "
                 f"`{protocol['results_file']}`.")
    lines.append("")
    Path(path).write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instances", default="eval/data/bench450.jsonl")
    p.add_argument("--expansions", type=int, default=1200)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--backward-policy", default=None)
    p.add_argument("--backward-value", default=None)
    p.add_argument("--forward-ckpts",
                   default="move_planner/checkpoints/best.ckpt")
    p.add_argument("--skip-backward", action="store_true")
    p.add_argument("--backward-anytime", action="store_true",
                   help="keep searching past complete plans that fail strict "
                        "realization (same expansion budget)")
    p.add_argument("--backward-prefix-check", action="store_true",
                   help="Lever A: prune child plans whose already-fixed "
                        "segments fail strict prefix realization before they "
                        "enter the frontier (works with and without "
                        "--backward-anytime; default OFF so baselines stay "
                        "reproducible)")
    p.add_argument("--backward-b1", action="store_true",
                   help="extended plan language: wall-less transient stoppers "
                        "(propose_b1) plus deterministic park repairs of failed "
                        "complete plans (with --backward-anytime); use with nets "
                        "trained on B1-vocabulary labels")
    p.add_argument("--backward-b2", action="store_true",
                   help="Lever B2 (implies --backward-b1's vocabulary): "
                        "supports-by-reference proposals plus generalized park "
                        "repairs (pairwise clearing, multi-slide destinations, "
                        "park cap 2); by-reference candidates are ranked "
                        "zero-shot by the B1 nets")
    p.add_argument("--dump-moves", action="store_true",
                   help="record each solved row's realized primitive-move "
                        "sequence ([color, direction] per slide; backward key "
                        "'moves', forward key 'moves_seq') for independent "
                        "replay certification by eval/replay_validate.py")
    p.add_argument("--count-slides", action="store_true",
                   help="count simulate.slide invocations per instance, split "
                        "by phase, in BOTH systems -- the matched physics-work "
                        "unit of publishability objection 1.1. Forward slides "
                        "are split into forward_search (legal_moves physics) "
                        "vs forward_encode (NN featurization; the majority -- "
                        "see eval/slide_counter.py's caveat before quoting "
                        "any cross-system ratio); calls under no bucket land "
                        "in the 'unbucketed' sentinel instead of being "
                        "dropped. Adds a Python call to the hottest function "
                        "in the codebase, so a counted run's wall-clock is "
                        "NOT comparable to an uncounted one: run accounting "
                        "passes separately from timing passes.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="eval/results/comparison.json")
    p.add_argument("--md", default="COMPARISON.md")
    a = p.parse_args()

    import torch
    torch.manual_seed(0)

    instances, sha, meta = load_instances(a.instances)
    # Beyond-oracle instance files carry d_star = 0 for every puzzle because no
    # optimum is known. Detect that (a real puzzle never has d_star = 0: the
    # target robot would already be on the goal) and null the derived fields
    # instead of publishing solution length as "regret" -- objection 0.6.
    # Beyond-oracle sets carry no optimum. Historically that was written as
    # d_star = 0; a set derived as "bench minus the graded half" carries None.
    # Both are placeholders and both must suppress regret -- None additionally
    # would crash `cost - d_star` outright.
    placeholder_d_star = bool(instances) and all(
        i.get("d_star") in (0, None) for i in instances)
    if placeholder_d_star:
        print(f"[compare] d_star placeholder detected in {a.instances}: "
              "regret / pct_optimal suppressed for this run")
    fwd_ckpts = [c for c in a.forward_ckpts.split(",") if c.strip()]
    run_back = not a.skip_backward and a.backward_policy and a.backward_value

    ckpt_info = {}
    for c in fwd_ckpts + ([a.backward_policy, a.backward_value] if run_back else []):
        ckpt_info[c] = time.strftime("%Y-%m-%dT%H:%M:%S",
                                     time.localtime(os.path.getmtime(c)))

    protocol = {
        "expansions": a.expansions,
        "k": a.k,
        "instances_file": a.instances,
        "instances_sha256": sha,
        "n_instances": len(instances),
        "instances_meta": meta,
        "checkpoints": ckpt_info,
        "device": a.device,
        "d_star_placeholder": placeholder_d_star,
        "count_slides": a.count_slides,
        "dump_moves": a.dump_moves,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results_file": a.out,
        "command": " ".join(["PYTHONPATH=. python -m eval.compare"]
                            + sys.argv[1:]),
        "expansion_definition": "one popped search node whose children are "
                                "generated (= 1 policy pass + 1 batched value "
                                "pass over <= k children, in both systems)",
    }

    systems, order = {}, []

    back_name = ("backward subgoal planner (anytime realization-checked)"
                 if a.backward_anytime else "backward subgoal planner")
    if a.backward_prefix_check:
        back_name += " (prefix-check)"
    if a.backward_b2:
        back_name += " [extended language B2]"
    elif a.backward_b1:
        back_name += " [extended language B1]"
    if run_back:
        print(f"[compare] backward: policy={a.backward_policy} "
              f"value={a.backward_value} anytime={a.backward_anytime} "
              f"prefix_check={a.backward_prefix_check} b1={a.backward_b1} "
              f"b2={a.backward_b2}")
        rows = run_backward(a.backward_policy, a.backward_value, instances,
                            a.k, a.expansions, a.device,
                            anytime=a.backward_anytime,
                            prefix_check=a.backward_prefix_check,
                            b1=a.backward_b1, b2=a.backward_b2,
                            dump_moves=a.dump_moves,
                            count_slides=a.count_slides,
                            placeholder_d_star=placeholder_d_star)
        systems[back_name] = {"kind": "backward",
                              "aggregate": aggregate(rows, "realized_strict"),
                              "rows": rows}
    else:
        systems[back_name] = {"kind": "pending"}
        print("[compare] backward skipped (placeholder row in report)")
    order.append(back_name)

    for ckpt in fwd_ckpts:
        name = f"forward move planner ({ckpt})"
        print(f"[compare] forward: {ckpt}")
        rows = run_forward(ckpt, instances, a.k, a.expansions, a.device,
                           dump_moves=a.dump_moves,
                           count_slides=a.count_slides,
                           placeholder_d_star=placeholder_d_star)
        systems[name] = {"kind": "forward", "aggregate": aggregate(rows),
                         "rows": rows}
        order.append(name)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"protocol": protocol,
               "systems": {nm: {k2: v2 for k2, v2 in s.items()} for nm, s in systems.items()}}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(a.md, protocol, systems, order)
    print(f"[compare] wrote {out} and {a.md}")
    for nm in order:
        s = systems[nm]
        if "aggregate" not in s:
            print(f"  {nm}: pending")
            continue
        ag = s["aggregate"]
        print(f"  {nm}: solved {ag['solved']}/{ag['n']} "
              f"mean_regret={_fmt(ag['mean_regret'])} "
              f"optimal%={_fmt(ag['pct_optimal'], '.1f')} "
              f"mean_exp={_fmt(ag['mean_expansions'], '.1f')} "
              f"s/inst={_fmt(ag['mean_seconds'])}")


if __name__ == "__main__":
    main()
