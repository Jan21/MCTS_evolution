"""Candidate-scored self-play record generation for the subgoal planner.

The current nets' own budgeted A* over partial-plan DAGs (a traced fork of
`eval.end2end.nn_astar`) is the expert. On a sampled instance it either finds a
complete plan or it does not:

- SOLVED   -> walk the parent-pointer chain of the winning node back to the root.
  Every decision on that chain kept a record of ALL the candidates it value-scored
  (policy top-k + occasional epsilon extras). For each decision: the chosen child's
  cost-to-go is exact from the winning plan (`win_cost - fixed_g(parent)`); every
  SIBLING child plan is commit-and-completed with a small-budget run of the same
  net-guided A*, giving `ctg = completion_cost - fixed_g(parent)`. Siblings whose
  completion exhausts the budget drop only their own record. `is_optimal` is the
  argmin of `cost_to_go` within the group.
- UNSOLVED -> DROP the instance entirely (the implicit curriculum).

Why candidate-scored rather than one-hot: both trainers are group-structured. The
value net's rank loss needs `is_optimal` inside multi-candidate groups, and the
policy's soft target needs `cost_to_go` for every candidate in the group. Scoring
the whole top-k gives full groups in the EXACT `nn/data/combined.jsonl` schema
(built by `nn.collect_search._record`), so `train.looped_pc` and `train.policy_tf`
datasets/models are imported and reused verbatim.

No oracle appears anywhere here: every label comes from the nets' own search.
"""
from __future__ import annotations

import heapq
import itertools
from collections import Counter

import torch

from GridEnv import GridEnv
from skeleton.astar import AStar, _initial_plan, _segment, _apply
from nn.generate import _context, _fixed_g
from nn.collect_search import _record
from train.policy_common import _ix
# _policy_logp / _value_cost / _hidx are the exact inference helpers the eval
# planner uses; importing them keeps ranking/scoring bit-identical with eval.
from eval.end2end import _policy_logp, _value_cost, _hidx

# eval.end2end globally disables autograd at import time (it is inference-only);
# undo that here so the training half of the loop still builds graphs.
torch.set_grad_enabled(True)


# -- one decision expansion (shared by traced expert / sibling completions) -----

def _pop_decision(env, state, solver, plan, env_id):
    """Apply forced exact-path fixes, then enumerate the first open decision.

    Returns `(plan, recs, cps)`: the (possibly advanced) plan, plus per-candidate
    records (combined.jsonl schema, labels zeroed) and committed child plans.
    `recs is None` iff the plan became complete. Mirrors eval.end2end.nn_astar's
    pop body exactly.
    """
    while not plan.is_complete():                       # free exact-path fixes
        parent, child = plan.open_edges()[0]
        seg = _segment(plan, state, parent, child)
        if env.compute_exact_shortest_path_length(seg.start, seg.end, seg.fix_support) is not None:
            plan = solver._expand(env, state, plan)[0]
        else:
            break
    if plan.is_complete():
        return plan, None, None
    parent, child = plan.open_edges()[0]
    seg = _segment(plan, state, parent, child)
    cands = solver.propose(env, seg.end, seg.mover, seg.helpers, seg.support)
    ctx = _context(plan)
    recs, cps = [], []
    for cand in cands:
        cp = _apply(env, plan, parent, child, seg, cand)
        if cp is None:
            continue
        r = _record(env_id, state, seg, ctx, cand, 0, 0, 0)   # labels patched later
        if _hidx(state, r["cand_helper"][0]) is None:
            continue
        recs.append(r)
        cps.append(cp)
    return plan, recs, cps


def _rank_and_score(policy, value, recs, state, dev, k, rng=None,
                    epsilon=0.0, extra_max=2):
    """Policy-rank the candidates (1 pass), keep top-k, value-score the kept set.

    With prob `epsilon` (GENERATION only) 1..`extra_max` random beyond-top-k
    candidates are appended before scoring, for ranking-blind-spot coverage.
    Returns `(kept_indices, costs)` or `(None, None)` when the policy can't rank.
    """
    logp, _ = _policy_logp(policy, recs, dev)
    if logp is None:
        return None, None
    keys = [(_ix(r["cand_bottleneck"]), _ix(r["cand_support"]),
             _hidx(state, r["cand_helper"][0])) for r in recs]
    order = sorted(range(len(recs)), key=lambda i: -logp.get(keys[i], -1e9))
    kept = order[:k]
    if rng is not None and epsilon > 0.0 and len(order) > k and rng.random() < epsilon:
        n_extra = rng.randint(1, extra_max)
        kept = kept + rng.sample(order[k:], min(n_extra, len(order) - k))
    costs = _value_cost(value, [recs[i] for i in kept], dev)
    return kept, costs


# -- (a) the traced expert: nn_astar fork with parent pointers ------------------

class _Node:
    """Frontier entry. `decision` (shared by all siblings of one expansion) holds
    the parent plan's fixed_g plus the scored candidates' records and child plans;
    `idx` is this child's position inside that scored set."""
    __slots__ = ("plan", "parent", "decision", "idx")

    def __init__(self, plan, parent=None, decision=None, idx=None):
        self.plan, self.parent, self.decision, self.idx = plan, parent, decision, idx


def nn_astar_traced(env, state, solver, policy, value, env_id, dev, cfg, rng,
                    realize_check=None, prefix_filter=None):
    """Fork of eval.end2end.nn_astar keeping per-push parent pointers.

    Same loop: pop plan -> free exact-path fixes -> propose -> policy ranks (1 pass)
    -> top-k (+ epsilon extras) -> value scores -> push children with
    f = fixed_g(child) + value. `decision_chain = [(decision, chosen_idx), ...]`
    root-to-leaf.

    With `realize_check` (generation-time option): a popped complete plan that fails
    the check is discarded and the search continues — generation wants only playable
    winners, there is no fallback plan.

    With `prefix_filter` (Lever A, generation-time option): a child plan whose
    already-fixed segments fail strict prefix realization is dropped before the
    push, so the expert never spends budget completing a doomed branch. Uniform
    return shape: `(plan | None, chain | None, n_rejected)`.
    """
    cnt = itertools.count()
    frontier = [(0.0, next(cnt), _Node(_initial_plan(env, state)))]
    iters = 0
    n_rejected = 0
    while frontier and iters < cfg.astar_iters:
        iters += 1
        _, _, node = heapq.heappop(frontier)
        plan, recs, cps = _pop_decision(env, state, solver, node.plan, env_id)
        if recs is None:                                  # complete: first pop wins
            if realize_check is not None and not realize_check(plan):
                n_rejected += 1
                continue
            chain = []
            n = node
            while n.parent is not None:
                chain.append((n.decision, n.idx))
                n = n.parent
            chain.reverse()
            return plan, chain, n_rejected
        if not recs:
            continue
        kept, costs = _rank_and_score(policy, value, recs, state, dev, cfg.k_top,
                                      rng=rng, epsilon=cfg.epsilon,
                                      extra_max=cfg.epsilon_extra_max)
        if kept is None:
            continue
        if prefix_filter is not None:                     # Lever A: drop doomed children
            live = [j for j, i in enumerate(kept) if prefix_filter(cps[i])]
            kept = [kept[j] for j in live]
            costs = [costs[j] for j in live]
            if not kept:
                continue
        decision = {"fixed_g": int(_fixed_g(plan)),
                    "recs": [recs[i] for i in kept],
                    "cps": [cps[i] for i in kept]}
        for j, (i, h) in enumerate(zip(kept, costs)):
            heapq.heappush(frontier, (float(_fixed_g(cps[i])) + h, next(cnt),
                                      _Node(cps[i], parent=node, decision=decision, idx=j)))
    return None, None, n_rejected


def nn_astar_from(env, state, solver, policy, value, env_id, dev,
                  start_plan=None, k=5, max_iters=300, prefix_filter=None):
    """Untraced nn_astar accepting an arbitrary start plan (no epsilon, no trace).

    Used for sibling commit-and-complete (small budget) and for the probe eval
    (start_plan=None -> full instance at the eval budget). With `prefix_filter`
    (Lever A) children with an unplayable fixed prefix are dropped before the
    push. Returns the complete PLAN object or None (callers take .cost() and/or
    realize it).
    """
    cnt = itertools.count()
    plan0 = start_plan if start_plan is not None else _initial_plan(env, state)
    frontier = [(0.0, next(cnt), plan0)]
    iters = 0
    while frontier and iters < max_iters:
        iters += 1
        _, _, plan = heapq.heappop(frontier)
        plan, recs, cps = _pop_decision(env, state, solver, plan, env_id)
        if recs is None:
            return plan
        if not recs:
            continue
        kept, costs = _rank_and_score(policy, value, recs, state, dev, k)
        if kept is None:
            continue
        for i, h in zip(kept, costs):
            if prefix_filter is not None and not prefix_filter(cps[i]):
                continue                                  # Lever A: doomed prefix
            heapq.heappush(frontier, (float(_fixed_g(cps[i])) + h, next(cnt), cps[i]))
    return None


# -- (b) commit-and-complete sibling scorer --------------------------------------

def label_chain(env, state, solver, policy, value, env_id, dev, win_cost, chain, cfg,
                prefix_filter=None):
    """Label every decision on the winning chain in the combined.jsonl schema.

    Chosen child: ctg is exact from the expert's own completion
    (`win_cost - fixed_g(parent)`). Siblings: small-budget net-guided completion
    from the committed child plan (with the same `prefix_filter` as the expert,
    when Lever A is on); a failed completion drops only that record.
    Returns `(records, group_sizes, n_sibling_fail)`.
    """
    records, group_sizes, n_fail = [], [], 0
    for depth, (decision, chosen) in enumerate(chain[:cfg.max_decisions_per_instance]):
        fixed_g = decision["fixed_g"]
        labeled = []
        for j, (rec, cp) in enumerate(zip(decision["recs"], decision["cps"])):
            if j == chosen:
                labeled.append((rec, win_cost - fixed_g))
                continue
            done = nn_astar_from(env, state, solver, policy, value, env_id, dev,
                                 start_plan=cp, k=cfg.k_top, max_iters=cfg.sibling_iters,
                                 prefix_filter=prefix_filter)
            if done is None:
                n_fail += 1
                continue
            labeled.append((rec, int(done.cost()) - fixed_g))
        if not labeled:
            continue
        best = min(c for _, c in labeled)
        for rec, ctg in labeled:
            out = dict(rec)
            out["cost_to_go"] = int(ctg)
            out["is_optimal"] = bool(ctg == best)
            out["depth"] = int(depth)
            records.append(out)
        group_sizes.append(len(labeled))
    return records, group_sizes, n_fail


def play_instance(env, state, solver, policy, value, env_id, dev, cfg, rng):
    """Expert search + sibling labeling for one instance.

    Returns `(records, win_cost, group_sizes, n_sibling_fail, extra)`. `win_cost is
    None` means the instance produced no training data — either no plan was found or
    the strict filter dropped it; `extra` disambiguates and carries the strict
    realization of the winner:
        extra = {"plan_found": bool, "stx": int | None, "abstract_cost": int | None}
    `stx` (strict moves) is computed for EVERY found winner so the loop always sees
    executability, filter on or off. A plan solved purely by exact-path pinning has
    an empty chain and yields 0 records but still counts as solved.

    `cfg.prefix_check` (Lever A) prunes doomed partial plans inside the expert and
    sibling searches AND enables the complete-pop strict check, so every winner this
    returns with the flag on is strictly playable by construction.
    """
    from eval.realize import strict_moves, prefix_playable, prefix_key

    complete_check = cfg.gen_realize_check or cfg.prefix_check
    check_cache = {}
    realize_check = None
    if complete_check:
        def realize_check(p, _cache=check_cache):
            m = strict_moves(env, state, p, log=None)
            _cache[id(p)] = m
            return m is not None
    prefix_filter = None
    if cfg.prefix_check:
        pfx_cache = {}                                    # per-instance memo
        def prefix_filter(p, _cache=pfx_cache):
            key = prefix_key(p)
            hit = _cache.get(key)
            if hit is None:
                hit = _cache[key] = prefix_playable(env, state, p)
            return hit
    win_plan, chain, n_rejected = nn_astar_traced(
        env, state, solver, policy, value, env_id, dev, cfg, rng,
        realize_check=realize_check, prefix_filter=prefix_filter)
    if win_plan is None:
        return [], None, [], 0, {"plan_found": False, "stx": None,
                                 "abstract_cost": None, "n_anytime_rejected": n_rejected}
    stx = (check_cache.get(id(win_plan)) if complete_check
           else strict_moves(env, state, win_plan, log=None))
    extra = {"plan_found": True, "stx": stx, "abstract_cost": int(win_plan.cost()),
             "n_anytime_rejected": n_rejected}
    if cfg.strict_filter and stx is None:
        # filter arm: only train on plans that execute legally under full physics
        return [], None, [], 0, extra
    win_cost = int(win_plan.cost())
    records, group_sizes, n_fail = label_chain(
        env, state, solver, policy, value, env_id, dev, win_cost, chain, cfg,
        prefix_filter=prefix_filter)
    return records, win_cost, group_sizes, n_fail, extra


# -- (c) one generation pass ------------------------------------------------------

def generate_iteration(cfg, policy, value, rng, boards=None):
    """Sample `cfg.instances_per_iter` instances over the train boards, solve each
    with the traced expert, sibling-score the winning chains -> `(records, stats)`."""
    from subgoal_selfplay.start_states import sample_instance

    boards = list(boards) if boards is not None else cfg.train_ids()
    dev = cfg.device
    policy = policy.to(dev).eval()
    value = value.to(dev).eval()
    solver = AStar(max_iters=cfg.solver_max_iters, max_frontier=cfg.solver_max_frontier)

    # Draw the board multiset up front, then iterate per board: one env load each.
    draw = Counter(rng.choice(boards) for _ in range(cfg.instances_per_iter))
    records, group_sizes, plan_costs = [], [], []
    strict_costs, strict_minus_abstract = [], []
    n_inst = n_plan_found = n_kept = n_strict_pass = n_sib_fail = n_anytime_rej = 0
    with torch.no_grad():
        for gid in sorted(draw):
            env, s0 = GridEnv.from_env(gid)
            colors = [s0.target_robot.color] + [h.color for h in s0.helpers]
            for _ in range(draw[gid]):
                st = sample_instance(env, colors, rng, cfg)
                if st is None:
                    continue
                n_inst += 1
                recs, win_cost, gs, nf, extra = play_instance(
                    env, st, solver, policy, value, gid, dev, cfg, rng)
                n_anytime_rej += extra.get("n_anytime_rejected", 0)
                if extra["plan_found"]:
                    n_plan_found += 1
                    if extra["stx"] is not None:
                        n_strict_pass += 1
                        strict_costs.append(extra["stx"])
                        strict_minus_abstract.append(extra["stx"] - extra["abstract_cost"])
                if win_cost is None:
                    continue                          # no plan, or filtered out
                n_kept += 1
                plan_costs.append(win_cost)
                records.extend(recs)
                group_sizes.extend(gs)
                n_sib_fail += nf
    stats = {
        "n_instances": n_inst,
        "n_plan_found": n_plan_found,
        "solve_rate": round(n_plan_found / n_inst, 3) if n_inst else 0.0,
        "n_kept": n_kept,
        "kept_rate": round(n_kept / n_inst, 3) if n_inst else 0.0,
        "strict_pass": n_strict_pass,
        "strict_pass_rate": round(n_strict_pass / n_plan_found, 3) if n_plan_found else 0.0,
        "mean_strict_cost": round(sum(strict_costs) / len(strict_costs), 2) if strict_costs else None,
        "mean_strict_minus_abstract": round(sum(strict_minus_abstract) / len(strict_minus_abstract), 2) if strict_minus_abstract else None,
        "mean_plan_cost": round(sum(plan_costs) / len(plan_costs), 2) if plan_costs else 0.0,
        "n_records": len(records),
        "n_groups": len(group_sizes),
        "mean_records_per_instance": round(len(records) / n_kept, 2) if n_kept else 0.0,
        "mean_group_size": round(sum(group_sizes) / len(group_sizes), 2) if group_sizes else 0.0,
        "n_sibling_fail": n_sib_fail,
        "n_anytime_rejected": n_anytime_rej,
    }
    return records, stats
