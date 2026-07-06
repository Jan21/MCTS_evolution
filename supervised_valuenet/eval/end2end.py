"""End-to-end system test: build WHOLE plans with the two transformer nets, vs optimal.

At each decision: enumerate valid candidates, PROPOSE = policy_tf ranks them (AR logprob)
-> keep top-k, SCORE = value net (masked looped) picks the lowest predicted cost-to-go,
commit, advance. Greedy (beam=1 over the policy's top-k). Measure achieved plan cost vs
the optimal (solve_plan from the root). Baseline: the hardcoded heuristic propose+score
greedy. The payoff question: does learned propose+score beat the heuristics on whole
puzzles, approaching optimal?

    python -m eval.end2end --boards 112-127,2400-2599 --per-board 4 --policy <ckpt> --value <ckpt>
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import random

import torch

from GridEnv import GridEnv
from skeleton.astar import AStar, _initial_plan, _segment, _apply
from nn.generate import random_instance, _context, _fixed_g, parse_graphs
from nn.collect_search import _record
from train.policy_common import _meta, _ix
from train.policy_tf import _adj, _x257 as _pol_x, PolicyTF
from train.looped_pc import _x257 as _val_x, LoopedValueNet

torch.set_grad_enabled(False)


def _hidx(state, helper_pos):
    for i, h in enumerate(state.helpers):
        if (int(h.position[0]), int(h.position[1])) == tuple(helper_pos):
            return i
    return None


def _policy_logp(policy, group, dev):
    """{(bn,sp,helper_idx): AR logprob} for the decision's candidates."""
    m = _meta(group)
    if m is None:
        return None, None
    x = _pol_x(group[0]).unsqueeze(0).to(dev)
    A_all, A_ind = _adj(group[0]["env_id"])
    h = policy._encode(x, A_all.unsqueeze(0).to(dev), A_ind.unsqueeze(0).to(dev))[0]
    g = policy.seg(torch.cat([h[m["seg_start"]], h[m["seg_end"]], h.mean(0)]))
    vbn = m["valid_bn"]
    bn_lp = torch.log_softmax(h[torch.tensor(vbn, device=dev)] @ policy.q_bn(g), 0)
    out = {}
    for bn in vbn:
        sl = m["sup_by_bn"][bn]
        sup_lp = torch.log_softmax(h[torch.tensor(sl, device=dev)] @ policy.q_sup(torch.cat([g, h[bn]])), 0)
        for sp in sl:
            hl = torch.log_softmax(h[torch.tensor(m["helper_cells"], device=dev)]
                                   @ policy.q_help(torch.cat([g, h[bn], h[sp]])), 0)
            for (b2, s2, hi2) in m["ctg_map"]:
                if b2 == bn and s2 == sp:
                    out[(bn, sp, hi2)] = float(bn_lp[vbn.index(bn)] + sup_lp[sl.index(sp)] + hl[hi2])
    return out, m


def _value_cost(value, recs, dev):
    """Predicted cost-to-go per candidate record (masked looped value net)."""
    x = torch.stack([_val_x(r) for r in recs]).to(dev)
    A_all, A_ind = _adj(recs[0]["env_id"])
    B = x.shape[0]
    b = dict(x=x, A_all=A_all.unsqueeze(0).expand(B, -1, -1).to(dev),
             A_ind=A_ind.unsqueeze(0).expand(B, -1, -1).to(dev),
             key=torch.stack([torch.tensor([_ix(r["cand_bottleneck"]), _ix(r["cand_support"]),
                                            _ix(r["cand_helper"][0]), _ix(r["seg_start"]),
                                            _ix(r["seg_end"])]) for r in recs]).to(dev))
    return value._value(value(b)).tolist()


def nn_rollout(env, state, solver, policy, value, env_id, dev, k=5, max_steps=40):
    plan = _initial_plan(env, state)
    steps = 0
    while not plan.is_complete() and steps < max_steps:
        steps += 1
        parent, child = plan.open_edges()[0]
        seg = _segment(plan, state, parent, child)
        if env.compute_exact_shortest_path_length(seg.start, seg.end, seg.fix_support) is not None:
            plan = solver._expand(env, state, plan)[0]
            continue
        cands = solver.propose(env, seg.end, seg.mover, seg.helpers, seg.support)
        ctx = _context(plan)
        recs, cps, keys = [], [], []
        for cand in cands:
            cp = _apply(env, plan, parent, child, seg, cand)
            if cp is None:
                continue
            r = _record(env_id, state, seg, ctx, cand, 0, 0, 0)
            hi = _hidx(state, r["cand_helper"][0])
            if hi is None:
                continue
            recs.append(r); cps.append(cp)
            keys.append((_ix(r["cand_bottleneck"]), _ix(r["cand_support"]), hi))
        if not recs:
            return None
        logp, _ = _policy_logp(policy, recs, dev)
        if logp is None:
            return None
        order = sorted(range(len(recs)), key=lambda i: -logp.get(keys[i], -1e9))[:k]
        costs = _value_cost(value, [recs[i] for i in order], dev)
        plan = cps[order[int(torch.tensor(costs).argmin())]]      # propose top-k -> value pick
    return int(plan.cost()) if plan.is_complete() else None


def nn_astar(env, state, solver, policy, value, env_id, dev, k=5, max_iters=3000):
    """A* over partial plans, f = g + value-net(commit). Best-first, propose top-k per
    node, return the first complete plan popped. Not greedy: explores + backtracks."""
    cnt = itertools.count()
    frontier = [(0.0, next(cnt), _initial_plan(env, state))]
    iters = 0
    while frontier and iters < max_iters:
        iters += 1
        _, _, plan = heapq.heappop(frontier)
        while not plan.is_complete():                         # apply forced exact fixes
            parent, child = plan.open_edges()[0]
            seg = _segment(plan, state, parent, child)
            if env.compute_exact_shortest_path_length(seg.start, seg.end, seg.fix_support) is not None:
                plan = solver._expand(env, state, plan)[0]
            else:
                break
        if plan.is_complete():
            return int(plan.cost())                           # first complete = solution
        parent, child = plan.open_edges()[0]
        seg = _segment(plan, state, parent, child)
        cands = solver.propose(env, seg.end, seg.mover, seg.helpers, seg.support)
        ctx = _context(plan)
        recs, cps = [], []
        for cand in cands:
            cp = _apply(env, plan, parent, child, seg, cand)
            if cp is None or _hidx(state, _record(env_id, state, seg, ctx, cand, 0, 0, 0)["cand_helper"][0]) is None:
                continue
            recs.append(_record(env_id, state, seg, ctx, cand, 0, 0, 0))
            cps.append(cp)
        if not recs:
            continue
        logp, _ = _policy_logp(policy, recs, dev)
        if logp is None:
            continue
        keys = [(_ix(r["cand_bottleneck"]), _ix(r["cand_support"]), _hidx(state, r["cand_helper"][0])) for r in recs]
        order = sorted(range(len(recs)), key=lambda i: -logp.get(keys[i], -1e9))[:k]
        costs = _value_cost(value, [recs[i] for i in order], dev)
        for i, h in zip(order, costs):
            heapq.heappush(frontier, (float(_fixed_g(cps[i])) + h, next(cnt), cps[i]))
    return None


def heur_rollout(env, state, solver, env_id, max_steps=40):
    plan = _initial_plan(env, state)
    steps = 0
    while not plan.is_complete() and steps < max_steps:
        steps += 1
        parent, child = plan.open_edges()[0]
        seg = _segment(plan, state, parent, child)
        if env.compute_exact_shortest_path_length(seg.start, seg.end, seg.fix_support) is not None:
            plan = solver._expand(env, state, plan)[0]
            continue
        cands = solver.propose(env, seg.end, seg.mover, seg.helpers, seg.support)
        best = None
        for cand in sorted(cands, key=lambda c: solver.score(env, c)):
            cp = _apply(env, plan, parent, child, seg, cand)
            if cp is not None:
                best = cp
                break
        if best is None:
            return None
        plan = best
    return int(plan.cost()) if plan.is_complete() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--boards", default="112-127,2400-2599")
    p.add_argument("--per-board", type=int, default=4)
    p.add_argument("--policy", required=True)
    p.add_argument("--value", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--astar-iters", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PolicyTF.load_from_checkpoint(a.policy, map_location=dev).to(dev).eval()
    value = LoopedValueNet.load_from_checkpoint(a.value, map_location=dev).to(dev).eval()
    solver = AStar(max_iters=4000, max_frontier=40_000)
    rng = random.Random(a.seed)

    boards = parse_graphs(a.boards)
    _, s0 = GridEnv.from_env(boards[0])
    colors = [s0.target_robot.color] + [h.color for h in s0.helpers]

    gr_reg, ast_reg, heur_reg, n, opt_fail = [], [], [], 0, 0
    gr_fail, ast_fail = 0, 0
    for gid in boards:
        env, _ = GridEnv.from_env(gid)
        for _ in range(a.per_board):
            st = random_instance(env, colors, rng)
            opt = solver.solve_plan(env, st, _initial_plan(env, st))
            if opt is None:
                opt_fail += 1
                continue
            oc = int(opt.cost())
            gc = nn_rollout(env, st, solver, policy, value, gid, dev, a.k)     # greedy
            ac = nn_astar(env, st, solver, policy, value, gid, dev, a.k, a.astar_iters)  # A*
            hc = heur_rollout(env, st, solver, gid)
            n += 1
            gr_reg.append(gc - oc) if gc is not None else (gr_fail := gr_fail + 1)
            ast_reg.append(ac - oc) if ac is not None else (ast_fail := ast_fail + 1)
            if hc is not None:
                heur_reg.append(hc - oc)
        print(f"board {gid}: n={n} greedy={sum(gr_reg)/max(len(gr_reg),1):.3f} "
              f"astar={sum(ast_reg)/max(len(ast_reg),1):.3f} "
              f"heur={sum(heur_reg)/max(len(heur_reg),1):.3f}", flush=True)

    print(f"\n=== END-TO-END ({n} instances, optimal-cost reference) ===")
    print(f"NN A* (value as h):     regret = {sum(ast_reg)/max(len(ast_reg),1):.3f} "
          f"(solved {len(ast_reg)}/{n}, fail {ast_fail})")
    print(f"NN greedy (beam=1):     regret = {sum(gr_reg)/max(len(gr_reg),1):.3f} "
          f"(solved {len(gr_reg)}/{n}, fail {gr_fail})")
    print(f"heuristic greedy:       regret = {sum(heur_reg)/max(len(heur_reg),1):.3f} "
          f"(solved {len(heur_reg)}/{n})")
    print(f"opt_fail={opt_fail}")


if __name__ == "__main__":
    main()
