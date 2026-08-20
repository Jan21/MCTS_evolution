"""Gumbel root exploration with sequential halving (Danihelka et al., "Policy
improvement by planning with Gumbel", ICLR 2022) -- generation-side only.

Control generation explores the root by Dirichlet-noised PUCT: visits crowd
onto the prior's favourite, and the policy target (certified costs of whatever
got explored) inherits that bias. Gumbel-AlphaZero instead samples the top-m
root actions WITHOUT replacement via the Gumbel-top-k trick (g_a + log p_a),
spends the budget on them in sequential-halving rounds (equal visits within a
round, halve the candidate set between rounds, ranking by g_a + log p_a +
sigma(q_a)), which guarantees a policy improvement in expectation at ANY budget
-- exactly the regime of our 300-expansion generation searches.

Adaptation to the subgoal tree (this file ships a modified copy of
spr.search.mcts, applied by monkeypatch in the generation workers only; the
arena bench is untouched):
  * root children = all candidates (root_all, as control); m0 = min(16, #cands)
    kept by g + log p; the rest stay in the tree for label extraction but are
    never selected;
  * within a round the allowed root child with the fewest visits is selected
    (uniform allocation); below the root, standard PUCT;
  * between rounds (equal slices of the expansion budget) the allowed set is
    halved by g + log p + (c_visit + maxN) * c_scale * normQ  (c_visit=50,
    c_scale=0.5, normQ in [0,1] higher=better);
  * root Dirichlet noise is OFF (the Gumbel sample IS the exploration).
Everything else (certification, park repairs, dead/closed bookkeeping,
stop-after-certified) is byte-identical to spr.search.mcts.
"""
import math
import random

import numpy as np

from variants import Variant

GUMBEL_M = 16
C_VISIT, C_SCALE = 50.0, 0.5

VARIANT = Variant(
    vid="v06_gumbel_root",
    axis="search",
    title="Gumbel root + sequential halving (generation)",
    hypothesis="Gumbel-top-m root exploration with sequential halving produces "
               "better-ranked root labels per expansion than Dirichlet-PUCT, "
               "improving the retrained policy's top-k on all exams.",
    mechanism="Worker-side hook replaces spr.search.mcts with a modified copy "
              "(Gumbel root, halving rounds, no Dirichlet). Bench protocol "
              "unchanged.",
    expected_failure="At 300 expansions with ~10-30 root candidates the "
                     "halving schedule starves deep subtrees -> fewer "
                     "certified completions, thinner labels, flat benches.",
    hooks=("selfplay",),
)


def mcts_gumbel(env, state, solver, ev, env_id, n, k, max_expansions,
                prefix_filter=None, best_at_budget=True, c_puct=1.5,
                backup="min", acct=None, dump_moves=False, root_noise=0.0,
                noise_alpha=0.3, rng=None, f_mode="parent",
                stop_after_certified=None, root_k=None, parks=False):
    from skeleton.astar import _initial_plan
    from nn.generate import _fixed_g
    from spr.search import Certifier, Child, Node, SearchResult, expand, forced_fixes

    rng = rng or random.Random(0)
    cert = Certifier(env, state, acct, dump_moves, parks=parks)
    root = Node(forced_fixes(env, state, solver, _initial_plan(env, state), acct))
    root.fixed_g = float(_fixed_g(root.plan))
    expansions = rejected = pruned = 0
    best = None
    first_failed = None
    first_cert_exp = None
    last_improve_exp = 0
    qmin, qmax = math.inf, -math.inf
    n_certified = 0
    gumbel = {}          # id(root child Node) -> g + log p
    allowed = None       # current sequential-halving candidate set (root)
    rounds = []          # expansion counts at which to halve

    def _norm(q):
        if qmax <= qmin:
            return 0.5
        return min(1.0, max(0.0, (qmax - q) / (qmax - qmin)))

    def _refresh(node):
        if not node.expanded:
            return
        live = [c for c in node.children if not c.dead]
        if not live:
            node.dead = node.closed = True
            return
        if backup == "min":
            node.Q = min(c.Q for c in live)
        else:
            node.Q = sum(c.Q * max(c.N, 1) for c in live) / sum(max(c.N, 1) for c in live)
        if all(c.closed for c in live):
            node.closed = True

    def _backup(path):
        nonlocal qmin, qmax
        for node in reversed(path):
            node.N += 1
            _refresh(node)
            if node.Q is not None and not node.dead:
                qmin, qmax = min(qmin, node.Q), max(qmax, node.Q)

    def _select_puct(node):
        sq = math.sqrt(node.N + 1)
        best_c, best_u = None, -math.inf
        for c in node.children:
            if c.closed:
                continue
            u = _norm(c.Q) + c_puct * c.prior * sq / (1 + c.N)
            if u > best_u:
                best_c, best_u = c, u
        return best_c

    def _sigma_score(c):
        max_n = max((x.N for x in allowed), default=0)
        return gumbel[id(c)] + (C_VISIT + max_n) * C_SCALE * _norm(c.Q)

    def _select_root():
        nonlocal allowed
        live = [c for c in allowed if not c.closed]
        if not live:                       # halved set exhausted -> widen once
            live = [c for c in root.children if not c.closed and id(c) in gumbel]
            allowed = live
            if not live:
                return None
        return min(live, key=lambda c: c.N)

    def _maybe_halve():
        nonlocal allowed
        while rounds and expansions >= rounds[0] and allowed and len(allowed) > 1:
            rounds.pop(0)
            keep = max(1, len(allowed) // 2)
            allowed = sorted(allowed, key=_sigma_score, reverse=True)[:keep]

    while expansions < max_expansions and not root.closed:
        if (best is not None and stop_after_certified is not None
                and expansions - last_improve_exp >= stop_after_certified):
            break
        if allowed is not None:
            _maybe_halve()
        node, path = root, [root]
        while node.expanded and not node.complete and not node.closed:
            nxt = _select_root() if node is root and allowed is not None else _select_puct(node)
            if nxt is None:
                node.closed = True
                break
            node = nxt
            path.append(node)
        if node.closed:
            _backup(path)
            continue
        if node.complete:
            m, mv = cert(node.plan)
            if m is None:
                node.cert = False
                rejected += 1
                if first_failed is None:
                    first_failed = node.plan
                rps = cert.repairs(node.plan) if not node.expanded else []
                if rps:
                    node.expanded = True
                    node.complete = False
                    for rp in rps:
                        c = Child(plan=rp, rec=None, key=None, prior=1.0 / len(rps),
                                  ctg_hat=0.0, fixed_g=float(rp.cost()), v_est=float(rp.cost()))
                        ch = Node(rp, parent=node, child_obj=c, depth=node.depth + 1)
                        ch.fixed_g = float(rp.cost())
                        node.children.append(ch)
                        qmin, qmax = min(qmin, ch.Q), max(qmax, ch.Q)
                else:
                    node.dead = node.closed = True
            else:
                node.cert = m
                node.best_abs = float(node.plan.cost())
                node.Q = node.best_abs
                node.solved = node.closed = True
                node.best_cert = m
                n_certified += 1
                if best is None or m < best[0]:
                    best = (m, node, mv)
                    last_improve_exp = expansions
                    if first_cert_exp is None:
                        first_cert_exp = expansions
                for a in reversed(path[:-1]):
                    if (a.best_cert is None or m < a.best_cert
                            or (m == a.best_cert and node.best_abs < a.best_abs)):
                        a.best_cert, a.best_abs = m, node.best_abs
            _backup(path)
            if best is not None and not best_at_budget:
                break
            if (best is not None and stop_after_certified is not None
                    and expansions - last_improve_exp >= stop_after_certified):
                break
            continue
        expansions += 1
        kk = k
        if node is root and root_k is not None:
            kk = root_k if root_k > 0 else 10 ** 6
        kids, pr = expand(env, state, solver, node.plan, env_id, n, ev, kk, prefix_filter,
                          acct, f_mode)
        pruned += pr
        node.expanded = True
        if not kids:
            node.dead = node.closed = True
            _backup(path)
            continue
        for c in kids:
            cp = forced_fixes(env, state, solver, c.plan, acct)
            ch = Node(cp, parent=node, child_obj=c, depth=node.depth + 1)
            ch.fixed_g = float(_fixed_g(cp))
            node.children.append(ch)
            qmin, qmax = min(qmin, ch.Q), max(qmax, ch.Q)
        if node is root:
            # Gumbel-top-m over ALL root candidates; sequential-halving schedule
            g_rng = np.random.default_rng(rng.randrange(1 << 30))
            for ch in node.children:
                p = max(float(ch.child_obj.prior), 1e-12)
                gumbel[id(ch)] = float(g_rng.gumbel()) + math.log(p)
            m0 = min(GUMBEL_M, len(node.children))
            allowed = sorted(node.children, key=lambda c: gumbel[id(c)],
                             reverse=True)[:m0]
            n_rounds = max(1, math.ceil(math.log2(m0))) if m0 > 1 else 1
            budget_left = max_expansions - expansions
            step = max(1, budget_left // (n_rounds + 1))
            rounds[:] = [expansions + step * (i + 1) for i in range(n_rounds)]
        _backup(path)

    res = SearchResult(expansions=expansions, rejected=rejected, pruned=pruned, root=root)
    if best is not None:
        m, bnode, mv = best
        res.plan, res.strict, res.moves = bnode.plan, m, mv
        res.extra = {"first_certified_expansion": first_cert_exp, "best_strict": m,
                     "root_N": root.N, "best_depth": bnode.depth,
                     "n_certified": n_certified, "root_closed": root.closed,
                     "gumbel_m": len(gumbel) and min(GUMBEL_M, len(gumbel))}
    else:
        res.plan = first_failed
        res.extra = {"root_N": root.N, "n_certified": 0, "root_closed": root.closed}
    return res


def apply_selfplay():
    import spr.search as S
    S.mcts = mcts_gumbel
