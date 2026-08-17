"""Search over subgoal decisions with the size-free nets: A* (own loop) and
MCTS (PUCT), sharing one expansion primitive and one certification contract.

Vocabulary (PROBLEM.md section 2/5): a node is a partial plan; a decision
resolves the plan's first open edge by choosing one candidate subgoal
(`heuristics.propose` + `skeleton.astar._apply`); a complete plan is a
terminal whose ground-truth value is its STRICT realized move count
(`eval.realize.strict_moves`, the arena's certifier) -- or "dead" when it
cannot be played. A backup value from an unrealized plan is a hypothesis;
only certified plans set the record (section 4.6).

Budget unit (section 6.5, = the arena's): one EXPANSION = one node whose
children are generated = one policy pass + one batched value pass over <= k
children. Forced exact fixes and physics (realization, prefix checks) are
free, exactly as in `eval/compare.py::_nn_astar_backward`.

    A*    f = fixed_g(child) + ctg_hat   ("child", the arena's f) or
          f = fixed_g(parent) + ctg_hat  ("parent": ctg labels are defined
          relative to the parent's fixed cost, nn/generate.py:93-104, so this
          is the label-consistent total-cost estimate); best-first; optional
          best-at-budget (anytime: keep popping while a cheaper abstract cost
          remains and return the cheapest CERTIFIED plan).
    MCTS  PUCT over subgoal decisions: children of an expanded node = the top-k
          candidates by policy prior, each born with its own value estimate
          v = fixed_g(parent) + ctg_hat (no first-play-urgency hack needed);
          Q normalized min-max over the tree (MuZero style) since values are
          costs; backup = min (default; deterministic single-agent) or mean;
          certified terminals are exact leaves; dead terminals are pruned and
          their parents re-evaluated; Dirichlet root noise + visit-count
          temperature available for self-play generation.
"""
from __future__ import annotations

import heapq
import itertools
import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch

from spr.nets import policy_features, policy_meta, flat


# ---------------------------------------------------------------------------
# records + evaluator
# ---------------------------------------------------------------------------

def _xy(p):
    return [int(p[0]), int(p[1])] if p is not None else None


def cand_record(env_id, state, seg, ctx, cand):
    """The frozen 18-field record layout minus labels (nn/generate.py:113-127)."""
    bn_ctx, sp_ctx, open_eps = ctx
    return {
        "env_id": env_id,
        "target": _xy(state.target),
        "target_robot": [_xy(state.target_robot.position), state.target_robot.color],
        "helpers": [[_xy(h.position), h.color] for h in state.helpers],
        "seg_start": _xy(seg.start), "seg_end": _xy(seg.end),
        "seg_support": _xy(seg.fix_support), "mover_color": seg.mover.color,
        "ctx_bottlenecks": bn_ctx, "ctx_supports": sp_ctx,
        "ctx_open_endpoints": open_eps,
        "cand_bottleneck": _xy(cand.subgoal.bottleneck.position),
        "cand_support": _xy(cand.subgoal.support.position),
        "cand_helper": [_xy(cand.subgoal.helper.position), cand.subgoal.helper.color],
        "cand_parent_support": _xy(cand.parent_support),
    }


def hidx(state, helper_pos):
    """Helper slot by START position (eval/end2end.py::_hidx, byref off)."""
    for i, h in enumerate(state.helpers):
        if (int(h.position[0]), int(h.position[1])) == tuple(helper_pos):
            return i
    return None


def adjacency_from_graph(G, n):
    """Dense A_all / A_ind masks from a slide graph (== nn_labeler.encode.adjacency
    on the pkl's grid_graph; built from the live env so lean/in-memory boards
    need no pkl read)."""
    size = n * n
    A_all = np.zeros((size + 1, size + 1), np.float32)
    A_ind = np.zeros((size + 1, size + 1), np.float32)
    for u, v, d in G.edges(data=True):
        s, t = u[1] * n + u[0], v[1] * n + v[0]
        A_all[t, s] = 1.0
        if "dependent" not in d:
            A_ind[t, s] = 1.0
    idx = np.arange(size + 1)
    A_all[idx, idx] = 1.0
    A_ind[idx, idx] = 1.0
    return A_all, A_ind


class Evaluator:
    """Policy prior + value cost-to-go for candidate record groups. Holds the
    per-board adjacency cache. `acct` counters mirror the arena's."""

    def __init__(self, policy, value, device="cpu"):
        from nn_labeler import encode as _enc
        self.policy, self.value, self.dev = policy, value, device
        self._adj = {}
        self._enc = _enc
        self.acct = {"nn_policy_calls": 0, "nn_value_calls": 0}

    def adjacency(self, env, env_id, n):
        key = (id(env), env_id, n)
        hit = self._adj.get(key)
        if hit is None:
            A_all, A_ind = adjacency_from_graph(env.G, n)
            hit = (torch.as_tensor(A_all).to(self.dev), torch.as_tensor(A_ind).to(self.dev))
            if len(self._adj) > 8:
                self._adj.clear()
            self._adj[key] = hit
        return hit

    @torch.no_grad()
    def policy_logp(self, recs, keys, env, env_id, n):
        """{key: logp} for the group's candidates (keys = (bn_flat, sp_flat, hslot))."""
        m = policy_meta(recs, n)
        if m is None:
            return None
        A_all, A_ind = self.adjacency(env, env_id, n)
        x = torch.from_numpy(policy_features(recs[0], n)).unsqueeze(0).to(self.dev)
        h = self.policy._encode(x, A_all.unsqueeze(0), A_ind.unsqueeze(0))[0]
        self.acct["nn_policy_calls"] += 1
        return self.policy.logp_map(h, m)

    @torch.no_grad()
    def value_ctg(self, recs, env, env_id, n):
        A_all, A_ind = self.adjacency(env, env_id, n)
        x = torch.from_numpy(np.stack([self._enc.node_features(r, n) for r in recs])).to(self.dev)
        key = torch.tensor([self._enc.key_indices(r, n) for r in recs], device=self.dev)
        R = len(recs)
        logits = self.value(x, A_all.unsqueeze(0).expand(R, -1, -1),
                            A_ind.unsqueeze(0).expand(R, -1, -1), n, key)
        self.acct["nn_value_calls"] += 1
        return [float(v) for v in self.value._value(logits)]


# ---------------------------------------------------------------------------
# expansion primitive
# ---------------------------------------------------------------------------

@dataclass
class Child:
    plan: object
    rec: dict
    key: tuple
    logp: float = -1e9
    prior: float = 0.0
    ctg_hat: float | None = None
    fixed_g: float = 0.0
    v_est: float | None = None      # estimated total plan cost (see f_mode)


def forced_fixes(env, state, solver, plan, acct=None):
    """Apply free exact fixes to the first open edge while possible
    (eval/compare.py:221-230). Returns the (possibly complete) plan."""
    from skeleton.astar import _segment
    while not plan.is_complete():
        parent, child = plan.open_edges()[0]
        seg = _segment(plan, state, parent, child)
        if env.compute_exact_shortest_path_length(seg.start, seg.end, seg.fix_support) is not None:
            if acct is not None:
                acct["free_exact_fix_expands"] = acct.get("free_exact_fix_expands", 0) + 1
            plan = solver._expand(env, state, plan)[0]
        else:
            break
    return plan


def expand(env, state, solver, plan, env_id, n, ev: Evaluator, k, prefix_filter=None,
           acct=None, f_mode="parent", score_all=False):
    """Generate the top-k children of an (incomplete) plan.

    One expansion: propose -> apply -> policy pass (prior over the valid
    candidates) -> top-k by prior -> value pass -> children carrying
    ctg_hat and their own fixed_g. `pruned` counts prefix-filter drops (Lever
    A, physics, free). Returns (children, pruned) -- children may be empty."""
    from skeleton.astar import _segment, _apply
    from nn.generate import _context, _fixed_g
    parent, child = plan.open_edges()[0]
    seg = _segment(plan, state, parent, child)
    cands = solver.propose(env, seg.end, seg.mover, seg.helpers, seg.support)
    ctx = _context(plan)
    kids = []
    for cand in cands:
        cp = _apply(env, plan, parent, child, seg, cand)
        if cp is None:
            continue
        r = cand_record(env_id, state, seg, ctx, cand)
        hi = hidx(state, r["cand_helper"][0])
        if hi is None:
            continue
        kids.append(Child(plan=cp, rec=r,
                          key=(flat(r["cand_bottleneck"], n), flat(r["cand_support"], n), hi)))
    if not kids:
        return [], 0
    logp = ev.policy_logp([c.rec for c in kids], [c.key for c in kids], env, env_id, n)
    if logp is None:
        return [], 0
    for c in kids:
        c.logp = logp.get(c.key, -1e9)
    mx = max(c.logp for c in kids)
    z = sum(math.exp(c.logp - mx) for c in kids)
    for c in kids:
        c.prior = math.exp(c.logp - mx) / z
    kids.sort(key=lambda c: -c.logp)
    top = kids if score_all else kids[:k]
    ctgs = ev.value_ctg([c.rec for c in top], env, env_id, n)
    g_parent = float(_fixed_g(plan))
    out, pruned = [], 0
    for c, h in zip(top, ctgs):
        c.ctg_hat = float(h)
        c.fixed_g = float(_fixed_g(c.plan))
        if prefix_filter is not None and not prefix_filter(c.plan):
            pruned += 1
            continue
        out.append(c)
    # value estimate of the child's whole plan (cost units): parent-consistent
    for c in out:
        c.v_est = (g_parent + c.ctg_hat) if f_mode == "parent" else (c.fixed_g + c.ctg_hat)
    return out, pruned


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    plan: object = None            # best certified plan (or first failed one)
    expansions: int = 0
    rejected: int = 0
    pruned: int = 0
    strict: int | None = None      # certified strict moves of `plan` (None if uncertified)
    moves: list | None = None      # realized [color, dir] sequence (dump)
    extra: dict = field(default_factory=dict)
    root: object = None            # MCTS tree root (for record extraction)


class Certifier:
    """strict_moves with memo + move dump + counters (the arena's realize_check)."""

    def __init__(self, env, state, acct=None, dump=False):
        from eval.realize import strict_moves
        self._sm, self.env, self.state = strict_moves, env, state
        self.acct, self.dump = acct, dump
        self.cache = {}

    def __call__(self, plan):
        hit = self.cache.get(id(plan))
        if hit is not None:
            return hit
        mv = [] if self.dump else None
        if self.acct is not None:
            self.acct["physics_calls_strict_realize"] = \
                self.acct.get("physics_calls_strict_realize", 0) + 1
        m = self._sm(self.env, self.state, plan, log=None, moves_out=mv)
        self.cache[id(plan)] = (m, mv if m is not None else None)
        return self.cache[id(plan)]


# ---------------------------------------------------------------------------
# A* (own loop)
# ---------------------------------------------------------------------------

def astar(env, state, solver, ev, env_id, n, k, max_expansions, prefix_filter=None,
          best_at_budget=False, f_mode="parent", acct=None, dump_moves=False):
    from skeleton.astar import _initial_plan
    cert = Certifier(env, state, acct, dump_moves)
    cnt = itertools.count()
    frontier = [(0.0, next(cnt), forced_fixes(env, state, solver, _initial_plan(env, state), acct))]
    expansions = rejected = pruned = 0
    best = None                     # (strict, plan, moves)
    first_failed = None
    first_cert_exp = None
    while frontier and expansions < max_expansions:
        f, _, plan = heapq.heappop(frontier)
        if best is not None and best_at_budget and f >= best[0]:
            break                   # nothing estimated cheaper than the certified best
        if plan.is_complete():
            m, mv = cert(plan)
            if m is None:
                rejected += 1
                if first_failed is None:
                    first_failed = plan
                continue
            if best is None or m < best[0]:
                best = (m, plan, mv)
                if first_cert_exp is None:
                    first_cert_exp = expansions
            if not best_at_budget:
                break
            continue
        expansions += 1
        kids, pr = expand(env, state, solver, plan, env_id, n, ev, k, prefix_filter,
                          acct, f_mode)
        pruned += pr
        for c in kids:
            cp = forced_fixes(env, state, solver, c.plan, acct)
            heapq.heappush(frontier, (float(c.v_est), next(cnt), cp))
    res = SearchResult(expansions=expansions, rejected=rejected, pruned=pruned)
    if best is not None:
        res.plan, res.strict, res.moves = best[1], best[0], best[2]
        res.extra = {"first_certified_expansion": first_cert_exp, "best_strict": best[0]}
    else:
        res.plan = first_failed
    return res


# ---------------------------------------------------------------------------
# MCTS (PUCT)
# ---------------------------------------------------------------------------

class Node:
    __slots__ = ("plan", "parent", "child_obj", "children", "N", "Q", "v_est", "prior",
                 "complete", "cert", "dead", "solved", "closed", "depth", "fixed_g",
                 "expanded", "best_cert")

    def __init__(self, plan, parent=None, child_obj=None, depth=0):
        self.plan, self.parent, self.child_obj = plan, parent, child_obj
        self.children = []
        self.N = 0
        self.Q = child_obj.v_est if child_obj is not None else None   # cost estimate
        self.v_est = self.Q
        self.prior = child_obj.prior if child_obj is not None else 1.0
        self.complete = plan.is_complete()
        self.cert = None            # None untested / int strict moves / False uncertifiable
        self.dead = False           # uncertifiable terminal or no live children
        self.solved = False         # certified terminal (exact leaf; kept in Q, not selected)
        self.closed = False         # nothing left to select below (dead, solved, or all
                                    # children closed); kept in Q unless dead
        self.depth = depth
        self.fixed_g = None
        self.expanded = False
        self.best_cert = None       # min certified strict cost in this subtree


def mcts(env, state, solver, ev, env_id, n, k, max_expansions, prefix_filter=None,
         best_at_budget=True, c_puct=1.5, backup="min", acct=None, dump_moves=False,
         root_noise=0.0, noise_alpha=0.3, rng=None, f_mode="parent",
         stop_after_certified=None):
    """PUCT tree search over subgoal decisions. Returns SearchResult with .root.

    Every loop iteration makes progress: it expands a leaf (budget), certifies
    a complete plan (free physics; the node becomes solved or dead), or closes
    a node whose subtree is exhausted -- so the loop terminates at the budget
    or when the root closes (the k-restricted tree is fully explored).

    `stop_after_certified` (int|None): in best-at-budget mode, stop once this
    many expansions have passed since the best certified plan last improved
    (self-play generation knob; the arena gate uses None = run to budget)."""
    from skeleton.astar import _initial_plan
    from nn.generate import _fixed_g
    rng = rng or random.Random(0)
    cert = Certifier(env, state, acct, dump_moves)
    root = Node(forced_fixes(env, state, solver, _initial_plan(env, state), acct))
    root.fixed_g = float(_fixed_g(root.plan))
    expansions = rejected = pruned = 0
    best = None                     # (strict, node, moves)
    first_failed = None
    first_cert_exp = None
    last_improve_exp = 0
    qmin, qmax = math.inf, -math.inf
    n_certified = 0

    def _norm(q):
        if qmax <= qmin:
            return 0.5
        return (qmax - q) / (qmax - qmin)          # lower cost -> closer to 1

    def _refresh(node):
        """Q from live (non-dead) children; dead when no live child; closed
        when no selectable (non-closed) child remains."""
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

    def _select(node):
        sq = math.sqrt(node.N + 1)
        best_c, best_u = None, -math.inf
        for c in node.children:
            if c.closed:
                continue
            u = _norm(c.Q) + c_puct * c.prior * sq / (1 + c.N)
            if u > best_u:
                best_c, best_u = c, u
        return best_c

    while expansions < max_expansions and not root.closed:
        # ---- select down to a selectable leaf / terminal
        node, path = root, [root]
        while node.expanded and not node.complete and not node.closed:
            nxt = _select(node)
            if nxt is None:
                node.closed = True
                break
            node = nxt
            path.append(node)
        if node.closed:
            _backup(path)
            continue
        # ---- terminal: certify once
        if node.complete:
            m, mv = cert(node.plan)
            if m is None:
                node.cert = False
                node.dead = node.closed = True
                rejected += 1
                if first_failed is None:
                    first_failed = node.plan
            else:
                node.cert = m
                node.Q = float(m)
                node.solved = node.closed = True
                node.best_cert = m
                n_certified += 1
                if best is None or m < best[0]:
                    best = (m, node, mv)
                    last_improve_exp = expansions
                    if first_cert_exp is None:
                        first_cert_exp = expansions
                for a in reversed(path[:-1]):
                    a.best_cert = m if a.best_cert is None else min(a.best_cert, m)
            _backup(path)
            if best is not None and not best_at_budget:
                break
            if (best is not None and stop_after_certified is not None
                    and expansions - last_improve_exp >= stop_after_certified):
                break
            continue
        # ---- expand leaf
        expansions += 1
        kids, pr = expand(env, state, solver, node.plan, env_id, n, ev, k, prefix_filter,
                          acct, f_mode)
        pruned += pr
        node.expanded = True
        if not kids:
            node.dead = node.closed = True
            _backup(path)
            continue
        if node is root and root_noise > 0:
            noise = np.random.default_rng(rng.randrange(1 << 30)).dirichlet(
                [noise_alpha] * len(kids))
            for c, e in zip(kids, noise):
                c.prior = (1 - root_noise) * c.prior + root_noise * float(e)
        for c in kids:
            cp = forced_fixes(env, state, solver, c.plan, acct)
            ch = Node(cp, parent=node, child_obj=c, depth=node.depth + 1)
            ch.fixed_g = float(_fixed_g(cp))
            node.children.append(ch)
        _backup(path)

    res = SearchResult(expansions=expansions, rejected=rejected, pruned=pruned, root=root)
    if best is not None:
        m, bnode, mv = best
        res.plan, res.strict, res.moves = bnode.plan, m, mv
        res.extra = {"first_certified_expansion": first_cert_exp, "best_strict": m,
                     "root_N": root.N, "best_depth": bnode.depth,
                     "n_certified": n_certified, "root_closed": root.closed}
    else:
        res.plan = first_failed
        res.extra = {"root_N": root.N, "n_certified": 0, "root_closed": root.closed}
    return res


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

def run(name, env, state, solver, policy, value_net, env_id, dev, k, max_expansions,
        prefix_filter=None, best_at_budget=False, f_mode="parent", c_puct=1.5,
        backup="min", acct=None, dump_moves=False, **kw):
    from simulate import _board_size
    n = _board_size(env.grid_data, None)
    ev = Evaluator(policy, value_net, dev)
    if name == "spr_astar":
        res = astar(env, state, solver, ev, env_id, n, k, max_expansions, prefix_filter,
                    best_at_budget, f_mode, acct, dump_moves)
    elif name == "mcts":
        res = mcts(env, state, solver, ev, env_id, n, k, max_expansions, prefix_filter,
                   best_at_budget, c_puct, backup, acct, dump_moves, f_mode=f_mode, **kw)
    else:
        raise ValueError(name)
    if acct is not None:
        acct["nn_policy_calls"] = ev.acct["nn_policy_calls"]
        acct["nn_value_calls"] = ev.acct["nn_value_calls"]
    return res
