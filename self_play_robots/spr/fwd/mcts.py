"""PUCT MCTS over PRIMITIVE MOVES, plus the A*/greedy arms it is measured against.

State = `positions` (the hashable robot-cell tuple of `move_planner.state`) for
one instance (`target_idx`, `target`, board walls `wr/wd` fixed for the search).
Actions = the legal `(robot_slot, dir_idx)` moves. The evaluator is a
`move_planner.evaluate.Guide` (MoveNet): one call returns `(values, policy
logits[N,R,4])` for a batch of states.

BUDGET UNIT (PROBLEM.md 6.5, = `eval/compare.py`'s):

    one EXPANSION = one node whose children are generated
                  = 1 policy pass on the node  +  1 batched value pass over its
                    top-k children by policy prior

which is exactly what `move_planner.evaluate.nn_astar` spends per expansion, so
`eval.compare._CountingGuide`'s `expansions = (calls - 1) // 2` holds for this
search too. Both counters are kept: MCTS increments its own `expansions` on the
same line that makes the two calls, and `bench.py` cross-checks it against the
Guide's call counter per instance (`accounting.expansions_from_calls`).

VALUES ARE COSTS. A child born at depth `g+1` under a parent at depth `g` gets

    v_est = (g + 1) + value(child)          [cost units: estimated TOTAL moves]

i.e. `nn_astar`'s `f = ng + value(child)`. Q is min-max normalized over the tree
(MuZero style, lower cost -> 1) before PUCT, since Q is not a bounded return.
Backup is `min` over live children by default (deterministic single-agent: the
value of a node is the value of its best continuation), `mean` optional.

TERMINALS ARE EXACT AND FREE. A child whose positions satisfy `is_goal` is a
terminal with cost exactly `g+1`; there is no realization gap to certify (unlike
the subgoal arm, where a complete plan is only a hypothesis until
`eval.realize.strict_moves` plays it). The path IS the certificate: every edge
is a legal, non-no-op slide by construction of `legal_moves`. We still (a)
replay the winning path through `apply_move` before accepting it
(`verify_path`, zero NN cost -- PROBLEM.md 4.6: nothing counts until replayed)
and (b) dump the `[color, direction]` sequence so `eval/replay_validate.py` can
certify it independently of this module.

TRANSPOSITIONS -- documented choice: a per-search `best_g: positions -> lowest g
at which this position has been attached to the tree`, exactly `nn_astar`'s
`best_g` closed set. A child is dropped when `g_child >= best_g[pos]` (a
dominated revisit: same position, no cheaper). This kills 2-cycles (the parent's
own position is in the table at a lower g) and keeps the tree finite. It is NOT
a full transposition table -- a strictly cheaper later path to a position is
allowed and creates a second node rather than rewiring the first, so the tree
can hold two nodes for one position with different g. That is deliberate: node
statistics stay a tree (no DAG backup bookkeeping, no double counting), and the
duplicate can only over-estimate, never fabricate a shorter solution. The prune
is applied AFTER the batched value pass so an expansion always costs exactly one
policy + one value pass, whatever the table says (budget parity with the arena).

BEST-AT-BUDGET. `best_at_budget=True` (default) keeps searching after the first
goal and returns the SHORTEST verified path found; `False` returns the first.
Because every edge costs 1, a node at depth g can only beat the incumbent
`best_cost` if `g + 1 < best_cost` (and a goal child if `g + 1 < best_cost`), so
such nodes are closed without spending budget -- an exact, not heuristic, prune.
`stop_after_certified=N` stops once N expansions have passed with no improvement
(a self-play generation knob; the arena gate uses None = run to budget).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

INF = 1 << 30


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Mirrors `spr.search.SearchResult` in role; fields are forward-native.

    `cost` is the realized primitive-move count (the metric of PROBLEM.md 1) and
    `path` the move list that achieves it -- for the forward arm those are the
    same object, so there is no `strict`/`abstract` split.
    """
    cost: int | None = None          # verified primitive moves of the best path
    path: list | None = None         # [(robot_slot, dir_idx), ...]
    expansions: int = 0
    nn_calls: int = 0
    extra: dict = field(default_factory=dict)
    root: object = None              # MCTS tree root (record extraction)

    @property
    def solved(self) -> bool:
        return self.cost is not None


def expansions_from_calls(calls: int) -> int:
    """The arena's accounting: 1 root call + 2 calls per expansion
    (`eval/compare.py::_CountingGuide`, lines 64-94)."""
    return max(0, calls - 1) // 2


def move_dump(path):
    """[[color, direction], ...] -- the `moves_seq` schema `eval.replay_validate`
    reads for forward rows (and `eval.compare.run_forward` emits)."""
    from move_planner.state import COLOR_ORDER
    from simulate import DIRECTIONS
    return [[COLOR_ORDER[s], DIRECTIONS[d]] for s, d in path]


def verify_path(start, path, target_idx, target, wr, wd, size=None):
    """Replay `path` under the real physics; (ok, reason). Free (no NN)."""
    from move_planner.state import apply_move, is_goal
    from nn.gen_grids import GRID
    size = GRID if size is None else size
    cur = start
    for j, (s, d) in enumerate(path):
        nxt = apply_move(cur, s, d, wr, wd, size)
        if nxt is None:
            return False, f"move {j}: slot {s} dir {d} is a no-op"
        cur = nxt
    if not is_goal(cur, target_idx, target):
        return False, "final state is not a goal"
    return True, f"{len(path)} moves"


# ---------------------------------------------------------------------------
# tree
# ---------------------------------------------------------------------------

class Node:
    __slots__ = ("pos", "parent", "move", "g", "children", "N", "Q", "v_est",
                 "prior", "goal", "dead", "closed", "expanded", "depth")

    def __init__(self, pos, parent=None, move=None, g=0, prior=1.0, q=None,
                 goal=False):
        self.pos = pos
        self.parent = parent
        self.move = move            # the (slot, dir) that produced this node
        self.g = g                  # moves from the root along the tree edge
        self.children = []
        self.N = 0
        self.Q = q                  # cost estimate (total moves), min-max normalized
        self.v_est = q
        self.prior = prior
        self.goal = goal
        self.dead = False           # no live child / pruned out
        self.closed = goal          # nothing selectable below (goals are exact leaves)
        self.expanded = False
        self.depth = g

    def path(self):
        out, nd = [], self
        while nd.parent is not None:
            out.append(nd.move)
            nd = nd.parent
        out.reverse()
        return out

    def child_visits(self):
        """[(slot, dir, N, prior, Q), ...] -- the AlphaZero policy target
        material for this decision (self-play record extraction)."""
        return [(c.move[0], c.move[1], c.N, c.prior, c.Q) for c in self.children
                if c.move is not None]


# ---------------------------------------------------------------------------
# the expansion primitive (one policy pass + one batched value pass)
# ---------------------------------------------------------------------------

def _expand_children(guide, env_id, node, target_idx, target, wr, wd, size, k,
                     acct=None):
    """Generate node's top-k children (unattached). Returns (kids, n_legal).

    kids = [(slot, dir, child_pos, prior, value)] in policy order. Exactly one
    policy pass and one batched value pass, matching `nn_astar` -- so an
    expansion here costs what an expansion there costs.
    """
    from move_planner.state import legal_moves
    from move_planner.evaluate import _ordered_moves
    succ = legal_moves(node.pos, wr, wd, size)
    if not succ:
        return [], 0
    _, plog = guide.eval_states(env_id, [node.pos], target_idx, target)   # policy pass
    row = plog[0]
    # prior = softmax over the LEGAL action set (illegal moves never enter)
    scores = [float(row[s, d]) for (s, d, _) in succ]
    mx = max(scores)
    ex = [math.exp(v - mx) for v in scores]
    z = sum(ex) or 1.0
    prior_of = {(s, d): e / z for (s, d, _), e in zip(succ, ex)}
    top = _ordered_moves(row, succ, k)                                    # arena's ordering
    vals, _ = guide.eval_states(env_id, [c for (_, _, c) in top],
                                target_idx, target)                       # value pass
    if acct is not None:
        acct["nn_policy_calls"] = acct.get("nn_policy_calls", 0) + 1
        acct["nn_value_calls"] = acct.get("nn_value_calls", 0) + 1
    return [(s, d, child, prior_of[(s, d)], float(v))
            for (s, d, child), v in zip(top, vals)], len(succ)


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

def mcts(guide, env_id, start, target_idx, target, wr, wd, size=None, k=5,
         max_expansions=1200, c_puct=1.5, backup="min", best_at_budget=True,
         root_noise=0.0, noise_alpha=0.3, rng=None, stop_after_certified=None,
         max_depth=64, root_k=None, acct=None) -> SearchResult:
    """PUCT tree search over primitive moves. Returns a SearchResult with .root.

    Every loop iteration makes progress: it expands a leaf (costs budget), or
    closes a node whose k-restricted subtree is exhausted / cannot improve on
    the incumbent (free) -- so the loop ends at the budget, at the first goal
    (`best_at_budget=False`), at `stop_after_certified`, or when the root
    closes (the whole k-restricted tree is explored).

    `root_k`: children at the ROOT only (0 = all legal moves); a generation knob
    so the depth-0 visit distribution covers the full action set. The arena gate
    uses None = k.
    """
    from move_planner.state import is_goal
    from nn.gen_grids import GRID
    import numpy as np

    size = GRID if size is None else size
    rng = rng or random.Random(0)
    acct = {} if acct is None else acct

    if is_goal(start, target_idx, target):
        return SearchResult(cost=0, path=[], expansions=0, nn_calls=0,
                            extra={"root_N": 0, "trivial": True})

    v0, _ = guide.eval_states(env_id, [start], target_idx, target)   # root value pass
    acct["nn_value_calls"] = acct.get("nn_value_calls", 0) + 1
    root = Node(start, q=float(v0[0]))
    best_g = {start: 0}
    qmin = qmax = root.Q
    expansions = 0
    nodes = 1
    prune_transposition = prune_dominated = prune_depth = 0
    best_cost, best_node = None, None
    last_improve = 0
    n_goals = 0
    first_goal_exp = None

    def _norm(q):
        if q is None:
            return 0.5
        if qmax <= qmin:
            return 0.5
        return min(1.0, max(0.0, (qmax - q) / (qmax - qmin)))     # lower cost -> 1

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
            w = sum(max(c.N, 1) for c in live)
            node.Q = sum(c.Q * max(c.N, 1) for c in live) / w
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

    def _useless(ng, goal):
        """True when nothing in this child's subtree can beat the incumbent."""
        if best_cost is None:
            return False
        return (ng >= best_cost) if goal else (ng + 1 >= best_cost)

    while expansions < max_expansions and not root.closed:
        if (best_cost is not None and stop_after_certified is not None
                and expansions - last_improve >= stop_after_certified):
            break
        # ---- select down to a leaf
        node, path = root, [root]
        while node.expanded and not node.closed:
            nxt = _select(node)
            if nxt is None:
                node.closed = True
                break
            node = nxt
            path.append(node)
        if node.closed:                      # exhausted subtree: free, still progress
            _backup(path)
            continue
        if node.depth >= max_depth or _useless(node.g, False):
            node.closed = True
            prune_depth += 1
            _backup(path)
            continue
        # ---- expand (the budget unit)
        expansions += 1
        kk = k
        if node is root and root_k is not None:
            kk = root_k if root_k > 0 else INF
        kids, n_legal = _expand_children(guide, env_id, node, target_idx, target,
                                         wr, wd, size, kk, acct)
        node.expanded = True
        if not kids:
            node.dead = node.closed = True
            _backup(path)
            continue
        if node is root and root_noise > 0:
            noise = np.random.default_rng(rng.randrange(1 << 30)).dirichlet(
                [noise_alpha] * len(kids))
            kids = [(s, d, c, (1 - root_noise) * p + root_noise * float(e), v)
                    for (s, d, c, p, v), e in zip(kids, noise)]
        ng = node.g + 1
        kept = []
        for (s, d, cpos, prior, val) in kids:
            goal = is_goal(cpos, target_idx, target)
            if _useless(ng, goal):
                prune_dominated += 1
                continue
            if ng >= best_g.get(cpos, INF):          # dominated revisit (nn_astar's rule)
                prune_transposition += 1
                continue
            best_g[cpos] = ng
            q = float(ng) if goal else float(ng) + val
            kept.append(Node(cpos, parent=node, move=(s, d), g=ng, prior=prior,
                             q=q, goal=goal))
        if not kept:
            node.dead = node.closed = True
            _backup(path)
            continue
        zp = sum(c.prior for c in kept) or 1.0    # renormalize over kept children
        for c in kept:
            c.prior /= zp
            node.children.append(c)
            nodes += 1
            qmin, qmax = min(qmin, c.Q), max(qmax, c.Q)
            if c.goal:
                # exact leaves are closed at birth and never re-selected, so they
                # would keep N=0 and vanish from the recorded visit distribution;
                # credit the one exact evaluation they did receive. (Visit counts
                # still UNDER-represent terminal actions -- `best_moves` on the
                # certified path, not visits, is the policy label this arm trains on.)
                c.N = 1
                n_goals += 1
                if best_cost is None or c.g < best_cost:
                    ok, _why = verify_path(start, c.path(), target_idx, target,
                                           wr, wd, size)
                    acct["physics_calls_verify"] = acct.get("physics_calls_verify", 0) + 1
                    if ok:
                        best_cost, best_node = c.g, c
                        last_improve = expansions
                        if first_goal_exp is None:
                            first_goal_exp = expansions
        _backup(path)
        if best_cost is not None and not best_at_budget:
            break

    extra = {"root_N": root.N, "nodes": nodes, "n_goals": n_goals,
             "root_closed": root.closed, "best_depth": best_cost,
             "first_goal_expansion": first_goal_exp,
             "prune_transposition": prune_transposition,
             "prune_dominated": prune_dominated, "prune_depth": prune_depth,
             "qmin": None if qmin == INF else round(float(qmin), 3),
             "qmax": None if qmax == -INF else round(float(qmax), 3)}
    res = SearchResult(expansions=expansions, extra=extra, root=root)
    if best_node is not None:
        res.cost, res.path = best_cost, best_node.path()
    return res


# ---------------------------------------------------------------------------
# the other arms (called unchanged where they exist upstream)
# ---------------------------------------------------------------------------

def astar(guide, env_id, start, target_idx, target, wr, wd, size=None, k=5,
          max_expansions=1200, acct=None, **_kw) -> SearchResult:
    """`move_planner.evaluate.nn_astar`, UNCHANGED (the F-M0 parity arm)."""
    from move_planner.evaluate import nn_astar
    c0 = getattr(guide, "calls", 0)
    cost, path = nn_astar(guide, env_id, start, target_idx, target, wr, wd,
                          k=k, max_iters=max_expansions)
    calls = getattr(guide, "calls", 0) - c0
    return SearchResult(cost=cost, path=path, expansions=expansions_from_calls(calls),
                        nn_calls=calls, extra={})


def greedy_policy(guide, env_id, start, target_idx, target, wr, wd, size=None,
                  max_expansions=1200, max_steps=40, acct=None, **_kw) -> SearchResult:
    """`move_planner.evaluate.nn_greedy_policy`, UNCHANGED: follow the policy
    argmax (beam 1). One policy pass per step, no value pass."""
    from move_planner.evaluate import nn_greedy_policy
    c0 = getattr(guide, "calls", 0)
    cost, path = nn_greedy_policy(guide, env_id, start, target_idx, target, wr, wd,
                                  max_steps=min(max_steps, max_expansions))
    calls = getattr(guide, "calls", 0) - c0
    return SearchResult(cost=cost, path=path, expansions=calls, nn_calls=calls,
                        extra={"steps": calls})


def greedy_value(guide, env_id, start, target_idx, target, wr, wd, size=None,
                 max_expansions=1200, max_steps=40, acct=None, **_kw) -> SearchResult:
    """`move_planner.evaluate.nn_greedy_value`, UNCHANGED: step to the child with
    the smallest predicted cost (beam 1). One batched value pass per step over
    ALL legal successors, no policy pass -- so a step is HALF an arena expansion
    in NN passes; `expansions` reports steps and `accounting.nn_calls` the truth."""
    from move_planner.evaluate import nn_greedy_value
    c0 = getattr(guide, "calls", 0)
    cost, path = nn_greedy_value(guide, env_id, start, target_idx, target, wr, wd,
                                 max_steps=min(max_steps, max_expansions))
    calls = getattr(guide, "calls", 0) - c0
    return SearchResult(cost=cost, path=path, expansions=calls, nn_calls=calls,
                        extra={"steps": calls})


# ---------------------------------------------------------------------------
# dispatcher (mirrors spr.search.run)
# ---------------------------------------------------------------------------

SEARCHES = ("astar", "mcts", "greedy_value", "greedy_policy")


def run(name, guide, env_id, start, target_idx, target, wr, wd, size=None, k=5,
        max_expansions=1200, acct=None, **kw) -> SearchResult:
    if name not in SEARCHES:
        raise ValueError(f"unknown search {name!r} (have {SEARCHES})")
    fn = {"astar": astar, "mcts": mcts, "greedy_value": greedy_value,
          "greedy_policy": greedy_policy}[name]
    c0 = getattr(guide, "calls", 0)
    res = fn(guide, env_id, start, target_idx, target, wr, wd, size=size, k=k,
             max_expansions=max_expansions, acct=acct, **kw)
    calls = getattr(guide, "calls", 0) - c0
    res.nn_calls = calls
    if acct is not None:
        acct["nn_calls"] = calls
        acct["expansions_from_calls"] = expansions_from_calls(calls)
        acct["expansions_own"] = res.expansions
    return res
