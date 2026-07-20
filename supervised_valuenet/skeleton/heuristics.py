"""The two swappable heuristics that drive the search.

Everything the A* skeleton needs from "intelligence" lives behind these two
functions. The default implementations wrap the hand-coded GridEnv heuristics;
later, each can be replaced by a neural network with the same signature.

    propose(env, goal, mover, helpers, support) -> list[Candidate]
        Which subgoals to consider for getting `mover` to cell `goal`.

    score(env, candidate) -> float
        A cheap, optimistic estimate of a candidate's added cost.

A Candidate is a fully-resolved proposal: a bottleneck the mover stops at, a
support cell a helper must occupy, the chosen helper, and the support position
that lets the mover reach the bottleneck's segment from its parent.
"""
from __future__ import annotations

from dataclasses import dataclass

from GridEnv import GridEnv, Robot_at, Subgoal, State


@dataclass
class Candidate:
    subgoal: Subgoal          # bottleneck + support + helper, from GridEnv
    parent_support: tuple | None   # support enabling bottleneck -> parent edge
    score: float


def _dependent_supports(env: GridEnv, pos):
    """Support cells of dependent edges leading into `pos`."""
    return {d["dependent"] for _, _, d in env.G.in_edges(pos, data=True)
            if "dependent" in d}


def propose(env: GridEnv, goal, mover: Robot_at, helpers, support: Robot_at | None,
            b1: bool = False):
    """Candidate subgoals for moving `mover` to cell `goal`.

    `support` is the helper already pinned for this segment (when the parent is
    itself a bottleneck), else None. Returns candidates whose bottleneck can
    actually reach the parent segment via some support. With `b1=True`
    (Lever B1, analysis/b1_design.md) transient-support candidates are
    included; the default path is unchanged.
    """
    segment = State(target=goal, target_robot=mover, helpers=helpers)
    raw = list(env.propose_subgoal_states(segment, support_robot=support,
                                          transient=b1))

    # Fallback: when `goal` cannot be reached as a plain stop, it must be
    # reached by first placing a helper at one of its dependent-edge supports.
    # Retry the proposal pinned to each such support so its subgoals surface.
    if not raw:
        for sup_pos in _dependent_supports(env, goal):
            for helper in helpers:
                pinned = Robot_at(position=sup_pos, color=helper.color)
                raw.extend(env.propose_subgoal_states(segment, support_robot=pinned,
                                                      transient=b1))

    # A bottleneck is only useful if it can reach the parent cell `goal` via
    # an exact (dependent-edge-free, or single-support) path.
    parent_supports = []
    if support is not None:
        parent_supports.append(support.position)
    parent_supports.append(None)
    parent_supports += [s for s in _dependent_supports(env, goal)
                        if s not in parent_supports]

    candidates = []
    for subgoal, score in raw:
        bn = subgoal.bottleneck.position
        for ps in parent_supports:
            if env.compute_exact_shortest_path_length(bn, goal, ps) is not None:
                candidates.append(Candidate(subgoal, ps, score))
                break
    return candidates


def propose_b1(env: GridEnv, goal, mover: Robot_at, helpers,
               support: Robot_at | None):
    """`propose` with the Lever B1 temporary-support vocabulary enabled —
    the drop-in for `AStar(propose=heuristics.propose_b1)`."""
    return propose(env, goal, mover, helpers, support, b1=True)


def score(env: GridEnv, candidate: Candidate) -> float:
    """Optimistic cost estimate for a candidate (the hand-coded heuristic)."""
    return env.subgoal_score(candidate.subgoal)
