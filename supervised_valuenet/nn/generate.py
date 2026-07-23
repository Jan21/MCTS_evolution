"""Generate value-network training records by solving instances with the A*.

For each instance we roll out one optimal trajectory. At every subgoal decision
we enumerate the candidate subgoals and commit-and-solve each one, recording its
exact cost-to-go (the admissible heuristic makes the first complete plan optimal,
so each commit-and-solve returns the true remaining cost). The cheapest candidate
is committed and the rollout advances; alternatives are kept as harder examples.

Each record is self-contained: instance, the open segment being decided, the
partial-plan context so far, the candidate, the integer cost-to-go, and whether
the candidate is optimal. The label is stored raw (integer move count) so any
classification binning / smoothing scheme can be applied later without
regenerating data.

    python -m nn.generate --graphs 0-99 --per-graph 20 --out nn/data/train.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from GridEnv import GridEnv, State, Robot_at
from skeleton.astar import (AStar, _initial_plan, _segment, _apply,
                            _reference_helpers)
from skeleton import heuristics


# -- instance sampling --------------------------------------------------------

def random_instance(env: GridEnv, colors, rng) -> State:
    """A random solvable-looking instance: distinct cells for robots + target."""
    cells = list(env.G.nodes())
    pts = rng.sample(cells, len(colors) + 1)
    robots = [Robot_at(position=pts[i], color=colors[i]) for i in range(len(colors))]
    return State(target=pts[-1], target_robot=robots[0], helpers=robots[1:])


# -- record extraction --------------------------------------------------------

def _xy(p):
    return [int(p[0]), int(p[1])] if p is not None else None


def _fixed_g(plan):
    return sum(d["cost"] for _, _, d in plan.g.edges(data=True)
              if d["status"] == "fixed" and d["cost"] is not None)


def _context(plan):
    """Cells already committed as bottlenecks / supports, and open endpoints."""
    bn = [_xy(d["pos"]) for _, d in plan.g.nodes(data=True)
          if d.get("ntype") == "bottleneck"]
    sp = [_xy(d["pos"]) for _, d in plan.g.nodes(data=True)
          if d.get("ntype") == "support"]
    open_eps = []
    for u, v, d in plan.g.edges(data=True):
        if d["status"] == "open":
            open_eps += [_xy(plan.g.nodes[u].get("pos")),
                         _xy(plan.g.nodes[v].get("pos"))]
    return bn, sp, open_eps


def rollout(env: GridEnv, state: State, solver: AStar, env_id: int,
            max_candidates=None):
    """One optimal trajectory; label every candidate at each decision.

    The candidate enumeration mirrors `AStar._expand` exactly, including the
    Lever B2 reference-helper injection when the solver was built with
    `by_reference=True` — so the labeled candidate set is the same set the
    solver's own search would generate at this decision.
    """
    records = []
    plan = _initial_plan(env, state)
    depth = 0
    while not plan.is_complete():
        parent, child = plan.open_edges()[0]
        seg = _segment(plan, state, parent, child)

        # No decision when the segment pins to an exact path; just fix it.
        if env.compute_exact_shortest_path_length(seg.start, seg.end, seg.fix_support) is not None:
            plan = solver._expand(env, state, plan)[0]
            continue

        if solver.by_reference:
            seg.helpers = seg.helpers + _reference_helpers(plan, seg.mover.color)
        cands = solver.propose(env, seg.end, seg.mover, seg.helpers, seg.support)
        if max_candidates is not None:
            cands = sorted(cands, key=lambda c: solver.score(env, c))[:max_candidates]

        fixed_g = _fixed_g(plan)
        bn_ctx, sp_ctx, open_eps = _context(plan)
        labeled, best = [], None
        for cand in cands:
            child_plan = _apply(env, plan, parent, child, seg, cand,
                                by_reference=solver.by_reference)
            if child_plan is None:
                continue
            done = solver.solve_plan(env, state, child_plan)
            if done is None:
                continue
            ctg = int(done.cost()) - int(fixed_g)
            labeled.append((cand, child_plan, ctg))
            if best is None or done.cost() < best[1]:
                best = (child_plan, done.cost())

        if not labeled:
            break
        best_ctg = min(c for _, _, c in labeled)
        for cand, _, ctg in labeled:
            records.append({
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
                "cost_to_go": ctg, "is_optimal": ctg == best_ctg, "depth": depth,
            })
        # advance along the optimal trajectory
        plan = next(cp for cand, cp, c in labeled if c == best_ctg)
        depth += 1
    return records


# -- driver -------------------------------------------------------------------

def parse_graphs(spec: str):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def make_solver(vocab="base", max_iters=4000, max_frontier=40_000):
    """Labeling solver for a plan-language vocabulary.

    "base": the original static-support language (default; unchanged path).
    "b1":   + transient supports (Lever B1, `heuristics.propose_b1`).
    "b2":   + supports-by-reference (Lever B2, `AStar(by_reference=True)`).
    Park repairs are NOT part of any labeling vocabulary: they are
    deterministic search-time repairs proposed from realization failures,
    never ranked by the networks (see analysis/b1_extension_notes.md), so
    there is nothing for a net to learn about them. Cost-to-go labels are
    vocabulary-relative; never mix vocabularies in one dataset.
    """
    if vocab not in ("base", "b1", "b2"):
        raise ValueError(f"unknown vocab {vocab!r}")
    propose = heuristics.propose if vocab == "base" else heuristics.propose_b1
    return AStar(propose=propose, max_iters=max_iters,
                 max_frontier=max_frontier, by_reference=(vocab == "b2"))


def generate(graphs, per_graph, out, seed=0, max_candidates=None,
             max_iters=4000, max_frontier=40_000, vocab="base"):
    rng = random.Random(seed)
    solver = make_solver(vocab, max_iters=max_iters, max_frontier=max_frontier)
    _, s0 = GridEnv.from_env(graphs[0])
    colors = [s0.target_robot.color] + [h.color for h in s0.helpers]

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rec = n_inst = 0
    t0 = time.time()
    with open(out, "w") as f:
        for gid in graphs:
            env, _ = GridEnv.from_env(gid)
            kept = 0
            attempts = 0
            while kept < per_graph and attempts < per_graph * 4:
                attempts += 1
                st = random_instance(env, colors, rng)
                try:
                    recs = rollout(env, st, solver, gid, max_candidates)
                except Exception:
                    recs = []
                if not recs:
                    continue
                for r in recs:
                    f.write(json.dumps(r) + "\n")
                n_rec += len(recs)
                n_inst += 1
                kept += 1
            print(f"graph {gid}: {kept} instances, {n_rec} records so far "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"\ndone: {n_inst} instances, {n_rec} records -> {out}")
    return n_inst, n_rec


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--graphs", default="0-19")
    p.add_argument("--per-graph", type=int, default=20)
    p.add_argument("--out", default="nn/data/train.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-candidates", type=int, default=None)
    p.add_argument("--vocab", default="base", choices=["base", "b1", "b2"],
                   help="plan-language vocabulary for labels (house rule: "
                        "never mix vocabularies in one dataset)")
    a = p.parse_args()
    generate(parse_graphs(a.graphs), a.per_graph, a.out,
             seed=a.seed, max_candidates=a.max_candidates, vocab=a.vocab)
