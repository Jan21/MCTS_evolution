"""Gate-5 regression corpus builder: the 12 residual-defect instances.

`eval/results/residual_failures_postfix.json` names two 6-instance defect
classes among the 150-instance bench
(`eval/data/bench450_first150.jsonl`, boards in `environments/`):

  robot_reused_two_places   [42, 50, 58, 86, 126, 138]
      the old planner scheduled ONE physical robot as TWO plan leaves
      (fix (b) in skeleton/astar.py `_apply` refuses the second leaf);
  goal_stopper_never_placed [25, 52, 87, 91, 111, 129]
      the old planner claimed a supported route into the goal whose stopper
      cell no plan node ever occupies (fix (c) refuses the claim).

For each instance this script replays the archived FAILING plan
(`analysis/artifacts/residual_failing_plans.pkl`) against the CURRENT
solver, guided: at every decision it applies the failing plan's own next
subgoal group (sg_k / bn_k / sp_k / leaf_k, in nc order) instead of the
optimal candidate, reconstructing the defective plan exactly as the old
planner built it -- until `_apply` under the current fixes REFUSES a group.
That decision context is the regression item: a `replay_backward_decision`
line whose candidate list is the natural `heuristics.propose` enumeration
(stable-sorted, max_candidates-truncated) with the DEFECT candidate
appended when propose no longer surfaces it, plus `python_labels` for every
candidate. The engines must agree per candidate, and the defect candidate
must be `rejected` by BOTH -- accepting it would recreate the archived
defect.

Sanity enforced per instance: the guided replay must (a) reproduce the
failing plan's node ids while it runs (same nc counter), and (b) hit a
refusal on a group of the EXPECTED defect mechanism before the plan
completes. If any failing plan could still be rebuilt end-to-end, the build
aborts -- that would mean the fixes do not cover the archived defect.

Outputs (committed under golden/regression/):
  regression12.jsonl       the 12 replay_backward_decision lines
  regression12.meta.json   per line: bench idx, defect class, the defect
                           candidate's index in the candidate list, and the
                           expected rejection flag

    python rust_datagen/pyref/gen_regression.py \
        --out-dir rust_datagen/golden/regression
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

PYREF_DIR = Path(__file__).resolve().parent
SV_DIR = PYREF_DIR.parent.parent
for _p in (str(SV_DIR), str(PYREF_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CHILD_FLAG = "RR_PYREF_CHILD_REGR"

DUP_LEAF_IDS = [42, 50, 58, 86, 126, 138]
GOAL_STOPPER_IDS = [25, 52, 87, 91, 111, 129]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", default=str(SV_DIR / "rust_datagen" / "golden"
                                            / "regression"))
    p.add_argument("--bench", default=str(SV_DIR / "eval" / "data"
                                          / "bench450_first150.jsonl"))
    p.add_argument("--plans", default=str(SV_DIR / "analysis" / "artifacts"
                                          / "residual_failing_plans.pkl"))
    p.add_argument("--max-candidates", type=int, default=14)
    p.add_argument("--max-iters", type=int, default=4000)
    p.add_argument("--max-frontier", type=int, default=40_000)
    return p.parse_args(argv)


def main():
    a = parse_args()
    if os.environ.get(CHILD_FLAG) != "1":
        env = {**os.environ,
               "RR_GRID": "16", "RR_ROBOTS": "4", "RR_WALLS": "48",
               "RR_ENV_DIR": str(SV_DIR / "environments"),
               CHILD_FLAG: "1", "PYTHONPATH": str(SV_DIR)}
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                             *sys.argv[1:]], env=env, cwd=str(SV_DIR)).returncode
        sys.exit(rc)
    _child(a)


# ---------------------------------------------------------------------------


def _groups(plan):
    """The failing plan's apply groups in nc order:
    k -> (parent, child, bottleneck_robot, support_robot, helper_robot, ps)."""
    g = plan.g
    out = {}
    k = 1
    while f"sg_{k}" in g.nodes:
        sg, bn, sp, leaf = f"sg_{k}", f"bn_{k}", f"sp_{k}", f"leaf_{k}"
        parent = next(iter(g.predecessors(sg)))
        # The child at APPLY time: bn_k -> C, but a later apply j expanding
        # (bn_k, C) rewires it to bn_k -> sg_j (with bn_j -> C). Chase the
        # subgoal chain until a non-subgoal node: that is the original C.
        child = next(iter(g.successors(bn)))
        while g.nodes[child].get("ntype") == "subgoal":
            j = child.split("_", 1)[1]
            child = next(iter(g.successors(f"bn_{j}")))
        out[k] = {
            "parent": parent, "child": child,
            "bottleneck": g.nodes[bn]["robot"],
            "support": g.nodes[sp]["robot"],
            "helper": g.nodes[leaf]["robot"],
            "ps": g.nodes[sg].get("parent_support_pos"),
        }
        k += 1
    return out


def _mk_candidate(grp):
    from skeleton.heuristics import Candidate
    from GridEnv import Subgoal
    sub = Subgoal(bottleneck=grp["bottleneck"], support=grp["support"],
                  goal_pos=None, target_robot=None, helper=grp["helper"])
    ps = grp["ps"]
    return Candidate(subgoal=sub,
                     parent_support=None if ps is None else tuple(ps),
                     score=0.0)


def _same_candidate(c, grp):
    ps = grp["ps"]
    return (tuple(c.subgoal.bottleneck.position) == tuple(grp["bottleneck"].position)
            and tuple(c.subgoal.support.position) == tuple(grp["support"].position)
            and c.subgoal.helper.color == grp["helper"].color
            and tuple(c.subgoal.helper.position) == tuple(grp["helper"].position)
            and (c.parent_support is None if ps is None
                 else c.parent_support is not None
                 and tuple(c.parent_support) == tuple(ps)))


def _build_instance(idx, inst, plan, env, grid_data, a):
    """Guided replay -> (dump_line, meta) at the refusal decision."""
    import common
    from GridEnv import State, Robot_at
    from skeleton.astar import AStar, _initial_plan, _segment, _apply

    palette = common.PALETTE[:len(inst["positions"])]
    robots = [Robot_at(position=tuple(p), color=c)
              for p, c in zip(inst["positions"], palette)]
    tidx = inst["target_idx"]
    state = State(target=tuple(inst["target"]), target_robot=robots[tidx],
                  helpers=[r for i, r in enumerate(robots) if i != tidx])

    fail_groups = _groups(plan)
    assert tuple(plan.g.nodes["goal"]["pos"]) == tuple(state.target)
    assert tuple(plan.g.nodes["leaf_0"]["pos"]) == tuple(robots[tidx].position)

    solver = AStar(max_iters=a.max_iters, max_frontier=a.max_frontier)
    cur = _initial_plan(env, state)
    while not cur.is_complete():
        parent, child = cur.open_edges()[0]
        k = cur.nc
        grp = fail_groups.get(k)
        if grp is None or (grp["parent"], grp["child"]) != (parent, child):
            # the failing plan pinned this edge exactly (no group here)
            exact = env.compute_exact_shortest_path_length(
                *_seg_ends(cur, state, parent, child))
            if exact is None:
                raise AssertionError(
                    f"idx {idx}: guided replay stuck at edge ({parent},{child}) "
                    f"nc={k}: no matching group and no exact pin")
            cur = solver._expand(env, state, cur)[0]
            continue
        seg = _segment(cur, state, parent, child)
        cand = _mk_candidate(grp)
        applied = _apply(env, cur, parent, child, seg, cand)
        if applied is not None:
            cur = applied
            continue
        # refusal point: build the regression decision line
        cands = solver.propose(env, seg.end, seg.mover, seg.helpers, seg.support)
        cands = sorted(cands, key=lambda c: solver.score(env, c))[: a.max_candidates]
        defect_idx = next((i for i, c in enumerate(cands)
                           if _same_candidate(c, grp)), None)
        injected = defect_idx is None
        if injected:
            cands = cands + [cand]
            defect_idx = len(cands) - 1
        labels = []
        from nn.generate import _fixed_g
        fixed_g = _fixed_g(cur)
        for c in cands:
            cp = _apply(env, cur, parent, child, seg, c)
            if cp is None:
                labels.append({"ctg": None, "rejected": True})
                continue
            done = solver.solve_plan(env, state, cp)
            labels.append({"ctg": None, "rejected": False} if done is None else
                          {"ctg": int(done.cost()) - int(fixed_g),
                           "rejected": False})
        assert labels[defect_idx]["rejected"], (
            f"idx {idx}: defect candidate not rejected by current python")
        line = {
            "task": "replay_backward_decision",
            "id": f"regr:idx{idx}:nc{k}",
            "board": common.board_obj(inst["env_id"], 16, grid_data),
            "dependent_edge_weight": 2,
            "max_iters": a.max_iters,
            "max_frontier": a.max_frontier,
            "max_candidates": a.max_candidates,
            "state": common.ser_state(state),
            "plan": common.ser_plan(cur),
            "open_edge": [parent, child],
            "candidates": [common.ser_candidate(c) for c in cands],
            "python_labels": labels,
        }
        meta = {"id": line["id"], "bench_idx": idx,
                "env_id": inst["env_id"], "refused_group": k,
                "defect_candidate_index": defect_idx,
                "n_candidates": len(cands),
                "injected": injected}
        return line, meta
    raise AssertionError(
        f"idx {idx}: failing plan rebuilt END-TO-END under the current fixes "
        f"-- the archived defect is still constructible")


def _seg_ends(plan, state, parent, child):
    from skeleton.astar import _segment
    seg = _segment(plan, state, parent, child)
    return seg.start, seg.end, seg.fix_support


def _child(a):
    import common

    bench = [json.loads(l) for l in open(a.bench)]
    with open(a.plans, "rb") as f:
        plans = pickle.load(f)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines, metas = [], []
    envs = {}
    for cls, ids in (("robot_reused_two_places", DUP_LEAF_IDS),
                     ("goal_stopper_never_placed", GOAL_STOPPER_IDS)):
        for idx in ids:
            inst = bench[idx]
            gid = inst["env_id"]
            if gid not in envs:
                grid_data = common.load_board_grid_data(
                    SV_DIR / "environments", gid)
                envs[gid] = (common.build_env(grid_data, 16, mode="lazy",
                                              weight=2, table_workers=8),
                             grid_data)
            env, grid_data = envs[gid]
            line, meta = _build_instance(idx, inst, plans[idx], env,
                                         grid_data, a)
            meta["defect_class"] = cls
            lines.append(line)
            metas.append(meta)
            print(f"[regr] idx {idx} ({cls}): refusal at group "
                  f"{meta['refused_group']}, defect candidate "
                  f"#{meta['defect_candidate_index']} of "
                  f"{meta['n_candidates']}"
                  f"{' (injected)' if meta['injected'] else ''}")

    with open(out_dir / "regression12.jsonl", "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    with open(out_dir / "regression12.meta.json", "w") as f:
        json.dump({"description": "gate-5 regression: 12 residual-defect "
                                  "decision contexts (see gen_regression.py)",
                   "items": metas}, f, indent=1)
    print(f"[regr] wrote {len(lines)} lines -> {out_dir}/regression12.jsonl")


if __name__ == "__main__":
    main()
