"""Pass 2: mechanism trace for support-absent failures + hold-rule realizer.

Regenerates plans for the 211 stored failures only (CPU), pickles them for
reuse, and records:
  - mechanism of support absence (placed-then-left / scheduled-later /
    initially-there-but-departed / never-exists)
  - whether the failing segment is reachable in the plan's own frame
    (declared start + declared support present)
  - counterfactual C: greedy realizer with a "don't move a robot off a pending
    support cell while alternatives exist" rule, no detours (zero extra moves)
  - counterfactual D: C + single-robot detour (<=3 slides)
"""
import os
import sys
import json
import time
import pickle
from collections import Counter, deque

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH)

import diag_failures as d1  # noqa: E402  (sets CPU env, repo path, chdir)
import torch  # noqa: E402

from GridEnv import GridEnv, State, Robot_at  # noqa: E402
from skeleton.astar import AStar  # noqa: E402
from train.policy_tf import PolicyTF  # noqa: E402
from train.looped_pc import LoopedValueNet  # noqa: E402
from move_planner.state import COLOR_ORDER  # noqa: E402
from eval.compare import _nn_astar_backward, load_instances  # noqa: E402
from eval.realize import _topo, _slide_bfs  # noqa: E402
from simulate import wall_sets, slide, DIRECTIONS  # noqa: E402

SIZE = 16


def replay_to_failure(env, state, plan):
    """Execute in strict_moves order; return (segs, order, fail_pos, executed,
    fail_idx, pos_at_failure). Assumes the plan fails (validated in pass 1)."""
    wr, wd = wall_sets(env.grid_data, SIZE)
    segs, deps, err = d1.build_segments(plan, state)
    assert err is None
    order = _topo(deps)
    assert order is not None
    pos = {r.color: tuple(int(x) for x in r.position) for r in state.all_robots}
    executed = []
    for step, i in enumerate(order):
        s = segs[i]
        color, end = s["color"], tuple(s["end"])
        blockers = frozenset(p for c, p in pos.items() if c != color)
        m = _slide_bfs(pos[color], end, blockers, wr, wd, SIZE)
        if m is None:
            return segs, order, step, executed, i, pos, (wr, wd)
        pos[color] = end
        executed.append(i)
    return segs, order, None, executed, None, pos, (wr, wd)


def support_mechanism(state, segs, executed, fail_idx, pos):
    """Why is nobody at the failing segment's support cell?"""
    s = segs[fail_idx]
    S = tuple(s["support"])
    execd = set(executed)
    init = {r.color: tuple(int(x) for x in r.position) for r in state.all_robots}
    ends_at_S_exec = [j for j in executed if tuple(segs[j]["end"]) == S]
    ends_at_S_pending = [j for j in range(len(segs))
                         if j not in execd and j != fail_idx
                         and tuple(segs[j]["end"]) == S]
    out = {"mech": None}
    if ends_at_S_exec:
        j = ends_at_S_exec[-1]
        placer = segs[j]["color"]
        placed_at = executed.index(j)
        left = any(segs[k]["color"] == placer
                   for k in executed[placed_at + 1:])
        out["mech"] = ("placed_then_left" if left
                       else "placed_then_unexplained")  # shouldn't happen
        out["placer"] = placer
    elif ends_at_S_pending:
        out["mech"] = "placement_scheduled_later"
    else:
        movers_from_S = [c for c, p in init.items() if p == S]
        if movers_from_S:
            c = movers_from_S[0]
            moved = any(segs[k]["color"] == c for k in executed)
            out["mech"] = ("initially_there_departed" if moved
                           else "initially_there_still")  # shouldn't happen
            out["placer"] = c
        else:
            out["mech"] = "never_exists"
    return out


def hold_rule_realize(env, state, plan, allow_detour, detour_slides=3):
    """Greedy work-list realizer with support-hold preference.

    Segment classes, tried in order while alternatives exist:
      1. mover NOT on a pending support cell, does NOT move target off goal
      2. mover NOT on a pending support cell
      3. anything ready+reachable (forced)
    With allow_detour, stalls trigger the same single-robot relocation search
    as pass 1 (robots pinned on pending support cells and the target robot on
    the goal are immovable)."""
    wr, wd = wall_sets(env.grid_data, SIZE)
    segs, deps, err = d1.build_segments(plan, state)
    if err:
        return {"status": err, "moves": None, "detours": 0}
    pos = {r.color: tuple(int(x) for x in r.position) for r in state.all_robots}
    target = tuple(int(x) for x in state.target)
    tcol = state.target_robot.color
    n = len(segs)
    execd = set()
    total, detours = 0, 0

    def pending_support_cells(exclude=None):
        return {tuple(segs[j]["support"]) for j in range(n)
                if j not in execd and j != exclude
                and segs[j]["support"] is not None}

    while len(execd) < n:
        ready = [i for i in range(n) if i not in execd and deps[i] <= execd]
        if not ready:
            return {"status": "stall_cyclic", "moves": None, "detours": detours}
        reach = {}
        for i in ready:
            s = segs[i]
            blockers = frozenset(p for c, p in pos.items() if c != s["color"])
            m = _slide_bfs(pos[s["color"]], tuple(s["end"]), blockers, wr, wd,
                           SIZE)
            if m is not None:
                reach[i] = m
        if reach:
            def klass(i):
                s = segs[i]
                on_pending_sup = pos[s["color"]] in pending_support_cells(
                    exclude=i)
                off_goal = (s["color"] == tcol and pos[tcol] == target
                            and tuple(s["end"]) != target)
                return (2 if on_pending_sup else 0) + (1 if off_goal else 0)
            i = min(reach, key=lambda i: (klass(i), i))
            total += reach[i]
            pos[segs[i]["color"]] = tuple(segs[i]["end"])
            execd.add(i)
            continue
        if not allow_detour:
            return {"status": "stall", "moves": None, "detours": detours}
        best = None
        pend = pending_support_cells()
        for c in pos:
            p = pos[c]
            if p in pend:
                continue
            if c == tcol and p == target:
                continue
            others = frozenset(q for cc, q in pos.items() if cc != c)
            seen = {p: 0}
            q = deque([(p, 0)])
            while q:
                cell, dd = q.popleft()
                if dd >= detour_slides:
                    continue
                for dr in DIRECTIONS:
                    nxt = slide(cell, dr, others, wr, wd, SIZE)
                    if nxt == cell or nxt in seen:
                        continue
                    seen[nxt] = dd + 1
                    q.append((nxt, dd + 1))
            del seen[p]
            for cell, dmov in seen.items():
                trial = dict(pos)
                trial[c] = cell
                for i in ready:
                    s = segs[i]
                    start = trial[s["color"]]
                    blockers = frozenset(q2 for cc, q2 in trial.items()
                                         if cc != s["color"])
                    m = _slide_bfs(start, tuple(s["end"]), blockers, wr, wd,
                                   SIZE)
                    if m is not None and (best is None or dmov + m < best[0]):
                        best = (dmov + m, c, cell, i, m)
        if best is None:
            return {"status": "stall_multi", "moves": None, "detours": detours}
        _, c, cell, i, m = best
        pos[c] = cell
        total += best[0]
        detours += 1
        pos[segs[i]["color"]] = tuple(segs[i]["end"])
        execd.add(i)
        if detours > 25:
            return {"status": "detour_limit", "moves": None, "detours": detours}
    if pos[tcol] != target:
        return {"status": "target_off_goal", "moves": None, "detours": detours}
    return {"status": "fixed", "moves": total, "detours": detours}


def main():
    torch.manual_seed(0)
    dev = "cpu"
    t0 = time.time()
    instances, _, _ = load_instances("eval/data/bench450.jsonl")
    stored = json.loads(open("eval/results/comparison_backward.json").read())
    rows = stored["systems"]["backward subgoal planner"]["rows"]
    fail_idxs = [i for i, r in enumerate(rows)
                 if r["plan_found"] and not r["solved"]]
    policy = PolicyTF.load_from_checkpoint(
        "checkpoints_backward/policy_v2.ckpt", map_location=dev).to(dev).eval()
    value = LoopedValueNet.load_from_checkpoint(
        "checkpoints_backward/value_v2.ckpt", map_location=dev).to(dev).eval()
    solver = AStar(max_iters=4000, max_frontier=40_000)

    out = []
    plans = {}
    cur_env_id, env = None, None
    for k, idx in enumerate(fail_idxs):
        inst = instances[idx]
        env_id = inst["env_id"]
        if env_id != cur_env_id:
            env, _ = GridEnv.from_env(env_id)
            cur_env_id = env_id
        positions = [tuple(p) for p in inst["positions"]]
        tidx = inst["target_idx"]
        st = State(
            target=tuple(inst["target"]),
            target_robot=Robot_at(position=positions[tidx],
                                  color=COLOR_ORDER[tidx]),
            helpers=[Robot_at(position=positions[j], color=COLOR_ORDER[j])
                     for j in range(len(positions)) if j != tidx])
        plan, _, _ = _nn_astar_backward(env, st, solver, policy, value,
                                        env_id, dev, 5, 1200)
        plans[idx] = plan
        segs, order, fail_step, executed, fail_idx_seg, pos, (wr, wd) = \
            replay_to_failure(env, st, plan)
        rec = {"idx": idx, "env_id": env_id, "d_star": inst["d_star"]}
        assert fail_step is not None, f"instance {idx} unexpectedly passed"
        s = segs[fail_idx_seg]
        sc = tuple(s["support"]) if s["support"] is not None else None
        # plan-frame reachability of the failing segment
        blk = frozenset({sc}) if sc else frozenset()
        rec["declared_frame_reachable"] = _slide_bfs(
            tuple(s["start"]), tuple(s["end"]), blk, wr, wd, SIZE) is not None
        sup_absent = (sc is not None
                      and sc not in {p for c, p in pos.items()
                                     if c != s["color"]})
        rec["class"] = ("support_absent" if sup_absent else
                        "pure_block" if _slide_bfs(
                            pos[s["color"]], tuple(s["end"]), frozenset(),
                            wr, wd, SIZE) is not None else "other")
        if sup_absent:
            rec["support_mech"] = support_mechanism(st, segs, executed,
                                                    fail_idx_seg, pos)
        rec["cf_hold"] = hold_rule_realize(env, st, plan, allow_detour=False)
        rec["cf_hold_detour"] = hold_rule_realize(env, st, plan,
                                                  allow_detour=True)
        out.append(rec)
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(fail_idxs)} ({time.time() - t0:.0f}s)",
                  flush=True)

    with open(os.path.join(SCRATCH, "diag_pass2.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(SCRATCH, "failing_plans.pkl"), "wb") as f:
        pickle.dump(plans, f)
    print(f"done ({time.time() - t0:.0f}s)")

    print("\nclass counts:", Counter(r["class"] for r in out))
    print("declared-frame reachable:",
          sum(1 for r in out if r["declared_frame_reachable"]),
          f"/{len(out)}")
    print("\nsupport-absent mechanisms:",
          Counter(r["support_mech"]["mech"] for r in out
                  if "support_mech" in r))
    print("\ncf_hold (ordering + hold rule, zero extra moves):",
          Counter(r["cf_hold"]["status"] for r in out))
    print("cf_hold_detour:",
          Counter(r["cf_hold_detour"]["status"] for r in out))
    fx = [r for r in out if r["cf_hold"]["status"] == "fixed"]
    if fx:
        regs = [r["cf_hold"]["moves"] - r["d_star"] for r in fx]
        print(f"cf_hold fixed: n={len(fx)} mean_regret={sum(regs)/len(regs):.2f}")
    fx = [r for r in out if r["cf_hold_detour"]["status"] == "fixed"]
    if fx:
        regs = [r["cf_hold_detour"]["moves"] - r["d_star"] for r in fx]
        print(f"cf_hold_detour fixed: n={len(fx)} "
              f"mean_regret={sum(regs)/len(regs):.2f} "
              f"mean_detours={sum(r['cf_hold_detour']['detours'] for r in fx)/len(fx):.2f}")
    print("\ncf_hold status x class:")
    ct = Counter((r["class"], r["cf_hold"]["status"]) for r in out)
    for key in sorted(ct):
        print(f"  {key}: {ct[key]}")
    print("\ncf_hold_detour status x class:")
    ct = Counter((r["class"], r["cf_hold_detour"]["status"]) for r in out)
    for key in sorted(ct):
        print(f"  {key}: {ct[key]}")


if __name__ == "__main__":
    main()
