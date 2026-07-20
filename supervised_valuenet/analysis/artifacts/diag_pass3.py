"""Pass 3:
(1) static plan audit — subgoals whose support robot == bottleneck robot
    ("self-support"), counted in failing (pickled) and passing (regenerated)
    plans;
(2) counterfactual E — two-phase realizer implementing the plan's own
    approach-then-place-support-then-bounce semantics (zero extra moves);
(3) counterfactual F — E + single-robot detour/recruit (<=3 slides).
"""
import os
import sys
import json
import time
import pickle
from collections import Counter, deque

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH)

import diag_failures as d1  # noqa: E402  (CPU env, repo path, chdir)
import torch  # noqa: E402

from GridEnv import GridEnv, State, Robot_at  # noqa: E402
from skeleton.astar import AStar  # noqa: E402
from train.policy_tf import PolicyTF  # noqa: E402
from train.looped_pc import LoopedValueNet  # noqa: E402
from move_planner.state import COLOR_ORDER  # noqa: E402
from eval.compare import _nn_astar_backward, load_instances  # noqa: E402
from eval.realize import _slide_bfs  # noqa: E402
from simulate import wall_sets, slide, DIRECTIONS  # noqa: E402

SIZE = 16


# ---------------------------------------------------------------------------
# static plan audit
# ---------------------------------------------------------------------------

def self_support_subgoals(plan):
    """Subgoal ids whose bottleneck robot color == support robot color."""
    g = plan.g
    out = []
    for n, d in g.nodes(data=True):
        if d.get("ntype") != "subgoal":
            continue
        bn = sp = None
        for c in g.successors(n):
            t = g.nodes[c].get("ntype")
            if t == "bottleneck":
                bn = g.nodes[c].get("robot")
            elif t == "support":
                sp = g.nodes[c].get("robot")
        if bn is not None and sp is not None and bn.color == sp.color:
            out.append(n)
    return out


def failing_seg_self_support(plan, state, fail_seg):
    """Does the failing segment's subgoal use the mover as its own support?"""
    g = plan.g
    u = fail_seg["edge"][0]
    if g.nodes[u].get("ntype") != "bottleneck":
        return False
    for sg in g.predecessors(u):
        if g.nodes[sg].get("ntype") == "subgoal":
            for sib in g.successors(sg):
                if g.nodes[sib].get("ntype") == "support":
                    return g.nodes[sib]["robot"].color == fail_seg["color"]
    return False


# ---------------------------------------------------------------------------
# two-phase realizer (counterfactuals E and F)
# ---------------------------------------------------------------------------

def _u_candidates(env, end, sup, wr, wd):
    """Pre-bounce cells u: dependent edges u->end via a robot at `sup`."""
    out = []
    supf = frozenset({sup})
    for u, _v, d in env.G.in_edges(tuple(end), data=True):
        if d.get("dependent") != tuple(sup):
            continue
        if any(slide(u, dr, supf, wr, wd, SIZE) == tuple(end)
               for dr in DIRECTIONS):
            out.append(tuple(u))
    return out


def two_phase_realize(env, state, plan, allow_detour, detour_slides=3):
    """Work-list realizer with supported segments split into approach + bounce.

    Unit (i,0): mover reaches end directly, OR (if impossible now and the
    segment has a support) reaches a pre-bounce cell u; unit (i,1): mover
    continues u -> end (legal BFS under whatever robots stand around).
    Support placement is required only before the bounce, not the approach.
    """
    wr, wd = wall_sets(env.grid_data, SIZE)
    segs, deps_seg, err = d1.build_segments(plan, state)
    if err:
        return {"status": err, "moves": None, "detours": 0, "detour_moves": 0}
    g = plan.g
    n = len(segs)
    idx_by_parent = {}
    for i, s in enumerate(segs):
        idx_by_parent.setdefault(s["edge"][0], []).append(i)

    split = [s["support"] is not None
             and bool(_u_candidates(env, s["end"], s["support"], wr, wd))
             for s in segs]
    ucands = [(_u_candidates(env, s["end"], s["support"], wr, wd)
               if split[i] else None) for i, s in enumerate(segs)]

    def final(i):
        return (i, 1) if split[i] else (i, 0)

    units = [(i, 0) for i in range(n)] + [(i, 1) for i in range(n) if split[i]]
    udeps = {u: set() for u in units}
    for i, s in enumerate(segs):
        # (a) mover chain: arrival at the segment's source node first
        if g.nodes[s["src"]].get("ntype") in ("bottleneck", "support"):
            for j in idx_by_parent.get(s["src"], []):
                udeps[(i, 0)].add(final(j))
        if split[i]:
            udeps[(i, 1)].add((i, 0))
        sp = s.get("support_node")
        if sp is not None:
            tgt = (i, 1) if split[i] else (i, 0)
            # (b) support placed before the bounce (not before the approach)
            for j in idx_by_parent.get(sp, []):
                udeps[tgt].add(final(j))
            # (c) the support robot departs only after the bounce is consumed
            for j2, s2 in enumerate(segs):
                if j2 != i and s2["src"] == sp:
                    udeps[(j2, 0)].add(tgt)

    pos = {r.color: tuple(int(x) for x in r.position) for r in state.all_robots}
    target = tuple(int(x) for x in state.target)
    tcol = state.target_robot.color
    done = set()
    at_u = {}                       # seg -> chosen u (approach executed)
    total, detours, detour_moves = 0, 0, 0

    def pending_support_cells():
        out = set()
        for i, s in enumerate(segs):
            if s["support"] is not None and final(i) not in done:
                out.add(tuple(s["support"]))
        return out

    def try_unit(u, p):
        """Executable now under positions `p`? -> (moves, new_mover_pos,
        completes_both) or None."""
        i, ph = u
        s = segs[i]
        c = s["color"]
        blockers = frozenset(q for cc, q in p.items() if cc != c)
        if ph == 1 or not split[i]:
            m = _slide_bfs(p[c], tuple(s["end"]), blockers, wr, wd, SIZE)
            return None if m is None else (m, tuple(s["end"]), False)
        # phase 0 of a split segment: direct completion, else approach to u
        m = _slide_bfs(p[c], tuple(s["end"]), blockers, wr, wd, SIZE)
        if m is not None:
            return (m, tuple(s["end"]), True)
        best = None
        for ucell in ucands[i]:
            mm = _slide_bfs(p[c], ucell, blockers, wr, wd, SIZE)
            if mm is not None and (best is None or mm < best[0]):
                best = (mm, ucell, False)
        return best

    while len(done) < len(units):
        ready = [u for u in units if u not in done and udeps[u] <= done]
        if not ready:
            return {"status": "stall_cyclic", "moves": None,
                    "detours": detours, "detour_moves": detour_moves}
        execable = {}
        for u in ready:
            r = try_unit(u, pos)
            if r is not None:
                execable[u] = r
        if execable:
            pend = pending_support_cells()

            def klass(u):
                i, _ph = u
                s = segs[i]
                on_pending = pos[s["color"]] in (pend - {tuple(s["support"])
                                                 if s["support"] else None})
                off_goal = (s["color"] == tcol and pos[tcol] == target
                            and execable[u][1] != target)
                return (2 if on_pending else 0) + (1 if off_goal else 0)
            u = min(execable, key=lambda x: (klass(x), x))
            m, newpos, both = execable[u]
            total += m
            pos[segs[u[0]]["color"]] = newpos
            done.add(u)
            if both and split[u[0]]:
                done.add((u[0], 1))
            elif u[1] == 0 and split[u[0]] and not both:
                at_u[u[0]] = newpos
            continue
        if not allow_detour:
            return {"status": "stall", "moves": None, "detours": detours,
                    "detour_moves": detour_moves}
        pend = pending_support_cells()
        best = None
        for c in pos:
            if pos[c] in pend:
                continue
            if c == tcol and pos[c] == target:
                continue
            others = frozenset(q for cc, q in pos.items() if cc != c)
            seen = {pos[c]: 0}
            q = deque([(pos[c], 0)])
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
            del seen[pos[c]]
            for cell, dmov in seen.items():
                trial = dict(pos)
                trial[c] = cell
                for u in ready:
                    r = try_unit(u, trial)
                    if r is not None and (best is None
                                          or dmov + r[0] < best[0]):
                        best = (dmov + r[0], c, cell, u, r)
        if best is None:
            return {"status": "stall_multi", "moves": None, "detours": detours,
                    "detour_moves": detour_moves}
        tot, c, cell, u, r = best
        pos[c] = cell
        detours += 1
        detour_moves += tot - r[0]
        total += tot
        m, newpos, both = r
        pos[segs[u[0]]["color"]] = newpos
        done.add(u)
        if both and split[u[0]]:
            done.add((u[0], 1))
        elif u[1] == 0 and split[u[0]] and not both:
            at_u[u[0]] = newpos
        if detours > 25:
            return {"status": "detour_limit", "moves": None,
                    "detours": detours, "detour_moves": detour_moves}
    if pos[tcol] != target:
        return {"status": "target_off_goal", "moves": None, "detours": detours,
                "detour_moves": detour_moves}
    return {"status": "fixed", "moves": total, "detours": detours,
            "detour_moves": detour_moves}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(0)
    dev = "cpu"
    t0 = time.time()
    instances, _, _ = load_instances("eval/data/bench450.jsonl")
    stored = json.loads(open("eval/results/comparison_backward.json").read())
    rows = stored["systems"]["backward subgoal planner"]["rows"]
    plans = pickle.load(open(os.path.join(SCRATCH, "failing_plans.pkl"), "rb"))
    p2 = {r["idx"]: r for r in json.load(
        open(os.path.join(SCRATCH, "diag_pass2.json")))}

    policy = PolicyTF.load_from_checkpoint(
        "checkpoints_backward/policy_v2.ckpt", map_location=dev).to(dev).eval()
    value = LoopedValueNet.load_from_checkpoint(
        "checkpoints_backward/value_v2.ckpt", map_location=dev).to(dev).eval()
    solver = AStar(max_iters=4000, max_frontier=40_000)

    out = []
    cur_env_id, env = None, None
    for idx, (row, inst) in enumerate(zip(rows, instances)):
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
        failing = not row["solved"]
        if failing:
            plan = plans[idx]
        else:
            plan, _, _ = _nn_astar_backward(env, st, solver, policy, value,
                                            env_id, dev, 5, 1200)
        ss = self_support_subgoals(plan)
        n_sg = sum(1 for _n, d in plan.g.nodes(data=True)
                   if d.get("ntype") == "subgoal")
        rec = {"idx": idx, "d_star": inst["d_star"], "failing": failing,
               "n_subgoals": n_sg, "n_self_support": len(ss),
               "class": p2[idx]["class"] if failing else None}
        if failing:
            segs, order, fstep, executed, fseg, _pos, _w = \
                d1_replay(env, st, plan)
            rec["fail_seg_self_support"] = failing_seg_self_support(
                plan, st, segs[fseg])
            rec["cfE"] = two_phase_realize(env, st, plan, allow_detour=False)
            rec["cfF"] = two_phase_realize(env, st, plan, allow_detour=True)
        out.append(rec)
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/450 ({time.time() - t0:.0f}s)", flush=True)

    with open(os.path.join(SCRATCH, "diag_pass3.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"done ({time.time() - t0:.0f}s)")

    fails = [r for r in out if r["failing"]]
    passes = [r for r in out if not r["failing"]]
    print("\n=== self-support subgoals (static plan property) ===")
    print("failing plans containing >=1:",
          sum(1 for r in fails if r["n_self_support"] > 0), f"/{len(fails)}")
    print("passing plans containing >=1:",
          sum(1 for r in passes if r["n_self_support"] > 0), f"/{len(passes)}")
    print("failing seg belongs to a self-support subgoal:",
          sum(1 for r in fails if r.get("fail_seg_self_support")),
          f"/{len(fails)}")
    print("  by class:",
          dict(Counter((r["class"], bool(r.get("fail_seg_self_support")))
                       for r in fails)))

    print("\n=== counterfactual E: two-phase realizer, zero extra moves ===")
    print(dict(Counter(r["cfE"]["status"] for r in fails)))
    print("by class:", dict(Counter((r["class"], r["cfE"]["status"])
                                    for r in fails)))
    fx = [r for r in fails if r["cfE"]["status"] == "fixed"]
    if fx:
        regs = [r["cfE"]["moves"] - r["d_star"] for r in fx]
        print(f"E-fixed: n={len(fx)} mean_regret={sum(regs)/len(regs):.2f}")

    print("\n=== counterfactual F: two-phase + detour/recruit ===")
    print(dict(Counter(r["cfF"]["status"] for r in fails)))
    print("by class:", dict(Counter((r["class"], r["cfF"]["status"])
                                    for r in fails)))
    fx = [r for r in fails if r["cfF"]["status"] == "fixed"]
    if fx:
        regs = [r["cfF"]["moves"] - r["d_star"] for r in fx]
        dts = [r["cfF"]["detours"] for r in fx]
        print(f"F-fixed: n={len(fx)} mean_regret={sum(regs)/len(regs):.2f} "
              f"mean_detours={sum(dts)/len(dts):.2f}")
        eonly = {r["idx"] for r in fails if r["cfE"]["status"] == "fixed"}
        print("F-fixed but not E-fixed:",
              sum(1 for r in fx if r["idx"] not in eonly))
    def dbin(d):
        return "1-3" if d <= 3 else "4-6" if d <= 6 else "7-9" if d <= 9 \
            else "10+"
    print("\ncfF status x d* bin:",
          dict(Counter((r["cfF"]["status"], dbin(r["d_star"]))
                       for r in fails)))


def d1_replay(env, st, plan):
    import diag_pass2
    return diag_pass2.replay_to_failure(env, st, plan)


if __name__ == "__main__":
    main()
