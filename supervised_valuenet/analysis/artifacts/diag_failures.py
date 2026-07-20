"""Diagnosis pass: why do backward subgoal plans fail strict realization.

Regenerates the backward plan for every bench450 instance (same nets/budget as
the stored comparison run, no realize_check), then runs an INSTRUMENTED copy of
eval.realize.strict_moves that records the exact failure channel and, for
BFS-unreachable segments, which robots block and who they are to the plan.
Also runs two counterfactual realizers (flexible order; flexible order +
single-robot detour) and a move-by-move legality replay on passing instances.

Read-only wrt the repo. Writes JSON results into the scratchpad only.
"""
import os
import sys
import json
import time
import itertools
from collections import Counter, deque

REPO = "/mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet"
SCRATCH = os.path.dirname(os.path.abspath(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = ""           # CPU only (no GPU allowed)
os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch  # noqa: E402

torch.set_num_threads(8)                          # polite on the shared box

from GridEnv import GridEnv, State, Robot_at  # noqa: E402
from skeleton.astar import AStar  # noqa: E402
from train.policy_tf import PolicyTF  # noqa: E402
from train.looped_pc import LoopedValueNet  # noqa: E402
from move_planner.state import COLOR_ORDER  # noqa: E402
from eval.compare import _nn_astar_backward, load_instances  # noqa: E402
from eval.realize import (_physical_segments, _mover_source,  # noqa: E402
                          _support_node_for, _topo, _slide_bfs)
from simulate import wall_sets, slide, DIRECTIONS  # noqa: E402

SIZE = 16


# ---------------------------------------------------------------------------
# segment/dependency construction (verbatim logic from eval.realize.strict_moves)
# ---------------------------------------------------------------------------

def build_segments(plan, state):
    """Returns (segs, deps, err). err in {None,'no_segments','no_mover'}."""
    g = plan.g
    segs = _physical_segments(plan)
    if not segs:
        return None, None, "no_segments"
    idx_by_parent = {}
    for i, s in enumerate(segs):
        idx_by_parent.setdefault(s["edge"][0], []).append(i)
    for s in segs:
        src = _mover_source(g, s["edge"][1])
        if src is None or "robot" not in g.nodes[src]:
            return None, None, "no_mover"
        s["src"] = src
        s["color"] = g.nodes[src]["robot"].color
    deps = [set() for _ in segs]
    for i, s in enumerate(segs):
        if g.nodes[s["src"]].get("ntype") in ("bottleneck", "support"):
            for j in idx_by_parent.get(s["src"], []):
                deps[i].add(j)
        sp = _support_node_for(g, s["edge"][0], s["edge"][1], s["support"])
        s["support_node"] = sp
        if sp is not None:
            for j in idx_by_parent.get(sp, []):
                deps[i].add(j)
            for j2, s2 in enumerate(segs):
                if j2 != i and s2["src"] == sp:
                    deps[j2].add(i)
    return segs, deps, None


def robot_seg_indices(segs):
    by_color = {}
    for i, s in enumerate(segs):
        by_color.setdefault(s["color"], []).append(i)
    return by_color


# ---------------------------------------------------------------------------
# blocking analysis for one unreachable segment
# ---------------------------------------------------------------------------

def analyze_block(plan, segs, executed, pos, s, wr, wd, state):
    """Who prevents `s`'s mover from reaching its end cell right now."""
    g = plan.g
    color, cur, end = s["color"], pos[s["color"]], tuple(s["end"])
    sc = tuple(s["support"]) if s["support"] is not None else None
    others = {c: p for c, p in pos.items() if c != color}
    blockers = frozenset(others.values())
    by_color_segs = robot_seg_indices(segs)
    exec_count = Counter(segs[j]["color"] for j in executed)

    out = {
        "mover": color,
        "mover_is_target": color == state.target_robot.color,
        "cur": list(cur), "declared_start": list(s["start"]),
        "end": list(end), "support_cell": list(sc) if sc else None,
        "support_node_assigned": s.get("support_node") is not None,
        "support_present": sc is not None and sc in blockers,
        "end_occupied_by": next((c for c, p in others.items() if p == end), None),
        "walls_only_reachable":
            _slide_bfs(cur, end, frozenset(), wr, wd, SIZE) is not None,
        "support_only_reachable": None,
        "goal_segment": g.nodes[s["edge"][0]].get("ntype") == "goal",
    }
    if sc is not None:
        out["support_only_reachable"] = (
            _slide_bfs(cur, end, frozenset({sc}), wr, wd, SIZE) is not None)

    execd = set(executed)

    def flags_for(c):
        p = others[c]
        own = by_color_segs.get(c, [])
        return {
            "color": c, "pos": list(p),
            "is_target_robot": c == state.target_robot.color,
            "plan_touches": bool(own),
            "already_moved": exec_count[c] > 0,
            "has_pending_own_segments": any(j not in execd for j in own),
            "at_pending_support_cell": any(
                j not in execd and segs[j]["support"] is not None
                and tuple(segs[j]["support"]) == p for j in range(len(segs))),
            "at_consumed_support_cell": any(
                j in execd and segs[j]["support"] is not None
                and tuple(segs[j]["support"]) == p for j in range(len(segs))),
        }

    # single critical blockers: removing that one robot alone unblocks
    crit = []
    for c, p in others.items():
        if _slide_bfs(cur, end, blockers - {p}, wr, wd, SIZE) is not None:
            crit.append(c)
    min_set = 1 if crit else None
    pair_blockers = []
    if not crit:
        cols = list(others)
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                rm = blockers - {others[cols[a]], others[cols[b]]}
                if _slide_bfs(cur, end, rm, wr, wd, SIZE) is not None:
                    min_set = 2
                    pair_blockers.append(
                        [flags_for(cols[a]), flags_for(cols[b])])
        if min_set is None:
            min_set = 3  # >=3 (only possible answer left with 4 robots)
    out["min_blocking_set"] = min_set
    out["critical_blockers"] = [flags_for(c) for c in crit]
    out["pair_blockers"] = pair_blockers
    return out


# ---------------------------------------------------------------------------
# instrumented strict realization (same semantics as eval.realize.strict_moves)
# ---------------------------------------------------------------------------

def strict_instrumented(env, state, plan):
    wr, wd = wall_sets(env.grid_data, SIZE)
    res = {"channel": None, "strict": None, "warnings": [], "n_segs": None}
    segs, deps, err = build_segments(plan, state)
    if err:
        res["channel"] = err
        return res
    res["n_segs"] = len(segs)
    order = _topo(deps)
    if order is None:
        res["channel"] = "cyclic"
        return res
    pos = {r.color: tuple(int(x) for x in r.position) for r in state.all_robots}
    target = tuple(int(x) for x in state.target)
    tcol = state.target_robot.color
    total, executed = 0, []
    goal_arrival_step = None
    for step, i in enumerate(order):
        s = segs[i]
        color, end = s["color"], tuple(s["end"])
        cur = pos[color]
        if tuple(s["start"]) != cur:
            res["warnings"].append({"step": step, "seg": i, "color": color,
                                    "declared": list(s["start"]),
                                    "cur": list(cur)})
        blockers = frozenset(p for c, p in pos.items() if c != color)
        m = _slide_bfs(cur, end, blockers, wr, wd, SIZE)
        if m is None:
            res["channel"] = "bfs_unreachable"
            res["fail_step"] = step
            res["fail_frac"] = step / max(1, len(order) - 1)
            res["fail"] = analyze_block(plan, segs, executed, pos, s, wr, wd,
                                        state)
            return res
        total += m
        pos[color] = end
        executed.append(i)
        if color == tcol and end == target:
            goal_arrival_step = step
    if pos[tcol] != target:
        res["channel"] = "target_off_goal"
        later = [step for step, i in enumerate(order)
                 if segs[i]["color"] == tcol
                 and (goal_arrival_step is None or step > goal_arrival_step)]
        res["fail"] = {
            "final_pos": list(pos[tcol]), "target": list(target),
            "goal_arrival_step": goal_arrival_step,
            "target_segments_after_arrival": len(later),
            "n_steps": len(order),
        }
        return res
    res["channel"] = "passed"
    res["strict"] = total
    return res


# ---------------------------------------------------------------------------
# counterfactual realizers
# ---------------------------------------------------------------------------

def flexible_realize(env, state, plan, allow_detour=False, detour_slides=3):
    """Greedy work-list realizer: run any dependency-ready segment that is
    currently BFS-reachable (min index; segments that would move the target
    robot off the reached goal are deferred while alternatives exist). With
    allow_detour, a stall triggers a search for ONE robot whose <=`detour_slides`
    relocation unblocks some ready segment (robots parked on a still-needed
    support cell, and the target robot once on goal, may not be detoured).
    Returns dict(status, moves, detours, detour_moves)."""
    wr, wd = wall_sets(env.grid_data, SIZE)
    segs, deps, err = build_segments(plan, state)
    if err:
        return {"status": err, "moves": None, "detours": 0, "detour_moves": 0}
    pos = {r.color: tuple(int(x) for x in r.position) for r in state.all_robots}
    target = tuple(int(x) for x in state.target)
    tcol = state.target_robot.color
    n = len(segs)
    execd = set()
    total, detours, detour_moves = 0, 0, 0

    def ready():
        return [i for i in range(n) if i not in execd and deps[i] <= execd]

    def reach(i):
        s = segs[i]
        blockers = frozenset(p for c, p in pos.items() if c != s["color"])
        return _slide_bfs(pos[s["color"]], tuple(s["end"]), blockers, wr, wd,
                          SIZE)

    while len(execd) < n:
        cand = []
        for i in ready():
            m = reach(i)
            if m is not None:
                cand.append((i, m))
        if cand:
            # defer moving the target robot off the goal while possible
            pref = [c for c in cand
                    if not (segs[c[0]]["color"] == tcol and pos[tcol] == target
                            and tuple(segs[c[0]]["end"]) != target)]
            i, m = (pref or cand)[0]
            total += m
            pos[segs[i]["color"]] = tuple(segs[i]["end"])
            execd.add(i)
            continue
        if not ready():
            return {"status": "stall_cyclic", "moves": None,
                    "detours": detours, "detour_moves": detour_moves}
        if not allow_detour:
            return {"status": "stall", "moves": None, "detours": detours,
                    "detour_moves": detour_moves}
        # detour search: cheapest (robot, cell) making some ready segment run
        best = None  # (dmoves+segmoves, color, cell, seg, segmoves)
        for c in pos:
            p = pos[c]
            pinned = any(j not in execd and segs[j]["support"] is not None
                         and tuple(segs[j]["support"]) == p for j in range(n))
            if pinned:
                continue
            if c == tcol and p == target:
                continue
            others = frozenset(q for cc, q in pos.items() if cc != c)
            # BFS over c's slides up to detour_slides
            seen = {p: 0}
            q = deque([(p, 0)])
            while q:
                cell, d = q.popleft()
                if d >= detour_slides:
                    continue
                for dr in DIRECTIONS:
                    nxt = slide(cell, dr, others, wr, wd, SIZE)
                    if nxt == cell or nxt in seen:
                        continue
                    seen[nxt] = d + 1
                    q.append((nxt, d + 1))
            del seen[p]
            for cell, dmov in seen.items():
                trial = dict(pos)
                trial[c] = cell
                for i in ready():
                    s = segs[i]
                    if s["color"] == c:
                        start = cell
                    else:
                        start = trial[s["color"]]
                    blockers = frozenset(q2 for cc, q2 in trial.items()
                                         if cc != s["color"])
                    m = _slide_bfs(start, tuple(s["end"]), blockers, wr, wd,
                                   SIZE)
                    if m is not None and (best is None
                                          or dmov + m < best[0]):
                        best = (dmov + m, c, cell, i, m)
        if best is None:
            return {"status": "stall_multi", "moves": None, "detours": detours,
                    "detour_moves": detour_moves}
        _, c, cell, i, m = best
        dmov = best[0] - m
        pos[c] = cell
        total += dmov
        detours += 1
        detour_moves += dmov
        total += m
        pos[segs[i]["color"]] = tuple(segs[i]["end"])
        execd.add(i)
        if detours > 25:
            return {"status": "detour_limit", "moves": None,
                    "detours": detours, "detour_moves": detour_moves}
    if pos[tcol] != target:
        return {"status": "target_off_goal", "moves": None, "detours": detours,
                "detour_moves": detour_moves}
    return {"status": "fixed", "moves": total, "detours": detours,
            "detour_moves": detour_moves}


# ---------------------------------------------------------------------------
# passing-instance move replay (legality spot check)
# ---------------------------------------------------------------------------

def slide_path(start, end, blockers, wr, wd):
    """BFS shortest slide path start->end; list of (direction, stop_cell)."""
    if start == end:
        return []
    seen = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        for d in DIRECTIONS:
            nxt = slide(cur, d, blockers, wr, wd, SIZE)
            if nxt == cur or nxt in seen:
                continue
            seen[nxt] = (cur, d)
            if nxt == end:
                path, node = [], end
                while seen[node] is not None:
                    p, dd = seen[node]
                    path.append((dd, node))
                    node = p
                return list(reversed(path))
            q.append(nxt)
    return None


def replay_check(env, state, plan):
    """Realize with recorded moves, then re-verify every move with slide()."""
    wr, wd = wall_sets(env.grid_data, SIZE)
    segs, deps, err = build_segments(plan, state)
    if err:
        return {"ok": False, "why": err}
    order = _topo(deps)
    if order is None:
        return {"ok": False, "why": "cyclic"}
    pos = {r.color: tuple(int(x) for x in r.position) for r in state.all_robots}
    moves = []  # (color, direction, expected_stop)
    for i in order:
        s = segs[i]
        color, end = s["color"], tuple(s["end"])
        blockers = frozenset(p for c, p in pos.items() if c != color)
        path = slide_path(pos[color], end, blockers, wr, wd)
        if path is None:
            return {"ok": False, "why": f"unreachable seg {i}"}
        for d, cell in path:
            moves.append((color, d, cell))
        pos[color] = end
    # independent replay: full joint physics, move by move
    pos2 = {r.color: tuple(int(x) for x in r.position) for r in state.all_robots}
    for k, (color, d, cell) in enumerate(moves):
        blockers = frozenset(p for c, p in pos2.items() if c != color)
        stop = slide(pos2[color], d, blockers, wr, wd, SIZE)
        if stop == pos2[color]:
            return {"ok": False, "why": f"move {k} is a no-op (illegal)"}
        if stop != cell:
            return {"ok": False, "why": f"move {k} stop {stop} != planned {cell}"}
        pos2[color] = stop
    tcol = state.target_robot.color
    if pos2[tcol] != tuple(int(x) for x in state.target):
        return {"ok": False, "why": "target robot not on target"}
    return {"ok": True, "n_moves": len(moves),
            "moves": [[c, d, list(cell)] for c, d, cell in moves]}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(0)                      # as in eval.compare.main
    dev = "cpu"
    t_start = time.time()

    instances, sha, _ = load_instances("eval/data/bench450.jsonl")
    stored = json.loads(open("eval/results/comparison_backward.json").read())
    rows = stored["systems"]["backward subgoal planner"]["rows"]
    assert len(rows) == len(instances) == 450
    for r, ins in zip(rows, instances):
        assert r["env_id"] == ins["env_id"] and r["d_star"] == ins["d_star"]

    policy = PolicyTF.load_from_checkpoint(
        "checkpoints_backward/policy_v2.ckpt", map_location=dev).to(dev).eval()
    value = LoopedValueNet.load_from_checkpoint(
        "checkpoints_backward/value_v2.ckpt", map_location=dev).to(dev).eval()
    solver = AStar(max_iters=4000, max_frontier=40_000)

    records = []
    cur_env_id, env = None, None
    n_pass_replayed = 0
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
        plan, expansions, _rej = _nn_astar_backward(
            env, st, solver, policy, value, env_id, dev, 5, 1200,
            realize_check=None)
        rec = {"idx": idx, "env_id": env_id, "d_star": inst["d_star"],
               "stored_solved": row["solved"],
               "stored_strict": row["realized_strict"],
               "stored_abstract_cost": row["plan_cost_abstract"],
               "plan_found": plan is not None}
        if plan is None:
            rec["channel"] = "no_plan_regenerated"
            records.append(rec)
            continue
        rec["regen_abstract_cost"] = float(plan.cost())
        rec["cost_matches_stored"] = (
            abs(float(plan.cost()) - row["plan_cost_abstract"]) < 1e-6)
        diag = strict_instrumented(env, st, plan)
        rec.update({k: diag.get(k) for k in
                    ("channel", "strict", "n_segs", "fail_step", "fail_frac",
                     "fail")})
        rec["n_start_mismatch"] = len(diag["warnings"])
        rec["regen_matches_stored_outcome"] = (
            (diag["channel"] == "passed") == row["solved"])
        if not row["solved"]:
            rec["cf_flex"] = flexible_realize(env, st, plan,
                                              allow_detour=False)
            rec["cf_detour"] = flexible_realize(env, st, plan,
                                                allow_detour=True)
        elif n_pass_replayed < 3 and diag["channel"] == "passed":
            rc = replay_check(env, st, plan)
            rec["replay"] = {k: v for k, v in rc.items() if k != "moves"}
            rec["replay_moves"] = rc.get("moves")
            rec["replay_strict_equals_stored"] = (
                rc.get("n_moves") == row["realized_strict"])
            n_pass_replayed += 1
        records.append(rec)
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/450  ({time.time() - t_start:.0f}s)",
                  flush=True)

    out_path = os.path.join(SCRATCH, "diag_all450.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=1)
    print(f"wrote {out_path}  ({time.time() - t_start:.0f}s total)")

    # ---------------- aggregate ----------------
    def dbin(d):
        return "1-3" if d <= 3 else "4-6" if d <= 6 else "7-9" if d <= 9 \
            else "10+"

    fails = [r for r in records if not r["stored_solved"]]
    passes = [r for r in records if r["stored_solved"]]
    print("\n=== determinism ===")
    print("regen plan found:", sum(1 for r in records if r["plan_found"]),
          "/450")
    print("abstract cost matches stored:",
          sum(1 for r in records if r.get("cost_matches_stored")), "/450")
    print("regen outcome matches stored solved flag:",
          sum(1 for r in records if r.get("regen_matches_stored_outcome")),
          "/450")
    print("stored-failing but regen PASSED strict:",
          sum(1 for r in fails if r.get("channel") == "passed"))
    print("stored-passing but regen FAILED strict:",
          sum(1 for r in passes if r.get("channel") not in (None, "passed")))

    print("\n=== failure channels (211 stored failures) ===")
    print(Counter(r.get("channel") for r in fails))
    print("\nchannel x d* bin:")
    ct = Counter((r.get("channel"), dbin(r["d_star"])) for r in fails)
    for key in sorted(ct):
        print(f"  {key}: {ct[key]}")

    bfs = [r for r in fails if r.get("channel") == "bfs_unreachable"]
    print(f"\n=== bfs_unreachable detail (n={len(bfs)}) ===")
    print("goal segment (last hop into target):",
          sum(1 for r in bfs if r["fail"]["goal_segment"]))
    print("mover is target robot:",
          sum(1 for r in bfs if r["fail"]["mover_is_target"]))
    print("end cell occupied by another robot:",
          sum(1 for r in bfs if r["fail"]["end_occupied_by"]))
    print("walls-only reachable (pure robot blockage):",
          sum(1 for r in bfs if r["fail"]["walls_only_reachable"]))
    print("needs support and support absent:",
          sum(1 for r in bfs if r["fail"]["support_cell"]
              and not r["fail"]["support_present"]))
    print("support node unassigned in plan:",
          sum(1 for r in bfs if r["fail"]["support_cell"]
              and not r["fail"]["support_node_assigned"]))
    print("min blocking set size:",
          Counter(r["fail"]["min_blocking_set"] for r in bfs))
    print("fail position frac (step/(n-1)) quartiles:",
          sorted(round(r["fail_frac"], 2) for r in bfs)[::max(1, len(bfs)//4)])
    print("fail on FIRST executed segment:",
          sum(1 for r in bfs if r["fail_step"] == 0))
    print("fail on LAST segment:",
          sum(1 for r in bfs if r["fail_step"] == (r["n_segs"] or 1) - 1))

    cats = Counter()
    tgt_block = 0
    for r in bfs:
        anyb = r["fail"]["critical_blockers"]
        if not anyb:  # min blocking set >= 2: use members of the first pair
            pp = r["fail"].get("pair_blockers") or []
            anyb = pp[0] if pp else []
        if any(b["is_target_robot"] for b in anyb):
            tgt_block += 1
        for b in anyb:
            if not b["plan_touches"]:
                cats["bystander (plan never touches it)"] += 1
            elif b["already_moved"] and b["at_pending_support_cell"]:
                cats["plan-parked support, bounce still pending"] += 1
            elif b["already_moved"] and b["at_consumed_support_cell"]:
                cats["plan-parked support, bounce consumed (stale)"] += 1
            elif b["already_moved"]:
                cats["plan-moved, at rest/in transit"] += 1
            else:
                cats["plan will move it later (still at start)"] += 1
    print("\ncritical blocker categories (per blocker):", dict(cats))
    print("instances where a critical blocker IS the target robot:", tgt_block)

    print("\n=== start-mismatch warnings ===")
    print("failing instances with >=1 warning:",
          sum(1 for r in fails if r.get("n_start_mismatch", 0) > 0),
          f"/{len(fails)}")
    print("passing instances with >=1 warning:",
          sum(1 for r in passes if r.get("n_start_mismatch", 0) > 0),
          f"/{len(passes)}")

    print("\n=== counterfactual realizers (on stored failures) ===")
    cf1 = Counter(r["cf_flex"]["status"] for r in fails if "cf_flex" in r)
    cf2 = Counter(r["cf_detour"]["status"] for r in fails if "cf_detour" in r)
    print("flexible order only:", dict(cf1))
    print("flexible + 1-robot detour(<=3 slides):", dict(cf2))
    fx = [r for r in fails if r.get("cf_detour", {}).get("status") == "fixed"]
    if fx:
        regs = [r["cf_detour"]["moves"] - r["d_star"] for r in fx]
        print(f"detour-fixed: n={len(fx)} mean_regret="
              f"{sum(regs)/len(regs):.2f} mean_detours="
              f"{sum(r['cf_detour']['detours'] for r in fx)/len(fx):.2f}")
    fx1 = [r for r in fails if r.get("cf_flex", {}).get("status") == "fixed"]
    if fx1:
        regs = [r["cf_flex"]["moves"] - r["d_star"] for r in fx1]
        print(f"flex-fixed: n={len(fx1)} mean_regret={sum(regs)/len(regs):.2f}")
    print("\ncf_detour status x d* bin:")
    ct = Counter((r["cf_detour"]["status"], dbin(r["d_star"]))
                 for r in fails if "cf_detour" in r)
    for key in sorted(ct):
        print(f"  {key}: {ct[key]}")

    print("\n=== passing-instance replay (legality spot check) ===")
    for r in records:
        if "replay" in r:
            print(f"  idx={r['idx']} env={r['env_id']} d*={r['d_star']} "
                  f"replay={r['replay']} "
                  f"strict==stored: {r.get('replay_strict_equals_stored')}")


if __name__ == "__main__":
    main()
