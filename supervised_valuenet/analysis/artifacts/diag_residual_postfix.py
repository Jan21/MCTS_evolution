"""Residual failure diagnosis AFTER the two post-fix changes (self-support
proposal fix in skeleton/astar.py; two-phase strict realizer default in
eval/realize.py).

Regenerates the backward plan for every instance of
eval/data/bench450_first150.jsonl with the CURRENT working tree +
checkpoints_backward/{policy_v2,value_v2}.ckpt (torch.manual_seed(0), k=5,
1200 expansions, CPU), verifies determinism against the stored run
(eval/results/comparison_backward_postfix2.json), then re-executes each of
the 38 strictly-failing plans with an INSTRUMENTED copy of the CURRENT
two-phase eval.realize.strict_moves that records the terminal failure
channel, the failing segment/phase, the joint robot positions at failure and
a minimal-blocking-set analysis (who occupies the blocking cells and what
role the plan gives them).

Also re-runs the anytime search for the frontier-exhausted anytime failures
(stored expansions < 1200 in comparison_backward_postfix2_anytime.json) and
diagnoses EVERY rejected completion; budget-exhausted rows (expansions ==
1200) are classified from the stored run (a re-run adds no information and
costs ~8 CPU-minutes each).

Read-only wrt existing repo files. Writes raw per-instance records to the
scratchpad-style JSON next to this script (diag_residual_postfix_raw.json +
residual_failing_plans.pkl); the final report is built by a separate
aggregation script.
"""
import os
import sys
import json
import time
import pickle
import itertools
from collections import Counter

REPO = "/mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("DIAG_OUTDIR", HERE)
os.environ["CUDA_VISIBLE_DEVICES"] = ""           # CPU only (no GPU allowed)
os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch  # noqa: E402

torch.set_num_threads(8)

from GridEnv import GridEnv, State, Robot_at  # noqa: E402
from skeleton.astar import AStar  # noqa: E402
from train.policy_tf import PolicyTF  # noqa: E402
from train.looped_pc import LoopedValueNet  # noqa: E402
from move_planner.state import COLOR_ORDER  # noqa: E402
from eval.compare import _nn_astar_backward, load_instances  # noqa: E402
from eval.realize import (_physical_segments, _mover_source,  # noqa: E402
                          _support_node_for, _topo, _slide_bfs,
                          _two_phase_ucands, strict_moves)
from simulate import wall_sets  # noqa: E402

SIZE = 16


# ---------------------------------------------------------------------------
# segment construction (verbatim semantics of eval.realize.strict_moves)
# ---------------------------------------------------------------------------

def build_segments(plan, wr, wd):
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
        s["support_node"] = _support_node_for(g, s["edge"][0], s["edge"][1],
                                              s["support"])
    return segs, idx_by_parent, None


# ---------------------------------------------------------------------------
# instrumented copy of eval.realize._execute_schedule (identical semantics,
# plus: returns the joint positions, executed units and warnings at failure)
# ---------------------------------------------------------------------------

def execute_schedule(g, state, segs, idx_by_parent, split, wr, wd):
    units = []
    for i in range(len(segs)):
        units.append((i, 0))
        if i in split:
            units.append((i, 1))
    units.sort()
    uix = {u: k for k, u in enumerate(units)}

    def final(i):
        return uix[(i, 1)] if i in split else uix[(i, 0)]

    udeps = [set() for _ in units]
    for i, s in enumerate(segs):
        if g.nodes[s["src"]].get("ntype") in ("bottleneck", "support"):
            for j in idx_by_parent.get(s["src"], []):
                udeps[uix[(i, 0)]].add(final(j))
        if i in split:
            udeps[uix[(i, 1)]].add(uix[(i, 0)])
        sp = s["support_node"]
        if sp is not None:
            bounce = final(i)
            for j in idx_by_parent.get(sp, []):
                udeps[bounce].add(final(j))
                if i in split:
                    udeps[uix[(j, 0)]].add(uix[(i, 0)])
            for j2, s2 in enumerate(segs):
                if j2 != i and s2["src"] == sp:
                    udeps[uix[(j2, 0)]].add(bounce)

    order = _topo(udeps)
    info = {"units": units, "order": order, "executed_units": [],
            "warnings": [], "pos_at_fail": None, "final_pos": None}
    if order is None:
        return None, None, "cyclic", info

    pos = {r.color: tuple(int(x) for x in r.position) for r in state.all_robots}
    total = 0
    for k in order:
        i, phase = units[k]
        s = segs[i]
        color, end = s["color"], tuple(s["end"])
        cur = pos[color]
        blockers = frozenset(p for c, p in pos.items() if c != color)
        if phase == 0:
            if tuple(s["start"]) != cur:
                info["warnings"].append(
                    {"seg": i, "color": color,
                     "declared": list(s["start"]), "cur": list(cur)})
            if i not in split:
                m = _slide_bfs(cur, end, blockers, wr, wd, SIZE)
                if m is None:
                    info["pos_at_fail"] = {c: list(p) for c, p in pos.items()}
                    return None, i, "atomic", info
                total += m
                pos[color] = end
                info["executed_units"].append([i, phase])
                continue
            if cur == end:
                info["executed_units"].append([i, phase])
                continue
            best = None
            for u in s["ucands"]:
                m = _slide_bfs(cur, u, blockers, wr, wd, SIZE)
                if m is not None and (best is None or (m, u) < best):
                    best = (m, u)
            if best is None:
                info["pos_at_fail"] = {c: list(p) for c, p in pos.items()}
                return None, i, "approach", info
            total += best[0]
            pos[color] = best[1]
        else:
            m = _slide_bfs(cur, end, blockers, wr, wd, SIZE)
            if m is None:
                info["pos_at_fail"] = {c: list(p) for c, p in pos.items()}
                return None, i, "bounce", info
            total += m
            pos[color] = end
        info["executed_units"].append([i, phase])

    info["final_pos"] = {c: list(p) for c, p in pos.items()}
    if pos[state.target_robot.color] != tuple(int(x) for x in state.target):
        return None, None, "target", info
    return total, None, None, info


# ---------------------------------------------------------------------------
# blocking analysis at the terminal failure
# ---------------------------------------------------------------------------

def analyze_block(state, segs, split, info, fail_seg, kind, wr, wd):
    """Who prevents the failing unit's mover from reaching its destination."""
    s = segs[fail_seg]
    pos = {c: tuple(p) for c, p in info["pos_at_fail"].items()}
    color, cur = s["color"], pos[s["color"]]
    # destination cells of the failing unit: the segment end for atomic/bounce
    # units, the certified pre-bounce cells for a failed approach leg
    targets = ([tuple(u) for u in s.get("ucands", [])] if kind == "approach"
               else [tuple(s["end"])])
    sc = tuple(s["support"]) if s["support"] is not None else None
    others = {c: p for c, p in pos.items() if c != color}
    blockers = frozenset(others.values())

    def reach(bl):
        return any(_slide_bfs(cur, t, bl, wr, wd, SIZE) is not None
                   for t in targets)

    executed = info["executed_units"]
    final_done = set()          # segments whose final (support-consuming) unit ran
    any_done_color = Counter()
    for i, phase in executed:
        any_done_color[segs[i]["color"]] += 1
        if (i in split and phase == 1) or (i not in split and phase == 0):
            final_done.add(i)

    def flags_for(c):
        p = others[c]
        own = [j for j, sj in enumerate(segs) if sj["color"] == c]
        return {
            "color": c, "pos": list(p),
            "is_target_robot": c == state.target_robot.color,
            "on_goal": (c == state.target_robot.color
                        and p == tuple(int(x) for x in state.target)),
            "plan_touches": bool(own),
            "already_moved": any_done_color[c] > 0,
            "has_pending_own_segments": any(j not in final_done for j in own),
            "at_pending_support_cell": any(
                j not in final_done and segs[j]["support"] is not None
                and tuple(segs[j]["support"]) == p for j in range(len(segs))),
            "at_consumed_support_cell": any(
                j in final_done and segs[j]["support"] is not None
                and tuple(segs[j]["support"]) == p for j in range(len(segs))),
        }

    out = {
        "mover": color,
        "mover_is_target": color == state.target_robot.color,
        "cur": list(cur), "declared_start": list(s["start"]),
        "targets": [list(t) for t in targets],
        "end": list(s["end"]),
        "support_cell": list(sc) if sc else None,
        "support_node_assigned": s.get("support_node") is not None,
        "support_present": sc is not None and sc in blockers,
        "end_occupied_by": next(
            (c for c, p in others.items() if p == tuple(s["end"])), None),
        "walls_only_reachable": reach(frozenset()),
        "support_only_reachable":
            (reach(frozenset({sc})) if sc is not None else None),
        "start_mismatch_at_fail": tuple(s["start"]) != cur,
    }

    crit = [c for c, p in others.items() if reach(blockers - {p})]
    min_set, pair_blockers = (1 if crit else None), []
    if not crit:
        cols = list(others)
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                rm = blockers - {others[cols[a]], others[cols[b]]}
                if reach(rm):
                    min_set = 2
                    pair_blockers.append([flags_for(cols[a]),
                                          flags_for(cols[b])])
        if min_set is None:
            min_set = 3
    out["min_blocking_set"] = min_set
    out["critical_blockers"] = [flags_for(c) for c in crit]
    out["pair_blockers"] = pair_blockers
    return out


# ---------------------------------------------------------------------------
# instrumented two-phase strict realization (mirrors eval.realize.strict_moves
# with two_phase=True, the current default)
# ---------------------------------------------------------------------------

def strict_instrumented(env, state, plan):
    wr, wd = wall_sets(env.grid_data, SIZE)
    g = plan.g
    segs, idx_by_parent, err = build_segments(plan, wr, wd)
    if err:
        return {"channel": err}
    res = {"n_segs": len(segs),
           "segs": [{"edge": list(s["edge"]), "start": list(s["start"]),
                     "end": list(s["end"]),
                     "support": (list(s["support"]) if s["support"] is not None
                                 else None),
                     "color": s["color"], "src": s["src"]} for s in segs]}
    split = set()
    attempts = 0
    while True:
        total, fail_seg, kind, info = execute_schedule(
            g, state, segs, idx_by_parent, split, wr, wd)
        attempts += 1
        if kind is None:
            res.update({"channel": "passed", "strict": total,
                        "n_split": len(split), "attempts": attempts})
            return res
        if kind == "cyclic":
            res.update({"channel": "cyclic", "attempts": attempts})
            return res
        if kind == "target":
            res.update({"channel": "target_off_goal", "attempts": attempts,
                        "final_pos": info["final_pos"]})
            return res
        if kind == "atomic":
            s = segs[fail_seg]
            if s["support"] is None:
                terminal = "atomic_unsupported"
            else:
                if "ucands" not in s:
                    s["ucands"] = _two_phase_ucands(env, s["end"],
                                                    s["support"], wr, wd, SIZE)
                if s["ucands"]:
                    split.add(fail_seg)     # retry two-phased, like realize.py
                    continue
                terminal = "atomic_no_ucands"
        else:
            terminal = kind                 # "approach" | "bounce"
        res.update({
            "channel": "bfs_unreachable", "terminal_kind": terminal,
            "attempts": attempts, "n_split": len(split),
            "fail_seg": fail_seg,
            "fail_edge": list(segs[fail_seg]["edge"]),
            "fail_unit_index": len(info["executed_units"]),
            "n_units": len(info["units"]),
            "warnings": info["warnings"],
            "pos_at_fail": info["pos_at_fail"],
            "fail": analyze_block(state, segs, split, info, fail_seg,
                                  "approach" if terminal == "approach"
                                  else "end", wr, wd),
        })
        return res


# ---------------------------------------------------------------------------
# static self-support audit (pass-3 detector; the proposal bug is fixed, this
# verifies the fix holds on regenerated plans)
# ---------------------------------------------------------------------------

def self_support_subgoals(plan):
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


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def make_state(inst):
    positions = [tuple(p) for p in inst["positions"]]
    tidx = inst["target_idx"]
    return State(
        target=tuple(inst["target"]),
        target_robot=Robot_at(position=positions[tidx],
                              color=COLOR_ORDER[tidx]),
        helpers=[Robot_at(position=positions[j], color=COLOR_ORDER[j])
                 for j in range(len(positions)) if j != tidx])


def main():
    torch.manual_seed(0)                      # as in eval.compare.main
    dev = "cpu"
    t0 = time.time()

    instances, sha, _ = load_instances("eval/data/bench450_first150.jsonl")
    stored = json.loads(
        open("eval/results/comparison_backward_postfix2.json").read())
    rows = stored["systems"]["backward subgoal planner"]["rows"]
    stored_any = json.loads(
        open("eval/results/comparison_backward_postfix2_anytime.json").read())
    rows_any = stored_any["systems"][
        "backward subgoal planner (anytime realization-checked)"]["rows"]
    assert len(rows) == len(instances) == len(rows_any) == 150
    for r, ra, ins in zip(rows, rows_any, instances):
        assert r["env_id"] == ins["env_id"] and r["d_star"] == ins["d_star"]
        assert ra["env_id"] == ins["env_id"]

    policy = PolicyTF.load_from_checkpoint(
        "checkpoints_backward/policy_v2.ckpt", map_location=dev).to(dev).eval()
    value = LoopedValueNet.load_from_checkpoint(
        "checkpoints_backward/value_v2.ckpt", map_location=dev).to(dev).eval()
    solver = AStar(max_iters=4000, max_frontier=40_000)

    # ---------------- pass 1: plain mode, all 150, determinism + diagnosis ---
    records, plans = [], {}
    cur_env_id, env = None, None
    for idx, (row, inst) in enumerate(zip(rows, instances)):
        env_id = inst["env_id"]
        if env_id != cur_env_id:
            env, _ = GridEnv.from_env(env_id)
            cur_env_id = env_id
        st = make_state(inst)
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
        # current-repo strict realizer (two-phase default), as the benchmark ran
        stx = strict_moves(env, st, plan, log=None)
        rec["regen_strict"] = stx
        rec["regen_matches_stored_outcome"] = (
            (stx is not None) == row["solved"])
        rec["n_self_support_subgoals"] = len(self_support_subgoals(plan))
        if not row["solved"]:
            diag = strict_instrumented(env, st, plan)
            rec["diag"] = diag
            plans[idx] = plan
        records.append(rec)
        if (idx + 1) % 25 == 0:
            print(f"  plain {idx + 1}/150  ({time.time() - t0:.0f}s)",
                  flush=True)

    # ---------------- pass 2: anytime failures ------------------------------
    # frontier-exhausted rows (stored expansions < 1200): cheap deterministic
    # re-run, diagnosing every rejected completion. budget-exhausted rows
    # (expansions == 1200): classified from the stored run.
    any_records = []
    cur_env_id, env = None, None
    for idx, (ra, inst) in enumerate(zip(rows_any, instances)):
        if ra["solved"]:
            continue
        arec = {"idx": idx, "env_id": inst["env_id"], "d_star": inst["d_star"],
                "stored_expansions": ra["expansions"],
                "stored_rejected": ra["plans_rejected"],
                "stored_plan_cost": ra["plan_cost_abstract"],
                "exit": ("budget_exhausted" if ra["expansions"] >= 1200
                         else "frontier_exhausted")}
        if ra["expansions"] < 1200:
            env_id = inst["env_id"]
            if env_id != cur_env_id:
                env, _ = GridEnv.from_env(env_id)
                cur_env_id = env_id
            st = make_state(inst)
            rejected_diags = []

            def realize_check(p, _env=env, _st=st, _sink=rejected_diags):
                m = strict_moves(_env, _st, p, log=None)
                if m is None:
                    d = strict_instrumented(_env, _st, p)
                    _sink.append({
                        "plan_cost": float(p.cost()),
                        "channel": d.get("channel"),
                        "terminal_kind": d.get("terminal_kind"),
                        "fail_edge": d.get("fail_edge"),
                        "fail": d.get("fail"),
                        "n_self_support": len(self_support_subgoals(p)),
                    })
                return m is not None

            plan, expansions, rejected = _nn_astar_backward(
                env, st, solver, policy, value, env_id, dev, 5, 1200,
                realize_check=realize_check)
            arec.update({
                "rerun_expansions": expansions,
                "rerun_rejected": rejected,
                "rerun_matches_stored": (rejected == ra["plans_rejected"]
                                         and expansions == ra["expansions"]),
                "rerun_returned_plan": plan is not None,
                "rejected_completions": rejected_diags,
            })
        any_records.append(arec)
        print(f"  anytime idx={idx} exit={arec['exit']} "
              f"({time.time() - t0:.0f}s)", flush=True)

    with open(os.path.join(OUTDIR, "diag_residual_postfix_raw.json"),
              "w") as f:
        json.dump({"plain": records, "anytime": any_records}, f, indent=1)
    with open(os.path.join(OUTDIR, "residual_failing_plans.pkl"), "wb") as f:
        pickle.dump(plans, f)

    # ---------------- quick console summary --------------------------------
    fails = [r for r in records if not r["stored_solved"]]
    print("\n=== determinism (plain, n=150) ===")
    print("plan regenerated:", sum(1 for r in records if r["plan_found"]))
    print("abstract cost matches stored:",
          sum(1 for r in records if r.get("cost_matches_stored")))
    print("strict outcome matches stored:",
          sum(1 for r in records if r.get("regen_matches_stored_outcome")))
    print("\n=== plain failures (n={}) ===".format(len(fails)))
    print("terminal:", Counter(
        (r.get("diag", {}).get("channel"),
         r.get("diag", {}).get("terminal_kind")) for r in fails))
    print("self-support subgoals in failing plans:",
          sum(1 for r in fails if r.get("n_self_support_subgoals", 0) > 0))
    print("\n=== anytime failures (n={}) ===".format(len(any_records)))
    print("exit:", Counter(a["exit"] for a in any_records))
    print("rerun matches stored:",
          [(a["idx"], a.get("rerun_matches_stored")) for a in any_records
           if "rerun_matches_stored" in a])
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
