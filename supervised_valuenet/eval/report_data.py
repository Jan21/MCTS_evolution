"""Load every result file and normalize it into the report's data model.

Everything rendered on the page comes from here; nothing is hardcoded
downstream.  Missing files surface as None (rendered as pending rows), so the
build never crashes when an experiment has not landed yet.

To add a future result (e.g. the B2-retrained networks), drop the file into
place and — if its name is already in FUTURE_SLOTS / RUNGS below — rebuild.
"""

import glob
import math
import os
import pickle
import sys
from collections import Counter, deque

from eval.report_util import (ROOT, rp, load_json, load_jsonl, SOURCES,
                              source_note, fact)

# ---------------------------------------------------------------------------
# Generic pickers for comparison-shaped files ({protocol, systems})
# ---------------------------------------------------------------------------


def systems_of_kind(comp, kind):
    if not comp or "systems" not in comp:
        return []
    return [(name, s) for name, s in comp["systems"].items()
            if s.get("kind") == kind and s.get("aggregate")]


def pick(comp, kind):
    """First system of `kind` with a non-empty aggregate, else None."""
    xs = systems_of_kind(comp, kind)
    return xs[0][1]["aggregate"] if xs else None


def pick_name(comp, kind):
    xs = systems_of_kind(comp, kind)
    return xs[0][0] if xs else None


def proto_date(comp):
    return (comp or {}).get("protocol", {}).get("date") or ""


KAROLINA_SINCE = "2026-07-20"


def machine_of(comp):
    """Which machine produced this file: 'karolina' or 'origin'.

    Derived from the run date in the file's own protocol block — the study
    moved to the Karolina cluster on 2026-07-20 (FINDINGS §16).  Wall-clock
    seconds are only comparable between files from the same machine.
    """
    d = proto_date(comp)
    return "karolina" if d[:10] >= KAROLINA_SINCE else "origin"


# ---------------------------------------------------------------------------
# The scaling ladder — every configuration the study runs, in order.
# Each slot names the file and which system inside it to read.  A missing
# file renders as a pending row.  `future` slots are expected results that
# render as "retraining in progress" until their file lands.
# ---------------------------------------------------------------------------

RUNGS = [
    {
        "key": "g16r4", "label": "16×16 board, 4 robots", "short": "16×16 · 4r",
        "base": True,  # files live under eval/results, handled separately
    },
    {
        "key": "g16r6", "label": "16×16 board, 6 robots", "short": "16×16 · 6r",
        "graded": {
            "bwd_old": ("comparison.json", "backward"),
            "bwd_b1": ("comparison_b1.json", "backward"),
            "bwd_b2": ("comparison_b2.json", "backward"),
            "fwd": ("comparison_forward_control.json", "forward"),
            "fwd_first": ("comparison.json", "forward"),
        },
        "frontier": {
            "bwd_old": ("comparison_ungraded.json", "backward"),
            "bwd_b1": ("comparison_ungraded_b1.json", "backward"),
            "bwd_b2": ("comparison_ungraded_b2.json", "backward"),
            "fwd": ("comparison_ungraded.json", "forward"),
        },
        "future": {"bwd_retrained": "comparison_b2retrained_cap20000.json",
                   "bwd_retrained_deficient": "comparison_b2retrained.json",
                   "bwd_retrained_frontier":
                       "comparison_ungraded_b2retrained_cap20000.json",
                   "bwd_retrained_frontier_deficient":
                       "comparison_ungraded_b2retrained.json"},
        "fwd_note": ("stability-controlled retrain (the stock recipe "
                     "destabilized; see the training-fragility section)"),
    },
    {
        "key": "g16r8", "label": "16×16 board, 8 robots", "short": "16×16 · 8r",
        "graded": {
            "bwd_old": ("comparison.json", "backward"),
            "bwd_b2": ("comparison_b2.json", "backward"),
            "fwd": ("comparison_forward_control.json", "forward"),
        },
        # the collapsed first forward run: loaded ONLY to assert it is never
        # rendered as a comparable result
        "fwd_withheld": ("comparison.json", "forward"),
        "frontier": {
            "bwd_old": ("comparison_ungraded.json", "backward"),
            "bwd_b2": ("comparison_ungraded_b2.json", "backward"),
            "fwd": ("comparison_ungraded.json", "forward"),
        },
        "future": {"bwd_retrained": "comparison_b2retrained_cap20000.json",
                   "bwd_retrained_deficient": "comparison_b2retrained.json",
                   "bwd_retrained_frontier":
                       "comparison_ungraded_b2retrained_cap20000.json",
                   "bwd_retrained_frontier_deficient":
                       "comparison_ungraded_b2retrained.json"},
        "fwd_note": ("stability-controlled retrain (the stock recipe "
                     "collapsed to near-random; see training fragility)"),
    },
    {
        "key": "g24r4", "label": "24×24 board, 4 robots", "short": "24×24 · 4r",
        "graded": {
            "bwd_old": ("comparison.json", "backward"),
            "bwd_b2": ("comparison_b2.json", "backward"),   # not yet measured
            "fwd": ("comparison.json", "forward"),
        },
        "frontier": {
            # no old-language frontier row was ever measured at this rung;
            # the 2026-07-29 chunked lane (g24r4_rows.slurm) produced the
            # first frontier file, carrying BOTH systems -- the forward
            # frontier cell reads from it, not from a comparison_ungraded.json
            # that never existed.
            "bwd_old": ("comparison_ungraded.json", "backward"),
            "bwd_b2": ("comparison_ungraded_b2.json", "backward"),
            "fwd": ("comparison_ungraded_b2.json", "forward"),
        },
        "future": {"bwd_retrained": "comparison_b2retrained_cap20000.json",
                   "bwd_retrained_deficient": "comparison_b2retrained.json",
                   "bwd_retrained_frontier":
                       "comparison_ungraded_b2retrained_cap20000.json",
                   "bwd_retrained_frontier_deficient":
                       "comparison_ungraded_b2retrained.json"},
        "fwd_note": "trained with the stock recipe (stable at this size)",
    },
    {
        "key": "g24r8", "label": "24×24 board, 8 robots", "short": "24×24 · 8r",
        "graded": {
            "bwd_old": ("comparison.json", "backward"),
            "bwd_b2": ("comparison_b2.json", "backward"),
            "fwd": ("comparison.json", "forward"),
        },
        "frontier": {
            "bwd_old": ("comparison_ungraded.json", "backward"),
            "bwd_b2": ("comparison_ungraded_b2.json", "backward"),
            "fwd": ("comparison_ungraded.json", "forward"),
        },
        "future": {"bwd_retrained": "comparison_b2retrained_cap20000.json",
                   "bwd_retrained_deficient": "comparison_b2retrained.json",
                   "bwd_retrained_frontier":
                       "comparison_ungraded_b2retrained_cap20000.json",
                   "bwd_retrained_frontier_deficient":
                       "comparison_ungraded_b2retrained.json"},
        "fwd_note": ("stability-controlled retrain (the stock recipe "
                     "collapsed; see training fragility)"),
    },
    {
        "key": "g32r4", "label": "32×32 board, 4 robots", "short": "32×32 · 4r",
        "graded": {
            "bwd_old": ("comparison.json", "backward"),
            "bwd_b2": ("comparison_b2.json", "backward"),
            "fwd": ("comparison.json", "forward"),
        },
        "frontier": {
            "bwd_old": ("comparison_ungraded.json", "backward"),
            "bwd_b2": ("comparison_ungraded_b2.json", "backward"),
            "fwd": ("comparison_ungraded.json", "forward"),
        },
        "future": {"bwd_retrained": "comparison_b2retrained_cap20000.json",
                   "bwd_retrained_deficient": "comparison_b2retrained.json",
                   "bwd_retrained_frontier":
                       "comparison_ungraded_b2retrained_cap20000.json",
                   "bwd_retrained_frontier_deficient":
                       "comparison_ungraded_b2retrained.json"},
        "fwd_note": ("stability-controlled retrain (the stock recipe "
                     "collapsed; see training fragility)"),
    },
]

# base-scale future slots (eval/results/...)
# Corpus-correct rows FIRST. The cap-5000 corpus stripped by-reference
# candidates and cost 22.4 points at the beyond-oracle set (FINDINGS 36), so a
# row trained on it is not the method's performance and is withheld -- the
# cell renders as pending until the cap-20000 base corpus lands.
BASE_FUTURE = {
    "bwd_retrained": ["final450_backward_b2_retrained_cap20000.json"],
}

# compute-fairness inputs (the "Is the comparison fair?" tab). Each entry:
# (D key, relpath, note shown in provenance while the file is absent)
FAIRNESS_FILES = [
    ("budget_by_rung", "eval/results/budget_curves_by_rung.json",
     "being reconstructed from archived per-instance expansion counts "
     "(main session)"),
    ("compute_accounting", "eval/results/compute_accounting.json",
     "measurement queued: instrumented counters (NN calls by head, physics "
     "slides in realization/prefix/park checks, free-fix expansions)"),
    ("fwd_probe_g24r8", "scaling/results/g24r8/forward_probe_e6000.json",
     "measurement queued: extended-budget (6,000-step) forward run on a "
     "frontier subsample"),
    ("fwd_probe_g32r4", "scaling/results/g32r4/forward_probe_e4800.json",
     "measurement queued: extended-budget (4,800-step) forward run on a "
     "frontier subsample"),
    ("fwd_probe_g16r6", "scaling/results/g16r6/forward_probe_e4800.json",
     "measurement queued: extended-budget (4,800-step) forward run on a "
     "frontier subsample"),
    ("fwd_probe_g16r8", "scaling/results/g16r8/forward_probe_e4800.json",
     "measurement queued: extended-budget (4,800-step) forward run on a "
     "frontier subsample"),
    ("budget_probe_summary", "eval/results/budget_probe_summary.json",
     "generated by eval/budget_probe_summary.py once a probe lands"),
    ("budget_table", "eval/results/budget_table.json",
     "generated by eval/budget_table.py (forward-vs-backward solve rate across "
     "the whole budget range, per rung)"),
    ("stats_tests", "eval/results/stats_tests.json",
     "generated by eval/stats_tests.py (paired McNemar + board-clustered "
     "bootstrap over every head-to-head cell)"),
    ("frontier_strata", "analysis/artifacts/frontier_strata.json",
     "generated by analysis/frontier_strata.py (oracle-independent hardness "
     "stratification of the frontier sets)"),
    ("dedup_audit", "analysis/artifacts/dedup_audit.json",
     "generated by analysis/dedup_audit.py (wall-layout near-duplicate audit "
     "across train/val/bench)"),
    ("data_budget", "analysis/artifacts/data_budget.json",
     "generated by analysis/data_budget.py (training records per system per "
     "configuration)"),
]


def load_rung_files(rung):
    """Load every distinct file a rung references; returns relpath->data."""
    files = {}
    slots = {}
    for group in ("graded", "frontier"):
        for slot, (fname, kind) in (rung.get(group) or {}).items():
            rel = os.path.join("scaling", "results", rung["key"], fname)
            if rel not in files:
                files[rel] = load_json(rel)
            slots[(group, slot)] = (rel, files[rel], kind)
    for slot, fname in (rung.get("future") or {}).items():
        rel = os.path.join("scaling", "results", rung["key"], fname)
        if os.path.exists(rp(rel)):
            data = load_json(rel)
        else:
            data = None
            SOURCES.setdefault(rel, {
                "status": "missing",
                "note": "expected from the B2 retraining in progress"})
        slots[("future", slot)] = (rel, data, "backward")
    if rung.get("fwd_withheld"):
        fname, kind = rung["fwd_withheld"]
        rel = os.path.join("scaling", "results", rung["key"], fname)
        if rel not in files:
            files[rel] = load_json(rel)
        slots[("withheld", "fwd")] = (rel, files[rel], kind)
    return slots


def cell(slots, group, slot):
    """-> dict(agg=..., src=..., machine=..., date=..., name=...) or None."""
    got = slots.get((group, slot))
    if not got:
        return None
    rel, comp, kind = got
    if not comp:
        return None
    agg = pick(comp, kind)
    if not agg:
        return None
    return {"agg": agg, "src": rel, "machine": machine_of(comp),
            "date": proto_date(comp)[:10], "name": pick_name(comp, kind)}


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def probe_counts(probe):
    """Counter of category values in a ceiling-probe list."""
    if not isinstance(probe, list):
        return None
    return Counter(r.get("category") for r in probe)


# ---------------------------------------------------------------------------
# Worked example (ported from the previous generator; verified logic)
# ---------------------------------------------------------------------------

DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def _slide(pos, d, blockers, wr, wd, size):
    x, y = pos
    dx, dy = DIRS[d]
    while True:
        if d == "up" and (y == 0 or (x, y - 1) in wd):
            return (x, y)
        if d == "down" and (y == size - 1 or (x, y) in wd):
            return (x, y)
        if d == "left" and (x == 0 or (x - 1, y) in wr):
            return (x, y)
        if d == "right" and (x == size - 1 or (x, y) in wr):
            return (x, y)
        nxt = (x + dx, y + dy)
        if nxt in blockers:
            return (x, y)
        x, y = nxt


def _bfs_solve(start, target_idx, goal, wr, wd, size, node_cap=300000):
    q = deque([(start, ())])
    seen = {start}
    while q:
        st, path = q.popleft()
        if st[target_idx] == goal:
            return path
        if len(seen) > node_cap:
            return None
        for i in range(len(st)):
            blockers = set(st) - {st[i]}
            for d in DIRS:
                np_ = _slide(st[i], d, blockers, wr, wd, size)
                if np_ == st[i]:
                    continue
                ns = list(st)
                ns[i] = np_
                ns = tuple(ns)
                if ns in seen:
                    continue
                seen.add(ns)
                q.append((ns, path + ((i, d, st[i], np_),)))
    return None


def worked_example(probe_rows, want_idx=333):
    """The idx-333 puzzle: real board, real optimum (recomputed at build
    time), and the exhaustive probe's verdict for the same puzzle."""
    insts = load_json("analysis/artifacts/ceiling_probe_instances.json")
    if not insts or not probe_rows:
        return None
    inst = next((i for i in insts if i.get("idx") == want_idx), None)
    verdict = next((r for r in probe_rows if r.get("idx") == want_idx), None)
    if inst is None or verdict is None:
        return None
    env_rel = os.path.join("environments", f"env_{inst['env_id']}.pkl")
    try:
        sys.path.insert(0, ROOT)
        from simulate import wall_sets  # stdlib-only module
        with open(rp(env_rel), "rb") as f:
            env = pickle.load(f)
        wr, wd = wall_sets(env["grid_data"])
        size = int(math.isqrt(len(env["grid_data"])))
    except Exception:
        return None
    SOURCES.setdefault(env_rel, {"status": "ok", "note": ""})
    source_note(env_rel, "board walls for the worked example")
    start = tuple(tuple(p) for p in inst["positions"])
    goal = tuple(inst["target"])
    tidx = inst["target_idx"]
    sol = _bfs_solve(start, tidx, goal, wr, wd, size)
    if sol is None:
        return None
    conflict = None
    li, ld, lfrm, lto = sol[-1]
    dx, dy = DIRS[ld]
    stop_cell = (lto[0] + dx, lto[1] + dy)
    pos = list(start)
    for (i, d, frm, to) in sol[:-1]:
        pos[i] = to
    stopper_slot = next((j for j, p in enumerate(pos)
                         if tuple(p) == stop_cell), None)
    passed_before = None
    for step_no, (i, d, frm, to) in enumerate(sol[:-1], start=1):
        if i != tidx:
            continue
        ddx, ddy = DIRS[d]
        x, y = frm
        while (x, y) != to:
            x, y = x + ddx, y + ddy
            if (x, y) == stop_cell:
                passed_before = step_no
                break
    stopper_step = None
    stopper_stop = None
    if stopper_slot is not None:
        run = list(start)
        for step_no, (i, d, frm, to) in enumerate(sol, start=1):
            run[i] = to
            if i == stopper_slot and tuple(to) == stop_cell:
                stopper_step = step_no
                sdx, sdy = DIRS[d]
                cell2 = (to[0] + sdx, to[1] + sdy)
                occ = list(start)
                for (i2, d2, frm2, to2) in sol[:step_no - 1]:
                    occ[i2] = to2
                blocker = next((j for j, p in enumerate(occ)
                                if tuple(p) == cell2 and j != i), None)
                if blocker is not None and 0 <= cell2[0] < size \
                        and 0 <= cell2[1] < size:
                    stopper_stop = (blocker, cell2)
                break
    if stopper_slot is not None and passed_before is not None:
        conflict = {"cell": stop_cell, "stopper_slot": stopper_slot,
                    "stopper_step": stopper_step,
                    "passed_step": passed_before,
                    "stopper_stop": stopper_stop}
    return {"inst": inst, "verdict": verdict, "walls": (wr, wd),
            "size": size, "solution": sol, "conflict": conflict,
            "env_rel": env_rel}


# ---------------------------------------------------------------------------
# collect() — the one entry point
# ---------------------------------------------------------------------------

def collect():
    D = {}

    # by-reference share of every merged B2 label corpus (the dose in the
    # label-budget dose-response; analysis/byref_shares.py)
    D["byref_shares"] = load_json("analysis/artifacts/byref_shares.json")
    # instrumented census of by-reference candidate supply at eval time
    # (FINDINGS 40; analysis/byref_topk_ablation.py, job 4599947)
    D["byref_topk_ablation"] = load_json(
        "analysis/artifacts/byref_topk_ablation.json")
    # seed-robustness summary (analysis/seed_spread.py; seeds 21/37/53)
    D["seed_spread"] = load_json("analysis/artifacts/seed_spread.json")
    # value-net mode diagnostic (analysis/valnet_modes.py; FINDINGS 44)
    D["valnet_modes"] = load_json("analysis/artifacts/valnet_modes.json")
    # failure gallery for the hardest measured pool (analysis/failure_examples.py)
    D["failure_examples"] = load_json("analysis/artifacts/failure_examples.json")

    # ---- base scale (16×16, 4 robots) ------------------------------------
    D["fwd450"] = load_json("eval/results/comparison_forward.json")
    D["bwd_before"] = load_json("eval/results/comparison_backward.json")
    D["bwd_fixed"] = load_json("eval/results/final450_backward_plain.json")
    D["bwd_any"] = load_json("eval/results/final450_backward_anytime.json")
    D["bwd_prefix"] = load_json("eval/results/final450_backward_prefix.json")
    D["bwd_b1"] = load_json("eval/results/final450_backward_b1.json")
    D["bwd_b2"] = load_json("eval/results/final450_backward_b2.json")
    D["bwd_heuristic"] = load_json(
        "eval/results/final450_backward_heuristic_baseline.json")
    D["bwd_retrained"] = None
    for fname in BASE_FUTURE["bwd_retrained"]:
        rel = "eval/results/" + fname
        if os.path.exists(rp(rel)):
            D["bwd_retrained"] = load_json(rel)
            D["bwd_retrained_src"] = rel
            break
    else:
        rel = "eval/results/" + BASE_FUTURE["bwd_retrained"][0]
        SOURCES.setdefault(rel, {
            "status": "missing",
            "note": "expected from the B2 retraining in progress"})

    # matched-150 fixes ladder
    D["postfix1"] = load_json("eval/results/comparison_backward_postfix.json")
    D["postfix2"] = load_json("eval/results/comparison_backward_postfix2.json")
    D["postfix3"] = load_json("eval/results/comparison_backward_postfix3.json")
    D["reorder_ab"] = load_json("eval/results/realizer_reorder_ab.json")
    D["twophase_ab"] = load_json("eval/results/realizer_twophase_ab.json")
    # the 150-slice of the pre-fix 450 run (same puzzles the postfix files use)
    rows450 = None
    if D["bwd_before"]:
        s = pick(D["bwd_before"], "backward")
        sysd = D["bwd_before"]["systems"].get("backward subgoal planner", {})
        rows450 = sysd.get("rows")
    if rows450 and len(rows450) >= 150:
        D["prefix150_slice_solved"] = sum(
            1 for r in rows450[:150] if r.get("solved"))
    else:
        D["prefix150_slice_solved"] = None

    D["prefix_ab"] = load_json("eval/results/prefix_check_ab.json")
    D["budget"] = load_json("eval/results/budget_curves.json")

    # self-play
    D["prefix150"] = load_json("eval/results/prefix150_prefix.json")
    D["arm5"] = load_json("eval/results/arm_prefix_iter5.json")
    D["arms"] = load_json("subgoal_selfplay/arms_index.json")
    D["selfplay_iters"] = []
    for i in range(0, 12):
        rel = f"subgoal_selfplay/runs_warm_prefix/iter{i}_stats.json"
        if not os.path.exists(rp(rel)):
            break
        D["selfplay_iters"].append(load_json(rel))

    # ceiling probes
    D["probe_base"] = load_json("analysis/artifacts/ceiling_probe_results.json")
    D["probe_b1"] = load_json("analysis/artifacts/ceiling_probe_results_b1.json")
    D["probe_b2"] = load_json("analysis/artifacts/ceiling_probe_results_b2.json")
    D["probe_b2_deep"] = load_json(
        "analysis/artifacts/ceiling_probe_results_b2_deep.json")
    D["probe_g16r6_old"] = load_json(
        "scaling/results/g16r6/ceiling_probe_old_vocab.json")
    D["probe_g16r6_b1"] = load_json("scaling/results/g16r6/ceiling_probe_b1.json")
    D["probe_g16r6_b2"] = load_json("scaling/results/g16r6/ceiling_probe_b2.json")
    D["cap_sens_base"] = load_json("analysis/artifacts/cap_sensitivity_base.json")
    D["cap_sens_g16r6"] = load_json("scaling/results/g16r6/cap_sensitivity.json")
    D["b2_solved40"] = load_json("analysis/artifacts/b2_solved40_results.json")
    D["recheck_default"] = load_json(
        "analysis/artifacts/ceiling_probe_default_recheck_post_b2.json")
    D["recheck_b1"] = load_json(
        "analysis/artifacts/ceiling_probe_b1_recheck_post_b2.json")
    D["realizer_b1_ab"] = load_json("eval/results/realizer_b1_ab.json")
    D["realizer_b2_ab"] = load_json("eval/results/realizer_b2_ab.json")

    # forward runs on the language-failure puzzles
    D["fwd_gallery_base"] = load_json(
        "analysis/failure_gallery/forward_solutions_base.json")
    D["fwd_gallery_g16r6"] = load_json(
        "analysis/failure_gallery/forward_solutions_g16r6.json")

    # plan structures + worked example
    D["plan_structs"] = load_json("eval/results/plan_structures_data.json")
    D["worked"] = worked_example(D["probe_base"])

    # benchmark metadata + oracle-death curve
    D["bench450_meta"] = load_json("eval/data/bench450.jsonl.meta.json")
    D["metas"] = {}
    for r in RUNGS:
        if r.get("base"):
            continue
        rel = os.path.join("scaling", "data", r["key"], "bench.jsonl.meta.json")
        D["metas"][r["key"]] = load_json(rel)
    D["cap_probe"] = load_jsonl("scaling/data/g16r8/cap_probe_10x_results.jsonl")

    # scaling rungs
    D["rungs"] = {}
    for r in RUNGS:
        if r.get("base"):
            continue
        D["rungs"][r["key"]] = load_rung_files(r)

    # compute-fairness inputs (each renders as a pending hook until it lands)
    for key, rel, note in FAIRNESS_FILES:
        if os.path.exists(rp(rel)):
            D[key] = load_json(rel)
        else:
            D[key] = None
            SOURCES.setdefault(rel, {"status": "missing", "note": note})

    # cross-file protocol sanity: every comparison file at the shared budget
    protos = []
    for key, slots in D["rungs"].items():
        for (group, slot), (rel, comp, kind) in slots.items():
            if comp and comp.get("protocol"):
                protos.append((rel, comp["protocol"]))
    for k in ("fwd450", "bwd_before", "bwd_prefix", "bwd_b1", "bwd_b2"):
        if D.get(k):
            protos.append((k, D[k]["protocol"]))
    bad = [rel for rel, p in protos
           if p.get("expansions") != 1200 or p.get("k") != 5]
    fact("every comparison file uses the shared budget "
         "(1200 search steps, top-5 proposals)", not bad)

    D["ladder"] = build_ladder(D)
    return D


def _base_cell(comp, kind="backward"):
    if not comp:
        return None
    agg = pick(comp, kind)
    if not agg:
        return None
    return {"agg": agg, "src": None, "machine": machine_of(comp),
            "date": proto_date(comp)[:10], "name": pick_name(comp, kind)}


def build_ladder(D):
    """Uniform per-rung view: for every rung, graded + frontier cells for
    bwd_old / bwd_b1 / bwd_b2 / bwd_retrained / fwd (None where missing)."""
    out = []
    for r in RUNGS:
        entry = {"key": r["key"], "label": r["label"], "short": r["short"],
                 "fwd_note": r.get("fwd_note", ""), "base": r.get("base", False),
                 "graded": {}, "frontier": {}}
        if r.get("base"):
            def with_src(c, rel):
                if c:
                    c["src"] = rel
                return c
            entry["graded"] = {
                "bwd_old": with_src(_base_cell(D["bwd_prefix"]),
                                    "eval/results/final450_backward_prefix.json"),
                "bwd_b1": with_src(_base_cell(D["bwd_b1"]),
                                   "eval/results/final450_backward_b1.json"),
                "bwd_b2": with_src(_base_cell(D["bwd_b2"]),
                                   "eval/results/final450_backward_b2.json"),
                "bwd_retrained": with_src(
                    _base_cell(D.get("bwd_retrained")),
                    D.get("bwd_retrained_src",
                          "eval/results/" + BASE_FUTURE["bwd_retrained"][0])),
                "fwd": with_src(_base_cell(D["fwd450"], "forward"),
                                "eval/results/comparison_forward.json"),
            }
            # base forward cell must be the BEST forward system, not the first
            if D.get("fwd450"):
                best = None
                for name, s in systems_of_kind(D["fwd450"], "forward"):
                    a = s["aggregate"]
                    if best is None or a["solve_rate"] > best["agg"]["solve_rate"]:
                        best = {"agg": a, "name": name,
                                "src": "eval/results/comparison_forward.json",
                                "machine": machine_of(D["fwd450"]),
                                "date": proto_date(D["fwd450"])[:10]}
                entry["graded"]["fwd"] = best
            entry["oracle_meta"] = D.get("bench450_meta")
            out.append(entry)
            continue
        slots = D["rungs"][r["key"]]
        for group in ("graded", "frontier"):
            for slot in ("bwd_old", "bwd_b1", "bwd_b2", "fwd"):
                entry[group][slot] = cell(slots, group, slot)
            fut = "bwd_retrained" if group == "graded" \
                else "bwd_retrained_frontier"
            entry[group]["bwd_retrained"] = cell(slots, "future", fut)
        entry["fwd_first"] = cell(slots, "graded", "fwd_first")
        entry["withheld"] = cell(slots, "withheld", "fwd")
        entry["meta"] = D["metas"].get(r["key"])
        out.append(entry)
    return out
