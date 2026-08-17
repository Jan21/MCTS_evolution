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


def newest_source_file():
    """(iso date, relpath) of the most recently WRITTEN file the build read.

    The masthead used to quote the newest `protocol.date` inside the loaded
    JSONs, which lags behind reality whenever a file is regenerated or a new
    artifact without a protocol block lands.  This reads the filesystem
    instead, so the date on the page can never be stale.
    """
    newest, newest_rel = None, None
    for rel, e in SOURCES.items():
        if e.get("status") != "ok":
            continue
        try:
            m = os.path.getmtime(rp(rel))
        except OSError:
            continue
        if newest is None or m > newest:
            newest, newest_rel = m, rel
    if newest is None:
        return "—", None
    import datetime as _dt
    return _dt.date.fromtimestamp(newest).isoformat(), newest_rel


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
# file renders as a pending row.  `future` slots are retrain results; every
# one that is absent today was NEVER LAUNCHED (no queued job) — the page says
# "not run", not "in progress" (see NOT_RUN_NOTE).
# ---------------------------------------------------------------------------

# The note attached to every retrain file that does not exist.  As of the last
# build no retraining job is queued or running for any of them, so nothing on
# the page may promise them as forthcoming.
NOT_RUN_NOTE = ("not run — this retrain was never launched (no queued or "
                "running job); the row fills in automatically if it ever is")

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


# ---------------------------------------------------------------------------
# Per-puzzle wall-clock (median / 90th percentile / total)
#
# Every comparison file carries a per-instance `rows` array with a `seconds`
# field for BOTH planners, at every rung and on both the gradable and the
# beyond-oracle sets.  The aggregates elsewhere on the page quote
# `mean_seconds`; a mean is the wrong summary here because the subgoal
# planner's cost distribution is heavy-tailed (FINDINGS 25), so this module
# recomputes median, 90th percentile and total straight from the rows.
#
# Wall-clock is only comparable WITHIN one machine (see machine_of), so each
# row pairs the forward system with the fullest backward language measured on
# the SAME machine, and records which fuller language had to be skipped.
# ---------------------------------------------------------------------------

WALL_PREF = ["bwd_b2", "bwd_b1", "bwd_old"]
WALL_LABEL = {"bwd_b2": "subgoals, full language",
              "bwd_b1": "subgoals, extended language (B1)",
              "bwd_old": "subgoals, original language"}


def pctile(xs, q):
    """Linear-interpolated percentile; q in [0, 1]. None for an empty list."""
    ys = sorted(xs)
    if not ys:
        return None
    if len(ys) == 1:
        return float(ys[0])
    pos = q * (len(ys) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return float(ys[lo])
    return float(ys[lo]) + (float(ys[hi]) - float(ys[lo])) * (pos - lo)


def _wall_pairs(comp, name):
    """[(seconds, solved)] for one named system inside a comparison file."""
    sysd = ((comp or {}).get("systems") or {}).get(name) or {}
    return [(float(r["seconds"]), bool(r.get("solved")))
            for r in (sysd.get("rows") or [])
            if isinstance(r.get("seconds"), (int, float))]


def wall_stats(pairs):
    """median / p90 / total / max plus a solved-within-t-seconds curve."""
    if not pairs:
        return None
    xs = [t for t, _ in pairs]
    n = len(xs)
    curve, solved = [], 0
    for t, ok in sorted(pairs):
        if ok:
            solved += 1
            curve.append((t, solved / n * 100))
    if not curve or curve[-1][0] < max(xs):
        curve.append((max(xs), solved / n * 100))
    return {"n": n, "median": pctile(xs, 0.5), "p90": pctile(xs, 0.9),
            "total": sum(xs), "mean": sum(xs) / n, "max": max(xs),
            "min": min(xs), "solved": solved, "solve_rate": solved / n * 100,
            "curve": curve}


def build_wallclock(ladder, comps):
    """One row per rung x set: same-machine backward/forward wall-clock."""
    out = []
    for e in ladder:
        for group, gname in (("graded", "gradable set"),
                             ("frontier", "beyond the oracle")):
            cells = e.get(group) or {}
            f = cells.get("fwd")
            if not f:
                continue
            avail = [k for k in WALL_PREF if cells.get(k)]
            same = [k for k in avail if cells[k]["machine"] == f["machine"]]
            row = {"key": e["key"], "label": e["label"], "short": e["short"],
                   "set": gname, "base": bool(e.get("base"))}
            if not same:
                row.update({"bwd": None, "fwd": None, "same_machine": False,
                            "skipped": WALL_LABEL.get(avail[0]) if avail
                            else None})
                out.append(row)
                continue
            b = cells[same[0]]
            bs = wall_stats(_wall_pairs(comps.get(b["src"]), b["name"]))
            fs = wall_stats(_wall_pairs(comps.get(f["src"]), f["name"]))
            if not bs or not fs:
                continue
            skipped = None
            if avail and avail[0] != same[0]:
                skipped = WALL_LABEL.get(avail[0])
            row.update({
                "bwd": bs, "fwd": fs, "same_machine": True,
                "machine": b["machine"], "bwd_slot": same[0],
                "bwd_label": WALL_LABEL.get(same[0], same[0]),
                "bwd_src": b["src"], "fwd_src": f["src"],
                "skipped": skipped,
                "r_median": (fs["median"] / bs["median"]
                             if bs["median"] else None),
                "r_p90": fs["p90"] / bs["p90"] if bs["p90"] else None,
                "r_total": fs["total"] / bs["total"] if bs["total"] else None,
            })
            out.append(row)
    return out or None


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
                "note": NOT_RUN_NOTE})
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
# By-reference zero-shot A/B (FINDINGS 78)
#
# The re-use ("by-reference") step type the plan-language ceiling depends on
# was never wired into the learned planner (FINDINGS 40); it is now wired
# behind a flag that defaults off.  Each entry below is one benchmark set:
# the SAME banked cap-20,000 B2 network pair run over the identical pinned
# 16x16 6-robot instances with the flag on and off, plus the file whose rows
# the flag-off arm must reproduce exactly (the regression check).
#
# NOTE on what that regression file IS: it is the cap-20,000 RETRAINED row at
# 16x16 6 robots (310/316 gradable, 105/134 beyond-oracle), because that is
# the network pair the A/B runs.  It is NOT the zero-shot production row of
# the headline tables (306/316, 108/134).  Any prose claiming the flag-off arm
# "reproduces every production row on this page" would be false.
#
# Rows pair by POSITION, not by env_id: these files are merged from 40 shard
# runs and env_ids repeat across shards (146 distinct ids over 316 graded
# rows).  Both arms were sharded identically, which is asserted below along
# with the instance-file sha256 before any pairing is done.
# ---------------------------------------------------------------------------

BYREF_AB_SETS = [
    ("graded", "gradable 316",
     "scaling/results/g16r6/comparison_b2retrained_cap20000_byref_on.json",
     "scaling/results/g16r6/comparison_b2retrained_cap20000_byref_off.json",
     "scaling/results/g16r6/comparison_b2retrained_cap20000.json"),
    ("frontier", "beyond-oracle 134",
     "scaling/results/g16r6/comparison_ungraded_b2retrained_cap20000_byref_on.json",
     "scaling/results/g16r6/comparison_ungraded_b2retrained_cap20000_byref_off.json",
     "scaling/results/g16r6/comparison_ungraded_b2retrained_cap20000.json"),
]


def _byref_arm(comp):
    """The measured planner inside a by-reference A/B file (not a control)."""
    for name, s in systems_of_kind(comp, "backward"):
        if "production control" in name.lower():
            continue
        if s.get("rows"):
            return s
    return None


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def build_byref_ab():
    """Paired on/off summary of the by-reference supply A/B, or None.

    Returns {"sets": [...], "pooled": {...}} with every number recomputed
    from the per-instance rows; the caller registers them through ck().
    """
    sets, pins_ok, regression_ok = [], True, True
    tot = {"n": 0, "on": 0, "off": 0, "b": 0, "c": 0}
    for key, label, rel_on, rel_off, rel_prod in BYREF_AB_SETS:
        c_on, c_off = load_json(rel_on), load_json(rel_off)
        a_on, a_off = _byref_arm(c_on), _byref_arm(c_off)
        if not a_on or not a_off or len(a_on["rows"]) != len(a_off["rows"]):
            return None
        r_on, r_off = a_on["rows"], a_off["rows"]
        ids_on = [r.get("env_id") for r in r_on]
        pins_ok = pins_ok and ids_on == [r.get("env_id") for r in r_off] and (
            c_on["protocol"].get("instances_sha256")
            == c_off["protocol"].get("instances_sha256")
            and bool(c_on["protocol"].get("instances_sha256")))
        # the flag-off arm must reproduce the production rows exactly
        a_prod = _byref_arm(load_json(rel_prod))
        if a_prod and len(a_prod["rows"]) == len(r_off):
            regression_ok = regression_ok and all(
                bool(p["solved"]) == bool(o["solved"])
                and p["realized_strict"] == o["realized_strict"]
                and p["expansions"] == o["expansions"]
                for p, o in zip(a_prod["rows"], r_off))
        else:
            regression_ok = False
        b = sum(1 for x, y in zip(r_on, r_off) if x["solved"] and not y["solved"])
        c = sum(1 for x, y in zip(r_on, r_off) if y["solved"] and not x["solved"])
        both = [(x, y) for x, y in zip(r_on, r_off) if x["solved"] and y["solved"]]
        acc = (a_on["aggregate"].get("accounting") or {})
        ent = {
            "key": key, "label": label, "n": len(r_on),
            "src_on": rel_on, "src_off": rel_off, "src_prod": rel_prod,
            "b": b, "c": c,
            "ranked": acc.get("byref_cands_ranked"),
            "topk": acc.get("byref_cands_topk"),
        }
        for arm, rows, both_i in (("on", r_on, 0), ("off", r_off, 1)):
            ent[arm] = {
                "solved": sum(1 for r in rows if r["solved"]),
                "mean_expansions": _mean([r["expansions"] for r in rows]),
                "median_seconds": pctile([r["seconds"] for r in rows], 0.5),
                "mean_len_both": _mean([p[both_i]["realized_strict"]
                                        for p in both]),
            }
        tot["n"] += ent["n"]
        tot["on"] += ent["on"]["solved"]
        tot["off"] += ent["off"]["solved"]
        tot["b"] += b
        tot["c"] += c
        sets.append(ent)
    if not sets:
        return None
    from eval.stats_tests import mcnemar_exact   # local: stats_tests imports us
    tot["p"] = mcnemar_exact(tot["b"], tot["c"])
    fact("by-reference A/B: both arms pin the identical instance file "
         "(sha256) and the identical row order, so rows pair one-to-one",
         pins_ok)
    fact("by-reference A/B: the flag-off arm reproduces the 16×16 · 6-robot "
         "cap-20,000 RETRAINED rows exactly (solved, plan length and "
         "expansions, both sets) — not the zero-shot production rows",
         regression_ok)
    return {"sets": sets, "pooled": tot}


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
    # seed replicates of the g16r8 zero-shot headline pair
    # (analysis/seed_headline_g16r8.py; FINDINGS 77a)
    D["seed_headline_g16r8"] = load_json(
        "analysis/artifacts/seed_headline_g16r8.json")
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
            "note": NOT_RUN_NOTE})

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

    # by-reference zero-shot supply A/B at 16x16 6r (FINDINGS 78)
    D["byref_ab"] = build_byref_ab()

    D["ladder"] = build_ladder(D)
    D["retrain"] = retrain_verdict(D)

    # per-puzzle wall-clock, recomputed from the row arrays (median + tail)
    comps = {}
    for _key, slots in D["rungs"].items():
        for _sl, (rel, comp, _kind) in slots.items():
            if comp:
                comps[rel] = comp
    for rel, key in (("eval/results/comparison_forward.json", "fwd450"),
                     ("eval/results/final450_backward_prefix.json",
                      "bwd_prefix"),
                     ("eval/results/final450_backward_b1.json", "bwd_b1"),
                     ("eval/results/final450_backward_b2.json", "bwd_b2")):
        if D.get(key):
            comps[rel] = D[key]
    D["wallclock"] = build_wallclock(D["ladder"], comps)
    return D


def _base_cell(comp, kind="backward"):
    if not comp:
        return None
    agg = pick(comp, kind)
    if not agg:
        return None
    return {"agg": agg, "src": None, "machine": machine_of(comp),
            "date": proto_date(comp)[:10], "name": pick_name(comp, kind)}


# ---------------------------------------------------------------------------
# The retraining story, derived (never asserted)
#
# Three separate places used to state, in prose, that "retraining did not
# improve on the zero-shot rows".  That is false at 32x32 4 robots and it was
# only ever true rung by rung, so the claim is now COMPUTED from the loaded
# ladder cells and rendered from this one structure.
# ---------------------------------------------------------------------------

RETRAIN_MATCH_BAND = 3.0   # percentage points: inside this, call it a wash

SET_LABEL = {"graded": "gradable", "frontier": "beyond-oracle"}


def retrain_verdict(D):
    """Per-rung verdict on the cap-20,000 retrain vs its zero-shot sibling.

    A rung counts as `improved` when every measured set gains more than
    RETRAIN_MATCH_BAND points, `regressed` when any set loses more than that,
    `matched` otherwise, and `not run` when no retrain file exists at all.
    """
    rungs = []
    for e in D["ladder"]:
        sets = []
        for group in ("graded", "frontier"):
            cells = e.get(group) or {}
            rc, zc = cells.get("bwd_retrained"), cells.get("bwd_b2")
            if not rc or not zc:
                continue
            sets.append({
                "set": group, "set_label": SET_LABEL[group],
                "retrained": rc, "zeroshot": zc,
                "delta": (rc["agg"]["solve_rate"]
                          - zc["agg"]["solve_rate"]) * 100})
        if not sets:
            verdict = "not run"
        else:
            lo = min(s["delta"] for s in sets)
            if lo > RETRAIN_MATCH_BAND:
                verdict = "improved"
            elif lo < -RETRAIN_MATCH_BAND:
                verdict = "regressed"
            else:
                verdict = "matched"
        rungs.append({"key": e["key"], "label": e["label"],
                      "short": e["short"], "sets": sets, "verdict": verdict})
    by = {}
    for r in rungs:
        by.setdefault(r["verdict"], []).append(r)
    out = {"rungs": rungs, "n_total": len(rungs),
           "n_landed": sum(1 for r in rungs if r["verdict"] != "not run"),
           "improved": by.get("improved", []),
           "matched": by.get("matched", []),
           "regressed": by.get("regressed", []),
           "not_run": by.get("not run", [])}
    fact("retraining story: a verdict is derived from result files for every "
         f"one of the {len(rungs)} configurations, and at least one rung is "
         "recorded as improved (so no text may claim retraining never helped)",
         len(rungs) == len(RUNGS) and bool(out["improved"]))
    return out


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
