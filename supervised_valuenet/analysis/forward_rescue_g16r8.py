"""The FORWARD planner's fair second chance at 16x16 / 8 robots (FINDINGS 77b).

The objection this answers is the reviewers' sharpest one: "you beat a weak
opponent -- the move-by-move planner got a single training run and one
emergency learning-rate fix, while the subgoal planner's pipeline was
developed for months."  So the forward net was retrained NINE times at the
rung where its control looked weakest --

    3 torch seeds {21, 37, 53}  x  3 learning rates {5e-5, 1e-4, 2e-4}

-- with the control's own recipe otherwise verbatim (8 epochs, batch 128),
and ONE arm was promoted to the benchmark.  The promotion used the training
pipeline's own validation metric, `val_policy_top1` on validation boards
700-899, which are disjoint from the benchmark boards 900-1049; no benchmark
number entered the choice.  The winner (seed 37, lr 1e-4, val top-1 0.8806
against the control of record's 0.8628) was then benched with the control's
protocol verbatim -- same pinned instances, 1200 expansions, top-5, every
solved plan replayed through the physics.

    scaling/runs/g16r8/forward-rescue/s<seed>-lr<lr>/BEST.json  per-arm val
    scaling/runs/g16r8/forward-rescue/SELECTED.json             the choice
    scaling/results/g16r8/comparison_forward_rescue.json        graded (266)
    scaling/results/g16r8/comparison_ungraded_forward_rescue.json
                                                              frontier (184)

This script writes analysis/artifacts/forward_rescue_g16r8.json:

  * the 9-arm validation table (seed x lr, val top-1, distance from the
    control of record, rank), plus the integrity check that the checkpoint
    actually benched IS the checkpoint the validation rule selected;
  * rescued forward vs the forward CONTROL OF RECORD, paired per set --
    the same-set rows pair by POSITION, permitted only when both files pin
    the identical instance file (sha256) and their env_id sequences agree;
  * rescued forward vs every backward arm of FINDINGS 84 (the published
    "seed of record" pair and the three fresh-seed replicates 21/37/53),
    per set and pooled, with the same positional pairing and exact McNemar;
  * the median-of-3 backward arm (the seed whose pooled count IS the
    median the rung now reports) named explicitly, so the headline margin
    is quoted against a specific, benched, replay-certified arm rather
    than against an arithmetic median with no rows behind it;
  * the pre-registered reading rule of FINDINGS 77b evaluated against the
    measured frontier count.

The forward CONTROL files are not hardcoded here: they are resolved through
eval.report_data.RUNGS, i.e. the same slots the report renders, so this
analysis can never quietly compare against a different control than the page.

Analysis only -- reads archived JSON, runs no planner.

    PYTHONPATH=. python -m analysis.forward_rescue_g16r8
    PYTHONPATH=. python -m analysis.forward_rescue_g16r8 --dry-run
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import re
import statistics
import sys

from eval.report_data import RUNGS
from eval.stats_tests import mcnemar_exact

CFG = "g16r8"
LABEL = "16×16 · 8 robots"
OUT = f"analysis/artifacts/forward_rescue_{CFG}.json"

SV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # supervised_valuenet
REPO = os.path.dirname(SV)                                         # MCTS_evolution

# the rescue grid, in the order the launcher ran it
SEEDS = (21, 37, 53)
LRS = (("5e-5", 5e-05), ("1e-4", 0.0001), ("2e-4", 0.0002))
GRID_DIR = f"scaling/runs/{CFG}/forward-rescue"
SELECTED = f"{GRID_DIR}/SELECTED.json"

# the benched rescue arm
RESCUE = {
    "graded": f"scaling/results/{CFG}/comparison_forward_rescue.json",
    "frontier": f"scaling/results/{CFG}/comparison_ungraded_forward_rescue.json",
}

# the backward arms of FINDINGS 84 (same rung, same protocol, same instances)
BWD_ARMS = ["production", "seed21", "seed37", "seed53"]
BWD_FILES = {
    "graded": {
        "production": f"scaling/results/{CFG}/comparison_b2.json",
        **{f"seed{s}": f"scaling/results/{CFG}/comparison_b2_seed{s}.json"
           for s in SEEDS}},
    "frontier": {
        "production": f"scaling/results/{CFG}/comparison_ungraded_b2.json",
        **{f"seed{s}":
           f"scaling/results/{CFG}/comparison_ungraded_b2_seed{s}.json"
           for s in SEEDS}},
}
SEED_ARMS = [f"seed{s}" for s in SEEDS]

SET_LABEL = {"graded": "gradable set (266)",
             "frontier": "beyond the oracle (184)",
             "pooled": "whole pinned pool (450)"}
SET_ORDER = ("graded", "frontier", "pooled")

# FINDINGS 77b, fixed before the grid ran.
PREREG = {
    "source": "FINDINGS 77b",
    "text": ("rescued frontier ≥ ~150/184 removes the +15.8-point pooled "
             "margin; ~120 keeps the margin with a best-of-9-by-validation "
             "caveat"),
    "kill_at_or_above": 150,
    "keep_with_caveat_near": 120,
}

# the bench log that certifies every solved rescue plan by replay
SEL_LOG_GLOB = f"runs/fwdgrid/rr-fwdsel-{CFG}*.out"
_REPLAY_RE = re.compile(
    r"replay_validate: (\d+) passed, (\d+) failed, (\d+) rows without")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _sv(rel):
    return os.path.join(SV, rel)


def _pick(payload, kind):
    hits = [s for s in (payload.get("systems") or {}).values()
            if s.get("kind") == kind and s.get("rows")]
    assert len(hits) == 1, (kind, len(hits))
    return hits[0]


def load_rows(rel, kind):
    """(rows, sha256, instances_file) for one system of one result file."""
    payload = json.load(open(_sv(rel)))
    system = _pick(payload, kind)
    rows = system["rows"]
    agg = system.get("aggregate") or {}
    n_solved = sum(bool(r.get("solved")) for r in rows)
    if "solved" in agg:
        assert agg["solved"] == n_solved, (rel, agg["solved"], n_solved)
    proto = payload.get("protocol") or {}
    return {"rows": rows, "sha256": proto.get("instances_sha256"),
            "instances_file": proto.get("instances_file"), "file": rel,
            "checkpoints": sorted(proto.get("checkpoints") or {})}


def control_slots():
    """The forward control files, taken from the report's own rung slots.

    report_data.RUNGS is the single place the page pins which file is the
    control of record at each rung; reading it here means the rescue can
    never be scored against a control the report does not show.
    """
    rung = next(r for r in RUNGS if r["key"] == CFG)
    out = {}
    for set_name in ("graded", "frontier"):
        fname, kind = rung[set_name]["fwd"]
        out[set_name] = (f"scaling/results/{CFG}/{fname}", kind)
    return out


def env_ids(rows):
    ids = [r.get("env_id") for r in rows]
    return ids if all(i is not None for i in ids) else None


def plan_len(row):
    """Move count of a solved plan on the real board (see seed_headline)."""
    rs = row.get("realized_strict")
    m = row.get("moves")
    if isinstance(m, list):
        assert rs is None or rs == len(m), (rs, len(m))
        return len(m)
    if rs is not None:
        return rs
    return m if isinstance(m, int) else None


def assert_pairable(x, y):
    """Rows of two result files may be paired by position only if ..."""
    assert x["sha256"] and y["sha256"], (x["file"], y["file"])
    assert x["sha256"] == y["sha256"], (x["file"], y["file"],
                                        x["sha256"], y["sha256"])
    assert len(x["rows"]) == len(y["rows"]), (x["file"], y["file"])
    ex, ey = env_ids(x["rows"]), env_ids(y["rows"])
    if ex and ey:
        assert ex == ey, (x["file"], y["file"])


# ---------------------------------------------------------------------------
# one paired cell
# ---------------------------------------------------------------------------

def cell(a_rows, b_rows, a_label, b_label):
    """Exact paired McNemar for A vs B; rows pair by position.

    `diff_points` is A minus B in percentage points, so the caller decides
    which arm is the positive direction by choosing which one is A.
    """
    assert len(a_rows) == len(b_rows), (len(a_rows), len(b_rows))
    a = [bool(r.get("solved")) for r in a_rows]
    b = [bool(r.get("solved")) for r in b_rows]
    n = len(a)
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    p = mcnemar_exact(a_only, b_only)
    both = [(ra, rb) for ra, rb, sa, sb in zip(a_rows, b_rows, a, b)
            if sa and sb]
    la = [plan_len(ra) for ra, _ in both if plan_len(ra) is not None]
    lb = [plan_len(rb) for _, rb in both if plan_len(rb) is not None]
    return {
        "a_label": a_label, "b_label": b_label, "n": n,
        "solved_a": sum(a), "solved_b": sum(b),
        "rate_a": sum(a) / n, "rate_b": sum(b) / n,
        "diff_solved": sum(a) - sum(b),
        "diff_points": (sum(a) - sum(b)) / n * 100.0,
        "discordant_a_only": a_only, "discordant_b_only": b_only,
        "n_discordant": a_only + b_only,
        "mcnemar_p": p, "significant_at_05": p < 0.05,
        "both_solved": len(both),
        "mean_len_a": (statistics.mean(la) if la else None),
        "mean_len_b": (statistics.mean(lb) if lb else None),
    }


# ---------------------------------------------------------------------------
# the 9-arm validation grid
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(r"epoch=(\d+)-step=(\d+)\.ckpt$")


def read_grid():
    """The 9 per-arm BEST.json files plus the launcher's own SELECTED.json.

    Every arm is read from its own BEST.json (the ModelCheckpoint monitor,
    mode and best score as Lightning recorded them); SELECTED.json is read
    only to cross-check the choice and to carry the control of record's
    validation score, and the two are asserted to agree.
    """
    sel = json.load(open(_sv(SELECTED)))
    assert sel["config"] == CFG, sel["config"]
    sel_by_cell = {row["cell"]: row for row in sel["table"]}

    arms = []
    for seed in SEEDS:
        for lr_tag, lr in LRS:
            cell_name = f"s{seed}-lr{lr_tag}"
            best = json.load(open(_sv(f"{GRID_DIR}/{cell_name}/BEST.json")))
            assert best["config"] == CFG and best["seed"] == seed, best
            assert abs(best["lr"] - lr) < 1e-12, (cell_name, best["lr"])
            # selection metric: maximize validation top-1, never a test score
            assert best["monitor"] == "val_policy_top1", best["monitor"]
            assert best["mode"] == "max", best["mode"]
            score = float(best["best_model_score"])
            mirror = sel_by_cell[cell_name]
            assert abs(mirror["val_policy_top1"] - score) < 1e-12, cell_name
            assert mirror["ckpt"] == best["best_model_path"], cell_name
            m = _STEP_RE.search(best["best_model_path"])
            arms.append({
                "cell": cell_name, "seed": seed, "lr": lr, "lr_tag": lr_tag,
                "val_policy_top1": score,
                "epochs_run": best.get("epochs_run"),
                "best_epoch": (int(m.group(1)) + 1) if m else None,
                "ckpt": best["best_model_path"],
                "trainer_args": best.get("trainer_args"),
            })
    assert len(arms) == 9, len(arms)

    ctrl = sel["control_of_record_g16r8"]
    ctrl_val = float(ctrl["val_policy_top1"])
    for a in arms:
        a["delta_vs_control"] = a["val_policy_top1"] - ctrl_val
        a["above_control"] = a["val_policy_top1"] > ctrl_val
    ranked = sorted(arms, key=lambda a: -a["val_policy_top1"])
    for i, a in enumerate(ranked):
        a["rank"] = i + 1
    winner = ranked[0]
    for a in arms:
        a["selected"] = a["cell"] == winner["cell"]
    assert winner["cell"] == sel["winner"]["cell"], (winner["cell"],
                                                     sel["winner"]["cell"])
    scores = [a["val_policy_top1"] for a in arms]
    return {
        "metric": "val_policy_top1",
        "rule": sel.get("rule"),
        "selection_reads_test": False,
        "n_arms": len(arms),
        "seeds": list(SEEDS),
        "learning_rates": [lr for _tag, lr in LRS],
        "recipe": "the control's recipe verbatim: 8 epochs, batch size 128",
        "val_boards": "700-899 (disjoint from the benchmark boards 900-1049)",
        "arms": arms,
        "ranked_cells": [a["cell"] for a in ranked],
        "winner": {k: winner[k] for k in
                   ("cell", "seed", "lr", "lr_tag", "val_policy_top1",
                    "epochs_run", "best_epoch", "ckpt")},
        "control_of_record": {"ckpt": ctrl["ckpt"],
                              "val_policy_top1": ctrl_val},
        "winner_minus_control": winner["val_policy_top1"] - ctrl_val,
        "n_arms_above_control": sum(1 for a in arms if a["above_control"]),
        "val_min": min(scores), "val_max": max(scores),
        "val_spread": max(scores) - min(scores),
        "collapsed_arms": [a["cell"] for a in arms
                           if a["val_policy_top1"] < 0.5],
    }


def replay_certificate():
    """Replay-validation counts for the rescue bench, out of its job log."""
    out = {}
    for path in sorted(glob.glob(os.path.join(REPO, SEL_LOG_GLOB))):
        for passed, failed, skipped in _REPLAY_RE.findall(
                open(path, errors="replace").read()):
            n = int(passed)
            key = ("graded" if n > 184 else "frontier")
            out[key] = {"passed": n, "failed": int(failed),
                        "no_move_dump": int(skipped),
                        "log": os.path.relpath(path, REPO)}
    return out or None


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def build():
    grid = read_grid()
    controls = control_slots()

    out = {
        "config": CFG,
        "label": LABEL,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "design": (
            "a fair second chance for the move-by-move planner at its "
            "weakest-looking rung: 3 torch seeds x 3 learning rates with "
            "the control's recipe otherwise verbatim, one arm promoted by "
            "validation top-1 alone, benched with the control's protocol"),
        "selection": grid,
        "replay_certified": replay_certificate(),
        "control_files": {k: v[0] for k, v in controls.items()},
        "rescue_files": dict(RESCUE),
        "backward_files": BWD_FILES,
        "backward_arms": BWD_ARMS,
        "sets": {},
    }

    # ---- load every arm, per set -----------------------------------------
    loaded = {}     # set -> {"rescued":..., "control":..., "bwd": {arm:...}}
    for set_name in ("graded", "frontier"):
        rescued = load_rows(RESCUE[set_name], "forward")
        ctrl_rel, ctrl_kind = controls[set_name]
        control = load_rows(ctrl_rel, ctrl_kind)
        bwd = {arm: load_rows(rel, "backward")
               for arm, rel in BWD_FILES[set_name].items()}
        assert_pairable(rescued, control)
        for arm in BWD_ARMS:
            assert_pairable(rescued, bwd[arm])
        loaded[set_name] = {"rescued": rescued, "control": control,
                            "bwd": bwd}

    # the benched net must BE the validation-selected arm
    benched = sorted({c for s in loaded.values()
                      for c in s["rescued"]["checkpoints"]})
    out["selection"]["benched_checkpoints"] = benched
    out["selection"]["benched_is_selected_arm"] = (
        benched == [grid["winner"]["ckpt"]])
    assert out["selection"]["benched_is_selected_arm"], benched

    # ---- per-set cells ----------------------------------------------------
    def entry(set_name, rescued_rows, control_rows, bwd_rows_by_arm,
              instances_file, sha, files):
        e = {
            "label": SET_LABEL[set_name],
            "n": len(rescued_rows),
            "instances_file": instances_file,
            "instances_sha256": sha,
            "files": files,
            "solved_rescued": sum(bool(r.get("solved")) for r in rescued_rows),
            "solved_control": sum(bool(r.get("solved")) for r in control_rows),
            "solved_backward": {
                arm: sum(bool(r.get("solved")) for r in rows)
                for arm, rows in bwd_rows_by_arm.items()},
            "vs_control": cell(rescued_rows, control_rows,
                               "rescued forward (best of 9 by validation)",
                               "forward control of record"),
            "vs_backward": {
                arm: cell(rows, rescued_rows,
                          f"subgoal planner ({arm})", "rescued forward")
                for arm, rows in bwd_rows_by_arm.items()},
        }
        counts = [e["solved_backward"][a] for a in SEED_ARMS]
        med = statistics.median(counts)
        e["backward_median3"] = med
        e["backward_band3"] = [min(counts), max(counts)]
        # the arm that IS the median (so the margin has benched rows behind it)
        med_arm = next((a for a in SEED_ARMS
                        if e["solved_backward"][a] == med), None)
        e["backward_median_arm"] = med_arm
        e["margin_median_over_rescued_points"] = (
            (med - e["solved_rescued"]) / e["n"] * 100.0)
        e["margin_production_over_rescued_points"] = (
            (e["solved_backward"]["production"] - e["solved_rescued"])
            / e["n"] * 100.0)
        return e

    for set_name in ("graded", "frontier"):
        L = loaded[set_name]
        out["sets"][set_name] = entry(
            set_name, L["rescued"]["rows"], L["control"]["rows"],
            {a: L["bwd"][a]["rows"] for a in BWD_ARMS},
            L["rescued"]["instances_file"], L["rescued"]["sha256"],
            {"rescued": L["rescued"]["file"], "control": L["control"]["file"],
             **{a: L["bwd"][a]["file"] for a in BWD_ARMS}})

    # ---- pooled 450 -------------------------------------------------------
    order = ("graded", "frontier")
    cat = lambda pick: [r for s in order for r in pick(loaded[s])]
    out["sets"]["pooled"] = entry(
        "pooled",
        cat(lambda L: L["rescued"]["rows"]),
        cat(lambda L: L["control"]["rows"]),
        {a: cat(lambda L, a=a: L["bwd"][a]["rows"]) for a in BWD_ARMS},
        " + ".join(loaded[s]["rescued"]["instances_file"] for s in order),
        " + ".join(loaded[s]["rescued"]["sha256"] for s in order),
        {"rescued": " + ".join(RESCUE[s] for s in order),
         "control": " + ".join(controls[s][0] for s in order),
         **{a: " + ".join(BWD_FILES[s][a] for s in order) for a in BWD_ARMS}})
    out["sets"]["pooled"]["component_sets"] = list(order)

    # ---- the pre-registered reading rule ----------------------------------
    fr = out["sets"]["frontier"]
    po = out["sets"]["pooled"]
    rescued_frontier = fr["solved_rescued"]
    verdict = ("margin removed" if rescued_frontier >= PREREG["kill_at_or_above"]
               else "margin stands, with a best-of-9-by-validation caveat")
    out["prereg"] = dict(
        PREREG,
        rescued_frontier=rescued_frontier,
        rescued_frontier_of=fr["n"],
        control_frontier=fr["solved_control"],
        backward_median_frontier=fr["backward_median3"],
        verdict=verdict,
        margin_removed=bool(rescued_frontier >= PREREG["kill_at_or_above"]),
        near_caveat_threshold=bool(
            rescued_frontier < PREREG["kill_at_or_above"]),
        pooled_margin_median_points=po["margin_median_over_rescued_points"],
        pooled_margin_production_points=(
            po["margin_production_over_rescued_points"]),
        pooled_backward_median=po["backward_median3"],
        pooled_backward_median_arm=po["backward_median_arm"],
        pooled_rescued=po["solved_rescued"],
        pooled_control=po["solved_control"],
    )
    out["backward_median_arm"] = po["backward_median_arm"]
    return out


def write(out):
    path = _sv(OUT)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, path)
    return OUT


def report(out):
    g = out["selection"]
    print(f"\n=== forward fair rescue, {out['config']} ({out['label']})")
    print(f"selection: {g['metric']}, {g['n_arms']} arms, control of record "
          f"{g['control_of_record']['val_policy_top1']:.4f}")
    for a in sorted(g["arms"], key=lambda a: -a["val_policy_top1"]):
        print(f"  {a['rank']}. {a['cell']:12s} seed {a['seed']:2d} "
              f"lr {a['lr_tag']:5s} val_top1 {a['val_policy_top1']:.4f} "
              f"({a['delta_vs_control']:+.4f})"
              + ("   <- selected" if a["selected"] else ""))
    print(f"  benched checkpoint is the selected arm: "
          f"{g['benched_is_selected_arm']}")
    for set_name in SET_ORDER:
        e = out["sets"][set_name]
        c = e["vs_control"]
        print(f"\n{set_name}  n={e['n']}")
        print(f"  rescued {e['solved_rescued']:4d}  control "
              f"{e['solved_control']:4d}  diff {c['diff_points']:+6.1f} pt  "
              f"discordant {c['discordant_a_only']}/{c['discordant_b_only']}  "
              f"McNemar p={c['mcnemar_p']:.4g}")
        for arm in BWD_ARMS:
            b = e["vs_backward"][arm]
            print(f"  bwd {arm:11s} {b['solved_a']:4d}  vs rescued "
                  f"{b['solved_b']:4d}  {b['diff_points']:+6.1f} pt  "
                  f"discordant {b['discordant_a_only']}/"
                  f"{b['discordant_b_only']}  p={b['mcnemar_p']:.4g}")
        print(f"  backward median-of-3 {e['backward_median3']:g} "
              f"(arm {e['backward_median_arm']}, band "
              f"{e['backward_band3'][0]}-{e['backward_band3'][1]}), "
              f"margin over rescued "
              f"{e['margin_median_over_rescued_points']:+.1f} pt")
    p = out["prereg"]
    print(f"\npre-registered rule ({p['source']}): {p['text']}")
    print(f"  measured rescued frontier {p['rescued_frontier']}/"
          f"{p['rescued_frontier_of']} -> {p['verdict']}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    out = build()
    report(out)
    if "--dry-run" in argv:
        print("\n(dry run — nothing written)")
    else:
        print(f"\nwrote {write(out)}")


if __name__ == "__main__":
    main()
