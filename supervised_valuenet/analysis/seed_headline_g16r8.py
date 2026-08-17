"""Seed replicates of the BACKWARD HEADLINE PAIR at 16x16 / 8 robots (FINDINGS 77a).

The g16r8 headline row is a ZERO-SHOT application of the base-trained B1 pair
(`checkpoints_backward/{policy,value}_b1.ckpt`).  That pair is a single draw:
policy 25 epochs cold, value 20 epochs warm-started from value_v2 with torch
seed 11.  `jobs/patterns/seed_headline_pair.slurm` repeats that recipe verbatim
under torch seeds 21/37/53 and re-benches with the headline protocol (budget
1200, k=5, --backward-anytime --backward-b2, replay-certified), producing

    scaling/results/g16r8/comparison_b2_seed<S>.json           graded, 266
    scaling/results/g16r8/comparison_ungraded_b2_seed<S>.json  frontier, 184
    eval/results/final450_backward_b2_seed<S>.json             base 450 (home rung)

This script pairs every arm against the SAME forward control the report uses,
per set, and writes analysis/artifacts/seed_headline_g16r8.json:

  * solved counts / rates per arm per set (graded, frontier, pooled 450, base450)
  * exact McNemar on the discordant pairs vs the forward control (rows pair by
    POSITION; the pairing is only permitted when both files record the same
    protocol.instances_sha256, and env_id sequences are asserted equal too)
  * median-of-3 (the seed replicates) and median-of-4 (incl. the production
    "seed of record"), with the min-max band, per set
  * mean solution length on the both-solved subset, both planners
  * the value net's val_regret per arm -- the pre-benchmark basin signal
    (seeds: the SEEDHL line in the job log; production: the ModelCheckpoint
    callback state banked inside checkpoints_backward/value_b1.ckpt)

Analysis only -- reads archived JSON/checkpoints, runs no planner.

    PYTHONPATH=. python -m analysis.seed_headline_g16r8
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import statistics

from eval.stats_tests import mcnemar_exact

OUT = "analysis/artifacts/seed_headline_g16r8.json"
SV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # .../supervised_valuenet
REPO = os.path.dirname(SV)                                          # .../MCTS_evolution

# ---------------------------------------------------------------------------
# what to read.  Production files are the ones eval/report_data.py's RUNGS
# table pins for g16r8 (bwd_b2 zero-shot slot + fwd slot) and stats_tests.py's
# BASE_SYSTEMS for the base pool.
# ---------------------------------------------------------------------------
ARMS = ["production", "seed21", "seed37", "seed53"]
SEED_ARMS = ["seed21", "seed37", "seed53"]

SETS = {
    "graded": {
        "label": "gradable set (266)",
        "backward": {
            "production": "scaling/results/g16r8/comparison_b2.json",
            "seed21": "scaling/results/g16r8/comparison_b2_seed21.json",
            "seed37": "scaling/results/g16r8/comparison_b2_seed37.json",
            "seed53": "scaling/results/g16r8/comparison_b2_seed53.json",
        },
        "forward": ("scaling/results/g16r8/comparison_forward_control.json",
                    "forward"),
    },
    "frontier": {
        "label": "beyond the oracle (184)",
        "backward": {
            "production": "scaling/results/g16r8/comparison_ungraded_b2.json",
            "seed21": "scaling/results/g16r8/comparison_ungraded_b2_seed21.json",
            "seed37": "scaling/results/g16r8/comparison_ungraded_b2_seed37.json",
            "seed53": "scaling/results/g16r8/comparison_ungraded_b2_seed53.json",
        },
        "forward": ("scaling/results/g16r8/comparison_ungraded.json", "forward"),
    },
    "base450": {
        "label": "base pool, 16x16 / 4 robots (450)",
        "backward": {
            "production": "eval/results/final450_backward_b2.json",
            "seed21": "eval/results/final450_backward_b2_seed21.json",
            "seed37": "eval/results/final450_backward_b2_seed37.json",
            "seed53": "eval/results/final450_backward_b2_seed53.json",
        },
        # report_data / stats_tests read the BEST forward system in this file
        # (candidate_scored.ckpt, 450/450), never the first one listed.
        "forward": ("eval/results/comparison_forward.json", "forward:best"),
    },
}
POOLED = ("graded", "frontier")     # the g16r8 whole-450 pool

# val_regret sources
VALUE_LOGS = {
    "seed21": "runs/seedhl/rr-seedhl-g16r8-s21-4670563.out",
    "seed37": "runs/seedhl/rr-seedhl-g16r8-s37-4670565.out",
    "seed53": "runs/seedhl/rr-seedhl-g16r8-s53-4670567.out",
}                                                   # relative to REPO
PROD_VALUE_CKPT = "checkpoints_backward/value_b1.ckpt"      # relative to SV
# the two clusters are 0.54-0.66 against 2.45; anything above this is the
# degenerate ("bad") basin of FINDINGS 44.
BASIN_THRESHOLD = 1.5


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _sv(rel):
    return os.path.join(SV, rel)


def _pick(payload, want):
    """Resolve a system: '<kind>:best' -> best solve_rate, else the unique kind."""
    systems = payload.get("systems", {})
    if want.endswith(":best"):
        kind = want[: -len(":best")]
        hits = [s for s in systems.values()
                if s.get("kind") == kind and s.get("rows")]
        assert hits, (want, list(systems))
        return max(hits, key=lambda s: s["aggregate"]["solve_rate"])
    hits = [s for s in systems.values()
            if s.get("kind") == want and s.get("rows")]
    assert len(hits) == 1, (want, len(hits))
    return hits[0]


def load_rows(rel, want):
    """(rows, sha256, instances_file) for one system of one result file."""
    payload = json.load(open(_sv(rel)))
    system = _pick(payload, want)
    rows = system["rows"]
    agg = system.get("aggregate") or {}
    n_solved = sum(bool(r.get("solved")) for r in rows)
    if "solved" in agg:
        assert agg["solved"] == n_solved, (rel, agg["solved"], n_solved)
    proto = payload.get("protocol", {})
    return {"rows": rows, "sha256": proto.get("instances_sha256"),
            "instances_file": proto.get("instances_file"), "file": rel}


def plan_len(row):
    """Move count of a solved plan, on the real board.

    The backward rows carry `realized_strict` (and, when the run was launched
    with --dump-moves, the move list itself -- asserted consistent); the
    forward rows carry `moves` as a plain count.
    """
    rs = row.get("realized_strict")
    m = row.get("moves")
    if isinstance(m, list):
        assert rs is None or rs == len(m), (rs, len(m))
        return len(m)
    if rs is not None:
        return rs
    return m if isinstance(m, int) else None


# ---------------------------------------------------------------------------
# one paired cell
# ---------------------------------------------------------------------------

def cell(bwd_rows, fwd_rows):
    """Paired stats for backward-vs-forward on one set (rows pair by position)."""
    assert len(bwd_rows) == len(fwd_rows), (len(bwd_rows), len(fwd_rows))
    a = [bool(r.get("solved")) for r in bwd_rows]
    b = [bool(r.get("solved")) for r in fwd_rows]
    n = len(a)
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    p = mcnemar_exact(a_only, b_only)
    both = [(ra, rb) for ra, rb, sa, sb in zip(bwd_rows, fwd_rows, a, b)
            if sa and sb]
    la = [plan_len(ra) for ra, _ in both if plan_len(ra) is not None]
    lb = [plan_len(rb) for _, rb in both if plan_len(rb) is not None]
    return {
        "n": n,
        "solved_backward": sum(a), "solved_forward": sum(b),
        "rate_backward": sum(a) / n, "rate_forward": sum(b) / n,
        "diff_points": (sum(a) - sum(b)) / n * 100.0,
        "discordant_backward_only": a_only, "discordant_forward_only": b_only,
        "mcnemar_p": p, "significant_at_05": p < 0.05,
        "both_solved": len(both),
        "mean_len_backward": (statistics.mean(la) if la else None),
        "mean_len_forward": (statistics.mean(lb) if lb else None),
    }


def env_ids(rows):
    ids = [r.get("env_id") for r in rows]
    return ids if all(i is not None for i in ids) else None


# ---------------------------------------------------------------------------
# val_regret
# ---------------------------------------------------------------------------

_SEEDHL_RE = re.compile(r"SEEDHL value best_model_score=([0-9.eE+-]+)")


def seed_val_regret(rel_log):
    path = os.path.join(REPO, rel_log)
    if not os.path.exists(path):
        return None, None
    hits = _SEEDHL_RE.findall(open(path, errors="replace").read())
    if not hits:
        return None, None
    assert len(set(hits)) == 1, (rel_log, set(hits))
    return float(hits[-1]), rel_log


def prod_val_regret():
    """val_regret of the production value_b1 net, from its banked callback state.

    The production pair was trained on the study's ORIGIN machine (its training
    log lived in a scratchpad that no longer exists, and the checkpoint's own
    paths still point at /mnt/raid/...), so no runs/*.out or metrics.csv is
    recoverable.  The number survives inside the checkpoint: Lightning banks
    the ModelCheckpoint callback state, including monitor + best_model_score.
    Verified against a seed whose log IS available (seed37: callback 0.5440 ==
    log 0.5440), so the two sources are the same quantity.
    """
    path = _sv(PROD_VALUE_CKPT)
    if not os.path.exists(path):
        return None, None
    try:
        import torch
    except ImportError:
        return None, None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for key, state in (ck.get("callbacks") or {}).items():
        if not isinstance(state, dict) or "best_model_score" not in state:
            continue
        assert state.get("monitor") == "val_regret", state.get("monitor")
        return float(state["best_model_score"]), PROD_VALUE_CKPT + " (ModelCheckpoint callback state)"
    return None, None


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main():
    out = {
        "config": "g16r8",
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "design": ("seed replicates of the zero-shot backward headline pair "
                   "(base-trained B1 policy+value, FINDINGS 77a); only "
                   "--torch-seed varies, bench protocol verbatim"),
        "basin_threshold_val_regret": BASIN_THRESHOLD,
        "arms": ARMS, "seed_arms": SEED_ARMS,
        "sets": {},
        "val_regret": {},
    }

    # ---- val_regret -------------------------------------------------------
    v, src = prod_val_regret()
    out["val_regret"]["production"] = {"val_regret": v, "source": src,
                                       "recoverable": v is not None}
    for arm in SEED_ARMS:
        v, src = seed_val_regret(VALUE_LOGS[arm])
        out["val_regret"][arm] = {"val_regret": v, "source": src,
                                  "recoverable": v is not None}
    for arm, e in out["val_regret"].items():
        e["basin"] = (None if e["val_regret"] is None else
                      ("bad" if e["val_regret"] > BASIN_THRESHOLD else "good"))

    # ---- per-set cells ----------------------------------------------------
    loaded = {}                     # set -> arm -> loaded backward rows
    for set_name, cfg in SETS.items():
        fwd = load_rows(*cfg["forward"])
        arms, missing = {}, []
        for arm in ARMS:
            rel = cfg["backward"][arm]
            if not os.path.exists(_sv(rel)):
                missing.append(arm)
                continue
            arms[arm] = load_rows(rel, "backward")
        loaded[set_name] = (fwd, arms)

        entry = {"label": cfg["label"], "forward_file": fwd["file"],
                 "instances_file": fwd["instances_file"],
                 "instances_sha256": fwd["sha256"],
                 "missing_arms": missing, "arms": {}}
        for arm, bw in arms.items():
            # the pairing guard: same pinned instance set, same order
            assert bw["sha256"] and fwd["sha256"], (bw["file"], fwd["file"])
            assert bw["sha256"] == fwd["sha256"], (bw["file"], fwd["file"])
            ea, eb = env_ids(bw["rows"]), env_ids(fwd["rows"])
            if ea and eb:
                assert ea == eb, (bw["file"], fwd["file"])
            c = cell(bw["rows"], fwd["rows"])
            c["file"] = bw["file"]
            entry["arms"][arm] = c
        entry["n"] = next(iter(entry["arms"].values()))["n"]
        entry["solved_forward"] = next(iter(entry["arms"].values()))["solved_forward"]
        out["sets"][set_name] = entry

    # ---- pooled 450 (graded + frontier of the g16r8 pool) ------------------
    pooled = {"label": "pooled whole 450 (gradable + beyond-oracle)",
              "component_sets": list(POOLED), "arms": {},
              "forward_file": " + ".join(SETS[s]["forward"][0] for s in POOLED),
              "instances_file": " + ".join(
                  loaded[s][0]["instances_file"] for s in POOLED),
              "instances_sha256": " + ".join(
                  loaded[s][0]["sha256"] for s in POOLED),
              "missing_arms": sorted(
                  set().union(*(set(out["sets"][s]["missing_arms"])
                                for s in POOLED)))}
    fwd_rows = [r for s in POOLED for r in loaded[s][0]["rows"]]
    for arm in ARMS:
        if any(arm not in loaded[s][1] for s in POOLED):
            continue
        bwd_rows = [r for s in POOLED for r in loaded[s][1][arm]["rows"]]
        c = cell(bwd_rows, fwd_rows)
        c["file"] = " + ".join(loaded[s][1][arm]["file"] for s in POOLED)
        pooled["arms"][arm] = c
    pooled["n"] = next(iter(pooled["arms"].values()))["n"]
    pooled["solved_forward"] = next(iter(pooled["arms"].values()))["solved_forward"]
    out["sets"]["pooled"] = pooled

    # ---- medians and bands ------------------------------------------------
    for set_name, entry in out["sets"].items():
        seeds = [entry["arms"][a]["solved_backward"]
                 for a in SEED_ARMS if a in entry["arms"]]
        alls = [entry["arms"][a]["solved_backward"]
                for a in ARMS if a in entry["arms"]]
        entry["summary"] = {
            "n_seed_arms": len(seeds),
            "median3_seeds": (statistics.median(seeds) if seeds else None),
            "min3_seeds": (min(seeds) if seeds else None),
            "max3_seeds": (max(seeds) if seeds else None),
            "band3_points": ((max(seeds) - min(seeds)) / entry["n"] * 100.0
                             if seeds else None),
            "median4_with_production": (statistics.median(alls) if alls else None),
            "min4": (min(alls) if alls else None),
            "max4": (max(alls) if alls else None),
            "production": entry["arms"].get("production", {}).get(
                "solved_backward"),
            "forward": entry["solved_forward"],
        }
        s = entry["summary"]
        for k, base in (("median3_seeds", "median3"),
                        ("median4_with_production", "median4")):
            if s[k] is not None:
                s[base + "_rate"] = s[k] / entry["n"]
                s[base + "_margin_points"] = (
                    (s[k] - s["forward"]) / entry["n"] * 100.0)

    os.makedirs(os.path.dirname(_sv(OUT)), exist_ok=True)
    tmp = _sv(OUT) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, _sv(OUT))

    # ---- readable summary -------------------------------------------------
    print("val_regret:", ", ".join(
        f"{a}={out['val_regret'][a]['val_regret']} "
        f"({out['val_regret'][a]['basin']})" for a in ARMS))
    for set_name in ("graded", "frontier", "pooled", "base450"):
        e = out["sets"][set_name]
        print(f"\n{set_name}  n={e['n']}  forward={e['solved_forward']}")
        for arm in ARMS:
            c = e["arms"].get(arm)
            if not c:
                continue
            print(f"  {arm:11s} {c['solved_backward']:4d}/{c['n']} "
                  f"({c['rate_backward'] * 100:5.1f}%)  "
                  f"diff {c['diff_points']:+6.1f} pt  "
                  f"discordant {c['discordant_backward_only']}/"
                  f"{c['discordant_forward_only']}  "
                  f"McNemar p={c['mcnemar_p']:.4g}  "
                  f"len {c['mean_len_backward']:.2f} vs "
                  f"{c['mean_len_forward']:.2f} on {c['both_solved']} "
                  "both-solved")
        s = e["summary"]
        print(f"  median-of-3 {s['median3_seeds']} "
              f"(band {s['min3_seeds']}-{s['max3_seeds']}), "
              f"median-of-4 {s['median4_with_production']}, "
              f"production {s['production']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
