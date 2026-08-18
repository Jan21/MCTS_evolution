"""Seed replicates of the BACKWARD HEADLINE PAIR (FINDINGS 77a, §80).

Two rungs are covered; the config is the script's only argument.

  g16r8 -- the 16x16 / 8-robot row is a ZERO-SHOT application of the
    base-trained B1 pair (`checkpoints_backward/{policy,value}_b1.ckpt`:
    policy 25 epochs cold, value 20 epochs warm-started from value_v2).
  g32r4 -- the 32x32 / 4-robot row is the per-config base-vocabulary pair
    (`scaling/runs/g32r4/backward-{policy,value}/...`: policy 30 epochs,
    value 25 epochs, both cold).

Either way the published pair is a single training draw.
`jobs/patterns/seed_headline_pair.slurm` repeats that config's recipe
verbatim under torch seeds 21/37/53 and re-benches with the headline
protocol (budget 1200, k=5, --backward-anytime --backward-b2,
replay-certified), producing

    scaling/results/<cfg>/comparison_b2_seed<S>.json           graded
    scaling/results/<cfg>/comparison_ungraded_b2_seed<S>.json  frontier
    eval/results/final450_backward_b2_seed<S>.json             base 450
                                                    (g16r8 home rung only)

This script pairs every arm against the SAME forward control the report
uses, per set, and writes analysis/artifacts/seed_headline_<cfg>.json:

  * solved counts / rates per arm per set (graded, frontier, pooled 450,
    and base450 where the config has one)
  * exact McNemar on the discordant pairs vs the forward control (rows pair
    by POSITION; the pairing is only permitted when both files record the
    same protocol.instances_sha256, and env_id sequences are asserted equal
    too)
  * median-of-3 (the seed replicates) and median-of-4 (incl. the production
    "seed of record"), with the min-max band, per set
  * mean solution length on the both-solved subset, both planners
  * the value net's val_regret per arm -- the pre-benchmark basin signal
    (seeds: the SEEDHL line in the job log; production: the ModelCheckpoint
    callback state banked inside the value checkpoint that the production
    comparison file's protocol.checkpoints names).  val_regret is a
    per-config quantity (it scales with the board's plan lengths) and is
    ONLY comparable within one config -- hence the per-config basin
    threshold, left None where no two-basin split exists.

Analysis only -- reads archived JSON/checkpoints, runs no planner.

    PYTHONPATH=. python -m analysis.seed_headline_g16r8 g16r8
    PYTHONPATH=. python -m analysis.seed_headline_g16r8 g32r4
    PYTHONPATH=. python -m analysis.seed_headline_g16r8 all

(The module keeps its original g16r8 name -- FINDINGS 80 cites it -- but
takes the config as an argument and defaults to g16r8.)
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import statistics
import sys

from eval.stats_tests import mcnemar_exact

OUT_TPL = "analysis/artifacts/seed_headline_{cfg}.json"
SV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # .../supervised_valuenet
REPO = os.path.dirname(SV)                                          # .../MCTS_evolution

SEEDS = (21, 37, 53)
SEED_ARMS = [f"seed{s}" for s in SEEDS]
ARMS = ["production"] + SEED_ARMS

# ---------------------------------------------------------------------------
# what to read.  Production files are the ones eval/report_data.py's RUNGS
# table pins for the config (bwd_b2 slot + fwd slot) and stats_tests.py's
# BASE_SYSTEMS for the base pool.  Seed files follow the launcher's naming.
# ---------------------------------------------------------------------------

def _scaling_sets(cfg, n_graded, n_frontier, graded_fwd=None,
                  frontier_fwd=None):
    """The graded + frontier set descriptors of one scaling rung."""
    d = f"scaling/results/{cfg}"
    return {
        "graded": {
            "label": f"gradable set ({n_graded})",
            "production": f"{d}/comparison_b2.json",
            "seed_tpl": d + "/comparison_b2_seed{seed}.json",
            "forward": graded_fwd or (f"{d}/comparison_forward_control.json",
                                      "forward"),
        },
        "frontier": {
            "label": f"beyond the oracle ({n_frontier})",
            "production": f"{d}/comparison_ungraded_b2.json",
            "seed_tpl": d + "/comparison_ungraded_b2_seed{seed}.json",
            "forward": frontier_fwd or (f"{d}/comparison_ungraded.json",
                                        "forward"),
        },
    }


CONFIGS = {
    # --- 16x16 / 8 robots: zero-shot B1 pair, dedicated forward control ----
    "g16r8": {
        "label": "16×16 · 8 robots",
        "design": ("seed replicates of the zero-shot backward headline pair "
                   "(base-trained B1 policy+value, FINDINGS 77a); only "
                   "--torch-seed varies, bench protocol verbatim"),
        "recipe": "policy 25 ep cold, value 20 ep warm-started from value_v2",
        "zero_shot": True,
        "sets": dict(
            _scaling_sets("g16r8", 266, 184),
            base450={
                "label": "base pool, 16x16 / 4 robots (450)",
                "production": "eval/results/final450_backward_b2.json",
                "seed_tpl": "eval/results/final450_backward_b2_seed{seed}.json",
                # report_data / stats_tests read the BEST forward system in
                # this file (candidate_scored.ckpt, 450/450), not the first.
                "forward": ("eval/results/comparison_forward.json",
                            "forward:best"),
            }),
        "pooled": ("graded", "frontier"),       # the whole-450 pool
        "value_logs": {
            f"seed{s}": f"runs/seedhl/rr-seedhl-g16r8-s{s}-{job}.out"
            for s, job in zip(SEEDS, (4670563, 4670565, 4670567))},
        # the two clusters are 0.54-0.66 against 2.45; anything above this
        # is the degenerate ("bad") basin of FINDINGS 44.
        "basin_threshold": 1.5,
        "basin_note": ("two clusters at this config: 0.54-0.66 (good) "
                       "against 2.45 (the degenerate basin of FINDINGS 44)"),
    },
    # --- 32x32 / 4 robots: per-config base-vocab pair ----------------------
    # this rung has no separate comparison_forward_control.json; report_data's
    # RUNGS reads the forward arm out of comparison{,_ungraded}.json.
    "g32r4": {
        "label": "32×32 · 4 robots",
        "design": ("seed replicates of the backward headline pair at 32x32 "
                   "/ 4 robots (per-config base-vocabulary policy+value, "
                   "FINDINGS 77a); only --torch-seed varies, bench protocol "
                   "verbatim"),
        "recipe": "policy 30 ep cold, value 25 ep cold",
        "zero_shot": False,
        "sets": _scaling_sets(
            "g32r4", 175, 275,
            graded_fwd=("scaling/results/g32r4/comparison.json", "forward")),
        "pooled": ("graded", "frontier"),
        "value_logs": {
            f"seed{s}": f"runs/seedhl/rr-seedhl-g32r4-s{s}-{job}.out"
            for s, job in zip(SEEDS, (4670569, 4670581, 4670583))},
        # no two-basin split is visible here -- all four runs land within a
        # tenth of a regret point -- so no threshold is claimed.
        "basin_threshold": None,
        "basin_note": ("no two-basin split at this config; val_regret is a "
                       "per-config scale, comparable only within the rung"),
    },
}
SET_ORDER = ("graded", "frontier", "pooled", "base450")


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


def prod_value_ckpt(cfg):
    """The production value checkpoint, as named by the benched file itself.

    Every comparison file records protocol.checkpoints (path -> mtime); the
    backward pair contributes exactly one policy and one value checkpoint,
    and every set of a config must have been benched with the same pair.
    """
    found = None
    for set_name in ("graded", "frontier"):
        spec = cfg["sets"].get(set_name)
        if not spec or not os.path.exists(_sv(spec["production"])):
            continue
        payload = json.load(open(_sv(spec["production"])))
        cks = list((payload.get("protocol") or {}).get("checkpoints") or {})
        vals = [c for c in cks if "value" in c.lower()]
        assert len(vals) == 1, (spec["production"], cks)
        assert found in (None, vals[0]), (found, vals[0])
        found = vals[0]
    return found


def prod_val_regret(rel_ckpt):
    """val_regret of the production value net, from its banked callback state.

    Lightning banks the ModelCheckpoint callback state -- monitor plus
    best_model_score -- inside every checkpoint it writes, so the number
    survives even when the training log does not (the g16r8 production pair
    was trained on the study's ORIGIN machine: its log lived in a scratchpad
    that no longer exists and the checkpoint's own paths still point at
    /mnt/raid/...).  Verified against a seed whose log IS available (g16r8
    seed37: callback 0.5440 == log 0.5440), so the two sources are the same
    quantity.
    """
    if not rel_ckpt:
        return None, None
    path = _sv(rel_ckpt)
    if not os.path.exists(path):
        return None, None
    try:
        import torch
    except ImportError:
        return None, None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for _key, state in (ck.get("callbacks") or {}).items():
        if not isinstance(state, dict) or "best_model_score" not in state:
            continue
        assert state.get("monitor") == "val_regret", state.get("monitor")
        return (float(state["best_model_score"]),
                rel_ckpt + " (ModelCheckpoint callback state)")
    return None, None


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def build(cfg_key):
    cfg = CONFIGS[cfg_key]
    out = {
        "config": cfg_key,
        "label": cfg["label"],
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "design": cfg["design"],
        "recipe": cfg["recipe"],
        "zero_shot": cfg["zero_shot"],
        "basin_threshold_val_regret": cfg["basin_threshold"],
        "basin_note": cfg["basin_note"],
        "arms": ARMS, "seed_arms": SEED_ARMS,
        "sets": {},
        "val_regret": {},
    }

    # ---- val_regret -------------------------------------------------------
    ckpt = prod_value_ckpt(cfg)
    v, src = prod_val_regret(ckpt)
    out["production_value_checkpoint"] = ckpt
    out["val_regret"]["production"] = {"val_regret": v, "source": src,
                                       "recoverable": v is not None}
    for arm in SEED_ARMS:
        v, src = seed_val_regret(cfg["value_logs"][arm])
        out["val_regret"][arm] = {"val_regret": v, "source": src,
                                  "recoverable": v is not None}
    thr = cfg["basin_threshold"]
    for _arm, e in out["val_regret"].items():
        e["basin"] = (None if (e["val_regret"] is None or thr is None) else
                      ("bad" if e["val_regret"] > thr else "good"))
    known = [e["val_regret"] for e in out["val_regret"].values()
             if e["val_regret"] is not None]
    out["val_regret_spread"] = (max(known) - min(known)) if known else None
    out["val_regret_min"] = min(known) if known else None
    out["val_regret_max"] = max(known) if known else None
    out["bimodal_by_val_regret"] = (
        None if thr is None else
        any(e["basin"] == "bad" for e in out["val_regret"].values()))

    # ---- per-set cells ----------------------------------------------------
    loaded = {}                     # set -> (forward, arm -> backward rows)
    for set_name, spec in cfg["sets"].items():
        fwd = load_rows(*spec["forward"])
        files = {"production": spec["production"]}
        files.update({f"seed{s}": spec["seed_tpl"].format(seed=s)
                      for s in SEEDS})
        arms, missing = {}, []
        for arm in ARMS:
            rel = files[arm]
            if not os.path.exists(_sv(rel)):
                missing.append(arm)
                continue
            arms[arm] = load_rows(rel, "backward")
        loaded[set_name] = (fwd, arms)

        entry = {"label": spec["label"], "forward_file": fwd["file"],
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

    # ---- pooled 450 (graded + frontier of this rung's pool) ---------------
    pool = cfg["pooled"]
    pooled = {"label": "pooled whole 450 (gradable + beyond-oracle)",
              "component_sets": list(pool), "arms": {},
              "forward_file": " + ".join(
                  cfg["sets"][s]["forward"][0] for s in pool),
              "instances_file": " + ".join(
                  loaded[s][0]["instances_file"] for s in pool),
              "instances_sha256": " + ".join(
                  loaded[s][0]["sha256"] for s in pool),
              "missing_arms": sorted(
                  set().union(*(set(out["sets"][s]["missing_arms"])
                                for s in pool)))}
    fwd_rows = [r for s in pool for r in loaded[s][0]["rows"]]
    for arm in ARMS:
        if any(arm not in loaded[s][1] for s in pool):
            continue
        bwd_rows = [r for s in pool for r in loaded[s][1][arm]["rows"]]
        c = cell(bwd_rows, fwd_rows)
        c["file"] = " + ".join(loaded[s][1][arm]["file"] for s in pool)
        pooled["arms"][arm] = c
    pooled["n"] = next(iter(pooled["arms"].values()))["n"]
    pooled["solved_forward"] = next(iter(pooled["arms"].values()))["solved_forward"]
    out["sets"]["pooled"] = pooled

    # ---- medians and bands ------------------------------------------------
    for _set_name, entry in out["sets"].items():
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
        # does every replicate reproduce (or beat) the published row?
        if seeds and s["production"] is not None:
            s["seeds_at_or_above_production"] = bool(
                s["min3_seeds"] >= s["production"])
            s["min3_minus_production"] = s["min3_seeds"] - s["production"]
    return out


def write(out):
    rel = OUT_TPL.format(cfg=out["config"])
    os.makedirs(os.path.dirname(_sv(rel)), exist_ok=True)
    tmp = _sv(rel) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, _sv(rel))
    return rel


def report(out):
    vr = out["val_regret"]
    print(f"\n=== {out['config']} ({out['label']}) — {out['recipe']}")
    print("val_regret:", ", ".join(
        f"{a}={vr[a]['val_regret']}"
        + (f" ({vr[a]['basin']})" if vr[a]["basin"] else "")
        for a in ARMS))
    if out["val_regret_spread"] is not None:
        print(f"  spread {out['val_regret_spread']:.4f} "
              f"({out['val_regret_min']}-{out['val_regret_max']}), "
              f"basin threshold {out['basin_threshold_val_regret']}")
    for set_name in SET_ORDER:
        e = out["sets"].get(set_name)
        if not e:
            continue
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
              f"production {s['production']}, "
              f"all seeds >= production: "
              f"{s.get('seeds_at_or_above_production')}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    dry = "--dry-run" in argv
    argv = [a for a in argv if not a.startswith("-")]
    which = argv[0] if argv else "g16r8"
    keys = list(CONFIGS) if which == "all" else [which]
    assert all(k in CONFIGS for k in keys), (which, list(CONFIGS))
    for key in keys:
        out = build(key)
        report(out)
        if dry:
            print(f"(dry run — nothing written for {key})")
        else:
            print(f"\nwrote {write(out)}")


if __name__ == "__main__":
    main()
