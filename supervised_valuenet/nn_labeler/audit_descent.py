"""Compare descent (NN-completion) labels against exact labels, decision by decision.

`nn_labeler/descent.py` prices candidates by greedy NN completion + realization,
so its `cost_to_go` is a CERTIFIED UPPER BOUND on the exact optimum. Run with
`--instances-from <exact.jsonl>` it replays an exact corpus's instances, which
makes the two files directly comparable: this tool matches DECISION GROUPS
across the files, matches candidates within each matched group, and reports how
far the upper bounds sit above the optimum and whether the descent labels still
point at an optimal candidate.

    exact-only candidates    descent dropped them (no completion / uncertified)
    descent-only candidates  should be IMPOSSIBLE (same `propose`) -- reported
                             loudly with examples
    gap = descent ctg - exact ctg   must be >= 0; negatives are a bug signal
    argmin_agreement         descent's argmin is an exact-optimal candidate
    optimal_set_jaccard      overlap of the two `is_optimal` sets
    depth-0 coverage         matched share of exact depth-0 groups whose
                             instance the descent file replayed at all

Deeper groups diverge by construction (the two labelers commit different
candidates and walk different trajectories); that is expected, not an error --
depth 0 is where the audit is meaningful.

    PYTHONPATH=. python -m nn_labeler.audit_descent \
        --exact scaling/data/g8r4/backward.rust.jsonl \
        --descent scaling/data/g8r4/backward.nnlab.jsonl \
        --out nn_labeler/results/descent_audit_g8r4.json

Output JSON is written atomically (tmp + rename). The last line on success is
`NNLAB DESCENT-AUDIT DONE <out>`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from scaling.configs import REPO

MAX_GROUP_ROWS = 5000  # per-group detail rows kept in the JSON


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO / p)


def _pt(v):
    """Coordinate (or any list) -> hashable tuple; None stays None."""
    return tuple(v) if v is not None else None


def instance_key(r):
    """The instance a record belongs to (what `--instances-from` replays)."""
    return (r["env_id"], _pt(r["target"]),
            _pt(r["target_robot"][0]), r["target_robot"][1],
            tuple((_pt(h[0]), h[1]) for h in r["helpers"]))


def group_key(r):
    """The decision group: instance + open segment + depth."""
    return instance_key(r) + (_pt(r["seg_start"]), _pt(r["seg_end"]), r["depth"])


def cand_key(r):
    """The candidate within a decision (as `propose` enumerates it)."""
    return (_pt(r["cand_bottleneck"]), _pt(r["cand_support"]),
            _pt(r["cand_helper"][0]), r["cand_helper"][1],
            _pt(r["cand_parent_support"]))


def _stats(vals):
    """mean/median/p90/max/min + share of exact zeros over a list of numbers."""
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None,
                "max": None, "min": None, "share_zero": None}
    s = sorted(vals)
    n = len(s)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {
        "n": n,
        "mean": sum(s) / n,
        "median": float(median),
        "p90": float(s[min(n - 1, math.ceil(0.9 * n) - 1)]),
        "max": float(s[-1]),
        "min": float(s[0]),
        "share_zero": sum(1 for v in s if v == 0) / n,
    }


def load_descent(path):
    """All descent records, grouped by decision key (file order preserved)."""
    groups: dict[tuple, list[dict]] = {}
    n_records = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        n_records += 1
        groups.setdefault(group_key(r), []).append(r)
    return groups, n_records


def load_exact(path, wanted_groups, wanted_instances):
    """Stream the exact corpus (it is the big one).

    Keeps full records only for groups the descent file also has; counts every
    exact group, and remembers which depth-0 exact groups belong to an instance
    the descent file replayed (the coverage denominator).
    """
    kept: dict[tuple, list[dict]] = {}
    all_group_keys: set[tuple] = set()
    depth0_shared_instance: set[tuple] = set()
    n_records = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        n_records += 1
        gk = group_key(r)
        all_group_keys.add(gk)
        if r["depth"] == 0 and instance_key(r) in wanted_instances:
            depth0_shared_instance.add(gk)
        if gk in wanted_groups:
            kept.setdefault(gk, []).append(r)
    return kept, len(all_group_keys), n_records, depth0_shared_instance


def _by_cand(records):
    """cand_key -> record, first occurrence wins; also returns duplicate count."""
    out, dups = {}, 0
    for r in records:
        k = cand_key(r)
        if k in out:
            dups += 1
            continue
        out[k] = r
    return out, dups


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--exact", required=True, help="exact-labeled corpus jsonl")
    p.add_argument("--descent", required=True,
                   help="descent-labeled corpus jsonl (same instances)")
    p.add_argument("--max-gap-examples", type=int, default=5,
                   help="full example records kept per anomaly bucket")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    exact_path, descent_path = _abs(a.exact), _abs(a.descent)
    dgroups, d_records = load_descent(descent_path)
    d_instances = {k[:5] for k in dgroups}
    egroups, n_groups_exact, e_records, depth0_denom = load_exact(
        exact_path, set(dgroups), d_instances)

    matched = sorted(set(egroups) & set(dgroups),
                     key=lambda k: (k[0], k[-1], repr(k)))

    rows = []                       # per-matched-group detail
    gaps, gaps_opt = [], []         # per matched candidate
    n_cand_exact = n_cand_descent = n_cand_matched = 0
    exact_only_total = descent_only_total = 0
    dup_exact = dup_descent = 0
    descent_only_depth0 = groups_with_descent_only = 0
    max_group_exact = 0
    argmin_hits = argmin_scored = argmin_unmatched = 0
    jaccards = []
    matched_by_depth: dict[int, int] = {}
    ex_descent_only, ex_negative = [], []
    ex_largest = []                 # (gap, group_key, cand_key), pruned as it grows
    negative_gaps = 0

    for gk in matched:
        erecs, drecs = egroups[gk], dgroups[gk]
        emap, ed = _by_cand(erecs)
        dmap, dd = _by_cand(drecs)
        dup_exact += ed
        dup_descent += dd
        both = set(emap) & set(dmap)
        eonly = set(emap) - set(dmap)
        donly = set(dmap) - set(emap)
        max_group_exact = max(max_group_exact, len(emap))
        n_cand_exact += len(emap)
        n_cand_descent += len(dmap)
        n_cand_matched += len(both)
        exact_only_total += len(eonly)
        descent_only_total += len(donly)
        depth = gk[-1]
        matched_by_depth[depth] = matched_by_depth.get(depth, 0) + 1

        if donly:
            groups_with_descent_only += 1
            if depth == 0:
                descent_only_depth0 += len(donly)
        for k in sorted(donly, key=repr):  # deterministic example selection
            if len(ex_descent_only) < a.max_gap_examples:
                ex_descent_only.append({
                    "group_key": repr(gk), "cand_key": repr(k),
                    "n_cand_exact": len(emap), "n_cand_descent": len(dmap),
                    "descent_record": dmap[k],
                    "exact_group_candidates": [repr(x) for x in emap],
                })

        g_gaps = []
        for k in sorted(both, key=repr):
            gap = int(dmap[k]["cost_to_go"]) - int(emap[k]["cost_to_go"])
            gaps.append(gap)
            g_gaps.append(gap)
            if emap[k]["is_optimal"]:
                gaps_opt.append(gap)
            if gap < 0:
                negative_gaps += 1
                if len(ex_negative) < a.max_gap_examples:
                    ex_negative.append({
                        "group_key": repr(gk), "cand_key": repr(k), "gap": gap,
                        "exact_record": emap[k], "descent_record": dmap[k],
                    })
            ex_largest.append((gap, gk, k))
        if len(ex_largest) > 4096:  # keep only the worst few, bounded memory
            ex_largest.sort(key=lambda t: (-t[0], repr(t[1]), repr(t[2])))
            del ex_largest[max(a.max_gap_examples, 1):]

        # decision level: descent argmin (ties -> descent file order)
        best = min(range(len(drecs)),
                   key=lambda i: (int(drecs[i]["cost_to_go"]), i))
        bk = cand_key(drecs[best])
        argmin_scored += 1
        if bk not in emap:
            argmin_unmatched += 1
            hit = False
        else:
            hit = bool(emap[bk]["is_optimal"])
        argmin_hits += hit

        eopt = {k for k, r in emap.items() if r["is_optimal"]}
        dopt = {k for k, r in dmap.items() if r["is_optimal"]}
        union = eopt | dopt
        jac = len(eopt & dopt) / len(union) if union else 1.0
        jaccards.append(jac)

        if len(rows) < MAX_GROUP_ROWS:
            rows.append({
                "env_id": gk[0], "depth": depth,
                "n_cand_exact": len(emap), "n_cand_descent": len(dmap),
                "n_cand_matched": len(both),
                "exact_only": len(eonly), "descent_only": len(donly),
                "argmin_optimal": hit, "optimal_set_jaccard": jac,
                "gap_mean": (sum(g_gaps) / len(g_gaps)) if g_gaps else None,
                "gap_max": max(g_gaps) if g_gaps else None,
            })

    ex_largest.sort(key=lambda t: (-t[0], repr(t[1]), repr(t[2])))
    largest = []
    for gap, gk, k in ex_largest[:a.max_gap_examples]:
        largest.append({
            "group_key": repr(gk), "cand_key": repr(k), "gap": gap,
            "exact_record": _by_cand(egroups[gk])[0][k],
            "descent_record": _by_cand(dgroups[gk])[0][k],
        })

    depth0_matched = sum(1 for k in matched if k[-1] == 0)
    coverage = (depth0_matched / len(depth0_denom)) if depth0_denom else None

    report = {
        "exact": str(exact_path), "descent": str(descent_path),
        "groups": {
            "n_groups_exact": n_groups_exact,
            "n_groups_descent": len(dgroups),
            "n_matched": len(matched),
            "n_records_exact": e_records, "n_records_descent": d_records,
            "matched_by_depth": {str(d): c
                                 for d, c in sorted(matched_by_depth.items())},
        },
        "candidates": {
            "n_candidates_exact_in_matched": n_cand_exact,
            "n_candidates_descent_in_matched": n_cand_descent,
            "n_candidate_matches": n_cand_matched,
            "exact_only_candidates": exact_only_total,
            "descent_only_candidates": descent_only_total,
            "descent_only_candidates_depth0": descent_only_depth0,
            "groups_with_descent_only": groups_with_descent_only,
            # If this equals the labelers' --max-candidates, descent-only
            # candidates are most likely tie-break differences at the shared
            # top-k truncation, not an enumeration mismatch.
            "max_exact_group_size_in_matched": max_group_exact,
            "duplicate_cand_keys_exact": dup_exact,
            "duplicate_cand_keys_descent": dup_descent,
        },
        "gaps": {
            "all_matched": _stats(gaps),
            "exact_optimal_only": _stats(gaps_opt),
            "negative_gaps": negative_gaps,
        },
        "decisions": {
            "n_groups_scored": argmin_scored,
            "argmin_agreement": (argmin_hits / argmin_scored)
            if argmin_scored else None,
            "argmin_candidate_unmatched": argmin_unmatched,
            "optimal_set_jaccard": (sum(jaccards) / len(jaccards))
            if jaccards else None,
        },
        "coverage": {
            "exact_depth0_groups_with_shared_instance": len(depth0_denom),
            "depth0_matched": depth0_matched,
            "depth0_coverage": coverage,
            "n_instances_descent": len(d_instances),
        },
        "examples": {
            "descent_only_candidates": ex_descent_only,
            "negative_gaps": ex_negative,
            "largest_gaps": largest,
        },
        "groups_detail": rows,
        "groups_detail_truncated": len(matched) > len(rows),
    }
    g_all, g_opt = report["gaps"]["all_matched"], report["gaps"]["exact_optimal_only"]
    report["summary"] = {
        "n_matched_groups": len(matched),
        "depth0_coverage": coverage,
        "n_candidate_matches": n_cand_matched,
        "exact_only_candidates": exact_only_total,
        "descent_only_candidates": descent_only_total,
        "negative_gaps": negative_gaps,
        "gap_mean": g_all["mean"], "gap_median": g_all["median"],
        "gap_p90": g_all["p90"], "gap_max": g_all["max"],
        "share_gap_zero": g_all["share_zero"],
        "gap_mean_on_exact_optimal": g_opt["mean"],
        "share_gap_zero_on_exact_optimal": g_opt["share_zero"],
        "argmin_agreement": report["decisions"]["argmin_agreement"],
        "optimal_set_jaccard": report["decisions"]["optimal_set_jaccard"],
        "clean": descent_only_total == 0 and negative_gaps == 0,
    }

    out = _abs(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=1, sort_keys=True))
    os.replace(tmp, out)

    def _f(v, spec=".3f"):
        return "n/a" if v is None else format(v, spec)

    print(f"[descent-audit] groups: exact={n_groups_exact} "
          f"descent={len(dgroups)} matched={len(matched)} "
          f"(depth0 {depth0_matched}/{len(depth0_denom)} = "
          f"{_f(coverage)} of replayed depth-0 exact groups)")
    print(f"[descent-audit] candidates in matched groups: exact={n_cand_exact} "
          f"descent={n_cand_descent} matched={n_cand_matched} "
          f"exact_only={exact_only_total} descent_only={descent_only_total}")
    print(f"[descent-audit] gap = descent - exact ctg over {g_all['n']} matched "
          f"candidates: mean={_f(g_all['mean'])} median={_f(g_all['median'])} "
          f"p90={_f(g_all['p90'])} max={_f(g_all['max'])} "
          f"share_zero={_f(g_all['share_zero'])}")
    print(f"[descent-audit] gap on exact-optimal candidates (n={g_opt['n']}): "
          f"mean={_f(g_opt['mean'])} median={_f(g_opt['median'])} "
          f"p90={_f(g_opt['p90'])} max={_f(g_opt['max'])} "
          f"share_zero={_f(g_opt['share_zero'])}")
    print(f"[descent-audit] decisions: argmin_agreement="
          f"{_f(report['decisions']['argmin_agreement'])} "
          f"(unmatched argmin {argmin_unmatched}) optimal_set_jaccard="
          f"{_f(report['decisions']['optimal_set_jaccard'])}")
    flag = "CLEAN" if report["summary"]["clean"] else "ANOMALIES"
    print(f"[descent-audit] {flag}: descent_only_candidates="
          f"{descent_only_total} (depth0 {descent_only_depth0}, in "
          f"{groups_with_descent_only} groups; largest exact group "
          f"{max_group_exact} -- compare to --max-candidates) "
          f"negative_gaps={negative_gaps}")
    print(f"NNLAB DESCENT-AUDIT DONE {out}", flush=True)


if __name__ == "__main__":
    main()
