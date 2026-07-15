"""Pairwise decision comparator for paired end-to-end backward rollouts.

Implements the DESIGN.md section 5.11 measurement (task C review section 3,
binding): given the Python and Rust record streams for IDENTICAL instances,
align the trajectories decision by decision and compare candidates MATCHED
across sides by (bottleneck, support, helper) -- never by position in a
multiset. Measures:

  (i)   cand_parent_support field-only divergence rate over matched record
        pairs whose labels (ctg / is_optimal) are equal -- the sign-off
        number (training consumes this field);
  (ii)  any ctg / is_optimal / record-SET divergence outside the two
        validated tie classes (truncation-boundary ties, optimal-tie forks):
        threshold ZERO;
  (iii) trajectory forks not justified by a genuine optimal-cost tie
        (>= 2 distinct candidates at the common minimum ctg): threshold ZERO.

Tie classes (the only forgiven differences):
  - truncation-boundary tie: the two sides keep different candidates at the
    max_candidates stable-sort cut. Validated only when the sides' minimum
    ctg is equal AND every unmatched record's candidate scores exactly at
    the truncation boundary (score recomputed from the board tables; this
    closes the task C review's boundary-tie validator hole -- candidates
    kept by BOTH sides are always ctg-compared pairwise, whatever their
    score). Comparison of the instance stops after such a decision unless
    both sides advance on the same candidate.
  - optimal-tie fork: both sides label the same candidate multiset with the
    same minimum ctg, but advance on different optimal candidates. Justified
    only when >= 2 distinct candidates carry the minimum. Comparison stops
    at the fork (inherent to forked trajectories).

Inputs, two modes:
  attempts mode (paired_rollouts.py output):
      --py <prefix>.py.jsonl --rust <prefix>.rust.jsonl [--work <prefix>.work.jsonl]
    Joined per attempt id; also cross-checks instance-level status
    (python ok/empty vs rust ok/empty).
  records mode (two combined.jsonl files, e.g. the g16r6 production data and
  its Rust regeneration):
      --records --py backward.jsonl --rust backward.rust.jsonl
    Records are grouped into instances by (env_id, target, target_robot,
    helpers); only instances present on BOTH sides are label-compared
    (one-sided instances are the gate-4 flip accounting, reported as counts).

Score recomputation builds a lazy env per board on demand (only needed when
candidate multisets differ). Boards resolve through --config's env dir.

Exit nonzero when any (ii)/(iii) violation is found.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

PYREF_DIR = Path(__file__).resolve().parent
SV_DIR = PYREF_DIR.parent.parent
for _p in (str(SV_DIR), str(PYREF_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CHILD_FLAG = "RR_PYREF_CHILD_CMP"
INF = 10_000


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--py", required=True)
    p.add_argument("--rust", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--records", action="store_true",
                   help="records mode (two combined.jsonl files)")
    p.add_argument("--out-report", default=None, help="json report path")
    p.add_argument("--examples", type=int, default=8)
    return p.parse_args(argv)


def main():
    a = parse_args()
    if os.environ.get(CHILD_FLAG) != "1":
        from scaling import configs
        cfg = configs.get(a.config)
        env = {**os.environ, **configs.env(cfg), CHILD_FLAG: "1",
               "PYTHONPATH": str(SV_DIR)}
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                             *sys.argv[1:]], env=env, cwd=str(SV_DIR)).returncode
        sys.exit(rc)
    _child(a)


# ---------------------------------------------------------------------------


def _key(rec):
    """Candidate identity: (bottleneck, support, helper). NOT parent_support."""
    return (tuple(rec["cand_bottleneck"]), tuple(rec["cand_support"]),
            (tuple(rec["cand_helper"][0]), rec["cand_helper"][1]))


def _ps(rec):
    ps = rec["cand_parent_support"]
    return None if ps is None else tuple(ps)


def _group_by_depth(recs):
    """[[records of decision 0], [decision 1], ...] in file order."""
    out = []
    for r in recs:
        d = r["depth"]
        while len(out) <= d:
            out.append([])
        out[d].append(r)
    return out


class EnvCache:
    def __init__(self, cfg):
        self.cfg = cfg
        self.envs = {}

    def get(self, env_id):
        if env_id not in self.envs:
            import common
            grid_data = common.load_board_grid_data(self.cfg.env_dir_abs, env_id)
            self.envs[env_id] = common.build_env(
                grid_data, self.cfg.grid, mode="lazy", weight=2,
                table_workers=8)
        return self.envs[env_id]


def _score(env, rec):
    """heuristics.score of a record's candidate, recomputed from the tables
    (GridEnv.subgoal_score formula). The subgoal's target_robot is the
    segment mover: at depth 0 the decision expands the goal->leaf_0 edge and
    the mover sits at its original cell (= seg_start); at depth >= 1 the
    parent is a bottleneck/support node whose stored robot is positioned AT
    the parent cell (= seg_end)."""
    bn = tuple(rec["cand_bottleneck"])
    sup = tuple(rec["cand_support"])
    helper = tuple(rec["cand_helper"][0])
    goal = tuple(rec["seg_end"])
    mover = tuple(rec["seg_start"] if rec["depth"] == 0 else rec["seg_end"])
    s1 = env.reachability_matrix[(bn, goal)]
    s2 = env.compute_relaxed_shortest_path_length(mover, bn, sup)
    s3 = env.compute_relaxed_shortest_path_length(helper, sup)
    return ((INF if s1 is None else s1) + (INF if s2 is None else s2)
            + (INF if s3 is None else s3))


def _resolve_ps_both_orders(env, rec):
    """Replay heuristics.propose's parent_support resolution for a record's
    candidate under BOTH engines' dependent-support tail orders (Python: set
    iteration order; Rust: in-edge insertion order, deduplicated). Returns
    (ps_python, ps_rust). When the two agree, an asymmetric fix-c rejection
    of this candidate is impossible, so a one-sided record can only come
    from the max_candidates truncation cut."""
    goal = tuple(rec["seg_end"])
    bn = tuple(rec["cand_bottleneck"])
    pinned = rec["seg_support"]
    pinned = None if pinned is None else tuple(pinned)
    # python tail: the heuristics._dependent_supports set, same expression
    tail_py = list({d["dependent"]
                    for _, _, d in env.G.in_edges(goal, data=True)
                    if "dependent" in d})
    # rust tail: in-edge insertion order, first-seen dedup
    tail_rust, seen = [], set()
    for _, _, d in env.G.in_edges(goal, data=True):
        dep = d.get("dependent")
        if dep is not None and dep not in seen:
            seen.add(dep)
            tail_rust.append(dep)

    def resolve(tail):
        pss = []
        if pinned is not None:
            pss.append(pinned)
        pss.append(None)
        pss += [s for s in tail if s not in pss]
        for ps in pss:
            if env.compute_exact_shortest_path_length(bn, goal, ps) is not None:
                return ps
        return "unresolvable"

    return resolve(tail_py), resolve(tail_rust)


class Stats:
    def __init__(self, max_examples):
        self.max_examples = max_examples
        self.c = Counter()
        self.examples = defaultdict(list)

    def add(self, kind, n=1, example=None):
        self.c[kind] += n
        if example is not None and len(self.examples[kind]) < self.max_examples:
            self.examples[kind].append(example)


def compare_instance(iid, env_id, py_recs, rust_recs, st, envs):
    """Align one instance's two record streams. Updates `st`."""
    pd, rd = _group_by_depth(py_recs), _group_by_depth(rust_recs)
    st.add("instances_compared")
    depth = 0
    while True:
        if depth >= len(pd) and depth >= len(rd):
            st.add("instances_full_length")
            return
        if depth >= len(pd) or depth >= len(rd):
            side = "python" if depth >= len(pd) else "rust"
            st.add("VIOLATION_trajectory_length",
                   example=f"{iid}: {side} stream ends at depth {depth}, "
                           f"other side continues")
            return
        p, r = pd[depth], rd[depth]
        pk, rk = Counter(map(_key, p)), Counter(map(_key, r))
        st.add("decisions_aligned")
        if any(rec["ctx_supports"] for rec in p):
            st.add("decisions_with_support_ctx")
        if depth >= 1:
            st.add("decisions_depth_ge1")

        pmin, rmin = min(x["cost_to_go"] for x in p), min(x["cost_to_go"] for x in r)

        if pk == rk:
            # matched multisets: pair per key, compare labels pairwise
            by_key_p, by_key_r = defaultdict(list), defaultdict(list)
            for x in p:
                by_key_p[_key(x)].append(x)
            for x in r:
                by_key_r[_key(x)].append(x)
            for k in by_key_p:
                lp = sorted(by_key_p[k], key=lambda x: (x["cost_to_go"], str(_ps(x))))
                lr = sorted(by_key_r[k], key=lambda x: (x["cost_to_go"], str(_ps(x))))
                for xp, xr in zip(lp, lr):
                    st.add("records_matched")
                    lab_eq = True
                    if xp["cost_to_go"] != xr["cost_to_go"]:
                        lab_eq = False
                        st.add("VIOLATION_ctg_matched_candidate",
                               example=f"{iid} d{depth} cand={k}: "
                                       f"py ctg={xp['cost_to_go']} "
                                       f"rust ctg={xr['cost_to_go']}")
                    if bool(xp["is_optimal"]) != bool(xr["is_optimal"]):
                        lab_eq = False
                        st.add("VIOLATION_is_optimal_matched_candidate",
                               example=f"{iid} d{depth} cand={k}: "
                                       f"py opt={xp['is_optimal']} "
                                       f"rust opt={xr['is_optimal']}")
                    if lab_eq and _ps(xp) != _ps(xr):
                        st.add("ps_field_divergence",
                               example=f"{iid} d{depth} cand={k}: "
                                       f"py ps={_ps(xp)} rust ps={_ps(xr)} "
                                       f"(labels equal, ctg={xp['cost_to_go']})")
        else:
            # candidate-set divergence: validate the truncation-tie class
            st.add("decisions_set_divergent")
            if pmin != rmin:
                st.add("VIOLATION_min_ctg_set_divergent",
                       example=f"{iid} d{depth}: py min={pmin} rust min={rmin}, "
                               f"py-only={sorted((pk - rk).elements())} "
                               f"rust-only={sorted((rk - pk).elements())}")
                return
            env = envs.get(env_id)
            only_p = pk - rk
            only_r = rk - pk
            div_p = [x for x in p if only_p.get(_key(x), 0) > 0]
            div_r = [x for x in r if only_r.get(_key(x), 0) > 0]
            scores_p = {_score(env, x) for x in div_p}
            scores_r = {_score(env, x) for x in div_r}
            shared0 = pk & rk
            matched_scores = [_score(env, x) for x in p
                              if shared0.get(_key(x), 0) > 0]
            # Truncation-tie rule: under the stable score sort both sides
            # keep identical sub-boundary sets, so every one-sided record
            # must carry ONE common score >= every matched record's score.
            div_scores = scores_p | scores_r
            boundary_ok = (len(div_scores) == 1
                           and (not matched_scores
                                or next(iter(div_scores)) >= max(matched_scores)))
            # Close the record-level ps hole: a one-sided record could also
            # come from an ASYMMETRIC fix-c rejection (different
            # parent_support resolution). Prove per divergent record that
            # both engines resolve the same parent_support.
            for x in div_p + div_r:
                ps_py, ps_rust = _resolve_ps_both_orders(env, x)
                if ps_py != ps_rust:
                    boundary_ok = False
                    st.add("VIOLATION_ps_resolution_asymmetry",
                           example=f"{iid} d{depth} cand={_key(x)}: "
                                   f"py-order ps={ps_py} rust-order ps={ps_rust}")
            # matched keys are still ctg-compared pairwise (boundary-score
            # candidates kept by BOTH sides included -- C review minor (a))
            shared = pk & rk
            by_key_p, by_key_r = defaultdict(list), defaultdict(list)
            for x in p:
                by_key_p[_key(x)].append(x)
            for x in r:
                by_key_r[_key(x)].append(x)
            for k, cnt in shared.items():
                lp = sorted(by_key_p[k], key=lambda x: (x["cost_to_go"], str(_ps(x))))[:cnt]
                lr = sorted(by_key_r[k], key=lambda x: (x["cost_to_go"], str(_ps(x))))[:cnt]
                for xp, xr in zip(lp, lr):
                    st.add("records_matched")
                    if xp["cost_to_go"] != xr["cost_to_go"]:
                        st.add("VIOLATION_ctg_matched_candidate",
                               example=f"{iid} d{depth} cand={k} (boundary "
                                       f"decision): py ctg={xp['cost_to_go']} "
                                       f"rust ctg={xr['cost_to_go']}")
                    elif _ps(xp) != _ps(xr):
                        st.add("ps_field_divergence",
                               example=f"{iid} d{depth} cand={k}: "
                                       f"py ps={_ps(xp)} rust ps={_ps(xr)}")
            if not boundary_ok:
                st.add("VIOLATION_record_set_not_boundary_tie",
                       example=f"{iid} d{depth}: divergent-record scores "
                               f"py={sorted(scores_p)} rust={sorted(scores_r)} "
                               f"matched-max={max(matched_scores, default=None)}, "
                               f"py-only={sorted(only_p.elements())} "
                               f"rust-only={sorted(only_r.elements())}")
                return
            st.add("tie_truncation_boundary",
                   example=f"{iid} d{depth}: boundary score "
                           f"{next(iter(div_scores))}, "
                           f"{sum(only_p.values())}+{sum(only_r.values())} "
                           f"one-sided records, min ctg {pmin} both sides")

        # advance parity: first record in file order with is_optimal true
        p_adv = next((x for x in p if x["is_optimal"]), None)
        r_adv = next((x for x in r if x["is_optimal"]), None)
        if p_adv is None or r_adv is None:
            # a decision whose records exist has >= 1 optimal by construction
            st.add("VIOLATION_no_optimal_record",
                   example=f"{iid} d{depth}: missing optimal record "
                           f"(py={p_adv is not None} rust={r_adv is not None})")
            return
        if _key(p_adv) == _key(r_adv):
            if _ps(p_adv) != _ps(r_adv):
                # same advance candidate but a different parent_support:
                # the applied child plans differ, so downstream records are
                # not comparable depth-by-depth. The ps difference itself
                # was already counted on the matched pair above.
                st.add("stopped_ps_divergent_advance",
                       example=f"{iid} d{depth}: advance {_key(p_adv)} "
                               f"py ps={_ps(p_adv)} rust ps={_ps(r_adv)}")
                return
            if pk != rk:
                st.add("boundary_tie_continued_same_advance")
            depth += 1
            continue
        # fork
        opt_keys = ({_key(x) for x in p if x["is_optimal"]}
                    | {_key(x) for x in r if x["is_optimal"]})
        if pmin == rmin and len(opt_keys) >= 2:
            st.add("fork_optimal_tie",
                   example=f"{iid} d{depth}: advance py={_key(p_adv)} "
                           f"rust={_key(r_adv)}, shared min ctg {pmin}, "
                           f"{len(opt_keys)} distinct optimal candidates")
        else:
            st.add("VIOLATION_unjustified_fork",
                   example=f"{iid} d{depth}: advance py={_key(p_adv)} "
                           f"rust={_key(r_adv)}, min py={pmin} rust={rmin}, "
                           f"distinct optimal={len(opt_keys)}")
        return


# ---------------------------------------------------------------------------


def _load_attempts(a):
    """attempts mode -> (paired list, stats-preload)."""
    py = {}
    with open(a.py) as f:
        for raw in f:
            ln = json.loads(raw)
            py[ln["iid"]] = ln
    rust = {}
    with open(a.rust) as f:
        for raw in f:
            r = json.loads(raw)
            rust[r["id"]] = r
    return py, rust


def _inst_key(rec):
    return (rec["env_id"], json.dumps(rec["target"]),
            json.dumps(rec["target_robot"]), json.dumps(rec["helpers"]))


def _load_records(path):
    inst = {}
    order = []
    with open(path) as f:
        for raw in f:
            rec = json.loads(raw)
            k = _inst_key(rec)
            if k not in inst:
                inst[k] = []
                order.append(k)
            inst[k].append(rec)
    return inst, order


def _child(a):
    from scaling import configs
    cfg = configs.get(a.config)
    envs = EnvCache(cfg)
    st = Stats(a.examples)

    if a.records:
        pi, _ = _load_records(a.py)
        ri, _ = _load_records(a.rust)
        shared = [k for k in pi if k in ri]
        st.add("instances_py_only", n=sum(k not in ri for k in pi))
        st.add("instances_rust_only", n=sum(k not in pi for k in ri))
        for k in shared:
            iid = f"e{k[0]}:{k[1]}"
            compare_instance(iid, k[0], pi[k], ri[k], st, envs)
    else:
        py, rust = _load_attempts(a)
        for iid, ln in py.items():
            r = rust.get(iid)
            if r is None:
                st.add("VIOLATION_missing_rust_result", example=iid)
                continue
            r_ok = r.get("status") == "ok" and r.get("records")
            if ln["status"] == "ok" and r_ok:
                compare_instance(iid, ln["env_id"], ln["records"],
                                 r["records"], st, envs)
            elif ln["status"] == "ok" and not r_ok:
                st.add("VIOLATION_py_ok_rust_not",
                       example=f"{iid}: rust status={r.get('status')}")
            elif ln["status"] == "empty":
                if r_ok:
                    st.add("VIOLATION_py_empty_rust_ok", example=iid)
                else:
                    st.add("instances_both_empty")
            elif ln["status"] == "timeout":
                st.add("instances_py_timeout_rust_ok" if r_ok
                       else "instances_py_timeout_rust_not", example=iid)
            else:
                st.add("instances_py_error", example=f"{iid}: {ln['status']}")

    # ---- report ------------------------------------------------------------
    matched = st.c.get("records_matched", 0)
    psd = st.c.get("ps_field_divergence", 0)
    violations = {k: v for k, v in st.c.items() if k.startswith("VIOLATION")}
    print(f"== compare_rollouts: {a.config} "
          f"{'records' if a.records else 'attempts'} mode")
    for k in sorted(st.c):
        print(f"  {k:<44} {st.c[k]}")
    rate = (psd / matched) if matched else 0.0
    print(f"  ps-field divergence rate: {psd}/{matched} = {rate:.6f}")
    print(f"  zero-threshold violations: {sum(violations.values())}")
    for k in sorted(violations):
        for ex in st.examples[k]:
            print(f"    [{k}] {ex}")
    for k in ("tie_truncation_boundary", "fork_optimal_tie",
              "ps_field_divergence"):
        for ex in st.examples[k]:
            print(f"    [{k}] {ex}")
    if a.out_report:
        with open(a.out_report, "w") as f:
            json.dump({"config": a.config, "counts": dict(st.c),
                       "ps_rate": rate,
                       "examples": {k: v for k, v in st.examples.items()}},
                      f, indent=1)
    sys.exit(1 if violations else 0)


if __name__ == "__main__":
    main()
