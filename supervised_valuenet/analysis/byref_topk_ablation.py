"""By-reference top-k ablation (FINDINGS §37's proposed ranking-side probe).

Question: does the learned backward planner, running the B2 language
(`--backward-b2` semantics: eval.compare.run_backward(anytime=True, b2=True)),
ever GENERATE a by-reference candidate, place one in a top-k proposal set,
EXPAND one, or USE one in its final solved plan?

Code-reading result this script verifies empirically (see the audit that
commissioned it): the production NN driver `eval.compare._nn_astar_backward`
calls `solver.propose(...)` WITHOUT injecting `skeleton.astar._reference_helpers`
and `_apply(...)` WITHOUT `by_reference=True` (eval/compare.py lines ~215/219),
unlike the label engine `nn/generate.py` (lines ~87-98) which does both. On top
of that, `eval.end2end._hidx` maps a candidate helper to its index by START
position only, so a by-reference candidate (helper standing at a planned,
non-start cell) would be dropped before ranking even if it were generated
(compare.py lines ~220-222). The as-is counters below are therefore expected to
be ZERO by construction; the SHADOW counters measure the counterfactual "what
the search would have been offered had the driver been wired like the label
engine", at zero behavioural change (shadow candidates are counted, never
pushed).

Everything is monkeypatched from here; no tracked search code is modified, and
the actual search behaviour is byte-identical to eval.compare's B2 path (the
shadow probe only adds read-only work).

Per instance it records:
  asis_byref_proposed     by-ref-shaped candidates reaching _apply   (expect 0)
  asis_byref_generated    ... accepted by _apply (child plan built)  (expect 0)
  asis_byref_ranked       ... surviving the _hidx filter into the
                          policy/value ranking pool                  (expect 0)
  asis_byref_topk         ... placed in a top-k proposal set         (expect 0)
  asis_byref_expanded     expanded plans already containing a by-ref
                          placement                                  (expect 0)
  final_plan_byref        final returned plan uses by-reference      (expect 0)
  final_plan_parks        park (step-aside) nodes in the final plan  (B2's live
                          machinery -- may be nonzero)
  shadow_byref_generated  by-ref candidates the search WOULD have been
                          offered per expansion group, had it been wired
                          like nn/generate.py (propose with
                          _reference_helpers + _apply(by_reference=True))
  shadow_byref_rankable   of those, how many the current featurization
                          could rank at all (_hidx != None; expect 0)
  shadow_groups_with_byref  expansion groups offering >= 1 such candidate

Usage (smoke):
  PYTHONPATH=. python -m analysis.byref_topk_ablation \
      --config g16r6 \
      --instances scaling/data/g16r6/bench.unsolved.jsonl \
      --policy 'scaling/runs/g16r6/backward-policy-b2-cap20000/lightning_logs/version_0/checkpoints/epoch=11-step=11016.ckpt' \
      --value  'scaling/runs/g16r6/backward-value-b2-cap20000/lightning_logs/version_0/checkpoints/epoch=29-step=59850.ckpt' \
      --expansions 200 --k 5 --limit 5 --device cpu \
      --out /tmp/byref_smoke.json

Full frontier run: --limit 0 --expansions 1200 (launched separately, not from
a login node).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=None,
                   help="scaling config name (sets RR_* env vars via "
                        "scaling.configs before repo imports); omit for base")
    p.add_argument("--instances", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--value", required=True)
    p.add_argument("--expansions", type=int, default=1200)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--limit", type=int, default=0,
                   help="run only the first N instances (0 = all)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-shadow", action="store_true",
                   help="skip the counterfactual shadow probe")
    p.add_argument("--out", required=True)
    return p.parse_args()


ARGS = parse_args()

# RR_* env vars must be set BEFORE any repo module import (module-level
# constants read them at import time). scaling.configs itself is import-safe.
if ARGS.config:
    from scaling.configs import get as _get_cfg, shell_env as _shell_env
    _cfg = _get_cfg(ARGS.config)
    for kv in _shell_env(_cfg).split():
        k, v = kv.split("=", 1)
        os.environ[k] = v

import eval.compare as compare              # noqa: E402
import eval.end2end as end2end              # noqa: E402
import skeleton.astar as astar              # noqa: E402
from skeleton import heuristics             # noqa: E402


# ---------------------------------------------------------------------------
# instrumentation state (reset per instance by the _nn_astar_backward wrapper)
# ---------------------------------------------------------------------------

CUR = {}
PER_INSTANCE = []


def _fresh_counters():
    return {
        "asis_apply_calls": 0,
        "asis_byref_proposed": 0,
        "asis_byref_generated": 0,
        "asis_byref_ranked": 0,
        "asis_byref_topk": 0,
        "asis_byref_expanded": 0,
        "asis_expansion_groups": 0,
        "shadow_byref_generated": 0,
        "shadow_byref_rankable": 0,
        "shadow_groups_with_byref": 0,
        "final_plan_byref": 0,
        "final_plan_parks": 0,
    }


def _starts_by_color(state):
    """color -> START cell, for the by-reference test (same definition as
    scaling/qc_byref.py: a helper standing anywhere that is not its robot's
    start position is a by-reference placement)."""
    d = {state.target_robot.color: tuple(state.target_robot.position)}
    for h in state.helpers:
        d[h.color] = tuple(h.position)
    return d


def _is_byref_robot(robot):
    sbc = CUR["starts_by_color"]
    return sbc.get(robot.color) is not None and \
        tuple(robot.position) != sbc[robot.color]


def _plan_contains_byref(plan):
    """True when the plan already carries a by-reference placement: a byref
    edge (shared support / target-as-stopper) or a support node relocating
    FROM an existing plan node (its successor is not a leaf)."""
    g = plan.g
    for _u, _v, d in g.edges(data=True):
        if d.get("byref"):
            return True
    for n, d in g.nodes(data=True):
        if d.get("ntype") == "support":
            for s in g.successors(n):
                if g.nodes[s].get("ntype") in ("support", "bottleneck"):
                    return True
    return False


# ---------------------------------------------------------------------------
# patch 1: skeleton.astar._apply -- candidate-level counting + shadow probe
# ---------------------------------------------------------------------------

_apply_orig = astar._apply


def _apply_instrumented(env, plan, parent, child, seg, cand,
                        by_reference=False):
    c = CUR.get("counters")
    if c is None:                       # outside an instrumented search
        return _apply_orig(env, plan, parent, child, seg, cand,
                           by_reference=by_reference)
    c["asis_apply_calls"] += 1
    is_byref = _is_byref_robot(cand.subgoal.helper)
    if is_byref:
        c["asis_byref_proposed"] += 1
    out = _apply_orig(env, plan, parent, child, seg, cand,
                      by_reference=by_reference)
    if is_byref and out is not None:
        c["asis_byref_generated"] += 1

    # One "expansion group" = the consecutive run of _apply calls for one
    # popped plan's open edge (compare.py calls _apply once per candidate).
    key = (id(plan), parent, child)
    if key != CUR.get("last_group_key"):
        CUR["last_group_key"] = key
        c["asis_expansion_groups"] += 1
        if _plan_contains_byref(plan):
            c["asis_byref_expanded"] += 1
        if not ARGS.no_shadow:
            _shadow_probe(env, plan, parent, child, seg, c)
    return out


def _shadow_probe(env, plan, parent, child, seg, c):
    """Counterfactual: what would this expansion have been offered had the
    driver injected _reference_helpers and applied with by_reference=True,
    exactly as nn/generate.py does for label generation? Read-only: children
    built here are counted and dropped, never pushed."""
    try:
        refs = astar._reference_helpers(plan, seg.mover.color)
    except Exception:
        return
    if not refs:
        return
    try:
        cands = heuristics.propose(env, seg.end, seg.mover,
                                   seg.helpers + refs, seg.support, b1=True)
    except Exception:
        return
    n_gen = n_rankable = 0
    state = CUR["state"]
    for cd in cands:
        if not _is_byref_robot(cd.subgoal.helper):
            continue                     # ordinary candidate, not by-reference
        cp = _apply_orig(env, plan, parent, child, seg, cd, by_reference=True)
        if cp is None:
            continue
        n_gen += 1
        if end2end._hidx(state, tuple(cd.subgoal.helper.position)) is not None:
            n_rankable += 1
    if n_gen:
        c["shadow_groups_with_byref"] += 1
        c["shadow_byref_generated"] += n_gen
        c["shadow_byref_rankable"] += n_rankable


# ---------------------------------------------------------------------------
# patch 2: eval.end2end._policy_logp -- ranking-pool + top-k membership
# ---------------------------------------------------------------------------

_policy_logp_orig = end2end._policy_logp


def _policy_logp_instrumented(policy, group, dev):
    logp, aux = _policy_logp_orig(policy, group, dev)
    c = CUR.get("counters")
    if c is None or logp is None:
        return logp, aux
    state = CUR["state"]
    sbc = CUR["starts_by_color"]

    def rec_is_byref(r):
        col = r["cand_helper"][1]
        return sbc.get(col) is not None and \
            tuple(r["cand_helper"][0]) != sbc[col]

    byref_ix = [i for i, r in enumerate(group) if rec_is_byref(r)]
    c["asis_byref_ranked"] += len(byref_ix)
    if byref_ix:
        # mirror compare.py's top-k selection (3 lines) to test membership
        from train.policy_common import _ix
        keys = [(_ix(r["cand_bottleneck"]), _ix(r["cand_support"]),
                 end2end._hidx(state, r["cand_helper"][0])) for r in group]
        order = sorted(range(len(group)),
                       key=lambda i: -logp.get(keys[i], -1e9))[:CUR["k"]]
        c["asis_byref_topk"] += sum(1 for i in order if i in set(byref_ix))
    return logp, aux


# ---------------------------------------------------------------------------
# patch 3: eval.compare._nn_astar_backward -- per-instance reset + final plan
# ---------------------------------------------------------------------------

_nn_orig = compare._nn_astar_backward


def _nn_instrumented(env, state, solver, policy, value, env_id, dev,
                     k, max_expansions, **kw):
    CUR["counters"] = _fresh_counters()
    CUR["state"] = state
    CUR["starts_by_color"] = _starts_by_color(state)
    CUR["k"] = k
    CUR["last_group_key"] = None
    plan, expansions, rejected, pruned = _nn_orig(
        env, state, solver, policy, value, env_id, dev, k, max_expansions,
        **kw)
    c = CUR["counters"]
    if plan is not None:
        c["final_plan_byref"] = int(_plan_contains_byref(plan))
        c["final_plan_parks"] = sum(
            1 for _n, d in plan.g.nodes(data=True) if d.get("ntype") == "park")
    c["env_id"] = env_id
    PER_INSTANCE.append(c)
    CUR["counters"] = None
    return plan, expansions, rejected, pruned


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    astar._apply = _apply_instrumented
    end2end._policy_logp = _policy_logp_instrumented
    compare._nn_astar_backward = _nn_instrumented

    instances, sha, _meta = compare.load_instances(ARGS.instances)
    if ARGS.limit:
        instances = instances[:ARGS.limit]
    placeholder = bool(instances) and not instances[0].get("d_star")

    t0 = time.time()
    rows = compare.run_backward(
        ARGS.policy, ARGS.value, instances, ARGS.k, ARGS.expansions,
        ARGS.device, log=lambda m: print(m, flush=True),
        anytime=True, b2=True, placeholder_d_star=placeholder)
    wall = time.time() - t0
    assert len(rows) == len(PER_INSTANCE), \
        f"row/counter mismatch: {len(rows)} vs {len(PER_INSTANCE)}"

    merged = []
    for row, c in zip(rows, PER_INSTANCE):
        assert row["env_id"] == c["env_id"]
        m = dict(row)
        m.pop("accounting", None)
        m.update(c)
        merged.append(m)

    keys = [k for k in _fresh_counters()]
    totals = {k: sum(m[k] for m in merged) for k in keys}
    summary = {
        "n": len(merged),
        "solved": sum(1 for m in merged if m["solved"]),
        "mean_expansions": (sum(m["expansions"] for m in merged)
                            / max(1, len(merged))),
        "wall_seconds": wall,
        "totals": totals,
        "instances_with_shadow_byref": sum(
            1 for m in merged if m["shadow_byref_generated"] > 0),
        "instances_final_plan_byref": sum(
            1 for m in merged if m["final_plan_byref"]),
        "instances_final_plan_parks": sum(
            1 for m in merged if m["final_plan_parks"] > 0),
    }

    out = {
        "protocol": {
            "script": "analysis/byref_topk_ablation.py",
            "config": ARGS.config,
            "instances_file": ARGS.instances,
            "instances_sha256": sha,
            "n_instances": len(instances),
            "expansions": ARGS.expansions,
            "k": ARGS.k,
            "device": ARGS.device,
            "shadow_probe": not ARGS.no_shadow,
            "checkpoints": {
                p: time.strftime("%Y-%m-%dT%H:%M:%S",
                                 time.localtime(Path(p).stat().st_mtime))
                for p in (ARGS.policy, ARGS.value)},
            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "search_mode": "eval.compare.run_backward(anytime=True, b2=True)"
                           " -- identical to the production --backward-b2 path;"
                           " shadow probe is read-only",
            "note": ("asis_* counters cover the driver as shipped; shadow_*"
                     " counters are the counterfactual candidate supply had"
                     " _nn_astar_backward been wired like nn/generate.py"
                     " (reference helpers + _apply(by_reference=True))."),
        },
        "summary": summary,
        "rows": merged,
    }
    outp = Path(ARGS.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(outp.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.rename(outp)

    print(json.dumps(summary, indent=1))
    print(f"BYREF_ABLATION DONE n={summary['n']} "
          f"asis_generated={totals['asis_byref_generated']} "
          f"asis_topk={totals['asis_byref_topk']} "
          f"shadow_generated={totals['shadow_byref_generated']} "
          f"-> {outp}")


if __name__ == "__main__":
    main()
