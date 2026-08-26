"""Apples-to-apples comparison engine for the report (owner directive 2026-08-21).

Problem this fixes: every payload's `aggregate.mean_moves` averages over the
instances THAT SYSTEM solved, so two systems' moves columns are not comparable
(the one that solves more hard puzzles looks worse). This module aligns N
result payloads that ran the SAME exam and computes:

  * solved / n                      (unchanged qualifier)
  * mean moves on the COMMON-SOLVED subset (instances every listed system
    solved) -- the one column on which all systems can be compared directly
  * pairwise vs designated reference systems: both-solved n, mean moves of
    each side on that both-solved set, delta, win/loss counts and the exact
    sign-test p (statistics reused from spr.gate, not reimplemented)
  * mean expansions (the budget column, from the payload aggregate)

Alignment contract (verified, 2026-08-21): all bench payloads in this repo --
eval.compare, spr.bench, spr.fwd and the v07 hybrid driver -- write one row
per instance IN INSTANCE-FILE ORDER (chunking is order-preserving through
eval/merge_compare_shards.py), carry `protocol.instances_sha256`, and every
row has `env_id`, `solved` and a realized playable-move count --
`realized_strict` on backward/spr rows, `moves` on the recorded forward rows
(for a move-level planner the move path IS the plan). env_id alone is NOT unique
(several instances per board), so rows are paired BY POSITION, guarded by
(a) identical instances_sha256, (b) identical row counts, (c) identical
env_id sequences. Any mismatch returns an error string instead of numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))            # self_play_robots/
from spr.gate import mcnemar_exact               # noqa: E402  (reused stats)


def pick_system(payload, prefer_kind=None, name_contains=None):
    """(name, system) of the first system with rows, honouring optional filters."""
    if not isinstance(payload, dict):
        return None, None
    for name, s in (payload.get("systems") or {}).items():
        if not (isinstance(s, dict) and s.get("rows")):
            continue
        if prefer_kind and s.get("kind") != prefer_kind:
            continue
        if name_contains and name_contains not in name:
            continue
        return name, s
    return None, None


class Entry:
    """One column of a comparison: a label + one system's rows."""

    def __init__(self, label, payload, prefer_kind=None, name_contains=None):
        self.label = label
        self.payload = payload if isinstance(payload, dict) else None
        self.sys_name, self.system = pick_system(self.payload, prefer_kind,
                                                 name_contains)
        self.rows = (self.system or {}).get("rows") or []
        self.agg = (self.system or {}).get("aggregate") or {}
        self.sha = ((self.payload or {}).get("protocol") or {}).get("instances_sha256")

    @property
    def ok(self):
        return bool(self.rows)


def _moves(r):
    """Realized playable moves of a solved row, across all row schemas."""
    v = r.get("realized_strict")
    return v if v is not None else r.get("moves")


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def compare(entries, ref_labels=(), dstar=None):
    """Align entries and compute the comparable columns.

    Returns {"error": str} if nothing aligns, else a dict:
      n, common_n, entries: [{label, ok, solved, n, moves_common, mean_exp,
                              delta_perfect, pairwise: {ref_label: {...}}}, ...]
    Entries whose payload is missing stay in the output with ok=False
    ("pending" rows). Alignment guards run over the present entries only.

    Perfect play: `dstar` is an optional per-position list of exact optima
    (None where unknown). When omitted, d* is read off the first entry whose
    rows carry real `d_star` values (graded exams). Each entry then gets
    `delta_perfect` = mean(moves - d*) over the common-solved positions with a
    known optimum (`perfect_n` of them; `perfect_mean` = mean d* there).
    """
    present = [e for e in entries if e.ok]
    if not present:
        return {"error": "no payloads on disk yet"}
    shas = {e.sha for e in present}
    if len(shas) != 1 or None in shas:
        return {"error": f"exam mismatch: instances_sha256 differ ({sorted(str(s)[:10] for s in shas)})"}
    ns = {len(e.rows) for e in present}
    if len(ns) != 1:
        return {"error": f"row-count mismatch: {sorted(ns)}"}
    n = ns.pop()
    seq0 = [r["env_id"] for r in present[0].rows]
    for e in present[1:]:
        if [r["env_id"] for r in e.rows] != seq0:
            return {"error": f"row-order mismatch between {present[0].label!r} and {e.label!r}"}

    common = [i for i in range(n) if all(e.rows[i]["solved"] for e in present)]
    refs = {e.label: e for e in present if e.label in set(ref_labels)}

    if dstar is None:                     # graded exams: d* lives on the rows
        for e in present:
            ds = [r.get("d_star") for r in e.rows]
            if any(ds):
                dstar = [d if d else None for d in ds]
                break
    perfect_idx = [i for i in common if dstar and dstar[i]] if dstar else []
    perfect_mean = _mean(dstar[i] for i in perfect_idx) if perfect_idx else None

    out_entries = []
    for e in entries:
        if not e.ok:
            out_entries.append({"label": e.label, "ok": False})
            continue
        rec = {
            "label": e.label, "ok": True, "system": e.sys_name,
            "n": n, "solved": sum(1 for r in e.rows if r["solved"]),
            "moves_common": _mean(_moves(e.rows[i]) for i in common),
            "mean_exp": e.agg.get("mean_expansions",
                                  _mean(r.get("expansions") for r in e.rows)),
            "delta_perfect": (_mean(_moves(e.rows[i]) - dstar[i]
                                    for i in perfect_idx)
                              if perfect_idx else None),
            "pairwise": {},
        }
        for rl, ref in refs.items():
            if ref is e:
                continue
            both = [i for i in range(n)
                    if e.rows[i]["solved"] and ref.rows[i]["solved"]]
            me = _mean(_moves(e.rows[i]) for i in both)
            them = _mean(_moves(ref.rows[i]) for i in both)
            wins = sum(1 for i in both
                       if _moves(e.rows[i]) < _moves(ref.rows[i]))
            losses = sum(1 for i in both
                         if _moves(e.rows[i]) > _moves(ref.rows[i]))
            rec["pairwise"][rl] = {
                "both_n": len(both), "mean_self": me, "mean_ref": them,
                "delta": (me - them) if None not in (me, them) else None,
                "wins": wins, "losses": losses,
                "sign_p": mcnemar_exact(wins, losses),
            }
        out_entries.append(rec)
    return {"n": n, "common_n": len(common), "sha": present[0].sha,
            "perfect_n": len(perfect_idx), "perfect_mean": perfect_mean,
            "entries": out_entries}
