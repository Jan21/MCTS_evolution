"""Paired A/B for forward payloads -- `spr.gate.paired` reused, not reimplemented.

`spr.gate.paired` is the promotion instrument of PROBLEM.md 4.7 (McNemar exact
test on the per-instance solved vectors + a sign test on moves over the
both-solved subset). It reads `payload["systems"][name]["rows"][i]["solved"]`
and `["realized_strict"]` -- which `spr.fwd.bench` writes -- but it locates the
system through `spr.arena._backward_system`, which requires
`system["kind"] == "backward"`. Forward payloads are `kind: "forward"` (that is
what `eval/merge_compare_shards.py` and `eval.compare.aggregate` key off, so it
must stay), hence this one-function shim: a shallow relabelled VIEW of the
payload is handed to `spr.gate.paired`. Nothing on disk is modified and no
statistic is re-implemented here.

    python -m spr.fwd.gate compare --a A.json --b B.json [--out X.json]

Cross-arm comparisons work too (A forward, B backward from `spr.bench`): both
sides expose `solved` and `realized_strict` per instance, and the instance
sha256 guard inside `spr.gate.paired` refuses payloads that are not the same
exam. Note the sha check makes a forward-vs-backward pairing valid only on a
shared instance file (`eval/data/bench450.jsonl` at g16r4).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _pairable_row(r):
    """Backfill `realized_strict` for forward rows that lack it.

    `spr.fwd.bench` writes it, but rows recorded by `eval.compare.run_forward`
    (e.g. the frozen baselines in `eval/results/comparison_forward.json`) carry
    only `moves`, which for a forward row IS the realized primitive-move count.
    Backward rows already have `realized_strict` and use `moves` for the [color,
    dir] dump, so they are left alone.
    """
    if "realized_strict" in r or isinstance(r.get("moves"), list):
        return r
    return {**r, "realized_strict": r.get("moves")}


def as_pairable(payload):
    """Shallow view whose row-bearing systems are labelled `backward` so
    `spr.arena._backward_system` (and therefore `spr.gate.paired`) finds them,
    with `realized_strict` backfilled on recorded forward rows. Nothing on disk
    is touched."""
    out = dict(payload)
    out["systems"] = {
        n: ({**s, "kind": "backward",
             "rows": [_pairable_row(r) for r in s["rows"]]} if s.get("rows") else s)
        for n, s in payload["systems"].items()}
    return out


def select(payload, wanted=None):
    """Narrow a payload to ONE system (by name substring).

    Needed because a recorded `eval.compare` file holds several systems --
    `eval/results/comparison_forward.json` carries four forward checkpoints plus
    a `pending` backward row -- while `spr.arena._backward_system` returns the
    FIRST row-bearing system it meets. Narrowing first makes the choice explicit
    instead of positional.
    """
    if not wanted:
        return payload
    from spr.fwd.bench import _pick_system
    n, s = _pick_system(payload, wanted)
    out = dict(payload)
    out["systems"] = {n: s}
    return out


def paired(a_payload, b_payload, a_system=None, b_system=None):
    from spr.gate import paired as _paired
    return _paired(as_pairable(select(a_payload, a_system)),
                   as_pairable(select(b_payload, b_system)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True, help="the challenger")
    c.add_argument("--b", required=True, help="the incumbent")
    c.add_argument("--a-system", default=None,
                   help="substring picking A's system when the payload has several")
    c.add_argument("--b-system", default=None,
                   help="substring picking B's system (e.g. a recorded ckpt name)")
    c.add_argument("--out", default=None)
    a = p.parse_args(argv)
    res = paired(json.loads(Path(a.a).read_text()), json.loads(Path(a.b).read_text()),
                 a.a_system, a.b_system)
    res["a"], res["b"] = a.a, a.b
    res["a_system"], res["b_system"] = a.a_system, a.b_system
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")
    print(f"SPR FWD GATE DONE {a.a} vs {a.b} "
          f"(mcnemar_p={res['mcnemar_p']:.4g} sign_p_moves={res['sign_p_moves']:.4g})",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
