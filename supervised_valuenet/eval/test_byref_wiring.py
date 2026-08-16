"""Regression + smoke tests for the by-reference (Lever B2) driver wiring.

Two checks, both cheap enough for a login node (a few CPU-minutes at the
default settings):

  A. FLAG OFF IS THE OLD DRIVER. `eval.compare.run_backward` at its defaults
     is compared, row for row, against the SAME function extracted from a git
     revision (default HEAD) -- same nets, same instances, same budget, same
     --backward-anytime --backward-b2 setting. Every field except wall-clock
     `seconds` must be equal, including the dumped primitive-move sequences.
     This is the guarantee that the existing results stay reproducible: the
     candidate list at every expansion, the ranking and the search outcome are
     untouched when `byref` is off.

  B. FLAG ON GENERATES BY-REFERENCE CANDIDATES AND REPLAYABLE PLANS. The same
     run with `byref=True` must (i) put by-reference candidates -- helpers
     standing at a cell that is not their robot's start -- in front of the
     policy net (`accounting.byref_cands_ranked` > 0 somewhere) and (ii)
     produce solved rows whose dumped move sequences pass INDEPENDENT replay
     certification (`eval.replay_validate`, physics-only imports).

Usage (from supervised_valuenet/, with the project venv active):

    PYTHONPATH=. python -m eval.test_byref_wiring            # both checks
    PYTHONPATH=. python -m eval.test_byref_wiring --n 6 --budget 400
    PYTHONPATH=. python -m eval.test_byref_wiring --only b   # skip the git leg

Exit status 0 = all requested checks passed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

BANK = "scaling/runs/b2_banked_cap20000.json"
REPO_REL = "supervised_valuenet/eval/compare.py"


def _load_ref_module(rev):
    """Import `eval/compare.py` as it exists at `rev` under a private name."""
    src = subprocess.run(["git", "show", f"{rev}:{REPO_REL}"],
                         capture_output=True, text=True, check=True,
                         cwd=Path(__file__).resolve().parents[2]).stdout
    tmp = Path(tempfile.mkdtemp(prefix="byref_ref_")) / "compare_ref.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("compare_ref", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strip(rows):
    """Rows minus the fields that legitimately differ between two runs."""
    out = []
    for r in rows:
        d = {k: v for k, v in r.items() if k != "seconds"}
        d["accounting"] = {k: v for k, v in (r.get("accounting") or {}).items()
                           if not k.startswith("byref_")}
        out.append(d)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instances", default="eval/data/bench450.jsonl")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--budget", type=int, default=250)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--pool", type=int, default=60,
                   help="byref candidate pool cap m (check B)")
    p.add_argument("--config", default="g16r4")
    p.add_argument("--policy", default=None)
    p.add_argument("--value", default=None)
    p.add_argument("--rev", default="HEAD",
                   help="git revision holding the pre-flag driver (check A)")
    p.add_argument("--env-dir", default="environments")
    p.add_argument("--only", choices=["a", "b"], default=None)
    a = p.parse_args()

    from eval import compare as new

    bank = json.loads(Path(BANK).read_text())[a.config]
    policy = a.policy or bank["policy"]
    value = a.value or bank["value"]
    instances = [json.loads(l) for l in
                 Path(a.instances).read_text().splitlines() if l.strip()][:a.n]
    print(f"[test] {len(instances)} instances, budget {a.budget}, k {a.k}\n"
          f"[test] policy={policy}\n[test] value={value}")

    common = dict(instances=instances, k=a.k, budget=a.budget, device="cpu",
                  log=None, anytime=True, b2=True, dump_moves=True)
    failures = []

    # ---- A: flag off == pre-flag driver -----------------------------------
    if a.only in (None, "a"):
        ref = _load_ref_module(a.rev)
        print(f"[test A] reference driver from {a.rev}")
        rows_ref = ref.run_backward(policy, value, **common)
        rows_new = new.run_backward(policy, value, **common)
        if _strip(rows_ref) == _strip(rows_new):
            print(f"[test A] PASS: {len(rows_new)} rows identical "
                  f"(solved {sum(r['solved'] for r in rows_new)}, "
                  f"expansions {sum(r['expansions'] for r in rows_new)})")
        else:
            for i, (x, y) in enumerate(zip(_strip(rows_ref), _strip(rows_new))):
                if x != y:
                    print(f"[test A] FAIL row {i}:\n  ref={x}\n  new={y}")
            failures.append("A")

    # ---- B: flag on supplies by-reference candidates, plans replay --------
    if a.only in (None, "b"):
        rows = new.run_backward(policy, value, byref=True, byref_pool=a.pool,
                                **common)
        ranked = sum(r["accounting"].get("byref_cands_ranked", 0) for r in rows)
        topk = sum(r["accounting"].get("byref_cands_topk", 0) for r in rows)
        solved = [r for r in rows if r["solved"]]
        print(f"[test B] byref candidates reaching the policy net: {ranked}; "
              f"selected into the top-k: {topk}; "
              f"solved {len(solved)}/{len(rows)}")
        if ranked <= 0:
            print("[test B] FAIL: no by-reference candidate was featurizable")
            failures.append("B-supply")
        if not solved:
            print("[test B] FAIL: no solved row to certify")
            failures.append("B-solved")
        else:
            tmp = Path(tempfile.mkdtemp(prefix="byref_rep_"))
            inst_f = tmp / "instances.jsonl"
            inst_f.write_text("".join(json.dumps(i) + "\n" for i in instances))
            comp = tmp / "comparison.json"
            comp.write_text(json.dumps({
                "protocol": {"instances_file": str(inst_f)},
                "systems": {"backward [by-reference]":
                            {"kind": "backward", "rows": rows}}}))
            rc = subprocess.run([sys.executable, "-m", "eval.replay_validate",
                                 "--compare", str(comp),
                                 "--env-dir", a.env_dir]).returncode
            if rc == 0:
                print(f"[test B] PASS: {len(solved)} solved rows replay-certified")
            else:
                print(f"[test B] FAIL: replay_validate rc={rc}")
                failures.append("B-replay")

    print("ALL CHECKS PASSED" if not failures
          else f"FAILED: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
