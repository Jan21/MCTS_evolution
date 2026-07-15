"""Gate-5 `make verify` leg: the full 12-instance regression, three-way.

Checks the committed `golden/regression/regression12.jsonl` corpus by
(1) replaying it through the CURRENT Python solver (`replay_python.py`) and
diffing against the corpus' stored `python_labels` -- proves the corpus
still matches the reference implementation; (2) replaying it through the
Rust engine (`datagen replay`) and diffing likewise; (3) asserting, from
`regression12.meta.json`, that every flagged defect candidate (whose
acceptance would recreate one of the two archived residual-defect classes)
is REJECTED by both engines. Exits nonzero on any failure.

    python rust_datagen/pyref/verify_regression.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PYREF_DIR = Path(__file__).resolve().parent
CRATE = PYREF_DIR.parent
DEFAULT_ENGINE = CRATE / "target" / "release" / "datagen"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--engine-bin", default=str(DEFAULT_ENGINE))
    p.add_argument("--skip-python", action="store_true",
                   help="rust + defect assertions only")
    a = p.parse_args()

    dump = CRATE / "golden" / "regression" / "regression12.jsonl"
    meta = json.loads((CRATE / "golden" / "regression"
                       / "regression12.meta.json").read_text())
    tmp = Path(tempfile.mkdtemp(prefix="regr12_"))
    py = sys.executable
    failures = 0

    def diff(results_path, tag):
        nonlocal failures
        rc = subprocess.run([py, str(PYREF_DIR / "diff_labels.py"),
                             "--dump", str(dump),
                             "--results", str(results_path)]).returncode
        print(f"[regr12] {tag} differ exit: {rc}")
        failures += rc != 0

    sides = {}
    if not a.skip_python:
        py_out = tmp / "py.jsonl"
        rc = subprocess.run([py, str(PYREF_DIR / "replay_python.py"),
                             "--dump", str(dump), "--out", str(py_out),
                             "--workers", "4"]).returncode
        if rc != 0:
            print("[regr12] python replay FAILED")
            failures += 1
        else:
            diff(py_out, "python-replay")
            sides["python"] = py_out

    rust_out = tmp / "rust.jsonl"
    rc = subprocess.run([a.engine_bin, "replay", "--work", str(dump),
                         "--out", str(rust_out), "--threads", "2",
                         "--quiet"]).returncode
    if rc != 0:
        print("[regr12] rust replay FAILED")
        failures += 1
    else:
        diff(rust_out, "rust-replay")
        sides["rust"] = rust_out

    for side, path in sides.items():
        results = {}
        with open(path) as f:
            for raw in f:
                r = json.loads(raw)
                results[r["id"]] = r
        for it in meta["items"]:
            lab = results[it["id"]]["labels"][it["defect_candidate_index"]]
            if lab["rejected"] is not True:
                print(f"[regr12] {side}: {it['id']} defect candidate "
                      f"#{it['defect_candidate_index']} "
                      f"({it['defect_class']}) NOT rejected")
                failures += 1
        print(f"[regr12] {side}: 12/12 defect candidates rejected"
              if failures == 0 else f"[regr12] {side}: FAILURES above")

    print("[regr12] RESULT:", "ALL GREEN" if failures == 0 else "FAIL")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
