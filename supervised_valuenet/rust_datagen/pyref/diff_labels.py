"""Gate-3 differ: dump file vs engine result file (DESIGN.md section 3 rules).

Backward (`replay_backward_decision` vs {"id","labels":[{"ctg","rejected"}]}):
per candidate, `ctg` equal (null==null) and `rejected` equal; `is_optimal`
recomputed on each side from its own ctg vector (ctg non-null and equal to
the min non-null ctg) and compared.

Forward (`replay_forward_state` vs {"id","cost_to_go","optimal_moves",
"legal_moves"}): `cost_to_go` equal; `legal_moves` equal as sets; if the
Python record has full=true, `best_moves` == engine `optimal_moves` as sets;
if full=false, Python's single best move must be a MEMBER of the engine's
optimal set (the taken move is tie-break-dependent; membership is the
tie-break-proof check).

Prints a per-(size x robots x task) pass/fail matrix; exits nonzero on any
diff or structural problem (missing/extra/errored ids, length mismatch).

    python rust_datagen/pyref/diff_labels.py --dump d.jsonl --results r.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def _opt_vector(labels):
    ctgs = [l["ctg"] for l in labels if l["ctg"] is not None]
    if not ctgs:
        return [False] * len(labels)
    best = min(ctgs)
    return [(l["ctg"] is not None and l["ctg"] == best) for l in labels]


def _moveset(moves):
    return frozenset((int(s), int(d)) for s, d in moves)


class Cell:
    __slots__ = ("lines", "units", "diffs", "examples")

    def __init__(self):
        self.lines = 0
        self.units = 0
        self.diffs = defaultdict(int)
        self.examples = []

    def diff(self, kind, line_id, detail):
        self.diffs[kind] += 1
        if len(self.examples) < 5:
            self.examples.append(f"{kind} @ {line_id}: {detail}")

    @property
    def total_diffs(self):
        return sum(self.diffs.values())


def diff_backward(cell, line, res):
    cands = line["candidates"]
    py = line["python_labels"]
    cell.units += len(cands)
    if "error" in res:
        cell.diff("struct", line["id"], res["error"])
        return
    eng = res.get("labels")
    if eng is None or len(eng) != len(py):
        cell.diff("struct", line["id"],
                  f"labels len {None if eng is None else len(eng)} != {len(py)}")
        return
    for i, (pl, el) in enumerate(zip(py, eng)):
        if bool(pl["rejected"]) != bool(el["rejected"]):
            cell.diff("rejected", line["id"],
                      f"cand {i}: py={pl['rejected']} eng={el['rejected']}")
        if pl["ctg"] != el["ctg"]:
            cell.diff("ctg", line["id"],
                      f"cand {i}: py={pl['ctg']} eng={el['ctg']}")
    for i, (po, eo) in enumerate(zip(_opt_vector(py), _opt_vector(eng))):
        if po != eo:
            cell.diff("is_optimal", line["id"], f"cand {i}: py={po} eng={eo}")


def diff_forward(cell, line, res):
    py = line["python_labels"]
    cell.units += len(py["legal_moves"])
    if "error" in res:
        cell.diff("struct", line["id"], res["error"])
        return
    if py["cost_to_go"] != res.get("cost_to_go"):
        cell.diff("value", line["id"],
                  f"py={py['cost_to_go']} eng={res.get('cost_to_go')}")
    if _moveset(py["legal_moves"]) != _moveset(res.get("legal_moves", [])):
        cell.diff("legal", line["id"],
                  f"py={sorted(_moveset(py['legal_moves']))} "
                  f"eng={sorted(_moveset(res.get('legal_moves', [])))}")
    eng_opt = _moveset(res.get("optimal_moves", []))
    if py["full"]:
        if _moveset(py["best_moves"]) != eng_opt:
            cell.diff("best", line["id"],
                      f"full: py={sorted(_moveset(py['best_moves']))} "
                      f"eng={sorted(eng_opt)}")
    else:
        if len(py["best_moves"]) != 1:
            cell.diff("struct", line["id"],
                      f"non-full record with {len(py['best_moves'])} best moves")
        elif tuple(py["best_moves"][0]) not in eng_opt:
            cell.diff("best", line["id"],
                      f"taken {py['best_moves'][0]} not in engine optimal set "
                      f"{sorted(eng_opt)}")


BACK_KINDS = ["ctg", "rejected", "is_optimal", "struct"]
FWD_KINDS = ["value", "legal", "best", "struct"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dump", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--quiet", action="store_true",
                   help="matrix only, no per-diff examples")
    a = p.parse_args()

    results = {}
    with open(a.results) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            r = json.loads(raw)
            results[r["id"]] = r

    cells = defaultdict(Cell)     # (n, robots, task) -> Cell
    seen = set()
    with open(a.dump) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            line = json.loads(raw)
            lid = line["id"]
            seen.add(lid)
            n = line["board"]["n"]
            if line["task"] == "replay_backward_decision":
                task = "backward"
                robots = len(line["state"]["helpers"]) + 1
            else:
                task = "forward"
                robots = len(line["positions"])
            cell = cells[(n, robots, task)]
            cell.lines += 1
            res = results.get(lid)
            if res is None:
                cell.diff("struct", lid, "no result line for id")
                continue
            if task == "backward":
                diff_backward(cell, line, res)
            else:
                diff_forward(cell, line, res)

    extra = set(results) - seen
    if extra:
        k = next(iter(cells)) if cells else (0, 0, "backward")
        cells[k].diff("struct", "<extra>",
                      f"{len(extra)} result ids not in dump, e.g. "
                      f"{sorted(extra)[:3]}")

    # -- matrix ------------------------------------------------------------
    hdr = (f"{'size':>4} {'robots':>6} {'task':<9} {'lines':>6} {'units':>7}  "
           f"{'mismatches':<44} status")
    print(hdr)
    print("-" * len(hdr))
    any_diff = False
    tot_lines = tot_units = 0
    for key in sorted(cells, key=lambda k: (k[0], k[1], k[2])):
        c = cells[key]
        n, robots, task = key
        kinds = BACK_KINDS if task == "backward" else FWD_KINDS
        mism = " ".join(f"{k}={c.diffs.get(k, 0)}" for k in kinds)
        status = "PASS" if c.total_diffs == 0 else "FAIL"
        any_diff |= c.total_diffs > 0
        tot_lines += c.lines
        tot_units += c.units
        print(f"{n:>4} {robots:>6} {task:<9} {c.lines:>6} {c.units:>7}  "
              f"{mism:<44} {status}")
    print("-" * len(hdr))
    if any_diff:
        print(f"RESULT: FAIL ({tot_lines} lines, {tot_units} units compared)")
        if not a.quiet:
            for key in sorted(cells, key=lambda k: (k[0], k[1], k[2])):
                for ex in cells[key].examples:
                    print(f"  [{key[0]}x{key[0]} r{key[1]} {key[2]}] {ex}")
        sys.exit(1)
    print(f"RESULT: ALL GREEN ({tot_lines} lines, {tot_units} units compared)")
    sys.exit(0)


if __name__ == "__main__":
    main()
