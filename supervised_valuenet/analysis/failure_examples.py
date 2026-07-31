"""Failure gallery data for 24×24 · 8 robots (owner request, 2026-07-31).

Reads the g24r8 beyond-oracle rows and writes
analysis/artifacts/failure_examples.json: a taxonomy of the backward
planner's failures there, plus a few deterministic concrete boards
(walls + robots + goal) so the report can DRAW what failure looks like.

Failure modes measurable from the rows:
  never_completed  plan_found=false, plans_rejected=0 — the search ran out
                   of budget without ever completing an abstract plan
  all_rejected     plans_rejected>0 and not solved — plans completed on
                   paper, every one failed the move-by-move play-out
Examples picked (first by env order, deterministic):
  fwd_solved       a puzzle the move-by-move planner solved and the
                   subgoal planner did not (the honest case)
  all_rejected     the paper-plans-refused case
  never_completed  the budget-exhausted case

    PYTHONPATH=. python3 -m analysis.failure_examples
"""
import json
import os
import pickle

from simulate import wall_sets

CFG = "g24r8"
INST = f"scaling/data/{CFG}/bench.unsolved.jsonl"
BWD = f"scaling/results/{CFG}/comparison_ungraded_b2.json"
FWD = f"scaling/results/{CFG}/comparison_ungraded.json"
ENVD = f"environments_{CFG}"
OUT = "analysis/artifacts/failure_examples.json"


def board(env_id, inst):
    with open(os.path.join(ENVD, f"env_{env_id}.pkl"), "rb") as fh:
        grid = pickle.load(fh)["grid_data"]
    wr, wd = wall_sets(grid)
    size = 24
    return {"size": size,
            "walls_right": sorted(map(list, wr)),
            "walls_down": sorted(map(list, wd)),
            "positions": inst["positions"],
            "target_idx": inst["target_idx"],
            "target": inst["target"]}


def main():
    insts = [json.loads(l) for l in open(INST) if l.strip()]
    bwd = [s for s in json.load(open(BWD))["systems"].values()
           if s.get("kind") == "backward"][0]["rows"]
    fwd = [s for s in json.load(open(FWD))["systems"].values()
           if s.get("kind") == "forward"][0]["rows"]
    assert len(insts) == len(bwd) == len(fwd)

    fails = [(i, r) for i, r in enumerate(bwd) if not r.get("solved")]
    tax = {
        "n_set": len(bwd),
        "n_bwd_failed": len(fails),
        "never_completed": sum(1 for _, r in fails
                               if not r.get("plan_found")
                               and not r.get("plans_rejected")),
        "all_rejected": sum(1 for _, r in fails if r.get("plans_rejected")),
        "fwd_solved_of_bwd_failed": sum(1 for i, _ in fails
                                        if fwd[i].get("solved")),
        "median_seconds_failed": sorted(
            r["seconds"] for _, r in fails)[len(fails) // 2],
    }

    def pick(pred):
        for i, r in fails:
            if pred(i, r):
                return i, r
        return None, None

    picks = {}
    for key, pred in (
            ("fwd_solved", lambda i, r: fwd[i].get("solved")),
            ("all_rejected", lambda i, r: r.get("plans_rejected")),
            ("never_completed", lambda i, r: not r.get("plan_found")
             and not r.get("plans_rejected"))):
        i, r = pick(pred)
        if r is None:
            continue
        picks[key] = {
            "row_index": i,
            "env_id": insts[i]["env_id"],
            "board": board(insts[i]["env_id"], insts[i]),
            "bwd": {k: r.get(k) for k in
                    ("expansions", "seconds", "plan_found",
                     "plans_rejected", "solved")},
            "fwd": {k: fwd[i].get(k) for k in
                    ("expansions", "seconds", "solved")},
        }

    payload = {"config": CFG, "taxonomy": tax, "examples": picks,
               "sources": [INST, BWD, FWD]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".tmp", "w") as fh:
        json.dump(payload, fh)
    os.replace(OUT + ".tmp", OUT)
    print("taxonomy:", json.dumps(tax))
    print("examples:", {k: v["env_id"] for k, v in picks.items()})
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
