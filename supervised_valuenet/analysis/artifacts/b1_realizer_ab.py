"""Realizer A/B for the Lever B1 changes (analysis/b1_design.md §4).

The B1 work touches eval/realize.py in three additive places (a position
snapshot in the failure context, a first-failure record for repair proposals,
and a scheduling dependency that only fires for plans containing `park`
nodes). No stored plan contains a park node, so every stored realization must
be IDENTICAL under the old and new module: same success/failure verdict, same
move count.

Inputs: the pre-change snapshot of eval/realize.py (argv[1]), the stored plan
pickles (all plans from the original 450-run that failed strict realization
at the time, plus the post-fix residual set), and eval/data/bench450.jsonl
for the matching instances. Output: eval/results/realizer_b1_ab.json.

Run:
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python3 analysis/artifacts/b1_realizer_ab.py \
        <realize_pre_b1.py> eval/results/realizer_b1_ab.json
"""
import importlib.util
import json
import os
import pickle
import sys

os.environ.setdefault("RR_ENV_DIR", "environments")
os.environ.setdefault("RR_GRID", "16")
os.environ.setdefault("RR_ROBOTS", "4")
sys.path.insert(0, ".")

from GridEnv import GridEnv, State, Robot_at          # noqa: E402
from move_planner.state import COLOR_ORDER            # noqa: E402
import eval.realize as new_realize                    # noqa: E402


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


old_realize = load_module(sys.argv[1], "old_realize")

bench = [json.loads(l) for l in open("eval/data/bench450.jsonl")]

results = {"n": 0, "n_equal": 0, "mismatches": [], "sources": {}}
for pkl in ("analysis/artifacts/failing_plans.pkl",
            "analysis/artifacts/residual_failing_plans.pkl"):
    plans = pickle.load(open(pkl, "rb"))
    results["sources"][pkl] = len(plans)
    for idx, plan in sorted(plans.items()):
        inst = bench[idx]
        env, _ = GridEnv.from_env(inst["env_id"])
        pos = [tuple(p) for p in inst["positions"]]
        tidx = inst["target_idx"]
        st = State(target=tuple(inst["target"]),
                   target_robot=Robot_at(position=pos[tidx],
                                         color=COLOR_ORDER[tidx]),
                   helpers=[Robot_at(position=pos[j], color=COLOR_ORDER[j])
                            for j in range(len(pos)) if j != tidx])
        m_old = old_realize.strict_moves(env, st, plan, log=None)
        m_new = new_realize.strict_moves(env, st, plan, log=None)
        results["n"] += 1
        if m_old == m_new:
            results["n_equal"] += 1
        else:
            results["mismatches"].append(
                {"pkl": pkl, "idx": idx, "old": m_old, "new": m_new})

results["identical"] = (results["n_equal"] == results["n"])
json.dump(results, open(sys.argv[2], "w"), indent=1)
print(f"[b1_realizer_ab] {results['n_equal']}/{results['n']} identical; "
      f"mismatches: {len(results['mismatches'])}")
print("wrote", sys.argv[2])
