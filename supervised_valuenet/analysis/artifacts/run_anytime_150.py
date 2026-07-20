"""Fallback: anytime backward benchmark on the FIRST 150 instances of bench450.

Scratch copy of eval.compare.main()'s backward path (repo files untouched);
same output paths, protocol marked as the 150-instance subset. CPU only.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet")
os.chdir("/mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet")

import torch
torch.manual_seed(0)
torch.set_num_threads(8)

from eval.compare import load_instances, run_backward, aggregate, write_markdown

N = 150
INSTANCES = "eval/data/bench450.jsonl"
POLICY = "checkpoints_backward/policy_v2.ckpt"
VALUE = "checkpoints_backward/value_v2.ckpt"
OUT = "eval/results/comparison_backward_anytime.json"
MD = "eval/results/COMPARISON_anytime.md"

instances, sha, meta = load_instances(INSTANCES)
instances = instances[:N]

protocol = {
    "expansions": 1200,
    "k": 5,
    "instances_file": INSTANCES + f" (FIRST {N} INSTANCES ONLY — CPU runtime cap)",
    "instances_sha256": sha,
    "n_instances": N,
    "instances_meta": meta,
    "checkpoints": {c: time.strftime("%Y-%m-%dT%H:%M:%S",
                                     time.localtime(os.path.getmtime(c)))
                    for c in (POLICY, VALUE)},
    "device": "cpu",
    "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "results_file": OUT,
    "command": (f"scratch run_anytime_150.py == eval.compare --backward-anytime "
                f"--device cpu on instances[:{N}] of {INSTANCES}"),
    "expansion_definition": "one popped search node whose children are "
                            "generated (= 1 policy pass + 1 batched value "
                            "pass over <= k children, in both systems)",
}

name = "backward subgoal planner (anytime realization-checked)"
rows = run_backward(POLICY, VALUE, instances, 5, 1200, "cpu", anytime=True)
systems = {name: {"kind": "backward",
                  "aggregate": aggregate(rows, "realized_strict"),
                  "rows": rows}}
out = Path(OUT)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"protocol": protocol, "systems": systems}, indent=2) + "\n")
write_markdown(MD, protocol, systems, [name])
print(f"[compare] wrote {OUT} and {MD} (n={N})")
ag = systems[name]["aggregate"]
print(f"solved {ag['solved']}/{ag['n']} regret={ag['mean_regret']}")
