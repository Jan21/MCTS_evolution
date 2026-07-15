"""Count gate-3 backward dump decisions that hit the propose FALLBACK path.

`heuristics.propose` has a fallback: when `propose_subgoal_states` returns
nothing, it retries pinned to each dependent-edge support of the goal (a
Python SET iteration -- the set-order-sensitive path task A's review flagged
for STOCK pickled boards). This tool re-runs the raw proposal for every
decision in a dump file and reports how many decisions took the fallback,
so the corpus can be shown to actually exercise it.

    python rust_datagen/pyref/count_fallback.py --dump out/gate_b16stock.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PYREF_DIR = Path(__file__).resolve().parent
SV_DIR = PYREF_DIR.parent.parent
for _p in (str(SV_DIR), str(PYREF_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CHILD_FLAG = "RR_PYREF_CHILD_FB"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dump", required=True)
    a = p.parse_args()
    if os.environ.get(CHILD_FLAG) != "1":
        env = {**os.environ,
               "RR_GRID": "16", "RR_ROBOTS": "4", "RR_WALLS": "48",
               "RR_ENV_DIR": str(SV_DIR / "environments"),
               CHILD_FLAG: "1", "PYTHONPATH": str(SV_DIR)}
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                             *sys.argv[1:]], env=env, cwd=str(SV_DIR)).returncode
        sys.exit(rc)

    import common
    from skeleton.astar import _segment
    from GridEnv import State

    envs = {}
    n_lines = n_fallback = 0
    fallback_ids = []
    per_board = defaultdict(int)
    with open(a.dump) as f:
        for raw in f:
            line = json.loads(raw)
            if line["task"] != "replay_backward_decision":
                continue
            n_lines += 1
            b = line["board"]
            key = common.grid_sha(b["grid_data"])
            if key not in envs:
                envs[key] = common.build_env(b["grid_data"], b["n"],
                                             mode="lazy", weight=2,
                                             table_workers=8)
            env = envs[key]
            state = common.de_state(line["state"])
            plan = common.de_plan(line["plan"])
            parent, child = line["open_edge"]
            seg = _segment(plan, state, parent, child)
            raw_props = list(env.propose_subgoal_states(
                State(target=seg.end, target_robot=seg.mover,
                      helpers=seg.helpers),
                support_robot=seg.support))
            if not raw_props:
                n_fallback += 1
                per_board[b["env_id"]] += 1
                if len(fallback_ids) < 20:
                    fallback_ids.append(line["id"])
    print(f"[fallback] {n_lines} backward decisions, {n_fallback} hit the "
          f"propose dependent-support fallback "
          f"({n_fallback / max(n_lines, 1):.1%})")
    print(f"[fallback] boards with fallback decisions: {dict(per_board)}")
    print(f"[fallback] example ids: {fallback_ids[:10]}")
    sys.exit(0 if n_fallback > 0 else 2)


if __name__ == "__main__":
    main()
