"""Backward (subgoal) labeler with a per-instance wall-time guard.

Same driver loop as `nn.generate.generate` (same RNG consumption, same
attempts cap, same solver settings) with two additions for larger boards:
`--max-candidates` defaults to 14, and every rollout runs under a SIGALRM
wall-time cap -- the candidate-capped rollout can wander, so instances that
exceed `--timeout` seconds are dropped and counted.

Run via `scaling.gen_data`, which spawns this module in a subprocess carrying
the config's RR_* environment (this module reads the config at import like
every other repo module; it never sets env vars itself).

    python -m scaling.backward_label --graphs 0-1 --per-graph 5 \
        --out scaling/data/g16r6/backward.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import signal
import time
from pathlib import Path

from GridEnv import GridEnv
from nn.generate import parse_graphs, random_instance, rollout
from skeleton.astar import AStar


class _Timeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _Timeout()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graphs", required=True)
    p.add_argument("--per-graph", type=int, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-candidates", type=int, default=14)
    p.add_argument("--timeout", type=int, default=120,
                   help="per-instance rollout wall-time cap (s)")
    a = p.parse_args()

    graphs = parse_graphs(a.graphs)
    rng = random.Random(a.seed)
    solver = AStar(max_iters=4000, max_frontier=40_000)
    _, s0 = GridEnv.from_env(graphs[0])
    colors = [s0.target_robot.color] + [h.color for h in s0.helpers]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGALRM, _raise_timeout)
    n_rec = n_inst = n_timeout = 0
    t0 = time.time()
    with open(out, "w") as f:
        for gid in graphs:
            env, _ = GridEnv.from_env(gid)   # cache build stays outside the alarm
            kept = attempts = 0
            while kept < a.per_graph and attempts < a.per_graph * 4:
                attempts += 1
                st = random_instance(env, colors, rng)
                signal.alarm(a.timeout)
                try:
                    recs = rollout(env, st, solver, gid, a.max_candidates)
                except _Timeout:
                    n_timeout += 1
                    recs = []
                except Exception:
                    recs = []
                finally:
                    signal.alarm(0)
                if not recs:
                    continue
                for r in recs:
                    f.write(json.dumps(r) + "\n")
                n_rec += len(recs)
                n_inst += 1
                kept += 1
            print(f"graph {gid}: {kept} instances, {n_rec} records, "
                  f"{n_timeout} timeouts so far ({time.time() - t0:.0f}s)",
                  flush=True)
    print(f"done: {n_inst} instances, {n_rec} records, {n_timeout} timeouts "
          f"-> {out}")


if __name__ == "__main__":
    main()
