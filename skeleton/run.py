"""Benchmark + realizability harness for the skeleton algorithm.

Keeps the original benchmark.py untouched. For each env it solves, structurally
validates, simulates for true realizability, and reports cost.

    python -m skeleton.run --n 20
"""
from __future__ import annotations

import argparse

from GridEnv import GridEnv
from validate_plan import validate
from evaluate_plan import evaluate_plan
from simulate import verify_plan
from skeleton.astar import AStar


def run(n=20, beam=None, verbose=True):
    algo = AStar(beam=beam)
    solved = real = 0
    costs = []
    for i in range(n):
        env, state = GridEnv.from_env(i)
        res = algo.solve(env, state)
        ok = validate(res.best_plan, env, state).passed
        rz = verify_plan(res.best_plan, env, state)[0] if ok else False
        cost = evaluate_plan(res.best_plan, env) if ok else None
        solved += ok
        real += rz
        if ok and rz:
            costs.append(cost)
        if verbose:
            print(f"env_{i:>3}: solved={str(ok)[0]} realizable={str(rz)[0]} cost={cost}")
    avg = round(sum(costs) / len(costs), 2) if costs else None
    print(f"\nsolved={solved}/{n}  realizable={real}/{n}  avg_cost={avg}")
    return solved, real, avg


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--beam", type=int, default=None)
    run(**vars(p.parse_args()))
