"""Materialize a config's shared benchmark instance file (+ d* references).

Sampling mirrors `eval.bench_instances.materialize`: one `random.Random(seed)`
consumed sequentially over the config's bench boards, `per_board` draws per
board, the same relaxed-reachability rejection. With no oracle failures the
output is line-identical to what `eval.bench_instances` would produce for the
same boards/per-board/seed. The difference: an instance whose exact move
oracle solve fails (expansion cap or wall-time cap) is KEPT with d_star=null
-- the oracle failure rate is itself a scaling datapoint and is stored in the
meta. A companion `<out stem>.solved.jsonl` (d_star non-null lines only) is
also written because `eval.compare` computes `cost - d_star` and cannot take
nulls.

    python -m scaling.bench --config g16r6 --per-board 3 --seed 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

from scaling.configs import REPO, apply_env, env, get, parse_ids


class _Timeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _Timeout()


def sample_and_label(wr, wd, rng, grid, robots, max_expansions, wall_time,
                     max_try=200):
    """One draw, mirroring `move_planner.evaluate.sample_instance`'s loop, but
    an oracle failure returns the instance with d=None instead of resampling
    (identical RNG consumption whenever the oracle succeeds)."""
    from move_planner import oracle as move_bfs

    cells_all = [(x, y) for y in range(grid) for x in range(grid)]
    for _ in range(max_try):
        cells = rng.sample(cells_all, robots + 1)
        positions = tuple(cells[:robots])
        target = cells[robots]
        tidx = rng.randrange(robots)
        hd = move_bfs.relaxed_target_dist(target, wr, wd, grid)
        if hd.get(positions[tidx], move_bfs.INF) >= move_bfs.INF:
            continue                     # unreachable even relaxed: resample
        signal.setitimer(signal.ITIMER_REAL, wall_time)
        try:
            d = move_bfs.solve(positions, tidx, target, wr, wd, grid, hd,
                               max_expansions=max_expansions)
        except _Timeout:
            d = None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        return positions, tidx, target, d
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--per-board", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--boards", default=None,
                   help="id spec override (default: the config's bench range)")
    p.add_argument("--max-expansions", type=int, default=200_000,
                   help="oracle expansion cap per instance")
    p.add_argument("--oracle-timeout", type=float, default=60.0,
                   help="oracle wall-time cap per instance (s)")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    cfg = get(a.config)
    apply_env(cfg)                       # must precede any repo import
    from eval.bench_instances import canonical_body
    from train.encode import walls_for

    boards = parse_ids(a.boards) if a.boards else cfg.ids("bench")
    missing = [i for i in boards if not (cfg.env_dir_abs / f"env_{i}.pkl").exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(boards)} board pkls missing from "
            f"{cfg.env_dir_abs} (first: {missing[:5]}); run "
            f"`python -m scaling.gen_boards --config {cfg.name}` first")

    signal.signal(signal.SIGALRM, _raise_timeout)
    import random
    rng = random.Random(a.seed)
    t0 = time.time()
    instances, n_failed = [], 0
    for bi, env_id in enumerate(boards):
        wr, wd = walls_for(env_id)
        for _ in range(a.per_board):
            inst = sample_and_label(wr, wd, rng, cfg.grid, cfg.robots,
                                    a.max_expansions, a.oracle_timeout)
            if inst is None:
                continue
            positions, tidx, target, d = inst
            if d is None:
                n_failed += 1
            instances.append({
                "env_id": int(env_id),
                "positions": [[int(x), int(y)] for (x, y) in positions],
                "target_idx": int(tidx),
                "target": [int(target[0]), int(target[1])],
                "d_star": None if d is None else int(d),
            })
        if (bi + 1) % 25 == 0:
            print(f"[{bi + 1}/{len(boards)}] instances={len(instances)} "
                  f"oracle_failed={n_failed} ({time.time() - t0:.0f}s)",
                  flush=True)

    out = (Path(a.out) if a.out
           else REPO / "scaling" / "data" / cfg.name / "bench.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_body(instances)
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    out.write_text(body)

    solved = [r for r in instances if r["d_star"] is not None]
    solved_path = out.with_name(f"{out.stem}.solved.jsonl")
    solved_body = canonical_body(solved)
    solved_path.write_text(solved_body)

    meta = {
        "protocol": {
            "sampler": "eval.bench_instances sampling loop (one "
                       "random.Random(seed) consumed sequentially over boards, "
                       "per_board draws per board); oracle failures keep the "
                       "instance with d_star=null instead of resampling",
            "config": cfg.name,
            "env": env(cfg),
            "boards": a.boards or cfg.board_ranges["bench"],
            "per_board": a.per_board,
            "seed": a.seed,
            "d_star": "exact move-optimal cost (move_planner.oracle.solve) or "
                      "null when the oracle exceeded its caps; reference only, "
                      "never used at inference",
            "oracle_caps": {"max_expansions": a.max_expansions,
                            "wall_time_s": a.oracle_timeout},
        },
        "n_instances": len(instances),
        "n_oracle_failed": n_failed,
        "oracle_failure_rate": n_failed / len(instances) if instances else None,
        "sha256": sha,
        "solved_file": str(solved_path),
        "solved_sha256": hashlib.sha256(solved_body.encode("utf-8")).hexdigest(),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path = Path(str(out) + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {len(instances)} instances ({n_failed} with d_star=null) "
          f"over {len(boards)} boards to {out} ({time.time() - t0:.0f}s)")
    print(f"sha256={sha}")
    print(f"solved-only companion ({len(solved)} instances) -> {solved_path}")
    print(f"meta -> {meta_path}")


if __name__ == "__main__":
    main()
