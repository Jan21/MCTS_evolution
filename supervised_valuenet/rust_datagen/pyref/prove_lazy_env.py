"""Prove LazyGridEnv == eager GridEnv, and the pyref table builder == the
reference nn.gen_grids tables (value equality), on real boards.

Checks per board (default: three 16x16 boards from environments/):

1. graph mirror: common.build_graph_n(grid_data) equals the pkl's stored
   grid_graph -- node LIST (order), edge LIST (order) and all attributes.
2. tables: pyref fast all-pairs == nn.gen_grids.all_pairs / independent_paths
   on the same graph, every ordered pair.
3. final components: lazy vs eager, equal as sets AND as iteration-order
   lists for every goal (order leaks into candidate enumeration).
4. propose_subgoal_states: lazy vs eager, ordered list equality (positions,
   colors, scores), for every goal cell and for pinned-support variants over
   every grouped (bottleneck, support) key.
5. heuristics.propose: ordered candidate equality (bottleneck, support,
   helper, parent_support, score) over random segments.
6. exact/relaxed path lengths over a random probe set of (start, end,
   support) triples (default 2000).
7. full nn.generate.rollout record streams identical on both envs for a set
   of random instances (the end-to-end check).

    python rust_datagen/pyref/prove_lazy_env.py                # 16x16 proof
    python rust_datagen/pyref/prove_lazy_env.py --env-dir environments_g24r4 \
        --board-ids 900 --n 24 --rollouts 5                    # 24x24 spot
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

PYREF_DIR = Path(__file__).resolve().parent
SV_DIR = PYREF_DIR.parent.parent
for _p in (str(SV_DIR), str(PYREF_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common  # noqa: E402
from lazy_env import LazyGridEnv  # noqa: E402

FAILS = []


def check(ok, what):
    if not ok:
        FAILS.append(what)
        print(f"  FAIL: {what}")
    return ok


def ser_sub(s):
    return (s.bottleneck.position, s.bottleneck.color,
            s.support.position, s.support.color, s.goal_pos,
            s.target_robot.position, s.target_robot.color,
            s.helper.position, s.helper.color)


def ser_cand(c):
    return (ser_sub(c.subgoal), c.parent_support, c.score)


def prove_board(env_dir, gid, n, probes, segments, rollouts, seed):
    import pickle
    import networkx as nx
    from GridEnv import GridEnv, Robot_at, State
    from nn.gen_grids import all_pairs, independent_paths
    from skeleton import heuristics
    from skeleton.astar import AStar
    from nn.generate import rollout, random_instance

    print(f"== board env_{gid} (n={n}) ==")
    with open(Path(env_dir) / f"env_{gid}.pkl", "rb") as f:
        board = pickle.load(f)
    grid_data = board["grid_data"]

    # 1. graph mirror vs stored pkl graph
    t0 = time.time()
    G = common.build_graph_n(grid_data, n)
    Gp = board["grid_graph"]
    check(list(G.nodes()) == list(Gp.nodes()), "node list/order vs pkl")
    check(list(G.edges(data=True)) == list(Gp.edges(data=True)),
          "edge list/order/attrs vs pkl")
    print(f"  graph mirror ok ({time.time() - t0:.1f}s)")

    # 2. tables: pyref fast builder vs gen_grids reference on the same graph
    t0 = time.time()
    common.reweight_dependent(G, 2)
    ind, allp = common.load_or_build_tables(G, grid_data, n, 2, workers=16,
                                            use_cache=False)
    ref_all = all_pairs(G, "weight")
    ref_ind = independent_paths(G)
    bad = sum(1 for k, v in ref_all.items() if allp[k] != v)
    bad += sum(1 for k, v in ref_ind.items() if ind[k] != v)
    check(bad == 0, f"tables: {bad} value mismatches vs gen_grids")
    print(f"  tables ok: {len(ref_all)} + {len(ref_ind)} pairs "
          f"({time.time() - t0:.1f}s)")

    # eager + lazy envs sharing the same graph object and tables
    t0 = time.time()
    eager = GridEnv(grid_graph=G, independent_paths=ind, all_paths=allp,
                    max_final_component_distance=None)
    t_eager = time.time() - t0
    t0 = time.time()
    lazy = LazyGridEnv(grid_graph=G, independent_paths=ind, all_paths=allp)
    t_lazy = time.time() - t0
    eager.grid_data = lazy.grid_data = list(grid_data)
    print(f"  env init: eager {t_eager:.1f}s, lazy {t_lazy:.2f}s")

    check(eager._wall_nodes == lazy._wall_nodes, "_wall_nodes")
    grouped_keys = list(eager._dependent_edge_cache.keys())
    check(all(k in lazy._dependent_edge_cache for k in grouped_keys)
          and len(grouped_keys) == len(lazy._dependent_edge_cache._entries),
          "dependent-edge cache key sets")
    bad = sum(1 for k in grouped_keys
              if eager._dependent_edge_cache[k]["edges"]
              != lazy._dependent_edge_cache[k]["edges"])
    check(bad == 0, f"dependent-edge groups: {bad} edge-list mismatches")

    # 3. final components (set AND order) for every goal
    t0 = time.time()
    bad_set = bad_order = 0
    for goal in G.nodes():
        fe = eager.precomputed_final_components[goal]
        fl = lazy.precomputed_final_components[goal]
        if fe != fl:
            bad_set += 1
        elif list(fe) != list(fl):
            bad_order += 1
    check(bad_set == 0, f"final components: {bad_set} set mismatches")
    check(bad_order == 0, f"final components: {bad_order} order mismatches")
    bad_fc = 0
    for k in grouped_keys:
        fe = eager._dependent_edge_cache[k]["final_component"]
        fl = lazy._dependent_edge_cache[k]["final_component"]
        if fe != fl or list(fe) != list(fl):
            bad_fc += 1
    check(bad_fc == 0,
          f"extended-graph final components: {bad_fc} mismatches "
          f"({len(grouped_keys)} keys)")
    print(f"  final components ok: {G.number_of_nodes()} goals + "
          f"{len(grouped_keys)} (bn,sup) keys ({time.time() - t0:.1f}s)")

    # 4. propose_subgoal_states: every goal, plus pinned-support variants
    rng = random.Random(seed)
    colors = common.PALETTE[:4]
    t0 = time.time()
    bad = 0
    for goal in G.nodes():
        cells = rng.sample(list(G.nodes()), 4)
        tr = Robot_at(position=cells[0], color=colors[0])
        helpers = [Robot_at(position=cells[i], color=colors[i])
                   for i in range(1, 4)]
        st = State(target=goal, target_robot=tr, helpers=helpers)
        re = [(ser_sub(s), sc) for s, sc in eager.propose_subgoal_states(st)]
        rl = [(ser_sub(s), sc) for s, sc in lazy.propose_subgoal_states(st)]
        if re != rl:
            bad += 1
    check(bad == 0, f"propose_subgoal_states plain: {bad}/{G.number_of_nodes()}")
    badp = 0
    for (bn, sup) in grouped_keys:
        cells = rng.sample(list(G.nodes()), 4)
        tr = Robot_at(position=cells[0], color=colors[0])
        helpers = [Robot_at(position=cells[i], color=colors[i])
                   for i in range(1, 4)]
        st = State(target=bn, target_robot=tr, helpers=helpers)
        pin = Robot_at(position=sup, color=colors[1])
        re = [(ser_sub(s), sc)
              for s, sc in eager.propose_subgoal_states(st, support_robot=pin)]
        rl = [(ser_sub(s), sc)
              for s, sc in lazy.propose_subgoal_states(st, support_robot=pin)]
        if re != rl:
            badp += 1
    check(badp == 0,
          f"propose_subgoal_states pinned: {badp}/{len(grouped_keys)}")
    print(f"  propose_subgoal_states ok ({time.time() - t0:.1f}s)")

    # 5. heuristics.propose over random segments
    t0 = time.time()
    bad = 0
    nodes = list(G.nodes())
    for _ in range(segments):
        cells = rng.sample(nodes, 5)
        goal = cells[0]
        mover = Robot_at(position=cells[1], color=colors[0])
        helpers = [Robot_at(position=cells[i], color=colors[i - 1])
                   for i in range(2, 5)]
        support = None
        if rng.random() < 0.3:
            deps = sorted(heuristics._dependent_supports(eager, goal))
            if deps:
                support = Robot_at(position=rng.choice(deps), color=colors[1])
        ce = [ser_cand(c)
              for c in heuristics.propose(eager, goal, mover, helpers, support)]
        cl = [ser_cand(c)
              for c in heuristics.propose(lazy, goal, mover, helpers, support)]
        if ce != cl:
            bad += 1
    check(bad == 0, f"heuristics.propose: {bad}/{segments} segments")
    print(f"  heuristics.propose ok ({segments} segments, "
          f"{time.time() - t0:.1f}s)")

    # 6. exact/relaxed probes
    t0 = time.time()
    sups = sorted({k[1] for k in grouped_keys})
    bad = 0
    for _ in range(probes):
        s, e = rng.sample(nodes, 2)
        r = rng.random()
        sup = None if r < 0.34 else (
            rng.choice(sups) if r < 0.67 and sups else rng.choice(nodes))
        if (eager.compute_exact_shortest_path_length(s, e, sup)
                != lazy.compute_exact_shortest_path_length(s, e, sup)):
            bad += 1
        if (eager.compute_relaxed_shortest_path_length(s, e, sup)
                != lazy.compute_relaxed_shortest_path_length(s, e, sup)):
            bad += 1
    check(bad == 0, f"exact/relaxed probes: {bad}/{probes * 2}")
    print(f"  exact/relaxed ok ({probes} triples, {time.time() - t0:.1f}s)")

    # 7. full rollouts: identical record streams
    t0 = time.time()
    solver = AStar(max_iters=4000, max_frontier=40_000)
    bad = done = 0
    rng_e = random.Random(seed + 1)
    rng_l = random.Random(seed + 1)
    for k in range(rollouts):
        st_e = random_instance(eager, colors, rng_e)
        st_l = random_instance(lazy, colors, rng_l)
        try:
            re = rollout(eager, st_e, solver, gid, 14)
        except Exception as ex:
            re = f"exc:{ex!r}"
        try:
            rl = rollout(lazy, st_l, solver, gid, 14)
        except Exception as ex:
            rl = f"exc:{ex!r}"
        if re != rl:
            bad += 1
        if isinstance(re, list) and re:
            done += 1
    check(bad == 0, f"rollout record streams: {bad}/{rollouts} differ")
    print(f"  rollouts ok: {rollouts} instances ({done} with records, "
          f"{time.time() - t0:.1f}s)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--env-dir", default=str(SV_DIR / "environments"))
    p.add_argument("--board-ids", default="2400,2401,2402")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--probes", type=int, default=2000)
    p.add_argument("--segments", type=int, default=200)
    p.add_argument("--rollouts", type=int, default=20)
    p.add_argument("--seed", type=int, default=11)
    a = p.parse_args()

    for gid in [int(x) for x in a.board_ids.split(",")]:
        prove_board(a.env_dir, gid, a.n, a.probes, a.segments, a.rollouts,
                    a.seed)
    if FAILS:
        print(f"\nPROOF FAILED: {len(FAILS)} checks")
        sys.exit(1)
    print("\nPROOF OK: lazy == eager on all checks")


if __name__ == "__main__":
    main()
