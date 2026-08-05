"""Parity gate for `nn_labeler.leanboard`: lean board == standard board.

    cd supervised_valuenet && PYTHONPATH=. python -m nn_labeler.test_leanboard_parity
    cd supervised_valuenet && PYTHONPATH=. python -m nn_labeler.test_leanboard_parity --measure

Read-only against the shipped boards; nothing is written outside a temp dir.
Runs on a login CPU in a couple of minutes at 16x16.

WHAT IS PROVEN, per board (3 boards of a 16x16 config by default):

  1. LAYOUT. `leanboard.make_board` reruns the generator's RNG stream
     (`gen_walls` then `random_instance`, `nn/gen_grids.py:123-133`) and must
     reproduce the pickled board exactly: `grid_data`, the slide graph's node
     AND edge insertion order with every edge attribute, and the instance.
     Independently, `leanboard.build_graph(pkl_grid_data, n)` must rebuild the
     pickled `grid_graph` edge-for-edge (this half also covers the stock boards,
     whose generator seed is unknown). Finally the arrays the encode path
     actually reads -- `nn_labeler.encode.adjacency`, A_all and A_ind -- must be
     identical when read from a lean pkl instead of the full one.

  2. DISTANCES, EXHAUSTIVELY. At 16x16 every one of the (n^2)^2 = 65 536 entries
     of BOTH tables is compared against WHAT `GridEnv.from_env` HANDS THE
     PLANNER, not against the raw pkl: `from_env` (GridEnv.py:132-146) reweights
     dependent edges to `dependent_edge_weight` and recomputes `all_paths` at
     that weight, passing `independent_paths` through verbatim. That step is not
     cosmetic -- the 128 stock g16r4 boards ship `grid_graph` dependent edges at
     weight 100 and an `all_paths` built at weight 100 with `inf` (not `None`)
     for unreachable, while `nn.gen_grids` boards ship weight 2 / `None`.
     `_from_env_tables` below replays the normalisation. Forward and backward
     rows are also cross-checked against each other.

  3. API + PLANNER. An eager `GridEnv` built from the pkl's own tables and the
     lean env must agree on every `compute_exact_shortest_path_length` /
     `compute_relaxed_shortest_path_length` query of the shapes descent makes,
     on `heuristics.propose` + `score` candidate lists (order included -- it
     leaks through set iteration), and on a full synthetic descent rollout:
     descent.py's `decision`/`nn_complete` loop with the hand scorer standing in
     for the value net, compared step by step, including `prefix_playable` and
     the final `plan.cost()`.

`--measure` additionally times lean construction and a descent-shaped query
burst at a list of sizes and prints the table.
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# A descent-shaped workload: descent.py's decision/completion loop with the
# hand scorer in place of the value net (deterministic, no checkpoint needed).
# ---------------------------------------------------------------------------

def _descent_machinery(env, state, solver, max_depth, prefix_check):
    """`decision` + greedy `complete`, verbatim from
    `nn_labeler/descent.py::decision` (163) and `nn_complete` (186), with
    `solver.score` standing in for the value net (deterministic, so two envs
    are comparable, and no checkpoint is needed)."""
    from skeleton.astar import _apply, _reference_helpers, _segment
    from eval.realize import prefix_playable

    def decision(plan):
        parent, child = plan.open_edges()[0]
        seg = _segment(plan, state, parent, child)
        if env.compute_exact_shortest_path_length(
                seg.start, seg.end, seg.fix_support) is not None:
            return None
        if solver.by_reference:
            seg.helpers = seg.helpers + _reference_helpers(plan, seg.mover.color)
        cands = solver.propose(env, seg.end, seg.mover, seg.helpers, seg.support)
        cands = sorted(cands, key=lambda c: solver.score(env, c))[:14]
        applied = []
        for cand in cands:
            cp = _apply(env, plan, parent, child, seg, cand,
                        by_reference=solver.by_reference)
            if cp is not None:
                applied.append((cand, cp))
        return seg, applied

    def complete(plan):
        for _ in range(max_depth):
            if plan.is_complete():
                return plan
            dec = decision(plan)
            if dec is None:
                exp = solver._expand(env, state, plan)
                if not exp:
                    return None
                plan = exp[0]
                continue
            _seg, applied = dec
            if not applied:
                return None
            for _cand, cp in applied:      # already score-sorted
                if prefix_check and not prefix_playable(env, state, cp, log=None):
                    continue
                plan = cp
                break
            else:
                return None
        return plan if plan.is_complete() else None

    return decision, complete


def greedy_completion(env, state, solver, max_depth=32, prefix_check=True):
    """descent's INNER loop only (`nn_complete` on the initial plan).

    The cheap, representative query burst: one greedy net-guided completion,
    which is what descent runs once per candidate at every decision.
    """
    from skeleton.astar import _initial_plan
    _dec, complete = _descent_machinery(env, state, solver, max_depth, prefix_check)
    done = complete(_initial_plan(env, state))
    return None if done is None else int(done.cost())


def descent_like_rollout(env, state, solver, max_depth=64, prefix_check=True):
    """Mirror of `nn_labeler/descent.py::nn_rollout` (216): price every
    candidate at every decision by greedy completion, advance on the best."""
    from skeleton.astar import _initial_plan
    decision, complete = _descent_machinery(env, state, solver, max_depth,
                                            prefix_check)
    trace = []
    plan = _initial_plan(env, state)
    depth = 0
    while not plan.is_complete() and depth < max_depth:
        dec = decision(plan)
        if dec is None:
            exp = solver._expand(env, state, plan)
            if not exp:
                break
            plan = exp[0]
            continue
        _seg, applied = dec
        priced = []
        for cand, cp in applied:
            done = complete(cp)
            if done is None:
                priced.append((cand, cp, None))
            else:
                priced.append((cand, cp, int(done.cost())))
        trace.append([(_xy(c.subgoal.bottleneck.position),
                       _xy(c.subgoal.support.position),
                       c.subgoal.helper.color, ctg) for c, _cp, ctg in priced])
        best = [(cp, ctg) for _c, cp, ctg in priced if ctg is not None]
        if not best:
            break
        bctg = min(c for _p, c in best)
        plan = next(p for p, c in best if c == bctg)
        depth += 1
    return trace, (int(plan.cost()) if plan.is_complete() else None)


def _xy(p):
    return None if p is None else (int(p[0]), int(p[1]))


# ---------------------------------------------------------------------------
# Section 1: board layout
# ---------------------------------------------------------------------------

def from_env_tables(pkl, dependent_edge_weight=2):
    """The two tables `GridEnv.from_env` actually constructs (GridEnv.py:132-146).

    Reweights the pickled graph's dependent edges IN PLACE (as `from_env` does),
    then recomputes `all_paths` at that weight with `None` for unreachable.
    `independent_paths` is passed through unchanged, exactly as `from_env` does.
    """
    import networkx as nx
    g = pkl["grid_graph"]
    for _u, _v, d in g.edges(data=True):
        if "dependent" in d:
            d["weight"] = dependent_edge_weight
    all_paths = {}
    nodes = sorted(g.nodes())
    lengths = dict(nx.all_pairs_dijkstra_path_length(g, weight="weight"))
    for src in nodes:
        sl = lengths.get(src, {})
        for dst in nodes:
            all_paths[(src, dst)] = sl.get(dst, None)
    return pkl.get("independent_paths", {}), all_paths


def check_layout(cfg, gid, pkl, n, robots, seed, tmpdir):
    import numpy as np
    import networkx as nx
    from nn_labeler import encode, leanboard

    fails = []

    # (a) full RNG replay: same seed -> byte-identical board
    lean = leanboard.make_board(gid, n, walls=cfg.walls, robots=robots, seed=seed)
    rng_ok = lean["grid_data"] == pkl["grid_data"]
    if rng_ok:
        _cmp_graphs(lean["grid_graph"], pkl["grid_graph"], "rng-replay graph", fails)
        li, pi = lean["instances"][0], pkl["instances"][0]
        if (li["target"] != pi["target"]
                or li["target_robot"] != pi["target_robot"]
                or li["helper_robots"] != pi["helper_robots"]):
            fails.append(f"env_{gid}: replayed instance differs")

    # (b) graph rebuild from the stored walls (works for any board).
    # Edge insertion order is only required to match for boards `nn.gen_grids`
    # actually wrote: the 128 stock g16r4 graphs come from older code that
    # emitted the same edges in another order. That is harmless here because
    # `leanboard.from_env` reuses the pickled `grid_graph` object whenever the
    # board has one -- `build_graph` runs only for boards it generates itself.
    rebuilt = leanboard.build_graph(pkl["grid_data"], n)
    _cmp_graphs(rebuilt, pkl["grid_graph"], f"env_{gid} rebuilt graph", fails,
                require_order=rng_ok)

    # (c) what the encode path reads, through a lean pkl
    lean_dir = Path(tmpdir) / "lean"
    lean_dir.mkdir(parents=True, exist_ok=True)
    with open(lean_dir / f"env_{gid}.pkl", "wb") as f:
        pickle.dump({"graph_idx": gid, "grid_data": pkl["grid_data"],
                     "grid_graph": rebuilt, "instances": pkl["instances"]}, f)
    a_ref = encode.adjacency(cfg.env_dir_abs, gid, n)
    a_lean = encode.adjacency(lean_dir, gid, n)
    for name, x, y in (("A_all", a_ref[0], a_lean[0]), ("A_ind", a_ref[1], a_lean[1])):
        if not np.array_equal(x, y):
            fails.append(f"env_{gid}: encode.adjacency {name} differs "
                         f"({int(np.abs(x - y).sum())} cells)")
    return rng_ok, fails


def _cmp_graphs(a, b, what, fails, require_order=True):
    """Same nodes and same attributed edges; insertion order optional."""
    if set(a.nodes()) != set(b.nodes()):
        fails.append(f"{what}: node SET differs")
        return
    da = {(u, v): tuple(sorted(d.items(), key=repr)) for u, v, d in a.edges(data=True)}
    db = {(u, v): tuple(sorted(d.items(), key=repr)) for u, v, d in b.edges(data=True)}
    if da != db:
        only_a = set(da) - set(db)
        only_b = set(db) - set(da)
        diff = [k for k in set(da) & set(db) if da[k] != db[k]]
        fails.append(f"{what}: edge sets/attrs differ "
                     f"({len(only_a)} extra, {len(only_b)} missing, "
                     f"{len(diff)} attr mismatches; e.g. "
                     f"{(sorted(only_a) or sorted(only_b) or sorted(diff))[:1]})")
        return
    if not require_order:
        return
    if list(a.nodes()) != list(b.nodes()):
        fails.append(f"{what}: node insertion order differs")
        return
    ea = [(u, v) for u, v in a.edges()]
    eb = [(u, v) for u, v in b.edges()]
    for i, (x, y) in enumerate(zip(ea, eb)):
        if x != y:
            fails.append(f"{what}: edge insertion order differs at {i}: {x} vs {y}")
            return


# ---------------------------------------------------------------------------
# Section 2: exhaustive distance parity
# ---------------------------------------------------------------------------

def check_tables_exhaustive(pkl, n, gid, tables):
    """Every (src, dst) of both tables, lean oracle vs what from_env produces."""
    import numpy as np
    from nn_labeler import leanboard

    ref_ind, ref_all = tables
    G = leanboard.build_graph(pkl["grid_data"], n)
    for _u, _v, d in G.edges(data=True):          # GridEnv.from_env:135-137
        if "dependent" in d:
            d["weight"] = leanboard.DEFAULT_DEP_WEIGHT
    edges = leanboard.edge_arrays(G, n)
    cells = [(x, y) for y in range(n) for x in range(n)]
    fails, checked = [], 0

    for tname, independent, ref in (("independent_paths", True, ref_ind),
                                    ("all_paths", False, ref_all)):
        tbl = leanboard.LazyDistTable(G, n, independent=independent,
                                      max_rows=4 * n * n, edges=edges)
        for s in cells:
            fwd = tbl.row_from(s)
            for t in cells:
                got = None if fwd[t[1] * n + t[0]] < 0 else int(fwd[t[1] * n + t[0]])
                want = ref[(s, t)]
                checked += 1
                if got != want and len(fails) < 10:
                    fails.append(f"env_{gid} {tname}[{s},{t}]: lean {got} != pkl {want}")
        # backward rows must agree with forward rows entry for entry
        for t in cells[::7]:
            bwd = tbl.row_to(t)
            col = np.array([tbl.row_from(s)[t[1] * n + t[0]] for s in cells])
            if not np.array_equal(bwd, col) and len(fails) < 10:
                fails.append(f"env_{gid} {tname}: backward row to {t} != forward column")
    return checked, fails


# ---------------------------------------------------------------------------
# Section 3: API + planner parity
# ---------------------------------------------------------------------------

def _eager_env(pkl, n, tables):
    """The env `GridEnv.from_env` would return, built in-process.

    Same construction as `from_env` (GridEnv.py:148-157) on the normalised
    tables; done without calling `from_env` only so the test writes nothing
    into the shared boards' `cache/` dir.
    """
    from GridEnv import GridEnv, State
    ind, allp = tables
    env = GridEnv(grid_graph=pkl["grid_graph"], independent_paths=ind,
                  all_paths=allp, max_final_component_distance=None)
    env.grid_data = pkl["grid_data"]
    inst = pkl["instances"][0]
    return env, State(target=inst["target"], target_robot=inst["target_robot"],
                      helpers=inst["helper_robots"])


def check_api(pkl, n, gid, n_queries, n_states, rng, tables):
    from nn_labeler import leanboard
    from nn.generate import make_solver, random_instance
    from skeleton import heuristics

    eager, _s0 = _eager_env(pkl, n, tables)
    lean = leanboard.build_env(pkl["grid_data"], n)
    fails = []

    # -- point queries in descent's shapes ---------------------------------
    cells = list(eager.G.nodes())
    supports = {}
    for u, v, d in eager.G.edges(data=True):
        if "dependent" in d:
            supports.setdefault(v, []).append(d["dependent"])
    nq = 0
    for _ in range(n_queries):
        s = rng.choice(cells)
        t = rng.choice(cells)
        sup = None
        if rng.random() < 0.5 and supports.get(t):
            sup = rng.choice(supports[t])
        a = eager.compute_exact_shortest_path_length(s, t, sup)
        b = lean.compute_exact_shortest_path_length(s, t, sup)
        c = eager.compute_relaxed_shortest_path_length(s, t, sup)
        e = lean.compute_relaxed_shortest_path_length(s, t, sup)
        nq += 2
        if a != b and len(fails) < 10:
            fails.append(f"env_{gid} exact({s},{t},{sup}): eager {a} != lean {b}")
        if c != e and len(fails) < 10:
            fails.append(f"env_{gid} relaxed({s},{t},{sup}): eager {c} != lean {e}")

    # -- propose/score candidate lists, order included ---------------------
    colors = leanboard.PALETTE[:1 + len(pkl["instances"][0]["helper_robots"])]
    states = [random_instance(eager, colors, rng) for _ in range(n_states)]
    for st in states:
        ca = heuristics.propose(eager, st.target, st.target_robot, st.helpers, None)
        cb = heuristics.propose(lean, st.target, st.target_robot, st.helpers, None)
        ka = [(_xy(c.subgoal.bottleneck.position), _xy(c.subgoal.support.position),
               c.subgoal.helper.color, _xy(c.parent_support),
               heuristics.score(eager, c)) for c in ca]
        kb = [(_xy(c.subgoal.bottleneck.position), _xy(c.subgoal.support.position),
               c.subgoal.helper.color, _xy(c.parent_support),
               heuristics.score(lean, c)) for c in cb]
        if ka != kb and len(fails) < 12:
            fails.append(f"env_{gid} propose/score differs at {st.target}: "
                         f"{len(ka)} vs {len(kb)} candidates")

    # -- full descent-shaped rollouts --------------------------------------
    solver = make_solver("base")
    n_roll = 0
    for st in states[:max(1, n_states // 2)]:
        ta, ca_ = descent_like_rollout(eager, st, solver)
        tb, cb_ = descent_like_rollout(lean, st, solver)
        n_roll += 1
        if ta != tb or ca_ != cb_:
            fails.append(f"env_{gid} descent rollout differs at target {st.target} "
                         f"(cost {ca_} vs {cb_}, {len(ta)} vs {len(tb)} decisions)")
    return nq, len(states), n_roll, lean, fails


# ---------------------------------------------------------------------------
# --measure
# ---------------------------------------------------------------------------

def measure(sizes, robots, instances, seed, max_rows, budget=90.0):
    """Time lean construction and a descent-shaped query burst per size.

    Only `gen_walls` + `build_graph` + the CSR build run per board -- never
    `nn.gen_grids.make_board`, whose all-pairs precompute is the wall this
    module exists to remove. The burst is `instances` greedy completions
    (descent's inner loop), abandoned after `budget` seconds so a big size
    still reports its per-completion rate.
    """
    from nn_labeler import leanboard
    from nn.generate import make_solver, random_instance

    rows = []
    solver = make_solver("base")
    hdr = (f"{'n':>4} {'edges':>8} {'board_s':>8} {'env_s':>7} {'lean_MB':>8} "
           f"{'compl':>6} {'burst_s':>8} {'s/compl':>8} {'queries':>9} "
           f"{'rows':>6} {'row_ms':>7}")
    print(hdr)
    print("-" * len(hdr))
    for n in sizes:
        walls = leanboard.default_walls(n)
        t0 = time.time()
        board = leanboard.make_board(0, n, walls=walls, robots=robots, seed=seed)
        t_board = time.time() - t0
        blob = pickle.dumps(board)
        t0 = time.time()
        env = leanboard.build_env(board["grid_data"], n,
                                  grid_graph=board["grid_graph"], max_rows=max_rows)
        t_env = time.time() - t0
        ind, allp = env.lean_tables

        # cost of one cold single-source row, both tables
        t0 = time.time()
        ind.row_from((0, 0)); allp.row_from((0, 0))
        row_ms = (time.time() - t0) * 500.0

        colors = leanboard.PALETTE[:robots]
        rng = random.Random(seed)
        t0 = time.time()
        done = 0
        for _ in range(instances):
            st = random_instance(env, colors, rng)
            greedy_completion(env, st, solver, max_depth=32)
            done += 1
            if time.time() - t0 > budget:
                break
        t_q = time.time() - t0
        q = ind.queries + allp.queries
        r = ind.rows_built + allp.rows_built
        rows.append({"n": n, "edges": env.G.number_of_edges(), "t_board": t_board,
                     "t_env": t_env, "t_burst": t_q, "completions": done,
                     "lean_mb": len(blob) / 1e6, "queries": q, "rows": r,
                     "row_ms": row_ms})
        print(f"{n:>4} {env.G.number_of_edges():>8} {t_board:>8.2f} {t_env:>7.2f} "
              f"{len(blob)/1e6:>8.2f} {done:>6} {t_q:>8.2f} "
              f"{t_q/max(done,1):>8.2f} {q:>9} {r:>6} {row_ms:>7.1f}", flush=True)
    return rows


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="g16r6",
                   help="a config whose environments_* pkls exist (16x16)")
    p.add_argument("--boards", default=None, help="board ids, e.g. 0,1,2")
    p.add_argument("--n-boards", type=int, default=3)
    p.add_argument("--queries", type=int, default=400)
    p.add_argument("--states", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--board-seed", type=int, default=0,
                   help="the --seed nn.gen_grids.main was run with")
    p.add_argument("--measure", action="store_true")
    p.add_argument("--measure-sizes", default="16,24,32,40,48,56,64,80,96")
    p.add_argument("--measure-instances", type=int, default=3)
    p.add_argument("--measure-robots", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=1024)
    a = p.parse_args()

    from scaling import configs
    cfg = configs.get(a.config)
    configs.apply_env(cfg)                     # before any repo import

    n, robots = cfg.grid, cfg.robots
    ids = (configs.parse_ids(a.boards) if a.boards
           else cfg.ids("train")[: a.n_boards])
    rng = random.Random(a.seed)
    t_start = time.time()
    all_fails = []

    print(f"config {cfg.name}: n={n} robots={robots} walls={cfg.walls} "
          f"boards {ids} from {cfg.env_dir_abs}")

    with tempfile.TemporaryDirectory(prefix="leanparity_") as tmp:
        for gid in ids:
            t0 = time.time()
            with open(cfg.env_dir_abs / f"env_{gid}.pkl", "rb") as f:
                pkl = pickle.load(f)
            t_load = time.time() - t0
            # from_env's normalisation, applied before anything is compared:
            # it reweights the pickled graph in place and rebuilds all_paths.
            t0 = time.time()
            tables = from_env_tables(pkl)
            t_norm = time.time() - t0

            rng_ok, f1 = check_layout(cfg, gid, pkl, n, robots, a.board_seed, tmp)
            all_fails += f1
            print(f"env_{gid}: pkl load {t_load:.1f}s, from_env normalisation "
                  f"{t_norm:.1f}s | layout: rng-replay="
                  f"{'MATCH' if rng_ok else 'n/a (not a gen_grids board)'}, "
                  f"graph rebuild + encode.adjacency "
                  f"{'OK' if not f1 else 'FAIL'}")

            t0 = time.time()
            checked, f2 = check_tables_exhaustive(pkl, n, gid, tables)
            all_fails += f2
            print(f"env_{gid}: distances: {checked} table entries "
                  f"(both tables, exhaustive) {'OK' if not f2 else 'FAIL'} "
                  f"in {time.time() - t0:.1f}s")

            t0 = time.time()
            nq, ns, nr, lean, f3 = check_api(pkl, n, gid, a.queries, a.states,
                                             rng, tables)
            all_fails += f3
            ind, allp = lean.lean_tables
            print(f"env_{gid}: api: {nq} point queries, {ns} propose/score states, "
                  f"{nr} descent rollouts {'OK' if not f3 else 'FAIL'} "
                  f"in {time.time() - t0:.1f}s "
                  f"[lean rows built: ind {ind.rows_built}/{ind.queries} q, "
                  f"all {allp.rows_built}/{allp.queries} q]")

    if all_fails:
        print(f"\nFAILURES ({len(all_fails)}):")
        for f in all_fails:
            print("  " + f)
        print("PARITY FAILED")
        return 1

    print(f"\nPARITY OK ({time.time() - t_start:.0f}s): lean boards are "
          f"byte-identical and the lazy oracle is value-identical.")

    if a.measure:
        sizes = [int(s) for s in a.measure_sizes.split(",")]
        print(f"\n-- lean construction + descent-shaped query burst "
              f"({a.measure_instances} rollouts, {a.measure_robots} robots) --")
        measure(sizes, a.measure_robots, a.measure_instances, a.seed, a.max_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
