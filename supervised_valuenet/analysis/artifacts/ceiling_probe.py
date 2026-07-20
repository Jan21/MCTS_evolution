"""Corrected ceiling probe: select failing INSTANCES by row index (bench450 has
3 instances per env_id; result rows align 1:1 with bench450.jsonl by index).

Input JSON: list of instance dicts, each {idx, env_id, d_star, positions, target,
target_idx, tag} where tag in {"npf","fu"}. Categorizes each via exhaustive
hand-coded anytime subgoal search (no NN).

Optional flags after the two positional arguments (defaults reproduce the
original behavior exactly):
    --b1              use the Lever B1 temporary-support vocabulary
                      (skeleton/heuristics.py:propose_b1; analysis/b1_design.md)
    --time-cap S      per-instance wall-clock cap (default 60)
    --max-iters N     per-instance expansion cap (default 200000)
"""
import os, sys, json, time, heapq

os.environ.setdefault("RR_ENV_DIR", "environments")
os.environ.setdefault("RR_GRID", "16")
os.environ.setdefault("RR_ROBOTS", "4")
sys.path.insert(0, ".")
from GridEnv import GridEnv, State, Robot_at
from skeleton.astar import AStar, _initial_plan, park_repairs
from skeleton import heuristics
from move_planner.state import COLOR_ORDER
from eval.realize import strict_moves
from simulate import wall_sets, _board_size

INSTS = json.load(open(sys.argv[1]))
MAX_ITERS, MAX_FRONTIER, TIME_CAP = 200_000, 400_000, 60.0
B1 = "--b1" in sys.argv[3:]
PARK_CAP = 1
argv = sys.argv[3:]
if "--time-cap" in argv:
    TIME_CAP = float(argv[argv.index("--time-cap") + 1])
if "--max-iters" in argv:
    MAX_ITERS = int(argv[argv.index("--max-iters") + 1])
if "--park-cap" in argv:
    PARK_CAP = int(argv[argv.index("--park-cap") + 1])
if "--max-frontier" in argv:
    MAX_FRONTIER = int(argv[argv.index("--max-frontier") + 1])
print(f"[probe] b1={B1} max_iters={MAX_ITERS} time_cap={TIME_CAP} "
      f"park_cap={PARK_CAP} max_frontier={MAX_FRONTIER}", flush=True)

def plan_key(p):
    nodes = tuple(sorted(
        (d.get("ntype"), tuple(d["pos"]) if d.get("pos") is not None else None,
         getattr(d.get("robot"), "color", None)) for _, d in p.g.nodes(data=True)))
    edges = tuple(sorted(
        (p.g.nodes[u].get("ntype"), p.g.nodes[v].get("ntype"), e.get("status"), e.get("cost"))
        for u, v, e in p.g.edges(data=True)))
    return (nodes, edges)

def probe(env, state, solver):
    start = _initial_plan(env, state)
    frontier = [(start.cost(), 0, start)]; tie = 1
    max_open = 2 * (len(state.helpers) + 2)
    seen = set(); t0 = time.time(); it = 0
    abstract_found = False; complete_tested = 0; realizable_moves = None; exhausted = True
    parks_used = 0
    size = _board_size(env.grid_data, None)
    wr, wd = wall_sets(env.grid_data, size)
    while frontier:
        it += 1
        if it > MAX_ITERS or len(frontier) > MAX_FRONTIER or (time.time() - t0) > TIME_CAP:
            exhausted = False; break
        _, _, cur = heapq.heappop(frontier)
        k = plan_key(cur)
        if k in seen: continue
        seen.add(k)
        if cur.is_complete():
            abstract_found = True; complete_tested += 1
            fi = {} if B1 else None
            m = strict_moves(env, state, cur, log=None, fail_info=fi)
            if m is not None:
                realizable_moves = m
                parks_used = sum(1 for _, d in cur.g.nodes(data=True)
                                 if d.get("ntype") == "park")
                break
            # Lever B1 repair step: a complete-but-unplayable plan may spawn
            # park-augmented variants (bounded by --park-cap); they carry
            # their own cost and re-enter the frontier like any other plan.
            if B1 and fi:
                for child in park_repairs(env, state, cur, fi, wr, wd, size,
                                          max_parks=PARK_CAP):
                    heapq.heappush(frontier, (child.cost(), tie, child)); tie += 1
            continue
        if len(cur.open_edges()) > max_open: continue
        for child in solver._expand(env, state, cur):
            heapq.heappush(frontier, (child.cost(), tie, child)); tie += 1
    cat = ("REALIZABLE_EXISTS" if realizable_moves is not None else
           "INCONCLUSIVE" if not exhausted else
           "NO_REALIZABLE_PLAN" if abstract_found else "NO_COMPLETE_PLAN")
    return dict(category=cat, abstract_found=abstract_found, complete_tested=complete_tested,
                realizable_moves=realizable_moves, exhausted=exhausted, iters=it,
                seconds=round(time.time() - t0, 2), parks=parks_used)

solver = AStar(propose=(heuristics.propose_b1 if B1 else heuristics.propose),
               max_iters=MAX_ITERS, max_frontier=MAX_FRONTIER, beam=None)
out = []
for inst in INSTS:
    env, _ = GridEnv.from_env(inst["env_id"])
    positions = [tuple(p) for p in inst["positions"]]
    tidx = inst["target_idx"]
    st = State(target=tuple(inst["target"]),
               target_robot=Robot_at(position=positions[tidx], color=COLOR_ORDER[tidx]),
               helpers=[Robot_at(position=positions[j], color=COLOR_ORDER[j])
                        for j in range(len(positions)) if j != tidx])
    res = probe(env, st, solver)
    res.update(idx=inst["idx"], env_id=inst["env_id"], d_star=inst["d_star"], tag=inst["tag"])
    out.append(res)
    print(f"idx {inst['idx']:3d} env {inst['env_id']:5d} d*={inst['d_star']:2d} [{inst['tag']:3s}] "
          f"{res['category']:18s} abs={res['abstract_found']!s:5} tested={res['complete_tested']:3d} "
          f"real={res['realizable_moves']} iters={res['iters']} t={res['seconds']}s", flush=True)

from collections import Counter
print("\n=== SUMMARY over", len(out), "failing instances ===", flush=True)
print("ALL:", dict(Counter(r["category"] for r in out)), flush=True)
print("npf:", dict(Counter(r["category"] for r in out if r["tag"] == "npf")), flush=True)
print("fu :", dict(Counter(r["category"] for r in out if r["tag"] == "fu")), flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
print("wrote", sys.argv[2], flush=True)
