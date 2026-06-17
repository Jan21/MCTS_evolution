"""Demonstrates the validate_plan module.

Builds several partial plans — one valid, the rest intentionally broken —
and shows what the validator catches.

Usage:
    python validate_example.py
"""

from math import isqrt

import networkx as nx

from GridEnv import GridEnv, Robot_at, State
from partial_plan import PartialPlan
from validate_plan import validate

N_ENVIRONMENTS = 10

# ── Helper to run and print a test ────────────────────────────────────────

def run_test(label, plan, grid_env, state):
    result = validate(plan, grid_env, state)
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {label}")
    for err in result.errors:
        print(f"       {err}")
    print()

# ══════════════════════════════════════════════════════════════════════════
# PART 1 — Verify GridEnv.from_env across N environments
# ══════════════════════════════════════════════════════════════════════════

print(f"=== Environment loading ({N_ENVIRONMENTS} environments) ===")
print()

envs: list[tuple[GridEnv, State]] = []
for i in range(N_ENVIRONMENTS):
    grid_env, state = GridEnv.from_env(i)
    envs.append((grid_env, state))

    errors: list[str] = []

    if not isinstance(grid_env.G, nx.DiGraph):
        errors.append("grid_graph is not a nx.DiGraph")
    grid_nodes = set(grid_env.G.nodes())
    # Infer the board side length from the grid: a full NxN board has N*N cells.
    grid_size = isqrt(len(grid_nodes))
    if grid_size * grid_size != len(grid_nodes):
        errors.append(f"grid_nodes has {len(grid_nodes)} nodes, "
                      f"which is not a perfect square (NxN) board")

    for pos in grid_nodes:
        if (not isinstance(pos, tuple) or len(pos) != 2
                or not (0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size)):
            errors.append(f"invalid grid node {pos}")
            break

    if not isinstance(state, State):
        errors.append("state is not a State instance")
    if not isinstance(state.target, tuple) or state.target not in grid_nodes:
        errors.append(f"target {state.target} not on the grid")
    if not isinstance(state.target_robot, Robot_at):
        errors.append("target_robot is not a Robot_at")
    for h in state.helpers:
        if not isinstance(h, Robot_at):
            errors.append(f"helper {h} is not a Robot_at")
            break

    for r in state.all_robots:
        if r.position not in grid_nodes:
            errors.append(f"robot {r.color} at {r.position} is off the grid")

    if not grid_env.reachability_matrix:
        errors.append("reachability_matrix is empty")
    if not grid_env.relaxed_reachability_matrix:
        errors.append("relaxed_reachability_matrix is empty")

    status_str = "PASS" if not errors else "FAIL"
    detail = "" if not errors else "  " + "; ".join(errors)
    print(f"  [{status_str}] env_{i}: target={state.target}, "
          f"target_robot={state.target_robot.color}, "
          f"helpers={[h.color for h in state.helpers]}{detail}")

print()

# ══════════════════════════════════════════════════════════════════════════
# PART 2 — Validate partial plans (using env_0)
# ══════════════════════════════════════════════════════════════════════════

print("=== Plan validation (env_0) ===")
print()

grid_env, s = envs[0]

target_robot = s.target_robot
red, blue, green = s.helpers


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1 — Valid complete plan
# ═══════════════════════════════════════════════════════════════════════════

p = PartialPlan()
p.add_node("goal",   "goal",       pos=(6, 15))
p.add_node("sg1",    "subgoal")
p.add_node("bn1",    "bottleneck", pos=(5, 15),              robot=target_robot)
p.add_node("sp1",    "support",    pos=(6, 14),              robot=red)
p.add_node("leaf_y", "leaf",       pos=target_robot.position, robot=target_robot)
p.add_node("leaf_r", "leaf",       pos=red.position,          robot=red)

p.add_edge("goal", "sg1",    status="fixed", cost=1)
p.add_edge("sg1",  "bn1",    status="fixed", cost=None)
p.add_edge("sg1",  "sp1",    status="fixed", cost=None)
p.add_edge("bn1",  "leaf_y", status="fixed", cost=3)
p.add_edge("sp1",  "leaf_r", status="fixed", cost=4)

run_test("Valid complete plan", p, grid_env, s)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2 — Cycle in the graph
# ═══════════════════════════════════════════════════════════════════════════

p2 = PartialPlan()
p2.add_node("goal", "goal",       pos=(6, 15))
p2.add_node("sg",   "subgoal")
p2.add_node("bn",   "bottleneck", pos=(5, 15), robot=target_robot)
p2.add_node("sp",   "support",    pos=(6, 14), robot=red)
p2.add_edge("goal", "sg",  status="fixed", cost=1)
p2.add_edge("sg",   "bn",  status="fixed", cost=None)
p2.add_edge("sg",   "sp",  status="fixed", cost=None)
p2.add_edge("bn",   "goal", status="fixed", cost=1)

run_test("Cycle in the graph", p2, grid_env, s)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3 — Wrong goal position
# ═══════════════════════════════════════════════════════════════════════════

p3 = PartialPlan()
p3.add_node("goal",   "goal",  pos=(0, 0))
p3.add_node("leaf_y", "leaf",  pos=target_robot.position, robot=target_robot)
p3.add_edge("goal", "leaf_y",  status="fixed", cost=1)

run_test("Wrong goal position", p3, grid_env, s)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4 — Bottleneck robot does not match target robot
# ═══════════════════════════════════════════════════════════════════════════

p4 = PartialPlan()
p4.add_node("goal", "goal",       pos=(6, 15))
p4.add_node("sg",   "subgoal")
p4.add_node("bn",   "bottleneck", pos=(5, 15), robot=red)
p4.add_node("sp",   "support",    pos=(6, 14), robot=green)
p4.add_edge("goal", "sg", status="fixed", cost=1)
p4.add_edge("sg",   "bn", status="fixed", cost=None)
p4.add_edge("sg",   "sp", status="fixed", cost=None)

run_test("Bottleneck robot != target robot", p4, grid_env, s)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5 — Subgoal missing its support child
# ═══════════════════════════════════════════════════════════════════════════

p5 = PartialPlan()
p5.add_node("goal", "goal",       pos=(6, 15))
p5.add_node("sg",   "subgoal")
p5.add_node("bn",   "bottleneck", pos=(5, 15), robot=target_robot)
p5.add_edge("goal", "sg", status="fixed", cost=1)
p5.add_edge("sg",   "bn", status="fixed", cost=None)

run_test("Subgoal missing support child", p5, grid_env, s)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6 — Leaf position doesn't match robot's actual position
# ═══════════════════════════════════════════════════════════════════════════

p6 = PartialPlan()
p6.add_node("goal",   "goal",  pos=(6, 15))
p6.add_node("leaf_y", "leaf",  pos=(0, 0), robot=target_robot)
p6.add_edge("goal", "leaf_y",  status="fixed", cost=1)

run_test("Leaf pos doesn't match robot's actual position", p6, grid_env, s)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7 — Invalid edge type (leaf → leaf)
# ═══════════════════════════════════════════════════════════════════════════

p7 = PartialPlan()
p7.add_node("goal",   "goal", pos=(6, 15))
p7.add_node("leaf_y", "leaf", pos=target_robot.position, robot=target_robot)
p7.add_node("leaf_r", "leaf", pos=red.position,          robot=red)
p7.add_edge("goal",   "leaf_y", status="fixed", cost=1)
p7.add_edge("leaf_y", "leaf_r", status="fixed", cost=1)

run_test("Invalid edge type (leaf -> leaf)", p7, grid_env, s)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 8 — Open edges (incomplete plan)
# ═══════════════════════════════════════════════════════════════════════════

p8 = PartialPlan()
p8.add_node("goal",   "goal",       pos=(6, 15))
p8.add_node("sg",     "subgoal")
p8.add_node("bn",     "bottleneck", pos=(5, 15),              robot=target_robot)
p8.add_node("sp",     "support",    pos=(6, 14),              robot=red)
p8.add_node("leaf_y", "leaf",       pos=target_robot.position, robot=target_robot)
p8.add_node("leaf_r", "leaf",       pos=red.position,          robot=red)

p8.add_edge("goal", "sg",      status="fixed", cost=1)
p8.add_edge("sg",   "bn",      status="fixed", cost=None)
p8.add_edge("sg",   "sp",      status="fixed", cost=None)
p8.add_edge("bn",   "leaf_y",  status="open",  cost=3)
p8.add_edge("sp",   "leaf_r",  status="open",  cost=4)

run_test("Open edges (incomplete plan)", p8, grid_env, s)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 9 — Position off the grid
# ═══════════════════════════════════════════════════════════════════════════

p9 = PartialPlan()
p9.add_node("goal", "goal", pos=(20, 20))
p9.add_edge("goal", "goal")

run_test("Position off the grid", p9, grid_env, s)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 10 — String instead of Robot_at instance
# ═══════════════════════════════════════════════════════════════════════════

p10 = PartialPlan()
p10.add_node("goal",   "goal",  pos=(6, 15))
p10.add_node("leaf_y", "leaf",  pos=target_robot.position, robot="Yellow")
p10.add_edge("goal", "leaf_y",  status="fixed", cost=1)

run_test("String instead of Robot_at instance", p10, grid_env, s)
