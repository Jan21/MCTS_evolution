"""Demonstrates the validate_plan module.

Builds several partial plans — one valid, the rest intentionally broken —
and shows what the validator catches.

Usage:
    python validate_example.py
"""

import networkx as nx

from game import Game, State
from partial_plan import PartialPlan
from robot import Robot
from validate_plan import validate

GRID_SIZE = 16
N_ENVIRONMENTS = 10

# ── Helper to run and print a test ────────────────────────────────────────

def run_test(label, plan, game):
    result = validate(plan, game)
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {label}")
    for err in result.errors:
        print(f"       {err}")
    print()

# ══════════════════════════════════════════════════════════════════════════
# PART 1 — Verify Game.from_env across N environments
# ══════════════════════════════════════════════════════════════════════════

print(f"=== Game loading ({N_ENVIRONMENTS} environments) ===")
print()

games: list[Game] = []
for i in range(N_ENVIRONMENTS):
    g = Game.from_env(i)
    games.append(g)

    errors: list[str] = []

    # grid_graph is a DiGraph with 256 nodes
    if not isinstance(g.grid_graph, nx.DiGraph):
        errors.append("grid_graph is not a nx.DiGraph")
    if len(g.grid_nodes) != GRID_SIZE * GRID_SIZE:
        errors.append(f"grid_nodes has {len(g.grid_nodes)} nodes, expected {GRID_SIZE**2}")

    # All grid positions are integer tuples in [0, 15]
    for pos in g.grid_nodes:
        if (not isinstance(pos, tuple) or len(pos) != 2
                or not (0 <= pos[0] < GRID_SIZE and 0 <= pos[1] < GRID_SIZE)):
            errors.append(f"invalid grid node {pos}")
            break

    # State is well-formed
    s = g.state
    if not isinstance(s, State):
        errors.append("state is not a State instance")
    if not isinstance(s.target, tuple) or s.target not in g.grid_nodes:
        errors.append(f"target {s.target} not on the grid")
    if not isinstance(s.target_robot, Robot):
        errors.append("target_robot is not a Robot")
    for h in s.helpers:
        if not isinstance(h, Robot):
            errors.append(f"helper {h} is not a Robot")
            break

    # Every robot position is on the grid
    for r in s.all_robots:
        if (r.x, r.y) not in g.grid_nodes:
            errors.append(f"robot {r.name} at ({r.x},{r.y}) is off the grid")

    # Precomputed path tables are present
    if not g.independent_paths:
        errors.append("independent_paths is empty")
    if not g.all_paths:
        errors.append("all_paths is empty")

    status = "PASS" if not errors else "FAIL"
    detail = "" if not errors else "  " + "; ".join(errors)
    print(f"  [{status}] env_{i}: target={s.target}, "
          f"target_robot={s.target_robot.name}, "
          f"helpers={[h.name for h in s.helpers]}{detail}")

print()

# ══════════════════════════════════════════════════════════════════════════
# PART 2 — Validate partial plans (using env_0)
# ══════════════════════════════════════════════════════════════════════════

print("=== Plan validation (env_0) ===")
print()

game = games[0]
s = game.state

target_robot = s.target_robot
red, blue, green = s.helpers


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1 — Valid complete plan
# ═══════════════════════════════════════════════════════════════════════════
#
#   goal(6,15) ──fixed,1──▶ sg1 ──struct──▶ bn1(5,15, Yellow) ──fixed,3──▶ leaf_y(12,9)
#                                ──struct──▶ sp1(6,14, Red)    ──fixed,4──▶ leaf_r(11,10)

p = PartialPlan()
p.add_node("goal",   "goal",       pos=(6, 15))
p.add_node("sg1",    "subgoal")
p.add_node("bn1",    "bottleneck", pos=(5, 15),  robot=target_robot)
p.add_node("sp1",    "support",    pos=(6, 14),  robot=red)
p.add_node("leaf_y", "leaf",       pos=(12, 9),  robot=target_robot)
p.add_node("leaf_r", "leaf",       pos=(11, 10), robot=red)

p.add_edge("goal", "sg1",    status="fixed", cost=1)
p.add_edge("sg1",  "bn1",    status="fixed", cost=None)  # structural
p.add_edge("sg1",  "sp1",    status="fixed", cost=None)  # structural
p.add_edge("bn1",  "leaf_y", status="fixed", cost=3)
p.add_edge("sp1",  "leaf_r", status="fixed", cost=4)

run_test("Valid complete plan", p, game)


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
p2.add_edge("bn",   "goal", status="fixed", cost=1)  # creates cycle

run_test("Cycle in the graph", p2, game)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3 — Wrong goal position
# ═══════════════════════════════════════════════════════════════════════════

p3 = PartialPlan()
p3.add_node("goal",   "goal",  pos=(0, 0))  # wrong — should be (6,15)
p3.add_node("leaf_y", "leaf",  pos=(12, 9), robot=target_robot)
p3.add_edge("goal", "leaf_y",  status="fixed", cost=1)

run_test("Wrong goal position", p3, game)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4 — Bottleneck robot does not match target robot
# ═══════════════════════════════════════════════════════════════════════════

p4 = PartialPlan()
p4.add_node("goal", "goal",       pos=(6, 15))
p4.add_node("sg",   "subgoal")
p4.add_node("bn",   "bottleneck", pos=(5, 15), robot=red)    # should be Yellow
p4.add_node("sp",   "support",    pos=(6, 14), robot=green)
p4.add_edge("goal", "sg", status="fixed", cost=1)
p4.add_edge("sg",   "bn", status="fixed", cost=None)
p4.add_edge("sg",   "sp", status="fixed", cost=None)

run_test("Bottleneck robot != target robot", p4, game)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5 — Subgoal missing its support child
# ═══════════════════════════════════════════════════════════════════════════

p5 = PartialPlan()
p5.add_node("goal", "goal",       pos=(6, 15))
p5.add_node("sg",   "subgoal")
p5.add_node("bn",   "bottleneck", pos=(5, 15), robot=target_robot)
p5.add_edge("goal", "sg", status="fixed", cost=1)
p5.add_edge("sg",   "bn", status="fixed", cost=None)

run_test("Subgoal missing support child", p5, game)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6 — Leaf position doesn't match robot's actual position
# ═══════════════════════════════════════════════════════════════════════════

p6 = PartialPlan()
p6.add_node("goal",   "goal",  pos=(6, 15))
p6.add_node("leaf_y", "leaf",  pos=(0, 0), robot=target_robot)  # Yellow is at (12,9)
p6.add_edge("goal", "leaf_y",  status="fixed", cost=1)

run_test("Leaf pos doesn't match robot's actual position", p6, game)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7 — Invalid edge type (leaf → leaf)
# ═══════════════════════════════════════════════════════════════════════════

p7 = PartialPlan()
p7.add_node("goal",   "goal", pos=(6, 15))
p7.add_node("leaf_y", "leaf", pos=(12, 9),  robot=target_robot)
p7.add_node("leaf_r", "leaf", pos=(11, 10), robot=red)
p7.add_edge("goal",   "leaf_y", status="fixed", cost=1)
p7.add_edge("leaf_y", "leaf_r", status="fixed", cost=1)  # invalid

run_test("Invalid edge type (leaf -> leaf)", p7, game)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 8 — Open edges (incomplete plan)
# ═══════════════════════════════════════════════════════════════════════════

p8 = PartialPlan()
p8.add_node("goal",   "goal",       pos=(6, 15))
p8.add_node("sg",     "subgoal")
p8.add_node("bn",     "bottleneck", pos=(5, 15),  robot=target_robot)
p8.add_node("sp",     "support",    pos=(6, 14),  robot=red)
p8.add_node("leaf_y", "leaf",       pos=(12, 9),  robot=target_robot)
p8.add_node("leaf_r", "leaf",       pos=(11, 10), robot=red)

p8.add_edge("goal", "sg",      status="fixed", cost=1)
p8.add_edge("sg",   "bn",      status="fixed", cost=None)
p8.add_edge("sg",   "sp",      status="fixed", cost=None)
p8.add_edge("bn",   "leaf_y",  status="open",  cost=3)   # not yet resolved
p8.add_edge("sp",   "leaf_r",  status="open",  cost=4)   # not yet resolved

run_test("Open edges (incomplete plan)", p8, game)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 9 — Position off the grid
# ═══════════════════════════════════════════════════════════════════════════

p9 = PartialPlan()
p9.add_node("goal", "goal", pos=(20, 20))  # outside 16×16 grid
p9.add_edge("goal", "goal")                # self-loop

run_test("Position off the grid", p9, game)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 10 — String instead of Robot instance
# ═══════════════════════════════════════════════════════════════════════════

p10 = PartialPlan()
p10.add_node("goal",   "goal",  pos=(6, 15))
p10.add_node("leaf_y", "leaf",  pos=(12, 9), robot="Yellow")  # string, not Robot
p10.add_edge("goal", "leaf_y",  status="fixed", cost=1)

run_test("String instead of Robot instance", p10, game)
