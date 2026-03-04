"""End-to-end validation pipeline example.

Shows the full flow:
  1. Load an environment through GridEnv.from_env
  2. Build a partial plan using Robot_at instances from the state
  3. Validate the plan
  4. Act on the result

Usage:
    python validation_pipeline_example.py
"""

from GridEnv import GridEnv
from partial_plan import PartialPlan
from validate_plan import validate

# ── 1. Load environment ───────────────────────────────────────────────────

grid_env, s = GridEnv.from_env(0)

print("Loaded env_0")
print(f"  Target : {s.target_robot.color} must reach {s.target}")
print(f"  Helpers: {[h.color for h in s.helpers]}")
print()

# ── 2. Build a plan ──────────────────────────────────────────────────────

yellow = s.target_robot
red = s.helpers[0]

plan = PartialPlan()

plan.add_node("goal",   "goal",       pos=s.target)
plan.add_node("sg1",    "subgoal")
plan.add_node("bn1",    "bottleneck", pos=(5, 15),          robot=yellow)
plan.add_node("sp1",    "support",    pos=(6, 14),          robot=red)
plan.add_node("leaf_y", "leaf",       pos=yellow.position,  robot=yellow)
plan.add_node("leaf_r", "leaf",       pos=red.position,     robot=red)

plan.add_edge("goal", "sg1",    status="fixed", cost=1)
plan.add_edge("sg1",  "bn1",    status="fixed", cost=None)
plan.add_edge("sg1",  "sp1",    status="fixed", cost=None)
plan.add_edge("bn1",  "leaf_y", status="fixed", cost=3)
plan.add_edge("sp1",  "leaf_r", status="fixed", cost=4)

# ── 3. Validate ──────────────────────────────────────────────────────────

result = validate(plan, grid_env, s)

# ── 4. Act on the result ─────────────────────────────────────────────────

if result.passed:
    print(f"Plan is valid  (cost={plan.cost()})")
    print(f"  Complete: {plan.is_complete()}")
    print(f"  Nodes:    {plan.g.number_of_nodes()}")
    print(f"  Edges:    {plan.g.number_of_edges()}")
else:
    print("Plan is INVALID:")
    for err in result.errors:
        print(f"  {err}")
