import json
import networkx as nx



class PartialPlan:
    """
    A DAG representing a partial plan for Ricochet Robots.

    Node types and their attributes:
        goal:       pos (target cell)
        subgoal:    (no extra attrs, just groups bottleneck + support)
        bottleneck: pos (cell the robot must reach), robot (robot id)
        support:    pos (cell the helper must occupy), robot (robot id)
        leaf:       pos (current position), robot (robot id)

    Edge attributes:
        status: "open" or "fixed"
        cost:   relaxed SPL (open) or exact SPL (fixed), or None
    """

    def __init__(self):
        self.g = nx.DiGraph()

    def add_node(self, nid, ntype, **attrs):
        self.g.add_node(nid, ntype=ntype, **attrs)

    def add_edge(self, parent, child, status="open", cost=None):
        self.g.add_edge(parent, child, status=status, cost=cost)

    def open_edges(self):
        return [(u, v) for u, v, d in self.g.edges(data=True)
                if d["status"] == "open"]

    def is_complete(self):
        return len(self.open_edges()) == 0

    def cost(self):
        return sum(d["cost"] for _, _, d in self.g.edges(data=True)
                   if d["cost"] is not None)

    def to_dict(self):
        nodes = []
        for nid, attrs in self.g.nodes(data=True):
            node = {"id": nid, "type": attrs["ntype"]}
            if "pos" in attrs:
                node["pos"] = list(attrs["pos"])
            if "robot" in attrs:
                node["robot"] = attrs["robot"]
            nodes.append(node)
        edges = []
        for u, v, d in self.g.edges(data=True):
            edges.append({
                "from": u,
                "to": v,
                "status": d.get("status", "open"),
                "cost": d.get("cost"),
            })
        return {"nodes": nodes, "edges": edges}

    def save(self, path="plan_data.json"):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def save_plans(plans, path="plan_data.json"):
        with open(path, "w") as f:
            json.dump([p.to_dict() for p in plans], f, indent=2)

    @classmethod
    def from_moves(cls, grid_env, state, moves):
        """Build a PartialPlan from a sequence of (robot, direction) moves.

        Simulates moves step by step, identifies robot-blocking dependencies,
        and constructs the plan DAG.
        """
        from GridEnv import State

        target_color = state.target_robot.color.lower()
        color_order = ["red", "blue", "green", "yellow"]

        # Build initial positions dict
        positions = {}
        for robot in state.all_robots:
            positions[robot.color.lower()] = robot.position
        for c in color_order:
            if c not in positions:
                positions[c] = (grid_env.grid_size[0] - 1, grid_env.grid_size[0] - 1)

        # Simulate all moves, recording trajectory + blockers
        trajectory = []  # (robot, direction, from_pos, to_pos, blocker)
        for robot, direction in moves:
            robot = robot.lower()
            from_pos = positions[robot]
            to_pos, blocker = grid_env.simulate_slide(positions, robot, direction)
            trajectory.append((robot, direction, from_pos, to_pos, blocker))
            positions[robot] = to_pos

        # Identify segments: group consecutive moves ending with a robot-block
        # Each time the target robot is blocked by another robot, that's a subgoal.
        # Walk through trajectory and find robot-blocking dependencies.
        p = cls()
        p.add_node("goal", "goal", pos=state.target)

        # Find all moves where a robot was blocked by another robot
        # These create the subgoal structure
        dependencies = []  # (move_idx, moving_robot, stop_pos, blocker_color, blocker_pos)
        for i, (robot, direction, from_pos, to_pos, blocker) in enumerate(trajectory):
            if blocker != "wall":
                blocker_pos = positions_at_step(trajectory, positions_initial(state, color_order, grid_env), blocker, i)
                dependencies.append((i, robot, to_pos, blocker, blocker_pos))

        if not dependencies:
            # No robot-robot blocking — direct path
            start_pos = positions_initial(state, color_order, grid_env)[target_color]
            leaf_id = f"leaf_{target_color}"
            p.add_node(leaf_id, "leaf", pos=start_pos, robot=target_color)
            p.add_edge("goal", leaf_id, status="fixed", cost=len(moves))
            return p

        # Build subgoals from dependencies
        # Process in reverse: last dependency is closest to goal
        sg_count = 0
        parent = "goal"
        # Track which move indices have been "claimed" as support setup
        claimed_support_moves = set()

        # Collect dependencies in reverse order for the target robot
        target_deps = [(i, r, sp, bl, bp) for i, r, sp, bl, bp in dependencies if r == target_color]

        # Also collect helper-robot dependencies (helpers blocked by other helpers)
        helper_deps = [(i, r, sp, bl, bp) for i, r, sp, bl, bp in dependencies if r != target_color]

        for dep_idx, (move_i, robot, stop_pos, blocker, blocker_pos) in enumerate(reversed(target_deps)):
            sg_count += 1
            sg_id = f"sg{sg_count}"
            bn_id = f"bn{sg_count}"
            sp_id = f"sp{sg_count}"

            p.add_node(sg_id, "subgoal")
            p.add_node(bn_id, "bottleneck", pos=stop_pos, robot=robot)
            p.add_node(sp_id, "support", pos=blocker_pos, robot=blocker)

            # Count target robot moves from this stop_pos to goal/next bottleneck
            target_moves_after = sum(1 for j, (r, d, fp, tp, b) in enumerate(trajectory)
                                     if j > move_i and r == target_color)
            p.add_edge(parent, sg_id, status="fixed", cost=target_moves_after + 1 if dep_idx == 0 else 1)
            p.add_edge(sg_id, bn_id, status="fixed")
            p.add_edge(sg_id, sp_id, status="fixed")

            # Find where the blocker started (before any of its moves leading to blocker_pos)
            init_positions = positions_initial(state, color_order, grid_env)
            blocker_start = find_robot_start_for_support(trajectory, init_positions, blocker, move_i)
            blocker_move_count = count_robot_moves(trajectory, blocker, 0, move_i)

            leaf_sp = f"leaf_{blocker}_{sg_count}"
            p.add_node(leaf_sp, "leaf", pos=blocker_start, robot=blocker)
            p.add_edge(sp_id, leaf_sp, status="fixed", cost=blocker_move_count)

            parent = bn_id

        # Leaf for target robot's starting position
        init_positions = positions_initial(state, color_order, grid_env)
        first_target_dep_move = target_deps[0][0] if target_deps else len(trajectory)
        target_start = init_positions[target_color]
        target_pre_moves = count_robot_moves(trajectory, target_color, 0, first_target_dep_move)

        leaf_target = f"leaf_{target_color}"
        p.add_node(leaf_target, "leaf", pos=target_start, robot=target_color)
        p.add_edge(parent, leaf_target, status="fixed", cost=target_pre_moves)

        return p


def positions_initial(state, color_order, grid_env):
    positions = {}
    for robot in state.all_robots:
        positions[robot.color.lower()] = robot.position
    for c in color_order:
        if c not in positions:
            positions[c] = (grid_env.grid_size[0] - 1, grid_env.grid_size[0] - 1)
    return positions


def positions_at_step(trajectory, init_positions, robot_color, before_step):
    """Get position of robot_color just before step `before_step`."""
    pos = init_positions[robot_color]
    for i, (robot, direction, from_pos, to_pos, blocker) in enumerate(trajectory):
        if i >= before_step:
            break
        if robot == robot_color:
            pos = to_pos
    return pos


def find_robot_start_for_support(trajectory, init_positions, robot_color, before_step):
    """Find where robot_color was before any of its moves up to before_step."""
    return init_positions[robot_color]


def count_robot_moves(trajectory, robot_color, from_step, to_step):
    """Count moves by robot_color in trajectory[from_step:to_step]."""
    return sum(1 for i, (r, d, fp, tp, b) in enumerate(trajectory)
               if from_step <= i < to_step and r == robot_color)


# === Example ===
# Board: target robot R0 at (0,0), helper robot R1 at (3,3), goal at (0,4).
# Plan: R1 moves to (0,3) as support so R0 slides from (0,0) to (0,3-1)=(0,2),
#        then from bottleneck (0,2) R0 can reach goal (0,4) without help.

if __name__ == "__main__":
    p = PartialPlan()

    # Nodes
    p.add_node("goal",  "goal",       pos=(0, 4))
    p.add_node("sg1",   "subgoal")
    p.add_node("bn1",   "bottleneck", pos=(0, 2), robot="R0")
    p.add_node("sp1",   "support",    pos=(0, 3), robot="R1")
    p.add_node("leaf0", "leaf",       pos=(0, 0), robot="R0")
    p.add_node("leaf1", "leaf",       pos=(3, 3), robot="R1")

    # Edges
    p.add_edge("goal", "sg1",   status="fixed", cost=2)   # bn(0,2)->goal(0,4): exact
    p.add_edge("sg1",  "bn1",   status="open",  cost=1)   # leaf0->bn1: relaxed
    p.add_edge("sg1",  "sp1",   status="open",  cost=3)   # leaf1->sp1: relaxed
    p.add_edge("bn1",  "leaf0", status="open",  cost=1)   # R0 (0,0)->(0,2)
    p.add_edge("sp1",  "leaf1", status="open",  cost=3)   # R1 (3,3)->(0,3)

    print("Nodes:", dict(p.g.nodes(data=True)))
    print("Edges:", list(p.g.edges(data=True)))
    print("Open edges:", p.open_edges())
    print("Complete:", p.is_complete())
    print("Cost:", p.cost())
