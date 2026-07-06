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

    def copy(self):
        """Cheap copy: networkx copies graph STRUCTURE + attr dicts but SHARES the attr
        VALUES (positions, Robot_at, costs) -- safe because they are never mutated in
        place (subgoals only ADD new nodes/edges). Much faster than copy.deepcopy."""
        p = PartialPlan()
        p.g = self.g.copy()
        if hasattr(self, "nc"):
            p.nc = self.nc
        return p

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
