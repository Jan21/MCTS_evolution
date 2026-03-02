import networkx as nx


class PartialPlan:
    """
    A DAG representing a partial plan for Ricochet Robots.

    Node types and their attributes:
        goal:       pos (target cell)
        subgoal:    entry_pos (cell where target enters final component
                    via the dependent edge — this is the position inside
                    the final component, used for goal->subgoal validation)
        bottleneck: pos (cell the target robot must reach before the
                    dependent edge — outside the final component),
                    robot (robot id)
        support:    pos (cell the helper must occupy), robot (robot id)
        leaf:       pos (current position), robot (robot id)

    Edge types:
        Physical edges (between positioned nodes): carry cost, validated
            against the grid graph's independent paths.
        Structural edges (subgoal -> bottleneck/support): grouping only,
            cost=None, not validated for physical traversability.

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

    def get_node(self, nid):
        """Return node attributes dict."""
        return dict(self.g.nodes[nid])

    def get_edge(self, parent, child):
        """Return edge attributes dict."""
        return dict(self.g.edges[parent, child])

    def nodes_by_type(self, ntype):
        """Return list of (nid, attrs) for all nodes of given type."""
        return [(n, d) for n, d in self.g.nodes(data=True)
                if d.get('ntype') == ntype]

    def children(self, nid):
        """Return list of child node ids (successors in the DiGraph)."""
        return list(self.g.successors(nid))

    def parents(self, nid):
        """Return list of parent node ids (predecessors in the DiGraph)."""
        return list(self.g.predecessors(nid))

    def is_structural_edge(self, parent, _child):
        """Check if an edge is structural (subgoal -> bottleneck/support)."""
        parent_type = self.g.nodes[parent].get('ntype')
        return parent_type == 'subgoal'

    def resolve_segment_positions(self, parent_nid, child_nid):
        """Resolve the physical (source_pos, dest_pos) for a DAG edge.

        Physical edges represent robot movements. The source position is
        where the robot starts and the destination is where it ends up.

        Resolution rules:
            - Both have pos (bn->leaf, sp->leaf): source=child.pos, dest=parent.pos
            - goal -> subgoal: source=subgoal.entry_pos, dest=goal.pos
              (path inside the final component from the entry point to the goal)
            - goal -> leaf (direct): source=leaf.pos, dest=goal.pos
            - Structural edges (subgoal -> bn/sp): returns None

        Returns:
            (source_pos, dest_pos) tuple, or None if structural/unresolvable.
        """
        if self.is_structural_edge(parent_nid, child_nid):
            return None

        parent_attrs = self.g.nodes[parent_nid]
        child_attrs = self.g.nodes[child_nid]

        parent_type = parent_attrs.get('ntype')
        child_type = child_attrs.get('ntype')

        # Both have pos: direct physical edge
        if 'pos' in parent_attrs and 'pos' in child_attrs:
            return (child_attrs['pos'], parent_attrs['pos'])

        # goal -> subgoal: path from entry_pos (inside FC) to goal
        if parent_type == 'goal' and child_type == 'subgoal':
            entry_pos = child_attrs.get('entry_pos')
            if entry_pos is None:
                return None
            return (entry_pos, parent_attrs['pos'])

        return None

    def open_edges(self):
        return [(u, v) for u, v, d in self.g.edges(data=True)
                if d["status"] == "open"]

    def validate_plan(self):
        return len(self.open_edges()) == 0

    def cost(self):
        return sum(d["cost"] for _, _, d in self.g.edges(data=True)
                   if d["cost"] is not None)


# === Example ===
# Board: target robot R0 at (0,0), helper robot R1 at (3,3), goal at (0,4).
# Subgoal: R1 moves to (0,3) as support. R0 slides from (0,0) and gets blocked
# by R1 at (0,3), stopping at (0,2). This is the dependent edge entry. From
# (0,2) inside the final component, R0 can reach goal (0,4) without help.
#
# Plan segments (physical edges):
#   1. goal -> sg1: entry_pos(0,2) -> goal(0,4), cost=2 (independent, inside FC)
#   2. bn1 -> leaf0: R0 from (0,0) -> (0,0) [already at bottleneck pos], cost=0
#      (In this example bottleneck pos = R0's current pos, so trivial)
#   3. sp1 -> leaf1: R1 from (3,3) -> (0,3), cost=3 (independent)
# The dependent edge (0,0) -> (0,2) via helper at (0,3) is implicit in sg1.

if __name__ == "__main__":
    p = PartialPlan()

    # Nodes
    p.add_node("goal",  "goal",       pos=(0, 4))
    p.add_node("sg1",   "subgoal",    entry_pos=(0, 2))
    p.add_node("bn1",   "bottleneck", pos=(0, 0), robot="R0")
    p.add_node("sp1",   "support",    pos=(0, 3), robot="R1")
    p.add_node("leaf0", "leaf",       pos=(0, 0), robot="R0")
    p.add_node("leaf1", "leaf",       pos=(3, 3), robot="R1")

    # Physical edges (carry cost, validated against independent paths)
    p.add_edge("goal", "sg1",   status="fixed", cost=2)   # entry(0,2)->goal(0,4)
    p.add_edge("bn1",  "leaf0", status="fixed", cost=0)   # R0 already at (0,0)
    p.add_edge("sp1",  "leaf1", status="open",  cost=3)   # R1 (3,3)->(0,3)

    # Structural edges (grouping only, cost=None)
    p.add_edge("sg1",  "bn1",   status="open")
    p.add_edge("sg1",  "sp1",   status="open")

    print("Nodes:", dict(p.g.nodes(data=True)))
    print("Edges:", list(p.g.edges(data=True)))
    print("Open edges:", p.open_edges())
    print("Complete:", p.validate_plan())
    print("Cost:", p.cost())
