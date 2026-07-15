//! `PartialPlan` — port of `partial_plan.py` with NetworkX-faithful
//! iteration orders.
//!
//! CRITICAL ORDER CONTRACT: `nx.DiGraph.edges()` iterates NODES in insertion
//! order and, per node, that node's OUT-edges in adjacency insertion order.
//! That is NOT global edge-creation order once `_apply` removes and re-adds
//! edges on interior nodes. `edges_nx()` reproduces the nested order exactly;
//! `open_edges()[0]` (the segment `_expand`/rollout works on) and the
//! `ctx_open_endpoints` record field depend on it.
//!
//! Nodes are never removed. `remove_edge` deletes exactly one (u, v) edge and
//! preserves the relative order of the rest (Python dict deletion). Costs and
//! sums are order-independent. Cloning is the search hot path: nodes and
//! edges are `Copy` structs in two flat `Vec`s, so `clone()` is two memcpys.

use crate::types::Cell;

use super::RobotAt;

/// Node id, mirroring the Python string ids ("goal", "sg_3", "bn_3", "sp_3",
/// "leaf_0"). Kept for replay-dump reconstruction and debugging.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeId {
    Goal,
    Sg(u32),
    Bn(u32),
    Sp(u32),
    Leaf(u32),
}

impl NodeId {
    /// Parse a Python plan-node id. Errors on anything outside the scheme
    /// (fail loud — DESIGN §7).
    pub fn parse(s: &str) -> Option<NodeId> {
        if s == "goal" {
            return Some(NodeId::Goal);
        }
        let (kind, num) = s.rsplit_once('_')?;
        let n: u32 = num.parse().ok()?;
        match kind {
            "sg" => Some(NodeId::Sg(n)),
            "bn" => Some(NodeId::Bn(n)),
            "sp" => Some(NodeId::Sp(n)),
            "leaf" => Some(NodeId::Leaf(n)),
            _ => None,
        }
    }

    pub fn to_string_id(self) -> String {
        match self {
            NodeId::Goal => "goal".to_string(),
            NodeId::Sg(n) => format!("sg_{n}"),
            NodeId::Bn(n) => format!("bn_{n}"),
            NodeId::Sp(n) => format!("sp_{n}"),
            NodeId::Leaf(n) => format!("leaf_{n}"),
        }
    }
}

/// `ntype` attribute values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NType {
    Goal,
    Subgoal,
    Bottleneck,
    Support,
    Leaf,
}

/// One plan node. `pos` is None only for subgoal nodes; `robot` only on
/// bottleneck/support/leaf nodes; `parent_support_pos` is the subgoal-node
/// attribute (itself an Option — `Some(None)`-style flattening is safe
/// because only subgoal nodes are ever read for it).
#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub id: NodeId,
    pub ntype: NType,
    pub pos: Option<Cell>,
    pub robot: Option<RobotAt>,
    pub parent_support_pos: Option<Cell>,
}

/// One edge; `u`/`v` are node indices into `PartialPlan::nodes`.
#[derive(Clone, Copy, Debug)]
pub struct EdgeRec {
    pub u: u32,
    pub v: u32,
    pub open: bool,
    pub cost: Option<i64>,
}

#[derive(Clone, Debug, Default)]
pub struct PartialPlan {
    pub nodes: Vec<Node>,
    /// Edges in CREATION order. NetworkX iteration order is derived by
    /// [`PartialPlan::edges_nx`]; within one `u` the creation order equals
    /// the adjacency insertion order, which is all NetworkX preserves.
    pub edges: Vec<EdgeRec>,
    /// Python `plan.nc` — node-id counter for `_apply`.
    pub nc: u32,
}

impl PartialPlan {
    pub fn new() -> Self {
        PartialPlan::default()
    }

    /// `add_node`; returns the node index.
    pub fn add_node(&mut self, node: Node) -> u32 {
        self.nodes.push(node);
        (self.nodes.len() - 1) as u32
    }

    /// `add_edge(parent, child, status, cost)`. The (u, v) pair must be new —
    /// plan construction never re-adds an existing edge (asserted in debug).
    pub fn add_edge(&mut self, u: u32, v: u32, open: bool, cost: Option<i64>) {
        debug_assert!(
            !self.edges.iter().any(|e| e.u == u && e.v == v),
            "duplicate edge {u}->{v}"
        );
        self.edges.push(EdgeRec { u, v, open, cost });
    }

    /// `g.remove_edge(u, v)` — removes the single (u, v) edge, preserving the
    /// relative order of the others.
    pub fn remove_edge(&mut self, u: u32, v: u32) {
        let i = self
            .edges
            .iter()
            .position(|e| e.u == u && e.v == v)
            .unwrap_or_else(|| panic!("remove_edge: no edge {u}->{v}"));
        self.edges.remove(i);
    }

    /// `g[u][v].update(status=..., cost=...)` — in-place attr update, order
    /// untouched.
    pub fn update_edge(&mut self, u: u32, v: u32, open: bool, cost: Option<i64>) {
        let e = self
            .edges
            .iter_mut()
            .find(|e| e.u == u && e.v == v)
            .unwrap_or_else(|| panic!("update_edge: no edge {u}->{v}"));
        e.open = open;
        e.cost = cost;
    }

    /// Edge indices in NetworkX `g.edges()` order: nodes in insertion order,
    /// each node's out-edges in adjacency insertion order.
    pub fn edges_nx(&self) -> impl Iterator<Item = &EdgeRec> {
        (0..self.nodes.len() as u32)
            .flat_map(move |u| self.edges.iter().filter(move |e| e.u == u))
    }

    /// `open_edges()` as (u, v) node-index pairs, in `edges_nx` order.
    pub fn open_edges(&self) -> Vec<(u32, u32)> {
        self.edges_nx()
            .filter(|e| e.open)
            .map(|e| (e.u, e.v))
            .collect()
    }

    /// `open_edges()[0]` without building the list.
    pub fn first_open_edge(&self) -> Option<(u32, u32)> {
        self.edges_nx().find(|e| e.open).map(|e| (e.u, e.v))
    }

    pub fn open_edge_count(&self) -> usize {
        self.edges.iter().filter(|e| e.open).count()
    }

    /// `is_complete()`.
    pub fn is_complete(&self) -> bool {
        !self.edges.iter().any(|e| e.open)
    }

    /// `cost()` — sum of all non-None edge costs (open and fixed).
    pub fn cost(&self) -> i64 {
        self.edges.iter().filter_map(|e| e.cost).sum()
    }

    /// `nn.generate._fixed_g` — sum of FIXED non-None edge costs.
    pub fn fixed_g(&self) -> i64 {
        self.edges
            .iter()
            .filter(|e| !e.open)
            .filter_map(|e| e.cost)
            .sum()
    }

    /// `_sibling_support(plan, bottleneck)`: the support-node robot attached
    /// to the (unique) subgoal parent of `bottleneck`. Predecessor order is
    /// immaterial: bottleneck nodes have exactly one in-edge (sg → bn) and a
    /// subgoal has exactly one support child.
    pub fn sibling_support(&self, bottleneck: u32) -> Option<RobotAt> {
        for e in &self.edges {
            if e.v == bottleneck && self.nodes[e.u as usize].ntype == NType::Subgoal {
                let sg = e.u;
                for e2 in &self.edges {
                    if e2.u == sg && self.nodes[e2.v as usize].ntype == NType::Support {
                        return self.nodes[e2.v as usize].robot;
                    }
                }
            }
        }
        None
    }

    /// Node index by id (replay-dump reconstruction only — never in search).
    pub fn node_index(&self, id: NodeId) -> Option<u32> {
        self.nodes.iter().position(|n| n.id == id).map(|i| i as u32)
    }
}
