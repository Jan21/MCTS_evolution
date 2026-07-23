//! Best-first search over partial plans — port of `skeleton/astar.py`
//! (CURRENT version, including the three fixes at ~lines 136/184/194).
//!
//! Control flow of `_search` is ported statement for statement: the
//! frontier-empty / max_frontier check sits at the TOP of the loop BEFORE
//! popping; completeness is tested on the popped plan; the max_open guard
//! discards popped plans; children are pushed with a monotonically
//! increasing tie counter so equal-cost plans pop FIFO — exactly Python's
//! `heapq` on `(cost, tie, plan)` tuples.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::types::{Cell, SUBGOAL_INF};

use super::grid_env::SubgoalEnv;
use super::plan::{NType, Node, NodeId, PartialPlan};
use super::{Candidate, RobotAt, State};

/// Segment description (`skeleton.astar._Seg`).
pub struct Seg {
    pub start: Cell,
    pub end: Cell,
    pub mover: RobotAt,
    pub helpers: Vec<RobotAt>,
    pub support: Option<RobotAt>,
    pub fix_support: Option<Cell>,
}

/// `_segment(plan, state, parent, child)` — describe the physical move an
/// open edge stands for. Fix (a): helpers = state.helpers minus the mover BY
/// COLOR; when the mover is not the target robot, the target robot is
/// appended at the END (order feeds candidate enumeration).
pub fn segment(plan: &PartialPlan, state: &State, parent: u32, child: u32) -> Seg {
    let pdata = &plan.nodes[parent as usize];
    let cdata = &plan.nodes[child as usize];
    let start = cdata.pos.expect("segment child has no pos");
    let end = pdata.pos.expect("segment parent has no pos");
    // Python: state.target_robot if parent is the goal else
    // pdata.get("robot", state.target_robot).
    let mover = if pdata.ntype == NType::Goal {
        state.target_robot
    } else {
        pdata.robot.unwrap_or(state.target_robot)
    };

    let support = if pdata.ntype == NType::Bottleneck {
        plan.sibling_support(parent)
    } else {
        None
    };
    let fix_support = support.map(|r| r.pos);

    let mut helpers: Vec<RobotAt> = state
        .helpers
        .iter()
        .filter(|h| h.color != mover.color)
        .copied()
        .collect();
    if mover.color != state.target_robot.color {
        helpers.push(state.target_robot);
    }

    Seg { start, end, mover, helpers, support, fix_support }
}

/// `_initial_plan(env, state)`.
pub fn initial_plan(env: &SubgoalEnv, state: &State) -> PartialPlan {
    let mut plan = PartialPlan::new();
    let goal = plan.add_node(Node {
        id: NodeId::Goal,
        ntype: NType::Goal,
        pos: Some(state.target),
        robot: None,
        parent_support_pos: None,
    });
    let leaf = plan.add_node(Node {
        id: NodeId::Leaf(0),
        ntype: NType::Leaf,
        pos: Some(state.target_robot.pos),
        robot: Some(state.target_robot),
        parent_support_pos: None,
    });
    plan.nc = 1;
    let exact = env.compute_exact(state.target_robot.pos, state.target, None);
    let relaxed = env.compute_relaxed(state.target_robot.pos, state.target, None);
    match (exact, relaxed) {
        (Some(e), r) if r.is_none() || e <= r.unwrap() => {
            plan.add_edge(goal, leaf, false, Some(e));
        }
        _ => {
            plan.add_edge(goal, leaf, true, Some(relaxed.unwrap_or(SUBGOAL_INF)));
        }
    }
    plan
}

/// `_reference_helpers(plan, mover_color)` — Lever B2 phantom helpers:
/// robots the plan already places, offered at their PLANNED cells (terminal
/// support nodes, or mid-chain bottlenecks they pass through). The mover is
/// excluded; nodes iterate in insertion order (Python dict order), dedup on
/// (color, pos) first-seen. (Python also excludes park-holding robots; park
/// nodes never occur in the label engine.)
pub fn reference_helpers(plan: &PartialPlan, mover_color: u8) -> Vec<RobotAt> {
    use rustc_hash::FxHashSet;
    let mut out: Vec<RobotAt> = Vec::new();
    let mut seen: FxHashSet<(u8, Cell)> = FxHashSet::default();
    for (i, node) in plan.nodes.iter().enumerate() {
        if node.ntype != NType::Support && node.ntype != NType::Bottleneck {
            continue;
        }
        let robot = node.robot.expect("support/bottleneck without robot");
        if robot.color == mover_color {
            continue;
        }
        if node.ntype == NType::Support && !plan.terminal_support(i as u32) {
            continue;
        }
        let pos = node.pos.expect("support/bottleneck without pos");
        if seen.insert((robot.color, pos)) {
            out.push(RobotAt { pos, color: robot.color });
        }
    }
    out
}

/// `_apply(env, plan, parent, child, seg, cand, by_reference)` — attach the
/// candidate's subgoal between parent and child, or None when an invariant
/// rejects it. Graph mutations follow the Python statement order exactly.
pub fn apply(
    env: &SubgoalEnv,
    plan: &PartialPlan,
    parent: u32,
    child: u32,
    seg: &Seg,
    cand: &Candidate,
    by_reference: bool,
) -> Option<PartialPlan> {
    let bn_pos = cand.subgoal.bottleneck.pos;
    let sp_pos = cand.subgoal.support.pos;

    // Fix (b): one physical robot, one leaf identity — a candidate whose
    // helper COLOR already owns a leaf is rejected; under Lever B2 it may
    // instead serve BY REFERENCE: the candidate helper stands at the robot's
    // planned cell and the subgoal wires to the existing plan node. Three
    // shapes: shared support, target-as-stopper, relocation.
    let helper_color = cand.subgoal.helper.color;
    let mut ref_node: Option<u32> = None;
    if plan.nodes.iter().any(|n| {
        n.ntype == NType::Leaf && n.robot.map(|r| r.color) == Some(helper_color)
    }) {
        if !by_reference {
            return None;
        }
        let hp = cand.subgoal.helper.pos;
        for (i, n2) in plan.nodes.iter().enumerate() {
            let nt2 = n2.ntype;
            if (nt2 == NType::Support || nt2 == NType::Bottleneck)
                && n2.robot.map(|r| r.color) == Some(helper_color)
                && n2.pos == Some(hp)
                && (nt2 == NType::Bottleneck || plan.terminal_support(i as u32))
            {
                ref_node = Some(i as u32);
                break;
            }
        }
        let Some(rn) = ref_node else {
            return None; // candidate not at a planned cell: stale
        };
        if plan.nodes[rn as usize].ntype == NType::Bottleneck && sp_pos != hp {
            return None; // a robot mid-chain can only serve in place
        }
        if plan.reaches(rn, parent) {
            // The referenced placement depends (via the DAG) on the very
            // segment that would consume it — a circular timing requirement
            // no schedule can satisfy. Wiring it would create a cycle.
            return None;
        }
    }

    // Fix (c): a real parent_support cell may only be claimed when it equals
    // the candidate's own support cell or an existing support node's cell.
    // Under B2 a referenced mid-chain bottleneck node also counts as placed.
    if let Some(ps) = cand.parent_support {
        if ps != sp_pos
            && !plan
                .nodes
                .iter()
                .any(|n| n.ntype == NType::Support && n.pos == Some(ps))
            && !(by_reference
                && plan
                    .nodes
                    .iter()
                    .any(|n| n.ntype == NType::Bottleneck && n.pos == Some(ps)))
        {
            return None;
        }
    }

    let parent_cost = env.compute_exact(bn_pos, seg.end, cand.parent_support)?;

    let mut out = plan.clone();
    out.remove_edge(parent, child);
    let n = out.nc;
    out.nc += 1;
    let sg = out.add_node(Node {
        id: NodeId::Sg(n),
        ntype: NType::Subgoal,
        pos: None,
        robot: None,
        parent_support_pos: cand.parent_support,
    });
    let bn = out.add_node(Node {
        id: NodeId::Bn(n),
        ntype: NType::Bottleneck,
        pos: Some(bn_pos),
        robot: Some(cand.subgoal.bottleneck),
        parent_support_pos: None,
    });
    out.add_edge(parent, sg, false, Some(parent_cost));
    out.add_edge(sg, bn, false, Some(0));

    // mover travels (child pos) -> bottleneck, stopping via the support.
    attach_move(env, &mut out, bn, child, seg.start, bn_pos, Some(sp_pos));

    match ref_node {
        None => {
            // Fresh recruitment (the only pre-B2 path): a new support node
            // delivered from the helper's own leaf.
            let sp = out.add_node(Node {
                id: NodeId::Sp(n),
                ntype: NType::Support,
                pos: Some(sp_pos),
                robot: Some(cand.subgoal.support),
                parent_support_pos: None,
            });
            out.add_edge(sg, sp, false, Some(0));
            let leaf = out.add_node(Node {
                id: NodeId::Leaf(n),
                ntype: NType::Leaf,
                pos: Some(cand.subgoal.helper.pos),
                robot: Some(cand.subgoal.helper),
                parent_support_pos: None,
            });
            attach_move(env, &mut out, sp, leaf, cand.subgoal.helper.pos, sp_pos, None);
        }
        Some(rn) if Some(sp_pos) == out.nodes[rn as usize].pos => {
            // Shared support / target-as-stopper: reference the existing
            // node. Zero extra moves; the realizer orders every bounce
            // consuming the node before its robot departs.
            out.add_edge_byref(sg, rn, Some(0));
        }
        Some(rn) => {
            // Relocation: the robot slides from its terminal support cell to
            // a new support cell — an ordinary costed plan edge hung off the
            // existing node.
            let sp = out.add_node(Node {
                id: NodeId::Sp(n),
                ntype: NType::Support,
                pos: Some(sp_pos),
                robot: Some(cand.subgoal.support),
                parent_support_pos: None,
            });
            out.add_edge(sg, sp, false, Some(0));
            let start = out.nodes[rn as usize].pos.expect("ref node without pos");
            attach_move(env, &mut out, sp, rn, start, sp_pos, None);
        }
    }
    Some(out)
}

/// `_attach_move`: fixed at the exact cost when one exists, else open at the
/// relaxed cost (INF when even that is unreachable).
fn attach_move(
    env: &SubgoalEnv,
    plan: &mut PartialPlan,
    parent: u32,
    child: u32,
    start: Cell,
    end: Cell,
    support: Option<Cell>,
) {
    if let Some(exact) = env.compute_exact(start, end, support) {
        plan.add_edge(parent, child, false, Some(exact));
    } else {
        let relaxed = env.compute_relaxed(start, end, support);
        plan.add_edge(parent, child, true, Some(relaxed.unwrap_or(SUBGOAL_INF)));
    }
}

/// Solver caps (`skeleton.astar.AStar.__init__` defaults) plus the
/// vocabulary levers: `b1` = transient supports in `propose`
/// (`heuristics.propose_b1`), `by_reference` = Lever B2 supports-by-reference
/// (`AStar(by_reference=True)`). Both default false — default path unchanged.
#[derive(Clone, Copy, Debug)]
pub struct AStar {
    pub max_iters: u64,
    pub beam: Option<usize>,
    pub max_open: Option<usize>,
    pub max_frontier: usize,
    pub b1: bool,
    pub by_reference: bool,
}

impl Default for AStar {
    fn default() -> Self {
        AStar {
            max_iters: 20_000,
            beam: None,
            max_open: None,
            max_frontier: 100_000,
            b1: false,
            by_reference: false,
        }
    }
}

/// Map a work-item `vocab` string to the two solver levers.
/// None / "base" → original vocabulary; "b1" → + transient supports;
/// "b2" → + supports-by-reference (implies b1).
pub fn vocab_flags(vocab: Option<&str>) -> anyhow::Result<(bool, bool)> {
    match vocab {
        None | Some("base") => Ok((false, false)),
        Some("b1") => Ok((true, false)),
        Some("b2") => Ok((true, true)),
        Some(other) => anyhow::bail!("unknown vocab {other:?} (want base|b1|b2)"),
    }
}

/// Search outcome: completed plan (or None) plus the number of loop
/// iterations ENTERED — the deterministic budget unit of DESIGN §3
/// (`budget.solver_iters`). An iteration that only trips the top-of-loop
/// frontier check still counts as entered.
pub struct SearchOutcome {
    pub plan: Option<PartialPlan>,
    pub iters: u64,
}

impl AStar {
    /// `solve_plan`: complete `start`, returning the optimal completion or
    /// None (Python: `res.best_plan if res.all_plans else None`).
    pub fn solve_plan(&self, env: &SubgoalEnv, state: &State, start: PartialPlan) -> SearchOutcome {
        self.search(env, state, start)
    }

    /// `_search` — statement-for-statement port.
    fn search(&self, env: &SubgoalEnv, state: &State, start: PartialPlan) -> SearchOutcome {
        // Arena keeps frontier plans; popped slots are vacated via mem::take
        // (each heap entry owns its index, so a slot is taken exactly once).
        let mut arena: Vec<PartialPlan> = Vec::new();
        let mut heap: BinaryHeap<Reverse<(i64, u64, u32)>> = BinaryHeap::new();
        let start_cost = start.cost();
        arena.push(start);
        heap.push(Reverse((start_cost, 0, 0)));
        let mut tie: u64 = 1;
        let max_open = self
            .max_open
            .unwrap_or(2 * (state.helpers.len() + 2));

        let mut entered: u64 = 0;
        for _it in 1..=self.max_iters {
            entered += 1;
            if heap.is_empty() || heap.len() > self.max_frontier {
                break;
            }
            let Reverse((_cost, _t, idx)) = heap.pop().unwrap();
            let cur = std::mem::take(&mut arena[idx as usize]);
            if cur.is_complete() {
                return SearchOutcome { plan: Some(cur), iters: entered };
            }
            if cur.open_edge_count() > max_open {
                continue;
            }
            let mut children = self.expand(env, state, &cur);
            if let Some(beam) = self.beam {
                // `sorted(children, key=cost)[:beam]` — stable sort, unused
                // in datagen but ported.
                children.sort_by_key(|p| p.cost());
                children.truncate(beam);
            }
            for child in children {
                let c = child.cost();
                arena.push(child);
                heap.push(Reverse((c, tie, (arena.len() - 1) as u32)));
                tie += 1;
            }
        }
        SearchOutcome { plan: None, iters: entered }
    }

    /// `_expand` — resolve the FIRST open edge (`open_edges()[0]`): pin it to
    /// an exact path when one exists (single child), else propose subgoals,
    /// stable-sorted by score, `_apply` each (rejections dropped). With
    /// `by_reference`, reference helpers are appended to the segment's
    /// helper list before proposing (Python `_expand` lines 105-106).
    pub fn expand(&self, env: &SubgoalEnv, state: &State, plan: &PartialPlan) -> Vec<PartialPlan> {
        let (parent, child) = plan
            .first_open_edge()
            .expect("_expand on a complete plan");
        let mut seg = segment(plan, state, parent, child);

        if let Some(exact) = env.compute_exact(seg.start, seg.end, seg.fix_support) {
            let mut out = plan.clone();
            out.update_edge(parent, child, false, Some(exact));
            return vec![out];
        }

        if self.by_reference {
            seg.helpers
                .extend(reference_helpers(plan, seg.mover.color));
        }
        let mut cands = env.propose(seg.end, &seg.mover, &seg.helpers, seg.support.as_ref(), self.b1);
        // Python: `cands.sort(key=lambda c: self.score(env, c))` — score()
        // recomputes subgoal_score, which equals the stored candidate score
        // by construction (heuristics.propose sets it from the same call).
        cands.sort_by_key(|c| c.score);
        cands
            .iter()
            .filter_map(|c| apply(env, plan, parent, child, &seg, c, self.by_reference))
            .collect()
    }
}
