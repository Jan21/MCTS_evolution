//! Unit tests for the subgoal machinery (Agent C): the three astar.py fixes
//! on hand-built minimal plans where each fix flips the outcome, the
//! extended-graph direction/blocker rules on a hand-drawn 5×5 board, the
//! initial-plan exact-vs-relaxed branch, and the deterministic rollout budget.

use rust_datagen::board::CompiledBoard;
use rust_datagen::subgoal::astar::{apply, initial_plan, segment, AStar};
use rust_datagen::subgoal::plan::{EdgeRec, NType, Node, NodeId, PartialPlan};
use rust_datagen::subgoal::rollout::{rollout, RolloutStatus};
use rust_datagen::subgoal::{Candidate, RobotAt, State, Subgoal, SubgoalEnv};
use rust_datagen::types::{Cell, SUBGOAL_INF};

/// Empty 5×5 board (border walls only). Slides stop only at the border, so
/// every interior cell is reachable ONLY via dependent edges — a crisp
/// setting for the extended-graph and exact-vs-relaxed distinctions.
fn board5() -> CompiledBoard {
    let g: Vec<String> = [
        "NW", "N", "N", "N", "NE", //
        "W", "", "", "", "E", //
        "W", "", "", "", "E", //
        "W", "", "", "", "E", //
        "SW", "S", "S", "S", "SE",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    CompiledBoard::compile(5, &g, 2)
}

/// 5×5 board with cell (2,2) walled on all four sides — an isolated cell,
/// unreachable even in the relaxed graph.
fn board5_isolated() -> CompiledBoard {
    let g: Vec<String> = [
        "NW", "N", "N", "N", "NE", //
        "W", "", "S", "", "E", //
        "W", "E", "NESW", "W", "E", //
        "W", "", "N", "", "E", //
        "SW", "S", "S", "S", "SE",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    CompiledBoard::compile(5, &g, 2)
}

fn fc_cells(fc: &[bool], n: u16) -> Vec<Cell> {
    let mut v: Vec<Cell> = fc
        .iter()
        .enumerate()
        .filter(|(_, &m)| m)
        .map(|(i, _)| ((i % n as usize) as u16, (i / n as usize) as u16))
        .collect();
    v.sort();
    v
}

// robots: slot 0 = Red, 1 = Blue, 2 = Green, 3 = Yellow
fn r(pos: Cell, slot: u8) -> RobotAt {
    RobotAt { pos, color: slot }
}

// ---------------------------------------------------------------------------
// extended graph: direction rule + blocker row/column removal (DESIGN §5.7)
// ---------------------------------------------------------------------------

#[test]
fn extended_graph_row_blocker_and_conversion() {
    let b = board5();
    let env = SubgoalEnv::new(&b);
    // bottleneck (2,2), support (3,2): rightward slides from (0,2)/(1,2)
    // stop at (2,2) thanks to the support -> their dependent edges CONVERT
    // to independent (edge direction opposite to main_vec = bn - sup).
    // Blocker removal: support (3,2) and everything past it on the shared
    // row, i.e. (4,2). Nothing else reaches (0,2)/(1,2) except each other.
    assert_eq!(
        fc_cells(&env.extended_final_component((2, 2), (3, 2)), 5),
        vec![(0, 2), (1, 2)]
    );
}

#[test]
fn extended_graph_column_blocker_and_conversion() {
    let b = board5();
    let env = SubgoalEnv::new(&b);
    // bottleneck (2,2), support (2,3): downward slides from (2,0)/(2,1).
    // Column rule removes (2,3) and (2,4).
    assert_eq!(
        fc_cells(&env.extended_final_component((2, 2), (2, 3)), 5),
        vec![(2, 0), (2, 1)]
    );
}

#[test]
fn plain_final_components_on_empty_board() {
    let b = board5();
    let env = SubgoalEnv::new(&b);
    // Interior cells have no independent in-edges -> empty final component.
    assert_eq!(fc_cells(&env.final_component((2, 2)), 5), vec![]);
    // Border cell (0,2): every cell on row 2 slides left into it; nothing
    // else stops there.
    assert_eq!(
        fc_cells(&env.final_component((0, 2)), 5),
        vec![(1, 2), (2, 2), (3, 2), (4, 2)]
    );
}

// ---------------------------------------------------------------------------
// initial plan: exact-vs-relaxed branch (skeleton.astar._initial_plan)
// ---------------------------------------------------------------------------

#[test]
fn initial_plan_exact_branch() {
    let b = board5();
    let env = SubgoalEnv::new(&b);
    // Robot (1,2) -> target (0,2): exact slide, cost 1 -> FIXED edge.
    let state = State { target: (0, 2), target_robot: r((1, 2), 0), helpers: vec![] };
    let plan = initial_plan(&env, &state);
    assert_eq!(plan.edges.len(), 1);
    assert!(!plan.edges[0].open);
    assert_eq!(plan.edges[0].cost, Some(1));
    assert!(plan.is_complete());
    assert_eq!(plan.nc, 1);
}

#[test]
fn initial_plan_relaxed_branch() {
    let b = board5();
    let env = SubgoalEnv::new(&b);
    // Robot (0,2) -> target (2,2): no independent path (exact None), but a
    // dependent edge of weight 2 -> OPEN edge at the relaxed estimate.
    let state = State { target: (2, 2), target_robot: r((0, 2), 0), helpers: vec![] };
    let plan = initial_plan(&env, &state);
    assert!(plan.edges[0].open);
    assert_eq!(plan.edges[0].cost, Some(2));
}

#[test]
fn initial_plan_inf_branch() {
    let b = board5_isolated();
    let env = SubgoalEnv::new(&b);
    // Target (2,2) is walled in: exact and relaxed both None -> OPEN at INF.
    let state = State { target: (2, 2), target_robot: r((0, 2), 0), helpers: vec![] };
    let plan = initial_plan(&env, &state);
    assert!(plan.edges[0].open);
    assert_eq!(plan.edges[0].cost, Some(SUBGOAL_INF));
}

// ---------------------------------------------------------------------------
// fix (a): _segment helpers minus mover BY COLOR, target appended at END
// ---------------------------------------------------------------------------

#[test]
fn fix_a_segment_helpers_by_color_target_last() {
    let b = board5();
    let _env = SubgoalEnv::new(&b);
    // State: Red is the target robot; helpers Blue and Green at their
    // ORIGINAL positions.
    let state = State {
        target: (0, 2),
        target_robot: r((4, 4), 0),
        helpers: vec![r((3, 3), 1), r((1, 1), 2)],
    };
    // Hand-built plan: the open segment hangs off a bottleneck whose robot
    // is BLUE at a PLANNED position (2,2) — different from Blue's original
    // (3,3), so equality-by-value would fail to drop it from the helpers.
    let mut plan = PartialPlan::new();
    let goal = plan.add_node(Node { id: NodeId::Goal, ntype: NType::Goal, pos: Some((0, 2)), robot: None, parent_support_pos: None });
    let leaf0 = plan.add_node(Node { id: NodeId::Leaf(0), ntype: NType::Leaf, pos: Some((4, 4)), robot: Some(r((4, 4), 0)), parent_support_pos: None });
    let sg = plan.add_node(Node { id: NodeId::Sg(1), ntype: NType::Subgoal, pos: None, robot: None, parent_support_pos: None });
    let bn = plan.add_node(Node { id: NodeId::Bn(1), ntype: NType::Bottleneck, pos: Some((2, 2)), robot: Some(r((2, 2), 1)), parent_support_pos: None });
    let sp = plan.add_node(Node { id: NodeId::Sp(1), ntype: NType::Support, pos: Some((3, 2)), robot: Some(r((3, 2), 2)), parent_support_pos: None });
    let leaf1 = plan.add_node(Node { id: NodeId::Leaf(1), ntype: NType::Leaf, pos: Some((1, 1)), robot: Some(r((1, 1), 2)), parent_support_pos: None });
    plan.add_edge(goal, sg, false, Some(1));
    plan.add_edge(sg, bn, false, Some(0));
    plan.add_edge(sg, sp, false, Some(0));
    plan.add_edge(bn, leaf0, true, Some(2));
    plan.add_edge(sp, leaf1, true, Some(3));
    plan.nc = 2;

    assert_eq!(plan.first_open_edge(), Some((bn, leaf0)));
    let seg = segment(&plan, &state, bn, leaf0);
    // Mover is the bottleneck's robot (Blue at its planned cell).
    assert_eq!(seg.mover, r((2, 2), 1));
    // Helpers: Blue dropped BY COLOR (though positions differ), Green kept,
    // target robot Red appended at the END.
    assert_eq!(seg.helpers, vec![r((1, 1), 2), r((4, 4), 0)]);
    // Pinned support = the sibling support node's robot.
    assert_eq!(seg.support, Some(r((3, 2), 2)));
    assert_eq!(seg.fix_support, Some((3, 2)));
    assert_eq!((seg.start, seg.end), ((4, 4), (2, 2)));
}

// ---------------------------------------------------------------------------
// fixes (b) and (c): _apply invariants
// ---------------------------------------------------------------------------

/// Initial-plan setting shared by the _apply tests: Red must reach (2,2)
/// from (0,2) — impossible without a helper, so the goal→leaf edge is open.
fn apply_setting(env: &SubgoalEnv) -> (State, PartialPlan) {
    let state = State {
        target: (2, 2),
        target_robot: r((0, 2), 0),
        helpers: vec![r((3, 3), 1), r((1, 1), 2)],
    };
    let plan = initial_plan(env, &state);
    assert_eq!(plan.first_open_edge(), Some((0, 1)));
    (state, plan)
}

/// A candidate whose bottleneck (0,2) reaches the goal (2,2) via the
/// candidate's own support (3,2) — parent_cost = 0 + 1.
fn cand_own_support(helper: RobotAt) -> Candidate {
    Candidate {
        subgoal: Subgoal {
            bottleneck: r((0, 2), 0),
            support: RobotAt { pos: (3, 2), color: helper.color },
            goal_pos: (2, 2),
            target_robot: r((0, 2), 0),
            helper,
        },
        parent_support: Some((3, 2)),
        score: 0,
    }
}

#[test]
fn fix_b_helper_color_owning_a_leaf_is_rejected() {
    let b = board5();
    let env = SubgoalEnv::new(&b);
    let (state, plan) = apply_setting(&env);
    let (parent, child) = plan.first_open_edge().unwrap();
    let seg = segment(&plan, &state, parent, child);

    // Baseline: Blue helper (owns no leaf) is accepted.
    assert!(apply(&env, &plan, parent, child, &seg, &cand_own_support(r((3, 3), 1))).is_some());

    // Add a leaf owned by GREEN (at any cell): a Green-helper candidate must
    // now be rejected — one robot, one leaf identity (compared BY COLOR).
    let mut plan2 = plan.clone();
    plan2.add_node(Node {
        id: NodeId::Leaf(9),
        ntype: NType::Leaf,
        pos: Some((1, 1)),
        robot: Some(r((1, 1), 2)),
        parent_support_pos: None,
    });
    assert!(apply(&env, &plan2, parent, child, &seg, &cand_own_support(r((1, 1), 2))).is_none());
    // ... while a Blue-helper candidate still passes on the same plan.
    assert!(apply(&env, &plan2, parent, child, &seg, &cand_own_support(r((3, 3), 1))).is_some());
}

#[test]
fn fix_c_parent_support_must_be_own_or_existing_support_cell() {
    let b = board5();
    let env = SubgoalEnv::new(&b);
    let (state, plan) = apply_setting(&env);
    let (parent, child) = plan.first_open_edge().unwrap();
    let seg = segment(&plan, &state, parent, child);

    // parent_support == the candidate's own support cell -> accepted.
    assert!(apply(&env, &plan, parent, child, &seg, &cand_own_support(r((3, 3), 1))).is_some());

    // parent_support (2,3) is a REAL dependent support of the goal, and the
    // bottleneck (2,0) reaches (2,2) through it — but no plan node ever
    // stands there and it is not the candidate's own support -> REJECTED.
    let cand_foreign_ps = Candidate {
        subgoal: Subgoal {
            bottleneck: r((2, 0), 0),
            support: RobotAt { pos: (3, 2), color: 1 },
            goal_pos: (2, 2),
            target_robot: r((0, 2), 0),
            helper: r((3, 3), 1),
        },
        parent_support: Some((2, 3)),
        score: 0,
    };
    assert_eq!(env.compute_exact((2, 0), (2, 2), Some((2, 3))), Some(1));
    assert!(apply(&env, &plan, parent, child, &seg, &cand_foreign_ps).is_none());

    // Same candidate, but the plan already carries a support node at (2,3)
    // -> fix (c) is satisfied and the candidate applies.
    let mut plan2 = plan.clone();
    plan2.add_node(Node {
        id: NodeId::Sp(9),
        ntype: NType::Support,
        pos: Some((2, 3)),
        robot: Some(r((2, 3), 2)),
        parent_support_pos: None,
    });
    assert!(apply(&env, &plan2, parent, child, &seg, &cand_foreign_ps).is_some());
}

// ---------------------------------------------------------------------------
// _apply node/edge shape and the deterministic budget
// ---------------------------------------------------------------------------

#[test]
fn apply_builds_python_node_and_edge_order() {
    let b = board5();
    let env = SubgoalEnv::new(&b);
    let (state, plan) = apply_setting(&env);
    let (parent, child) = plan.first_open_edge().unwrap();
    let seg = segment(&plan, &state, parent, child);
    let out = apply(&env, &plan, parent, child, &seg, &cand_own_support(r((3, 3), 1))).unwrap();

    // Nodes appended in Python order: sg_1, bn_1, sp_1, leaf_1.
    let ids: Vec<NodeId> = out.nodes.iter().map(|n| n.id).collect();
    assert_eq!(
        ids,
        vec![NodeId::Goal, NodeId::Leaf(0), NodeId::Sg(1), NodeId::Bn(1), NodeId::Sp(1), NodeId::Leaf(1)]
    );
    assert_eq!(out.nc, 2);
    // Edge creation order: goal->sg fixed(parent_cost=1), sg->bn 0, sg->sp 0,
    // bn->leaf_0 (mover move), sp->leaf_1 (helper move).
    let shape: Vec<(u32, u32, bool, Option<i64>)> =
        out.edges.iter().map(|e: &EdgeRec| (e.u, e.v, e.open, e.cost)).collect();
    // mover (0,2) -> bn (0,2) VIA SUPPORT (3,2): the supported computation
    // goes through the dependent-edge group ((0,2),(3,2)), which does not
    // exist, so exact AND relaxed are None even though start == end — the
    // edge opens at INF (faithful Python oddity: compute_*_shortest_path_
    // length with a support never consults the plain tables).
    // helper (3,3) -> sp (3,2): slide up passes through to (3,0) -> no
    // exact; relaxed via dependent edge = 2 -> open.
    assert_eq!(
        shape,
        vec![
            (0, 2, false, Some(1)),           // goal -> sg_1
            (2, 3, false, Some(0)),           // sg_1 -> bn_1
            (2, 4, false, Some(0)),           // sg_1 -> sp_1
            (3, 1, true, Some(SUBGOAL_INF)),  // bn_1 -> leaf_0 (no group)
            (4, 5, true, Some(2)),            // sp_1 -> leaf_1 (relaxed)
        ]
    );
}

/// 5×5 board with a wall stop at (3,2): 'E' on (3,2) makes rightward row-2
/// slides stop there (so (3,2) is a wall node and can anchor a support), and
/// 'S' on (3,2) lets a robot sliding down column 3 stop AT (3,2).
fn board5_wallstop() -> CompiledBoard {
    let g: Vec<String> = [
        "NW", "N", "N", "N", "NE", //
        "W", "", "", "", "E", //
        "W", "", "", "ES", "EW", //
        "W", "", "", "N", "E", //
        "SW", "S", "S", "S", "SE",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    CompiledBoard::compile(5, &g, 2)
}

#[test]
fn rollout_budget_exhaustion_is_deterministic() {
    let b = board5_wallstop();
    let env = SubgoalEnv::new(&b);
    // Red must reach (2,2), which needs a helper standing on (3,2); Blue at
    // (3,1) can slide straight down onto it. The first decision therefore
    // has labelable candidates and commit-and-solve runs.
    let state = State {
        target: (2, 2),
        target_robot: r((0, 2), 0),
        helpers: vec![r((3, 1), 1), r((1, 1), 2)],
    };
    let solver = AStar { max_iters: 4000, max_frontier: 40_000, ..AStar::default() };
    // Zero budget: the first commit-and-solve overruns it.
    let res = rollout(&env, &state, &solver, 0, Some(14), Some(0));
    assert_eq!(res.status, RolloutStatus::BudgetExhausted);
    // Unlimited budget on the same instance completes with records.
    let res2 = rollout(&env, &state, &solver, 0, Some(14), None);
    assert_eq!(res2.status, RolloutStatus::Ok);
    assert!(!res2.records.is_empty());
    // Iteration accounting is stable run to run (determinism smoke).
    let res3 = rollout(&env, &state, &solver, 0, Some(14), None);
    assert_eq!(res2.iters_used, res3.iters_used);
    let recs2: Vec<String> = res2.records.iter().map(|r| serde_json::to_string(r).unwrap()).collect();
    let recs3: Vec<String> = res3.records.iter().map(|r| serde_json::to_string(r).unwrap()).collect();
    assert_eq!(recs2, recs3);
}

#[test]
fn node_id_parse_roundtrip() {
    for s in ["goal", "sg_3", "bn_12", "sp_0", "leaf_7"] {
        assert_eq!(NodeId::parse(s).unwrap().to_string_id(), s);
    }
    assert!(NodeId::parse("frob_1").is_none());
    assert!(NodeId::parse("leaf_x").is_none());
}
