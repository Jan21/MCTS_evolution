//! Commit-and-solve labeling — port of `nn/generate.py::rollout` (record
//! extraction incl. `_context`/`_xy`/`_fixed_g`) and the
//! `replay_backward_decision` executor (DESIGN §3).
//!
//! Budget: Python caps rollouts with a 120 s SIGALRM; the deterministic
//! replacement (DESIGN §3) counts loop iterations ENTERED by every
//! `solve_plan` search inside one rollout. The check runs immediately after
//! each `solve_plan` returns; exceeding the budget aborts the rollout with
//! `BudgetExhausted` (records gathered so far are returned but the caller
//! must not emit them as a completed rollout).

use anyhow::{anyhow, bail, Context as _};
use serde::Serialize;
use serde_json::Value;

use crate::types::Cell;

use super::astar::{apply, initial_plan, reference_helpers, segment, AStar};
use super::grid_env::SubgoalEnv;
use super::plan::{EdgeRec, NType, Node, NodeId, PartialPlan};
use super::{color_slot, Candidate, RobotAt, State, Subgoal};

/// `[[x, y], "Color"]` in record JSON.
pub type RobotJson = ((u16, u16), String);

fn robot_json(r: &RobotAt) -> RobotJson {
    (r.pos, r.color_str().to_string())
}

/// One `combined.jsonl` record — 18 fields, declaration order matches
/// `nn/generate.py` write order exactly (DESIGN §4).
#[derive(Clone, Debug, Serialize)]
pub struct BackwardRecord {
    pub env_id: i64,
    pub target: Cell,
    pub target_robot: RobotJson,
    pub helpers: Vec<RobotJson>,
    pub seg_start: Cell,
    pub seg_end: Cell,
    pub seg_support: Option<Cell>,
    pub mover_color: String,
    pub ctx_bottlenecks: Vec<Cell>,
    pub ctx_supports: Vec<Cell>,
    pub ctx_open_endpoints: Vec<Option<Cell>>,
    pub cand_bottleneck: Cell,
    pub cand_support: Cell,
    pub cand_helper: RobotJson,
    pub cand_parent_support: Option<Cell>,
    pub cost_to_go: i64,
    pub is_optimal: bool,
    pub depth: i64,
}

/// `nn.generate._context`: committed bottleneck / support cells (node
/// insertion order) and open-edge endpoint positions (NetworkX edge order;
/// a subgoal endpoint would serialize as null, like Python's `_xy(None)`).
fn context(plan: &PartialPlan) -> (Vec<Cell>, Vec<Cell>, Vec<Option<Cell>>) {
    let bn = plan
        .nodes
        .iter()
        .filter(|n| n.ntype == NType::Bottleneck)
        .map(|n| n.pos.expect("bottleneck without pos"))
        .collect();
    let sp = plan
        .nodes
        .iter()
        .filter(|n| n.ntype == NType::Support)
        .map(|n| n.pos.expect("support without pos"))
        .collect();
    let mut open_eps = Vec::new();
    for e in plan.edges_nx() {
        if e.open {
            open_eps.push(plan.nodes[e.u as usize].pos);
            open_eps.push(plan.nodes[e.v as usize].pos);
        }
    }
    (bn, sp, open_eps)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RolloutStatus {
    /// Records produced (Python: instance kept).
    Ok,
    /// No records (Python: `if not recs: continue` — instance dropped).
    Empty,
    /// Deterministic iteration budget exhausted mid-rollout.
    BudgetExhausted,
}

pub struct RolloutResult {
    pub status: RolloutStatus,
    pub records: Vec<BackwardRecord>,
    /// Total `solve_plan` iterations consumed (budget accounting).
    pub iters_used: u64,
}

/// Audit trace of one decision (verification only — lets the fixture harness
/// prove that a Python-vs-Rust record difference is the documented
/// truncation-tie class rather than a porting bug). Not part of any output.
pub struct DecisionTrace {
    pub depth: i64,
    /// Full stable-sorted candidate list BEFORE max_candidates truncation.
    pub all_candidates: Vec<Candidate>,
}

/// `nn.generate.rollout` — one optimal trajectory; label every candidate at
/// each decision; advance to the FIRST labeled candidate (in candidate
/// order) whose ctg equals the minimum. Forced exact fixes advance without
/// records and do not increment `depth`.
pub fn rollout(
    env: &SubgoalEnv,
    state: &State,
    solver: &AStar,
    env_id: i64,
    max_candidates: Option<usize>,
    budget: Option<u64>,
) -> RolloutResult {
    rollout_traced(env, state, solver, env_id, max_candidates, budget, None)
}

/// [`rollout`] with an optional per-decision audit trace.
pub fn rollout_traced(
    env: &SubgoalEnv,
    state: &State,
    solver: &AStar,
    env_id: i64,
    max_candidates: Option<usize>,
    budget: Option<u64>,
    mut trace: Option<&mut Vec<DecisionTrace>>,
) -> RolloutResult {
    let mut records = Vec::new();
    let mut plan = initial_plan(env, state);
    let mut depth: i64 = 0;
    let mut used: u64 = 0;

    while !plan.is_complete() {
        let (parent, child) = plan.first_open_edge().expect("incomplete plan has an open edge");
        let mut seg = segment(&plan, state, parent, child);

        // No decision when the segment pins to an exact path; just fix it.
        if env
            .compute_exact(seg.start, seg.end, seg.fix_support)
            .is_some()
        {
            plan = solver.expand(env, state, &plan).swap_remove(0);
            continue;
        }

        // Lever B2: mirror `AStar._expand` / the extended `nn.generate.rollout`
        // — reference helpers join the segment's helper list before proposing.
        if solver.by_reference {
            seg.helpers.extend(reference_helpers(&plan, seg.mover.color));
        }
        let mut cands = env.propose(seg.end, &seg.mover, &seg.helpers, seg.support.as_ref(), solver.b1);
        if let Some(mc) = max_candidates {
            // Python sorts (stably) and truncates ONLY when max_candidates
            // is given; otherwise the unsorted propose order is used.
            cands.sort_by_key(|c| c.score);
            if let Some(t) = trace.as_deref_mut() {
                t.push(DecisionTrace { depth, all_candidates: cands.clone() });
            }
            cands.truncate(mc);
        } else if let Some(t) = trace.as_deref_mut() {
            t.push(DecisionTrace { depth, all_candidates: cands.clone() });
        }

        let fixed_g = plan.fixed_g();
        let (bn_ctx, sp_ctx, open_eps) = context(&plan);
        let mut labeled: Vec<(Candidate, PartialPlan, i64)> = Vec::new();
        for cand in &cands {
            let Some(child_plan) =
                apply(env, &plan, parent, child, &seg, cand, solver.by_reference)
            else {
                continue;
            };
            let out = solver.solve_plan(env, state, child_plan.clone());
            used += out.iters;
            if let Some(b) = budget {
                if used > b {
                    return RolloutResult {
                        status: RolloutStatus::BudgetExhausted,
                        records,
                        iters_used: used,
                    };
                }
            }
            let Some(done) = out.plan else { continue };
            let ctg = done.cost() - fixed_g;
            labeled.push((*cand, child_plan, ctg));
        }

        if labeled.is_empty() {
            break;
        }
        let best_ctg = labeled.iter().map(|(_, _, c)| *c).min().unwrap();
        for (cand, _, ctg) in &labeled {
            records.push(BackwardRecord {
                env_id,
                target: state.target,
                target_robot: robot_json(&state.target_robot),
                helpers: state.helpers.iter().map(robot_json).collect(),
                seg_start: seg.start,
                seg_end: seg.end,
                seg_support: seg.fix_support,
                mover_color: seg.mover.color_str().to_string(),
                ctx_bottlenecks: bn_ctx.clone(),
                ctx_supports: sp_ctx.clone(),
                ctx_open_endpoints: open_eps.clone(),
                cand_bottleneck: cand.subgoal.bottleneck.pos,
                cand_support: cand.subgoal.support.pos,
                cand_helper: robot_json(&cand.subgoal.helper),
                cand_parent_support: cand.parent_support,
                cost_to_go: *ctg,
                is_optimal: *ctg == best_ctg,
                depth,
            });
        }
        // advance along the optimal trajectory (first labeled at best ctg)
        let pos = labeled
            .iter()
            .position(|(_, _, c)| *c == best_ctg)
            .unwrap();
        plan = labeled.swap_remove(pos).1;
        depth += 1;
    }

    let status = if records.is_empty() {
        RolloutStatus::Empty
    } else {
        RolloutStatus::Ok
    };
    RolloutResult { status, records, iters_used: used }
}

// ---------------------------------------------------------------------------
// replay_backward_decision (DESIGN §3)
// ---------------------------------------------------------------------------

/// One dumped candidate (`pyref/common.ser_candidate` shape). Colors are
/// structural: the bottleneck always carries the segment mover's color and
/// the support the helper's color, so the dump fields are lossless.
#[derive(Clone, Debug)]
pub struct ReplayCandidate {
    pub bottleneck: Cell,
    pub support: Cell,
    pub helper: RobotAt,
    pub parent_support: Option<Cell>,
}

/// A parsed `replay_backward_decision` work item (board handled separately).
/// `b1` / `by_reference` come from the item's optional `vocab` field
/// (default base — both false, behavior unchanged).
pub struct ReplayDecision {
    pub state: State,
    pub plan: PartialPlan,
    pub open_edge: (NodeId, NodeId),
    pub candidates: Vec<ReplayCandidate>,
    pub max_iters: u64,
    pub max_frontier: usize,
    pub b1: bool,
    pub by_reference: bool,
}

/// Per-candidate replay label: `rejected` = `_apply` refused it; `ctg: None`
/// with `rejected: false` = `solve_plan` found no completion within caps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct CandidateLabel {
    pub ctg: Option<i64>,
    pub rejected: bool,
}

/// Re-run `_apply` + `solve_plan` for every dumped candidate.
/// `parent_support` is taken FROM THE DUMP, never re-resolved, which makes
/// replay immune to the documented parent-support order divergence.
pub fn replay_decision(env: &SubgoalEnv, item: &ReplayDecision) -> anyhow::Result<Vec<CandidateLabel>> {
    let (parent, child) = item
        .plan
        .first_open_edge()
        .ok_or_else(|| anyhow!("replay plan has no open edge"))?;
    let (pu, pv) = (
        item.plan.nodes[parent as usize].id,
        item.plan.nodes[child as usize].id,
    );
    if (pu, pv) != item.open_edge {
        bail!(
            "open_edge mismatch: dump=({:?},{:?}) reconstructed=({:?},{:?})",
            item.open_edge.0, item.open_edge.1, pu, pv
        );
    }

    let solver = AStar {
        max_iters: item.max_iters,
        max_frontier: item.max_frontier,
        b1: item.b1,
        by_reference: item.by_reference,
        ..AStar::default()
    };
    let seg = segment(&item.plan, &item.state, parent, child);
    let fixed_g = item.plan.fixed_g();

    let mut labels = Vec::with_capacity(item.candidates.len());
    for c in &item.candidates {
        let cand = Candidate {
            subgoal: Subgoal {
                bottleneck: RobotAt { pos: c.bottleneck, color: seg.mover.color },
                support: RobotAt { pos: c.support, color: c.helper.color },
                goal_pos: seg.end,
                target_robot: seg.mover,
                helper: c.helper,
            },
            parent_support: c.parent_support,
            score: 0,
        };
        let Some(child_plan) =
            apply(env, &item.plan, parent, child, &seg, &cand, item.by_reference)
        else {
            labels.push(CandidateLabel { ctg: None, rejected: true });
            continue;
        };
        match solver.solve_plan(env, &item.state, child_plan).plan {
            None => labels.push(CandidateLabel { ctg: None, rejected: false }),
            Some(done) => labels.push(CandidateLabel {
                ctg: Some(done.cost() - fixed_g),
                rejected: false,
            }),
        }
    }
    Ok(labels)
}

// ---------------------------------------------------------------------------
// JSON parsing of the DESIGN §3 replay/state shapes (shared with Agent D)
// ---------------------------------------------------------------------------

pub fn parse_cell(v: &Value) -> anyhow::Result<Cell> {
    let a = v.as_array().filter(|a| a.len() == 2).ok_or_else(|| anyhow!("bad cell {v}"))?;
    Ok((
        a[0].as_u64().ok_or_else(|| anyhow!("bad cell x {v}"))? as u16,
        a[1].as_u64().ok_or_else(|| anyhow!("bad cell y {v}"))? as u16,
    ))
}

pub fn parse_cell_opt(v: &Value) -> anyhow::Result<Option<Cell>> {
    if v.is_null() {
        Ok(None)
    } else {
        Ok(Some(parse_cell(v)?))
    }
}

/// `[[x, y], "Color"]` → RobotAt (palette colors only; fail loud otherwise).
pub fn parse_robot(v: &Value) -> anyhow::Result<RobotAt> {
    let a = v.as_array().filter(|a| a.len() == 2).ok_or_else(|| anyhow!("bad robot {v}"))?;
    let pos = parse_cell(&a[0])?;
    let color_str = a[1].as_str().ok_or_else(|| anyhow!("bad robot color {v}"))?;
    let color = color_slot(color_str).ok_or_else(|| anyhow!("non-palette color {color_str:?}"))?;
    Ok(RobotAt { pos, color })
}

/// DESIGN §3 `state` object.
pub fn parse_state(v: &Value) -> anyhow::Result<State> {
    Ok(State {
        target: parse_cell(&v["target"]).context("state.target")?,
        target_robot: parse_robot(&v["target_robot"]).context("state.target_robot")?,
        helpers: v["helpers"]
            .as_array()
            .ok_or_else(|| anyhow!("state.helpers not a list"))?
            .iter()
            .map(parse_robot)
            .collect::<anyhow::Result<_>>()?,
    })
}

/// DESIGN §3 `plan` object — nodes and edges in DUMP order (Python insertion
/// order), which fixes `open_edges()[0]` and all downstream iteration.
pub fn parse_plan(v: &Value) -> anyhow::Result<PartialPlan> {
    let mut plan = PartialPlan::new();
    for nv in v["nodes"].as_array().ok_or_else(|| anyhow!("plan.nodes not a list"))? {
        let row = nv.as_array().filter(|a| a.len() == 3).ok_or_else(|| anyhow!("bad node {nv}"))?;
        let id_str = row[0].as_str().ok_or_else(|| anyhow!("bad node id {nv}"))?;
        let id = NodeId::parse(id_str).ok_or_else(|| anyhow!("unknown node id {id_str:?}"))?;
        let ntype_str = row[1].as_str().ok_or_else(|| anyhow!("bad ntype {nv}"))?;
        let attrs = &row[2];
        let node = match ntype_str {
            "goal" => Node {
                id,
                ntype: NType::Goal,
                pos: Some(parse_cell(&attrs["pos"])?),
                robot: None,
                parent_support_pos: None,
            },
            "subgoal" => Node {
                id,
                ntype: NType::Subgoal,
                pos: None,
                robot: None,
                parent_support_pos: parse_cell_opt(&attrs["parent_support_pos"])?,
            },
            "bottleneck" | "support" | "leaf" => Node {
                id,
                ntype: match ntype_str {
                    "bottleneck" => NType::Bottleneck,
                    "support" => NType::Support,
                    _ => NType::Leaf,
                },
                pos: Some(parse_cell(&attrs["pos"])?),
                robot: Some(parse_robot(&attrs["robot"])?),
                parent_support_pos: None,
            },
            other => bail!("unknown ntype {other:?}"),
        };
        plan.add_node(node);
    }
    for ev in v["edges"].as_array().ok_or_else(|| anyhow!("plan.edges not a list"))? {
        // 4-element rows are the original dump shape; a 5th element `true`
        // marks a Lever B2 `byref` edge (emitted by `ser_plan` only for b2
        // plans, so old corpora parse unchanged).
        let row = ev
            .as_array()
            .filter(|a| a.len() == 4 || a.len() == 5)
            .ok_or_else(|| anyhow!("bad edge {ev}"))?;
        let uid = NodeId::parse(row[0].as_str().ok_or_else(|| anyhow!("bad edge u {ev}"))?)
            .ok_or_else(|| anyhow!("unknown edge u id {ev}"))?;
        let vid = NodeId::parse(row[1].as_str().ok_or_else(|| anyhow!("bad edge v {ev}"))?)
            .ok_or_else(|| anyhow!("unknown edge v id {ev}"))?;
        let u = plan.node_index(uid).ok_or_else(|| anyhow!("edge u not a node {ev}"))?;
        let vv = plan.node_index(vid).ok_or_else(|| anyhow!("edge v not a node {ev}"))?;
        let status = row[2].as_str().ok_or_else(|| anyhow!("bad edge status {ev}"))?;
        let open = match status {
            "open" => true,
            "fixed" => false,
            other => bail!("unknown edge status {other:?}"),
        };
        let cost = if row[3].is_null() {
            None
        } else {
            Some(row[3].as_i64().ok_or_else(|| anyhow!("bad edge cost {ev}"))?)
        };
        let byref = if row.len() == 5 {
            row[4].as_bool().ok_or_else(|| anyhow!("bad edge byref {ev}"))?
        } else {
            false
        };
        plan.edges.push(EdgeRec { u, v: vv, open, cost, byref });
    }
    plan.nc = v["nc"].as_u64().ok_or_else(|| anyhow!("plan.nc missing"))? as u32;
    Ok(plan)
}

/// Parse a full `replay_backward_decision` line (minus the board, which the
/// caller compiles/caches separately). The optional `vocab` field selects
/// the plan-language levers (absent → base, both flags false).
pub fn parse_replay_decision(v: &Value) -> anyhow::Result<ReplayDecision> {
    let (b1, by_reference) = super::astar::vocab_flags(v.get("vocab").and_then(|x| x.as_str()))?;
    let oe = v["open_edge"].as_array().filter(|a| a.len() == 2).ok_or_else(|| anyhow!("bad open_edge"))?;
    let ou = NodeId::parse(oe[0].as_str().ok_or_else(|| anyhow!("bad open_edge u"))?)
        .ok_or_else(|| anyhow!("bad open_edge u id"))?;
    let ov = NodeId::parse(oe[1].as_str().ok_or_else(|| anyhow!("bad open_edge v"))?)
        .ok_or_else(|| anyhow!("bad open_edge v id"))?;
    let candidates = v["candidates"]
        .as_array()
        .ok_or_else(|| anyhow!("candidates not a list"))?
        .iter()
        .map(|c| -> anyhow::Result<ReplayCandidate> {
            Ok(ReplayCandidate {
                bottleneck: parse_cell(&c["bottleneck"])?,
                support: parse_cell(&c["support"])?,
                helper: parse_robot(&c["helper"])?,
                parent_support: parse_cell_opt(&c["parent_support"])?,
            })
        })
        .collect::<anyhow::Result<_>>()?;
    Ok(ReplayDecision {
        state: parse_state(&v["state"])?,
        plan: parse_plan(&v["plan"])?,
        open_edge: (ou, ov),
        candidates,
        max_iters: v["max_iters"].as_u64().ok_or_else(|| anyhow!("max_iters missing"))?,
        max_frontier: v["max_frontier"].as_u64().ok_or_else(|| anyhow!("max_frontier missing"))?
            as usize,
        b1,
        by_reference,
    })
}
