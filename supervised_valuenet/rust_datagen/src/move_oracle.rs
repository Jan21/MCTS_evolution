//! Forward move oracle — exact port of `move_planner/oracle.py` plus the
//! record/driver semantics of `move_planner/generate.py::label_board/_rec`
//! (DESIGN §3 forward_instance / replay_forward_state, §4 forward record,
//! §5.16–5.18).
//!
//! Parity notes (the Python line is the spec — see oracle.py):
//! - `relaxed_target_dist`: reverse BFS from the goal; from each popped cell,
//!   walk rays outward one step at a time (`_can_step` physics); first-seen
//!   wins; FIFO queue.
//! - `solve`: A* over joint robot states with priority tuple `(f, g, counter)`
//!   where the counter increments per push, so among equal `(f, g)` entries
//!   the pop order is FIFO by push order — exactly Python `heapq`. The
//!   counter makes the order total, so Rust's `BinaryHeap` (max-heap wrapped
//!   in `Reverse`) pops the identical sequence. Stale entries are skipped via
//!   `gc != g[cur]`; the goal test happens on POP; `expansions` increments
//!   AFTER the goal test and aborts when `expansions > max_expansions`;
//!   a child is accepted only when `ng < g.get(child)` and then pruned when
//!   `h >= INF || ng + h > cost_cap`; successor order = robot slot-major,
//!   then `Dir` order (physics::successors). With all of this mirrored, even
//!   CAPPED behaviour (expansion cap, cost cap) is bit-identical to Python.
//!
//! Determinism (DESIGN §8): the `g`/`came` maps are hash maps keyed by the
//! packed joint state, used ONLY for keyed lookups — nothing ever iterates
//! them, so their ordering cannot leak into any output. All output orders
//! come from the heap (total order) and the successor enumeration (fixed).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::physics::{apply_move, slide, successors, Walls};
use crate::types::{Cell, Dir, DIRECTIONS, ORACLE_INF};

// ---------------------------------------------------------------------------
// joint-state packing (private): cell index in 12 bits, robot slots side by
// side in a u128. Invertible given (n, robot count); supports n <= 64, R <= 10.
// ---------------------------------------------------------------------------

const CELL_BITS: u32 = 12;
const CELL_MASK: u128 = (1 << CELL_BITS) - 1;

#[inline]
fn pack(positions: &[Cell], n: u16) -> u128 {
    // Hard asserts (release too): every search enters through pack() exactly
    // once (patch() only rewrites pack-built keys), so an out-of-envelope
    // work item fails loudly instead of silently corrupting packed keys
    // (DESIGN §7 fail-loud). Cost: one call per solve, not per node.
    assert!(n <= 64, "state packing supports n <= 64, got n = {n}");
    assert!(
        positions.len() as u32 * CELL_BITS <= 128,
        "state packing supports at most {} robots, got {}",
        128 / CELL_BITS,
        positions.len()
    );
    let mut key = 0u128;
    for (i, &(x, y)) in positions.iter().enumerate() {
        key |= ((y as u128 * n as u128) + x as u128) << (CELL_BITS * i as u32);
    }
    key
}

#[inline]
fn unpack(key: u128, r: usize, n: u16, out: &mut Vec<Cell>) {
    out.clear();
    for i in 0..r {
        let idx = ((key >> (CELL_BITS * i as u32)) & CELL_MASK) as u32;
        out.push(((idx % n as u32) as u16, (idx / n as u32) as u16));
    }
}

/// Replace slot `i`'s cell in a packed key.
#[inline]
fn patch(key: u128, i: usize, cell: Cell, n: u16) -> u128 {
    let shift = CELL_BITS * i as u32;
    (key & !(CELL_MASK << shift)) | (((cell.1 as u128 * n as u128) + cell.0 as u128) << shift)
}

// ---------------------------------------------------------------------------
// relaxed_target_dist (oracle.py::relaxed_target_dist)
// ---------------------------------------------------------------------------

/// One-cell step with wall/edge check — port of `oracle.py::_can_step`
/// (same physics as one iteration of `simulate.slide`).
#[inline]
fn can_step(x: u16, y: u16, d: Dir, walls: &Walls) -> Option<Cell> {
    let n = walls.n();
    match d {
        Dir::Up => (y > 0 && !walls.has_down(x, y - 1)).then(|| (x, y - 1)),
        Dir::Down => (y < n - 1 && !walls.has_down(x, y)).then(|| (x, y + 1)),
        Dir::Left => (x > 0 && !walls.has_right(x - 1, y)).then(|| (x - 1, y)),
        Dir::Right => (x < n - 1 && !walls.has_right(x, y)).then(|| (x + 1, y)),
    }
}

/// Admissible heuristic field: min slides for the target robot ALONE to reach
/// `target` assuming a blocker at every cell. Reverse BFS from the goal, rays
/// walked outward from each popped cell; first-seen wins; FIFO.
/// `result[y*n + x]` = distance of cell (x, y), `None` = unreachable.
pub fn relaxed_target_dist(target: Cell, walls: &Walls) -> Vec<Option<i64>> {
    let n = walls.n();
    let nu = n as usize;
    let mut dist: Vec<Option<i64>> = vec![None; nu * nu];
    dist[target.1 as usize * nu + target.0 as usize] = Some(0);
    let mut q: VecDeque<Cell> = VecDeque::new();
    q.push_back(target);
    while let Some(v) = q.pop_front() {
        let dv = dist[v.1 as usize * nu + v.0 as usize].unwrap();
        for &d in DIRECTIONS.iter() {
            let (mut x, mut y) = v;
            while let Some((nx, ny)) = can_step(x, y, d, walls) {
                x = nx;
                y = ny;
                let i = y as usize * nu + x as usize;
                if dist[i].is_none() {
                    dist[i] = Some(dv + 1);
                    q.push_back((x, y));
                }
            }
        }
    }
    dist
}

// ---------------------------------------------------------------------------
// solve (oracle.py::solve)
// ---------------------------------------------------------------------------

/// A move: `(robot_slot, dir)`.
pub type Action = (usize, Dir);

/// One entry of a reconstructed optimal path: the joint state and the action
/// `(robot_slot, dir)` that PRODUCED it (`None` for the start entry).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathEntry {
    pub positions: Vec<Cell>,
    pub action: Option<Action>,
}

/// A* optimal move count (no path). `None` = unsolvable, cost-capped away, or
/// `max_expansions` exceeded — exactly oracle.py::solve's `None`.
/// Pass `cost_cap = ORACLE_INF` for the Python default.
pub fn solve(
    positions: &[Cell],
    target_idx: usize,
    target: Cell,
    walls: &Walls,
    hdist: &[Option<i64>],
    max_expansions: i64,
    cost_cap: i64,
) -> Option<i64> {
    solve_inner(positions, target_idx, target, walls, hdist, max_expansions, cost_cap, false)
        .map(|(cost, _)| cost)
}

/// A* with path reconstruction (oracle.py::solve with `want_path=True`).
/// The path runs start → goal; the goal entry carries the final action, the
/// start entry `action = None`.
pub fn solve_with_path(
    positions: &[Cell],
    target_idx: usize,
    target: Cell,
    walls: &Walls,
    hdist: &[Option<i64>],
    max_expansions: i64,
    cost_cap: i64,
) -> Option<(i64, Vec<PathEntry>)> {
    solve_inner(positions, target_idx, target, walls, hdist, max_expansions, cost_cap, true)
        .map(|(cost, path)| (cost, path.expect("want_path")))
}

#[allow(clippy::too_many_arguments)]
fn solve_inner(
    positions: &[Cell],
    target_idx: usize,
    target: Cell,
    walls: &Walls,
    hdist: &[Option<i64>],
    max_expansions: i64,
    cost_cap: i64,
    want_path: bool,
) -> Option<(i64, Option<Vec<PathEntry>>)> {
    let n = walls.n();
    let nu = n as usize;
    let r = positions.len();
    let h_of = |c: Cell| hdist[c.1 as usize * nu + c.0 as usize].unwrap_or(ORACLE_INF);

    let h0 = h_of(positions[target_idx]);
    if h0 >= ORACLE_INF || h0 > cost_cap {
        return None;
    }
    if positions[target_idx] == target {
        let path = want_path.then(|| {
            vec![PathEntry { positions: positions.to_vec(), action: None }]
        });
        return Some((0, path));
    }

    let start = pack(positions, n);
    let mut cnt: u64 = 0;
    // Priority tuple (f, g, push-counter, state-key): the counter is unique
    // per push, so the order is total and the trailing key never compares —
    // identical pop sequence to Python heapq on (f, g, cnt, positions).
    let mut pq: BinaryHeap<Reverse<(i64, i64, u64, u128)>> = BinaryHeap::new();
    pq.push(Reverse((h0, 0, cnt, start)));
    let mut g: FxHashMap<u128, i64> = FxHashMap::default();
    g.insert(start, 0);
    // came: child -> (parent, action); only needed for path reconstruction
    // (Python fills it unconditionally, but it never influences the search).
    let mut came: FxHashMap<u128, (Option<u128>, Option<Action>)> = FxHashMap::default();
    if want_path {
        came.insert(start, (None, None));
    }
    let mut expansions: i64 = 0;

    let mut pos_buf: Vec<Cell> = Vec::with_capacity(r);
    let mut blockers: Vec<Cell> = Vec::with_capacity(r.saturating_sub(1));

    while let Some(Reverse((_f, gc, _c, cur))) = pq.pop() {
        if gc != *g.get(&cur).unwrap_or(&ORACLE_INF) {
            continue; // stale
        }
        unpack(cur, r, n, &mut pos_buf);
        if pos_buf[target_idx] == target {
            if !want_path {
                return Some((gc, None));
            }
            let mut path: Vec<PathEntry> = Vec::new();
            let mut s = Some(cur);
            let mut sbuf: Vec<Cell> = Vec::with_capacity(r);
            while let Some(sk) = s {
                let (ps, act) = came[&sk];
                unpack(sk, r, n, &mut sbuf);
                path.push(PathEntry {
                    positions: sbuf.clone(),
                    action: if ps.is_none() { None } else { act },
                });
                s = ps;
            }
            path.reverse();
            return Some((gc, Some(path)));
        }
        expansions += 1;
        if expansions > max_expansions {
            return None;
        }
        let ng = gc + 1;
        // Successor enumeration: robot slot-major, then Dir order — the exact
        // order of physics::successors / state.py::legal_moves, inlined on
        // packed keys to avoid per-child Vec allocations.
        for i in 0..r {
            blockers.clear();
            for (j, &p) in pos_buf.iter().enumerate() {
                if j != i {
                    blockers.push(p);
                }
            }
            for &d in DIRECTIONS.iter() {
                let nxt = slide(pos_buf[i], d, &blockers, walls);
                if nxt == pos_buf[i] {
                    continue; // no-op -> illegal
                }
                let child = patch(cur, i, nxt, n);
                if ng < *g.get(&child).unwrap_or(&ORACLE_INF) {
                    let tcell = if i == target_idx { nxt } else { pos_buf[target_idx] };
                    let h = h_of(tcell);
                    if h >= ORACLE_INF || ng + h > cost_cap {
                        continue;
                    }
                    g.insert(child, ng);
                    cnt += 1;
                    if want_path {
                        came.insert(child, (Some(cur), Some((i, d))));
                    }
                    pq.push(Reverse((ng + h, ng, cnt, child)));
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// label_trajectory (oracle.py::label_trajectory)
// ---------------------------------------------------------------------------

/// One labelled decision state on an optimal rollout (oracle.py record dict).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrajRecord {
    pub positions: Vec<Cell>,
    pub cost_to_go: i64,
    pub best_moves: Vec<Action>,
    pub legal_moves: Vec<Action>,
    pub depth: i64,
    pub full: bool,
}

/// True iff `child` lies on an optimal solution from a state whose exact
/// cost-to-go is `here` (oracle.py::label_trajectory::is_optimal_child).
fn is_optimal_child(
    child: &[Cell],
    target_idx: usize,
    target: Cell,
    walls: &Walls,
    hdist: &[Option<i64>],
    max_expansions: i64,
    here: i64,
) -> bool {
    if child[target_idx] == target {
        return here == 1;
    }
    solve(child, target_idx, target, walls, hdist, max_expansions, here - 1) == Some(here - 1)
}

/// One optimal rollout with per-decision labels: `(d_star, records)`, records
/// only for pre-goal on-path states; `None` when the instance is unsolvable /
/// capped. Mirrors oracle.py::label_trajectory line for line.
#[allow(clippy::too_many_arguments)]
pub fn label_trajectory(
    positions: &[Cell],
    target_idx: usize,
    target: Cell,
    walls: &Walls,
    hdist: &[Option<i64>],
    max_expansions: i64,
    full_policy: bool,
    full_policy_max_ctg: Option<i64>,
) -> Option<(i64, Vec<TrajRecord>)> {
    let (d_star, path) = solve_with_path(
        positions, target_idx, target, walls, hdist, max_expansions, ORACLE_INF,
    )?;
    let mut records: Vec<TrajRecord> = Vec::new();
    for (depth, entry) in path.iter().enumerate() {
        let state = &entry.positions;
        let here = d_star - depth as i64; // exact cost-to-go on the optimal path
        if state[target_idx] == target {
            break; // goal: no decision
        }
        // the move actually taken leaves `state` -> the NEXT entry's action.
        let taken = if depth + 1 < path.len() { path[depth + 1].action } else { None };
        let succ = successors(state, walls);
        let legal: Vec<Action> = succ.iter().map(|&(i, d, _)| (i, d)).collect();
        let do_full = full_policy && full_policy_max_ctg.is_none_or(|cap| here <= cap);
        let best: Vec<Action> = if do_full {
            succ.iter()
                .filter(|(_, _, child)| {
                    is_optimal_child(child, target_idx, target, walls, hdist, max_expansions, here)
                })
                .map(|&(i, d, _)| (i, d))
                .collect()
        } else {
            taken.into_iter().collect()
        };
        records.push(TrajRecord {
            positions: state.clone(),
            cost_to_go: here,
            best_moves: best,
            legal_moves: legal,
            depth: depth as i64,
            full: do_full,
        });
    }
    Some((d_star, records))
}

// ---------------------------------------------------------------------------
// moves.jsonl record (generate.py::_rec — 9 fields, DESIGN §4 order)
// ---------------------------------------------------------------------------

/// One `moves.jsonl` record. Field DECLARATION order is the serialization
/// order and must stay exactly as `generate.py::_rec` writes it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoveRecord {
    pub env_id: i64,
    pub robots: Vec<[i64; 2]>,
    pub target: [i64; 2],
    pub target_idx: usize,
    pub cost_to_go: i64,
    pub best_moves: Vec<[i64; 2]>,
    pub legal_moves: Vec<[i64; 2]>,
    pub depth: i64,
    pub full: bool,
}

fn moves_to_pairs(moves: &[Action]) -> Vec<[i64; 2]> {
    moves.iter().map(|&(s, d)| [s as i64, d as i64]).collect()
}

#[allow(clippy::too_many_arguments)] // mirrors generate.py::_rec's 9 fields
fn make_rec(
    env_id: i64,
    positions: &[Cell],
    target: Cell,
    target_idx: usize,
    ctg: i64,
    best_moves: &[Action],
    legal_moves: &[Action],
    depth: i64,
    full: bool,
) -> MoveRecord {
    MoveRecord {
        env_id,
        robots: positions.iter().map(|&(x, y)| [x as i64, y as i64]).collect(),
        target: [target.0 as i64, target.1 as i64],
        target_idx,
        cost_to_go: ctg,
        best_moves: moves_to_pairs(best_moves),
        legal_moves: moves_to_pairs(legal_moves),
        depth,
        full,
    }
}

// ---------------------------------------------------------------------------
// label_instance (generate.py::label_board per-instance loop, DESIGN §3)
// ---------------------------------------------------------------------------

/// Outcome of labelling one forward instance (DESIGN §3 result statuses).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LabelOutcome {
    /// Target cell unreachable even in the relaxation (consumes an attempt).
    RelaxedUnreachable,
    /// Expansion cap hit / no path.
    Unsolved,
    Solved { d_star: i64, records: Vec<MoveRecord> },
}

/// The full forward_instance work-item semantics: relaxed-unreachable
/// pre-check, `label_trajectory`, and (with `score_candidates`) one
/// value-only record per legal move of every recorded state — each
/// trajectory record first, then its candidate records, exactly the order
/// generate.py::label_board emits them.
#[allow(clippy::too_many_arguments)]
pub fn label_instance(
    env_id: i64,
    positions: &[Cell],
    target_idx: usize,
    target: Cell,
    walls: &Walls,
    max_expansions: i64,
    full_policy: bool,
    full_policy_max_ctg: Option<i64>,
    score_candidates: bool,
) -> LabelOutcome {
    let nu = walls.n() as usize;
    let hdist = relaxed_target_dist(target, walls);
    let start_t = positions[target_idx];
    if hdist[start_t.1 as usize * nu + start_t.0 as usize].unwrap_or(ORACLE_INF) >= ORACLE_INF {
        return LabelOutcome::RelaxedUnreachable;
    }
    let Some((d_star, recs)) = label_trajectory(
        positions, target_idx, target, walls, &hdist, max_expansions,
        full_policy, full_policy_max_ctg,
    ) else {
        return LabelOutcome::Unsolved;
    };
    let mut out: Vec<MoveRecord> = Vec::new();
    for r in &recs {
        out.push(make_rec(
            env_id, &r.positions, target, target_idx, r.cost_to_go,
            &r.best_moves, &r.legal_moves, r.depth, r.full,
        ));
        if !score_candidates {
            continue;
        }
        // score every candidate move by its resulting state's EXACT cost-to-go
        for &(slot, d) in &r.legal_moves {
            let Some(child) = apply_move(&r.positions, slot, d, walls) else {
                continue; // unreachable for a legal move; mirrored anyway
            };
            let c_ctg = if child[target_idx] == target {
                0
            } else {
                match solve(&child, target_idx, target, walls, &hdist, max_expansions, ORACLE_INF)
                {
                    Some(c) => c,
                    None => continue, // child unsolvable within budget -> skip
                }
            };
            let c_legal: Vec<Action> =
                successors(&child, walls).iter().map(|&(i, d2, _)| (i, d2)).collect();
            out.push(make_rec(
                env_id, &child, target, target_idx, c_ctg, &[], &c_legal,
                r.depth + 1, false, // value-only (policy skips it)
            ));
        }
    }
    LabelOutcome::Solved { d_star, records: out }
}

// ---------------------------------------------------------------------------
// replay_forward_state (DESIGN §3 gate-3 task)
// ---------------------------------------------------------------------------

/// Gate-3 replay for one forward state: exact cost-to-go (`None` = null),
/// the FULL optimal-move set (every legal child tested via cost-capped
/// solve), and the legal moves.
pub fn replay_forward_state(
    positions: &[Cell],
    target_idx: usize,
    target: Cell,
    walls: &Walls,
    max_expansions: i64,
) -> (Option<i64>, Vec<Action>, Vec<Action>) {
    let hdist = relaxed_target_dist(target, walls);
    let succ = successors(positions, walls);
    let legal: Vec<Action> = succ.iter().map(|&(i, d, _)| (i, d)).collect();
    let ctg = solve(positions, target_idx, target, walls, &hdist, max_expansions, ORACLE_INF);
    let optimal: Vec<Action> = match ctg {
        Some(here) if here > 0 => succ
            .iter()
            .filter(|(_, _, child)| {
                is_optimal_child(child, target_idx, target, walls, &hdist, max_expansions, here)
            })
            .map(|&(i, d, _)| (i, d))
            .collect(),
        _ => Vec::new(), // unsolved or already on goal: no optimal set
    };
    (ctg, optimal, legal)
}

// ---------------------------------------------------------------------------
// work items (DESIGN §3) — module-local; io.rs wires dispatch in Wave 3
// ---------------------------------------------------------------------------

/// Inline board description (DESIGN §3). Sidecar resolution is io.rs
/// territory (Wave 3); the oracle layer only consumes inline boards.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoardInline {
    pub env_id: i64,
    pub n: u16,
    pub grid_data: Vec<String>,
}

fn default_true() -> bool {
    true
}

fn default_max_expansions() -> i64 {
    40_000 // generate.py --max-expansions default
}

/// task: "forward_instance" (DESIGN §3).
#[derive(Clone, Debug, Deserialize)]
pub struct ForwardInstanceItem {
    pub id: String,
    pub board: BoardInline,
    pub robots: Vec<Cell>,
    pub target_idx: usize,
    pub target: Cell,
    #[serde(default = "default_max_expansions")]
    pub max_expansions: i64,
    #[serde(default = "default_true")]
    pub full_policy: bool,
    #[serde(default)]
    pub full_policy_max_ctg: Option<i64>,
    #[serde(default)]
    pub score_candidates: bool,
}

/// Result line for task "forward_instance".
#[derive(Clone, Debug, Serialize)]
pub struct ForwardInstanceResult {
    pub id: String,
    pub status: &'static str, // "solved" | "relaxed_unreachable" | "unsolved"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub d_star: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub records: Option<Vec<MoveRecord>>,
}

pub fn run_forward_instance(item: &ForwardInstanceItem) -> ForwardInstanceResult {
    let walls = Walls::from_grid_data(&item.board.grid_data, item.board.n);
    match label_instance(
        item.board.env_id, &item.robots, item.target_idx, item.target, &walls,
        item.max_expansions, item.full_policy, item.full_policy_max_ctg,
        item.score_candidates,
    ) {
        LabelOutcome::Solved { d_star, records } => ForwardInstanceResult {
            id: item.id.clone(),
            status: "solved",
            d_star: Some(d_star),
            records: Some(records),
        },
        LabelOutcome::RelaxedUnreachable => ForwardInstanceResult {
            id: item.id.clone(),
            status: "relaxed_unreachable",
            d_star: None,
            records: None,
        },
        LabelOutcome::Unsolved => ForwardInstanceResult {
            id: item.id.clone(),
            status: "unsolved",
            d_star: None,
            records: None,
        },
    }
}

/// task: "replay_forward_state" (DESIGN §3, gate 3).
#[derive(Clone, Debug, Deserialize)]
pub struct ReplayForwardStateItem {
    pub id: String,
    pub board: BoardInline,
    pub positions: Vec<Cell>,
    pub target_idx: usize,
    pub target: Cell,
    #[serde(default = "default_max_expansions")]
    pub max_expansions: i64,
}

/// Result line for task "replay_forward_state" (`cost_to_go` null when
/// unsolved; moves as `[slot, dir_idx]`).
#[derive(Clone, Debug, Serialize)]
pub struct ReplayForwardStateResult {
    pub id: String,
    pub cost_to_go: Option<i64>,
    pub optimal_moves: Vec<[i64; 2]>,
    pub legal_moves: Vec<[i64; 2]>,
}

pub fn run_replay_forward_state(item: &ReplayForwardStateItem) -> ReplayForwardStateResult {
    let walls = Walls::from_grid_data(&item.board.grid_data, item.board.n);
    let (ctg, optimal, legal) = replay_forward_state(
        &item.positions, item.target_idx, item.target, &walls, item.max_expansions,
    );
    ReplayForwardStateResult {
        id: item.id.clone(),
        cost_to_go: ctg,
        optimal_moves: moves_to_pairs(&optimal),
        legal_moves: moves_to_pairs(&legal),
    }
}

// ---------------------------------------------------------------------------
// unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(rows: &[&str]) -> Vec<String> {
        rows.iter().map(|s| s.to_string()).collect()
    }

    /// 3x3 board, border walls only.
    fn border3() -> Walls {
        Walls::from_grid_data(&grid(&["NW", "N", "NE", "W", "", "E", "SW", "S", "SE"]), 3)
    }

    #[test]
    fn heap_tuple_order_matches_python_heapq() {
        // Python heapq pops tuples in ascending lexicographic order; the push
        // counter breaks (f, g) ties FIFO. Verify the Reverse-wrapped
        // BinaryHeap pops the identical sequence on a hand-built example.
        let entries: Vec<(i64, i64, u64, u128)> = vec![
            (5, 2, 3, 90), // pushed first, ties (5,2) with cnt 7
            (4, 7, 2, 91),
            (5, 1, 4, 92),
            (5, 2, 7, 93),
            (4, 7, 6, 94),
        ];
        let mut pq: BinaryHeap<Reverse<(i64, i64, u64, u128)>> = BinaryHeap::new();
        for e in &entries {
            pq.push(Reverse(*e));
        }
        let mut popped = Vec::new();
        while let Some(Reverse(e)) = pq.pop() {
            popped.push(e);
        }
        // Python: sorted(entries) == heapq pop order.
        let mut expect = entries.clone();
        expect.sort();
        assert_eq!(popped, expect);
        // among equal (f, g): FIFO by push counter
        assert_eq!(popped[0], (4, 7, 2, 91));
        assert_eq!(popped[1], (4, 7, 6, 94));
        assert_eq!(popped[3], (5, 2, 3, 90));
        assert_eq!(popped[4], (5, 2, 7, 93));
    }

    #[test]
    fn relaxed_target_dist_hand_example() {
        // Border-only 3x3, target (1,0). One-move predecessors: (0,0) [slides
        // right into it], (2,0) [left], (1,1)/(1,2) [up]. Everything else is 2.
        let walls = border3();
        let d = relaxed_target_dist((1, 0), &walls);
        let at = |x: usize, y: usize| d[y * 3 + x];
        assert_eq!(at(1, 0), Some(0));
        assert_eq!(at(0, 0), Some(1));
        assert_eq!(at(2, 0), Some(1));
        assert_eq!(at(1, 1), Some(1));
        assert_eq!(at(1, 2), Some(1));
        assert_eq!(at(0, 1), Some(2));
        assert_eq!(at(2, 2), Some(2));
    }

    #[test]
    fn solve_start_on_target_is_zero_with_single_entry_path() {
        let walls = border3();
        let pos = [(1u16, 0u16), (2, 2)];
        let hd = relaxed_target_dist((1, 0), &walls);
        assert_eq!(solve(&pos, 0, (1, 0), &walls, &hd, 40_000, ORACLE_INF), Some(0));
        let (c, path) =
            solve_with_path(&pos, 0, (1, 0), &walls, &hd, 40_000, ORACLE_INF).unwrap();
        assert_eq!(c, 0);
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].positions, vec![(1, 0), (2, 2)]);
        assert_eq!(path[0].action, None);
    }

    /// d* = 2 instance on the border-only 3x3: robot 0 must stop at (1,0),
    /// which needs a blocker at (2,0) first (helper slides up), so the
    /// optimum is 2 moves.
    fn d2_instance() -> ([Cell; 2], Cell, Vec<Option<i64>>) {
        let walls = border3();
        let hd = relaxed_target_dist((1, 0), &walls);
        ([(0, 0), (2, 2)], (1, 0), hd)
    }

    #[test]
    fn cost_cap_pruning_boundary() {
        let walls = border3();
        let (pos, tgt, hd) = d2_instance();
        assert_eq!(solve(&pos, 0, tgt, &walls, &hd, 40_000, ORACLE_INF), Some(2));
        assert_eq!(solve(&pos, 0, tgt, &walls, &hd, 40_000, 2), Some(2)); // cap == d*
        assert_eq!(solve(&pos, 0, tgt, &walls, &hd, 40_000, 1), None); // cap == d*-1
        assert_eq!(solve(&pos, 0, tgt, &walls, &hd, 40_000, 0), None); // h0 > cap pre-check
    }

    #[test]
    fn expansions_cap_boundary() {
        let walls = border3();
        let (pos, tgt, hd) = d2_instance();
        // Find the minimal K that solves (success is monotone in the cap:
        // the search is identical until the abort). Then K-1 must fail.
        let mut k = None;
        for cap in 1..=100 {
            if solve(&pos, 0, tgt, &walls, &hd, cap, ORACLE_INF).is_some() {
                k = Some(cap);
                break;
            }
        }
        let k = k.expect("instance must solve within 100 expansions");
        assert!(k >= 2, "d*=2 needs at least 2 expansions, got K={k}");
        assert_eq!(solve(&pos, 0, tgt, &walls, &hd, k, ORACLE_INF), Some(2));
        assert_eq!(solve(&pos, 0, tgt, &walls, &hd, k - 1, ORACLE_INF), None);
        // Python-agreement at the boundary is covered by the golden fixtures
        // (solve cases with min_expansions K and K-1 dumped from oracle.py).
    }

    #[test]
    fn label_trajectory_d2_records() {
        let walls = border3();
        let (pos, tgt, hd) = d2_instance();
        let (d, recs) =
            label_trajectory(&pos, 0, tgt, &walls, &hd, 40_000, true, None).unwrap();
        assert_eq!(d, 2);
        assert_eq!(recs.len(), 2); // pre-goal states only
        assert_eq!(recs[0].positions, vec![(0, 0), (2, 2)]);
        assert_eq!(recs[0].cost_to_go, 2);
        assert_eq!(recs[0].depth, 0);
        assert!(recs[0].full);
        assert_eq!(recs[1].cost_to_go, 1);
        // every best move must be legal
        for r in &recs {
            for m in &r.best_moves {
                assert!(r.legal_moves.contains(m));
            }
        }
    }

    #[test]
    fn label_instance_relaxed_unreachable_on_sealed_target() {
        // Seal (1,1) on the 3x3 board: N/E/S/W on the cell + mirrors.
        let g = grid(&["NW", "NS", "NE", "WE", "NESW", "WE", "SW", "NS", "SE"]);
        let walls = Walls::from_grid_data(&g, 3);
        let out = label_instance(7, &[(0, 0), (2, 2)], 0, (1, 1), &walls, 40_000, true,
                                 Some(6), true);
        assert_eq!(out, LabelOutcome::RelaxedUnreachable);
    }

    #[test]
    fn move_record_serializes_nine_fields_in_design_order() {
        let rec = make_rec(3, &[(1, 2), (3, 4)], (5, 6), 1, 7,
                           &[(0, Dir::Down)], &[(0, Dir::Down), (1, Dir::Left)], 2, true);
        let s = serde_json::to_string(&rec).unwrap();
        assert_eq!(
            s,
            "{\"env_id\":3,\"robots\":[[1,2],[3,4]],\"target\":[5,6],\
             \"target_idx\":1,\"cost_to_go\":7,\"best_moves\":[[0,1]],\
             \"legal_moves\":[[0,1],[1,2]],\"depth\":2,\"full\":true}"
        );
    }

    /// Out-of-envelope boards must fail LOUDLY in release too (the asserts in
    /// pack() are hard, not debug): n = 65 would silently corrupt packed keys.
    #[test]
    #[should_panic(expected = "state packing supports n <= 64")]
    fn solve_panics_loudly_on_board_larger_than_64() {
        let n = 65u16;
        let grid: Vec<String> = vec![String::new(); n as usize * n as usize];
        let walls = Walls::from_grid_data(&grid, n); // physics itself is unbounded
        let hd = relaxed_target_dist((1, 0), &walls);
        // start not on target and relaxed-reachable, so the pre-checks pass
        // and the search reaches pack() -> must panic, never truncate.
        let _ = solve(&[(0, 0), (2, 2)], 0, (1, 0), &walls, &hd, 40_000, ORACLE_INF);
    }

    /// 11 robots exceed the 128-bit key (12 bits x 11): loud failure, release too.
    #[test]
    #[should_panic(expected = "state packing supports at most 10 robots")]
    fn solve_panics_loudly_on_more_than_ten_robots() {
        let walls = border3();
        // 11 in-bounds cells (duplicates are irrelevant; pack asserts first)
        let pos: Vec<Cell> = (0..11u16).map(|i| (i % 3, (i / 3) % 3)).collect();
        let hd = relaxed_target_dist((1, 0), &walls);
        let _ = solve(&pos, 0, (1, 0), &walls, &hd, 40_000, ORACLE_INF);
    }

    #[test]
    fn pack_patch_unpack_roundtrip_64() {
        let n = 64u16;
        let pos: Vec<Cell> = vec![(0, 0), (63, 63), (17, 42), (63, 0), (0, 63), (31, 31), (1, 62), (62, 1)];
        let key = pack(&pos, n);
        let mut back = Vec::new();
        unpack(key, pos.len(), n, &mut back);
        assert_eq!(back, pos);
        let key2 = patch(key, 2, (5, 7), n);
        unpack(key2, pos.len(), n, &mut back);
        assert_eq!(back[2], (5, 7));
        assert_eq!(back[0], (0, 0));
        assert_eq!(back[7], (62, 1));
    }

    #[test]
    fn replay_forward_state_full_optimal_set() {
        let walls = border3();
        let (pos, tgt, _) = d2_instance();
        let (ctg, optimal, legal) = replay_forward_state(&pos, 0, tgt, &walls, 40_000);
        assert_eq!(ctg, Some(2));
        assert!(!optimal.is_empty());
        for m in &optimal {
            assert!(legal.contains(m));
        }
        // start-on-goal: ctg 0, empty optimal set
        let (ctg0, opt0, _) = replay_forward_state(&[(1, 0), (2, 2)], 0, tgt, &walls, 40_000);
        assert_eq!(ctg0, Some(0));
        assert!(opt0.is_empty());
    }
}
