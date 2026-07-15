//! Subgoal machinery (Agent C): exact port of the backward labeler —
//! `GridEnv.py` (final components, extended graphs, dependent-edge groups,
//! propose/score), `partial_plan.py`, `skeleton/astar.py` (CURRENT version,
//! all three fixes), `skeleton/heuristics.py`, and `nn/generate.py::rollout`.
//!
//! Parity ground rules (DESIGN §5.5–5.15, §5.18, §8):
//! - When Python does something odd, the oddity is ported, never "fixed".
//! - No HashMap/HashSet iteration order ever leaks into output; the two
//!   places Python leaks *set* iteration order (candidate enumeration order
//!   in `propose_subgoal_states`, `_dependent_supports` order in
//!   `heuristics.propose`) are replaced by DOCUMENTED deterministic orders:
//!   final-component cells in cell-index order (y-major, `idx = y*n + x`),
//!   in-edges in graph insertion order, dedup first-seen. The candidate SET
//!   is identical to Python's; the candidate ORDER may differ (DESIGN §5.11).
//! - Costs are `i64`; unreachable is `None`; subgoal infinity is `10_000`.
//!
//! Robot identity is the COLOR (slot in `types::PALETTE`); positions carried
//! in a `RobotAt` are whatever Python stores there (original instance
//! positions in `State`, planned positions on plan nodes).

pub mod astar;
pub mod grid_env;
pub mod plan;
pub mod rollout;

pub use astar::AStar;
pub use grid_env::SubgoalEnv;
pub use plan::PartialPlan;

use crate::types::{Cell, PALETTE};

/// Compact robot: `color` is the slot index into [`PALETTE`]. Mirrors
/// `GridEnv.Robot_at` (position, color-string); the string form exists only
/// at the I/O boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RobotAt {
    pub pos: Cell,
    pub color: u8,
}

impl RobotAt {
    pub fn color_str(&self) -> &'static str {
        PALETTE[self.color as usize]
    }
}

/// Slot index for a palette color string; `None` for non-palette colors
/// (fail-loud at the parsing boundary, DESIGN §7).
pub fn color_slot(color: &str) -> Option<u8> {
    PALETTE.iter().position(|c| *c == color).map(|i| i as u8)
}

/// Mirrors `GridEnv.State`: one puzzle instance.
#[derive(Clone, Debug)]
pub struct State {
    pub target: Cell,
    pub target_robot: RobotAt,
    pub helpers: Vec<RobotAt>,
}

/// Mirrors `GridEnv.Subgoal` (all five fields are carried, as Python does).
#[derive(Clone, Copy, Debug)]
pub struct Subgoal {
    pub bottleneck: RobotAt,
    pub support: RobotAt,
    pub goal_pos: Cell,
    pub target_robot: RobotAt,
    pub helper: RobotAt,
}

/// Mirrors `skeleton.heuristics.Candidate`. `score` is an i64: Python stores
/// the int result of `GridEnv.subgoal_score` (sums of table ints / 10_000).
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub subgoal: Subgoal,
    pub parent_support: Option<Cell>,
    pub score: i64,
}
