//! Shared types (DESIGN.md §2). Owned by the lead; agents extend by
//! appending, never by repurposing existing definitions.

use serde::{Deserialize, Serialize};

/// (x, y) = (column, row); `idx = y*n + x` into `grid_data`.
pub type Cell = (u16, u16);

/// Matches Python `simulate.DIRECTIONS = ("up", "down", "left", "right")`.
/// `dir_idx` values in records use exactly this numbering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dir {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

pub const DIRECTIONS: [Dir; 4] = [Dir::Up, Dir::Down, Dir::Left, Dir::Right];

impl Dir {
    /// Unit step (dx, dy), matching `nn/gen_grids.DIRS`.
    pub fn step(self) -> (i32, i32) {
        match self {
            Dir::Up => (0, -1),
            Dir::Down => (0, 1),
            Dir::Left => (-1, 0),
            Dir::Right => (1, 0),
        }
    }
}

/// Canonical robot palette (`nn/gen_grids.PALETTE`); slot i = PALETTE[i].
pub const PALETTE: [&str; 8] = [
    "Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Cyan", "Magenta",
];

/// Subgoal-layer infinity (`GridEnv.py` / `skeleton/astar.py` INF).
pub const SUBGOAL_INF: i64 = 10_000;
/// Forward-oracle infinity (`move_planner/oracle.py` INF = 1<<30).
pub const ORACLE_INF: i64 = 1 << 30;
