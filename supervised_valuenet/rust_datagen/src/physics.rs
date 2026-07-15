//! Board physics — exact port of `simulate.py::wall_sets` / `::slide`
//! (DESIGN §5.1–5.2) plus the move-level successor enumeration of
//! `move_planner/state.py::legal_moves` / `::apply_move`.
//!
//! Parity notes (the Python line is the spec):
//! - `wall_sets`: `E` at (x,y) → wall right of (x,y); `S` → wall below (x,y);
//!   `W` mirrors to (x-1,y) only when x>0; `N` mirrors to (x,y-1) only when
//!   y>0. Border rows/cols carry explicit chars AND the slide rule clamps at
//!   0 / n-1 regardless.
//! - `slide`: walks cell by cell; stops when the next cell is across a
//!   wall/edge or occupied; returns the start cell when it cannot move at all
//!   (callers treat that as a no-op).

use serde::{Deserialize, Serialize};

use crate::types::{Cell, Dir, DIRECTIONS};

/// Wall lookup built from `grid_data` (port of `simulate.wall_sets`).
///
/// `right[y*n+x]` = wall between (x,y) and (x+1,y);
/// `down[y*n+x]`  = wall between (x,y) and (x,y+1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Walls {
    n: u16,
    right: Vec<bool>,
    down: Vec<bool>,
}

impl Walls {
    pub fn from_grid_data(grid_data: &[String], n: u16) -> Self {
        let nn = n as usize * n as usize;
        assert_eq!(
            grid_data.len(),
            nn,
            "grid_data length {} != n*n = {}",
            grid_data.len(),
            nn
        );
        let mut right = vec![false; nn];
        let mut down = vec![false; nn];
        for (idx, cell) in grid_data.iter().enumerate() {
            let col = idx % n as usize;
            let row = idx / n as usize;
            if cell.contains('E') {
                right[row * n as usize + col] = true;
            }
            if cell.contains('S') {
                down[row * n as usize + col] = true;
            }
            // W/N mirror onto the neighbour only when one exists (simulate.py).
            if cell.contains('W') && col > 0 {
                right[row * n as usize + (col - 1)] = true;
            }
            if cell.contains('N') && row > 0 {
                down[(row - 1) * n as usize + col] = true;
            }
        }
        Walls { n, right, down }
    }

    /// Board side length.
    #[inline]
    pub fn n(&self) -> u16 {
        self.n
    }

    /// Wall between (x,y) and (x+1,y)?
    #[inline]
    pub fn has_right(&self, x: u16, y: u16) -> bool {
        self.right[y as usize * self.n as usize + x as usize]
    }

    /// Wall between (x,y) and (x,y+1)?
    #[inline]
    pub fn has_down(&self, x: u16, y: u16) -> bool {
        self.down[y as usize * self.n as usize + x as usize]
    }
}

/// Port of `simulate.slide`: slide from `pos` until a wall, the board edge or
/// a blocker stops the robot. Returns the stop cell (== `pos` when it cannot
/// move at all).
pub fn slide(pos: Cell, dir: Dir, blockers: &[Cell], walls: &Walls) -> Cell {
    let n = walls.n;
    let (mut x, mut y) = pos;
    loop {
        // "can_leave" mirrors the Python guard: board edge OR wall on the
        // boundary being crossed.
        let can_leave = match dir {
            Dir::Up => y > 0 && !walls.has_down(x, y - 1),
            Dir::Down => y < n - 1 && !walls.has_down(x, y),
            Dir::Left => x > 0 && !walls.has_right(x - 1, y),
            Dir::Right => x < n - 1 && !walls.has_right(x, y),
        };
        if !can_leave {
            return (x, y);
        }
        let nxt = match dir {
            Dir::Up => (x, y - 1),
            Dir::Down => (x, y + 1),
            Dir::Left => (x - 1, y),
            Dir::Right => (x + 1, y),
        };
        if blockers.contains(&nxt) {
            return (x, y);
        }
        (x, y) = nxt;
    }
}

/// All non-no-op moves from `positions` — exact semantics and ORDER of
/// `move_planner/state.py::legal_moves`: robot slot-major, then `Dir` order
/// (up, down, left, right); a slide that cannot leave its cell is dropped.
///
/// Returns `(robot_slot, dir, new_positions)` triples.
pub fn successors(positions: &[Cell], walls: &Walls) -> Vec<(usize, Dir, Vec<Cell>)> {
    let mut out = Vec::new();
    let mut blockers: Vec<Cell> = Vec::with_capacity(positions.len().saturating_sub(1));
    for (i, &pos) in positions.iter().enumerate() {
        blockers.clear();
        blockers.extend(
            positions
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &c)| c),
        );
        for &d in DIRECTIONS.iter() {
            let nxt = slide(pos, d, &blockers, walls);
            if nxt == pos {
                continue; // blocked immediately -> illegal (no-op dropped)
            }
            let mut np = positions.to_vec();
            np[i] = nxt;
            out.push((i, d, np));
        }
    }
    out
}

/// Port of `move_planner/state.py::apply_move`: apply one `(slot, dir)` move;
/// `None` when the move is a no-op.
pub fn apply_move(positions: &[Cell], robot_slot: usize, dir: Dir, walls: &Walls) -> Option<Vec<Cell>> {
    let pos = positions[robot_slot];
    let blockers: Vec<Cell> = positions
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != robot_slot)
        .map(|(_, &c)| c)
        .collect();
    let nxt = slide(pos, dir, &blockers, walls);
    if nxt == pos {
        return None;
    }
    let mut np = positions.to_vec();
    np[robot_slot] = nxt;
    Some(np)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(rows: &[&str]) -> Vec<String> {
        rows.iter().map(|s| s.to_string()).collect()
    }

    /// 3x3 board, border walls only.
    fn border3() -> Vec<String> {
        grid(&["NW", "N", "NE", "W", "", "E", "SW", "S", "SE"])
    }

    #[test]
    fn wall_mirroring_w_at_border_column_does_not_mirror() {
        // 'W' at x=0 must NOT create walls_right at x=-1 (simulate.py: col>0).
        let walls = Walls::from_grid_data(&border3(), 3);
        // Only explicit E walls on the right border column.
        assert!(walls.has_right(2, 0));
        assert!(walls.has_right(2, 1));
        assert!(walls.has_right(2, 2));
        assert!(!walls.has_right(0, 0));
        assert!(!walls.has_right(1, 1));
    }

    #[test]
    fn wall_mirroring_n_at_border_row_does_not_mirror() {
        // 'N' at y=0 must NOT create walls_down at y=-1 (simulate.py: row>0).
        let walls = Walls::from_grid_data(&border3(), 3);
        assert!(walls.has_down(0, 2)); // explicit S on bottom row
        assert!(!walls.has_down(0, 0));
        assert!(!walls.has_down(1, 1));
    }

    #[test]
    fn wall_mirroring_interior() {
        // Interior 'W' at (1,1) mirrors to right-wall of (0,1);
        // interior 'N' at (1,1) mirrors to down-wall of (1,0).
        let g = grid(&["NW", "N", "NE", "W", "NW", "E", "SW", "S", "SE"]);
        let walls = Walls::from_grid_data(&g, 3);
        assert!(walls.has_right(0, 1));
        assert!(walls.has_down(1, 0));
    }

    #[test]
    fn slide_no_move_returns_start() {
        let walls = Walls::from_grid_data(&border3(), 3);
        // Against the top border: cannot move at all.
        assert_eq!(slide((1, 0), Dir::Up, &[], &walls), (1, 0));
        // Immediately blocked by a robot in the next cell.
        assert_eq!(slide((0, 0), Dir::Right, &[(1, 0)], &walls), (0, 0));
    }

    #[test]
    fn slide_runs_to_border_and_stops_before_blockers() {
        let walls = Walls::from_grid_data(&border3(), 3);
        assert_eq!(slide((0, 0), Dir::Right, &[], &walls), (2, 0));
        assert_eq!(slide((0, 0), Dir::Down, &[], &walls), (0, 2));
        // Blocker mid-ray: stop in the last free cell before it.
        assert_eq!(slide((0, 0), Dir::Right, &[(2, 0)], &walls), (1, 0));
    }

    #[test]
    fn slide_stops_at_interior_wall() {
        // 'S' at (1,0): wall below (1,0) stops a downward slide from (1,0)
        // and an upward slide from (1,2) at (1,1).
        let g = grid(&["NW", "NS", "NE", "W", "N", "E", "SW", "S", "SE"]);
        let walls = Walls::from_grid_data(&g, 3);
        assert_eq!(slide((1, 0), Dir::Down, &[], &walls), (1, 0));
        assert_eq!(slide((1, 2), Dir::Up, &[], &walls), (1, 1));
    }

    #[test]
    fn successors_order_is_slot_major_then_dir_order() {
        let walls = Walls::from_grid_data(&border3(), 3);
        // Robot 0 at (0,0) (up/left are no-ops), robot 1 at (2,2).
        let moves = successors(&[(0, 0), (2, 2)], &walls);
        let key: Vec<(usize, Dir)> = moves.iter().map(|&(s, d, _)| (s, d)).collect();
        assert_eq!(
            key,
            vec![
                (0, Dir::Down),
                (0, Dir::Right),
                (1, Dir::Up),
                (1, Dir::Left),
            ]
        );
        // Robots block each other: robot 0 sliding right stops before (2,0)?
        // No robot there — it reaches the wall; robot 1 sliding up stops at
        // (2,1)? No — (2,0) is free, robot 0 is at (0,0). Check real physics:
        assert_eq!(moves[1].2, vec![(2, 0), (2, 2)]);
        assert_eq!(moves[2].2, vec![(0, 0), (2, 0)]);
    }

    #[test]
    fn apply_move_matches_successors_and_rejects_noops() {
        let walls = Walls::from_grid_data(&border3(), 3);
        let positions = [(0, 0), (2, 2)];
        assert_eq!(apply_move(&positions, 0, Dir::Up, &walls), None);
        assert_eq!(
            apply_move(&positions, 0, Dir::Down, &walls),
            Some(vec![(0, 2), (2, 2)])
        );
    }
}
