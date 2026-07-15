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
//!
//! Performance notes (Agent P — all bit-exact, DESIGN §5.16 unchanged):
//! - Slides inside the A* use per-(board,dir) wall-stop ray tables
//!   ([`Rays`]) + an O(R) nearest-blocker scan instead of walking cell by
//!   cell with a linear `blockers.contains` per step. Provably the same stop
//!   cell as `physics::slide` (unit-fuzzed against it).
//! - The heap priority packs `(f, g)` into one `u64` word: both are
//!   non-negative and < 2^31 (`h < ORACLE_INF = 2^30` is enforced by the
//!   pruning rule, `g <= expansions + 1 <= max_expansions + 1`, and
//!   `max_expansions < 2^30` is hard-asserted), so the u64 compare is
//!   EXACTLY the `(f, g)` lexicographic tuple compare; `cnt` is unique per
//!   push, so the trailing state key still never decides an ordering.
//! - One [`SearchCtx`] per work item reuses the heap / `g` / `came`
//!   allocations across the item's many child solves and holds the hdist
//!   field as a flat `i64` vec (`ORACLE_INF` = the Python dict-miss).
//! - `label_instance` memoizes uncapped candidate-child solves by packed
//!   state within the item (same board, target, hdist and expansion budget
//!   -> identical result by determinism; duplicates still emit their own
//!   records exactly as before).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::physics::{apply_move, successors, Walls};
use crate::types::{Cell, Dir, DIRECTIONS, ORACLE_INF};

// ---------------------------------------------------------------------------
// joint-state packing (private): each cell as (x | y << cb) with
// cb = coord_bits(n) bits per coordinate, robot slots side by side in one
// integer. u64 when 2*cb*R <= 64 (every production config: 16x16 R8 = 64
// bits, 32x32 R4 = 40, 64x64 R4 = 48), u128 otherwise. Keys are INTERNAL:
// they never decide any ordering (the heap push counter is unique) and the
// maps keyed by them are never iterated, so the key type/encoding cannot
// leak into any output (DESIGN §8). Envelope: n <= 64, R <= 10, as before.
// ---------------------------------------------------------------------------

const CELL_BITS: u32 = 12; // 2 * max coord_bits — defines the R <= 10 envelope

/// Max robots the u128 packing supports; also the fixed size of the stack
/// buffers used inside the search (no per-node heap allocation).
const MAX_ROBOTS: usize = 128 / CELL_BITS as usize;

/// Bits per coordinate in a packed cell (x and y each).
#[inline]
fn coord_bits(n: u16) -> u32 {
    if n <= 16 {
        4
    } else if n <= 32 {
        5
    } else {
        6
    }
}

/// Whether the joint state of `r` robots on an n x n board fits a u64 key.
#[inline]
fn key_fits_u64(n: u16, r: usize) -> bool {
    2 * coord_bits(n) as usize * r <= 64
}

/// Hard asserts (release too): every search enters through Key::pack exactly
/// once (patch only rewrites pack-built keys), so an out-of-envelope work
/// item fails loudly instead of silently corrupting packed keys (DESIGN §7
/// fail-loud). Cost: one call per solve, not per node.
#[inline]
fn assert_envelope(n: u16, r: usize) {
    assert!(n <= 64, "state packing supports n <= 64, got n = {n}");
    assert!(
        r as u32 * CELL_BITS <= 128,
        "state packing supports at most {} robots, got {}",
        128 / CELL_BITS,
        r
    );
}

/// Packed joint-state key (see module header note). Implemented for u64 and
/// u128; the search is monomorphized over the two.
trait Key: Copy + Eq + Ord + std::hash::Hash {
    fn pack(positions: &[Cell], n: u16, cb: u32) -> Self;
    fn patch(self, i: usize, cell: Cell, cb: u32) -> Self;
    fn unpack(self, r: usize, cb: u32, out: &mut [Cell]);
}

macro_rules! impl_key {
    ($t:ty) => {
        impl Key for $t {
            #[inline]
            fn pack(positions: &[Cell], n: u16, cb: u32) -> $t {
                assert_envelope(n, positions.len());
                debug_assert!(positions.len() as u32 * 2 * cb <= <$t>::BITS);
                let mut key: $t = 0;
                for (i, &(x, y)) in positions.iter().enumerate() {
                    key |= (((y as $t) << cb) | x as $t) << (2 * cb * i as u32);
                }
                key
            }

            #[inline]
            fn patch(self, i: usize, cell: Cell, cb: u32) -> $t {
                let shift = 2 * cb * i as u32;
                let mask: $t = ((1 as $t) << (2 * cb)) - 1;
                (self & !(mask << shift))
                    | ((((cell.1 as $t) << cb) | cell.0 as $t) << shift)
            }

            #[inline]
            fn unpack(self, r: usize, cb: u32, out: &mut [Cell]) {
                let cmask: $t = ((1 as $t) << cb) - 1;
                for (i, slot) in out.iter_mut().enumerate().take(r) {
                    let v = self >> (2 * cb * i as u32);
                    *slot = ((v & cmask) as u16, ((v >> cb) & cmask) as u16);
                }
            }
        }
    };
}
impl_key!(u64);
impl_key!(u128);

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

/// Wall-stop ray tables: for every cell, the coordinate a blocker-free slide
/// stops at. `up`/`down` hold the stop Y for the cell's column, `left`/`right`
/// the stop X for its row. Built with the exact `can_leave` rule of
/// `physics::slide` (one recurrence per direction), so
/// `slide(c, d, &[], walls) == ray stop` by construction — unit-fuzzed too.
struct Rays {
    up: Vec<u16>,
    down: Vec<u16>,
    left: Vec<u16>,
    right: Vec<u16>,
}

impl Rays {
    fn new(walls: &Walls) -> Rays {
        let n = walls.n();
        let nu = n as usize;
        let mut up = vec![0u16; nu * nu];
        let mut down = vec![0u16; nu * nu];
        let mut left = vec![0u16; nu * nu];
        let mut right = vec![0u16; nu * nu];
        for x in 0..n {
            for y in 0..n {
                // can_leave up from (x,y): y > 0 && no wall between y-1 and y
                up[y as usize * nu + x as usize] = if y > 0 && !walls.has_down(x, y - 1) {
                    up[(y - 1) as usize * nu + x as usize]
                } else {
                    y
                };
                let yd = n - 1 - y; // descending for down
                down[yd as usize * nu + x as usize] = if yd < n - 1 && !walls.has_down(x, yd) {
                    down[(yd + 1) as usize * nu + x as usize]
                } else {
                    yd
                };
            }
        }
        for y in 0..n {
            for x in 0..n {
                left[y as usize * nu + x as usize] = if x > 0 && !walls.has_right(x - 1, y) {
                    left[y as usize * nu + (x - 1) as usize]
                } else {
                    x
                };
                let xd = n - 1 - x;
                right[y as usize * nu + xd as usize] = if xd < n - 1 && !walls.has_right(xd, y) {
                    right[y as usize * nu + (xd + 1) as usize]
                } else {
                    xd
                };
            }
        }
        Rays { up, down, left, right }
    }
}

/// Nearest-blocker scans, one per direction: `wall` is the blocker-free stop
/// coordinate from the ray table; the slices hold the OTHER robots'
/// coordinates on the mover's column (`ys`) / row (`xs`). A robot strictly
/// inside the ray segment shortens the slide to the cell just before it —
/// exactly where the cell-by-cell walk of `physics::slide` stops (a robot
/// beyond the wall stop is never reached by the walk). A robot adjacent to
/// the mover yields the mover's own coordinate = no-op, like the walk.
#[inline]
fn stop_up(wall: u16, py: u16, ys: &[u16]) -> u16 {
    let mut sy = wall;
    for &by in ys {
        if by < py && by >= sy {
            sy = by + 1; // nearest-so-far blocker: stop just below it
        }
    }
    sy
}

#[inline]
fn stop_down(wall: u16, py: u16, ys: &[u16]) -> u16 {
    let mut sy = wall;
    for &by in ys {
        if by > py && by <= sy {
            sy = by - 1;
        }
    }
    sy
}

#[inline]
fn stop_left(wall: u16, px: u16, xs: &[u16]) -> u16 {
    let mut sx = wall;
    for &bx in xs {
        if bx < px && bx >= sx {
            sx = bx + 1;
        }
    }
    sx
}

#[inline]
fn stop_right(wall: u16, px: u16, xs: &[u16]) -> u16 {
    let mut sx = wall;
    for &bx in xs {
        if bx > px && bx <= sx {
            sx = bx - 1;
        }
    }
    sx
}

/// Slide `robots[skip]` from `pos` in direction `d` — identical result to
/// `physics::slide(pos, d, blockers, walls)` with `blockers` = all robots
/// except slot `skip` (unit-fuzzed against it). The search loop inlines the
/// same [`stop_up`]/... helpers with the column/row classification hoisted
/// out of the per-direction loop; this composed form exists so the fuzz test
/// exercises exactly those helpers.
#[cfg(test)]
fn fast_slide(rays: &Rays, n: u16, pos: Cell, d: Dir, robots: &[Cell], skip: usize) -> Cell {
    let (px, py) = pos;
    let i0 = py as usize * n as usize + px as usize;
    let mut col_ys = [0u16; MAX_ROBOTS];
    let mut ncol = 0usize;
    let mut row_xs = [0u16; MAX_ROBOTS];
    let mut nrow = 0usize;
    for (j, &(bx, by)) in robots.iter().enumerate() {
        if j != skip {
            if bx == px {
                col_ys[ncol] = by;
                ncol += 1;
            }
            if by == py {
                row_xs[nrow] = bx;
                nrow += 1;
            }
        }
    }
    match d {
        Dir::Up => (px, stop_up(rays.up[i0], py, &col_ys[..ncol])),
        Dir::Down => (px, stop_down(rays.down[i0], py, &col_ys[..ncol])),
        Dir::Left => (stop_left(rays.left[i0], px, &row_xs[..nrow]), py),
        Dir::Right => (stop_right(rays.right[i0], px, &row_xs[..nrow]), py),
    }
}

/// Pack the A* priority `(f, g)` into one u64 whose integer order equals the
/// `(f, g)` tuple order. Sound because both are non-negative and < 2^31:
/// `h < ORACLE_INF = 2^30` (pruned otherwise), `g <= expansions + 1` and
/// `max_expansions < 2^30` (hard-asserted at search entry), so
/// `f = g + h < 2^31`.
#[inline]
fn pack_prio(f: i64, g: i64) -> u64 {
    ((f as u64) << 32) | (g as u64)
}

/// Per-(board, target) search context: ray tables + flat hdist + reusable
/// search buffers, so one work item's many `solve` calls share allocations.
/// Building one per public `solve` call keeps the old API cheap too.
struct SearchCtx<K: Key> {
    n: u16,
    cb: u32,
    rays: Rays,
    /// `hdist[y*n + x]`; `ORACLE_INF` = unreachable (the Python dict-miss).
    hdist: Vec<i64>,
    // Heap entry: (packed (f,g), push counter, state key). The counter is
    // unique per push, so the order is total and the trailing key never
    // compares — identical pop sequence to Python heapq on (f, g, cnt, ...).
    pq: BinaryHeap<Reverse<(u64, u64, K)>>,
    g: FxHashMap<K, i64>,
    // came: child -> (parent, action as (slot << 2) | dir); the start entry
    // is its own parent (action ignored). Only needed for path reconstruction
    // (Python fills it unconditionally, but it never influences the search).
    came: FxHashMap<K, (K, u8)>,
}

impl<K: Key> SearchCtx<K> {
    fn new(walls: &Walls, hdist: &[Option<i64>]) -> SearchCtx<K> {
        SearchCtx {
            n: walls.n(),
            cb: coord_bits(walls.n()),
            rays: Rays::new(walls),
            hdist: hdist.iter().map(|d| d.unwrap_or(ORACLE_INF)).collect(),
            pq: BinaryHeap::new(),
            g: FxHashMap::default(),
            came: FxHashMap::default(),
        }
    }

    fn solve(
        &mut self,
        positions: &[Cell],
        target_idx: usize,
        target: Cell,
        max_expansions: i64,
        cost_cap: i64,
    ) -> Option<i64> {
        self.solve_inner(positions, target_idx, target, max_expansions, cost_cap, false)
            .map(|(cost, _)| cost)
    }

    fn solve_with_path(
        &mut self,
        positions: &[Cell],
        target_idx: usize,
        target: Cell,
        max_expansions: i64,
        cost_cap: i64,
    ) -> Option<(i64, Vec<PathEntry>)> {
        self.solve_inner(positions, target_idx, target, max_expansions, cost_cap, true)
            .map(|(cost, path)| (cost, path.expect("want_path")))
    }

    fn solve_inner(
        &mut self,
        positions: &[Cell],
        target_idx: usize,
        target: Cell,
        max_expansions: i64,
        cost_cap: i64,
        want_path: bool,
    ) -> Option<(i64, Option<Vec<PathEntry>>)> {
        let n = self.n;
        let nu = n as usize;
        let r = positions.len();
        let hd = &self.hdist;
        let h_of = |hd: &[i64], c: Cell| hd[c.1 as usize * nu + c.0 as usize];

        let h0 = h_of(hd, positions[target_idx]);
        if h0 >= ORACLE_INF || h0 > cost_cap {
            return None;
        }
        if positions[target_idx] == target {
            let path = want_path.then(|| {
                vec![PathEntry { positions: positions.to_vec(), action: None }]
            });
            return Some((0, path));
        }
        // Envelope guard for the packed (f, g) priority (see pack_prio): out
        // of range fails loudly instead of silently mis-ordering the heap.
        assert!(
            max_expansions < (1 << 30),
            "solve supports max_expansions < 2^30, got {max_expansions}"
        );

        let start = K::pack(positions, n, self.cb);
        let mut cnt: u64 = 0;
        self.pq.clear();
        self.g.clear();
        self.came.clear();
        self.pq.push(Reverse((pack_prio(h0, 0), cnt, start)));
        self.g.insert(start, 0);
        if want_path {
            self.came.insert(start, (start, 0)); // self-parent = start marker
        }
        let mut expansions: i64 = 0;

        let mut pos_arr = [(0u16, 0u16); MAX_ROBOTS];

        while let Some(Reverse((prio, _c, cur))) = self.pq.pop() {
            let gc = (prio & 0xFFFF_FFFF) as i64;
            if gc != *self.g.get(&cur).unwrap_or(&ORACLE_INF) {
                continue; // stale
            }
            let pos_buf = &mut pos_arr[..r];
            cur.unpack(r, self.cb, pos_buf);
            if pos_buf[target_idx] == target {
                if !want_path {
                    return Some((gc, None));
                }
                let mut path: Vec<PathEntry> = Vec::new();
                let mut sk = cur;
                let mut sbuf = [(0u16, 0u16); MAX_ROBOTS];
                loop {
                    let (ps, code) = self.came[&sk];
                    sk.unpack(r, self.cb, &mut sbuf[..r]);
                    let is_start = ps == sk;
                    path.push(PathEntry {
                        positions: sbuf[..r].to_vec(),
                        action: if is_start {
                            None // start entry (Python's came[start] = None)
                        } else {
                            Some(((code >> 2) as usize, DIRECTIONS[(code & 3) as usize]))
                        },
                    });
                    if is_start {
                        break;
                    }
                    sk = ps;
                }
                path.reverse();
                return Some((gc, Some(path)));
            }
            expansions += 1;
            if expansions > max_expansions {
                return None;
            }
            let ng = gc + 1;
            // h of the target robot's cell changes only when the target robot
            // itself moves — hoist the "some other robot moved" value.
            let h_stay = h_of(&self.hdist, pos_buf[target_idx]);
            // Successor enumeration: robot slot-major, then Dir order — the
            // exact order of physics::successors / state.py::legal_moves,
            // inlined on packed keys to avoid per-child Vec allocations.
            //
            // Python evaluates `ng < g.get(child)` first and the h/cost_cap
            // prune second; both are side-effect-free filters and the state
            // is only mutated when BOTH pass, so evaluating the cheap h-prune
            // before the hash lookup accepts the identical child set with
            // identical priorities — bit-exact, just fewer map probes.
            for i in 0..r {
                let (px, py) = pos_buf[i];
                let i0 = py as usize * nu + px as usize;
                // same-column / same-row robots of the mover, classified once
                // for all four directions (see fast_slide).
                let mut col_ys = [0u16; MAX_ROBOTS];
                let mut ncol = 0usize;
                let mut row_xs = [0u16; MAX_ROBOTS];
                let mut nrow = 0usize;
                for (j, &(bx, by)) in pos_buf.iter().enumerate() {
                    if j != i {
                        if bx == px {
                            col_ys[ncol] = by;
                            ncol += 1;
                        }
                        if by == py {
                            row_xs[nrow] = bx;
                            nrow += 1;
                        }
                    }
                }
                for &d in DIRECTIONS.iter() {
                    let nxt = match d {
                        Dir::Up => (px, stop_up(self.rays.up[i0], py, &col_ys[..ncol])),
                        Dir::Down => (px, stop_down(self.rays.down[i0], py, &col_ys[..ncol])),
                        Dir::Left => (stop_left(self.rays.left[i0], px, &row_xs[..nrow]), py),
                        Dir::Right => {
                            (stop_right(self.rays.right[i0], px, &row_xs[..nrow]), py)
                        }
                    };
                    if nxt == pos_buf[i] {
                        continue; // no-op -> illegal
                    }
                    let h = if i == target_idx { h_of(&self.hdist, nxt) } else { h_stay };
                    if h >= ORACLE_INF || ng + h > cost_cap {
                        continue;
                    }
                    let child = cur.patch(i, nxt, self.cb);
                    match self.g.entry(child) {
                        std::collections::hash_map::Entry::Occupied(mut e) => {
                            if ng < *e.get() {
                                e.insert(ng);
                                cnt += 1;
                                if want_path {
                                    self.came.insert(child, (cur, ((i as u8) << 2) | d as u8));
                                }
                                self.pq.push(Reverse((pack_prio(ng + h, ng), cnt, child)));
                            }
                        }
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert(ng);
                            cnt += 1;
                            if want_path {
                                self.came.insert(child, (cur, ((i as u8) << 2) | d as u8));
                            }
                            self.pq.push(Reverse((pack_prio(ng + h, ng), cnt, child)));
                        }
                    }
                }
            }
        }
        None
    }
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
    if key_fits_u64(walls.n(), positions.len()) {
        SearchCtx::<u64>::new(walls, hdist)
            .solve(positions, target_idx, target, max_expansions, cost_cap)
    } else {
        SearchCtx::<u128>::new(walls, hdist)
            .solve(positions, target_idx, target, max_expansions, cost_cap)
    }
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
    if key_fits_u64(walls.n(), positions.len()) {
        SearchCtx::<u64>::new(walls, hdist)
            .solve_with_path(positions, target_idx, target, max_expansions, cost_cap)
    } else {
        SearchCtx::<u128>::new(walls, hdist)
            .solve_with_path(positions, target_idx, target, max_expansions, cost_cap)
    }
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
fn is_optimal_child<K: Key>(
    ctx: &mut SearchCtx<K>,
    child: &[Cell],
    target_idx: usize,
    target: Cell,
    max_expansions: i64,
    here: i64,
) -> bool {
    if child[target_idx] == target {
        return here == 1;
    }
    ctx.solve(child, target_idx, target, max_expansions, here - 1) == Some(here - 1)
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
    if key_fits_u64(walls.n(), positions.len()) {
        let mut ctx = SearchCtx::<u64>::new(walls, hdist);
        label_trajectory_ctx(
            &mut ctx, positions, target_idx, target, walls, max_expansions,
            full_policy, full_policy_max_ctg,
        )
    } else {
        let mut ctx = SearchCtx::<u128>::new(walls, hdist);
        label_trajectory_ctx(
            &mut ctx, positions, target_idx, target, walls, max_expansions,
            full_policy, full_policy_max_ctg,
        )
    }
}

/// [`label_trajectory`] on an existing context (buffer reuse across an item).
#[allow(clippy::too_many_arguments)]
fn label_trajectory_ctx<K: Key>(
    ctx: &mut SearchCtx<K>,
    positions: &[Cell],
    target_idx: usize,
    target: Cell,
    walls: &Walls,
    max_expansions: i64,
    full_policy: bool,
    full_policy_max_ctg: Option<i64>,
) -> Option<(i64, Vec<TrajRecord>)> {
    let (d_star, path) =
        ctx.solve_with_path(positions, target_idx, target, max_expansions, ORACLE_INF)?;
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
                    is_optimal_child(ctx, child, target_idx, target, max_expansions, here)
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
    if key_fits_u64(walls.n(), positions.len()) {
        label_instance_inner::<u64>(
            env_id, positions, target_idx, target, walls, &hdist, max_expansions,
            full_policy, full_policy_max_ctg, score_candidates,
        )
    } else {
        label_instance_inner::<u128>(
            env_id, positions, target_idx, target, walls, &hdist, max_expansions,
            full_policy, full_policy_max_ctg, score_candidates,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn label_instance_inner<K: Key>(
    env_id: i64,
    positions: &[Cell],
    target_idx: usize,
    target: Cell,
    walls: &Walls,
    hdist: &[Option<i64>],
    max_expansions: i64,
    full_policy: bool,
    full_policy_max_ctg: Option<i64>,
    score_candidates: bool,
) -> LabelOutcome {
    let mut ctx = SearchCtx::<K>::new(walls, hdist);
    let Some((d_star, recs)) = label_trajectory_ctx(
        &mut ctx, positions, target_idx, target, walls, max_expansions,
        full_policy, full_policy_max_ctg,
    ) else {
        return LabelOutcome::Unsolved;
    };
    // Uncapped candidate-child solves memoized by packed state: within this
    // item the board, target, hdist and expansion budget are all fixed, and
    // solve() is deterministic, so a repeated child state MUST produce the
    // same result — the duplicate's record is still emitted exactly as
    // before, only the recomputation is skipped.
    let mut cand_memo: FxHashMap<K, Option<i64>> = FxHashMap::default();
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
                let key = K::pack(&child, walls.n(), ctx.cb);
                let solved = match cand_memo.get(&key) {
                    Some(&v) => v,
                    None => {
                        let v =
                            ctx.solve(&child, target_idx, target, max_expansions, ORACLE_INF);
                        cand_memo.insert(key, v);
                        v
                    }
                };
                match solved {
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
    if key_fits_u64(walls.n(), positions.len()) {
        replay_forward_state_inner::<u64>(positions, target_idx, target, walls, &hdist, max_expansions)
    } else {
        replay_forward_state_inner::<u128>(positions, target_idx, target, walls, &hdist, max_expansions)
    }
}

fn replay_forward_state_inner<K: Key>(
    positions: &[Cell],
    target_idx: usize,
    target: Cell,
    walls: &Walls,
    hdist: &[Option<i64>],
    max_expansions: i64,
) -> (Option<i64>, Vec<Action>, Vec<Action>) {
    let mut ctx = SearchCtx::<K>::new(walls, hdist);
    let succ = successors(positions, walls);
    let legal: Vec<Action> = succ.iter().map(|&(i, d, _)| (i, d)).collect();
    let ctg = ctx.solve(positions, target_idx, target, max_expansions, ORACLE_INF);
    let optimal: Vec<Action> = match ctg {
        Some(here) if here > 0 => succ
            .iter()
            .filter(|(_, _, child)| {
                is_optimal_child(&mut ctx, child, target_idx, target, max_expansions, here)
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
    use crate::physics::slide;

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
        // 64x64 with 8 robots needs 12 bits/cell x 8 = 96 -> u128 key.
        let n = 64u16;
        assert!(!key_fits_u64(n, 8));
        let cb = coord_bits(n);
        let pos: Vec<Cell> = vec![(0, 0), (63, 63), (17, 42), (63, 0), (0, 63), (31, 31), (1, 62), (62, 1)];
        let key = <u128 as Key>::pack(&pos, n, cb);
        let mut back = [(0u16, 0u16); MAX_ROBOTS];
        key.unpack(pos.len(), cb, &mut back[..pos.len()]);
        assert_eq!(&back[..pos.len()], &pos[..]);
        let key2 = key.patch(2, (5, 7), cb);
        key2.unpack(pos.len(), cb, &mut back[..pos.len()]);
        assert_eq!(back[2], (5, 7));
        assert_eq!(back[0], (0, 0));
        assert_eq!(back[7], (62, 1));
    }

    #[test]
    fn pack_patch_unpack_roundtrip_u64_key() {
        // 16x16 with 8 robots fits exactly: 8 bits/cell x 8 = 64.
        let n = 16u16;
        assert!(key_fits_u64(n, 8));
        let cb = coord_bits(n);
        let pos: Vec<Cell> = vec![(0, 0), (15, 15), (7, 12), (15, 0), (0, 15), (8, 8), (1, 14), (14, 1)];
        let key = <u64 as Key>::pack(&pos, n, cb);
        let mut back = [(0u16, 0u16); MAX_ROBOTS];
        key.unpack(pos.len(), cb, &mut back[..pos.len()]);
        assert_eq!(&back[..pos.len()], &pos[..]);
        let key2 = key.patch(5, (3, 9), cb);
        key2.unpack(pos.len(), cb, &mut back[..pos.len()]);
        assert_eq!(back[5], (3, 9));
        assert_eq!(back[0], (0, 0));
        assert_eq!(back[7], (14, 1));
        // wider boards: 24x24 (5-bit coords) and 64x64 (6-bit) with 4 robots
        for &(nn, robots) in &[(24u16, 4usize), (33, 4), (64, 5)] {
            let cb = coord_bits(nn);
            assert!(key_fits_u64(nn, robots));
            let pos: Vec<Cell> = (0..robots)
                .map(|i| ((i as u16 * 7) % nn, (nn - 1) - (i as u16 * 5) % nn))
                .collect();
            let key = <u64 as Key>::pack(&pos, nn, cb);
            let mut back = [(0u16, 0u16); MAX_ROBOTS];
            key.unpack(robots, cb, &mut back[..robots]);
            assert_eq!(&back[..robots], &pos[..], "n={nn} r={robots}");
        }
    }

    /// The packed u64 priority must order EXACTLY like the (f, g) tuple for
    /// all in-envelope values, so the pop sequence is unchanged.
    #[test]
    fn packed_priority_orders_like_tuple() {
        let vals: [i64; 7] = [0, 1, 2, 999, 10_000, (1 << 30) - 1, (1 << 31) - 1];
        for &f1 in &vals {
            for &g1 in &vals {
                for &f2 in &vals {
                    for &g2 in &vals {
                        assert_eq!(
                            pack_prio(f1, g1).cmp(&pack_prio(f2, g2)),
                            (f1, g1).cmp(&(f2, g2)),
                            "(f,g) order mismatch at ({f1},{g1}) vs ({f2},{g2})"
                        );
                    }
                }
            }
        }
    }

    /// The u64 and u128 key paths must run the IDENTICAL search: same cost,
    /// same reconstructed path, same capped behaviour (keys are opaque — only
    /// the counter orders ties, so the packing width cannot leak).
    #[test]
    fn u64_and_u128_key_searches_agree() {
        let walls = border3();
        let (pos, tgt, hd) = d2_instance();
        assert!(key_fits_u64(3, pos.len()));
        for cap in [ORACLE_INF, 2, 1] {
            for max_exp in [40_000i64, 3, 2, 1] {
                let a = SearchCtx::<u64>::new(&walls, &hd)
                    .solve_with_path(&pos, 0, tgt, max_exp, cap);
                let b = SearchCtx::<u128>::new(&walls, &hd)
                    .solve_with_path(&pos, 0, tgt, max_exp, cap);
                assert_eq!(a, b, "cap {cap} max_exp {max_exp}");
                let a = SearchCtx::<u64>::new(&walls, &hd).solve(&pos, 0, tgt, max_exp, cap);
                let b = SearchCtx::<u128>::new(&walls, &hd).solve(&pos, 0, tgt, max_exp, cap);
                assert_eq!(a, b, "(no path) cap {cap} max_exp {max_exp}");
            }
        }
    }

    /// Rays + fast_slide must reproduce physics::slide exactly: exhaustive
    /// over every (cell, dir) blocker-free, plus a deterministic pseudo-random
    /// fuzz with robot sets on a walled board.
    #[test]
    fn fast_slide_matches_physics_slide() {
        // interior walls in all four directions + border
        let g = grid(&[
            "NW", "N", "NS", "NE", //
            "W", "E", "", "E", //
            "WS", "", "NW", "E", //
            "SW", "S", "SE", "SE",
        ]);
        let walls = Walls::from_grid_data(&g, 4);
        let rays = Rays::new(&walls);
        // blocker-free: ray table == slide for every cell and direction
        for y in 0..4u16 {
            for x in 0..4u16 {
                for &d in DIRECTIONS.iter() {
                    let expect = slide((x, y), d, &[], &walls);
                    let got = fast_slide(&rays, 4, (x, y), d, &[(x, y)], 0);
                    assert_eq!(got, expect, "blocker-free at ({x},{y}) {d:?}");
                }
            }
        }
        // fuzz: 4 robots, every position mix from a deterministic LCG
        let mut s: u64 = 42;
        let mut rnd = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % 16) as u16
        };
        for _ in 0..2000 {
            let mut pos: Vec<Cell> = Vec::new();
            while pos.len() < 4 {
                let c = (rnd() % 4, rnd() % 4);
                if !pos.contains(&c) {
                    pos.push(c);
                }
            }
            for i in 0..4 {
                let blockers: Vec<Cell> =
                    pos.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, &c)| c).collect();
                for &d in DIRECTIONS.iter() {
                    let expect = slide(pos[i], d, &blockers, &walls);
                    let got = fast_slide(&rays, 4, pos[i], d, &pos, i);
                    assert_eq!(got, expect, "at {:?} slot {i} {d:?} of {:?}", pos[i], pos);
                }
            }
        }
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
