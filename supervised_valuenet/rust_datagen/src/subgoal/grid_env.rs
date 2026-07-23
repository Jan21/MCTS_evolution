//! `SubgoalEnv` — port of `GridEnv.py` (final components, extended graphs,
//! dependent-edge groups, exact/relaxed lengths, propose/score) plus
//! `skeleton/heuristics.py::propose`.
//!
//! Caching strategy: LAZY with memoization (the `pyref/lazy_env.py` model,
//! proven value-identical to the eager `GridEnv.__init__` by
//! `pyref/prove_lazy_env.py`). Identical semantics argument:
//! - every cache is looked up by key only, never iterated, so lazy-vs-eager
//!   cannot change any observable order;
//! - the eager `(goal, None)` pairs entry exists only when `goal` has an
//!   independent in-edge, but `propose_subgoal_states`' cache-miss branch
//!   recomputes the very same expression (DESIGN §5.9), so presence/absence
//!   of the entry is unobservable.
//!
//! `max_final_component_distance` is always `None` in datagen (DESIGN §5.6);
//! work items requesting otherwise are rejected upstream. The port therefore
//! implements only the `None` behaviour: final component = ancestors.
//!
//! Deterministic orders replacing Python set iteration (DESIGN §5.11,
//! documented contract):
//! - pair collection iterates `fc ∪ {goal}` cells in cell-index order
//!   (y-major, `idx = y*n + x`), then each cell's in-edges in graph insertion
//!   order; pairs dedup first-seen;
//! - `dependent_supports` returns in-edge insertion order, dedup first-seen.

use std::cell::RefCell;
use std::rc::Rc;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::board::CompiledBoard;
use crate::types::{Cell, SUBGOAL_INF};

use super::{Candidate, RobotAt, State, Subgoal};

/// Final component as a cell-index bitmap (`fc[idx] == true` ⇔ member).
pub type FcSet = Vec<bool>;

/// A (bottleneck, support) pair list in the documented deterministic order.
pub type PairList = Vec<(Cell, Cell)>;

struct Caches {
    /// goal -> ancestors on the dependent-edge-free graph
    /// (`GridEnv.precomputed_final_components`).
    final_components: FxHashMap<Cell, Rc<FcSet>>,
    /// (bottleneck, support) -> extended-graph final component
    /// (`_dependent_edge_cache[..]['final_component']`, also the
    /// `propose_subgoal_states` cache-miss recompute for non-group keys).
    extended: FxHashMap<(Cell, Cell), Rc<FcSet>>,
    /// (goal, support?) -> bottleneck/support pairs
    /// (`_bottleneck_support_pairs_cache`, unified with the recompute branch).
    pairs: FxHashMap<(Cell, Option<Cell>), Rc<PairList>>,
    /// Lever B1: (goal, support?) -> transient (wall-less-support) pairs
    /// (`GridEnv._transient_pairs_cache`). Same collection rule as `pairs`
    /// with the wall-adjacency test inverted; the two sets are disjoint.
    transient_pairs: FxHashMap<(Cell, Option<Cell>), Rc<PairList>>,
}

/// Wrapper over a [`CompiledBoard`] mirroring `GridEnv`.
pub struct SubgoalEnv<'a> {
    board: &'a CompiledBoard,
    n: usize,
    /// Dependent-edge groups (`GridEnv._dependent_edge_cache[..]['edges']`):
    /// key (head v, support cell) -> edge tails u, in in-edge insertion order.
    /// Only `min` over the tails is ever consumed, so the difference from
    /// Python's `G.edges` grouping order is unobservable (stock boards have a
    /// different OUT-edge order; in-edge order is the stable surface).
    groups: FxHashMap<(Cell, Cell), Vec<Cell>>,
    caches: RefCell<Caches>,
}

impl<'a> SubgoalEnv<'a> {
    pub fn new(board: &'a CompiledBoard) -> Self {
        let n = board.n() as usize;
        let mut groups: FxHashMap<(Cell, Cell), Vec<Cell>> = FxHashMap::default();
        for y in 0..board.n() {
            for x in 0..board.n() {
                let v = (x, y);
                for e in board.in_edges(v) {
                    if let Some(sup) = e.dependent {
                        groups.entry((v, sup)).or_default().push(e.u);
                    }
                }
            }
        }
        SubgoalEnv {
            board,
            n,
            groups,
            caches: RefCell::new(Caches {
                final_components: FxHashMap::default(),
                extended: FxHashMap::default(),
                pairs: FxHashMap::default(),
                transient_pairs: FxHashMap::default(),
            }),
        }
    }

    #[inline]
    pub fn board(&self) -> &CompiledBoard {
        self.board
    }

    #[inline]
    fn idx(&self, c: Cell) -> usize {
        c.1 as usize * self.n + c.0 as usize
    }

    /// `GridEnv._has_adjacent_wall` / `_wall_nodes` membership.
    #[inline]
    pub fn has_adjacent_wall(&self, pos: Cell) -> bool {
        self.board.is_wall_node(pos)
    }

    // -- exact / relaxed shortest-path lengths (GridEnv.compute_*) ----------

    /// `compute_exact_shortest_path_length`: independent table, or the
    /// dependent-edge-group variant (min over group tails of table+1).
    pub fn compute_exact(&self, start: Cell, end: Cell, support: Option<Cell>) -> Option<i64> {
        match support {
            None => self.board.independent(start, end),
            Some(sup) => self.group_min(start, end, sup, false),
        }
    }

    /// `compute_relaxed_shortest_path_length`: all-pairs table variant.
    pub fn compute_relaxed(&self, start: Cell, end: Cell, support: Option<Cell>) -> Option<i64> {
        match support {
            None => self.board.all_pairs(start, end),
            Some(sup) => self.group_min(start, end, sup, true),
        }
    }

    fn group_min(&self, start: Cell, end: Cell, sup: Cell, relaxed: bool) -> Option<i64> {
        let tails = self.groups.get(&(end, sup))?;
        let mut best: Option<i64> = None;
        for &u in tails {
            let d = if relaxed {
                self.board.all_pairs(start, u)
            } else {
                self.board.independent(start, u)
            };
            if let Some(d) = d {
                let len = d + 1;
                if best.is_none_or(|b| len < b) {
                    best = Some(len);
                }
            }
        }
        best
    }

    // -- final components ----------------------------------------------------

    /// `precomputed_final_components[goal]`: ancestors of `goal` on the
    /// dependent-edge-free graph (goal itself excluded, as `nx.ancestors`).
    pub fn final_component(&self, goal: Cell) -> Rc<FcSet> {
        if let Some(fc) = self.caches.borrow().final_components.get(&goal) {
            return Rc::clone(fc);
        }
        let mut fc = vec![false; self.n * self.n];
        // Reverse BFS over independent in-edges.
        let mut stack = vec![goal];
        let mut seen = vec![false; self.n * self.n];
        seen[self.idx(goal)] = true;
        while let Some(v) = stack.pop() {
            for e in self.board.in_edges(v) {
                if e.dependent.is_none() && !seen[self.idx(e.u)] {
                    seen[self.idx(e.u)] = true;
                    fc[self.idx(e.u)] = true;
                    stack.push(e.u);
                }
            }
        }
        // goal excluded: fc[goal] never set (seen guard from the start).
        let fc = Rc::new(fc);
        self.caches
            .borrow_mut()
            .final_components
            .insert(goal, Rc::clone(&fc));
        fc
    }

    /// The extended-graph final component
    /// (`_compute_final_component(remove_dependent_edges(get_extended_graph(
    /// support, bottleneck)), bottleneck, is_extended_graph=True,
    /// blocker_pos=support)`), built as a filtered reverse reachability — no
    /// graph copy (DESIGN §5.7):
    /// - nodes removed: `support` plus cells strictly past it on the
    ///   bottleneck's row/column (side away from the bottleneck);
    /// - edges: all independent edges of G, PLUS the in-edges of `bottleneck`
    ///   with `dependent == support` whose main-axis direction component has
    ///   the sign OPPOSITE to `bottleneck - support` (they become independent
    ///   in the extended graph). Same-sign / zero-component edges are removed
    ///   in the extended graph but are dependent anyway, so they never enter
    ///   the dependent-free view.
    pub fn extended_final_component(&self, bottleneck: Cell, support: Cell) -> Rc<FcSet> {
        let key = (bottleneck, support);
        if let Some(fc) = self.caches.borrow().extended.get(&key) {
            return Rc::clone(fc);
        }
        assert_ne!(
            bottleneck, support,
            "extended final component with support == bottleneck \
             (Python raises here; unreachable through plan-built inputs)"
        );
        let nn = self.n * self.n;
        let mut removed = vec![false; nn];
        removed[self.idx(support)] = true;
        // Blocker row/column rule of _compute_final_component. Python
        // computes nodes_to_remove with the row check first, then REASSIGNS
        // it in the column check (both fire only when blocker == goal, which
        // the assert above excludes) — the port keeps one active rule.
        let (bx, by) = (support.0, support.1);
        let (gx, gy) = (bottleneck.0, bottleneck.1);
        let mut mark = |pred: &dyn Fn(Cell) -> bool| {
            for y in 0..self.n as u16 {
                for x in 0..self.n as u16 {
                    if pred((x, y)) {
                        removed[y as usize * self.n + x as usize] = true;
                    }
                }
            }
        };
        if bx == gx {
            // Same column: drop cells past the blocker vertically.
            if by > gy {
                mark(&|c: Cell| c.1 > by && c.0 == bx);
            } else if by < gy {
                mark(&|c: Cell| c.1 < by && c.0 == bx);
            }
        } else if by == gy {
            // Same row: drop cells past the blocker horizontally.
            if bx > gx {
                mark(&|c: Cell| c.0 > bx && c.1 == by);
            } else if bx < gx {
                mark(&|c: Cell| c.0 < bx && c.1 == by);
            }
        }

        // Main-axis sign rule for the bottleneck's own dependent-on-support
        // in-edges (get_extended_graph): opposite sign -> independent.
        let main = (
            bottleneck.0 as i32 - support.0 as i32,
            bottleneck.1 as i32 - support.1 as i32,
        );
        let converts = |u: Cell, v: Cell| -> bool {
            let ev = (v.0 as i32 - u.0 as i32, v.1 as i32 - u.1 as i32);
            if main.0 != 0 {
                ev.0 * main.0 < 0
            } else if main.1 != 0 {
                ev.1 * main.1 < 0
            } else {
                false
            }
        };

        let mut fc = vec![false; nn];
        let mut seen = vec![false; nn];
        seen[self.idx(bottleneck)] = true;
        let mut stack = vec![bottleneck];
        while let Some(v) = stack.pop() {
            for e in self.board.in_edges(v) {
                let usable = match e.dependent {
                    None => true,
                    Some(dep) => v == bottleneck && dep == support && converts(e.u, e.v),
                };
                if usable && !removed[self.idx(e.u)] && !seen[self.idx(e.u)] {
                    seen[self.idx(e.u)] = true;
                    fc[self.idx(e.u)] = true;
                    stack.push(e.u);
                }
            }
        }
        let fc = Rc::new(fc);
        self.caches.borrow_mut().extended.insert(key, Rc::clone(&fc));
        fc
    }

    // -- bottleneck/support pair collection ----------------------------------

    /// `_collect_bottleneck_support_pairs(fc | {goal})` with fc chosen per
    /// `propose_subgoal_states` (plain final component for `support == None`,
    /// extended-graph final component otherwise — identical for group and
    /// non-group keys, DESIGN §5.9). Deterministic first-seen order per the
    /// module contract; the pair SET equals Python's.
    pub fn pairs_for(&self, goal: Cell, support: Option<Cell>) -> Rc<PairList> {
        let key = (goal, support);
        if let Some(p) = self.caches.borrow().pairs.get(&key) {
            return Rc::clone(p);
        }
        let fc = match support {
            None => self.final_component(goal),
            Some(sup) => self.extended_final_component(goal, sup),
        };
        let goal_idx = self.idx(goal);
        let member = |idx: usize| fc[idx] || idx == goal_idx;
        let mut pairs: Vec<(Cell, Cell)> = Vec::new();
        let mut seen: FxHashSet<(Cell, Cell)> = FxHashSet::default();
        for y in 0..self.n as u16 {
            for x in 0..self.n as u16 {
                let v = (x, y);
                if !member(self.idx(v)) {
                    continue;
                }
                for e in self.board.in_edges(v) {
                    if member(self.idx(e.u)) {
                        continue;
                    }
                    if let Some(sup) = e.dependent {
                        if self.has_adjacent_wall(sup) && seen.insert((e.v, sup)) {
                            pairs.push((e.v, sup));
                        }
                    }
                }
            }
        }
        let pairs = Rc::new(pairs);
        self.caches.borrow_mut().pairs.insert(key, Rc::clone(&pairs));
        pairs
    }

    /// Lever B1 (`GridEnv._transient_support_pairs`): the same
    /// final-component crossing rule as [`Self::pairs_for`] with the
    /// wall-adjacency test INVERTED — only support cells WITHOUT an adjacent
    /// wall. Python unions the two sets and iterates the union (set order);
    /// the documented deterministic replacement appends the transient pairs
    /// (in the same cell-index collection order) after the static ones. The
    /// two sets are disjoint (the wall test partitions), so the union SET
    /// equals Python's.
    pub fn transient_pairs_for(&self, goal: Cell, support: Option<Cell>) -> Rc<PairList> {
        let key = (goal, support);
        if let Some(p) = self.caches.borrow().transient_pairs.get(&key) {
            return Rc::clone(p);
        }
        let fc = match support {
            None => self.final_component(goal),
            Some(sup) => self.extended_final_component(goal, sup),
        };
        let goal_idx = self.idx(goal);
        let member = |idx: usize| fc[idx] || idx == goal_idx;
        let mut pairs: Vec<(Cell, Cell)> = Vec::new();
        let mut seen: FxHashSet<(Cell, Cell)> = FxHashSet::default();
        for y in 0..self.n as u16 {
            for x in 0..self.n as u16 {
                let v = (x, y);
                if !member(self.idx(v)) {
                    continue;
                }
                for e in self.board.in_edges(v) {
                    if member(self.idx(e.u)) {
                        continue;
                    }
                    if let Some(sup) = e.dependent {
                        if !self.has_adjacent_wall(sup) && seen.insert((e.v, sup)) {
                            pairs.push((e.v, sup));
                        }
                    }
                }
            }
        }
        let pairs = Rc::new(pairs);
        self.caches
            .borrow_mut()
            .transient_pairs
            .insert(key, Rc::clone(&pairs));
        pairs
    }

    // -- scoring and proposal -------------------------------------------------

    /// `GridEnv.subgoal_score`: independent[bn→goal] + relaxed(target→bn via
    /// the candidate's own support) + relaxed(helper→support); None → 10_000.
    pub fn subgoal_score(&self, sg: &Subgoal) -> i64 {
        let bn_goal = self.board.independent(sg.bottleneck.pos, sg.goal_pos);
        let tgt_bn = self.compute_relaxed(sg.target_robot.pos, sg.bottleneck.pos, Some(sg.support.pos));
        let hlp_sup = self.compute_relaxed(sg.helper.pos, sg.support.pos, None);
        bn_goal.unwrap_or(SUBGOAL_INF) + tgt_bn.unwrap_or(SUBGOAL_INF) + hlp_sup.unwrap_or(SUBGOAL_INF)
    }

    /// `GridEnv.propose_subgoal_states`: (Subgoal, score) per pair × helper.
    /// The bottleneck robot carries the segment mover's color, the support
    /// robot the helper's color; positions on both are the PLANNED cells.
    /// With `transient` (Lever B1) the pair list additionally includes the
    /// wall-less-support pairs, appended after the static ones.
    pub fn propose_subgoal_states(
        &self,
        state: &State,
        support_robot: Option<&RobotAt>,
        transient: bool,
    ) -> Vec<(Subgoal, i64)> {
        let goal = state.target;
        let target_color = state.target_robot.color;
        let sup_key = support_robot.map(|r| r.pos);
        let static_pairs = self.pairs_for(goal, sup_key);
        let transient_pairs = if transient {
            Some(self.transient_pairs_for(goal, sup_key))
        } else {
            None
        };
        let n_pairs =
            static_pairs.len() + transient_pairs.as_ref().map_or(0, |p| p.len());
        let mut results = Vec::with_capacity(n_pairs * state.helpers.len());
        let all_pairs = static_pairs
            .iter()
            .chain(transient_pairs.iter().flat_map(|p| p.iter()));
        for &(bn_pos, sup_pos) in all_pairs {
            for helper in &state.helpers {
                let subgoal = Subgoal {
                    bottleneck: RobotAt { pos: bn_pos, color: target_color },
                    support: RobotAt { pos: sup_pos, color: helper.color },
                    goal_pos: goal,
                    target_robot: state.target_robot,
                    helper: *helper,
                };
                let score = self.subgoal_score(&subgoal);
                results.push((subgoal, score));
            }
        }
        results
    }

    /// `skeleton.heuristics._dependent_supports`: support cells of dependent
    /// in-edges of `pos` — in-edge insertion order, dedup first-seen (Python
    /// returns a set; documented deterministic replacement, DESIGN §5.11).
    pub fn dependent_supports(&self, pos: Cell) -> Vec<Cell> {
        let mut out: Vec<Cell> = Vec::new();
        for e in self.board.in_edges(pos) {
            if let Some(sup) = e.dependent {
                if !out.contains(&sup) {
                    out.push(sup);
                }
            }
        }
        out
    }

    /// `skeleton.heuristics.propose`: raw proposals (with the empty-raw
    /// retry pinned to each dependent support of `goal` × each helper), then
    /// parent_support resolution per candidate in order
    /// [pinned, None, dependent-supports-of-goal] — first exact hit wins.
    /// `b1` threads the Lever B1 transient-support vocabulary
    /// (`heuristics.propose(..., b1=True)` / `propose_b1`).
    pub fn propose(
        &self,
        goal: Cell,
        mover: &RobotAt,
        helpers: &[RobotAt],
        support: Option<&RobotAt>,
        b1: bool,
    ) -> Vec<Candidate> {
        let segment = State {
            target: goal,
            target_robot: *mover,
            helpers: helpers.to_vec(),
        };
        let mut raw = self.propose_subgoal_states(&segment, support, b1);

        // Fallback: goal only reachable via a helper on one of its dependent
        // supports; retry pinned to each. NOTE the Python oddity, ported: the
        // pinned robot's color varies per helper but is unused, so this adds
        // len(helpers) identical copies of each pinned proposal set.
        if raw.is_empty() {
            for sup_pos in self.dependent_supports(goal) {
                for helper in helpers {
                    let pinned = RobotAt { pos: sup_pos, color: helper.color };
                    raw.extend(self.propose_subgoal_states(&segment, Some(&pinned), b1));
                }
            }
        }

        let mut parent_supports: Vec<Option<Cell>> = Vec::new();
        if let Some(s) = support {
            parent_supports.push(Some(s.pos));
        }
        parent_supports.push(None);
        for s in self.dependent_supports(goal) {
            if !parent_supports.contains(&Some(s)) {
                parent_supports.push(Some(s));
            }
        }

        let mut candidates = Vec::with_capacity(raw.len());
        for (subgoal, score) in raw {
            let bn = subgoal.bottleneck.pos;
            for &ps in &parent_supports {
                if self.compute_exact(bn, goal, ps).is_some() {
                    candidates.push(Candidate { subgoal, parent_support: ps, score });
                    break;
                }
            }
        }
        candidates
    }
}
