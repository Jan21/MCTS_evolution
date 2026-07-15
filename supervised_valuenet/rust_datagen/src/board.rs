//! Compiled board: slide graph, all-pairs tables, wall nodes, canonical table
//! hashes and the bincode sidecar (DESIGN §5.3–5.5, §3).
//!
//! Ground truth: `nn/gen_grids.py::build_graph` / `::all_pairs` /
//! `::independent_paths` and the re-weighting block of `GridEnv.from_env`
//! (GridEnv.py) plus `GridEnv.__init__`'s `_wall_nodes`.
//!
//! Ordering contracts (DESIGN §8 — these leak into candidate enumeration
//! downstream, do not "clean them up"):
//! - Out-edges per node are stored in `build_graph` insertion order: cells
//!   y-major then x, directions up/down/left/right; per direction the
//!   independent edge (cell→stop, weight 1) FIRST, then one dependent edge per
//!   intermediate cell in ray order (nearest first), stop excluded,
//!   `dependent` = intermediate + step.
//! - In-edges per node are stored in global edge-creation order — identical to
//!   NetworkX `G.in_edges(v)` iteration order for a graph built in that order.
//!   (Verified: the stock pickled graphs have the same in-edge order even
//!   though their per-node OUT-edge order differs — an older builder emitted
//!   the stop edge last within a ray; in-edge order is ray-interleaving-proof.)

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::physics::{slide, Walls};
use crate::types::{Cell, DIRECTIONS};

/// One slide-graph edge. `dependent = None` → independent edge (pure wall
/// stop); `Some(support)` → the cell a helper robot must occupy so the slide
/// stops at `v`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    pub u: Cell,
    pub v: Cell,
    pub weight: i64,
    pub dependent: Option<Cell>,
}

/// Board compiled from `(n, grid_data, dependent_edge_weight)`.
///
/// Tables are flat `Vec<Option<i64>>` of size (n*n)^2 indexed by
/// `src_idx * n*n + dst_idx` with `idx = y*n + x`; `None` = unreachable.
#[derive(Clone, Serialize, Deserialize)]
pub struct CompiledBoard {
    n: u16,
    grid_data: Vec<String>,
    dependent_edge_weight: i64,
    walls: Walls,
    out_edges: Vec<Vec<Edge>>,
    in_edges: Vec<Vec<Edge>>,
    all_pairs: Vec<Option<i64>>,
    independent: Vec<Option<i64>>,
    wall_nodes: Vec<bool>,
    n_edges: usize,
}

impl CompiledBoard {
    /// Build the slide graph and both distance tables.
    ///
    /// The graph mirrors `build_graph` with dependent edges created directly
    /// at `dependent_edge_weight` — equivalent to building at weight 2 and
    /// re-weighting as `GridEnv.from_env` does (only edges carrying a
    /// `dependent` attr are re-weighted; independent edges stay at 1).
    pub fn compile(n: u16, grid_data: &[String], dependent_edge_weight: i64) -> Self {
        let walls = Walls::from_grid_data(grid_data, n);
        let nn = n as usize * n as usize;
        let idx = |c: Cell| c.1 as usize * n as usize + c.0 as usize;

        let mut out_edges: Vec<Vec<Edge>> = vec![Vec::new(); nn];
        let mut in_edges: Vec<Vec<Edge>> = vec![Vec::new(); nn];
        let mut n_edges = 0usize;
        // build_graph order: y-major cells, then dict order up/down/left/right.
        for y in 0..n {
            for x in 0..n {
                let c = (x, y);
                for &d in DIRECTIONS.iter() {
                    let stop = slide(c, d, &[], &walls);
                    if stop == c {
                        continue; // zero slide -> no edges
                    }
                    let (dx, dy) = d.step();
                    let step = |p: Cell| -> Cell {
                        (
                            (p.0 as i32 + dx) as u16,
                            (p.1 as i32 + dy) as u16,
                        )
                    };
                    // Independent edge first (G.add_edge(c, stop, weight=1)),
                    // then dependents in ray order, stop excluded (path[:-1]).
                    let mut push = |e: Edge| {
                        out_edges[idx(e.u)].push(e);
                        in_edges[idx(e.v)].push(e);
                        n_edges += 1;
                    };
                    push(Edge { u: c, v: stop, weight: 1, dependent: None });
                    let mut cur = step(c);
                    while cur != stop {
                        push(Edge {
                            u: c,
                            v: cur,
                            weight: dependent_edge_weight,
                            dependent: Some(step(cur)),
                        });
                        cur = step(cur);
                    }
                }
            }
        }

        // _wall_nodes (GridEnv.__init__): any in-edge of weight == 1 AFTER
        // re-weighting. NOT "independent heads": at dependent_edge_weight == 1
        // dependent heads genuinely qualify (DESIGN §5.5).
        let wall_nodes: Vec<bool> = in_edges
            .iter()
            .map(|es| es.iter().any(|e| e.weight == 1))
            .collect();

        // CSR adjacency (target idx, weight) for the Dijkstra sweeps, built
        // straight from the out-edge lists (no per-node Vec intermediate).
        let mut offs_all: Vec<u32> = Vec::with_capacity(nn + 1);
        let mut csr_all: Vec<(u32, i64)> = Vec::with_capacity(n_edges);
        let mut offs_ind: Vec<u32> = Vec::with_capacity(nn + 1);
        let mut csr_ind: Vec<(u32, i64)> = Vec::new();
        offs_all.push(0);
        offs_ind.push(0);
        for es in &out_edges {
            for e in es {
                csr_all.push((idx(e.v) as u32, e.weight));
                if e.dependent.is_none() {
                    csr_ind.push((idx(e.v) as u32, e.weight));
                }
            }
            offs_all.push(csr_all.len() as u32);
            offs_ind.push(csr_ind.len() as u32);
        }
        let all_pairs = par_all_sources_dijkstra(&offs_all, &csr_all, nn);
        let independent = par_all_sources_dijkstra(&offs_ind, &csr_ind, nn);

        CompiledBoard {
            n,
            grid_data: grid_data.to_vec(),
            dependent_edge_weight,
            walls,
            out_edges,
            in_edges,
            all_pairs,
            independent,
            wall_nodes,
            n_edges,
        }
    }

    #[inline]
    pub fn n(&self) -> u16 {
        self.n
    }

    #[inline]
    pub fn grid_data(&self) -> &[String] {
        &self.grid_data
    }

    #[inline]
    pub fn dependent_edge_weight(&self) -> i64 {
        self.dependent_edge_weight
    }

    #[inline]
    pub fn walls(&self) -> &Walls {
        &self.walls
    }

    #[inline]
    pub fn n_edges(&self) -> usize {
        self.n_edges
    }

    #[inline]
    fn node_idx(&self, c: Cell) -> usize {
        c.1 as usize * self.n as usize + c.0 as usize
    }

    /// Out-edges of `cell` in `build_graph` insertion order
    /// (== NetworkX `G.edges(cell)` order for a freshly built graph).
    #[inline]
    pub fn out_edges(&self, cell: Cell) -> &[Edge] {
        &self.out_edges[self.node_idx(cell)]
    }

    /// In-edges of `cell` in edge-creation order
    /// (== NetworkX `G.in_edges(cell, data=True)` order).
    #[inline]
    pub fn in_edges(&self, cell: Cell) -> &[Edge] {
        &self.in_edges[self.node_idx(cell)]
    }

    /// Re-weighted all-pairs distance (`GridEnv.from_env` recomputation);
    /// `None` = unreachable.
    #[inline]
    pub fn all_pairs(&self, src: Cell, dst: Cell) -> Option<i64> {
        self.all_pairs[self.node_idx(src) * self.n as usize * self.n as usize
            + self.node_idx(dst)]
    }

    /// Independent-edges-only distance (`nn/gen_grids.independent_paths`);
    /// `None` = unreachable.
    #[inline]
    pub fn independent(&self, src: Cell, dst: Cell) -> Option<i64> {
        self.independent[self.node_idx(src) * self.n as usize * self.n as usize
            + self.node_idx(dst)]
    }

    /// `GridEnv._wall_nodes`: any in-edge of weight 1 after re-weighting.
    #[inline]
    pub fn is_wall_node(&self, cell: Cell) -> bool {
        self.wall_nodes[self.node_idx(cell)]
    }

    /// All edges sorted by (ux, uy, vx, vy) — the golden-dump comparison order
    /// and the `emit:"graph"` export order (DESIGN §3).
    pub fn all_edges_sorted(&self) -> Vec<Edge> {
        let mut edges: Vec<Edge> = self
            .out_edges
            .iter()
            .flat_map(|es| es.iter().copied())
            .collect();
        edges.sort_by_key(|e| (e.u.0, e.u.1, e.v.0, e.v.1));
        edges
    }

    /// Canonical SHA-256 of the re-weighted all-pairs table (DESIGN §3).
    pub fn all_pairs_sha256(&self) -> String {
        self.table_sha256(&self.all_pairs)
    }

    /// Canonical SHA-256 of the independent-paths table (DESIGN §3).
    pub fn independent_sha256(&self) -> String {
        self.table_sha256(&self.independent)
    }

    /// SHA-256 over `"sx,sy,tx,ty,D\n"` for every ordered pair in
    /// lexicographic (s, t) order — s and t as (x, y) tuples, so x-major —
    /// with D the integer distance or the literal `None`. Identical to the
    /// Python dumper (`pyref/dump_reference.py`).
    fn table_sha256(&self, table: &[Option<i64>]) -> String {
        let n = self.n as usize;
        let mut hasher = Sha256::new();
        let mut line = String::with_capacity(24);
        for sx in 0..n {
            for sy in 0..n {
                let s_idx = sy * n + sx;
                for tx in 0..n {
                    for ty in 0..n {
                        let t_idx = ty * n + tx;
                        line.clear();
                        match table[s_idx * n * n + t_idx] {
                            Some(d) => {
                                let _ = writeln!(line, "{sx},{sy},{tx},{ty},{d}");
                            }
                            None => {
                                let _ = writeln!(line, "{sx},{sy},{tx},{ty},None");
                            }
                        }
                        hasher.update(line.as_bytes());
                    }
                }
            }
        }
        hex_string(&hasher.finalize())
    }

    /// Write a bincode sidecar of this compiled board.
    pub fn save_sidecar(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut w = BufWriter::new(File::create(path)?);
        bincode::serialize_into(&mut w, self)?;
        Ok(())
    }

    /// Load a compiled board from a bincode sidecar.
    pub fn load_sidecar(path: &Path) -> anyhow::Result<CompiledBoard> {
        let r = BufReader::new(File::open(path)?);
        Ok(bincode::deserialize_from(r)?)
    }

    /// Content-address of this board for caching (DESIGN §3): the compiled
    /// board is fully determined by (env_id, sha256(grid_data), weight).
    pub fn cache_key(&self, env_id: u64) -> String {
        cache_key(env_id, &self.grid_data, self.dependent_edge_weight)
    }
}

/// SHA-256 of `grid_data` (cells joined by ','; cells only contain NESW).
pub fn grid_data_sha256(grid_data: &[String]) -> String {
    let mut hasher = Sha256::new();
    for (i, cell) in grid_data.iter().enumerate() {
        if i > 0 {
            hasher.update(b",");
        }
        hasher.update(cell.as_bytes());
    }
    hex_string(&hasher.finalize())
}

/// Content-address for a compiled board: `(env_id, sha256(grid_data), weight)`.
pub fn cache_key(env_id: u64, grid_data: &[String], dependent_edge_weight: i64) -> String {
    format!(
        "{env_id}_{}_{dependent_edge_weight}",
        grid_data_sha256(grid_data)
    )
}

fn hex_string(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Largest edge weight the Dial (bucket-queue) Dijkstra handles; datagen
/// weights are only 1 and `dependent_edge_weight` (2 in practice).
const DIAL_MAX_W: i64 = 8;

/// Shortest paths from every source over the CSR adjacency (`offs`/`edges`);
/// row `s` of the result is the distance vector from source `s`. Sources run
/// in parallel via rayon (still capped by the engine's pinned pool); the
/// output is order-independent (each source writes its own disjoint chunk,
/// and shortest-path DISTANCES are unique regardless of settle order).
///
/// When every weight is in `1..=DIAL_MAX_W` (the datagen case: weights 1 and
/// dependent_edge_weight = 2) a Dial/bucket queue over a CSR adjacency
/// replaces the binary heap — same metric, identical distances (gate 2's
/// golden table hashes prove it), ~2-4x faster per source. Any other weight
/// (a nonstandard `dependent_edge_weight` in a work item) falls back to the
/// original binary-heap Dijkstra.
fn par_all_sources_dijkstra(
    offs: &[u32],
    edges: &[(u32, i64)],
    nn: usize,
) -> Vec<Option<i64>> {
    let mut flat: Vec<Option<i64>> = vec![None; nn * nn];
    let maxw = edges.iter().map(|&(_, w)| w).max().unwrap_or(1);
    let minw = edges.iter().map(|&(_, w)| w).min().unwrap_or(1);
    if minw >= 1 && maxw <= DIAL_MAX_W {
        let nb = maxw as usize + 1; // bucket window: distances live in [d, d+maxw]
        // 8-byte edge records halve the sweep's memory traffic (the edge
        // array is re-read once per source).
        let edges32: Vec<(u32, u32)> = edges.iter().map(|&(v, w)| (v, w as u32)).collect();
        flat.par_chunks_mut(nn).enumerate().for_each_init(
            || (vec![i32::MAX; nn], vec![Vec::<u32>::new(); nb]),
            |(dist, buckets), (s, out)| {
                dial_into(offs, &edges32, s, nb, dist, buckets, out);
            },
        );
    } else {
        flat.par_chunks_mut(nn)
            .enumerate()
            .for_each(|(s, out)| out.copy_from_slice(&dijkstra(offs, edges, nn, s)));
    }
    flat
}

/// One Dial (bucket queue) single-source pass, writing `out[v] = dist(src,v)`.
/// `dist`/`buckets` are reusable per-thread scratch (rayon `for_each_init`);
/// buckets are fully drained on return. Correctness of the circular window:
/// a relaxation from distance `d` pushes `nd <= d + maxw < d + nb`, so the
/// bucket being drained (`d % nb`) is never pushed into, and every entry in
/// flight lives in `[d, d + maxw]`. Stale entries (node re-pushed with a
/// smaller tentative distance) are skipped via the `dist[u] == d` check —
/// the same lazy-deletion rule as the heap version.
fn dial_into(
    offs: &[u32],
    edges: &[(u32, u32)],
    src: usize,
    nb: usize,
    dist: &mut [i32],
    buckets: &mut [Vec<u32>],
    out: &mut [Option<i64>],
) {
    for d in dist.iter_mut() {
        *d = i32::MAX;
    }
    dist[src] = 0;
    buckets[0].push(src as u32);
    let mut remaining = 1usize;
    let mut d = 0i32;
    while remaining > 0 {
        let b = d as usize % nb;
        while let Some(u) = buckets[b].pop() {
            remaining -= 1;
            let u = u as usize;
            if dist[u] != d {
                continue; // stale entry
            }
            for &(v, w) in &edges[offs[u] as usize..offs[u + 1] as usize] {
                let nd = d + w as i32;
                if nd < dist[v as usize] {
                    dist[v as usize] = nd;
                    buckets[nd as usize % nb].push(v);
                    remaining += 1;
                }
            }
        }
        d += 1;
    }
    for (o, &dv) in out.iter_mut().zip(dist.iter()) {
        *o = if dv == i32::MAX { None } else { Some(dv as i64) };
    }
}

fn dijkstra(offs: &[u32], edges: &[(u32, i64)], nn: usize, src: usize) -> Vec<Option<i64>> {
    let mut dist: Vec<Option<i64>> = vec![None; nn];
    let mut heap: BinaryHeap<Reverse<(i64, u32)>> = BinaryHeap::new();
    dist[src] = Some(0);
    heap.push(Reverse((0, src as u32)));
    while let Some(Reverse((d, u))) = heap.pop() {
        if dist[u as usize] != Some(d) {
            continue; // stale entry
        }
        for &(v, w) in &edges[offs[u as usize] as usize..offs[u as usize + 1] as usize] {
            let nd = d + w;
            if dist[v as usize].is_none_or(|cur| nd < cur) {
                dist[v as usize] = Some(nd);
                heap.push(Reverse((nd, v)));
            }
        }
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Dir;

    /// 4x4 board: border walls plus 'S' at (1,0) (wall below (1,0)) and
    /// 'E' at (2,2) (wall right of (2,2)).
    fn board4() -> CompiledBoard {
        let g: Vec<String> = [
            "NW", "NS", "N", "NE", //
            "W", "", "", "E", //
            "W", "", "E", "E", //
            "SW", "S", "S", "SE",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        CompiledBoard::compile(4, &g, 2)
    }

    #[test]
    fn out_edge_insertion_order_matches_build_graph() {
        let b = board4();
        // (0,0): up no slide; down -> stop (0,3), intermediates (0,1),(0,2);
        // left none; right -> stop (3,0), intermediates (1,0),(2,0).
        let e: Vec<(Cell, i64, Option<Cell>)> =
            b.out_edges((0, 0)).iter().map(|e| (e.v, e.weight, e.dependent)).collect();
        assert_eq!(
            e,
            vec![
                ((0, 3), 1, None),          // independent FIRST (down)
                ((0, 1), 2, Some((0, 2))),  // dependents in ray order
                ((0, 2), 2, Some((0, 3))),
                ((3, 0), 1, None),          // independent FIRST (right)
                ((1, 0), 2, Some((2, 0))),
                ((2, 0), 2, Some((3, 0))),
            ]
        );
    }

    #[test]
    fn in_edge_order_is_global_creation_order() {
        let b = board4();
        // In-edges of (1,0) come from cells in y-major creation order.
        let heads: Vec<Cell> = b.in_edges((1, 0)).iter().map(|e| e.u).collect();
        let mut sorted_by_creation = heads.clone();
        sorted_by_creation.sort_by_key(|c| (c.1, c.0)); // y-major
        // Every u appears at most once per head here, and creation order is
        // y-major cell order (directions nested inside a single cell).
        assert_eq!(heads, sorted_by_creation);
    }

    #[test]
    fn wall_below_1_0_stops_slides() {
        let b = board4();
        // 'S' at (1,0): sliding down from (1,0) cannot move.
        assert_eq!(slide((1, 0), Dir::Down, &[], b.walls()), (1, 0));
        // Sliding up from (1,3) stops at (1,1).
        assert_eq!(slide((1, 3), Dir::Up, &[], b.walls()), (1, 1));
    }

    #[test]
    fn tables_match_hand_computed_distances() {
        let b = board4();
        // Independent: (0,0) -> (3,0) is one wall slide.
        assert_eq!(b.independent((0, 0), (3, 0)), Some(1));
        // All-pairs: (0,0) -> (1,0) only via the dependent edge (weight 2).
        assert_eq!(b.all_pairs((0, 0), (1, 0)), Some(2));
        assert_eq!(b.all_pairs((0, 0), (0, 0)), Some(0));
    }

    #[test]
    fn wall_nodes_have_weight_one_in_edges() {
        let b = board4();
        for y in 0..4u16 {
            for x in 0..4u16 {
                let expect = b.in_edges((x, y)).iter().any(|e| e.weight == 1);
                assert_eq!(b.is_wall_node((x, y)), expect);
            }
        }
        // (1,1) is a wall stop (upward slides from column 1 stop below the
        // 'S' wall at (1,0)).
        assert!(b.is_wall_node((1, 1)));
    }

    #[test]
    fn dependent_weight_one_makes_dependent_heads_wall_nodes() {
        // DESIGN §5.5: at weight 1 the Python set includes dependent heads.
        let g: Vec<String> = [
            "NW", "NS", "N", "NE", //
            "W", "", "", "E", //
            "W", "", "E", "E", //
            "SW", "S", "S", "SE",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let b1 = CompiledBoard::compile(4, &g, 1);
        let b2 = CompiledBoard::compile(4, &g, 2);
        let count1 = (0..16).filter(|&i| b1.wall_nodes[i]).count();
        let count2 = (0..16).filter(|&i| b2.wall_nodes[i]).count();
        assert!(count1 > count2, "weight-1 wall nodes {count1} !> {count2}");
    }

    /// The Dial (bucket queue) path must produce EXACTLY the heap Dijkstra's
    /// distances on real board adjacencies — including weight = DIAL_MAX_W
    /// (largest bucket window) and the dependent-free independent subgraph.
    #[test]
    fn dial_matches_heap_dijkstra_on_board_adjacency() {
        for weight in [1i64, 2, DIAL_MAX_W] {
            let g: Vec<String> = [
                "NW", "NS", "N", "NE", //
                "W", "", "", "E", //
                "W", "", "E", "E", //
                "SW", "S", "S", "SE",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            let b = CompiledBoard::compile(4, &g, weight);
            let nn = 16usize;
            let idx = |c: Cell| c.1 as usize * 4 + c.0 as usize;
            for dependent_only in [false, true] {
                let mut offs: Vec<u32> = vec![0];
                let mut csr: Vec<(u32, i64)> = Vec::new();
                for i in 0..nn {
                    for e in b.out_edges[i]
                        .iter()
                        .filter(|e| !dependent_only || e.dependent.is_none())
                    {
                        csr.push((idx(e.v) as u32, e.weight));
                    }
                    offs.push(csr.len() as u32);
                }
                let flat = par_all_sources_dijkstra(&offs, &csr, nn); // dial path
                for s in 0..nn {
                    let row = dijkstra(&offs, &csr, nn, s); // heap reference
                    assert_eq!(
                        &flat[s * nn..(s + 1) * nn],
                        &row[..],
                        "weight {weight} dependent_only {dependent_only} src {s}"
                    );
                }
            }
        }
    }

    #[test]
    fn sidecar_roundtrip_preserves_everything() {
        let b = board4();
        let dir = std::env::temp_dir().join(format!(
            "rust_datagen_sidecar_test_{}",
            std::process::id()
        ));
        let path = dir.join(format!("{}.bin", b.cache_key(0)));
        b.save_sidecar(&path).unwrap();
        let b2 = CompiledBoard::load_sidecar(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(b.all_edges_sorted(), b2.all_edges_sorted());
        assert_eq!(b.all_pairs_sha256(), b2.all_pairs_sha256());
        assert_eq!(b.independent_sha256(), b2.independent_sha256());
        assert_eq!(b.wall_nodes, b2.wall_nodes);
        assert_eq!(b.n_edges(), b2.n_edges());
    }
}
