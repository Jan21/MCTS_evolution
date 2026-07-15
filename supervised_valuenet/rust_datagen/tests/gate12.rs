//! Gates 1–2 (DESIGN §9): Rust CompiledBoard vs the Python golden corpus.
//!
//! For every golden board: edges must match the dump EXACTLY (count,
//! endpoints, weight, dependent), the canonical table hashes must be equal,
//! and every dumped slide case must agree.
//!
//! Weight normalisation: stock pickles store dependent edges at weight 100;
//! `GridEnv.from_env` re-weights them to `dependent_edge_weight` before any
//! use. The dump keeps the raw pickled weight, so the expected weight here is
//! `dependent_edge_weight` for dependent edges and the raw weight otherwise —
//! exactly the from_env rule. Fresh dumps are built at weight 2, making the
//! rule a no-op for them.
//!
//! `golden_corpus` runs the committed mini-corpus (hermetic, no network, path
//! relative to the crate). `gate_full` (#[ignore]) runs a regenerated corpus
//! pointed at by RUST_DATAGEN_GATE12_DIR — see pyref/verify_full_gate12.sh.

use std::io::Read;
use std::path::{Path, PathBuf};

use rust_datagen::board::CompiledBoard;
use rust_datagen::physics::slide;
use rust_datagen::types::{Cell, Dir};

/// Shared 128-core box: cap the rayon pool at 16 threads (task brief).
fn init_rayon() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build_global()
            .ok();
    });
}

fn read_json(path: &Path) -> serde_json::Value {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let text = if path.extension().is_some_and(|e| e == "gz") {
        let mut s = String::new();
        flate2::read::GzDecoder::new(&bytes[..])
            .read_to_string(&mut s)
            .unwrap_or_else(|e| panic!("gunzip {path:?}: {e}"));
        s
    } else {
        String::from_utf8(bytes).unwrap()
    };
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
}

fn board_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}"))
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let name = p.file_name().unwrap().to_string_lossy();
            name.ends_with(".json") || name.ends_with(".json.gz")
        })
        .collect();
    files.sort();
    files
}

fn as_cell(v: &serde_json::Value) -> Cell {
    (v[0].as_u64().unwrap() as u16, v[1].as_u64().unwrap() as u16)
}

struct BoardCounts {
    edges: usize,
    slides: usize,
    tables: usize,
}

/// Run every gate 1–2 check for one dumped board; panics on any mismatch.
fn check_board(doc: &serde_json::Value, ctx: &str) -> BoardCounts {
    let n = doc["n"].as_u64().unwrap() as u16;
    let weight = doc["dependent_edge_weight"].as_i64().unwrap();
    let grid_data: Vec<String> = doc["grid_data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let board = CompiledBoard::compile(n, &grid_data, weight);

    // -- gate 1: node and edge parity ------------------------------------
    assert_eq!(
        doc["n_nodes"].as_u64().unwrap() as usize,
        n as usize * n as usize,
        "{ctx}: node count"
    );
    let dumped = doc["edges"].as_array().unwrap();
    assert_eq!(
        doc["n_edges"].as_u64().unwrap() as usize,
        dumped.len(),
        "{ctx}: n_edges vs dumped rows"
    );
    let rust_edges = board.all_edges_sorted();
    assert_eq!(rust_edges.len(), dumped.len(), "{ctx}: edge count");
    for (i, (re, row)) in rust_edges.iter().zip(dumped.iter()).enumerate() {
        let row = row.as_array().unwrap();
        let u: Cell = (
            row[0].as_u64().unwrap() as u16,
            row[1].as_u64().unwrap() as u16,
        );
        let v: Cell = (
            row[2].as_u64().unwrap() as u16,
            row[3].as_u64().unwrap() as u16,
        );
        let raw_w = row[4].as_i64().unwrap();
        let dependent: Option<Cell> = if row[5].is_null() {
            assert!(row[6].is_null(), "{ctx}: edge {i} half-null dependent");
            None
        } else {
            Some((
                row[5].as_u64().unwrap() as u16,
                row[6].as_u64().unwrap() as u16,
            ))
        };
        // GridEnv.from_env re-weighting rule (see module docs).
        let expect_w = if dependent.is_some() { weight } else { raw_w };
        assert_eq!(re.u, u, "{ctx}: edge {i} tail");
        assert_eq!(re.v, v, "{ctx}: edge {i} head");
        assert_eq!(re.dependent, dependent, "{ctx}: edge {i} dependent");
        assert_eq!(re.weight, expect_w, "{ctx}: edge {i} weight");
    }

    // -- gate 2: canonical table hashes -----------------------------------
    let mut tables = 0;
    if let Some(h) = doc["all_pairs_sha256"].as_str() {
        assert_eq!(board.all_pairs_sha256(), h, "{ctx}: all_pairs hash");
        tables += 1;
    }
    if let Some(h) = doc["independent_sha256"].as_str() {
        assert_eq!(board.independent_sha256(), h, "{ctx}: independent hash");
        tables += 1;
    }

    // -- gate 1: slide cases ----------------------------------------------
    let mut slides = 0;
    if let Some(cases) = doc["slides"].as_array() {
        for (i, case) in cases.iter().enumerate() {
            let pos = as_cell(&case["pos"]);
            let dir = Dir::from_name(case["dir"].as_str().unwrap())
                .unwrap_or_else(|| panic!("{ctx}: slide {i} bad dir"));
            let blockers: Vec<Cell> = case["blockers"]
                .as_array()
                .unwrap()
                .iter()
                .map(as_cell)
                .collect();
            let expect = as_cell(&case["stop"]);
            let got = slide(pos, dir, &blockers, board.walls());
            assert_eq!(
                got, expect,
                "{ctx}: slide {i} pos {pos:?} dir {dir:?} blockers {blockers:?}"
            );
            slides += 1;
        }
    }

    BoardCounts {
        edges: dumped.len(),
        slides,
        tables,
    }
}

fn run_corpus(dir: &Path) {
    init_rayon();
    let files = board_files(dir);
    assert!(!files.is_empty(), "no golden boards in {dir:?}");
    let (mut boards, mut edges, mut slides, mut tables) = (0usize, 0usize, 0usize, 0usize);
    for path in &files {
        let doc = read_json(path);
        let ctx = doc["board_id"].as_str().unwrap_or("?").to_string();
        let c = check_board(&doc, &ctx);
        boards += 1;
        edges += c.edges;
        slides += c.slides;
        tables += c.tables;
        println!(
            "PASS {ctx}: {} edges, {} slides, {} table hashes",
            c.edges, c.slides, c.tables
        );
    }
    println!(
        "gate 1-2 corpus PASS: {boards} boards, {edges} edges, {slides} slides, {tables} table hashes ({dir:?})"
    );
}

/// Committed mini-corpus: 12 boards (4x16 incl. 2 stock, 4x24, 4x32), each
/// with graph dump + table hashes + 200 slide cases.
#[test]
fn golden_corpus() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("golden");
    let files = board_files(&dir);
    assert_eq!(
        files.len(),
        12,
        "mini-corpus must hold exactly 12 boards, found {}",
        files.len()
    );
    run_corpus(&dir);
}

/// Full gate 1–2 sweep over a regenerated corpus
/// (pyref/verify_full_gate12.sh). Run with:
///   RUST_DATAGEN_GATE12_DIR=... cargo test --release -- --ignored gate_full --nocapture
#[test]
#[ignore]
fn gate_full() {
    let dir = std::env::var("RUST_DATAGEN_GATE12_DIR")
        .expect("set RUST_DATAGEN_GATE12_DIR to the dumped corpus directory");
    run_corpus(Path::new(&dir));
}
