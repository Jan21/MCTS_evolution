//! Gate 3, forward half: Rust move oracle vs the Python golden fixtures
//! (`golden/forward/forward_<config>.json.gz`, dumped by
//! `pyref/gen_fixtures_forward.py` straight from `move_planner/oracle.py`).
//!
//! Every fixture case must be reproduced EXACTLY:
//! - `relaxed_target_dist`: full per-cell field equal (null == unreachable);
//! - `solve`: returned cost equal, INCLUDING capped None-vs-None — the A*
//!   expansion order is mirrored (push-counter tie-break), so cap-boundary
//!   cases (`min_expansions` K vs K-1) must agree bit-for-bit;
//! - `label_trajectory`: records field-for-field, positions and best/legal
//!   move lists in Python's exact LIST order;
//! - instance record streams (label_board semantics with score_candidates):
//!   the full interleaved stream equal record-for-record.

use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::Value;

use rust_datagen::move_oracle::{
    label_instance, label_trajectory, relaxed_target_dist, solve, solve_with_path,
    LabelOutcome, TrajRecord,
};
use rust_datagen::physics::{apply_move, Walls};
use rust_datagen::types::{Cell, Dir, ORACLE_INF};

fn read_json(path: &Path) -> Value {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let mut s = String::new();
    flate2::read::GzDecoder::new(&bytes[..])
        .read_to_string(&mut s)
        .unwrap_or_else(|e| panic!("gunzip {path:?}: {e}"));
    serde_json::from_str(&s).unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
}

fn fixture_files() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("golden/forward");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}"))
        .map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().ends_with(".json.gz"))
        .collect();
    files.sort();
    files
}

fn as_cell(v: &Value) -> Cell {
    (v[0].as_u64().unwrap() as u16, v[1].as_u64().unwrap() as u16)
}

fn as_positions(v: &Value) -> Vec<Cell> {
    v.as_array().unwrap().iter().map(as_cell).collect()
}

fn as_opt_i64(v: &Value) -> Option<i64> {
    if v.is_null() { None } else { Some(v.as_i64().unwrap()) }
}

/// Fixture move list [[slot, dir_idx], ...] == Rust (slot, Dir) list,
/// element for element (exact LIST order).
fn assert_moves_eq(rust: &[(usize, Dir)], fx: &Value, ctx: &str) {
    let fx = fx.as_array().unwrap();
    assert_eq!(rust.len(), fx.len(), "{ctx}: move list length");
    for (k, (&(s, d), v)) in rust.iter().zip(fx.iter()).enumerate() {
        assert_eq!(s, v[0].as_u64().unwrap() as usize, "{ctx}: move {k} slot");
        assert_eq!(d as u64, v[1].as_u64().unwrap(), "{ctx}: move {k} dir");
    }
}

fn assert_traj_record_eq(rust: &TrajRecord, fx: &Value, ctx: &str) {
    assert_eq!(rust.positions, as_positions(&fx["positions"]), "{ctx}: positions");
    assert_eq!(rust.cost_to_go, fx["cost_to_go"].as_i64().unwrap(), "{ctx}: cost_to_go");
    assert_moves_eq(&rust.best_moves, &fx["best_moves"], &format!("{ctx}: best_moves"));
    assert_moves_eq(&rust.legal_moves, &fx["legal_moves"], &format!("{ctx}: legal_moves"));
    assert_eq!(rust.depth, fx["depth"].as_i64().unwrap(), "{ctx}: depth");
    assert_eq!(rust.full, fx["full"].as_bool().unwrap(), "{ctx}: full");
}

struct Counts {
    hdist: usize,
    solve: usize,
    traj: usize,
    traj_records: usize,
    instances: usize,
    instance_records: usize,
}

fn check_fixture(doc: &Value, name: &str) -> Counts {
    let n = doc["n"].as_u64().unwrap() as u16;
    let walls: Vec<Walls> = doc["boards"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| {
            let gd: Vec<String> = b["grid_data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            Walls::from_grid_data(&gd, n)
        })
        .collect();

    // -- (a) relaxed_target_dist ------------------------------------------
    let hdist_cases = doc["hdist_cases"].as_array().unwrap();
    for (i, case) in hdist_cases.iter().enumerate() {
        let ctx = format!("{name}: hdist case {i}");
        let w = &walls[case["board"].as_u64().unwrap() as usize];
        let got = relaxed_target_dist(as_cell(&case["target"]), w);
        let want = case["dist"].as_array().unwrap();
        assert_eq!(got.len(), want.len(), "{ctx}: field size");
        for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(*g, as_opt_i64(w), "{ctx}: cell idx {j}");
        }
    }

    // -- (b) solve ----------------------------------------------------------
    let solve_cases = doc["solve_cases"].as_array().unwrap();
    for (i, case) in solve_cases.iter().enumerate() {
        let ctx = format!("{name}: solve case {i}");
        let bi = case["board"].as_u64().unwrap() as usize;
        let w = &walls[bi];
        let pos = as_positions(&case["positions"]);
        let ti = case["target_idx"].as_u64().unwrap() as usize;
        let tgt = as_cell(&case["target"]);
        let max_exp = case["max_expansions"].as_i64().unwrap();
        let cost_cap = as_opt_i64(&case["cost_cap"]).unwrap_or(ORACLE_INF);
        let want = as_opt_i64(&case["cost"]);
        let hd = relaxed_target_dist(tgt, w);
        let got = solve(&pos, ti, tgt, w, &hd, max_exp, cost_cap);
        assert_eq!(got, want, "{ctx}: cost (cap {cost_cap}, max_exp {max_exp})");

        // uncapped solved cases: cost IS d_star, and the reconstructed path
        // must replay move-for-move under the physics.
        if let Some(cost) = want {
            if case["cost_cap"].is_null() {
                if let Some(d_star) = as_opt_i64(&case["d_star"]) {
                    assert_eq!(cost, d_star, "{ctx}: cost vs d_star");
                }
                let (c2, path) =
                    solve_with_path(&pos, ti, tgt, w, &hd, max_exp, ORACLE_INF).unwrap();
                assert_eq!(c2, cost, "{ctx}: want_path cost");
                assert_eq!(path.len() as i64, cost + 1, "{ctx}: path length");
                assert_eq!(path[0].positions, pos, "{ctx}: path start");
                assert_eq!(path[0].action, None, "{ctx}: start action");
                for k in 1..path.len() {
                    let (slot, dir) = path[k].action.unwrap();
                    let next = apply_move(&path[k - 1].positions, slot, dir, w)
                        .unwrap_or_else(|| panic!("{ctx}: path step {k} is a no-op"));
                    assert_eq!(next, path[k].positions, "{ctx}: path step {k}");
                }
                assert_eq!(path.last().unwrap().positions[ti], tgt, "{ctx}: path ends on goal");
            }
        }
    }

    // -- (c) label_trajectory -------------------------------------------------
    let traj_cases = doc["trajectory_cases"].as_array().unwrap();
    let mut traj_records = 0usize;
    for (i, case) in traj_cases.iter().enumerate() {
        let ctx = format!("{name}: trajectory case {i}");
        let w = &walls[case["board"].as_u64().unwrap() as usize];
        let pos = as_positions(&case["positions"]);
        let ti = case["target_idx"].as_u64().unwrap() as usize;
        let tgt = as_cell(&case["target"]);
        let max_exp = case["max_expansions"].as_i64().unwrap();
        let full_policy = case["full_policy"].as_bool().unwrap();
        let cap = as_opt_i64(&case["full_policy_max_ctg"]);
        let hd = relaxed_target_dist(tgt, w);
        let got = label_trajectory(&pos, ti, tgt, w, &hd, max_exp, full_policy, cap);
        let want = &case["result"];
        match (got, want.is_null()) {
            (None, true) => {}
            (None, false) => panic!("{ctx}: Rust None, Python solved"),
            (Some(_), true) => panic!("{ctx}: Rust solved, Python None"),
            (Some((d, recs)), false) => {
                assert_eq!(d, want["d_star"].as_i64().unwrap(), "{ctx}: d_star");
                let wrecs = want["records"].as_array().unwrap();
                assert_eq!(recs.len(), wrecs.len(), "{ctx}: record count");
                for (k, (r, wr)) in recs.iter().zip(wrecs.iter()).enumerate() {
                    assert_traj_record_eq(r, wr, &format!("{ctx} record {k}"));
                }
                traj_records += recs.len();
            }
        }
    }

    // -- (d) instance record streams ------------------------------------------
    let inst_cases = doc["instance_cases"].as_array().unwrap();
    let mut instance_records = 0usize;
    for (i, case) in inst_cases.iter().enumerate() {
        let ctx = format!("{name}: instance case {i}");
        let w = &walls[case["board"].as_u64().unwrap() as usize];
        let env_id = case["env_id"].as_i64().unwrap();
        let pos = as_positions(&case["positions"]);
        let ti = case["target_idx"].as_u64().unwrap() as usize;
        let tgt = as_cell(&case["target"]);
        let max_exp = case["max_expansions"].as_i64().unwrap();
        let full_policy = case["full_policy"].as_bool().unwrap();
        let cap = as_opt_i64(&case["full_policy_max_ctg"]);
        let score = case["score_candidates"].as_bool().unwrap();
        let got = label_instance(env_id, &pos, ti, tgt, w, max_exp, full_policy, cap, score);
        match case["status"].as_str().unwrap() {
            "relaxed_unreachable" => {
                assert_eq!(got, LabelOutcome::RelaxedUnreachable, "{ctx}: status")
            }
            "unsolved" => assert_eq!(got, LabelOutcome::Unsolved, "{ctx}: status"),
            "solved" => {
                let LabelOutcome::Solved { d_star, records } = got else {
                    panic!("{ctx}: expected solved, got {got:?}");
                };
                assert_eq!(d_star, case["d_star"].as_i64().unwrap(), "{ctx}: d_star");
                let wrecs = case["records"].as_array().unwrap();
                assert_eq!(records.len(), wrecs.len(), "{ctx}: record stream length");
                for (k, (r, wr)) in records.iter().zip(wrecs.iter()).enumerate() {
                    let rv = serde_json::to_value(r).unwrap();
                    assert_eq!(&rv, wr, "{ctx}: record {k}");
                }
                instance_records += records.len();
            }
            s => panic!("{ctx}: unknown fixture status {s:?}"),
        }
    }

    Counts {
        hdist: hdist_cases.len(),
        solve: solve_cases.len(),
        traj: traj_cases.len(),
        traj_records,
        instances: inst_cases.len(),
        instance_records,
    }
}

/// The committed forward corpus: 5 configs (16x16 r4/r6/r8, 24x24 r4,
/// 32x32 r4), ~20 hdist fields, 200 solve cases (incl. cost_cap and
/// expansion-boundary), 40 trajectories, 21 instance streams.
#[test]
fn gate3_forward_golden() {
    let files = fixture_files();
    assert_eq!(files.len(), 5, "expected 5 forward fixture files, found {}", files.len());
    let (mut h, mut s, mut t, mut tr, mut i, mut ir) = (0, 0, 0, 0, 0, 0);
    for path in &files {
        let doc = read_json(path);
        let name = doc["config"].as_str().unwrap().to_string();
        let c = check_fixture(&doc, &name);
        println!(
            "PASS {name}: {} hdist, {} solve, {} traj ({} records), {} instances ({} records)",
            c.hdist, c.solve, c.traj, c.traj_records, c.instances, c.instance_records
        );
        h += c.hdist;
        s += c.solve;
        t += c.traj;
        tr += c.traj_records;
        i += c.instances;
        ir += c.instance_records;
    }
    assert!(h >= 20 && s >= 200 && t >= 40 && i >= 20, "fixture corpus shrank: {h}/{s}/{t}/{i}");
    println!(
        "gate 3 forward PASS: {h} hdist fields, {s} solve cases, {t} trajectories \
         ({tr} records), {i} instance streams ({ir} records)"
    );
}

/// Quick micro-bench (report material, not a formal benchmark — Agent D owns
/// those). Labels 100 instances at 16x16 with score_candidates, like
/// generate.py's inner loop. Run:
///   cargo test --release --test gate3_forward -- --ignored bench --nocapture
#[test]
#[ignore]
fn bench_label_100_instances_16x16() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("golden/forward/forward_g16r4.json.gz");
    let doc = read_json(&path);
    let n = doc["n"].as_u64().unwrap() as u16;
    let boards: Vec<Walls> = doc["boards"]
        .as_array()
        .unwrap()
        .iter()
        .take(4) // real boards only (skip the sealed synthetic)
        .map(|b| {
            let gd: Vec<String> = b["grid_data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            Walls::from_grid_data(&gd, n)
        })
        .collect();
    // Deterministic test-local LCG (the engine itself has no RNG).
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    let mut next = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as usize
    };
    let r = 4usize;
    let t0 = std::time::Instant::now();
    let (mut solved, mut attempts, mut records) = (0usize, 0usize, 0usize);
    while solved < 100 {
        attempts += 1;
        let w = &boards[next() % boards.len()];
        // distinct cells for R robots + target
        let mut cells: Vec<Cell> = Vec::new();
        while cells.len() < r + 1 {
            let c = ((next() % 16) as u16, (next() % 16) as u16);
            if !cells.contains(&c) {
                cells.push(c);
            }
        }
        let target = cells.pop().unwrap();
        let ti = next() % r;
        match label_instance(0, &cells, ti, target, w, 40_000, true, Some(6), true) {
            LabelOutcome::Solved { records: recs, .. } => {
                solved += 1;
                records += recs.len();
            }
            _ => continue,
        }
    }
    let dt = t0.elapsed();
    println!(
        "bench: 100 solved instances ({attempts} attempts, {records} records, \
         score_candidates=true) in {dt:?} ({:.1} inst/s, {:.0} rec/s)",
        100.0 / dt.as_secs_f64(),
        records as f64 / dt.as_secs_f64()
    );
}
