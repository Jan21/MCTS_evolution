//! Gate 5 — the 12-instance residual-defect regression corpus (Agent E2).
//!
//! `golden/regression/regression12.jsonl` holds one `replay_backward_decision`
//! line per residual-defect bench instance (two classes of 6: a helper color
//! recruited for a second leaf; a claimed parent_support cell no plan node
//! occupies), built by `pyref/gen_regression.py` from the archived failing
//! plans. Each line carries `python_labels` from the current Python solver.
//!
//! This test runs the REAL binary's `replay` on the corpus and asserts
//! (a) per-candidate label parity with `python_labels` (ctg + rejected),
//! (b) the flagged defect candidate — the one whose acceptance would
//!     recreate the archived defect — is REJECTED, per
//!     `regression12.meta.json`.

use std::path::PathBuf;
use std::process::Command;

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn gate5_regression12_rejection_parity() {
    let dump = crate_dir().join("golden/regression/regression12.jsonl");
    let meta = crate_dir().join("golden/regression/regression12.meta.json");
    let out = std::env::temp_dir().join(format!(
        "rust_datagen_gate5_{}.jsonl",
        std::process::id()
    ));

    let status = Command::new(env!("CARGO_BIN_EXE_datagen"))
        .args([
            "replay",
            "--work",
            dump.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--threads",
            "2",
            "--quiet",
        ])
        .status()
        .expect("spawn datagen");
    assert!(status.success(), "datagen replay failed on the corpus");

    // results by id
    let mut results = std::collections::HashMap::new();
    for line in std::fs::read_to_string(&out).unwrap().lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        results.insert(v["id"].as_str().unwrap().to_string(), v);
    }

    // (a) per-candidate parity with python_labels
    let mut n_lines = 0usize;
    let mut n_labels = 0usize;
    for line in std::fs::read_to_string(&dump).unwrap().lines() {
        let d: serde_json::Value = serde_json::from_str(line).unwrap();
        let id = d["id"].as_str().unwrap();
        let py = d["python_labels"].as_array().unwrap();
        let res = results
            .get(id)
            .unwrap_or_else(|| panic!("no result for {id}"));
        let eng = res["labels"].as_array().unwrap();
        assert_eq!(eng.len(), py.len(), "{id}: label count");
        for (i, (p, e)) in py.iter().zip(eng.iter()).enumerate() {
            assert_eq!(
                p["ctg"], e["ctg"],
                "{id} cand {i}: ctg py={} eng={}",
                p["ctg"], e["ctg"]
            );
            assert_eq!(
                p["rejected"], e["rejected"],
                "{id} cand {i}: rejected py={} eng={}",
                p["rejected"], e["rejected"]
            );
            n_labels += 1;
        }
        n_lines += 1;
    }
    assert_eq!(n_lines, 12, "corpus must hold exactly 12 decisions");

    // (b) the defect candidate is rejected on the Rust side
    let m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&meta).unwrap()).unwrap();
    let items = m["items"].as_array().unwrap();
    assert_eq!(items.len(), 12);
    for it in items {
        let id = it["id"].as_str().unwrap();
        let di = it["defect_candidate_index"].as_u64().unwrap() as usize;
        let lab = &results[id]["labels"][di];
        assert_eq!(
            lab["rejected"],
            serde_json::Value::Bool(true),
            "{id}: defect candidate #{di} ({}) must be rejected",
            it["defect_class"]
        );
    }
    println!(
        "gate 5 regression PASS: 12 decisions, {n_labels} candidate labels, \
         12 defect candidates rejected"
    );
    let _ = std::fs::remove_file(&out);
}
