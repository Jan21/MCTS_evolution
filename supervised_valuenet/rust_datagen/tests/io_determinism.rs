//! Agent D — engine determinism + CLI contract (DESIGN §7/§8).
//!
//! Runs the REAL `datagen` binary (CARGO_BIN_EXE_datagen) on the committed
//! golden selftest corpus:
//! - `--threads 1`, same work file twice -> byte-identical output, results
//!   in work-file order;
//! - `--threads 16` -> equal line multiset (lines may reorder, bytes per
//!   line unchanged);
//! - malformed input (unknown field) -> nonzero exit, no partial silence;
//! - `replay` rejects generation tasks; `boards` writes + reloads sidecars.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn work_file() -> PathBuf {
    crate_dir().join("golden/selftest_work.jsonl")
}

fn tmp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "rust_datagen_io_test_{tag}_{}",
        std::process::id()
    ));
    fs::create_dir_all(&d).unwrap();
    d
}

fn run_datagen(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_datagen"))
        .args(args)
        .output()
        .expect("spawn datagen");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn ids_of(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).expect("result line");
            v["id"].as_str().unwrap().to_string()
        })
        .collect()
}

#[test]
fn threads1_is_byte_reproducible_and_in_work_order() {
    let dir = tmp_dir("t1");
    let out1 = dir.join("out1.jsonl");
    let out2 = dir.join("out2.jsonl");
    for out in [&out1, &out2] {
        let (ok, _, err) = run_datagen(&[
            "run",
            "--work",
            work_file().to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--threads",
            "1",
            "--quiet",
        ]);
        assert!(ok, "run failed: {err}");
    }
    let b1 = fs::read(&out1).unwrap();
    let b2 = fs::read(&out2).unwrap();
    assert_eq!(b1, b2, "--threads 1 must be byte-reproducible");

    let work_ids: Vec<String> = ids_of(&fs::read_to_string(work_file()).unwrap());
    let out_ids = ids_of(std::str::from_utf8(&b1).unwrap());
    assert_eq!(work_ids, out_ids, "--threads 1 must emit results in work-file order");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn threads16_equal_line_multiset() {
    let dir = tmp_dir("t16");
    let out1 = dir.join("seq.jsonl");
    let outn = dir.join("par.jsonl");
    let (ok, _, err) = run_datagen(&[
        "run", "--work", work_file().to_str().unwrap(),
        "--out", out1.to_str().unwrap(), "--threads", "1", "--quiet",
    ]);
    assert!(ok, "{err}");
    let (ok, _, err) = run_datagen(&[
        "run", "--work", work_file().to_str().unwrap(),
        "--out", outn.to_str().unwrap(), "--threads", "16", "--quiet",
    ]);
    assert!(ok, "{err}");
    let mut l1: Vec<String> = fs::read_to_string(&out1).unwrap().lines().map(String::from).collect();
    let mut ln: Vec<String> = fs::read_to_string(&outn).unwrap().lines().map(String::from).collect();
    l1.sort();
    ln.sort();
    assert_eq!(l1, ln, "threaded run must produce the same line multiset");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn malformed_input_exits_nonzero() {
    let dir = tmp_dir("bad");
    // unknown field on an otherwise valid line + a following valid line
    let mut lines: Vec<String> = fs::read_to_string(work_file())
        .unwrap()
        .lines()
        .map(String::from)
        .collect();
    let mut v: serde_json::Value = serde_json::from_str(&lines[1]).unwrap();
    v["mystery_field"] = serde_json::json!(1);
    lines[1] = v.to_string();
    let bad = dir.join("bad.jsonl");
    fs::write(&bad, lines.join("\n")).unwrap();
    let out = dir.join("out.jsonl");
    let (ok, _, err) = run_datagen(&[
        "run", "--work", bad.to_str().unwrap(),
        "--out", out.to_str().unwrap(), "--threads", "1", "--quiet",
    ]);
    assert!(!ok, "malformed input must exit nonzero");
    assert!(err.contains("line 2"), "error should name the line: {err}");
    assert!(err.contains("mystery_field"), "error should name the field: {err}");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn replay_mode_rejects_generation_tasks() {
    let dir = tmp_dir("replay");
    let out = dir.join("out.jsonl");
    let (ok, _, err) = run_datagen(&[
        "replay", "--work", work_file().to_str().unwrap(),
        "--out", out.to_str().unwrap(), "--threads", "1", "--quiet",
    ]);
    assert!(!ok, "replay must reject board/instance tasks");
    assert!(err.contains("not accepted"), "{err}");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn stdout_mode_and_stdin_mode_work() {
    // --out -
    let (ok, stdout, err) = run_datagen(&[
        "run", "--work", work_file().to_str().unwrap(),
        "--out", "-", "--threads", "1", "--quiet",
    ]);
    assert!(ok, "{err}");
    assert_eq!(ids_of(&stdout).len(), 11);

    // --work - (stdin)
    use std::io::Write as _;
    let mut child = Command::new(env!("CARGO_BIN_EXE_datagen"))
        .args(["run", "--work", "-", "--out", "-", "--threads", "2", "--quiet"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let work = fs::read_to_string(work_file()).unwrap();
    child.stdin.take().unwrap().write_all(work.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    assert_eq!(ids_of(&String::from_utf8_lossy(&out.stdout)).len(), 11);
}

#[test]
fn boards_subcommand_writes_and_reloads_sidecars() {
    let dir = tmp_dir("sidecar");
    // pull the board item from the corpus, add sidecar_out
    let corpus = fs::read_to_string(work_file()).unwrap();
    let mut board_item: serde_json::Value = corpus
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
        .find(|v| v["task"] == "board")
        .expect("corpus has a board item");
    board_item["sidecar_out"] = serde_json::json!("env_0.bin");
    board_item["emit"] = serde_json::json!("tables");
    let work = dir.join("boards.jsonl");
    fs::write(&work, board_item.to_string() + "\n").unwrap();
    let manifest = dir.join("manifest.jsonl");
    let (ok, _, err) = run_datagen(&[
        "boards", "--work", work.to_str().unwrap(),
        "--out", manifest.to_str().unwrap(),
        "--sidecar-dir", dir.to_str().unwrap(),
        "--threads", "1", "--quiet",
    ]);
    assert!(ok, "{err}");
    assert!(dir.join("env_0.bin").exists(), "sidecar written");
    let mline: serde_json::Value =
        serde_json::from_str(fs::read_to_string(&manifest).unwrap().lines().next().unwrap())
            .unwrap();
    let ap_sha = mline["all_pairs_sha256"].as_str().unwrap().to_string();

    // an item referencing the sidecar must produce the same table hashes
    let env_id = board_item["board"]["env_id"].clone();
    let by_sidecar = serde_json::json!({
        "task": "board", "id": "via_sidecar",
        "board": {"env_id": env_id, "sidecar": "env_0.bin"},
        "emit": "tables",
    });
    let work2 = dir.join("boards2.jsonl");
    fs::write(&work2, by_sidecar.to_string() + "\n").unwrap();
    let manifest2 = dir.join("manifest2.jsonl");
    let (ok, _, err) = run_datagen(&[
        "boards", "--work", work2.to_str().unwrap(),
        "--out", manifest2.to_str().unwrap(),
        "--sidecar-dir", dir.to_str().unwrap(),
        "--threads", "1", "--quiet",
    ]);
    assert!(ok, "{err}");
    let m2: serde_json::Value =
        serde_json::from_str(fs::read_to_string(&manifest2).unwrap().lines().next().unwrap())
            .unwrap();
    assert_eq!(m2["all_pairs_sha256"].as_str().unwrap(), ap_sha);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn selftest_subcommand_passes() {
    let out = Command::new(env!("CARGO_BIN_EXE_datagen"))
        .args(["selftest", "--threads", "4"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "selftest failed: {stdout}");
    assert!(stdout.contains("selftest PASS"), "{stdout}");
}
