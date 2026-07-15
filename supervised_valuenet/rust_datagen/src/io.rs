//! JSONL work-item streaming, dispatch, and the shared board cache (Agent D,
//! DESIGN §3/§7/§8).
//!
//! One JSON object per input line. Every item carries `"task"` and `"id"`;
//! the engine echoes `id` in the result line so the caller can reassociate
//! results (multi-threaded runs may reorder OUTPUT LINES only — each line's
//! bytes are deterministic, DESIGN §8). Unknown fields are a hard error
//! (`deny_unknown_fields` on every item struct); the THREE documented
//! auxiliary fields of pyref dumps — `max_candidates` on
//! `replay_backward_decision`, `full_policy_max_ctg` and `python_labels` on
//! the replay items — are explicitly modeled and accepted (DESIGN §3;
//! candidate lists in dumps are already truncated and are NEVER re-truncated
//! here).
//!
//! Malformed input (unparseable line, unknown field, inconsistent board,
//! bad robot color, out-of-range index) aborts the whole run with a nonzero
//! exit and the offending line number — fail loud, no silent skips (DESIGN
//! §7). Solver failures are ordinary result rows (`status` fields).

use std::collections::HashMap;
use std::io::{BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context as _};
use rayon::iter::{ParallelBridge, ParallelIterator};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::board::{cache_key, CompiledBoard};
use crate::move_oracle::{label_instance, replay_forward_state, LabelOutcome, MoveRecord};
use crate::physics::Walls;
use crate::subgoal::rollout::{
    parse_replay_decision, parse_robot, replay_decision, rollout, BackwardRecord,
    CandidateLabel, RolloutStatus,
};
use crate::subgoal::{AStar, State, SubgoalEnv};
use crate::types::Cell;

/// Default deterministic rollout budget (`budget.solver_iters`, DESIGN §3)
/// applied when a `backward_rollout` item carries no explicit budget.
///
/// Calibration (Agent D, rust_datagen/BENCH.md §8): iteration counts are
/// engine-independent (same algorithm/caps), so pairing Python wall time
/// with the engine's iteration count on the SAME instance converts the
/// production 120 s SIGALRM into iterations. Over 67 paired instances
/// across g16r4/g16r6/g16r8/g24r4/g32r4
/// (pyref/cache/bench/*_bwd_calibration.jsonl) the highest observed rate
/// was 2,632 iterations per Python-second (median 350), so 120 s of
/// Python work corresponds to at most ~316k iterations; x10 headroom is
/// ~3.2M, rounded UP to 10M (>= 31x the worst-case conversion, >280x the
/// median). Cross-check: the full g16r6 backward regeneration (25,200
/// rollouts) never exceeded 83,085 iterations — 120x below this default —
/// and had zero budget_exhausted results.
pub const DEFAULT_SOLVER_ITERS: u64 = 10_000_000;

// ---------------------------------------------------------------------------
// board reference + cache
// ---------------------------------------------------------------------------

/// DESIGN §3 board description: inline (`n` + `grid_data`) or by sidecar
/// reference — exactly one of the two forms.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoardRef {
    pub env_id: i64,
    #[serde(default)]
    pub n: Option<u16>,
    #[serde(default)]
    pub grid_data: Option<Vec<String>>,
    #[serde(default)]
    pub sidecar: Option<String>,
}

impl BoardRef {
    fn validate(&self) -> anyhow::Result<()> {
        match (&self.grid_data, &self.sidecar) {
            (Some(gd), None) => {
                let n = self.n.ok_or_else(|| anyhow!("inline board needs \"n\""))?;
                if gd.len() != (n as usize) * (n as usize) {
                    bail!("grid_data has {} cells, expected n*n = {}", gd.len(), (n as usize).pow(2));
                }
                Ok(())
            }
            (None, Some(_)) => Ok(()),
            (Some(_), Some(_)) => bail!("board carries BOTH grid_data and sidecar"),
            (None, None) => bail!("board carries neither grid_data nor sidecar"),
        }
    }
}

type BoardSlot = Arc<OnceLock<Result<Arc<CompiledBoard>, String>>>;

/// Compiled-board cache, content-hash keyed (DESIGN §3): compilation happens
/// once per (env_id, sha256(grid_data), weight) within a run; concurrent
/// requests for the same board block on the same `OnceLock` until the first
/// compilation finishes. Failures are cached too (same board fails the same
/// way; the run aborts on the first one anyway).
pub struct BoardCache {
    map: Mutex<HashMap<String, BoardSlot>>,
    sidecar_dir: Option<PathBuf>,
}

impl BoardCache {
    pub fn new(sidecar_dir: Option<PathBuf>) -> Self {
        BoardCache { map: Mutex::new(HashMap::new()), sidecar_dir }
    }

    fn sidecar_path(&self, p: &str) -> PathBuf {
        let path = Path::new(p);
        match (&self.sidecar_dir, path.is_relative()) {
            (Some(dir), true) => dir.join(path),
            _ => path.to_path_buf(),
        }
    }

    /// Compile (or sidecar-load) the referenced board at `weight`, caching it.
    pub fn resolve(&self, b: &BoardRef, weight: i64) -> anyhow::Result<Arc<CompiledBoard>> {
        b.validate()?;
        let key = match (&b.grid_data, &b.sidecar) {
            (Some(gd), None) => cache_key(b.env_id as u64, gd, weight),
            (None, Some(p)) => format!("sidecar:{p}|w{weight}"),
            _ => unreachable!("validated"),
        };
        let slot: BoardSlot = {
            let mut map = self.map.lock().unwrap();
            map.entry(key).or_default().clone()
        };
        let res = slot.get_or_init(|| match (&b.grid_data, &b.sidecar) {
            (Some(gd), None) => {
                Ok(Arc::new(CompiledBoard::compile(b.n.unwrap(), gd, weight)))
            }
            (None, Some(p)) => {
                let path = self.sidecar_path(p);
                let cb = CompiledBoard::load_sidecar(&path)
                    .map_err(|e| format!("sidecar {}: {e}", path.display()))?;
                if cb.dependent_edge_weight() != weight {
                    return Err(format!(
                        "sidecar {} was compiled at dependent_edge_weight {}, item wants {}",
                        path.display(), cb.dependent_edge_weight(), weight
                    ));
                }
                if let Some(n) = b.n {
                    if cb.n() != n {
                        return Err(format!(
                            "sidecar {} is a {}x{} board, item says n={}",
                            path.display(), cb.n(), cb.n(), n
                        ));
                    }
                }
                Ok(Arc::new(cb))
            }
            _ => unreachable!("validated"),
        });
        res.clone().map_err(|e| anyhow!(e))
    }

    /// Walls only (forward tasks): inline boards skip table compilation.
    fn resolve_walls(&self, b: &BoardRef) -> anyhow::Result<Walls> {
        b.validate()?;
        match (&b.grid_data, &b.sidecar) {
            (Some(gd), None) => Ok(Walls::from_grid_data(gd, b.n.unwrap())),
            (None, Some(_)) => {
                // Weight is irrelevant for walls; use the default key so a
                // sidecar shared with backward items is loaded only once.
                let cb = self.resolve(b, 2)?;
                Ok(Walls::from_grid_data(cb.grid_data(), cb.n()))
            }
            _ => unreachable!("validated"),
        }
    }
}

// ---------------------------------------------------------------------------
// work items (DESIGN §3) — strict schemas
// ---------------------------------------------------------------------------

fn d_weight() -> i64 {
    2
}
fn d_max_iters() -> u64 {
    4_000 // scaling.backward_label AStar setting
}
fn d_max_frontier() -> usize {
    40_000 // scaling.backward_label AStar setting
}
fn d_max_expansions() -> i64 {
    40_000 // move_planner.generate default
}
fn d_true() -> bool {
    true
}

/// task: "board" — precompute + optional sidecar/table export.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkBoard {
    pub task: String,
    pub id: String,
    pub board: BoardRef,
    #[serde(default = "d_weight")]
    pub dependent_edge_weight: i64,
    #[serde(default)]
    pub sidecar_out: Option<String>,
    /// "none" (default) | "graph" | "tables"
    #[serde(default)]
    pub emit: Option<String>,
    #[serde(default)]
    pub full_tables: bool,
}

/// task: "forward_instance".
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkForwardInstance {
    pub task: String,
    pub id: String,
    pub board: BoardRef,
    pub robots: Vec<Cell>,
    pub target_idx: usize,
    pub target: Cell,
    #[serde(default = "d_max_expansions")]
    pub max_expansions: i64,
    #[serde(default = "d_true")]
    pub full_policy: bool,
    #[serde(default)]
    pub full_policy_max_ctg: Option<i64>,
    #[serde(default)]
    pub score_candidates: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub solver_iters: u64,
}

/// task: "backward_rollout".
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkBackwardRollout {
    pub task: String,
    pub id: String,
    pub board: BoardRef,
    pub target: Cell,
    /// `[[x, y], "Color"]`
    pub target_robot: Value,
    /// list of `[[x, y], "Color"]`
    pub helpers: Value,
    #[serde(default)]
    pub max_candidates: Option<usize>,
    #[serde(default = "d_max_iters")]
    pub max_iters: u64,
    #[serde(default = "d_max_frontier")]
    pub max_frontier: usize,
    #[serde(default = "d_weight")]
    pub dependent_edge_weight: i64,
    #[serde(default)]
    pub budget: Option<Budget>,
    /// Parity experiments only (DESIGN §3): wall-clock caps are
    /// non-deterministic, so THIS engine refuses items that set it — use
    /// `budget.solver_iters`.
    #[serde(default)]
    pub timeout_s: Option<f64>,
    /// Datagen always runs with `max_final_component_distance = None`
    /// (DESIGN §5.6); anything non-null is rejected.
    #[serde(default)]
    pub max_final_component_distance: Option<Value>,
}

/// task: "replay_backward_decision" (gate 3; produced by pyref dumpers).
/// `max_candidates` and `python_labels` are the DOCUMENTED auxiliary dump
/// fields (DESIGN §3): informational / differ-side data. The candidate list
/// in the dump is already truncated — it is never re-truncated here.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkReplayBackward {
    pub task: String,
    pub id: String,
    pub board: BoardRef,
    #[serde(default = "d_weight")]
    pub dependent_edge_weight: i64,
    #[serde(default = "d_max_iters")]
    pub max_iters: u64,
    #[serde(default = "d_max_frontier")]
    pub max_frontier: usize,
    pub state: Value,
    pub plan: Value,
    pub open_edge: Value,
    pub candidates: Value,
    #[serde(default)]
    pub max_candidates: Option<Value>,
    #[serde(default)]
    pub python_labels: Option<Value>,
}

/// task: "replay_forward_state" (gate 3). `full_policy_max_ctg` and
/// `python_labels` are the documented auxiliary dump fields (DESIGN §3).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkReplayForward {
    pub task: String,
    pub id: String,
    pub board: BoardRef,
    pub positions: Vec<Cell>,
    pub target_idx: usize,
    pub target: Cell,
    #[serde(default = "d_max_expansions")]
    pub max_expansions: i64,
    #[serde(default)]
    pub full_policy_max_ctg: Option<Value>,
    #[serde(default)]
    pub python_labels: Option<Value>,
}

// ---------------------------------------------------------------------------
// result envelopes
// ---------------------------------------------------------------------------

/// `[ux, uy, vx, vy, weight, sx|null, sy|null]` (DESIGN §3).
type EdgeRow = (u16, u16, u16, u16, i64, Option<u16>, Option<u16>);

#[derive(Serialize)]
struct TablesOut {
    /// Flattened src-major: entry `s_idx * n*n + t_idx`, `idx = y*n + x`.
    all_pairs: Vec<Option<i64>>,
    independent: Vec<Option<i64>>,
}

#[derive(Serialize)]
struct BoardResult {
    id: String,
    task: &'static str,
    env_id: i64,
    n: u16,
    n_nodes: usize,
    n_edges: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    edges: Option<Vec<EdgeRow>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    all_pairs_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    independent_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tables: Option<TablesOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sidecar: Option<String>,
}

#[derive(Serialize)]
struct ForwardInstanceResult {
    id: String,
    status: &'static str, // "solved" | "relaxed_unreachable" | "unsolved"
    #[serde(skip_serializing_if = "Option::is_none")]
    d_star: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    records: Option<Vec<MoveRecord>>,
}

#[derive(Serialize)]
struct BackwardRolloutResult {
    id: String,
    status: &'static str, // "ok" | "empty" | "budget_exhausted"
    #[serde(skip_serializing_if = "Option::is_none")]
    records: Option<Vec<BackwardRecord>>,
    /// Total solve_plan iterations consumed (budget audit / calibration).
    iters: u64,
}

#[derive(Serialize)]
struct ReplayBackwardResult {
    id: String,
    labels: Vec<CandidateLabel>,
}

#[derive(Serialize)]
struct ReplayForwardResult {
    id: String,
    cost_to_go: Option<i64>,
    optimal_moves: Vec<[i64; 2]>,
    legal_moves: Vec<[i64; 2]>,
}

fn moves_to_pairs(moves: &[(usize, crate::types::Dir)]) -> Vec<[i64; 2]> {
    moves.iter().map(|&(s, d)| [s as i64, d as i64]).collect()
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

/// Which tasks a subcommand admits (DESIGN §7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// `datagen run`: every task type.
    Run,
    /// `datagen replay`: replay_backward_decision / replay_forward_state only.
    Replay,
    /// `datagen boards`: board tasks only.
    Boards,
}

impl Mode {
    fn allows(self, task: &str) -> bool {
        match self {
            Mode::Run => true,
            Mode::Replay => matches!(task, "replay_backward_decision" | "replay_forward_state"),
            Mode::Boards => task == "board",
        }
    }
}

struct Ctx {
    cache: BoardCache,
    mode: Mode,
}

/// Process one work line into one result line (compact JSON, no newline).
fn process_line(line: &str, ctx: &Ctx) -> anyhow::Result<String> {
    let v: Value = serde_json::from_str(line).context("unparseable JSON")?;
    let task = v
        .get("task")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing \"task\" field"))?
        .to_string();
    if !ctx.mode.allows(&task) {
        bail!("task {task:?} is not accepted by this subcommand ({:?} mode)", ctx.mode);
    }
    match task.as_str() {
        "board" => run_board(serde_json::from_value(v).context("board item")?, ctx),
        "forward_instance" => {
            run_forward(serde_json::from_value(v).context("forward_instance item")?, ctx)
        }
        "backward_rollout" => {
            run_backward(serde_json::from_value(v).context("backward_rollout item")?, ctx)
        }
        "replay_backward_decision" => run_replay_backward(
            serde_json::from_value(v).context("replay_backward_decision item")?,
            ctx,
        ),
        "replay_forward_state" => run_replay_forward(
            serde_json::from_value(v).context("replay_forward_state item")?,
            ctx,
        ),
        other => bail!("unknown task {other:?}"),
    }
}

fn run_board(item: WorkBoard, ctx: &Ctx) -> anyhow::Result<String> {
    let emit = item.emit.as_deref().unwrap_or("none");
    if !matches!(emit, "none" | "graph" | "tables") {
        bail!("bad emit {emit:?} (want none|graph|tables)");
    }
    let board = ctx.cache.resolve(&item.board, item.dependent_edge_weight)?;
    let n = board.n();
    let nn = n as usize * n as usize;

    let mut sidecar = None;
    if let Some(out) = &item.sidecar_out {
        let path = ctx.cache.sidecar_path(out);
        board
            .save_sidecar(&path)
            .with_context(|| format!("writing sidecar {}", path.display()))?;
        sidecar = Some(path.display().to_string());
    }

    let edges = (emit == "graph").then(|| {
        board
            .all_edges_sorted()
            .into_iter()
            .map(|e| {
                let (sx, sy) = match e.dependent {
                    Some(c) => (Some(c.0), Some(c.1)),
                    None => (None, None),
                };
                (e.u.0, e.u.1, e.v.0, e.v.1, e.weight, sx, sy)
            })
            .collect::<Vec<EdgeRow>>()
    });
    let (ap_sha, ind_sha, tables) = if emit == "tables" {
        let full = item.full_tables.then(|| {
            let mut ap = Vec::with_capacity(nn * nn);
            let mut ind = Vec::with_capacity(nn * nn);
            for s_idx in 0..nn {
                let s = ((s_idx % n as usize) as u16, (s_idx / n as usize) as u16);
                for t_idx in 0..nn {
                    let t = ((t_idx % n as usize) as u16, (t_idx / n as usize) as u16);
                    ap.push(board.all_pairs(s, t));
                    ind.push(board.independent(s, t));
                }
            }
            TablesOut { all_pairs: ap, independent: ind }
        });
        (
            Some(board.all_pairs_sha256()),
            Some(board.independent_sha256()),
            full,
        )
    } else {
        (None, None, None)
    };

    let res = BoardResult {
        id: item.id,
        task: "board",
        env_id: item.board.env_id,
        n,
        n_nodes: nn,
        n_edges: board.n_edges(),
        edges,
        all_pairs_sha256: ap_sha,
        independent_sha256: ind_sha,
        tables,
        sidecar,
    };
    Ok(serde_json::to_string(&res)?)
}

fn check_forward_envelope(robots: usize, target_idx: usize, n: u16) -> anyhow::Result<()> {
    if robots == 0 || robots > 10 {
        bail!("forward tasks support 1..=10 robots, got {robots}");
    }
    if target_idx >= robots {
        bail!("target_idx {target_idx} out of range for {robots} robots");
    }
    if n > 64 {
        bail!("forward tasks support n <= 64, got {n}");
    }
    Ok(())
}

fn run_forward(item: WorkForwardInstance, ctx: &Ctx) -> anyhow::Result<String> {
    let walls = ctx.cache.resolve_walls(&item.board)?;
    check_forward_envelope(item.robots.len(), item.target_idx, walls.n())?;
    let res = match label_instance(
        item.board.env_id,
        &item.robots,
        item.target_idx,
        item.target,
        &walls,
        item.max_expansions,
        item.full_policy,
        item.full_policy_max_ctg,
        item.score_candidates,
    ) {
        LabelOutcome::Solved { d_star, records } => ForwardInstanceResult {
            id: item.id,
            status: "solved",
            d_star: Some(d_star),
            records: Some(records),
        },
        LabelOutcome::RelaxedUnreachable => ForwardInstanceResult {
            id: item.id,
            status: "relaxed_unreachable",
            d_star: None,
            records: None,
        },
        LabelOutcome::Unsolved => ForwardInstanceResult {
            id: item.id,
            status: "unsolved",
            d_star: None,
            records: None,
        },
    };
    Ok(serde_json::to_string(&res)?)
}

fn run_backward(item: WorkBackwardRollout, ctx: &Ctx) -> anyhow::Result<String> {
    if item.timeout_s.is_some() {
        bail!(
            "timeout_s is a non-deterministic parity knob this engine does not \
             implement; use budget.solver_iters (DESIGN §3/§8)"
        );
    }
    if let Some(mfcd) = &item.max_final_component_distance {
        if !mfcd.is_null() {
            bail!("max_final_component_distance must be null in datagen (DESIGN §5.6)");
        }
    }
    let board = ctx.cache.resolve(&item.board, item.dependent_edge_weight)?;
    let state = State {
        target: item.target,
        target_robot: parse_robot(&item.target_robot).context("target_robot")?,
        helpers: item
            .helpers
            .as_array()
            .ok_or_else(|| anyhow!("helpers is not a list"))?
            .iter()
            .map(parse_robot)
            .collect::<anyhow::Result<_>>()
            .context("helpers")?,
    };
    let solver = AStar {
        max_iters: item.max_iters,
        max_frontier: item.max_frontier,
        ..AStar::default()
    };
    let budget = item.budget.map(|b| b.solver_iters).unwrap_or(DEFAULT_SOLVER_ITERS);
    let env = SubgoalEnv::new(&board);
    let out = rollout(
        &env,
        &state,
        &solver,
        item.board.env_id,
        item.max_candidates,
        Some(budget),
    );
    let (status, records) = match out.status {
        RolloutStatus::Ok => ("ok", Some(out.records)),
        RolloutStatus::Empty => ("empty", None),
        RolloutStatus::BudgetExhausted => ("budget_exhausted", None),
    };
    let res = BackwardRolloutResult {
        id: item.id,
        status,
        records,
        iters: out.iters_used,
    };
    Ok(serde_json::to_string(&res)?)
}

fn run_replay_backward(item: WorkReplayBackward, ctx: &Ctx) -> anyhow::Result<String> {
    let board = ctx.cache.resolve(&item.board, item.dependent_edge_weight)?;
    // Reassemble the §3 shape for the shared parser (defaults applied above).
    let payload = json!({
        "state": item.state,
        "plan": item.plan,
        "open_edge": item.open_edge,
        "candidates": item.candidates,
        "max_iters": item.max_iters,
        "max_frontier": item.max_frontier,
    });
    let decision = parse_replay_decision(&payload)?;
    let env = SubgoalEnv::new(&board);
    let labels = replay_decision(&env, &decision)?;
    let res = ReplayBackwardResult { id: item.id, labels };
    Ok(serde_json::to_string(&res)?)
}

fn run_replay_forward(item: WorkReplayForward, ctx: &Ctx) -> anyhow::Result<String> {
    let walls = ctx.cache.resolve_walls(&item.board)?;
    check_forward_envelope(item.positions.len(), item.target_idx, walls.n())?;
    let (ctg, optimal, legal) = replay_forward_state(
        &item.positions,
        item.target_idx,
        item.target,
        &walls,
        item.max_expansions,
    );
    let res = ReplayForwardResult {
        id: item.id,
        cost_to_go: ctg,
        optimal_moves: moves_to_pairs(&optimal),
        legal_moves: moves_to_pairs(&legal),
    };
    Ok(serde_json::to_string(&res)?)
}

// ---------------------------------------------------------------------------
// streaming engine
// ---------------------------------------------------------------------------

pub struct EngineOpts {
    /// Worker threads. 1 = sequential, byte-reproducible, results in
    /// work-file order (DESIGN §8). >1 sizes an explicit rayon pool; result
    /// LINES may reorder, bytes per line are unchanged.
    pub threads: usize,
    pub mode: Mode,
    /// Base dir for relative sidecar paths (both loads and `sidecar_out`).
    pub sidecar_dir: Option<PathBuf>,
    /// Progress lines to stderr every ~5 s.
    pub progress: bool,
    /// Flush the output after every result line (stdout co-process mode).
    pub flush_each: bool,
    /// Known item count (for ETA); None when reading a stream.
    pub total_hint: Option<u64>,
}

pub struct RunStats {
    pub items: u64,
}

struct Progress {
    done: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Progress {
    fn start(total: Option<u64>, enabled: bool) -> Progress {
        let done = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let handle = enabled.then(|| {
            let done = done.clone();
            let stop = stop.clone();
            std::thread::spawn(move || {
                let t0 = Instant::now();
                loop {
                    for _ in 0..50 {
                        if stop.load(Ordering::Relaxed) {
                            return;
                        }
                        std::thread::sleep(Duration::from_millis(100));
                    }
                    let d = done.load(Ordering::Relaxed);
                    let dt = t0.elapsed().as_secs_f64();
                    let rate = d as f64 / dt.max(1e-9);
                    match total {
                        Some(t) if rate > 0.0 => {
                            let eta = (t.saturating_sub(d)) as f64 / rate;
                            eprintln!(
                                "[datagen] {d}/{t} items, {rate:.1} items/s, ETA {eta:.0}s"
                            );
                        }
                        _ => eprintln!("[datagen] {d} items done, {rate:.1} items/s"),
                    }
                }
            })
        });
        Progress { done, stop, handle }
    }

    fn finish(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Stream work items from `input` to result lines on `output`.
///
/// Reads line-by-line (never the whole file). Aborts with an error naming
/// the line number on any malformed item.
pub fn run_work<R, W>(input: R, output: W, opts: &EngineOpts) -> anyhow::Result<RunStats>
where
    R: BufRead + Send,
    W: Write + Send,
{
    let ctx = Ctx {
        cache: BoardCache::new(opts.sidecar_dir.clone()),
        mode: opts.mode,
    };
    let writer = Mutex::new(BufWriter::new(output));
    let progress = Progress::start(opts.total_hint, opts.progress);
    let done = progress.done.clone();

    let result: anyhow::Result<()> = if opts.threads <= 1 {
        // Sequential: results in work-file order, byte-reproducible. Runs
        // INSIDE a 1-thread rayon pool so any rayon parallelism nested in
        // the engine (the board-table Dijkstra in `CompiledBoard::compile`
        // uses par_iter) is pinned to this one thread instead of escaping
        // to the global pool — `--threads N` is a hard cap on compute
        // threads for the whole process (shared-box discipline, and it
        // keeps single-core benchmark numbers honest).
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .context("building rayon pool")?;
        pool.install(|| -> anyhow::Result<()> {
            for (i, line) in input.lines().enumerate() {
                let line = line.context("reading work input")?;
                if line.trim().is_empty() {
                    continue;
                }
                let res = process_line(&line, &ctx)
                    .with_context(|| format!("work item at line {}", i + 1))?;
                let mut w = writer.lock().unwrap();
                writeln!(w, "{res}")?;
                if opts.flush_each {
                    w.flush()?;
                }
                done.fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        })
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(opts.threads)
            .build()
            .context("building rayon pool")?;
        let first_err: Mutex<Option<anyhow::Error>> = Mutex::new(None);
        let abort = AtomicBool::new(false);
        let record_err = |i: usize, e: anyhow::Error| {
            abort.store(true, Ordering::Relaxed);
            let mut g = first_err.lock().unwrap();
            // Keep the first-recorded error for a stable message.
            if g.is_none() {
                *g = Some(e.context(format!("work item at line {}", i + 1)));
            }
        };
        pool.install(|| {
            input.lines().enumerate().par_bridge().for_each(|(i, line)| {
                if abort.load(Ordering::Relaxed) {
                    return;
                }
                let out = line
                    .context("reading work input")
                    .and_then(|l| {
                        if l.trim().is_empty() {
                            Ok(None)
                        } else {
                            process_line(&l, &ctx).map(Some)
                        }
                    });
                match out {
                    Ok(None) => {}
                    Ok(Some(res)) => {
                        let mut w = writer.lock().unwrap();
                        let wr = writeln!(w, "{res}").and_then(|_| {
                            if opts.flush_each {
                                w.flush()
                            } else {
                                Ok(())
                            }
                        });
                        drop(w);
                        match wr {
                            Ok(()) => {
                                done.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(e) => record_err(i, e.into()),
                        }
                    }
                    Err(e) => record_err(i, e),
                }
            });
        });
        match first_err.into_inner().unwrap() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    };

    let items = done.load(Ordering::Relaxed);
    progress.finish();
    result?;
    writer.lock().unwrap().flush()?;
    Ok(RunStats { items })
}

// ---------------------------------------------------------------------------
// selftest (embedded golden mini-corpus)
// ---------------------------------------------------------------------------

const SELFTEST_WORK: &str = include_str!("../golden/selftest_work.jsonl");
const SELFTEST_EXPECTED: &str = include_str!("../golden/selftest_expected.jsonl");

fn lines_by_id(text: &str) -> anyhow::Result<HashMap<String, Value>> {
    let mut map = HashMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line)?;
        let id = v["id"].as_str().ok_or_else(|| anyhow!("line without id"))?.to_string();
        if map.insert(id.clone(), v).is_some() {
            bail!("duplicate id {id:?}");
        }
    }
    Ok(map)
}

/// `datagen selftest`: run the committed golden work corpus through the full
/// dispatch (both sequential and threaded), compare against committed
/// expected results as parsed JSON per id, and check the §8 determinism
/// contract (threads=1 byte-stable + in order; threaded = same line set).
pub fn selftest(threads: usize) -> anyhow::Result<()> {
    let expected = lines_by_id(SELFTEST_EXPECTED)?;
    let mut opts = EngineOpts {
        threads: 1,
        mode: Mode::Run,
        sidecar_dir: None,
        progress: false,
        flush_each: false,
        total_hint: None,
    };

    // sequential, twice: byte-identical and in work order
    let mut out1: Vec<u8> = Vec::new();
    run_work(SELFTEST_WORK.as_bytes(), &mut out1, &opts)?;
    let mut out2: Vec<u8> = Vec::new();
    run_work(SELFTEST_WORK.as_bytes(), &mut out2, &opts)?;
    if out1 != out2 {
        bail!("selftest: --threads 1 output is not byte-reproducible");
    }
    let work_ids: Vec<String> = SELFTEST_WORK
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Value = serde_json::from_str(l).unwrap();
            v["id"].as_str().unwrap().to_string()
        })
        .collect();
    let out_ids: Vec<String> = std::str::from_utf8(&out1)?
        .lines()
        .map(|l| {
            let v: Value = serde_json::from_str(l).unwrap();
            v["id"].as_str().unwrap().to_string()
        })
        .collect();
    if work_ids != out_ids {
        bail!("selftest: --threads 1 results are not in work-file order");
    }

    // threaded: same result lines as a set of bytes
    opts.threads = threads.max(2);
    let mut out_mt: Vec<u8> = Vec::new();
    run_work(SELFTEST_WORK.as_bytes(), &mut out_mt, &opts)?;
    let mut set1: Vec<&str> = std::str::from_utf8(&out1)?.lines().collect();
    let mut set2: Vec<&str> = std::str::from_utf8(&out_mt)?.lines().collect();
    set1.sort_unstable();
    set2.sort_unstable();
    if set1 != set2 {
        bail!("selftest: threaded output line multiset differs from sequential");
    }

    // value comparison against the committed expected results
    let got = lines_by_id(std::str::from_utf8(&out1)?)?;
    if got.len() != expected.len() {
        bail!("selftest: {} results, expected {}", got.len(), expected.len());
    }
    for (id, want) in &expected {
        let have = got.get(id).ok_or_else(|| anyhow!("selftest: no result for {id:?}"))?;
        if have != want {
            bail!(
                "selftest: result mismatch for {id:?}\n  want: {want}\n  got:  {have}"
            );
        }
    }
    println!(
        "selftest PASS: {} items ({} tasks), determinism + expected values",
        got.len(),
        {
            let mut tasks: Vec<String> = SELFTEST_WORK
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| {
                    let v: Value = serde_json::from_str(l).unwrap();
                    v["task"].as_str().unwrap().to_string()
                })
                .collect();
            tasks.sort();
            tasks.dedup();
            tasks.join(",")
        }
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bordered n x n grid_data with no interior walls.
    fn bordered(n: usize) -> Vec<String> {
        let mut g = vec![String::new(); n * n];
        for x in 0..n {
            g[x].push('N');
            g[(n - 1) * n + x].push('S');
        }
        for y in 0..n {
            let idx = y * n;
            let mut s: Vec<char> = g[idx].chars().collect();
            s.push('W');
            s.sort_unstable();
            g[idx] = s.into_iter().collect();
            let idx = y * n + n - 1;
            let mut s: Vec<char> = g[idx].chars().collect();
            s.push('E');
            s.sort_unstable();
            g[idx] = s.into_iter().collect();
        }
        g
    }

    fn ctx() -> Ctx {
        Ctx { cache: BoardCache::new(None), mode: Mode::Run }
    }

    fn board_json(n: usize) -> Value {
        json!({"env_id": 7, "n": n, "grid_data": bordered(n)})
    }

    #[test]
    fn unknown_field_is_rejected() {
        let line = json!({
            "task": "forward_instance", "id": "x", "board": board_json(4),
            "robots": [[0,0],[1,1]], "target_idx": 0, "target": [3,3],
            "surprise": 1
        })
        .to_string();
        let err = process_line(&line, &ctx()).unwrap_err();
        assert!(format!("{err:#}").contains("surprise"), "{err:#}");
    }

    #[test]
    fn documented_aux_fields_are_accepted() {
        // replay_forward_state with full_policy_max_ctg + python_labels
        let line = json!({
            "task": "replay_forward_state", "id": "aux1", "board": board_json(4),
            "positions": [[0,0],[1,1]], "target_idx": 0, "target": [3,3],
            "max_expansions": 1000,
            "full_policy_max_ctg": 6,
            "python_labels": {"cost_to_go": 1}
        })
        .to_string();
        let res = process_line(&line, &ctx()).unwrap();
        let v: Value = serde_json::from_str(&res).unwrap();
        assert_eq!(v["id"], "aux1");
        assert!(v.get("cost_to_go").is_some());
    }

    #[test]
    fn board_needs_exactly_one_source() {
        let line = json!({
            "task": "board", "id": "b",
            "board": {"env_id": 1, "n": 4, "grid_data": bordered(4), "sidecar": "x.bin"}
        })
        .to_string();
        assert!(process_line(&line, &ctx()).is_err());
        let line = json!({"task": "board", "id": "b", "board": {"env_id": 1}}).to_string();
        assert!(process_line(&line, &ctx()).is_err());
    }

    #[test]
    fn grid_data_length_checked() {
        let mut gd = bordered(4);
        gd.pop();
        let line = json!({
            "task": "board", "id": "b", "board": {"env_id": 1, "n": 4, "grid_data": gd}
        })
        .to_string();
        let err = process_line(&line, &ctx()).unwrap_err();
        assert!(format!("{err:#}").contains("expected n*n"));
    }

    #[test]
    fn timeout_s_and_mfcd_are_rejected() {
        let base = json!({
            "task": "backward_rollout", "id": "r", "board": board_json(4),
            "target": [3,3], "target_robot": [[0,0],"Red"], "helpers": [[[1,1],"Blue"]],
        });
        let mut with_timeout = base.clone();
        with_timeout["timeout_s"] = json!(120.0);
        let err = process_line(&with_timeout.to_string(), &ctx()).unwrap_err();
        assert!(format!("{err:#}").contains("timeout_s"));
        let mut with_mfcd = base.clone();
        with_mfcd["max_final_component_distance"] = json!(3);
        let err = process_line(&with_mfcd.to_string(), &ctx()).unwrap_err();
        assert!(format!("{err:#}").contains("max_final_component_distance"));
        // null mfcd is fine (datagen semantics)
        let mut with_null = base;
        with_null["max_final_component_distance"] = Value::Null;
        process_line(&with_null.to_string(), &ctx()).unwrap();
    }

    #[test]
    fn replay_mode_rejects_generation_tasks() {
        let line = json!({
            "task": "backward_rollout", "id": "r", "board": board_json(4),
            "target": [3,3], "target_robot": [[0,0],"Red"], "helpers": [],
        })
        .to_string();
        let ctx = Ctx { cache: BoardCache::new(None), mode: Mode::Replay };
        let err = process_line(&line, &ctx).unwrap_err();
        assert!(format!("{err:#}").contains("not accepted"));
    }

    #[test]
    fn board_cache_compiles_once_per_content() {
        let c = ctx();
        let b: BoardRef = serde_json::from_value(board_json(4)).unwrap();
        let a1 = c.cache.resolve(&b, 2).unwrap();
        let a2 = c.cache.resolve(&b, 2).unwrap();
        assert!(Arc::ptr_eq(&a1, &a2), "same content must hit the cache");
        let a3 = c.cache.resolve(&b, 1).unwrap();
        assert!(!Arc::ptr_eq(&a1, &a3), "weight is part of the key");
    }

    #[test]
    fn bad_color_fails_loud() {
        let line = json!({
            "task": "backward_rollout", "id": "r", "board": board_json(4),
            "target": [3,3], "target_robot": [[0,0],"Mauve"], "helpers": [],
        })
        .to_string();
        let err = process_line(&line, &ctx()).unwrap_err();
        assert!(format!("{err:#}").contains("Mauve"));
    }

    #[test]
    fn target_idx_bounds_checked() {
        let line = json!({
            "task": "forward_instance", "id": "f", "board": board_json(4),
            "robots": [[0,0],[1,1]], "target_idx": 2, "target": [3,3],
        })
        .to_string();
        let err = process_line(&line, &ctx()).unwrap_err();
        assert!(format!("{err:#}").contains("target_idx"));
    }
}
