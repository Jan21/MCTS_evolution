//! datagen — JSONL work-item CLI over the rust_datagen engine (DESIGN §7).
//!
//! ```text
//! datagen run    --work items.jsonl --out results.jsonl --threads 16
//! datagen replay --work decisions.jsonl --out replayed.jsonl --threads 16
//! datagen boards --work boards.jsonl --out manifests.jsonl --sidecar-dir DIR
//! datagen selftest
//! ```
//!
//! `--work -` reads work items from stdin (streamed; used by the Python
//! bridge as a co-process) and `--out -` writes results to stdout (flushed
//! per line). Progress goes to stderr every ~5 s. Exit is nonzero on any
//! malformed work item (DESIGN §7).

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use rust_datagen::io::{run_work, selftest, EngineOpts, Mode};

#[derive(Parser)]
#[command(name = "datagen", about = "Ricochet Robots datagen engine (JSONL work items)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args)]
struct IoArgs {
    /// Work-item JSONL file, or '-' for stdin (streamed line-by-line).
    #[arg(long)]
    work: String,
    /// Result JSONL file, or '-' for stdout (flushed per line).
    #[arg(long)]
    out: String,
    /// Worker threads (1 = sequential + results in work order; shared box,
    /// be considerate).
    #[arg(long, default_value_t = 16)]
    threads: usize,
    /// Base directory for relative sidecar paths.
    #[arg(long)]
    sidecar_dir: Option<PathBuf>,
    /// Suppress the ~5 s progress lines on stderr.
    #[arg(long)]
    quiet: bool,
}

#[derive(Subcommand)]
enum Cmd {
    /// Process any work-item task (board / forward_instance /
    /// backward_rollout / replay_*).
    Run(IoArgs),
    /// Replay-only mode: accepts replay_backward_decision /
    /// replay_forward_state items (pyref dumps).
    Replay(IoArgs),
    /// Board precompute: accepts "board" items only; writes sidecars via
    /// sidecar_out (resolved under --sidecar-dir) and manifest result lines.
    Boards(IoArgs),
    /// Run the embedded golden mini-corpus end-to-end and verify results +
    /// the determinism contract.
    Selftest {
        /// Threads for the multi-threaded leg of the check.
        #[arg(long, default_value_t = 4)]
        threads: usize,
    },
}

/// Count newline bytes for the progress ETA (file inputs only; the work
/// stream itself is still consumed line-by-line).
fn count_lines(path: &str) -> Option<u64> {
    let f = File::open(path).ok()?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let mut total = 0u64;
    let mut saw_any = false;
    let mut last: u8 = b'\n';
    loop {
        let buf = r.fill_buf().ok()?;
        if buf.is_empty() {
            break;
        }
        total += buf.iter().filter(|&&b| b == b'\n').count() as u64;
        last = *buf.last().unwrap();
        saw_any = true;
        let len = buf.len();
        r.consume(len);
    }
    if saw_any && last != b'\n' {
        total += 1; // unterminated final line
    }
    Some(total)
}

fn run_io(args: &IoArgs, mode: Mode) -> anyhow::Result<()> {
    let total_hint = if args.work == "-" { None } else { count_lines(&args.work) };
    let input: Box<dyn BufRead + Send> = if args.work == "-" {
        Box::new(BufReader::new(std::io::stdin()))
    } else {
        Box::new(BufReader::with_capacity(
            1 << 20,
            File::open(&args.work).map_err(|e| anyhow::anyhow!("--work {}: {e}", args.work))?,
        ))
    };
    let to_stdout = args.out == "-";
    let output: Box<dyn Write + Send> = if to_stdout {
        Box::new(std::io::stdout())
    } else {
        Box::new(File::create(&args.out).map_err(|e| anyhow::anyhow!("--out {}: {e}", args.out))?)
    };
    let opts = EngineOpts {
        threads: args.threads.max(1),
        mode,
        sidecar_dir: args.sidecar_dir.clone(),
        progress: !args.quiet,
        flush_each: to_stdout,
        total_hint,
    };
    let stats = run_work(input, output, &opts)?;
    if !args.quiet {
        eprintln!("[datagen] done: {} items", stats.items);
    }
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let res = match &cli.cmd {
        Cmd::Run(a) => run_io(a, Mode::Run),
        Cmd::Replay(a) => run_io(a, Mode::Replay),
        Cmd::Boards(a) => run_io(a, Mode::Boards),
        Cmd::Selftest { threads } => selftest(*threads),
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("datagen: error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
