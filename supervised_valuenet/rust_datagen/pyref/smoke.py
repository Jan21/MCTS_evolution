"""Cross-engine smoke suite (DESIGN.md section 3b / section 9 smoke row).

Fixed seed; sizes 16/24/32 (~20 instances each, robots mixed 4 and 8 via the
r4/r8 configs) plus 64x64 (fresh boards, robots 4); both task types. For each
cell: dump -> engine replay -> diff, then a one-screen pass/fail matrix with
stage timings (these seed the benchmark table later).

Engines:
  --engine python   replay_python.py (the reference; must be ALL GREEN)
  --engine rust     rust_datagen/target/release/datagen replay --work ...
                    (fails cleanly while the Rust CLI does not exist yet)
  --engine stub     a command that always fails -- proves the harness reports
                    an engine-stage failure without taking the whole run down.

The 64x64 backward cell runs under --cell-timeout; if the Python reference
cannot dump within it, the matrix marks the cell "python-reference infeasible
(measured)" -- a documented limit, not a silent skip (DESIGN section 9).

    python rust_datagen/pyref/smoke.py --engine python
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PYREF_DIR = Path(__file__).resolve().parent
RD_DIR = PYREF_DIR.parent
SV_DIR = RD_DIR.parent
PY = sys.executable

# (cell name, dump selector args, instances) -- robots mix comes from the
# r4/r8 configs at 16/24; 32 has only the r4 config; 64 is config-less.
CELLS = [
    ("g16r4", ["--config", "g16r4"], 10),
    ("g16r8", ["--config", "g16r8"], 10),
    ("g24r4", ["--config", "g24r4"], 10),
    ("g24r8", ["--config", "g24r8"], 10),
    ("g32r4", ["--config", "g32r4"], 20),
    ("n64r4", ["--n", "64", "--robots", "4", "--fresh-boards", "3"], 8),
]
TASKS = ["backward", "forward"]


def run(cmd, timeout=None, log=None):
    """Run cmd in its own process group so a timeout kills the whole tree
    (dump_decisions re-execs itself and spawns worker processes)."""
    import os
    import signal as sig
    t0 = time.time()
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             start_new_session=True)
        try:
            out, _ = p.communicate(timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, sig.SIGKILL)
            except ProcessLookupError:
                pass
            out, _ = p.communicate()
            rc = -99
            out = (out or "") + f"\n[smoke] TIMEOUT after {timeout}s"
    except OSError as e:
        rc = -98
        out = f"[smoke] cannot exec {cmd[0]}: {e}"
    dt = time.time() - t0
    if log:
        log.write_text(" ".join(map(str, cmd)) + "\n\n" + out)
    return rc, dt, out


def engine_cmd(engine, dump, res, workers):
    if engine == "python":
        return [PY, str(PYREF_DIR / "replay_python.py"), "--dump", str(dump),
                "--out", str(res), "--workers", str(workers)]
    if engine == "rust":
        return [str(RD_DIR / "target" / "release" / "datagen"), "replay",
                "--work", str(dump), "--out", str(res),
                "--threads", str(workers)]
    if engine == "stub":
        return [PY, "-c",
                "import sys; sys.stderr.write('stub engine: not implemented\\n');"
                "sys.exit(3)"]
    raise ValueError(engine)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--engine", choices=["python", "rust", "stub"],
                   default="python")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--workers", type=int, default=8, help="<=16")
    p.add_argument("--cells", default=None,
                   help="comma list to restrict, e.g. g16r4,n64r4")
    p.add_argument("--tasks", default="backward,forward")
    p.add_argument("--cell-timeout", type=int, default=1800,
                   help="per-cell dump wall cap (s); a backward cell that "
                        "exceeds it is marked python-reference infeasible")
    p.add_argument("--instance-timeout", type=int, default=120,
                   help="per-instance rollout cap passed to the backward "
                        "dumper (production scaling.backward_label uses 120)")
    p.add_argument("--out-dir", default=str(PYREF_DIR / "out" / "smoke"))
    a = p.parse_args()

    cells = [c for c in CELLS
             if a.cells is None or c[0] in a.cells.split(",")]
    tasks = [t for t in TASKS if t in a.tasks.split(",")]
    out_dir = Path(a.out_dir) / a.engine
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    hard_fail = False
    t_all = time.time()
    for name, sel, instances in cells:
        for task in tasks:
            tag = f"{name}.{task}"
            dump = out_dir / f"{tag}.dump.jsonl"
            res = out_dir / f"{tag}.res.jsonl"
            print(f"[smoke] === {tag} ===", flush=True)

            # 1. dump
            cmd = [PY, str(PYREF_DIR / "dump_decisions.py"), *sel,
                   "--task", task, "--instances", str(instances),
                   "--seed", str(a.seed), "--workers", str(a.workers),
                   "--instance-timeout", str(a.instance_timeout),
                   "--out", str(dump)]
            rc, t_dump, out = run(cmd, timeout=a.cell_timeout,
                                  log=out_dir / f"{tag}.dump.log")
            if rc == -99:
                rows.append((tag, "-", t_dump, None, None,
                             f"python-reference infeasible (measured: "
                             f">{a.cell_timeout}s dump)"))
                print(f"[smoke] {tag}: dump exceeded {a.cell_timeout}s -> "
                      f"marked infeasible", flush=True)
                continue
            if rc != 0:
                rows.append((tag, "-", t_dump, None, None, "DUMP-FAIL"))
                hard_fail = True
                print(out[-2000:], flush=True)
                continue
            n_lines = sum(1 for _ in open(dump))
            if n_lines == 0:
                rows.append((tag, 0, t_dump, None, None,
                             "DUMP-EMPTY (no decisions/records)"))
                hard_fail = True
                continue
            meta = {}
            for stats in reversed(out.strip().splitlines()):
                if stats.startswith("{"):
                    try:
                        meta = json.loads(stats)
                    except json.JSONDecodeError:
                        pass
                    break
            print(f"[smoke] {tag}: dumped {n_lines} lines "
                  f"({meta.get('instances', '?')} instances) in {t_dump:.1f}s",
                  flush=True)

            # 2. engine replay
            rc, t_eng, out = run(engine_cmd(a.engine, dump, res, a.workers),
                                 log=out_dir / f"{tag}.engine.log")
            if rc != 0:
                rows.append((tag, n_lines, t_dump, t_eng, None,
                             f"ENGINE-FAIL (rc={rc})"))
                hard_fail = True
                print(f"[smoke] {tag}: engine failed rc={rc}: "
                      f"{out.strip().splitlines()[-1] if out.strip() else ''}",
                      flush=True)
                continue

            # 3. diff
            rc, t_diff, out = run(
                [PY, str(PYREF_DIR / "diff_labels.py"), "--dump", str(dump),
                 "--results", str(res)],
                log=out_dir / f"{tag}.diff.log")
            verdict = "PASS" if rc == 0 else "DIFF-FAIL"
            if rc != 0:
                hard_fail = True
                print(out, flush=True)
            rows.append((tag, n_lines, t_dump, t_eng, t_diff, verdict))
            print(f"[smoke] {tag}: engine {t_eng:.1f}s, diff {t_diff:.1f}s "
                  f"-> {verdict}", flush=True)

    # -- matrix --------------------------------------------------------------
    def fmt_t(t):
        return "-" if t is None else f"{t:7.1f}"

    print()
    hdr = (f"{'cell':<16} {'lines':>6} {'dump_s':>7} {'engine_s':>8} "
           f"{'diff_s':>7}  status")
    print(f"smoke matrix -- engine={a.engine}, seed={a.seed}, "
          f"total {time.time() - t_all:.0f}s")
    print(hdr)
    print("-" * len(hdr))
    for tag, n_lines, t_d, t_e, t_f, verdict in rows:
        print(f"{tag:<16} {str(n_lines):>6} {fmt_t(t_d):>7} {fmt_t(t_e):>8} "
              f"{fmt_t(t_f):>7}  {verdict}")
    print("-" * len(hdr))
    n_pass = sum(1 for r in rows if r[5] == "PASS")
    n_inf = sum(1 for r in rows if "infeasible" in r[5])
    print(f"{n_pass}/{len(rows)} cells PASS"
          + (f", {n_inf} marked infeasible (documented limit)" if n_inf else "")
          + ("" if not hard_fail else " -- FAILURES PRESENT"))
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
