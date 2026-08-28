"""Exact cost-to-go from the project's own engine, at labelling speed.

`move_planner/oracle.py::solve` is the reference: A* over joint robot states
with the classic admissible relaxed-target heuristic, so its answer is the true
move optimum. In pure Python one call costs of the order of a second on a 16x16
four-robot instance, which is far too slow for the ~400k child states Stage 3
must label. `supervised_valuenet/rust_datagen` is that same oracle ported to
Rust and gate-verified against the Python reference (`rust_datagen/
VERIFICATION.md`); its `replay_forward_state` work item returns exactly

    {"cost_to_go": int | null, "optimal_moves": [...], "legal_moves": [...]}

for an arbitrary (board, positions, target robot, target cell). This module is
a thin driver for it:

  * `ensure_sidecars` compiles the boards once into bincode sidecars so the
    per-state work items stay ~120 bytes instead of carrying `grid_data`;
  * `ctg_batch` labels a list of states in one engine run (file in / file out),
    through the `forward_instance` task with `full_policy=false`, which is one
    exact A* per state;
  * `ExactCTG` keeps the engine alive on a pipe for the oracle-heuristic
    diagnostic, which needs answers inside a search loop.

`cost_to_go` is None when the instance is unsolvable or the expansion cap is
hit; callers must treat that as "no label", never as a number.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPR = HERE.parent
REPO = SPR.parent
SV = REPO / "supervised_valuenet"

DATAGEN = SV / "rust_datagen/target/release/datagen"
ENV_DIR = SV / "environments"
MAX_EXPANSIONS = 400_000


def grid_data(env_id, env_dir=ENV_DIR):
    with open(Path(env_dir) / f"env_{env_id}.pkl", "rb") as f:
        return pickle.load(f)["grid_data"]


def ensure_sidecars(env_ids, sidecar_dir, n=16, env_dir=ENV_DIR, threads=16):
    """Compile every board once; returns {env_id: relative sidecar path}."""
    sidecar_dir = Path(sidecar_dir)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    want = {int(e): f"env_{int(e)}.bin" for e in sorted(set(int(e) for e in env_ids))}
    todo = [e for e, p in want.items() if not (sidecar_dir / p).exists()]
    if todo:
        work = sidecar_dir / "boards_work.jsonl"
        with open(work, "w") as fh:
            for e in todo:
                fh.write(json.dumps({"task": "board", "id": f"b{e}",
                                     "board": {"env_id": e, "n": n,
                                               "grid_data": list(grid_data(e, env_dir))},
                                     "dependent_edge_weight": 2,
                                     "sidecar_out": want[e],
                                     "emit": "none"}) + "\n")
        subprocess.run([str(DATAGEN), "boards", "--work", str(work), "--out",
                        str(sidecar_dir / "boards_manifest.jsonl"),
                        "--sidecar-dir", str(sidecar_dir), "--threads", str(threads),
                        "--quiet"], check=True)
    return want


def item(qid, env_id, sidecar, positions, target_idx, target,
         max_expansions=MAX_EXPANSIONS):
    """`replay_forward_state`: returns `cost_to_go` plus the full optimal-move
    set. Small result lines, so this is what the streaming driver uses."""
    return {"task": "replay_forward_state", "id": qid,
            "board": {"env_id": int(env_id), "sidecar": sidecar},
            "positions": [[int(p[0]), int(p[1])] for p in positions],
            "target_idx": int(target_idx),
            "target": [int(target[0]), int(target[1])],
            "max_expansions": int(max_expansions)}


def solve_item(qid, env_id, sidecar, positions, target_idx, target,
               max_expansions=MAX_EXPANSIONS):
    """`forward_instance` with `full_policy=false`: ONE exact A* per state and
    no per-child optimal-move set, which is what bulk labelling needs. Measured
    1.8x cheaper than `replay_forward_state` for the same `cost_to_go`
    (89 ms vs 160 ms of one core per 16x16 four-robot state, cap 400k)."""
    return {"task": "forward_instance", "id": qid,
            "board": {"env_id": int(env_id), "sidecar": sidecar},
            "robots": [[int(p[0]), int(p[1])] for p in positions],
            "target_idx": int(target_idx),
            "target": [int(target[0]), int(target[1])],
            "max_expansions": int(max_expansions),
            "full_policy": False, "score_candidates": False}


def ctg_batch(queries, sidecar_dir, work_dir, threads=16,
              max_expansions=MAX_EXPANSIONS, tag="ctg", quiet=True):
    """queries: [(env_id, positions, target_idx, target), ...]
    returns a list of `int | None`, aligned with `queries`."""
    sidecars = ensure_sidecars([q[0] for q in queries], sidecar_dir,
                               threads=threads)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    work = work_dir / f"{tag}_work.jsonl"
    res = work_dir / f"{tag}_res.jsonl"
    t0 = time.time()
    with open(work, "w") as fh:
        for i, (eid, pos, tidx, tgt) in enumerate(queries):
            fh.write(json.dumps(solve_item(str(i), eid, sidecars[int(eid)], pos,
                                           tidx, tgt, max_expansions)) + "\n")
    cmd = [str(DATAGEN), "run", "--work", str(work), "--out", str(res),
           "--sidecar-dir", str(sidecar_dir), "--threads", str(threads)]
    if quiet:
        cmd.append("--quiet")
    subprocess.run(cmd, check=True)
    out = [None] * len(queries)
    with open(res) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            out[int(d["id"])] = d.get("d_star") if d.get("status") == "solved" else None
    print(f"[rustexact] {len(queries)} exact cost-to-go labels in "
          f"{time.time() - t0:.1f}s on {threads} threads", flush=True)
    return out


class ExactCTG:
    """The engine kept alive on a pipe, for the oracle-heuristic diagnostic.

    One `ask()` writes a whole round of queries and reads exactly that many
    result lines back, so the pipe never deadlocks on a partially-filled
    buffer. Results are reassociated by their `id`, because the engine's worker
    threads may reorder them.
    """

    def __init__(self, sidecar_dir, threads=16, max_expansions=MAX_EXPANSIONS):
        self.sidecar_dir = str(sidecar_dir)
        self.max_expansions = max_expansions
        self.n_calls = 0
        self.p = subprocess.Popen(
            [str(DATAGEN), "run", "--work", "-", "--out", "-",
             "--sidecar-dir", self.sidecar_dir, "--threads", str(threads),
             "--quiet"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._seq = 0

    def ask(self, queries, sidecars):
        """queries: [(env_id, positions, target_idx, target)]; -> [int|None]."""
        if not queries:
            return []
        ids = []
        for eid, pos, tidx, tgt in queries:
            self._seq += 1
            qid = f"q{self._seq}"
            ids.append(qid)
            self.p.stdin.write(json.dumps(solve_item(qid, eid, sidecars[int(eid)],
                                                     pos, tidx, tgt,
                                                     self.max_expansions)) + "\n")
        self.p.stdin.flush()
        got = {}
        while len(got) < len(ids):
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("rust engine closed its output early")
            if not line.strip():
                continue
            d = json.loads(line)
            got[d["id"]] = d.get("d_star") if d.get("status") == "solved" else None
        self.n_calls += len(ids)
        return [got[i] for i in ids]

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()
