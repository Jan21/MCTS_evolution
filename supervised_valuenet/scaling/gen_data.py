"""Label a config's boards into training records (one subprocess per run).

The labeler runs in a child process whose environment carries the config's
RR_* variables (one config per process; repo modules read them at import):

- backward: `scaling.backward_label` -- the `nn.generate` rollout with
  `--max-candidates 14` and a mandatory 120s SIGALRM per-instance guard
  (the candidate-capped rollout can wander on larger boards).
- forward: `move_planner.generate` (its main CLI, unmodified).

Boards must already exist (run `scaling.gen_boards` first): the forward
labeler would otherwise create lean pkls that the backward stack cannot load.

    python -m scaling.gen_data --config g16r6 --system backward --per-graph 20
    python -m scaling.gen_data --config g16r6 --system forward --per-graph 30 \
        --score-candidates --nshards 4 --shard 0
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from scaling.configs import REPO, compress_ids, env, get


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--system", choices=["forward", "backward"], required=True)
    p.add_argument("--per-graph", type=int, default=20,
                   help="instances labeled per board")
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--splits", default="train,val,test",
                   help="which board_ranges splits to label")
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N boards of this shard (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16, help="forward only")
    p.add_argument("--timeout", type=int, default=120,
                   help="backward per-instance wall-time cap (s)")
    p.add_argument("--max-candidates", type=int, default=14,
                   help="backward only")
    p.add_argument("--score-candidates", action="store_true",
                   help="forward only: also label each candidate move's child state")
    p.add_argument("--out", default=None)
    p.add_argument("--engine", choices=["python", "rust"], default="python",
                   help="python (default): the unmodified labelers; rust: "
                        "scaling.rust_bridge over the rust_datagen engine "
                        "(same sampling stream; --timeout is replaced by the "
                        "engine's deterministic iteration budget)")
    a = p.parse_args()
    cfg = get(a.config)
    if not 0 <= a.shard < a.nshards:
        raise SystemExit("--shard must be in [0, --nshards)")

    ids = sorted({i for s in a.splits.split(",") for i in cfg.ids(s)})
    ids = ids[a.shard::a.nshards]
    if a.limit is not None:
        ids = ids[:a.limit]
    missing = [i for i in ids if not (cfg.env_dir_abs / f"env_{i}.pkl").exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(ids)} board pkls missing from "
            f"{cfg.env_dir_abs} (first: {missing[:5]}); run "
            f"`python -m scaling.gen_boards --config {cfg.name}` first")

    stem = f"{a.system}.rust" if a.engine == "rust" else a.system
    out = (Path(a.out) if a.out
           else REPO / "scaling" / "data" / cfg.name / f"{stem}.jsonl")
    if a.out is None and a.nshards > 1:
        out = out.with_name(f"{out.stem}.shard{a.shard}of{a.nshards}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    spec = compress_ids(ids)
    if a.engine == "rust":
        cmd = [sys.executable, "-m", "scaling.rust_bridge",
               "--config", cfg.name, "--system", a.system,
               "--per-graph", str(a.per_graph),
               "--nshards", str(a.nshards), "--shard", str(a.shard),
               "--splits", a.splits, "--seed", str(a.seed),
               "--threads", str(min(a.workers, 16)),
               "--max-candidates", str(a.max_candidates),
               "--out", str(out)]
        if a.limit is not None:
            cmd += ["--limit", str(a.limit)]
        if a.score_candidates:
            cmd.append("--score-candidates")
    elif a.system == "backward":
        cmd = [sys.executable, "-m", "scaling.backward_label",
               "--graphs", spec, "--per-graph", str(a.per_graph),
               "--seed", str(a.seed), "--max-candidates", str(a.max_candidates),
               "--timeout", str(a.timeout), "--out", str(out)]
    else:
        cmd = [sys.executable, "-m", "move_planner.generate",
               "--graphs", spec, "--per-board", str(a.per_graph),
               "--workers", str(a.workers), "--seed", str(a.seed),
               "--out", str(out)]
        if a.score_candidates:
            cmd.append("--score-candidates")

    child = {**os.environ, **env(cfg)}
    child["PYTHONPATH"] = str(REPO) + (
        os.pathsep + child["PYTHONPATH"] if child.get("PYTHONPATH") else "")
    print(f"[gen_data] {cfg.name}/{a.system}: {len(ids)} boards -> {out}")
    print(f"[gen_data] exec: {' '.join(cmd)}", flush=True)
    raise SystemExit(subprocess.run(cmd, cwd=REPO, env=child).returncode)


if __name__ == "__main__":
    main()
