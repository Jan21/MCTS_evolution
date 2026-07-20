"""Print (or execute with --execute) one config's full pipeline, in order:

    gen_boards -> gen_data (backward, forward) -> train (backward value,
    backward policy, forward) -> bench -> eval.compare

Default is a dry-run print of copy-pasteable commands (run from the repo root
with PYTHONPATH=.). GPU stages carry a CUDA_VISIBLE_DEVICES=<FREE_GPU>
placeholder: check `nvidia-smi` and substitute an idle GPU index at launch
time (shared machine -- a GPU index is never hardcoded). With --execute the
selected steps run sequentially; GPU steps then require --gpu N, and the
compare step resolves checkpoint placeholders to the newest checkpoint under
scaling/runs/<config>/<system>/.

    python -m scaling.run_config --config g16r6
    python -m scaling.run_config --config g16r6 --steps boards,data-backward
    python -m scaling.run_config --config g16r6 --execute --gpu 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from scaling.configs import REPO, get, shell_env

# Measured on A100-40GB (looped value net, d=192, rec=12): peak memory scales
# ~linearly with in-flight records at ~0.32 GB each at G=24 and ~0.85 GB at
# G=32, so batch-size x max-per-group must shrink as the grid grows.
VALUE_BATCH = {16: "--batch-size 8 --max-per-group 32",
               24: "--batch-size 4 --max-per-group 16",
               32: "--batch-size 2 --max-per-group 16"}
# Optimal costs grow with the board; the default 50 value bins saturate at
# G=32 (num_classes has no looped_pc CLI flag -- the scaling.train wrapper
# injects it). Forward MoveNet defaults to 64 bins; widen it the same way.
VALUE_BINS = {32: 96}
FORWARD_BINS = {32: "--num-classes 96"}

STEP_ORDER = ["boards", "data-backward", "data-forward", "train-backward-value",
              "train-backward-policy", "train-forward", "bench", "compare"]

GPU_NOTE = ("# GPU step: run `nvidia-smi` first and replace <FREE_GPU> with an "
            "idle GPU index (shared machine -- never hardcode one).")


def latest_ckpt(run_dir: Path):
    """Newest checkpoint under a scaling run dir (preferring the monitored
    best over last.ckpt)."""
    cands = sorted(run_dir.glob("lightning_logs/version_*/checkpoints/*.ckpt"),
                   key=lambda q: q.stat().st_mtime)
    best = [q for q in cands if q.name != "last.ckpt"]
    return (best or cands)[-1] if cands else None


def build_steps(cfg, a):
    data = f"scaling/data/{cfg.name}"
    results = f"scaling/results/{cfg.name}"
    g = cfg.grid
    steps = {}

    steps["boards"] = dict(gpu=False, cmd=(
        f"python -m scaling.gen_boards --config {cfg.name}"),
        notes=["parallelize across CPUs with --nshards S --shard i"])

    steps["data-backward"] = dict(gpu=False, cmd=(
        f"python -m scaling.gen_data --config {cfg.name} --system backward "
        f"--per-graph {a.per_graph_backward}"),
        notes=["single-process exact solver; shard it: --nshards 16 --shard i, "
               f"then `cat {data}/backward.shard*.jsonl > {data}/backward.jsonl`"])

    steps["data-forward"] = dict(gpu=False, cmd=(
        f"python -m scaling.gen_data --config {cfg.name} --system forward "
        f"--per-graph {a.per_graph_forward} --score-candidates"),
        notes=[])

    vb = VALUE_BATCH.get(g, VALUE_BATCH[16])
    vbins = f"--num-classes {VALUE_BINS[g]} " if g in VALUE_BINS else ""
    steps["train-backward-value"] = dict(gpu=True, cmd=(
        f"CUDA_VISIBLE_DEVICES=<FREE_GPU> python -m scaling.train "
        f"--config {cfg.name} --system backward-value {vbins}-- "
        f"--data {data}/backward.jsonl --epochs {a.epochs} {vb}"),
        notes=[])

    steps["train-backward-policy"] = dict(gpu=True, cmd=(
        f"CUDA_VISIBLE_DEVICES=<FREE_GPU> python -m scaling.train "
        f"--config {cfg.name} --system backward-policy -- "
        f"--data {data}/backward.jsonl --epochs {a.epochs}"),
        notes=[])

    fbins = f" {FORWARD_BINS[g]}" if g in FORWARD_BINS else ""
    steps["train-forward"] = dict(gpu=True, cmd=(
        f"CUDA_VISIBLE_DEVICES=<FREE_GPU> python -m scaling.train "
        f"--config {cfg.name} --system forward -- "
        f"--data {data}/forward.jsonl --epochs {a.epochs} --patience 5{fbins}"),
        notes=[])

    steps["bench"] = dict(gpu=False, cmd=(
        f"python -m scaling.bench --config {cfg.name} "
        f"--per-board {a.per_board} --seed {a.bench_seed}"),
        notes=["oracle failures are kept as d_star=null (rate in the meta); "
               "compare must then use bench.solved.jsonl (eval.compare does "
               "d_star arithmetic and cannot take nulls)"])

    # realization is size-agnostic since 2026-07-12 (eval/realize.py infers the board
    # side from grid_data), so the backward planner is scored at every grid size
    skip_back = False
    backward_args = ("--backward-policy <BACKWARD_POLICY_CKPT> "
                     "--backward-value <BACKWARD_VALUE_CKPT>")
    compare_notes = ["<..._CKPT> = best checkpoint under "
                     f"scaling/runs/{cfg.name}/<system>/lightning_logs/ "
                     "(--execute resolves them automatically)",
                     "swap bench.jsonl for bench.solved.jsonl if the bench "
                     "meta shows n_oracle_failed > 0"]
    steps["compare"] = dict(gpu=True, cmd=(
        f"CUDA_VISIBLE_DEVICES=<FREE_GPU> {shell_env(cfg)} PYTHONPATH=. "
        f"python -m eval.compare --instances {data}/bench.jsonl "
        f"--expansions {a.expansions} --k {a.k} {backward_args} "
        f"--forward-ckpts <FORWARD_CKPT> --device cuda "
        f"--out {results}/comparison.json --md {results}/COMPARISON.md"),
        notes=compare_notes)
    return steps


def resolve_for_execute(cfg, name, cmd, gpu_idx):
    """Substitute placeholders at run time; returns None to abort."""
    if "<FREE_GPU>" in cmd:
        cmd = cmd.replace("<FREE_GPU>", str(gpu_idx))
    if name == "compare":
        bench = REPO / "scaling" / "data" / cfg.name / "bench.jsonl"
        meta = Path(str(bench) + ".meta.json")
        if meta.exists() and json.loads(meta.read_text())["n_oracle_failed"] > 0:
            cmd = cmd.replace("bench.jsonl", "bench.solved.jsonl")
            print("[run_config] bench has d_star=null instances -> comparing "
                  "on bench.solved.jsonl")
        for ph, system in (("<BACKWARD_POLICY_CKPT>", "backward-policy"),
                           ("<BACKWARD_VALUE_CKPT>", "backward-value"),
                           ("<FORWARD_CKPT>", "forward")):
            if ph not in cmd:
                continue
            ck = latest_ckpt(REPO / "scaling" / "runs" / cfg.name / system)
            if ck is None:
                print(f"[run_config] no checkpoint found for {cfg.name}/"
                      f"{system}; train it first", file=sys.stderr)
                return None
            cmd = cmd.replace(ph, str(ck))
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--steps", default=",".join(STEP_ORDER),
                   help=f"comma subset of: {','.join(STEP_ORDER)}")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--gpu", type=int, default=None,
                   help="GPU index for --execute (pick a free one via nvidia-smi)")
    p.add_argument("--per-graph-backward", type=int, default=20)
    p.add_argument("--per-graph-forward", type=int, default=30)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--per-board", type=int, default=3)
    p.add_argument("--bench-seed", type=int, default=1)
    p.add_argument("--expansions", type=int, default=1200)
    p.add_argument("--k", type=int, default=5)
    a = p.parse_args()
    cfg = get(a.config)
    chosen = [s.strip() for s in a.steps.split(",") if s.strip()]
    unknown = [s for s in chosen if s not in STEP_ORDER]
    if unknown:
        raise SystemExit(f"unknown steps {unknown}; valid: {STEP_ORDER}")
    chosen = [s for s in STEP_ORDER if s in chosen]
    steps = build_steps(cfg, a)

    print(f"=== scaling pipeline: {cfg.name} (grid {cfg.grid}, robots "
          f"{cfg.robots}, walls {cfg.walls}) ===")
    print(f"env: {shell_env(cfg)}")
    print(f"# run from {REPO} with PYTHONPATH=.")
    if cfg.legacy:
        print("# NOTE: legacy config -- the live 16x16 pipeline already owns "
              "its boards (environments/), data and checkpoints; these "
              "commands only build independent copies under scaling/ paths "
              "(gen_boards is a no-op; nothing pre-existing is overwritten).")
    print()
    for i, name in enumerate(chosen, 1):
        st = steps[name]
        print(f"[{i}/{len(chosen)}] {name}")
        if st["gpu"]:
            print(f"    {GPU_NOTE}")
        for note in st["notes"]:
            print(f"    # {note}")
        print(f"    {st['cmd']}")
        print()

    if not a.execute:
        return
    child = {**os.environ, "PYTHONPATH": str(REPO)}
    for name in chosen:
        st = steps[name]
        if st["gpu"] and a.gpu is None:
            raise SystemExit(f"step {name} needs a GPU: check `nvidia-smi` "
                             f"and rerun with --gpu <idx>")
        cmd = resolve_for_execute(cfg, name, st["cmd"], a.gpu)
        if cmd is None:
            raise SystemExit(1)
        print(f"[run_config] executing {name}: {cmd}", flush=True)
        rc = subprocess.run(["bash", "-c", cmd], cwd=REPO, env=child).returncode
        if rc != 0:
            raise SystemExit(f"step {name} failed (exit {rc})")
    print("[run_config] all selected steps completed")


if __name__ == "__main__":
    main()
