"""Run one of the three training entrypoints under a config.

Thin wrapper around the unmodified trainers -- `train.looped_pc`
(backward-value), `train.policy_tf` (backward-policy), `move_planner.net`
(forward) -- that supplies the per-config plumbing they cannot:

1. sets the config's RR_* env vars before any repo import (nets and encoders
   read them at import/construction time);
2. rebinds `nn.benchmark.SPLITS` to the config's board ranges for non-legacy
   configs (the stock constant encodes the legacy 16x16 board ids; a new
   config's ids 0-1199 would otherwise land in the wrong splits);
3. chdirs into scaling/runs/<config>/<system>/ so lightning_logs (checkpoints,
   metrics) stay per-config and never mix with another config's runs.

Everything after `--` is forwarded verbatim to the underlying trainer CLI.
GPU selection is the operator's: prefix with CUDA_VISIBLE_DEVICES=<idx> after
checking `nvidia-smi` (shared machine).

    CUDA_VISIBLE_DEVICES=3 python -m scaling.train --config g16r6 \
        --system backward-value -- --data scaling/data/g16r6/backward.jsonl
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

from scaling.configs import REPO, apply_env, get

ENTRYPOINTS = {
    "backward-value": "train.looped_pc",
    "backward-policy": "train.policy_tf",
    "forward": "move_planner.net",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--system", choices=sorted(ENTRYPOINTS), required=True)
    p.add_argument("--run-dir", default=None,
                   help="default scaling/runs/<config>/<system>")
    p.add_argument("--num-classes", type=int, default=None,
                   help="backward-value only: override LoopedValueNet's bin "
                        "count (it has no CLI flag; larger grids exceed the "
                        "default 50 bins). Forward passes --num-classes "
                        "through after `--` instead.")
    p.add_argument("rest", nargs=argparse.REMAINDER,
                   help="args after `--` go to the underlying trainer")
    a = p.parse_args()
    cfg = get(a.config)
    apply_env(cfg)                       # must precede any repo import

    rest = a.rest[1:] if a.rest and a.rest[0] == "--" else list(a.rest)
    # resolve --data before the chdir below (paths are repo-root-relative)
    for i, tok in enumerate(rest[:-1]):
        if tok == "--data" and not Path(rest[i + 1]).is_absolute():
            rest[i + 1] = str((REPO / rest[i + 1]).resolve())

    run_dir = Path(a.run_dir) if a.run_dir else (
        REPO / "scaling" / "runs" / cfg.name / a.system)
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)                    # lightning_logs/ lands here

    import nn.benchmark as benchmark
    if not cfg.legacy:
        benchmark.SPLITS = {s: cfg.ids(s) for s in ("train", "val", "test")}
        print(f"[scaling.train] SPLITS rebound to {cfg.name} board ranges "
              f"{ {s: cfg.board_ranges[s] for s in ('train', 'val', 'test')} }")

    if a.num_classes is not None:
        if a.system != "backward-value":
            raise SystemExit("--num-classes is a wrapper flag for "
                             "backward-value only (forward: pass it after --)")
        import functools
        import train.looped_pc as looped_pc
        looped_pc.LoopedValueNet = functools.partial(
            looped_pc.LoopedValueNet, num_classes=a.num_classes)
        print(f"[scaling.train] LoopedValueNet num_classes={a.num_classes}")

    modname = ENTRYPOINTS[a.system]
    mod = importlib.import_module(modname)
    print(f"[scaling.train] {cfg.name}/{a.system}: python -m {modname} "
          f"{' '.join(rest)}  (cwd={run_dir})", flush=True)
    sys.argv = [modname] + rest
    mod.main()


if __name__ == "__main__":
    main()
