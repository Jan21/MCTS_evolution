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
    p.add_argument("--warm-start", default=None,
                   help="backward-value only: load this checkpoint's "
                        "state_dict into the freshly built net before "
                        "training (cold value retrains are proven "
                        "seed-unstable; the b1_retrain_chain WarmValueNet "
                        "pattern). Path is repo-root-relative unless "
                        "absolute. The architecture must match — combine "
                        "with --num-classes when the predecessor used "
                        "non-default bins (g32r4: 96).")
    p.add_argument("--lr", type=float, default=None,
                   help="backward systems only: override the trainer's "
                        "learning-rate default (no CLI flag exists there; "
                        "the >=6-robot low-lr launcher pattern, typically "
                        "1e-4).")
    p.add_argument("--torch-seed", type=int, default=None,
                   help="torch.manual_seed before the trainer starts "
                        "(the b1 retrain chain used 11)")
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

    if a.num_classes is not None and a.system != "backward-value":
        raise SystemExit("--num-classes is a wrapper flag for "
                         "backward-value only (forward: pass it after --)")
    if a.warm_start is not None and a.system != "backward-value":
        raise SystemExit("--warm-start applies to backward-value only "
                         "(proposal nets retrain cold by policy)")
    if a.lr is not None and a.system == "forward":
        raise SystemExit("--lr is a wrapper flag for the backward systems; "
                         "forward uses the train_fwd_lowlr launcher pattern")

    if a.torch_seed is not None:
        import torch
        torch.manual_seed(a.torch_seed)
        print(f"[scaling.train] torch.manual_seed({a.torch_seed})")

    if a.system == "backward-value" and (a.num_classes is not None
                                         or a.lr is not None
                                         or a.warm_start is not None):
        import torch
        import train.looped_pc as looped_pc
        _orig_value = looped_pc.LoopedValueNet
        extra = {}
        if a.num_classes is not None:
            extra["num_classes"] = a.num_classes
        if a.lr is not None:
            extra["lr"] = a.lr
        warm = None
        if a.warm_start is not None:
            warm = Path(a.warm_start)
            if not warm.is_absolute():
                warm = (REPO / warm).resolve()
            if not warm.exists():
                raise SystemExit(f"--warm-start checkpoint not found: {warm}")

        class WrappedValueNet(_orig_value):
            def __init__(self, *args, **kw):
                super().__init__(*args, **{**extra, **kw})
                if warm is not None:
                    ck = torch.load(warm, map_location="cpu")
                    self.load_state_dict(ck["state_dict"])
                    print(f"[scaling.train] value net warm-started from "
                          f"{warm}", flush=True)

        looped_pc.LoopedValueNet = WrappedValueNet
        print(f"[scaling.train] LoopedValueNet wrapper: {extra}, "
              f"warm_start={warm}")
    elif a.system == "backward-policy" and a.lr is not None:
        import train.policy_tf as policy_tf
        _orig_policy = policy_tf.PolicyTF
        _lr = a.lr

        class WrappedPolicyTF(_orig_policy):
            def __init__(self, *args, **kw):
                kw.setdefault("lr", _lr)
                super().__init__(*args, **kw)

        policy_tf.PolicyTF = WrappedPolicyTF
        print(f"[scaling.train] PolicyTF lr={_lr}")

    modname = ENTRYPOINTS[a.system]
    mod = importlib.import_module(modname)
    print(f"[scaling.train] {cfg.name}/{a.system}: python -m {modname} "
          f"{' '.join(rest)}  (cwd={run_dir})", flush=True)
    sys.argv = [modname] + rest
    mod.main()


if __name__ == "__main__":
    main()
