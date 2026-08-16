"""Forward (move-level) planner training launcher for the seed x lr rescue grid.

Generalises jobs/launcher_patterns/train_fwd_lowlr_<cfg>.py (the recipe that
produced the forward controls of record: lightning_logs/version_44 = g16r8,
lr 1e-4, torch seed 7) into a parametrised, resumable form. It does exactly
what those launchers did -- apply the config env, rebind nn.benchmark.SPLITS
to the config's board ranges, seed torch, force MoveNet's lr default -- and
adds three things the grid needs:

  * --run-dir: chdir there so lightning_logs/ lands per (cfg, seed, lr) instead
    of in the repo root (the run_config/scaling.train convention);
  * resume: with RR_RESUME=1 and a last.ckpt under the run dir, Trainer.fit is
    given ckpt_path=<newest last.ckpt> (move_planner/net.py already saves
    save_last=True but has no resume flag; explicit and env-gated, never
    automatic -- FINDINGS 38/43 rule);
  * BEST.json: after fit, the ModelCheckpoint's best_model_score (monitor =
    val_policy_top1, mode max, the forward pipeline's own selection metric)
    and best_model_path are written next to lightning_logs/, so selection by
    validation never has to parse tfevents or guess version_N.

The trainer itself (move_planner/net.py) is untouched: model, loss, data
pipeline, checkpoint monitor are the production ones.

    python jobs/patterns/fwd_train_launcher.py --config g16r8 --seed 21 --lr 1e-4 \
        --run-dir scaling/runs/g16r8/forward-rescue/s21-lr1e-4 \
        -- --data scaling/data/g16r8/forward.jsonl --epochs 8 --batch-size 128 --num-workers 8
"""
from __future__ import annotations

import argparse
import glob
import inspect
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # .../supervised_valuenet
sys.path.insert(0, str(REPO))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--run-dir", required=True, help="repo-root-relative or absolute")
    p.add_argument("rest", nargs=argparse.REMAINDER, help="args after -- go to move_planner.net")
    a = p.parse_args()

    from scaling.configs import CONFIGS, apply_env
    cfg = CONFIGS[a.config]
    apply_env(cfg)                                    # BEFORE any repo import

    rest = a.rest[1:] if a.rest and a.rest[0] == "--" else list(a.rest)
    for i, tok in enumerate(rest[:-1]):
        if tok == "--data" and not Path(rest[i + 1]).is_absolute():
            rest[i + 1] = str((REPO / rest[i + 1]).resolve())

    run_dir = Path(a.run_dir)
    if not run_dir.is_absolute():
        run_dir = (REPO / run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)

    import torch
    torch.manual_seed(a.seed)
    print(f"[fwdgrid] torch.manual_seed({a.seed})", flush=True)

    import nn.benchmark as benchmark
    if not cfg.legacy:
        benchmark.SPLITS = {s: cfg.ids(s) for s in ("train", "val", "test")}
        print(f"[fwdgrid] SPLITS rebound to {cfg.name}", flush=True)

    import move_planner.net as mn
    sig = inspect.signature(mn.MoveNet.__init__)
    names = [q.name for q in sig.parameters.values() if q.default is not inspect.Parameter.empty]
    defaults = list(mn.MoveNet.__init__.__defaults__)
    defaults[names.index("lr")] = a.lr
    mn.MoveNet.__init__.__defaults__ = tuple(defaults)
    print(f"[fwdgrid] MoveNet lr default forced to {a.lr}", flush=True)

    import pytorch_lightning as pl
    _fit = pl.Trainer.fit

    def fit(self, *args, **kw):
        if os.environ.get("RR_RESUME") == "1" and "ckpt_path" not in kw:
            lasts = sorted(glob.glob(str(run_dir / "lightning_logs/version_*/checkpoints/last.ckpt")),
                           key=os.path.getmtime)
            if lasts:
                kw["ckpt_path"] = lasts[-1]
                print(f"[fwdgrid] RR_RESUME=1 -> resuming from {lasts[-1]}", flush=True)
            else:
                print("[fwdgrid] RR_RESUME=1 but no last.ckpt found -> fresh run", flush=True)
        try:
            return _fit(self, *args, **kw)
        finally:
            cb = getattr(self, "checkpoint_callback", None)
            if cb is not None:
                score = cb.best_model_score
                rec = {"config": cfg.name, "seed": a.seed, "lr": a.lr,
                       "monitor": cb.monitor, "mode": cb.mode,
                       "best_model_score": float(score) if score is not None else None,
                       "best_model_path": cb.best_model_path or None,
                       "epochs_run": int(self.current_epoch),
                       "trainer_args": rest}
                tmp = run_dir / "BEST.json.tmp"
                tmp.write_text(json.dumps(rec, indent=1))
                os.replace(tmp, run_dir / "BEST.json")
                print(f"[fwdgrid] BEST.json: {rec}", flush=True)

    pl.Trainer.fit = fit

    sys.argv = ["net"] + rest
    print(f"[fwdgrid] {cfg.name} seed={a.seed} lr={a.lr} cwd={run_dir}: python -m move_planner.net {' '.join(rest)}",
          flush=True)
    mn.main()


if __name__ == "__main__":
    main()
