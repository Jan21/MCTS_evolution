"""Offline val metrics (value MAE + policy top-1) per checkpoint, on the SHARED
supervised val set (moves.jsonl, env 1800-2399). Reuses net.py's exact
validation_step / on_validation_epoch_end via Trainer.validate.

Run:  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python eval_offline_fill.py
"""
import warnings
warnings.filterwarnings("ignore")
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from move_planner.net import MoveNet, MoveDataset, collate
from nn.benchmark import load, by_split

VAL = by_split(load("move_planner/data/moves.jsonl"), "val")
print(f"val states = {len(VAL)}", flush=True)

CKPTS = {
    "supervised (sanity, expect ~0.263/0.960)": "move_planner/checkpoints/best.ckpt",
    "candidate-scored": "lightning_logs/version_0/checkpoints/epoch=11-step=28884.ckpt",
    "self-play warm (iter0)": "move_planner_v2/runs_warm/iter0.ckpt",
    "self-play from-scratch (iter15)": "move_planner_v2/runs_scratch_v5/iter15.ckpt",
}

for name, c in CKPTS.items():
    va = MoveDataset(VAL, True)  # with_slide=True (14-channel nets)
    dl = DataLoader(va, batch_size=256, collate_fn=collate, num_workers=8)
    model = MoveNet.load_from_checkpoint(c, map_location="cpu")
    tr = pl.Trainer(accelerator="gpu", devices=1, logger=False,
                    enable_progress_bar=False, enable_model_summary=False)
    r = tr.validate(model, dl, verbose=False)[0]
    print(f"RESULT | {name:42s} | val_mae={r['val_mae']:.4f} | "
          f"policy_top1={r['val_policy_top1']:.4f} | {c}", flush=True)
print("DONE", flush=True)
