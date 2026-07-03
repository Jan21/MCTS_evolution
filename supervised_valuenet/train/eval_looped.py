"""Evaluate a trained looped-transformer value net on a split: regret + MAE + top1.

    python -m train.eval_looped --ckpt lightning_logs/version_X/checkpoints/best.ckpt \
        --data nn/data/combined.jsonl --split test

Mirrors train.eval_gnn but for LoopedValueNet. The regret/top1/MAE definitions are
identical to the in-training validation loop (train/looped_pc.py), so the number
printed here for --split val matches val_regret; --split test gives the headline
0.356 test number reported in docs/value-net.md.
"""
from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader

from train.looped_pc import LoopedValueNet, DenseDataset, collate
from nn.benchmark import load, by_split, group_by_decision


def evaluate(model, groups, device, max_per_group=500):
    ds = DenseDataset(groups, max_per_group=max_per_group, sample=False, frac=1.0)
    dl = DataLoader(ds, batch_size=8, collate_fn=collate, num_workers=4)
    rows = []
    model.eval()
    with torch.no_grad():
        for b in dl:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            val = model._value(model(b))
            for i in range(val.shape[0]):
                rows.append((int(b["group"][i]), bool(b["opt"][i]),
                             float(val[i]), float(b["ctg"][i])))
    gr = {}
    for g, o, p, c in rows:
        gr.setdefault(g, []).append((p, o, c))
    top1 = sum(min(v)[1] for v in gr.values()) / len(gr)
    regret = sum(min(v)[2] - min(c for _, _, c in v) for v in gr.values()) / len(gr)
    mae = sum(abs(p - c) for _, _, p, c in rows) / len(rows)
    return mae, regret, top1, len(gr), len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LoopedValueNet.load_from_checkpoint(a.ckpt, map_location=device).to(device)

    recs = load(a.data)
    if a.split:
        recs = by_split(recs, a.split)
    groups = group_by_decision(recs)
    mae, regret, top1, nd, nr = evaluate(model, groups, device)
    print(f"{a.data.split('/')[-1]}[{a.split}]: decisions={nd} records={nr} | "
          f"value_MAE={mae:.3f} regret={regret:.3f} top1={top1:.3f}")


if __name__ == "__main__":
    main()
