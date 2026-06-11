"""Lightning data module for the value-network benchmark.

Reads the JSONL records, encodes each as a grid tensor on the fly, and builds the
HL-Gauss soft classification target. Splits are by graph id (see nn.benchmark)
so val/test boards are unseen. Each sample also carries its decision-group id and
`is_optimal`, so the `top1_optimal` metric can be computed at validation time.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from nn.benchmark import load, by_split, SPLITS, group_by_decision
from nn.labels import soft_label
from train.encode import encode, CHANNELS


class ValueDataset(Dataset):
    def __init__(self, records, encoder, num_classes, sigma, group_ids):
        self.records = records
        self.encoder = encoder
        self.num_classes = num_classes
        self.sigma = sigma
        self.group_ids = group_ids  # parallel list: decision-group id per record

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        x = torch.from_numpy(encode(r, self.encoder))
        y = torch.tensor(soft_label(r["cost_to_go"], self.num_classes, self.sigma),
                         dtype=torch.float32)
        return {
            "x": x,
            "y": y,
            "cost_to_go": float(r["cost_to_go"]),
            "is_optimal": bool(r["is_optimal"]),
            "group": self.group_ids[i],
        }


def _assign_groups(records):
    """Stable integer id per decision state (env+instance+segment+depth)."""
    groups = group_by_decision(records)
    gid = {}
    for k, g in enumerate(groups):
        for r in g:
            gid[id(r)] = k
    return [gid[id(r)] for r in records]


class ValueDataModule(pl.LightningDataModule):
    def __init__(self, data_path, encoder="grid_v1", num_classes=32, sigma=1.0,
                 batch_size=256, num_workers=4):
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = CHANNELS[encoder]

    def setup(self, stage=None):
        records = load(self.hparams.data_path)
        self.sets = {}
        for split in ("train", "val", "test"):
            rs = by_split(records, split)
            if not rs:
                continue
            self.sets[split] = ValueDataset(
                rs, self.hparams.encoder, self.hparams.num_classes,
                self.hparams.sigma, _assign_groups(rs))

    def _loader(self, split, shuffle):
        return DataLoader(self.sets[split], batch_size=self.hparams.batch_size,
                          shuffle=shuffle, num_workers=self.hparams.num_workers,
                          persistent_workers=self.hparams.num_workers > 0)

    def train_dataloader(self):
        return self._loader("train", True)

    def val_dataloader(self):
        return self._loader("val", False) if "val" in self.sets else None

    def test_dataloader(self):
        return self._loader("test", False) if "test" in self.sets else None
