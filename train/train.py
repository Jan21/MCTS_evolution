"""Hydra + Lightning training entrypoint for the value network.

    python -m train.train                          # defaults
    python -m train.train model.depth=12 sigma=1.5 # override anything
    python -m train.train -m model.depth=4,8,12    # sweep

Everything is config-driven so encoders, architectures, and label schemes can be
swept without code changes. Metrics (val_mae, val_acc, val_top1_optimal) are
logged each epoch; `val_top1_optimal` is the one that predicts downstream beam
performance.
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from train.data import ValueDataModule
from train.model import DepthRecurrentTransformer


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed, workers=True)

    dm = ValueDataModule(
        data_path=cfg.data_path, encoder=cfg.encoder.name,
        num_classes=cfg.num_classes, sigma=cfg.sigma,
        batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    dm.setup()

    model = DepthRecurrentTransformer(
        in_channels=dm.in_channels, num_classes=cfg.num_classes,
        **OmegaConf.to_container(cfg.model, resolve=True))

    has_val = "val" in dm.sets
    callbacks = [LearningRateMonitor(logging_interval="epoch")]
    trainer_kw = OmegaConf.to_container(cfg.trainer, resolve=True)
    if has_val:
        callbacks.append(ModelCheckpoint(
            monitor="val_top1_optimal", mode="max", save_top_k=1,
            filename="best-{epoch}-{val_top1_optimal:.3f}"))
    else:
        trainer_kw["limit_val_batches"] = 0
        trainer_kw["num_sanity_val_steps"] = 0

    trainer = pl.Trainer(callbacks=callbacks, **trainer_kw)
    trainer.fit(model, dm)

    if dm.test_dataloader() is not None:
        trainer.test(model, dm)


if __name__ == "__main__":
    main()
