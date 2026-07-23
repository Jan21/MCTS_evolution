"""g32r4 forward control retrain, lr 1e-4, with the config's split rebinding."""
import inspect
import sys
import torch

torch.manual_seed(7)
from scaling.configs import CONFIGS
import nn.benchmark as benchmark
cfg = CONFIGS["g32r4"]
benchmark.SPLITS = {s: cfg.ids(s) for s in ("train", "val", "test")}
print("[ctl32] SPLITS rebound", flush=True)

import move_planner.net as mn
sig = inspect.signature(mn.MoveNet.__init__)
names = [p.name for p in sig.parameters.values() if p.default is not inspect.Parameter.empty]
defaults = list(mn.MoveNet.__init__.__defaults__)
defaults[names.index("lr")] = 1e-4
mn.MoveNet.__init__.__defaults__ = tuple(defaults)
print("[ctl32] lr default forced to 1e-4", flush=True)

sys.argv = ["net", "--data", "scaling/data/g32r4/forward.jsonl",
            "--epochs", "4", "--batch-size", "16", "--num-workers", "8"]
mn.main()
