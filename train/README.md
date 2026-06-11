# train — value network training (PyTorch Lightning + Hydra)

Trains the **value network** (heuristic #2): given a candidate subgoal for an open
segment, predict its cost-to-go, as a classifier over discrete cost bins with
HL-Gauss smoothing. Everything is config-driven so encoders, architectures, and
label schemes are swept without code changes.

## Layout

```
train/
  encode.py    state+candidate record -> grid tensor [C,16,16]  (swappable channel schemes)
  data.py      LightningDataModule over nn/data/*.jsonl, builds soft targets
  model.py     DepthRecurrentTransformer (LightningModule)
  train.py     hydra entrypoint
  configs/     config.yaml + data/ model/ encoder/ trainer/
```

Data and the benchmark live in `../nn/` (records, splits, metrics, baseline).

## Run

```bash
# generate data first (see nn/README.md):
python -m nn.generate --graphs 0-127 --per-graph 6 --max-candidates 50 --out nn/data/all.jsonl

# train with defaults:
python -m train.train

# override anything:
python -m train.train model.depth=12 sigma=1.5 data.batch_size=512

# hydra multirun sweep:
python -m train.train -m model.depth=4,8,12 model.d_model=64,128
```

Outputs (checkpoints, logs) go to `train/outputs/<timestamp>/`.

## Model

Depth-recurrent transformer (Universal-Transformer style): each of the 256 grid
cells is a token; one transformer block is applied `depth` times with shared
weights, so depth = propagation rounds (what shortest-path reasoning needs). Head
is a softmax over `num_classes` cost bins; loss is soft cross-entropy against the
HL-Gauss target; the scalar prediction is the expected bin.

## Metrics (logged each epoch)

- `val_loss` — soft cross-entropy
- `val_mae` — abs error of predicted cost-to-go
- `val_acc` — hard-bin accuracy
- **`val_top1_optimal`** — fraction of decisions whose cheapest-predicted candidate
  is truly optimal. This is the number that predicts downstream beam quality;
  checkpoints are selected on it. Heuristic baseline ≈ 0.60 (see `nn.benchmark`).

## What to sweep

- **Encoder** (`encoder.name`): add variants to `train/encode.py` `ENCODERS`
  (channel sets, token schemes). `grid_v1` = 15 channels (walls, segment,
  candidate, committed context, robots).
- **Architecture** (`model.*`): `depth`, `d_model`, `nhead`, `dim_ff`, `pool`.
- **Labels** (`num_classes`, `sigma`): binning resolution and smoothing width.

## Next step: downstream eval (not yet built)

The real target is: plug the trained net into `skeleton.heuristics.score` and run
`beam=1`, measuring avg plan cost vs the full-A* optimum (~7.40). Note this needs
the score interface to see the partial-plan context the encoder uses; either
extend `score(env, candidate, plan, seg)` or add a context-free encoder variant
that `score(env, candidate)` can build from the candidate alone. See
`docs/journal/` for the design rationale.
