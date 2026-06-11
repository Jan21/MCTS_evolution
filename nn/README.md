# nn — value-network benchmark and data pipeline

Tooling to train and benchmark the **value network** (heuristic #2): given a
candidate subgoal for an open segment, predict its **cost-to-go**. Trained as a
classifier over discrete cost bins with smoothing, not a regressor.

Encoders and architectures are deliberately *not* fixed here — the point is to
sweep them against a stable benchmark.

## Pipeline

```
nn/generate.py   instances -> A* -> (state, candidate, cost-to-go) records   (JSONL)
nn/labels.py     raw integer cost-to-go -> classification target (bins + HL-Gauss smoothing)
nn/benchmark.py  fixed graph splits + framework-agnostic metrics + heuristic baseline
```

The encoder (`state, candidate -> tensor`) and the model (depth-recurrent
transformer) plug in on top; they are intentionally left open so several
variants can be compared on the same data and metrics.

## Records

One record per `(decision state, candidate)`. Self-contained and JSON-native:

- instance: `target`, `target_robot`, `helpers`
- open segment being decided: `seg_start`, `seg_end`, `seg_support`, `mover_color`
- partial-plan context: `ctx_bottlenecks`, `ctx_supports`, `ctx_open_endpoints`
- candidate: `cand_bottleneck`, `cand_support`, `cand_helper`, `cand_parent_support`
- **`cost_to_go`** (raw integer label), `is_optimal` (on the optimal plan?), `depth`

`cost_to_go` is stored raw so binning/smoothing stays a sweepable knob.
`is_optimal` also serves the future policy network.

How labels are made: each instance is rolled out along one optimal trajectory;
at every subgoal decision every candidate is commit-and-solved for its exact
cost-to-go (the admissible A* makes the first complete plan optimal). The
cheapest is committed and the rollout advances.

## Generate

```bash
python -m nn.generate --graphs 0-127 --per-graph 6 --out nn/data/all.jsonl
# --max-candidates K   cap commit-solves per decision (faster, may bias)
```

The solver is bounded (`max_iters`, `max_frontier`, `max_open`) so pathological
random instances cannot OOM the generator.

## Benchmark

Graphs are split by id so val/test boards are unseen (cross-board
generalization):

```python
SPLITS = {"train": 0-95, "val": 96-111, "test": 112-127}
```

Metrics (predictions are scalar cost-to-go per record):

- `value_mae` — absolute error of predicted cost-to-go (clipped to class range)
- `accuracy` — hard-bin classification accuracy
- `top1_optimal` — **the one that matters**: fraction of decisions whose
  cheapest-predicted candidate is truly optimal. If this is high, beam=1 with the
  net reaches the optimum.

Baseline every model must beat (the hand-coded `subgoal_score` as a predictor):

```bash
python -m nn.benchmark --data nn/data/all.jsonl --num-classes 32
```

## Files

- `generate.py` — instance sampler + rollout + record writer
- `labels.py` — `to_class`, `soft_label` (HL-Gauss), `expected_value`
- `benchmark.py` — `SPLITS`, `load`, metrics, `heuristic_baseline`
- `data/` — generated JSONL (git-ignored if large)
