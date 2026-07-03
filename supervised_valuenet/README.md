# Supervised value net (the baseline)

Self-contained replication of the **supervised cost-to-go value network** for the
Ricochet-Robots A\* planner. The net is an **edge-masked looped transformer**
(`train/looped_pc.py`); on held-out test boards it reaches **regret 0.356 / MAE
2.065 / top1 0.827**, with **no hand-crafted features** (for reference, an earlier
GNN value net reached 0.633 and the hand-coded heuristic 1.99; those are not
included here). See `docs-value-net.md` for the full architecture writeup.

Everything needed to train and evaluate the net lives in this one folder.

## What the value net does

The planner solves Ricochet Robots by A\* over **partial plans** (DAGs of subgoals).
A **subgoal candidate** = `(bottleneck cell, support cell, helper robot)`. The value
net estimates the candidate's **cost-to-go**: extra moves to optimally finish the
whole partial plan if this candidate is committed. It (1) ranks candidates within a
decision and (2) supplies a calibrated `h` for the A\* frontier.

## Layout

```
supervised_valuenet/
  train/
    looped_pc.py     # the value net: edge-masked looped transformer (regret 0.356)
    encode.py        # board encoders + graph/node-feature builders; ENV_DIR
    eval_looped.py   # eval a trained checkpoint on any split
  nn/
    benchmark.py     # dataset loader, fixed train/val/test SPLITS, metrics
    labels.py        # HL-Gauss soft-label / cost-to-go binning
    data/
      combined.jsonl # THE dataset (88 MB, committed): 181,768 labelled decisions
  environments/      # board pkls (2.0 GB, gitignored, see Data below)
  GridEnv.py         # env loader (used by the benchmark's heuristic reference)
  simulate.py        # slide-graph physics (wall_sets, slide)
  validate_plan.py   # plan realizability check (dep of simulate.verify_plan)
  requirements.txt
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

- **`nn/data/combined.jsonl`** (88 MB, committed): 181,768 labelled decisions over
  1,295 distinct boards. Labels are exact optimal cost-to-go from commit-and-solve
  A\* (the "oracle"). Fixed splits (`nn/benchmark.py::SPLITS`):
  - `train` = env 0-95 + 1000-1799  (86,778 decisions)
  - `val`   = env 1800-2399         (42,473 decisions)
  - `test`  = env 112-127 + 2400-2999 (48,835 decisions)
- **`environments/*.pkl`** (2.0 GB, 1,295 files, **gitignored** for size): each pkl
  holds the board's `grid_data` (walls), `grid_graph` (slide graph), and precomputed
  distances. The net reads `grid_graph` (adjacency = attention mask) and `grid_data`
  (wall planes). Provenance: copied from `jonathan:~/Astar_evolution/environments/`
  (the boards `combined.jsonl` references). Not regenerable to the same geometries
  (the 128 stock boards ship without a generator).

  If cloning this branch elsewhere, re-fetch the pkls into `environments/`:
  ```bash
  python3 -c 'import json;ids=sorted({json.loads(l)["env_id"] for l in open("nn/data/combined.jsonl")});open("/tmp/refenv.txt","w").write("\n".join(f"env_{i}.pkl" for i in ids)+"\n")'
  ssh jonathan 'cd ~/Astar_evolution/environments && tar cf - -T -' < /tmp/refenv.txt | tar xf - -C environments/
  ```

## Reproduce

Run from this folder (imports resolve as `train.*` / `nn.*`).

### 1. Train (regret 0.356)

```bash
python -m train.looped_pc --data nn/data/combined.jsonl \
    --epochs 50 --d-model 192 --recurrence 12 --heads 4 \
    --batch-size 8 --max-per-group 32 --warmup 8
```

Best checkpoint (min `val_regret`) lands in `lightning_logs/version_*/checkpoints/`.
~1 h on one A100.

### 2. Evaluate on the held-out test split (the headline number)

```bash
python -m train.eval_looped \
    --ckpt lightning_logs/version_0/checkpoints/<best>.ckpt \
    --data nn/data/combined.jsonl --split test
# -> value_MAE~2.07 regret~0.356 top1~0.83
```

## Result (held-out test boards)

| model                                | regret | MAE   | top1  |
|--------------------------------------|--------|-------|-------|
| **looped transformer (`looped_pc`)** | 0.356  | 2.065 | 0.827 |

**regret** = extra moves from greedily following the model vs optimal (primary metric).

## Note on the "0.676" number

Elsewhere the project quotes a supervised pipeline at **regret 0.676 / 99.6% solved**.
That is the *same* net measured differently: run **end-to-end inside A\* search**
(regret over the solved subset, plus solve rate), together with the proposal net
(NN#1). The **0.356** here is the value net's **offline per-decision regret** on the
benchmark. Errors compound over a full search, so the in-search number is higher.
Reproducing the full-search 0.676/99.6% additionally needs the proposal net and the
A\* harness (not included in this value-net-only folder).
