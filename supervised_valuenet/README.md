# Supervised A* planner (proposal net + value net)

Self-contained replication of the **supervised** learned planner for Ricochet Robots:
A\* over partial plans, guided by **two transformer nets** trained on optimal-cost
labels. No hand-crafted value features, no GNN, no torch_geometric.

- **Proposal net (NN#1)** `train/policy_tf.py` -- given the open segment, generates the
  subgoal candidate autoregressively (bottleneck -> support -> helper) and ranks the
  valid candidates. Keeps A\*'s branching small.
- **Value net (NN#2)** `train/looped_pc.py` -- predicts a candidate's **cost-to-go**
  (extra moves to optimally finish the whole partial plan). Edge-masked looped
  transformer; on held-out test decisions **regret 0.356 / MAE 2.065 / top1 0.827**.
- **Search** `skeleton/astar.py` + `eval/end2end.py` -- A\* with `f = g + value`, expand
  the proposal net's top-k per node, first complete plan pops as the solution.

Both nets share one encoder (edge-masked looped transformer, 257 tokens = 256 cells +
1 global, three attention head families: global + A_all-masked + A_ind-masked). Whole
system measured end-to-end reaches **regret 0.676 / 99.6% solved** (see the note at the
bottom on why that is higher than the value net's offline 0.356). See
`docs-value-net.md` for the architecture writeup.

## Layout

```
supervised_valuenet/
  solve.py             # solve ONE puzzle end-to-end, print the plan  <-- start here
  eval/end2end.py      # benchmark the learned A* vs optimal + heuristic over many boards
  train/
    policy_tf.py       # PROPOSAL net (NN#1): autoregressive subgoal generator
    looped_pc.py       # VALUE net (NN#2): edge-masked looped transformer (regret 0.356)
    policy_common.py   # shared proposal helpers (_ix/_features/_meta), GNN-free
    encode.py          # board encoders + graph/node-feature builders; ENV_DIR
    eval_looped.py     # eval a value-net checkpoint on any split
  skeleton/
    astar.py           # A* over partial plans: propose / score / expand / solve_plan
    heuristics.py      # hand-coded propose+score baseline (no NN)
  A_star/base.py       # abstract solver base classes used by skeleton/astar
  partial_plan.py      # PartialPlan DAG (subgoal graph, cost, open edges)
  nn/
    generate.py        # random instances on a board; board-id spec parsing
    gen_grids.py       # generate BRAND-NEW boards (walls + slide graph + distances)
    collect_search.py  # decision-record builder (_record), used by the solver glue
    benchmark.py       # dataset loader, fixed train/val/test SPLITS, metrics
    labels.py          # HL-Gauss soft-label / cost-to-go binning
    data/combined.jsonl# THE dataset (88 MB, committed): 181,768 labelled decisions
  environments/        # board pkls (2.0 GB, gitignored, see Data below)
  GridEnv.py           # env loader + State/Robot
  simulate.py          # slide-graph physics (wall_sets, slide)
  validate_plan.py     # plan realizability check
  requirements.txt
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch, pytorch-lightning, numpy, networkx
```

## Data

- **`nn/data/combined.jsonl`** (88 MB, committed): 181,768 labelled decisions over 1,295
  distinct boards; labels are exact optimal cost-to-go from commit-and-solve A\*. Fixed
  splits (`nn/benchmark.py::SPLITS`): `train` = env 0-95 + 1000-1799, `val` = 1800-2399,
  `test` = 112-127 + 2400-2999.
- **`environments/*.pkl`** (2.0 GB, 1,295 files, **gitignored**): each pkl holds the
  board `grid_data` (walls), `grid_graph` (slide graph = attention mask) and precomputed
  distances. If cloning this branch elsewhere, re-fetch:
  ```bash
  python3 -c 'import json;ids=sorted({json.loads(l)["env_id"] for l in open("nn/data/combined.jsonl")});open("/tmp/refenv.txt","w").write("\n".join(f"env_{i}.pkl" for i in ids)+"\n")'
  ssh jonathan 'cd ~/Astar_evolution/environments && tar cf - -T -' < /tmp/refenv.txt | tar xf - -C environments/
  ```
  Brand-new boards need no fetch -- `nn/gen_grids.py` generates them (see below).

## 1. Train the two nets

Both train from `combined.jsonl` on one A100 in ~1 h each. Checkpoints land in
`lightning_logs/version_*/checkpoints/` (gitignored).

```bash
# VALUE net (NN#2)  -> best on min val_regret
python -m train.looped_pc --data nn/data/combined.jsonl \
    --epochs 50 --d-model 192 --recurrence 12 --heads 4 --batch-size 8

# PROPOSAL net (NN#1) -> best on regret@k
python -m train.policy_tf --data nn/data/combined.jsonl \
    --epochs 50 --d-model 192 --recurrence 12 --heads 4
```

Sanity-check the value net alone on the held-out split (the headline 0.356):

```bash
python -m train.eval_looped --ckpt lightning_logs/version_0/checkpoints/<best>.ckpt \
    --data nn/data/combined.jsonl --split test        # -> regret~0.356 MAE~2.07 top1~0.83
```

## 2. Solve puzzles

Both checkpoints required.

```bash
# an instance on an existing board:
python solve.py --board 112 --value <val.ckpt> --policy <pol.ckpt> --compare-optimal

# a BRAND-NEW random board (walls generated on the fly):
python solve.py --new --seed 7 --value <val.ckpt> --policy <pol.ckpt> --compare-optimal
```

Prints the solved subgoal DAG (bottleneck / support moves), the plan's move cost, and
(with `--compare-optimal`) the optimal cost + regret.

Generate a batch of fresh boards up front instead:

```bash
python -m nn.gen_grids --n 50 --start 900000 --out environments   # env_900000..env_900049.pkl
```

## 3. Benchmark the whole system

```bash
python -m eval.end2end --boards 112-127,2400-2599 --per-board 4 \
    --value <val.ckpt> --policy <pol.ckpt> --k 5 --astar-iters 3000
```

Reports, over many instances: **NN A\*** (value as h) regret + solve rate, **NN greedy**
(beam=1), and the **hand-coded heuristic** propose+score baseline, all vs optimal.

## Note on 0.676 vs 0.356

`0.356` is the value net's **offline per-decision regret** (`eval_looped`, one committed
decision at a time against the benchmark labels). `0.676 / 99.6% solved` is the **whole
system run inside A\*** (`eval.end2end`): errors compound across a full multi-decision
search, so the in-search regret is higher while the solve rate stays near-perfect.
