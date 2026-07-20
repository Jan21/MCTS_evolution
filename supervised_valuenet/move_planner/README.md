# `move_planner` — move-based policy + value planner

A self-contained reformulation of the supervised Ricochet-Robots baseline. Where the
parent project searches over **subgoal DAGs** (bottleneck / support / helper) and scores
subgoal candidates, this package searches over **primitive moves** and learns, for a
*board state*, two quantities:

- **policy** — which `(robot, direction)` to play, `direction ∈ {up, down, left, right}`;
- **value** — the **exact optimal number of moves** from this state to the goal.

Because a Ricochet-Robots state is Markov, the whole move history can be forgotten: the
value and the best move depend only on the current board. That makes this the standard
AlphaZero `(policy, value)` interface — it drops directly into self-play, which is built
and run in [`../move_planner_v2/`](../move_planner_v2/README.md) (plain Expert Iteration
with the net's own A\* as the expert; see [§12](#12-self-play-built--move_planner_v2)).

> For a diagram-first overview and the comparison to the original subgoal net, see
> **[`SUMMARY.md`](SUMMARY.md)**. This file is the detailed reference.

```
 board state ─▶ [ shared looped-transformer encoder ] ─┬─▶ value : optimal cost-to-go
                                                        └─▶ policy: 4 robots × 4 dirs
```

---

## 1. The game as an MDP

| element | definition |
|---|---|
| **State** | positions of all 4 robots (colours `Red, Blue, Green, Yellow`, fixed slot order) + which robot is the *target* + the target cell. 16×16 board. |
| **Action** | `(robot_slot, direction)`, `direction ∈ (up, down, left, right)` — 4 robots × 4 dirs = **16** actions. |
| **Transition** | the chosen robot **slides** until a wall or another robot stops it (`simulate.slide`, with *every other robot* a blocker — true multi-robot physics). A slide that cannot leave its cell is a **no-op / illegal** move and is pruned. |
| **Goal** | the target robot stands on the target cell. |
| **Cost** | every move costs **1**; the value target is the minimum number of moves to the goal. |

State/transition helpers live in `state.py`
(`legal_moves`, `apply_move`, `is_goal`, canonical `COLOR_ORDER = [Red, Blue, Green, Yellow]`).

---

## 2. What the network sees (input encoding)

Everything is expressed on the board's `16×16 = 256` cells plus **one global/scratchpad
token** (row 256) — 257 tokens total. Built in `encode.py`.

### 2a. Node features `x` — `[257, 14]`
Per-cell channels (`move_node_features`); row 256 (global token) is all-zeros:

| ch | meaning |
|----|---------|
| 0–3 | per-slot robot occupancy: Red, Blue, Green, Yellow cell |
| 4 | the **target robot's** cell (which robot must reach the goal) |
| 5 | the **target cell** (the goal) |
| 6–13 | **slide-displacement fields**: for each of the 4 directions, the normalised `(dx, dy)` from every cell to its wall-slide stop — the board's one-step move dynamics (board-only, cached per board) |

Channels 0–5 describe the current position/goal; 6–13 hand the transformer the raw
slide physics so it can learn reachability and cost-to-go itself rather than rediscover
the geometry. (Drop 6–13 with `--no-slide` → `C = 6`.)

### 2b. Slide-graph adjacency `A_all`, `A_ind` — `[257, 257]`
Binary attention **masks** (not features), from the board's precomputed slide graph
(`_adj` → parent `train.encode._graph`). `A[i,j]=1` if a robot at cell `j` can slide and
stop at cell `i`. `A_ind` = independent slides (stop at a wall alone); `A_all` = all
slides incl. dependent ones (stop only because a blocker is placed). Self-loops on every
row; row/col 256 has only a self-loop, so the global token is reached only by the global
attention head.

### 2c. Readout index tensors
- `robot_cells [4]` — each robot slot's cell (policy gather + value's target-robot cell).
- `dest_cells [4,4]` — for every `(slot, dir)`, the cell the robot would **slide to**
  (`slide` with all other robots as blockers). This is the **1-step lookahead** the
  policy head needs (see [§4](#4-the-two-heads)).
- `val_cells [2]` — `(goal cell, target-robot cell)` for the value readout.

---

## 3. Encoder (reused verbatim from the parent value net)

`net.py` reuses `train.looped_pc.LoopedLayer` — the **edge-masked looped transformer**:

```
enc      = Linear(14 → d_model=192)                     # node-feature embedding
pos      = learned positional embedding [257, 192]
X0       = enc(x) + pos
for _ in range(recurrence=12):                          # ONE weight-tied block, looped
    X = LoopedLayer(X, A_all, A_ind)
```

Each `LoopedLayer` sums three attention-head families (+ residual, LayerNorm, MLP):
`Head_global` (full attention), `Head_all` (masked to `A_all` neighbours), `Head_ind`
(masked to `A_ind` neighbours). heads = 4. The masking + looping is an
algorithmic-reasoning bias: message passing along slide edges ≈ a Bellman-Ford /
reachability computation. This is the encoder the parent project found best (regret
0.356 vs the GNN's 0.633); we changed only the input width and the readout heads.

---

## 4. The two heads

**Value head** — reads the pooled **global token** plus the **goal-cell** and
**target-robot-cell** embeddings (`[3·d]`) → MLP → **64 bin logits**. The scalar
cost-to-go is the *expected bin* `Σ_b softmax(logits)_b · b` (HL-Gauss head, exactly as
the parent value net). Reading the two goal-relevant cells + the global summary gives the
value direct access to the goal geometry.

**Policy head** — for each `(slot, dir)`, concatenates the **robot-cell embedding** with
its **slide-destination-cell embedding** (`[2·d]`) → MLP → 1 logit → `[4,4]` logits;
illegal moves masked to `-∞`, softmax over the legal set.

> The destination embedding is essential. Picking the optimal move is a **1-step
> lookahead** ("if I slide this robot, *where does it stop*, and does that lower
> cost-to-go?"). A static per-robot-cell readout has no lookahead and does **not learn**
> — in an overfit test it stayed at random (top1 ≈ 0.1). Feeding the destination cell's
> embedding, the same head overfits perfectly (top1 → 1.0) and reaches **0.95 on
> held-out boards**. This is the one architectural departure from the parent readout.

---

## 5. Losses

`loss = value_loss + policy_weight · policy_loss`  (`policy_weight = 1.0`)

- **value_loss** — HL-Gauss smoothed cross-entropy: the target is a Gaussian over the 64
  bins centred on the true cost-to-go (`sigma = 1`), normalised; soft CE against the bin
  logits. Predicting a calibrated distribution (not plain MSE) makes the scalar value a
  tight, usable A\* heuristic.
- **policy_loss** — cross-entropy of the masked policy logits against a soft target that
  is **uniform over the optimal-move set** (`best_moves`). Records whose optimal set is
  incomplete (deep states, see [§6](#6-the-optimal-cost-oracle)) still supply a single
  behaviour-cloning move; goal states (no move) are skipped.

---

## 6. The optimal-cost oracle (`oracle.py`)

Labels are the **exact** optimal move count. A blind BFS over 4-robot joint states is
hopeless (a reachable component is >150 k states; optimal solutions run past 10 moves; an
unsolvable instance costs ~70 s). So we use **A\* with the classic Ricochet-Robots
admissible heuristic**:

- `relaxed_target_dist(target)` — the slides the **target robot alone** would need if a
  blocker were available at *every* cell (so a slide may stop anywhere along its ray). A
  reverse BFS from the goal; a lower bound on the target's own moves, hence on the total
  move count → admissible. A\* (`solve`) then returns the true optimum expanding very few
  nodes, and prunes truly-unreachable targets instantly.

`label_trajectory` rolls out **one** optimal path and, at each state on it, records:
- `cost_to_go` — exact optimal moves from that state (**value target**);
- `best_moves` — the **optimal-move set** (**policy target**). A successor is optimal iff
  its cost-to-go is `here-1`; since a successor always has cost-to-go `≥ here-1`, this is
  decided by a **cost-capped A\*** of radius `here-1` — cheap for small cost-to-go. For
  deep states (`cost_to_go > full_policy_max_ctg`, default 6) the radius-`here` checks get
  expensive, so we fall back to the single move taken on the path and flag the record
  `full=false` (value stays exact; excluded from the policy top-1 metric).

Unsolvable / too-hard instances (≈35 % of random samples; target can't be stopped on the
cell, or the search exceeds `max_expansions`) are skipped — as the parent generator drops
its failures. Correctness was cross-checked against an independent blind BFS: 0 mismatches
on cost, 0 wrong or missing optimal moves.

`optimal_cost(...)` is the cheap public alias used as the **evaluation reference**.

---

## 7. Data pipeline (`generate.py`) and record schema

For each board: generate fresh geometry (`nn.gen_grids` walls + slide graph, saved as
`environments/env_<id>.pkl` so the encoder can read it back — the project ships **no**
boards, so this stays self-contained), sample random **solvable** instances (random robot
cells + a **randomised** target robot/cell, for generalisation across target identities),
solve each with the oracle, and stream one JSONL record per decision on the optimal path.
Generation is **embarrassingly parallel over boards** (one worker per board).

One record (`move_planner/data/moves.jsonl`):
```json
{"env_id":1000, "robots":[[x,y],[x,y],[x,y],[x,y]], "target":[x,y], "target_idx":0,
 "cost_to_go":6, "best_moves":[[slot,dir],...], "legal_moves":[[slot,dir],...],
 "depth":2, "full":true}
```
`robots` are in `COLOR_ORDER` slots; `slot` indexes that order, `dir` indexes
`(up,down,left,right)`; `target_idx` is the slot of the robot that must reach `target`.

**Splits** are by `env_id` (reused from parent `nn.benchmark.SPLITS`), so val/test boards
are geometries never trained on:

| split | board ids used | boards | records |
|---|---|---|---|
| train | `1000–1599` | 600 | 92,316 |
| val | `1800–1999` | 200 | 30,402 |
| test | `2400–2599` | 200 | 30,907 |

Dataset actually generated: **153,625 records / 25,000 solved instances / 1000 boards**;
cost-to-go 1–13 (mean 3.9); 85 % of records carry a full optimal-move set. Generation
took ~28 min on 40 CPU workers.

---

## 8. Training setup (`net.py`)

| | |
|---|---|
| model | `MoveNet` (PyTorch-Lightning); ≈0.95 M params, `d_model=192, recurrence=12, heads=4, num_classes=64` |
| dataset | `MoveDataset` — one item per labelled state (no decision-grouping / ranking needed) |
| optimiser | AdamW, `lr 3e-4`, `weight_decay 1e-4` |
| batch | 128 states; `num_workers 12` |
| schedule | ≤20 epochs, **early-stop patience 4 on `val_policy_top1`** (value MAE converges in ~1 epoch; the policy is the slower, harder metric) |
| checkpoint | best `val_policy_top1` (max); `last.ckpt` also saved |
| device | one A100 (`CUDA_VISIBLE_DEVICES=<empty gpu>`), ~2.1 it/s, ~5.5 min/epoch |

**Metrics** (logged each epoch): `val_mae` = |predicted − true cost-to-go|;
`val_policy_top1` = fraction of complete-label decisions whose argmax **legal** move is
optimal.

**Result** (held-out val boards, best checkpoint):

| epoch | val_mae | val_policy_top1 |
|---|---|---|
| 1 | 0.748 | 0.808 |
| 4 | 0.541 | 0.896 |
| 8 | 0.389 | 0.936 |
| 12 | 0.337 | 0.951 |
| 16 | 0.296 | 0.951 |
| 20 (best) | **0.263** | **0.960** |

The value net predicts the optimal cost-to-go to within ~0.26 moves on unseen boards
(parent subgoal value net: MAE ~2.07, though a different quantity), and the policy picks
an optimal move 96 % of the time. Both heads share the encoder, and the policy's
destination-lookahead gradients also sharpened it — the value MAE fell from 1.85 (static
policy head, earlier run) to 0.26.

---

## 9. Evaluation (`evaluate.py`)

Three planners host the net inside search and are scored against the exact optimum
(`oracle.optimal_cost`) over sampled test-board instances:

- **NN A\*** — best-first `f = g + value(child)`; the **policy** orders/prunes the ≤16
  legal moves to top-`k`; goal test = target-on-target; a closed set on the state tuple
  (move search revisits states). Value is a learned, non-admissible heuristic, so this is
  best-first, not provably optimal (matches the parent `nn_astar`).
- **NN greedy (policy)** — follow the policy argmax, beam 1.
- **NN greedy (value)** — step to the child with the smallest predicted value, beam 1
  (pure value + 1-step lookahead; uses no policy).

Reported per method: solve rate, **mean regret** (extra moves over the optimum), and the
fraction solved exactly optimally. Single-puzzle demo:
`python -m move_planner.evaluate --ckpt <best> --demo --boards 2400`.

**Result** — 450 instances over 150 held-out test boards (`2400–2549`, `k=5`,
`astar-iters=1200`), vs the exact optimum:

| planner | solved | mean regret | solved exactly optimally |
|---|---|---|---|
| **NN A\*** (policy top-k + value `f=g+value`) | 381/450 (85%) | **0.354** | 277/381 (73%) |
| **NN greedy (policy)** (argmax policy, beam 1) | 388/450 (86%) | 0.649 | 291/388 (75%) |
| NN greedy (value only) (argmin value child) | 30/450 (7%) | 11.13 | 6/30 |

Reading these:
- The **policy** is the reliable action selector — greedily following it solves 86 % with
  only ~0.65 extra moves; adding value + a closed set (**NN A\***) cuts regret to **0.354**
  (comparable to the parent subgoal system's in-search regret ~0.676, a different search).
- **Value-only greedy collapses** (7 % solved). This is the exact
  [distribution-shift limitation](#13-known-limitations): the value net is trained on
  on-optimal-path states, so ranking *all* children — including wild off-path ones — it
  misjudges and wanders. It works *inside* NN A\* precisely because the **policy prunes to
  the top-k near-optimal children first**, keeping value on states it has seen. This
  validates the two-head design: policy proposes, value ranks the proposals.
- The remaining ~15 % failures are the **same off-path value mis-ranking**, deep in the
  search where A\* meets states far from any optimal path. The direct fix is
  **candidate-scoring** ([`../RESULTS.md`](../RESULTS.md) §5) — labelling every candidate's
  resulting state — which lifts solve rate to **100 % / regret 0.067**; self-play fixes it
  the same way. Note the sweep is slow (per-node GPU forwards) — ~20 min per 100 instances.

---

## 10. File layout & reuse

Everything in `move_planner/` is new; it **reuses** the parent baseline's shared code
(run modules from the `supervised_valuenet/` directory so both resolve):

```
move_planner/
  state.py      # hashable state; legal_moves / apply_move / is_goal (on simulate.slide)
  oracle.py     # optimal-move oracle: heuristic A* + per-decision value/policy labels
  generate.py   # parallel instance sampling -> JSONL
  encode.py     # state node features + robot/dest/val index tensors
  net.py        # MoveNet (two-headed LightningModule) + MoveDataset + training
  evaluate.py   # move-A* + greedy rollouts + regret-vs-optimal benchmark
  data/         # generated moves.jsonl
```

| reused from parent | for |
|---|---|
| `simulate.slide / wall_sets / DIRECTIONS` | the move primitive + walls |
| `nn.gen_grids` (`gen_walls`, `build_graph`) | fresh boards (no shipped boards needed) |
| `train.looped_pc.LoopedLayer`, `_adj` | the looped-transformer encoder + attention masks |
| `train.encode` (`walls_for`, `_slide_fields`, `_graph`, `COLOR_ORDER`) | board planes + slide graph |
| `nn.benchmark.SPLITS / load / by_split` | fixed cross-board splits |

The subgoal machinery (`partial_plan.py`, `skeleton/astar.py`, the proposal net,
`GridEnv` subgoal methods) is **not** used.

---

## 11. Reproduce end-to-end

```bash
cd supervised_valuenet
export PYTHONPATH=.

# 1. boards + labels (parallel; writes environments/env_*.pkl too)
python -m move_planner.generate --graphs 1000-1599,1800-1999,2400-2599 \
    --per-board 25 --workers 40 --out move_planner/data/moves.jsonl

# 2. train the two-headed net (pick an EMPTY gpu — the DGX is shared)
CUDA_VISIBLE_DEVICES=1 python -m move_planner.net \
    --data move_planner/data/moves.jsonl \
    --epochs 20 --patience 4 --batch-size 128 --num-workers 12

# 3. solve one puzzle / benchmark vs the optimal oracle
#    a trained checkpoint (val_mae 0.263, policy_top1 0.960) ships at
#    move_planner/checkpoints/best.ckpt — or use your own from lightning_logs/
CKPT=move_planner/checkpoints/best.ckpt
CUDA_VISIBLE_DEVICES=1 python -m move_planner.evaluate --ckpt $CKPT --demo --boards 2400
CUDA_VISIBLE_DEVICES=1 python -m move_planner.evaluate --ckpt $CKPT \
    --boards 2400-2599 --per-board 4 --k 5 --astar-iters 2000
```

A trained checkpoint is included (`move_planner/checkpoints/best.ckpt`, ~12 MB,
gitignored) so the demo/benchmark run without retraining.

---

## 12. Self-play (built — `../move_planner_v2/`)

The net is already the AlphaZero `(policy, value)` interface, so the supervised baseline is
one step from self-play — and that step is **implemented and run** in
[`../move_planner_v2/`](../move_planner_v2/README.md). The method is **plain Expert
Iteration**:

1. **Expert = the net's own A\*.** The current net's budgeted `evaluate.nn_astar` (larger
   budget than eval) is the expert — it out-solves the greedy net, and that gap is the
   learning signal. MCTS/PUCT was considered and **rejected**: at branching ≤16 with a
   cost-to-go value head, A\*-as-expert dominates and collapses MCTS to the same thing.
2. **Targets from that search, not the oracle** — on a solved path, value = the plain
   remaining length and policy = the committed move; unsolved instances are dropped.
   Nothing else changes: self-play emits the *exact same record schema*, so `net.py`,
   `MoveDataset`, `collate` and the loss are reused verbatim. The heuristic-A\* oracle stays
   useful only as an *evaluation* upper bound (the true optimum).
3. **Iterate** — generate → retrain → repeat; net-solvability is the implicit curriculum.

**Result:** from **random weights** (no oracle, no tricks) self-play reaches **87.6% solved /
regret 0.213**, *beating* this supervised baseline (85% / 0.354); warm-starting the baseline
reaches 97.6% / 0.055. The strongest model of all is a supervised variant —
**candidate-scored** (label every candidate move's resulting state, fixing the §13
off-path value gap) — at 100% / 0.067. Full comparison, metrics and difficulty breakdown:
[`../RESULTS.md`](../RESULTS.md).

---

## 13. Known limitations

- **Value distribution shift.** Records are states on optimal paths; at search time the
  value net is queried on off-path children too. The policy prunes to top-`k`, which
  mitigates this, but off-path value coverage is thinner than on-path. (Self-play fixes
  this by training on the states actually visited.)
- **~35 % instance skip.** Hard/unsolvable random instances are dropped, biasing the data
  slightly toward easier boards. `max_expansions` trades coverage for generation speed.
- **Deep-state policy labels** (`cost_to_go > 6`, ~15 % of records) are single
  behaviour-cloning moves, not full optimal sets — a speed/completeness trade in the
  labeler, not a modelling limit.
- **Search is best-first, not provably optimal** — the learned value is not admissible.
