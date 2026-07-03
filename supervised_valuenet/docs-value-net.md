# Value network (NN#2): the subgoal scorer

## Role

The project solves Ricochet Robots by A\* over the space of **partial plans** (DAGs of
subgoals). Each open segment of a plan is a move "robot X must reach cell C", and a
**subgoal candidate** = `(bottleneck cell, support cell, helper robot)` proposes how to
realize it. The value network estimates, for a candidate, the **cost-to-go**: the number
of extra moves to optimally complete the whole partial plan if this candidate is committed.

It serves two purposes in search:
1. **rank** candidates within a decision (which subgoal to commit), and
2. provide a **calibrated `h`** for the A\* frontier (the relaxed heuristic is a loose
   lower bound, MAE ~15.5 vs true cost-to-go; the value net is MAE ~2 and tight).

Winning architecture: an **edge-masked looped transformer** over the board's slide-graph
(file `train/looped_pc.py`). It beats the GNN value net (regret 0.356 vs 0.633).

## Input encoding

The board is `16x16 = 256` cells. The model sees **257 tokens**: the 256 cells plus one
**global / scratchpad token** (row 256). Two things are fed in:

### 1. Node features `x` : `[257, 9]`
Per-cell binary markers (`train/gnn.py:_node_features`, row 256 left zero):

| ch | marker |
|----|--------|
| 0 | segment mover (current robot position, `seg_start`) |
| 1 | segment goal (`seg_end`) |
| 2 | candidate bottleneck cell |
| 3 | candidate support cell |
| 4 | candidate helper cell |
| 5 | all robot positions |
| 6 | other open-segment endpoints (the rest of the partial plan to finish) |
| 7 | already-committed bottleneck cells |
| 8 | already-committed support cells |

Channels 0-5 describe the current segment + candidate; **6-8 are the partial-plan
context** (`cost_to_go` is whole-plan completion, so the model must see the rest of the
plan, not just this candidate). Each candidate of a decision is a separate input (its own
`x`), so candidates are encoded with their cells marked.

### 2. Slide-graph adjacency `A_all`, `A_ind` : `[257, 257]` (binary, self-loops)
The Ricochet-Robots move structure (`train/looped_pc.py:_adj`). `A[i,j]=1` if a robot at
cell `j` can slide and stop at cell `i`. Two edge types:
- `A_ind`: **independent** slides (stop at a wall on their own) = exact reachability.
- `A_all`: **all** slides incl. **dependent** ones (only stop because a blocker robot is
  placed) = relaxed reachability.

These are not stored as features; they are used as **attention masks** (see below). Row/
column 256 (global token) has only a self-loop, so it is reached only by the global head.

## Architecture

Encoder = one **weight-tied block applied `recurrence` (=12) times** (the "loop"). Each
block has three attention-head families, summed with a residual:

```
attn(X) =  Head_global(X)                  # I:     full attention, every cell <-> every cell
         + Head_all(X, A_all)              # A_all: attention MASKED to all-edge neighbors
         + Head_ind(X, A_ind)              # A_ind: attention MASKED to independent neighbors
X = LayerNorm(X + proj(attn));  X = LayerNorm(X + MLP(X))
```

Each head is standard multi-head attention `softmax(QK^T / sqrt(d_k)) V`; the graph heads
add `masked_fill(A == 0, -inf)` before the softmax, so a cell attends **only to its
slide-neighbors** (this local sharpness is what makes the readout discriminate
candidates; pure global attention collapses to the mean). `enc = Linear(9, d_model=192)`
plus a learned positional embedding `[257, d]`. The masking + looping is the
algorithmic-reasoning bias: message passing along slides = a Bellman-Ford / reachability
computation.

Readout: **gather the 5 candidate cells** (bottleneck, support, helper, mover, goal) from
the final embeddings, concatenate `[5 * d]`, feed an MLP to `num_classes = 50` logits.

Value (calibrated cost-to-go) = **expected bin**:
`value = sum_b softmax(logits)_b * b`.

## Loss

The label is the exact `cost_to_go` (true optimal completion cost, from commit-and-solve
in `nn/generate.py`). Two terms:

1. **HL-Gauss smoothed classification** (calibration). Target is a Gaussian over the 50
   bins centred on the true cost-to-go (`sigma = 1`), normalised; loss is soft
   cross-entropy `-(target * log_softmax(logits)).sum()`. Predicting a distribution +
   smoothing calibrates the magnitude far better than plain MSE regression.
2. **Listwise ranking** (sharp ordering). Within each decision group, with `opt` the
   optimal candidate(s):
   `-(opt / opt.sum() * log_softmax(-value)).sum()` (lower value = better = higher rank).

`loss = ranking + class_weight * soft_CE`. Ranking gives the order needed to pick the
best candidate; HL-Gauss gives the calibrated number needed for the A\* `h`.

## Metrics and result

- **regret** (primary): extra moves from greedily following the model vs optimal.
- **value MAE**: predicted vs true cost-to-go.
- **top1**: fraction of decisions where the argmin-value candidate is optimal.

On held-out test boards: **regret 0.356, MAE 2.065, top1 0.827** (heuristic baseline
1.99 / 15.5 / 0.59; constant-mean MAE 4.24; GNN value net 0.633 / 2.352 / 0.748). No
hand-crafted features: the model learns the shortest-path quantities from the graph.

See also `proposal-net.md` (NN#1), and the journals under `docs/journal/`.
