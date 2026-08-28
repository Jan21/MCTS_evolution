# Learned subgoal discovery — Stage 2

`PLAN_SUBGOAL_DISCOVERY.md` §4, run 2026-08-28. Stage 2 asks one question: **can
a network, given a board state and a goal of the form "robot R on cell C",
predict the cost of that goal well enough to RANK candidate goals?** Everything
below is recomputed from payloads produced this session; nothing is copied from
a project document.

    PYTHONPATH=supervised_valuenet:self_play_robots python -m subgoal.stage2 gen|train|eval|beam|report|verify
    PYTHONPATH=supervised_valuenet:self_play_robots python -m subgoal.optset --n 20 --stride 23 --out ...

## Verdict

**Gate: FAIL.** On boards it never trained on the network ranks goals far better
than the baseline the plan names — Spearman **0.749 vs 0.431** for the any-stop
relaxation on the 450 bench450 root states, +0.318 — but it does **not** beat the
stronger learning-free baseline, the exact lone-robot slide BFS, which scores
**0.773**. The pre-registered margin was +0.05; the measured margin against the
better baseline is **−0.024** (and −0.016 on the second held-out board set).
Per the plan's rule the stage stops here and reports the negative result. No
tuning was done after the gate was read.

Two things were learned that matter more than the gate itself:

1. **The pre-registered configuration collapsed, and the cause is the encoder's
   recurrence depth.** At the production 12 weight-tied steps the net sat on the
   constant-value plateau for 30 epochs — within-group prediction spread exactly
   0.00, ρ ≈ 0, reachability head frozen at the 82.4% base rate. At 4 steps it
   learns (spread ≈ 1.9, ρ ≈ 0.75) with either positional signal. This is the
   collapse mode `supervised_valuenet/FINDINGS.md` §50 and §57 already record
   for this net family (7 of 8 cold trainings on the plateau; `val_group_spread`
   exactly 0.00), now reproduced on a new task with an unambiguous cause.
2. **Ranking 1024 candidate subgoals by their cost — even by their EXACT cost —
   does not keep the optimal subgoal inside a top-5 beam.** With exact costs the
   optimal-preserving subgoal survives a k=5 prune at the root of only **61.2%**
   of instances; the network's own ranking survives **66.7%**. Because k=5 is a
   hard prune, the root number is an upper bound on Stage 3's optimal-%. Adding
   the relaxed remaining-distance term raises it to 77.3% (exact) / 71.8%
   (network). This is a property of the plan's *selection rule*, not of the
   network, and no cost network can fix it.

---

## 1. The cost target

    c(s, R, C) = the number of SLIDES robot R needs to come to rest on cell C
                 from joint state s, with EVERY OTHER ROBOT FROZEN where it
                 stands (they are blockers; they never move).

This is `subgoal/space.py::rest_cells`, the exact joint-state BFS Stage 1 used.
It was chosen over the "other robots may move too" variant for three reasons,
and Stage 3 must use the same one:

1. **It is what the search's node expansion does.** A macro edge in
   `space.py::search` moves one robot while the others stand still, so the edge
   weight *is* this number. The free-robots variant is not an edge weight at
   all: "R ends up on C" does not name a unique successor joint state, so a
   best-first search cannot take it as a child and cannot add its cost to `g`.
2. **Legality by construction.** Because each edge moves one robot with the rest
   frozen, the concatenation of a path's edges is a legal primitive move
   sequence — the property that made all 40 Stage 1 solutions replay-certify
   with no realization step.
3. **The labels are ground truth, not bounds.** One BFS per (state, robot) gives
   the exact cost of all 256 goals of that robot; measured at 2.74 ms per
   (state, robot) in pure Python, so 10.9 ms for a decision's whole 1024.

Cells R cannot come to rest on are **unreachable** and have no cost; R's own cell
is not a candidate (standing still is not a subgoal). Both are excluded from
every cost metric. At 16×16 with four robots, **17.2–17.7% of a state's 1024
candidates are reachable** (mean 177–190 per state), mean cost ≈ 5.5 slides, max
26 — the target has real spread and is not dominated by one-move goals.

**A consequence worth stating plainly.** Reason 3 cuts both ways: at 10.9 ms per
decision, the quantity the network is asked to learn is computable exactly, and
faster than the network computes it, by the same BFS a Stage 3 expansion must
run anyway to build its children. A learned
predictor of `c` is therefore never needed for its own sake; the only reason to
have one is if its *ranking* is used before physics is consulted. That is what
§6 measures.

## 2. The data

| corpus | boards | id range | states | how the states were drawn |
|---|---|---|---|---|
| train | 700 | 1000–1699 | 2800 | 4 per board: a uniform random placement of 4 robots, then 0–3 random macro subgoals (a random robot to a random cell it can stop on), so the corpus holds planner-reachable descendants, not only uniform placements. seed 11 |
| val (model selection) | 50 | 1800–1849 | 200 | same recipe, seed 12 |
| test A (**headline**) | 150 | 2400–2549 | 450 | the **root states of bench450 itself** — the exact inputs a Stage 3 planner sees |
| test B | 100 | 1900–1999 | 400 | same recipe as training, seed 13 — separates "different boards" from "different state distribution" |

Board ids follow `supervised_valuenet/scaling/configs.py::CONFIGS["g16r4"]`:
train `0-95,1000-1799`, val `1800-2399`, test `112-127,2400-2999`, bench
`2400-2549`. Training draws only from a 700-board prefix of the train split, so
val and both test sets are boards no training state ever came from.
`subgoal.stage2 verify` prints the board-id overlap — **0, 0, 0** — and
re-derives 240 label groups by brute-force enumeration of every direction
sequence up to length 3: **0 mismatches**.

Each state stores the labels of **all 1024 candidates** (4 robots × 256 cells) as
an int8 array with −1 for unreachable, so the 2800 training states are
**2.87 M exactly-labelled (state, robot, cell) examples**.

## 3. The architecture change

**Encoder: unchanged.** `nn_labeler.model.LoopedLayer` — the size-free encoder
`spr/nets.py` re-exports — stacked with weight tying, d_model 192, 4 heads,
edge-masked attention over the board's slide graph
(`nn_labeler.encode.adjacency`), `pe="none"`. Board side enters only through the
token count, so the net stays size-free.

**Conditioning: the input channels are the change.** The old value net's 9
channels describe a hand-written (bottleneck, support, helper) candidate. The
physics of `c(s, R, C)` depends on nothing but the query robot's cell, the other
robots' cells and the walls, so the input is **two** channels — 0 the query
robot's cell, 1 every other robot's cell (colour is irrelevant to this cost) —
and the walls arrive, as before, only through the attention masks.

**The goal cell does not enter the encoder; it enters the readout.** One encoder
pass over a (state, robot) pair therefore scores all 256 goal cells of that
robot, and four passes score a decision's whole 1024-candidate set — the "one
batched network pass" `PLAN_SUBGOAL_DISCOVERY.md` §3 asks for. Measured
end-to-end: all 450 × 1024 candidates scored, *and* both CPU baselines
recomputed for them, in 35 s on one A100 — 78 ms per decision end to end, of
which the exact-label BFS alone is 10.9 ms.

**Head: the 96-bin distributional cost head is kept** — same depth and width,
same HL-Gauss target (σ = 1), same softmax-expectation readout, same listwise
ranking loss with the cheapest candidates as positives. The only change is its
gather: the old head reads 5 tokens (bottleneck, support, helper, seg_start,
seg_end); this one reads the 3 tokens a (state, R, C) query has — the query
robot's cell, the goal cell, and the global scratchpad token — so the first
Linear is 3·d_model wide instead of 5·d_model.

**One head was added.** Cost is undefined for a goal a robot cannot stop on, so
the 96-bin head is trained masked to reachable goals and a 1-logit reachability
head carries "can R stop here at all". Ranking over the physics-filtered
reachable set uses the cost head's expectation `cost_hat`; ranking over all 1024
uses `p_reach · cost_hat + (1 − p_reach) · 50`.

1.02 M parameters at recurrence 12 (encoder 740 K, cost head 166 K, reachability
head 110 K); the recurrence is weight-tied, so the count is the same at 4.
Trained in fp32 (the reference recipe; the edge-masked attention fills with
−inf, which fp16 cannot represent), AdamW, lr 3e-4, weight decay 1e-4, batch 16
groups (= 4096 labelled candidates), 30 epochs, seed 11, checkpoint selected by
best `val_spearman` on the 1800–1849 val corpus.

## 4. The collapse, and its cause

The first submission ran the pre-registered production recipe — `pe="none"`,
**recurrence 12** — for 30 epochs (job 4863358, 20.7 min of A100). It never
learned anything:

| epoch | 0 | 5 | 10 | 15 | 20 | 25 | 29 |
|---|---|---|---|---|---|---|---|
| val ρ | −0.002 | −0.003 | 0.012 | 0.011 | −0.078 | −0.002 | −0.022 (best over all 30: **0.052**) |
| val top-1 | 0.085 | 0.066 | 0.104 | 0.089 | 0.083 | 0.094 | random rate 0.083 |
| val reach-acc | 0.824 | 0.824 | 0.824 | 0.824 | 0.824 | 0.824 | = the "everything unreachable" base rate |

Its held-out numbers are the signature of a constant predictor: pooled ρ
**−0.003**, top-1 **7.1%** (random), reachability precision and recall **0.0%**
— it predicted "unreachable" for all 460 800 candidates of the bench450 root
set (tp 0, fp 0, fn 81 680, tn 379 120). "Random" here is exact: the mean
fraction of a group's reachable candidates that carry its minimum cost is
**0.083** per (state, robot) and 0.078 pooled, computed from the label arrays.

A four-arm diagnostic (job 4863393, 9.5 min, 4 epochs each on the same corpora,
only two structural knobs moved) isolates the cause. `val_group_spread` is the
collapse detector `SizeFreeValueNet.on_validation_epoch_end` already computes —
the mean within-group standard deviation of the predictions — introduced after
the battery-1 collapses of `supervised_valuenet/FINDINGS.md` §50 and the one
that caught §57's mid-training fall.

| arm | val ρ, epochs 0–3 | val_group_spread | val reach-acc |
|---|---|---|---|
| pe=none, **recurrence 12** (pre-registered) | −0.002, −0.001, 0.004, 0.000 | **0.00, 0.00, 0.00, 0.00** | 0.824 (base rate) |
| pe=sin2d, **recurrence 12** | 0.033, 0.047, 0.055, 0.056 | **0.00, 0.00, 0.00, 0.00** | 0.824 (base rate) |
| pe=none, **recurrence 4** | −0.059, 0.434, **0.583**, 0.016 | 2.07, 2.00, 1.91, 1.91 | 0.904 → 0.918 |
| pe=sin2d, **recurrence 4** | 0.043, 0.348, **0.556**, 0.421 | 2.02, 1.91, 1.84, 1.87 | 0.927 → 0.945 |

The positional signal makes no difference; the **depth does**. Twelve weight-tied
steps of edge-masked attention over a field whose non-robot cells all start from
an identical embedding drive every token to the same vector, and a readout that
must tell 256 goal cells apart then has nothing to read. Four steps do not.
The second full run (job 4863454) is identical to the first except
`--recurrence 4`, and is the run everything below reports. Both runs' payloads
are kept.

Repairing a collapse is not tuning: a net whose predictions have zero spread
cannot rank at all, so the first run measures nothing about the question Stage 2
asks. **No knob was touched after the second run's numbers were read**, and the
gate threshold was committed (d9032e1) before either run was submitted.

## 5. Ranking quality on boards never trained on

Both learning-free baselines read only the board and the robot positions:

- **B1 — the any-stop relaxation**, the heuristic the plan and Stage 1 name. One
  move may end on **any** cell of the row/column segment it can see, walls only,
  other robots ignored; B1(s, R, C) is the BFS distance from R's cell to C in
  that relaxed graph. Every real slide is a relaxed move, so B1 ≤ c — Stage 1's
  admissible A* heuristic. It is blind to robots, so it cannot tell a reachable
  goal from an unreachable one.
- **B2 — the lone-robot exact BFS**: the same BFS as the label, but with the
  board emptied of the other robots — real slide physics for a robot alone.
  Stronger than B1 and still learning-free; neither an upper nor a lower bound,
  because other robots both block and help. Cells the lone robot cannot stop on
  are scored with the stated constant **20**, above any lone-robot distance on a
  16×16 board.

The gate is set against the **better** of the two.

### Test A — the 450 bench450 root states (boards 2400–2549), pooled over all four robots
450 groups, mean 181.5 reachable candidates per group.

| scorer | Spearman ρ | top-1 accuracy | MAE (moves) |
|---|---|---|---|
| **goal-conditioned cost net** | **0.749** | **100.0%** | 2.83 |
| B1 any-stop relaxation | 0.431 | 97.6% | 2.82 |
| B2 lone-robot exact BFS | **0.773** | 98.0% | **2.70** |

One group per (state, robot) instead (1797 groups, mean 45.5 candidates): net
0.750 / 99.8% / 2.84; B1 0.406 / 97.7% / 2.77; B2 0.772 / 98.4% / 2.58.

### Test B — 400 random states on boards 1900–1999, pooled
400 groups, mean 176.6 reachable candidates.

| scorer | Spearman ρ | top-1 accuracy | MAE (moves) |
|---|---|---|---|
| **goal-conditioned cost net** | **0.723** | **100.0%** | 2.83 |
| B1 any-stop relaxation | 0.392 | 95.8% | 3.06 |
| B2 lone-robot exact BFS | **0.739** | 97.5% | 3.20 |

Per (state, robot): net 0.718 / 99.7% / 2.85; B1 0.377 / 95.5% / 2.98;
B2 0.738 / 97.7% / 3.13.

### Gate arithmetic

| | test A (bench450 roots) | test B (boards 1900–1999) |
|---|---|---|
| net ρ | 0.749 | 0.723 |
| better baseline (B2) ρ | 0.773 | 0.739 |
| margin | **−0.024** | **−0.016** |
| required | +0.05 | +0.05 |
| net top-1 vs best baseline top-1 | 100.0% vs 98.0% | 100.0% vs 97.5% |
| **verdict** | **FAIL** | **FAIL** |

Read honestly, the network **decisively beats the baseline the plan actually
names** (B1: +0.318 and +0.331) and **ties the stronger one it does not**
(B2: −0.024, −0.016). It is the better ranker at the very top of the list
(top-1 100% on both sets, against 98.0% and 97.5%) and the worse one on absolute
calibration (MAE 2.83 against B2's 2.70 on test A, and against 2.5 for a
constant predictor — the cost head is a poor regressor even where its ordering
is good). Where B2 is defined at all (37.5 of 45.5 candidates per group), B2's
ρ is 0.905 and its MAE 0.50 against the net's 0.791 and 2.86: the other robots
usually do not change the answer, and when they do not, exact lone-robot physics
is nearly the exact answer.

The reachability head, at 4 recurrent steps, reaches 92.3% accuracy with 86.7%
precision and 66.7% recall on test A (base rate 82.3% unreachable) — useful but
well short of the physics it is approximating.

Training remained unstable at recurrence 4: val ρ swings between −0.07 and 0.73
across the 30 epochs while `val_group_spread` stays flat at ≈ 1.9, so the
checkpoint is a best-of-30 selection on the val corpus. That is legitimate early
stopping on data disjoint from both test sets, but it means the reported 0.749
is the top of a noisy band, not a stable operating point — which makes the −0.024
shortfall, if anything, generous to the network.

## 6. Beam survival — the number that predicts Stage 3

`subgoal/optset.py` computes the ground truth this needs. For each of the 39
distinct instances of Stage 1's two 20-instance samples it re-runs the state
subgoal search with `d*` known, prunes every child with f > d*, keeps **all**
tight parents, and runs until the frontier's f exceeds d*. A reverse pass from
the goal states marks every state on every optimal path and gives each one its
**complete set of optimal-preserving next subgoals** — not just the subgoal some
particular optimal plan used. All 39 completed inside the caps (0 capped, max
59 087 states stored, max 36.1 s), all 39 reproduce `d*`, and all 40 rows of
Stage 1's two payloads agree with them on (d*, optimal move count).

Measured at every state of the **fewest-subgoal** optimal path (86 decisions;
mean 2.21 decisions per instance, matching Stage 1's 2.0–2.45). On average
**4.34 of the 1024 candidates preserve optimality**, so a uniformly random top-5
would survive **2.1%** of the time.

Six rankings, all over the same 1024 candidates. `+h` adds the relaxed
remaining-distance of the target robot in the *resulting* state — i.e. the
`f = g + c + h` priority Stage 1's exhaustive A* used, which costs one extra BFS
per instance and no network.

**Expected survival under uniform random tie-breaking** (integer scorers tie
massively, so this is the fair summary; the optimistic and pessimistic bounds
are in the payload):

| ranking | top-1 | top-3 | **top-5** | top-10 | top-20 |
|---|---|---|---|---|---|
| network cost | 14.0% | 39.5% | **59.3%** | 76.7% | 83.7% |
| **exact cost** (oracle for what the net predicts) | 13.5% | 36.0% | **54.3%** | 88.0% | 100.0% |
| B1 relaxation | 0.0% | 0.0% | **2.5%** | 14.3% | 34.1% |
| network cost + h | 33.7% | 51.2% | **62.8%** | 79.1% | 83.7% |
| **exact cost + h** (Stage 1's A* priority) | 43.3% | 62.1% | **74.2%** | 92.1% | 99.2% |
| B1 + h | 10.9% | 29.3% | **43.8%** | 57.4% | 69.7% |

At the **root** only (39 instances, the one decision no search can route around):

| ranking | top-1 | top-3 | **top-5** | top-10 | top-20 |
|---|---|---|---|---|---|
| network cost | 17.9% | 43.6% | **66.7%** | 82.1% | 89.7% |
| exact cost | 17.2% | 43.0% | **61.2%** | 88.6% | 100.0% |
| network cost + h | 43.6% | 56.4% | **71.8%** | 87.2% | 89.7% |
| exact cost + h | 48.5% | 67.9% | **77.3%** | 91.6% | 99.5% |
| B1 + h | 10.8% | 28.5% | **42.1%** | 54.6% | 69.1% |

Chained over the whole fewest-subgoal optimal path (the product of the
per-decision survivals, one path per instance):

| ranking | whole path survives a k=5 prune |
|---|---|
| network cost | 30.8% |
| exact cost | 28.0% |
| network cost + h | 38.5% |
| **exact cost + h** | **54.6%** |
| B1 + h | 34.3% |

**How to read this.** `PLAN_SUBGOAL_DISCOVERY.md` §3 says the expansion "scores
all 1024 goals, keeps the top k = 5". That is a *hard* prune: if no
optimality-preserving subgoal is in the root's top 5, no d\*-length solution
exists anywhere in the tree, whatever the expansion budget. So the root row is an
**upper bound on Stage 3's headline optimal-%**, and the chained row is a lower
bound (other optimal paths may survive where this one does not). Three readings:

1. **Cost alone is the wrong ranking, and the network is not what limits it.**
   Ranking by *exact* cost caps Stage 3 at 61.2% and the chained estimate is
   28.0%. The learned ranking is *better* than the exact one at k ≤ 5 (66.7% vs
   61.2% at the root; 59.3% vs 54.3% per decision) purely because its continuous
   scores break the massive cost ties better than chance — a real, if backhanded,
   argument for the network. But no cost predictor, however perfect, lifts this
   above the low 60s.
2. **Adding a remaining-distance term is worth more than learning the cost.**
   `exact cost + h` reaches 77.3% at the root and 54.6% chained, using a
   board-only BFS and no network at all. `network cost + h` reaches 71.8% and
   38.5%.
3. **The band brackets Stage 3's gate.** Stage 3 must beat the backward
   planner's 53.3% of 450 (STAGE01.md). Under a strict k=5 prune the best
   ranking measured here sits between 54.6% (chained lower bound) and 77.3%
   (root upper bound), and the *network* ranking sits between 38.5% and 71.8%.
   Stage 3 is not excluded, but it has no margin, and it will live or die on the
   ranking function rather than on cost accuracy.

## 7. The plan's table, unchanged

The plan asks for the table at every stage even when the new row is empty. It is
regenerated here by `python -m subgoal.table`, which replays every payload row's
own move dump; all three rows still reproduce their payloads (gate PASS, PASS,
PASS). Stage 2 trains no planner, so the fourth row stays empty.

| model | optimal % of 450 | solved / 450 | extra moves (on its solves) |
|---|---|---|---|
| forward (move-by-move, supervised) | 94.2% (424/450) | 450/450 | 0.067 (n=450) |
| backward (subgoal, supervised) | 53.3% (240/450) | 432/450 | 1.840 (n=432) |
| current self-play line (v14_stack nets) | 51.3% (231/450) | 439/450 | 1.995 (n=439) |
| **new: learned subgoal discovery** | — | — | — |

## 8. What Stage 3 should carry forward

- Keep the cost definition of §1; it is the only one that is an edge weight.
- **Do not spend the network on the edge cost.** Stage 3's expansion must run
  `rest_cells` anyway to build children, which yields the exact cost for free —
  10.9 ms for all 1024 candidates, against 78 ms per decision for the whole
  network-plus-baselines evaluation step.
  Consult physics first, then rank the reachable children.
- The ranking that decides the beam must contain a *remaining-distance* term.
  The cheapest version costs one BFS per instance and beats every cost-only
  ranking measured here.
- If a network is used at all, the object worth learning is that
  remaining-distance term — "how many more moves from this state to the target
  robot on the target cell" — not "what does this subgoal cost". Stage 2 as
  specified learned the quantity that is already free.

## 9. Compute

| what | job | partition | elapsed | node-hours |
|---|---|---|---|---|
| GPU smoke of train/eval/beam | 4863328 | qgpu_exp, 1 GPU | 1 m 49 s | 0.0038 |
| main leg, pre-registered recurrence 12 (collapsed) | 4863358 | qgpu_exp, 1 GPU | 22 m 37 s | 0.0471 |
| four-arm collapse diagnostic | 4863393 | qgpu_exp, 1 GPU | 9 m 31 s | 0.0198 |
| main leg, recurrence 4 (the reported run) | 4863454 | qgpu_exp, 1 GPU | 12 m 27 s | 0.0259 |
| corpus generation, verification, cross-checks | — | login node, 1 core | ≈ 2 min | 0 |
| **total** | | | **46 m 24 s of 1 A100** | **0.097** |

Job 4863331 (the same leg on `qgpu`, 2 h) was cancelled unstarted: 63 of the 72
GPU nodes were in maintenance and `squeue` put its start at 2026-08-30, so the
leg was re-cut to fit the 1 h `qgpu_exp` cap. Stage 2's budget was 1 node-hour;
it used **0.097**.

---

## Provenance appendix

Every number above, with the file it came from and how it was read.

### Code

| file | what it fixes |
|---|---|
| `self_play_robots/subgoal/costnet.py` | the cost target, the exact labeller, the two baselines, the network, the ranking metrics |
| `self_play_robots/subgoal/stage2.py` | `gen` / `train` / `eval` / `beam` / `report` / `verify`; `GATE_RHO_MARGIN = 0.05` |
| `self_play_robots/subgoal/optset.py` | the optimal-preserving subgoal sets |
| `self_play_robots/jobs/subgoal_stage2.slurm`, `..._smoke.slurm`, `..._diag.slurm` | the three jobs |

All committed as d9032e1 **before** either training job was submitted, so the
gate threshold is on record ahead of any result.

### Data

| number | file | how it was read |
|---|---|---|
| 2800 / 200 / 400 / 450 states; 17.24 / 17.51 / 17.25 / 17.73% reachable | `results/subgoal/stage2/{train_b1000-1699,val_b1800-1849,test_b1900-1999,test_bench450_roots}.npz` | `subgoal.stage2 verify`, which prints them from the arrays |
| board overlap train↔val, train↔test B, train↔test A = 0, 0, 0 | the same four npz | `verify`: set intersection of the `env_id` arrays |
| labels correct: 240 groups, 0 mismatches | `test_bench450_roots.npz` | `verify --groups 60 --depth 3`: brute-force enumeration of all 4^1 + 4^2 + 4^3 direction sequences per group, compared with `{cell : 0 < cost ≤ 3}` |
| `cost_row` == Stage 1's `rest_cells`, `relaxed_dist` == `relaxed_h` | `subgoal/space.py` | direct comparison on the first 15 bench instances × 4 robots: 0 disagreements (login node, this session) |
| test A's states are bench450's own rows | `supervised_valuenet/eval/data/bench450.jsonl` | `gen --from-bench`: `env_id` and `positions` copied verbatim, 450 rows |

### Training

| number | file | how it was read |
|---|---|---|
| recurrence-12 curve (ρ ≈ 0, top-1 at the random rate, reach-acc 0.824 for 30 epochs) | `runs/spr/subgoal/stage2_cost/lightning_logs/version_0/metrics.csv` | the `val_*` rows, one per epoch |
| recurrence-4 curve (ρ −0.07…0.73, spread ≈ 1.9), best epoch 7 | `runs/spr/subgoal/stage2_rec4/lightning_logs/version_*/metrics.csv`, `.../best.txt` | same; the checkpoint name carries its own `val_spearman` |
| the four diagnostic arms, incl. `val_group_spread` exactly 0.00 at recurrence 12 | `runs/spr/subgoal/stage2_diag/*/lightning_logs/version_*/metrics.csv` | the `val_*` rows of each arm |
| 78 ms per 1024-candidate decision, end to end | `runs/spr/spr-sg-stage2r4-4863454.out` | "eval bench450_roots took 35s" over 450 states; each state is 4 encoder passes plus both CPU baselines plus the label BFS |
| 2.74 ms per (state, robot) / 10.9 ms per decision for the exact label BFS | `subgoal/costnet.py::cost_row` | timed on the login node over the first 200 states of `test_bench450_roots.npz` (800 calls, 2.19 s), boards pre-cached |
| random top-1 rate 0.083 / 0.078 | `results/subgoal/stage2/test_bench450_roots.npz` | mean over groups of (#candidates at the group's minimum cost) / (#reachable) |

### Evaluation

| number | file | how it was read |
|---|---|---|
| test A table, gate margin −0.024 | `results/subgoal/stage2/eval_rec4_bench450_roots.json` | `subgoal.stage2 report --eval`; `per_state` / `per_robot` / `b2_defined_subset` / `reachability` / `gate` |
| test B table, gate margin −0.016 | `results/subgoal/stage2/eval_rec4_b1900-1999.json` | same |
| the collapsed run's ρ −0.003, top-1 7.1%, reach precision/recall 0.0% | `results/subgoal/stage2/eval_cost_bench450_roots.json`, `eval_cost_b1900-1999.json` | same |
| Spearman, top-1, MAE definitions | `subgoal/costnet.py::group_metrics`, `spearman` | tie-corrected average ranks; top-1 = the argmin of the prediction has the group's minimum true cost; MAE over the group's reachable candidates |

### Beam survival

| number | file | how it was read |
|---|---|---|
| 39 instances, 0 capped, all reproduce d\*, mean 2.21 decisions, mean optimal set 4.34 | `results/subgoal/stage2/optsets.json` | `summary` plus a scan of `rows` for `capped` / `ok` |
| the 40 Stage 1 rows agree on (d\*, optimal moves) | `results/subgoal/stage1_state_space_first20.json`, `..._stride23.json` | row-by-row join on `idx` against `optsets.json`: 40 checked, 40 agreeing, 0 differing |
| all six survival tables | `results/subgoal/stage2/beam_rec4.json` (`beam_cost.json` for the collapsed run) | `subgoal.stage2 report --beam`; `survival.{all,reach,root,reach_root}` |
| the chained (whole-path) row | the same file | product over `per_instance[*].decisions[*]["all:<scorer>"]["5"][2]` (the expected-survival component), averaged over the 39 instances |
| random top-5 = 2.1% | the same file | `random_top5.all` = 5 × mean(|optimal set|) / 1024 |

### The plan's table

| number | file | how it was read |
|---|---|---|
| the three rows, gate PASS × 3 | `results/fwd_m0/astar_candidate_scored.json`, `results/m0/g16r4_b1s21_b2flags.json`, `results/subgoal/v14_stack_g16r4_bench450_astar.json` | `python -m subgoal.table` re-run this session: every row's own move dump replayed under the joint-game rules, replayed length compared with `d*` |

### Compute

`sacct -j 4863328,4863331,4863358,4863393,4863454 -X` for the elapsed times;
node-hours = elapsed × 1/8 (one A100 of an eight-GPU node). Allocation
`open-37-42`, `qgpu_exp`, `--gpus 1`.
