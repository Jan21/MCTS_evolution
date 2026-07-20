# `move_planner` — visual summary & comparison

One-page picture of how the move-based planner works **now**, and how it compares to the
original subgoal value net. Full detail: [`README.md`](README.md).

---

## In one sentence

Given a board state, one shared transformer predicts **which `(robot, direction)` to
play** (policy) and **the exact optimal number of moves to the goal** (value); an A\* uses
the policy to propose moves and the value to score the states they lead to.

---

## 1. The reformulation at a glance

```
        ORIGINAL  (subgoal DAG)                    NEW  (primitive moves)
   ──────────────────────────────────       ──────────────────────────────────
   state                                      state
     │  propose SUBGOAL candidates              │  the ≤16 legal MOVES
     │   (bottleneck, support, helper)          │   robot × {up,down,left,right}
     │                                          │   robot slides until wall/robot
     ▼                                          ▼
   value = cost to COMPLETE the partial       value = OPTIMAL # MOVES from this
           plan  (weighted, dep.edge=2)               state to the goal (exact)
     │                                          │
     ▼                                          ▼
   A* over partial-plan DAGs                  A* over board STATES  (Markov:
   (needs subgoal machinery:                   forget how you got here — this is
   reachability, dependent edges,              the AlphaZero policy+value setup,
   final components)                           the direct bridge to self-play)
```

The **encoder is the same** looped transformer in both. What changed: the *problem*
(subgoal → primitive move), the *readout heads*, and the *labels* (approximate subgoal
cost → exact optimal move count).

---

## 2. What the network is (input → encoder → two heads)

```
 INPUT  (one board state)
 ─────────────────────────────────────────────────────────────────────────
  node features  x : [257, 14]            (256 board cells + 1 global token)
      ch 0..3   robot occupancy  Red / Blue / Green / Yellow
      ch 4      target robot's cell
      ch 5      target cell (goal)
      ch 6..13  slide-displacement fields  (dx,dy to wall-stop, per direction)
  adjacency masks  A_all , A_ind : [257, 257]   (slide-graph = attention masks)
  readout indices  robot_cells[4] , dest_cells[4,4] , val_cells[2]
                                    │
                                    ▼
 ENCODER   (reused from train/looped_pc.py — unchanged)
 ┌───────────────────────────────────────────────────────────────────────┐
 │  Linear(14 → 192)  +  learned positional embedding                      │
 │  ┌── LoopedLayer  ── applied 12× (weight-tied) ───────────────────────┐ │
 │  │   Head_global (full attn) + Head_all (A_all-masked)                │ │
 │  │                            + Head_ind (A_ind-masked)               │ │
 │  │   → LayerNorm(X+attn) → LayerNorm(X+MLP)                           │ │
 │  └────────────────────────────────────────────────────────────────────┘ │
 └───────────────────────────────────────────────────────────────────────┘
                                    │  X : [257, 192]  (an embedding per cell)
             ┌──────────────────────┴───────────────────────┐
             ▼                                               ▼
 VALUE head                                       POLICY head
   in = [ global token ⊕ goal-cell ⊕              per (robot,dir):
          target-robot-cell ]   [3·192]             in = [ robot-cell ⊕ DEST-cell ] [2·192]
   → MLP → 64 bin logits                            → MLP → 1 logit  → [4×4]
   value = Σ softmax(bins)·b                        mask illegal → softmax
         = optimal cost-to-go                       = P(robot, direction)
```

> **Why the policy reads the destination cell.** Picking the optimal move is a *1-step
> lookahead* ("if this robot slides, where does it stop, does cost drop?"). A static
> per-robot readout can't do it — it wouldn't even overfit (top1 ≈ 0.1). Feeding each
> move's slide-**destination** embedding, the head hits **0.96** on held-out boards. This
> is the only architectural change from the parent readout.

---

## 3. How search uses the two heads (one A\* step)

```
   ┌───────────────────────── frontier (min-heap by f) ─────────────────────────┐
   pop state with lowest  f = g + value(state)
        │
        ├─ target robot on target cell?  ── yes ──►  return g   (solution length)
        │
        ├─ expand: legal_moves(state)                       (≤ 16 children)
        │        │
        │        ├─ POLICY(state) ranks the moves ─► keep top-k       (prune)
        │        │
        │        ├─ VALUE(child) for those k children             (score states)
        │        │
        │        └─ push each child with  f = (g+1) + value(child)
        │            (closed set on the state tuple — move search revisits states)
        └──────────────────────────────────────────────────────────────────────┘
```

`policy proposes, value ranks the proposals.` This division is exactly why the value net
is usable in search even though value-alone-greedy fails (see [§5](#5-results)).

---

## 4. The pipeline (data → train → evaluate)

```
  nn.gen_grids ──► random solvable ──► ORACLE  ──────────► move_planner/data/moves.jsonl
   (fresh boards)   instances          heuristic-A* +      (state, cost_to_go,
        │           (random target)    exact optimal        best_moves, legal_moves)
        ▼                              move count                 │
  environments/env_*.pkl  ◄── walls + slide graph (encoder reads back)
        │                                                          ▼
        └────────────────────────────────────────►  train MoveNet  (value + policy)
                                                               │  best.ckpt
                                                               ▼
                                        evaluate:  move-A* / greedy  vs  ORACLE
                                                   → regret + solve rate
```

Oracle = A\* with the classic Ricochet-Robots admissible heuristic (target-robot
"blocker-anywhere" distance). Blind BFS is intractable here; the heuristic makes exact
labels cheap. Correctness cross-checked vs independent blind BFS: 0 mismatches.

---

## 5. Results

Held-out **test** boards (never trained on). MoveNet best checkpoint:

```
  value MAE (optimal cost-to-go)   0.263 moves        (target range 1–13, mean 3.9)
  policy top-1 (optimal move)      0.960

  end-to-end, 450 instances / 150 boards, vs the exact optimum:
    NN A*  (policy top-k + value)  regret 0.354   solved 85%   (73% exactly optimal)
    NN greedy (policy)             regret 0.649   solved 86%
    NN greedy (value only)         regret 11.1    solved  7%   ◄ see note
```

**The value-only collapse is the headline lesson, not a bug.** Value is trained on
on-optimal-path states; ranking *all* children (including wild off-path ones) it misjudges
and wanders. Inside NN A\* it works because the **policy prunes to the top-k near-optimal
children first**, keeping value on states it has seen. → validates the two-head design.

---

## 6. Comparison to the original subgoal net

> ⚠ **The offline numbers measure different quantities and are NOT directly comparable.**
> The original value predicts a *subgoal's weighted plan-completion cost* (dependent edges
> cost 2; targets range 2–44, mean 14.4). The new value predicts the *raw optimal move
> count* (range 1–13, mean 3.9). So MAE 0.263 vs 2.065 is **not** an 8× claim — the tasks
> and target scales differ. Read the table as *formulation differences*, and compare
> end-to-end regret/solve with that caveat.

| aspect | Original — subgoal net | New — move planner |
|---|---|---|
| value predicts | cost to **complete a partial plan** (weighted) | **optimal # moves** from a state (exact) |
| policy predicts | next **subgoal** (bottleneck→support→helper, autoregressive) | next **(robot, direction)** (direct, dest-lookahead) |
| search space | partial-plan **DAGs** | board **states** (Markov) |
| node branching | few big semantic steps | ≤16 primitive moves, longer horizon |
| labels | A\* commit-and-solve on subgoals (abstraction-optimal) | heuristic-A\* over true joint states (**ground-truth optimal**) |
| needs subgoal machinery | **yes** (reachability, dep. edges, final components) | **no** |
| encoder | edge-masked looped transformer | **same** (reused verbatim) |
| offline value MAE | 2.065 *(subgoal cost, mean≈14.4)* | 0.263 *(move count, mean≈3.9)* |
| offline pick-optimal top-1 | 0.827 *(over subgoal candidates)* | 0.960 *(over ≤16 moves)* |
| **end-to-end regret** | **0.676** | **0.354** |
| **end-to-end solved** | **99.6%** | **85%** (NN A\*, `iters=1200`) |
| self-play ready | needs adaptation | **yes** — already the AlphaZero interface |

**Honest read of the trade-off:**
- **Lower regret (0.354 vs 0.676):** when the move planner solves, its plans are closer to
  optimal. The exact move-count value is a tighter guide than the weighted subgoal cost.
- **Lower solve rate (85% vs 99.6%):** the subgoal abstraction gives the original a compact
  search space with an admissible heuristic → near-complete search. The move space is
  16-branching over ~8-move horizons with a *learned, non-admissible* value, so best-first
  can wander on hard instances. The bottleneck is **value off-path mis-ranking**: the value
  net is trained only on optimal-path states, so it misjudges the off-path children A\* must
  rank (value-greedy alone solves just 7%). The fix is direct — **candidate-scoring** (label
  every candidate's resulting state; §5 of `../RESULTS.md`) lifts this to **100% / regret
  0.067**, and self-play does the same by training on visited states. Policy top-1 is 0.96,
  so it was never a policy failure.
- **The real win is structural:** exact labels,
  Markov states (forget history), no subgoal machinery, and a policy+value net that plugs
  straight into **self-play** — built and run in [`../move_planner_v2/`](../move_planner_v2/README.md)
  as plain Expert Iteration (the net's own A\* as the expert), which fixes the value's
  off-path blind spot by training on the states the search actually visits. From random
  weights it reaches **87.6% / 0.213**, beating this baseline.

---

## 7. Where things live

```
move_planner/
  state.py     oracle.py   generate.py   encode.py   net.py   evaluate.py
  data/moves.jsonl              checkpoints/best.ckpt
  README.md    ← full detail        SUMMARY.md ← you are here
```
Run from `supervised_valuenet/` with `PYTHONPATH=.` (e.g. `python -m move_planner.evaluate
--demo ...`). Detailed method, schema, losses, CLI: [`README.md`](README.md).
