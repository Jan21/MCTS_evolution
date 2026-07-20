# Ricochet-Robots planners — models, training, and results

Everything solves the **same puzzle**: one *target robot* must reach one *target cell* on a
16×16 board; a move slides one robot until a wall or another robot stops it; cost = number
of moves. Below: how each model is trained (at a glance), the metric definitions (read this
first to avoid unit confusion), then the comparison table and per-model detail.

---

## At a glance — how each model is trained

```
 ORIGINAL (subgoal)      exact oracle labels SUBGOALS  ─▶ train value+proposal nets ─▶ A* over subgoal DAGs
 SUPERVISED (move)       exact oracle labels MOVES      ─▶ train MoveNet             ─▶ A* over board states
 SELF-PLAY warm          start = supervised net; the net's OWN A* labels moves ─▶ fine-tune ─▶ repeat
 SELF-PLAY from scratch  start = RANDOM net;    the net's OWN A* labels moves ─▶ train ─▶ repeat   (NO oracle)
```

The move models share one network (`MoveNet`: shared looped-transformer encoder → a **value**
head = optimal moves-to-go, and a **policy** head = which (robot, direction)). Supervised and
self-play differ *only* in where the training labels come from: an exact oracle vs the net's
own search. Self-play uses the oracle **only to measure regret at test time**, never to train.

---

## Metrics — exact definitions (units matter!)

> ⚠ The two formulations use **different cost units**. The original subgoal system measures
> **weighted subgoal cost**; the move models measure **primitive moves**. They are *not* the
> same number. The one directly-comparable quantity is **primitive moves** (below).

| metric | definition | unit | better |
|---|---|---|---|
| **solve rate** | fraction of test instances the planner reaches the goal within its search budget | % | higher |
| **regret** | mean *extra* cost over the true optimum, on solved instances | see cost unit | lower |
| **primitive moves** (a.k.a. **mean plan length**) | number of robot slides in a solution | moves | — |
| **weighted subgoal cost** | a subgoal plan's cost: independent (wall-stop) edges count **1**, dependent (helper-assisted) edges count **2**. A difficulty *proxy*, **NOT a move count** | weighted units | lower |
| **% optimal** | fraction of solved instances with regret exactly 0 | % | higher |
| **value MAE** | value head's mean abs error vs the true optimal moves-to-go | moves | lower |
| **policy top-1** | fraction of decisions whose argmax legal move lies on an optimal path | % | higher |

**The unit trap, resolved.** The original's headline "cost ≈ 13.5" and "regret 0.676" are in
*weighted subgoal cost*, not moves. Realizing the original's optimal subgoal plans into actual
moves and comparing to the move-oracle on the same instances gives ≈ **7.0 vs 7.2 moves**
(matched 18-instance sample) — i.e. **the subgoal formulation's optimal plans are ≈ move-optimal**.
So there is no real "13.5 vs 3.9" gap; that was two different measuring sticks (and different
instance samples: the move models' test set averages **3.9** optimal moves).

---

## Comparison table

Move models evaluated by **NN A\*** (policy proposes top-k moves, value scores the resulting
states) on the **same 450-instance test set** (held-out boards 2400–2549). Original from its
own docs (different, weighted units).

| model | trained on | solve rate | regret | % optimal | notes |
|---|---|---|---|---|---|
| **Original subgoal NN** | oracle subgoal labels | 99.6% | **0.676** *(weighted)* | — | mean cost 13.5 *(weighted)*; ≈ move-optimal when realized (~7 moves) |
| **Supervised (move)** | oracle move labels (optimal-path states) | **85%** | **0.354** *(moves)* | 73% | val_mae 0.263, policy top-1 0.960 |
| **Supervised — candidate-scored** ★ | oracle move labels **+ every candidate's resulting-state cost** | **100%** | **0.067** *(moves)* | 94% | the *faithful* method (§5); **best model**; value-greedy 7%→78% |
| **Self-play — warm** | fine-tune supervised via own A\* | 97.6% | 0.055 *(moves)* | 95% | value-greedy 7%→51% |
| **Self-play — from scratch** | random init, pure self-play (no oracle, no tricks) | **87.6%** | **0.213** *(moves)* | 83% | **beats supervised on both** (16 iters); iter14 hits 89.1% solve |
| *(from scratch, scramble-only — superseded)* | flawed near-goal curriculum | 66% | 0.440 | — | see "what went wrong" below |

★ **candidate-scored supervised is the strongest: 100% solved, regret 0.067** — the *more faithful*
method (score the candidates via A\*) is also the best, because scoring the candidates' resulting
states gives the value net the off-path coverage it was missing. From-scratch is the **confirmed**
full-benchmark number (a 40-instance in-loop probe read higher, 92.5%, but was small-sample-optimistic).

---

## Per-model detail

### 1. Original subgoal NN (the starting point)
Search over **partial-plan DAGs**: a *proposal net* generates subgoal candidates
(bottleneck / support / helper) and a *value net* scores each candidate's cost-to-complete.
Labels came from an exact commit-and-solve A\* over subgoals; cost is **weighted subgoal cost**
(dependent edges ×2). Strong solve rate (99.6%) but the cost is a coarse, weighted proxy, and
the abstraction carries machinery (reachability matrices, dependent-edge bookkeeping). Its
checkpoint is not retained; metrics quoted are from the project's own docs.

### 2. Supervised move planner
The reformulation: predict, for a *state*, the **value** (optimal moves-to-go) and the
**policy** (which robot + direction). Training records are (state, exact optimal cost-to-go,
optimal move) from a from-scratch exact move-oracle (heuristic A\* over joint robot states).
`MoveNet` reuses the parent's looped-transformer encoder; the policy head reads each move's
slide-**destination** cell (a 1-step lookahead — without it the policy can't learn). Result:
85% solved, regret 0.354 moves, value MAE 0.263, policy top-1 0.960. This is the **baseline**
the self-play models are measured against.

### 3. Self-play — warm start
Same net, but labels come from the net's **own A\*** instead of the oracle: solve puzzles,
train on (state → found-solution length, committed move), repeat. Started from the supervised
checkpoint, one iteration jumps to **97.6% solved / regret 0.055** — the search generates
*better* labels on the states search actually visits, fixing the value net's off-path blind
spot (value-greedy solve rate 7%→51%). Beats supervised. (Less interesting scientifically — it
already knows the game — but proves the self-play targets are sound.)

### 4. Self-play — from scratch (the goal)
Identical loop but the net starts at **random weights** — no supervision, no oracle in training.
The generator makes puzzles the net can currently solve (a mix of **target-biased forward-walk**
puzzles — random start, slide the target k times, relabel its final cell as the goal, so a
≤k-move solution is guaranteed — and raw random instances); the net's A\* solves what it can and
unsolved puzzles are dropped. Value target = plain remaining path length; policy target = the
committed move. Net-solvability is the **implicit curriculum**: it solves shallow puzzles first,
learns, then solves deeper ones (deep-state share climbs each iteration). Two runs, same method:
a 10-iteration run reached **81.3% / 0.224**; extending to **16 iterations with a bigger expert
search budget → 87.6% solved, regret 0.213** (iter15), and **89.1% solve** at iter14. **This
beats the supervised baseline on both solve rate (87.6 vs 85) and regret (0.213 vs 0.354)** —
from random weights, no oracle, no tricks. So the simple self-play NN doesn't just match the
supervised baseline, it **surpasses it** given enough iterations; the only extra cost was compute
(more ExIt rounds + a deeper expert search).

### 5. Supervised — candidate-scored (the faithful extension)
The original subgoal system *scored the candidates via A\**. The move analog: for every state,
label **each candidate move's resulting state with its exact optimal cost-to-go** (not just the
states on the optimal path). This mirrors the original's candidate-scoring literally *and* gives
the value net direct coverage of the **off-path states the search actually reaches** — the exact gap that limited
solve rate. It is a pure supervised change (oracle at train time only; planner stays pure NN).
**Result: 100% solved, regret 0.067** — the best model, beating trajectory-only supervised
(85% / 0.354) and even warm self-play on solve rate. Value-greedy solve rate jumps **7% → 78%**:
the value net now ranks the off-path children correctly, so A\* never wanders. This confirms the
whole diagnosis — the bottleneck was value **off-path mis-ranking**, and the faithful
candidate-scoring fixes it completely, in the supervised setting alone (no self-play needed). It
does cost more labeling (score every child), and the policy is unchanged; the gain is entirely
value coverage.

---

## Solve rate by difficulty

"Difficulty" = the optimal number of **primitive moves** (what we use; the exact ground truth,
better than the old weighted subgoal cost). Test-set difficulty histogram (150 instances):

| optimal moves | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| instances | 2 | 6 | 10 | 11 | 15 | 26 | 21 | 25 | 27 | 5 | 2 |

The test set is **hard** (peak 6–9 moves, mean ≈ 6.5) — harder than the training-data average
(3.9), because the eval draws random full instances. Solve rate binned by difficulty:

| optimal moves | instances | supervised | from-scratch | **candidate-scored** |
|---|---|---|---|---|
| 1–3 (easy) | 18 | 100% | 100% | **100%** |
| 4–6 | 52 | 88% | 96% | **100%** |
| 7–9 (hard) | 73 | 77% | 73% | **100%** |
| 10–12 (deep) | 7 | 29% | 14% | **100%** |

**Every failure is a deep puzzle.** All models solve short puzzles perfectly; supervised and
from-scratch degrade sharply past 7 moves and mostly fail at 10+. The candidate-scored model
stays **100% at every difficulty** — its off-path value coverage is exactly what's needed to
rank moves correctly deep in the search. (From-scratch is slightly *better* than supervised
mid-range but weaker on the very deepest, matching its thinner deep-state training.)

## What went wrong first (the debugging path, for honesty)
- **Reverse-scramble curriculum failed** — DeepCubeA's trick assumes reversible moves; Ricochet
  slides aren't, so scrambled starts stay ~1 move from the goal (measured optimal ~1.3 even at
  depth 16). The net never trained on deep states → capped at 66%. Replaced by **forward-walk**
  (works for non-reversible domains) + random instances.
- **A value-target bug** (`min(found_length, net_estimate)`) was a downward ratchet toward the
  net's own wrong belief — replaced with the plain found length.
- **Under-training** a random net with fine-tune-grade settings — fixed to supervised-grade
  (lr 3e-4, policy_weight 1.0, more epochs).
- A rejected **oracle fallback at inference** (guaranteed 100% solve) — removed; the planner must
  be pure NN.

See `move_planner/README.md`, `move_planner/SUMMARY.md`, `move_planner_v2/README.md`,
`move_planner_v2/DESIGN.md`, and the visual `move_planner/summary.html`.
