# move_planner_v2 — self-play (plain Expert Iteration)

The self-play successor to the supervised `move_planner/`. Same net, same search — but the
training targets come from the **net's own A\* search** instead of the oracle, so it
improves at runtime. Full method + rationale: **[`DESIGN.md`](DESIGN.md)**.

## Paradigm (one line)
**The current net's own budgeted A\* (`evaluate.nn_astar`) is the Expert-Iteration expert;
the net is the apprentice.** Solve puzzles with the net's search, train the net on the
solutions, repeat. Runs **from a random net** (pure self-play) or **warm** from
`move_planner/checkpoints/best.ckpt`. No oracle in training (oracle → eval-only regret).

## How it works (per iteration)
1. **generate** — sample **target-biased forward-walk** puzzles on train boards (random
   start + `k~U(1,16)` slides, relabel the target's final cell as the goal → guaranteed
   solvable), plus a 25% mix of raw random eval-distribution instances. Run the budgeted
   expert `nn_astar` (k=8, iters=4000 > eval budget). **Solved** paths → value = plain
   remaining length, policy = committed move. **Unsolved** → dropped.
2. **train** — load `MoveNet` from the latest ckpt (a fresh random net when `--from-scratch`),
   fit at **supervised-grade** settings (lr 3e-4, policy_weight 1.0, 6 epochs) on a replay
   window — **reusing `net.py`, `MoveDataset`, `collate`, and the exact loss unchanged**
   (self-play emits the same JSONL record schema).
3. **eval** — `evaluate.benchmark`: regret vs the exact oracle on a small held-out probe.

No difficulty knob to advance: net-solvability is the implicit curriculum, so the solvable
frontier widens on its own (deep-state share of the data climbs each iteration).

## Files
```
config.py         Config dataclass — every knob; pkl-backed split id lists
start_states.py   forward-walk + relabel generator (+ random eval-distribution mix-in)
selfplay.py       target generation (ExIt solved paths -> records; drop unsolved)
train_iterate.py  the outer loop + from_scratch/warm + CLI (entry point)
DESIGN.md         full design (paradigm, MDP, search, targets, curriculum, history)
```

## Run
```bash
cd supervised_valuenet && export PYTHONPATH=.
# from scratch — pure self-play, random init, no oracle
python -m move_planner_v2.train_iterate --from-scratch \
    --n-iters 18 --instances-per-iter 8000 --device auto
# warm start — fine-tune the supervised checkpoint
python -m move_planner_v2.train_iterate \
    --n-iters 6 --instances-per-iter 8000 --device auto
```

## Results (confirmed, full 450-instance test benchmark)

Same test set (held-out boards 2400–2549), NN A\* (k=5). Supervised baseline for reference.

| model | solved | regret | %optimal | notes |
|---|---|---|---|---|
| supervised baseline | 85% | 0.354 | 73% | oracle labels |
| **self-play — from scratch** | **87.6%** | **0.213** | 83% | random init, no oracle, no tricks (16 iters; iter14 hits 89.1% solve) |
| self-play — warm | 97.6% | 0.055 | 95% | fine-tune supervised (1 iter) |

**The headline: from random weights, pure self-play beats the supervised baseline on both
solve rate (87.6 vs 85) and regret (0.213 vs 0.354)** — no oracle, no tricks. A simple
self-play NN doesn't just match the supervised baseline, it surpasses it given enough
iterations; the only extra cost is compute (more ExIt rounds + a deeper expert search).

**Warm-start** confirms the mechanism: one iteration jumps to 97.6% / 0.055 because the
search generates better labels on the states it actually visits, fixing the value net's
off-path blind spot — **value-greedy solve rate 7% → 51%**.

The strongest model overall is a *supervised* variant — **candidate-scored** (label every
candidate move's resulting state with its exact cost-to-go): **100% solved, regret 0.067**.
It's the faithful analog of the original subgoal system's "score the candidates via A\*",
and it fixes the same off-path value gap in the supervised setting alone. Full comparison,
metric definitions and difficulty breakdown: **[`../RESULTS.md`](../RESULTS.md)**.

Checkpoints land in `runs_scratch*/` (from-scratch) and `runs_warm/` (warm). Deliberately no
MCTS/PUCT (branching ≤16 + a cost-to-go value make A\*-as-expert dominant).

Related: the supervised baseline `move_planner/`
([README](../move_planner/README.md), visual [summary.html](../move_planner/summary.html)).
