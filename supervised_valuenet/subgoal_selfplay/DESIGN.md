# subgoal_selfplay — self-play training for the subgoal planner

## 0. Summary

**The planner teaches itself.** Each round, the current networks solve a batch of random
puzzles with their own search (a traced fork of the eval planner,
`eval.end2end.nn_astar`). Every solved puzzle is turned into training data: at each
decision point along the winning plan, the chosen candidate's cost label comes free from
the winning plan itself, and each of the other top-ranked candidates is finished off with
a short run of the same net-guided search to measure what it would have cost. That makes
each decision a fully labeled group of candidates in the **exact `nn/data/combined.jsonl`
format**, so the existing supervised trainers (`train.looped_pc`, `train.policy_tf`) are
imported and reused unchanged. Puzzles the search cannot solve are simply dropped. Then
the networks are retrained on this data and the cycle repeats.

No exact-solver labels are ever used for training; the exact solver
(`skeleton.astar.AStar.solve_plan`) appears **only** inside the progress check as the
reference for how far from optimal the found plans are.

## 1. Why every candidate gets a label (the core design decision)

The obvious cheap variant — label only the move the search actually took — silently
breaks both trainers, because both are built around *comparing candidates within one
decision*:

- The value network's ranking loss compares all candidates of a decision and needs to
  know which one was best; a decision with a single labeled candidate contributes
  nothing (`train/looped_pc.py`, `_rank`).
- The proposal network's training target is a probability distribution spread over all
  candidates of a decision according to their costs (`train/policy_tf.py`); it needs a
  cost for every candidate, not just the winner.

So self-play must label whole decisions. For each decision on the winning plan:

```
cost(chosen)  = winning plan's total cost − cost already fixed before this decision
cost(sibling) = completed sibling plan's total cost − cost already fixed before this decision
best          = the candidate with the lowest such cost
```

A sibling whose short completion run fails is left out (its cost is unknown, not
infinite). This mirrors how the supervised labels were made (`nn/generate.py`
`rollout()`: every candidate is committed and solved to completion, then costed), with
the networks' own search standing in for the exact solver. Sibling costs found by search
are over-estimates at first and shrink as the networks improve, so the labels correct
themselves over rounds.

## 2. The loop

```
policy, value = warm-start checkpoints | fresh random networks    # --from-scratch
buffer = last 5 rounds of records
for each round:
    solve a batch of random puzzles with the current networks (traced search)
    label every decision on each winning plan (as above)
    add the records to the buffer
    retrain the value network  on the buffer   (same dataset/loss code as supervised)
    retrain the proposal network on the buffer (same dataset/loss code as supervised)
    save round checkpoints
    progress check: solve a fixed probe set, compare plan costs to the exact solver
```

### 2.1 The search that generates the data (`selfplay.nn_astar_traced`)

A fork of `eval.end2end.nn_astar` that remembers, for every plan it builds, which
decision produced it — otherwise line-for-line the same loop: take the most promising
partial plan; fill in any segments that have a known exact path (no decision needed);
list the candidate subgoals (`heuristics.propose`); rank them with the proposal network
in one pass; score the top `k_top` with the value network; queue the resulting child
plans. When a complete plan is reached, walking its recorded parents recovers the full
chain of decisions.

**Budget rule**: data generation searches harder than the evaluation setting
(`k_top=8`, `astar_iters=1500` versus eval `k=5`, `1200`), so the plans found during
training are a bit better than what the eval-budget planner would produce — that gap is
what the networks learn from. Sibling completions use a small budget
(`sibling_iters=300`) because they only finish an almost-committed plan.

### 2.2 Occasional wildcard candidates (data generation only)

With probability 0.15 per decision (`epsilon` in the config), one or two randomly chosen
candidates from outside the top ranks are added to the scored group. This gives the
training data occasional examples the current ranking would never surface, so a blind
spot in the ranking cannot silently persist forever. The evaluation search never does
this.

### 2.3 Where the puzzles come from (`start_states.py`)

Puzzles are sampled exactly like the supervised data generator
(`nn.generate.random_instance`), with one cheap filter: the target must be reachable on
the board graph at all (a connectivity check, not a solve). **Unsolved puzzles are
dropped**: as the networks improve, the set of puzzles they can solve grows on its own —
no difficulty schedule needed. An optional dial (`walk_relabel_prob`, default OFF) can
bias early from-scratch rounds toward easier targets; it only changes which puzzles are
sampled, never how they are labeled. Two optional flags are OFF by default and not part
of the baseline: that dial, and `strict_filter` (which would drop solved puzzles whose
plan does not execute legally move-by-move).

Deliberately absent, because they measurably failed in this codebase's earlier self-play
work: training on artificially scrambled near-goal states; relabeling failed attempts as
successes toward wherever they ended up; blending the network's own guess into the
training label (which drags labels toward the network's existing errors).

### 2.4 Memory between rounds and training strength

The buffer keeps the last 5 rounds of records so retraining does not forget the previous
rounds. Training reuses the supervised datasets and losses unmodified; only the strength
differs by mode: **warm start** fine-tunes gently (learning rate 1e-4, 4 passes over the
data), **from scratch** trains at full supervised strength (3e-4, 6 passes). The trainers
run in-process and write checkpoints into the run directory.

### 2.5 Progress check

Each round, a fixed set of 50 held-out puzzles (25 boards × 2) is solved with the plain
evaluation search at the evaluation budget (`k=5`, 1200), and the found plan costs are
compared against the exact solver's optimal plan costs. This is the only place the exact
solver appears. It is a trend indicator; final claims come from the shared 450-instance
benchmark.

## 3. Data format

Records are emitted by `nn.collect_search._record` and are field-for-field identical to
`nn/data/combined.jsonl`:

```
{env_id, target, target_robot, helpers, seg_start, seg_end, seg_support, mover_color,
 ctx_bottlenecks, ctx_supports, ctx_open_endpoints, cand_bottleneck, cand_support,
 cand_helper, cand_parent_support, cost_to_go, is_optimal, depth}
```

`depth` is the position of the decision along the winning plan. Decisions are re-grouped
for training by `nn.benchmark.group_by_decision`, the same grouping the supervised
trainers use. Each round's records are also written to `out_dir/iterN_records.jsonl`.

## 4. Modules

```
subgoal_selfplay/
  config.py         # Config dataclass — every knob in one place; split ids gated on which boards exist
  start_states.py   # random puzzle sampling + reachability filter + optional easy-target dial
  selfplay.py       # traced search, sibling completion labeling, one generation round
  train_iterate.py  # outer loop: generate -> buffer -> retrain both networks -> progress check; CLI
  DESIGN.md         # this file
```

Imported unchanged (never copied): `skeleton.astar.{AStar, _initial_plan, _segment,
_apply}`, `skeleton.heuristics.propose` (via the solver), `nn.generate.{random_instance,
_context, _fixed_g}`, `nn.collect_search._record`, `nn.benchmark.group_by_decision`,
`train.looped_pc.{LoopedValueNet, DenseDataset, collate}`, `train.policy_tf.{PolicyTF,
PolicyTFDataset, collate}`, `train.policy_common._ix`, and the inference helpers
`eval.end2end.{_policy_logp, _value_cost, _hidx}` (so ranking and scoring behave exactly
as they do at evaluation). The only copied-and-modified code is the search loop itself
(it needs parent tracking, an arbitrary starting plan, and the wildcard candidates —
none of which can be added from outside).

## 5. Settings (defaults in `Config`)

| group | value |
|---|---|
| mode | warm start (requires `--policy` + `--value`) or `--from-scratch` |
| boards | training `(1000, 1799)`, progress-check `(1800, 2399)` — only boards whose files exist |
| outer loop | 6 rounds, 2000 puzzles per round, at most 8 labeled decisions per puzzle |
| data-generation search | `k_top=8`, `astar_iters=1500` (harder than eval 5 / 1200) |
| sibling completions | `sibling_iters=300`; a failed completion drops only that record |
| wildcard candidates | probability 0.15, 1–2 per firing, data generation only |
| puzzles | reachability-filtered random; easy-target dial OFF; strict-execution filter OFF |
| buffer | last 5 rounds |
| training | warm: 1e-4, 4 passes; scratch: 3e-4, 6 passes; value batch 8 decisions, proposal batch 16, at most 32 candidates per decision |
| progress check | 25 boards × 2, `k=5`, 1200, every round |

## 6. Success criterion

Match or beat the retrained supervised version of this planner at the evaluation budget
(`k=5`, `astar_iters=1200`) on the shared 450-instance benchmark — same boards, seeds,
and budgets. From scratch is the stretch goal: even with random networks the planner
solves some puzzles, because segments with a known exact path complete without any
learned decision, so data generation produces records from round one and the loop can
start without any supervised warm start.
