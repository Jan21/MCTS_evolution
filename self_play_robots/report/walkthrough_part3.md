# Part 3 — The self-play loop, what it bought, and the action-space breakthrough

This part follows Part 1 (setups, metrics, non-learned solvers) and Part 2 (the two
action spaces and the plan-language extensions). It answers three questions.

1. What does one round of the self-play loop actually do, step by step?
2. What did the loop measurably buy, and what did other changes buy instead?
3. What is the "breakthrough", and why does it work?

Every number in this part was read from a file in the repository during the writing of
this part. Section 9 lists each number with its file and the way it was read. Where the
project's own prose and the code disagree, this part follows the code and flags the
disagreement in section 10.

---

## 0. A compact glossary

Part 1 and Part 2 define these terms. This list repeats them so that this part reads
without a second window.

| term | meaning |
|---|---|
| **instance** (or puzzle) | one board, one start cell per robot, one target robot, one goal cell |
| **slide** (or move) | one robot moves in one direction until a wall or another robot stops it |
| **d\*** | the exact minimum number of slides for an instance, from an exhaustive solver |
| **strict moves** | the number of real slides in a legal joint-state execution of a plan (`supervised_valuenet/eval/realize.py::strict_moves`) |
| **regret** | strict moves minus d\*, on one instance |
| **partial plan** | the subgoal planner's state: a small graph of committed subgoals with some edges still open |
| **decision** | resolving the first open edge of a partial plan by choosing one candidate subgoal |
| **base vocabulary** | the original subgoal language (static supports only) |
| **B2 vocabulary** | the extended subgoal language: transient supports, supports named by robot identity, and deterministic "park" repairs at certification |
| **expansion** | the budget unit: one node whose children are generated, which costs one policy pass and one batched value pass over at most k children |
| **graded exam** | `scaling/data/g24r4/bench.solved.jsonl`, 232 instances on boards 900-1049, each with a known d\* |
| **frontier exam** | `scaling/data/g24r4/bench.unsolved.jsonl`, 218 harder instances on the same boards, no known d\* |
| **unseen exam** | `results/variants/exam/g24r4_unseen.jsonl`, 200 instances on 50 fresh boards, ids 20000-20049, built after the loop finished |
| **certified** | a complete plan that `strict_moves` executed successfully under full joint physics |
| **replay-validated** | the emitted move list was re-executed by an independent checker that imports only the physics layer (`eval/replay_validate.py`) |
| **A\* row / MCTS row** | the same nets benched with best-first search or with PUCT tree search |

All exams below use the pinned protocol: budget 1200 expansions, k = 5 children per
expansion.

---

## 1. Section A — one round of the self-play loop, mechanically

### 1.1 The files

| file | role |
|---|---|
| `self_play_robots/spr/selfplay.py` | generation: fresh boards, random instances, search, record extraction |
| `self_play_robots/spr/search.py` | the searches: `expand`, `mcts`, `astar`, `greedy`, and `Certifier` |
| `self_play_robots/spr/buffer.py` | assembles the training buffer from the last few rounds |
| `self_play_robots/spr/train.py` | trains both nets, warm start, checkpoint on minimum validation regret |
| `self_play_robots/spr/gauge.py` | fidelity gauge: compares the round's labels against exact labels |
| `self_play_robots/spr/bench.py`, `spr/arena.py` | the exam runner and its replay certification |
| `self_play_robots/spr/gate.py` | paired statistics against the previous round and against a fixed baseline |
| `self_play_robots/jobs/selfplay_iter.slurm` | the job script that runs one round end to end |
| `self_play_robots/jobs/selfplay_mix_iter.slurm` | the same round across three board configurations |

### 1.2 Three loops actually ran

The job script has defaults, and the runs overrode them. The table below states the
values that the runs actually used. These come from the `search` block of each round's
`generation.manifest.json`, not from the script defaults.

| loop | rounds | boards per round | board ids | instances per board | search budget | stop rule | vocabulary | anchors |
|---|---|---|---|---|---|---|---|---|
| base-vocabulary loop | 2 | 120 | 5000-5239 | 8 | 600 expansions | stop 150 | base | yes |
| B2 loop at 24x24 | 5 | 60 | 8000-8299 | 8 | 300 expansions | stop 80 | B2 | no |
| mixed-size curriculum | 3 | 30 per configuration | 9000-9089 | 8 | 300 expansions | stop 80 | B2 | no |

"Stop 80" means the search halts once 80 expansions have passed without an improvement
to the best certified plan. "Anchors" means the exact supervised corpora are added to
the training data. The job script forces anchors off whenever the vocabulary is B2.

The rest of this section describes the B2 loop at 24x24, because that is the loop the
project's headline claims rest on. Differences for the other two loops are noted.

### 1.3 One round, numbered

**Step 1 — write fresh boards.**
`spr.selfplay` writes 60 new boards through `nn_labeler.leanboard.write_board`, using
the round number as the random seed. Round k uses ids `8000 + 60*(k-1)` to
`8000 + 60*k - 1`. The boards live in `runs/spr/boards/g24r4_b2_iter<k>`.

**Step 2 — refuse any board id that could leak into an exam.**
Before anything else, `main` computes `bad = [i for i in ids if i <= 1199]` and raises
`SystemExit` if that list is not empty. Every pinned exam board of every configuration
lives in the range 0-1199. The graded and frontier exams use ids 900-1049. So an exam
board cannot enter the training data. This is a hard failure, not a warning.

**Step 3 — start W worker processes.**
The B2 loop used 6 workers. Each worker is a fresh spawned process. Each worker loads
its own copy of the policy net and the value net onto the GPU, builds the B2 solver
(`nn.generate.make_solver("b2")`), and builds an `Evaluator` with by-reference helper
resolution enabled.

**Step 4 — draw instances.**
For each board, a worker draws instances with `nn.generate.random_instance`. That
function samples `len(colors) + 1` distinct cells uniformly from the board graph. The
first cell is the target robot, the rest are helper robots, and the last is the goal.
There is no difficulty filter and no curriculum. The worker keeps drawing until 8
instances have produced at least one training record, and it gives up after 24 attempts.

**Step 5 — search one instance.**
The worker runs `spr.search.mcts` with these settings:

- budget 300 expansions, k = 5 children per expansion,
- `best_at_budget=True`, so the search keeps the cheapest certified plan rather than the first,
- `c_puct = 1.5`, backup `min`,
- Dirichlet root noise with weight 0.25 and alpha 0.3,
- `root_k = 0`, which means "all candidates at the root", not the top 5,
- `stop_after_certified = 80`,
- park repairs enabled (B2),
- a per-instance wall-clock cap of 600 seconds, enforced by `SIGALRM`.

PUCT selection uses `norm(Q) + c_puct * prior * sqrt(N_parent + 1) / (1 + N_child)`.
Q is a cost, so `norm` maps the lowest cost in the tree to 1 and the highest to 0. A
child is born with its own estimate `v_est = fixed_g(parent) + ctg_hat`, so there is no
first-play-urgency hack.

**Step 6 — certify every complete plan.**
When selection reaches a complete plan, `mcts` calls the `Certifier`. The `Certifier`
calls `eval.realize.strict_moves`, which topologically orders the plan's physical
segments and executes them one at a time on the full joint state with real slide
physics. It returns the total slide count, or `None` if any segment cannot be executed.

- If it returns a number, the node becomes `solved`. Its abstract plan cost becomes its
  Q value. The number and the abstract cost propagate up the path as `best_cert` and
  `best_abs`.
- If it returns `None`, the node is dead, unless the B2 park repairs produce repaired
  variants. Those variants become children of the failed node and re-enter the search
  as complete plans with a physics-derived cost.

**Step 7 — extract records from the tree (`_extract_records`).**
This is where a plan becomes training data.

1. Follow the principal path down from the root. At each node, the next node is the
   child whose `best_cert` equals the node's own `best_cert`. This is the chain of
   decisions that produced the best certified plan.
2. At each node on that path, examine every child that was a real decision. Park-repair
   children are excluded, because they carry no candidate record.
3. A child is labelled if and only if `best_cert` is not `None`, which means some
   complete plan below that child was strictly realized.
4. If a child was never certified, and the `--complete-siblings` flag is on (it was on
   in every run), the worker attempts a greedy completion of that child:
   `_greedy_complete` repeatedly expands, takes the child with the smallest predicted
   cost-to-go, and applies free exact fixes, for at most 16 steps. If the result is a
   complete plan, it is passed to the same `Certifier`. If certification fails, the
   sibling stays unlabelled and is dropped.
5. A decision emits a training group only if at least 2 of its children were labelled.
6. The label of each labelled child is

   ```
   cost_to_go = abstract cost of the certified plan  -  fixed_g(the decision's plan)
   is_optimal = (that abstract cost equals the minimum over the group)
   ```

   The strict move count of the same certified plan travels with the record as
   `strict_total`, and the abstract cost as `abstract_total`.

Each record also carries `visits` (the child's visit count), `prior` (the policy
probability), `ctg_hat` (the value net's prediction), `label_source = "spr_mcts"`,
`vocab`, `iter`, `label_model` and `boards_dir`.

**Step 8 — write the round's data.**
The parent process writes `records.jsonl` atomically, plus a manifest and a per-instance
statistics file. The five B2 rounds produced:

| round | instances attempted | instances solved | records | mean strict moves of the solved plans | mean expansions | generation wall time |
|---|---|---|---|---|---|---|
| 1 | 681 | 598 | 5,822 | 13.72 | 108.1 | 4,301 s |
| 2 | 655 | 588 | 5,694 | 13.78 | 102.9 | 4,064 s |
| 3 | 660 | 582 | 6,301 | 14.10 | 109.2 | 4,197 s |
| 4 | 675 | 600 | 6,069 | 14.15 | 102.9 | 4,136 s |
| 5 | 676 | 609 | 6,661 | 13.83 | 98.5 | 3,868 s |

So one round costs about 1.1 hours of one GPU for generation and yields about 6,000
labelled decisions from about 600 solved instances.

**Step 9 — the fidelity gauge.**
`spr.gauge` samples up to 200 depth-0 decisions from the round's records, labels the same
decisions exactly with the Rust engine, and runs `nn_labeler.audit_descent` to compare.
The key output is `argmin_agreement`: how often the round's own best candidate is also
an exactly-optimal candidate. Section 1.6 reports the numbers.

**Step 10 — build the training buffer.**
`spr.buffer` concatenates the record files of the last 3 rounds. Inside each round it
sorts the board ids and assigns the last 15% of the ids to validation and the rest to
training. Validation is therefore board-disjoint from training by construction. The
buffer writes a `--splits` string for the trainer and a map from board id to board
directory.

**Step 11 — train both nets.**
`spr.train` runs twice, once for the policy and once for the value net.

| setting | policy | value |
|---|---|---|
| warm start | previous round's policy checkpoint | previous round's value checkpoint |
| epochs | 6 | 6 |
| learning rate | 1e-4 | 1e-4 |
| optimiser | AdamW, weight decay 1e-4, no scheduler | AdamW, weight decay 1e-4, no scheduler |
| batch | 8 decision groups | 4 decision groups, at most 8 records per group |
| extra | `--grad-clip 1.0`, `--byref` | none |
| target | softmax of `-cost_to_go` over the group's candidates, temperature 1.0 | 96-bin distribution with an HL-Gauss soft target, plus a within-group ranking loss |
| checkpoint | minimum `val_regret` | minimum `val_regret` |
| guard | none | `CollapseStop`: halt after 3 epochs with `val_group_spread` below 0.05 |

`val_regret` is the within-group regret of the net's own argmin. The scalar value of the
value net is the expectation of the 96-bin distribution.

The learning rate default in `spr.train` is 3e-4 for a cold start and 1e-4 for a warm
start. The job scripts pass 1e-4 explicitly.

**Step 12 — bench and gate.**
The round's new nets are benched on the graded exam and on the frontier exam, with two
searches: the arena A\* loop and PUCT best-at-budget. Both use 1200 expansions and
k = 5. The bench runner merges the shards and then runs `eval.replay_validate` on the
merged payload. If replay fails, the result file is quarantined and the job fails.
Finally `spr.gate compare` produces a paired comparison against the previous round:
McNemar's exact test on the solved vectors, and a sign test on moves over the instances
that both rounds solved.

The mixed-size job is identical, except that generation, buffering and benching run for
three configurations (24x24 with 4 robots, 32x32 with 4 robots, 24x24 with 8 robots),
and one single size-free net pair trains on the union of the three buffers.

### 1.4 Where do the training answers come from? Measurement.

The answer is measurement, not network prediction. The evidence is in
`_extract_records` in `self_play_robots/spr/selfplay.py`.

- The label is `int(ab - fixed_g)`, where `ab` is `c.best_abs`.
- `c.best_abs` is set in `mcts` only inside the branch where `cert(node.plan)` returned a
  number. That number comes from `eval.realize.strict_moves`.
- The value net's own prediction, `ctg_hat`, is written into the record, but under the
  key `ctg_hat`. It is provenance. It is never the label. Nothing reads it back during
  training.
- `is_optimal` compares measured quantities against each other inside one group.

The network does choose *which* plans get measured. The policy ranks candidates, the
value net drives PUCT selection, and the greedy sibling completion follows the value
net. So the network shapes the search. It never supplies a number that becomes an answer.

One nuance about units. The label is the *abstract plan cost* of the certified plan, not
its strict move count. Abstract cost is the plan's own cost model. Strict moves is the
metric the exams score. They differ. The project treated this as a design flaw and tested
the fix as a lab variant, `v09_strict_value`, which rescales each label by
`strict_total / abstract_total` before training. That variant became the flagship's value
net. The main loop rounds 1-5 kept the abstract units.

### 1.5 What makes a wrong label structurally impossible

A "wrong label" would be a claim that a plan reaches the goal in N moves when it does not.
Three code facts exclude that.

1. **A label needs a realized plan.** In `spr/search.py::mcts`, a node is marked
   `solved` and given `best_cert` only inside `if m is not None`, where
   `m, mv = cert(node.plan)`. `Certifier.__call__` calls `eval.realize.strict_moves`,
   which executes the plan segment by segment on the full joint state with real slide
   physics and returns `None` on the first segment it cannot execute. A plan that cannot
   be played produces no label at all.
2. **Only certified subtrees propagate.** `best_cert` and `best_abs` move up the path
   only from a node that certified. `_extract_records` skips any child whose `best_cert`
   is `None`. A child with no certified descendant produces nothing.
3. **The greedy sibling path uses the same gate.** `_greedy_complete` returns a plan, and
   `_extract_records` then calls `m, _mv = cert(done)`. If `m` is `None`, the counter
   `sibling_failed` increases and the sibling is dropped.

A second, independent check applies to every exam row. `spr.arena.bench` runs
`eval.replay_validate` on the merged payload before it accepts the result. That module
imports only `simulate.slide` and `simulate.wall_sets` plus the standard library. It
reads the board's raw wall strings straight from the pickle. It replays the move list
from the instance's start positions and requires three things: every move is a real slide
that changes the robot's cell, the sequence length equals the claimed move count, and the
target robot ends on the goal. It never touches the planner or the realizer. If it fails,
`spr.arena` renames the file to `.json.uncertified` and exits with an error.

So the loop can be wrong about *which* plan is best. It cannot be wrong about *whether*
a plan works or about how many moves it costs.

### 1.6 What error can still enter the data

The remaining error is a valid but longer-than-necessary plan. The label is a certified
upper bound on the true cost-to-go. If the search never finds the shortest plan through a
candidate, the candidate's label is too large. If the error is uneven across candidates
in a group, then `is_optimal` can point at the wrong candidate.

The fidelity gauge measures exactly this. It labels the same depth-0 decisions with the
exact engine and compares. `gap = self-play label - exact label`. Negative gaps would be
a bug, because the self-play label is an upper bound.

| loop | round | argmin agreement | mean gap | median gap | negative gaps | matched decisions |
|---|---|---|---|---|---|---|
| base | 1 | 0.915 | 0.99 | 0.0 | 0 | 200 |
| base | 2 (hard-instance filter) | 0.850 | 1.29 | 0.0 | 0 | 200 |
| B2 | 1 | 0.513 | 3.93 | 2.0 | 0 | 197 |
| B2 | 2 | 0.500 | 4.12 | 2.0 | 0 | 74 |
| B2 | 3 | 0.568 | 3.98 | 1.0 | 0 | 74 |
| B2 | 4 | 0.451 | 4.73 | 3.0 | 0 | 195 |
| B2 | 5 | 0.460 | 4.76 | 3.0 | 0 | 189 |

Two readings follow.

- **No negative gap ever appeared.** Across all seven gauge runs, `negative_gaps` is 0.
  That is the empirical confirmation of section 1.5.
- **In the B2 vocabulary the labels are loose.** The gauge's own calibration, recorded in
  its module docstring, is that about 91% agreement is downstream-equivalent to exact
  labels, about 89% keeps the solve rate but loses optimality, and about 82% means
  collapse. The B2 rounds sit at 0.45-0.57. The base rounds sit at 0.85-0.92.

The project reads the B2 gauge as uninformative rather than as a collapse signal, because
the exact engine's B2 rollouts are capped, so the "exact" side is itself weaker in B2. I
could not verify that reading from the files, and I flag it in section 10. What the
numbers do support without interpretation is the direction of the error: the loop's B2
labels are on average 4 cost units above the exact answer, and never below it. That is
consistent with the loop's measured outcome. It learned to find *a* plan faster and more
often. It never learned to find a *shorter* plan.

### 1.7 The other labelling mechanism, and why the loop does not use it

The project has a second, different labeller, and the two are easy to confuse.

**The certified-descent labeller**, `supervised_valuenet/nn_labeler/descent.py`, prices a
candidate by *greedy network completion* instead of by exhaustive solving. It commits the
candidate, then repeatedly takes the value net's best-scored candidate until the plan is
complete, then realizes the result. Its records carry `label_source = "nnlab_descent"`.
Its purpose is to remove the cost wall of the exact labeller. It produced the "twin"
corpora `nn_labeler/results/twin_<config>.jsonl`, and the per-size supervised planners were
retrained on them. The retrained planner is recorded in
`scaling/results/g24r4/comparison_nntwin.json`: 212 of 232 solved, mean regret 3.047,
against the exact-labelled planner's 205 of 232 and mean regret 4.220.

**The self-play loop does not use it.** `spr.selfplay` never imports `nn_labeler.descent`.
It writes `label_source = "spr_mcts"`. Its labels come from plans that a tree search found
and that the certifier executed.

There is one honest overlap. The loop's `--complete-siblings` option runs
`_greedy_complete`, which uses the same *idea* as the descent labeller: follow the value
net greedily to a complete plan. Every run used this option. But the resulting plan is
still certified by `strict_moves` before it can carry a label, and the label is still the
measured plan's cost. So the network picks the plan. The physics engine supplies the
number. In neither mechanism does a network output become a training target.

---

## 2. Section B — what self-play measurably bought, decomposed

The headline claim to test is: the loop bought solve rate and search efficiency, it did
not shorten solutions, the large move-quality gain came from rebuilding the networks
before self-play started, and switching to the richer vocabulary made the *supervised*
planner worse.

All four hold. The decomposition below quotes only rows on the same 232-instance graded
exam, so every pair of numbers differs in exactly one thing.

### 2.1 The decomposition table (graded exam, 232 instances, 1200 expansions, k = 5)

| # | system | what changed vs the row above | solved | mean moves | mean regret | % optimal | mean expansions |
|---|---|---|---|---|---|---|---|
| 1 | per-size supervised backward pair, base vocabulary, A\* | — (the starting point) | 205 | 11.81 | 4.220 | 38.5 | 26.9 |
| 2 | per-size supervised backward pair, **B2 vocabulary**, A\* anytime | vocabulary only | **199** | 12.33 | **4.889** | 36.7 | 53.5 |
| 3 | **size-free rebuilt** pair (`mixed_value_warm_s21`), base vocabulary, A\* | net architecture and training data, same vocabulary as row 1 | **215** | 10.00 | **2.433** | 54.4 | 5.3 |
| 4 | same rebuilt pair, **B2 vocabulary**, A\* anytime | vocabulary only, vs row 3 | **223** | 9.51 | 1.946 | 55.6 | 52.8 |
| 5 | same rebuilt pair, B2 vocabulary, **PUCT best-at-budget** | search only, vs row 4 | **228** | 9.19 | 1.553 | 58.8 | 490.3 |
| 6 | after **5 self-play rounds**, B2, A\* anytime | training data only, vs row 4 | **227** | 9.85 | 2.233 | 54.2 | **30.3** |
| 7 | after 5 self-play rounds, B2, PUCT | training data only, vs row 5 | 228 | 9.18 | 1.548 | 57.5 | 543.1 |
| 8 | mixed-size curriculum, round 3, B2, A\* anytime | training data (three board sizes) | **230** | 9.87 | 2.200 | 55.2 | 24.1 |
| 9 | *reference*: exhaustive B2 language optimum | no network at all | 228 proven realizable | 8.81 | 1.171 | 62.7 | — |
| 10 | *reference*: supervised forward move planner | different action space entirely | 220 | 7.57 | 0.068 | 94.1 | 191.0 |

Rows 1 and 2 use the per-size supervised networks. Rows 3 to 8 use the size-free
networks. Row 1 was also re-run on this machine and reproduced at 205 solved and mean
regret 4.205, so the cross-machine comparison is sound.

### 2.2 The rebuild bought the move quality (row 1 to row 3)

The size-free rebuild changed the network family, not the search and not the vocabulary.
It moved regret from 4.220 to 2.433, moves from 11.81 to 10.00, optimality from 38.5% to
54.4%, and solves from 205 to 215. It also cut the expansions from 26.9 to 5.3. This
happened before any self-play data existed. The nets of row 3 are exactly the nets that
seeded the B2 loop as its round 0.

### 2.3 The richer vocabulary made the supervised planner worse (row 1 to row 2)

Row 2 is the same per-size supervised pair, under the B2 inference convention. Solves
fall from 205 to 199. Mean regret rises from 4.220 to 4.889. A paired comparison over the
232 instances is sharper: on the 186 instances both arms solve, the B2 arm writes a
shorter solution on 15 and a longer one on 41. The base arm alone solves 19 instances, the
B2 arm alone solves 13.

The same vocabulary switch *helps* the rebuilt size-free nets (row 3 to row 4): 215 to 223
solves and regret 2.433 to 1.946. So the vocabulary is not bad. The per-size supervised
nets could not rank its extra candidates.

### 2.4 The base-vocabulary loop: a clean null result

Two rounds ran in the base vocabulary before the project pivoted.

| round | graded A\* solved | graded A\* regret | graded PUCT solved | graded PUCT regret | frontier A\* solved |
|---|---|---|---|---|---|
| seed nets (row 3) | 215 | 2.433 | — | — | 99 of 218 |
| 1 | 216 | 2.477 | 216 | 1.856 | — |
| 2 | 215 | 2.298 | 215 | 1.679 | 100 of 218 |

Nothing moves. The reason is measured, not guessed: the exhaustive base-language search
finds a realizable plan on only 216 of the 232 graded instances and 103 of the 218
frontier instances. The seed nets already solve 215 and 99. There is nothing left in that
language for a loop to learn.

### 2.5 The B2 loop, round by round, with denominators

Round 0 is the seed pair of row 3, benched under B2 flags.

| round | records | gauge | graded A\* (of 232) | graded A\* regret | graded A\* expansions | graded PUCT (of 232) | graded PUCT regret | frontier A\* (of 218) | frontier PUCT (of 218) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | — | — | 223 | 1.946 | 52.8 | 228 | 1.553 | 133 | 171 |
| 1 | 5,822 | 0.513 | 226 | 1.991 | 43.0 | 229 | 1.563 | 144 | 169 |
| 2 | 5,694 | 0.500 | 227 | 2.207 | 35.5 | 229 | 1.581 | 139 | 172 |
| 3 | 6,301 | 0.568 | 225 | 1.867 | 35.9 | 228 | 1.504 | **158** | 171 |
| 4 | 6,069 | 0.451 | **228** | 2.083 | **25.9** | 228 | **1.465** | 153 | **178** |
| 5 | 6,661 | 0.460 | 227 | 2.233 | 30.3 | 228 | 1.548 | 153 | 174 |

Paired tests, round 5 against round 0, computed for this document with the project's own
gate arithmetic:

| comparison | round 5 solved | round 0 solved | 5-only | 0-only | McNemar p | both solved | mean moves 5 vs 0 | moves wins 5 / 0 | sign p |
|---|---|---|---|---|---|---|---|---|---|
| frontier A\* | 153 | 133 | 27 | 7 | **0.00082** | 126 | 19.04 vs 19.14 | 22 / 27 | 0.57 |
| graded A\* | 227 | 223 | 6 | 2 | 0.29 | 221 | 9.46 vs 9.47 | 16 / 17 | 1.00 |
| graded PUCT | 228 | 228 | 2 | 2 | 1.00 | 226 | 9.12 vs 9.05 | 13 / 18 | 0.47 |
| frontier PUCT | 174 | 171 | 12 | 9 | 0.66 | 162 | 18.72 vs 18.73 | 23 / 25 | 0.89 |

The shape is unambiguous.

- **Solve rate on the hard exam rose and the rise is significant.** 133 to 153 of 218 on
  the A\* row, p = 0.00082. Round 3 alone reaches 158, p = 4.7e-6 against round 0.
- **Search efficiency improved a lot.** The cheap first-solution A\* went from 52.8
  expansions to 25.9-30.3, while its solve count rose from 223 to 227-228. On the graded
  exam the A\* row caught the 490-expansion PUCT row. This is search distilled into the
  network, which is the AlphaZero mechanism.
- **Move quality did not improve.** In every paired test, the move sign test is not
  significant, and the A\* regret drifts upward from 1.946 to 2.233 as the planner solves
  harder instances on the first attempt. The PUCT regret stays inside 1.46-1.58.
- **The 1200-expansion reach did not move.** The frontier PUCT row is 171 at round 0 and
  174 at round 5, p = 0.66.

### 2.6 The mixed-size curriculum

One iteration of generation on three board distributions at once produced the largest
single jump of the loop line, and then flattened.

| round | g24r4 graded (232) | g24r4 frontier (218) | g32r4 graded (175) | g32r4 frontier (275) | g24r8 graded (161) | g24r8 frontier (289) |
|---|---|---|---|---|---|---|
| 1 | 226, regret 2.040 | **170** | **173**, regret 2.613 | **235** | 159, regret 2.346 | **269** |
| 2 | 227, regret 2.189 | 165 | 170, regret 2.147 | 224 | 159, regret 2.277 | 265 |
| 3 | **230**, regret 2.200 | 165 | 172, regret 2.227 | 225 | 159, regret 2.346 | 268 |

Round 1's 170 on the g24r4 frontier is above the 24x24-only loop's best of 158. Every
round-3 against round-2 gate is not significant, with McNemar p at or above 0.25 and every
move sign test at or above 0.56. Again, solves rose and regret did not fall.

The g24r4 graded 230 of 232 at round 3 is the best pure-subgoal solve count in the project.
It also proves that at least 230 of the 232 graded instances have a realizable B2 plan,
which is above the exhaustive probe's 228 proven cases. The probe left 4 instances
inconclusive at its caps.

### 2.7 The from-scratch control, and exactly what it controls for

`self_play_robots/variants/v08_cold_start.py` runs one matched round with a single change:
`spr.train` runs **without** `--init`, so both networks start from random initialization.
The value net gets the cold recipe, meaning 4 curriculum warmup epochs and learning rate
3e-4, and `CollapseStop` guards it. Generation is unchanged and uses the frozen seed nets,
so the data of the cold arm and the data of the control arm are the same kind of data.

That is the point of the design. The arm isolates **the training prior**. It does not
isolate "self-play without any supervision", because the *generation* nets are still the
supervised-descended seed pair. It answers one question: at one round's data volume, how
much of the planner's competence comes from the supervised warm start?

The round produced 329 instances, 306 solved, 3,787 records.

| exam | cold start | matched control (v00, same protocol, same seed) | paired |
|---|---|---|---|
| graded (232) | 200 | 228 | control-only 29, cold-only 1, McNemar p = 5.8e-8 |
| frontier (218) | 96 | 160 | control-only 70, cold-only 6, McNemar p = 6.3e-15 |
| unseen (200) | 132 | 172 | control-only 41, cold-only 1, McNemar p = 2.0e-11 |

The reference point that matters is the unseen exam. The per-size supervised backward pair
solves 134 of 200 there. The cold arm solves 132. So one round of certified self-play data,
from random weights, reaches the supervised planner's level on fresh boards. The
supervised warm start is worth about 40 unseen solves on top of that.

Two limits on this control. It is one round only, and its generation used trained nets.

---

## 3. Section C — the breakthrough, step by step

### 3.1 Why only an action-space change could help

Part 2 describes the plan languages. The relevant fact is a *measurement*, and it is worth
restating precisely, because the whole of section 3 exists because of it.

`self_play_robots/spr/ceiling.py` runs an exhaustive best-first search over partial plans,
ordered by abstract plan cost, with **no network anywhere**. It strictly realizes every
complete plan it pops. It records `best_realizable_moves`, the cheapest certified plan it
finds before the popped abstract cost reaches that minimum. On the graded exam:

| vocabulary | instances with a realizable plan | mean d\* | mean best moves | mean regret | % reaching d\* |
|---|---|---|---|---|---|
| base | 216 of 232 | 7.59 | 9.31 | **1.722** | 57.4 |
| B2 | 228 of 232 | 7.64 | 8.81 | **1.171** | 62.7 |

The forward move planner, which plans in the primitive-move space, sits at mean regret
0.068 and 94.1% optimal on the same exam.

So a perfect ranker with unlimited search inside the B2 subgoal language still writes
solutions about 1.17 moves longer than optimal. Better networks cannot pass that. Better
search cannot pass that. Only a change to the set of actions can.

Two honest qualifications. The floor is a measurement, not a proof. The module's own
docstring notes that a plan whose strict count undercuts its abstract cost could sit
beyond the search's bound. And the B2 probe capped 7 of 232 instances, of which 3 still
found a realizable plan, so their "best" is not proven.

### 3.2 The algorithm

The file is `self_play_robots/variants/v07_hybrid_actions.py`. It is a standalone bench
driver. Generation and training are untouched. The whole algorithm is inline in
`bench_main`.

The constants below are the **flagship** constants, taken from
`jobs/v07_flagship.slurm` and from leg A of `jobs/v07_wave4.slurm`, and confirmed in the
`search` block of every row of the result payloads. They are **not** the module defaults.

| constant | flagship (depth 2) | depth-1 arm | depth-3 probe |
|---|---|---|---|
| `--b0` (budget for the search on the untouched position) | 500 | 600 | 460 |
| `--top-m` (prefix candidates that get a sub-search) | 8 | 6 | 10 |
| `--sub` (budget per sub-search) | 80 | 100 | 70 |
| `--prefix-depth` | 2 | 1 | 3 |
| `--d2-from` (how many prefixes of each depth are extended) | 4 | — | 4 |
| `--expansions` (overall cap) | 1200 | 1200 | 1200 |
| `--k` | 5 | 5 | 5 |
| structural maximum spend | 500 + 8x80 = 1140 | 600 + 6x100 = 1200 | 460 + 10x70 = 1160 |

Networks: the value net and the policy net are used exactly as in the control. Root noise
is 0, because `mcts` is called without `root_noise`. The B2 solver and by-reference helper
resolution are always on.

**The procedure, for one instance:**

1. **Search the untouched position.** Run the standard PUCT search on the instance as
   given, with budget `b0 = 500`, best-at-budget, `c_puct = 1.5`, backup `min`, park
   repairs on, move dumping on. Call the result `r0`. If `r0` certified a plan, the
   current best becomes `(r0.strict, r0.moves, "subgoal", r0.plan)`. Add `r0.expansions`
   to the running total `spent`.

2. **Enumerate every legal single slide.** Call `move_planner.state.legal_moves(positions,
   walls_right, walls_down, n)`. That function tries all 4 directions for all robots, with
   every other robot acting as a blocker, and drops any slide that cannot leave its cell.
   On the graded exam at 24x24 with 4 robots this yields on average 13.72 legal slides per
   instance, with a minimum of 8 and a maximum of 16.

3. **Score each slid position with `plan_cost_of`.** For a moved position, `plan_cost_of`
   does exactly three things.
   - It builds the moved state as a fresh instance with the same goal and the same target
     robot.
   - It calls `skeleton.astar._initial_plan`, which creates a two-node plan (goal, target
     robot) whose single edge is *fixed* at the exact shortest path length when the engine
     can compute one, and *open* at the relaxed bound otherwise.
   - It calls `spr.search.forced_fixes`, which repeatedly pins the first open edge to its
     exact shortest path whenever the engine returns one.

   It then returns `pl0.cost()`, the abstract cost of the resulting plan.

   **It builds no tree. It calls no network. It certifies nothing.** It is a cheap
   lower-bound-flavoured estimate of how hard the moved position looks. If it raises, the
   candidate is dropped.

4. **Form the depth-1 candidate list.** Each legal slide becomes a candidate with rank key
   `plan_cost_of(moved state) + 1`. The `+1` charges the slide as a real move at ranking
   time. Every legal slide enters the list.

5. **Extend the most promising depth-1 states (depth 2).** Sort the depth-1 states by their
   raw `plan_cost_of` value, take the best `d2_from = 4`, and enumerate every legal slide
   from each of them. Each resulting two-slide position becomes a candidate with rank key
   `plan_cost_of(state) + 2`. Deduplication uses a `seen` set that starts as
   `{original positions}` and grows as depth-2 positions are added. Depth 3 repeats the
   same construction from the best 4 depth-2 states.

6. **Merge and sort.** All candidates, depth 1 and depth 2 together, live in one list and
   are sorted by their rank key. Depth-1 and depth-2 prefixes therefore compete directly
   for the same slots, with the depth-2 ones carrying a `+2` handicap instead of `+1`.

7. **Run a sub-search on the top `top_m = 8` candidates.** For each, in rank order:
   - the sub-budget is `min(sub, max(0, expansions - spent))`, so 80 while budget remains,
   - if the budget is exhausted, stop the loop,
   - **the pruning rule**: `if best is not None and cost0 >= best[0] + 3: continue`. The
     candidate is skipped when its rank key is at least 3 above the current best total. The
     code comments that this is a hopeless margin only, because `cost0` is in abstract plan
     units and `best[0]` is in strict move units. A skipped candidate still consumes one of
     the 8 slots. The budget is not reassigned.
   - otherwise run the same standard PUCT search on the moved state, with the same
     settings, and add its expansions to `spent`.

8. **Compete.** A sub-search result replaces the best only when
   `plen + r2.strict < best[0]`, where `plen` is the number of slides in the prefix. Ties
   go to the incumbent, which means to the pure-subgoal result when one exists. The
   winning move list is `[prefix slides] + r2.moves`, and the winner tag records the exact
   slides, for example `slide:Red:up` or `slide2:Blue:left+Red:down`.

9. **Report.** The row records the total spend, the composed move list, the winner tag, and
   the four search constants.

**How a slide is charged.** A prefix of length `plen` costs exactly `plen` strict moves.
It is added at ranking time (step 4 and step 5) and again at comparison time (step 8). The
replay checker then counts the composed list, so the charge is enforced by an independent
program.

### 3.3 The two-layer verification

**Layer 1, inside the search.** The sub-search runs on the *moved* position. Its
`Certifier` was constructed with that moved state. So `strict_moves` executes the sub-plan
against the position that exists after the slides. This is the same certifier the control
uses, unchanged.

**Layer 2, after the run.** The job scripts call
`python -m eval.replay_validate --compare <payload> --env-dir <boards>`. That checker knows
nothing about slides or prefixes. It reads the composed `[color, direction]` list, replays
it from the **original** instance's start positions under full joint physics, and requires
that each move is a real slide, that the length equals the claimed count, and that the
target robot ends on the goal. Every hybrid job treats a non-zero exit as a job failure.

Every hybrid payload in the repository has a passing replay log:

| payload | passed | failed | skipped (no move dump) |
|---|---|---|---|
| graded depth-2 | 231 | 0 | 1 |
| graded depth-3 | 231 | 0 | 1 |
| graded depth-1, second net pair | 230 | 0 | 2 |
| unseen depth-2 | 188 | 0 | 12 |
| frontier depth-2 | 177 | 0 | 41 |
| g32r4 graded depth-2 | 174 | 0 | 1 |
| g32r4 frontier depth-2 | 251 | 0 | 24 |
| g24r8 graded depth-2 | 159 | 0 | 2 |
| g24r8 frontier depth-2 | 276 | 0 | 13 |

The skipped rows are the unsolved instances, which carry no move list.

### 3.4 Results with controls

The control is the **same two networks**, at the **same 1200-expansion protocol**, in the
**same B2 vocabulary**, under standard PUCT best-at-budget. I confirmed that the hybrid
payload and the control payload name the identical two checkpoint files.

Networks for the three 24x24 exams: `v09_strict_value_s8`, a lab arm whose value net was
trained on realized strict moves.

**Aggregate rows.**

| exam | arm | solved | mean moves | mean regret | % optimal | mean expansions |
|---|---|---|---|---|---|---|
| graded (232) | standard PUCT control | 230 | 9.083 | 1.417 | 59.6 | 499.0 |
| graded (232) | hybrid depth 1 | 231 | 8.688 | 1.009 | 66.2 | 569.7 |
| graded (232) | **hybrid depth 2 (flagship)** | **231** | **8.623** | **0.944** | **68.4** | 581.6 |
| graded (232) | hybrid depth 3 | 231 | 8.541 | 0.861 | 70.6 | 603.8 |
| frontier (218) | standard PUCT control | 174 | 19.052 | — | — | 962.9 |
| frontier (218) | hybrid depth 1 | 176 | 17.983 | — | — | 1084.8 |
| frontier (218) | **hybrid depth 2** | **177** | 17.994 | — | — | 1049.4 |
| unseen (200) | standard PUCT control | 182 | 14.132 | — | — | 689.1 |
| unseen (200) | hybrid depth 1 | 186 | 13.398 | — | — | 807.8 |
| unseen (200) | **hybrid depth 2** | **188** | 13.335 | — | — | 798.5 |

**Paired statistics against the same-nets control.** `a_only` means instances the hybrid
solves and the control does not. `moves wins` counts instances where the arm writes a
strictly shorter solution, among instances both arms solve.

| exam | arm vs control | solved a / b | a_only / b_only | McNemar p | both solved | mean moves a / b | moves wins a / b | sign p |
|---|---|---|---|---|---|---|---|---|
| graded | depth 1 | 231 / 230 | 1 / 0 | 1.00 | 230 | 8.643 / 9.083 | **30 / 0** | **1.9e-9** |
| graded | depth 2 | 231 / 230 | 1 / 0 | 1.00 | 230 | 8.578 / 9.083 | **40 / 0** | **1.8e-12** |
| frontier | depth 1 | 176 / 174 | 4 / 2 | 0.69 | 172 | 17.866 / 18.791 | **44 / 1** | **2.6e-12** |
| frontier | depth 2 | 177 / 174 | 5 / 2 | 0.45 | 172 | 17.826 / 18.791 | **47 / 2** | **4.4e-12** |
| unseen | depth 1 | 186 / 182 | 5 / 1 | 0.22 | 181 | 13.276 / 13.646 | **31 / 1** | **1.5e-8** |
| unseen | depth 2 | 188 / 182 | 7 / 1 | 0.070 | 181 | 13.072 / 13.646 | **39 / 1** | **7.5e-11** |

The pattern is the same on all three exams. Solve rate rises a little and does not reach
significance. Move length falls, and the sign test is decisive.

**Absolute grounding on the unseen exam.** An exhaustive solver established d\* for 137 of
the 200 unseen instances. Mean d\* on those 137 is 8.71.

| system | solves among the 137 with a known d\* | mean regret on those |
|---|---|---|
| supervised backward per-size pair | 115 | 4.844 |
| frozen seed nets, A\* | 131 | 2.931 |
| cold start (label-free from zero) | 109 | 2.890 |
| same nets, standard PUCT | 132 | 1.939 |
| hybrid depth 1 | 135 | 1.674 |
| **hybrid depth 2** | **134** | **1.470** |
| hybrid depth 3 | 134 | 1.425 |
| supervised forward move planner | 101 | 0.119 |

### 3.5 The floor comparison, done properly

The project states its headline as "graded regret 0.944, below the pure-B2 language floor
of 1.171". Those two numbers have different denominators: 231 solved instances against 228
realizable instances. I recomputed the comparison instance by instance.

Restrict to the instances that the exhaustive B2 search **completed without hitting a
cap** and that have a known d\*. That is 225 of the 232 graded instances.

| system | instances | mean regret | shorter than the exhaustive B2 optimum | longer than it |
|---|---|---|---|---|
| exhaustive B2 language optimum (no network) | 225 | **0.978** | — | — |
| same nets, standard PUCT | 225 | 1.191 | 4 | 17 |
| **hybrid depth 2** | 225 | **0.760** | **27** | 7 |

Over the full 228 realizable instances the same comparison reads: language optimum 1.171,
standard PUCT 1.351, hybrid depth 1 0.939, hybrid depth 2 0.873, hybrid depth 3 0.789,
with the hybrid beating the exhaustive optimum on 22, 29 and 32 instances respectively.

This is the cleanest statement of the result. The pure-subgoal search with these networks
sits **above** the language's exhaustive optimum, as it must. The hybrid sits **below** it,
on a paired basis, on 27 individual instances. No amount of training or search inside the
subgoal language could produce that row.

### 3.6 Replication with a different network pair

The searches are deterministic at bench settings, because root noise is off. So repeating
the run with a different random seed would produce the same numbers. The honest replicate
is a different pair of networks. The project used `v14_stack`, a different adopted lab arm.

| exam | hybrid depth 1 with `v14` nets | control with `v14` nets | both solved | mean moves | moves wins | sign p |
|---|---|---|---|---|---|---|
| graded (232) | 230 | 229 | 229 | 8.699 vs 9.166 | **33 / 0** | **2.3e-10** |
| unseen (200) | 186 | 184 | 183 | 13.448 vs 14.333 | **36 / 0** | **2.9e-11** |

The direction, the magnitude and the significance all repeat. The gain belongs to the
search, not to one checkpoint.

### 3.7 Transfer to board sizes and robot counts the hybrid never ran on

These runs use a third network pair, `mix_b2mix_iter3`, the mature mixed-size curriculum
pair. Neither the search nor the networks had ever seen a slide at these sizes. Both arms
in each row use the identical two checkpoint files.

| exam | hybrid depth 2 | standard PUCT control | a_only / b_only | McNemar p | both solved | mean moves | moves wins | sign p |
|---|---|---|---|---|---|---|---|---|
| g32r4 graded (175) | **174**, regret 1.603 | 172, regret 1.866 | 2 / 0 | 0.50 | 172 | 9.267 vs 9.663 | **17 / 0** | **1.5e-5** |
| g32r4 frontier (275) | **251** | 239 | 12 / 0 | **0.00049** | 239 | 20.314 vs 21.481 | **77 / 3** | **1.4e-19** |
| g24r8 graded (161) | 159, regret 1.239 | 159, regret 1.818 | 0 / 0 | 1.00 | 159 | 6.849 vs 7.428 | **23 / 0** | **2.4e-7** |
| g24r8 frontier (289) | 276 | **279** | 2 / 5 | 0.45 | 274 | 14.248 vs 15.296 | **82 / 2** | **3.7e-22** |

Three of the four cells show more or equal solves and clearly shorter solutions. The
32x32 frontier cell is the strongest: 12 more solves with McNemar p = 0.00049, and 77
shorter solutions against 3 longer ones. The 8-robot frontier cell is the exception and
is discussed in section 3.9.

The 8-robot graded cell shows the largest regret cut of any size, from 1.818 to 1.239.
Crowded boards are where a single free slide changes the most.

### 3.8 Depth 3

| exam | depth 3 vs depth 2 | both solved | mean moves | moves wins | sign p |
|---|---|---|---|---|---|
| graded (232) | 231 vs 231 | 231 | 8.541 vs 8.623 | 7 / 0 | 0.016 |
| unseen (200) | 187 vs 188 | 187 | 13.214 vs 13.225 | 6 / 2 | 0.29 |

The graded curve is still descending, from 1.009 at depth 1 to 0.944 at depth 2 to 0.861
at depth 3. The unseen exam has saturated. The project stopped there.

### 3.9 Caveats, all of them

**1. Most of the depth-2 gain is wider screening, not two-slide plans.**
The depth-2 configuration changes three things at once against depth 1: `prefix_depth`
1 to 2, `top_m` 6 to 8, and `sub` 100 to 80, plus `b0` 600 to 500. It is not a clean
depth ablation. I cross-tabulated every instance where depth 2 writes a shorter solution
than depth 1, by the kind of plan that won.

| exam | depth-2 wins over depth 1 | won by a one-slide plan | won by a genuine two-slide plan | depth-2 losses |
|---|---|---|---|---|
| graded | 10 | **10** | **0** | 1 |
| unseen | 11 | 7 | 4 | 0 |
| frontier | 10 | 7 | 3 | 2 |
| total | 31 | 24 (77%) | 7 (23%) | 3 |

On the graded exam, not one of the 10 improvements came from a two-slide prefix. Across
the three exams, 24 of the 31 improvements are still one-slide plans found because 8
candidates were screened instead of 6. Counting winners rather than improvements gives
the same picture: the depth-2 graded run has 38 one-slide winners and 2 two-slide winners,
and the depth-2 unseen run has 41 one-slide winners and 5 two-slide winners.

**2. The move-by-move planner still writes shorter solutions on shared instances.**
The supervised forward planner plans in the primitive-move space. On instances both
systems solve, it wins.

| exam | hybrid arm | both solved | hybrid mean | forward mean | hybrid wins / forward wins | hybrid-only | forward-only |
|---|---|---|---|---|---|---|---|
| graded | standard PUCT | 219 | 8.826 | 7.553 | 4 / 85 | 11 | 1 |
| graded | hybrid depth 2 | 219 | 8.338 | 7.553 | 6 / 64 | 12 | 1 |
| graded | hybrid depth 3 | 219 | 8.251 | 7.553 | 7 / 59 | 12 | 1 |
| unseen | standard PUCT | 99 | 9.081 | 7.778 | 4 / 38 | 83 | 2 |
| unseen | hybrid depth 2 | 100 | 8.680 | 7.790 | 5 / 29 | 88 | 1 |

The hybrid narrows the gap. It does not close it. The hybrid solves far more instances
than the forward planner on fresh boards, 188 against 101 of 200.

**3. The forward baseline is the original network.** The project's own log records that a
re-tuned forward planner gains about 5 graded and 8 frontier puzzles at a different board
configuration, and that no re-tuned 24x24 forward planner exists. Any move gap quoted
against the forward planner should be read against a possibly slightly stronger opponent.

**4. On the 8-robot hard exam the control solves more.** 279 against 276 of 289. The
McNemar p is 0.45, so the difference is not significant, but the direction is real and
it is the only cell where the hybrid solves fewer.

**5. The wall-clock gap is not explained by the algorithm.** The payloads report mean
seconds per instance of 21.1 for the hybrid on the graded exam and 103.4 for the control.
That comparison is invalid. The control ran through `spr.arena`, which splits the exam
into chunks and runs `width = 8` worker processes on one GPU with `OMP_NUM_THREADS = 2`.
The hybrid ran as a single process with `OMP_NUM_THREADS = 16`. So the control's
per-instance seconds include 8-way contention on one GPU. In total job wall time the
control graded run took 3,618 seconds for 232 instances and the hybrid's per-row seconds
sum to 4,898 seconds. Nobody has measured the two systems at equal concurrency. The
5-times speed advantage that the payload columns appear to show is a measurement artifact.

**6. The two systems cannot reach the same number of search steps, and the asymmetry runs
both ways.**

| exam | arm | structural cap | mean expansions | median | rows at the cap |
|---|---|---|---|---|---|
| graded | control | 1200 | 499.0 | 74.5 | 91 of 232 |
| graded | hybrid depth 2 | 1140 | 581.6 | 613.0 | 81 of 232 |
| unseen | control | 1200 | 689.1 | 1200 | 105 of 200 |
| unseen | hybrid depth 2 | 1140 | 798.5 | 1127 | 100 of 200 |
| frontier | control | 1200 | 962.9 | 1200 | 165 of 218 |
| frontier | hybrid depth 2 | 1140 | 1049.4 | 1140 | 165 of 218 |

- The hybrid's cap is 1140, below the control's 1200. It can never spend the full budget.
- The hybrid's search on the untouched position is capped at 500, which is 42% of the
  control's 1200. On the graded exam 91 control rows spent all 1200 on the untouched
  position, so this cap binds often.
- Against that, the hybrid's *average* spend is higher on every exam, by 17% on graded,
  16% on unseen and 9% on frontier. The control terminates early on easy instances,
  because its tree closes. The hybrid always pays for its sub-searches. Its graded median
  is 613 expansions against the control's 74.5.
- The depth-1 arm is the only exactly budget-matched configuration: 600 + 6 x 100 = 1200.
  Its graded moves result is 30 wins to 0, p = 1.9e-9, so the headline survives at an
  exactly equal cap.

**7. The candidate screening is free under the project's accounting and not free in
reality.** `plan_cost_of` calls the exact shortest-path engine repeatedly through
`_initial_plan` and `forced_fixes`. It is invoked once per legal slide, which is 13.72
times per graded instance on average, plus once per depth-2 position, which the top-4
expansion adds. It passes `acct=None`, so these calls appear in no counter, not even in
`free_exact_fix_expands`. The project's budget unit counts only network passes, so this is
consistent with the convention. A reader who thinks of the budget as "work" should know
that the hybrid performs an uncounted amount of exact-engine work that the control does
not.

**8. The pruning rule compares two different units.** The rule is
`cost0 >= best[0] + 3`, where `cost0` is an abstract plan cost plus the prefix length and
`best[0]` is a strict move count. The code comments on this. The margin of 3 is a
constant with no derivation in the file. A pruned candidate also consumes one of the
`top_m` slots, so pruning reduces the number of sub-searches instead of redirecting the
budget to the next candidate.

**9. Deduplication is partial.** The `seen` set starts as the original position and grows
only with depth-2 and depth-3 positions. Depth-1 positions are never added. So a two-slide
prefix that lands on a position also reachable in one slide is not detected as a
duplicate. Nothing measures how often that happens.

**10. Ties favour the incumbent.** The replacement test is a strict inequality,
`plen + r2.strict < best[0]`. A slide plan that ties the pure-subgoal plan never wins. So
the "slide winner" counts are a lower bound on how often slides find an equally good plan.

**11. The floor it beat is measured, not proved.** See section 3.1. The exhaustive probe
capped 7 of the 232 graded instances, and its `best_realizable_moves` is described in its
own docstring as a tight upper bound on the true language optimum rather than a proof.
Section 3.5 restricts the comparison to the 225 uncapped instances to reduce this risk.

**12. The d\* coverage of the unseen exam is partial.** 137 of the 200 unseen instances
carry an exact optimum. The flagship solves 134 of those 137. So the "+1.47 against
perfect play" figure is an average over 134 instances, not 137, and not 200.

**13. The hybrid is inference only.** Generation and training were never changed. Two
attempts to teach the networks about slides both failed. `v15_slide_training` was flat at
two seeds. `v16_ranked_slide_training` showed a frontier signal at one seed, 38 wins to
17, and flatly failed to repeat it at the second seed, 28 wins to 29.

---

## 4. Section D — what is still open

**The loop still cannot shorten solutions.** Across five B2 rounds and three mixed-size
rounds, no paired move test reached significance in the loop's favour. The only two things
that moved move quality were the network rebuild, which happened before self-play, and the
action-space change, which happens at inference and is not in the loop.

**Nothing in the training pipeline knows about slides.** The hybrid is a portfolio of
independent searches with no shared tree, no slide prior and no slide value head. Both
attempts to build one failed. The project's own explanation, that the networks already
generalize to post-slide positions and had no gap to close, is plausible and untested.

**The depth curve is nearly flat, and nobody knows why.** Graded regret improves 1.009 to
0.944 to 0.861 across depths 1, 2 and 3. Unseen solves stop at about 188. No deeper probe
was run.

**Most of the depth-2 improvement is screening width, and screening width was never swept.**
The runs change `top_m`, `sub` and `b0` together with `prefix_depth`. A pure `top_m` sweep
at fixed depth 1 would separate the two effects. It does not exist.

**The two systems were never timed fairly.** See caveat 5.

**One exam disagrees on solves.** The 8-robot frontier exam. 276 against 279. Not
significant, and unexplained.

**The B2 fidelity gauge is uninterpreted.** The loop's B2 labels sit at 0.45-0.57 argmin
agreement with the exact engine, well inside the band the project's own calibration calls
collapse, yet the loop's exam rows improved. Either the calibration does not transfer to
B2, or the exact side of the comparison is itself weak because of its caps, or the loop's
gains do not depend on label fidelity. The files do not settle this.

**The cold-start control is one round deep.** It shows that random weights plus one round
of certified self-play data reach the supervised baseline on fresh boards, 132 against
134 of 200. It does not show what a chain of such rounds does, and its generation still
used trained nets.

**No pinned exam exists above 32x32.** The transfer claim covers 24x24 with 4 and 8
robots and 32x32 with 4 robots. Larger sizes were only audited at the level of individual
decisions.

**The move-by-move planner is still ahead on solution length**, 64 wins to 6 on the graded
exam against the flagship, and the forward baseline has not been re-tuned at 24x24.

---

## 5. Summary in one paragraph

The self-play loop takes fresh boards no exam uses, draws random puzzles, searches them
with the current networks, and keeps only the decisions that lie on a path to a plan the
physics engine actually executed. Its answers are measurements. Over five rounds it raised
the hard-exam solve rate from 133 to 153 of 218 and cut the cheap planner's search cost
from 53 expansions to 26, but it never shortened a solution, because the subgoal plan
language has an exhaustively measured floor of about 1.0-1.2 extra moves that no ranking
can pass. The breakthrough is a search-time change that steps outside that language: before
planning, try letting one or two robots slide, price each resulting position with a cheap
network-free plan estimate, spend a slice of the same expansion budget planning from the
best few, and charge each slide as a real move. It works because the floor is a property of
what the language can *express*, not of how well the networks rank it, so a plan that
begins with an ordinary move is simply not in the set the floor bounds. On the standard
exam it reaches 0.94 extra moves against the language's own exhaustive 0.98, beating the
exhaustive optimum on 27 individual puzzles, and it repeats with two other network pairs
and on two board types it never practised on.

---

## 6. Reader's map of the claims

| claim | verified against | verdict |
|---|---|---|
| the loop bought solve rate | `results/selfplay/g24r4_b2_iter*/bench_g24r4_frontier_astar.json` | yes, 133 to 153 of 218, McNemar p = 0.00082 |
| the loop bought search efficiency | same files, `mean_expansions` | yes, 52.8 to 25.9-30.3 expansions at equal or better solves |
| the loop did not shorten solutions | the paired gates in section 2.5 | yes, every move sign test is not significant |
| the big move-quality gain came from the rebuild | `results/m1/mixed_value_warm_s21_g24r4.json` vs `scaling/results/g24r4/comparison.json` | yes, regret 4.220 to 2.433 before any self-play |
| the richer vocabulary hurt the supervised planner | `comparison.json` vs `comparison_b2.json` | yes, 205 to 199 solves, regret 4.220 to 4.889 |
| the hybrid scores below the language floor | section 3.5, paired over 225 instances | yes, 0.760 against 0.978, 27 instances strictly better |
| the hybrid transfers | `results/variants/v07_transfer/` | yes on 3 of 4 cells, with one solve-count exception |

---

## 7. Command reference

Every number below can be reproduced from the repository root with:

```
ml Python/3.11.5-GCCcore-13.2.0
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
export PYTHONPATH=supervised_valuenet:self_play_robots
```

Aggregate rows live at `payload["systems"][<name>]["aggregate"]`. Per-instance rows live
at `payload["systems"][<name>]["rows"]`, one row per instance, in instance-file order.
Paired statistics can be recomputed with `python -m spr.gate compare --a A.json --b B.json`.

---

## 8. Files this part read

Code: `spr/selfplay.py`, `spr/search.py`, `spr/train.py`, `spr/buffer.py`, `spr/bench.py`,
`spr/arena.py`, `spr/gate.py`, `spr/gauge.py`, `spr/ceiling.py`, `spr/nets.py`,
`variants/v07_hybrid_actions.py`, `variants/v08_cold_start.py`, `variants/v09_strict_value.py`,
`variants/v04_deep_emit.py`, `variants/v14_stack.py`, `variants/__init__.py`,
`supervised_valuenet/eval/realize.py`, `supervised_valuenet/eval/replay_validate.py`,
`supervised_valuenet/nn/generate.py`, `supervised_valuenet/nn_labeler/descent.py`,
`supervised_valuenet/nn_labeler/model.py`, `supervised_valuenet/nn_labeler/audit_descent.py`,
`supervised_valuenet/skeleton/astar.py`, `supervised_valuenet/move_planner/state.py`,
`supervised_valuenet/scaling/configs.py`.

Job scripts: `jobs/selfplay_iter.slurm`, `jobs/selfplay_mix_iter.slurm`,
`jobs/variant_iter.slurm`, `jobs/v07_bench.slurm`, `jobs/v07_wave4.slurm`,
`jobs/v07_flagship.slurm`, `jobs/v07_d3.slurm`, `jobs/v07_transfer.slurm`.

Results: `results/selfplay/`, `results/variants/`, `results/m0/`, `results/m1/`,
`results/ceiling/`, `results/transfer/`, `supervised_valuenet/scaling/results/g24r4/`.

Project prose, read for cross-checking only: `self_play_robots/DESIGN.md`,
`self_play_robots/FINDINGS.md`, `self_play_robots/variants/FINDINGS.md`.

---

## 9. Appendix — every number, with its provenance

All paths are relative to `/scratch/project/open-37-42/petrhyner/MCTS_evolution`.
"agg" means `payload["systems"][<name>]["aggregate"]`. "rows" means
`payload["systems"][<name>]["rows"]`, one entry per instance in instance-file order.
"gate" means the flat JSON object written by `spr.gate compare`.

### 9.1 Loop parameters and yields

| value | number | file | how read |
|---|---|---|---|
| B2 loop board ids, round 1 | 8000-8059 | `self_play_robots/results/selfplay/g24r4_b2_iter1/generation.manifest.json` | key `board_ids` |
| B2 loop search settings | expansions 300, stop_after 80, k 5, c_puct 1.5, backup min, root_noise 0.25, complete_siblings true, root_all true, emit path, vocab b2, timeout 600 | same file | key `search` |
| B2 loop workers | 6 | same file | key `workers` |
| B2 round 1 instances / solved / records | 681 / 598 / 5,822 | same file | keys `instances`, `solved`, `records` |
| B2 round 2 | 655 / 588 / 5,694 | `.../g24r4_b2_iter2/generation.manifest.json` | same keys |
| B2 round 3 | 660 / 582 / 6,301 | `.../g24r4_b2_iter3/generation.manifest.json` | same keys |
| B2 round 4 | 675 / 600 / 6,069 | `.../g24r4_b2_iter4/generation.manifest.json` | same keys |
| B2 round 5 | 676 / 609 / 6,661 | `.../g24r4_b2_iter5/generation.manifest.json` | same keys |
| B2 round mean strict moves 13.72 / 13.78 / 14.10 / 14.15 / 13.83 | as listed | the five manifests | key `mean_strict` |
| B2 round generation seconds 4301 / 4064 / 4197 / 4136 / 3868 | as listed | the five manifests | key `seconds` |
| base loop round 1 | 120 boards, ids 5000-5119, expansions 600, stop 150, 1,774 instances, 1,268 solved, 12,244 records | `.../g24r4_iter1/generation.manifest.json` | keys `board_ids`, `search`, `instances`, `solved`, `records` |
| base loop round 2 | ids 5120-5239, `min_expansions` 3, 2,869 instances, 2,041 solved, 4,268 records, 1,525 dropped as trivial | `.../g24r4_iter2/generation.manifest.json` | same keys plus `status_counts` |
| mixed loop board ids | 9000-9029, 9030-9059, 9060-9089 | `.../mix_b2mix_iter{1,2,3}/generation_g24r4.manifest.json` | key `board_ids` |
| mixed loop search settings | expansions 300, stop 80, vocab b2, timeout 900 | `.../mix_b2mix_iter1/generation_g24r4.manifest.json` | key `search` |
| mixed loop records per round per configuration | 3,283 / 5,770 / 2,862 (round 1), 4,025 / 7,455 / 3,173 (round 2), 3,791 / 6,699 / 2,970 (round 3) | the nine `generation_*.manifest.json` files | key `records` |
| training recipe of a round | epochs 6 and 6, lr 1e-4, policy `--batch-size 8 --grad-clip 1.0 --byref`, value `--batch-size 4 --max-per-group 8`, seed 21+K, window 3 | `self_play_robots/jobs/selfplay_iter.slurm` | shell variables `EP_P`, `EP_V`, `LR`, function `train_one`, `WINDOW` |
| validation fraction | 0.15 of board ids, last ids per round | `self_play_robots/spr/buffer.py` | argument `--val-frac` default and the `cut` computation |
| optimiser | AdamW, weight decay 1e-4, no scheduler | `self_play_robots/spr/nets.py` line 248, `supervised_valuenet/nn_labeler/model.py` line 252 | `configure_optimizers` |
| policy target | softmax of `-cost_to_go` at temperature 1.0 | `self_play_robots/spr/nets.py` | `_soft_targets` |
| value loss | ranking loss plus HL-Gauss cross-entropy over 96 bins | `supervised_valuenet/nn_labeler/model.py` | `training_step` |
| checkpoint selection | minimum `val_regret` | `self_play_robots/spr/train.py` | `pl.callbacks.ModelCheckpoint(monitor=monitor, mode="min")` |
| collapse guard | `val_group_spread` below 0.05 for 3 epochs | `self_play_robots/spr/train.py` | class `CollapseStop` |
| board-id guard | rejects any id at or below 1199 | `self_play_robots/spr/selfplay.py` | `bad = [i for i in ids if i <= 1199]` then `raise SystemExit` |
| exam board ids | 900-1049 for graded and frontier, 20000-20049 for unseen | `scaling/data/g24r4/bench.solved.jsonl`, `bench.unsolved.jsonl`, `results/variants/exam/g24r4_unseen.jsonl` | min and max of `env_id` over all lines |
| standard config splits | train 0-699, val 700-899, test and bench 900-1049, pool 0-1199 | `supervised_valuenet/scaling/configs.py` | `_STD_RANGES` and `_std` |
| exam sizes | graded 232, frontier 218, unseen 200 | the three instance files | line count |

### 9.2 Fidelity gauge

| value | number | file | how read |
|---|---|---|---|
| base round 1 argmin agreement / mean gap / negative gaps | 0.915 / 0.987 / 0 | `results/selfplay/g24r4_iter1/gauge.json` | `audit_summary.argmin_agreement`, `.gap_mean`, `.negative_gaps` |
| base round 2 | 0.850 / 1.292 / 0 | `results/selfplay/g24r4_iter2/gauge.json` | same keys |
| B2 rounds 1-5 argmin agreement | 0.513 / 0.500 / 0.568 / 0.451 / 0.460 | `results/selfplay/g24r4_b2_iter{1..5}/gauge.json` | same keys |
| B2 rounds 1-5 mean gap | 3.926 / 4.121 / 3.980 / 4.729 / 4.763 | same files | `audit_summary.gap_mean` |
| B2 rounds 1-5 negative gaps | 0 in all five | same files | `audit_summary.negative_gaps` |
| B2 rounds 1-5 matched decisions | 197 / 74 / 74 / 195 / 189 | same files | `audit_summary.n_matched_groups` |
| gauge calibration bands | about 0.91 equivalent, about 0.89 keeps solves, about 0.82 collapse | `self_play_robots/spr/gauge.py` | module docstring |
| gap definition | descent ctg minus exact ctg, negatives are a bug signal | `supervised_valuenet/nn_labeler/audit_descent.py` lines 14-15 | module docstring |

### 9.3 The language ceiling

| value | number | file | how read |
|---|---|---|---|
| g24r4 graded, base vocabulary: realizable, mean d\*, mean best moves, mean regret, % optimal | 216 of 232, 7.593, 9.315, **1.722**, 57.4 | `results/ceiling/g24r4_base.json` | `summary.n_realizable`, `.mean_d_star`, `.mean_best_moves`, `.mean_gap_best`, `.pct_best_optimal` |
| g24r4 graded, B2 vocabulary | 228 of 232, 7.640, 8.811, **1.171**, 62.7 | `results/ceiling/g24r4_b2.json` | same keys |
| B2 probe capped instances | 7, of which 3 still realizable | same file | `summary.capped`, and counting rows with `category == "REALIZABLE_EXISTS"` and `capped == true` |
| B2 probe uncapped realizable rows | 225 | same file | counting rows with `category == "REALIZABLE_EXISTS"` and `capped == false` |
| B2 probe bound-proven rows | 158 of 228 | same file | `summary.best_bounded` |
| g24r4 frontier, base ceiling | 103 of 218 | `results/ceiling/g24r4_frontier_base.json` | `summary.n_realizable` |
| g24r4 frontier, B2 ceiling | 114 proven, 104 inconclusive | `results/ceiling/g24r4_frontier_b2.json` | `summary.categories` |
| `best_realizable_moves` is an upper bound, not a proof | statement | `self_play_robots/spr/ceiling.py` | module docstring |

### 9.4 The decomposition table of section 2.1

| row | file | how read |
|---|---|---|
| 1, supervised per-size backward, base | `supervised_valuenet/scaling/results/g24r4/comparison.json` | agg of the system whose name starts with `backward` |
| 1b, same pair re-run on this machine (205, 11.795, 4.205) | `self_play_robots/results/m0/g24r4_exact_prefix.json` | agg |
| 2, supervised per-size backward, B2 | `supervised_valuenet/scaling/results/g24r4/comparison_b2.json` | agg of the `backward` system |
| 3, size-free rebuild, base | `self_play_robots/results/m1/mixed_value_warm_s21_g24r4.json` | agg |
| 4, same nets under B2, A\* anytime | `self_play_robots/results/selfplay/g24r4_b2_iter0/m1mixed_b2_g24r4_bench_solved_astar.json` | agg |
| 5, same nets under B2, PUCT | `.../m1mixed_b2_g24r4_bench_solved_mcts.json` | agg |
| 6, round 5 A\* | `.../g24r4_b2_iter5/bench_g24r4_astar.json` | agg |
| 7, round 5 PUCT | `.../g24r4_b2_iter5/bench_g24r4_mcts.json` | agg |
| 8, mixed round 3 A\* | `.../mix_b2mix_iter3/bench_g24r4_astar.json` | agg |
| 9, exhaustive B2 optimum | `results/ceiling/g24r4_b2.json` | `summary` |
| 10, supervised forward move planner | `supervised_valuenet/scaling/results/g24r4/comparison.json` | agg of the system whose name starts with `forward` |
| rounds 3 to 8 use the same nets lineage | `results/selfplay/g24r4_b2_iter0/*_astar.json` `protocol.checkpoints` names `runs/spr/m1/mixed_policy_s21` and `runs/spr/m1/mixed_value_warm_s21` | direct string comparison with `results/m1/mixed_value_warm_s21.nets.txt` |

Other M1 rows, read from `self_play_robots/results/m1/` agg blocks:
`g16_value_warm_s21_g24r4` 213 of 232 at regret 2.526,
`g24_value_warm_s21_g24r4` 216 at 2.310,
`mixed_value_cold_s21_g24r4` 215 at 2.233,
`mixed_value_warm_s37_g24r4` 215 at 2.312.
At 16x16 on 450 instances the pairs read: `g16_value_warm_s21` 397, `g24_value_warm_s21` 402, `mixed_value_cold_s21` 401, `mixed_value_warm_s21` 401, `mixed_value_warm_s37` 402.

### 9.5 Paired result: supervised B2 against supervised base

| value | number | how computed |
|---|---|---|
| both solved | 186 | zipping the `backward` rows of `comparison_b2.json` and `comparison.json` |
| mean moves, B2 vs base, on those 186 | 12.247 vs 10.661 | mean of `realized_strict` |
| B2 shorter / base shorter | 15 / 41 | strict comparison of `realized_strict` |
| B2 only / base only | 13 / 19 | one solved and the other not |

### 9.6 The B2 loop series

Aggregates from `results/selfplay/g24r4_b2_iter{1..5}/bench_g24r4_{astar,mcts,frontier_astar,frontier_mcts}.json`,
key `aggregate`. Round 0 aggregates from
`results/selfplay/g24r4_b2_iter0/m1mixed_b2_g24r4_bench_{solved,unsolved}_{astar,mcts}.json`.

Paired tests of section 2.5 were recomputed for this document. Method: zip the `rows`
arrays of the two payloads, count discordant `solved` pairs, apply the exact two-sided
McNemar formula from `self_play_robots/spr/gate.py::mcnemar_exact`, and apply the same
formula to the counts of strictly shorter solutions on the both-solved subset. Results:

| comparison | a_only / b_only | McNemar p | moves wins a / b | sign p |
|---|---|---|---|---|
| frontier A\* round 5 vs round 0 | 27 / 7 | 0.000821 | 22 / 27 | 0.568 |
| frontier A\* round 3 vs round 0 | 28 / 3 | 4.65e-6 | 27 / 24 | 0.78 |
| graded A\* round 5 vs round 0 | 6 / 2 | 0.289 | 16 / 17 | 1.0 |
| graded PUCT round 5 vs round 0 | 2 / 2 | 1.0 | 13 / 18 | 0.473 |
| frontier PUCT round 5 vs round 0 | 12 / 9 | 0.664 | 23 / 25 | 0.885 |

Round-to-round gates already on disk, read from
`results/selfplay/g24r4_b2_iter{1..5}/gate_vs_prev.json`, all show `mcnemar_p` at or above
0.25 and `sign_p_moves` at or above 0.30.

### 9.7 The mixed-size curriculum

Aggregates from `results/selfplay/mix_b2mix_iter{1,2,3}/bench_{g24r4,g32r4,g24r8}_{astar,frontier_astar}.json`,
key `aggregate`. Round-3-against-round-2 gates from
`results/selfplay/mix_b2mix_iter3/gate_*_vs_prev.json`: `mcnemar_p` values 0.25, 1.0, 1.0,
0.664, 0.5 and 1.0, `sign_p_moves` values 0.557, 0.795, 1.0, 0.723, 1.0 and 0.826.

### 9.8 The cold-start control

| value | number | file | how read |
|---|---|---|---|
| generation: instances / solved / records | 329 / 306 / 3,787 | `results/variants/v08_cold_start/generation.manifest.json` | keys `instances`, `solved`, `records` |
| generation nets | `mix_b2mix_iter2` policy and value | same file | keys `policy`, `value` |
| board ids | 42400-42429 | same file | key `board_ids` |
| graded / frontier / unseen solves | 200 of 232 / 96 of 218 / 132 of 200 | `results/variants/v08_cold_start/bench_{graded,frontier,unseen}_astar.json` | agg `solved` |
| control solves | 228 / 160 / 172 | `results/variants/v00_control/bench_*_astar.json` | agg `solved` |
| paired McNemar p | 5.77e-8 / 6.31e-15 / 1.96e-11 | `results/variants/v08_cold_start/gate_*_vs_control.json` | key `mcnemar_p` |
| supervised per-size pair on the unseen exam | 134 of 200 | `results/variants/baselines/supervised_persize_unseen.json` | agg `solved` |
| what the arm changes | `spr.train` without `--init`, value warmup 4, lr 3e-4 | `variants/v08_cold_start.py` and `jobs/variant_iter.slurm` | `env={"COLD": 1, "EP": 20}` and the `COLD` branch of `train_one` |

### 9.9 The hybrid, constants

| value | number | file | how read |
|---|---|---|---|
| module defaults | B0 600, TOP_M 6, SUB 100 | `variants/v07_hybrid_actions.py` line 39 | `B0, TOP_M, SUB = 600, 6, 100` |
| flagship frontier leg | `--prefix-depth 2 --b0 500 --top-m 8 --sub 80` | `jobs/v07_flagship.slurm` | the driver invocation |
| depth-2 graded and unseen legs | `--prefix-depth 2 --b0 500 --top-m 8 --sub 80` | `jobs/v07_wave4.slurm` leg A | the `hyb` calls |
| depth-1 legs | defaults, so 600 / 6 / 100 | `jobs/v07_bench.slurm` | `B0`, `TOPM`, `SUB` shell defaults |
| depth-3 probe | `--prefix-depth 3 --b0 460 --top-m 10 --sub 70 --d2-from 4` | `jobs/v07_d3.slurm` | the driver invocation |
| transfer legs | `--prefix-depth 2 --b0 500 --top-m 8 --sub 80` | `jobs/v07_transfer.slurm` | the driver invocation |
| constants confirmed per row | depth-2 rows carry `{"b0":500,"top_m":8,"sub":80,"prefix_depth":2}`, depth-1 rows carry `{"b0":600,"top_m":6,"sub":100}` and no `prefix_depth` key | every hybrid payload | `rows[i]["search"]` |
| root noise off | 0 | `variants/v07_hybrid_actions.py` | `mcts(...)` is called without `root_noise`, and `spr/search.py::mcts` defaults it to 0.0 |
| legal single slides per graded instance | mean 13.72, min 8, max 16 | computed this session | `move_planner.state.legal_moves` applied to every line of `scaling/data/g24r4/bench.solved.jsonl` with the board's own `wall_sets` |
| `plan_cost_of` uses no network | statement | `variants/v07_hybrid_actions.py` lines 138-143 | the function body calls only `_initial_plan`, `forced_fixes` and `pl0.cost()` |
| `forced_fixes` pins exact shortest paths | statement | `spr/search.py::forced_fixes` and `skeleton/astar.py::AStar._expand` step 1 | `env.compute_exact_shortest_path_length` then `status="fixed"` |
| pruning rule | `cost0 >= best[0] + 3` | `variants/v07_hybrid_actions.py` line 177 | source |
| replacement rule | `plen + r2.strict < best[0]` | same file line 183 | source |
| deduplication | `seen = {positions}`, grown only at depth 2 and above | same file lines 155-167 | source |

### 9.10 The hybrid, results

All aggregates from `payload["systems"][<name>]["aggregate"]`. All paired numbers from
`spr.gate compare` output files, except where marked "recomputed".

| row | file |
|---|---|
| graded control 230, 9.083, regret 1.417, 59.6% | `results/variants/v07_hybrid_actions/bench_graded_stdmcts.json` |
| graded depth 1: 231, 8.688, 1.009, 66.2% | `.../bench_graded_hybrid.json` |
| graded depth 2: 231, 8.623, 0.944, 68.4% | `.../bench_graded_hybrid_d2.json` |
| graded depth 3: 231, 8.541, 0.861, 70.6% | `.../bench_graded_hybrid_d3.json` |
| frontier control 174, 19.052 | `.../bench_frontier_stdmcts.json` |
| frontier depth 1: 176, 17.983 | `.../bench_frontier_hybrid.json` |
| frontier depth 2: 177, 17.994 | `.../bench_frontier_hybrid_d2.json` |
| unseen control 182, 14.132 | `.../bench_unseen_stdmcts.json` |
| unseen depth 1: 186, 13.398 | `.../bench_unseen_hybrid.json` |
| unseen depth 2: 188, 13.335 | `.../bench_unseen_hybrid_d2.json` |
| unseen depth 3: 187, 13.214 | `.../bench_unseen_hybrid_d3.json` |
| graded depth 1 with `v14` nets: 230, 8.839, 1.170 | `.../bench_graded_hybrid_v14nets.json` |
| graded control with `v14` nets: 229, 9.166, 1.511 | `.../bench_graded_stdmcts_v14nets.json` |
| unseen depth 1 with `v14` nets: 186, 13.462 | `.../bench_unseen_hybrid_v14nets.json` |
| unseen control with `v14` nets: 184, 14.429 | `.../bench_unseen_stdmcts_v14nets.json` |

Gate files, all in `results/variants/v07_hybrid_actions/`:

| gate | a_only / b_only | McNemar p | moves wins a / b | sign p |
|---|---|---|---|---|
| `gate_graded_hybrid_vs_std.json` | 1 / 0 | 1.0 | 30 / 0 | 1.86e-9 |
| `gate_graded_d2_vs_std.json` | 1 / 0 | 1.0 | 40 / 0 | 1.82e-12 |
| `gate_graded_d2_vs_d1.json` | 0 / 0 | 1.0 | 10 / 1 | 0.0117 |
| `gate_graded_d3_vs_d2.json` | 0 / 0 | 1.0 | 7 / 0 | 0.0156 |
| `gate_graded_v14_hyb_vs_std.json` | 1 / 0 | 1.0 | 33 / 0 | 2.33e-10 |
| `gate_unseen_hybrid_vs_std.json` | 5 / 1 | 0.219 | 31 / 1 | 1.54e-8 |
| `gate_unseen_d2_vs_std.json` | 7 / 1 | 0.0703 | 39 / 1 | 7.46e-11 |
| `gate_unseen_d2_vs_d1.json` | 3 / 1 | 0.625 | 11 / 0 | 0.000977 |
| `gate_unseen_d3_vs_d2.json` | 0 / 1 | 1.0 | 6 / 2 | 0.289 |
| `gate_unseen_v14_hyb_vs_std.json` | 3 / 1 | 0.625 | 36 / 0 | 2.91e-11 |
| `gate_frontier_hyb_vs_std.json` | 4 / 2 | 0.688 | 44 / 1 | 2.61e-12 |
| `gate_frontier_d2_vs_stdmcts.json` | 5 / 2 | 0.453 | 47 / 2 | 4.36e-12 |
| `gate_frontier_d2_vs_hybrid.json` | 1 / 0 | 1.0 | 10 / 2 | 0.0386 |

Transfer gates, all in `results/variants/v07_transfer/`:

| gate | solved a / b | a_only / b_only | McNemar p | moves wins a / b | sign p |
|---|---|---|---|---|---|
| `gate_g32r4_graded_hyb_vs_std.json` | 174 / 172 | 2 / 0 | 0.5 | 17 / 0 | 1.53e-5 |
| `gate_g32r4_frontier_hyb_vs_std.json` | 251 / 239 | 12 / 0 | 0.000488 | 77 / 3 | 1.41e-19 |
| `gate_g24r8_graded_hyb_vs_std.json` | 159 / 159 | 0 / 0 | 1.0 | 23 / 0 | 2.38e-7 |
| `gate_g24r8_frontier_hyb_vs_std.json` | 276 / 279 | 2 / 5 | 0.453 | 82 / 2 | 3.69e-22 |

Transfer regret values, from the same payloads' aggregates: g32r4 graded hybrid 1.603
against control 1.866, g24r8 graded hybrid 1.239 against control 1.818.

Both arms of each transfer cell name the identical checkpoints
`runs/spr/selfplay/mix_b2mix_iter3/policy/.../epoch=2-step=2361.ckpt` and
`.../value/.../epoch=0-step=1573.ckpt`, matching `results/selfplay/mix_b2mix_iter3/nets.txt`.

Both arms of each 24x24 cell name
`runs/spr/variants/v09_strict_value_s8/policy/.../epoch=0-step=41.ckpt` and
`.../value/.../epoch=1-step=162.ckpt`, matching
`results/variants/v09_strict_value_s8/nets.txt`.

### 9.11 The floor comparison of section 3.5 (recomputed)

Method: align `results/ceiling/g24r4_b2.json` rows with the bench rows by index, because
both are emitted in `bench.solved.jsonl` order and both have 232 entries. Then compare
`row["realized_strict"]` against `ceiling_row["best_realizable_moves"]` and
`ceiling_row["d_star"]`.

| subset | system | n | mean regret | strictly shorter than the exhaustive optimum | strictly longer |
|---|---|---|---|---|---|
| 228 realizable rows with known d\* | exhaustive B2 optimum | 228 | 1.171 | — | — |
| | standard PUCT | 228 | 1.351 | 5 | 18 |
| | hybrid depth 1 | 228 | 0.939 | 22 | 8 |
| | hybrid depth 2 | 228 | 0.873 | 29 | 8 |
| | hybrid depth 3 | 228 | 0.789 | 32 | 5 |
| 225 uncapped realizable rows | exhaustive B2 optimum | 225 | 0.978 | — | — |
| | standard PUCT | 225 | 1.191 | 4 | 17 |
| | hybrid depth 2 | 225 | 0.760 | 27 | 7 |

### 9.12 Winner-type counts (recomputed)

Method: for each row, read `row["search"]["winner"]`. The tag is `subgoal`, or
`slide:<color>:<dir>`, or `slide2:...`, or `slide3:...`, or absent when unsolved.

| payload | subgoal | one slide | two slides | three slides | unsolved |
|---|---|---|---|---|---|
| graded depth 1 | 201 | 30 | 0 | 0 | 1 |
| graded depth 2 | 191 | 38 | 2 | 0 | 1 |
| graded depth 3 | 186 | 41 | 2 | 2 | 1 |
| unseen depth 1 | 150 | 36 | 0 | 0 | 14 |
| unseen depth 2 | 142 | 41 | 5 | 0 | 12 |
| unseen depth 3 | 139 | 44 | 3 | 1 | 13 |
| frontier depth 1 | 129 | 47 | 0 | 0 | 42 |
| frontier depth 2 | 126 | 48 | 3 | 0 | 41 |
| g32r4 graded depth 2 | 155 | 18 | 1 | 0 | 1 |
| g32r4 frontier depth 2 | 164 | 83 | 4 | 0 | 24 |
| g24r8 graded depth 2 | 136 | 22 | 1 | 0 | 2 |
| g24r8 frontier depth 2 | 192 | 82 | 2 | 0 | 13 |

Cross-tabulation of depth-2 improvements over depth 1 by the depth-2 winner type:

| exam | improvements | one slide | two slides | regressions |
|---|---|---|---|---|
| graded | 10 | 10 | 0 | 1 (a one-slide plan) |
| unseen | 11 | 7 | 4 | 0 |
| frontier | 10 | 7 | 3 | 2 (subgoal plans) |

Depth-3 improvements over depth 2: graded 7, of which 6 one-slide and 1 three-slide, with
0 regressions. Unseen 6, all one-slide, with 2 regressions.

### 9.13 Budget accounting (recomputed)

Method: read `row["expansions"]` from each payload.

| payload | structural cap | mean | median | max | rows at the cap |
|---|---|---|---|---|---|
| graded depth 1 | 1200 | 569.7 | 526.0 | 1200 | 74 of 232 |
| graded depth 2 | 1140 | 581.6 | 613.0 | 1140 | 81 of 232 |
| graded depth 3 | 1160 | 603.8 | 689.0 | 1160 | 79 of 232 |
| graded control | 1200 | 499.0 | 74.5 | 1200 | 91 of 232 |
| unseen depth 2 | 1140 | 798.5 | 1127.0 | 1140 | 100 of 200 |
| unseen control | 1200 | 689.1 | 1200 | 1200 | 105 of 200 |
| frontier depth 2 | 1140 | 1049.4 | 1140 | 1140 | 165 of 218 |
| frontier control | 1200 | 962.9 | 1200 | 1200 | 165 of 218 |
| g32r4 frontier depth 2 | 1140 | 995.4 | 1140 | 1140 | 193 of 275 |
| g32r4 frontier control | 1200 | 877.4 | 1200 | 1200 | 186 of 275 |

No row of any payload exceeds its structural cap.

Concurrency: the control payloads carry `payload["spr"] = {"width": 8, "omp_threads": 2,
"device": "cuda", "wall_seconds": 3618.3}` for the graded leg and 3937.3 for the unseen
leg. The hybrid payloads have no `spr` block, because they come from the standalone
driver, which the job scripts run as one process with `OMP_NUM_THREADS=16`. Summing
`row["seconds"]` gives 4,898 seconds for graded depth 2 and 6,915 seconds for unseen
depth 2.

### 9.14 The unseen exam, exact optima

| value | number | file | how read |
|---|---|---|---|
| instances with a known d\* | 137 of 200 | `results/variants/exam/g24r4_unseen.dstar.jsonl` | counting lines whose `d_star` is not null |
| mean regret, hybrid depth 1 | 1.674 over 135 | that sidecar plus `bench_unseen_hybrid.json` | zip rows with sidecar lines, average `realized_strict - d_star` over solved rows with a known d\* |
| hybrid depth 2 | 1.470 over 134 | plus `bench_unseen_hybrid_d2.json` | same |
| hybrid depth 3 | 1.425 over 134 | plus `bench_unseen_hybrid_d3.json` | same |
| standard PUCT | 1.939 over 132 | plus `bench_unseen_stdmcts.json` | same |
| depth-1 with `v14` nets | 1.612 over 134 | plus `bench_unseen_hybrid_v14nets.json` | same |
| control with `v14` nets | 2.145 over 131 | plus `bench_unseen_stdmcts_v14nets.json` | same |
| frozen seed nets | 2.931 over 131 | plus `results/variants/baselines/seed_nets_unseen.json` | same |
| supervised per-size backward | 4.844 over 115 | plus `.../supervised_persize_unseen.json` | same |
| supervised forward | 0.119 over 101 | plus `.../forward_movenet_unseen.json` | same |
| cold start | 2.890 over 109 | plus `results/variants/v08_cold_start/bench_unseen_astar.json` | same |
| unseen exam solve counts | forward 101, supervised backward 134, seed nets 175 | the three baseline files | agg `solved` |

### 9.15 Hybrid against the forward planner (recomputed)

Method: zip the hybrid rows and the forward rows of the same exam, in file order. Move
count is `realized_strict` for backward rows and `moves` for forward rows.

| exam | hybrid arm | both solved | hybrid mean | forward mean | hybrid wins / forward wins | two-sided sign p | hybrid only | forward only |
|---|---|---|---|---|---|---|---|---|
| graded | standard PUCT | 219 | 8.826 | 7.553 | 4 / 85 | 8.3e-21 | 11 | 1 |
| graded | depth 2 | 219 | 8.338 | 7.553 | 6 / 64 | 2.4e-13 | 12 | 1 |
| graded | depth 3 | 219 | 8.251 | 7.553 | 7 / 59 | 2.4e-11 | 12 | 1 |
| unseen | standard PUCT | 99 | 9.081 | 7.778 | 4 / 38 | 5.7e-8 | 83 | 2 |
| unseen | depth 1 | 101 | 8.941 | 7.782 | 5 / 31 | 1.3e-5 | 85 | 0 |
| unseen | depth 2 | 100 | 8.680 | 7.790 | 5 / 29 | 3.9e-5 | 88 | 1 |
| unseen | depth 3 | 100 | 8.630 | 7.790 | 5 / 27 | 1.1e-4 | 87 | 1 |

Hybrid depth 2 against the supervised backward planner on the graded exam: 205 both
solved, means 8.517 against 11.810, 90 wins to 0, and the hybrid solves 26 the supervised
planner does not.

### 9.16 Replay validation

Counts from the tail line of each `*.replay.log` file in
`results/variants/v07_hybrid_actions/` and `results/variants/v07_transfer/`, which reads
`replay_validate: <n> passed, <m> failed, <k> rows without move dumps skipped`. Every one
of the nine hybrid logs reports 0 failed. Section 3.3 lists the counts.

### 9.17 The other labelling mechanism

| value | number | file | how read |
|---|---|---|---|
| label source of the descent labeller | `nnlab_descent` | `supervised_valuenet/nn_labeler/descent.py` line 259 | the emitted record dictionary |
| label source of the loop | `spr_mcts` | `self_play_robots/spr/selfplay.py` | `_extract_records` record update |
| planner retrained on descent labels | 212 of 232, regret 3.047, 45.8% optimal | `supervised_valuenet/scaling/results/g24r4/comparison_nntwin.json` | agg of the `backward` system |
| second seed of the same | 204 of 232, regret 4.225 | `.../comparison_nntwin-seed21.json` | agg |

### 9.18 Lab arms cited in passing

| value | number | file | how read |
|---|---|---|---|
| `v15_slide_training`, unseen, seeds 7 and 8 | 172 and 170 of 200 | `results/variants/v15_slide_training{,_s8}/bench_unseen_astar.json` | agg `solved` |
| `v16_ranked_slide_training`, unseen, seeds 7 and 8 | 171 and 174 of 200 | `results/variants/v16_ranked_slide_training{,_s8}/bench_unseen_astar.json` | agg `solved` |
| `v00_control` seed 7, graded / frontier / unseen | 228 / 160 / 172 | `results/variants/v00_control/bench_*_astar.json` | agg `solved` |
| `v00_control` seed 8 | 224 / 155 / 170 | `results/variants/v00_control_s8/bench_*_astar.json` | agg `solved` |
| variant runner generation defaults | BOARDS 30, PER_BOARD 8, EXP 300, STOP 80, NOISE 0.25, EP 6, LR 1e-4, WORKERS 6, EMIT path, BACKUP min, SIB 1 | `python -m variants describe v09_strict_value --format env` | ran this session |
| seed nets for every lab arm | `mix_b2mix_iter2` | `jobs/variant_iter.slurm` | the `SEEDNETS` assignment |

---

## 10. Appendix — where the code and the project's documents disagree

The rule for this part is that the code wins. These are the places where following the
code changed what this part says.

**10.1 The v07 module docstring names a function that does not exist.**
`variants/v07_hybrid_actions.py` line 15 says "Search (mcts_root_slides, used by the
standalone driver ...)". There is no `mcts_root_slides` anywhere in the repository. A grep
over `self_play_robots/` returns only that docstring line. The algorithm is written inline
inside `bench_main`. This part describes the inline code.

**10.2 The v07 docstring describes only the depth-1 version.**
The docstring lists four steps and states the constants as "budget B0 (default 600)",
"top M (default 6)" and "budget SUB each (default 100)". The code implements
`--prefix-depth` 1, 2 or 3, a `--d2-from` frontier width, a merged candidate list in which
depth-1 and depth-2 prefixes compete, and a pruning rule. None of that appears in the
docstring. All landed flagship rows use depth 2 at 500 / 8 / 80, not the documented
600 / 6 / 100.

**10.3 The docstring's budget statement is only true for the depth-1 arm.**
It says "B0 + M*SUB <= 1200 = the arena convention". Depth 1 gives exactly 1200. The
flagship depth-2 configuration gives 1140 and the depth-3 probe gives 1160. Both are below
the control's 1200 and neither is stated anywhere in the project's prose.

**10.4 The pruning rule is not documented at all.**
`if best is not None and cost0 >= best[0] + 3: continue` is a real behavioural filter that
reduces the number of sub-searches actually run. It appears in no design note, no FINDINGS
entry and no docstring. Its constant 3 has no derivation in the file.

**10.5 The bench payload's `byref` field is always false, even for B2 runs.**
`spr/bench.py` hard-codes `"byref": False` in the protocol block. Meanwhile
`spr/search.py::run` sets `byref = vocab == "b2"` and passes it into the `Evaluator`. So
every B2 payload in this repository claims `byref: false` while by-reference helper
resolution was in fact active. The field is legacy and misleading. This part reports the
vocabulary from the `vocab` field and from `search_options`, not from `byref`.

**10.6 DESIGN.md states board ids and budgets the runs did not use.**
`DESIGN.md` section 3 says "Fresh lean boards per iteration (ids 5000+120(k-1)...)" and the
`spr.selfplay` docstring shows `--expansions 600 --stop-after 150 --per-board 8` with 100
boards. The B2 loop, which is the loop the headline claims rest on, used 60 boards from id
8000, 300 expansions and stop 80. Section 4b of `DESIGN.md` does record the id 8000 change.
The budget change is recorded only in the manifests. This part reports the manifests.

**10.7 DESIGN.md section 4 says the buffer includes exact anchors. The B2 loop had none.**
`jobs/selfplay_iter.slurm` sets `ANCHOR=0` whenever `VOCAB` is `b2`. So rounds 1 to 5 of
the B2 loop and all three mixed rounds trained on self-play records only. Section 4b of
`DESIGN.md` states this correctly. Section 4 does not.

**10.8 The flagship VERDICT rounds a denominator.**
`results/variants/v07_hybrid_actions/VERDICT.json` says "New-boards exam (unseen, 200
puzzles): 188 solved, with 1.47 extra moves vs perfect on the 137 puzzles with known
optima". 137 is the number of unseen instances that carry an exact optimum. The flagship
solves 134 of those 137. The 1.47 average is over 134 instances. I reproduced 1.4701 over
134. The `variants/FINDINGS.md` entry 17 states this correctly as "134 solved".

**10.9 The floor comparison in the project's prose mixes denominators.**
`variants/FINDINGS.md` entry 8 and the VERDICT compare the hybrid's graded mean regret,
which is an average over the instances the hybrid solved, against the ceiling probe's
`mean_gap_best`, which is an average over the instances the exhaustive search found
realizable. Those are 231 and 228 instances. The comparison is directionally right and the
paired recomputation in section 3.5 supports it more strongly than the prose does, at
0.760 against 0.978 over the same 225 instances. But the two published numbers are not a
paired comparison.

**10.10 The project reads the B2 fidelity gauge as uninformative. The files do not say so.**
`FINDINGS.md` entry 19(c) says "The fidelity gauge (0.51 -> 0.46) stays uninformative in B2
(section 14b)". The gauge output itself carries no such flag. Its `clean` field is `false`
for every B2 round, and its own docstring places 0.45-0.57 well inside the collapse band.
The proposed explanation, that the exact side is weakened by the `--solver-iters 100000`
cap, is plausible: `spr/gauge.py`'s own help text says that cap "keeps ~99% of rollouts".
I could not verify the claim that the remaining 1% explains a drop from 0.91 to 0.46.
This part reports the gauge numbers and marks the interpretation as open.

**10.11 A small inconsistency inside the ceiling summary.**
`results/ceiling/g24r4_b2.json` reports `summary.categories` as 228 `REALIZABLE_EXISTS`
and 4 `INCONCLUSIVE`, but `summary.capped` is 7. Reading the rows resolves it: 3 instances
hit a cap and still found a realizable plan, so they are counted as realizable and as
capped. Their "best" is not proven. Section 3.5 excludes them.

**10.12 The exhaustive probe underestimates the B2 solve ceiling.**
The probe proves 228 of 232 graded instances realizable in B2 and leaves 4 inconclusive.
The mixed-size curriculum's round-3 A\* row solves 230 of 232 with replay-certified B2
plans, which proves the ceiling is at least 230. `FINDINGS.md` entry 22 states this as
"the B2 ceiling probe says >=229 of 232 reachable". The probe alone supports only 228. The
230 comes from a planner run, not from the probe.
