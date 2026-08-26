# The self-play planner, explained

*An in-depth, plain-English account of the `self_play_robots/` project: what was
built, how it works, how big it is, how it was trained, and what it achieved.
Written for a smart reader with no machine-learning background. Every number
here is taken from a file in this repository; the source is named next to it.
Companion documents: `PROBLEM.md` (the brief), `DESIGN.md` (as-built design),
`FINDINGS.md` (the results log), `variants/FINDINGS.md` (the experiment lab).*

---

## Contents

1. [The game, and why it is hard](#1-the-game-and-why-it-is-hard)
2. [The planner's vocabulary: sub-goals instead of moves](#2-the-planners-vocabulary-sub-goals-instead-of-moves)
3. [What "self-play with a physics referee" means here](#3-what-self-play-with-a-physics-referee-means-here)
4. [The two networks](#4-the-two-networks)
5. [How the search works](#5-how-the-search-works)
6. [The training loop, end to end](#6-the-training-loop-end-to-end)
7. [The journey to the current best planner](#7-the-journey-to-the-current-best-planner)
8. [The scoreboard, in absolute terms](#8-the-scoreboard-in-absolute-terms)
9. [What is honestly still unknown](#9-what-is-honestly-still-unknown)

---

## 1. The game, and why it is hard

### 1.1 Rules

**Ricochet Robots** is played on a square grid — here `n × n`, with `n` between
16 and 96 — with walls on some cell edges. A handful of robots (4 or 8 in this
project) sit on distinct cells. One robot is the **target robot**; one cell is
the **target cell**.

A single **move** picks one robot and one of the four directions. The robot then
**slides in a straight line until something stops it** — a wall or another
robot. It cannot stop halfway, and it must travel at least one cell (a move that
would not slide at all is illegal). The puzzle is solved when the target robot
comes to rest on the target cell. The score is the **number of moves used**;
fewer is better.

The naming convention used throughout is `g<N>r<R>`: `g24r4` means a 24×24 board
with 4 robots (`GridEnv.py`, `supervised_valuenet/scaling/configs.py`).

### 1.2 Why it is hard

Three properties make this much harder than its tidy rules suggest.

**Sliding destroys local reasoning.** You cannot "step towards" the target. A
robot goes as far as it can go. So the question is never "which direction is the
goal in" but "what is out there that will stop me in the right place".

**The other robots are the tools.** The three or seven non-target robots (called
**helpers** in this codebase) are not obstacles to be avoided — they are the
walls you build. Optimal play routinely spends moves parking a helper on an
empty cell purely so the target robot can bounce off it later. That means the
solution to a puzzle is usually not a path; it is a small *construction
project*.

**The horizon is long and the interactions are non-local.** At the sizes studied
here, optimal solutions run roughly 8 to 30 moves (`PROBLEM.md` §2). On the
pinned 24×24 exam the average perfect solution is **7.59 moves**
(`results/ceiling/g24r4_base_slack12.json`, field `mean_d_star`); on the fresh
"unseen" exam it is **8.71 moves** over the 137 puzzles where perfect play is
known (`results/variants/exam/g24r4_unseen.dstar.jsonl`, verified by
re-computation). Every move changes what every future move can do, because every
robot is a potential wall for every other.

**How hard, concretely?** The project ships an exact optimal solver (a Rust
program, `supervised_valuenet/rust_datagen/`). It is the referee for anything
"gradeable", but it has two hard limits: it only runs at `n ≤ 64`, and even
below that it cannot always finish. The pinned 24×24 benchmark is split exactly
by that failure: **232 puzzles it solved** ("graded" — we know perfect play) and
**218 it could not** ("frontier" — we do not). On the 200-puzzle unseen exam,
the exact solver — given 200,000 expansions and 60 seconds per puzzle, then a
second pass at 5× those caps — cracked only **137 of 200**
(`variants/exam_dstar.py`, `variants/FINDINGS.md` entry 10).

That is the difficulty in one sentence: *on a third of these puzzles, nobody
knows the right answer, including the exact solver.*

### 1.3 What "solved" and "certified" mean here

A claimed solution counts only when it has been **replayed move by move against
the physics** and observed to finish on the target cell
(`supervised_valuenet/eval/realize.py::strict_moves`, and independently
`eval/replay_validate.py` after every benchmark run). This is called
**certification**, and it is absolute in this project: no number in any results
file comes from a network's opinion about whether a plan works. Validity is
checkable at *any* board size; only *optimality* needs the exact solver, and
therefore only exists at `n ≤ 64`.

---

## 2. The planner's vocabulary: sub-goals instead of moves

### 2.1 Two ways to think

You can search this game two ways.

- **Forwards, move by move.** Actions are the primitive slides (at most `4R` of
  them). The tree is narrow but very deep (8–30 levels). This is the "natural"
  AlphaZero framing, and the project keeps a planner in this space as a
  comparison arm (`spr/fwd/`).
- **Backwards, in sub-goals.** Reason from the goal: *what would have to be true
  for the target robot to arrive?* Actions are **sub-goal decisions**. The tree
  is wide (up to ~50 choices) but very shallow — 2 to 6 decisions
  (`PROBLEM.md` §2). This is the main line of the project.

The main line is backwards because the horizon is short enough to search
properly, because the strongest inherited baseline lives there, and because the
realize-and-certify machinery already existed.

### 2.2 What a sub-goal actually is

The planner's state is a **partial plan**: a little graph of *segments*, each
saying "robot X must get from cell A to cell B". The plan starts with exactly
one segment — "the target robot must get from where it is to the target cell"
(`skeleton/astar.py::_initial_plan`).

A segment is resolved in one of two ways:

- **For free, if it is already possible.** If there is an exact slide route from
  A to B, the segment is simply pinned to that route and costs what it costs.
  The code does this automatically wherever it can, before spending any thinking
  (`spr/search.py::forced_fixes`) — these are called *forced exact fixes* and
  they cost nothing against the search budget.
- **Otherwise, by choosing a sub-goal.** A sub-goal is a triple:

  > **bottleneck cell B** — an intermediate cell the mover will stop at, chosen
  > because *from B the mover can reach the segment's end*;
  > **support cell S** — the cell a robot must be standing on so that the mover
  > actually *stops* at B rather than sliding past it;
  > **helper robot H** — which robot gets sent to stand on S.

  In one sentence: **"send helper robot H to cell S, so that the mover stops at
  cell B, from which it can finish the segment."**

Choosing a sub-goal replaces one open segment with the resolved piece *plus two
new open segments*: "the mover must get from its start to B" and "helper H must
get from where it is to S". Those are then resolved the same way, recursively.
The plan is **complete** when no open segments remain, and its **abstract plan
cost** is the sum of all its segment costs.

The whole vocabulary — which sub-goals may even be proposed — comes from a
hand-written generator: `GridEnv.propose_subgoal_states` wrapped by
`skeleton/heuristics.py::propose`. This is important, and the project measured
its consequences (§2.4).

### 2.3 A real worked example

Board 42700 (a freshly generated 24×24 board from the self-play pool), one
random puzzle from it, run through the actual flagship networks on the login
node:

```
target cell: (2, 3)          target robot: Red, currently at (19, 13)
helpers:     Blue @ (10, 6), Green @ (20, 16), Yellow @ (1, 2)

open segment: Red must travel from (19, 13) to (2, 3)
              -> no exact slide route exists, so a sub-goal is needed
candidates the generator proposes: 12  (4 support cells x 3 helpers)
the policy network's top 5, with the value network's estimate of each:

  bottleneck (2,3)  support (1,3)  helper Blue   @(10,6)   prior 0.223   est. 25.3
  bottleneck (2,3)  support (3,3)  helper Blue   @(10,6)   prior 0.204   est. 23.9
  bottleneck (2,3)  support (3,3)  helper Green  @(20,16)  prior 0.198   est. 25.4
  bottleneck (2,3)  support (2,2)  helper Blue   @(10,6)   prior 0.196   est. 24.3
  bottleneck (2,3)  support (3,3)  helper Yellow @(1,2)    prior 0.180   est. 26.6
```

Read that as follows. Red cannot slide to (2,3) and stop there — nothing on the
board would halt it. So the generator reasons: the bottleneck must be the target
cell itself, and for Red to stop on it, some robot has to be standing on one of
its **four neighbours** — (2,4), (3,3), (2,2) or (1,3), one for each direction
Red might arrive from. Any of the three helpers could be that robot. Four
support cells × three helpers = exactly the **12 candidates** the generator
emits, and all 12 survive the plan-legality check.

The **policy network** scores all 12 and the search keeps the five it likes best
(`prior` is its probability for each, and they are close — 0.18 to 0.22, meaning
the policy has no strong opinion here). The **value network** then estimates,
for each of those five, how much more plan-cost is needed to finish the whole
job. It prefers *"send Blue to (3,3)"* at 23.9 over *"send Yellow to (3,3)"* at
26.6 — sensible, since Yellow at (1,2) is already sitting right beside the goal
and spending it as a blocker wastes a well-placed robot.

Notice also what has *not* been decided: whichever candidate is picked, the plan
still owes two new segments — "Red must get from (19,13) to (2,3)" and "Blue
must get from (10,6) to (3,3)" — and each of those will need its own sub-goal
decision unless a direct slide route happens to exist. That recursion is the
tree the search explores.

That single act — propose, rank with the policy, keep 5, price with the value —
is **one expansion**, the unit everything in this project is budgeted in (§5.1).

### 2.4 The ceiling: why no sub-goal planner can be perfect

Before training anything, the project asked a question that bounds the whole
enterprise: *if you had a perfect ranker and unlimited search, how good could a
sub-goal planner possibly be?* This is answerable exactly — enumerate the plan
space with no network at all, physically certify every complete plan, and
compare to the exact optimum. That is `spr/ceiling.py`
(`FINDINGS.md` §3, re-tightened in §14).

The answer on the pinned 24×24 graded exam:

| plan vocabulary | puzzles reachable at all | best certified plan | vs perfect play |
|---|---|---|---|
| base | 216 of 232 | 9.22 moves | **+1.63 moves** |
| extended ("B2") | 228 of 232 | 8.81 moves | **+1.17 moves** |

*(`results/ceiling/g24r4_base_slack12.json`, `results/ceiling/g24r4_b2.json`.)*

This is a **language limit, not a learning limit**. A sub-goal plan says "stop
here, using that robot as a blocker"; the actual optimal move sequence sometimes
does something a sub-goal simply cannot express, so the cheapest expressible
plan is longer. For comparison, the move-by-move (forward) planner in the same
repository averages **+0.07 moves** over perfect play on its exam
(`FINDINGS.md` §6). The gap is an order of magnitude, and no amount of training
inside the sub-goal language closes it. Recognising this early is what
eventually produced the project's best result (§7.6).

### 2.5 The extended vocabulary ("B2")

The base vocabulary saturated (§7.3), so the loop moved to an enriched one,
called **B2** throughout the code and logs. Three additions, all in
`skeleton/astar.py` and `skeleton/heuristics.py`:

- **Transient supports.** Normally a support cell needs a wall next to it, so
  the blocking robot has somewhere to stop. B2 also allows support cells with no
  adjacent wall — the blocker is only there *transiently*, held in place by
  timing or by another robot (`GridEnv.Subgoal.transient`,
  `heuristics.propose_b1`).
- **By-reference helpers.** If the plan already sends a robot somewhere, that
  robot can serve as a blocker *where it already is*, at zero extra cost,
  instead of being recruited a second time (`astar._reference_helpers`,
  `_apply(by_reference=True)`). Three shapes exist: sharing an existing support,
  using the target robot mid-journey as a stopper, and relocating a robot from
  one support cell to another.
- **Park repairs.** When a finished plan fails its physics replay because some
  robot is standing in the way, the code deterministically finds robots whose
  removal would unblock the failing slide, tries the cells they could slide to,
  and re-enters the repaired plans into the search at their real cost
  (`astar.py::park_repairs`). "Parking" is exactly what a human player does:
  shove the nuisance robot aside first.

B2 buys +12 reachable puzzles on the graded exam and lowers the best-plan floor
from +1.63 to +1.17 moves, at roughly 5× the search cost per puzzle.
**House rule:** a training corpus is either base or B2, never mixed — the two
have different candidate-set semantics (`PROBLEM.md` §10).

---

## 3. What "self-play with a physics referee" means here

### 3.1 The AlphaZero recipe, and what is missing

AlphaZero's loop is: a network plays games against itself; a tree search
improves on the network's raw instincts; the games' outcomes (win/loss) become
training labels; the network is retrained and the cycle repeats. The engine of
improvement is that **search is better than the network alone**, so the network
can be trained to imitate its own search — and then search on top of the
improved network is better still.

Two AlphaZero ingredients do not exist here:

- **There is no opponent.** This is a one-player puzzle. Nothing generates
  variety by trying to beat you.
- **There is no win/loss.** The outcome is a *cost* — the number of moves — and
  the goal is to minimise it, not to maximise a probability.

### 3.2 What replaces them

**The referee is physics.** Instead of "did I win", the ground truth is "when I
actually execute this plan on the board, does the target robot end up on the
target cell, and in how many moves". The simulator answers that exactly and
cheaply. So the loop's contract (`PROBLEM.md` §4.6, enforced in
`spr/search.py::Certifier` and `spr/selfplay.py`) is:

> A value backed up from an unrealised plan is a **hypothesis**.
> Only a **replayed, certified** plan sets the record.

This has a consequence worth stating plainly: **label corruption is structurally
impossible**. A self-play loop's classic failure is the network convincing
itself that something works and then training on its own delusion. Here it
cannot — an illegal shortcut fails the replay and produces no training record at
all. The only error that can creep in is *suboptimality*: certified plans that
are legal but longer than necessary. And even that is measurable, because up to
`n ≤ 64` the exact solver can be asked what the right answer was.

**Variety comes from the world, not from an opponent.** Every iteration
generates brand-new random boards (the "lean board" generator,
`nn_labeler/leanboard.py`, produces a 32×32 board in 0.24 s and a 96×96 one in
3.8 s) and brand-new random robot placements on them. Board IDs are namespaced
so that self-play boards can never collide with the benchmark pool (IDs 0–1199)
or the unseen exam (IDs 20000+) — `spr/selfplay.py` refuses to start otherwise.
Extra exploration inside a single puzzle comes from **Dirichlet noise** injected
into the policy's opinion at the root of the search tree (weight 0.25), the same
device AlphaZero uses.

### 3.3 The early-warning instrument

Because exact answers exist below 64×64, the loop can grade its own labels. The
**fidelity gauge** (`spr/gauge.py`) samples 200 first-move decisions per
iteration, has the exact Rust engine label them, and reports how often the
loop's own labels pick the same best candidate. The inherited calibration
(`supervised_valuenet/FINDINGS.md` §74) is: ~91% agreement is
downstream-equivalent to exact labels, ~89% keeps solve rate but loses
optimality, ~82% collapses.

The base-vocabulary loop scored **0.915 on its first iteration**
(`results/selfplay/g24r4_iter1/gauge.json`) — inside the safe band immediately.

In the B2 vocabulary the gauge became uninformative and this was diagnosed
honestly rather than ignored (`FINDINGS.md` §14b): exact B2 labels price plans
that *cannot actually be played* — only 13 of 28 sampled exact-optimal B2
completions physically realise — so the "exact" reference is the weaker labeller
there. In B2 the instrument is the certified benchmark itself. (A second finding
from the same review: the gauge was calling the exact engine with no iteration
budget, defaulting to 10 million iterations per rollout — 2.5 hours and 25 GB of
memory. Capping it at 100,000 kept 197 of 198 answers and finished in minutes.)

---

## 4. The two networks

The planner is two neural networks that share an identical **encoder** (the part
that looks at the board) and differ in what they do with it. Together they are
**~2.21 million parameters** — small by any modern standard, and deliberately so.

### 4.1 What the networks see: the board as tokens

Both networks turn the situation into `n² + 1` **tokens** — one per board cell,
plus one extra "global" token that acts as a scratchpad (its input row is all
zeros; `nn_labeler/encode.py::node_features`). Each token starts as a short list
of yes/no flags describing that cell.

**The value network's 9 channels** (`encode.py::node_features`, verbatim order):

| # | channel | what it means on the board |
|---|---|---|
| 0 | `seg_start` | where the moving robot is right now |
| 1 | `seg_end` | the cell it has to reach |
| 2 | `cand_bottleneck` | the cell this candidate proposes it should stop at |
| 3 | `cand_support` | the cell a blocker must occupy for that stop to happen |
| 4 | `cand_helper` | where the proposed blocker robot currently stands |
| 5 | all helper robots | every non-moving robot's position (one shared channel) |
| 6 | `ctx_open_endpoints` | endpoints of the plan's *other* still-unresolved segments |
| 7 | `ctx_bottlenecks` | bottleneck cells the plan has already committed to elsewhere |
| 8 | `ctx_supports` | support cells the plan has already committed to elsewhere |

Channels 6–8 are the reason this network can be trusted with a *whole-plan*
estimate: its job is "how much more will it cost to finish **this plan**", and
without the context channels it could not see the rest of the plan it is being
asked to complete.

**The policy network's 7 channels** (`spr/nets.py::policy_features`):

| # | channel | what it means on the board |
|---|---|---|
| 0 | mover now (`seg_start`) | where the moving robot is |
| 1 | goal (`seg_end`) | where it must get to |
| 2 | pinned support (`seg_support`) | a blocker already committed for *this* segment, if any |
| 3 | all robots | target robot and every helper (one shared channel) |
| 4 | `ctx_open_endpoints` | the plan's other unresolved segment endpoints |
| 5 | `ctx_bottlenecks` | already-committed bottleneck cells |
| 6 | `ctx_supports` | already-committed support cells |

The crucial difference: **the candidate is not marked**. The value network is
shown one candidate and asked "how good is this?"; the policy network is shown
none and asked "which one should we try?" — it has to *generate* the answer.

**The walls do not get a channel — they get the wiring.** Alongside the tokens,
each board contributes two `(n²+1) × (n²+1)` **attention masks**
(`encode.py::adjacency`): `A_all` connects cell *j* to cell *i* whenever a robot
could slide from *j* to *i*, and `A_ind` keeps only the slides that do **not**
require a helper as a blocker. Self-connections are added everywhere, including
on the global token. These masks are the board's geometry: information inside
the network only flows **along legal slides**. Two cells that are adjacent on
screen but unreachable from one another are, to the network, not neighbours at
all.

### 4.2 The shared encoder: one layer, twelve times

First, once, a linear map lifts each token's 7 or 9 channels to a
**192-dimensional vector** (`d_model = 192`). Then the block —
`nn_labeler/model.py::LoopedLayer`, identical in both networks — does this:

1. **Three families of attention run in parallel**, 4 heads each:
   - `g` — *global*: unmasked, every token may look at every token (this is
     what makes the scratchpad token useful);
   - `a` — masked by `A_all`: a token may only look at cells one slide away;
   - `i` — masked by `A_ind`: only slides that need no helper as a blocker.
   Their three 192-vectors are concatenated (576) and projected back to 192.
2. Residual connection + LayerNorm, then a small feed-forward network
   (192 → 768 → 192, GELU), residual + LayerNorm again.

That is a fairly ordinary transformer block. The unusual part is what happens
next:

> **The same block is applied 12 times** (`recurrence = 12`), reusing the *same
> weights* every time.

This is the "recurrence trick". Twelve rounds of message-passing along the slide
graph let information travel twelve slides across the board — deep enough to
reason about multi-hop routes — while costing the parameters of **one** layer.
Measured: the shared block is **740,928 parameters**; an ordinary untied
12-layer stack of the same block would be **8,891,136**. That is a 12× parameter
saving, and it also means the depth is a *hyper-parameter of inference*, not a
shape baked into the weights.

### 4.3 Why one network works at any board size

This is the project's most distinctive architectural choice, and it is a
*subtraction*.

The inherited supervised networks (`supervised_valuenet/train/policy_tf.py`,
`train/looped_pc.py`) each carry a learned **positional table** — one 192-number
vector per board cell, `self.pos` of shape `(n² + 1) × 192`. At 24×24 that is
**110,784 parameters** (confirmed by loading
`assets/g24r4_backward_policy_exact.ckpt`). It is the only tensor in those
networks whose *shape* depends on the board size — and it makes the checkpoint
useless at any other size.

The networks used here set `pe = "none"`: **no positional embeddings at all**
(`nn_labeler/model.py`, `spr/nets.py`). Where does position information come
from, then? From the attention masks. The network does not need to be told "this
is cell (7, 12)"; it needs to know *what can reach what*, and the mask says
exactly that. So the checkpoint contains nothing board-shaped, and **one file
runs at 16×16, 24×24, 32×32, 64×64 or 96×96 without modification.**

The evidence that this is not merely legal but *good*. The value network,
trained on 16×16 and 24×24 boards only, was audited against exact ground truth
at 32, 40, 48, 56 and 64: it picks the optimal candidate **86.3% of the time at
32×32 and 83.8% at 64×64**, beating the 8×8–16×16-trained labeller that
initialised it at every one of those sizes (85.3% and 81.4%) —
`FINDINGS.md` §17, `results/audit/b2it0_m1mixed.json`. And the *planner pair*,
trained at 16×16 and 24×24 only, beat the **purpose-built** 32×32 supervised
planner at 32×32 by 9 puzzles using 7× fewer search steps (`FINDINGS.md` §9).

The one price: because the masks are dense `(n²+1)²` matrices, memory grows as
`n⁴`, so batches must shrink at larger sizes and a training batch may never mix
board sizes. Both are handled explicitly (§6.4).

### 4.4 The value network — exact size and output

`nn_labeler/model.py::SizeFreeValueNet`, checkpoint
`assets/labeler_prod_v1_s11.ckpt` (and the flagship's retrained copy, identical
in shape):

```
hyper-parameters: d_model=192, recurrence=12, heads=4, num_classes=96,
                  pe='none', in_channels=9

input Linear   (9 -> 192)                          1,920
LoopedLayer    (applied 12x, weights shared)     740,928
readout head   (960 -> 192 -> 192 -> 96)         240,096
------------------------------------------------------------
trainable parameters                             982,944
(plus a 96-element non-trainable "bins" buffer)
```

**The readout** gathers exactly five token vectors — the bottleneck cell, the
support cell, the helper's cell, the segment start and the segment end
(`encode.py::key_indices`) — concatenates them (5 × 192 = 960) and pushes them
through a small three-layer network.

**The output is not a number — it is a 96-way distribution.** The final layer
produces 96 logits, one per integer "cost-to-go" bin from 0 to 95; a softmax
turns them into probabilities and the reported value is the expectation
`Σ p_i · i`. Predicting a *distribution* rather than a single number is a
well-established stabiliser for value learning, and it also lets the network
express uncertainty ("probably 6, maybe 9") instead of averaging into a
meaningless 7.4.

Ninety-six bins, not the 50 the older per-size nets used, because realized costs
at 32×32 exceed 50 and the older code silently clamped them
(`PROBLEM.md` §4.5).

**The two training losses** (both inherited verbatim from the supervised value
net, `model.py::training_step`):

- **HL-Gauss classification.** The true integer cost is not a one-hot target;
  it is smeared over neighbouring bins with a Gaussian of width σ = 1, and the
  network is trained to match that smeared target. "Being one off" is therefore
  a small error, not a total miss.
- **A ranking loss.** Within one decision group (all candidates for the same
  segment), the network's costs are turned into a softmax of their negatives and
  pushed towards the candidates that were actually optimal. This is what makes
  the network good at *choosing*, which is all the search ever asks of it.

Validation metrics are `val_regret` (how many extra moves you would pay by
trusting its top pick), `val_top1_optimal`, and `val_group_spread` — the last
being a collapse alarm (§6.4).

### 4.5 The policy network — exact size and the three pointer heads

`spr/nets.py::SizeFreePolicyNet`, flagship checkpoint
`runs/spr/variants/v09_strict_value_s8/policy/.../epoch=0-step=41.ckpt`:

```
hyper-parameters: d_model=192, recurrence=12, heads=4, pe='none', temp=1.0

input Linear   (7 -> 192)                          1,536
LoopedLayer    (applied 12x; same design as the value net's,
                own weights, initialised from the labeller)
                                                 740,928
seg   MLP      (576 -> 192 -> 192)               147,840
q_bn  MLP      (192 -> 192 -> 192)                74,112
q_sup MLP      (384 -> 192 -> 192)               110,976
q_help MLP     (576 -> 192 -> 192)               147,840
------------------------------------------------------------
trainable parameters                           1,223,232
```

**The three heads pick the three parts of a sub-goal, in order.** Recall from
§2.2 that a sub-goal is *(bottleneck, support, helper)*. The network chooses
them one at a time, each choice conditioned on the previous ones — this is what
"autoregressive" means:

1. **Summarise the segment.** `seg` takes the token vectors of the segment start
   and end plus the average of all tokens, and produces one 192-dimensional
   *query vector* `g` meaning "this is the journey I need to arrange".
2. **Which bottleneck?** `q_bn(g)` produces a query; it is dot-multiplied with
   the token vector of every *valid* bottleneck cell, and a softmax over those
   dot products gives a probability per cell.
3. **Which support?** `q_sup([g, h(bottleneck)])` — now conditioned on the
   bottleneck just chosen — queries the support cells compatible with it.
4. **Which helper?** `q_help([g, h(bottleneck), h(support)])` queries the cells
   the helper robots currently stand on.

A candidate's **prior** is the product of the three probabilities (the sum of
three log-probabilities), renormalised over the candidate set that the generator
actually produced (`spr/nets.py::heads_logp_map`, `spr/search.py::expand`).

**Why these are called "pointer" heads, and why they matter.** A conventional
classifier has a fixed list of outputs — 10 digits, 1000 image classes, 361 Go
points. A pointer head has none. It emits a *query vector* and scores whatever
candidates are actually on the table by dot product. So the same weights work
with 3 candidates or 50, on a 16×16 board or a 96×96 one, with 4 robots or 8.
Combined with the absence of positional embeddings, this is what makes the whole
planner genuinely size-free.

**Training target.** For each decision, the certified costs of the candidates
are turned into a softmax of their negatives (temperature 1) and that soft
distribution is decomposed onto the three heads
(`nets.py::_soft_targets`). Note what this is *not*: it is not AlphaZero's
"train the policy on the search's visit counts". That alternative was tested
explicitly and lost catastrophically (§7.6, `v01_visit_policy`).

### 4.6 Size summary

| network | trainable parameters |
|---|---|
| value (`SizeFreeValueNet`, 9 channels, 96 bins) | **982,944** |
| policy (`SizeFreePolicyNet`, 7 channels, 3 pointer heads) | **1,223,232** |
| **the whole planner** | **2,206,176** |
| *for comparison:* per-size supervised policy (24×24) | 1,334,016 — of which 110,784 is the board-shaped `pos` table |
| *for comparison:* per-size supervised value (24×24) | 1,084,900 — of which 110,784 is `pos` |
| *for comparison:* the same encoder if its 12 rounds were untied | 8,891,136 for the trunk alone |

*(All counted by loading the checkpoints on CPU and summing `p.numel()` over
`model.parameters()`.)*

---

## 5. How the search works

### 5.1 One expansion — the unit of everything

Every budget, every comparison and every cost figure in this project is
denominated in **expansions**, and an expansion is defined identically to the
inherited benchmark harness (`spr/search.py::expand`,
`PROBLEM.md` §6.5):

> **One expansion** = take one partial plan → ask the generator for every legal
> candidate sub-goal → **one policy pass** to score them all → keep the top
> **k = 5** by policy score → **one batched value pass** over those 5 → five
> child plans are born, each carrying its cost estimate.

What is deliberately **free** (not counted):

- forced exact fixes — segments that can be pinned without any choice;
- all physics: playability pre-checks, full replay, certification;
- park repairs in the B2 vocabulary.

That is the accounting convention of the frozen benchmark harness
(`supervised_valuenet/eval/compare.py`), preserved exactly so that every number
this project reports is comparable, row for row, with the numbers the supervised
campaign recorded. The benchmark budget is **1200 expansions per puzzle, k = 5**
throughout.

Two cost estimates exist for each child, and the difference mattered:
`f = fixed_cost(parent) + estimate` (label-consistent) versus
`f = fixed_cost(child) + estimate` (the inherited harness's, which
double-counts the new segment's own cost). The double-counting turned out to be
*helpful* at 24×24 — it penalises deep commitments — so it was kept rather than
"fixed" (`FINDINGS.md` §5b).

### 5.2 MCTS: how the two networks combine

`spr/search.py::mcts` is a **PUCT** tree search — the AlphaZero selection rule,
adapted for costs.

**Selection.** From the root, repeatedly pick the child maximising

```
    norm(Q)  +  c_puct · P · √(N(parent) + 1) / (1 + N(child))
```

- `Q` is the child's current cost estimate. Because *low* cost is good, `Q` is
  min-max normalised across the whole tree so that the cheapest becomes 1 and
  the dearest 0 (the MuZero convention, clamped to [0,1]).
- `P` is the policy's prior for that child — the first term says "go where it
  looks good", the second says "go where the policy expected good things but we
  have not looked much".
- `c_puct = 1.5`.
- Unvisited children are *not* a special case: they already carry their own
  value estimate from the batched value pass, so no "first-play urgency" hack is
  needed.

**Backup — minimum, not average.** In AlphaZero a node's value is the *average*
over its subtree, because an opponent gets to reply. Here there is no opponent
and no dice: whatever the search finds, the planner may simply choose to do. So
a node's value is the **minimum** over its live children — the best thing
findable below it. (The average was implemented and tested as an option; on
these shallow trees it makes no measurable difference — `FINDINGS.md` §5d.)

**Terminals are settled by physics, not opinion.**

- A complete plan is replayed. If it works, the node becomes a **certified
  exact leaf**: its value is a fact, and it is never selected again.
- If it fails, the node is **dead** — removed from its parent's value entirely,
  and the parent re-evaluated. In the B2 vocabulary, before dying it gets one
  chance: the deterministic **park repairs** become its children.
- A node whose children are all closed is itself closed.

Because every iteration either expands, certifies or closes something, the
search always terminates — at the budget, or when the root closes, meaning the
entire top-5 tree has been exhausted.

**A measured caveat, recorded honestly** (`FINDINGS.md` §14d): with `k = 5` the
tree closes long before 1200 expansions — the flag `root_closed` is true on
100% of graded instances. So "MCTS at 1200 expansions" is in practice *an
exhaustive certified enumeration of the top-5 tree*, roughly 5 physical
realizations per puzzle instead of A\*'s 1, at 3–5× the wall time. The 1200 cap
is not binding, and the value network's role in that regime is to order the
visits rather than to prune.

### 5.3 The other searches

- **Greedy** — one expansion per decision, take the value network's cheapest
  child, never look back. This is the labelling rule inherited from the
  supervised project and the "no search at all" baseline.
- **A\*** — classic best-first over partial plans, ordered by estimated total
  cost. Plain mode returns the first complete plan found (this reproduces the
  frozen harness row for row); *anytime* mode certifies at each pop, discards
  unplayable plans and keeps going; *best-at-budget* keeps going while anything
  in the frontier still estimates cheaper than the best certified plan so far.
  A\* is the cheap workhorse: it is what the loop's headline "26 expansions per
  puzzle" figures refer to.

### 5.4 The hybrid search — the flagship's search

`variants/v07_hybrid_actions.py`. This is the change that broke the ceiling of
§2.4, and its idea fits in one line: **let the planner make one or two ordinary
robot moves *before* it starts sub-goal planning.**

Why that helps: the ceiling is a property of the *language*. A sub-goal plan
starting from position P may be unable to express the optimal solution. But a
sub-goal plan starting from position P′ — P after one ordinary slide — is a
*different* plan space, and it may contain something cheaper, even after paying
one extra move for the slide.

The mechanism is a **portfolio**: several independent searches, cheapest
certified answer wins.

1. Run standard MCTS on the original position with budget `B0` (flagship: 500
   expansions).
2. Enumerate every legal single-robot slide. For each resulting position, build
   its initial plan (free) and read off its abstract cost. Rank by that. At
   depth 2, take the best 4 and expand each by one more slide.
3. Keep the best `top_m` prefixes (flagship: 8) and run MCTS on each with a
   small budget `sub` (flagship: 80). A prefix of length L pays **+L strict
   moves** on top of whatever its sub-plan costs.
4. Return the cheapest **certified** total.

Budget bookkeeping is honest: `B0 + top_m × sub ≤ 1200`, actual expansions are
summed and reported per puzzle (measured mean on the graded exam: **582**), and
the composed move sequence `[slides] + [sub-plan moves]` is replay-validated
against the **original** position, not the shifted one. The comparison
throughout is against *the same networks under standard search at the same
budget*, which is the only fair control.

Depth results (`variants/FINDINGS.md` entries 8, 14a, 17, 20), graded exam:

| search | mean extra moves vs perfect play | % solved perfectly |
|---|---|---|
| standard MCTS, same nets, same budget | 1.42 | 59.6% |
| hybrid, depth 1 | 1.01 | 66.2% |
| hybrid, depth 2 (**flagship**) | **0.944** | **68.4%** |
| hybrid, depth 3 | 0.861 | 70.6% |
| *the pure sub-goal language's proven floor* | *1.17* | *62.7%* |

Depth 2 beats depth 1 on every exam where both ran (graded 10/1, p = 0.012;
unseen 11/0, p = 0.001; frontier 10/2, p = 0.039) and depth 3 beats depth 2 on
the graded exam (7/0, p = 0.016), but the curve is clearly flattening and the
unseen exam stopped improving at depth 2, so no deeper probes were run. Two honesty notes
are recorded in the log: part of depth 2's gain comes from screening more
prefixes (`top_m` 8 vs 6), not from genuinely two-slide plans — only 5 of 46
unseen slide-wins used both slides; and the depth-3 number is a graded-exam-only
result.

---

## 6. The training loop, end to end

Below is one iteration of the mixed-size curriculum
(`jobs/selfplay_mix_iter.slurm`), which is the loop in its final form. Every
setting quoted is the job script's actual value, and every measured figure comes
from **iteration 1**'s own manifests
(`results/selfplay/mix_b2mix_iter1/generation_*.manifest.json`).

### 6.1 A — Generate

For each of three configurations (24×24 with 4 robots, 32×32 with 4 robots,
24×24 with 8 robots):

- **30 fresh boards** are generated from scratch (IDs 9000+, seeded so the
  iteration is reproducible), plus **8 random puzzles per board**.
- Each puzzle is attacked with **MCTS in the B2 vocabulary**: 300 expansions
  maximum; stop early if the best certified plan has not improved for 80
  expansions; `c_puct` 1.5; Dirichlet root noise 0.25; **all** root candidates
  scored rather than only the top 5, so the first decision's labels cover the
  full candidate set; **greedy sibling completion** on, meaning candidates the
  tree never got round to certifying are finished off greedily and certified so
  they too can be labelled.
- 6 worker processes (4 at 32×32), each holding its own copy of the networks on
  the GPU; 900-second per-puzzle cap; records streamed back and written
  atomically with a manifest.

Measured, iteration 1:

| config | puzzles attempted | solved | training records | wall clock |
|---|---|---|---|---|
| g24r4 | 309 | 290 (94%) | 3,283 | 1,865 s |
| g32r4 | 301 | 281 (93%) | 2,862 | 2,778 s |
| g24r8 | 331 | 293 (89%) | 5,770 | 4,667 s |
| **total** | **941** | **864** | **11,915** | ~2.6 h |

### 6.2 B — Turn certified plans into labels

This is the heart of the contract. Walking the finished search tree
(`spr/selfplay.py::_extract_records`):

- For each **decision node** the search expanded, look at its children.
- A child is **labelled only if a complete plan through it was physically
  certified** — either found inside its subtree, or produced by greedy sibling
  completion and then certified. Anything else is discarded, silently and
  deliberately.
- Its label is `cost_to_go = (abstract cost of the cheapest certified plan below
  it) − (fixed cost already committed at this decision)`. These are exactly the
  units the older supervised labellers used, so the records are drop-in
  compatible.
- `is_optimal` marks the argmin of the group. **A decision with fewer than two
  labelled candidates is dropped entirely** — with only one candidate there is
  nothing to learn about ranking.
- Every record also carries the **realized strict move count** of the plan it
  came from (`strict_total`), the abstract total, the search's visit count, the
  policy's prior, the value net's estimate, the iteration number, and which
  checkpoints produced it. Eighteen frozen fields plus provenance.

By default only decisions on the **principal path** (the line leading to the
best certified plan) are emitted. The lab later showed that emitting **all**
expanded decisions is better (§7.6, `v04`).

### 6.3 C — Buffer

A rolling window over the last 3 iterations of that configuration
(`spr/buffer.py`), one buffer per configuration.

### 6.4 D/E — Retrain

**One** policy/value pair is trained on the **union** of the three
configurations' buffers (`spr/train.py`):

| | policy | value |
|---|---|---|
| initialisation | **warm start** from the previous iteration's checkpoint | same |
| epochs | 6 | 6 |
| learning rate | 1e-4 | 1e-4 |
| batch | 8 decision groups | 4 groups × at most 8 records each |
| extras | gradient clipping 1.0, `--byref` | — |
| batch shrinking | `--batch-ref-n 24` | same |

Five safety mechanisms are wired in from day one, every one of them the scar
tissue of an earlier failure (`PROBLEM.md` §4.4):

- **Warm starting.** Cold retrains are seed-unstable; warm ones are not.
- **`CollapseStop`.** Value networks in this family are *bistable*: they can
  fall into a mode where they emit the same number for everything. The detector
  is `val_group_spread` — the average spread of predictions within a decision
  group. Three consecutive epochs below 0.05 and training stops. (The production
  labeller itself collapsed at epoch 2; its banked checkpoint is the epoch-0
  snapshot.)
- **Checkpoint on minimum `val_regret`.** A collapse can therefore never destroy
  the best epoch.
- **`save_last` + `RR_RESUME`.** Every job is idempotent and resumable, because
  the cluster's walltime limits are hard kills.
- **Size-bucketed batches.** A batch may never mix board sizes (the attention
  masks are dense and cannot be padded cheaply), and above 24×24 the batch size
  is scaled down by the mask-memory ratio `((24²+1)/(n²+1))²` so that one memory
  envelope covers the whole curriculum. At 32×32 that turns a value batch of 4
  into 1 and a policy batch of 8 into 2. Without this the job simply runs out of
  GPU memory — an early attempt hit 13 GB per step.

Training takes about half an hour.

### 6.5 F/G — Bench and gate

- **Bench**: the A\* planner on every configuration's graded and frontier exam,
  1200 expansions, k = 5, every solved row replay-certified. ~2.7 hours.
- **Gate**: paired statistical tests against the previous iteration and against
  the frozen supervised baselines (`spr/gate.py`) — McNemar's test on the
  per-puzzle solved/unsolved vectors, and a sign test on move counts over
  puzzles *both* systems solved. Paired tests, not aggregate comparisons,
  because the measured seed-to-seed noise on this protocol is 3.5 solve-rate
  points and 6.6 optimality points — large enough to fabricate a convincing
  fake improvement (`PROBLEM.md` §4.7). A one-point gain is not a result.

### 6.6 Cost

**≈ 5–6 hours on a single A100 per mixed iteration = 0.65–0.85 node-hours.**
The 24×24-only iterations were cheaper, 2.1–2.5 hours ≈ 0.3 node-hours, of which
the frontier benchmark alone was 40%. The entire five-iteration 24×24 series
cost ≈ 2.9 node-hours; the whole variants lab ≈ 20.8 of its 50-node-hour budget
(`variants/FINDINGS.md` entry 17c).

---

## 7. The journey to the current best planner

### 7.0 The chain in one line

```
labeller value net (trained on 8-16 boards only)
  -> M1 size-free pair (trained on existing exact corpora at 16x16 + 24x24)
  -> 5 self-play iterations at 24x24 in the extended B2 vocabulary
  -> 3 mixed-size curriculum iterations (24x24 + 32x32 + 8-robot)
  -> one lab iteration with real-move value targets (v09, seed 8)
  -> + the depth-2 hybrid search at inference
  = the flagship planner
```

Every arrow after the second is **label-free**: no human annotation, no exact
solver, only the loop's own certified searches.

### 7.1 The bootstrap (M1): rebuild size-free, and beat the specialists

The inherited assets were per-size supervised planners and one size-free *value*
network (the "labeller", trained only on 8×8–16×16 boards). The first step was
to rebuild both networks on the size-free recipe and prove they matched the
specialists — the gate being "within 3.5 solve-rate points and 6.6 optimality
points", the measured seed noise.

**Two things had to be discovered first.** The policy network did not train at
all under the supervised recipe: at learning rate 3e-4 with no gradient
clipping, it peaked at epoch 0 and then *degraded* (validation regret 4.74 →
6.3–7.75 by epochs 6–24). Without positional embeddings, gradients through 12
weight-tied recurrences blow up. Clipping restored monotone training, and
initialising the encoder from the labeller's was better still. Adopted recipe:
**25 epochs, lr 1e-4, gradient clip 1.0, encoder initialised from the labeller**
(`FINDINGS.md` §4).

Result (`FINDINGS.md` §7): the gate was not merely met, it was overshot.

| | g16r4 exam (450) | g24r4 graded (232) |
|---|---|---|
| per-size supervised pair | 401 solved, +2.14 vs perfect, 7.1 expansions | 205 solved, +4.20, 26.9 expansions |
| one size-free pair, both sizes | 401, +2.04, 8.9 | **215, +2.43, 5.3** |

One network replaced two, lost nothing at 16×16, and at 24×24 gained 10 puzzles
and 1.8 moves while using **five times less search**. Trained on 16×16 alone it
still beat the 24×24 specialist at 24×24; trained on 24×24 alone it was the best
16×16 planner in the table. A second seed replicated this to within 0.1 moves
(`FINDINGS.md` §18).

### 7.2 Search beats greed (M2) — and immediately hits a wall

With those networks frozen, MCTS was compared to greedy descent and to the
inherited A\* on identical puzzles and budgets. It won decisively: 80/0 and 24/0
paired move wins against greedy, 59/0 and 27/0 against A\*
(`FINDINGS.md` §8).

And then the ceiling probe was overlaid on the same table, and MCTS's regret sat
**0.06–0.12 moves above the exhaustive language optimum** at both sizes.

> **The base sub-goal language was already saturated by the supervised networks
> plus search.** No self-play iteration could ever register a gain on those
> exams — the remaining headroom was smaller than the noise.

This was confirmed on every pinned exam, frontier sets included
(`FINDINGS.md` §11), and it was arguably the most valuable finding of the early
project: it redirected everything.

### 7.3 The clean negative result

Two full self-play iterations were run in the base vocabulary anyway
(`FINDINGS.md` §10, §12a) — to exercise the machinery end to end, to measure
generation cost, and to check the label quality gauge (0.915 on iteration 1,
comfortably inside the safe band). They moved nothing, in either direction.

That is a **clean negative result with an explanation**, and it is logged as
such. It also produced a small warning worth keeping: iteration 2 tried
filtering to "hard" instances only, and the label fidelity gauge dropped from
0.915 to 0.85 — harder puzzles get looser certified labels, exactly in the band
where the inherited dose-response curve says quality starts to cost you.

### 7.4 The B2 pivot, and what the loop actually learns

The loop moved to the extended vocabulary. Five iterations at 24×24
(`FINDINGS.md` §15, §19). The headline trend, on the frontier exam (the 218
puzzles the exact solver could not crack — the only exam with room):

```
iteration:   0      1      2      3      4      5
A* solves:  133 -> 144 -> 139 -> 158 -> 153 -> 153     (it5 vs it0: +27/-7, p=8e-4)
expansions:  53 ->  43 ->  35 ->  36 ->  26 ->  30     per puzzle, graded exam
```

Against the frozen supervised B2 planner: 227 vs 199 on the graded exam
(p = 2e-7) and 153 vs 125 on the frontier (p < 1e-4). The M4 gate — "the final
network beats the supervised baseline beyond seed noise" — was met by a wide
margin.

**But read the second row.** What the loop learned was not longer reach; it was
**efficiency**. The cheap first-solution A\* planner, after five iterations,
reached in 26–30 expansions what the iteration-0 networks had needed a
500-expansion MCTS to reach. That is precisely the AlphaZero mechanism —
*search distilled into the network* — and it is stated that way in the log.
What did **not** move was realized move count, on any pairing. On the project's
headline metric the 24×24 B2 loop was flat, and it saturated after three
iterations.

### 7.5 The curriculum: distribution, not capacity

The plateau was diagnosed as a *data-distribution* limit: six thousand records
per iteration, all from the same 24×24 4-robot distribution, plus a measurable
drift — after four iterations of 24-only self-play the networks were 1–3 points
*worse* than their own initialisation on 32×32-to-64×64 decisions
(`FINDINGS.md` §17c). Specialisation.

The fix was to self-play on all three target distributions at once. **One
iteration of the mixed curriculum beat everything before it**
(`FINDINGS.md` §20):

| exam | best 24-only result | mixed iteration 1 | paired |
|---|---|---|---|
| g24r4 frontier (218) | 158 | **170** | +30/−13, p = 0.002 |
| g32r4 frontier (275) | 208 | **235** | +38/−11, p = 2e-5 |
| g24r8 frontier (289) | 171 | **269** | +108/−0 vs the supervised B2 arm |

The 8-robot jump (269 of 289 = 93% solved, against 161 for the recorded
supervised arm) has a clean explanation: B2's parking and by-reference machinery
matters most on crowded boards, and nobody had ever trained a ranking for it on
8-robot boards. The specialisation drift also disappeared — the same single
network now held all six exams.

Iterations 2 and 3 were flat (`FINDINGS.md` §22). This is the loop's recurring
shape: **one big step when something genuinely changes, then flat**. Iteration 3
did deliver the best graded solve count of the plain-search line, 230 of 232 —
one or two short of exhausting that exam.

### 7.6 The variants lab: 21 arms, four adoptions, and some instructive failures

At this point the project opened a controlled experiment lab — twelve designs,
twenty-one arms once seed replicates and chained runs are counted
(`variants/`, `variants/DESIGN.md`) — under the owner's directive: structurally
different mechanisms, not knob tweaking; every claimed win replicated at a
second random seed; every arm run under an identical matched protocol from the
same frozen starting networks; a new **unseen exam** of 200 puzzles on 50 fresh
boards, pinned once and never regenerated, that nothing ever trains on.

**What was adopted:**

- **v09 — train the value network on the metric you are graded on.** The entire
  stack, supervised and self-play alike, trained the value network on *abstract
  plan cost*, then graded it on *realized moves*. The two diverge precisely
  where plans get subtle. Self-play makes the fix possible for the first time,
  because every certified record carries the plan's real move count. Mechanism:
  rescale each record's target by `strict_total / abstract_total`. Result:
  better on every exam at both seeds (frontier Fisher p = 0.0015), best graded
  regret of the program at the time, best unseen row ever recorded. **This was
  the single strongest network change of the project.**
- **v04 — keep everything the search examined.** The control trained only on
  the winning line. The search certifies many off-line decisions whose labels
  are equally physics-certified, and AlphaZero trains on every visited state.
  Switching to "emit all" roughly doubles the records per unit of generation.
  Replicated: frontier +11 and +21 at two seeds (Fisher p = 3.4e-5). Caveat
  logged: on 8-robot trees the extraction cost is ~6× higher.
- **v14 — their stack**, adopted as the lab recipe; wins the unseen exam at both
  seeds (Fisher p ≈ 0.018).
- **v07 — the hybrid search** (§5.4), the flagship.

**What was killed, and why that matters:**

- **v01 — AlphaZero's own policy target.** Training the policy on the search's
  *visit counts*, the textbook recipe, was catastrophic here: every exam
  collapsed, p ≤ 1e-7, regret doubled. The counterfactual is the point — it
  proves the certified-cost softmax target is load-bearing rather than
  incidental. (It is also, in fairness, closer to Gumbel-AlphaZero's
  "completed-Q" target than to plain visit counts, which is why the comparison
  was worth making.)
- **v12 — the hard-puzzle curriculum.** Screening each puzzle with a quick
  attempt and practising only on failures: flat after one iteration, and still
  flat after three chained iterations. Retired.
- **v06 — Gumbel root exploration.** A convincing seed-7 win (19/6, p = 0.015)
  that reversed at seed 8 (14/16, p = 0.86). Downgraded to "not replicated" —
  this is exactly what the two-seed rule exists to catch.
- **v15 and v16 — teaching the networks about slides.** Two attempts to train
  the networks specifically for the hybrid search: with uniform random nudges
  (v15) and with the search's own top-ranked slide (v16). Both flat at both
  seeds. The explanation arrived later from the transfer result (§8.3): the
  networks already generalise to post-slide positions, so there was no gap to
  close. Both closed honestly as failures.

**The control that turned into a headline: v08, cold start.** Every other arm
warm-starts from supervised networks. This one started from **random
initialisation** — no supervised weights, no exact labels, nothing but one
iteration of the loop's own certified self-play data (3,787 certified records from 306 solved puzzles). It
reached **132 of 200 on the unseen exam**, against the fully-supervised
per-size pair's **134 of 200** — statistically indistinguishable. The
supervised head start is worth about 40 unseen puzzles to the warm arms, but it
turns out not to be *needed* to reach supervised-level competence.

### 7.7 The flagship

`variants/FINDINGS.md` entries 15, 17. The final combination is:

> **the v09 strict-value networks + the depth-2 root-slides hybrid search.**

Chosen by rule, not by taste: v15 added nothing so it was excluded, depth 2 beat
depth 1 on both exams where they were both measured, so it was included. The
networks come from one lab iteration (seed 8, 30 boards, 328 puzzles, 290
solved, 3,370 certified records, 32 minutes of generation) on top of the mature
mixed-curriculum networks.

---

## 8. The scoreboard, in absolute terms

All rows below: 1200 expansions, k = 5, every solved puzzle replay-certified.

### 8.1 The graded exam — 232 puzzles where perfect play is known

| planner | solved | mean moves | extra vs perfect | % perfect |
|---|---|---|---|---|
| **flagship** (v09 nets + depth-2 hybrid) | **231 / 232** | **8.62** | **+0.944** | **68.4%** |
| same networks, standard search, same budget | 230 / 232 | 9.08 | +1.42 | 59.6% |
| *the pure sub-goal language's proven floor* | *228* | *8.81* | *+1.17* | *62.7%* |
| supervised backward planner (B2 arm, recorded) | 199 / 232 | 12.33 | +4.89 | 36.7% |
| supervised backward planner (base arm, recorded) | 205 / 232 | 11.80 | +4.20 | 38.5% |

Perfect play on this exam averages 7.59 moves. The flagship uses 0.94 more —
**below the proven floor of any pure sub-goal planner**, which is only possible
because the hybrid search is not a pure sub-goal planner. Against its own
matched control the paired result is 30/0 move wins at depth 1 (p = 1.9e-9) and
40/0 at depth 2 (p = 1.8e-12). Mean cost: 582 expansions, 21 seconds per puzzle.

*Two caveats on the floor row, both from the project's own audit
(`FINDINGS.md` §14 (iii)): the ceiling probe averages over the 228 puzzles it
proved reachable while the flagship averages over the 231 it solved, and the
probe's search order assumes realized moves never come in below abstract plan
cost, which certified plans occasionally violate. The honest reading is
"the hybrid is below the pure-sub-goal floor by clearly more than the floor's
own ±0.05 uncertainty", not "below it by exactly 0.23".*

### 8.2 The frontier exam — 218 puzzles the exact solver could not crack

| planner | solved | mean moves |
|---|---|---|
| **flagship** (depth 2) | **177 / 218** | **17.99** |
| hybrid depth 1, same nets | 176 / 218 | 17.98 |
| standard MCTS, same nets, same budget | 174 / 218 | 19.05 |
| best plain-A\* arm of the loop (v04, seed 8) | 176 / 218 | 20.76 |
| supervised backward (B2 arm, recorded) | 125 / 218 | 22.54 |

Against the matched standard-search control: 47/2 paired move wins
(p = 4e-12; `gate_frontier_d2_vs_stdmcts.json`). And the hybrid at depth 1
against the best plain-A\* arm, on the puzzles both solve: 74/9 (p = 8e-14,
17.48 vs 20.07 moves) — about **two fewer moves per puzzle on the hardest
puzzles in the set**, at the same solve count.

### 8.3 The unseen exam — 200 brand-new puzzles, 50 fresh boards

Nothing in the project has ever trained on these boards.

| planner | solved | mean moves |
|---|---|---|
| **flagship** | **188 / 200** | **13.34** |
| same networks, standard search | 182 / 200 | 14.13 |
| best lab arm without the hybrid (v09, seed 8) | 177 / 200 | 14.32 |
| the frozen networks every arm started from | 175 / 200 | 14.49 |
| supervised backward planner | 134 / 200 | 14.78 |
| **label-free from random initialisation** (v08) | 132 / 200 | 12.97 (on its smaller, easier solved set) |
| supervised forward (move-by-move) planner | 101 / 200 | 7.78 |

And on the 137 of those 200 where perfect play is known — **perfect play needs
8.71 moves on average** (re-verified by recomputation from
`results/variants/exam/g24r4_unseen.dstar.jsonl` against each system's rows):

| system | solved of 137 | % solved perfectly | extra moves vs perfect |
|---|---|---|---|
| **flagship** | **134** | **57%** | **+1.47** |
| same networks, standard search | 132 | 50% | +1.94 |
| the frozen seed networks (A\*) | 131 | 47% | +2.93 |
| label-free from random initialisation | 109 | 45% | +2.89 |
| supervised backward planner | 115 | 36% | +4.84 |
| supervised forward planner | 101 | 90% | +0.12 |

**How to read the last two rows.** The supervised backward planner is the
project's real opponent: the flagship solves **19 more** of these puzzles and,
where both succeed, uses **a third of its excess moves**. The forward
move-by-move planner is a different kind of animal: when it solves a puzzle it
is nearly perfect (+0.12), but it solves only 101 of the 200 — half the exam —
and it burns 688 expansions per puzzle doing it. The project's stated success criterion — "solve at
least as many as the backward-supervised baseline, with fewer moves" — is
**met on brand-new boards, with no human labels anywhere in the loop**. The
stretch half of it — matching the forward planner's *solution lengths* — is not
met, and §2.4 explains why it was never reachable in the pure sub-goal language;
the hybrid closed the gap from about 2.2 moves to **0.87 moves** on the graded
exam and from 2.17 to 1.16 on the unseen one, but did not close it.

*(One caveat recorded in the log: the forward baseline here is the original
24×24 network. A validated re-tune gained the forward planner ~8 frontier
puzzles at a different board configuration, so moves-versus-forward comparisons
should be read as "versus the original network" —
`variants/FINDINGS.md` entry 5 and `FINDINGS.md` §24.)*

### 8.4 Transfer: does any of this survive a change of board?

The hybrid search and the networks it runs on were developed entirely at 24×24
with 4 robots. Run unchanged elsewhere (`variants/FINDINGS.md` entry 19), with
the mature mixed-curriculum networks that had never seen a slide-first search:

| exam | hybrid depth 2 | matched standard-search control | paired move wins |
|---|---|---|---|
| 32×32 graded (175) | **174 / 175**, +1.60 vs perfect | 172, +1.87 | **17 / 0**, p = 1.5e-5 |
| 24×24 8-robot graded (161) | **159 / 161**, +1.24 vs perfect | 159, +1.82 | **23 / 0**, p = 2.4e-7 |
| 24×24 8-robot frontier (289) | **276 / 289** — program record | — | — |

The largest gain is on the crowded 8-robot boards, which is the same pattern the
B2 vocabulary showed one level down: when the board is full of robots, being
allowed to shove one out of the way first is worth the most.

Separately (§4.3), the value network's raw judgement was audited against exact
ground truth at 32, 40, 48, 56 and 64 — sizes far beyond anything it trained on
(`FINDINGS.md` §17) — and it beats the labeller that initialised it at *every*
one of them. Degradation from 32×32 to 64×64 is 0.863 → 0.838, against the
labeller's 0.853 → 0.814: the size-free bet holds well past the curriculum.

---

## 9. What is honestly still unknown

The project's log makes a point of recording what did not work, and the same
discipline applies here.

1. **Realized moves never improved from self-play training alone.** Across five
   B2 iterations and three curriculum iterations, the loop bought solve rate and
   search efficiency; it never bought a statistically significant reduction in
   move count on shared solves. Every move-count gain in this document came from
   the **search** change (v07), not from the training loop.
2. **Iterating a good recipe does not compound.** Chained iterations of v14,
   v12 and the main line all show the same shape: one step, then flat. At this
   loop's maturity, one round of a better recipe is worth more than five rounds
   of the same one — and nobody knows why the ceiling arrives so fast.
3. **The forward planner still wins on solution length.** The gap is down to
   0.87 moves on the graded exam, but it is real, and closing it further
   probably means a deeper hybrid or a learned slide proposer rather than more
   sub-goal training (v15 and v16 both failed at the latter).
4. **The fidelity gauge does not work in the B2 vocabulary.** The instrument
   that was supposed to give early warning of label degradation is only valid in
   the base vocabulary, because exact B2 optima routinely price unplayable
   plans. In B2 the only real instrument is the certified benchmark.
5. **Most results are single-seed on the network side.** The lab replicates its
   claims at two seeds; the main line's curriculum networks are not multi-seed.
   The M1 seed control (`FINDINGS.md` §18) suggests the family's actual seed
   noise is much smaller than the conservative bars used for gating, but that
   was measured at one rung.
6. **Depth-3 hybrid was measured on the graded exam only** and the unseen exam
   had already saturated at depth 2, so the flagship stops at depth 2.

---

### Where the numbers live

| claim type | file |
|---|---|
| the loop's results log | `self_play_robots/FINDINGS.md` |
| the experiment lab's log | `self_play_robots/variants/FINDINGS.md` |
| benchmark payloads (per-puzzle rows) | `self_play_robots/results/**/bench_*.json` |
| language ceilings | `self_play_robots/results/ceiling/*.json` |
| generation manifests | `self_play_robots/results/**/generation*.manifest.json` |
| the unseen exam and its exact optima | `self_play_robots/results/variants/exam/` |
| the code | `self_play_robots/spr/`, `supervised_valuenet/{GridEnv,skeleton,nn_labeler}/` |

*Every solved row in every payload cited above has been replayed move by move
against the physics simulator. That is the one guarantee this project never
compromised.*
