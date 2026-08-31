# Stage 5 probe: how deep is an optimal plan in the state subgoal space at 24x24?

Written **before** the run it judges, per the plan's rule. Committed with the
job script; the result is appended afterwards and nothing above the result
section is edited.

## Why this and why first

`PLAN_SUBGOAL_DISCOVERY.md` Stage 5 says the search settings must be re-derived
per configuration and that "the open problem is coverage, not quality", with the
gap expected to widen first at 24x24. Stage 3 §(f) located the residual failure
precisely: **depth, not vocabulary**. The learned planner is perfect through
`d*` <= 5 and falls to 46/73 at `d*` = 8.

At 16x16 an optimal solution needs **1 to 5 state subgoals, mean 2.0**
(`subgoal/STAGE01.md`). If that count grows with the board, Stage 5 walks
straight into the regime Stage 3 already measured as the failure mode, and the
settings work has to come before any self-play round is submitted. If it does
not grow, the depth worry is unfounded and Stage 5 is worth its budget.

Nothing here trains or evaluates a network. It is Stage 1's probe, unchanged,
pointed at a different configuration.

## Pre-registered design

- **Instances.** `supervised_valuenet/scaling/data/g24r4/bench.solved.jsonl`,
  the 232-instance graded exam, every instance carrying an exact `d*`.
  **Stride 8, n = 29**: indices 0, 8, 16, ... 224, spanning the whole difficulty
  range. Fixed here; no instance is chosen for its result.
- **Caps.** 900 s and 200,000 pops per instance, whichever binds first. An
  instance that caps is reported as capped and contributes to **no** depth or
  ceiling statistic.
- **Comparison.** The hand-written vocabulary's recorded ceiling at this exact
  configuration, `results/ceiling/g24r4_{base,b2}.json`, joined by instance
  index. Its published summary over all 232 is +1.171 mean extra moves for B2
  over its 228 realizable instances, 62.7% optimal.
- **Certification.** Every solution the probe reports is replayed under the real
  joint-game rules by `space.py::certify` before it counts, as at 16x16.
- **Compute.** CPU only, 16 shards in parallel. One GPU is requested solely
  because the allocation has no CPU partition; no GPU work is done.

## The read-out, fixed in advance

Among instances that resolve within the caps:

| measurement | decision |
|---|---|
| mean subgoals per optimal plan <= 3.5 **and** mean extra moves = 0.000 | the depth worry is unfounded: the vocabulary carries to 24x24 unchanged, and Stage 5's obstacle is search cost, not expressiveness |
| mean subgoals >= 5 | depth grows with the board: fix the beam rule and the admissible bound at 24x24 **before** any self-play round is submitted |
| between 3.5 and 5 | inconclusive on depth; report as such and widen the sample rather than tuning |

**The capped fraction is reported separately and is not evidence about the
vocabulary.** This probe ranks with the trivial any-stop relaxation, the weakest
heuristic in the project, not with the learned cost-to-go that Stages 3 and 4
used. A high capped fraction means this heuristic cannot search 24x24 cheaply,
which is a statement about the heuristic and directly supports the plan's own
Stage 5 item 1.

## Result

**Job 4908943, 2026-08-31, 6.4 minutes wall clock on 16 CPU shards of one GPU
node (the allocation has no CPU partition). Code at `d53588f`, which is the
commit that carries this pre-registration.**

### The read-out, against the rule fixed above

| measurement | value | rule |
|---|---|---|
| mean subgoals per optimal plan | **2.43** | <= 3.5 required |
| mean extra moves over `d*` | **0.000** | = 0.000 required |

**The depth worry is unfounded on this exam.** The state subgoal space still
expresses a certified, exactly-optimal solution at 24x24, and it needs about the
same number of decisions to do it as at 16x16.

| | 16x16 (`STAGE01.md`) | 24x24 (this probe) |
|---|---|---|
| instances | 40 | 29 (28 resolved, 1 capped) |
| mean `d*` | 5.6 | 7.83 |
| extra moves, state space | 0.000 (40/40 optimal) | **0.000 (28/28 optimal)** |
| subgoals per plan, mean / median / range | 2.0 / 2 / 1-5 | **2.43 / 2 / 1-6** |
| primitive moves per subgoal | ~3.1 | **3.13** |
| hand-written B2, extra moves | +1.000 / +0.632 | **+1.276** |
| hand-written base, extra moves | +2.000 / +2.059 | **+2.483** |

Subgoal histogram: 1 -> 5 instances, 2 -> 11, 3 -> 9, 4 -> 2, 6 -> 1.
Soundness: **0 solutions shorter than `d*`**, and every reported solution was
replayed under the real joint-game rules before it counted.

### Three things this says

1. **The abstraction ratio is stable, so depth tracks solution length rather
   than board area.** 3.13 primitive moves per subgoal at 24x24 against about
   3.1 at 16x16. Going from a 16x16 board to a 24x24 board, 2.25x the cells,
   cost 0.43 extra decisions per plan.
2. **The vocabulary gap widened, in the new space's favour.** The hand-written
   B2 language sits at +1.276 extra moves on this sample (its published figure
   over all 232 is +1.171); the state space sits at 0.000. On 4 of the 29 the
   hand-written best is 4 or more moves longer than optimal: `d*` 5 against 9,
   8 against 12, 10 against 15, 11 against 16.
3. **The cost, not the vocabulary, is what grew.** Mean 63.9 s per instance,
   median 37.3 s, max 315 s, mean 35,410 pops, against roughly 10 s per
   instance at 16x16. The one unresolved instance (idx 128, `d*` 14) hit the
   200,000-pop cap at 258 s, well inside the 900 s time cap. As stated above,
   this is a fact about the trivial any-stop relaxation this probe ranks with,
   not about the space: Stages 3 and 4 showed the learned cost-to-go searches
   this space far more cheaply than the relaxation does.

### The caveat that should gate Stage 5, stated as inference

Depth is flat in board size but **not** in difficulty. Bucketing the resolved
instances by `d*`: 0-2 -> 1.00 subgoals (n=2), 3-5 -> 1.60 (n=5), 6-8 -> 2.78
(n=9), 9-11 -> 2.75 (n=12). At the constant 3.13 moves per subgoal, a 24x24
**frontier** instance, where solutions run 17 to 20 moves, would need roughly 5
to 6 subgoals. That is precisely the regime Stage 3 §(f) measured as the failure
mode (46/73 optimal at `d*` = 8, 28/62 at `d*` = 9).

So the recommendation this probe supports is narrow and specific: **the graded
24x24 exam is safe to carry the method to, and the frontier exam is the thing to
probe before assuming it carries there.** Running this same probe on
`scaling/data/g24r4/bench.unsolved.jsonl` would answer it, at a cost this run
puts at a few CPU-minutes per instance.

Payload: `results/subgoal/stage5_probe/merged.json` (29 rows, per-instance
subgoals, move sequences and certification), per-shard payloads and logs beside
it. Nothing here trained or evaluated a network.
