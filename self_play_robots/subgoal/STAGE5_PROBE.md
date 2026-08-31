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

_Appended after the run._
