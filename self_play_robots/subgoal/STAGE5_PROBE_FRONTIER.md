# Stage 5 probe, part 2: how deep is an optimal plan on the 24x24 FRONTIER?

Written **before** the run it judges. Committed with the job script; the result
is appended afterwards and nothing above the result section is edited.

## Why

The graded probe (`STAGE5_PROBE.md`, job 4908943) passed its gate: at 24x24 the
state subgoal space still expresses exactly-optimal solutions, in 2.43 subgoals
on average, at 3.13 primitive moves per subgoal against about 3.1 at 16x16. It
closed with one inference, flagged as inference: **depth is flat in board size
but not in difficulty.** Bucketed by optimum, subgoal count ran 1.00 / 1.60 /
2.78 / 2.75 across `d*` 0-2 / 3-5 / 6-8 / 9-11. Extrapolated at the constant
3.13 ratio, a frontier instance whose solution runs 17 to 20 moves would need
roughly 5 to 6 subgoals, which is the regime Stage 3 §(f) measured as the
failure mode.

This probe tests that extrapolation directly instead of leaving it as an
inference. It is the same probe, same code, pointed at the frontier exam.

## What the frontier exam is

`supervised_valuenet/scaling/data/g24r4/bench.unsolved.jsonl`, 218 instances on
the same boards, every one carrying `d_star: None`: the project's exact Rust
solver **failed to solve them within its budget**. So there is no ground truth
to score against, and two consequences follow.

1. The usual "extra moves over optimal" metric does not exist here. The
   measurements are the subgoal count and the absolute move count.
2. **Any instance this probe resolves yields a proven optimum the project does
   not currently have.** The search is A* over joint states with true slide
   counts as edge costs and the any-stop relaxation as `h`, which is admissible
   and consistent, so the first goal state popped is optimal in the space. The
   graded run supports the stronger reading that optimal-in-the-space equals
   optimal-in-the-game: on 28 of 28 resolved instances it returned exactly `d*`,
   never above it, and never below it.

## Pre-registered design

- **Instances.** Stride 7, n = 32: indices 0, 7, 14, ... 217, spanning the whole
  file. Fixed here; no instance is chosen for its result.
- **Caps.** 900 s and **500,000 pops**, whichever binds first. The graded run
  used 200,000 pops, which bound at 258 s and left 642 s of the allowed time
  unspent; here the pop cap is deliberately moved out of the way so wall clock
  is the single binding limit and as many instances resolve as the time allows.
  This cap difference is stated wherever the two runs are compared.
- **Comparison.** `results/ceiling/g24r4_frontier_{base,b2}.json`, the
  hand-written language's recorded ceiling on this same exam, joined by index.
  Its own summary: 114 of 218 realizable, 104 inconclusive, 115 capped.
- **Certification.** Every reported solution is replayed under the real
  joint-game rules before it counts.
- **Compute.** CPU only, 16 parallel shards; the GPU is requested only because
  the allocation has no CPU partition.

## The read-out, fixed in advance

Primary, among instances that resolve:

| mean subgoals per optimal plan | reading |
|---|---|
| <= 3.5 | the extrapolation was wrong; depth stays shallow even on hard instances and Stage 5 has no depth problem |
| 3.5 to 5 | as predicted; the beam rule and the admissible bound must be re-derived at 24x24 before any self-play round is submitted |
| >= 5 | the extrapolation holds and understates; depth is the binding constraint at the frontier and the settings work is mandatory, not optional |

**Falsification test, run on the same rows.** Wherever the hand-written ceiling
payload records a `best_realizable_moves` for an instance this probe resolves,
the probe's optimum must be **less than or equal to** it. The hand-written
language is a subset of what the state space can express, so a single instance
where the probe's answer is longer falsifies the completeness assumption the
optimality claim rests on, and the claim is withdrawn rather than argued.

**The resolved fraction is a statement about the heuristic, not the space.**
This probe ranks with the trivial any-stop relaxation, the weakest heuristic in
the project. Stages 3 and 4 showed the learned cost-to-go searches this space
far more cheaply. A low resolved fraction is evidence for the plan's own Stage 5
item 1, that settings must be re-derived per configuration, and is not evidence
about expressiveness.

## Result

_Appended after the run._
