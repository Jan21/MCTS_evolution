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

**Job 4910255, 2026-09-01. The run did NOT answer its primary question, for two
separate reasons, one of them my error. Reported here rather than re-run and
quietly replaced.**

### What went wrong

**12 of the 16 shards were killed by the out-of-memory reaper** (`sacct` state
`OUT_OF_MEMORY`, 12 `oom_kill` events). The cause is in the pre-registered design
above: I raised `max_pops` from the graded run's 200,000 to 500,000 while keeping
16 concurrent shards. The search's `best_g` and `parent` dictionaries grow with
pops, so per-shard memory grew by the same factor and 16 copies exceeded the
job's allowance. The graded run was safe at 200,000; this one was not, and I did
not check it before submitting.

**4 shards survived and covered 8 of the 32 pre-registered instances.**

### What the 8 completed instances say

| measurement | value |
|---|---|
| instances resolved | **0 of 8** |
| instances capped | 8 of 8 |
| mean seconds | 879.6 (median 900.0, the cap) |
| mean nodes popped | 412,752 |
| subgoal depth | **not measurable, nothing resolved** |
| new optima established | 0 |
| falsification test | could not run: no instance where both vocabularies have an answer |

Six hit the 900-second time cap; two hit the 500,000-pop cap first, at 779 s and
882 s. Not one produced a certified solution.

### The primary question is still open, and the secondary one is answered

**Open: how deep is an optimal frontier plan.** Nothing resolved, so there is no
subgoal count. The extrapolation from the graded probe (roughly 5 to 6 subgoals)
remains an extrapolation.

**Answered: whether this space is searchable at the frontier with a weak
heuristic. It is not.** Eight instances, 400,000 nodes each, fifteen minutes
each, zero solutions. Set against the graded exam, where the median instance
resolved in 37 seconds, the difficulty cliff between the two exams is steep.

The comparison that matters is with the OLD vocabulary on the same exam. Its own
exhaustive probe (`results/ceiling/g24r4_frontier_b2.json`, job 4685187)
resolved **114 of 218** frontier instances with a 120-second cap and 200,000
iterations, and finished all 218 in 481 seconds of 8-worker wall clock. On this
exam the hand-written vocabulary is cheap to search and the complete one is not.

**So at the frontier the trade reverses.** On the graded exam the complete space
costs about 6x more search and returns perfect solutions. On the frontier, with
this heuristic, it returns nothing at all while the lossy vocabulary keeps
working. Expressiveness bought at this price is not usable, and the binding
constraint is the heuristic, not the action space: FINDINGS §32(e) already
measured that substituting the exact engine's cost-to-go collapses the same
search from 1200 expansions to 20.

### What should happen instead of re-running this

Re-running the same design with fewer shards would cost another node-hour and,
on the evidence of the 8 completed instances, would most likely return the same
"0 resolved". Two better instruments, both cheaper than a corrected re-run:

1. **Measure depth without searching for it.** Any certified move sequence
   decomposes into macro-moves by grouping consecutive moves of the same robot,
   so an existing good frontier solution gives its subgoal depth for free. This
   needs a planner run that emits move dumps: I checked
   `results/variants/v07_hybrid_actions/bench_frontier_*.json` and
   `results/selfplay/*/bench_g24r4_frontier_astar.json` and **none of the stored
   24x24 frontier payloads carry a move list**, only aggregate counts.
2. **Give the probe a real heuristic.** The point of the frontier probe was
   depth, not cost, and the trivial relaxation is the wrong instrument for a
   frontier search. Running it with the learned cost-to-go, or with the exact
   engine as a diagnostic ceiling, would resolve instances and yield the depth
   distribution.

Payload: `results/subgoal/stage5_frontier/merged.json`, 8 rows, plus the 4
surviving shard logs. The 12 killed shards left no payload.
