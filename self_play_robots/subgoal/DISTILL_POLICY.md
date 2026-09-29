# Replicating the proposer with a cheaper student (distillation harness)

Code: `spr/distill_policy.py`, launcher `jobs/distill.slurm` (env VARIANT
"s,k", STEPS, TAG, EXTRA). Results `results/subgoal/distill/policy_<s>+<k>*.json`.
Started 2026-09-12 on the user's proposal.

## The idea

The teacher (size-free policy net, 12 weight-tied rounds over the board with
7 plan marks per cell, candidate log-probs read off the cell embeddings)
costs 233 ms per decision at 96x96, all of it the 12 rounds
(`SEARCH_SPEED.md`). Its behaviour is free to generate in any quantity, so a
cheaper student can be judged on "ranks candidates like the teacher" in
minutes, before any search run.

Student shape: `s` STATIC rounds on the walls alone (6 geometry channels),
once per board, then the plan marks fused in (added to the static embeddings)
and `k` DYNAMIC rounds per decision. Weight-tied layer warm-started from the
teacher for both halves (`--untied` gives the dynamic half its own copy).
Readout heads start as the teacher's. Per-decision cost = k rounds
(19 ms each at 96x96); the static half is amortised over the ~300 decisions
of a hard puzzle.

## Supervision (online, no stored targets)

Per decision: squared error on the final cell embeddings (9217 x 192
targets at 96x96) + KL(teacher || student) over the enumerated candidates.
Data: decisions from the self-play buffers at 24, 32 (mix_b2mix_iter3), 64
(g64r4_b2_iter1) plus g24r8 as train-only; validation 400 held-out decisions
by the buffers' board splits.

## Pre-registered

Metric that matters: top-5 overlap with the teacher (the search keeps five),
then top-1 agreement, Spearman, KL. Lines: integrate into the search only if
top-5 overlap >= 0.9 (4.5 of 5); below 0.8 the split is dead.

Variants (jobs 5036631-5036637, 25k steps, lr 1e-4, one decision per step):

| variant | static s | dynamic k | role | cost per decision at 96 |
|---------|----------|-----------|------|-------------------------|
| 0+12 step 0 | 0 | 12 | must be 100% / KL 0: validates the readout replica | 233 ms |
| 0+12 trained | 0 | 12 | does training in this loop preserve the teacher | 233 ms |
| 6+6 | 6 | 6 | the proposal | ~117 ms |
| 8+4 | 8 | 4 | cheaper | ~78 ms |
| 4+8 | 4 | 8 | dearer | ~156 ms |
| 0+6 | 0 | 6 | baseline: is the static half doing anything | ~117 ms |
| 6+6 untied | 6 | 6 | does the dynamic half want its own weights | ~117 ms |

If 6+6 passes: (a) plug the student into the search behind the proposer
interface (static cache per board in `Evaluator`), same puzzle-for-puzzle
check as the value net; (b) the same split for the value net's shared pass,
sharing the static half; (c) extra dynamic rounds per chosen candidate for
the value readout, as the user suggested.

## Results (2026-09-12, jobs 5036660-5036665; 6+6 untied pending)

| student | top-1 agree | top-5 overlap | Spearman | KL | emb mse | ms/decision at 96 |
|---------|-------------|---------------|----------|-----|---------|-------------------|
| 0+12 step 0 (teacher) | 100.0% | 1.000 | 1.000 | 0.000 | 0.000 | 232 |
| 0+12 trained | 91.0% | 0.982 | 0.969 | 0.008 | 0.001 | 230 |
| 4+8 | 81.7% | 0.958 | 0.858 | 0.051 | 0.010 | 154 |
| 6+6 | 69.6% | 0.929 | 0.747 | 0.131 | 0.031 | 116 |
| 0+6 | 73.1% | 0.930 | 0.769 | 0.133 | 0.028 | 115 |
| 8+4 | 57.3% | 0.910 | 0.654 | 0.232 | 0.061 | 77 |

Reading: 6+6 == 0+6 within noise; the static rounds on the walls contribute
nothing. Quality is a function of the number of rounds that see the plan:
4 -> 57%, 6 -> 70-73%, 8 -> 82%, 12 -> 91% top-1 (the trained 0+12 control
drifts from 100% to 91%, which bounds what the loop holds). Same lesson as
the value net's per-board attempt: the work is propagating the plan through
the walls, and it has to start from the plan.

What it buys: a plain 6-round student keeps 0.93 of the teacher's top-5 sets
at half the cost; the teacher's own weights truncated to 4 rounds were far
worse for the value net (60%). Next: (1) the 0+6 student inside the search
(tag `_p6`) against `_fast2`; (2) 0+4 and 0+8 for the plain round curve;
(3) the same distillation for the value net's shared pass at 6 rounds.

## The 6-round student inside the search (jobs 5036707-5036719, tag `_p6`)

Same fast configuration as `_fast2` (split value, wrappers, axial), proposer
replaced by `policy_0+6.pt` via `spr.nets.load_policy_student`:

| size | n  | teacher (12 rounds) | student (6 rounds) | gained | lost | s/inst |
|------|----|---------------------|--------------------|--------|------|--------|
| 64   | 40 | 35                  | 38                 | idx 4, 14, 28 | none | 12.4 -> 7.6 |
| 96   | 24 | 18                  | 18                 | none   | none | 41 -> 37 |

The search path differs (93% top-5 agreement is not 100%), so outcomes can
move either way; here they moved up at 64 (three budget-exhausted puzzles
solved, two of them in under 10 s) and not at all at 96. Untied 6+6: 70.4% /
0.927, same as tied. Pending: 0+4, 0+8 (the plain round curve), and the
6-round value student (`split-sw-dec0d6`).

## Round curve without static parts, and the value net at 6 rounds

| rounds | proposer top-1 / top-5 | ms at 96 |
|--------|------------------------|----------|
| 4 (0+4) | 54.8% / 0.889 | 77 |
| 6 (0+6) | 73.1% / 0.930 | 115 |
| 8 (0+8) | 77.9% / 0.945 | 154 |
| 12 (0+12 trained) | 91.0% / 0.982 | 230 |

(4+8 gave 81.7% / 0.958, 0+8 77.9% / 0.945: within about one standard
error; 8+4 and 0+4 agree at 55-57%.)

Value net, per-decision shared pass at 6 rounds, distilled with the g24r8
buffer (job 5036706, `dec0-sw-r6-lr1e-4-x8-distill`): 59.7% top-1, MAE 4.42,
Spearman 0.511 (12 rounds: 67.9%). Search-level check with both 6-round
students: tag `_p6v6`.

## Both 6-round students in the search (jobs 5036753-5036765, tag `_p6v6`)

6-round proposer student + 6-round per-decision value student (59.7% top-1
in the harness), axial attention, fixed wrappers:

| size | n  | solved | s/inst | (12-round pair, same config: solved, s/inst) |
|------|----|--------|--------|----------------------------------------------|
| 64   | 40 | 38     | 6      | 35, 12                                       |
| 96   | 24 | 18     | 29     | 18, 41                                       |

Identical outcomes to the 6-round-proposer-only run (`_p6`); the weaker
value student (59.7% vs 67.9% top-1) changed no outcome on 64 instances.
Whole chain from the scaling table's `net` column: 324 -> 29 s per instance
at 96x96 (11x), 40 -> 6 s at 64x64, three instances gained at 64, none lost.
