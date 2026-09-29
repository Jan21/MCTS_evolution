# Encode once: a static board encoder plus a cheap per-decision head

Code: `spr/split_value.py`, launcher `jobs/split.slurm` (env: VARIANT, STEPS,
WARM, EXTRA, TAG). Results: `results/subgoal/split/*.json`, logs
`runs/split-<jobname>-<jobid>.out`.

## Why

At 96x96 the sub-goal planner makes about 2000 value calls per instance and
each call runs the full size-free value net: 12 recurrences of one looped
attention layer over 9217 tokens. The board geometry is identical in every one
of those calls; only the roles (robots, plan cells) change. The idea is to run
the geometry part ONCE per board and make the per-decision computation small.

Load-bearing assumption: a per-decision head that sees far fewer tokens than
the full board (or one shallow pass over it) can reach the accuracy of the
monolithic net on the same decisions.

## Variants (same static encoder, same 5-cell readout, differ in the head)

- `sparse`: head is a 2-layer transformer over the 5 readout cells + every
  marked cell (robots, plan cells) + the global row. About 8 tokens.
- `reach`: as sparse, plus every cell one walls-only slide away from any
  marked cell. About 100 tokens at 96x96.
- `inject`: roles are added to every cell's static embedding and ONE masked
  looped-layer step runs over the whole board (9217 tokens, but one pass, not
  twelve).
- `mono`: the harness control. The reference net rebuilt inside the same
  training loop, roles as the only input, looped encoder over all cells.
  It has to reproduce the baseline, or nothing else in the file means anything.

Baseline: the current size-free value net of the mix_b2mix_iter3 pair
(`epoch=0-step=1573.ckpt`), called through the project's own collate, on the
same 318 validation decisions (24x24 and 32x32 self-play records, the splits
that trained the pair).

Kill criterion (pre-registered before the warm-started runs): a variant is
worth integrating into the planner only if its top-1 on these decisions is
within 5 points of the baseline's 77.4%. Below 65% it is dead.

## What went wrong first (both invalid, kept for the record)

1. Cold training, 25k steps: every variant AND the mono control plateaued at
   loss 3.4 while the baseline sits at 2.48 under the identical HL-Gauss loss.
   Cold training of this family collapses to the label marginal; the project's
   FINDINGS 50/57 already said so and its own loop always warm-starts.
   Numbers (top-1): sparse 49.1, reach 36.8, inject 49.1, mono ~50. Invalid.
2. Warm start + rank loss, 25k steps at lr 3e-4, but `--static-rec 4`:
   mono 64.8, sparse 45.3, reach 48.4, inject 51.6. Invalid: the baseline
   checkpoint's hparams say `recurrence: 12`. The "control" was the baseline's
   weights run at a third of their depth.

## Step-0 check (2026-09-11, jobs 5032759 / 5032760): harness validated

Warm-started mono, no training, on the 318 decisions:

| recurrences | MAE  | Spearman | top-1 |
|-------------|------|----------|-------|
| 12          | 3.43 | 0.659    | 77.4% |
| 4           | 6.90 | 0.415    | 59.7% |
| baseline    | 3.45 | 0.662    | 77.4% |

At 12 recurrences the control IS the baseline (the residual is the 32-candidate
cap per decision). At 4 it loses 18 points, which is the whole earlier gap.
The forward path, the role features, the readout and the metric code are
therefore correct. Everything before this table is superseded.

Per-decision cost of the full net at batch 8 (this is what the variants have
to beat): 96x96 1.80 s, 64x64 0.38 s, 32x32 0.04 s. Earlier head-only
measurements, same batch: sparse 0.05 s, reach 0.05 s, inject 0.20 s at 96x96.

## Running (2026-09-11)

- mono, rec 12, warm, 25k steps, lr 3e-4 (5032761) and lr 1e-4 (5032762):
  does training in this loop keep or improve the baseline?
- sparse / reach / inject, static encoder rec 12, warm, 25k steps, lr 1e-4
  (5032764-66): the real test of the three heads.

## Where the network time goes in the planner (from results/subgoal/sgnet)

Per instance, `nn_policy_calls == nn_value_calls`: every expansion makes ONE
policy pass over the board and ONE value pass over the k=5 (about 4-5)
candidates. So one expansion costs about 6 full-board forwards; at 96x96 that
is 1.3-1.5 s per expansion, 300-400 expansions on the instances that need
search, about 600 s per instance.

Consequences for the design:

- A value-only split (this file's variants) removes at most 5 of the 6
  forwards per expansion: about 6x on network time, not 30x.
- The policy makes the same full pass and reads the same board. The static
  encoder should be SHARED: one geometry encoding per board, a value head
  and a policy head on top. The policy head is a different readout (per-cell
  logits over the marked candidates) but the same token selection applies.
- Integration point is `spr/search.py: Evaluator` (value_ctg, policy_logp),
  which already keeps a per-board adjacency cache; a per-board static
  embedding cache goes next to it. Nothing else in the planner changes.

## Result: the three per-board variants (2026-09-11, jobs 5032764-66)

Static encoder at 12 recurrences, warm-started, 25k steps, lr 1e-4, rank +
HL-Gauss loss, same 318 decisions. Kill line was 65% top-1.

| variant | tokens in head (96x96) | MAE  | Spearman | top-1 |
|---------|------------------------|------|----------|-------|
| sparse  | ~8                     | 5.36 | 0.367    | 48.7% |
| reach   | ~100                   | 5.36 | 0.339    | 46.5% |
| inject  | 9217, one masked step  | 5.12 | 0.370    | 48.1% |
| baseline| 9217, twelve steps     | 3.45 | 0.662    | 77.4% |

All three dead, and the training losses (about 4.0 against 2.7 for the control
at the same learning rate) show it is a capacity or information gap, not an
optimisation accident. A geometry embedding that has never seen where the
robots are does not let a small head reconstruct the value; the reference net
propagates robot and plan information through twelve masked hops, and the
earlier "17% far-cell shift" finding said the same thing from the other side.

## Next probe: encode once PER DECISION (dec0 / dec1)

Observation from the corpus and the planner: within one decision every
candidate shares channels 0,1,5,6,7,8 of the value input (segment endpoints,
all helper robots, plan context) and differs only in 2,3,4 (its bottleneck,
support, helper). The policy's own input is that shared context. So the
expensive twelve-hop encoding can be done once per decision, with the robots
in it, and each candidate is a readout on top.

- `dec0`: shared context through the (warm-started) looped encoder, 12
  recurrences, once per decision; each candidate = the 5-cell readout on
  those embeddings at its own cells. Head cost per candidate: an MLP.
- `dec1`: as dec0, then each candidate adds its 3 channels and runs ONE
  masked looped step before the readout.

Cost per expansion at 96x96 if it works: 1 full pass (about 0.22 s) + R
cheap heads instead of R full passes (about 1.1 s for R=5); and the policy
pass could share the same encoding, which brings the per-expansion network
cost to one pass instead of six.

Kill line: top-1 within 5 points of 77.4% (that is, at least 72%) on the
same 318 decisions. Below 65%: dead, and then the remaining option is the
dynamic-depth sweep (how many of the twelve hops need the candidate).
Jobs: step-0 5032777 (dec0) / 5032779 (dec1); 25k steps 5032778 / 5032780.

## Result: dec0 (2026-09-11, job 5032778)

| variant | per-decision cost, 8 candidates, 96x96 | MAE  | Spearman | top-1 |
|---------|----------------------------------------|------|----------|-------|
| dec0    | 0.28 s (one shared pass + MLP)          | 4.18 | 0.547    | 63.2% |
| baseline| 1.80 s                                  | 3.45 | 0.662    | 77.4% |

Step 0 (warm-started, untrained readout on candidate-agnostic embeddings) was
38.7%. So the per-decision shared encoding carries most of the signal, and
the readout alone recovers 63 of the 77 points at 6.4x lower cost. Not enough
to integrate (line is 72%), but it changes the picture: the per-board
variants were dead at 47-49%, this one is a capacity question.

## Pre-registered follow-up: how many candidate-aware hops does the head need?

`dec1 --dyn-steps k`: after the shared 12-hop encoding, each candidate adds
its 3 channels and runs k masked looped steps (warm-started layer weights,
one shared copy). k=1 is job 5032780; k=2 is 5032784; k=4 is 5032785.
Cost per candidate grows linearly in k (about 0.02 s per hop per candidate at
96x96, so k=4 is roughly 0.4 s + 0.28 s per 8 candidates, still 2.7x under
the full net). Read the curve: if 72% needs k >= 6 the saving is gone and the
idea is dead; if k <= 2 reaches it, integrate.

## Controls after training (jobs 5032761 / 5032762): the loop is validated

mono, 12 recurrences, warm, 25k steps: lr 3e-4 -> 78.6% top-1 (MAE 3.55,
Spearman 0.678); lr 1e-4 -> 78.3% (MAE 3.46, Spearman 0.626). Baseline 77.4%.
The harness neither loses nor invents accuracy; every variant number in this
file after the step-0 table is a real comparison.

## dec1, one candidate-aware hop (job 5032780): 61.9%

MAE 4.14, Spearman 0.493, cost 0.43 s per 8 candidates at 96x96. Level with
dec0 (63.2%), so one hop of candidate-to-context propagation adds nothing.
The 2-hop and 4-hop runs (5032784 / 5032785) decide whether depth matters.

## Pre-registered: dec0f (job 5032786), candidate geometry as features

dec0 plus a side path fed with the exact directed slide-graph distances among
the candidate's five readout cells (20 ordered pairs, BFS horizon 8, computed
on the GPU in 8 masked matmuls per decision, `slide_pair_features`). Last
layer zero-initialised, so step 0 is exactly dec0. The hypothesis: what the
readout lacks is not more propagation but the pairwise relation between the
candidate's own cells (can the helper reach the support, does the bottleneck
lead to the segment end), which the twelve-hop baseline computes implicitly.
Same line: 72% to integrate, below 65% dead.

## dec0f (job 5032786): 62.9%

MAE 4.21, Spearman 0.506, cost unchanged (0.29 s per 8 candidates at 96x96).
Exact pairwise slide distances among the candidate's cells, added to the
logits through a side MLP, buy nothing over dec0 (63.2%). Three per-decision
designs (dec0, dec1 one hop, dec0f) now sit at 62-63%.

## Pre-registered: dec0g (job 5032799), the same distances INSIDE the readout

The side path could only shift logits; it could not condition the readout's
reading of the five embeddings on the distances. dec0g widens the readout's
first layer (5d+40 -> d), copies the baseline's weights into the first 5d
columns and zero-initialises the rest, so step 0 is dec0 and the distances
interact with the embeddings from the first nonlinearity on. If this is also
62-63%, the per-decision family is capped by what the readout can recover
from candidate-agnostic embeddings, and only candidate-aware propagation
(the k-hop sweep) remains.

## dec1 two hops (job 5032784): 55.3%

MAE 4.21, Spearman 0.481, cost 0.58 s per 8 candidates at 96x96. Worse than
one hop (61.9%) and than no hop (63.2%). Candidate-aware hops on top of the
shared encoding hurt at this data size; the four-hop run (5032785) is the
last point of that curve.

## A data gap in the harness, and the control for it (job 5032802)

The baseline pair's own DATA.json: it trained on THREE buffers, g24r4 (1013
train decisions), g32r4 (1025) and g24r8 (1176, eight robots), with no anchor
corpora (the B2 recipe sets ANCHOR=0). This harness loaded only the first two:
2038 train decisions against the baseline's 3214. The mono control cannot
reveal that (it starts trained), but every dec variant has to learn a new
function from the data it sees. `--extra-train g24r8=...` adds the third
buffer to training only; validation stays the same 318 decisions.
dec0x8 = dec0 + g24r8, 35k steps, lr 1e-4. If it stays at 63%, the per-
decision family is capped by architecture, not data.

## dec1 four hops (job 5032785): 66.4%, and the hop curve

| candidate-aware hops | top-1 | MAE  | cost per 8 candidates, 96x96 |
|----------------------|-------|------|------------------------------|
| 0 (dec0)             | 63.2% | 4.18 | 0.28 s                       |
| 1                    | 61.9% | 4.14 | 0.43 s                       |
| 2                    | 55.3% | 4.21 | 0.58 s                       |
| 4                    | 66.4% | 3.90 | 0.87 s                       |
| 12 (baseline)        | 77.4% | 3.45 | 1.80 s                       |

The standard error on 318 decisions is about 2.7 points, so the curve is
flat within noise from 0 to 4 hops. Extrapolating, the 72% line would need
8 or more hops, at which point the per-candidate cost is most of the full
net and the saving is gone. Depth is not the lever.

## dec0g (job 5032799): 59.4%

MAE 4.08, Spearman 0.502. The distances inside the readout do no better than
outside it (dec0f 62.9%, dec0 63.2%; all within noise of each other). The
readout, however it is fed, recovers about 60-66% from candidate-agnostic
embeddings. Six per-decision designs, one answer.

## Pre-registered: distillation (job 5032830)

Every student so far learned from the same noisy self-play cost labels the
baseline learned from, one label per candidate. The baseline itself can label
every candidate with a full distribution over cost bins. dec0d = dec0 + the
g24r8 buffer + KL(student || baseline logits) with weight 1 on every
candidate, 35k steps, lr 1e-4. Cost is unchanged. This is the standard way a
cheap student closes on a teacher; if it does not move the number, the
per-decision family is closed: the value of a candidate is not a function of
context embeddings read at the candidate's cells.

## dec0 with the baseline's full training data (job 5032802): 65.1%

MAE 3.98, Spearman 0.554, 35k steps. Adding the g24r8 buffer (3214 train
decisions, the baseline's own set) lifts dec0 by two points, inside the
noise band. Data is not what separates 63% from 77%.

## dec0 distilled (job 5032830): 67.9%

MAE 3.97, Spearman 0.565. The teacher's soft targets add three points over
the same run without them (65.1%). Best per-decision number so far, cost
unchanged. Levers stack slowly: 63.2 (dec0) -> 65.1 (+g24r8) -> 67.9
(+distillation). Still four points under the 72% line.

## Pre-registered: the planner-level test (jobs 5032885-5032897)

Top-1 on 318 decisions is a proxy. The question the paper asks is solve
rate against wall clock at scale, so the 67.9% distilled dec0 goes into the
sub-goal search itself: `spr.nets.SplitValue` behind the value net's
positional signature (bit-exact against the harness on real decisions),
`spr.bench --arch split`, the full-size policy unchanged. Same protocol as
the ladder's network rows (1200 expansions, k=5, anytime, B2, lean boards):
64x64 on the first 40 ladder instances, 96x96 on the same 24 instances the
full pair ran (12 shards of 2).

Read against the full pair on the same instances (64: 87.5% at 40 s;
96: 75.0% at 324 s): the split value wins if it solves at least as many at
clearly lower time per instance, or more per unit time. If it loses solves
at 64 and gains nothing at 96, the encode-once thread closes with the
harness numbers as its result and the cost/accuracy curve as the finding.

## Planner-level result at 96x96 (jobs 5032886-5032897): same solves, 1.8x faster

Same 24 instances, same protocol, distilled dec0 (67.9% top-1) as the value
net, full policy unchanged. `subgoal/cmp96.py` reproduces this:

| value net        | solved | s per instance | total s | expansions |
|------------------|--------|----------------|---------|------------|
| full pair        | 18/24  | 324            | 7774    | 313        |
| split (dec0)     | 18/24  | 179            | 4285    | 320        |

Per-instance agreement on solved: 24/24 (the same 18 solved, the same 6
unsolved; all 6 hit the 1200-expansion cap in both arms). On those 6 the
split arm spends 658 s against 1228 s for the same 1200 expansions: 0.55 s
per expansion instead of 1.02 s. Easy instances: 18.6 s against 22.6 s, fixed
costs dominate there.

Reading: at this size the search outcome is insensitive to the value net's
top-1 dropping from 77% to 68% (on this sample), while the value pass went
from most of the expansion cost to a rounding error. What is left per
expansion is one full policy pass, one full shared-context pass for the
value (the same context the policy reads), and about 0.3 s of non-network
work (candidate generation and ranking, physics). Sharing ONE encoding
between the policy head and the value readouts would remove one of the two
remaining passes; that is a joint-model build (policy recipe, byref logp
map) and the next step if the thread continues.

The pre-registered 72% top-1 line was a proxy; the planner-level criterion
(no lost solves, clearly lower time) is met.

## Planner-level result at 64x64 (job 5032885): same solves, 1.6x faster

First 40 ladder instances: full pair 35/40 at 40.3 s per instance, split
value 35/40 at 25.7 s, per-instance agreement 40/40. On the 6 instances the
full pair needed more than 60 s: 254 s -> 159 s for the same expansions.

Both sizes now say the same thing: the cheaper value changes no outcome on
64 instances and takes 1.6-1.8x less wall clock per instance; the whole
saving comes from the value pass, the policy pass and the CPU-side work are
untouched.

## Longer and heavier distillation (jobs 5032882 / 5032883): no gain

| run                        | top-1 | MAE  | Spearman |
|----------------------------|-------|------|----------|
| distill 1.0, 35k steps     | 67.9% | 3.97 | 0.565    |
| distill 1.0, 55k steps     | 64.8% | 4.02 | 0.542    |
| distill 3.0, 55k steps     | 67.6% | 3.91 | 0.577    |

All within the noise band of one another (standard error about 2.7 points).
The per-decision student plateaus at 65-68% top-1 on this corpus; neither
teacher weight nor training length moves it. The 110k-step run (5032884) is
the last point. The checkpoint used in the planner stays the 35k one.

## 110k-step distillation (job 5032884): 65.1%

MAE 4.08, Spearman 0.541. The plateau is confirmed: 64.8 / 65.1 / 67.6 /
67.9 across 35k to 110k steps and teacher weights 1 to 3. Explainer for the
whole thread: artifact "One Look Per Decision".
