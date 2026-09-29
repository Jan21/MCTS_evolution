# Journal: making the guided sub-goal search fast, and what it showed

Branch `probe/stage5-depth-24x24`, clone `/scratch/project/open-37-42/janhula/mcts_stage5_probe` on Karolina.
Every entry names the files that hold the evidence. Nothing here is committed or pushed yet.
Companion notes with the full tables: `subgoal/SCALING_LADDER.md`, `subgoal/SEARCH_SPEED.md`,
`subgoal/DISTILL_POLICY.md`, `subgoal/ENCODE_ONCE.md`. Explainers written for a reader outside
the project: "One Look Per Decision" and "One Search Iteration" (artifact links in the memory
file `artifact-series-mcts`).

---

## 2026-09-10 · Where the time goes

**Question.** The network-guided sub-goal search solved 75% of the 96x96 ladder puzzles but took
324 s per puzzle, five minutes each, which made anything beyond 24 puzzles at that size
unaffordable. Where does the time go?

**Method.** Per-instance accounting written by `spr/bench.py` into every result row
(`results/subgoal/sgnet/*.json`), then `cProfile` around one hard 96x96 puzzle for 300
expansions (`jobs/profile96.slurm`).

**Findings.**
- One expansion (one step of the search) costs 1.66 s. The value call is 1.56 s of that (94%),
  the proposer call 0.30 s, everything the processor does 0.07 s. The networks are the cost; the
  recipe logic, physics checks and bookkeeping are noise.
- `nn_policy_calls == nn_value_calls`: every expansion makes one proposer pass over the board and
  one value pass over the k=5 kept candidates, each candidate a full pass. So one expansion is
  about six twelve-round passes over 9,217 cells.
- 95% of all benchmark time at 96x96 goes to the six puzzles that exhaust the 1,200-expansion
  budget without a solution. Solved puzzles finish in about 18 expansions.
- The search loop the benchmark actually runs is `eval/compare.py::_nn_astar_backward`, not
  `spr/search.py` (that one is the `spr_astar` option). Worth knowing before touching anything.

---

## 2026-09-11 · Step 1: score candidates from one shared reading

**Idea (Jan's).** Encode the board once, do only small per-decision work.

**What died.** Encoding the walls once per board (a static geometry encoder, twelve rounds) and
giving a small head the robots and the candidate: three head shapes (8 cells, 100 cells, one
masked round over all cells) all landed at 46-49% top-1 ranking accuracy against 77.4% for the
original value net. The expensive work is propagating where the robots are through the wall
structure, and a walls-only encoding leaves all of it to the head. Two earlier result rounds
were invalid (cold training collapses this family; the control ran 4 recurrences instead of the
checkpoint's 12) and are recorded as such in `ENCODE_ONCE.md`.

**What worked.** Within one decision every candidate shares every input channel except its own
three (stop cell, park cell, helper). So the twelve rounds run once per decision on the shared
picture, with the robots in it, and each candidate is a readout of its five cells. Plain: 63.2%
top-1; with the eight-robot training buffer the original also used: 65.1%; distilled from the
original's outputs: 67.9%. Depth (1, 2, 4 candidate-aware hops), pairwise slide-distance
features (two ways), longer training and heavier distillation all stayed on that plateau.

**The test that mattered.** Inside the real search (`spr/nets.py: SplitValue`, `spr.bench --arch
split`), same puzzles: 64x64 35/40 in both arms at 25.7 s instead of 40.3 s per puzzle; 96x96
18/24 in both arms at 179 s instead of 324 s; per-puzzle agreement 24/24 and 40/40. The search
keeps five candidates and closes most lines for free once pointed the right way; it does not
need the last ten points of ranking accuracy.

Files: `spr/split_value.py` (harness), `results/subgoal/split/`, checkpoint used:
`dec0-sw-r12-lr1e-4-x8-distill.pt`.

---

## 2026-09-11 · Step 2: two wrapper costs that were not the networks (exact)

The profile of the search's call wrappers (`eval/end2end.py`) showed:
- `_value_cost` expanded the two 9,218 x 9,218 adjacency tables to the candidate batch on the
  CPU and then copied the expanded tensor to the card: 1.7 GB per value call at 96x96.
  `_policy_logp` copied the unexpanded tables every call.
- `_policy_logp` scored candidates in a Python triple loop with one small product and one host
  sync per candidate, about 200 per expansion, plus a linear scan per pair.

Fix: adjacency tables cached on the device (`_adj_dev`), expansion is a view; the readout's
three chained softmaxes computed for every (stop, park) pair in batched products with one
transfer back. Equality check against the original loop: keys identical, max difference 2.4e-6.
Same instance, same 300 expansions, identical path: 1.66 s -> 1.19 s per expansion with the
original value net, 0.85 s -> 0.42 s with the per-decision one. Original kept as
`eval/end2end.py.orig`.

---

## 2026-09-12 · Step 3: axial attention for the masked families (exact)

A stage-by-stage profile of one real proposer call (`subgoal/prof_policy.py`, 180 candidates,
card-synchronised at every boundary): 225 of 240 ms in the three attention families of the
twelve rounds, 19 ms per round; readout over all candidates 7 ms; features, meta, transfers,
norms, feed-forward together under 16 ms. Kernel level: the fused attention kernel 183 ms
(fp32; half precision had been rejected earlier on accuracy), mask preparation for the two
masked families 50 ms, all linear layers 12 ms. Batching boards per call is not a lever at 96x96:
one board saturates the card (231 ms per board alone, 219 at batch 8).

Every slide edge lies in a row or a column, so the two masked families attend over about 190
keys per cell, not 9,217. The existing `axial_head` (row pass + column pass merged exactly by
log-sum-exp, global token dense) was only 2x because it materialised a [n, n, n, heads, dk]
tensor. `axial_head_fast` computes the same scores as batched matrix products (`RR_AXIAL=1`,
`nn_labeler/model.py`, masks cached per board, gated to n >= 48 where it pays).

Exactness and speed of a full twelve-round pass: 24x24 max diff 4e-6, 9.6 -> 38 ms (slower);
64x64 7e-6, 58 -> 38 ms; 96x96 6e-6, 231 -> 93 ms. In the search, same instance, same 300
expansions: 1.19 -> 0.50 s per expansion with the original pair, 0.42 -> 0.21 s with the
per-decision value. What remains of a pass at 96 is the one unmasked all-to-all family (55 ms).

Bug found later the same day and fixed: the mask cache kept a reference to the adjacency
tensor it was called with; in training every step brings fresh copies to the card, so the cache
pinned them until the card was full. Now no caching while gradients are enabled.

---

## 2026-09-12 · Step 4: six-round students taught by the twelve-round networks

**Idea (Jan's).** Collect the input-output behaviour of the current proposer, train cheaper
students to replicate it, judge them in minutes before any search run; in particular test
six static rounds on the walls followed by six on the plan.

**Harness.** `spr/distill_policy.py`: teacher outputs (final cell embeddings plus the candidate
distribution) generated online; student warm-started from the teacher; students `s+k` = s
static rounds on the walls once per board, k dynamic rounds per decision; metric = top-5 overlap
with the teacher on 398 held-out decisions (the search keeps five). Step-0 control `0+12` is the
teacher exactly (100%, KL 0); the trained `0+12` drifts to 91% / 0.982, which bounds what the
loop holds. (Slurm splits `--export` on commas, so the variant is written `6+6`.)

**Result.** Static rounds add nothing: `6+6` (69.6% / 0.929) equals `0+6` (73.1% / 0.930), tied
or untied. Quality is a function of the rounds that see the plan: 4 rounds 55%, 6 rounds 73%,
8 rounds 78%, 12 rounds 91% top-1 agreement. Same lesson as the value net's per-board attempt.

**What it buys anyway.** A plain six-round student keeps 93% of the teacher's top-5 sets at
half the cost, where the teacher's own weights truncated to four rounds had lost 17 points on
the value side. Inside the search (`spr.nets.load_policy_student`, tag `_p6`): 64x64 38/40
against the teacher's 35, none lost; 96x96 18/24, same. The value net's shared pass got the
same treatment (six rounds, distilled: 59.7% top-1 against 67.9%); with both students
(tag `_p6v6`) the outcomes are unchanged and one expansion is about 0.15 s.

---

## 2026-09-12 · The chain, and the real test

Seconds per puzzle, same puzzles, identical or better solves at every step:

| configuration | 96x96 (24 puzzles) | 64x64 (40 puzzles) |
|---|---|---|
| original, as in the scaling table | 324 | 40 |
| + per-decision value net | 179 | 26 |
| + wrapper fixes (exact) | 89 | 16 |
| + axial attention (exact) | 41 | 12 |
| + six-round students | 29 | 6 |

Eleven times less time per puzzle at 96x96, no puzzle lost at either size, three gained at 64.

That made the full ladder affordable. Fast configuration on the first 100 ladder puzzles at
every size, the same puzzles as the network-free arms (44 shards, tag `_full`; the network
never saw a board above 64x64):

| size | network-free at a 20 s cap (m0 / m8 / m32) | network-free at the guided arm's own time (cap) | guided, fast | s per puzzle |
|---|---|---|---|---|
| 24 | 78 / 80 / 80 | 76 (5 s) | 88 | 4.9 |
| 32 | 70 / 73 / 75 | 69 (5 s) | 90 | 4.3 |
| 40 | 60 / 63 / 65 | 60 (6 s) | 85 | 5.8 |
| 48 | 62 / 62 / 63 | 65 (14 s) | 81 | 13.2 |
| 56 | 70 / 71 / 74 | 69 (10 s) | 90 | 9.8 |
| 64 | 69 / 70 / 77 | 67 (9 s) | 92 | 8.2 |
| 80 | 55 / 58 / 61 | 60 (21 s) | 78 | 20.5 |
| 96 | 55 / 56 / 58 | 59 (32 s) | 77 | 31.3 |

The matched-time column is the network-free planner with its iteration cap lifted and the
guided arm's mean seconds per puzzle as its only limit (`jobs/matched.slurm`, `_tc<N>` files).
Extra time barely moves it: it is bound by what its fixed heuristic explores, not by the clock.
At equal wall clock the guided sub-goal search leads by 12 to 25 points at every size from
24x24 to 96x96. This is the table for the paper (`ladder_table.py`, view "All 100 ladder
instances per size").

Caveats written next to it: 100 puzzles per size from one seed; one draw per board at 24-64,
eight per board over 20 boards at 80 and 96; the cap is the guided arm's mean including its
budget-exhausting failures, so the comparison is, if anything, generous to the baseline.

---

## 2026-09-11 to 09-13 · Self-play at new sizes (the curriculum question)

One B2 round at 64x64 (60 boards, 5,207 records), pair warm-started from the zero-shot pair:
64x64 35 -> 37 of 40, 96x96 18 -> 20 of 24, none lost, half the expansions per solved puzzle.
The policy barely moved (best epoch was the first); the gain is the value net having seen 64x64.

Two rounds at 96x96 (60 boards each, 4,668 and 4,330 records): B from the zero-shot pair,
C from the 64x64 pair. Both: 64x64 38/40, 96x96 20/24, identical gains. The curriculum arm is
indistinguishable from training directly at the target size, and the 64-only round already had
both 96 gains. The four 96x96 puzzles that stay unsolved resist every pair and the network-free
search alike; they are a budget question, not a network question.

What a round at 96 needs on this cluster (all in `SCALING_LADDER.md`): the B2 vocabulary
(the launcher's default is base and produced an incomparable first round), one board per
generation shard on the fast partition, a merge that accepts 58 of 60 boards, no data-loader
workers and 250 GB of host memory, `RR_AXIAL=1` for training, and `RR_GROUP_HARD_CAP=1`
(`nn_labeler/dataset.py`: the frozen dataset keeps every tied-optimal candidate past the
per-group cap, which at 96x96 put thirty boards of adjacency masks in one batch).

---

## What is open

1. **One shared pass with two readouts.** An expansion is now two six-round passes over the same
   board with the same plan marks, one for the proposer and one for the value's shared reading.
   A joint model would roughly halve what is left; the distillation harness is the tool to
   build it.
2. **The budget on unsolved puzzles.** 95% of the time at 96x96 went to puzzles that were not
   solved anyway. A cheap network-free first pass, or a smarter give-up, moves the mean more
   than any further change to a pass.
3. **The original pair on all 100 puzzles.** The 100-puzzle rows are with the students; the
   original pair was compared on 24 and 40 puzzles. With the exact speedups only it costs about
   41 s per puzzle at 96, an hour of card time in shards, if the stronger "same accuracy" claim
   is wanted.
4. **Bigger boards.** 96 is the largest size with a board generator and instances today.
