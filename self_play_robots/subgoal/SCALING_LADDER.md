# Scaling ladder: sub-goal search from 24x24 to 96x96

The paper's central table. Regenerate with `python subgoal/ladder_table.py`
(reads `results/subgoal/ladder`, `ladder_matched`, `sgnet`).

## Protocol (identical at every size)

Instances: `supervised_valuenet/scaling/data/g<N>r4/ladder.jsonl`, drawn by
`scaling/bench.py` with the project's `eval.bench_instances` sampler, seed 1,
one draw per board (24-64: boards 900-1049, 150 instances; 80 and 96: boards
0-19, 8 draws per board, 160 instances). The exact solver runs with tight caps
(2000 expansions, 2 s) and its failures are KEPT with d_star=null, so the set
is not filtered to what the oracle can do: 23/150 graded at 24, 25 at 32, 9
at 40, 12 at 48, 5 at 56, 5 at 64, 3 at 80, 2 at 96. d* is reference only.
"Solved" = the planner returned a plan that the physics realizes, verified.

Arms (4 robots, B2 vocabulary, relaxed heuristic, lean board loader):

- m0 / m8 / m32: network-free sub-goal search (`subgoal.hybrid`), 400 main
  iterations, 200 per sub-search, 20 s per instance, with 0 / 8 / 32
  macro-move prefix candidates tried before the sub-goal search. 100 instances.
- matched: the same network-free search with the iteration cap lifted and the
  wall clock per instance set to what the trained pair used at that size
  (25 / 40 / 30 / 30 / 150 / 400 s at 40-96, 10 s at 24 and 32). Same
  instances as the network arm.
- net: sub-goal search guided by the size-free policy + value pair of
  mix_b2mix_iter3 (trained on 24 and 32 only, run zero-shot above that),
  1200 expansions, k=5, anytime, one GPU. First 40 instances at 40-64, 12 at
  80, 24 at 96 (cost), all 100 at 24 and 32.

## Table (2026-09-11), solve rate on the instances the network arm ran on

| size | n   | m0   | m8   | m32  | matched | net  | net s/inst |
|------|-----|------|------|------|---------|------|------------|
| 24   | 100 | 78.0 | 80.0 | 80.0 | 78.0    | 89.0 | 5          |
| 32   | 100 | 70.0 | 73.0 | 75.0 | 71.0    | 89.0 | 5          |
| 40   | 40  | 60.0 | 62.5 | 67.5 | 70.0    | 87.5 | 13         |
| 48   | 40  | 55.0 | 55.0 | 57.5 | 60.0    | 75.0 | 33         |
| 56   | 40  | 75.0 | 77.5 | 82.5 | 77.5    | 92.5 | 21         |
| 64   | 40  | 72.5 | 72.5 | 77.5 | 77.5    | 87.5 | 40         |
| 80   | 12  | 75.0 | 75.0 | 75.0 | 75.0    | 91.7 | 69         |
| 96   | 24  | 66.7 | 66.7 | 70.8 | 79.2    | 75.0 | 324        |

Network-free arms on all 100 instances (m0 / m8 / m32, mean seconds):
24: 78/80/80 (2.3/12.9/45.7 s); 32: 70/73/75; 40: 60/63/65; 48: 62/62/63;
56: 70/71/74; 64: 69/70/77; 80: 55/58/61; 96: 55/56/58 (9.9/23.4/60.6 s).

## What it says

1. The network-free sub-goal search does not collapse with size: 55-78% at a
   20 s cap from 24x24 to 96x96, at 2-10 s per instance. Macro-move search on
   the same ladder fell to 10-24% by 96 (PLAN_HYBRID_ACTION_SPACE.md).
2. The size-free networks, trained only at 24 and 32, add 10-17 points over
   the matched-time network-free control from 24 up to 80 (89 vs 78, 89 vs
   71, 87.5 vs 70, 75 vs 60, 92.5 vs 77.5, 87.5 vs 77.5, 91.7 vs 75).
3. At 96 the zero-shot pair stops helping (75 vs 79 on 24 instances) and costs
   324 s per instance. That is the motivation for (a) self-play at the new
   size (round 1 at 64x64 running, then 96) and (b) the encode-once network
   (`ENCODE_ONCE.md`), which attacks the per-instance cost.
4. The macro-move prefix (m8, m32) buys 0-8 points at 3-6x the time; it is
   not where the scaling comes from.

Caveats: 12 and 24 instances at 80 and 96 are small samples (one instance is
4-8 points); the 40-64 rows are 40 instances. The matched control at 24/32
used a 10 s cap, twice the network's 5 s.

## Self-play at 64x64: correction (2026-09-11)

The first round (jobs 5030294 / 5031174 / 5032277 / 5032783) ran with the
launcher's default `VOCAB=base` while the pair it fine-tunes (mix_b2mix_iter3)
and every network row above are B2. Its buffer (998 records, vocab=base) and
its policy checkpoint are not comparable and are not used. Resubmitted as
job 5032801 with `VOCAB=b2 WORKERS=4 BOARDS=60 VBS=2` (value batch halved:
32 boards of 64x64 with gradients exceeded 40 GB). Fresh buffer under
`runs/spr/selfplay/g64r4_b2_iter1`, internal bench on the first 40 ladder
instances. The comparable number afterwards comes from `jobs/sgnet_smoke.sh`
with the new pair at 64 and 96, same protocol as the table.

## Split value at 96x96 (2026-09-11)

Column `net_split` in the table: the distilled per-decision value net
(`ENCODE_ONCE.md`) inside the same search. 96x96, same 24 instances: 75.0%
solved in both arms, 179 s per instance against 324 s. 64x64 pending.
64x64, first 40 instances: 87.5% in both arms, 25.7 s against 40.3 s;
agreement 40/40.

## Self-play at 64x64 with B2: how the round is actually run (2026-09-11)

The long partition did not start the 6 h job in three hours (700+ jobs
ahead), and the B2 generation phase is slow: 5 of 60 boards in 47 min at
4 workers, about 9 h for the set, written only at the end. So the round is
split to fit the fast partition (58 min per job):

1. `jobs/gen_shard.slurm` x 15 (jobs 5033553-5033567): the generation
   phase with the round's exact flags, 4 boards each (ids 8000-8059), output
   `runs/spr/selfplay/g64r4_b2_iter1/records_<shard>.jsonl`.
2. `jobs/merge_gen.slurm` (5033570, afterany on all shards): concatenates
   the shards into `records.jsonl`, merges the manifests, marks `DONE.gen`.
   Fails if a shard is missing, which blocks step 3.
3. `jobs/selfplay_iter.slurm` x 5 chained (5033571-5033575, afterok on the
   merge, then afterany): gauge, buffer, policy, value, bench. Phases are
   idempotent and training resumes from `last.ckpt`, so each link continues
   where the previous one was cut.

Afterwards: `jobs/sgnet_smoke.sh` with the new pair (VALUE and POLICY
overrides) at 64 and 96 on the ladder instances, same protocol as the table.

Progress (2026-09-11 15:00): shards done in about 30 min each; merge gave
5207 records from 673 instances (598 solved by the search). Gauge phase
failed non-fatally: it launches the exact Rust engine, whose binary is not
built in this clone; it is a diagnostic only. Policy trained in 13 min.
Value training OOMs at 16 boards per step (VBS=2: 38 GB; the SDPA mask is
kept per head and per recurrence for backward), so the chain was resubmitted
as jobs 5033671-5033673 with VBS=1 VMPG=8 (8 boards per step, about 12 min
per epoch, resumes across links).

Round finished its training (2026-09-11 17:16). Pair in
`results/selfplay/g64r4_b2_iter1/nets.txt`: policy epoch 0 (validation got
worse after the first epoch, so the recipe kept it), value epoch 5 (top-1 on
the round's 9 validation boards 80.5%). The round's own `spr.arena` bench
(40 instances, two searches) does not fit a 58-min link and cannot resume,
so it is skipped; the comparison uses the ladder-protocol benches instead:
`jobs/bench_newpair.slurm` (jobs 5033799-5033811, tag `_sp64`, column
`net_sp64` in `ladder_table.py`), 64x64 first 40 and 96x96 same 24.

## Self-play at 64x64: result (2026-09-11 evening, jobs 5033799-5033811)

One B2 round at 64x64 (60 boards, 5207 records), pair warm-started from the
zero-shot pair. Same instances and protocol as the `net` column.
`subgoal/cmp_sp64.py` reproduces this.

| size | n  | zero-shot pair | new pair | gained | lost | s/inst zero-shot -> new |
|------|----|----------------|----------|--------|------|-------------------------|
| 64   | 40 | 35 (87.5%)     | 37 (92.5%) | idx 4, 14 | none | 40 -> 32 |
| 96   | 24 | 18 (75.0%)     | 20 (83.3%) | idx 13, 15 | none | 324 -> 233 |

On the instances both solve, the new pair needs fewer expansions (64: 12 ->
5; 96: 18 -> 12) at the same plan length (64: 19.8 -> 19.5; 96: 26.0 -> 26.8).
The gained instances at 96 are two the zero-shot pair had exhausted its
1200-expansion budget on (980 s and 1530 s); the new pair solves them in 48
and 624 expansions (82 s and 501 s).

Reading: one round of self-play at a new size improves the size it trained
on AND transfers one rung up, with no loss on either. The policy barely
moved (its best checkpoint is epoch 0, one pass over the records), so the
gain is mostly the value net having seen 64x64 boards. Caveat: 40 and 24
instances; two gained instances each is about 5-8 points and within a
one-instance swing of noise at 96.

This is the curriculum arm "C: 64 then 96" half done. Next: a round at 96
starting from this pair (arm C), and a round at 96 starting from the
zero-shot pair (arm B) for the comparison the plan asked for.

## Curriculum arms at 96x96 (launched 2026-09-11 20:00, `jobs/launch_arm96.sh`)

Both arms are one B2 round at 96x96 (60 boards, ids 8000-8059, 8 draws
each), 30 generation shards of 2 boards, merge, three chained links
(`VBS=1 VMPG=2 PBS=2`: two boards per value step and two per policy step,
the most a 40 GB card holds at 9217 tokens with gradients; VMPG=2 truncates
groups to two candidates, a caveat for the rank loss), then the ladder-
protocol benches at 96 (same 24) and 64 (first 40).

- Arm C, "64 then 96": starts from the 64x64 round's pair. Dir
  `g96r4_b2c_iter1`, bench tag `_sp96c`, jobs 5034118-5034164.
- Arm B, "direct 96": starts from the zero-shot pair. Dir `g96r4_b2_iter1`,
  bench tag `_sp96b`, jobs 5034165-5034211.
- Arm A, "zero-shot": the existing `net` column (75.0% at 96).

Pre-registered reading: C beats B on the 24 instances at 96 = the curriculum
helps beyond what one round at the target size gives; C or B beats A = self-
play at the target size helps at all. Two instances is the noise band.

## Fast configuration (2026-09-11 late): same solves, 2.5-3.6x less time

Per-decision value net (`ENCODE_ONCE.md`) plus the two exact wrapper fixes in
`eval/end2end.py` (`SEARCH_SPEED.md`): 64x64 first 40: 35 solved at 16 s per
instance (was 40.3 s); 96x96 same 24: 18 solved at 89 s (was 324 s).
Identical expansion counts per instance. This is the configuration the
paper's time column should quote for the network arm.

## 96x96 arms: first generation attempt overran (2026-09-12)

Two boards per shard did not fit the 58-min partition at 96x96: 8 of 30
shards finished (45-57 min each), 22-23 per arm were cut, both merges
refused, the chains became unsatisfiable and were cancelled with their
benches. Resubmitted via `jobs/launch_arm96_fill.sh`: the missing shards as
ONE board per job (44 for arm C, 46 for arm B; merges 5036464 / 5036527,
links 5036465-67 / 5036528-30, 13 benches each). `jobs/merge_gen.slurm` now
checks board coverage (60 distinct env_ids) instead of shard indices.
Generation uses `spr.search.Evaluator` (full nets, its own readout loop), so
the wrapper fixes in `eval/end2end.py` do not apply to it yet.

## Fast configuration, second step: axial attention (2026-09-12)

`RR_AXIAL=1` (exact row+column attention for the two masked families,
`SEARCH_SPEED.md`): 64x64 first 40: 35 solved at 12 s per instance; 96x96
same 24: 18 solved at 41 s (was 324 s in the table's `net` column). Column
`+ axial attention` in `ladder_table.py`. Identical expansion counts.

## Fast configuration, third step: 6-round distilled students (2026-09-12)

`DISTILL_POLICY.md`: proposer and per-decision value net both replaced by
6-round students distilled from the 12-round nets. 64x64 first 40: 38 solved
(teacher 35) at 6 s per instance; 96x96 same 24: 18 solved at 29 s (table:
324 s). Columns `+ 6-round proposer` and `+ 6-round value` in
`ladder_table.py`. This is now the configuration to quote for the network
arm's time column; the solve column is unchanged or better.

## 96x96 arms, third relaunch (2026-09-12 evening)

One-board shards covered 59 of 60 boards per arm (one shard each hit the
hour); merge threshold relaxed to 58 (`NBOARDS=58`), arm C merged 4668
records from 726 instances (602 solved). Policy training then died of HOST
memory: the job had 124 GB, MaxRSS 127 GB, a DataLoader worker OOM-killed
(dense 9217x9217 adjacencies, 340 MB per mask per board, replicated across
4 loader workers). Relaunched with `NW=0` (loader workers, new env knob in
`selfplay_iter.slurm`) and `--mem=250G`: arm C links 5036926-28, arm B
links 5036942-44 (after its merge 5036819), 13 benches each.

## The real test: fast configuration on all 100 ladder instances per size

Launched 2026-09-12 (44 shards, tag `_full`): both 6-round students, axial
attention, fixed wrappers, on the first 100 ladder instances at 24, 32, 40,
48, 56, 64, 80, 96, the same instances as the network-free arms. View
"All 100 ladder instances per size" in `ladder_table.py`.

96x96 arms, fourth relaunch: value training OOMed on the card at 2 boards
per step (dense masked kernel), then again with `RR_AXIAL=1` because of the
mask-cache bug (`SEARCH_SPEED.md`). Fixed; links 5037483-85 (arm C, policy
already trained) and 5037499-5037501 (arm B), 13 benches each, all with
`RR_AXIAL=1 NW=0 --mem=250G VBS=1 VMPG=2 PBS=2`.

96x96 arms, fifth relaunch: the value OOM was the dataset, not the model.
`GroupDataset` keeps EVERY is_optimal record even past max_per_group (a
frozen quirk); at 96x96 a self-play group with 30 tied-optimal candidates is
30 boards x 2 masks x 340 MB in one batch. New env `RR_GROUP_HARD_CAP=1`
(nn_labeler/dataset.py) truncates to the cap. Relaunched with it.

## HEADLINE: fast guided search on all 100 ladder instances per size (2026-09-12)

Both 6-round distilled students (trained/distilled on 24, 32, 64 decisions
only), axial attention, fixed wrappers, 1200 expansions, k=5, one A100.
Network-free arms at a 20 s cap on the SAME instances. `ladder_table.py`,
view "All 100 ladder instances per size"; result files
`results/subgoal/sgnet/g<N>r4_<lim>_<off>_full.json`.

| size | network-free m0 / m8 / m32 (%) | guided, fast (%) | s per instance, m0 -> guided |
|------|--------------------------------|------------------|------------------------------|
| 24   | 78 / 80 / 80                   | 88               | 2.3 -> 4.9                   |
| 32   | 70 / 73 / 75                   | 90               | 3.2 -> 4.3                   |
| 40   | 60 / 63 / 65                   | 85               | 3.9 -> 5.8                   |
| 48   | 62 / 62 / 63                   | 81               | 4.3 -> 13.2                  |
| 56   | 70 / 71 / 74                   | 90               | 4.4 -> 9.8                   |
| 64   | 69 / 70 / 77                   | 92               | 5.5 -> 8.2                   |
| 80   | 55 / 58 / 61                   | 78               | 8.3 -> 20.5                  |
| 96   | 55 / 56 / 58                   | 77               | 9.9 -> 31.3                  |

Reading: the guided sub-goal search solves 77-92% of random puzzles from
24x24 to 96x96, 10-25 points above the network-free search at every rung,
at 4-31 s per puzzle. The networks never saw a board above 64x64 (the
proposer student was distilled on 24/32/64 decisions, the value student on
24/32 plus the 8-robot 24 buffer). The drop from 92% at 64 to 77-78% at 80
and 96 is where the size curriculum (arms B and C) is aimed.

Caveats: 100 instances per size, one draw per board (24-64) or eight per
board over 20 boards (80, 96); the network-free arms had a 20 s cap while
the guided arm had an expansion budget (its unsolved instances run 1200
expansions, about 180 s at 96), so the time columns are not a matched-time
comparison at the top end. The matched-time control at 96 (`_tc400`, 24
instances) is the reference for that: 79% at 400 s.

## Matched-time controls on the same 100 instances (jobs 5039536-5039543)

Network-free planner, iteration cap lifted, wall-clock cap per instance =
the guided arm's mean seconds per instance at that size:

| size | guided, fast (%) | net-free at the same time (%) | cap (s) | delta |
|------|------------------|-------------------------------|---------|-------|
| 24   | 88               | 76                            | 5       | +12   |
| 32   | 90               | 69                            | 5       | +21   |
| 40   | 85               | 60                            | 6       | +25   |
| 48   | 81               | 65                            | 14      | +16   |
| 56   | 90               | 69                            | 10      | +21   |
| 64   | 92               | 67                            | 9       | +25   |
| 80   | 78               | 60                            | 21      | +18   |
| 96   | 77               | 59                            | 32      | +18   |

Giving the network-free search the guided arm's time changes its solve rate
by at most a few points from its 20 s-cap numbers (it is iteration-bound,
not time-bound, on these instances), so the guided arm's lead holds at equal
wall clock: +12 to +25 points at every size. Column "matched-time net-free"
in the full view of `ladder_table.py`.

96x96 arms: both pairs exist (2026-09-12 late). Arm C (from the 64 pair):
policy epoch 1, value trained 54 min at 96x96 with the hard cap. Arm B
(from the zero-shot pair): value trained 55 min. The idle last links were
cancelled so the 13 ladder benches per arm (tags `_sp96c`, `_sp96b`) could
start; they run the full 12-round pairs with axial attention.

## Curriculum arms: result (2026-09-13 early)

Same instances (64: first 40; 96: same 24), full 12-round pairs. A = the
zero-shot pair (mix_b2mix_iter3); 64-pair = one round at 64; B = one round
at 96 from A; C = one round at 96 from the 64-pair. B and C benched with
axial attention on (their seconds are not comparable to A's).

| arm            | 64x64 solved / 40 | 96x96 solved / 24 | gained vs A (64) | gained vs A (96) |
|----------------|-------------------|-------------------|------------------|------------------|
| A zero-shot    | 35                | 18                | -                | -                |
| 64-pair        | 37                | 20                | 4, 14            | 13, 15           |
| B direct 96    | 38                | 20                | 4, 14, 28        | 13, 15           |
| C 64 then 96   | 38                | 20                | 4, 14, 28        | 13, 15           |

Nothing lost anywhere. Reading against the pre-registration: self-play at a
new size helps (+2 of 24 at 96, +3 of 40 at 64, and a round at 96 also
lifts 64); the curriculum arm C is indistinguishable from direct-96 B on
these instances, and even the 64-only round already had both 96 gains. So
"train on the biggest boards you can generate" is as good as the ladder of
sizes, on this sample. The four unsolved 96x96 instances (idx 1, 3, 6, 11)
resist every pair and the network-free search alike; they are the budget
question, not a network question.
