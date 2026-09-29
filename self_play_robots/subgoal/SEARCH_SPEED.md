# Where one search iteration spends its time, and the fixes

Explainer: artifact "One Search Iteration". Profile job: `jobs/profile96.slurm`
(cProfile around `spr.bench --search arena_astar --vocab b2 --anytime`, one
hard 96x96 ladder instance, idx 2, 300 expansions, one A100).

## Measured before any fix (job 5034269, 2026-09-11)

| arm            | wall for 300 exp | per exp | value call (x257/231) | proposer call | propose | attach | exact-path checks |
|----------------|------------------|---------|-----------------------|---------------|---------|--------|-------------------|
| ladder pair    | 497 s            | 1.66 s  | 401 s = 1.56 s each   | 77 s = 0.30 s | 12.8 s  | 5.3 s  | 2.4 s (103k calls)|
| split value    | 256 s            | 0.85 s  | 169 s = 0.73 s each   | 70 s = 0.30 s | 11.3 s  | 4.6 s  | 2.0 s             |

Shares with the ladder pair: value 81%, proposer 16%, everything on the CPU
about 4% (propose 2.6%, attach 1%, plan copies, records, exact-path table
lookups). About 205 candidates are attached per expansion on this instance
(61k `_apply` calls / 300), of which 5 reach the value net.

Two things in the call wrappers (`eval/end2end.py`) that are not the
networks:

1. `_value_cost` expanded the two [9218 x 9218] adjacency masks to the
   candidate batch on the CPU and then copied the expanded tensor to the
   card: 5 x 85M floats x 2 masks = 1.7 GB per value call. That is why the
   split value call costs 0.73 s when its shared pass is about 0.3 s.
   `_policy_logp` copied the unexpanded masks every call (0.34 GB).
2. `_policy_logp` scored candidates in a Python triple loop with one small
   matmul and one host sync (`float()`) per candidate, plus a linear scan of
   `ctg_map` per (bottleneck, support) pair: quadratic in candidates.

## Fix (2026-09-11, `eval/end2end.py`, original kept as `end2end.py.orig`)

- `_adj_dev(env_id, dev)`: the masks live on the device, cached; expanding
  a device tensor to the batch is a view.
- `_policy_logp`: the same three chained softmaxes computed for all
  (bottleneck, support) pairs in batched products, one `.tolist()` per call.
  The original loop is kept as `_policy_logp_loop`; `RR_POLICY_CHECK=n`
  compares the two on the first n calls.

Pre-registered expectation: the split-value arm drops from 0.85 s to about
0.4-0.45 s per expansion (value call to about 0.3 s, proposer call to about
0.15 s); the ladder-pair arm from 1.66 s to about 1.3 s. Same solved /
expansions on the instance (readout differences at float rounding only).
Re-profile: job 5034278 (`_fast` tag).

## Measured after the fix (job 5034278, same instance, same 300 expansions)

Equality check on the first 5 proposer calls: keys identical, max |diff|
2.4e-6. Both arms report the same expansions and accounting as before, so
the search took the same path.

| arm            | wall  | per exp | value call | proposer call | propose | attach |
|----------------|-------|---------|------------|---------------|---------|--------|
| ladder pair    | 357 s | 1.19 s  | 1.07 s     | 0.24 s        | 13.5 s  | 5.3 s  |
| split value    | 126 s | 0.42 s  | 0.23 s     | 0.24 s        | 11.4 s  | 4.6 s  |

Per expansion, ladder configuration to split value + fixed wrappers: 1.66 s
-> 0.42 s, 4x, nothing changed in what is computed. What is left at 0.42 s:
proposer pass 0.24 (its own twelve rounds at batch 1 plus `_meta` over about
200 candidates in Python), value shared pass 0.23 (the same twelve rounds),
propose 0.04, attach 0.02, the rest under 0.02. The joint model (one shared
pass, two readouts) would merge the two 0.23-0.24 s items into one.

Benches with this configuration (tag `_fast`, ARCH=split, patched wrappers):
64x64 first 40 and 96x96 same 24, launched 2026-09-11 22:15.

## Ladder-protocol benches in the fast configuration (jobs 5034280-5034292)

Split value + fixed wrappers, same instances as the `net` column, identical
solves and identical expansion counts on every instance:

| size | n  | solved | s/inst ladder pair | s/inst split alone | s/inst split + fixed wrappers |
|------|----|--------|--------------------|--------------------|-------------------------------|
| 64   | 40 | 35     | 40.3               | 25.7               | 16                            |
| 96   | 24 | 18     | 324                | 179                | 89                            |

Per instance at 96x96: 3.6x faster than the configuration in the scaling
table, same answers. Column `split+fast wrappers` in `ladder_table.py`.

## One proposer call, stage by stage (job 5036419, real 96x96 decision, 180 candidates)

Card-synchronised at every boundary, mean of 5:

| stage | ms |
|---|---|
| adjacency tables to the card (once per board, cached) | 50.4 |
| candidate meta on the CPU (180 candidates) | 0.3 |
| featurise the board, transfer marks | 0.2 |
| input projection | 0.1 |
| 12 rounds | 233.0 |
| of which attention, global family x12 | 54.6 |
| of which attention, all-edges (masked) x12 | 85.0 |
| of which attention, independent-edges (masked) x12 | 85.0 |
| of which projection + norm x12 | 2.5 |
| of which feed-forward + norm x12 | 5.9 |
| (mask build A != 0, 24 per pass) | 7.6 |
| readout, all 180 candidates, batched | 6.9 |
| scores back to the CPU | 0.0 |
| total | 240.5 |

Every round costs the same 19 ms. 96% of the call is the three attention
families; the two masked ones cost 1.6x the unmasked global one, so the
boolean-mask kernel path is slower than plain flash attention. Everything
that is not attention (readout, meta, features, transfers, norms, MLP) is
under 16 ms. Kernel-level split and batch sweep: job profpol2.

## Kernel-level split and batch sweep (job 5036544)

One 96x96 encode pass, card time by op: the fused attention kernel
(memory-efficient, fp32) 183 ms over 36 calls (12 rounds x 3 families); mask
preparation for the two masked families (pad, where, copy) about 50 ms; all
linear layers 12 ms. Batch sweep: 231 ms per board alone, 219 ms per board at
batch 8 at 96x96 (one board saturates the card); 55 -> 46 ms per board at
64x64. Batching expansions is not a lever at 96.

## Pre-registered: exact axial attention for the masked families (job 5036640)

Every slide edge lies in a row or a column, so the two masked families
attend over row cells + column cells only (about 190 keys, not 9217). The
existing `axial_head` (row pass + column pass merged by log-sum-exp, global
token dense) is exact but materialised [B,n,n,n,h,dk] and was only 2x.
`axial_head_fast` computes the same scores as batched matrix products
(`RR_AXIAL=1`, masks cached per board). Expectation: masked families from
85 ms each to a few ms, pass from 233 to about 80-100 ms; max |diff| of the
12-round embeddings against the SDPA path below 1e-4 (float rounding only).
If the diff is larger, the merge has a bug and the path stays off.

## Axial attention result (job 5036655): exact, 2.5x at 96x96

| size | one masked head max diff | 12 rounds max diff (rel) | pass dense -> axial |
|------|--------------------------|--------------------------|---------------------|
| 24   | 3.3e-7 | 4.1e-6 (2.4e-7) | 9.6 -> 38.1 ms (0.3x) |
| 64   | 7.8e-7 | 6.7e-6 (3.1e-7) | 58.0 -> 38.2 ms (1.5x) |
| 96   | 7.8e-7 | 5.7e-6 (3.4e-7) | 231.0 -> 92.9 ms (2.5x) |

Gated to n >= 48 (`RR_AXIAL=1`; below that the dense fused kernel wins).
What remains of the 93 ms at 96: the unmasked global family (about 55 ms,
genuinely all-to-all), the two axial families (about 30 ms), norms and MLP
(8 ms). Both nets use the same layer, so the value's shared pass shrinks the
same way. Next: profile with RR_AXIAL=1, ladder benches tagged `_fast2`.

## Search profile with RR_AXIAL=1 (job 5036667, same instance, same 300 expansions)

| arm | wall | per exp | value call | proposer call |
|-----|------|---------|------------|---------------|
| ladder pair + wrappers + axial | 151 s | 0.50 s | 0.41 s | 0.10 s |
| split value + wrappers + axial | 63 s  | 0.21 s | 0.10 s | 0.10 s |

Same expansions and accounting as every earlier run of this instance. The
whole chain, all exact: 1.66 s -> 1.19 (wrappers) -> 0.42 (split value)
-> 0.21 s per expansion (axial), 8x. What remains: two 93-ms passes
dominated by the unmasked global family, and about 0.02 s of CPU work.
Ladder benches with this configuration: tag `_fast2`.

## Ladder benches with axial attention (jobs 5036668-5036680, tag `_fast2`)

Identical solves and expansion counts on every instance. Seconds per
instance, whole chain:

| size | n  | ladder pair | + split value | + wrappers | + axial |
|------|----|-------------|---------------|------------|---------|
| 64   | 40 | 40.3        | 25.7          | 16         | 12      |
| 96   | 24 | 324         | 179           | 89         | 41      |

96x96: 7.9x less time per instance than the configuration in the scaling
table, same answers. This is the configuration to quote (`RR_AXIAL=1`,
`--arch split`, patched `eval/end2end.py`).

## 6-round distilled proposer in the search (tag `_p6`)

Adds to the `_fast2` configuration: 64x64 38/40 (teacher 35) at 7.6 s per
instance (was 12.4); 96x96 18/24 (same) at 37 s (was 41). Details in
`DISTILL_POLICY.md`. Whole chain at 96: 324 -> 37 s per instance, 8.8x, one
instance's outcome unchanged at 96 and three gained at 64.

## Final configuration so far (tag `_p6v6`): 29 s per instance at 96x96

Two 6-round distilled students (proposer and per-decision value), axial
attention, fixed wrappers. 64x64 38/40 at 6 s; 96x96 18/24 at 29 s. Chain:
324 (table) -> 179 (split value) -> 89 (wrappers) -> 41 (axial) -> 37
(6-round proposer) -> 29 s (6-round value). Nothing lost at either size.

## Axial path in TRAINING: a cache bug, fixed (2026-09-12 evening)

`_axial_masks_cached` kept a reference to the adjacency tensor it was called
with (to keep the data_ptr key valid). In inference those are the few
long-lived per-board tensors; in training every step moves fresh copies to
the card, so the cache pinned up to 16 x 1.4 GB at 96x96 and the value
training of the 96x96 self-play arms went OOM after a minute. Now: no cache
while gradients are enabled (masks rebuilt per call, cheap), cache only
under no_grad. Inference behaviour unchanged.
