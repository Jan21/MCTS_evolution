# By-reference integration — implemented behind a default-OFF flag (2026-08-16)

**Nothing submitted, nothing committed.** All three FINDINGS-40 blocks are now
wired; every default path is byte-identical to HEAD.

## What changed

- `eval/compare.py::_nn_astar_backward` (+ `run_backward`, `main`) — with
  `byref` on it injects `skeleton/astar.py::_reference_helpers`, applies with
  `by_reference=True`, and bounds the pool to top-m by `solver.score`
  (`byref_pool`), the same ordering `AStar._expand` / `nn/generate.py` use.
  Adds observational `byref_cands_{ranked,topk}` accounting; run name gains
  "[by-reference]"; `protocol` records both settings.
- `eval/end2end.py::_hidx(state, pos, helper_color=None)` — colour (robot
  identity) first, position fallback. Strict superset: a robot at its own start
  resolves identically; a target-robot helper keeps the old behaviour.
- `eval/end2end.py::_policy_logp(..., byref=)` → `train/policy_common.py::_meta(group, byref=None)`
  — the record filter is now flag-controlled. Needed at eval too: without it
  by-reference candidates miss `ctg_map` and fall to the −1e9 default.

**Flags** (all OFF by default): `--backward-byref` / `RR_BYREF=1`;
`--backward-byref-pool M` / `RR_BYREF_POOL`; `RR_BYREF_RECORDS=1` (or
`_meta(byref=True)`) for training. Eval passes `byref` explicitly, so
`RR_BYREF_RECORDS` in a job env cannot leak into a benchmark run.

## Tests — `eval/test_byref_wiring.py` (21 s / ~1 min, login node)

- **A, off = old driver:** `run_backward` vs the same function extracted from
  HEAD, 12 instances × 600 expansions, anytime+b2: **12/12 rows identical**
  (all fields but `seconds`, including dumped move sequences).
- **B, on:** 159 by-reference candidates reached the policy net, 9 entered a
  top-k, 12/12 solved, all **replay-certified** by `eval.replay_validate`.
- CLI/env path smoke-run through `eval.compare`: `byref=True pool=100`, 3/3
  solved, counters land in the aggregate.

## Census — `scaling/data/g16r6/backward_b2.cap20000.rust.jsonl`

236,037 records / 24,019 decision groups. Flipping the filter **gains 20,093
records (8.5%; 8.7% of the trainable set)**; **1,867 groups (7.8%) become
usable at all**, 4,889 more grow. 5,950 records stay dropped — helper is the
*target robot*, which has no slot in `state.helpers` (pre-existing, unchanged).
Caveat measured on a 20k-record slice: the labelled **optimal target changes in
12.6% of groups**, so a retrain is not purely additive.

## A/B — `jobs/patterns/byref_ab.slurm` (written, not submitted)

g16r6 pooled 450 (graded 316 + frontier 134), 4 lanes, same banked cap-20000
nets, 1200/k=5, anytime+b2, replay-certified, outputs `*_byref_{on,off}.json`
(never a `report_data.py` filename). From the measured cap-20000 lanes
(--gpus 2 = 0.25 nh/h; graded 8–22 min, frontier 24–45 min): **~0.9 node-hours
expected, 2.0 nh if lanes sit to the 8 h walltime.** Pool default m=100 = the
as-shipped mean applications/expansion, so the new candidates substitute rather
than add physics; if the on-arm loses, rerun unbounded before concluding.

## Left as design, not code

The policy net cannot *name* the referenced cell: a by-reference candidate is
scored at its robot's start-cell embedding. An 8th `_features` channel would
change `_x257`'s width and break `PolicyTF.load_from_checkpoint` on every
banked net, so it is written up in `train/policy_common.py::_features`
(migration sketch included) and deliberately not implemented — decide after the
zero-shot A/B.

Unrelated: `eval/report_*.py` show as modified in `git diff`; not by me.
