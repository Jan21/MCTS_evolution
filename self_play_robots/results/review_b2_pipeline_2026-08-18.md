# Review: B2 self-play code path + fidelity gauge (2026-08-18, read-only)

Scope: `self_play_robots/spr/{search,selfplay,bench,nets,train,gauge}.py`,
`jobs/selfplay_iter.slurm` (VOCAB=b2), compared against
`supervised_valuenet/eval/compare.py`, `skeleton/astar.py`, `nn/generate.py`,
`scaling/rust_bridge.py`, `rust_datagen/src/{io.rs,subgoal/rollout.rs}`,
`jobs/patterns/b2_labels_*.slurm`. No code was changed. Small login-node
checks (seconds) were run on the existing B2 iteration-1 artefacts; their
numbers are quoted below with the command that produced them.

## 0. Headline (read this first)

1. **The B2 gauge already has a number, and it is 0.51.** The killed gauge of
   job 4685442 had in fact returned 198/200 instances (all `status: ok`) before
   the two wanderers were SIGTERMed; only the Python side (`ep.collect`) threw.
   Running `nn_labeler.audit_descent` on those 198 exact results vs the saved
   `selfplay_subset.jsonl` gives **argmin agreement 0.510, optimal-set Jaccard
   0.41, mean gap +3.9 (+4.5 on exact-optimal candidates), share of zero gaps
   0.40, negative gaps 0** (files:
   `results/selfplay/g24r4_b2_iter1/gauge.work/engine.results.jsonl`, audit in
   the session scratchpad `audit_cap64.json`). Base iteration 1 was 0.915 /
   0.86 / gap 0.99. On FINDINGS 74's bands this is far below "collapse", BUT:
2. **The exact B2 reference is largely unrealizable, so the band does not
   transfer.** On 16 sampled depth-0 decisions (28 exact-optimal candidates),
   the Python solver reproduces the Rust ctg exactly (parity holds) and
   `strict_moves` certifies only **13/28 (46 %)** of those exact-optimal
   completions (transient supports the abstract language accepts but physics
   does not). The self-play labels are certified plan costs; the exact B2 labels
   price unplayable plans and carry no park cost (`nn/generate.py:147-163`
   docstring: parks are not part of any labeling vocabulary). Half of the
   "disagreement" is therefore the reference, not the loop. This is the same
   pathology that made exact-B2-trained nets weak in the supervised track
   (FINDINGS 40/78) and it means **the fidelity gauge needs a certified
   reference for B2** (see B below) before its number can steer anything.
3. **Why the gauge exploded:** `spr/gauge.py:112-120` builds the
   `backward_rollout` item WITHOUT `"budget": {"solver_iters": N}`, so the engine
   applies `DEFAULT_SOLVER_ITERS = 10_000_000` per rollout
   (`rust_datagen/src/io.rs:57,612`). Every supervised B2 label job set
   `--budget-iters` (5 000 production, 20 000 for the `cap20000` corpora;
   `jobs/patterns/b2_labels_shard.slurm:53-56`, `scaling/rust_bridge.py:255-261`)
   with `max_candidates 14`. On the killed run the 198 finished rollouts used
   median 3.5 k, p90 26 k, max 115 k iterations (sum 2.05 M; 170/198 <= 20 k,
   189 <= 50 k, 197 <= 100 k); the two wanderers burned ~2.4 h and 25 GB. One
   added key fixes the cost; the `--max-candidates 24` cap does not (it changes
   the label semantics instead, see B).
4. **Nothing found that invalidates tonight's iterations 2-3** in the search /
   record / training path (details in A). The B2 code mirrors the arena's byref
   + park_hook path faithfully; the record and label semantics are internally
   consistent. Two operational risks: (i) VOCAB/BOARDS/EXP/STOP/WORKERS reach
   4689627/4689628 only through the submitting shell's environment (SubmitLine
   shows no `--export`); if VOCAB was not exported, `selfplay_iter.slurm:40`
   silently runs the BASE loop into `runs/spr/selfplay/g24r4_iter2` (exists,
   DONE markers -> no-op) and a base iteration 3 with anchors; (ii) qgpu is
   currently 48 maint / 16 drained / 8 draining, Slurm's start estimate for
   4689626 is 2026-08-20 18:05 — the "tonight" window may not exist.

## 1. Prioritized findings

Severity: H = would corrupt data/decisions, M = biases a measurement or wastes
budget, L = latent / cosmetic.

### H1 — gauge reference semantics: exact B2 labels are not the yardstick the loop optimizes
- Where: `spr/gauge.py` (whole design), `nn/generate.py:147-163`,
  `rust_datagen/src/subgoal/rollout.rs:168-183`.
- Scenario: `audit_descent` compares self-play argmins against `is_optimal` of
  abstract B2 completions that are unplayable ~50 % of the time (measured
  13/28 above). Any B2 gauge number computed this way is uninterpretable
  against the base bands; a "good" loop iteration could read as 0.5.
- Minimal fix: keep the exact rollout for `cost_to_go` but add a
  realizability pass and report agreement on the CLEAN subset (decisions whose
  exact optimum strict-realizes) plus the share of clean decisions; or, better,
  reference = certified language optimum from `spr.ceiling` machinery. A cheap
  Python-only variant is feasible: the depth-0 solves above ran 28 candidates
  in 2 s.

### H2 — gauge item has no rollout budget (cost explosion, and the time-cap fallback selects the easy subset)
- Where: `spr/gauge.py:112-120` (item), `:127-143` (time cap);
  `rust_datagen/src/io.rs:57,612`; `scaling/rust_bridge.py:255-261` (how the
  supervised track sends it).
- Scenario: wanderers run to 10 M iterations; with `--time-cap` the audit is
  taken over whichever instances finished — those are the quick (easy)
  rollouts, so the sample is biased toward agreement. Also
  `RolloutStatus::BudgetExhausted` returns NO records at all (`io.rs:626`), so a
  budget cap drops the instance rather than yielding its depth-0 labels.
- Minimal fix: add `"budget": {"solver_iters": 100000}` to the item (1 line);
  keep `--time-cap` only as a safety net. Longer term use the engine's
  `replay_backward_decision` task (depth-0 only, per-candidate cap 4000, no
  wandering: `rust_datagen/DESIGN.md:164-190`), which is exactly the depth-0
  slice the gauge consumes.

### M1 — `--max-candidates` on the exact side changes the label, not just the cost
- Where: `spr/gauge.py:53-55,117`; `rollout.rs:152-158` (score-sort then
  truncate BEFORE labeling; `is_optimal` is over the truncated set).
- Measured: root B2 candidate sets at g24r4 have mean 24, median 12, max 204;
  8 % of instances exceed 64, 18 % exceed 24 (60 sampled instances, Python
  `propose_b1` + `_reference_helpers`). One sampled depth-0 group had 198
  labeled self-play candidates. With cap 24: 6 unmatched argmins vs 2 at cap
  64 (audit_cap24 vs audit_cap64), and exact `is_optimal` can point at a
  candidate that is not the true optimum when the optimum is beyond the cap.
- Fix: cap >= 64 (or none) and control cost through `budget.solver_iters`.

### M2 — depth-0 gauge cannot see by-reference candidates at all
- Where: `skeleton/astar.py:193-222` (`_reference_helpers` needs placed
  robots; empty at the initial plan), `spr/gauge.py:86` (depth == 0 only).
- Scenario: the B2-specific vocabulary (by-reference; 71/5822 records at
  depth >= 1) is never audited; the gauge measures only the transient-support
  half. Not fixable at depth 0; note it in FINDINGS and, if a by-ref audit is
  wanted, use `replay_backward_decision` on depth-1/2 groups (their plan is
  reconstructible from the record's ctx fields only approximately — would
  need the plan serialized into the record; today it is not).

### M3 — the queued gauge job 4689629 (80 / 24 / 2400 s) inherits H2 and M1
- It will still send 10 M-iteration rollouts; with 16 threads and ~2 wanderers
  it will most likely finish ~78-80 instances by the cap and report a biased
  number in the 0.5 region. Its result should not be logged as "the B2 gauge"
  without the caveats in H1/M1. Cheaper and better: cancel it and instead run
  `audit_descent` on the existing 198 results (already done above; 0.510) plus
  the realizability pass.

### M4 — job env hand-off (VOCAB etc.) is invisible and silently defaults
- Where: `jobs/selfplay_iter.slurm:34-40`; SubmitLines of 4689627/4689628
  carry no `--export`. Iteration 1 provably ran with BOARDS=60 EXP=300 STOP=80
  WORKERS=6 VOCAB=b2 (manifest), so the shell had them; if the chained jobs
  were submitted from another shell they run BASE (`TAGV=""`, ids 5000+,
  ANCHOR=1). The first log line (`SPR ITER ... boards=8060-8119`) reveals it
  within a minute of start — worth a watch.sh trigger. Minimal fix for the
  future: pass VOCAB/BOARDS/... as positional args or bake `--export=ALL,VOCAB=b2`.

### M5 — resume job 4689626 (4 h) will probably not finish the frontier MCTS bench
- Iteration-0 frontier A\* took 3474 s and graded MCTS 2724 s; frontier MCTS
  on 218 mostly-unsolved instances at 1200 expansions best-at-budget is
  plausibly > 2 h. `bench_one` frontier rows are `|| true` and not resumable
  mid-run, so the it1 frontier_mcts row may be missing; iterations 2/3 (8 h)
  will each spend ~3-4 h on the four benches. Not fatal; budget only.

### L1 — helper-slot resolution order differs between search key and policy meta (inherited from the arena)
- Where: `spr/search.py:73-84` (`hidx`: colour first, then position) vs
  `spr/nets.py:83-86` (`policy_meta`: position first, then colour); same pair
  as `eval/end2end.py:32-52` vs `train/policy_common.py:110-116`.
- Scenario: a by-reference helper whose planned cell coincides with ANOTHER
  helper's start cell resolves to different slots on the two sides -> the
  key misses (`search.py:222`, logp = -1e9) -> candidate never enters top-k;
  in training the group's `ctg_map` gets the wrong slot. Rare (needs a robot
  to vacate its start cell first) and identical to the arena, so parity holds;
  fix later by making both colour-first.
- Related, by design: a by-reference helper that is the TARGET robot
  ("target-as-stopper", `skeleton/astar.py:271-289`) has no slot in
  `state.helpers`, so `hidx` returns None and the candidate is dropped by
  both spr and the arena (`compare.py:260`). One of B2's three shapes is thus
  unreachable to the nets. Same as the arena; note in FINDINGS.

### L2 — duplicate policy keys for by-reference candidates
- Where: `spr/search.py:214-215`, `spr/nets.py:88-108`.
- Two by-ref candidates that differ only in the referenced cell (same robot
  offered at its terminal support AND a mid-chain bottleneck; same bn/sup)
  share `(bn, sup, slot)`; in search both get the same prior (harmless), in
  training `ctg_map` keeps the last one and `cands` lists both (double weight
  in the soft target). Known design gap ("policy cannot name the referenced
  cell", `train/policy_common.py:38-61`); the value net does see the cell
  (`nn_labeler/encode.py:85`).

### L3 — sibling completion re-certifies a failed terminal instead of trying its park children
- Where: `spr/selfplay.py:137-151`, `:215` (Certifier without `parks=`).
- A failed terminal that received park children but whose children were
  never selected has `best_cert None`, is not dead, so `_greedy_complete`
  returns the same complete plan and the second (memo-less) certifier fails
  it again (`sibling_failed`). Wasted physics, no wrong label. Cheap gain:
  certify its park children (1-2 `strict_moves`) before giving up.

### L4 — provenance: `protocol["byref"] = False` hard-coded in bench payloads
- Where: `spr/bench.py:235`. Under `--vocab b2` the arena path runs with
  `byref=True` (`bench.py:92,170`), but the payload says False; readers that
  key on `protocol.byref` (the supervised report tooling) will mislabel the
  rows. `protocol.vocab` is correct.

### L5 — `_extract_records` counts a repaired failed terminal as a "decision"
- Where: `spr/selfplay.py:116-127,131-138,156`. The principal-path walk can
  descend into a failed terminal (expanded=True after repairs) whose children
  all have `rec=None`; the group is skipped but `stats["decisions"] += 1`.
  Stats only.

## A. Correctness of the B2 search path (answers)

A1 By-reference candidates and records. `expand()` injects
`_reference_helpers(plan, mover)` (`search.py:201-202`, = `compare.py:246-248`
and `AStar._expand`), applies with `by_reference=True` (`:207`), resolves the
slot by colour (`:211`), and keys the policy map with that slot (`:215`);
`Evaluator.policy_logp` passes `byref` into `policy_meta` (`:131`). The record
stores `cand_helper = [planned cell, colour]` (`:68`), so the value net marks
the planned cell (`nn_labeler/encode.py:85`, `key_indices` :131) — the same
featurization the exact B2 corpora use — and the policy sees the robot's
start-cell slot (design gap L2, same as the arena). Training keeps them via
`PolicyGroupDataset(byref=True)` (`train.py:300-301`, `--byref` set at
`selfplay_iter.slurm:109`). Verdict: well-defined and trainable; parity with
the arena. Caveats L1/L2.

A2 Park repairs in MCTS. Failed terminal: `cert=False`, `expanded=True`,
`complete=False`, children = repair plans with `rec=None`, `Q=v_est=rp.cost()`
(`search.py:539-555`); `_backup -> _refresh` recomputes Q from live children
and closes/deads the node correctly (`:484-498`); the select loop treats it as
internal (`:525`); a repair child, once selected, is certified exactly once
and becomes solved/closed or dead/closed or (one more level, single-park only)
gets its own repairs — `park_repairs` enforces `max_parks=2` itself
(`skeleton/astar.py:459-462,506`), so repair depth is bounded. `best_cert /
best_abs` propagate through the failed terminal to the decision node
(`:570-573`, path includes the terminal), so the candidate that led to the
failed plan is labeled with the park-augmented cost. `_extract_records` skips
`rec=None` children (`selfplay.py:137-138`) and never reads `cert`. No double
certification found; the Certifier memo keeps plans alive so `id()` keys are
safe (`search.py:291-293`). Verdict: consistent. Only L3/L5.

A3 A\* priorities. Repairs are pushed at `rp.cost()` (`search.py:392-393`);
other entries carry `v_est = fixed_g(parent|child) + ctg_hat` (`:241,409`).
Both are estimates of total ABSTRACT plan cost, `rp.cost()` being the exact
one for a complete plan — same units, same convention as
`compare.py:235-241`. The best-at-budget stop (`:375`) compares against
`best_abs = plan.cost()`, also abstract. OK.

A4 `Certifier.fails` / re-repair. `fails[id(plan)]` is set only after the
plan is memoized (kept alive), so no id recycling. A park child that fails can
be repaired again (single park only: `n_parks + 2 > max_parks` blocks the
pairwise branch), and a 2-park plan returns `[]`. Bounded; matches the arena's
`max_parks=2 if b2` (`compare.py:394-397`).

A5 Label semantics. `cost_to_go = int(round(plan.cost())) - fixed_g(decision)`
(`selfplay.py:163-167`) with the certified plan's cost including park edges
(`apply_park` adds a fixed costed edge, `skeleton/astar.py:434-435`). The
exact B2 labeler prices the cheapest ABSTRACT completion, parks excluded,
realizability ignored (`nn/generate.py:96-104`, `rollout.rs:174-183`). So the
self-play labels are certified upper bounds in the exact labeler's units —
the same relation as base (descent >= exact, `audit_descent` docstring) but
with a much larger and structurally different gap: measured mean +3.9 at
depth 0, 40 % zero-gap, and ~54 % of exact optima unplayable. `is_optimal`
argmins are over abstract cost among certified candidates (`:163-167`, note
the search itself picks by STRICT moves, `search.py:565`), which is the
correct unit for the trainer; comparability with the exact B2 argmin is what
H1 is about — do not read FINDINGS 74's thresholds into B2 numbers.

## B. Fidelity gauge for B2 — recommendation with numbers

Facts. Supervised B2 exact labeling: `scaling.rust_bridge --system backward
--vocab b2 --per-graph 10 --threads 16 --budget-iters 5000` (production;
`20000` for the `cap20000` corpora used by b2_payoff), `max_candidates` 14
(default, `rust_bridge.py:70`), `max_iters 4000`, `max_frontier 40000`; smoke
calibration 36.7 s/graph at g16r4 base (10 keepers, ~40 attempts) i.e. on the
order of 10^5 iterations/s aggregate on 16 threads
(`jobs/patterns/b2_labels_shard.slurm:12-25`). Killed gauge: 198/200 ok,
median 3.5 k, p90 26 k, max 115 k iterations, sum 2.05 M; keep-rates at caps
5 k/20 k/50 k/100 k = 112/170/189/197 of 198.

Recommended settings (all fit in minutes, not 30):
- Item: `"budget": {"solver_iters": 100000}` (keeps ~99 % of instances,
  worst case 200 x 100 k = 20 M iterations ~ 3-4 min at 16 threads),
  `max_candidates 64` (or omit; 8 % of roots exceed 64), `--sample 200`,
  `--threads 16`, `--time-cap 900` as a safety net only. Log the
  `budget_exhausted` count as part of the gauge JSON (selection bias signal).
- Reference: report TWO numbers — (a) argmin agreement vs exact-abstract (as
  now, comparable to the supervised track's own B2 numbers), and (b) agreement
  on the CLEAN subset where the exact-optimal completion strict-realizes
  (Python: `solver.solve_plan` on the exact-optimal candidate(s) + `strict_moves`,
  ~0.1 s per candidate at g24r4 depth 0), plus the clean share. (b) is the
  number to compare with the base 0.915 band. If clean share is ~50 %, a
  sample of 200 gives ~100 clean decisions (SE ~ 0.05) — use 300-400 if the
  gauge is meant to drive decisions.
- Structural fix (post-tonight): switch the exact side to
  `replay_backward_decision` at depth 0 (per-candidate 4000-iteration cap,
  ALL candidates, no wandering, deterministic); it labels exactly the decision
  the self-play group is about.
- `--time-cap` consistency: the audit only scores matched groups, so partial
  results do not break `audit_descent`; the JSON's `sample` vs
  `exact_labeled` shows the shortfall. The bias (finished = easy) is the
  problem, not consistency. `ep.proc.kill()` on an already-exited engine is
  safe (Popen.send_signal polls first); if the engine crashes early the loop
  still waits to the deadline (wasted wall only).
- Tonight: iteration 2/3 gauges (`selfplay_iter.slurm:69`: 80 / 24 / 1800 s)
  will produce biased ~0.5 numbers and burn <= 30 min each; harmless to the
  loop (non-fatal), do not log them as fidelity without the H1/M1 caveats.
  Job 4689629 can be cancelled: the same information (198 instances at cap
  64) is already on disk and audited (0.510).

## C. Training in B2 with no base anchors

- Forgetting risk is low per iteration but real over the chain: iteration 1
  fine-tuned 6 epochs at lr 1e-4 on 5.8 k records (policy 80 s, value 210 s
  of GPU), the picked checkpoints were epoch 1 of 6 for both nets (val regret
  rose after epoch 1 — 109 val groups, noisy), warm from a pair trained on
  ~10^5-10^6 records. Iterations 2/3 train on the window (up to ~17 k records
  labeled by three different net generations). Drift shows first as bench
  regression on the graded set (already measured each iteration) — that is
  the right monitor; the gauge is not (H1).
- Mixing base anchors: labels ARE vocabulary-relative (base optimum >= B2
  optimum for the same decision) and base groups contain no
  transient/by-ref candidates, so base `is_optimal` targets can contradict B2
  targets. But note the self-play labels themselves sit +3.9 above the exact
  B2 optimum on average, i.e. the vocabulary offset (typically 0-2) is
  smaller than the loop's own label looseness. If stability is needed, a
  small base anchor (e.g. 10-20 % of the batch, value net only) is unlikely
  to hurt in practice — yet it does violate the house rule in a way that
  matters for the POLICY (candidate-set semantics), so prefer B2 anchors:
- B2 exact anchors available (`scaling/data/<cfg>/backward_b2*.rust.jsonl`,
  same 18-field schema, board ids = the config's standard pool 0-2999 /
  0-1049): g16r4 cap20000 413 k records (12.7 % by-ref), g16r4 cap5000
  377 k (7.1 %), g24r8 cap20000 226 k (10.5 %), g16r6/g16r8/g32r4 cap20000
  ~110-140 MB each. No g24r4 B2 corpus exists. Caveat (H1 again): these are
  exact-abstract labels — ~half of their optima are unplayable, which is what
  made the supervised B2 nets rank the vocabulary badly. Anchoring the value
  net on them would pull it back toward the reference the loop is trying to
  leave. Recommendation: keep ANCHOR=0 for the B2 loop; if drift appears,
  anchor with the loop's own earlier B2 iterations (WINDOW already does) or
  with a small CERTIFIED B2 seed (e.g. `nn_labeler.descent --vocab b2` output,
  `nn_labeler/jobs/b2_payoff.slurm` part 2, which is certified by
  construction) rather than exact-abstract corpora. If a by-ref-rich signal is
  wanted for the policy specifically, g16r4 cap20000 (12.7 % by-ref) is the
  best available corpus and the size-free nets accept 16x16 boards.

## D. Anything else that could invalidate tonight

- Auto hand-off: `nets.txt` written to both IT and RES (`selfplay_iter.slurm:121`),
  `best_ckpt` = newest `epoch=*.ckpt` (top-1 by val_regret, so newest = best);
  PREV_BENCH for k=2 defaults to it1's `bench_g24r4_astar.json` (exists). OK.
- B2 bench flags: `--vocab b2 --anytime` (`:135`) map to
  `make_solver('b2')` + `park_hook(max_parks=2, pairwise, multi_slide)` +
  `byref=True` and no prefix filter (`bench.py:87-92,137-151,164-170`),
  mirroring `compare.py:329-332,386-405` (arena `--backward-anytime
  --backward-b2 --backward-byref`). `wall_sets`/size handling equals the
  arena's. OK. Only L4 (payload byref flag).
- Board ids: 8000 + 60(k-1) if BOARDS=60 is exported (8060-8119, 8120-8179);
  with the default 120 they are 8120-8239 / 8240-8359 — no overlap with
  iteration 1 either way. Result dirs `<cfg>_b2_iter<k>` are separate from the
  base loop's. OK.
- Records/buffer: `spr.buffer --tag _b2` concatenates only B2 iterations. OK.
- Operational: M4 (env hand-off), M5 (4 h resume job), and the qgpu state
  (48 maint / 16 drained / 8 draining at review time; cooling reservations
  end 18:00 daily; Slurm estimates 4689626 at 2026-08-20 18:05).

## E. Commands behind the numbers (all login-node, seconds)

- Audit of the existing 198 exact results (cap 64) vs the saved self-play subset:
  `python -m nn_labeler.audit_descent --exact <depth-0 records extracted from
  results/selfplay/g24r4_b2_iter1/gauge.work/engine.results.jsonl> --descent
  results/selfplay/g24r4_b2_iter1/gauge.work/selfplay_subset.jsonl` -> 0.510 /
  0.411 / gap 3.92 / 40 % zero / 2 unmatched argmins; with the exact side
  truncated to its first 24 score-ordered labeled candidates -> 0.510 / 0.414 /
  6 unmatched.
- Iteration histogram from `engine.results.jsonl` (`iters` field).
- Root candidate counts: Python `propose_b1 + _reference_helpers` on 60 sampled
  instances from the same subset (lean boards `runs/spr/boards/g24r4_b2_iter1`).
- Realizability of exact optima: Python `make_solver('b2').solve_plan` on the
  exact-optimal candidates of 16 instances (28 solves, 2 s) + `strict_moves`:
  13/28 realizable; Python plan cost == Rust ctg in all 28.
