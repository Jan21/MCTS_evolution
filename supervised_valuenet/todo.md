# todo — subgoal self-play: Phase A parity, then Phase B scaling

Goal: **Phase A** — on bench450 (16×16, 4 robots) the backward subgoal planner matches the
forward move planner in *strict realized moves* (a subgoal plan converted into an actual legal
move sequence under full game physics; the honest metric per `COMPARISON.md`). **Phase B** —
show the backward advantage *grows* with scale (robot axis, grid axis).

**Why this order.** The loop cannot currently see the deciding metric, so instrumentation comes
first (1). Two measured defects account for ~70% of today's strict failures — a proposal bug
(self-support, 44% of failures) and a realizer gap (atomic segments, ~25%) — both are hours of
work, so they land next (2, 3) and every downstream sizing shifts after them. Then training
pressure targets what remains (4), the official gate is re-run with literature-grade reporting
(5, 6), and only then does scaling spend real compute (7, 8) — scaling before the executability
fix would worsen the comparison, since the failure mode deepens with depth (strict-solve
95% → 20% from d\* 1–3 to 10+). The vocabulary extension (9) is last and evidence-gated; 10 is
standing hygiene.

## Constraints (every item respects these)

- **Frozen methodology**: plans stay subgoal DAGs found by propose → policy-rank → value-score →
  commit best-first search. Allowed: tuning, training, evaluation, bug fixes (item 2 is a bug
  fix in the proposal filter), realization improvements that execute the plan's own declared
  semantics (item 3), new candidate types inside the propose step (item 9 — last,
  evidence-gated). Not allowed: redesign or pivot to move-level decisions.
- **Pure NN at inference**: no exact-solver calls in the planner. Physics checks
  (`simulate.slide`-based realization) are the game rules, not an oracle — allowed, already part
  of the eval protocol. Oracle only for training labels and the eval reference d\* (the exact
  optimal move count per benchmark instance).
- **GPU**: the whole project holds AT MOST ONE GPU, only if free; the self-play loop holds that
  slot now. Any GPU item reuses that single slot (between rounds, or by takeover at an iteration
  boundary) — never a second GPU. Benchmarks/evals of these ~1M-param nets run on CPU
  (`OMP_NUM_THREADS=8`, `torch.set_num_threads(8)`).
- **Machine-readable outputs**: every result lands as JSON/JSONL alongside any markdown — a
  later deliverable is an extended HTML comparison viewer built from these files.
- **Never compare abstract plan cost to d\*** — abstract cost (`plan.cost()`, the relaxed count
  that assumes blockers away) can undercut the true optimum (65/450 plans do). Exploratory
  `eval.compare` runs pass explicit `--out`/`--md`; root `COMPARISON.md` is rewritten only by
  the official rerun (item 5).
- **Plain language, neutral tone**; define technical terms at first use; cite file:line.

---

## 1. In-loop strict metrics (L1) — do first, ~30 lines

**What**: every generation round and probe reports the fraction of winning plans passing strict
realization, their strict costs, and a strict-regret probe; persist per-iteration stats as JSON.
**Why**: executability is measured nowhere in the loop — it can reinforce unplayable plan
patterns invisibly (`analysis/pass1_selfplay.md` §1, claims 1–4 verified; plan in §2).
**How** (all in `subgoal_selfplay/`; full line-level plan in pass1 §2):
- `selfplay.py::play_instance` (222–242) computes `stx = strict_moves(...)` for every winner and
  returns it plus an explicit plan-found indicator; `generate_iteration` (247–291) adds
  `strict_pass_rate`, `mean_strict_cost`, `mean_strict_minus_abstract` with explicit
  denominators `n_plan_found` vs `n_kept` — fixing the conflation where "found but filtered"
  would silently deflate `solve_rate` (pass1 §2a step 3).
- `nn_astar_from` (157–182) returns the completed plan object, not `int(plan.cost())` (line
  174); two call sites adapt (`label_chain` 203–208; `probe_eval` `train_iterate.py:136–140`).
- `probe_eval` adds `strict_solved`, `strict_solve_rate`, `mean_strict_regret`, `ref_fail`;
  reference = strict realization of the exact solver's plan — eval-only oracle, trend gauge.
- Replace `getattr(cfg, "strict_filter", True)` at `selfplay.py:233` with `cfg.strict_filter` —
  the fallback `True` silently turns filtering ON for any config lacking the field (pass1 §1).
- Write each iteration's stats + probe dicts to `<out-dir>/iterN_stats.json`.
**Measure**: next run prints `gen_strict`/`probe_strict` and writes `iterN_stats.json`;
first-iteration gen strict-pass should land near **0.40** (pass1 §6 measured 40.2%). No
training-behavior change.

## 2. Self-support proposal fix + re-measure executability

**What**: fix the bug that lets a plan schedule a robot to bounce off *itself*, then re-run the
plain and anytime backward benches on CPU.
**Why**: the dominant mechanism (`analysis/pass2_executability.md` §4): 92/211 strict failures
declare a support cell occupied by the mover itself — physically impossible, yet rewarded by
the abstract cost model (self-placement costs 0); 101/211 failing plans carry the pattern vs
2/239 passing — a near-perfect static predictor of failure.
**How**: `skeleton/astar.py:133–135` — the helper filter compares `Robot_at` dataclasses
(position AND color), so a mover carrying its *planned* position survives in its own helper
list; filter by **color** instead (2 lines; same fix for the target-robot append, line 134).
Optional: a static "support color == bottleneck mover color" reject in plan validation. Then
re-bench (CPU): `head -150 eval/data/bench450.jsonl > eval/data/bench450_first150.jsonl`, then
`OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python -m eval.compare
--instances eval/data/bench450_first150.jsonl --expansions 1200 --k 5
--backward-policy checkpoints_backward/policy_v2.ckpt --backward-value checkpoints_backward/value_v2.ckpt
--forward-ckpts "" --device cpu --out eval/results/comparison_backward_postfix.json
--md eval/results/COMPARISON_postfix.md`, and again with `--backward-anytime` (out
`..._postfix_anytime.json`). Full 450 when time permits (pre-fix anytime ~3 h CPU).
**Measure**: postfix JSONs vs pass3's matched-150 baselines — plain 54.0% / regret 1.383 / 2.1
expansions; anytime 83.3% / 3.048 / 70.0 (`analysis/pass3_anytime_bench.md` §2). Re-size the
unrecoverable residue here (feeds item 9).
**Expected**: large strict-solve gain; anytime rejections/expansions collapse (the doomed
family stops eating budget). Caveat: the nets were trained on the old proposal stream, so
quality may wobble until item 4 retrains — the measurement stands either way.

## 3. Two-phase strict realizer (A/B offline, then adopt)

**What**: execute supported segments as approach → place support → bounce (the semantics the
plan's own cost model already uses) instead of one atomic segment with the support pre-placed.
**Why**: pass2 §5 variant D fixes 52/211 failures (~25%) with **no repair moves added** — these
are realizer false-negatives, not bad plans; fixed instances realize at mean regret 1.90
(competitive with 1.20 on currently-solved).
**How**: `eval/realize.py::strict_moves` forces support-before-segment (deps, realize.py:204–217)
and runs segments atomically; split supported segments to mirror `_abstract_segment_moves`'
second route (realize.py:44–62) / `segment_realizable` (simulate.py:88–98). A/B offline first on
the 211 pickled failing plans: `analysis/artifacts/failing_plans.pkl` (classifier
`analysis/artifacts/diag_failures.py`; regeneration is deterministic — pass2 §2 reproduced the
stored run exactly on CPU). Regression bar: all 239 passing plans still realize with identical
move counts. Output: `eval/results/realizer_twophase_ab.json`.
**Measure**: A/B JSON shows ≈52 newly-fixed at ≈+1.90 regret, 0 regressions; then re-run item
2's 150-instance protocol with both fixes live (`..._postfix2.json`) — the pre-training baseline.
**If clean**: adopt as THE realizer; document as a realizer improvement (output remains a
verified, fully legal joint-physics move sequence — fair under the protocol); both systems get
re-baselined at item 5.

## 4. Training pressure toward playable plans

**What**: primary arm — anytime-in-generation (L3: the generator discards a completed plan that
fails strict realization and keeps searching, returning the first playable winner). Cheap
ablation arm — `--strict-filter`. Strict-cost labels only if both plateau.
**Why**: even a 100%-solve anytime planner misses the regret bar unless the nets rank playable
plans *first* — rescues average regret 6.11 and deep bins cost 130–165 expansions (pass3 §6).
Check-inside-search is the literature-standard architecture (`analysis/pass5_research.md`
Q1/L3). Filter-alone likely starves and skews shallow: strict-pass on the training distribution
is 40.2%, i.e. ~60% of records dropped (pass1 §6, §7.4) — it is the ablation, not the bet.
**How (L3, primary — pass1 §4 plan)**: `subgoal_selfplay/config.py` adds
`gen_realize_check: bool = False`; CLI `--gen-realize-check`; `nn_astar_traced`
(selfplay.py:118–154) gains `realize_check=None` — the completion branch (133–140) discards
failing complete plans and keeps popping; return `None` on exhaustion (NOT compare.py's
first-failed fallback); `play_instance` builds a caching closure over `strict_moves`, reusing
the cached result as `stx`; count `n_anytime_rejected`. Run — taking over the single GPU slot
from the control run at an iteration boundary (the control arm's value is its logged history):
`PYTHONPATH=. python3 -m subgoal_selfplay.train_iterate --policy checkpoints_backward/policy_v2.ckpt
--value checkpoints_backward/value_v2.ckpt --out-dir subgoal_selfplay/runs_warm_anytime --gen-realize-check`
**How (L2, ablation)**: add `--strict-filter` to argparse (`train_iterate.py` after `--epsilon`,
~line 205) + `strict_filter=a.strict_filter` in `Config(...)` (~222–234); run
`runs_warm_strict/` with `--instances-per-iter 4000-5000` to restore iter0-scale volume.
Sequential in the same GPU slot after the anytime arm.
**Strict-cost labels (only if arms plateau below the item 5 bar)**: the 6-function change in
pass1 §3b (`strict_labels` config+CLI; `nn_astar_from` returns plan — done in item 1;
`play_instance` `win_cost = stx`; `label_chain` realizes sibling completions, counter
`n_sibling_unrealizable`); unit-sound per pass1 §3b. Value-net bins are safe — 50 bins,
observed cost-to-go max 44, targets clamp at 49; log a `ctg >= 49` counter; do **NOT** raise
`num_classes` on warm runs (breaks checkpoint loading; pass1 §5).
**Measure**: item 1's `iterN_stats.json` trajectories — gen `strict_pass_rate` up, probe
strict-regret down, vs the control arm (`runs_warm/`) over the same iteration count.

## 5. Parity gate on bench450 (the Phase A bar)

**What**: official full-450 rerun once 1–4 have landed.
**Why**: pass3's verdict — anytime alone = 83.3% / 3.05 / 70 expansions — is short on all three
criteria; items 2–4 target exactly the rescue regret and deep-bin expansions.
**How**: `eval.compare` on `eval/data/bench450.jsonl`, 1200 expansions, k=5, best-arm
checkpoints; backward in **anytime mode** as the official row (fair per pass3 §5's audit: the
check costs zero NN passes; wall-time disclosed), plain mode as an ablation row; include the
four forward checkpoints from the `COMPARISON.md` run line so one artifact carries both systems
under one protocol stamp. CPU (8 threads), background, ~a day wall-clock. The only run allowed
to rewrite root `COMPARISON.md`. Outputs `eval/results/comparison_*.json`.
**Measure / bar**: strict solve ≥ 97.5%, strict regret ≤ 0.35, mean expansions ≥ an order of
magnitude below forward's 36 — with item 6's accounting attached. Watch: regret on rescued
instances (was 6.11) and expansions at d\* ≥ 7 (was 130–165).

## 6. Reporting upgrade (required for any headline claim)

**What**: solve-rate-vs-budget curves at several expansion caps; per-system total accounting
(NN passes, realization-check wall-time, seconds/instance); solve-rate-by-d\* curves — all JSON
for the HTML viewer.
**Why**: single-budget-point comparisons and subgoal-level-only counting are what the
literature discounts; the "2.9 expansions" headline must carry total accounting or it is the
inflated number the methodology study warns about (pass5 Q3).
**How**: run `eval.compare` at `--expansions` ∈ {10, 30, 100, 300, 1200} (both systems, k=5,
CPU, background over days); add a per-instance realization-wall-time counter to `compare.py`'s
backward path (eval tooling — allowed). NN passes are derivable: 1 policy + 1 batched value
pass per expansion in both systems (`COMPARISON.md` protocol); rejected complete-plan pops cost
0 NN passes and land in wall-time — disclose `plans_rejected` alongside. Aggregate to
`eval/results/budget_curves.json` + `eval/results/accounting.json` (totals + by-d\*-bin tables).
**Measure**: the two JSONs exist and back every number the item 5 rerun quotes.

## 7. Robot axis: g16r6 (16×16, 6 robots)

**What**: train the 3 nets, build the full bench, run the comparison.
**Why**: data complete (129,423 backward / 1,038,827 forward records); everything after data is
TODO; ~one working day inside the single GPU slot (`analysis/pass4_scaling.md` §1, §5, §7).
Robot count is the theory-backed hardness knob (pass5 Q4: W[SAT]-hard in robots), and the exact
solver already strains at r6 (3/6 smoke instances failed at 200k-expansion/60 s caps) —
oracle-failure rate is itself a headline scaling result.
**How** (pass4 §5 exact commands; GPU steps reuse the single slot after/around item 4; run
after parity, or interleaved whenever the slot is idle): trainings via
`scaling.train --config g16r6 --system {backward-value,backward-policy,forward}` (value:
`--batch-size 8 --max-per-group 32`; forward: `--patience 5`; all `--epochs 50`) or the
one-liner `python -m scaling.run_config --config g16r6 --steps
train-backward-value,train-backward-policy,train-forward,bench,compare --execute --gpu <idx>`;
bench `python -m scaling.bench --config g16r6 --per-board 3 --seed 1` — CPU, **single process**
(the sampler consumes one seeded RNG sequentially; sharding would change the instance set),
overlappable with training; compare with the env prefix run_config prints (`RR_GRID=16
RR_ROBOTS=6 RR_WALLS=48 RR_ENV_DIR=.../environments_g16r6`), instances
`scaling/data/g16r6/bench.solved.jsonl`, out `scaling/results/g16r6/comparison.json` + md.
**Measure**: `scaling/results/g16r6/comparison.json` + bench meta's `n_oracle_failed`; report
labeling cost alongside quality (pass4 §3 anchors: backward ≈7.8 s·core/instance at G=16).

## 8. Grid axis unlock, then g24r4

**What**: make realization size-agnostic (~15 lines), flip the gate, run the g24r4 row.
**Why**: the single piece of engineering left in the study (pass4 §5); g24r4 data is complete
and waiting; the grid axis is the likelier crossover axis (addendum §4).
**How** (pass2 §7 patch): change the four env-carrying entry points to `size=None` and infer
`size = isqrt(len(grid_data))` (assert perfect square): `simulate.wall_sets` (simulate.py:20),
`simulate.verify_plan` (:102) + `segment_realizable` (:74), `eval.realize.strict_moves`
(realize.py:182) + `abstract_moves` (:65). Low-level helpers keep explicit `size`; **no caller
changes needed** (`eval/compare.py:209,226–229`, `subgoal_selfplay/selfplay.py:236–237` become
correct automatically). Inferring from the board is safer than threading `RR_GRID` (a
disagreeing `RR_GRID` mis-decodes boards silently). Regression bar: a bench450 slice re-runs
unchanged at G=16; a handful of G=24 strict realizations replay move-by-move legal. Then flip
`skip_back = g != 16` at `scaling/run_config.py:106` and run the g24r4 row (same pattern as
item 7; backward-value `--batch-size 4 --max-per-group 16`).
**Measure**: G=16 regression diff = 0; `scaling/results/g24r4/comparison.json` with backward rows.

## 9. L4 vocabulary extension — only for the measured residue

**What**: a blocker-clearing candidate type in `skeleton/heuristics.py::propose`, built only if
a coherent blocker-shaped failure class survives items 2–4.
**Why**: the pre-fix unrecoverable set was 16.7% (18 frontier-exhausted + 7 budget-exhausted of
150; pass3 §4) — but pure robot blockage was only 14% of pre-fix failures (pass2 §6), so the
original blocker hypothesis over-weighted, and items 2–4 shrink and reshape the residue.
**How**: after item 2's postfix anytime run (and again after item 4), classify residual
failures with the pass2 tooling pattern (`analysis/artifacts/diag_failures.py`) into
`eval/results/residual_failures_postfix.json`, recording the build decision and measured class
sizes. If built: a new candidate type extends the propose step's vocabulary — no loop change
(allowed under the freeze, last in line).
**Default**: if the residue falls below ~3% of bench450, do not build it.

## 10. Run hygiene / control arm (standing)

- Keep `subgoal_selfplay/runs_warm/` frozen as the control arm ("self-play on the abstract
  objective" vs "on the strict objective" is itself a reportable ablation): log, checkpoints,
  every `iterN_records.jsonl`.
- Every new arm gets a distinct out dir (`runs_warm_anytime/`, `runs_warm_strict/`, ...); never
  concatenate records across arms — abstract vs strict labels differ by a systematic
  ~+0.6-move shift and the grouping key is instance-blind (pass1 §7.1).
- One boards dir per configuration; never point `RR_ENV_DIR` across configs. Maintain
  `subgoal_selfplay/arms_index.json`: run dir, flags, label semantics, iterations, headline
  probe numbers per arm.

---

## Success bars

- **Phase A (bench450, `COMPARISON.md` protocol)**: strict solve ≥ 97.5% AND strict regret
  ≤ 0.35 — inside the forward band (naive 84.7%/0.354; candidate-scored 100%/0.067) — while
  keeping mean expansions at least an order of magnitude below forward's 36. Stretch: ≥ 99% /
  ≤ 0.1. The claim ships with honest total accounting (item 6): multi-budget solve-rate curves,
  per-system NN-pass counts, realization-check wall-time disclosed, seconds/instance,
  solve-rate-by-d\* curves.
- **Phase B**: at ≥ 1 scaled configuration, backward (strict moves) beats the best forward
  system at matched budget — or matches it with ≥ 10× fewer expansions and materially cheaper
  training labels. Labeling CPU-time and oracle-failure rate are first-class results. A clean
  negative (no crossover on either axis) is also a reportable outcome.

## Pointer map

- Rationale + lever definitions (L0–L4): `addendum.md` (this file extends and partially
  supersedes its sequencing).
- Evidence: `analysis/pass1_selfplay.md` (loop verification, L1–L3 line plans, label safety);
  `analysis/pass2_executability.md` (failure taxonomy, self-support bug, two-phase realizer,
  grid audit); `analysis/pass3_anytime_bench.md` (anytime bench, fairness audit, residue);
  `analysis/pass4_scaling.md` (readiness, commands, costs); `analysis/pass5_research.md`
  (literature grounding, reporting requirements).
- A/B artifacts: `analysis/artifacts/` (failing-plans pickle, diagnosis scripts/JSONs,
  150-instance runner) — regenerable deterministically per pass2 §2.
- Official numbers + protocol: `COMPARISON.md` (rewritten only by item 5); forward-side model
  detail: `RESULTS.md`; machine-readable per-instance rows: `eval/results/*.json`.
