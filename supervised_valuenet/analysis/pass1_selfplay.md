# Deep pass 1 — the subgoal self-play loop (`subgoal_selfplay/`)

Scope: the self-play training loop and its integration with the trainers and the eval
planner. Builds on `addendum.md`. No repo file was modified; the one scratch script and
its output live in this scratchpad directory. All paths below are relative to
`/mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet/`.

---

## 1. Verification of the addendum's claims about the loop

| # | Claim (addendum §2) | Verdict | Evidence |
|---|---|---|---|
| 1 | `strict_filter` is OFF for the CLI-started run | **VERIFIED** | Field default `False` at `subgoal_selfplay/config.py:50`. The gate reads it at `subgoal_selfplay/selfplay.py:233`. The CLI (`subgoal_selfplay/train_iterate.py:190-234`) exposes **no** `--strict-filter` flag, so no run started through `main()` can turn it on. The live run's command line (PID 1405327) is `python3 -m subgoal_selfplay.train_iterate --policy checkpoints_backward/policy_v2.ckpt --value checkpoints_backward/value_v2.ckpt --out-dir subgoal_selfplay/runs_warm` — no such flag. `subgoal_selfplay/DESIGN.md:95-97` documents the flag as "OFF by default and not part of the baseline". |
| 2 | All training labels are abstract `plan.cost()` units | **VERIFIED** | Winner: `win_cost = int(win_plan.cost())` at `selfplay.py:239`; chosen-child label `win_cost - fixed_g` at `selfplay.py:201`; sibling label `done - fixed_g` at `selfplay.py:208`, where `done` is `nn_astar_from`'s return `int(plan.cost())` (`selfplay.py:174`). `plan.cost()` sums plan-edge costs (`partial_plan.py:46-48`) — the relaxed blocker-clearing model (`eval/realize.py:8-14`). |
| 3 | Probe regret is abstract-vs-abstract | **VERIFIED** | Probe achieved cost `ac` comes from `nn_astar_from` (abstract, `selfplay.py:174`), called at `train_iterate.py:136-137`; reference is `int(opt.cost())` of the exact solver's plan (`train_iterate.py:131,140`) — also abstract. `regrets.append(ac - int(opt.cost()))` at `train_iterate.py:140`. |
| 4 | No executability stat anywhere in the loop | **VERIFIED** | `strict_moves`/`strict_filter` appear in the package only at `selfplay.py:233-237` (the default-off drop gate; when off, `strict_moves` is never called). Generation stats (`selfplay.py:280-290`) hold n_instances / n_solved / solve_rate / mean_plan_cost / n_records / n_groups / mean_records_per_instance / mean_group_size / n_sibling_fail — nothing about realization. Probe stats (`train_iterate.py:141-144`) hold n / solved / solve_rate / mean_regret / optimal / opt_fail — likewise nothing. |

Context numbers quoted by the addendum also check out against the live artifacts:
generation solve rate 0.966 and probe regret 1.48 → 1.18 appear verbatim in the running
job's log (`[gen 0]`, `[probe warm-baseline]`, `[probe iter0]` lines); bench450 backward
239/450 solved (53.1%), regret 1.201, 2.9 mean expansions, 0.34 s/inst, and 65
negative-abstract-regret instances are in `eval/results/comparison_backward.json`;
forward candidate-scored 450/450 / 0.067 / 36.0 in `eval/results/comparison_forward.json`.

One latent quirk found while verifying claim 1: the gate is written
`if getattr(cfg, "strict_filter", True):` (`selfplay.py:233`) — the *fallback* default is
`True`, opposite to the `Config` default `False`. Any caller that passes a config object
lacking the field (an ad-hoc namespace, an older pickled config) silently gets filtering
ON. Harmless for `Config`-driven runs, but worth normalizing to `cfg.strict_filter`
whenever this code is next touched.

---

## 2. L1 — add strict-executability metrics to the loop (change plan, not applied)

Goal: every generation round and every probe reports how many winning plans are legally
playable, at negligible cost (one `strict_moves` call per winner — pure CPU slide/BFS
physics, `eval/realize.py:182-246`).

### 2a. Generation stats

1. `subgoal_selfplay/selfplay.py` — module imports (near line 36-41): add
   `from eval.realize import strict_moves` (currently imported lazily inside the gate at
   `selfplay.py:236`; the lazy import can then go).
2. `play_instance(env, state, solver, policy, value, env_id, dev, cfg, rng)`
   (`selfplay.py:222-242`): compute the winner's strict cost unconditionally and return
   it. Replace lines 233-239 logic with:
   - `stx = strict_moves(env, state, win_plan, log=None)`
   - `if cfg.strict_filter and stx is None: return [], None, [], 0, None`
   - `win_cost = int(win_plan.cost())`
   - return becomes `(records, win_cost, group_sizes, n_fail, stx)` (5-tuple; the
     unsolved early-return at `selfplay.py:231` becomes `[], None, [], 0, None`).
   Sole caller: `generate_iteration` at `selfplay.py:271-272`.
3. `generate_iteration(cfg, policy, value, rng, boards=None)` (`selfplay.py:247-291`):
   unpack the 5th element; accumulate `n_strict_pass`, `strict_costs` (passers) and
   the passers' abstract costs. Extend the stats dict (after `"n_sibling_fail"`,
   `selfplay.py:289`) with:
   - `"strict_pass"`: count of winners whose `stx is not None`
   - `"strict_pass_rate"`: `round(strict_pass / n_plan_found, 3)`
   - `"mean_strict_cost"`: mean of `strict_costs`
   - `"mean_strict_minus_abstract"`: mean of `stx - win_cost` over passers (scale-shift
     monitor for L2).
   Denominator caveat: once `strict_filter` is ON, "plan found" and "kept for training"
   diverge — the current gate returns the same `win_cost=None` for both "search failed"
   and "found but dropped" (`selfplay.py:233-238` vs `230-231`), which would silently
   deflate `solve_rate`. The 5-tuple should therefore carry `stx` alongside a
   plan-found indicator (e.g. return `win_cost` for any found plan and let
   `generate_iteration` count `n_plan_found`, `n_kept = plans surviving the filter`,
   and `strict_pass` with explicit denominators; only kept plans contribute records).

### 2b. Probe

4. `subgoal_selfplay/selfplay.py::nn_astar_from` (`selfplay.py:157-182`): change the
   return at line 174 from `int(plan.cost())` to the completed `plan` object (docstring
   lines 159-164 updated). Signature unchanged:
   `nn_astar_from(env, state, solver, policy, value, env_id, dev, start_plan=None, k=5, max_iters=300)`.
   Exactly two call sites adapt:
   - `label_chain` sibling completion (`selfplay.py:203-208`): `done` is now a plan;
     use `int(done.cost()) - fixed_g` (behavior identical).
   - `probe_eval` (`train_iterate.py:136-140`): `ac_plan = nn_astar_from(...)`;
     `ac = int(ac_plan.cost())` where previously `ac` was the int.
5. `subgoal_selfplay/train_iterate.py::probe_eval` (`train_iterate.py:107-146`): add
   `from eval.realize import strict_moves` (module top) and, in the solved branch
   (after line 138):
   - `stx = strict_moves(env, st, ac_plan, log=None)` → count `strict_solved`;
   - `ref = strict_moves(env, st, opt, log=None)` (the oracle plan from line 131,
     realized the same way — still eval-only oracle use) → if both present,
     `strict_regrets.append(stx - ref)`; count `ref_fail` when `ref is None`.
   Extend the out dict (`train_iterate.py:141-144`) with `"strict_solved"`,
   `"strict_solve_rate"` (denominator `n`), `"mean_strict_regret"`, `"ref_fail"`.
   Caveat to log alongside: the reference is the strict realization of the exact
   solver's *plan*, not the move-optimal d\* — the honest headline metric stays the
   bench450 protocol; this probe number is a trend gauge.
6. `iterate` summary line (`train_iterate.py:181-185`): append
   `gen_strict={stats['strict_pass_rate']}` and `probe_strict={probe.get('strict_solve_rate','-')}`.

Total ≈ 30 lines, matching the addendum's estimate. No training-behavior change.

---

## 3. L2 — strict-filter run and strict-cost labels (change plan, not applied)

### 3a. `strict_filter=True` variant — CLI plumbing IS needed

The field exists (`config.py:50`) and the gate works (`selfplay.py:233`), but
`main()` (`train_iterate.py:190-234`) never sets it, so a flag must be added:

1. `train_iterate.py` argparse (after `--epsilon`, line 205):
   `p.add_argument("--strict-filter", action="store_true", default=d.strict_filter, help="drop winners failing strict legal realization before labeling")`
2. `Config(...)` construction (`train_iterate.py:222-234`): add
   `strict_filter=a.strict_filter,` (e.g. with the epsilon/walk args on line 228).
3. Launch as a separate run, distinct out dir (control arm stays running):
   `PYTHONPATH=. python3 -m subgoal_selfplay.train_iterate --policy checkpoints_backward/policy_v2.ckpt --value checkpoints_backward/value_v2.ckpt --out-dir subgoal_selfplay/runs_warm_strict --strict-filter`
   (Note: under the 1-GPU project cap this waits for the current run's slot, or runs
   `--device cpu` at reduced throughput.)
4. Optional hygiene in the same touch: replace `getattr(cfg, "strict_filter", True)` at
   `selfplay.py:233` with `cfg.strict_filter` (see §1 quirk).

With L1 in place, filtering ON reuses the already-computed `stx` (§2a step 2) — no
double realization.

### 3b. Strict-cost labels (stronger variant) — every touched function

Label = legal move count instead of relaxed plan cost. Enumerated changes:

1. `subgoal_selfplay/config.py` — new field under `strict_filter` (line ~53):
   `strict_labels: bool = False`. Semantics: implies the winner must realize (a `None`
   strict cost cannot label), i.e. it subsumes `strict_filter` for the winner.
2. `subgoal_selfplay/selfplay.py::nn_astar_from` (`selfplay.py:157-182`) — must return
   the completed **plan object**, not its cost (return at line 174; same change as L1
   §2b step 4, which is a prerequisite): signature stays
   `nn_astar_from(env, state, solver, policy, value, env_id, dev, start_plan=None, k=5, max_iters=300)`,
   return type becomes `PartialPlan | None`.
3. `subgoal_selfplay/selfplay.py::play_instance(env, state, solver, policy, value, env_id, dev, cfg, rng)`
   (`selfplay.py:222-242`) — winner cost source:
   `win_cost = stx if cfg.strict_labels else int(win_plan.cost())`, with the instance
   dropped when `cfg.strict_labels` and `stx is None` (the L1 structure already computes
   `stx`).
4. `subgoal_selfplay/selfplay.py::label_chain(env, state, solver, policy, value, env_id, dev, win_cost, chain, cfg)`
   (`selfplay.py:187-219`) — sibling branch (lines 203-208) becomes:
   - `done = nn_astar_from(..., start_plan=cp, k=cfg.k_top, max_iters=cfg.sibling_iters)`
   - `if done is None: n_fail += 1; continue`
   - `if cfg.strict_labels:` → `c = strict_moves(env, state, done, log=None)`;
     `if c is None: n_unrealizable += 1; continue` (recommend a separate counter from
     `n_fail`, surfaced in stats as `"n_sibling_unrealizable"`)
   - `else: c = int(done.cost())`
   - `labeled.append((rec, c - fixed_g))`
   Return grows to `(records, group_sizes, n_fail, n_unrealizable)`; sole caller is
   `play_instance` (`selfplay.py:240-241`).
5. `subgoal_selfplay/train_iterate.py::main` — `p.add_argument("--strict-labels", action="store_true", default=d.strict_labels)` + `strict_labels=a.strict_labels` in the
   `Config(...)` call.
6. `subgoal_selfplay/train_iterate.py::probe_eval(cfg: Config, policy: PolicyTF, value: LoopedValueNet, tag: str) -> dict`
   — only the adaptation to `nn_astar_from` returning a plan (L1 §2b step 5); probe
   reporting is unchanged by the label switch. `main() -> None` gains the two flags
   listed above; `generate_iteration(cfg, policy, value, rng, boards=None)` and
   `label_chain`'s caller unpack the widened return tuples (§2a step 3, §3b step 4).

Unit analysis (why `c - fixed_g` is sound): `fixed_g` is the abstract cost of the
already-fixed prefix (`nn/generate.py:46-48`), while `c` is the strict total of the whole
completed plan; the difference is "strict remaining + prefix's strict-vs-abstract
discrepancy". Since the SAME `fixed_g` is subtracted from every candidate in a group
(`selfplay.py:197-208`), within-group *differences* — all that `is_optimal`
(`selfplay.py:211-215`), the value rank loss (`train/looped_pc.py:173-181`), and the
policy soft target (`train/policy_tf.py:138-153`) consume — are pure strict-total
differences. At inference `f = fixed_g(child) + value` (`selfplay.py:152`,
`eval/end2end.py:148`) then approximates the plan's strict total, which is a *more*
coherent ranking target than today's mixed abstract f. The addendum's two cautions
(mixed-unit f during transition; warm value net trained on the abstract scale) stand;
sequencing filter-first, labels-only-if-plateau is sensible.

---

## 4. L3 — anytime realization check inside generation (change plan, not applied)

Mirror of `eval/compare.py::_nn_astar_backward` (`eval/compare.py:110-173`): a popped
complete plan failing `realize_check` is discarded and the search continues
(`compare.py:142-147`), with the check built as a caching closure per instance
(`compare.py:205-211`, cache keyed by `id(plan)`).

1. `subgoal_selfplay/config.py`: new field, e.g. `gen_realize_check: bool = False`;
   CLI flag `--gen-realize-check` in `train_iterate.py::main` + `Config(...)` entry.
2. `subgoal_selfplay/selfplay.py::nn_astar_traced(env, state, solver, policy, value, env_id, dev, cfg, rng)`
   (`selfplay.py:118-154`): add trailing parameter `realize_check=None`. In the
   completion branch (`selfplay.py:133-140`), before building the chain:
   - `if realize_check is not None and not realize_check(plan): continue`
   (discard the node, keep popping). On budget exhaustion return `None` — deliberately
   NOT `compare.py`'s `first_failed` fallback (`compare.py:131,146-147,173`), because
   eval must keep `plan_found` truthful while generation wants only playable winners;
   returning `None` simply drops the instance (`selfplay.py:230-231`).
   Optionally count rejections and surface as `"n_anytime_rejected"` in stats.
3. `subgoal_selfplay/selfplay.py::play_instance` (`selfplay.py:229`): build and pass the
   closure when enabled:
   - `cache = {}`; `def check(p): m = strict_moves(env, state, p, log=None); cache[id(p)] = m; return m is not None`
   - `out = nn_astar_traced(..., realize_check=check if cfg.gen_realize_check else None)`
   - when combined with `strict_labels`, reuse `cache[id(win_plan)]` as `stx` (the
     returned plan is the same object the check saw — `_pop_decision`'s advanced plan,
     `selfplay.py:132`), avoiding a second realization.
4. Budget accounting note: `nn_astar_traced` counts every pop in `iters`
   (`selfplay.py:129-130`), including completion pops, whereas `_nn_astar_backward`
   counts only propose-reaching pops as expansions (`compare.py:142-149`). Under anytime,
   each rejected complete plan costs one `iters` tick plus one realization (~ms). With
   completions typically found in ~3 pops against a 1500-pop budget, immaterial — but
   worth one code comment so the two accountings aren't conflated later.

Interaction: with L3 ON, every returned winner passes the check, so `strict_filter`
becomes a no-op for winners — keep the two flags independent so the filter-only arm
remains runnable as an ablation. L3 keeps the instance (more data, on-distribution with
an anytime eval planner) where the filter throws it away; sibling completions still
produce abstract (or strict, per L2) labels from `nn_astar_from`, which performs no
realization check — a known asymmetry worth logging (`n_sibling_unrealizable`) if L2's
strict labels are also on.

---

## 5. Label-scale compatibility with the trainers

**Value net (`train/looped_pc.py::LoopedValueNet`).** HL-Gauss classification over
`num_classes=50` bins 0..49 (`looped_pc.py:136-138,149`), `sigma=1.0`; the live warm
checkpoint confirms `num_classes: 50, sigma: 1.0` (hyper_parameters of
`checkpoints_backward/value_v2.ckpt`). Targets are built from
`ctg.clamp(0, num_classes-1)` (`looped_pc.py:167-171`), and the scalar prediction is the
bin expectation, range [0, 49] (`looped_pc.py:164-165`). So strict costs **cannot
overflow anything — they saturate**: a label ≥ 49 degenerates to top-bin mass, no error.
Headroom check on real data: iter0 self-play records (25,290 in
`subgoal_selfplay/runs_warm/iter0_records.jsonl`) have abstract `cost_to_go` min 2 / mean
12.23 / p95 22 / p99 27 / max 44, with 9 records ≥ 40 and none ≥ 45. The strict shift
measured in §6 (mean +0.60 moves per plan on passers; per-record ctg is bounded by the
plan total) leaves the distribution essentially inside the bins, with at most a handful of deep-tail
records at risk of clamping — acceptable; log a `ctg >= 49` counter if strict labels go
live. Do NOT raise `num_classes` for warm runs: the head's final layer is
`[d_model, num_classes]` (`looped_pc.py:146-148`), so changing it breaks
`load_from_checkpoint` of the v2 warm start; a wider head is only an option for
from-scratch scaled runs.

**Policy net (`train/policy_tf.py::PolicyTF`).** The soft target is
`softmax(-cost/temp)` over the group's candidates (`policy_tf.py:138-153`, temp=1.0 per
the checkpoint hparams) — shift-invariant, no bins, no range limit; `_meta` just takes
`int(r["cost_to_go"])` and the group min (`train/policy_common.py:44-73`). Strict costs
change only the *gaps* between candidates (typically equal or larger), making targets
somewhat sharper. No compatibility issue.

**Grouping.** Both trainers re-group the replay records with
`nn.benchmark.group_by_decision` (`train_iterate.py:84,96`), keyed on
`(env_id, target, target_robot pos, seg_start, seg_end, depth)` (`nn/benchmark.py:47-49`)
— no helper positions, no instance nonce. Cross-instance collisions would merge groups
with incomparable labels; measured on iter0's records: **0** of 3,685 groups mix
different helper sets, so today this is a theoretical risk (see §7 for the buffer-scale
version). 1,070 of 3,685 groups have multiple `is_optimal=True` — those are cost ties
(`selfplay.py:215`), handled by the rank loss's split target (`looped_pc.py:176-180`),
not a bug.

---

## 6. Measurement — strict executability on the training distribution

Script: `measure_strict_pass.py` in this scratchpad dir (CPU-only per the project's
1-GPU cap; `torch.set_num_threads(8)`, `OMP_NUM_THREADS=8`). It samples training-board
instances exactly like generation does (`Config.train_ids()` → boards 1000-1799,
`subgoal_selfplay.start_states.sample_instance`, seed 12345), runs the warm v2
checkpoints through `subgoal_selfplay.selfplay.nn_astar_traced` at the default GEN
budget (`k_top=8, astar_iters=1500, epsilon=0.15`), and realizes each winner with
`eval.realize.strict_moves`.

The job was externally stopped at 90 of the planned 100 instances (~50 min CPU; the
heavy tail is unsolved instances, which burn the full 1500-pop budget — several minutes
each on CPU). Results over the 90 sampled instances (cumulative progress log in
`measure_out.log`, same scratchpad dir):

| quantity | value |
|---|---|
| instances sampled (boards 1000-1799) | 90 |
| solved by the traced GEN-budget search | 87/90 = **0.967** (externally consistent with the live run's `[gen 0]` solve_rate 0.966) |
| winners passing `strict_moves` | 35/87 = **0.402** |
| mean abstract `plan.cost()` of passers | 6.97 |
| mean strict move count of passers | 7.57 |
| mean strict − abstract shift (passers) | **+0.60** |

Implications:

- **`strict_filter=True` would discard ~60% of winners on the training distribution** —
  noticeably worse than the bench450 pass rate (53.1%); the generation budget differs
  (k_top=8 / 1500 pops / epsilon extras vs eval k=5 / 1200) and the board split differs,
  so the two rates are not directly comparable, but the loop as configured is training
  on a majority of unplayable winning plans.
- Volume per round at fixed `instances_per_iter=2000` would fall from ~25.3k records to
  roughly ~10k — likely less, since passers skew shallow and shallow chains yield fewer
  decisions per instance. `--instances-per-iter` around 4000-5000 restores iter0-scale
  volume (at ~2.5× generation cost, still oracle-free).
- The strict scale shift is small (+0.60 mean on passers; +0.64 on bench-solved, §8),
  supporting §5's conclusion that strict labels sit comfortably inside the 50-bin value
  head at the current depth distribution.
- Note the early-sample means were much higher (abstract 12.25 / strict 14.25 at n=10)
  — deep-plan instances dominate small samples; treat per-decade running means as noisy
  and the n=87 figures as the estimate. The stopped run never printed its chain-length
  breakdown; the per-depth decay is documented on bench450 instead (95% → 20% from d\*
  1-3 to 10+, per the addendum's table).

---

## 7. Risks and interactions

1. **Replay-buffer label mixing.** The buffer is in-memory only
   (`deque(maxlen=cfg.replay_iters)`, `train_iterate.py:156`; `replay_iters=5`,
   `config.py:60`) — a fresh strict run starts with an empty buffer, so abstract and
   strict labels can only mix if flags were flipped mid-process (impossible via CLI) or
   if someone seeds a run by concatenating `iterN_records.jsonl` files across runs.
   That must be ruled out: the two label kinds differ by a systematic shift (+0.64
   moves per plan on bench-solved, +0.60 on training-distribution passers, §6) and would
   corrupt within-group comparisons wherever
   `group_by_decision`'s instance-blind key (§5) merged them. Same-run cross-iteration
   merges (same key sampled in two iterations, labels from different-age nets) remain
   possible but were measured at 0 collisions in iter0's 25,290 records.
2. **Warm value net trained on the abstract scale.** Strict labels shift the target up
   slightly; fine-tuning at lr 1e-4 / 4 epochs will track it, but expect the first
   strict iteration's probe to wobble before settling — keep the L1 metrics running so
   the trend is visible. The predicted value's ceiling of 49 (§5) is untouched by the
   shift at current depths.
3. **Data volume under `strict_filter`.** Volume shrinks to the strict-pass rate on the
   *training* distribution — measured 0.402 (§6; bench-distribution pass is 53.1%). At
   iter0 volumes (25,290 records / 3,685 groups from 1,932 winners, per the run log and
   records file) that cuts records per round to ~40% or below; the CLI
   already exposes the remedy (`--instances-per-iter`, `train_iterate.py:200`). Prefer
   raising instances over touching lr/epochs: each record already sees up to
   `replay_iters × epochs = 20` gradient passes across its buffer lifetime, so more
   epochs would amplify over-fitting to stale, older-net labels rather than add signal.
   Note also that in-loop training has no validation loader, no early stopping, and no
   best-checkpoint selection — `trainer.fit(net, dl)` is train-only and the checkpoint
   saved is simply the last epoch (`train_iterate.py:72-78,87-89,99-101`); the probe
   only logs, it gates nothing. A shrinking-data strict run leans harder on this
   unguarded fine-tune, which is another reason to keep volume up rather than epochs.
4. **Selection bias of the filter.** Strict-pass decays with depth (bench: 95% at d\*
   1-3 down to 20% at 10+), so filtering skews training toward shallow plans — exactly
   the regime that is already fine — and starves the deep regime the fix is for. L3
   counters this by keeping the instance and searching for a playable winner instead of
   discarding it; that also matches the anytime eval planner's distribution. This makes
   "filter alone" a slightly self-defeating long-run objective and strengthens the
   addendum's sequencing (filter as the unit-safe first step, anytime + strict labels as
   the real lever).
5. **Group shrinkage under strict sibling labels.** Sibling completions already fail at
   a visible rate (`n_sibling_fail=1476` at iter0 vs ~23.1k sibling attempts — 25,290
   records minus 3,685 chosen, plus the failures — i.e. ~6.4%);
   strict-labeling adds a second drop reason (completion realizes to `None`). Groups
   that shrink to a single candidate contribute nothing to the rank loss (a one-element
   `log_softmax` is 0, `looped_pc.py:176-181`) and only feed the HL-Gauss term — watch
   `mean_group_size` in the stats.
6. **`getattr` fallback trap** at `selfplay.py:233` (§1): fallback `True` vs field
   default `False`. Normalize when touching the file.
7. **Throughput.** `strict_moves` is per-plan milliseconds (BFS over 256 cells per
   segment) — invisible next to sibling completions (300-pop searches per sibling,
   which dominate the ~20 s/instance generation cost observed in the running job's
   iter0 timing). L1/L3 add no meaningful cost; L2's sibling realization is also
   negligible. The binding constraint is the project's 1-GPU cap: a strict variant
   either waits for the current run's GPU slot or runs CPU-only. §6's CPU timing for
   the traced search alone: ~13 s per solved instance, but minutes per UNSOLVED
   instance (full 1500-pop budget) — the 3 unsolved instances consumed well over half
   of the 50-minute wall time; sibling completions would multiply the solved-instance
   cost several-fold.
8. **Footnote.** `checkpoints_backward/value_v2.ckpt` predates the `grid/robots`
   hparams (absent from its hyper_parameters; present in `iter0_value.ckpt`), so loading
   it relies on the process's default 16×16/4 — fine here, a portability trap for scaled
   configs.

---

## 8. Addendum-claim scorecard (this pass's scope)

- strict_filter OFF in the running loop (`config.py:50`, gate `selfplay.py:233`, no CLI flag): **VERIFIED**
- all labels abstract `plan.cost()` (`selfplay.py:239,201,208,174`): **VERIFIED**
- probe regret abstract-vs-abstract (`train_iterate.py:131,140`): **VERIFIED**
- no executability stat in generation or probe (`selfplay.py:280-290`, `train_iterate.py:141-144`): **VERIFIED**
- "~30 lines" for L1: **VERIFIED** (§2 lands within that envelope)
- L2 "flag already exists, small change to `nn_astar_from`": **VERIFIED with a caveat** — the flag exists in `Config` but is *not reachable from the CLI*; plumbing (§3a) is required. The `nn_astar_from` change is as small as claimed (one return statement + two call sites).
- bench context numbers (53.1%, 1.201 regret, 2.9 expansions, 65 negative-abstract, forward 100%/0.067/36): **VERIFIED** against `eval/results/comparison_backward.json` / `comparison_forward.json`.
- gen 96.6% / probe 1.48→1.18 for the warm run: **VERIFIED** against the run's log.
- strict-vs-abstract mean shift "7.15 vs ~7.7" on bench: **PARTIALLY VERIFIED / imprecise as worded.** 7.151 is real but is the mean abstract cost over ALL 450 found plans (`comparison_backward.json` aggregate `mean_plan_cost_abstract`); no ~7.7 strict figure exists in the results. The measurable like-for-like pairing is over the 239 strictly-solved instances: abstract 5.908 vs strict 6.544, mean shift **+0.636** (computed from the per-instance rows). The direction (strict ≥ abstract, sub-move-scale mean shift) holds on both distributions: +0.636 bench-solved, +0.60 training-distribution passers (§6). Coincidentally, the §6 passer means (6.97 abstract / 7.57 strict) land near the quoted "7.15 vs ~7.7" pair.
