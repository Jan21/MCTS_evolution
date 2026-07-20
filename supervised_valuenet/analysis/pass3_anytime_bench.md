# Pass 3 — first benchmark of the anytime realization-checked backward planner

**One-line result.** Letting the backward planner discard unplayable plans and keep searching
(the `--backward-anytime` mode, never measured before) raises the strict solve rate from
54.0% to **83.3%** on the same instances, rescues **44 of 69** previous failures, loses
nothing it previously solved — and does so at a heavy price in search effort (mean expansions
2.1 → 70.0, now above the forward planner's 36) and with long rescue plans (rescued instances
average 6.1 moves over optimal). It does **not** reach the Phase A parity bar on any of the
three criteria.

Throughout: "solved" = the plan survives strict realization (a legal move sequence under full
game physics, `eval/realize.py::strict_moves`); "regret" = achieved moves minus d\*, the exact
per-instance optimum stored in the benchmark file (oracle used only for that label); regret is
averaged over solved instances only.

---

## 1. What was run

```
eval.compare --instances eval/data/bench450.jsonl --expansions 1200 --k 5
  --backward-policy checkpoints_backward/policy_v2.ckpt
  --backward-value  checkpoints_backward/value_v2.ckpt
  --backward-anytime --forward-ckpts "" --device cpu
  --out eval/results/comparison_backward_anytime.json
  --md  eval/results/COMPARISON_anytime.md
```

- `--forward-ckpts ""` cleanly yields zero forward systems (`compare.py:416` filters empty
  strings), so only the backward planner ran; no forward re-measurement, no root
  `COMPARISON.md` touched. The two files above are new; no existing repo file was modified.
- Same checkpoints, instance file (sha `1b384dfd...`), budget (1200), k (5) as the plain
  baseline run — a pure A/B on the anytime flag.

**Two deviations from the original plan, stated plainly:**

1. **CPU, not GPU.** A first attempt on GPU 0 was killed a few minutes in by machine-owner
   policy (all four A100s are reserved for training jobs; strict 1-job-per-GPU). The
   measurement therefore ran CPU-only (`CUDA_VISIBLE_DEVICES=""`, 8 threads,
   `--device cpu`). Expansion counts are hardware-independent; wall-clock times are not
   directly comparable with the GPU baseline (see the timing note in §3).
2. **First 150 instances of bench450, not all 450.** At CPU pace the full 450 projected to
   ~3 h (over the agreed ~2.5 h cap), so the full-file run was stopped at 100/450 and the
   measurement rerun on `instances[:150]` via a scratch copy of `eval.compare`'s backward
   loop (identical protocol; repo untouched; the produced report marks the subset
   explicitly). All plain-baseline numbers below were recomputed on the same first 150 rows
   of `eval/results/comparison_backward.json`, matched by row order (env_id and d\* verified
   equal row-by-row), so every comparison is apples-to-apples. The subset is representative:
   plain failure rate 46.0% vs 46.9% on the full 450; mean d\* 6.51 vs 6.37.

   Neither killed attempt wrote any output (`eval.compare` writes results only at the very
   end, `compare.py:468-473`), so the final JSON/markdown come from the single clean CPU run.

---

## 2. Headline: anytime vs plain, same 150 instances

| system | strict solve | mean regret | % optimal | mean strict moves | mean expansions | mean s/inst |
|---|---|---|---|---|---|---|
| plain backward (full 450, reference) | 239/450 (53.1%) | 1.201 | 64.0% | 6.54 | 2.9 | 0.34 (GPU) |
| plain backward (same 150) | 81/150 (54.0%) | 1.383 | 59.3% | 6.79 | 2.1 | 0.32 (GPU) |
| **anytime backward (150)** | **125/150 (83.3%)** | **3.048** | 45.6% | 9.30 | **70.0** | 28.8 (CPU) |
| forward best (450, reference) | 100% | 0.067 | — | — | 36 | — |

Plan-found rate stays 150/150 — every instance still yields a complete plan; the change is
in how many survive realization.

**Rejected-plan distribution** (`plans_rejected` per instance): 81 of 150 instances needed 0
rejections — exactly the 81 the plain run solved, i.e. the first plan popped is identical and
already playable. 69 instances (46%) rejected at least one complete plan; max = 1,183
rejections on a single instance; mean 29.3 overall, but sharply split: mean 6.1 among solved
vs 145.4 among failed. Histogram: 0×81, 1×19, 2×6, 3×9, 4-9×15, 11-50×9, >50×11.

**The regret jump 1.38 → 3.05 is pure composition, not degradation.** On the 81 instances
both runs solve, the anytime run returns bit-identical plans (0 of 81 differ in strict move
count) — regret there stays exactly 1.383. The 44 newly solved instances average regret
6.114, and (81×1.383 + 44×6.114)/125 = 3.048 exactly. The %-optimal drop (59.3 → 45.6) is
the same effect: 59% of both-solved instances are optimal vs 20% of rescues. Nothing the
plain planner did well got worse; the average now includes hard instances solved with long,
salvaged plans.

---

## 3. Depth breakdown: does anytime close the executability gap?

Solve rate and regret by puzzle difficulty (d\* = optimal move count), matched 150 instances:

| d\* bin | n | plain solve | anytime solve | plain regret | anytime regret |
|---|---|---|---|---|---|
| 1-3 | 14 | 100% (14/14) | 100% (14/14) | 0.43 | 0.43 |
| 4-6 | 58 | 78% (45/58) | **91% (53/58)** | 1.40 | 1.53 |
| 7-9 | 70 | 30% (21/70) | **76% (53/70)** | 2.00 | 4.91 |
| 10+ | 8 | 12% (1/8) | **62% (5/8)** | 1.00 | 6.80 |

(Full-450 plain bins for orientation: 95% / 67% / 34% / 20%; the 150-subset bins 100/78/30/12
sit close.)

What it costs, per bin (anytime; plain in parentheses; forward-best expansions from the
full-450 forward run for scale):

| d\* bin | expansions (plain) | mean rejections | s/inst CPU (plain, GPU) | forward-best expansions |
|---|---|---|---|---|
| 1-3 | 0.1 (0.1) | 0.0 | 0.07 (0.01) | ~3 |
| 4-6 | 0.8 (0.7) | 0.4 | 0.32 (0.09) | ~11 |
| 7-9 | 130.5 (3.6) | 60.6 | 53.8 (0.55) | ~50 |
| 10+ | 165.1 (3.4) | 16.8 | 65.9 (0.46) | ~159 |

Reading: the depth-dependent executability gap **narrows a lot but does not close** — at
d\* 7-9 solve rate jumps 30% → 76%, at 10+ 12% → 62%, still far from forward's 100%. And the
backward planner's signature "~3 expansions regardless of depth" is gone exactly where the
gap was: at d\* 7-9 it now spends 130 expansions (2.6× forward's 50 in that bin), at 10+ 165
(about equal to forward's 159). The cheap bins stay cheap: shallow plans mostly pass on the
first or second try (mean rejections ≤ 0.4 below d\* 7).

**Conversion of plain failures.** Of the 69 instances the plain run failed, anytime solves
44 (64%): by bin, 8 of 13 at d\* 4-6, 32 of 49 at 7-9, 4 of 7 at 10+ (and 0 lost anywhere:
no instance flipped solved → failed, confirming the search is deterministic up to the first
complete plan). Rescues are usually shallow: 24 of the 44 needed ≤ 3 rejected plans; the
tail is long (one rescue took 345 rejections). Rescued-instance quality is poor, though:
mean regret 6.11, only 9/44 optimal.

**Timing note (devices differ).** The plain baseline ran on an A100; this run on 8 CPU
threads. On the identical first-75-instance prefix, the killed GPU attempt took ~270 s wall
vs ~1,680 s summed on CPU (~6-7.5× factor), so the honest GPU-equivalent estimate for the
anytime mode is roughly 4-5 s/inst mean — still ~15× the plain mode's 0.32 s, driven by NN
passes on retries, not by the realization checks themselves. The failure tail dominates
wall-time: the 25 unsolved instances average 137 s each (max 537 s) and the 7 budget-
exhausted ones alone account for ~57 of the run's 72 total minutes. Solved instances average
7.1 s (CPU).

---

## 4. The residual failure set: 25 instances, two distinct modes

d\* profile of the 25 failures: 5×(d\*5-6), 17×(d\*7-9), 3×(d\*10). All 25 still produce a
complete plan (`plan_found` 150/150; the row keeps the first failed plan, correctly marked
unsolved). The failure modes split cleanly on expansions used:

- **Frontier exhaustion — 18 of 25.** The search dies after 0-13 expansions, having rejected
  only 1-13 complete plans (one outlier: 53), in under ~5 s. The proposal step simply stops
  producing alternatives: every plan variant the proposal vocabulary can express for these
  boards fails realization, and the frontier empties. Five of these die at literally 0
  expansions — the initial plan completes through the free exact-fix path, fails the check,
  and there is nothing else to try.
- **Budget exhaustion — 7 of 25.** The other extreme: all 1,200 expansions consumed,
  83-1,183 complete plans rejected per instance (420-537 s each on CPU). The vocabulary
  generates endless variations, but all of them break under real physics.

Both modes say the same thing: for ~1 in 6 instances the current plan language/nets cannot
express a playable plan at all, no matter how long the search retries. That is the addendum's
L4 diagnosis (proposal vocabulary lacks "move the blocking robot out of the way"), now with a
measured size: 16.7% of instances, concentrated at d\* ≥ 7.

---

## 5. Fairness/correctness audit of the anytime implementation

Scope: `eval/compare.py::_nn_astar_backward` (lines 110-173) and `run_backward` (176-237);
`eval/realize.py::strict_moves` (182-246); forward protocol `move_planner/evaluate.py::
nn_astar` (97-123).

**The realization check is free in the expansion count — defensible, with one disclosure.**
A popped complete plan is tested at `compare.py:142-147`; failing pops do not increment
`expansions` (the increment at `compare.py:149` sits after the completeness branch). This is
internally consistent with the protocol's stated unit — "one popped node whose children are
generated (= 1 policy pass + 1 batched value pass), in both systems" (`compare.py:437-439`) —
because a rejected complete plan generates no children and costs zero NN passes. The forward
planner's goal-pop is likewise free (`evaluate.py:107-111`: the goal test returns before
`iters += 1`). What must be disclosed: a forward search can never pop an unplayable goal, so
it has no analog of hundreds of free complete-plan pops; the size of that concession is
exactly `plans_rejected`, which is reported per instance (max observed: 1,183). If one
charged them anyway, the price in the protocol's own currency (NN passes) is still zero.

**The check leaks no privileged knowledge.** `strict_moves` is a topological ordering of plan
segments plus one BFS over single-robot slides per segment (`realize.py:163-179, 182-246`),
built on the same `simulate.slide` rules (`simulate.py:40`) the forward planner exercises at
every expansion via `legal_moves` (`move_planner/state.py:49`; applied at
`evaluate.py:112`). No oracle, no d\*, no solver state. The forward planner uses this
physics far more often; giving the backward planner an accept/reject test on complete plans
is symmetric in kind. The check provides no search gradient — it does not rescore the
frontier or explain failures; it only forbids returning an unplayable plan.

**CPU cost is not hidden.** The check's cost lands in wall-clock: `t0` is taken before the
check closure is built and `dt` after the search returns (`compare.py:204, 215`), so all
in-search `strict_moves` calls are inside the reported seconds. On a 16×16 board one check is
bounded by ~256 cells × 4 directions per segment — microseconds-to-milliseconds. The
expansion metric stays an NN-compute budget; seconds carry the verifier cost. Both are
reported above.

**`plan_found` stays truthful on exhaustion.** When the budget or frontier runs out after at
least one complete-but-unplayable plan, the first failed plan is returned
(`compare.py:145-147, 173`); `plan_found=True` is recorded (`compare.py:219`) — truthful, a
complete plan was found — while `solved` comes from the strict result cached for that exact
plan (None → unsolved, regret None; `compare.py:227-233`). Verified in the data: all 25
failures have `plan_found=true`, `solved=false`. Two minor implementation notes, neither
affecting this run's correctness: the per-instance check cache is keyed by `id(plan)`
(`compare.py:208-211, 227`) — safe here because the only id ever looked up belongs to the
still-alive returned plan whose own check was the last write to that key, but a fragile
pattern in general; and there is no plan-content deduplication, so an identical plan reached
twice is re-checked (CPU waste only, counted in seconds).

**Pre-existing asymmetries, unchanged by the anytime flag** (apply equally to the plain
baseline; disclosed in the generated report's footnotes): one backward expansion commits a
whole subgoal vs one primitive move (`compare.py:370-377`), and the backward environment
carries precomputed exact-distance tables used by the free exact-fix loop
(`compare.py:134-140, 378-382`). Both make the shared expansion cap generous to the backward
system — which is why the Phase A bar demands backward stay well below forward's expansion
count, a margin this run no longer has.

**Metric caveats.** Regret averages over solved instances only (`compare.py:251-256`), so it
rises mechanically as harder instances enter the solved pool — hence the composition analysis
in §2. And the anytime search accepts the *first* plan that passes, in abstract-cost order
(`nn/generate.py:46-48`), with no pressure toward shorter playable plans — visible in the
rescued plans' 6.1 mean regret.

---

## 6. Verdict against the Phase A bar

Bar (addendum §5): strict solve ≥ 97.5%, strict regret ≤ 0.35, mean expansions ~an order of
magnitude below forward's 36.

| criterion | bar | anytime alone (150) | verdict |
|---|---|---|---|
| strict solve | ≥ 97.5% | 83.3% | short by ~14 points |
| mean strict regret | ≤ 0.35 | 3.05 (1.38 on both-solved, 6.11 on rescues) | far short |
| mean expansions | ~3-4 (order below 36) | 70.0 (16.6 on solved, 337 on failed) | lost — now *above* forward |

**Anytime alone does not reach parity, and is not close.** What the measurement establishes:

1. Roughly two-thirds of the plain planner's failures were *recoverable* — the nets' top
   candidates do include playable plans, just not first — so simply forbidding unplayable
   answers buys +29 solve points for zero training work. That is the single biggest jump any
   lever has produced on the honest metric.
2. The remaining 16.7% is not recoverable by search: 18 of 25 residual failures exhaust the
   proposal vocabulary within 13 expansions. More budget will not help (only 7 even reached
   the cap). Closing this needs either training pressure toward executable plans (L2/L3) or
   the vocabulary fix (L4) — most likely both, since L2/L3 also cannot express a plan the
   vocabulary lacks.
3. The rescue quality (regret 6.1, 20% optimal) means even a 100%-solve anytime planner
   would miss the regret bar badly without something that prefers short playable plans —
   e.g. strict-cost labels (L2 strong form) or anytime-in-generation (L3) so the nets learn
   to rank playable plans first, restoring the ~3-expansion profile that made the backward
   approach attractive.

Recommended next step per the addendum's ordering: run L1+L2 (loop metrics + strict filter)
with this run as the calibration point; the anytime mode's per-instance `plans_rejected` and
the residual-failure list in `eval/results/comparison_backward_anytime.json` give exact
targets to test the vocabulary hypothesis on.

---

## Files

- Results JSON (per-instance rows): `eval/results/comparison_backward_anytime.json` (new)
- Generated protocol report: `eval/results/COMPARISON_anytime.md` (new; n=150 marked)
- Plain baseline used for matching: `eval/results/comparison_backward.json` (untouched)
- Run log (all three attempts, timestamped): scratchpad `anytime_run.log`; analysis script:
  scratchpad `analyze_anytime.py`; 150-instance runner: scratchpad `run_anytime_150.py`
