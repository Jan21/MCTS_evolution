# rust_datagen — verification report

This document records the full verification battery for the Rust port of the
training-data generation core (DESIGN.md). Every number below comes from a
command that was actually run on 2026-07-15 (shared 128-core box, background
load 31–47 throughout; all verification work capped at 16 workers). Raw logs
live under the paths given per section; larger artifacts under
`pyref/out/` (git-ignored) and the session work directory.

Verdict summary (details per section):

| gate | content | verdict |
|---|---|---|
| 1–2 | board physics: slide graph edge-for-edge, all-pairs tables | **PASS** — 328 boards, 2,936,558 edges, 10,168 slides, 400 table hashes, fresh sweep, 0 diffs |
| 3 (replay) | ≥10,000 decision contexts replayed, zero label diffs | **PASS** — 20,649 contexts / 258,999 labels this battery (8,983 structured + 11,049 fuzz + special legs), **0 diffs** |
| 3(ii)/(iii) via §5.11 | paired end-to-end rollouts: label / record-set / fork divergence outside the two validated tie classes must be ZERO | **FAIL — BLOCKER** (§4): 82 violations / 1,650 instances; the backward labels are not order-independent — both engines are faithful ports, the reference under-determines the label (proven: Python disagrees with itself across board representations) |
| §5.11 (i) | `cand_parent_support` field-only divergence rate (sign-off number) | **0 / 90,130 matched record pairs** (18,930 paired-fresh + 71,200 production-scale) |
| 4 | cap-boundary divergence < 2 % of instances, never a differing label value | forward **PASS** (0 divergence in 3,074 attempts incl. candidate-scoring); backward solvability-at-the-cap **0 / 600 (< 2 % budget)** and per-candidate cap parity exact (5,912 labels, 0 diffs) — but the "never a differing label value" clause is hit by the §4 blocker class (not cap-induced) |
| 5 | three fixes + 12-instance regression corpus | **PASS** — 12/12 defect candidates rejected by BOTH engines, 165/165 label parity, wired into `cargo test` + `make verify` |
| 8b | backward flip classification (g16r6 regeneration) | **RESOLVED** — exact attempt-level reconstruction: 30 genuine flips, 30/30 replay past the 120 s production timeout; the earlier "fast flips" were alignment artifacts |
| smoke | 12-cell cross-engine matrix | **PASS** — 12/12, 525 s steady-state |
| determinism | byte-reproducible single-thread, multiset multi-thread | **PASS** — 9,460-item file: t1 twice byte-identical, t16 multiset-equal; engine test suite 7/7 |
| CI wiring | `cargo test` / `make smoke` / `make verify` | **PASS** — all three run from committed inputs; `make verify` exit 0 in 538 s |
| perf gates | quoted per BENCH.md (post-optimization) | backward ≥50× MET (52.7–178.9×, g16r4 at the boundary), forward ≥50× MET post-optimization (58.4–79.9×; paired), precompute §9 |

Terms used: a **decision** is one step of the backward labeler where several
candidate subgoals are evaluated; each evaluated candidate produces one
training **record** with a `cost_to_go` label (the number of moves the rest
of the plan needs) and an `is_optimal` flag (whether it ties the cheapest
candidate). **Replay** (gate 3) feeds the Rust engine the exact candidate
list Python enumerated; **paired rollouts** (§4) let both engines enumerate
candidates themselves on identical instances — the difference between those
two modes is what separates the green gate 3 from the §4 blocker.

---

## 1. Gates 1–2 — board physics and tables (fresh full sweep)

Re-run end-to-end into a fresh temp corpus (not reusing any committed
evidence): `pyref/verify_full_gate12.sh <scratch>/gate12` (PROCS=8, seed 7).
The script dumps 128 stock boards + 100 fresh 16×16 + 50 fresh 24×24 + 50
fresh 32×32 from the Python reference (`pyref/dump_reference.py`), with
canonical table hashes on 50 boards per size and 31 slide cases per board,
then runs the Rust side (`cargo test --release -- --ignored gate_full`).

Result (log: session scratch `logs/gate12_sweep.log`, exit 0):

```
gate 1-2 corpus PASS: 328 boards, 2936558 edges, 10168 slides, 400 table hashes
```

328/328 per-board PASS lines; edge-for-edge equality (endpoints, weight,
dependent-support annotation), 10,168 slide cases, and 400 canonical
SHA-256 table hashes (both the all-pairs and the no-dependent-edges tables)
all equal. Rust side 24.6 s.

## 2. Gate 3 — replayed decisions, zero label diffs

All legs use the same pipeline: `pyref/dump_decisions.py` (Python labels
every decision and dumps the full decision context) → `datagen replay`
(Rust recomputes every label from the dumped context) → `pyref/diff_labels.py`
(per-candidate `cost_to_go`/`rejected`/`is_optimal` for backward; value /
legal-move set / optimal-move set rules for forward).

Solver settings are production's (backward max_iters 4,000 / max_frontier
40,000 / max_candidates 14; forward max_expansions 40,000 /
full-policy cap 6) unless stated. "Lines" are decision contexts (backward:
one decision with its full candidate list; forward: one recorded state);
"units" are per-candidate labels (backward: `cost_to_go` + `rejected` +
recomputed `is_optimal`) or per-move comparisons (forward: value +
legal-move set + optimal-move set/membership).

**Structured corpora** (seed 20250715; boards from each config's bench
split unless noted; logs `logs/diff_s_*.log` in the session scratch,
corpora under `pyref/out/gate3/`):

| leg | boards | instances | lines | units | verdict |
|---|---|---|---|---|---|
| g16r4 backward | 40 | 920 | 1,325 | 12,876 | ALL GREEN |
| g16r4 forward | 20 | 200 | 1,250 | 14,922 | ALL GREEN |
| g16r4 backward, STOCK pickled boards 0–49 | 50 | 400 | 547 | 5,155 | ALL GREEN |
| g16r4 backward, `--env-source pkl` (labels computed on production pickle-loaded boards, replayed on the engine's rebuilt boards) | 30 | 210 | 288 | 2,620 | ALL GREEN |
| g16r6 backward | 30 | 300 | 504 | 5,942 | ALL GREEN |
| g16r6 forward | 20 | 160 | 828 | 15,651 | ALL GREEN |
| g16r8 backward | 20 | 60 | 93 | 1,197 | ALL GREEN |
| g16r8 forward | 20 | 100 | 492 | 12,426 | ALL GREEN |
| g24r4 backward | 30 | 420 | 593 | 5,789 | ALL GREEN |
| g24r4 forward | 20 | 200 | 1,325 | 16,556 | ALL GREEN |
| g24r8 backward | 15 | 30 | 51 | 623 | ALL GREEN |
| g24r8 forward | 15 | 83 | 423 | 11,235 | ALL GREEN |
| g32r4 backward | 25 | 200 | 305 | 2,933 | ALL GREEN |
| g32r4 forward | 20 | 112 | 813 | 10,442 | ALL GREEN |
| 64×64 r4 backward (fresh boards) | 3 | 12 | 17 | 174 | ALL GREEN |
| 64×64 r4 forward (fresh boards) | 3 | 17 | 129 | 1,663 | ALL GREEN |
| **structured total** | | **3,424** | **8,983** | **120,204** | **0 diffs** |

The STOCK leg matters because pickled stock boards are where the proposal
fallback (retry pinned to the goal's dependent supports — the
set-order-sensitive path) lives: measured with `pyref/count_fallback.py`,
**67 of the 547 stock decisions (12.2 %) hit the fallback**, spread over
41 of the 50 boards — the corpus genuinely exercises it, with zero diffs.
The `--env-source pkl` leg closes the inherited pkl-vs-rebuild check at
the replay level (zero diffs); a same-seed dump pair additionally
quantifies the REPRESENTATION effect on the Python side itself: of 288
decision lines dumped from pickle-loaded vs rebuilt boards, 245 are
byte-identical and 43 differ (order/cut of the enumerated candidates) —
see §4/§6 for why this matters.

**Fuzz corpora** (leg 8 of the battery: fresh seeds 555–563, plus
config-less fresh-board cells covering the robots axis beyond the standard
configs; ≥10,000 fuzz decisions required beyond the structured corpora):

| leg | seed | instances | lines | units | verdict |
|---|---|---|---|---|---|
| g16r4 backward | 555 | 1,500 | 2,258 | 22,384 | ALL GREEN |
| g16r4 forward | 555 | 520 | 3,103 | 37,456 | ALL GREEN |
| g16r4 backward | 563 | 520 | 781 | 7,578 | ALL GREEN |
| g16r6 backward | 556 | 300 | 579 | 6,895 | ALL GREEN |
| g16r6 forward | 563 | 100 | 470 | 8,896 | ALL GREEN |
| g24r4 backward | 557 | 525 | 744 | 7,226 | ALL GREEN |
| g24r4 forward | 557 | 200 | 1,273 | 15,907 | ALL GREEN |
| 24×24 r6 backward (fresh boards) | 559 | 100 | 172 | 2,015 | ALL GREEN |
| 24×24 r6 forward (fresh boards) | 559 | 60 | 316 | 6,183 | ALL GREEN |
| g32r4 backward | 558 | 160 | 233 | 2,017 | ALL GREEN |
| g32r4 forward | 558 | 104 | 710 | 8,978 | ALL GREEN |
| 32×32 r8 backward (fresh boards) | 560 | 24 | 34 | 434 | ALL GREEN |
| 32×32 r8 forward (fresh boards) | 560 | 32 | 158 | 4,196 | ALL GREEN |
| 64×64 r4 backward (fresh boards) | 561 | 21 | 32 | 286 | ALL GREEN |
| 64×64 r4 forward (fresh boards) | 561 | 24 | 170 | 2,229 | ALL GREEN |
| 64×64 r8 backward (fresh boards) | 562 | 8 | 16 | 203 | ALL GREEN |
| **fuzz total** | | **4,198** | **11,049** | **132,883** | **0 diffs** |

**Handcrafted forward cases** (`pyref/gen_handcrafted_forward.py`,
inherited fixture gaps): start-on-target (incl. inside a sealed pocket),
`cost_to_go` 1 and 2 states (every non-goal child's capped solve runs the
`h0 > cost_cap` pre-check), sealed-pocket relaxed-unreachable targets and
robots, tight-cap (500 expansions) unsolved states, and 64×64
short-distance + start-on-target states — **24 cases / 335 units, ALL
GREEN** (`pyref/out/handcrafted/`). These are also the previously-missing
end-to-end Python-parity coverage for the `replay_forward_state` task
shape, on top of the forward legs above.

**Re-verification of the committed corpora on the final binary**: all six
earlier gate corpora (`pyref/out/gate_{b16,f16,b24,f24,b64,f64}.jsonl`) and
the earlier pkl A/B corpus replayed through the current engine — 1,497 +
92 lines / 16,469 + 900 units, ALL GREEN.

**Gate-3 grand total: 20,649 decision contexts / 258,999 compared labels
(this battery's runs alone), ZERO label diffs.** Including the
re-verified committed corpora, handcrafted cases, the tight-cap replay
leg (§5) and the regression corpus (§3): 22,274 contexts / 276,868 units,
zero diffs. Backward instance-timeout drops during dumping (18 across the
r8 cells) affect corpus size only, not comparisons.

One reading note: this zero is an EMPIRICAL zero, not structural
immunity. Pinning the candidate list (with its `parent_support`) in the
replay item removes the enumeration-order surfaces of candidate
GENERATION, but each engine still runs its own inner plan search, whose
tie order can change the returned `cost_to_go` under the §4
non-admissible-estimate corner — the committed §4 repro
(`golden/s511_evidence/repro_e910_replay.jsonl`) diverges through this
very replay path (Python 11 vs Rust 10). At the measured rate of that
shape (about 1 per 19,000 record labels on rebuilt boards) the sampled
gate-3 corpora simply did not contain such a case.

## 3. Gate 5 — the three fixes and the 12-instance regression corpus

The two six-instance residual-defect classes come from
`eval/results/residual_failures_postfix.json` over the bench instances in
`eval/data/bench450_first150.jsonl` (boards in `environments/`): six plans
that recruited one physical robot for two plan roles
(`robot_reused_two_places`, bench indices 42, 50, 58, 86, 126, 138) and six
plans that claimed a helper-supported route into the goal without ever
placing the helper (`goal_stopper_never_placed`, indices 25, 52, 87, 91,
111, 129). `pyref/gen_regression.py` replays each archived failing plan
(`analysis/artifacts/residual_failing_plans.pkl`) against the CURRENT
solver, applying the failing plan's own subgoal sequence until the current
invariants refuse a step, and captures that decision context — including
the defect-recreating candidate — as a `replay_backward_decision` line.

Results:

- All 12 failing plans are REFUSED by the current Python solver before
  completion — every duplicate-robot plan at its duplicate-recruitment
  step, every missing-stopper plan at its very first step. None of the
  archived defects is still constructible.
- 7 of the 12 defect candidates are still enumerated naturally by the
  proposal heuristic (they appear inside the kept candidate list); the
  other 5 were appended explicitly so the corpus always exercises the
  refusal.
- Cross-engine: `datagen replay` on the corpus matches Python's labels on
  all **165 candidate labels across the 12 decisions (0 diffs)**, and the
  flagged defect candidate is `rejected` by BOTH engines in **12/12**
  cases.

Permanent wiring: corpus committed at `golden/regression/regression12.jsonl`
(+ `regression12.meta.json` naming each defect candidate's index);
`tests/gate5_regression.rs` runs the real binary on it inside `cargo test`
(asserts full label parity + the 12 rejections); `make regression`
(included in `make verify`) additionally replays the corpus through the
CURRENT Python solver via `pyref/verify_regression.py`, a three-way check
that also guards against reference drift.

## 4. The §5.11 measurement — paired end-to-end rollouts (BLOCKER found)

**What was measured.** DESIGN §5.11 documents one place where the Rust
engine deliberately does not reproduce Python: the ORDER in which candidate
subgoals are enumerated (Python iterates internal `set` objects; Rust uses
the graph's edge-insertion order; the candidate SET is proven identical).
The design assumed labels are order-independent. To test that assumption at
scale, `pyref/paired_rollouts.py` sampled identical instances for both
engines, ran the real Python labeler (`nn.generate.rollout`) and the Rust
engine (`backward_rollout` work items) independently on each instance, and
`pyref/compare_rollouts.py` aligned the two record streams decision by
decision, matching candidates across sides by (bottleneck, support, helper)
— never by list position — and comparing `cost_to_go` pairwise. Production
solver settings (max_candidates 14, plan-search caps 4000/40,000), seed
4242, bench-split boards.

**Corpus and headline numbers** (logs: session scratch `logs/s511.log`;
per-config JSON reports committed under `golden/s511_evidence/paired_
<cfg>.report.json` (copies of `pyref/out/s511/<cfg>.report.json`); raw
paired streams `pyref/out/s511/<cfg>.{py,rust,work}.jsonl`):

| config | instances compared | aligned decisions | matched record pairs | ps-field divergence (i) | validated boundary ties | justified forks | zero-threshold violations (ii)+(iii) |
|---|---|---|---|---|---|---|---|
| g16r4 | 500 | 784 | 4,275 | 0 | 158 | 9 | 20 |
| g16r6 | 450 | 846 | 6,524 | 0 | 227 | 15 | 27 |
| g16r8 | 300 | 545 | 4,906 | 0 | 166 | 14 | 20 |
| g24r4 | 400 | 622 | 3,225 | 0 | 98 | 15 | 15 |
| **total** | **1,650** | **2,797** | **18,930** | **0** | 649 | 53 | **82** |

Supporting counts (totals): Python-side timeouts 0 (none excluded);
instances where Python's rollout was empty: Rust agreed empty on all 1,012
(no instance-level solvability flip at production caps); 1,516 of 1,650
instances (91.9 %) have fully identical record streams end to end; 1,147
aligned decisions sit at depth ≥ 1 with at least one committed support
node (the high-exposure population the task-C review required — r6/r8
included for exactly this reason). The 82 violations decompose as 70
candidate-set divergences that move the labeled minimum, 1 matched-candidate
value difference, 11 trajectory-length differences.

The two validated tie classes behaved as validated: every
truncation-boundary tie was verified to sit on a single common score at the
kept-list cut with equal minimum cost on both sides AND a per-candidate
proof that both engines resolve the same `parent_support` (so an
asymmetric acceptance cannot be hiding in the class); every accepted fork
carries a genuine optimal-cost tie with at least two distinct candidates.
The `cand_parent_support` field-only divergence rate — the §5.11 sign-off
number, since training consumes this field — measured **ZERO** over all
matched record pairs.

**The blocker.** A residual set of decisions diverges in LABEL CONTENT,
outside both validated tie classes — the zero-threshold classes of this
gate. Three shapes, one root cause:

1. **Candidate-set divergence that moves the labeled minimum** (the
   dominant shape). The labeler keeps only the 14 best-scored candidates
   (a stable sort by a heuristic score, then a cut). When the cut falls
   inside a group of equal-scored candidates — very common, including
   large "junk" plateaus at score ≥ 10,000 — the two engines keep
   different members of that group. The score is only a heuristic: a
   cut-group member can carry the genuinely cheapest `cost_to_go`. When it
   does, the two engines label different minima, the `is_optimal` flags
   point at different candidates, and the trajectories fork at unequal
   costs. Minimal example (instance `e906:i15` of the g16r6 corpus, first
   decision, identical plan on both sides): 13 of 14 kept candidates are
   shared with IDENTICAL labels; Python's 14th is a candidate with
   `cost_to_go` 6 (its optimum), Rust's 14th is a different equal-scored
   candidate with `cost_to_go` 9 — Python's labeled minimum is 6, Rust's
   is 7.

2. **Same candidate, different `cost_to_go` value** (rare: 1 of 18,930
   matched pairs in the fresh paired corpus; 12 of 71,200 at production
   scale, §6). Fully root-caused with a committed minimal repro
   (`e910:i8`, board 910 of environments_g16r6, first decision, candidate
   bottleneck (0,11) / support (0,10) / helper Yellow, identical
   `parent_support=None` on both sides): Python labels 11, Rust labels 10
   — reproducible through the gate-3 replay path itself. The minimal
   repro is COMMITTED: `golden/s511_evidence/repro_e910_replay.jsonl` (one
   replay work item whose `python_labels` say 11) and
   `repro_e910_rust_output.jsonl` (the engine's answer, 10). The plan-cost
   estimate used by the inner plan search is NOT admissible, contrary to
   the comment in `skeleton/astar.py`: an open segment's optimistic
   estimate uses the all-pairs table, which prices a helper-supported hop
   at weight 2, while the eventual decomposed route prices the same hop at
   1 (the supported-route tables add `+1` past the stopper) — measured
   directly: the cheapest child plan of that decision is pushed at
   estimated cost 11 but completes at cost 10. With an over-estimating
   heuristic the first completed plan popped is not always the cheapest,
   so WHICH completion the search returns depends on the heap tie order —
   which inherits the candidate enumeration order, the exact surface
   DESIGN declared order-free. Python's own 11 is the objectively
   non-optimal value here; Rust's 10 is the cheaper completion.

3. **Different trajectory length**: after identically-labeled decisions
   and identical advances, one side's next kept-14 contains a candidate
   that survives labeling and the other side's does not, so one rollout
   ends (the labeler stops when no candidate labels) while the other
   continues — the shape-1 mechanism at the trajectory's end.

**Adjudication.** This is NOT a defect in the Rust port: gate 3 (§2)
replays the same decisions with Python's candidate list pinned and shows
ZERO label diffs across tens of thousands of candidates, and shape 2's
repro shows each engine returning a cost that is correct for its own
enumeration order. The finding is that the REFERENCE under-determines the
label: `cost_to_go`/`is_optimal` are functions of CPython's set-iteration
order, not of the decision alone. No amount of faithful porting can make
an independent enumerator reproduce them exactly; the only exact-match
options are (a) bit-matching CPython's set order inside Rust, or (b)
making the reference order-free: a full deterministic content-based
tie-break on the candidate sort AND the plan-search heap AND a
deterministic dependent-support resolution order for `parent_support`
(so no `set` iteration order remains anywhere labels can see). This
changes Python's own outputs and would require regenerating reference
data. Choosing between them —
or explicitly accepting a measured divergence class — is a design
adjudication, not a verification call. Per the gate's escalation clause
this is recorded as a **BLOCKER**: threshold ZERO, measured nonzero.

**Scale context.** The same phenomenon bounds what the production
regeneration can achieve: the g16r6 whole-config record-level comparison
(§6) measures 28.6 % exactly-identical instances, ~64 % differing only
within the validated tie classes, and 7.4 % in the blocker class — with
an additional proven order surface (the pickled boards' stored edge order
differs from the rebuilt order, and Python's own labels change with it).
The forward pipeline is unaffected (bit-mirrored expansion order,
whole-config byte-identity — §5, §9).

## 5. Gate 4 — cap-boundary divergence

The gate: on capped runs, instance-level divergence must stay under 2 %
and consist ONLY of solvability-at-the-cap (one side labels, the other
returns unsolved/over-budget) — never a differing returned label value.

**Forward** (`pyref/paired_forward.py`; per-instance status + full record
streams compared as parsed values; reports
`pyref/out/gate4/f*.report.json`):

| leg | max_expansions | attempts | status diffs | record-stream diffs |
|---|---|---|---|---|
| g16r4 | 2,000 | 1,072 (292 solved / 777 unsolved / 3 unreachable) | **0** | **0** (1,322 records) |
| g16r4 | 5,000 | 748 (300 / 447 / 1) | **0** | **0** (1,448 records) |
| g24r4 | 2,000 | 949 (151 / 798 / 0) | **0** | **0** (720 records) |

Forward divergence at tightened caps: **0 of 2,769 attempts** — the
expected result (the move-level search is mirrored decision-for-decision,
so both engines exhaust the same budget on the same expansion sequence).
An additional production-cap leg with the candidate-scoring records
enabled (g16r6, max_expansions 40,000, `--score-candidates`, seed 779):
305 attempts / 150 solved instances / **14,681 records — 0 status diffs, 0
record diffs** (`pyref/out/gate4/f16r6_sc_fixed.report.json`).

**Backward** (`pyref/paired_rollouts.py` at tightened caps
max_iters=1000 / max_frontier=8,000, seed 8484; production caps
4000/40,000 are covered by the §4 corpus at 1,650 instances):

| leg | instances | instance-level solvability divergence | ps divergence | validated ties / forks | label-class divergences |
|---|---|---|---|---|---|
| g16r4 | 300 (+173 both-empty agreed) | **0** | 0/2,461 | 77 / 13 | 20 |
| g16r6 | 300 (+134 both-empty agreed) | **0** | 0/4,061 | 142 / 17 | 21 |

- **Solvability-at-the-cap divergence: 0 of 600 instances (0 %, gate
  budget < 2 %)** — likewise 0 of 1,650 at production caps (§4) and 0 of
  3,160 python-dropped attempts in the full-config reconstruction (§6).
  The ONLY instance-outcome divergence found anywhere in the battery is
  the 30 production-timeout flips of §6 (0.32 % of attempts), which are
  the intended wall-clock-vs-deterministic-budget difference, not a
  solver-cap effect.
- **The "never a differing label value" clause** is violated only by the
  §4 blocker class, at the same rate as at production caps (41/600 =
  6.8 % of instances) — it is not cap-induced; it counts under the §4
  zero-threshold, where it is reported.
- Per-candidate cap parity through replay (tight caps mi=1000/mf=8,000,
  python contexts pinned; g16r4 seed 9911): 617 decisions / **5,912
  candidate labels, 1,173 of them unsolved-within-caps (`ctg` null) —
  ZERO diffs** (`pyref/out/gate4/replay_b16r4_cap.jsonl`). Cap semantics
  are exact given identical candidate lists; every instance-level
  difference at caps traces to the §4 enumeration-order classes, not to
  the caps.

## 6. Backward flip classification (g16r6 regeneration) — complete, attempt-level

The earlier whole-config comparison (BENCH.md §5) reported 90.27 %
kept-instance overlap between the shipped Python `backward.jsonl` and its
Rust regeneration, attributed the flips to production's 120 s wall-clock
kill, and left two loose ends: three of five sampled flips replayed fast
(0.0/0.7/17.7 s — a timeout cannot explain those), and record CONTENT of
shared instances was never compared (the 90.27 % counted instance keys).
Both are settled here by reconstructing Python's ENTIRE attempt history:
`pyref/classify_flips.py --stage reconstruct` re-simulates the production
sampling loop per shard (one shared random stream, board order) and aligns
every draw against the shipped kept stream. The reconstruction is exact —
**9,460 attempts across 8 shards; all 6,300 shipped kept instances matched
in order on all 1,050 boards** — so every attempt Python actually made is
known, immune to the flip-cascade resampling that made board-level diffs
ambiguous. The Rust engine then judged the very same 9,460 attempts
(`datagen run`, §7's determinism runs), and per-attempt outcomes were
joined (`--stage classify`, `--stage replay`; outputs under
`pyref/out/flips/`).

**Outcome flips: 30, all production timeouts.**

- 3,130 of 3,160 python-dropped attempts: Rust agrees (rollout genuinely
  empty). **Zero** instances where Python produced records and Rust did
  not.
- 30 attempts (on 27 boards; 24 boards with one flip, 3 with two):
  Python dropped, Rust labels. Replaying ALL 30 in today's Python
  (`--stage replay`, 150 s cap; per-flip detail committed at
  `golden/s511_evidence/flips_classification.json`): **30/30 exceed the
  cap** (i.e. all are > 120 s of Python work even unloaded) — the
  production-timeout class,
  exactly as designed (the deterministic iteration budget deliberately
  labels what the wall clock killed; DESIGN §3). None of the earlier
  "fast flip" boards (155/163/191) carries a genuine flip: at those
  boards the reconstruction shows Python never attempted the draws the
  board-level alignment had paired — they were resampling artifacts, not
  flips. Flip rate: 30/9,460 attempts = 0.32 %.

**Record content on the 6,300 kept instances** (production Python vs Rust
on identical instances — the §4 measurement at production scale):

- 1,801 instances (28.6 %): record streams exactly identical.
- 4,499 instances: some difference; adjudicated by the §4 comparator
  (report committed at
  `golden/s511_evidence/production_recmismatch.report.json`): 8,858 aligned
  decisions, **71,200 matched record pairs, `cand_parent_support`
  divergence 0**, 3,087 validated truncation-boundary ties, 249 justified
  optimal-tie forks — and **466 label-affecting divergences** (12
  matched-candidate value differences, 397 moved minima, 57 trajectory
  lengths), i.e. 7.4 % of kept instances, the §4 blocker class at
  production scale.
- The much lower exact-identity rate here (28.6 % vs 91.9 % in §4's
  paired runs) has a proven additional cause: production Python runs on
  the PICKLED board graphs, whose stored edge order differs from the
  order both the rebuilt Python graphs and the Rust engine use (a legacy
  of an older board generator; the edge SET is identical). Direct proof
  on instance `b0:a1`: today's Python on the pickle-loaded board
  reproduces the shipped production records byte-for-byte, while the SAME
  Python code on the SAME board rebuilt from its wall list produces a
  different (tie-class) record stream. Sampled at scale (150 random kept
  instances relabeled by today's Python on rebuilt boards): **125
  reproduce the shipped records exactly, 25 (16.7 %) differ** — Python
  disagrees with itself across two representations of the same board at
  a rate comparable to the cross-engine rate, the strongest possible
  demonstration that the §4 order-sensitivity is a property of the
  reference, not of the port.

## 7. Determinism

Two layers, both green:

- **Engine test suite** (`tests/io_determinism.rs`, drives the real
  binary): re-run as part of `cargo test --release` — 7/7 passed
  (byte-identity + work-file order at `--threads 1`, line-multiset
  equality at `--threads 16`, malformed-input abort naming line and
  field, replay-mode filter, stdin/stdout modes, sidecar round-trip).
- **Fresh 10k-item check on real work** (this battery): the 9,460-item
  backward work file from §6 (every rollout the production g16r6 run
  attempted) through `datagen run`:
  - `--threads 1`, twice: **byte-identical** outputs —
    46,327,714 bytes / 9,460 result lines (407.3 s and 405.8 s wall);
  - `--threads 16`: **line multiset equal** to the single-thread output
    (35.8 s wall — 11.4× the single-thread run on the shared box).
  Log: session scratch `logs/battery_summary.log`.

## 8. Smoke matrix

`make smoke` (= `pyref/smoke.py --engine rust --workers 8`, seed 7), run
as part of `make verify` on the final tree, warm caches:

```
smoke matrix -- engine=rust, seed=7, total 525s
cell              lines  dump_s engine_s  diff_s  status
--------------------------------------------------------
g16r4.backward       13     1.1      0.1     0.0  PASS
g16r4.forward        75     6.6      0.1     0.0  PASS
g16r8.backward       19   108.6      0.6     0.0  PASS
g16r8.forward        53    21.0      0.2     0.0  PASS
g24r4.backward       10     5.9      0.1     0.0  PASS
g24r4.forward        63    15.1      0.1     0.0  PASS
g24r8.backward       12   237.0      0.7     0.0  PASS
g24r8.forward        39    37.5      0.2     0.0  PASS
g32r4.backward       29    12.5      0.2     0.0  PASS
g32r4.forward       137    36.9      0.1     0.0  PASS
n64r4.backward       14    10.8      1.8     0.0  PASS
n64r4.forward        15    27.3      0.1     0.0  PASS
--------------------------------------------------------
12/12 cells PASS
```

**12/12 cells PASS, 525 s steady-state (< 15-minute target).** The wall
time is Python-dump-dominated (the two 8-robot backward cells account for
~346 s of dumping); the Rust engine stage totals 4.2 s across all 12
cells. Log: session scratch `logs/make_verify.log` (per-cell diffs under
`pyref/out/smoke/rust/`).

## 9. Benchmarks (quoted from BENCH.md; every number traces to a log there)

All ratios are PAIRED (identical work items through both engines, trimmed
to what the production loop would consume) and per-core (Rust
`--threads 1`, thread pool pinned). Python baselines cross-check against
the phase-0 measurements in `.superpowers/sdd/progress.md` (from_env cold:
8.04 s at 16×16 r6 / 62.22 s at 24×24 / 242.09 s at 32×32 — consistent
within load variance).

- **Backward labeling** (gate ≥ 50× per core: MET): 52.7× (g16r4) /
  178.9× (g16r6) / 169.2× (g16r8) / 108.0× (g24r4) / 79.9× (g32r4).
  Caveat carried from the task-D review: the g16r4 margin is
  load-sensitive — an independent re-run measured 50.4× on the same work —
  so quote it as "at the boundary", not as precise. Outcome agreement
  126/126 attempts. (BENCH.md §2.)
- **Forward labeling** (gate ≥ 50× per core): the honest paired figure
  before optimization was **25.5–34.3×** (BENCH.md §3) — an earlier ~66×
  micro-benchmark was UNPAIRED and composition-sensitive and is not to be
  quoted. After the optimization pass (BENCH.md "Post-optimization"), the
  same paired battery measures **58.4–79.9×** across all five configs —
  gate MET — with outputs verified byte-identical to the pre-optimization
  engine on all five work files. Outcome agreement 110/110.
- **Board precompute**: production surface (`GridEnv.from_env` cold)
  1549× / 2184× / 3831× at n = 16/24/32 post-optimization. The isolated
  all-pairs table slice: 75.5× / 132.2× / 160.7× — the ≥ 100× aspiration
  is met at n = 24 and 32; n = 16 lands at 75–95× depending on
  methodology, documented in BENCH.md as an arithmetic floor of the
  algorithm class (≈2M edge relaxations at ~1.5 ns inside a 4.4 ms
  compile), not slack.
- **End-to-end g16r6 regeneration** (production settings, 32 threads,
  loaded box): backward 141 s (6,300 instances / 98,246 records, zero
  budget exhaustions) + forward 194 s post-optimization (10,500 instances /
  1,038,827 records) ≈ **5.6 min** for the full config (target < 10 min).
  Forward output verified byte-identical to the pre-optimization run,
  which itself matches the shipped production `forward.jsonl` on
  1050/1050 boards. 32×32: ~36 min EXTRAPOLATED from 30-board samples
  (target < 1 h). (BENCH.md §4–6 + post-opt section.)
- **Deterministic budget calibration** (replaces the production 120 s
  wall-clock kill): over 67 paired instances across five configs, plan
  search speed measured 12–2,632 iterations per Python-second (median
  350, p90 1,216); 120 s of Python work therefore corresponds to at most
  ~316k iterations. The shipped default `DEFAULT_SOLVER_ITERS =
  10,000,000` is ≥ 31× that worst case (> 280× the median). Cross-check:
  the full g16r6 backward regeneration (25,200 rollouts) peaked at 83,085
  iterations — 120× under the default — with zero budget exhaustions.
  Raw pairs: `pyref/cache/bench/*_bwd_calibration.jsonl`. (BENCH.md §8.)
- **Board sidecars**: the pipeline recomputes tables from inline board
  text (compile 0.04 s at 32×32 / 0.64 s at 64×64 on 16 threads, at or
  below sidecar-load cost) rather than shipping table files (12.8 MB /
  196 MB per board; ~230 GB per 64×64 config). Sidecars remain available
  for serial replay/debug workflows where they are 3–4× faster than
  compiling. (BENCH.md §7.)

## 10. CI wiring

Three entry points, all exercised on this tree:

- `cargo test --release` — hermetic; runs entirely on committed corpora
  (gates 1–2 golden mini-corpus, forward/backward golden fixtures, the
  gate-5 regression corpus added by this battery, the engine determinism
  and CLI-contract tests, and the library unit tests). No network, no
  absolute paths, no un-committed inputs.
  Run on this tree: lib 41 passed, backward_fixtures 5 (+2 ignored),
  backward_units 12, gate12 1 (+1 ignored full-sweep variant, run in §1),
  gate3_forward 1 (+1 ignored), **gate5_regression 1 (new)**,
  io_determinism 7 — all ok, 0 failures.
- `make smoke` — the 12-cell cross-engine matrix (§8). Needs the Python
  environment and, on first run, builds the 32×32/64×64 table caches
  under `pyref/cache/` (git-ignored; one-time warm-up outside the
  15-minute steady-state budget) and generates the three fresh 64×64
  boards there.
- `make verify` — `cargo test` + `make smoke` + `make regression` (the
  gate-5 three-way leg, §3) + `datagen selftest` (the committed 11-item
  golden work file through the real dispatch plus determinism and
  expected-value checks).
  Run on this tree end-to-end: **exit 0, 538 s wall** (cargo suite +
  the §8 smoke matrix + the three-way regression leg + `selftest PASS:
  11 items`). No uncommitted artifact is needed beyond the git-ignored
  `pyref/cache/` warm-up noted above.

The deeper battery legs of this report (the §4 paired-rollout
measurement, §5 capped runs, §6 flip classification, the gate-3 corpus
dumps) are driven by the committed pyref tools
(`paired_rollouts.py` + `compare_rollouts.py`, `paired_forward.py`,
`classify_flips.py`, `dump_decisions.py`/`replay_python.py`/
`diff_labels.py`, `gen_handcrafted_forward.py`, `count_fallback.py`,
`verify_regression.py`, `gen_regression.py`) — commands are quoted in
each section; they need the Python reference and real board directories,
so they are deliberately not part of `make verify`.

## 11. Known limits

- **The §4 blocker** is the limit that matters: backward `cost_to_go` /
  `is_optimal` labels are not a pure function of the decision context in
  the reference implementation (candidate-order sensitivity through the
  14-candidate cut and, rarely, through the plan search's tie order under
  a non-admissible cost estimate). Until adjudicated, Rust-regenerated
  backward data cannot be expected to match the SHIPPED Python data
  record-for-record beyond ~29 % of instances (the pickled boards' edge
  order adds a second order surface, §6) — or beyond ~92 % against Python
  run on rebuilt boards; distribution-level equality (BENCH.md §5) is
  unaffected, and ~64 % of the differing instances differ only inside the
  two validated tie classes.
- **64×64 Python reference costs**: the eager Python board build is
  ~72 min per 64×64 board, so all 64×64 Python-side verification runs on
  the lazily-computed equivalent (`pyref/lazy_env.py`), proven
  result-identical (`pyref/prove_lazy_env.py`). The Rust engine itself
  compiles a 64×64 board in 0.64 s at 16 threads.
- **Low forward yield at 64×64**: most random 64×64 forward instances
  exceed the 40,000-expansion search budget (~11 solvable per 120
  attempts measured), so 64×64 forward corpora are small by nature; the
  cells still verify with zero diffs.
- **Smoke wall time** is dominated by the Python dump stage (the Rust
  engine stage is seconds); the 12-cell matrix fits the 15-minute budget
  only with warm 32×32/64×64 table caches (`pyref/cache/`), which the
  first run builds.
- **Production runs leave no timeout audit trail**: the backward labeler
  swallows exceptions and drops timed-out instances without recording
  which ones (`scaling/backward_label.py`), so flip attribution required
  reconstructing the whole sampling stream (§6). The reconstruction
  resolved every flip as a genuine > 120 s timeout, but pipelines that
  log their drops would make this class auditable directly.
- **Sidecar files at 64×64** are impractically large (196 MB/board); the
  pipeline recomputes tables instead (decision recorded in BENCH.md §7).
