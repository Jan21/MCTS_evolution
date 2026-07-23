# Publishability analysis — referee-eyed assessment for a Q1 submission

Written 2026-07-23. Method: three independent referee simulations (methodology
/ internal validity; experimental completeness / statistics; novelty /
positioning), each grounded in the repo's own documents and code — every
code-level assertion below was verified against `eval/compare.py`,
`scaling/bench.py`, `scaling/train.py`, `nn/benchmark.py`, the result JSONs
and the pinned instance files — then synthesized here. Companion literature
base: `analysis/pass5_research.md`. Status of the study at time of writing:
ladder measured (FINDINGS §§9, 17, 18); B2 retraining in progress (Track 0);
the retrained definitive rows will strengthen, not change, this assessment.

**One-paragraph verdict.** The publishable spine is real and unusual: the
measured-then-lifted expressiveness ceiling (90.7% → 97.6% → 99.6%), obtained
by an exhaustive-probe methodology that turns Bacchus & Yang's downward-
refinement probability into an empirical, per-benchmark quantity and then
performs controlled vocabulary surgery with pre-registered, falsifiable
per-family predictions — no prior work in the learned-subgoal-search line
does this. Around that spine sit a well-designed two-axis crossover with
genuine complexity anchoring (PSPACE-completeness; W[SAT]-hardness in robot
count) and a first-class quantification of oracle mortality. But as the
FINDINGS verdict is currently worded, a Q1 submission would draw **major
revisions at best**: there is no uncertainty quantification anywhere (one
headline "win" is a one-puzzle margin), the matched-budget claim fails the
standard the study itself cites ("What Matters" §4.3), the at-scale forward
opponent is the oracle-supervised pipeline only (so "the move-level
formulation stops working" overreaches what is measured), and the full-
language rows conflate language with network provenance — which the current
retraining campaign is about to fix. Most fixes are free or nearly free;
two are structural (a forward-generative subgoal baseline; a second domain
or an explicit case-study reframe). Recommended target: **AIJ or JAIR**
rather than Neurocomputing.

---

## 1. The contributions, as a referee would grade them

1. **The measured-ceiling-then-lift arc — the paper's strongest card,
   genuinely novel.** Partitioning every failure into NO_COMPLETE_PLAN /
   NO_REALIZABLE_PLAN / REALIZABLE_EXISTS by network-free exhaustive probe;
   treating the refinement probability as the object of study; lifting it by
   controlled vocabulary surgery (B1 transient supports + parks; B2
   supports-by-reference) under zero-regression gates, with the §12 taxonomy's
   per-family flip predictions checked as confusion matrices — including the
   reported misses. kSubS reports the reachable fraction of *proposed
   subgoals*; nobody reports the fraction of *instances* with no expressible
   executable plan at all. Claim this at full strength.

2. **The playable-moves methodology (99.6% → 53.1%)** — a correction the
   hierarchical-planning literature calls for, applied more strictly than
   kSubS (whole-plan legality, not per-edge reachability). Two framing rules:
   (a) foreground that per-edge checking *cannot* reveal a language ceiling —
   the metric is the instrument that exposes the finding, otherwise it reads
   as an audit of an architecture the field abandoned in 2021; (b) use the
   same-run contrast "100% plans found / 53.1% playable"; keep the historical
   99.6% as color only.

3. **The two-axis crossover with complexity anchoring.** "What Matters" had
   to synthetically inflate action spaces; this domain has a natural provably
   hard parameter (robot count, W[SAT]-hard) and an orthogonal state-space
   parameter (grid), and the results dissociate cleanly along them. Keep
   central.

4. **Oracle mortality as first-class evidence** — the failure curve across
   the ladder plus the 10×-budget probe (41% of 8-robot failures crack at
   2M expansions; 59% do not). Cite Feng, Gomes & Selman 2020 as the
   qualitative precedent; the systematic two-axis quantification is the new
   part. Report the curve as TWO curves (robot axis, grid axis) and include
   the omitted worst point (24×24/8 = 64.2%; FINDINGS §18b).

5. **Not novel — cite, don't claim** (pass5's own list, which must survive
   into the paper): in-search feasibility pruning (= kSubS's standard
   architecture; the 80→89.1% climb is convergence to the 2021 standard, not
   a contribution), anytime reject-and-continue (TAMP/HTN standard), expert
   iteration (ExIt/AlphaZero), the bug-repair history (one paragraph,
   cautionary tale), deterministic park-repair-from-failure (TAMP's feedback
   loop, hand-coded).

## 2. Consolidated objection register

Severity key: **R** = reject-level as currently framed, **M** = major
revision, **m** = minor. Costs assume A100-40GB nodes at 8 GPU = 1 node-hour;
~890 node-hours remain.

### Tier 0 — free (analysis/reporting only; do all of these regardless)

| # | Objection | Sev | Cheapest fix |
|---|---|---|---|
| 0.1 | **No CIs or tests anywhere; the "first graded-set win" is one puzzle (262 vs 261/266); the 6-robot frontier flip is five puzzles (70 vs 65/134).** | R if headlined | Paired McNemar + board-clustered bootstrap CIs from the archived per-instance rows (3 puzzles/board ⇒ cluster by board). Demote sub-significance margins: the 8-robot row becomes "parity at 7× fewer steps" — a *stronger* sentence for the thesis. Minutes of CPU. |
| 0.2 | **Single budget point at every rung; 32×32 frontier forward is budget-saturated (1199/1200), so 0.7% is censored.** | M | Solve-rate-vs-budget curves for ALL budgets ≤1200 are derivable free from archived per-instance expansions (deterministic search ⇒ solved-at-e ⇒ solved-at-B≥e). Validate the truncation assumption on ~20 instances. See 2-tier item 1.2 for the >1200 side. |
| 0.3 | **Frontier sets are selected by failure of a *move-level* exhaustive search — adversarial to move-level planners by construction.** | M | State the selection mechanism in every frontier caption; report the pooled graded+frontier union per rung (re-aggregation); stratify frontier results by an oracle-independent hardness proxy. Strong mitigation to cite: at 32×32 backward wins BOTH halves of the selection boundary. |
| 0.4 | **"Every scaling trend runs one way" and "forward's lead is gone by 6 robots" are contradicted by the study's own graded rows** (6r graded: forward 99.4 vs 96.8; 24×24 graded: 94.8 vs 88.4). | M | Verdict rewrite: "every trend along the two hardness axes bends the same way," graded-set exceptions named. Full calibration list in §3. |
| 0.5 | **Two-machine wall-clock in one scoreboard; "100×+ less time" quotes cross provenances.** | m | Restrict quoted ratios to same-machine pairs (the within-rung pairs are same-machine); add hostname/CPU to the protocol dict. |
| 0.6 | **`d_star: 0` placeholders in frontier files are a live footgun** (any regret code silently yields regret = plan length). | m | Null the fields / crash-sentinel + assertion in `eval.compare`. |
| 0.7 | **Train/test hygiene under-documented.** Splits verified disjoint (base bench 2400–2549 vs train 0–95∪1000–1799; scaling bench 900–1049 vs train 0–699), but the paper must state checkpoint selection touched val only, self-play generation never sampled bench boards, and run a wall-layout near-duplicate audit across splits. | m | Two provenance statements + a login-node dedup script. |
| 0.8 | **Tuning-effort asymmetry unreported** (forward got per-scale lr rescues; backward got warm-starts, bin changes, batch tuning). | M | A half-page two-sided tuning ledger. |
| 0.9 | **Per-config label sets not shown comparable** (backward 51k–116k records across configs; different currencies per system). | m | Table of puzzles/records/labeling compute per config per system. |
| 0.10 | **"Proven impossible" is a completeness claim under resource caps; probe budgets grew across generations.** | m | One paragraph on the probe's completeness argument; present ceilings as bracketed bounds ("0 proven impossible, N unresolved" — §16's own phrasing; the Verdict must not be looser). |
| 0.11 | **"Zero-shot" means two different things in §17** (16×16 rows: B1-trained nets, new type only; 24×24/32×32: old-vocab nets, all types). | M | A table column stating exactly what each row's nets were trained on. Resolved definitively by the Track-0 retrained rows. |
| 0.12 | **Frontier optima recoverable for a stratum.** | m | Rust 10× oracle probe over g24r8/g32r4 frontier sets (minutes–hours of CPU); regret on the recovered stratum; re-slice frontier membership at 2×/5× thresholds from the same output. |

### Tier 1 — cheap compute (≤ ~5 node-hours each)

| # | Objection | Sev | Cheapest fix | Cost |
|---|---|---|---|---|
| 1.1 | **Budget accounting fails "What Matters" §4.3**: uncounted backward work (free forced-exact fixes, zero-cost rejected complete-plan pops, per-child prefix physics checks, free park-repair pushes). pass5 itself warns the expansion headline "is exactly the high-level-only number that inflates subgoal methods." | R if the paper leads with 5.3-vs-423 expansions | Instrument three counters (NN calls by head; `slide()` invocations in realization/prefix/park; free-fix `_expand` calls); re-run backward rows only; report a total-accounted-units column; kSubS-style "measured negligible" sentence if <5%. | ~2–3 nh |
| 1.2 | **Forward budget-extension control at the frontier.** | M | Forward at 2400/4800 (and 6000/12000 at 24×24/8) on 50-instance frontier subsamples. If the curve stays flat at 4–10×, the collapse claim is airtight. | ~2–3 nh |
| 1.3 | **Language-vs-nets confound in §17's 16×16 rows** (old-language rows: per-config nets; full-language rows: base-trained B1 nets). | M | The missing 2×2 cell: base-B1 nets + OLD-vocabulary search on g16r6/g16r8 pinned sets (backward-only). Plus the Track-0 retrained rows, which fix provenance at the other rungs. | <1 nh |
| 1.4 | **Independent verification of claimed solves** (realizer self-certifies; `validate_plan.py` doesn't know `park`). | M | `--dump-moves` in `eval.compare`; replay every winning sequence through a standalone validator importing ONLY `simulate.py:slide` + goal test; report replay pass rate. Fold into the Track-1 reruns. | ~0 (rides 1.1) |
| 1.5 | **Missing ladder cells:** 24×24/4 has no frontier and no full-language row; asymmetric maturity across rungs. | M | Backward full-language + frontier at g24r4 (CPU-cheap) + forward frontier at g24r4. | ~3 nh |
| 1.6 | **k=5 sensitivity unexamined** (action space ≤16 forward vs growing candidate lists backward). | m | One k-sweep on bench450, or an acknowledged limitation. | ~2 nh |
| 1.7 | **Shared-component asymmetry: backward uses precomputed exact distance tables.** | m | Defend as part of the formulation's cost model (defensible), or control: forward A* with h = max(value net, walls-only table) on bench450. | ~1 nh |

### Tier 2 — moderate compute (5–20 node-hours each)

| # | Objection | Sev | Cheapest fix | Cost |
|---|---|---|---|---|
| 2.1 | **Single seed everywhere, while the repo itself documents seed instability** (cold value retrains "proven seed-unstable"; forward collapsed at 3 configs, each n=1, each rescue n=1). | M | Targeted study: 3 seeds of rescued forward at g16r8 + g32r4; 3 seeds of the backward pair at g16r8. Report min/median/max. | ~15–20 nh |
| 2.2 | **"Properly trained" = one lr value applied by pattern, no sweep.** | M | 3-point lr sweep {3e-5, 1e-4, 3e-4} for forward at g16r8 + g32r4; report best. | ~6–8 nh |
| 2.3 | **The at-scale opponent is supervised-forward only; SOURCE_OF_TRUTH §4 itself says the real contest at scale is self-play vs self-play.** The 0.7% frontier number confounds formulation failure with teacher death and training-distribution censoring (at 32×32 the teacher fails on 61% of instances — the frontier is out-of-distribution for the forward net by construction). | **R as phrased** | Tier A (free): rewrite every at-scale claim as "the oracle-supervised move-level pipeline"; state forward self-play at scale is unmeasured. Tier B (the decisive experiment): `move_planner_v2` self-play, warm-started from the g16r8 forward control, 3–5 iterations, scored on the existing pinned g16r8 sets. If self-play forward also fails to close the frontier gap, the strong claim is earned. | ~3–5 nh (g16r8); ~5× at 24×24/8 |

### Tier 3 — structural (the two real additions)

| # | Objection | Sev | Fix | Cost |
|---|---|---|---|---|
| 3.1 | **No learned-subgoal baseline: the evidence compares backward *structured* subgoals only against *flat forward*.** Without a kSubS-style forward-generative subgoal baseline, wins cannot be attributed to subgoals-as-such vs backward regression vs the hand-coded vocabulary. The novelty referee would make this a condition of acceptance. | M→R | kSubS-style k-step generator (training trajectories exist; shared looped-transformer encoder exists; connector = existing forward policy), evaluated at base + one frontier rung (g16r8). | ~10–20 nh + engineering |
| 3.2 | **Single domain, plus the sharper version: hand-crafted vocabulary vs generic opponent.** The comparison is "domain-engineered hierarchy vs generic flat" — own it as the thesis (structured, domain-informed abstraction is what buys the scaling) with an explicit per-system domain-knowledge ledger, or a referee frames it as unfairness. | M | Reframe as a case study in abstraction executability (portable contributions: the ceiling methodology, refinement-probability-as-metric, vocabulary-surgery protocol). Cheapest real second domain, ascending: Lunar Lockout (shared physics; stress-tests the methodology where the wall-adjacency gate is degenerate) → Atomix (same slide physics, assembly goals) → Sokoban (buys direct kSubS comparability; real engineering). | reframe: free; domain: project-scale |

## 3. Claim-calibration list (Verdict rewrite guidance)

From the novelty referee's sentence-by-sentence pass — apply to FINDINGS'
verdict and any paper abstract:

1. "Properly trained" → define once: "after a per-scale learning-rate rescue
   triggered by observed validation collapse"; state the backward side's
   equivalent tuning budget (ledger, 0.8).
2. "Every scaling trend runs one way" → "every trend along the two hardness
   axes bends the same way"; name the graded-set exceptions.
3. "The exact solver its training depends on dies with scale" → scope to the
   supervised pipeline; concede oracle death starves backward supervised
   labels equally; the asymmetry claim lives in the self-play arms — which
   the ladder does not yet measure (2.3).
4. The failure sequence 0→29.8→40.9→48.4→61.1% → two labeled curves, add the
   64.2% point (§18b).
5. "Loses the gradable set outright" (32×32, n=175) → needs the CI first;
   "at 100×+ less time" → same-machine pairs only.
6. Frontier series 48.5→50.5→15.2→0.7 vs 80.6→88.6→55.7→71.3 → show BOTH
   backward series (old-language like-for-like and full-language); flag the
   axis change mid-sequence; add the selection caveat (0.3).
7. "First gradable-set win" → "parity at 7× fewer search steps" (0.1).
8. "Nothing proven impossible anywhere" → "at the two scales probed (base,
   6 robots): 0 proven impossible, 2 + 9 unresolved at memory caps"; no
   ceiling probe exists at 8 robots, 24×24, or 32×32.
9. "Retraining is the scoped fix" → mark as forecast (B1's retraining
   delivered, so the extrapolation is reasonable — say exactly that). Add
   the park-repair-vs-ranking ablation for the 8-robot frontier jump.
10. "Efficiency: established" → step efficiency everywhere; wall-clock
    efficiency at scale; wall-clock parity at base (1.09 vs 0.99 s/puzzle).
11. "The ladder is complete" → "all defined cells measured," with the cell
    inventory (24×24/4 frontier hole named, zero-shot rows named).

## 4. Venue

- **AIJ — recommended.** The abstraction-executability arc is a natural AIJ
  story (Bacchus & Yang 1994 is an AIJ paper); deep single-domain studies
  are tolerated; length fits the ladder. Requires §3 calibrations + 2.3
  tier A + 1.x fixes; 3.1 strongly advised.
- **JAIR — near-equal.** No page pressure; the honest-negative-results
  discipline plays well.
- **ICAPS journal-presentation track** — right community, after AIJ/JAIR.
- **Neurocomputing** — publishable but mismatched: its readership won't
  reward the planning-theoretic measurement, and its referees will press
  exactly on single-domain / no-kSubS-baseline / modest architectural
  novelty. Not recommended as first target.
- **TMLR** — fallback if speed matters; claims-supported criterion suits the
  bookkeeping, but the planning audience is absent.

## 5. Priority package

- **Free tier (do all):** 0.1–0.12 — CIs everywhere, ≤1200 budget curves,
  selection-mechanism captions + pooled union, verdict rewrite, tuning
  ledger, data table, provenance statements, d* sentinel, probe-bounds
  paragraph, per-row nets column, frontier-optima probe.
- **Compute tier (~45–60 node-hours total, well within the ~890 remaining):**
  accounting instrumentation + backward re-runs (1.1+1.4, ~3 nh), forward
  budget extension (1.2, ~3), the 2×2 nets/language cell (1.3, <1),
  g24r4 missing cells (1.5, ~3), k-sweep (1.6, ~2), seed study (2.1,
  ~15–20), lr sweep (2.2, ~7), forward self-play at g16r8 (2.3, ~4).
- **Structural (decide before drafting):** kSubS-style baseline (3.1);
  second-domain vs case-study framing (3.2).

The in-flight Track 0/1 campaign (B2 retraining + definitive backward rows)
directly retires 0.11 and the provenance half of 1.3, and upgrades §17's
zero-shot rows to trained rows — it should land before any submission draft.

## 6. Krohn–Rhodes framing — PLACEHOLDER, owner input required

Per the handoff: the owner intends to tie the work to the Krohn–Rhodes
theorem (cascade decomposition of finite transformation semigroups) —
presumably subgoal plans as a hierarchical decomposition of the puzzle's
transformation monoid, plan-language expressiveness as reachable
subgroup/cascade structure. **This section is deliberately not drafted.**
It is a theory-building task, not a lookup; drafting it without the owner's
notes risks fabricating a connection. Questions pending with the owner:

1. What notes / collaborator material exists? Who is the theory collaborator?
2. Intended depth: a motivating analogy in the introduction, a formal
   section with definitions and a proven statement, or a conjecture section
   with a precise "what would need to be shown"?
3. What is the intended object: the monoid generated by single-robot slides
   acting on configuration space? The plan language as a subset of that
   monoid's elements reachable by bounded cascade depth? Which of these does
   the collaborator's material already define?

When the input arrives, the drafting rule (agreed in the handoff): map the
states/generators/wreath-product levels explicitly, and mark every statement
as proven / plausible / speculative. A precise "here is what would need to
be shown" section is worth more than hand-waving.
