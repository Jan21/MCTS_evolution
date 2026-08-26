# Certified Self-Play Breaks the Expressiveness Ceiling of Subgoal Planning

*A full-stack study on Ricochet Robots: supervised scaling, a size-free neural
labeler, label-free self-play, and the action-space extension that passes the
plan language's proven quality floor.*

**Status: LOCAL DRAFT — not for distribution.** Written 2026-08-26 against
`supervised_valuenet/FINDINGS.md` (§1–§87), `self_play_robots/FINDINGS.md`
(§1–§26), `self_play_robots/variants/FINDINGS.md` (1–22) and
`variants/DESIGN.md` §7. Every number below carries a source tag:
**[SV§n]** = supervised FINDINGS entry n, **[SP§n]** = self_play_robots
FINDINGS entry n, **[VL§n]** = variants-lab FINDINGS entry n, or a repo file
path. Numbers in the tables of §3–§5 were recomputed for this draft directly
from the cited result files via `self_play_robots/report/compare.py`
(2026-08-26); nothing is quoted from memory.

**Relation to `supervised_valuenet/PAPER_PLAN.md`:** that plan splits the
supervised campaign into paper A (the expressiveness ceiling) and paper B (the
neural labeler's fidelity threshold). This draft is the JOINT narrative that
adds the self-play arc; §2 condenses A, §2.3 condenses B, and §4–§5 are new
material. Whether to publish as one paper or as A + B + a self-play paper C is
an open owner decision (see Open Questions).

---

## Abstract

Learned planners need answer keys, and exact solvers stop producing them as
problems grow. We study both halves of that bind on Ricochet Robots, a
PSPACE-complete sliding-robot puzzle, under one strict rule: a puzzle counts as
solved only if the plan replays, move by move, under the full game physics.
First, supervised planners: a move-by-move searcher is near-optimal but its
oracle teacher fails on most large instances, while a subgoal-based searcher
stays cheap and wins every pooled benchmark above base scale — yet an
exhaustive, network-free probe proves its plan *language* caps solution quality
about 1.2 moves above optimal, no matter how well it is trained. Second, we
replace the answer key entirely: an AlphaZero-style loop whose only supervision
is physics certification. On fresh, never-trained boards it solves 177/200
versus the supervised planner's 134, with fewer moves — and a from-scratch run
matches the supervised baseline without a single labeled example. Third, we
break the proven ceiling: letting the search consider one or two ordinary
moves before subgoal planning reaches 0.86 extra moves versus perfect play,
beats a same-budget control 30/0 on paired moves, and transfers zero-shot to
board sizes and robot counts it never ran on. (198 words)

---

## 1. Setup: the domain, the metric, and the certification rule

**Domain.** Ricochet Robots: robots slide on a grid until they hit a wall or
another robot; one robot must reach a goal cell. Optimal play is
PSPACE-complete in general and W[SAT]-hard in the number of robots (Hesterberg
& Kopinsky; Balanza-Martinez 2024 — see `PAPER_PLAN.md` §3), giving two
natural hardness axes: board size (16→96) and robot count (4→8). All
benchmarks are pinned per configuration (450 puzzles at base; graded +
frontier splits at scale) and never trained on.

**Two formulations.** A *forward* planner searches move by move (policy +
value over slides, guided A\*/MCTS). A *backward* planner searches over
*subgoals* — a hand-built vocabulary of "place robot X at cell Y so robot Z
can bounce" decisions — and only converts a finished plan to moves at the end.

**The metric and the honesty rule.** We score *realized moves to terminal*:
every claimed solution is replayed under the full physics
(`eval/replay_validate.py`); a plan that does not replay is a failure. This
rule is not cosmetic: the backward planner's historical "99.6% solved" counted
paper plans, and fell to **53.1%** when forced to play them out [SV§1]. Four
realizer bug-fixes and an in-search playability check rebuilt it honestly to
**89.1%** at ~7 search expansions per puzzle [SV§2, SV§3].

**Budget parity.** All head-to-heads run at an identical search budget (1200
expansions, top-k 5), where one expansion = one policy pass + one batched
value pass [SP§2; PROBLEM.md §7].

---

## 2. The supervised campaign

### 2.1 The oracle dies; the scale ranking flips

Supervised forward training needs an exact solver for labels. At its practical
budget (200k expansions + time cap) that solver fails on **0% / 29.8% / 40.9%**
of benchmark puzzles at 4/6/8 robots and **48.4% / 61.1%** at 24×24/32×32
[SV§6; `scaling/data/*/bench.jsonl.meta.json`]. A 10× budget re-run of all 184
8-robot failures recovers only 41% — a cost explosion, not an implementation
defect [SV§6].

On the puzzles the oracle *can* grade, the forward planner keeps the quality
crown everywhere (e.g. 24×24: 220/232 solved, 0.068 extra moves — but at 191
expansions and 275 s/puzzle versus backward's 27 and 7.8 s [SV§6]). On the
whole pinned pool (graded + frontier union, no selection caveat), the
full-language backward planner wins **every rung above base scale, all
p < 0.0005** [SV§22, SV§46; `eval/results/stats_tests.json`]:

| rung | backward (B2) | forward | Δ (95% CI) |
|---|---|---|---|
| 16×16·4r (base) | 430/450 | **450/450** | forward wins |
| 16×16·6r | 414/450 = 92.0% | 379/450 | +7.8 [+4.2, +11.6] |
| 16×16·8r | 425/450 = 94.4% | 354/450 | +15.8 [+11.6, +20.0] |
| 24×24·4r | — | — | +19.8 [+14.4, +25.1] |
| 24×24·8r | 309/450 = 68.7% | 201/450 | +24.0 [+19.6, +28.7] |
| 32×32·4r | 350/450 = 77.8% | 135/450 | +47.8 [+42.7, +52.7] |

Seed replication: the 32×32 headline is a 3-seed median of +50.2 points (band
9 puzzles wide, 357–366/450) [SV§83]; the 16×16·8r rung is bimodal and
reported as median-of-3 with its wide band [SV§80/§84]. The forward planner
was given a fair second chance — a 9-arm tune (3 seeds × 3 LRs, selection by
validation only): it perfects the gradable base-scale set (266/266) but gains
only 8 frontier puzzles; the pooled margin stands at +11.3 points, p=1.7e-09
[SV§86; `scaling/results/g16r8/comparison*_forward_rescue.json`].

### 2.2 The ceiling is a language property

A network-free exhaustive probe over the subgoal vocabulary shows the
remaining base-scale gap is *not* trainable away: for 9.3% of base puzzles no
playable subgoal plan exists at all — ceiling 90.7% [SV§4;
`analysis/artifacts/ceiling_probe_results.json`]. Controlled vocabulary
surgery lifts the *solvability* ceiling 90.7 → 97.6 (B1) → 99.6% (B2 —
reusing already-placed robots, parked stoppers, by-reference helpers) [SV§16].

The self-play campaign re-measured the ceiling on the *moves* metric
[SP§3; `self_play_robots/results/ceiling/`]: even with a perfect ranker and
unlimited search, the best playable plan in the base vocabulary averages
**+1.42** extra moves at 16×16 and **+1.72** at 24×24; the extended B2
vocabulary lowers this to **+0.90 / +1.17** — an order of magnitude above the
forward planner's +0.07 [SV§5]. This number — **+1.17 at 24×24, the B2
language's best-plan floor** — is the wall §4 breaks.

### 2.3 The size-free labeler (condensed paper B)

Removing the one size-locked tensor from a graph-transformer value net yields
a *size-free* labeler: trained only on ≤16×16 boards, it labels certified
descents at every size. Against exact ground truth its argmin agreement stays
in **86.4–92.6% across all twenty rungs from 17×17 to 64×64** with no cliff —
90.1% at 64×64, four times its training size — while absolute calibration
slowly drifts (ordering holds, calibration drifts) [SV§59, SV§65, SV§70].
Downstream, twin planners trained on NN labels versus exact labels of the same
boards are indistinguishable at 91.0% and 89.4% agreement and collapse at
82.2% [SV§71–§74] — but a pre-registered causality experiment showed the
mechanism is *not* argmin fidelity itself: corrupting labels to exactly the
collapse dose (82.2%) with everything else held fixed caused **no harm**
[SV§85]. Argmin agreement is a validated cheap *predictor* of label damage,
not the causal lever. The labeler costs 9–250× more compute than the exact
solver where both run [SV§59] — it buys reach, not savings: it labeled the
first UNVERIFIABLE boards (80×80, 96×96 — 6× its training size) with zero
timeouts, warranted only by the flat 17→64 curve [SV§72c].

---

## 3. Label-free self-play

### 3.1 The loop

AlphaZero minus the opponent, plus a physics certifier: fresh lean boards each
iteration (never the pinned pools), PUCT MCTS over subgoal decisions with the
policy/value pair, and **labels only from certified plans** — cost-to-go =
certified plan cost minus the decision's fixed cost; a plan that fails replay
produces no label, so label *corruption* is structurally impossible (only
suboptimality can creep in, and the exact-audit gauge measures that up to
64×64) [SP§1; PROBLEM.md §9; `self_play_robots/DESIGN.md`]. Nets are the
size-free family (pe=none policy + value, 96 cost bins) warm-started from the
supervised stack — allowed but, as §3.4 shows, not required.

Milestones, each gated and logged: **M0** arena parity with the recorded
supervised rows [SP§2]; **M1** the size-free rebuild *beats* the per-size
supervised pair at 24×24 (215/232, regret 2.43, 54.4% optimal vs 205/232,
4.22, 38.5%), replicated exactly at a second seed (215/232, 2.31) [SP§7,
SP§18; `results/m1/`]; **M2** search beats greedy [SP§5, SP§8]; **M3/M4** five
B2-vocabulary self-play iterations at 24×24: frontier solves 133→158
(p=8e-4 vs iteration 0), the graded exam pinned at the language's solve
ceiling with expansions cut 53→26, and the final nets beat the frozen
supervised B2 rows on both exams at p<1e-4 [SP§19;
`results/selfplay/g24r4_b2_iter*/`]. The honest mechanism finding: the loop
*distills search into the nets* — the cheap first-solution planner learns to
reach what a 500-expansion MCTS reached at iteration 0 — but realized moves on
shared solves never improved beyond noise inside the fixed language [SP§19b].

**M5, mixed-size curriculum:** one iteration of self-play *on* the target
distributions (24×24 + 32×32 + 8 robots, one size-free pair) broke the
24-only plateau everywhere: frontier solves 170/218, 235/275, and **269/289 at
8 robots — versus 161/289 for the recorded supervised B2 arm** (McNemar
p≈1e-17 class) [SP§20; `results/selfplay/mix_b2mix_iter1/`]. Iterations 2–3
were flat: the distribution switch, not repetition, carried the gain [SP§22].

### 3.2 Far-size audits: the student beats its teacher

The M1 pair (trained at 16/24 only) audited zero-shot against exact decision
corpora [SP§17; `results/audit/`]: its value net picks an optimal candidate
**more often than the labeler that initialized it at every size** — 0.863 vs
0.853 (32×32), 0.863 vs 0.852 (40), 0.854 vs 0.842 (48), 0.842 vs 0.829 (56),
0.838 vs 0.814 (64×64) — and its policy's top-5 contains an optimum in 97–99%
of decisions, so the arena's k=5 filter costs ≈0.05 regret.

### 3.3 The unseen-board exam

The generalization instrument: 200 uniform instances on 50 fresh 24×24 boards
(ids pinned, never trainable), all systems at the same 1200/k=5 budget, with
the supervised backward *and* forward baselines benched on the *same* boards
[VL§1; `results/variants/exam/`, `results/variants/baselines/`]. Exact optima
exist for 137/200 (mean d\*=8.71; the rest defeat the exact solver's caps)
[VL§10; `exam/g24r4_unseen.dstar.jsonl`]. Moves are compared only on the
common-solved subset (93 puzzles all systems solved, all with known optima) —
per-system means over different solve sets are not comparable.

| system (unseen exam, n=200) | solved | moves (93-common) | Δ vs perfect | moves w/l vs backward | vs forward |
|---|---|---|---|---|---|
| forward supervised (original ckpt) | 101 | 7.76 | **+0.13** | 52/1 | — |
| backward supervised (per-size pair) | 134 | 11.45 | +3.82 | — | 1/52 |
| self-play seed nets (mix-iter2) | 175 | 9.84 | +2.20 | 37/19 | 4/44 |
| v09 strict-value (seed 8) | 177 | 9.68 | +2.04 | 38/16 | 3/43 |
| v14 stack | 179 | 10.04 | +2.41 | 36/16 | 4/46 |
| **hybrid d2 (flagship search)** | **188** | **8.54** | **+0.90** | **67/0** | 5/29 |
| hybrid d3 | 187 | 8.48 | +0.85 | 68/0 | 5/27 |

(Recomputed for this draft from the payloads via `report/compare.py`.) The
label-free line meets the project's success criterion's first half **out of
distribution**: ≥ backward solves *and* fewer moves (both-solved vs backward
p=0.004 at wave 2 [VL§5]). The second half — moves versus forward — is §4.

### 3.4 Cold start: no human labels anywhere

A control arm trained **from random initialization** — no supervised warm
start, no anchors — reached 132/200 on the unseen exam after one label-free
iteration, statistically indistinguishable from the fully supervised per-size
pair's 134/200 [VL§4; `results/variants/v08_cold_start/`]. The supervised
prior is worth ≈40 unseen solves versus the mature line, but supervised-level
competence needs no labels at all.

### 3.5 The forward arm

The primitive-move self-play loop at 24×24 (two iterations, exact-anchor
regularized) improved optimality 94.1→96.3% (regret 0.068→0.041; the
project's only significant paired *moves* gain inside a fixed action space,
p=0.039) at solve parity with its supervised teacher [SP§21;
`results/fwd_selfplay/`].

---

## 4. Breaking the ceiling: the hybrid action space

§2.2's floor (+1.17 at 24×24) is a property of *what plans can say*. The fix
(lab arm v07) widens what the *search* considers: a portfolio search that
spends its budget on the standard subgoal tree **plus** sub-searches rooted at
the states reached by the top-m single robot slides (depth 2 = two-slide
prefixes; slide moves are paid at full strict cost; composed
`[slides]+subgoal-plan` solutions replay-certified like everything else)
[VL§8; `self_play_robots/variants/v07_hybrid_actions.py`].

**Results at 24×24** (graded exam, exact optima known) [VL§8, VL§14, VL§17;
`results/variants/v07_hybrid_actions/`]:

| search (same v09 nets) | solved /232 | Δ vs perfect | % optimal | expansions |
|---|---|---|---|---|
| std-MCTS, matched budget (control) | 230 | +1.42 | 59.6 | 499 |
| hybrid, depth-1 slides | 231 | +1.01 | 66.2 | 570 |
| hybrid, depth-2 (**flagship**) | 231 | **+0.944** | 68.4 | 582 |
| hybrid, depth-3 | 231 | **+0.861** | 70.6 | 604 |
| *pure-B2-language best-plan floor* | — | *+1.17* | — | *(exhaustive)* [SP§3] |

The paired verdicts: hybrid vs same-budget control 30/0 move wins on graded
(p=1.9e-9) and 31/1 on unseen (p=1.5e-8) [VL§8]; d2>d1 10/2 (p=0.039), d3
still descending on graded, saturated on unseen [VL§20]. The break is
**net-independent**: with a different net family (v14) the hybrid beats its
control 33/0 / 36/0 [VL§16]. Honesty note: most of depth-2's gain is broader
screening of slide-starts; genuine two-slide plans are a minority (5/46 wins)
[VL§14].

**Zero-shot transfer** — neither the search nor the nets ever touched slides
at these sizes [VL§19; `results/variants/v07_transfer/`]:

| exam | hybrid d2 | std control (same nets/budget) | paired moves |
|---|---|---|---|
| 32×32 graded (175) | **174/175**, regret 1.60 | 172, 1.87 | **17/0, p=1.5e-5** |
| 24×24·8r graded (161) | **159/161**, regret 1.24 | 159, 1.82 | **23/0, p=2.4e-7** |
| 24×24·8r frontier (289) | **276/289** (program record; prior best 269) | pending (control re-running) | pending |

The largest regret cut is at 8 robots (1.82→1.24) — the crowded mode is where
one-robot slides unlock most, the same pattern the B2 vocabulary showed one
level up [SP§20b].

**Against the forward planner**, the both-solved moves gap shrinks from 2.2
(A\* line) to **0.87** on graded (hybrid 8.42 vs 7.55, while solving 231 vs
220) [VL, wave-3 processing; `bench_graded_hybrid.json` vs
`scaling/results/g24r4/comparison.json`] and the hybrid solves 188 unseen
puzzles to forward's 101. Forward keeps the pure-quality crown (+0.13 vs
+0.90); the hybrid is the best overall planner by a wide margin at a fraction
of the search cost.

**Head-to-head at 24×24, pinned graded exam** (moves on the 198 puzzles all
three solved; recomputed from payloads for this draft):

| system | solved /232 | moves (198-common) | Δ vs perfect | expansions |
|---|---|---|---|---|
| backward supervised | 205 | 11.36 | +3.92 | 27 |
| forward supervised | 220 | 7.52 | +0.07 | 191 |
| self-play line (A\*, mix-iter3) | 230 | 9.25 | +1.80 | 24 |
| flagship hybrid d2/d3 (v09 nets) | 231 | 8.5–8.6 (own set) | +0.94/+0.86 | ~600 |

---

## 5. The variants lab: methodology as a contribution

Sixteen design variants were tested under one matched protocol — same seed
boards, same generation budget, same warm start, single delta per arm, paired
gates versus a common control, **two seeds before any "win" is claimed** —
with results in a fixed plain-language card format (what we tested / why /
result / conclusion) [VL§1; `variants/DESIGN.md`]. Verdicts, including the
negatives [`variants/DESIGN.md` §7]:

- **Adopted:** v09 (value trained on realized strict moves — the benchmark
  metric itself; 2-seed frontier win, Fisher p=0.0015 [VL§5]); v04 (train on
  every certified decision the search explored, not just the winning line;
  Fisher 3.4e-5); v14 (their stack; unseen win at both seeds, p≈0.018); v07
  (the hybrid, §4).
- **Killed with evidence:** v01 visit-count policy targets (the AlphaZero
  default!) collapses everything at p≤1e-7 — the certified-cost softmax
  target is load-bearing [VL§4]; v12 frontier curriculum, flat at one *and*
  three chained iterations [VL§9].
- **Honest non-replications:** v06 Gumbel-root's seed-7 moves win reversed at
  seed 8 [VL§5]; v15/v16 slide-*training* (uniform and search-ranked nudges)
  flat at two seeds each — the nets already generalize to slid states; the
  search supplies what training was supposed to [VL§21].
- **Boundary results:** the v14 recipe does not compound over iterations and
  does not move a mature main line (174 vs 177, n.s.) — it is a fresh-loop
  tool [VL§11, VL§16].

The program's three durable lessons, verbatim from the closeout
[`variants/DESIGN.md` §7]: **train on the number you are graded on; keep
everything the search examined; search in subgoals plus ordinary moves.**

---

## 6. Limitations and future work

1. **Forward still owns pure move quality** (+0.13 vs the hybrid's +0.86–0.94
   where optima are known) — the claim is best *overall* planner (solves ×
   moves × cost), not best moves-when-it-solves.
2. **Frozen-baseline caveat:** the forward baseline in the self-play tables is
   the original checkpoint; §SV86's 9-arm retune (at 16×16·8r) gained the
   forward planner 8/184 frontier puzzles — a retuned forward opponent at
   24×24 would likely narrow the reported gaps slightly [SP§24, SV§86].
3. **One domain**; the kSubS-style learned-subgoal baseline and the Rush Hour
   port remain unrun [`PAPER_PLAN.md` §4.5/§4.11]. Our subgoal side has a
   hand-written vocabulary — "subgoals beat flat search" is not claimable.
4. **Unseen exam scope:** one board size (24×24), uniform instances, 63/200
   optima unknown (reported only on the known subset).
5. **M6 — beyond the oracle — is in progress:** self-play at 80×80/96×96,
   where only certification exists and the flat 17→64 fidelity curve
   [SV§70] is the entire warranty; feasibility phase running at draft time
   (marked IN PROGRESS, no results claimed).
6. Two transfer legs (8-robot frontier control, 32×32 frontier) were still
   completing at draft time — marked pending in §4's table.
7. **Ceiling-probe budget mismatch:** the base-scale B2 solvability ceiling is
   99.6% under the supervised probe's caps [SV§16] but 98.0–98.3% under the
   self-play re-probe's tighter caps [SP§3]; both count unresolved as
   failures. Cite per-probe, never blended.

## 7. Reproducibility

Every result row is a JSON payload with per-instance rows, protocol hash, and
producing Slurm job id; every solve is replay-certified; the three FINDINGS
logs record every experiment with sources and costs; the experiment tracker
holds 74+ runs with idea lineage. Total compute across both campaigns: ≈90
GPU-node-hours of a 1000 nh allocation (supervised ≈37 [SV closeout];
self-play milestones ≈25 [SP entries]; variants lab ≈28 of its 50-nh budget
[VL§12, VL§17, VL§18–22 ledgers]).

---

## Open questions for the owner

1. **One paper or three?** PAPER_PLAN argues A/B split for the supervised
   half; the self-play arc (§3–§5) could be paper C or fold into A as its
   second act. The ceiling-break story reads strongest joint (A + §4).
2. Venue for the joint story if merged (PAPER_PLAN suggests AIJ/JAIR for A,
   TMLR for B; the self-play+ceiling-break arc might retarget NeurIPS/ICML).
3. The two pending transfer legs and M6's first results: hold the draft for
   them or freeze here?
4. Author-facing: the missing figures (PAPER_PLAN's A1/A5/B1/B5 "MISSING"
   items) still need drawing/consolidation.

## Known number discrepancies (audit trail)

- Backward g24r4 regret: 4.22 [SV§6] vs 4.20/4.22 in SP§7's quotation —
  rounding of the same payload (`scaling/results/g24r4/comparison.json`);
  this draft uses 4.22.
- v09 unseen 173 (seed 7) vs 177 (seed 8): tables use the seed-8 arm,
  labeled as such [VL§5].
- B2 base ceiling 99.6% [SV§16] vs 98.0% [SP§3]: different probe budgets
  (see Limitation 7).
- Earlier chat-quoted "+1.47 vs perfect" for the flagship unseen row is its
  own-solved∩known-optima stat (135/137); the table's +0.90 is the
  93-common-subset figure — both correct, different subsets.
