# PAPER_PLAN.md — what to write, from what, and what we may not say

Written 2026-08-16 from the fourteen reviewer memos (§75, archived under
`analysis/review_2026-08-16/`) and `analysis/publishability.md`. Every number is
quoted with its FINDINGS entry; nothing here is new measurement.

## 1. Two papers

A asks *which way of formulating the search scales*. B asks *how good a
synthetic answer key must be before it can replace an exact solver*. B is
domain-general and would be buried inside A; A is the thesis and has the bigger
holes. Split them.

### A — *Executable Abstraction: Measuring and Lifting the Expressiveness Ceiling of Subgoal Planning*
**Venue: Artificial Intelligence (AIJ) first, JAIR near-equal** — deep
single-domain studies are welcome there, and the arc descends from Bacchus &
Yang's downward-refinement work, itself an AIJ paper.

*Abstract.* We compare two learned planners on Ricochet Robots — one searching
move by move, one searching backwards over a vocabulary of subgoals — at an
identical budget on identical pinned pools, counting a puzzle solved only when
its plan replays under the full physics, a test that cuts the subgoal planner's
historical 99.6% to 53.1% (§1). Bug fixes plus checking playability inside the
search recover 89.1% at about 7 search steps (§3), and a network-free
exhaustive probe shows the rest is a property of the plan language, not of
training: for 9.3% of puzzles no playable plan exists in the vocabulary at all
(§4). Controlled surgery on the vocabulary lifts that ceiling 90.7 → 97.6 →
99.6% (§16). On the whole pinned pool the subgoal planner wins every rung above
base scale by +8.0 to +47.8 points (§46, all p < 0.0005), while the move-level
planner keeps the quality crown wherever an exact solver can still grade — and
that solver dies with scale, failing on 0 → 29.8 → 40.9% of puzzles along the
robot axis and 48.4 → 64.2 → 61.1% along the grid axis (§18b). We report the
counterweights: efficiency in a shared physics unit (median 64 vs 5,776 `slide`
calls, §25), budget curves showing the 16×16 collapse is partly a cap artifact
(§28), and a negative result — retraining on extended-vocabulary labels is
bistable and usually hurts (§44/§45/§47).

### B — *Good Enough Answer Keys: A Size-Free Neural Labeler and the Fidelity Threshold for Replacing an Exact Solver*
**Venue: TMLR first (claims-supported reviewing suits a calibration study),
NeurIPS Datasets & Benchmarks as alternative.** Venue choice here is our
judgement, not a reviewer's.

*Abstract.* Training a planner needs an answer key, and exact solvers stop
producing one as problems grow. We remove the one size-locked tensor from a
graph-transformer value network and use it as a certified-descent labeler, so a
single net trained only on boards up to 16×16 labels boards from 17×17 to
96×96. Against exact ground truth it picks the same best candidate 86.4–92.6%
of the time at every one of twenty rungs from 17 to 64 with no cliff (§65,
§70) — 90.1% at 64×64, four times its training size — while its absolute value
calibration slowly drifts, an ordering-holds/calibration-drifts split (§59). We
then test the labels downstream, training twin planners on NN labels and on
exact labels of the same boards: at 91.0% and 89.4% agreement the two are
indistinguishable on solve rate, and at 82.2% the NN-trained planner falls from
89.4% to 68.3% solved (§71, §73, §74). Label fidelity gates downstream utility
with a threshold in the 82–91% band; the labeler costs 9–250× more compute than
the exact solver it replaces (§59), so it buys reach, not savings.

## 2. Figures and tables (max 6 each), with the file holding the data

**A**

| # | Content | File |
|---|---|---|
| A1 | Schematic: slide physics, the two formulations | MISSING (to draw) |
| A2 | Oracle mortality, two curves + 10× probe (§18b) | `scaling/data/g*/bench.jsonl.meta.json` |
| A3 | Headline: per rung graded/frontier/pooled, McNemar + clustered CIs (§22, §46) | `eval/results/stats_tests.json` |
| A4 | Ceiling arc 53.1 → 89.1 → 90.7 → 97.6 → 99.6 + failure families (§4, §13, §16) | `analysis/artifacts/ceiling_probe_results{,_b1,_b2}.json` |
| A5 | Two currencies: steps vs median `slide` calls, plus solve-vs-budget (§20, §25) | `eval/results/compute_accounting.json`, `budget_curves_by_rung.json`; **iso-wall-clock at all six rungs MISSING** |
| A6 | Retraining bistability: val_regret vs frontier solve (§44/§45) | `analysis/artifacts/seed_spread.json`, `valnet_modes.json` |

**B**

| # | Content | File |
|---|---|---|
| B1 | The flat 17→64 ladder, one net, one protocol (§65, §69, §70) | `nn_labeler/results/dgate_ladder_g{17..31}r4.json`, `capgate_g{24,32}r4.json`, `coarsegate_g{40,48,56,64}r4.json`; **consolidated plotting table MISSING** |
| B2 | Three-cell dose-response: fidelity vs downstream solve (§74) | `nn_labeler/results/twin_gate_g{24r4,32r4,24r8}.json` + `scaling/results/g{24r4,32r4,24r8}/comparison*_nntwin.json` |
| B3 | 2×2 seed square at g24r4 — twin seeds straddle the exact arm (§71, §72a) | `scaling/results/g24r4/comparison_nntwin{,-seed21}.json`, `comparison_exactseed21.json` |
| B4 | Ordering holds while calibration drifts, 8→32 (§54, §59) | `nn_labeler/results/audit_c2_mix_none_s11_fullcurve.json`, `capgate_g{24,32}r4.json` |
| B5 | Cost: NN descent vs Rust exact per instance (§58, §59) | MISSING as a table (numbers in FINDINGS §59, `runs/nnlab/capstone_g32_4616709.out`) |
| B6 | Beyond verification: 80/96 yields, zero timeouts, warranty (§72c) | `nn_labeler/results/beyond_UNVERIFIABLE_g{80,96}r4.jsonl.manifest.json` |

## 3. Related work — what exists, what we add

| Prior work | What it did | What this adds |
|---|---|---|
| kSubS / AdaSubS | Transformer subgoal generator + best-first search, per-edge reachability | Physics-grounded hand-built vocabulary; whole-plan playability; instance-level ceiling measurement plus controlled vocabulary surgery |
| "What Matters in Hierarchical Search" (2406.03361) | Sets the matched-budget fairness standard | We adopt it and report where we fail it (§25, §28); not a baseline |
| Bacchus & Yang 1994 | Refinement probability as theory | Makes it a measured per-benchmark quantity (§4, §16) |
| DeepCubeA-style learned heuristics | Value nets guiding A* | Nothing novel; our value net plays that role |
| TAMP/HTN; ExIt/AlphaZero | In-search pruning, repair loops, self-play | Cite, do not claim — standard machinery reused |
| Ricochet Robots theory (Hesterberg & Kopinsky; Balanza-Martinez 2024) | Complexity only (PSPACE-complete, W[SAT]-hard in robots) | First learned planner on the domain; a natural hard parameter, not a synthetic one |
| Weak-to-strong / self-distillation / noisy-label thresholds | Learned supervision replacing stronger supervision | Paper B is a calibrated dose-response for it, with a size-free labeler and a measured threshold (§74) |

## 4. Missing experiments, ranked, with rough cost

1. **Fair rescue for the move-level planner** at g16r8 and g32r4 — 3 seeds × 3 learning rates plus self-play at g16r8; the first thing a hostile reader attacks (§75). ~20–30 nh. **IN FLIGHT.**
2. **Seed replicates of the backward headline pair** at two or three rungs; measured seed swing reaches 28 points (§44). ~12–15 nh. **IN FLIGHT.**
3. **By-reference wiring + zero-shot A/B** — the machinery is dead code in the eval driver (§40), so no B2 row is causal. ~2–3 nh. **IN FLIGHT.**
4. **Efficiency chapter rewrite** — median-plus-tail everywhere, same-machine wall-clock only, iso-cost rows. Analysis only. **IN FLIGHT.**
5. **kSubS-style learned-subgoal baseline**, base + one frontier rung; without it we compare our vocabulary, not subgoals. ~10–20 nh plus engineering.
6. **Symmetric budget sweep**, both planners, 1×–10×, every rung. ~15 nh.
7. **Two more seeds on the g32r4 and g24r8 twin cells** — every Paper B caveat is single-seed. ~4 nh.
8. **Size-matched control for B** — subsample exact corpora to twin size and retrain, decoupling fidelity from corpus shrinkage. ~2 nh.
9. **Beam or multi-restart descent at 8 robots**: does pushing fidelity past 89% recover the lost 21 points? ~5 nh.
10. **Planner-side feasibility above 32×32, then one downstream cell at 40 or 48** — the 80/96 labels have no downstream test. ~1 nh probe, then large.
11. **Rush Hour port** — reuses the plan DAG, realizer and self-play loop nearly verbatim; turns a domain result into a method result. ~1 nh.

## 5. Claims we must NOT make

- **"Move-level planning stops working at scale."** Only one seed of an *oracle-supervised* forward planner was measured; forward self-play at scale is unmeasured (§27, §75). Scope to the supervised pipeline.
- **"The 16×16 collapse is not a budget artifact."** At 4× budget forward gains +21.9 and +31.2 points and the backward lead loses significance (§28). Only the grid axis survives.
- **"Subgoals beat flat search."** No learned-subgoal baseline exists; our side also has a hand-written vocabulary, a hand-written repair pass and exact distance tables (§75).
- **"The full language delivers 99.6%."** That is the exhaustive probe's ceiling (§16); learned rows sit at 95.6% and the by-reference step never reached the planner (§40).
- **"The NN labeler saves compute."** It costs 9–250× more than the Rust exact labeler (§59) and the B2 payoff was negative (§68). It buys reach.
- **"The fidelity threshold is 89%."** Three cells, mostly one seed, with fidelity, corpus size and robot count moving together (§73, §74). Say "in the 82–91% band, not yet isolated".
- **"Labels above 64×64 are validated."** UNVERIFIABLE by construction; the flat 17–64 curve is their only warranty (§72c).
- **"Retraining is the scoped fix."** Bistable, and no good basin has ever been seen at an 8-robot rung (§44, §45, §47).
- **Mean compute ratios, or wall-clock across machines.** Median plus tail, same-machine pairs only (§25, §27).
