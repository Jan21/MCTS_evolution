# CLAIMS.md — every headline claim, its strongest evidence, its honest caveat

Reviewer-proofing checklist for `paper/DRAFT.md` (2026-08-26). One line per
claim: **claim → evidence → caveat**. Source tags as in DRAFT.md.

## Supervised campaign

1. **Replay certification changes the science: the backward planner's real base solve rate was 53.1%, not 99.6%.** → `eval/results/comparison_backward.json` [SV§1]. → Caveat: none; this is the paper's honesty rule, applied to ourselves first.
2. **The exact oracle dies with scale (61.1% label failure at 32×32; 10× budget recovers only 41% of 8-robot failures).** → `scaling/data/*/bench.jsonl.meta.json`, 10× Rust re-run [SV§6]. → Caveat: "fails" = resource caps on a complete solver; the wall is cost, not impossibility.
3. **Above base scale the subgoal planner wins every pooled rung, +7.8→+47.8 pts, all p<0.0005.** → `eval/results/stats_tests.json` [SV§22, SV§46]. → Caveat: pooled union avoids frontier selection bias, but the forward side is the *supervised* pipeline only — forward self-play at scale is unmeasured; and the 16×16 collapse is partly budget-limited [SV§28].
4. **The 32×32 headline is seed-robust: median-of-3 +50.2 pts, band 9 puzzles.** → `analysis/artifacts/seed_headline_g32r4.json` [SV§83]. → Caveat: the 16×16·8r rung is bimodal (band 361–424) — one bad basin draw [SV§80/§84].
5. **A fairly retuned forward planner (9 arms, validation selection) still loses pooled by +11.3 pts (p=1.7e-09).** → `comparison*_forward_rescue.json` [SV§86]. → Caveat: rescue ran at 16×16·8r only; other rungs keep the original forward checkpoint (frozen-baseline caveat).
6. **The base-scale quality gap is a plan-LANGUAGE property: 9.3% of puzzles have no playable subgoal plan; vocabulary surgery lifts the ceiling 90.7→97.6→99.6%.** → `analysis/artifacts/ceiling_probe_results{,_b1,_b2}.json` [SV§4, SV§16]. → Caveat: 99.6% is the exhaustive probe under its caps (98.0% under the self-play re-probe's tighter caps [SP§3]); the learned planner reaches 95.6%.
7. **A single size-free value net labels near-optimally from 17×17 to 64×64: argmin agreement 86.4–92.6% at all 20 rungs, no cliff.** → `nn_labeler/results/{dgate_ladder,capgate,coarsegate}_*.json` [SV§65, SV§70]. → Caveat: absolute calibration drifts with size (ordering holds); labeler costs 9–250× the exact solver where both run [SV§59] — reach, not savings.
8. **Label fidelity gates downstream planners in the 82–91% argmin band (91% equivalent, 82% collapses).** → twin suite [SV§71–§74]. → Caveat: pre-registered causality arms show argmin fidelity is a *predictor*, NOT the mechanism — 82% corrupted-label training caused no harm at 4 robots [SV§85]; the g24r8 collapse is robot-count/error-structure linked.
9. **80×80/96×96 boards were labeled with zero timeouts.** → `beyond_UNVERIFIABLE_*` manifests [SV§72c]. → Caveat: named UNVERIFIABLE — the flat 17→64 curve is their only warranty; no downstream test exists yet (M6 in progress).

## Self-play campaign

10. **A label-free self-play loop (physics certification only) beats the frozen supervised subgoal planner on both pinned 24×24 exams, p<1e-4.** → `results/selfplay/g24r4_b2_iter*/` vs `scaling/results/g24r4/comparison*_b2.json` [SP§19]. → Caveat: the mechanism is search distillation (cheap search reaches what deep search reached); realized moves on shared solves did not improve inside the fixed language.
11. **On 200 never-trained boards the self-play line solves 177/200 vs supervised backward's 134 and forward's 101, with fewer both-solved moves than backward (p=0.004).** → `results/variants/baselines/*, v09_strict_value_s8/*` [VL§5]; recomputed via `report/compare.py`. → Caveat: one board size (24×24), uniform instances; forward baseline is the original checkpoint.
12. **From RANDOM initialization, one label-free iteration matches the supervised per-size pair on unseen boards (132 vs 134, n.s.).** → `results/variants/v08_cold_start/` [VL§4]. → Caveat: one seed, one iteration; the supervised prior is still worth ≈40 unseen solves vs the mature line.
13. **The size-free self-play value net beats the labeler that initialized it at every audited size up to 64×64 (0.838 vs 0.814 at 64).** → `results/audit/b2it0_m1mixed.json` vs `labeler_prod_v1_s11_g56g64.json` + `nn_labeler/results/audit_*.json` [SP§17]. → Caveat: base-vocabulary decisions; the B2-loop nets pay 1–3 pts there (specialization), fixed by the mixed curriculum.
14. **One mixed-size curriculum iteration set frontier records everywhere, incl. 269/289 at 8 robots vs 161 recorded (p≈1e-17 class).** → `results/selfplay/mix_b2mix_iter1/` [SP§20]. → Caveat: gains came from the distribution switch; iterations 2–3 were flat [SP§22].

## The ceiling break (flagship)

15. **Widening the search space with 1–3 leading robot slides breaks the proven language floor: +0.94 (d2) and +0.86 (d3) extra moves vs the exhaustive +1.17 floor at 24×24.** → `results/variants/v07_hybrid_actions/bench_graded_hybrid_d{2,3}.json` vs `results/ceiling/g24r4_b2.json` [VL§8, VL§20, SP§3]. → Caveat: floor and planner measured on the same pinned exam but by different instruments (exhaustive probe vs learned search); the probe had 4 inconclusive puzzles.
16. **The break is the idea, not compute: same nets at matched budget under standard search stay above the floor (+1.42); paired moves 30/0, p=1.9e-9.** → `bench_graded_stdmcts.json` [VL§8]. → Caveat: hybrid uses ~17% more expansions than the control's 499 (570–604); the 30/0 pairing dwarfs this, but iso-expansion exactness is approximate.
17. **The break is net-independent (v14 nets: 33/0 graded, 36/0 unseen) and transfers zero-shot across size and robot count (17/0 at 32×32, 23/0 at 8 robots; 8-robot frontier record 276/289).** → `bench_*_v14nets.json`, `results/variants/v07_transfer/` [VL§16, VL§19]. → Caveat: two transfer control legs pending at draft time; depth-2's gain is mostly broader slide screening, not true multi-slide plans [VL§14].
18. **Against the near-optimal forward planner the both-solved gap shrinks to 0.87 moves while solving 231 vs 220 (graded).** → wave-3 pairing, `bench_graded_hybrid.json` vs `scaling/results/g24r4/comparison.json` [VL§8 processing]. → Caveat: forward keeps the pure-quality crown (+0.07); claim is best *overall* planner, never best moves-when-it-solves.

## Methodology

19. **AlphaZero's visit-count policy target is HARMFUL here (collapse at p≤1e-7); the certified-cost softmax target is load-bearing.** → `results/variants/v01_visit_policy/` [VL§4]. → Caveat: one domain, one architecture; stated as a domain finding, not a general law.
20. **Training the value net on the graded metric itself (realized strict moves) is the best single network change (2-seed win, Fisher 0.0015).** → `results/variants/v09_strict_value{,_s8}/` [VL§5]. → Caveat: gains are search-side; it did not compound over chained iterations [VL§16].
21. **Slide-aware TRAINING is unnecessary: uniform and search-ranked slide nudges both flat at 2 seeds each — the search supplies what training was meant to.** → `results/variants/v15*,v16*/` [VL§21]. → Caveat: only nudge-style injection tested; full hybrid-generation self-play (slides inside the generation search) was not run.
22. **Matched-protocol, single-delta, 2-seed-gated variant testing found 4 adopted / 4 killed / 2 non-replicated ideas in ≈28 nh.** → `variants/DESIGN.md` §7, `variants/FINDINGS.md` [VL§1–§22]. → Caveat: the unseen exam that powers it exists at one size; "wins" are one-iteration effects unless chained (and chaining twice showed non-compounding).
