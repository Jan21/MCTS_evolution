# Findings — subgoal planning vs move-by-move planning

Written 2026-07-14. Plain-language record of what has been established, in the order it
was established. Every claim cites the file that proves it. For protocol detail see
`COMPARISON.md`; for the canonical audit see `SOURCE_OF_TRUTH.md`; the live web view is
the comparison page (claude.ai artifact 785268c8).

## The goal

Test the claim: **planning with subgoals (the "backward" planner) is more efficient and
more useful than planning one move at a time with a neural network (the "forward"
planner), especially as puzzles grow** (more robots, larger boards). Constraints
throughout: no exact solver at solve time (networks + search only), no change to how the
subgoal method fundamentally works, all comparisons on identical puzzles at identical
search budgets, and a puzzle counts as solved only if the plan **plays out legally move
by move** on the real board.

## The findings

1. **The backward planner had been grading itself on paper plans.** Its historical
   "99.6% solved" counted plans complete in its internal notation that often could not
   be played (e.g. bouncing off a robot that is not really there). Forced to play every
   plan out, it scored **53.1%** (239/450) — the honest starting point.
   Source: `eval/results/comparison_backward.json`.

2. **Most of that gap was bugs, not a flawed idea.** Four defects found and fixed, each
   verified zero-regression: (a) a robot could be scheduled to bounce off itself;
   (b) one robot could hold two plan roles at once; (c) a plan could claim a stopper
   cell without ever placing a robot there; (d) the plan-to-moves converter executed
   steps in a stricter order than the plan's own cost model assumes (two separate
   ordering corrections). Playable rate on the matched 150-puzzle slice:
   **54.0% → 63.3% → 74.7% → 79.3% → 80.0%**.
   Sources: `eval/results/comparison_backward_postfix*.json`,
   `eval/results/realizer_twophase_ab.json`, `eval/results/realizer_reorder_ab.json`,
   fix history in `COMPARISON.md`.

3. **Checking playability inside the search finished the climb: 89.1% at ~7 search
   steps.** Each partial plan is physics-checked as it is built; doomed branches are
   discarded immediately (verified: zero false discards). Official full-450 result:
   **401/450 = 89.1% playable, 2.15 extra moves on average, 7.1 search steps,
   1.07 s per puzzle**. The forward planners need **36–424 search steps** on the same
   puzzles. Sources: `eval/results/final450_backward_prefix.json`,
   `eval/results/prefix_check_ab.json`.

4. **The remaining gap is a measured property of the plan language, not of training.**
   For **42 of 450 puzzles (9.3%)** no playable subgoal plan exists at all: 22 cannot
   even be decomposed into subgoals, 20 decompose but no decomposition survives legal
   playback, and only 7 failures are the networks' fault. Ceiling: **90.7%**; the
   planner sits at 89.1%. Lifting it requires extending what a plan can say — the
   scoped next step is a stopper robot that may vacate/return after its bounce is
   consumed. Sources: `analysis/artifacts/ceiling_probe_results.json`,
   `SOURCE_OF_TRUTH.md` §5.

5. **At the base scale (16×16, 4 robots) the forward planner wins on raw quality.**
   Its best version: 450/450 solved, 0.067 extra moves — against backward's 89.1% and
   2.15. Backward wins efficiency by one to two orders of magnitude; at a 10-step
   budget it solves 78.7% vs forward's 2.4–34.7%. Sources:
   `eval/results/comparison_forward.json`, `eval/results/budget_curves.json`.

6. **As puzzles scale, the picture flips — the core of the thesis.**
   - **The forward approach's teacher dies.** Supervised forward training needs an
     exact solver for labels; that solver fails (at its practical budget of 200,000
     search expansions plus a time cap) on **0%** of benchmark puzzles at 4 robots,
     **29.8%** at 6 robots, **40.9%** at 8 robots, **48.4%** on 24×24 boards,
     **61.1%** on 32×32 boards (`scaling/data/g32r4/bench.jsonl.meta.json`) — past
     that point the exact method fails on the majority of puzzles.
     Precision on what "fails" means: the solver is complete in principle — these are
     resource caps being hit, and the required compute grows exponentially with scale.
     Measured directly (all 184 8-robot failures re-run at 10× budget, 2,000,000
     expansions, via the Rust engine in 87 seconds): **75/184 (41%) become solvable
     with 10× the compute** (their optima are 6–10 moves); **109/184 (59%) remain out
     of reach**, leaving a 24% failure rate at even the 10× budget. The wall is real
     and moves outward with scale; it is a cost explosion, not an implementation
     defect. Source: session artifact `cap_probe_rust_results.jsonl` (copied to
     `scaling/data/g16r8/cap_probe_10x_results.jsonl`).
     Curve sources: `scaling/data/g16r6/bench.jsonl.meta.json`,
     `scaling/data/g16r8/bench.jsonl.meta.json`,
     `scaling/data/g24r4/bench.jsonl.meta.json`.
   - **The "tie at 6 robots" did not survive its control test — corrected here.** The
     first 6-robot forward network's training destabilized after one pass, and against
     that weak baseline the two systems tied at ~87%. A control retrain at a lower
     learning rate fixed the instability (validation top-1 0.71 → 0.878), and the
     corrected head-to-head reads: forward **314/316 (99.4%), 0.096 extra moves,
     64.2 search steps** vs backward 275/316 (87.0%), 2.22, 72.9. So on the puzzles the
     oracle can grade, a properly trained forward planner still wins at 6 robots, and
     backward's search-step advantage disappears there (its candidate lists grow with
     robot count while its playability drops). Sources:
     `scaling/results/g16r6/comparison.json` (first run),
     `scaling/results/g16r6/comparison_forward_control.json` (corrected forward row).
   - **Beyond the oracle's reach, the ranking flips — the thesis's strongest direct
     evidence.** The 134 6-robot puzzles the exact solver could not grade (the hardest
     subset) were run with no reference — solving is self-certifying there, since a
     plan that plays out legally to the goal proves itself. Result: **backward 70/134
     (52.2%) solved vs forward 65/134 (48.5%)**, with backward using **2.8× fewer
     search steps (296 vs 835) and 3.5× less time (80 vs 281 s)**. Where both solve,
     forward's solutions are shorter (mean 9.8 vs 14.9 moves). This is the first
     measured regime where the subgoal planner beats a properly trained move-by-move
     planner on solve rate — and it is exactly the regime that matters as puzzles
     outgrow exact methods. Source: `scaling/results/g16r6/comparison_ungraded.json`
     (the d-star fields in that file are placeholders; only solve rate, steps, time
     and plan lengths are meaningful).
   - **24×24 head-to-head (232 gradable puzzles):** forward wins quality again —
     **220/232 (94.8%), 0.068 extra moves** vs backward 205/232 (88.4%), 4.22 — but its
     cost explodes with board size: **191 search steps and 275 seconds per puzzle vs
     backward's 26.9 steps and 7.8 seconds** (7× fewer steps, 35× faster). The
     efficiency gap that closed on the robot axis re-opens wide on the grid axis:
     subgoal plans stay a few decisions deep however large the board, while move-level
     search grows with travel distance. And the oracle could grade only 52% of the
     puzzles at this size. Source: `scaling/results/g24r4/comparison.json`.

7. **Self-play works for the backward planner — exactly where it matters.** Since the
   exact solver dies at scale, self-play (the planner learning from its own solved
   puzzles, no teacher) is the only training that survives there. With generation
   restricted to plans that really play out, six self-play rounds cut the probe's
   average excess moves from **~1.10 to ~0.21** (playable-plan excess **1.04 → 0.22**).
   Benchmark verdict (matched 150 puzzles, both systems using the in-search check):
   the self-play-improved networks match the supervised ones on solve rate (130/150 —
   both pressed against the language ceiling), with slightly better move quality
   (2.146 vs 2.169 extra moves) and **18% fewer search steps** (5.8 vs 7.1). At the
   base scale self-play is a wash on rate and a modest efficiency gain; its decisive
   property is that it keeps working where supervised training cannot.
   Sources: `subgoal_selfplay/runs_warm_prefix/iter*_stats.json`,
   `eval/results/arm_prefix_iter5.json`, `eval/results/prefix150_prefix.json`,
   `subgoal_selfplay/arms_index.json`.

8. **Data generation was ported to Rust and adopted for production (2026-07-15).**
   A five-agent team built `rust_datagen/`; its verification replayed 20,649 decision
   contexts with **zero label differences** on pinned inputs, passed a 12-cell
   cross-engine smoke matrix (16×16 through 64×64, re-run green in this environment),
   and measured 50–180× per-core speedups — the 32×32 dataset that was projected at
   8–20 hours generated in minutes. Two findings from that verification matter beyond
   the port and are disclosed here:
   - **The Python labeler's outputs were never canonical.** At equal-score tie points
     its labels depend on Python's internal set-iteration order — it disagrees with
     itself across board representations. About 7% of instances are affected at the
     trajectory level (which decisions get labeled), never the correctness of a label
     for its own trajectory. Adjudicated and accepted for production in
     `rust_datagen/ADOPTION.md`.
   - **A real pre-existing bug in the Python subgoal search**: its cost estimate is
     not "optimistic-only" as its comment claims (a helper-supported hop is priced 2
     in the estimate but 1 when decomposed), so in rare cases (~0.02% of records,
     12/71,200 measured) the "exact" label is slightly off — in the root-caused
     example, the Rust engine's answer was the correct one. Effect on training is
     negligible; recorded for honesty. Sources: `rust_datagen/VERIFICATION.md` §4,
     `rust_datagen/ADOPTION.md`, `rust_datagen/BENCH.md`.
   - Post-swap sanity: Rust-generated label distributions match the partial
     Python-generated shards within noise (24×24/8-robot config, 60k-record samples).

8b. **8-robot head-to-head, first half (2026-07-16).** On the 266 oracle-gradable
   puzzles: backward **230/266 (86.5%), 1.97 extra moves, 53.8 search steps**. The
   forward row is withheld: its network's training collapsed again (validation top-1
   ≈ 0.06, barely above random — same failure signature the 6-robot run showed before
   its learning-rate fix), scoring a meaningless 9.4%. Per the integrity rule
   established at 6 robots, a lower-learning-rate control retrain is queued and the
   row publishes only against a properly trained opponent. Pattern worth naming: the
   forward training recipe that works out of the box at 4 robots has now destabilized
   at BOTH 6 and 8 robots and needs per-scale tuning to function — the tuning cost and
   fragility are themselves scaling evidence, with the caveat that each point has so
   far been rescuable. Source: `scaling/results/g16r8/comparison.json`.

9. **Ladder scoreboard (updated 2026-07-17).** Measured vs planned; "–" = experiment
   defined, numbers pending. Backward numbers are final where shown; forward rows wait
   for their stability-controlled retrains per the integrity rule (§8b).

   | configuration | experiment | backward solved | forward solved |
   |---|---|---|---|
   | 16×16, 6 robots | graded head-to-head | 275/316 (87.0%) | 314/316 (99.4%) |
   | 16×16, 6 robots | beyond the oracle (134) | 70 (52.2%) | 65 (48.5%) |
   | 16×16, 8 robots | graded head-to-head (266) | 230 (86.5%) | 261 (98.1%) after the stability control (`scaling/results/g16r8/comparison_forward_control.json`) |
   | 16×16, 8 robots | beyond the oracle (184) | 88 (47.8%) at 264 steps / 88 s | 93 (50.5%) at 819 steps / 257 s |
   | 24×24, 4 robots | graded head-to-head (232) | 205 (88.4%) | 220 (94.8%) |
   | 24×24, 8 robots | graded head-to-head | – | – |
   | 24×24, 8 robots | beyond the oracle (289) | 154 (53.3%) at 260 steps / 201 s | 44 (15.2%) at 1101 steps / 1166 s |
   | 32×32, 4 robots | graded head-to-head (175 gradable) | 147 (84.0%) at 13.8 steps / 10.2 s | 133 (76.0%) at 469 steps / 1379 s |
   | 32×32, 4 robots | beyond the oracle (275) | 127 (46.2%) at 29.7 steps / 14.6 s | 2 (0.7%) at 1199 steps / 2380 s |

   All backward networks for every rung are trained; the pending cells wait on the
   forward retrains (24×24/8 and 32×32 in progress on the two granted GPUs) and the
   evaluation runs that follow. The comparison page shows the same scoreboard with
   placeholder rows.

10. **The language ceiling widens with scale — measured (2026-07-17).** The ceiling
   probe (exhaustive no-network search + legal playback) re-run on all 105 of the
   backward planner's 6-robot failures: **48 have no complete subgoal plan at all,
   23 have plans but none survives physics** (71/105 = 68% structural), 28 have a
   playable plan the networks missed (guidance headroom), 6 inconclusive. As a share
   of the whole benchmark, the language-structural gap grows from 9.3% at base scale
   to ~16% at 6 robots — so the planned plan-language extension (the vacate/return
   stopper, in implementation now) is the project's highest-leverage remaining change:
   it targets exactly the regime where the backward planner should win. Source:
   `scaling/results/g16r6/ceiling_probe_old_vocab.json`.

11. **The search caps do not shape any result — verified (2026-07-17).** Both systems
   carry practical caps (search-step budgets; the backward planner also limits how
   many unresolved segments a partial plan may hold at once; both value networks clamp
   their predictions at 49/63 moves-to-go). Checks: (a) every "no complete plan
   exists" ceiling verdict re-probed at 4× the plan-width cap — 0 of 70 became
   solvable (one 6-robot instance moved between the two structural categories, still
   a language failure); (b) measured optimal solutions max out at 12–14 moves even on
   32×32 boards (bigger boards lengthen slides, not solutions), far below the value
   clamps. Sources: `analysis/artifacts/cap_sensitivity_base.json`,
   `scaling/results/g16r6/cap_sensitivity.json`.

12. **The language failures decompose into 7 measured families (2026-07-18)** — and
   every base-scale one is a solvable puzzle: the forward planner solved **42/42** of
   them (33/71 at 6 robots; the 38 unsolved are all beyond-oracle and defeat both
   planners at standard budget). Families (base / 6-robot counts): the only sayable
   plan walks through a robot (11/11); a stopper must park on a wall-less cell —
   the worked-example mechanism (10/9); one parked robot must stop two different
   sliders (8/4); the target robot itself must be the stopper (7/6); the same robot
   must stop twice in sequence (4/1); the target must first step out of the way
   (1/0); plan-assembly residual (1/2). **Actionable finding: the vacate/return
   extension in implementation covers ~50% of the ceiling** (21/42 base instances);
   the other half needs robot-ROLE flexibility, not the vacate maneuver. Falsifiable
   predictions for the extension's re-probe are recorded per family. Sources:
   `analysis/failure_families.md`, `eval/results/failure_gallery.html`,
   `analysis/failure_gallery/forward_solutions_*.json`.

13. **The plan-language extension landed and moved the ceiling (2026-07-18).** Two
   additions, both staying inside the subgoal formalism (opt-in, default paths proven
   byte-identical): stoppers may now park on wall-less cells (their delivery recursing
   through the existing machinery), and a plan may schedule a robot to slide aside
   before a named segment (parks, costed honestly — a park plan's cost equals its
   legal move count, demonstrated). Measured with the exhaustive probe:
   **base-scale ceiling 90.7% → 97.6%** (structural failures 42 → 11; 7 recoveries
   move-optimal); **6-robot ceiling 84.2% → at least 96.9%** (proven-impossible
   71 → 2, with 12 probes unresolved at their memory caps). Verification: 249/249
   stored plans realize byte-identically; the default-vocabulary probe re-run is
   per-row identical to the frozen baseline. Cross-validated per-instance against the
   §12 taxonomy: measured coverage 74% of the base ceiling vs the predicted 50% —
   the wall-less stopper type also unlocks alternate plan shapes around most
   robot-role conflicts. The remaining 11 base failures are scoped as bookkeeping
   extensions (a proposal referencing an existing plan node as its support), not new
   vocabulary. The learned planner does NOT yet use the new language — labels and
   retraining are next; until then the benchmark rows stand. Sources:
   `analysis/b1_design.md`, `analysis/b1_extension_notes.md`,
   `analysis/artifacts/ceiling_probe_results_b1*.json`,
   `scaling/results/g16r6/ceiling_probe_b1.json`, `eval/results/realizer_b1_ab.json`.

14. **The beyond-oracle frontier is contested (2026-07-18).** At 8 robots, on the 184
   puzzles the exact solver cannot grade, the properly trained forward planner edges
   the backward planner on solve rate — 93/184 (50.5%) vs 88/184 (47.8%) — reversing
   the 6-robot frontier result (backward 52.2% vs 48.5%). The efficiency pattern
   holds: backward used 3.1× fewer search steps and 2.9× less time, and where both
   solve, forward's solutions are shorter (mean 8.8 vs 13.2 moves; no true optimum
   exists on this set). Caveat that matters: both rows use the OLD backward
   vocabulary — the language-extension networks were mid-training at measurement
   time, and the extension specifically targets the failure classes that dominate
   here. Source: `scaling/results/g16r8/comparison_ungraded.json`.

15. **The extended-language networks deliver — the frontier flips decisively
   (2026-07-19).** The backward networks were retrained on extended-vocabulary labels
   (`nn/data/combined_b1.jsonl`, 108,902 records; proposal net cold, value net
   warm-started per the known instability) and benchmarked with the extended search
   (`--backward-b1`: wall-less stoppers in the proposal step + deterministic park
   repairs of failed complete plans inside the checked search):
   - **Base 450: 95.3% playable (429/450), 9.7 search steps** — up from 89.1% at 7.6;
     regret 2.03. The gap to the 97.6% language ceiling is now ~2 points.
   - **6-robot graded (316): 96.2% (304/316) at 27 steps** — up from 87.0%,
     approaching the forward control's 99.4% at comparable step counts (64).
   - **6-robot beyond the oracle (134): 80.6% (108/134) at 270 steps / 42 s** — up
     from 52.2%, versus the forward planner's **48.5% at 835 steps / 281 s**. On the
     hardest puzzles measurable, the extended backward planner now solves **1.7× as
     many puzzles at 3× fewer search steps and 6.7× less time.** (Solution lengths on
     this set run long — mean ~16.9 moves with no true optimum known; quality-focused
     training/self-play is the scoped follow-up.)
   Sources: `eval/results/final450_backward_b1.json`,
   `scaling/results/g16r6/comparison_b1.json`,
   `scaling/results/g16r6/comparison_ungraded_b1.json`,
   `checkpoints_backward/{policy_b1,value_b1}.ckpt`.

16. **The second language extension (B2) removes nearly all of what remained of
   the ceiling (2026-07-20; first results produced on Karolina).** B1 left 11
   base-benchmark puzzles (2.4%) with no playable plan in the language. B2 adds
   no new vocabulary — it lets a plan *reuse* a robot it has already placed
   (the same parked robot stopping two different sliders; the target robot
   itself serving as a stopper mid-route; a placed helper sliding on to a
   second post, costed as ordinary moves) and generalizes the step-aside
   repair (clearing two robots at once; two-slide step-asides). Everything is
   opt-in (`--b2` on the probe, `--backward-b2` on the benchmark driver;
   `AStar(by_reference=True)`), with the same zero-regression burden as B1,
   all green: default probe per-row identical
   (`analysis/artifacts/ceiling_probe_default_recheck_post_b2.json`, 49/49),
   B1 probe per-row identical (`ceiling_probe_b1_recheck_post_b2.json`,
   49/49), all 249 stored realizations identical
   (`eval/results/realizer_b2_ab.json`). Measured with the exhaustive probe on
   the 11 residual puzzles (`analysis/artifacts/ceiling_probe_results_b2.json`):
   **9 of 11 now have legal, played-out solutions** (idx 156 at exactly its
   8-move optimum; idx 76 — the known two-robot-clearing case — via two
   scheduled step-asides; the other 7 via reuse of placed robots),
   **0 proven impossible**, 2 unresolved with the probe out of search memory
   (idx 405, 427 — an open search-budget question, not a proven wall).
   Base-scale ceiling: 90.7% (original) → 97.6% (B1) → **99.6%** (B2,
   counting the unresolved probes as failures). Superset check: 39/40 of the
   solved sample re-proven under B2; the 40th is expressible by construction
   (B2 is a strict superset of B1's language; the busier search did not
   re-find its plan within caps; `analysis/artifacts/b2_solved40_results.json`).
   The learned planner (unchanged B1-trained nets, zero-shot ranking of the
   new candidate type) with the B2 machinery on the pinned 450:
   **430/450 = 95.6% at 9.7 search steps**
   (`eval/results/final450_backward_b2.json`) — exactly one puzzle above the
   B1 row (idx 76, recovered by the deterministic repair step), no
   regressions. Stated plainly: the language now permits ~99.6%; the achieved
   95.6% is a training gap (the nets have never seen by-reference
   candidates), and label regeneration + retraining is the scoped next step,
   as it was for B1. Machine note: measured on Karolina (CPU lanes on the
   A100 nodes' EPYC 7763, torch 2.13); solve rates, step and move counts are
   machine-independent; wall-clock seconds are not comparable to
   origin-machine rows.

   **At 6 robots the same extension ends all proven impossibility.** The B2
   probe on the 14 instances B1 had not recovered
   (`scaling/results/g16r6/ceiling_probe_b2.json`): the two puzzles B1 had
   PROVEN impossible both flipped — (gra,115), the two-robot-clearing case,
   is solved at exactly its 4-move optimum with two scheduled step-asides,
   and (gra,251), the two-slide-park case, in 8 moves — and 3 of the 12
   frontier-bound beyond-oracle instances also resolved to playable ((bey,53)
   31 moves, (bey,69) 11, (bey,126) 16). The remaining 9 (all beyond-oracle)
   stay INCONCLUSIVE at 2M-frontier/900 s caps. Ceiling at 6 robots: 84.2%
   (original) → ≥96.9% (B1) → **≥98.0%** (B2, conservative), with
   **proven-impossible now 0** (originally 71, then 2 after B1). The two base
   unresolved instances (idx 405, 427) were also re-probed on a 1 TB node at
   a 5M-plan frontier and 30-minute caps and remain INCONCLUSIVE
   (`analysis/artifacts/ceiling_probe_results_b2_deep.json`) — deciding them
   needs the memory-shaped (depth-bounded) probe redesign already scoped in
   the B1 notes, not more wall-clock.

17. **The full-language backward planner, zero-shot, contests every rung it has
   been tried on (2026-07-21; series in progress).** Per the owner's fairness
   directive, the ladder gains "same trained networks, richer plan language"
   backward rows at matched budgets (1200 / top-5, anytime, `--backward-b2`;
   the 16×16 rungs use the base-trained B1 net pair, which generalizes across
   robot counts — the nets rank the new candidate types zero-shot). Landed so
   far, both 16×16 rungs (sources:
   `scaling/results/g16r6/comparison{_b2,_ungraded_b2}.json`,
   `scaling/results/g16r8/comparison{_b2,_ungraded_b2}.json`; measured on
   Karolina — seconds not comparable to origin rows):
   - **6 robots graded (316): 306 (96.8%) at 27.1 steps** — up from 96.2%
     (B1) and 87.0% (old language); forward control 99.4% at 64 steps.
     Frontier (134): 108 (80.6%) at 269 steps — unchanged from B1, vs
     forward's 48.5% at 835 steps.
   - **8 robots graded (266): 262 (98.5%) at 8.6 steps — the backward
     planner's first graded-set win** (forward control 261 = 98.1% at 62.4
     steps, i.e. 7× more search). Old-language backward: 230 (86.5%).
     **Frontier (184): 163 (88.6%) at 183 steps vs forward's 93 (50.5%) at
     819 steps** — +70 puzzles over the old-language row (88), +75 over
     forward, at 4.5× fewer steps. This replaces the §14 "contested frontier"
     verdict: with its full language the subgoal planner wins the 8-robot
     frontier decisively.
   - Honest constants: where both solve, forward's solutions stay shorter
     (backward graded regret ~2.4–2.6 moves); frontier solution lengths run
     long (no true optima exist there — only solve rate / steps / time are
     meaningful, per the house rule).
   - **32×32 graded (175): 154 (88.0%) at 34.5 steps**; frontier (275 — the
     hardest pool in the study, oracle failure 61.1%): **196 (71.3%) at 96.1
     steps**, ~58 s/puzzle (sources:
     `scaling/results/g32r4/comparison{_b2,_ungraded_b2}.json`; per-config
     nets + full language, zero-shot). Their old-language and forward
     opponents are computing in the baseline lanes and complete this rung's
     head-to-head when they land.
   - **24×24/8 graded (161): 148 (91.9%) at 94.9 steps** (source:
     `scaling/results/g24r8/comparison_b2.json`; per-config nets + full
     language, zero-shot) — up from 89.4% old-language; forward control 97.5%
     at 152 steps. Frontier (289): **161 (55.7%) at 635 steps** (source:
     `scaling/results/g24r8/comparison_ungraded_b2.json`) — vs 53.3%
     old-language and the forward control's 15.2%. The zero-shot gain is
     small at this scale (the per-config nets have never seen the new
     candidate types); the series' consistent reading is that the language
     ceiling has moved far ahead of what unretrained ranking can reach —
     retraining on the extended vocabulary is the scoped next step.

## Verdict so far against the goal

> **SUPERSEDED 2026-07-25 — read §27 instead.** The paragraph below is kept as
> the record of what was claimed on 2026-07-23. Three of its sentences do not
> survive the significance testing of §22 and the axis audit of §24. §27 is the
> current verdict and states exactly what changed.

**One-paragraph summary of where the thesis stands — the ladder now fully measured.**
On small puzzles the properly trained move-by-move planner remains the quality
champion (100% at base scale, near-optimal solutions), and it keeps a solve-rate edge
on oracle-gradable sets up to 24×24. But every scaling trend runs one way. The exact
solver its training depends on dies with scale (0% → 29.8% → 40.9% → 48.4% → 61.1%
failures). Its per-puzzle cost explodes on the grid axis (23 minutes per gradable
32×32 puzzle), and at 32×32 it loses the gradable set outright (76.0% vs the subgoal
planner's 84.0% old-language / 88.0% full-language at 100×+ less time). Beyond the
oracle's reach — the regime that defines scale — it collapses: 48.5% → 50.5% → 15.2%
→ 0.7% down the ladder, while the subgoal planner with its full language reads 80.6%
→ 88.6% → 55.7% → 71.3% at 3–40× fewer search steps. With the full language the
subgoal planner also takes its first gradable-set win (8 robots: 98.5% vs 98.1% at 7×
fewer steps). The move-by-move planner's remaining advantage is solution quality
where it solves; the subgoal planner's remaining handicaps are longer frontier
solutions and the measured gap between what its language permits (99.6% base ceiling,
nothing proven impossible anywhere) and what its unretrained networks reach (95.6%
base; small zero-shot gains at scale) — retraining on the extended vocabulary is the
scoped fix.

- **Efficiency: established.** Same puzzles, same rules: subgoals plan in a handful of
  search steps where move-by-move needs hundreds; at tiny budgets the gap is 79% vs
  2–35% solved.
- **Usefulness at scale: strong evidence, accumulating.** The forward pipeline's
  training collapses at 6+ robots while the backward pipeline keeps working, and
  forward's small-scale solve-rate lead is gone by 6 robots.
- **Honest weaknesses, stated plainly:** on small puzzles forward still produces
  shorter solutions and a perfect solve rate; frontier solutions run long (mean ~16.9
  moves at the 6-robot frontier); the extended language's ceiling is realized by the
  exhaustive probe, not yet by the trained networks (95.6% achieved vs 99.6%
  permitted).

18. **The ladder is complete (2026-07-23).** The last cell — 32×32 beyond the
   oracle, the 275 hardest puzzles in the study — reads: **backward (old
   language) 127 (46.2%) at 29.7 steps / 14.6 s vs the properly trained
   forward control 2 (0.7%) at 1199 steps (budget-saturated) / 2380 s** —
   forty minutes per puzzle to solve almost nothing. With the full language
   (§17) the backward planner reaches **196 (71.3%) at 96 steps / 58 s**. On
   the grid axis the move-by-move formulation is not merely slower — at the
   frontier it has effectively stopped working, while the subgoal
   formulation keeps solving at two orders of magnitude less cost. Source:
   `scaling/results/g32r4/comparison_ungraded.json`.

18b. **Errata (2026-07-23).** Three discrepancies between this log's prose and
   the source JSONs, surfaced by the rebuilt report generator's self-check
   pass and re-verified by hand; in every case the JSON wins:
   - §3's "2.15 extra moves" for the checked 450-puzzle row: the source
     (`eval/results/final450_backward_prefix.json`) reads mean regret
     **2.1446** — correct rounding is 2.14, not 2.15.
   - §7's self-play probe arc "~1.10 → ~0.21": the cited per-round files
     (`subgoal_selfplay/runs_warm_prefix/iter*_stats.json`) read strict
     probe regret **0.724 (iter 0) → 0.222 (iter 5)**; the ~1.10 figure
     matches a different arm's index headline, not the cited source.
   - §6's oracle-failure curve (0% → 29.8% → 40.9% → 48.4% → 61.1%) omits
     the 24×24/8 point, which at **64.2%** (289/450,
     `scaling/data/g24r8/bench.jsonl.meta.json`) is the worst measured —
     and the sequence concatenates two axes (robot count at 16×16, then
     grid size); it should be read as two curves, not one ladder.

19. **The Rust label engine speaks the full B2 vocabulary — gated and adopted
   (2026-07-25; work of 2026-07-23/24).** Both labeling engines were extended
   to the current plan language: the Python labeler (`nn/generate.py
   --vocab b2`, mirrored in `scaling/backward_label.py` and the gate dumper)
   now emits transient-support AND supports-by-reference candidates with
   exact committed costs — the labeled candidate set equals what the
   solver's own `_expand` generates at each decision. Park repairs are
   deliberately NOT labeled, matching B1 precedent: they are deterministic
   realization-failure repairs the networks never rank. The Rust engine
   gained the same vocabulary (transient pair enumeration; the three
   by-reference shapes in `apply`; `byref` plan edges; a `vocab` field on
   work items) and passed the VERIFICATION.md gate-3 pattern on NEW pinned
   corpora: four legs (base b2, base b1, 16×16/8 b2, 24×24/8 b2), **68
   pinned decisions, 894 candidate labels replayed Python-vs-Rust, zero
   differences** (job 4591708, 21.5 min; corpora archived at
   `rust_datagen/golden/b2gate/`). Default paths proven unchanged: the full
   cargo suite including the original golden-corpus gates stays green, and
   the base-vocabulary Python driver emits a record set identical to the
   pre-change driver (the only difference is candidate order within
   equal-score ties — the process-level set-iteration class already
   adjudicated in `rust_datagen/ADOPTION.md`, reproduced by a HEAD-vs-HEAD
   control). Operational calibration recorded for reuse: legitimate base B2
   rollouts finish in ~60–1,700 solver iterations, while wandering rollouts'
   per-iteration cost explodes with plan size — label jobs therefore carry
   per-config iteration budgets (50k/100k/200k) and the bridge sends
   attempts in waves (~4× less wanderer exposure; sampling order unchanged).
   Setup validation on this machine also passed: the smoke + shard-
   equivalence job scored 19/20 on both planners with byte-equal
   sharded-vs-unsharded rows (job 4591696). Sources:
   `runs/b2gate/rr-b2gate-4591708.out`, `rust_datagen/golden/b2gate/*.gz`,
   `runs/smoke/rr-smoke-4591696.out`, commits 278bd6e..5f33330.

20. **Budget curves for every budget ≤1200, reconstructed at zero compute
   (2026-07-24).** Because both search families are deterministic and the
   budget only truncates, solve-rate-vs-budget curves are derivable from the
   archived per-instance expansion counts: 42 system curves across 29 result
   files, every curve's 1200-point verified equal to its stored aggregate
   (`eval/budget_curves_from_rows.py` →
   `eval/results/budget_curves_by_rung.json`). Two honest readings, now in
   the report's compute-accounting tab: (a) at the 32×32 frontier the
   backward planner reaches 43.6% by budget 100 while the forward control is
   at 0.0% even at budget 400; (b) the 16×16 frontier forward curves are
   STILL CLIMBING at the 1200 cap (+4–6 points over the last 200 steps)
   while 24×24/8 (+1.0) and 32×32 (+0.4) are nearly flat — so
   budget-saturation is demonstrated on the grid axis but not the robot
   axis; the extended-budget probe jobs (4592278/4592279) exist to decide
   it. A second same-machine reading the report now states plainly: at base
   scale wall-clock favors the FORWARD planner (0.99 vs 1.23 s/puzzle) —
   the subgoal time advantage is real only at scale (1.4×–45×).

21. **Compute-accounting instrumentation adopted, and claimed solves are now
   independently certified (2026-07-25).** `eval/compare.py` + `eval/realize.py`
   gained observational counters (NN calls split policy/value, free forced-exact
   `_expand` calls, rejected complete-plan pops, park plans pushed, and entry
   calls into the prefix-check / strict-realization / park-repair layers) plus
   `--dump-moves`, which records every solved row's realized primitive-move
   sequence. `eval/replay_validate.py` replays those sequences through the
   `simulate.py` physics layer ALONE — no realizer, no plan classes, no
   GridEnv — and requires every move to be a real slide, the length to equal
   the claimed move count, and the target robot to finish on the goal.
   Verification, reproduced this session on a 20-instance bench450 slice
   against a `git worktree` checkout of the pre-instrumentation commit
   (f1a0be9), three lanes — backward anytime-B2, the same plus prefix-check,
   and forward:
   - **3/3 lanes byte-identical** after stripping only the additive
     `accounting` key and wall-clock/provenance fields (canonical
     sort-key JSON, SHA-256 compared).
   - **`--dump-moves` changes no result**: both backward lanes again identical
     to their non-dumping counterparts once the added `moves` list is removed.
   - **Replay certification 19/19 PASS** on each backward lane (the 20th row
     is unsolved and correctly skipped).
   Sources: `eval/results/instrumentation_ab/` (all nine result JSONs, the
   pinned slice, and `ab_compare.py`, the normaliser that produced the
   verdicts).
   **Scope note, stated because the wording elsewhere promised more:** the
   three `physics_calls_*` counters count ENTRY calls into those layers, not
   individual `simulate.slide` invocations, so they are not yet a work unit
   commensurable with the forward planner's physics (which the branch does not
   count at all). A matched `slide`-level counter is the scoped follow-up;
   until it lands, no "total accounted units" column may be published, and
   `eval/report_data.py`'s provenance string overstates what is measured.

22. **Every head-to-head cell now carries a paired test and a clustered
   confidence interval — and two published claims do not survive them
   (2026-07-25).** `eval/stats_tests.py` → `eval/results/stats_tests.json`:
   exact two-sided McNemar on the discordant pairs, plus a 95% percentile
   bootstrap CI (10,000 resamples, seed 0) that resamples BOARDS, not puzzles,
   because up to three puzzles share one board's wall layout (bench450:
   exactly 3 on each of 150 boards). Pairing is refused unless both result
   files record the same `protocol.instances_sha256`. 40 cells; 8 are not
   significant at 0.05. The two that matter:
   - **§17's "the backward planner's first graded-set win" (8 robots, 262 vs
     261) is a 5-vs-4 discordant split: +0.4 points, 95% CI [−1.9, +2.7],
     p = 1.000.** It is parity, not a win. The honest sentence — parity at 7×
     fewer search steps — is the stronger one anyway.
   - **§6's "first measured regime where the subgoal planner beats a properly
     trained move-by-move planner on solve rate" (6-robot frontier,
     old language, 70 vs 65) is 24 vs 19 discordant: +3.7 points, CI
     [−6.1, +13.8], p = 0.542** — not significant. The same rung's 8-robot
     counterpart runs the other way and is equally insignificant (88 vs 93,
     p = 0.583). The regime claim is earned only by the FULL-language rows
     (+32.1 and +38.0 points, both p < 0.0001).
   Also newly non-significant: B2-vs-old-language backward at 24×24/8 on both
   sets (p = 0.42 graded, p = 0.46 frontier) and at 32×32 graded (p = 0.25) —
   consistent with §17's own reading that zero-shot ranking gains little at
   scale; and the old-language 32×32 graded row (84.0% vs 76.0%, p = 0.076),
   so the Verdict's "at 32×32 it loses the gradable set outright" is supported
   by the full-language row (88.0% vs 76.0%, p = 0.0038) but NOT by the
   old-language row it also cites.
   **The pooled graded+frontier union is the stronger headline.** Frontier
   sets are selected by failure of a move-level exhaustive search, which is
   adversarial to move-level planners by construction (objection 0.3); their
   union with the graded half is the whole pinned 450-puzzle pool at that
   rung and carries no such selection. Full-language backward wins it at
   every rung measured, all p < 0.0005:
   | rung | backward B2 | forward | difference (95% CI) |
   |---|---|---|---|
   | 16×16 · 6r | 414/450 = 92.0% | 379/450 = 84.2% | +7.8 [+4.2, +11.6] |
   | 16×16 · 8r | 425/450 = 94.4% | 354/450 = 78.7% | +15.8 [+11.6, +20.0] |
   | 24×24 · 8r | 309/450 = 68.7% | 201/450 = 44.7% | +24.0 [+19.6, +28.7] |
   | 32×32 · 4r | 350/450 = 77.8% | 135/450 = 30.0% | +47.8 [+42.7, +52.7] |
   The old-language backward planner loses the same union at both 16×16 rungs
   (−7.6 and −8.0, p ≤ 0.0003) and wins it at 24×24/8 (+21.6) and 32×32
   (+30.9) — the crossover, with no selection caveat to answer.
   **One caveat on that old-language row, added 2026-07-27 after an
   adversarial re-check:** at g16r6 its two halves were run in different
   planner modes — the graded half anytime realization-checked
   (`comparison.json`), the frontier half prefix-check
   (`comparison_ungraded.json`) — so that single pooled row mixes two backward
   configurations. The other three rungs use prefix-check on both halves. The
   direction is safe (anytime is the stronger mode, so a like-for-like row
   would only deepen the loss) but the row is not as clean as the sentence
   above implies. Every row involving `bwd_b2` or the forward control, which
   is all of the headline table, was verified protocol-matched across halves:
   identical checkpoints, expansions 1200, k 5.

23. **No wall-layout leakage between training and benchmark boards
   (2026-07-25).** The ID ranges were already known disjoint; what had never
   been checked is whether a bench board's wall layout is a near-copy of a
   training board's, which would leak the test set regardless of IDs because
   the networks see geometry, not IDs. `analysis/dedup_audit.py` compares
   interior wall-segment sets (border excluded) by Jaccard similarity across
   train/val/bench for all six configurations: **0 exact duplicates and 0
   pairs at J ≥ 0.9 in every split pair, maximum J = 0.185 (base config,
   train vs bench)**; the worst value anywhere is 0.200. Layouts are
   effectively independent. Recorded alongside it, because it explains why the
   16×16 rungs report identical figures: **configurations that differ only in
   robot count share their board pool exactly** (`grid_data` identical between
   `environments_g16r6`/`g16r8` and between `environments_g24r4`/`g24r8`), so
   the robot axis is measured on the same walls — a property in the study's
   favour that was never written down. Source:
   `analysis/artifacts/dedup_audit.json`.

24. **Correction to §20: the queued extended-budget probes test the grid axis,
   not the robot axis they were justified by (2026-07-25).** §20 concluded
   that budget-saturation is demonstrated on the grid axis but not the robot
   axis, and stated that jobs 4592278/4592279 "exist to decide it". They do
   not: those two jobs probe g24r8 and g32r4 — the two rungs already flat at
   the cap. Reading the stored curves
   (`eval/results/budget_curves_by_rung.json`), forward frontier solve rate
   gained over the last 200 expansions before the 1200 cap:
   **g16r6 +6.0 points (42.5% → 48.5%), g16r8 +4.3 (46.2% → 50.5%), g24r8
   +1.0 (14.2% → 15.2%), g32r4 +0.4 (0.4% → 0.7%)**. The unsaturated rungs
   had no probe, and g16r8 is precisely where §17 claims a decisive frontier
   win (+75 puzzles over forward). The existing two jobs are kept — they turn
   a flat extrapolation into a measurement at the rungs whose numbers (15.2%,
   0.7%) attract the censoring objection hardest — and two more were submitted
   at the rungs that actually decide the robot axis: **g16r6 and g16r8,
   forward-only, budget 4800, 32-instance seeded frontier subsamples (jobs
   4593491 and 4593492, ~1.2 node-hours together)** via the same
   `jobs/patterns/fwd_budget_probe.slurm` recipe, using each rung's own stored
   forward control checkpoint so the probe extends that exact row's budget and
   nothing else. Until they land, the "collapse is not a cap artifact" claim
   is proven on the grid axis only, and must be worded that way.

25. **A matched physics-work unit exists, and it does not say what the
   expansion headline says — in either direction (2026-07-25).** `simulate.slide`
   is the single primitive both stacks bottom out in (forward successor
   generation; backward realization / prefix-check / park-repair BFS), so
   counting it measures both planners in one unit. `eval/slide_counter.py` +
   `eval.compare --count-slides` does that, split into disjoint buckets, and
   is proven inert on results (see below). Pilot, base scale, **the same five
   bench450 puzzles run through both systems**:
   | system | expansions | slide calls / puzzle (median) |
   |---|---|---|
   | backward, full language, anytime | 3.4 | 64 |
   | forward (`best.ckpt`) | 337.0 | 40,448 |
   **CORRECTED 2026-07-27 — the 40,448 figure is not physics.** An adversarial
   review found that ~96 of the ~112 slides per forward expansion come from
   `dest_cells` one-step-lookahead featurization of the states fed to the
   networks (`move_planner/encode.py`), not from generating successors, while
   the backward planner's counterpart featurization reads precomputed
   graph/distance tables and calls `slide` never. The counter was correct; the
   *interpretation* conflated "physics work" with "how each planner happens to
   featurize states". `eval.compare --count-slides` now splits the forward
   attribution into `forward_search` (physics) and `forward_encode`
   (featurization), and the backward loop's diagnostic re-costing of a found
   plan gets its own `abstract_scoring` bucket instead of leaking uncounted.
   Re-measured on the SAME five puzzles:
   | system | physics only (median) | including featurization / diagnostics |
   |---|---|---|
   | backward | **64** | 983 |
   | forward | **5,776** | 40,448 |
   So the honest physics-only ratio is **≈90×, not ≈630×** — against ≈99× on
   expansions. The corrected reading is that the efficiency advantage
   **survives the matched unit essentially unchanged**, neither widening nor
   shrinking; the earlier "in fact widens" was an artifact of the conflation.
   That is still a clear answer to publishability objection 1.1 — the concern
   was that a matched unit would erase the advantage, and it does not — but it
   is a materially weaker statement than the one first recorded here, and the
   sentinel bucket now proves nothing is dropped from either total. But the mean tells a different story
   and the difference is the finding: over the 20-instance slice the backward
   planner's mean is **5,961 slide calls with a median of 74.5**, because a
   single **unsolved** puzzle (env 2405) spent **106,817 slide calls, 96% of
   them inside generalized park repair** — 89.6% of the whole slice's physics
   work, and more than the most expensive puzzle the forward planner solved
   (93,888). Two consequences, both to be honoured in the paper:
   - Compute accounting must be reported as **median plus tail**, never mean
     alone; a mean would hand a referee a number that misrepresents both
     systems.
   - **Park repair is the physics cost centre**, and it is the one component
     the networks never rank (it fires only after a complete plan fails, and
     its repairs re-enter the frontier on raw abstract cost with no value-net
     score — verified in `_nn_astar_backward`). That makes the §19 decision
     not to label park repairs defensible as a *labelling* decision and
     simultaneously identifies where the planner's unbounded cost lives.
   These are pilot numbers on 5–20 base puzzles with `best.ckpt` as the forward
   arm; the definitive per-rung figures come from the Track 1 accounting pass.
   Sources: the corrected split figures are
   `eval/results/instrumentation_ab/v3_split_{backward5,forward5}.json`
   (re-measured 2026-07-27 with `forward_search`/`forward_encode`/
   `abstract_scoring` separated); the original pre-split pilot is retained as
   `v2_slidepilot_{backward5,forward5}.json` and `v2_n4_both.json`, whose
   `forward_search` bucket lumps featurization in and must not be quoted as
   physics.

   **Verification of the whole accounting branch, round 2** (same harness as
   §21, baseline `instrumentation_ab/after.L1.json`): plain, `--dump-moves`,
   `--count-slides` and both flags together are **4/4 byte-identical** on
   results; `--dump-moves` no longer costs an extra BFS (the approach leg
   reuses the candidate BFS's own path) and all 20 dumped sequences are
   byte-identical to the re-derivation version; replay certification passes
   **19/19** on both dumping lanes, **4/4** on a 16×16/8 frontier lane and
   **5/5** on a forward lane — forward solves are now independently certified
   too. The `d_star` placeholder sentinel (objection 0.6) fires on that
   frontier lane: `d_star`/`regret` null per row, `mean_regret`, `pct_optimal`
   and `n_negative_abstract_regret` suppressed, `protocol.d_star_placeholder`
   recorded. This retires a live footgun — the stored g16r8 frontier B2 file
   publishes `mean_regret: 17.81`, which is mean solution LENGTH, not regret.
   Operational note for Track 1: `eval.replay_validate` needs `--env-dir`
   (or `RR_ENV_DIR`) for any non-base configuration; it defaults to
   `environments/` and will otherwise fail on a missing board.

26. **The frontier margin is not a selection artifact: it holds in every
   stratum (2026-07-25).** Frontier sets are the puzzles a move-level
   exhaustive search failed to grade, so membership is defined by an oracle's
   failure — adversarial to move-level planners by construction
   (publishability objection 0.3). `analysis/frontier_strata.py` partitions
   each frontier set by two proxies computed from the instance and its board
   ALONE, never from d*, solver status or any result: whether the goal cell
   has a wall on any side (an unbacked goal can only be reached by parking a
   helper first — the domain's classic hardness signal) and the target robot's
   Manhattan travel, split at each set's own terciles. Across 4 rungs and 44
   strata the full-language backward planner leads the forward control in
   **43 and ties the 44th** (the n=7 cell named below); **39 are
   significant at 0.05** and every insignificant
   cell has n ≤ 13. No reversal anywhere. Two readings worth keeping:
   - The margin is *largest* where the mechanism predicts. At both 16×16 rungs
     the "goal OPEN" class (no wall behind the goal, so a helper must be
     parked) is where the backward vocabulary should pay, and it does:
     g16r8 goal-OPEN 87.6% vs 47.1% (+40.5, n=153) against goal-walled
     93.5% vs 67.7% (+25.8, n=31).
   - The easiest cell behaves as expected: g16r6 "goal walled / travel low"
     (n=7) is 100% vs 100% — parity, not a manufactured win.
   Source: `analysis/artifacts/frontier_strata.json`.

   **Training-data budget, both systems, per configuration** (objection 0.9;
   `analysis/data_budget.py` → `analysis/artifacts/data_budget.json`). The
   units are not interchangeable — a backward record is one candidate at one
   search decision, a forward record is one state on an optimal move
   trajectory — so the table states the unit rather than pretending one number
   compares. With that caveat the asymmetry is large and runs AGAINST the
   backward planner:
   | config | backward (decisions) | forward (moves) | ratio |
   |---|---|---|---|
   | 16×16 · 4r (base, old vocab) | 181,768 | 512,752 | 2.8× |
   | 16×16 · 6r | 97,779 | 1,038,827 | 10.6× |
   | 16×16 · 8r | 116,271 | 1,284,864 | 11.1× |
   | 24×24 · 4r | 53,789 | 916,526 | 17.0× |
   | 24×24 · 8r | 109,180 | 1,376,431 | 12.6× |
   | 32×32 · 4r | 51,353 | 951,030 | 18.5× |
   Every non-base configuration trains both systems on the same 700 boards;
   the forward planner receives 10–19× more supervision records and still
   loses the pooled union at 24×24/8 and 32×32 (§22). Stated with its unit
   caveat this strengthens rather than weakens the comparison, and it is the
   honest way to answer 0.9 — the sets are NOT comparable, and the imbalance
   is not in the winner's favour.

27. **Verdict, restated (2026-07-25).** Supersedes the 2026-07-23 paragraph
   above. Every number here is from a result JSON named in §§21–26; the
   changes from the previous wording are itemised at the end so nothing is
   quietly dropped.

   **Where the thesis stands.** On the puzzles an exact solver can still
   grade, the properly trained move-by-move planner remains the quality
   champion — 100% at base scale with near-optimal solutions, and a
   solve-rate edge that is statistically real at 16×16/6 (−2.5 points,
   p = 0.039) and 24×24/8 (−5.6, p = 0.035). At 16×16/8 the two systems are
   **at parity** (98.5% vs 98.1%, p = 1.000) with the subgoal planner using 7×
   fewer search steps. But on the whole pinned pool at each rung — graded and
   beyond-oracle puzzles together, which is the only view free of any
   selection — the full-language subgoal planner wins **every rung measured**,
   by +7.8, +15.8, +24.0 and +47.8 points (16×16/6, 16×16/8, 24×24/8,
   32×32/4), all p < 0.0005, with the margin growing monotonically along both
   hardness axes. Beyond the oracle's reach the gap is 80.6 / 88.6 / 55.7 /
   71.3% against forward's 48.5 / 50.5 / 15.2 / 0.7%, it holds in all 44
   oracle-independent hardness strata with no reversal (§26), and it is
   achieved while training on 10–19× fewer supervision records from the same
   boards. The exact solver that the forward pipeline's supervision depends on
   fails on 0 → 29.8 → 40.9% of puzzles along the robot axis and 48.4 → 64.2 →
   61.1% along the grid axis — two curves, not one ladder (§18b).

   **Efficiency: established, and it survives a matched unit.** Subgoal plans
   take a handful of search steps where move-by-move needs hundreds, and the
   advantage does not evaporate when both systems are measured in the same
   physics unit: on identical base puzzles, 64 vs 40,448 median `slide` calls
   (§25). The honest qualifications are that the backward planner's compute is
   heavy-tailed — a single unsolved puzzle can cost more physics than the
   forward planner's most expensive solved one, almost all of it in park
   repair — and that at base scale wall-clock actually favours forward
   (0.99 vs 1.23 s/puzzle); the time advantage is real only at scale.

   **Honest weaknesses.** Forward still produces shorter solutions wherever
   both solve, and a perfect base-scale solve rate. Frontier solutions run
   long and have no known optimum. The extended language's ceiling (99.6% at
   base) is realised by the exhaustive probe, not by the trained networks
   (95.6%) — retraining is the scoped fix and is queued, not done. Budget
   saturation is demonstrated on the grid axis only; the 16×16 forward
   frontier curves are still climbing at the cap and their probes are pending
   (§24). And the at-scale forward opponent is the oracle-supervised pipeline
   only — forward self-play at scale is unmeasured, so no claim here is a
   claim about move-level planning as such.

   **What changed from the 2026-07-23 verdict, and why.**
   - *"takes its first gradable-set win (8 robots: 98.5% vs 98.1%)"* →
     **parity at 7× fewer search steps.** A 5-vs-4 discordant split,
     p = 1.000 (§22). The replacement sentence is stronger and true.
   - *"forward's small-scale solve-rate lead is gone by 6 robots"* →
     **false as written**; forward's graded lead at 16×16/6 is significant
     (p = 0.039). What is gone by 6 robots is the lead on the *pooled* pool.
   - *"at 32×32 it loses the gradable set outright (76.0% vs 84.0%
     old-language / 88.0% full-language)"* → keep the full-language half
     (p = 0.0038), **drop the old-language half** (p = 0.076, not
     significant).
   - *"every scaling trend runs one way"* → **"every trend along the two
     hardness axes bends the same way"**, with the graded-set exceptions named
     above; the original sentence is contradicted by the study's own graded
     rows.
   - The frontier series is now reported with its selection mechanism stated,
     its stratified breakdown (§26), and the pooled union as the headline —
     because the union needs no caveat and says more.

28. **The 16×16 frontier wins are budget-dependent — a negative result
   (2026-07-26).** The probes §24 argued for landed (jobs 4593491/4593492,
   COMPLETED in 1.6 h and 1.4 h; forward-only, budget 4800, seeded 32-instance
   frontier subsamples; sources `scaling/results/g16r6/forward_probe_e4800.json`,
   `scaling/results/g16r8/forward_probe_e4800.json`). On the SAME 32 instances,
   matched against the stored rows:
   | rung | backward B2 @1200 | forward @1200 | forward @4800 | bwd vs fwd@1200 | bwd vs fwd@4800 |
   |---|---|---|---|---|---|
   | 16×16 · 6r | 23/32 = 71.9% | 13/32 = 40.6% | **20/32 = 62.5%** | +31.2, p = 0.021 | +9.4, **p = 0.581** |
   | 16×16 · 8r | 30/32 = 93.8% | 14/32 = 43.8% | **24/32 = 75.0%** | +50.0, p < 0.0001 | +18.8, **p = 0.070** |
   Forward gains **+21.9 and +31.2 points from 4× budget**, so the 1200-cap
   frontier rows at 16×16 are heavily censored, exactly as the still-climbing
   curves predicted. Consequences, stated plainly:
   - The **matched-budget** claim stands untouched — that is the protocol, and
     at 1200/1200 both margins are significant.
   - The claim that the frontier collapse "is not a cap artifact" is now
     **proven false on the robot axis**. At 16×16 it substantially IS a cap
     artifact. §17's "with its full language the subgoal planner wins the
     8-robot frontier decisively" must be reworded to name the budget.
   - What survives is the grid axis, where the curves are flat (+1.0 and +0.4
     points over the last 200 expansions) and the extended probes
     (4592278/4592279) are still running.
   Honest limits of this probe: n = 32 gives modest power, so p = 0.070 at
   g16r8 is underpowered rather than evidence of parity — the point estimate
   still favours backward by 18.8 points; and backward was NOT given 4800, so
   this is a robustness probe against a deliberately over-budgeted opponent,
   not a like-for-like row. Both readings belong in the paper.

29. **The 2×2 nets/language cell closes objection 1.3: the gain is language,
   not net provenance (2026-07-26).** §17's 16×16 rows used base-trained B1
   nets with the full language, while their old-language comparators used
   per-config nets — confounding the two. Job 4593535 filled the missing cell
   (base-B1 nets + OLD vocabulary, flags identical to `comparison_b2.json`
   minus `--backward-b2`; 1 h 59 m, ~0.2 nh). Holding the nets fixed at the
   base-B1 pair, the **language** effect is:
   | rung / set | old vocabulary | B2 vocabulary | difference (95% CI) | p |
   |---|---|---|---|---|
   | 16×16 · 6r graded | 282/316 = 89.2% | 306 = 96.8% | +7.6 [+4.2, +11.2] | <0.0001 |
   | 16×16 · 6r frontier | 80/134 = 59.7% | 108 = 80.6% | +20.9 [+13.3, +28.8] | <0.0001 |
   | 16×16 · 8r graded | 232/266 = 87.2% | 262 = 98.5% | +11.3 [+7.4, +15.5] | <0.0001 |
   | 16×16 · 8r frontier | 95/184 = 51.6% | 163 = 88.6% | +37.0 [+29.9, +43.9] | <0.0001 |
   Holding the LANGUAGE fixed at the old vocabulary, the **net-provenance**
   effect (base-B1 nets vs the rung's own per-config nets) is +2.2 (p = 0.065),
   +7.5 (p = 0.006), +0.8 (p = 0.727) and +3.8 (p = 0.092) — small, and every
   sign FAVOURS the base nets. So the borrowed base nets were not a handicap
   the language gain had to overcome; if anything they were slightly better
   than the per-config nets even at 6 and 8 robots. The language effect is
   **2.8–15×** the provenance effect at every cell. Objection 0.11's "zero-shot means
   two different things" is retired for the 16×16 rungs by measurement rather
   than by the pending retraining. All 689 solved rows the job produced were
   replay-certified, 0 failures. Sources:
   `scaling/results/g16r{6,8}/comparison{,_ungraded}_basenets_oldvocab.json`,
   `eval/results/stats_tests.json` (52 cells).

30. **The B2 label campaign as configured is infeasible, and the cause is
   measured (2026-07-26).** All five `rr-b2lab-*` jobs were run; two hit the
   24 h walltime (4591709 base, 4591710 g16r6) and the rest were cancelled at
   0.6–2.9% completion. Rates from the jobs' own logs — not extrapolated from
   their opening graphs, which are 10–45× faster than the steady state and are
   what misled the first projection:
   | config | boards | done | s/graph | projected | node-hours |
   |---|---|---|---|---|---|
   | g16r4 | 2112 | 121 (5.7%) | 692 | 406 h | 50.7 |
   | g16r6 | 1050 | 9 (0.9%) | 2,696 | 786 h | 98.3 |
   | g16r8 | 1050 | 30 (2.9%) | 2,418 | 705 h | 88.1 |
   | g24r8 | 1050 | 6 (0.6%) | 4,579 | 1,335 h | 166.9 |
   | g32r4 | 1050 | 10 (1.0%) | 1,013 | 295 h | 36.9 |
   **~441 node-hours for labels alone against ~879 remaining.** This is the
   Rust engine at 16 threads, so there is no faster implementation to switch
   to. The cause is diagnosable at zero compute from the engine's own
   per-attempt records (`scaling/data/<cfg>/rust_work/backward_b2.results.jsonl`,
   which carry `status` and `iters`):
   | config | budget | `ok` | `empty` | `budget_exhausted` | share of ALL iterations spent on `budget_exhausted` |
   |---|---|---|---|---|---|
   | g16r4 | 50,000 | 66.6% | 24.6% | 8.8% | **65.1%** |
   | g16r8 | 100,000 | 66.0% | 17.9% | 16.1% | **91.3%** |
   Legitimate rollouts are cheap — median **342** iterations at base and
   **174** at g16r8, p95 ≈ 16–20k — so the iteration budget sits 30–100× above
   the legitimate median and a wandering rollout is free to burn all of it.
   §19's calibration ("~60–1,700 iterations") described the median correctly
   but the budget was set from the tail, and an iteration budget does not
   bound wall-time. Capping the per-rollout budget buys, per the same records:
   | cap | base: `ok` kept / speed-up | g16r8: `ok` kept / speed-up |
   |---|---|---|
   | 2,000 | 72.8% / **9.2×** | 81.8% / **24.7×** |
   | 5,000 | 83.3% / 4.6× | 87.6% / 11.9× |
   | 10,000 | 89.7% / 2.8× | 91.4% / 6.7× |
   A cap does not cost instances one-for-one: the bridge draws `per_graph * 4`
   = 80 attempts to keep 20, so at a 66% success rate there is ~2.6× headroom.
   Per-board yield under a cap (from the recorded attempt streams, which are
   truncated once 20 keepers are found and therefore pessimistic): at cap
   10,000, 93.7% of base boards and 92% of g16r8 boards still reach 20
   keepers. The second lever is `--per-graph`: B2 emits ~19.5 records per
   instance, so 10 instances/board over all 2112 base boards still yields
   ~410k records — nearly 4× `nn/data/combined_b1.jsonl` (108,902). What a cap
   DOES change is which instances are labelled — it biases the training set
   toward puzzles whose rollout succeeds quickly — and that must be stated
   wherever the retrained rows are reported. Recommended configuration, to be
   smoke-tested with `--limit` before any full submission: cap 5,000–10,000,
   `--per-graph` 10, sharded ≤16 h per the cooling reservation.

   **Operating rule adopted, at the cost of ~11 node-hours:** never size a
   generation job from its opening graphs. Run `--limit`-bounded smoke tests
   and project from the steady-state rate. The first projection here (58
   s/graph from 8 graphs) was wrong by 12× at base and 42× at 24×24/8.

31. **The grid axis is budget-sensitive too — "saturated" was my inference,
   not a measurement, and it was wrong (2026-07-27).** §20 read the
   reconstructed curves as showing the 24×24/8 and 32×32 forward frontier
   curves "nearly flat" at the 1200 cap (+1.0 and +0.4 points over the last
   200 expansions) and §24 concluded saturation was *demonstrated* on the grid
   axis. The extended probe (job 4592278, COMPLETED in 12.4 h) measures it
   directly on the same 32 frontier instances at 24×24/8:
   | | solve rate | vs backward @1200 |
   |---|---|---|
   | forward @1200 | 5/32 = 15.6% | +31.2, p = 0.0063 |
   | **forward @6000** | **9/32 = 28.1%** | +18.8, **p = 0.146** |
   | backward @1200 | 15/32 = 46.9% | — |
   **+12.5 points at 5× budget**, and the backward lead stops being
   significant on this subsample. Mean expansions used at the 6000 cap is
   4,974, so a large share of instances are still exhausting the budget and
   the curve is very likely still climbing.
   **Where the reasoning failed, stated precisely.** The +1.0-point gain over
   1000→1200 was real; calling it "nearly flat" was the error. That is ~1
   point per 200 expansions, and the measured 1200→6000 gain averages ~0.52
   points per 200 — the *same order*. A small per-window gain integrated over
   a 4,800-step extension compounds into 12.5 points. Local flatness over a
   200-step window is not saturation, and no budget curve should be read that
   way again.
   **Consequence for the thesis.** Combined with §28, forward's frontier
   "collapse" is substantially budget-limited at **every rung probed so far**
   — g16r6 +21.9, g16r8 +31.2, g24r8 +12.5 — so the claim must be retired in
   general, not merely scoped to one axis. What remains true and is enough:
   (a) at **matched budget**, which is the study's protocol, the backward lead
   is real and significant at every rung; (b) the backward planner reaches its
   rate at 1200 expansions while forward needs 5,000+ and still trails; (c)
   the pooled-union result (§22) is unaffected, since it is a matched-budget
   comparison. The sentence "at the frontier the move-level formulation has
   effectively stopped working" (§18) is not supportable as written — the
   supportable sentence is that it needs 4–5× the search budget to reach a
   rate the subgoal planner reaches at 1×, and still does not catch up.
   The 32×32 probe (4592279) is still running and will complete the set;
   32×32 forward at 1200 solves 2/275, so even a large relative gain there
   leaves the qualitative picture intact.
   Source: `scaling/results/g24r8/forward_probe_e6000.json`.

32. **The B2 label campaign is complete under the recalibrated settings — and
   the cap that made it affordable depletes the very vocabulary it exists to
   teach (2026-07-27).** All five configurations generated with the Rust
   engine at `--budget-iters 5000 --per-graph 10` (base as 8 shards, merged):
   | config | records | boards | boards at full keeper target | by-reference share |
   |---|---|---|---|---|
   | g16r4 | 376,912 | 2112 | 2110/2112 | 7.1% |
   | g16r6 | 210,686 | 1050 | 1050/1050 | 5.2% |
   | g16r8 | 218,248 | 1050 | 1050/1050 | 4.8% |
   | g24r8 | 210,607 | 1050 | 1050/1050 | 5.1% |
   | g32r4 | 176,593 | 1050 | — | — |
   1.19M records in total, every manifest reading `engine: rust, vocab: b2,
   per_graph: 10`. Cost ~5 node-hours against the ~441 the original settings
   projected (§30).

   **The finding that matters more than the throughput.** The handoff's QC
   gate expects a by-reference share of roughly 5–20% (base measured 13%).
   Every new set sits at the very bottom of that band, and the cause is the
   cap. Measured on the **same 111 base boards**. The cap-50,000 run also used
   `--per-graph 20` against the production 10, so this is not a single-variable
   comparison; the bridge keeps the first `per_graph` successes in attempt
   order, and restricting the old file to its first 10 keepers per board gives
   **14.5%**, so the per-graph difference biases the comparison CONSERVATIVELY
   — at matched settings the depletion is larger, not smaller. The §33
   cap-20,000-vs-5,000 comparison is per-graph-matched and fully clean:
   | cap | by-reference share |
   |---|---|
   | 50,000 (original) | 5,698/42,348 = **13.5%** |
   | 5,000 (recalibrated) | 889/18,568 = **4.8%** |
   A ~3× depletion of exactly the candidate type the retraining exists to
   teach. By-reference plans reuse a robot the plan has already placed, and
   they evidently surface in longer rollouts, so a tight iteration cap removes
   them first. This is a sharper version of the bias recorded in §30: the cap
   does not merely change *which instances* are labelled, it changes the
   *composition of the label set* against the new vocabulary.
   **Why this had to be caught before retraining, not after.** Had the chain
   run through, the likely outcome would have been a modest improvement in
   by-reference ranking and the conclusion "retraining delivers less than B1
   did". That conclusion would have been an artifact of label generation, not
   a property of the method — and it would have been very hard to distinguish
   from the real thing after the fact. Higher-cap label sets are being
   generated at the two cheapest configurations to measure how much of the
   13.5% returns and at what cost, before any retrained row is published.
   Sources: `scaling/data/*/backward_b2.rust.jsonl` and their
   `rust_work/*.manifest.json`; QC by `scaling/qc_byref.py`.

33. **The cap trade-off measured, and a banking criterion of mine retracted
   (2026-07-27).** Two results, one about the data and one about my own method.

   **(a) A 4× cap more than doubles the by-reference yield.** Regenerating two
   configurations at `--budget-iters 20000` against the production 5000:
   | config | cap 5,000 | cap 20,000 | cost ratio |
   |---|---|---|---|
   | g16r6 | 10,900/210,686 = 5.2% | 26,003/236,037 = **11.0%** | 9.3× |
   | g16r8 | 10,446/218,248 = 4.8% | 25,567/246,762 = **10.4%** | 12.5× |
   So §32's depletion is real and largely recoverable: the richer setting
   restores roughly the 13% the original 50,000-iteration budget produced,
   at 2.4× the absolute number of by-reference examples. Cost is trivial for
   the scaling rungs (~0.6–0.8 node-hours each) and ~50 node-hours for base,
   which is the only configuration where the decision has a price. Whether it
   is worth paying is an empirical question — retraining costs ~0.6 node-hours
   per configuration, so the cheap test is to retrain g16r6 on both label sets
   and compare their benchmark rows, rather than reasoning about it.

   **(b) The banking criterion I pre-registered in commit e6c0451 was
   invalid, and I am retracting it.** It required a retrained value net to
   beat or approach the `val_regret` of the checkpoint it warm-started from.
   Those two numbers are computed on **different validation splits** — the
   incumbent's on old-vocabulary labels, the retrain's on B2 labels that
   contain by-reference candidates the old split does not have. A higher
   number can therefore mean "harder validation set", not "worse network".
   Applied as a gate it flagged g16r6 (+0.1617) and g16r8 (+0.2287) as
   failures requiring a seed rerun, which would have wasted compute chasing a
   measurement artifact — while g16r4 showed −0.4144 against a differently
   derived incumbent, so the direction is not even consistent.
   `scaling/bank_b2.py` now reports the comparison as context and gates on a
   signature that IS valid without a matched split: a value run whose **best
   epoch is 0**, i.e. one that never improved on its warm-start at all. By
   that test no run is unstable — best epochs are 13 (g16r4), 12 (g16r6) and
   4 (g16r8, still training). The decisive test remains the benchmark itself.
   Recorded because a pre-registered criterion that turns out unsound has to
   be retracted in public, not quietly replaced.

34. **The first Track 1 definitive row: retraining on the cap-5000 labels makes
   the planner WORSE, and the frontier loss is severe (2026-07-27).** g16r6,
   retrained B2 networks against the pinned sets, all rows replay-certified
   (301/301 graded, 75/75 frontier, zero failures — the numbers are real, not
   a harness artifact):
   | set | retrained | zero-shot B2 | change | p |
   |---|---|---|---|---|
   | graded (316) | 301 = 95.3% | 306 = 96.8% | −1.6 | 0.332 |
   | **frontier (134)** | **75 = 56.0%** | **108 = 80.6%** | **−24.6** | **<0.0001** |
   | pooled (450) | 376 = 83.6% | 414 = 92.0% | −8.4 | <0.0001 |
   Against the forward control the retrained planner still leads the frontier
   (+7.5, p = 0.143) but no longer wins the pooled union (−0.7, p = 0.788),
   where the zero-shot B2 planner won by +7.8.

   **This is the failure §32 predicted, and it is the strongest evidence yet
   that the label-generation cap is the problem.** The retrained networks were
   trained on labels containing 5.2% by-reference candidates, against ~14% at
   the original iteration budget (§33a). The damage is concentrated exactly
   where by-reference plans matter most — the beyond-oracle set, where a
   quarter of the previously-solved puzzles are lost — while the gradable set,
   which the old vocabulary already handled, barely moves. A net trained on a
   corpus that under-represents a candidate type appears to have learned to
   under-rank it, undoing the zero-shot advantage the B2 machinery gave it.
   **No retrained row may be published as a definitive result until this is
   resolved.** §17's zero-shot rows remain the study's best full-language
   numbers for now, and the framing "retraining is the scoped fix" (§16, §27)
   is now a claim the evidence contradicts at this rung.
   The decisive test is running: the same configuration retrained on the
   cap-20000 label set (11.0% by-reference), job 4597769, ~0.6 node-hours. If
   the frontier loss reverses, the cap is confirmed as the cause and base
   labels must be regenerated at the higher budget (~50 node-hours). If it
   does not, the fault lies in the retraining recipe rather than the data, and
   that is a different and larger problem.
   **Update, same day, as the other rungs landed — the pattern tracks the
   by-reference share, which is the hypothesis's own prediction.** All rows
   replay-certified (base 433/433, zero failures anywhere):
   | rung | by-ref in its labels | retrained vs zero-shot B2 | p |
   |---|---|---|---|
   | g16r4 (base), graded 450 | **7.1%** | 96.2% vs 95.6% = **+0.7** | 0.508 |
   | g16r6 graded | 5.2% | 95.3% vs 96.8% = −1.6 | 0.332 |
   | g16r6 frontier | 5.2% | 56.0% vs 80.6% = **−24.6** | <0.0001 |
   | g16r8 graded | 4.8% | 93.6% vs 98.5% = **−4.9** | 0.0010 |
   Base — the only configuration whose labels retained a by-reference share
   near the original — is the only one that did NOT regress; it improved
   slightly, though not significantly. The two configurations with the
   thinnest by-reference corpora regressed, most severely on the set where
   that vocabulary matters most. That is the ordering the depletion
   hypothesis predicts, on data collected before the hypothesis was tested,
   and it is now the primary reason to believe the cap rather than the
   retraining recipe is at fault.
   **Two confounds, one measured by this study and large enough to matter**
   (added after an adversarial re-read; neither was named in the first
   version of this entry):
   - **Net lineage inflates the 16×16 regressions.** The zero-shot comparator
     at both 16×16 rungs is the BASE-B1 net pair, which §29 measured as
     *stronger* than the per-config lineage the retrained nets descend from —
     **+7.5 points on this exact g16r6 frontier set** (p = 0.006), +2.2
     graded, +3.8/+0.8 at g16r8. So of the −24.6 frontier points, roughly 7–8
     are plausibly lineage rather than label depletion. The regression is
     still large after that allowance, but "−24.6 caused by label depletion"
     overstates it and must not be quoted that way.
   - **Base cannot exhibit the signature.** The base configuration has no
     beyond-oracle set, so its "+0.7, no regression" row is measured only on
     the graded axis — where g16r6 also barely moved (−1.6, p = 0.33). The
     dose-response reading therefore rests on the graded series alone
     (+0.7 / −1.6 / −4.9), which is consistent with the hypothesis but weaker
     than the frontier collapse makes it look, and is further confounded by
     g16r8's zero-shot comparator sitting at 98.5%, near ceiling.
   Also unequal: base has 2112 boards and 377k records against 1050 and
   ~210k, and warm-started from a B1 net rather than an old-vocabulary
   per-config one. Excluded as causes by the evidence: checkpoint selection
   (cannot manufacture −24.6) and evaluation artifacts (protocols matched at
   1200/k=5 with equal instance hashes, every row replay-certified). Job
   4597769 — same configuration, same lineage, same recipe, cap-20000 corpus
   — remains the controlled test, and is the only reading that holds lineage
   fixed.
   Sources: `scaling/results/g16r{6,8}/comparison{_b2retrained,_ungraded_b2retrained}.json`,
   `eval/results/final450_backward_b2_retrained.json`, `eval/results/stats_tests.json`.

35. **The last budget probe lands, and it partially restores the collapse claim
   — at 32×32 the move-level planner really has stopped working (2026-07-28).**
   §31 concluded from the 24×24/8 probe that budget-sensitivity was general and
   the "not a cap artifact" claim should be retired everywhere. The 32×32 probe
   (job 4596461, completed) shows that was one rung too sweeping. All four
   probes, each on its rung's own seeded frontier subsample, backward always at
   the standard 1,200 budget:
   | rung | fwd @1200 | fwd @4×–5× | climb | bwd @1200 | bwd − fwd @probe |
   |---|---|---|---|---|---|
   | 16×16 · 6r | 40.6% | 62.5% | +21.9 | 71.9% | +9.4, p = 0.58 |
   | 16×16 · 8r | 43.8% | 75.0% | +31.2 | 93.8% | +18.8, p = 0.070 |
   | 24×24 · 8r | 15.6% | 28.1% | +12.5 | 46.9% | +18.8, p = 0.146 |
   | **32×32 · 4r** | **0/24 = 0.0%** | **1/24 = 4.2%** | **+4.2** | 50.0% | **+45.8, p = 0.0010** |
   At 32×32 four times the budget buys **one puzzle out of twenty-four**, mean
   expansions used 4,733 of 4,800 — the search is still saturating, it simply
   has nowhere to go. The backward planner's lead is the only one of the four
   that remains significant against an opponent given 4× the search.
   **The corrected claim, which is narrower than the original and wider than
   §31's retraction:** budget-sensitivity is large on the robot axis and at
   24×24, so no frontier margin at those rungs may be described as a collapse —
   §31 stands there. At the largest grid scale the collapse is real and
   survives the control. The general sentence "beyond the oracle the move-level
   formulation has effectively stopped working" is supportable **only for
   32×32**, and must be scoped to it rather than asserted across the ladder.
   Source: `eval/results/budget_probe_summary.json` (regenerated with all four
   probes by `eval/budget_probe_summary.py`).

## Still open

- Retraining the backward networks on the extended (B2) vocabulary — the
  measured gap between what the language permits (99.6% base ceiling) and what
  unretrained ranking achieves (95.6% base; small zero-shot gains at scale)
  — plus quality-focused self-play on the extended stack (frontier solution
  length). Requires teaching `rust_datagen` the extended vocabulary first
  (labels at 24×24/32×32 are impractical in Python).
- Deciding the 2 frontier-bound base probe instances (idx 405, 427) via a
  memory-shaped (depth-bounded) probe; `validate_plan.py` and the `park`
  node type.
