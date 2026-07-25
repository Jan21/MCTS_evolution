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
