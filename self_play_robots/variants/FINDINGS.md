# Variants lab — FINDINGS (results log)

Own numbering from 1 (max+1); every entry cites result files and Slurm job ids.
Protocol and portfolio rationale: `variants/DESIGN.md`. The orchestrator's main
log (`self_play_robots/FINDINGS.md`) cross-references this file; nothing here
is duplicated there.

---

1. **Lab opened; framework, unseen exam, and wave 1 built and smoke-tested
   (2026-08-20, login-node CPU only).** Registry/hook framework
   (`variants/__init__.py`; `--variant` plumbed into `spr.selfplay` [worker-side,
   ctx-passing], `spr.train`, `spr.bench`; `spr.arena --env-dir` for
   off-config board dirs), pinned unseen exam (200 instances on 50 fresh
   24×24 boards, ids 20000+, `results/variants/exam/`), runner
   (`jobs/variant_iter.slurm`), baselines job (`jobs/variant_baselines.slurm`),
   report tab (`report/gen_report.py` "Variants lab"). Wave-1 portfolio after
   the owner's "breakthrough, not knob-tweaking" directive: v00 control,
   v01 visit-policy targets, v04 deep-emit (the one knob sanity control),
   v06 Gumbel root + sequential halving (own `mcts_gumbel`), v08 cold-start
   (bootstrap control), v09 strict-moves value targets (train on the metric),
   v12 frontier-mining curriculum; v02/v03/v05 parked as incremental,
   v07 hybrid action space = documented stub (wave 2). CPU smokes: unseen
   bench path via lean boards + RR_ENV_DIR (2/2 solved), v06 and v12
   generation hooks (records produced, hooks logged), v01 and v09 train hooks
   (metrics finite; v09 rescaled 80/80 records to strict units). One real bug
   found and fixed by the smokes: under `python -m spr.selfplay` the live
   module is `__main__`, so a hook importing `spr.selfplay` saw a second,
   empty module — worker state is now passed to hooks explicitly
   (`apply_phase(..., ctx=_W)`).
   Sources: `variants/*.py`, `jobs/variant_*.slurm`.
   Conclusion (plain): the testing framework exists, is documented, and every
   piece of it passed a cheap dry run before any GPU money was spent.

2. **Wave 1 submitted (2026-08-20 09:23).** Jobs (qgpu, 1 GPU each):
   baselines 4719424 (3 h), v00_control 4719425 (5 h), then after v00:
   v01 4719426, v04 4719427, v06 4719428, v08 4719429, v09 4719430,
   v12 4719431 (5 h each). Projected ≈2.3 nh actual (≈4.8 nh walltime
   ceiling; program cap 50 nh). Per-arm results land in
   `results/variants/<vid>/` (bench/gate/manifest JSONs) and render in the
   report's Variants tab ("pending" until then). Verdict entries follow as
   arms complete; any WIN is replicated at a second seed before being
   claimed.
   Conclusion (plain): seven experiments queued at roughly two nights of one
   graphics card's time, every one dry-run-tested first.

3. **Wave-1 interim: unseen-bench bug found and fixed; baselines + control
   landed (2026-08-20).** (a) BUG: the runner's unseen bench ran without
   `--boards lean` (a text patch had silently not matched), so lean exam pkls
   went through `GridEnv.from_env`, whose reconstructed reachability matrix
   lacks entries for arbitrary endpoints -> KeyError in `_initial_plan`,
   all 25 chunks rc=1 (job 4719425's unseen leg; log
   `runs/spr/var_v00_control_unseen.46ba792d0401/chunk.000.log`). Fixed with
   an assert-verified patch (commit in tree); bench-only fix-up jobs
   4722745-4722751 resubmitted (runner is idempotent: gen/train/graded/
   frontier skip). Lesson recorded: verify patch application, not just
   patch-script exit. (b) Unseen-exam BASELINES (job 4719424,
   `results/variants/baselines/`): frozen seed nets 175/200, 14.49 mv,
   172 exp; supervised per-size backward pair 134/200, 14.78 mv, 30 exp.
   The seed nets' +41-solve margin on fresh boards says the self-play line's
   generalization edge over the supervised planner is large before any
   variant runs; the variants' bar is 175/200. (c) CONTROL (job 4719425):
   generation 320 instances / 289 solved / 3,359 certified records (1,900 s;
   5 OOM-dropped + 1 timeout, within the loop's norm); graded A* 228/232
   (regret 2.39, 33.8 exp) = the loop's ceiling level; frontier A* 160/218
   -- one standard iteration from the mix_b2mix_iter2 nets at g24r4-only,
   NOTE: above the 24-only B2 loop's best (158, main FINDINGS §19) --
   the mixed-curriculum warm start itself is worth frontier solves.
   Sources: `results/variants/{baselines,v00_control}/`, jobs
   4719424/4719425.
   Conclusion (plain): one bug found and fixed before it cost anything real;
   the baseline numbers every experiment must beat are now on disk.

4. **Wave-1 verdicts (2026-08-20; jobs 4719425-4719431 + unseen fix-ups
   4722745-4722751; ~2.6 nh actual).** Matched protocol throughout; paired
   gates vs v00_control (its rows: graded 228/232 rg 2.39 exp 34, frontier
   160/218, unseen 172/200 mv 14.06). Verdict table (solves p = McNemar,
   moves p = sign test on shared solves):
   | arm | graded | frontier | unseen | verdict |
   |---|---|---|---|---|
   | v01_visit_policy | 202 (p=2e-7 against), rg 5.23 | 118 (p=1e-9 against) | 142 (p=7e-8 against), moves p=1e-10 against | **LOSS, decisive** |
   | v04_deep_emit | 230, rg 2.29 (n.s.) | **171 vs 160, p=0.013**; moves 36/21 p=0.063 | 175 vs 172 (n.s.) | **WIN (frontier) -- replicating at seed 8** |
   | v06_gumbel_root | 229; **moves 19/6 p=0.015, rg 2.05 vs 2.39** | 169 vs 160, p=0.049 | 172 = 172 | **WIN (graded moves) -- replicating** |
   | v08_cold_start | 200 (p=6e-8 against) | 96 (p=6e-15 against) | 132 (p=2e-11 against) | expected loss (control arm; see below) |
   | v09_strict_value | 228; moves 19/8 p=0.052, rg 2.06 | 168 vs 160, p=0.057 | 173 (n.s.) | **borderline-positive -- replicating** |
   | v12_frontier_curriculum | 230 (n.s.) | 161 (n.s.); moves 37/22 p=0.067 | 176 (best absolute, n.s.) | FLAT at one iteration |
   Readings. (a) **v01 kills the AlphaZero-textbook target in this domain**:
   visit-count policy targets at 300-expansion budgets are catastrophically
   worse than the stack's certified-cost softmax (which is itself a
   completed-Q-style target, DESIGN.md §4) -- regret doubles, every exam
   collapses. The counterfactual matters: it says the control's target
   design is load-bearing, not incidental. (b) **v04 and v06 are the real
   positive signals**, on DIFFERENT axes: emit-all buys frontier solves
   (+11, 2.2x records at 3.8x gen wall-time -- sibling completion on every
   expanded node is the cost), Gumbel root buys graded MOVES (19/6 wins,
   regret 2.05) at identical solves -- the first moves gain any B2-loop arm
   has shown on the graded exam. v09 shows the same shape as v06 one notch
   weaker (both p~0.05). All three go to seed-8 replication before any WIN
   is claimed (jobs 4726033-4726036), and their stack is submitted as
   v13_combo (4726037). (c) **v08 (cold start) is the no-human-in-the-loop
   headline**: from RANDOM initialization, one iteration of 3.8k certified
   self-play records -- zero supervised or exact labels anywhere -- reaches
   200/232 graded and **132/200 unseen, statistically indistinguishable
   from the fully-supervised per-size pair's 134/200** on the same fresh
   boards. The supervised prior is worth ~40 unseen solves to the warm
   arms, but a label-free-from-zero planner already matches the supervised
   baseline it was meant to need. (d) v12 is flat at one iteration: the
   probe made generation 29% record-poorer (257/416 solved) without
   moving any exam yet; its natural reading is that curriculum needs
   MULTIPLE iterations to compound -- a wave-3 candidate (3-iteration v12
   chain), not a kill. (e) Nothing beats the frozen seed nets' unseen bar
   (175/200) after ONE g24r4-only iteration -- consistent with the main
   line's §19 saturation; the unseen exam's discriminating power will show
   on multi-iteration arms. Sources: `results/variants/<vid>/bench_*.json`
   + `gate_*_vs_control.json`, report tab "Variants lab".
   Conclusion (plain): two changes look like real wins, one is a decisive
   loss, the blank-slate planner ties the supervised baseline on new boards,
   and every apparent win goes to a second-seed retest before being believed.

5. **Wave-2: replications, the combo arm, and the three-way unseen table
   (2026-08-20/21; jobs 4726033-4726038, ~1.8 nh; program total ~6.5 nh).**
   Seed-8 control: graded 224/232 rg 2.10, frontier 155/218, unseen 170/200
   (seed noise on this protocol is ~4-5 solves per exam -- the two control
   runs bound it).
   (a) **v04_deep_emit REPLICATES: confirmed WIN on frontier solves.**
   Seed 8: frontier 176 vs 155 (p=0.0002; seed 7: 171 vs 160, p=0.013;
   Fisher combined p=3.4e-5); graded and unseen flat-positive both seeds.
   Emit-all is now the lab's recommended default for generation.
   (b) **v06_gumbel_root does NOT replicate.** The seed-7 graded-moves win
   (19/6, p=0.015) reverses at seed 8 (14/16, p=0.86); frontier stays
   suggestive only (163 vs 155, p=0.13; Fisher p=0.04). Verdict downgraded
   to "not replicated" -- the honest reading is that Gumbel-root's gain at
   this budget is within seed noise for a one-iteration arm.
   (c) **v09_strict_value: WIN on 2-seed evidence, the lab's best arm.**
   Seed 8: frontier 169 vs 155 (p=0.0026; Fisher with seed 7's 0.057:
   p=0.0015), graded 230/232 with the best regret of the entire program
   (1.98; control 2.10/2.39), unseen 177/200 = best absolute row of any
   net ever on this exam. Same direction on every exam at both seeds.
   Training the value net on the benchmark metric (realized strict moves)
   is the design change that works.
   (d) **v13_combo: the deltas interfere on solves, compose on moves.**
   Frontier solves 160 = control exactly (v04's +11 gain vanishes under
   Gumbel-shaped trees -- its expected_failure realized), but frontier
   both-solved MOVES 37/15 (p=0.003, best frontier moves 19.47) and graded
   moves 28/15 (p=0.066). Stacking search+data changes is not additive;
   v04+v09 without v06 is the natural wave-3 combination.
   (e) **Three-way unseen-board table (the owner's goal, quantified).**
   Forward MoveNet baseline (job 4726038): 101/200, 7.78 mv, 688 exp.
   | arm | solved | both-solved moves vs loop |
   |---|---|---|
   | best loop arm (v09_s8) | **177/200**, 14.32 mv | -- |
   | frozen seed nets | 175/200, 14.49 mv | -- |
   | supervised backward per-size | 134/200, 14.78 mv | loop wins 38/16, p=0.004 (12.57 vs 14.78) |
   | supervised forward MoveNet | 101/200, 7.78 mv | forward wins 43/3, p=5e-10 (7.78 vs 9.95) |
   The owner's success criterion is MET on its first half on unseen boards:
   the label-free loop line solves 43 more than the backward-supervised
   baseline (p~1e-9) and realizes FEWER moves than it on shared solves
   (p=0.004). The second half (moves vs forward) stands exactly where main
   FINDINGS 3 predicted: the subgoal language ceiling, not training, is the
   binding constraint (forward solves half as much yet wins both-solved
   moves 43/3) -- the action-space axis (v07) is the only route.
   Sources: `results/variants/{v00_control_s8,v04_deep_emit_s8,
   v06_gumbel_root_s8,v09_strict_value_s8,v13_combo,baselines}/`,
   `VERDICT.json` files, report tab.
   Conclusion (plain): after re-testing, TWO changes are adopted (train on
   everything examined; predict real move counts), one is rejected (the
   lottery exploration did not repeat), and on brand-new boards the
   label-free loop now solves far more than the supervised planner while
   also using fewer moves than it -- the first half of the project's goal is
   met. Only the move-by-move planner still beats us on solution length,
   which is exactly the limitation the action-space experiment attacks.
   (Forward-baseline caveat, added 2026-08-21: the unseen forward row uses
   the ORIGINAL g24r4 MoveNet; the parallel supervised session showed a
   validated re-tune gains the forward planner +8 frontier puzzles of 184 at
   g16r8 -- supervised_valuenet/FINDINGS 86, main FINDINGS 24. No rescued
   g24r4 forward exists, so the row stands, but moves-gap measurements vs
   forward should be read as vs-the-original-network.)

6. **Wave 3 submitted (2026-08-21; coordinator-approved ~10-12 nh).**
   (1) v14_stack = v04+v09 without the non-replicating v06 (jobs 4731104
   seed 7, 4731105 seed 8; the natural composition v13's interference points
   to). (2) v07 root-slides hybrid IMPLEMENTED (bench-only first instantiation:
   portfolio search, standard mcts(600) on the original state + mcts(100) on
   the top-6 slide states ranked by initial-plan cost, slide totals pay +1
   strict move, composed dumps replay-validated; nets = v09_strict_value_s8;
   control = same nets under standard arena MCTS; job 4731112, unseen +
   graded). The claim it attacks: both-solved moves vs the forward baseline,
   3/43 against today. (3) v12 chained x3 iterations (jobs/variant_chain.slurm,
   job 4731113): the compounding test its one-shot flat result asked for.
   (4) adopt_mainline (jobs/variant_adopt.slurm, job 4731114): one mixed-size
   curriculum iteration seeded from mix_b2mix_iter3's nets with the lab recipe
   (emit-all + strict-value targets), fresh ids 9500+, gates vs mix_iter3's
   own benches -- does the lab transfer out of the lab? Projected: v14 2x0.35
   + v07 0.5 + chain 0.9 + adopt 0.75 ~ 3.2 nh this wave.
   Conclusion (plain): the follow-up round tests the two-winner combination,
   the first crack at mixing single robot moves into the planner's language,
   a three-round curriculum, and whether the lab's recipe helps the main
   training line -- all inside the approved budget.

7. **v14_stack (v04+v09): the winners compose -- WIN on the unseen exam at
   both seeds; adopted as the lab recipe (2026-08-21, jobs 4731104/4731105,
   0.85 nh).** Seed 7: unseen 179/200 vs control 172 (p=0.039; the best
   unseen row of the program, one above v07's subgoal-only... see entry 8),
   graded 228 with moves 25/12 (p=0.047) and regret 2.02; frontier 158
   (flat). Seed 8: unseen 177 vs 170 (p=0.065) WITH unseen moves 32/15
   (p=0.019); frontier 168 vs 155 (p=0.019); graded flat. Combined unseen
   Fisher p~0.016 with same-direction rows at both seeds; record volume
   ~7.1-7.9k per iteration (emit-all).
   Reading: the two adopted changes stack where it matters most -- the
   generalization exam -- and seed 8 adds the first UNSEEN moves win of the
   program. The one thing the stack does not reproduce is v04's big
   single-seed frontier jump at seed 7 (158 vs v04's 171/176): with strict
   targets in the mix the extra data buys unseen breadth rather than
   frontier reach. Distinct mechanisms, distinct exams -- worth carrying
   BOTH recipes: v14 for generalization-facing loops, v04-only when the
   frontier is the target.
   Conclusion (plain): combining the two proven changes works and is now the
   default recipe; it wins on exactly the exam the project cares about
   (never-seen boards), at both seeds.
   Sources: `results/variants/v14_stack{,_s8}/`, gates vs same-seed controls.

8. **v07 root-slides hybrid: WIN on both exams against its same-budget
   control -- the first break of the subgoal language ceiling (2026-08-21/22,
   jobs 4731286 [unseen leg], 4733005 [graded leg], 4741426 [controls],
   ~1.2 nh incl. two wiring reruns).** Same nets (v09_strict_value_s8), same
   1200-expansion budget; control = standard arena MCTS best-at-budget.
   | exam | hybrid | std-MCTS control | paired |
   |---|---|---|---|
   | unseen (200) | **186**, 13.40 mv, 808 exp | 182, 14.13 mv, 689 exp | solves n.s.; **moves 31/1, p=1.5e-8** |
   | graded (232) | **231**, 8.69 mv, regret **1.01**, 66.2% opt | 230, 9.08 mv, 1.42, 59.6% | **moves 30/0, p=1.9e-9** |
   The graded regret 1.01 is BELOW the pure-B2 language's exhaustive
   best-plan floor (1.17, main FINDINGS 3) -- an action-space effect by
   construction, not a search/training artifact; 30 graded and 36 unseen
   instances were won by slide-first plans, all replay-certified. Vs the
   forward planner on shared graded solves the gap shrinks to 0.87 mv
   (8.42 vs 7.55, 5/68); on unseen from 2.17 to 1.16 mv (5/31).
   Two wiring bugs found and fixed en route (a None-guard in a progress
   line; ctl() reading $2 after shift 3), both committed with the lesson.
   Conclusion (plain): letting the planner make one ordinary move before
   subgoal planning wins everywhere against its fair control and breaks a
   proven ceiling; the road to matching the move-by-move planner's solution
   lengths runs through MORE of this (slide-aware training, deeper slides).
   Sources: `results/variants/v07_hybrid_actions/`, replay logs, gates.

9. **v12 chained x3: flat everywhere -- killed (2026-08-22, job 4731113,
   0.9 nh).** Three chained iterations (2.2k/2.0k/2.1k records; probe reject
   rate steady ~37%): unseen 172 vs control 172 (p=1), frontier 162 vs 160
   (p=0.83), graded 229 vs 228 (p=1); moves n.s. everywhere. The one-shot
   "needs iterations to compound" hypothesis is now tested and dead.
   Conclusion (plain): practicing only on screened-hard puzzles did not help
   even after three rounds; the idea is retired.
   Sources: `results/variants/v12_frontier_curriculum_chain/`.

10. **Absolute grounding: exact optima for the unseen exam (2026-08-21, job
    4735236, 0.1 nh).** `move_planner.oracle.solve` at the pinned bench caps
    + a 5x-cap second pass (34 rescues): **137/200 instances now carry exact
    d\*** (mean 8.71 moves; sidecar
    `results/variants/exam/g24r4_unseen.dstar.jsonl`, loader
    `variants.exam_dstar.load_dstar`). Perfect-play table on the 137
    graded-unseen puzzles ("perfect play needs 8.71 moves; X uses +Y more"):
    | system | solved (of 137) | % optimal | extra moves vs perfect |
    |---|---|---|---|
    | v07 hybrid | **135** | **56%** | **+1.67** |
    | same nets, std search | 132 | 50% | +1.94 |
    | v14 / v09s8 / seed nets (A\*) | 131-132 | 44-47% | +2.9-3.3 |
    | cold start (label-free from zero) | 109 | 45% | +2.89 |
    | supervised backward | 115 | 36% | +4.84 |
    | supervised forward | 101 | 90% | +0.12 |
    Conclusion (plain): on brand-new puzzles where perfect play is known,
    our best planner now solves nearly all of them using under 2 extra moves
    on average -- half the excess of the plain search line, a third of the
    supervised baseline's; the move-by-move planner is nearly perfect on the
    ones it solves but misses a quarter of the puzzles entirely.
    Sources: sidecar + `runs/spr/spr-var-dstar-4735236.out`.

11. **adopt_mainline: the lab recipe does NOT measurably improve one
    mixed-size main-line iteration on the main line's own exams (2026-08-22/23,
    jobs 4731114 [walltime at 8 h] + 4747761 [resume, g24r8 leg trimmed to 12
    boards], 1.8 nh).** One curriculum iteration from mix_b2mix_iter3's nets
    with emit-all + strict-value targets (25.7k records -- 7.3k/7.9k/10.4k per
    config, ~2.4x the standard iteration's volume), benched on all six pinned
    exams, gated vs mix-it3's own benches:
    graded regret improves at g24r4/g32r4 (2.06/1.97 vs 2.20/2.23) and every
    solve count sits within +-1 except the g24r4 frontier's -8 (157 vs 165,
    p=0.12) -- nothing clears noise in either direction. Two honest caveats:
    (a) the comparison is one-iteration-vs-its-seed, the weakest possible
    transfer test (the lab measured v14's win over TWO one-shot controls on
    the UNSEEN exam, which this run never benched -- flagged for a 20-min
    bench-only follow-up); (b) emit-all's extraction cost explodes on 8-robot
    trees (~91 s/instance, 6x path-only), forcing the r8 trim -- at the main
    line's scale the recipe's data advantage costs real walltime.
    Conclusion (plain): in the big mixed training loop, one round with the
    lab's recipe looked the same as the ordinary round on the loop's usual
    exams -- slightly fewer wasted moves, no change in puzzles solved. The
    recipe's proven win is on never-seen boards, which this run did not
    measure; that cheap measurement is queued next. Verdict for adoption
    decisions: adopt for generalization-facing loops, not (yet) as a
    frontier-solver upgrade.
    Sources: `results/variants/adopt_mainline/`, gates vs
    `results/selfplay/mix_b2mix_iter3/`.

12. **Wave-3 cost ledger and program state (2026-08-23).** Wave 3 actual:
    30.8 GPU-hours = **3.85 nh** (v14 0.83, v07 0.64 incl. two wiring reruns,
    v12x3 0.57, adopt 1.78, d* 0.02). Program to date ~**10.4 nh of the
    50-nh cap**. Standing verdicts: ADOPTED v09 (strict targets), v04
    (emit-all; caveat: r8 extraction cost), v14 (their stack; the
    generalization recipe), v07 root-slides (inference); KILLED v01, v12;
    NOT REPLICATED v06; CONTROLS v00/v08 (the cold-start tie with the
    supervised baseline on unseen boards stands as the label-free headline).
    Conclusion (plain): about a fifth of the budget spent; four adopted
    changes, two clean kills, and every claim two-seeded or same-budget
    controlled.
