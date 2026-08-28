# self_play_robots — FINDINGS (results log)

Owner's choice (2026-08-17): this project keeps its own log, numbered from 1;
`supervised_valuenet/FINDINGS.md` gets one cross-reference entry. Every number
here cites a result file (path) and the Slurm job that produced it; the local
report `report/selfplay.html` regenerates its tables from the same files.
Milestones/gates: `PROBLEM.md` §8. House rules: `PROBLEM.md` §10.

---

1. **Project opened, wiring built, first jobs queued (2026-08-17).** Read
   PROBLEM.md/ASSETS.md and the cited background (FINDINGS §§7, 10–13, 53, 74,
   79, 80, 81; the 80/96 feasibility brief; the prior self-play attempts
   `subgoal_selfplay/` and `move_planner_v2/`). Built `spr/` (see README.md):
   arena wrapper (calls `eval.compare` chunked + merge + replay-certify), the
   plan-language ceiling probe on the MOVES metric, the size-free policy net
   (PolicyTF's heads minus `self.pos`), one warm-start/collapse-guarded trainer
   for both nets, the arena driver for size-free nets (imports the arena's own
   `_nn_astar_backward`), the searches (greedy / A\* with two f-modes and
   best-at-budget / PUCT MCTS with certified terminals), the self-play generator
   (fresh lean boards, labels in the exact labeler's units, provenance), the
   fidelity gauge (Rust exact engine + `nn_labeler.audit_descent`), gates
   (M1 bars, paired McNemar). CPU smoke tests pass end to end; `spr_astar
   --f-mode child` reproduces the arena loop row-for-row on 6 g24r4 instances;
   an independent code review (Opus agent) found one high-severity bug
   (`Certifier` id()-keyed cache could certify a freed plan's successor —
   fixed by keeping the plan alive) and eight medium ones (unit mixing of
   strict vs abstract costs in MCTS Q / A\* bound, unclamped PUCT term,
   stop-after check placement, `--emit all` group aliasing, gauge
   candidate-set comparability, silent `--splits` parse failures, gate None
   handling) — all fixed the same day.
   **Owner decisions (2026-08-17):** §6.1 → BOTH action spaces from the start
   (subgoal main line + primitive-move MCTS at g16r4 in parallel); §6.2 →
   research it: train BOTH net families at two sizes (g16r4, g24r4) and
   compare; log here.
   **Observation from reading the arena:** `eval/compare.py::_nn_astar_backward`
   orders its frontier by `fixed_g(child) + ctg_hat` while `ctg` labels are
   defined relative to the PARENT's fixed cost (`nn/generate.py:93-104`), so
   the baseline's f double-counts each candidate's own fixed increment; a
   label-consistent variant (`fixed_g(parent) + ctg_hat`) is one of the M2 arms.
   No recorded baseline is changed.
   Jobs: M0 arena parity 4679719 (qgpu_exp), ceiling 4679720 (qgpu, 4 arms),
   GPU smoke 4680209 (qgpu_exp), M1 training 4680210 (mixed: policy +
   value_warm + value_cold) / 4680211 (g16 only) / 4680212 (g24 only), M1
   bench 4680213 (afterany the three); M2 search-arm jobs for the PER-SIZE family (no training needed: the supervised pairs) 4680221 (g16r4 v2 pair) / 4680222 (g24r4 exact pair), gated afterok on M0. Projected M1 cost ≈ 1.5–2 nh; note the
   GPU nodes sit in a daily 10:00–18:00 cooling maintenance reservation
   (`scontrol show reservation`), so GPU work runs overnight.
   Sources: `spr/*.py`, `jobs/*.slurm`, `results/status.json`.

2. **M0 arena parity PASSES — the harness reproduces the recorded rows; the
   one non-identical arm is float drift across machines, not wiring
   (2026-08-17, jobs 4679719 = 0.09 nh, 4680465 = 0.02 nh; GPU smoke 4680209
   also green).**
   | arm (spr.arena / spr.fwd.arena, 1200 exp, k=5, CPU 8-wide unless noted) | new | recorded | per-row diffs |
   |---|---|---|---|
   | g16r4 B1-seed21 pair, B2 flags, bench450 (ref recorded on Karolina 2026-08-17) | 432/450, regret 1.840, 55.6% opt, 21.96 exp | 432/450, 1.840, 55.6%, 21.96 | **0/450** |
   | g24r4 exact pair, prefix-check, bench.solved (ref recorded on the origin machine 2026-07-14) | 205/232, regret 4.205, 38.5% opt, 26.909 exp | 205/232, 4.220, 38.5%, 26.909 | 34/232: 31 rows expansions ±1–5, 3 rows realized_strict ±1 |
   | forward candidate_scored.ckpt, `move_planner.evaluate.nn_astar` via spr.fwd, bench450, GPU | 450/450, regret 0.0667, 94.2% opt, 6.436 moves | same | **0/450** |
   Reading. (a) Both references recorded on this machine reproduce
   bit-for-bit (subgoal arena on CPU, forward arena on GPU), so the wiring
   (chunk → merge → replay-certify → parity) is proven. (b) The g24r4 drift
   is numeric: re-running two of the differing instances at 1/2/8 OMP threads
   flips env 913 between 63 and 64 expansions on the SAME machine and code
   (near-tie candidate rankings; 24×24 nets accumulate more float noise than
   16×16), while every aggregate matches (identical solve count and %optimal;
   regret 4.205 vs 4.220 from three ±1 strict-move differences). Verdict:
   M0 gate passed; from here on, same-machine paired comparisons are the
   instrument (spr.gate), and cross-machine references are read at aggregate
   level. Every solved row in all three files is replay-certified.
   Sources: `results/m0/{g16r4_b1s21_b2flags,g24r4_exact_prefix}.json`,
   `results/fwd_m0/astar_candidate_scored{,.parity}.json`,
   `runs/spr/spr-m0-4679719.out`, `runs/spr/spr-fwd-m0-4680465.out`,
   `runs/spr/spr-smoke-4680209.out` (train → arena → MCTS → self-play → gauge
   on CUDA, all DONE).

3. **The subgoal plan language's ceiling on the MOVES metric is far above the
   exact optimum — no subgoal-space planner can approach the forward planner
   on moves; the owner's "both action spaces" call is vindicated, and the
   subgoal loop's honest target is the 2.5-move gap between the supervised
   planner and its own language ceiling (2026-08-17, job 4679720 = 0.06 nh
   + two slack re-runs on the login node).** `spr.ceiling`: exhaustive
   no-network best-first search over partial plans in abstract-cost order,
   every complete plan strictly realized with the arena's certifier, on the
   WHOLE pinned bench; `best` = cheapest certified plan popped before the
   abstract cost reaches it (+ a `slack` re-run to check tightness), `first` =
   the first certified plan popped (what a perfect-ranking first-solution
   search returns). No probe hit its caps in the base arms.
   | bench (n) | vocab | solve ceiling | mean d\* | best plan moves (regret) | % reach d\* | first plan (regret) |
   |---|---|---|---|---|---|---|
   | g16r4 bench450 (450) | base | 408 = **90.7%** (22 no plan, 20 unplayable — reproduces FINDINGS §4 exactly) | 6.26 | 7.68 (**+1.42**; slack-4 re-run 7.66/+1.40) | 61.8% | 8.08 (+1.82) |
   | g16r4 bench450 (450) | B2 (+parks) | 441 = 98.0% (9 inconclusive at 60 s/200k caps) | 6.32 | 7.22 (+0.90) | 66.9% | 7.76 (+1.44) |
   | g24r4 bench.solved (232) | base | 216 = **93.1%** (9 no plan, 7 unplayable) | 7.59 | 9.31 (**+1.72**; slack-4 9.28/+1.69) | 57.4% | 9.73 (+2.13) |
   | g24r4 bench.solved (232) | B2 (+parks) | 228 = 98.3% (4 inconclusive) | 7.64 | 8.81 (+1.17) | 62.7% | 9.35 (+1.71) |
   Reading. (a) **Moves ceiling.** Even with a perfect ranker and unlimited
   search, the base subgoal language cannot get below ≈1.4 (g16r4) / ≈1.7
   (g24r4) mean regret or above ≈60% optimal; the extended B2 language
   lowers this to ≈0.9 / ≈1.2 — all an order of magnitude above the forward
   planner's 0.07 regret / 94% optimal (`comparison_forward.json`,
   `scaling/results/g24r4/comparison.json`). PROBLEM.md §7's "moves closer to
   optimal than forward" is therefore unreachable in the subgoal action space
   at these sizes; it lives in the primitive-move arm (`spr/fwd`, F-M0 passed
   today). (b) **Headroom for the subgoal loop.** The supervised backward
   planner sits at 205/232, regret 4.22, 38.5% opt (g24r4) and 401/450, 2.14,
   50.4% (g16r4 v2 pair): 2.5 / 0.7 moves and 11 / 7 solves above its own
   ceiling — that gap (ranking + search), not d\*, is what M2–M4 can recover.
   (c) **Anytime search is worth ≈0.4 moves at both sizes** even under perfect
   ranking (first vs best): the arena's first-solution convention leaves
   moves on the table, so best-at-budget arms are in M2. (d) The probe is
   cheap (6–15 s per bench for base, ~100 s for B2) and reproduces the
   supervised track's solve ceilings (90.7% base; B2 98.0% here vs the 99.6%
   recorded with 4× caps), so it can be re-run at every size ≤ 64 as the
   moves-ceiling reference for M5.
   Sources: `results/ceiling/{g16r4,g24r4}_{base,base_slack4,b2}.json`,
   `runs/spr/spr-ceiling-4679720.out`.

4. **The size-free policy net does not train under the supervised recipe
   (lr 3e-4, no clipping): it peaks at epoch 0 and degrades; gradient
   clipping fixes it, lr 1e-4 + labeler-encoder init is best (2026-08-17,
   jobs 4680211/4680212 [first M1 attempt] + probe 4680628 ≈ 0.25 nh).**
   First M1 attempt (PolicyTF's recipe verbatim minus `self.pos`): g16r4
   legacy corpus val_regret@1 4.74 (epoch 0) → 6.3–7.75 by epochs 6–24,
   recall@5 0.57 → 0.37; g24r4 4.21 (epoch 0) → 4.9–5.5 afterwards; the
   per-size PolicyTF improves monotonically on the same data (e.g. the B1
   seed-21 policy 2.14 → 1.14 over 25 epochs). Probe on the g16r4 legacy
   corpus, 8 epochs each, seed 21 (val_regret@1 / recall@5 at the best epoch):
   A lr 1e-4 + clip 1.0 → **2.99 / 0.75**; B pe=sin2d, lr 3e-4 + clip → 3.11 /
   0.73 (noisier; FINDINGS 53's sin2d verdict holds for the policy too);
   C encoder (LoopedLayer) initialized from the labeler prod_v1_s11 + lr 1e-4
   + clip → **2.90 / 0.77** (best from epoch 0 on); D lr 3e-4 + clip → 3.07 /
   0.74. Reading: without positional embeddings the pointer heads' gradients
   through 12 weight-tied recurrences blow up at 3e-4; clipping alone
   restores monotone training, and the size-free VALUE encoder transfers to
   the policy (recipe C adopted for M1: 25 epochs, lr 1e-4, clip 1.0,
   encoder init). The size-free value net warm-started from the labeler
   trains cleanly on the legacy g16r4 corpus (val_regret 2.21 → 2.02, top-1
   0.49, spread 1.1–1.5, no collapse; `runs/spr/m1/g16_value_warm_s21`).
   Housekeeping: the unstable runs are kept as `runs/spr/m1/*_lr3e4_unstable`
   (the g24 one was renamed while its job was still logging, which killed
   that arm's CSV logger — irrelevant to the result; retrains queued as
   4680913/4680914/4680915, bench 4680916, size-free M2 4680917).
   Sources: `runs/spr/m1/recipe_g16_*/lightning_logs/version_0/metrics.csv`,
   `runs/spr/spr-m1-recipe-4680628.out`, `runs/spr/spr-m1-g16-4680211.out`.

5. **M2 PASSES for the per-size family at both sizes: MCTS beats greedy AND
   the arena A\* on realized moves at the same solve set and budget; the
   arena's "double-counting" f is not a bug to fix but a depth penalty
   that helps at 24×24 (2026-08-17, jobs 4680480/4680481/4681012/4681029
   ≈ 0.35 nh; per-size supervised pairs, GPU, 1200 expansions, k=5,
   prefix-check, replay-certified; all searches share `spr.search.expand`).**
   | g16r4 bench450, v2 pair | solved | mean moves | regret | % opt | mean exp |
   |---|---|---|---|---|---|
   | greedy (labeler's rule) | 366/450 | 8.03 | 1.94 | 53.3 | 1.3 |
   | A\* f=child, first plan (= arena; matches the CPU record 401/8.38/2.14/7.1 exactly) | 401/450 | 8.38 | 2.14 | 50.4 | 7.1 |
   | A\* f=parent (label-consistent) | 401/450 | 8.36 | 2.12 | 50.1 | 5.1 |
   | A\* f=parent, best-at-budget | 401/450 | 8.27 | 2.03 | 50.6 | 5.2 |
   | **MCTS** (PUCT c=1.5, min backup, best-at-budget) | 401/450 | **7.92** | **1.68** | **55.1** | 28.3 |
   | MCTS mean backup | 401/450 | 7.92 | 1.68 | 55.1 | 28.3 (identical decisions; trees differ by 1–2 visits) |
   | g24r4 bench.solved, exact pair | | | | | |
   | greedy | 135/232 | 10.95 | 3.96 | 43.0 | 1.9 |
   | A\* f=child, first plan (= arena; 11.80/4.20 on CPU, 3 float-drift rows) | 205/232 | 11.84 | 4.25 | 38.5 | 26.6 |
   | A\* f=parent | 205/232 | 12.80 | 5.20 | 34.6 | 15.7 |
   | A\* f=parent, best-at-budget | 205/232 | 12.42 | 4.83 | 34.6 | 16.3 |
   | **MCTS** min (= mean) | 205/232 | **11.48** | **3.89** | 38.5 | 38.6 |
   Paired tests (spr.gate, both-solved subsets, sign test on moves; McNemar
   on solved vectors): MCTS vs greedy g16r4 7.44 vs 8.03 (58 wins / 0 losses,
   p=7e-18; +35 solves, p=6e-11), g24r4 9.06 vs 10.95 (29/0, p=4e-9; +70
   solves, p=2e-21) → **M2 gate met at both sizes** (strictly better moves
   at higher solve rate). MCTS vs the arena A\* on the identical solved sets:
   g16r4 57/0 (p=1e-17), g24r4 15/0 (p=6e-5).
   Reading. (a) Search over subgoal decisions has real headroom: MCTS
   removes 0.46 (g16r4) / 0.36 (g24r4) moves per solved puzzle from the
   supervised A\* rows without any retraining, at ~4×/1.5× the expansions
   (still ≤3% of the 1200 cap) — the loop's data will therefore be better
   than the descent labels it was bootstrapped from. (b) The label-consistent
   f (fixed_g(parent)+ctg) is WORSE at 24×24 (+0.96 regret) and neutral at
   16×16: the arena's extra fixed-increment term penalizes deep commitments
   and buys plan quality with more expansions; kept as-is (§1's observation
   closed: not a bug worth fixing). (c) No search variant changes the solve
   SET (401 / 205 in every arm): with these nets and k=5 the unsolved
   instances are pruned by the nets, not by search — solve-rate gains must
   come from training (M3+) or wider k, and the language ceiling (§3) is
   408 / 216. (d) min vs mean backup are indistinguishable on these shallow
   trees. Sources: `results/m2/persize_{v2_g16r4,exact_g24r4}_*.json`
   (+ `.vs_greedy.json` paired tables), `runs/spr/spr-m2-ps{16,24}-*.out`.

6. **Forward (primitive-move) F-M2: MCTS beats greedy but not A\* — at g16r4
   the forward A\* is already near-optimal and ~17× cheaper than an MCTS that
   matches it (2026-08-17, job 4680466 ≈ 0.4 nh; `spr/fwd`, candidate_scored
   MoveNet, bench450, 1200 expansions, k=5, GPU, replay-certified).**
   | search | solved | mean moves | regret | % opt | mean exp | s/inst |
   |---|---|---|---|---|---|---|
   | A\* (`move_planner.evaluate.nn_astar`, = recorded row) | 450/450 | 6.436 | 0.067 | 94.2 | 36.0 | 3.5 |
   | greedy policy | 371/450 | 6.69 | 0.66 | 73.6 | 12.5 | 0.3 |
   | greedy value | 353/450 | 8.40 | 2.47 | 73.9 | 15.2 | 1.3 |
   | MCTS min backup, first solution | 450/450 | 6.77 | 0.40 | 79.8 | 30.6 | 3.1 |
   | MCTS mean backup, first solution | 450/450 | 6.70 | 0.33 | 79.8 | 46.9 | 4.9 |
   | MCTS min, best-at-budget | 450/450 | 6.429 | 0.060 | 96.2 | 600.3 | 69.4 |
   Paired MCTS-best vs A\* (450 both solved): 12 wins / 5 losses, sign p=0.14
   — not significant, at 17× the expansions (the move tree never closes, so
   best-at-budget runs to the cap or `stop_after`). Reading: in the
   primitive-move space at 16×16 the learned cost-to-go makes A\* the
   efficient expert (as `move_planner_v2/DESIGN.md` argued); MCTS's only
   advantage is exploration, worth ~0.007 moves here. F-M2 gate vs greedy:
   met (450 vs 371 solved, better moves); vs A\*: not met. Consequence for the
   forward loop: use A\* (or MCTS with a stop-after budget) as the data
   expert; the forward arm's headroom lies at 24×24+ (forward A\* 220/232,
   0.068 regret but 191 expansions / 275 s per puzzle) where search cost,
   not quality, is the target.
   Sources: `results/fwd_m2/cand_*.json` (+ `.vs_*.json` paired tables),
   `runs/spr/spr-fwd-m2-4680466.out`.

7. **M1 PASSES — and the size-free rebuild does not merely match the per-size
   supervised planners, it beats them at 24×24 (+4–5 solve pts, −1.8 moves,
   +11–16 optimality pts, 5× fewer expansions) and transfers zero-shot across
   sizes in both directions (2026-08-17, training jobs 4680211/4680212
   [values], 4680914/4681011/4680953/4680954/4680955 [recipe-fixed policies,
   mixed value warm/cold], bench 4680956; ≈ 2.2 nh incl. the discarded first
   attempt).** Every pair benched with the arena's own A\* loop
   (`eval.compare._nn_astar_backward` via spr.bench, 1200 expansions, k=5,
   prefix-check, replay-certified) on both pinned exams; gate = within 3.5
   solve / 6.6 optimality points of the per-size base-vocab pairs.
   | pair (policy: recipe §4; value: warm from labeler unless noted) | g16r4 bench450 (ref v2 pair 401/450, 8.38 mv, 2.14 regret, 50.4% opt, 7.1 exp) | g24r4 bench.solved (ref exact pair 205/232, 11.80, 4.20, 38.5%, 26.9 exp) |
   |---|---|---|
   | **mixed** (g16r4+g24r4 corpora, ONE net for both sizes) | 401/450, 8.26, 2.04, 52.1%, 8.9 exp — PASS (paired 44/37 mv, n.s.) | **215/232, 10.00, 2.43, 54.4%, 5.3 exp** — PASS (+10 solves, McNemar p=0.006; 61/12 mv wins, p=5e-9) |
   | mixed, value COLD (collapsed: constant output, CollapseStop at ep 3) | 401/450, 8.14, 1.92, 53.4%, 14.8 exp | 215/232, 9.80, 2.23, 54.9%, 10.7 exp |
   | g16-only (zero-shot at g24) | 397/450, 8.24, 2.03, 52.6%, 9.2 exp — PASS | 213/232, 10.08, 2.53, 49.3%, 8.6 exp — PASS (+8, p=0.04; 58/10, p=2e-9) |
   | g24-only (zero-shot at g16) | 402/450, 8.15, 1.91, 54.0%, 5.1 exp — PASS (39/21 mv, p=0.03) | 216/232, 9.90, 2.31, 53.7%, 4.1 exp — PASS (+11, p=0.001; 59/11, p=4e-9) |
   (Paired tests above use the same-machine M0 re-run of the per-size pair,
   `results/m0/g24r4_exact_prefix.json`, as §2 recommends; the `.gate.json`
   files pair against the recorded origin-machine file and read 62/11, 58/8,
   60/10 with p ≤ 1e-9 — same verdicts. The recorded reference itself is
   205/232, 11.81 mv, 4.22 regret; the M0 re-run 11.80 / 4.20.)
   Reading. (a) **Gate met by all four pairs at both sizes** (all deltas
   positive except g16-only at g16r4: −0.9 solve pts, inside the bar). One
   size-free pair (mixed) replaces two per-size pairs with no loss at 16 and a
   large gain at 24; per-size val metrics of the mixed nets equal the
   single-size ones (policy 2.97/0.93, value 2.02/0.42). (b) **At 24×24 the
   size-free planner sits within 1 solve of the base-language ceiling
   (215–216 vs 216, §3) and closes 70% of the regret gap to it (4.20 → 2.2–2.5
   vs the ceiling's 1.72), using 4–11 expansions instead of 27.** (c)
   **Cross-size transfer is real in both directions**: trained on 16×16 only,
   the pair beats the 24×24-trained supervised pair at 24×24; trained on
   24×24 only, it is the best 16×16 planner in the table. This is the
   labeler-track headline (FINDINGS 53/54/70) carried into the policy. (d)
   **The value net contributes little to the arena A\* here**: the collapsed
   constant-value control scores the same as the warm value (A\* then orders
   by fixed cost alone over the policy's top-5, whose val recall@5 is 0.98 at
   g24r4 / 0.77 at legacy g16r4) — the size-free POLICY (labeler-encoder
   init + pointer heads) is the improvement; the value net's job in this
   loop is MCTS leaf evaluation and labeling (M2-sf, M3). (e) Bench cost:
   ~1 min per exam 8-wide on one GPU (vs 15–27 min on CPU); the whole M1
   bench (8 exams) took 8 min.
   Sources: `results/m1/{mixed_value_warm,mixed_value_cold,g16_value_warm,
   g24_value_warm}_s21_{g16r4,g24r4}.json` (+ `.gate.json`),
   `runs/spr/spr-m1-bench-4680956.out`, `runs/spr/m1/*/lightning_logs/`.

8. **M2 PASSES for the size-free family too — and MCTS with the M1 nets sits
   at the base language's moves ceiling at both sizes, so the subgoal loop's
   measurable headroom on these two exams is nearly gone (2026-08-17, job
   4680957 ≈ 0.25 nh; mixed size-free pair, same protocol as §5).**
   | exam | search | solved | mean moves | regret | % opt | mean exp |
   |---|---|---|---|---|---|---|
   | g16r4 bench450 | greedy | 348/450 | 7.94 | 1.99 | 52.3 | 1.4 |
   | | A\* f=child (arena) | 401/450 | 8.26 | 2.04 | 52.1 | 8.9 |
   | | A\* f=parent / best-at-budget | 401/450 | 8.39 / 8.28 | 2.17 / 2.06 | 48.6 / 48.9 | 5.7 / 5.8 |
   | | **MCTS** (min = mean) | 401/450 | **7.76** | **1.54** | **57.4** | 30.7 |
   | | language ceiling (§3, over 408 solvable) | 408 | 7.68 | 1.42 | 61.8 | — |
   | g24r4 bench.solved | greedy | 205/232 | 9.44 | 1.97 | 57.1 | 1.0 |
   | | A\* f=child (arena) | 215/232 | 10.00 | 2.43 | 54.4 | 5.3 |
   | | A\* f=parent / best-at-budget | 215/232 | 9.96 / 9.90 | 2.39 / 2.33 | 54.4 | 4.2 |
   | | **MCTS** (min = mean) | 215/232 | **9.35** | **1.78** | **57.2** | 19.5 |
   | | language ceiling (§3, over 216 solvable) | 216 | 9.31 | 1.72 | 57.4 | — |
   Paired: MCTS vs greedy g16r4 80/0 (p=2e-24; +53 solves), g24r4 24/0
   (p=1e-7; +10 solves); MCTS vs arena A\* on identical solved sets g16r4 59/0
   (p=3e-18), g24r4 27/0 (p=1e-8); size-free MCTS vs per-size MCTS (§5)
   g16r4 42/27 (p=0.09), g24r4 9.35 vs 11.48 (same 215 vs 205 solves).
   → **M2 gate met for both net families at both sizes.**
   Reading. (a) With the size-free nets, greedy descent alone already matches
   the per-size supervised A\* solve count at g24r4 (205) with a 2.2-move
   lower regret on its subset — the labels the loop will generate start
   from a far better place than the descent labeler's. (b) **The ceiling is
   reached**: MCTS's regret sits 0.06–0.12 above the exhaustive
   language optimum (§3) at both sizes and its solve counts are within 1
   (g24r4) / 7 (g16r4) of the solvable set; % optimal equals the ceiling's at
   g24r4 (57.2 vs 57.4). Any further gain a self-play iteration could show on
   THESE exams is ≤ 0.1 moves and ≤ 7 solves — below the seed-noise bars of
   §4.7 by construction. (c) Consequence for M3+: iteration 1 (job 4681263,
   g24r4) still runs to validate the loop end-to-end and to measure
   generation cost and label fidelity, but the promotion evidence must come
   from exams with headroom: the frontier sets (bench.unsolved: 218 g24r4
   instances the supervised planner solves ~90 of), 32×32 and 8 robots
   (zero-shot transfer + ceiling probes queued as job 4681264), the B2
   vocabulary (ceiling 0.90/1.17), or the primitive-move space at 24×24+.
   Sources: `results/m2/sizefree_mixed_warm_{g16r4,g24r4}_*.json`,
   `runs/spr/spr-m2-sf-4680957.out`.

9. **Zero-shot, the one size-free pair (trained on 16×16 + 24×24 with 4 robots)
   matches or beats every per-size supervised backward planner at 32×32 and
   at 8 robots, on graded AND frontier sets, and sits at the base language's
   solve ceiling there too; the frontier sets are the only exams with
   material headroom left for the subgoal loop (2026-08-17, job 4681264
   ≈ 0.2 nh; mixed pair of §7; arena protocol; recorded rows from
   `scaling/results/<cfg>/comparison{,_ungraded}.json`).**
   | exam (n) | supervised per-size pair (recorded) | size-free A\* (zero-shot) | size-free MCTS | base-language ceiling (`spr.ceiling`) |
   |---|---|---|---|---|
   | g24r4 frontier (218, no d\*) | 90/218 (twin pair, base vocab), 19.97 mv | **99/218**, 18.93 mv (McNemar p=0.02) | 99/218, 18.17 mv | — (not probed yet) |
   | g32r4 graded (175) | 147/175, regret 2.37, 51.7% opt, 13.8 exp | **156/175, 1.84, 56.4%, 1.9 exp** (p=0.004; 20/4 mv wins) | 156/175, **1.63, 57.1%**, 13.0 exp | **156/175 = 89.1%**; best-plan regret 1.56, 57.7% |
   | g32r4 frontier (275) | 127/275, 21.28 mv | **141/275**, 20.40 mv (p=1e-4) | — | — |
   | g24r8 graded (161) | 144/161, 2.49, 50.7%, 44.0 exp | 147/161, 2.29, 53.7%, 14.8 exp (p=0.25; 17/8) | 147/161, **1.92, 57.1%** (22/5 mv wins vs recorded, p=0.0015) | 148/161 = 91.9%; regret 1.58, 58.8% |
   | g24r8 frontier (289) | 154/289, 15.68 mv | **171/289**, 15.21 mv (p=9e-4; 46 mv wins) | — | — |
   Reading. (a) **Cross-size AND cross-robot-count transfer** without a
   single training example at 32×32 or with 8 robots: the pair beats the
   32×32-trained supervised pair by 9 solves and 0.5 regret with 7× fewer
   expansions, and matches the 8-robot-trained pair — the "size-free
   self-play" bet of PROBLEM.md §9(2) holds already at the supervised
   stage, in the policy as well as the value (FINDINGS 53/54/70 for the
   labeler). (b) **Saturation**: at g32r4 the zero-shot A\* solve count equals
   the exhaustive language ceiling (156) and MCTS's regret is 0.07 above the
   ceiling's; at g24r8 it is 1 solve short and 0.3 regret above (a little
   room; 8 robots is where PROBLEM.md §4.4 warned about value-net collapse —
   note the size-free value trained on 4-robot corpora only). (c) **Where
   the loop can still show gains**: the frontier sets (+9/+14/+17 solves
   over the recorded rows and 100+ instances still unsolved) — beyond-oracle
   instances have no d\*, so their gate is solve rate + paired McNemar and
   both-solved moves; and the extended vocabulary/primitive-move space for
   the moves-vs-forward question (§3). Iteration 1 of the loop is being
   evaluated on exactly these sets (frontier bench added to the iteration
   job; the same probe re-run on the retrained nets).
   Sources: `results/transfer/*.json`, `results/ceiling/{g32r4,g24r8}_base.json`,
   `runs/spr/spr-transfer-4681264.out`.

10. **M3, iteration 1 (g24r4): the loop runs end to end — 1,774 fresh
    instances searched, 12,244 certified records, fidelity gauge 0.915,
    warm retrain, arena bench — in 1 h 51 min (0.23 nh); on the saturated
    graded exam the retrained pair equals the M1 pair (no gain, no
    regression), as §8 predicted (2026-08-17/18, job 4681263).**
    Generation: 120 fresh lean 24×24 boards (ids 5000–5119), 8 instances per
    board (up to 3× attempts), MCTS best-at-budget (600 expansions, stop 150
    after the last improvement, c=1.5, root Dirichlet 0.25, ALL root
    candidates scored, greedy sibling completion), 8 GPU workers: 1,268 of
    1,774 instances certified (71.5%), mean 49.6 expansions / 6.8 s per
    instance, 12,244 records (1,160 training + 203 validation decision
    groups), 4 transient CUDA OOMs caught per instance (root-all value passes
    with 8 workers on one A100). **Fidelity gauge** (`spr.gauge`, 200 depth-0
    decisions exact-labeled by the Rust engine, `nn_labeler.audit_descent`):
    argmin agreement **0.915**, optimal-set Jaccard 0.86, mean gap 0.99 (0.82
    on exact-optimal candidates), no negative gaps — inside FINDINGS 74's
    "downstream-equivalent" band from the first iteration. Training: policy
    + value warm from the M1 pair, 6 epochs at lr 1e-4 (clip 1.0) on the
    buffer + exact anchors; val regret 1.97 → 2.00 (policy), 1.24 → 1.22
    (value), no collapse. Bench (arena A\*): **216/232, 10.07 mv, regret 2.48,
    54.6% opt** vs M1's 215/232, 10.00, 2.43, 54.4% (paired: +1 solve, moves
    8/7, n.s.); MCTS: 216/232, 9.45, 1.86, 57.4% vs 215, 9.35, 1.78, 57.2%
    (5/3, n.s.). Reading: the loop's data is certified and near-exact
    (0.915), its cost is small (0.23 nh/iteration at 24×24), and — exactly as
    §8 said — the graded g24r4 exam cannot register improvement (216 = the
    language ceiling; regret 0.1 above it). The M3 verdict is therefore
    deferred to the exams with headroom: the iteration-1 nets are being
    probed on the frontier sets and zero-shot at 32×32 / 8 robots (job
    4682171, same protocol as §9), and iteration 2 (job 4682173) generates
    with a hard-instance filter (`--min-expansions 3`) so the data targets
    the frontier-like tail (PROBLEM.md §6.4).
    Sources: `results/selfplay/g24r4_iter1/{generation.manifest.json,
    gauge.json,bench_g24r4_{astar,mcts}.json,gate_vs_supervised.json,
    gate_vs_prev.json,nets.txt}`, `runs/spr/selfplay/g24r4_iter1/`,
    `runs/spr/spr-it1-g24-4681263.out`.

11. **The base subgoal language is saturated EVERYWHERE, frontier sets included:
    the size-free supervised planner sits at (or 1–4 solves from) the
    exhaustive language ceiling on every pinned exam, so no self-play
    iteration in the base vocabulary can register a gain on any of them
    (2026-08-18, jobs 4682172 [frontier ceilings], 4682184 [iteration-1 nets
    on the frontier]).**
    | exam | base-language solve ceiling (`spr.ceiling`) | size-free A\* (M1 nets) | iteration-1 nets |
    |---|---|---|---|
    | g24r4 frontier (218) | **103** (99 no complete plan, 16 unplayable) | 99 | 100 (paired +1, n.s.) |
    | g32r4 frontier (275) | **142** (122 / 11) | 141 | (probing) |
    | g24r4 graded (232) | 216 | 215 → MCTS 215; iter-1 216 | 216 |
    | g32r4 graded (175) | 156 | 156 | — |
    | g24r8 graded (161) | 148 | 147 | — |
    Reading. The recorded B2-vocabulary arm on the g24r4 frontier solves
    125/218 (`comparison_ungraded_b2.json`) — 22 more than the base language
    can EVER solve; the extended language (transient/park stoppers,
    supports-by-reference) is where the remaining solve-rate headroom is,
    and its moves ceiling is 0.5 move lower on the graded exams (§3). The
    supervised track showed the B2 supply exists but the nets never learned
    to rank it (FINDINGS 40/78: "supply is not the bottleneck, the training
    signal is") — which is precisely what a self-play loop can manufacture at
    any size. Decision (pending owner steer, proceeding on it): the subgoal
    loop switches to the B2 vocabulary (search over `propose_b1` + by-reference
    candidates, park repairs at certification, records with by-reference
    helpers; nets warm from the M1 pair, ranking B2 candidates zero-shot at
    first — exactly FINDINGS 78's starting point); the base-vocabulary loop
    is complete as a negative result (nothing to learn) and its iteration 2
    (hard-instance filter, job 4682185) runs only to measure that filter's
    data yield. The primitive-move arm's next target is 24×24, where the
    forward planner's search cost (191 expansions / 275 s per puzzle) is the
    headroom.
    Sources: `results/ceiling/{g24r4,g32r4}_frontier_base.json`,
    `results/selfplay/g24r4_iter1/transfer/`, `results/transfer/`.

12. **Base-vocabulary loop, iteration 2 (hard-instance filter) is flat, as the
    ceilings dictate; the extended B2 vocabulary opens the headroom the loop
    needs — the M1 nets already lift the g24r4 graded row to 223/232 (A\*) and
    228/232 = the B2 solve ceiling (MCTS), and the frontier to 133/218
    (2026-08-18, jobs 4682185, 4682184, 4682786, 4682172, 4685187 ≈ 1.2 nh).**
    (a) Iteration 2 (`--min-expansions 3`: only instances whose first
    certified plan needed ≥3 expansions are kept — 1,525 of 2,869 dropped as
    trivial): 4,268 records; **fidelity gauge 0.85** (vs 0.915 unfiltered) —
    harder instances get looser certified labels, right where FINDINGS 74
    puts the danger band; retrained pair: A\* 215/232, 9.87 mv, MCTS 215/232,
    9.25 (1.68) vs iteration 1's 216/10.07 and 216/9.45 (paired 8/7 and n.s.),
    frontier 100/218 = unchanged. Iteration-1 nets on the transfer exams: equal
    to the M1 nets within ±1 solve everywhere (g32r4 156=156, g24r8 146 vs
    147, g32r4 frontier 140 vs 141). Verdict: two base-vocabulary iterations
    neither help nor hurt any pinned exam — a clean negative result explained
    by §8/§11 (the base language is saturated), with the loop machinery,
    certification, gauge and gate all exercised.
    (b) **B2 vocabulary with the M1 (base-trained) nets, zero-shot ranking**
    (`spr.bench --vocab b2 --anytime`: propose_b1 + by-reference candidates,
    park repairs at failed certification, no prefix-check — the arena's B2
    convention; `eval.realize.prefix_key` cannot order park/by-ref plans):
    | exam | search | solved | mean moves | regret | % opt | mean exp |
    |---|---|---|---|---|---|---|
    | g24r4 graded (232) | supervised B2 arm (recorded, per-size B1 nets) | 199/232 | 12.33 | 4.89 | 36.7 | 53.5 |
    | | size-free M1 pair, A\* anytime | **223/232** | 9.51 | 1.95 | 55.6 | 52.8 |
    | | size-free M1 pair, MCTS best-at-budget | **228/232 = B2 ceiling** | 9.19 | 1.55 | 58.8 | 490 |
    | | B2 language ceiling (§3) | 228 | 8.81 | 1.17 | 62.7 | — |
    | g24r4 frontier (218) | supervised B2 arm (recorded) | 125/218 | 22.54 | — | — | 110 |
    | | size-free M1 pair, A\* anytime | **133/218** | 19.43 | — | — | 515 |
    | | base-language ceiling | 103 | | | | |
    | | B2 probe (8 workers, 100k frontier, 120 s): 114 proven, 104 inconclusive → ceiling ≥ 133 | | | | | |
    | g32r4 graded (175) | B2 language ceiling | 166/175 (base 156); regret 1.28 (base 1.56) | | | | |
    Reading. The B2 language adds +12 graded solves and ≥ +30 frontier solves
    over the base ceiling at g24r4 (+34 over what the M1 nets reach in base), and the base-trained size-free nets
    already exploit it zero-shot (they had never seen a transient-support or
    by-reference candidate: FINDINGS 78's "supply is not the bottleneck"),
    but at a large search cost (490–515 expansions vs 5–20 in base) and 0.4
    regret above the B2 ceiling. **That is the loop's job now**: learn to
    rank the extended vocabulary so the same solve rate comes at base-like
    expansions and the regret gap closes — B2 iteration 1 (job 4685442:
    self-play generation in B2, records with by-reference helpers, policy
    trained with `--byref`, no base anchors) is queued behind the iteration-0
    reference bench (4682786).
    Sources: `results/selfplay/g24r4_iter2/`, `results/selfplay/g24r4_iter1/transfer/`,
    `results/selfplay/g24r4_b2_iter0/`, `results/ceiling/{g24r4_frontier_b2,g32r4_graded_b2}.json`.

13. **B2 loop, iteration 1 (g24r4): the first self-play data in the extended
    vocabulary moves the planner in the right direction on the graded exam —
    +3 solves (A\*), −19% expansions, moves unchanged — not yet beyond noise
    after one small iteration; the exact B2 gauge is too expensive as-is and
    was bounded (2026-08-18, job 4685442, ~0.6 nh incl. the wasted gauge
    hours).** Generation in B2 (`spr.selfplay --vocab b2`, 60 fresh boards,
    ids 8000–8059, MCTS 300 expansions / stop 80, 6 workers): 681 instances,
    **598 certified (87.8% vs 71.5% in base)**, 5,822 records of which 71 are
    by-reference (helper standing on a planned cell), 108 expansions / 37 s
    per instance (park repairs and by-reference candidates make B2 search
    ~5× dearer than base). Gauge: the Rust engine's exact B2 rollouts on 200
    depth-0 decisions with all candidates did not finish in 2.5 h (25 GB
    RSS) and were killed; `spr.gauge` now takes `--max-candidates` and a
    `--time-cap` (queued as job 4689629 with 80 instances / 24 candidates /
    40 min). Training: policy + value warm from the M1 pair, 6 epochs on the
    5,822 B2 records only (no base anchors: one vocabulary per dataset,
    PROBLEM.md §10), policy `--byref`; val regret 1.74 (B2 val, 109 groups),
    value spread 2.5 (no collapse). Bench, B2 arena convention (anytime,
    park repairs, by-reference, no prefix-check), 1200 expansions:
    | g24r4 graded (232) | solved | mean moves | regret | % opt | mean exp |
    |---|---|---|---|---|---|
    | iteration 0 = M1 nets, A\* | 223 | 9.51 | 1.95 | 55.6 | 52.8 |
    | **iteration 1, A\*** | **226** | 9.61 | 1.99 | 55.8 | **43.0** |
    | iteration 0, MCTS | 228 | 9.19 | 1.55 | 58.8 | 490 |
    | **iteration 1, MCTS** | **229** | 9.22 | 1.56 | 57.6 | 450 |
    Paired: A\* +4/−1 solves (McNemar p=0.38), moves 12/10; MCTS +2/−1,
    11/10 — direction consistent (more solves, fewer expansions), size below
    the seed-noise bar; 229 exceeds the capped B2 probe's 228, so the true
    B2 ceiling here is ≥229. Frontier benches did not fit the walltime
    (resumed as job 4689626); iterations 2 and 3 are chained behind it
    (4689627/4689628, `auto` net hand-off) so tonight's window yields three
    B2 iterations for the M3/M4 trend on the graded AND frontier exams.
    Sources: `results/selfplay/g24r4_b2_iter1/`, `runs/spr/selfplay/g24r4_b2_iter1/`,
    `runs/spr/spr-b2-it1-4685442.out`.

14. **Three parallel review/audit passes (2026-08-18, agents; memos in
    `results/audit_claims_2026-08-18.md`, `results/review_b2_pipeline_2026-08-18.md`,
    `results/audit_findings_vs_files_2026-08-18.md`).** (a) FINDINGS-vs-files:
    ~260 quoted numbers re-derived; 10 mismatches, none changing a conclusion
    (three §7 paired counts were computed against the same-machine M0 re-run
    rather than the recorded file — both now stated; §12's "+34" was over the
    M1 base nets, +30 over the ceiling — corrected). Untraceable-by-design
    items (node-hours, wall times, training metrics) live under `runs/`.
    (b) B2 pipeline review: no invalidating defect in the by-reference /
    park-repair search, record and training path (arena-parity confirmed:
    slot resolution, `_apply(by_reference)`, park children semantics, label
    units); latent low-severity items listed in the memo. Two substantive
    findings on the **B2 fidelity gauge**: (i) the exact engine was called
    without a `budget.solver_iters` (its default is 10 M iterations per
    rollout — hence 2.5 h / 25 GB); the supervised track always sent
    5k–20k; measured on the 198 finished rollouts, a 100k budget keeps 197/198
    and finishes in minutes → `spr.gauge --solver-iters 100000` (default now;
    job 4691066 re-runs the B2 iteration-1 gauge with all candidates).
    (ii) The 198 finished exact B2 rollouts give **argmin agreement 0.51**
    (Jaccard 0.41, mean gap +3.9) against the loop's certified B2 labels —
    but only 13/28 sampled exact-optimal B2 completions strictly REALIZE:
    exact B2 labels price unplayable plans (and exclude park cost), so the
    FINDINGS-74 bands do not transfer to B2 and the exact side is the
    weaker reference there. Consequence: for the B2 loop, drift is monitored
    through the graded/frontier benches (certified, the metric itself); the
    gauge is reported on the subset where the exact optimum realizes once
    that filter exists. (c) Anchoring: keep the B2 loop anchor-free (base
    anchors would change candidate-set semantics; the existing exact B2
    corpora — g16r4/g16r6/g16r8/g24r8/g32r4 cap20000, none at g24r4 — are
    exact-abstract with ~half unplayable optima); a certified B2 seed via
    `nn_labeler.descent --vocab b2` is the alternative if drift appears.
    (d) The claims audit (like-for-like protocol, leakage, statistics,
    ceiling soundness) is appended below when it lands.
    (d) **Claims audit (adversarial, independent re-derivation): no blockers;
    every quoted statistic reproduces; four caveats now attached to the
    headlines.** (i) `row["search"]["root_closed"]` is true on 100% of graded
    instances in every MCTS payload: with k=5 the tree closes long before
    1200 expansions, so "MCTS best-at-budget" is an exhaustive certified
    enumeration of the top-5 tree (≈5 strict realizations per instance vs 1
    for A\*; 3–5× wall time) — the 1200 cap is not binding, min = mean backup
    is then tautological, and the value net only orders visits. The strongest
    A\* control (arena f + best-at-budget) had not been run; queued (job
    4691082) at both sizes for the size-free pair. (ii) A stronger per-size
    baseline exists at g24r4 than the exact pair: the base-vocab twin pair
    `scaling/results/g24r4/comparison_nntwin.json` scores 212/232, regret
    3.05 — vs it the size-free A\* gain is +3 solves (n.s.) while the moves
    gain stays significant (39/13, p=4e-4; MCTS 50/1); all size-free nets are
    single-seed (21). (iii) The moves ceiling (`spr.ceiling`) assumes strict ≥
    abstract, which certified plans violate on 4/215 g24r4 and 3/401 g16r4
    instances (by up to 8–11 moves); a full-enumeration re-probe lowers the
    ceiling regret by ≈0.04–0.06 — the size of the reported gap to it — so
    the honest wording is "within ~0.1–0.2 moves of the exhaustive optimum of
    the top-5 tree"; base probes are being re-run with `--slack 12`
    (`results/ceiling/*_base_slack12.json`). (iv) The recorded g32r4/g24r8/
    frontier reference rows carry no move dumps (only the new rows are
    replay-certified), and "beats every per-size supervised backward planner"
    holds within base vocabulary + prefix-check (recorded per-size B2 arms
    reach 154–171/175 at g32r4). Confirmed clean: same instances_sha256,
    1200/k=5 and prefix-check on both sides of every comparison; 46/46 new
    payloads replay-certified (6 re-validated: 0 failures); zero bench
    boards/instances in any training corpus; labeler encoder init trained on
    g8–g16 only; the g16r4 base ceiling reproduces supervised FINDINGS §4
    (22 no-plan + 20 unplayable) and no planner ever solves a probe-
    "unrealizable" instance in 21–27 payloads per exam.
    Slack-12 re-probe of the base ceilings (login node, minutes): g16r4 best
    7.65 / regret 1.39 / 62.5% (was 7.68 / 1.42 / 61.8%), g24r4 9.22 / 1.63 /
    57.9% (was 9.31 / 1.72 / 57.4%); no probe capped. So the size-free MCTS
    rows (§8: 1.54 and 1.78) sit 0.15 above the tightened ceilings — the
    "within ~0.1–0.2 moves" wording of (iii) is the one to quote.
    Sources: `results/ceiling/{g16r4,g24r4}_base_slack12.json`.
    The missing A\* control (arena f=child + best-at-budget, job 4691253) is
    IDENTICAL to first-plan A\* at both sizes (g24r4 215/232, 10.00 mv, 5.3
    exp; g16r4 401/450, 8.26, 8.9): with the arena's over-estimating f the
    best-at-budget bound (frontier f ≥ certified abstract cost) triggers at
    the first certified plan, so no A\* variant explores the tree further; the
    MCTS advantage (27/0 and 59/0 paired wins) is exactly the certified
    enumeration of the top-5 tree that A\*'s bound cannot reach.
    Sources: `results/m2/sizefree_mixed_warm_{g24r4,g16r4}_astar_child_best.json`.

15. **M3 PASSES and M4 is on track in the B2 loop: three self-play iterations
    at 24×24 climb on the exam with headroom — frontier solves 133 → 144 → 139
    → 158 (iteration 3 vs 0: +28/−3, McNemar p=5e-6) at 22% fewer expansions,
    hold the graded exam at its B2 solve ceiling with slightly better regret,
    cut A\* search cost by a third, and the final nets beat the recorded
    supervised B2 planner beyond any seed bar (2026-08-18/19, jobs 4685442,
    4691251, 4689627, 4689628 ≈ 2.1 nh incl. reruns).** Protocol: iteration
    0 = the M1 size-free pair benched under B2 (§12); each iteration = 60
    fresh lean boards (ids 8000+), MCTS generation (300 expansions, stop 80,
    root noise 0.25, all root candidates, sibling completion), ~5.7–6.3k
    certified B2 records, warm retrain (6 epochs, lr 1e-4, clip, `--byref`,
    no anchors), B2 arena bench (anytime, park repairs, by-reference, no
    prefix-check, 1200 expansions), gates paired vs iteration k−1 and 0.
    | iter | certified / instances → records | A\* graded (232) | MCTS graded | A\* frontier (218) | MCTS frontier |
    |---|---|---|---|---|---|
    | 0 (M1 pair) | — | 223, 9.51 mv, regret 1.95, 53 exp | 228, 9.19, 1.55, 490 exp | 133, 19.43 mv, 515 exp | (bench timed out) |
    | 1 | 598/681 → 5,822 | 226, 9.61, 1.99, 43 | 229, 9.22, 1.56, 450 | 144, 19.34, 467 | 169, 19.71, 1019 |
    | 2 | 588/655 → 5,694 | 227, 9.83, 2.21, 35 | 229, 9.23, 1.58, 468 | 139, 18.80, 476 | 172, 19.72, 1012 |
    | 3 | 582/660 → 6,301 | 225, 9.48, 1.87, 36 | 228, 9.14, **1.50**, 520 | **158**, 19.99, 402 | 171, 19.44, 992 |
    Paired tests (spr.gate). Frontier A\* iteration 3 vs 0: +28/−3 solves,
    p=5e-6 (moves on shared solves 27/24, n.s.); vs iteration 1: n.s. (the
    frontier flips ~15 instances between neighbouring iterations, so
    single-step gates are underpowered — the M4 "monotone-ish" reading is
    133→144→139→158). Graded A\* iteration 3 vs 0: 225 vs 223 (+4/−2, n.s.),
    moves 14/12 n.s., expansions −32%; graded MCTS regret 1.55→1.50 (best of
    the series; 0.33 above the B2 ceiling 1.17). Frontier MCTS 169→172→171
    (all within noise; the 1200-expansion cap binds there, ~1000 mean).
    **Versus the frozen supervised baselines** (recorded per-size B2 arm,
    `scaling/results/g24r4/comparison{,_ungraded}_b2.json`): graded 225 vs
    199 (McNemar p=2e-7; 76/9 move wins on shared solves, mean 9.23 vs
    12.30), frontier 158 vs 125 (p=4e-5; 61/13 move wins, 18.65 vs 21.94) —
    and versus the base-vocabulary supervised pair (205 graded / 90–99
    frontier) the gap is larger still. Fidelity gauge vs the exact-B2
    reference stays 0.50–0.57 (informational only, §14b: half of the exact
    B2 optima are unplayable). Reading: (a) the loop climbs where the language
    leaves room and holds where it does not; the graded exam is pinned at the
    B2 solve ceiling (228–229 of ≥229) so its signal is regret and cost. (b)
    What the loop learns is to rank the extended vocabulary: A\* reaches the
    same solve set with 53→36 expansions and MCTS's frontier solves rise
    from 133 (base-trained nets zero-shot in B2) to 158–172. (c) M4's
    literal gate — "final net beats the supervised backward baseline beyond
    seed noise" — is met against both recorded baselines on both exams; the
    "monotone-ish" part holds on the frontier trend with one dip (iteration
    2) inside the per-step noise. Cost: 2.1–2.5 h per iteration on one GPU
    (generation ~70 min at 36 s/instance; benches dominate — the frontier
    MCTS bench alone is ~1 h), i.e. ≈0.3 nh per iteration.
    Sources: `results/selfplay/g24r4_b2_iter{0,1,2,3}/`, `spr.trend --config
    g24r4 --tag _b2`, `runs/spr/spr-b2-it{1,1r,2,3}-*.out`.

16. **The B2 loop's gains transfer zero-shot to 32×32 and 8 robots (M5
    signal) and iteration 4 keeps the trend (2026-08-19, jobs 4698737,
    4698743 [transfer], 4698741 [iteration 4] ≈ 0.9 nh).** Same protocol as
    §15; "iteration 0" = the M1 pair, "iteration 3" = its nets after three
    B2 self-play iterations at 24×24 only; benches under the B2 convention.
    | exam | recorded per-size supervised B2 arm | iteration-0 nets | iteration-3 nets | paired it3 vs it0 |
    |---|---|---|---|---|
    | g32r4 graded (175), A\* | 154/175, regret 2.84, 34.5 exp | 170/175, 1.90, 35 exp | 170/175, 2.36, 48 exp | solves =, moves 6/15 (p=0.08) |
    | g32r4 graded, MCTS | — | 171/175, 1.75, 297 exp | 172/175, 2.10, 304 exp | 5/12 (n.s.) |
    | g32r4 frontier (275), A\* | 196/275, 24.6 mv | 194/275, 20.6 mv, 397 exp | **208/275**, 21.9 mv, 355 exp | **+23/−9, p=0.02**; moves 37/36 |
    | g24r8 graded (161), A\* | 148/161, regret 2.34, 95 exp | 157/161, 2.21, 26 exp | **159/161 (= B2 ceiling probe)**, 2.68, **2.2 exp** | +2/−0; moves 12/18 (n.s.) |
    | g24r8 graded, MCTS | — | (row) | (row) | — |
    Reading. (a) The nets that self-played only at 24×24 solve more of the
    32×32 frontier (+14, p=0.02) and reach the 8-robot B2 ceiling with an
    order of magnitude fewer expansions (26 → 2.2) — the loop's learned
    ranking of the extended vocabulary carries across size and robot count,
    as the size-free bet (§7/§9) predicted. (b) The price is a small,
    consistent regret increase on the graded 32×32/8-robot exams (n.s.
    individually: 6/15, 5/12, 12/18) — the loop specializes to what it
    trains on (24×24 boards, 4 robots); a mixed-size self-play curriculum
    (M5's plan) is the natural fix and the M1 mixed training already showed
    that mixing costs nothing per size. (c) Iteration 4 at 24×24: A\* graded
    **228/232** (best of the series; 26 expansions vs 53 at iteration 0),
    MCTS graded 228, regret **1.46** (best), frontier A\* 153/218 (vs 133:
    +25/−5, p=3e-4; the trend 133→144→139→158→153 is a plateau of ≈+20
    solves after iteration 3). Fidelity gauge vs the exact-B2 reference
    keeps drifting down (0.51→0.50→0.57→0.45), consistent with §14b (the
    reference is the weaker labeler in B2), while every certified metric
    holds or improves — the certified benches, not the gauge, are the
    instrument in the B2 loop.
    Sources: `results/selfplay/g24r4_b2_iter{0,3}/transfer/`,
    `results/selfplay/g24r4_b2_iter4/`, `spr.trend --config g24r4 --tag _b2`.

17. **M5 far-size clause: the size-free pair, trained at 16/24 only, audits
    zero-shot against exact ground truth at 32–64 at the labeler's level or
    above; the B2 loop's nets pay 1–2 points on these base-vocabulary decisions
    (specialization, cf. §16b); and the iteration-0 frontier-MCTS reference
    shows the loop's frontier gains are A\*'s, not MCTS's (2026-08-19, jobs
    4704311/4704312 [audits, 0.2 nh], 4704317 [iteration-0 frontier MCTS,
    0.1 nh], 4707188 [labeler reference 56/64, 0.05 nh]).** Instrument
    `spr.audit` (new): on the exact-labeled decision corpora (g32r4 test split
    of `scaling/data/g32r4/backward.jsonl`; 40/48/56/64 =
    `backward_audit.rust.jsonl`, the labeler's own fidelity-curve sets,
    FINDINGS 53/70; base vocabulary, all depths, groups with ≥2 candidates)
    score the value net's argmin (= `nn_labeler.audit`), the policy's top-k,
    and the planner's greedy decision (value argmin over the policy's top-5 =
    what one arena expansion decides). "it0" = the M1 mixed pair (§7), "it4" =
    the B2 loop's iteration-4 nets (§16); labeler = prod_v1_s11 value
    (`nn_labeler/results/audit_prod_v1_s11_full.json`, `audit_coarse_g{40,48}r4.json`,
    and for 56/64 `results/audit/labeler_prod_v1_s11_g56g64.json`, run now with the
    same instrument; the labeler's group counts are a few % higher because
    `spr.audit` drops 1-candidate groups).
    | exam (exact groups) | labeler value top1 / regret | it0: value · policy top1 / r@1 / recall@5 · pair top1 / regret | it4: value · policy · pair |
    |---|---|---|---|
    | g32r4 (1292) | 0.853 / 0.56 | **0.863** / 0.48 · 0.687 / 1.33 / 0.985 · 0.852 / 0.55 | 0.837 / 0.64 · 0.671 / 1.33 / 0.974 · 0.828 / 0.66 |
    | g40r4 (2100) | 0.852 / 0.59 | **0.863** / 0.52 · 0.691 / 1.36 / 0.982 · 0.851 / 0.55 | 0.847 / 0.58 · 0.665 / 1.57 / 0.975 · 0.830 / 0.62 |
    | g48r4 (2082) | 0.842 / 0.63 | **0.854** / 0.53 · 0.677 / 1.33 / 0.988 · 0.848 / 0.56 | 0.846 / 0.62 · 0.643 / 1.57 / 0.980 · 0.836 / 0.68 |
    | g56r4 (2084) | 0.829 / 0.71 | 0.842 / 0.62 · 0.658 / 1.49 / 0.982 · 0.831 / 0.65 | 0.839 / 0.64 · 0.620 / 1.71 / 0.971 · 0.820 / 0.73 |
    | g64r4 (2049) | 0.814 / 0.67 | 0.838 / 0.57 · 0.644 / 1.60 / 0.980 · 0.829 / 0.59 | 0.837 / 0.65 · 0.616 / 1.81 / 0.973 · 0.822 / 0.71 |
    Reading. (a) The M1 value net beats the labeler that initialized it at
    every size — +1.0/+1.1/+1.2 points at 32/40/48 and +1.3/+2.4 at 56/64
    (its extra 24×24 exact data helps 2–3× beyond its training size; even
    the B2-loop nets are above the labeler at 56/64) — and degrades only
    0.863→0.838 from 32 to 64 against the labeler's 0.853→0.814: the
    size-free bet (§6.2/§9) holds far past the curriculum. (b) The policy's
    top-1 is a weak oracle (0.64–0.69) but its top-5 contains an optimum in
    97–99% of decisions, so the arena's k=5 filter costs ≈0.05 regret — the
    value net does the choosing (pair ≈ value − 1 pt at every size; the pair
    loses to the value alone in 20–35 decisions per set and beats it in
    4–12). (c) The B2 loop's nets are 1–3 points worse than their own
    initialization on these *base-vocabulary* decisions at every size (value
    0.837 vs 0.863 at 32; policy top-1 −2 to −3 pts, r@1 1.33→1.57–1.81),
    the same specialization cost §16b measured on the graded 32/8-robot
    exams: four iterations of 24×24 B2-only self-play (no exact anchors, by
    design of §11) drift the nets toward the B2 ranking — the motive for the
    mixed-size curriculum now running (job chain 4704314–16). Depth split
    (it0): depth-0 decisions are the hard ones (value top1 0.82–0.84; depth
    ≥1 0.85–0.99), as in the labeler's own audits. (d) **Frontier MCTS
    reference.** The iteration-0 pair with MCTS best-at-budget on the g24r4
    frontier (the row §15 lacked) solves **171/218** (19.50 mv, 1038 exp) —
    so the frontier MCTS series is 171 → 169 → 172 → 171 → 178 (iteration 4
    vs 0: +12/−5, McNemar p=0.14; moves 23/31 n.s.): flat within noise. The
    loop's frontier gain is A\*'s (133→158, p=5e-6): what four iterations
    bought is a ranking under which the cheap first-solution search reaches
    what the 1200-expansion MCTS could already reach with the iteration-0
    nets (graded: A\* 223→228 = MCTS's 228, at 26 vs 500 expansions) — search
    distilled into the nets, the AlphaZero mechanism, rather than a higher
    language reach. §15's "monotone-ish" reading stands for A\*; for MCTS the
    honest statement is "held at the language's reach with slightly better
    regret (1.55→1.46)". Sources: `results/audit/b2it0_m1mixed.json`,
    `results/audit/b2it4.json`,
    `results/selfplay/g24r4_b2_iter0/m1mixed_b2_g24r4_bench_unsolved_mcts.json`,
    `spr.gate compare` pairs, `runs/spr/spr-audit-it{0,4}-470431{1,2}.out`.

18. **Control (asked for by the 2026-08-18 audits, §14): a second seed of the
    M1 mixed size-free pair replicates §7 to the instance — the "size-free beats
    per-size at 24×24" claim is not a seed artefact (2026-08-19, jobs 4704318
    [train 3.6 h] + 4704319 [bench], ≈0.5 nh).** Same recipe as §7 (policy 25 ep
    lr 1e-4 clip 1.0 labeler-encoder init; value warm from the labeler 12 ep),
    torch seed 37 instead of 21; arena protocol (1200/k5, prefix-check,
    replay-certified).
    | exam | seed 21 (§7) | seed 37 | paired 37 vs 21 | per-size supervised pair |
    |---|---|---|---|---|
    | g16r4 bench450 | 401/450, 8.26 mv, regret 2.04, 52.1% opt, 8.9 exp | 402/450, 8.29, 2.06, 52.0%, 9.4 exp | — | 400/450 (v2 pair; gate +0.2 solve / +1.6 opt pts) |
    | g24r4 graded (232) | 215/232, 10.01 mv, 2.43, 54.4%, 5.3 exp | 215/232, 9.88, **2.31**, 55.3%, 5.0 exp | same solve set (McNemar p=1), moves 10/8 (n.s.) | 205/232, 4.20, 38.5% (gate +4.3 solve / +16.8 opt pts) |
    Reading: two independently seeded size-free pairs land within 0.1 move
    and 1 optimality point of each other at both sizes, and both clear the
    per-size supervised g24r4 pair by 10 solves and ≈1.8 regret — §4.7's seed
    bars (3.5 / 6.6 pts) are generous relative to this family's actual seed
    noise. The seed-37 pair is banked as an alternative iteration-0 for
    future loop-seed controls. Sources: `results/m1/mixed_value_warm_s37_*.json`
    (+ `.gate.json`), `runs/spr/spr-m1-{mixed,bench}-s37-470431{8,9}.out`.

19. **M4 verdict after five B2 self-play iterations at 24×24: the gate's two
    clauses are met — the final nets beat the frozen supervised backward
    baseline beyond seed noise on both exams, and the bench trend is
    monotone-ish on the frontier A\* metric — but the loop saturates after
    iteration 3, and what it learned is search efficiency (the cheap
    first-solution planner catches the 1200-expansion MCTS), not lower
    realized moves (2026-08-19, iteration 5 = job 4706184 ≈ 0.6 nh; the
    series = jobs 4685442, 4691251, 4689627, 4689628, 4698741, 4706184).**
    Full trend (`spr.trend --config g24r4 --tag _b2`; iteration 0 = the M1
    pair under B2 flags; frontier MCTS at iteration 0 from §17d):
    | iter | certified / instances → records | gauge | A\* graded (232) | MCTS graded | A\* frontier (218) | MCTS frontier |
    |---|---|---|---|---|---|---|
    | 0 | — | — | 223, 9.51 mv, regret 1.95, 53 exp | 228, 9.19, 1.55, 490 | 133, 19.43 mv, 515 exp | 171, 19.50, 1038 |
    | 1 | 598/681 → 5,822 | 0.51 | 226, 9.61, 1.99, 43 | 229, 9.22, 1.56, 450 | 144, 19.34, 467 | 169, 19.71, 1019 |
    | 2 | 588/655 → 5,694 | 0.50 | 227, 9.83, 2.21, 35 | 229, 9.23, 1.58, 468 | 139, 18.80, 476 | 172, 19.72, 1012 |
    | 3 | 582/660 → 6,301 | 0.57 | 225, 9.48, 1.87, 36 | 228, 9.14, **1.50**, 520 | **158**, 19.99, 402 | 171, 19.44, 992 |
    | 4 | 600/675 → 6,069 | 0.45 | **228**, 9.72, 2.08, **26** | 228, 9.10, **1.46**, 500 | 153, 19.84, 405 | **178**, 20.34, 975 |
    | 5 | 609/676 → 6,661 | 0.46 | 227, 9.85, 2.23, 30 | 228, 9.18, 1.55, 543 | 153, 19.79, 398 | 174, 19.79, 1017 |
    Paired tests, iteration 5 vs 0 (spr.gate): frontier A\* +27/−7 solves,
    McNemar **p=8e-4** (moves on shared solves 22/27, n.s.); graded A\* 227 vs
    223 (+6/−2, p=0.29; moves 16/17); graded MCTS 228 = 228 (moves 13/18,
    n.s.); frontier MCTS 174 vs 171 (p=0.66). Iteration 5 vs 4: nothing moves
    (graded A\* −1, frontier =, p=1). Versus the frozen supervised B2 planner
    (§15): graded 227 vs 199, frontier 153 vs 125 — both p<1e-4, as at
    iteration 3. M1 gate vs the base-vocab per-size pair: +9.5 solve / +15.6
    optimality pts (job log).
    Verdict. (a) **Gate clauses**: "final net beats the supervised backward
    baseline beyond seed noise" — yes, by a wide margin (and the seed bar is
    now known to be generous, §18); "monotone-ish bench improvement" — yes for
    frontier A\* (133→144→139→158→153→153, two plateaus), flat for everything
    else. **M4 PASS, with the caveat that the climb is three iterations long.**
    (b) **What the loop learned** is a ranking under which the first-solution
    A\* reaches, in 26–30 expansions, what the iteration-0 nets needed
    500-expansion MCTS for (graded: A\* 223→227–228 = MCTS 228; frontier: A\*
    133→153–158 toward MCTS's 171–178). Search was distilled into the nets —
    the AlphaZero mechanism — but the 1200-expansion reach of the B2
    language on this frontier (≈171–178 solves, the MCTS row) did not move,
    and realized moves on shared solves never improved beyond noise in any
    pairing (the A\* regret even drifts 1.95→2.23 as the planner solves harder
    instances first-shot; MCTS regret 1.55→1.46→1.55). On the project's
    headline metric, moves-to-terminal, the 24×24 B2 loop is flat. (c) The
    fidelity gauge (0.51→0.46) stays uninformative in B2 (§14b). (d) Why it
    saturates: the graded exam sits at its B2 solve ceiling (228 of ≥229,
    §3), the frontier's remaining ≈40 instances are beyond the language's
    1200-expansion reach, and a 60-board iteration adds ≈6k records on the
    same 24×24 distribution — consistent with the 24-only nets' far-size
    audit drifting down (§17c). The lever that moved the frontier again is
    the **mixed-size curriculum** (first iteration, §20): g24r4 frontier 170
    and g32r4 frontier 235 — which is where the loop continues. Cost of the
    5-iteration series: ≈2.9 nh incl. reruns (0.5–0.75 nh per iteration; the
    frontier MCTS bench is 40% of it).
    Sources: `results/selfplay/g24r4_b2_iter{0..5}/`, `spr.trend`, `spr.gate
    compare`, `runs/spr/spr-b2-it5-4706184.out`.

20. **M5, mixed-size curriculum iteration 1: one iteration of B2 self-play ON
    the target distributions (24×24 + 32×32 + 8 robots, one size-free pair)
    beats everything that came before it on every frontier — g24r4 170/218
    (best 24-only iteration: 158), g32r4 235/275 (24-only loop's transfer:
    208), g24r8 269/289 (recorded supervised B2 arm: 161; base-vocab
    size-free row: 171) — at 20–45% fewer expansions, while holding the
    graded exams at their ceilings (2026-08-19, job 4704314 = 5.2 h ≈ 0.65 nh;
    iterations 2–3 chained: 4704315/4704316).** Setup (`jobs/selfplay_mix_iter.slurm`,
    DESIGN §4c): seed nets = the 24-only B2 loop's iteration-4 pair; per config
    30 fresh lean boards (ids 9000+), B2 MCTS generation as in §15; per-config
    window buffers; ONE policy+value pair warm-retrained on the union (11,915
    records: 3.3k/2.9k/5.8k; `--batch-ref-n 24` shrinks batches above 24×24);
    benches = B2 arena A\* (graded + frontier, all three configs).
    | exam | best prior planner row | mix iteration 1 | paired |
    |---|---|---|---|
    | g24r4 graded (232) | 228/232, regret 2.08, 26 exp (it4, §16) | 226/232, 2.04, 33 exp | −2 solves (n.s.) |
    | g24r4 frontier (218) | **158** (it3) / 153 (it4/5 seed) | **170**, 20.06 mv, 288 exp | vs it4: +30/−13, **p=0.002** |
    | g32r4 graded (175) | 170/175, 2.36 (it3 transfer, §16) | **173/175**, 2.61, 26 exp | +3 |
    | g32r4 frontier (275) | 208 (it3 transfer) / 194 (it0) | **235**, 22.80 mv, 226 exp | vs it3: +38/−11, **p=2e-5**; moves 38/36 |
    | g24r8 graded (161) | 159/161, 2.68, 2.2 exp (it3 transfer) | 159/161, 2.35, 2.5 exp | regret −0.33 |
    | g24r8 frontier (289) | 171 (base-vocab size-free §9); recorded B2 arm 161 | **269**, 17.47 mv, 132 exp | vs recorded: +108/−0, **p≈0**; moves 53/38 (n.s.) |
    Reading. (a) The 24-only loop's frontier plateau (§19d) was a data-
    distribution limit, not a language or capacity limit: 30 boards per
    target size in ONE iteration bought +12/+27/+98 frontier solves over the
    best 24-only rows. (b) The g24r8 frontier jump (171→269 of 289, 93%
    solved) says the B2 vocabulary's park/by-reference machinery is most
    valuable in the crowded 8-robot mode — where the supervised B2 arm,
    trained on exact labels that ignore playability, got 161 — and that
    nobody had ever trained a B2 ranking on 8-robot boards before this job.
    (c) Specialization cost is gone: the same single pair now holds all six
    exams (cf. §17c's drift warning — the far-size audit of these nets after
    iteration 3 will check the base-vocab cost). (d) Regret on the graded
    32×32 exam drifts up (1.90 it0 → 2.61) as solves rise — same moves-vs-
    solves trade as §19b; MCTS rows for the final nets (bench_pair) will say
    whether the moves gap closes with search. Iterations 2–3 are chained;
    gates vs this iteration will use `gate_<cfg>_<t>_vs_prev.json`.
    Sources: `results/selfplay/mix_b2mix_iter1/` (benches, generation
    manifests, nets.txt), `runs/spr/spr-mix-it1-4704314.out`, `spr.gate
    compare` pairs above.

21. **F-M3/F-M4, forward (primitive-move) loop at 24×24, two iterations: the
    self-play MoveNet closes the supervised planner's moves gap where MCTS
    had one (iteration 2 MCTS: 224/232, regret 0.36 vs iteration 1's 0.46;
    A\* regret 0.096→0.041, 96.3% optimal, moves better than iteration 1
    p=0.04) — but solve rate is pinned at the supervised level (219–227 vs
    220) and the moves-vs-supervised sign test still favours the frozen
    baseline on MCTS rows (2026-08-20, jobs 4703446 + 4703467 ≈ 1.9 nh;
    `jobs/fwd_iter.slurm` generalized to CFG, commit 962ffb3).** Setup: 120
    fresh 24×24 boards/iteration (ids 5000+), PUCT MCTS generation (800
    expansions, stop 200, root noise, 50% eval-distribution instances),
    ~6k replay-certified move records/iteration + 40k exact anchor records,
    MoveNet warm from the supervised g24r4 ckpt (batch 64; 128+ OOMs);
    benches on the pinned graded exam, 1200/k5, replay-certified.
    | row | solved | mean moves | regret | % opt | exp |
    |---|---|---|---|---|---|
    | supervised A\* (frozen row, §F-M0) | 220/232 | 7.60 | 0.068 | 94.1 | 191 |
    | it1 A\* | 219/232 | 7.598 | 0.096 | 92.2 | 190 |
    | it1 MCTS best-at-budget | **227/232** | 8.09 | 0.463 | 85.5 | 307 |
    | it2 A\* | 219/232 | 7.539 | **0.041** | **96.3** | 187 |
    | it2 MCTS | 224/232 | 7.95 | 0.362 | 87.9 | 310 |
    Gates. it2-A\* vs it1-A\*: same solves, moves 8/1 wins, sign p=0.039 (the
    one significant moves gain of the whole project); it2-A\* vs supervised:
    219 vs 220 (p=1), moves 6/2 (n.s.) — parity on solves, slightly better
    optimality (96.3 vs 94.1%). MCTS rows solve +4..+7 over the supervised A\*
    (p=0.065/0.34) but pay 0.3–0.4 regret on shared solves (sign p≤0.0015
    against) — same shape as F-M2 (§6): exploration buys solves, costs moves,
    at 1.6× expansions. Reading: in the primitive-move space the supervised
    planner was already ≈1 point off the exact optimum, so the loop's room
    was optimality, and two iterations bought half of it (0.068→0.041 regret
    among solved; 94.1→96.3% optimal). F-M4's "beats the supervised baseline
    beyond seed noise" is NOT met on solves (parity) — met on optimality only.
    The honest comparison with the subgoal arm stands as in §3: the forward
    planner's regret 0.04–0.10 is 20–50× below the subgoal language ceiling.
    Sources: `results/fwd_selfplay/g24r4_iter{1,2}/` (benches, gates,
    manifests), `runs/spr/spr-fwd-g24-it{1,2}-470344{6,7}*.out`.

22. **M5, mixed-size curriculum iterations 2–3: the curriculum's gains hold
    but do not compound — after iteration 1's jump every exam is flat within
    noise, and the 3-iteration loop ends with all six exams at or near their
    reachable ceilings (2026-08-20/21, jobs 4704315 ≈ 0.85 nh, 4704316
    ≈ 0.75 nh; protocol as §20).** Trend (A\* graded | frontier per config):
    | iter | g24r4 | g32r4 | g24r8 |
    |---|---|---|---|
    | 1 | 226, rg 2.04 \| **170**/218 | **173**/175, rg 2.61 \| **235**/275 | 159/161, rg 2.35 \| **269**/289 |
    | 2 | 227, rg 2.19 \| 165 | 170, rg **2.15** \| 224 | 159, rg 2.28 \| 265 |
    | 3 | **230**, rg 2.20 \| 165 | 172, rg 2.23 \| 225 | 159, rg 2.35 \| 268 |
    Every iteration-3-vs-2 gate is n.s. (McNemar p≥0.25, all move sign tests
    p≥0.56). Iteration 2's g32r4 frontier dip (235→224, p=0.03 vs iter 1)
    does not recover at iteration 3 (225): iteration 1's 235 stands as the
    series peak there, within-noise of 225 on a paired basis across the pair
    of later iterations. Readings: (a) the curriculum's value was the
    distribution switch (§20), delivered almost entirely in one iteration —
    the same one-big-step-then-flat shape as the 24-only loop (§19) but at a
    much higher level; (b) g24r4 graded 230/232 is the best graded solve
    count of the entire project (at 24 expansions; the B2 ceiling probe says
    ≥229 of 232 reachable — the loop is 1–2 solves from exhausting the
    graded exam); (c) the moves-vs-solves trade stays: regret never improves
    while solves rise. Where the remaining headroom is: frontier moves (the
    MCTS rows and the variants lab's v06/v09 moves-positive arms), not
    solves. The curriculum loop stops here; its iteration-1 nets (the
    frontier-solves peak) and iteration-3 nets (graded peak) are both kept.
    Sources: `results/selfplay/mix_b2mix_iter{2,3}/` (benches, gates,
    manifests), `runs/spr/spr-mix-it{2,3}-470431{5,6}.out`, §20 table.

23. **Variants lab, waves 1–2 (cross-reference; full log in
    `variants/FINDINGS.md` entries 1–5, tables in the report's Variants tab;
    jobs 4719424–31, 4722745–51, 4726033–38 ≈ 6.5 nh): two replicated design
    wins, one honest non-replication, and the project's headline first half
    met on unseen boards by a fully label-free line (2026-08-20/21).**
    (a) **v09 strict-value — the lab's best arm, 2-seed win**: training the
    value net on realized STRICT moves (the benchmark metric) instead of
    abstract plan cost improves every exam in both seeds — frontier Fisher
    p=0.0015, best graded regret of the program (1.98), best unseen row ever
    (177/200). (b) **v04 emit-all — 2-seed win** on frontier solves (Fisher
    3.4e-5); new recommended generation default. (c) v06 Gumbel-root's
    seed-7 graded-moves win did NOT replicate (19/6 → 14/16) — downgraded.
    (d) v01 visit-count policy targets lose decisively: the certified-cost
    softmax target is load-bearing. (e) v08 cold start: from RANDOM init,
    one label-free iteration matches the fully-supervised per-size pair on
    unseen boards (132 vs 134/200) — the supervised prior is worth ≈40
    unseen solves but is not needed to reach supervised-level competence.
    (f) **Unseen-board exam** (200 fresh instances, ids 20000+, all systems
    at 1200/k5): loop line 175–177/200 vs backward supervised 134 vs forward
    101; both-solved moves vs backward 38/16 (12.57 vs 14.78, p=0.004) —
    PROBLEM.md §7's success statement's first half ("≥ backward solves,
    fewer moves") holds OUT OF DISTRIBUTION with no human labels in the
    loop; the second half (moves vs forward: 3/43 against, p=5e-10) is the
    §3 language ceiling measured out-of-distribution and is wave 3's target
    (hybrid action space). Wave 3 approved ≈10–12 nh: v04+v09 stack, v07
    hybrid subgoal+slide space, v12 chained, adoption run into the M5
    curriculum recipe.

24. **Cross-reference — the parallel supervised campaign closed its three
    review objections (supervised_valuenet/FINDINGS §83–§86, 2026-08-20/21,
    ~7 nh, other session): 3-seed replication of both headline rungs (32×32
    rock-solid, +50 median; 16×16/8r bimodal, honest median +14), a 9-way
    tuned "fair second chance" forward planner, and a by-reference A/B
    (+3/450 n.s. — the ceiling is a language property).** What matters for
    THIS project: the rescued forward planner (selection by validation only)
    takes the g16r8 gradable crown outright (266/266) but gains only 8
    frontier puzzles — the backward scale margin stands (+11.3 pts,
    p=1.7e-09). Implication for our headline: our frozen forward baselines
    (arena rows, and the variants lab's unseen-exam forward row, which uses
    the original g24r4 MoveNet) survive the "weak opponent" objection at the
    rung where it was tested, but any future moves-vs-forward claim at other
    rungs should note the forward planner improves ~5 graded/+8 frontier
    under a 9-draw tune — the v07 hybrid's Δ1.16 both-solved moves gap
    should be read against a potentially ~slightly-stronger forward
    opponent. kSubS baseline, forward self-play and a second domain remain
    the successor project's documented future work (their PAPER_PLAN.md).

25. **Variants lab, wave 3 (cross-reference; full log `variants/FINDINGS.md`
    entries 8–10; jobs 4731104/05, 4731113, 4731286, 4733005, 4741426,
    4735236 ≈ 3 nh): the hybrid action space breaks the subgoal-language
    ceiling and survives its same-budget control; the unseen exam is now
    grounded in exact optima (2026-08-21/22).** (a) **v07 hybrid
    (slide-first portfolio search)**: graded 231/232 with regret **1.01 —
    below the pure-B2 language's proven best-plan floor of 1.17 (§3)** — and
    vs the SAME nets at matched budget under standard MCTS (230/232, regret
    1.42): moves 30/0 (p=1.9e-9) graded, 31/1 (p=1.5e-8) unseen (186 vs 182
    solves). First arm to buy realized moves below the language ceiling;
    every slide plan replay-certified. Both-solved gap to the forward
    planner: 0.87 moves graded (was ≈2.2 for the A\* line). (b) **v14
    (emit-all + strict-value)** adopted as the loop recipe (2-seed unseen
    win, §23). (c) **v12 frontier curriculum killed** after a fair
    3-iteration chain (flat, all p>0.8). (d) **Perfect-play grounding of the
    unseen exam**: exact optima for 137/200 instances (mean d\* 8.71; 63
    defeat the exact solver's caps). Extra moves vs perfect on those 137:
    forward +0.12 (but 101/200 solves), v07 hybrid **+1.67** (135/137
    solved, 56% perfectly), same-nets standard search +1.94, A\* line
    +2.9–3.3, supervised backward +4.84. (e) Pending: the adoption run
    (lab recipe into the M5 curriculum line, resumed after a walltime
    timeout) and the wave-4 plan (slide-aware TRAINING — the nets have
    never seen slide-first data; v07 is inference-only so far).

26. **Variants lab, wave 4 + flagship (cross-reference; `variants/FINDINGS.md`
    entries 13–17; ≈10.4 nh incl. two walltime hits — per-leg jobs hereafter;
    program ≈20.8/50): the hybrid's ceiling break is depth-monotone,
    net-independent, and holds on all three exams; the "current best
    planner" row is frozen (2026-08-23).** Flagship = v09 strict-value nets
    + depth-2 root-slides, all solutions replay-certified: graded
    **231/232, regret 0.944, 68.4% optimal** (pure-subgoal floor 1.17 — those
    two averages cover different sets; the paired headline over the same 225
    instances is **0.760 vs 0.978**, §28);
    frontier **177/218** with 47/2 shared-solve move wins vs its matched
    control under the same 1200 cap (which the hybrid cannot reach: its lanes
    sum to 1140, §28) (p=4e-12), d2>d1 10/2 (p=0.039; caveat: most d2
    gain is broader slide screening — genuine two-slide plans are a
    minority); unseen **188/200, +1.47 moves vs perfect** over the **134**
    of the 137 exactly-labeled instances that it solves (57% of those 134
    solved optimally). Robustness: with a
    different net family (v14) the hybrid beats its control 33/0 graded /
    36/0 unseen — the break is a property of the search space, not a
    checkpoint. Negative results logged honestly: v15 slide-training flat
    (uniform nudges ≠ search-picked slides; sharper v16 defined), the v14
    recipe does not compound over chained iterations, and the lab recipe
    is a fresh-loop tool (mature main-line nets already sit at its one-shot
    level, entry 11/174-vs-177 n.s.). Wave 5 approved (~3 nh): cross-size
    transfer of the ceiling break (g32r4/g24r8), depth-3 probe, v16
    search-ranked slide training, program closeout.

27. **Variants lab CLOSED (final cross-reference; full record `variants/FINDINGS.md`
    entries 1–22; program total ≈30.4 of 50 nh): the transfer matrix ends with
    every cell in the hybrid's favor — the last leg, the 32×32 frontier (never
    practiced), is the strongest of the study: 251/275 solved (program record;
    prior best 235; standard search under the same cap 239 — the hybrid's own
    lanes reach only 1140 of that 1200, §28) with moves won on 77 of 80
    differing shared solutions (20.31 vs 21.48, p=1.4e-19) — more solves AND
    fewer moves at a size the method never trained on (2026-08-27, jobs
    4829780/4835373).** Standing record: 16 matched-protocol arms — 4 adopted
    (v04 emit-all, v09 strict-value, v14 stack, v07 hybrid search), 3 killed,
    2 training bets closed as clean double negatives; every claim behind
    replication or a matched control; the unseen exam and its exact-optima
    sidecar remain as permanent instruments; the three durable lessons in
    variants/DESIGN.md §7: train on the number you are graded on; keep
    everything the search examined; search in subgoals plus ordinary moves.
    The lab stands down to maintenance; the main line consumes the adopted
    recipe; M6 continues separately.

28. **Correction pass on the flagship's published claims (2026-08-28, no new
    compute; recomputed from payloads already on disk).** A line-by-line audit
    of the code against the payloads
    (`self_play_robots/report/walkthrough_part3.md` §10) found published claims
    the data does not support. All of them are fixed in the report
    generators, `report/story.html`, this log, `variants/FINDINGS.md` and
    `results/variants/v07_hybrid_actions/VERDICT.json`.
    (a) **The wall-clock claim was an artifact — withdrawn.** The report said
    "the hybrid is also faster in wall clock, at 21 seconds per puzzle against
    103 … about one fifth of the running time". The two runs were never timed
    at equal concurrency. The control's payload records `spr.width` 8 and
    `spr.omp_threads` 2 (`bench_graded_stdmcts.json`, job 4741426,
    `spr.wall_seconds` 3618.3 for 232 instances, so its 232 × 103.36 s of
    process time ran 8-wide). The hybrid ran as ONE process with
    `OMP_NUM_THREADS=16` and no width flag at all — `jobs/v07_wave4.slurm` and
    the driver, which has no width or pool code (job log 4763976: 4957 s for
    the same 232 instances, ≈ its 232 × 21.11 s plus replay). Per-instance
    `mean_seconds` therefore measures different amounts of hardware. Job
    throughput in fact favours the CONTROL: 3618/232 = 15.6 s per puzzle
    against 4898/232 = 21.1 s. No wall-clock claim between these two arms is
    supportable. Every surface now says so, the seconds column was dropped from
    the story page's planner table, and `report/gen_report.py` now prints
    `spr.width`/`spr.omp_threads` on every protocol line and warns that
    `s/inst` does not compare across rows.
    (b) **Denominator error on the unseen row — fixed.** "+1.47 extra moves on
    the 137 puzzles with known optima" is an average over the **134** of those
    137 that the flagship solves (recomputed: 1.4701 over 134, 56.7% optimal;
    `bench_unseen_hybrid_d2.json` rows aligned by index with
    `results/variants/exam/g24r4_unseen.dstar.jsonl`).
    (c) **The floor comparison is now paired, and stronger.** The published
    form set 0.944 (over the hybrid's 231 solves) against 1.171 (over the
    probe's 228 realizable rows) — different populations. Paired over the 225
    graded rows where `results/ceiling/g24r4_b2.json` reports
    `REALIZABLE_EXISTS`, `capped == false` and a known d\*, all of which both
    arms solve: exhaustive B2 optimum **0.978**, standard PUCT 1.191, hybrid d1
    0.836, **hybrid d2 0.760**, hybrid d3 0.676. The flagship is strictly
    shorter than the language's best plan on **27** of the 225 and strictly
    longer on 7. Over the wider 228 (capped rows included) the same table reads
    1.171 / 1.351 / 0.939 / 0.873 / 0.789. The paired numbers are the headline;
    the unpaired pair is kept as labelled context.
    (d) **Budget asymmetry, against the result — now stated.** Both arms run
    under the 1200-expansion arena cap, but the hybrid's lanes sum to
    `b0 + top_m * sub`: **1140** at depth 2 (500 + 8*80) and **1160** at depth 3
    (460 + 10*70). The payload rows confirm it — the largest `expansions` in
    `bench_graded_hybrid_d2.json` is 1140 and in `..._d3.json` is 1160, while
    the control reaches 1200. The hybrid wins while unable to reach the
    control's ceiling. The same 1140 applies to every transfer leg (§27).
    (e) **Two smaller items.** The `variants/v07_hybrid_actions.py` module
    docstring named a function that does not exist (`mcts_root_slides`) and
    described only the depth-1 defaults; it now describes the inline algorithm,
    the depth-2/3 frontier, the undocumented `cost0 >= best + 3` pruning rule
    and the real budgets. And `spr/bench.py` hard-coded `"byref": False` into
    every protocol block while `spr/search.py::run` sets `byref = vocab ==
    "b2"`; the field is now written from the vocabulary. **Every B2 payload on
    disk that `spr.bench` wrote — which includes every `spr.arena` payload,
    since the arena shells out to the bench — carries `protocol.byref: false`
    and that value is wrong.** By-reference helper resolution was active in all
    of them. (The `variants/v07_hybrid_actions.py` payloads are unaffected: that
    driver writes `"byref": True` itself.) Read the vocabulary from
    `protocol.vocab` (or `search_options`), never from `byref`, on any
    bench/arena payload dated before 2026-08-28.

29. **Subgoal-discovery plan, Stages 0–1: the 16×16 table is rebuilt from move
    dumps and both gates PASS — but two corrections fall out of it, and the
    state subgoal space ("robot R comes to rest on cell C") reaches the exact
    optimum on 40/40 probed instances where the hand-written language cannot
    (2026-08-28, job 4862862 = 0.028 nh + 412 CPU-seconds on the login node;
    full write-up `subgoal/STAGE01.md`).** Harness `subgoal/table.py`: every
    number is recomputed by replaying each payload row's own move dump under
    the joint-game rules (`simulate.slide`, a no-op slide illegal — the physics
    of `eval/replay_validate.py`) and requiring the replayed length to equal
    `d*`; no aggregate field and no document was copied. Headline = percent of
    all 450 solved move-optimally.
    | model (bench450, 1200 exp, k=5) | optimal % of 450 | solved / 450 | extra moves on its solves |
    |---|---|---|---|
    | forward (move-by-move, supervised, `candidate_scored.ckpt`) | **94.2%** (424/450) | 450/450 | 0.067 |
    | backward (subgoal, supervised, B1-s21 pair under B2 flags) | 53.3% (240/450) | 432/450 | 1.840 |
    | self-play, v14_stack nets (benched at 16×16 for the first time) | 51.3% (231/450) | 439/450 | 1.995 |
    | self-play, mix_b2mix_iter3 nets (same) | 52.4% (236/450) | 438/450 | 2.011 |
    Every row reproduces its payload exactly: 0 misaligned rows, 0 replay
    failures, 0 length disagreements against `realized_strict`, 0 rows shorter
    than `d*`, mean extra == `mean_regret` to machine precision. The two rows
    with recorded references also match them row for row (`comparison_forward.json`
    candidate_scored system, `final450_backward_b2_seed21.json`).
    (a) **`pct_optimal` is a percentage of SOLVES, not of n.**
    `eval/compare.py::aggregate` divides by `len(solved)` (lines 507–509). The
    recorded 16×16 backward figure "55.6% optimal" is 240/432; per 450 it is
    **53.3%**. The forward row is unaffected only because it solves all 450.
    Nothing on disk is wrong, but no optimal-% figure in this log is comparable
    across arms with different solve counts without restating the denominator.
    (b) **No self-play checkpoint had ever been benched at 16×16.** Of the 493
    payloads under `results/`, 28 are on `bench450` and all 28 are supervised /
    M1 / M2 arms; every `results/selfplay/` and `results/variants/` payload is
    g24r4, g24r8 or g32r4. The size-free nets were therefore run unchanged at
    g16r4 under the backward row's own protocol (arena A\*, B2, anytime,
    replay-certified; `jobs/subgoal_stage0_selfplay16.slurm`, job 4862862,
    qgpu_exp, 13 min). Result: at 16×16 the self-play line solves MORE than the
    supervised backward planner it grew out of (439/438 vs 432) and is slightly
    WORSE on move quality (231/236 vs 240 optimal; 1.995/2.011 vs 1.840 extra)
    — the same "search was distilled, moves were not" pattern §19 recorded at
    24×24, now measured on the graded 16×16 exam as well.
    (c) **Stage 1 PASS, at the ceiling.** `subgoal/space.py` searches the state
    subgoal space exhaustively with no network: a node is the joint state, a
    child is any (robot, cell) the joint-state slide BFS says that robot can
    come to rest on (all 1024 candidates realized against the real rules, true
    slide counts as edge costs), ordered by an admissible "any-stop" relaxation
    of the target robot's distance, so the first goal popped is optimal IN THE
    SPACE. Two deterministic 20-instance samples (the first 20; every 23rd,
    spanning the bench):
    | sample | state space | hand-written base (§3) | hand-written B2 (§3) |
    |---|---|---|---|
    | first 20 | **0.000 extra, 20/20 optimal** | 2.000 over 20, 13/20 | 1.000 over 19, 13/20 |
    | stride 23 | **0.000 extra, 20/20 optimal** | 2.059 over 17, 11/20 | 0.632 over 19, 15/20 |
    All 40 solutions are replay-certified and all 40 equal `d*`. The
    hand-written language is strictly worse on 7/20 and 9/20 (base) and 7/20
    and 5/20 (B2, inconclusive rows counted), and **better on none of the 40**;
    the gaps are large (base needs 15 where `d*` is 5, 28 where it is 7, 22
    where it is 3). The §3 floor of +1.42 / +0.90 is a property of that
    vocabulary, not of subgoal planning: in the state space the floor is 0.00.
    The optimal solutions use 1–5 subgoals (mean 2.0 / 2.45), so a planner here
    makes very few decisions per puzzle. Caveat carried into Stage 2/3: this
    probe popped up to 60,685 nodes with a perfect admissible heuristic, while
    the arena budget is 1200 expansions at k=5 — "the space contains the
    optimum" is not "a k=5 beam over network scores finds it".
    Sources: `subgoal/STAGE01.md` (provenance appendix gives every number a
    path), `subgoal/{table,space}.py`,
    `results/subgoal/{stage0_table,stage1_state_space_first20,
    stage1_state_space_stride23,v14_stack_g16r4_bench450_astar,
    mix_b2mix_iter3_g16r4_bench450_astar}.json`,
    `runs/spr/spr-sg-stage0-4862862.out`.

30. **Subgoal-discovery plan, Stage 2: the goal-conditioned cost net FAILS its
    gate — it beats the plan's named baseline by +0.32 Spearman but only ties
    the stronger learning-free one (−0.02) — and the beam measurement shows the
    binding constraint is the plan's selection rule, not the network: ranking
    1024 candidate subgoals by their EXACT cost keeps an optimality-preserving
    subgoal in a top-5 beam at only 61.2% of roots (2026-08-28, jobs 4863328 /
    4863358 / 4863393 / 4863454 = 0.097 node-hours of a 1 node-hour budget;
    full write-up `subgoal/STAGE2.md`).**
    Cost target, fixed in `subgoal/costnet.py` and binding on Stage 3:
    `c(s, R, C)` = slides for robot R to come to rest on cell C with every other
    robot FROZEN — i.e. `subgoal/space.py::rest_cells`, the exact edge weight of
    the macro expansion. The "other robots may move" variant was rejected
    because it names no unique successor state and therefore is not an edge
    weight at all. Labels are exact (240 groups re-derived by brute-force
    enumeration of all slide sequences to length 3, 0 mismatches). Architecture:
    the size-free `LoopedLayer` encoder unchanged; conditioning moved to two
    input channels (query robot / other robots) with the GOAL CELL MOVED OUT OF
    THE ENCODER INTO THE READOUT, so one encoder pass scores all 256 goals of a
    robot and four score a decision's 1024; the 96-bin HL-Gauss cost head kept
    with a 3-token gather instead of 5; one 1-logit reachability head added.
    Trained on 2800 states from train-split boards 1000-1699 only (2.87 M
    labelled candidates); zero board overlap with either eval set.
    (a) **The pre-registered production recipe COLLAPSED, and the cause is
    recurrence depth, not the positional signal.** At recurrence 12 the net sat
    on the constant-value plateau for 30 epochs: `val_group_spread` exactly
    0.00, rho ~ 0, top-1 at the 0.083 random rate, and the reachability head
    predicted "unreachable" for all 460 800 bench450-root candidates. A
    four-arm diagnostic (job 4863393, 4 epochs each) gives rho after 4 epochs of
    0.004 (pe=none, rec 12) / 0.056 (sin2d, rec 12) / **0.583** (none, rec 4) /
    **0.556** (sin2d, rec 4), with spread 0.00 / 0.00 / 1.91 / 1.84. Twelve
    weight-tied steps over a field whose non-robot cells start from identical
    embeddings drive every token to one vector, and a readout that must separate
    256 goal cells then has nothing to read. This is the constant-value-plateau
    mode of `supervised_valuenet/FINDINGS.md` sections 50 and 57 (7 of 8 cold
    trainings; `val_group_spread` exactly 0.00) on a new task and with an
    unambiguous cause. The reported run is
    identical except `--recurrence 4`; both payloads are kept.
    (b) **Gate FAIL, on both held-out board sets.** Spearman over a state's
    truly reachable candidates, all four robots pooled:
    | scorer | 450 bench450 roots (boards 2400-2549) | 400 states, boards 1900-1999 |
    |---|---|---|
    | goal-conditioned cost net | 0.749 (top-1 100.0%, MAE 2.83) | 0.723 (100.0%, 2.83) |
    | B1 any-stop relaxation (the plan's baseline) | 0.431 (97.6%, 2.82) | 0.392 (95.8%, 3.06) |
    | B2 lone-robot exact slide BFS | **0.773** (98.0%, 2.70) | **0.739** (97.5%, 3.20) |
    Required margin over the better baseline was +0.05, committed (d9032e1)
    before either job was submitted; measured **-0.024** and **-0.016**. The net
    is the better ranker at the head of the list (top-1 100% vs 98.0/97.5) and
    the worse regressor (MAE 2.83 vs 2.70; a constant predictor scores 2.49).
    Where B2 is defined, B2 reaches rho 0.905 / MAE 0.50 against 0.791 / 2.86 —
    the other robots usually do not change the answer.
    (c) **The beam number, which predicts Stage 3 better than any of the above.**
    `subgoal/optset.py` recomputes, for each of the 39 distinct Stage 1
    instances, the FULL set of optimality-preserving next subgoals at every
    state of a fewest-subgoal optimal path (A* restricted to f <= d*, all tight
    parents kept, reverse pass from the goal states; 0 capped, all 39 reproduce
    d*, and all 40 Stage 1 rows agree). 4.34 of 1024 candidates preserve
    optimality on average, so a random top-5 survives 2.1%. Expected survival
    under uniform random tie-breaking, k=5, over 86 decisions / 39 roots /
    chained over a whole optimal path:
    | ranking | per decision | at the root | whole path |
    |---|---|---|---|
    | network cost | 59.3% | 66.7% | 30.8% |
    | EXACT cost (the oracle for this net) | 54.3% | 61.2% | 28.0% |
    | network cost + relaxed h of the child | 62.8% | 71.8% | 38.5% |
    | EXACT cost + relaxed h (Stage 1's A* priority) | **74.2%** | **77.3%** | **54.6%** |
    | B1 + relaxed h | 43.8% | 42.1% | 34.3% |
    Because k=5 is a HARD prune, the root column is an upper bound on Stage 3's
    optimal-% and the chained column a lower bound. Three consequences.
    (i) Cost-only ranking caps Stage 3 in the low 60s however good the network
    is; the learned ranking is actually BETTER than the exact one at k<=5
    (66.7 vs 61.2 at the root) purely because continuous scores break the
    massive cost ties better than chance. (ii) Adding a remaining-distance term
    is worth more than learning the cost, and costs one board-only BFS per
    instance. (iii) Stage 3's bar is 53.3% of 450 (section 29); the best ranking
    measured here brackets it at 54.6-77.3%, the network's at 38.5-71.8% — not
    excluded, no margin.
    (d) **The target was the wrong object.** At 10.9 ms per decision the exact
    cost is cheaper than the network that predicts it, and a Stage 3 expansion
    must compute it anyway to build children. What is worth learning in this
    space is the remaining distance to the final goal, not the subgoal's own
    cost. Budget: 0.097 node-hours (job 4863331 on `qgpu` was cancelled
    unstarted — 63 of 72 GPU nodes in maintenance, start estimated 2026-08-30 —
    and the leg re-cut to the 1 h `qgpu_exp` cap).
    Sources: `subgoal/STAGE2.md` (provenance appendix gives every number a
    path), `subgoal/{costnet,stage2,optset}.py`,
    `jobs/subgoal_stage2{,_smoke,_diag}.slurm`,
    `results/subgoal/stage2/{optsets,eval_rec4_bench450_roots,
    eval_rec4_b1900-1999,beam_rec4,eval_cost_bench450_roots,
    eval_cost_b1900-1999,beam_cost}.json`,
    `results/subgoal/stage2/*.npz`,
    `runs/spr/spr-sg-stage2{-4863358,r4-4863454}.out`,
    `runs/spr/spr-sg-s2diag-4863393.out`.
