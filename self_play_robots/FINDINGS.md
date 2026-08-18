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
