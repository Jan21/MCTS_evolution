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
