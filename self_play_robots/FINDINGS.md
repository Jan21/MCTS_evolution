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
