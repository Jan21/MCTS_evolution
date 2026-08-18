# Audit: self_play_robots/FINDINGS.md numbers vs the result files (2026-08-18)

Scope: every quantitative claim in FINDINGS §2, §3, §5, §7, §8, §9, §10, §12, §13,
checked against the file each entry cites (comparison payloads read as
`spr.arena.summarize` does: `systems[<first backward system with rows>].aggregate`,
i.e. `mean_moves`/`mean_regret`/`pct_optimal` over solved rows, `mean_expansions`
over all rows; ceiling files via `summary`; gauge files via `audit_summary`;
generation manifests; `.gate.json` / `.vs_greedy.json` / `gate_vs_*.json` paired
blocks). Paired tests that FINDINGS quotes without a saved file (§5, §8, §9, §10,
§12, §13) were recomputed on the login node with `spr.gate.paired` from the two
payloads named in the text. Method: one Python pass over `results/**.json` (and
`supervised_valuenet/{eval,scaling}/results/*.json` for the recorded references),
then claim-by-claim comparison. Nothing was edited except this memo and the
report generator.

**Headline: 3 substantive mismatches, 4 rounding-level ones, no wrong
conclusions.** Every solve count, mean-moves, regret, %-optimal, expansions,
ceiling, gauge, manifest and job-id figure in the nine sections reproduces
from its file (≈ 260 numbers checked; the full pass is summarised per section
below). The mismatches are all in derived/paired figures.

## Mismatch table

| # | section | claim (as written) | file(s) | value in file | verdict |
|---|---|---|---|---|---|
| 1 | §7 table, mixed pair @ g24r4 | "61/12 mv wins, p=5e-9" | `results/m1/mixed_value_warm_s21_g24r4.gate.json` `paired` | `moves_wins_a` 62 / `moves_wins_b` 11, `sign_p_moves` 9.1e-10 | **wrong reference file.** The quoted numbers reproduce exactly when the pair is compared with the M0 CPU re-run `results/m0/g24r4_exact_prefix.json` (61/12, p=4.8e-9) instead of the recorded `scaling/results/g24r4/comparison.json` that the cited `.gate.json` uses (3 rows differ by ±1 strict move between the two references, FINDINGS §2). Conclusion unchanged. |
| 2 | §7 table, g16-only pair @ g24r4 | "+8, p=0.04; 58/10, p=2e-9" | `results/m1/g16_value_warm_s21_g24r4.gate.json` | +8 (10 vs 2), p=0.039 ✓; wins **58/8**, p=**1.8e-10** | same cause as #1 (vs the M0 re-run: 58/10, p=2.4e-9). |
| 3 | §7 table, g24-only pair @ g24r4 | "+11, p=0.001; 59/11, p=4e-9" | `results/m1/g24_value_warm_s21_g24r4.gate.json` | +11, p=0.00098 ✓; wins **60/10**, p=**8e-10** | same cause as #1 (vs the M0 re-run: 59/11, p=4.5e-9). |
| 4 | §7 table header (also `status.json` M1 note) | "ref exact pair 205/232, **11.80, 4.20**, 38.5%, 26.9 exp" | cited `.gate.json` `ref` block = `scaling/results/g24r4/comparison.json` | mean_moves **11.81** (11.8098), mean_regret **4.22** (4.2195) | rounding/provenance: 11.80/4.20 are the M0 CPU re-run's values (`results/m0/g24r4_exact_prefix.json`: 11.795, 4.205), not the recorded reference the gate compares against. §5's "11.80/4.20 on CPU" is the correct attribution. |
| 5 | §12 reading | "The B2 language adds +12 graded solves and **≥ +34** frontier solves over the base ceiling at g24r4" | `results/ceiling/g24r4_frontier_base.json` (103), `results/selfplay/g24r4_b2_iter0/m1mixed_b2_g24r4_bench_unsolved_astar.json` (133) | 133 − 103 = **+30** (+34 is 133 − 99, i.e. over the M1 base-vocabulary nets' frontier row, not over the ceiling) | **arithmetic / wording error** (+12 graded = 228 − 216 is right). |
| 6 | §5 (a) | "MCTS removes 0.46 (g16r4) … moves per solved puzzle from the supervised A* rows" | `results/m2/persize_v2_g16r4_{astar_child,mcts_min}.json` | 8.384 − 7.915 = **0.469** (0.47) | rounding of rounded operands (8.38 − 7.92); g24r4's 0.36 is exact. |
| 7 | §5 (a) | MCTS expansions "still ≤3% of the 1200 cap" | `results/m2/persize_exact_g24r4_mcts_min.json` | 38.56 / 1200 = **3.2%** | slightly over the stated bound (g16r4: 2.4%). |
| 8 | §5 (b) | "label-consistent f is WORSE at 24×24 (**+0.96** regret)" | `results/m2/persize_exact_g24r4_astar_{parent,child}.json` | 5.205 − 4.254 = **0.951** (0.95) | rounding, 0.01. |
| 9 | §9 (b) | "at g32r4 … MCTS's regret is **0.07** above the ceiling's" | `results/transfer/g32r4_graded_mcts.json`, `results/ceiling/g32r4_base.json` | 1.635 − 1.558 = **0.077** (0.08) | rounding of rounded operands (1.63 − 1.56). |
| 10 | §10 reading | "216 = the language ceiling; regret **0.1** above it" (MCTS) | `results/selfplay/g24r4_iter1/bench_g24r4_mcts.json`, `results/ceiling/g24r4_base.json` | 1.856 − 1.722 = **0.13** | loose rounding (A* row is 0.75 above; the sentence refers to MCTS). |

Everything else in the nine sections matches its file to the printed precision.
Notable confirmations (the ones most likely to be doubted):

- §2: g16r4 arm 0/450 per-row diffs; g24r4 arm 34/232 diffs = 31 expansions-only rows (deltas 1–5) + 3 rows with realized_strict −1, solved-vectors identical; forward arm 450/450, 0.0667, 94.22%, 6.436 = `comparison_forward.json` candidate_scored row.
- §3: all 12 ceiling cells per arm (n, solve ceiling, categories, mean d*, best/first moves and gaps, % optimal, slack re-runs, wall 5.8/9.6/98.6/103 s, `capped`=0 in base arms) match; derived 2.5/0.7 moves and 11/7 solves headroom, 0.40/0.42 first-vs-best gaps.
- §5, §8: every aggregate row and every quoted paired result (`.vs_greedy.json`, plus MCTS-vs-arena-A* 57/0, 15/0, 59/0, 27/0 and size-free-vs-per-size MCTS 42/27 p=0.09 recomputed) matches.
- §9: all recorded rows (`comparison*.json` of g24r4/g32r4/g24r8, twin pair for the g24r4 frontier), all transfer aggregates, McNemar/sign figures (p=0.02, 0.004, 1e-4, 0.25, 9e-4; 20/4, 17/8, 22/5, 46 wins) and both ceilings (156/175, 1.56, 57.7%; 148/161, 1.58, 58.8%) reproduce.
- §10: manifest (1,774 / 1,268 = 71.5% / 12,244 / 49.6 exp / 6.8 s / 4 OOM / boards 5000–5119 / 600-150-1.5-0.25 search), gauge (0.915, Jaccard 0.859, gap 0.99 / 0.82, 0 negative, n=200), decision groups 1,160 train + 203 val (counted from `runs/spr/selfplay/g24r4_iter1/records.jsonl` with the buffer split 5000–5101 / 5102–5119), bench + paired (8/7, 5/3), val regret 2.00 (policy best epoch) / 1.22 (value) in the iteration's `metrics.csv`.
- §12: iteration-2 manifest (2,869 / 1,525 trivial / 4,268), gauge 0.85, benches, paired 8/7, iteration-1 transfer rows (156, 146, 140), the whole B2 zero-shot table (223/9.51/1.95/55.6/52.8; 228/9.19/1.55/58.8/490; 133/19.43/515; recorded 199 and 125 rows), the frontier B2 probe (114 proven, 104 inconclusive, 8 workers, 100k frontier, 120 s) and the g32r4 B2 ceiling (166/175, 1.28).
- §13: manifest (60 boards 8000–8059, 300/80, 6 workers, 681 / 598 = 87.8% / 5,822 / 108 exp / 36.9 s), benches, paired (+4/−1 p=0.375, 12/10; +2/−1, 11/10), −19% expansions (42.96 vs 52.75), policy val regret 1.74 at 109 val groups and value spread 2.2–2.7 in `runs/spr/selfplay/g24r4_b2_iter1/*/metrics.csv`.
- Every Slurm job id quoted in §2–§13 appears as `spr.slurm_job_id` / `slurm_job_id` in the corresponding result files (4679719, 4680465, 4679720, 4680480/4680481/4681012/4681029, 4680956, 4680957, 4680466, 4681264, 4681263, 4682184, 4682172, 4682185, 4682786, 4685187, 4685442).

## Numbers that could not be traced to any result file

| section | number | where it would live | note |
|---|---|---|---|
| §2, §3, §5, §7, §8, §9, §10, §12, §13 | node-hour costs (0.09, 0.02, 0.06, 0.35, ≈2.2, 0.25, 0.2, 0.23, ≈1.2, ~0.6 nh) and §10 "1 h 51 min" | `sacct` / job accounting, not files | not checked (no Slurm queries in this audit); the generation phase alone is `seconds` 1554 in the iteration-1 manifest. |
| §3 (d) | "B2 98.0% here vs the **99.6%** recorded with 4× caps" | `supervised_valuenet/FINDINGS.md` | external to this project's files. |
| §7 (a), (d), (e) | per-size val metrics of the mixed nets "policy 2.97/0.93, value 2.02/0.42", "val recall@5 0.98 at g24r4 / 0.77 at legacy g16r4", bench cost "~1 min per exam", "8 min" | `runs/spr/m1/*/lightning_logs/*/metrics.csv`, `runs/spr/spr-m1-bench-4680956.out` | outside `results/`; not re-derived. |
| §10 | training start values "1.97" (policy) / "1.24" (value) | the M1 nets' `metrics.csv` under `runs/spr/m1/` | only the iteration's end values (2.00 / 1.22) were verified. |
| §13 | "**71** are by-reference (helper standing on a planned cell)" | `runs/spr/selfplay/g24r4_b2_iter1/records.jsonl` | records carry no by-reference flag (`cand_helper` is always a position; `vocab: b2` on all 5,822); a helper-on-planned-cell heuristic counts 109, and the job log has no such line — the 71 is unverifiable as stored. |
| §13 | gauge attempt "did not finish in 2.5 h (25 GB RSS)" | job log / `sstat` | no result file (the bounded gauge is job 4689629, pending). |
| §12 (a) | "harder instances get looser certified labels" (0.85 vs 0.915) | — | verified as numbers; the causal reading is not a file claim. |

## Suggested one-line fixes (for the owner; FINDINGS.md was not edited)

1. §7: either quote the `.gate.json` paired figures (62/11 p=9e-10; 58/8 p=2e-10; 60/10 p=8e-10) or say the paired test used the M0 CPU re-run as the reference; make the header's reference read 11.81 / 4.22 (or label 11.80/4.20 as the M0 re-run). Same for `results/status.json` M1 note ("4.20").
2. §12: "≥ +30 frontier solves over the base ceiling (+34 over the M1 base-vocabulary nets)".
3. §5 (a): "0.47 (g16r4) / 0.36 (g24r4)"; "≈3% of the 1200 cap".
