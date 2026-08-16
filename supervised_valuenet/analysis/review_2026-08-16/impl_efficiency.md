# Efficiency chapter: wall-clock robustness (implementation memo)

## 1. What data existed (audit)
Every comparison JSON — `eval/results/final450_*.json`, `comparison_forward.json`,
and all `scaling/results/g*/comparison{,_ungraded}*.json` — carries a per-instance
`rows` array with `seconds` for **both** planners, at **all six rungs**, on both the
gradable and beyond-oracle sets (100% coverage: 450/316/266/232/161/175 graded,
134/184/218/289/275 frontier). No slide/physics-call counts exist in any row; the
matched-unit data is only the 5-puzzle pilot in `compute_accounting.json`. Machine
tag comes from the run date (origin < 2026-07-20 ≤ Karolina); a same-machine
backward/forward pair exists at every rung and set (11 rows).

## 2. What was added
`report_data.build_wallclock` recomputes median / p90 / total / solved-within-t
curves from the rows; `sec_fair_wallclock` (eval/report_sections_scale.py) now
renders one 9-column table + 11 solve-rate-vs-wall-clock log-x panels + an honest
reading. `budget_curve_panel` gained generic x-axis params; `.sep` CSS added; the
32×32 KPI tile now prints medians beside its means.

### Table (median | p90 | total, backward vs forward; same-machine rows only)
| rung · set | bwd med/p90/total | fwd med/p90/total | ratio med / total |
|---|---|---|---|
| 16×16·4r graded | 0.38 s / 3.19 s / 9 min | 0.38 s / 1.97 s / 7 min | 1.0× / **0.8×** |
| 16×16·6r graded | 0.39 / 3.75 / 27 min | 5.38 / 43.8 / 114 min | 13.8× / 4.3× |
| 16×16·6r frontier | 4.58 / 173 / 94 min | 326 / 418 / 10.5 h | 71.2× / 6.7× |
| 16×16·8r graded | 0.43 / 11.4 / 65 min | 5.55 / 41.0 / 94 min | 12.9× / 1.4× |
| 16×16·8r frontier | 1.87 / 384 / 4.5 h | 334 / 393 / 13.1 h | 179× / 2.9× |
| 24×24·4r graded | 2.90 / 22.9 / 30 min | 65.9 / 1041 / 17.7 h | 22.8× / 35.2× |
| 24×24·4r frontier | 54.0 / 66.2 / 3.0 h | 2562 / 2945 / 148.4 h | 47.4× / 49.6× |
| 24×24·8r graded | 1.18 / 22.8 / 91 min | 59.5 / 427 / 8.2 h | 50.6× / 5.4× |
| 24×24·8r frontier | 424 / 945 / 34.6 h | 1228 / 1766 / 93.6 h | 2.9× / 2.7× |
| 32×32·4r graded | 8.80 / 94.7 / 89 min | 656 / 4033 / 67.0 h | 74.5× / 45.0× |
| 32×32·4r frontier | 59.4 / 93.4 / 4.4 h | 2377 / 2453 / 181.8 h | 40.0× / 41.2× |

Reading: forward is slower at median, p90 and total at **every** scaling rung
(2.9×–179× median, 1.4×–49.6× total), but at base scale wall-clock **favours
forward** (p90 3.19 vs 1.97 s; total 9 vs 7 min) — FINDINGS 27 confirmed. The
backward tail is heavy (g16r8 frontier p90 = 205× its own median; tightest p90
margin 1.0×), so median and total are always printed together.

## 3. The "17.0× and 17.0×" bug
In `report_sections_hygiene.py`: the records-ratio loop required *exactly one*
backward corpus per config, so the four configs with two vocabularies were dropped
and only g24r4 survived → min = max. Fixed to use the largest backward corpus per
config (most generous to backward), plus a degenerate-range guard. Now reads
"between 4.9× and 17.0× … (smallest at 16×16·6r, largest at 24×24·4r)".

## 4. Build
`python -m eval.build_report` passes: **1444 checks + 51 assertions** (was 1390 + 50).

## Not done
- Matched-unit (slide-call) accounting beyond the 5-puzzle pilot: no per-rung
  physics counters exist in any result file; that needs new instrumented runs.
- Not committed. Note: `eval/compare.py` / `eval/end2end.py` show as modified in
  git — not by me (concurrent session).
