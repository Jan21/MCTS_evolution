# Devil's advocate: the headline is not yet convincing

## Attacks the repo already answers (concede these)
Selection of the frontier set (pooled union + 44 strata, §22/§26), test leakage
(§23), "it's just the hand-written labeler replayed" (§49: nets add 8–23 points,
and the heuristic loses the pooled union to forward), self-certified solves
(replay validation), matched physics unit (§25: ~90×), data-budget asymmetry
running against the winner, and no CIs (118 cells). The tuning
ledger is genuinely two-sided. This is unusually honest work.

## What still fails a hostile read
**1. The budget unit is the thing being claimed.** COMPARISON.md footnote 1
concedes a shared expansion cap is "generous to the backward system"; prefix
checks, free exact fixes, zero-cost pops and park repair — the measured,
heavy-tailed, network-free cost centre — sit outside the counter. The probes then
show forward gaining +21.9/+31.2/+12.5 points at 4–5× budget, three of four
margins going insignificant (§28/§31/§35). Only 32×32 survives. Backward was
never given 4–5×, no iso-wall-clock row exists, and at base wall-clock favours
forward.

**2. The opponent is weak by construction, and never given a fair rescue.**
Forward's teacher fails on 30–61% of instances, so the frontier is
out-of-distribution for it by definition; it got one lr applied by pattern after
observed collapse, one seed at every scale, and no self-play (the repo's own
objection 2.3 calls this reject-level). Backward meanwhile gets exact
precomputed distance tables at inference, a hand-written vocabulary and a
hand-written repair pass: engineered hierarchy vs generic flat learner.

**3. Seeds.** §44 measures a 28-point frontier swing from the value seed alone,
bimodal, ~50% degenerate — and the headline zero-shot pair is a single draw with
the policy net unseeded. Also, "zero-shot is the best configuration" was chosen
after seeing benchmark rows.

**4. Metric.** Forward wins quality wherever both solve and wins four of six
graded sets; the headline is solve-rate-only, on the half backward wins.

(Minor: the report renders the data ratio as "between 17.0× and 17.0×".)

## What would move me
1. **Forward given a real chance** — self-play or 3 seeds × 3 lrs at g16r8 and
   g32r4. Medium (~20 nh). Kills the strongest objection.
2. **Iso-cost rows** — equal wall-clock and equal slide+NN calls, backward also
   at 4800. Medium (~15 nh).
3. **3 seeds of the headline backward pair** at three rungs. Medium (~12 nh).
   A kSubS-style learned-subgoal baseline is the real fix, but that is large.
