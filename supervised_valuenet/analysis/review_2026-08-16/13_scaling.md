# The scaling story — memo

**1. Axes tested, and where each ladder stops.**
Two separate ladders exist, testing different things. (a) The *planner* (backward
vs forward, head-to-head decision quality) ladder covers grid {16,24,32} × robots
{4,6,8}, six configs (`scaling/configs.py`), capped at g32r4/g24r8. It stops there
because board construction (`gen_grids.make_board`, O(G⁴) all-pairs Dijkstra) was
infeasible past 32 until the lean-board fix (FINDINGS 62) — a data-generation cost
wall, not an architecture cap — and because the exact solver (Rust) hard-asserts
n≤64, R≤10, so nothing above 32×32/8 has ever had graded ground truth for the
planner comparison. Robot count never exceeds 8 anywhere in the repo. (b) The *NN
labeler* grid ladder is separate and goes much further: 17→64 verified against
exact labels (flat 86–92% argmin agreement, FINDINGS 65/70), then 80/96
UNVERIFIABLE (certified-valid, no ground truth exists past 64, FINDINGS 72c). This
ladder is single-robot-axis (r4 only) and tests label fidelity, not planner
solve-rate — it stops at 96 because that's already 6× the net's training size and
yield is dropping (28% at 96).

**2. Planner (not labeler) at 48–64: what's needed, and vs. more robots.**
The controlled "downstream equivalence" suite (FINDINGS 71–74) only retrains and
benches planners at g24r4/g24r8/g32r4 — the planner-side search/realize/descent
loop above 32×32 is explicitly unprobed (§74's "remaining owner-approved work").
Needed: (i) lean-board-based bench-instance generation at 48/64 (labeler side is
done, planner-side realize.py + descent search haven't been timed/memory-tested at
that scale), (ii) a twin corpus + retrain + benched comparison, same recipe as
g32r4. The robot axis is arguably the higher-value next rung: FINDINGS 73/74 show
label fidelity gates downstream utility, and the one place equivalence *broke*
(g24r8, 82.2% argmin) was the 8-robot cell, not a big-grid cell — so 8-12 robots is
where the real risk and the real research question live, while 48/64 is mostly an
engineering scale-out of a mechanism already validated at 32.

**3. Next-step ideas.**
- **Push robot axis to 10-12 at fixed 24×24** (medium: reuses lean-board infra, ~1
  twin corpus + retrain, ~4-6 nh) — tests whether the fidelity-gates-utility curve
  continues past 82%, the most theoretically interesting open question.
- **Planner-side 48×48 deployment run** (large, ~2 days per §'s ledger) — extends
  the validated mechanism, lower risk, mostly confirms rather than discovers.
- **Combine grid+robot stress cell (e.g. g48r8)** (large) — cheapest way to find
  where both axes compound, since g24r8 already shows compounding is the real
  failure mode.
- **Exact-arm multi-seed replicates at g24r8** (small, ~1 nh) — needed before
  trusting the 21-point equivalence-break number, per §73's own caveat.
