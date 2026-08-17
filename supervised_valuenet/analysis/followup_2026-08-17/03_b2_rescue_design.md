# B2 rescue seed-corpus design brief (2026-08-17) — NOT LAUNCHED

Design + pricing (Sonnet agent, this session). Decision: **not funded from
the labeler track** — it revives an economic claim PAPER_PLAN.md:114
correctly drops ("the NN labeler saves compute"), and its main beneficiary
is the by-reference planner thread (FINDINGS §78's "next lever"). Archived
here for the Track-1 session / owner.

## Key correction

The 13.5% uncapped by-reference share is measured at **g16r4 (base)**, not
g16r6 — FINDINGS §32: same 111 boards, cap 50,000 → 13.5%, cap 5,000 →
4.8%. g16r4 is also the CHEAPEST config to solve uncapped (692 s/board vs
2,696 at g16r6; robots dominate cost, not grid size). So the seed corpus
should be g16r4: the only config with a like-for-like uncapped target.

## Pricing (N=300 boards, budget-iters 50,000, per-graph 10)

| step | nh | basis |
|---|---|---|
| uncapped Rust solve, 300 boards | ~7.2 | measured 692 s/board (§30) ÷ 8 (1-GPU qgpu_free billing = 16 cores) |
| labeler fine-tune (warm-start v1/B2 net) | ~0.3–0.6 EST | analogy to §33 retrains |
| descent label + 600-inst gate verify | ~0.1–0.4 | §64-§66 rates |
| **total** | **~8** | vs ~32 nh burned on the failed distillation (§68) |

Yield: ~58,500 records, ~7,900 by-reference. Alternative sizes: 200 boards
→ 4.8 nh; 500 → 12 nh.

## Success/failure criteria (no further spend)

- By-reference share of FRESH descent labels (boards outside the seed):
  success ≥8–10% (vs 1.3%/0.3% distillation floor, 13.5% target);
  failure = pinned at 1–3% → bottleneck is the feature gap (§77c: policy
  net cannot name the referenced cell), not the data.
- Argmin gate on the seed's held-out split: must clearly beat the B2 net's
  53.6%/50.0% floor (§68).

## Dual-use

Same corpus + `RR_BYREF_RECORDS=1` feeds the planner-side by-reference
retrain (§77c/§78) for ~1 nh incremental. One solve, two experiments.

## Risks

- Rust engine has NO wall-clock timeout by design (io.rs rejects
  `timeout_s`); fat right tail (p95 16–20k iters vs median 342) → small
  samples can overrun. Mitigate: `--limit` smoke test first (§30's rule).
- The 13.5% target is itself a ~111-board point estimate.
- Clean data may be necessary but insufficient without the 8th feature
  channel (checkpoint-breaking migration, out of scope).
