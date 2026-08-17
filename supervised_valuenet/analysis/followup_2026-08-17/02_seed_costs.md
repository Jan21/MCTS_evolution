# Seed-replicate cost/benefit brief (2026-08-17)

Measured from job logs + sacct (Sonnet agent, this session). 1 GPU-h = 0.125 nh.

## Seed inventory (FINDINGS 67/71/72/73/74/79)

| config | exact | twin | nndeploy |
|---|---|---|---|
| g24r4 | 2 seeds | 2 seeds | — |
| g24r8 | 1 (pre-campaign) | 1 | — |
| g32r4 | 1 (pre-campaign) | 1 | 1 |

## Measured per-seed costs

| cell/arm | wall | nh | basis |
|---|---|---|---|
| g24r4 twin | 4h17–4h20 | 0.54 | jobs 4618961 / 4625100 |
| g24r4 exact | 5h00 | 0.63 | job 4629540 |
| g32r4 twin | ≈13.1 h | 1.63 | 4621819(partial)+4632986, stage-traced |
| g32r4 nndeploy (train+bench only) | 14h18 | 1.79 | job 4670119 |
| g24r8 twin | ≈30.0 h | 3.75 | 4629917+4621819, incl. 19.6 h frontier bench |
| g32r4 exact | — | ~2.0 EST | corpus-scaled, no log exists |
| g24r8 exact | — | ~3.6–4.4 EST | two scaling methods agree; no script exists (exact_replicate.slurm is hardcoded g24r4) |

Cross-check: campaign retrain subtotal reconstructed = 5.91 nh, matches
FINDINGS 74's "~6 nh retrains" line item.

## Benefit ranking

1. **g32r4 twin (1.63 nh)** — the optimality-slip claim (45.9 vs 51.7 %opt,
   a 5.8-pt gap) is WITHIN the 6.6-pt %opt spread measured between g24r4's
   two twin seeds. The study's most noise-vulnerable claim; cheapest test.
   → **LAUNCHED 2026-08-17 as job 4678373** (TWIN_SEED=21).
2. g32r4 nndeploy (1.79) — hardens the closing pipeline-equivalence claim.
3. g24r8 twin (3.75) — the collapse cell; 21-pt gap dwarfs the 3.5-pt noise
   band, but 8-robot cold value retrains are a documented bistable regime
   (§44/§56), a mechanism-level reason a draw could masquerade as collapse.
4. g24r8/g32r4 exact arms (EST) — symmetric square completion; scripts need
   generalizing first.

Items 2–4 deferred (not launched): insurance on effects that dwarf noise,
or estimates without scripts. Revisit after the corruption arms read out.
