# Plan: (A) seed replicates of the backward headline pair, (B) fair forward rescue

New files in `supervised_valuenet/jobs/patterns/`: `seed_headline_pair.slurm`, `fwd_train_launcher.py`,
`fwd_rescue_grid.slurm`, `fwd_rescue_select_bench.slurm`, `seeds_forward_submit.sh` (dry-run; `--go`
submits), `seeds_forward_summary.py`. Nothing submitted, no existing code touched. Balance 787 nh.

## The headline pair (from `protocol.checkpoints` of the headline JSONs)
- g16r8 (also g16r6/base450): base-trained **B1 pair** `checkpoints_backward/{policy,value}_b1.ckpt`;
  recipe `b1_retrain_chain.sh` — policy `train.policy_tf --data nn/data/combined_b1.jsonl --epochs 25`
  (cold, unseeded), value `--epochs 20 --warmup 0` warm-started from `value_v2.ckpt`, seed 11.
- g32r4: **per-config base-vocab pair** (policy version_0/epoch=1, value version_1/epoch=15): policy 30 ep
  bs 8, value 25 ep bs 2 mpg 8, both cold, default lr (= twin_retrain's g32r4 arm).
"Zero-shot" = no B2 retraining; benched `--backward-anytime --backward-b2`, 1200/k5, replay-certified.

## (A) `seed_headline_pair.slurm <g16r8|g32r4> <21|37|53>` — 1 GPU, 16 h, resumable
Same recipe with `--torch-seed S` on both nets (seed_study convention); DONE sentinels, RR_RESUME,
8-wide idempotent chunks, merge, `replay_validate`. Outputs `scaling/results/<cfg>/comparison_b2_seed<S>.json`,
`comparison_ungraded_b2_seed<S>.json`, `eval/results/final450_backward_b2_seed<S>.json` (g16r8 only, ~3 min).
Cost (sacct): g16r8 — `rr-b2rt-g16r4` 4h28m for 25/20 ep on 3.5× more records → ≤4.5 h train + ~0.5 h
bench (`rr-b2r-g16r8` 1h26m unsharded): 3 × ≤5 h × 0.125 = **≤1.9 nh**. g32r4 — twin logs: policy
~3.5 h, value 21,315 s = 5.9 h; bench ~2 h (`rr-b2r-g32r4` 4h32m unsharded): 3 × ~12 h × 0.125 =
**~4.5 nh** (+~1 if a resume link fires). Jobs: 6 + 6 afterany resume links (no-ops when DONE).

## (B) forward rescue at g16r8 (g24r4 optional)
Validation metric: `move_planner/net.py` checkpoints on **`val_policy_top1`** (max; `val_mae` also
logged) over val boards 700-899 (bench = 900-1049). Selection-by-val is possible and is what the
pipeline already does; control of record (version_44) has val_top1 0.8628, lr 1e-4, seed 7.
- `fwd_rescue_grid.slurm --array=0-8`: seeds {21,37,53} × lr {5e-5, **1e-4**, 2e-4} at g16r8 (8 ep,
  bs 128 = version_44 recipe); g24r4 {1e-4, **3e-4**, 6e-4} (8 ep, bs 64). Launcher adds run-dir,
  seed, lr, RR_RESUME resume, `BEST.json` (best val_top1 + path).
- `fwd_rescue_select_bench.slurm`: refuses unless all 9 DONE; max val_top1 → `SELECTED.json`; benches the
  winner with the control protocol (`--skip-backward --forward-ckpts`, 1200/k5, certified) →
  `scaling/results/<cfg>/comparison_forward_rescue.json`, `comparison_ungraded_forward_rescue.json`.
Cost: g16r8 train 9 × ~6 h (version_44 16:10→22:10) = 54 GPU-h = **6.75 nh**; bench 14.7 lane-h
(5,641+47,215 s) on 2 GPUs ≈ 4 h = **1 nh** ⇒ **~8 nh**, 18 tasks + 2 jobs. g24r4: 9 × ~14 h =
**15.75 nh** + bench 166 lane-h (41 min/puzzle) ≈ 25-30 h on 2 GPUs = **6-8 nh** ⇒ ~23 nh; graded-only
0.75 nh (`FWD_BENCH_SETS=graded`). Recommend deferring. **g32r4 forward excluded**: 249 lane-h per
bench (40 min/puzzle), version_45 needed 19 h for 4 epochs (grid ≈ 21 nh training alone), and §35
already shows 4× budget buys 1/24 there.

## What would change the claim
(A) any seed under the forward control (g16r8 frontier <93/184; g32r4 <2/275) kills that rung; a
bimodal draw (§44-style) forces "median of 3" headlines plus a zero-shot bistability caveat; the g16r8
graded 262-vs-261 "parity" flips with ±2 puzzles (expected). Spread of a few puzzles ⇒ defensible.
(B) rescued g16r8 frontier ≥ ~150/184 ⇒ pooled +15.8 gone; 93→~120 ⇒ claim holds with a
"best-of-9-by-val forward" caveat; winner ≈ control (val 0.86) ⇒ asymmetric-tuning objection answered.

## Unsure / flags
g24r4 forward of record ran 8 of 15 requested epochs (cause unknown); Karolina A100 assumed as fast as
the origin GPU (no forward training has run here yet); g32r4 policy time not cleanly measured — job may
need its resume link; seed varies both nets (production policy was unseeded — exact reproduction is
impossible); the g16r8 value warm start is fixed across seeds (that is the recipe); forward
`--dump-moves` replay path assumed per `replay_validate`'s docstring (`moves_seq`).
