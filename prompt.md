# Continuation prompt — retraining, the definitive head-to-head, publishability, and the report

You are picking up the Ricochet Robots planner study on **Karolina** (IT4Innovations).
Working directory: `/scratch/project/open-37-42/petrhyner/MCTS_evolution` — this IS
the working repository now (branch `b1-language-extension-scaling-study`, https origin
with working credentials: commit and `git push` directly). All gitignored payload
(board sets, labels, checkpoints, lightning logs) has been placed at its in-tree paths
under `supervised_valuenet/`. A machine-level operations hook is auto-loaded from
`~/.claude/CLAUDE.md` (partitions, accounting, storage, pitfalls) — trust it.
The previous working copy at `.../karolina_bundle/` is retired (same data via
hardlinks; pristine archive at `/mnt/proj1/open-37-42/karolina_bundle.tar.gz`).

## Read first (all under `supervised_valuenet/`)
1. `PRIMER.md` — the study in plain language.
2. `SOURCE_OF_TRUTH.md` — canonical audit (its §5 ceiling story is since superseded
   by FINDINGS §§13/16 — the ceiling was lifted twice).
3. `FINDINGS.md` — THE single running log. §§13–18 are the language-extension and
   scaling-completion arc; **every new result appends here first**.
4. `analysis/b1_extension_notes.md` + `analysis/b1_design.md` — how the two language
   extensions (B1, B2) work and were verified.
5. `eval/results/report.html` + `eval/results/plan_structures.html` — the current
   deliverable pages (regenerate via `PYTHONPATH=. python3 -m eval.build_report` and
   `-m eval.build_plan_viz`).

## Where the study stands (2026-07-23, all measured, committed, pushed)
- Backward planner repaired (53.1% → 89.1%), language extended twice: measured
  expressiveness ceiling 90.7% → 97.6% (B1) → **99.6%** (B2); **nothing is proven
  impossible anywhere** (base or 6-robot). B2 = supports-by-reference + generalized
  step-asides, all opt-in (`--backward-b2`, probe `--b2`), zero-regression proven.
- The full scaling ladder is measured at matched budgets (1200 expansions, top-5,
  playable-moves scoring). Frontier (beyond-oracle) ladder, backward-full-language vs
  properly-trained forward: **80.6/88.6/55.7/71.3% vs 48.5/50.5/15.2/0.7%** (6r, 8r,
  24×24/8, 32×32). First backward graded-set wins: 8 robots (98.5% vs 98.1% at 7×
  fewer steps) and 32×32 (84.0/88.0% vs 76.0% at 100×+ less time).
- KEY LIMIT (your mission): the current networks rank the new plan-language
  candidate types **zero-shot** — they were never trained on them. Base: 95.6%
  achieved vs 99.6% permitted. At scale the zero-shot gains are small. The language
  has outrun the training.

## Mission (in order)

### Track 0 — retrain the backward networks on the extended vocabulary
The pipeline, gated at every step:
1. **Extend the Python labeler** (`nn/` label driver; B1 labels were Python-made:
   `nn/data/combined_b1.jsonl`, 108,902 records) to also emit B2 candidates
   (by-reference supports, generalized parks). The search machinery already
   generates them (`skeleton/astar.py`, `AStar(by_reference=True)`,
   `park_repairs(pairwise=, multi_slide=)`); labeling means recording them as
   candidates with exact committed costs, same as B1 did for transient stoppers.
2. **Teach `rust_datagen` the extended vocabulary** (it currently knows only the
   original one). Gate exactly per `rust_datagen/VERIFICATION.md`'s pattern: replay
   pinned decision contexts against the Python reference, zero label differences
   required, then adopt per `ADOPTION.md`. Python labeling at 24×24/32×32 is
   impractical (that is why Rust exists) — do NOT try to brute-force big-board
   labels in Python.
3. **Generate extended-vocabulary labels per config** (base, g16r6, g16r8, g24r8,
   g32r4) with the gated Rust engine. NEVER mix vocabularies in one dataset; name
   every dataset/run/checkpoint by vocabulary (house rule; cost-to-go is
   vocabulary-relative).
4. **Retrain per config**: proposal net cold, value net **warm-started** from the
   banked predecessor (`checkpoints_backward/value_b1.ckpt` for base; per-config
   value ckpts under `scaling/runs/<cfg>/` — cold value retrains are proven
   seed-unstable). Use the low-lr launcher patterns for anything ≥6 robots
   (`launcher_patterns` history is in the repo docs; lr 1e-4 with the config's
   SPLITS rebinding). Batch caps on 40 GB A100: 24×24 ≤32–64, 32×32 value batch 2 /
   group 8. Bank best ckpts as `checkpoints_backward/{policy,value}_b2.ckpt` (base)
   and `scaling/runs/<cfg>/backward-{policy,value}-b2/...`.
5. Training jobs: `-p qgpu --gpus 1`, 24 h walltime, one GPU per job.

### Track 1 — the definitive head-to-head, same budgets
With retrained nets, rerun the BACKWARD rows only (the forward control rows exist and
stand): base 450 (`--backward-anytime --backward-b2`, k=5, 1200), then per config
graded + beyond-oracle. Compare against the §9/§17 rows. The proven execution
pattern for long CPU evals is in `supervised_valuenet/jobs/patterns/` (shard the
instance file, many small `--gpus 1..2` jobs with 24 h walltimes, idempotent chunk
outputs, merge with `eval/merge_compare_shards.py` — row-identity vs unsharded was
proven in the smoke job). Backward-only rows are cheap (~5–15 node-hours total);
budget estimate before submitting anything, report it to the owner in chat
(house rule), and never let a chunk job run with a walltime shorter than double its
estimate — killed chunks waste the hours already burned.
Append every result to FINDINGS §19+ (numbers only from result JSONs, source file
named), rebuild the report, commit, push.

### Track 2 — publishability analysis (target: Q1, e.g. Neurocomputing)
Write `analysis/publishability.md`: an honest referee-eyed assessment.
- Contributions to argue: the playable-moves evaluation methodology (paper plans vs
  played plans — the 99.6%→53.1% correction); the measured-then-lifted
  expressiveness ceiling with zero-regression verification discipline (downward
  refinement, Bacchus & Yang 1994); the language-extension mechanism (temporary
  supports, supports-by-reference); the full-ladder empirical crossover (both axes)
  against properly trained move-level baselines at matched budgets.
- Gaps a Q1 referee will hit: single domain; single seed per training run (no error
  bars); no comparison to published learned-subgoal-search baselines (kSubS,
  Kujanpää et al. — cited in `analysis/pass5_research.md`); solution-quality gap on
  the frontier; compute normalization argument (expansions vs wall-clock).
  Propose the cheapest experiment closing each gap.
- **Krohn–Rhodes framing — get the owner's input before writing this section.** The
  owner intends to tie the work to the Krohn–Rhodes theorem (cascade decomposition
  of finite transformation semigroups) — presumably: subgoal plans as a hierarchical
  decomposition of the puzzle's transformation monoid, plan-language expressiveness
  as the reachable subgroup/cascade structure. This is a THEORY-BUILDING task, not a
  lookup: ask the owner what prior notes/collaborator material exists, draft the
  mapping explicitly (states, generators, what the wreath-product levels correspond
  to), and mark clearly what is proven vs. speculative. Do not fabricate a
  connection; a precise "here is what would need to be shown" section is worth more
  than hand-waving.

### Track 3 — the report, rebuilt from scratch by a dedicated agent
Spawn a dedicated parallel agent (Fable) whose SOLE job is the HTML, and keep it
iterating until it is genuinely excellent:
- **One self-contained file** (`eval/results/report.html`), HTML+CSS-first, inline
  vanilla JS permitted for tabs/interaction only — NO external assets, fonts, CDNs,
  or frameworks; must render offline. Produced by a GENERATOR
  (`eval/build_report.py` may be rewritten from scratch) that reads every number
  from the result JSONs — nothing hardcoded — and self-verifies (the current
  generator's check-table pattern: every rendered headline cross-checked against
  its source aggregate; keep or improve it).
- Design brief: professional, academic-grade; tabbed, with a GUIDED reading path
  (a reader who knows nothing is led: what the puzzle is → the two ideas → the
  headline table → how the plans changed (keep the inline plan-structure diagrams)
  → the scaling story → methods/provenance). Publication-quality inline-SVG charts
  (proper axes, units, captions, consistent palette, light+dark). Plain language
  everywhere, jargon only in the provenance appendix. Tables: clearly named
  experiments (never internal run names), winner emphasis only where both sides are
  measured, honest pending/caveat tags.
- The plan-structure diagrams and their extractor
  (`eval/plan_viz_core.py`, `eval/build_plan_viz.py`,
  `eval/results/plan_structures_data.json`) are assets to reuse.
- **NO publishing to claude.ai artifacts or anywhere else — local files only**
  (owner's standing order).

## House rules (owner-set; violations have been reverted before)
- Playable-moves scoring only; never compare abstract plan cost to move optima;
  beyond-oracle sets have placeholder optima (only solve rate/steps/time/length are
  meaningful there).
- The subgoal methodology stays frozen: fixes, tuning, and new candidate types
  inside the propose step are allowed; redesigns and raw-move fallbacks are not.
- No exact solver at solve time. Vocabulary separation absolute.
- Slurm: account `open-37-42` (lowercase), **`qgpu` partition ONLY** (never
  `qgpu_free`/`qgpu_preempt` — owner's explicit order), 24 h walltimes on
  everything, `scancel` never `pkill`. Charge = elapsed × GPUs/8 node-hours; ~895
  node-hours remain; state projected cost before big runs.
- Env: `ml Python/3.11.5-GCCcore-13.2.0` BEFORE `source
  /scratch/project/open-37-42/petrhyner/venv/bin/activate` (libbz2 comes from the
  module). Evals on CPU inside GPU jobs (`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8`),
  `PYTHONUNBUFFERED=1`. Never mix torch builds within a comparison (venv is torch
  2.13; backward step counts shift across builds via float tie-breaks).
- Session conduct: never block the CLI with waiting calls — arm background
  watchers/monitors, give a one-line status, end the turn; lead every status answer
  with the status; report failures honestly the moment they happen.
- Commit + push often (this checkout pushes directly); `/scratch` purges files
  untouched 90 days — after milestones copy new checkpoints/results to
  `/mnt/proj1/open-37-42/`.

## Known pitfalls (each has already bitten this project once)
1. Value-net cold retrains are seed-unstable — always warm-start.
2. Forward-style training diverges at ≥6 robots on stock lr (not your lane, but the
   pattern generalizes: watch validation top-1 from epoch 1).
3. Per-config board dirs (`environments_<cfg>/`, `RR_ENV_DIR`) must never be
   crossed — silent geometry mismatch.
4. `lightning_logs` version numbers are shared across run types — identify runs by
   hparams+mtime, never by number.
5. Long CPU evals print nothing for hours (buffering) — check `sstat`/CPU% before
   calling anything hung.
6. Probe artifacts at 6 robots join on `(tag, idx)`, not `idx`.
7. Daily 10:00–18:00 "cooling" maintenance reservations kill jobs scheduled onto
   reserved nodes at the boundary (signal 15 within seconds) — if a job dies
   instantly, check the node against `scontrol show reservations` and resubmit
   (`--begin=18:00:00` or `--exclude=<node>`).
8. The eval driver writes its JSON only at the END of a lane — a lane killed at
   walltime banks nothing; that is why everything runs as small idempotent chunks.

## Ask the owner before/while starting
1. Krohn–Rhodes: what notes/material exists? Who is the theory collaborator?
2. Track-0 budget green-light: label generation + 5×2 retrains + reruns ≈ 30–60
   node-hours projected — confirm.
3. Whether the retired `karolina_bundle/` directory may be deleted (its payload now
   lives here via hardlinks; the pristine archive is on `/mnt/proj1`).
