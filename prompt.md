# Continuation prompt — 2026-07-25 — the retraining chain is queued and gated; finish Tracks 0/1, close the publishability gaps

You are picking up the Ricochet Robots planner study on **Karolina**
(IT4Innovations). Working directory:
`/scratch/project/open-37-42/petrhyner/MCTS_evolution` — this IS the working
repository (branch `b1-language-extension-scaling-study`, https origin;
commit and push directly — see "Git push" under Operational knowledge). All
gitignored payload (board sets, labels, checkpoints, lightning logs) is at
its in-tree paths under `supervised_valuenet/`. The machine-level operations
hook auto-loads from `~/.claude/CLAUDE.md` (partitions, accounting, storage,
pitfalls) — trust it. The previous handoff (2026-07-23) is in git history at
this path; this document supersedes it.

## Read first (all under `supervised_valuenet/` unless noted)
1. `PRIMER.md` — the study in plain language.
2. `FINDINGS.md` — THE single running log; items 13–20 are the
   language-extension → gating arc. **Every new result appends here first**,
   numbers only from result JSONs, source file named.
3. `analysis/publishability.md` — the referee-eyed assessment (2026-07-23);
   its objection register drives much of the remaining work below.
4. `jobs/patterns/` — Slurm recipes. **Only four are current, tested-this-
   session recipes**: `b2_labels_one.slurm`, `b2_retrain_one.slurm`,
   `b2_vocab_gate_qgpu.slurm`, `fwd_budget_probe.slurm` (plus the smoke).
   The rest (`lanes_one`, `b2_bench450_qgpu`, `b2_rows_one`,
   `b2row_g24u_shards`, …) are HISTORICAL: repointed to this repo on
   2026-07-25 but they still use the PRE-retrain nets and the old output
   filenames — author fresh scripts for Track 1 (see below), using them only
   as structural templates (sharding/idempotence/merge idiom).
5. `eval/report_data.py` — the report generator's config lists define the
   EXACT filenames pending results must use (listed below; do not improvise
   names). Note `"base": True` on the g16r4 entry — see the g16r4=base
   warning below.

## Where the study stands (all measured, committed, pushed)

Everything in the 2026-07-23 handoff still holds (ladder fully measured;
frontier verdict backward 80.6/88.6/55.7/71.3% vs forward 48.5/50.5/15.2/0.7%
at 1200 expansions / top-5; base ceiling 90.7→97.6→99.6%; the trained-vs-
permitted gap is the mission). Since then, this session (2026-07-23..25):

- **Track 0a DONE.** Both Python labelers emit the full B2 vocabulary:
  `nn/generate.py --vocab {base,b1,b2}` (`make_solver`), mirrored in
  `scaling/backward_label.py` and `rust_datagen/pyref/dump_decisions.py`.
  The rollout mirrors `AStar._expand` exactly (reference-helper injection +
  `_apply(by_reference=)`), so labeled candidate sets equal the solver's
  own. **Scoping decision (flagged to owner, unchallenged): park repairs are
  NOT labeled** — matching B1 precedent; they are deterministic
  realization-failure repairs the nets never rank. Base-vocab default path
  proven record-set-identical to the pre-change driver (order jitter within
  equal-score ties is the adjudicated ADOPTION.md set-iteration class;
  HEAD-vs-HEAD control reproduces it).
- **Track 0b DONE.** The Rust engine (`rust_datagen/`) speaks B2: transient
  pair enumeration, the three by-reference shapes in `apply`
  (shared-support / target-as-stopper / relocation), `byref` plan edges,
  `vocab` on work items. Full cargo suite + original golden gates green
  (default path unchanged). **Formally gated (job ran 2026-07-24, adopted
  and logged 2026-07-25): 4 legs, 68 pinned decisions, 894 labels replayed
  Python-vs-Rust, ZERO diffs** (job 4591708; corpora archived
  `rust_datagen/golden/b2gate/`; FINDINGS item 19).
- **Setup validated:** smoke + shard-equivalence job scored 19/20 both
  planners, sharded-vs-unsharded rows byte-equal (job 4591696).
- **Track 2 WRITTEN** (`analysis/publishability.md`): synthesis of three
  independent referee simulations; verdict "major revisions as worded";
  costed objection register (free/cheap/moderate/structural tiers); venue
  advice AIJ/JAIR over Neurocomputing. **Krohn–Rhodes section is a
  deliberate placeholder — owner input required first (see Owner-pending).**
- **Track 3 REBUILT** (twice): `eval/build_report.py` + eight `report_*`
  modules → one self-contained offline `eval/results/report.html`, six
  guided tabs, **477 machine-checked numbers + 47 assertions**, light/dark
  validated, plan-structure diagrams embedded. Tab 5 "Is it fair?" is the
  owner-REQUIRED compute-accounting section (matched-unit statement,
  unmetered-work ledger, budget-curve small multiples, same-machine
  wall-clock, live hooks for pending measurements). Regenerate:
  `PYTHONPATH=. python3 -m eval.build_report` (needs `ml bzip2/1.0.8-GCCcore-13.2.0`
  besides the usual env). **Local file only — NEVER publish anywhere
  (owner's standing order).**
- **Budget curves ≤1200 reconstructed at zero compute** (FINDINGS §20):
  `eval/budget_curves_from_rows.py` → `eval/results/budget_curves_by_rung.json`
  (42 curves, endpoints verified against stored aggregates). Two honest
  readings recorded: 32×32 frontier backward 43.6% at budget 100 vs forward
  0.0% at 400; and the 16×16 frontier forward curves STILL CLIMB at the cap
  (+4–6 pts/last 200 steps) while 24×24/8 and 32×32 are flat — saturation is
  proven on the grid axis only, hence the probe jobs below.
- **FINDINGS §18b errata** (prose-vs-JSON, all verified): base checked
  regret is 2.14 not 2.15; self-play arc 0.724→0.222 per the cited files;
  the oracle-death curve omits its worst point (24×24/8 = 64.2%) and
  conflates two axes.

## LIVE SLURM STATE — 12 queued jobs, do not resubmit blindly

All on `qgpu` (house rule: qgpu ONLY, never qgpu_free/preempt). Charge =
elapsed × GPUs/8 node-hours. ~896 node-hours remain; total spent this
session ≈ 0.14. Check with `squeue --me`; job outputs land under
`/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/<family>/`.

**Preconditions & session facts:**
- **No watchers are running.** The previous session's background monitors
  died with it. Arm your own (pattern below) before doing anything else.
- The Rust engine binary (`rust_datagen/target/release/datagen`) is present
  and selftest-verified as of 2026-07-25 — but `target/` is gitignored and
  can vanish (fresh clone, cleanup). The label jobs do NOT build it and
  compute nodes have no internet: before the `rr-b2lab-*` jobs run, confirm
  `./rust_datagen/target/release/datagen selftest` passes on the login node;
  rebuild with `cargo build --release` (~36 s) if missing. A missing binary
  makes each label job FAIL fast (clear Python error), which cancels its
  `afterok` retrain child — recoverable but wasteful.
- **`g16r4` IS the base configuration** (16×16/4, bench450, boards
  2400–2549; `scaling/README.md`, `report_data.py:"base": True`). The
  retrain chain covers 5 configs INCLUDING g16r4; the Track 1 per-config
  list below has 4 because base is handled by the separate bench450 command.
  Misreading this stalls the headline row — see Banking step 2.

| jobs | what | depends on | expected |
|---|---|---|---|
| 4591709..13 `rr-b2lab-<cfg>` | B2 label datasets via the gated Rust engine → `scaling/data/<cfg>/backward_b2.rust.jsonl` (+ rust_work manifests recording engine+vocab) | gate 4591708 (SATISFIED — now waiting on queue priority only) | ~0.5–2 nh each; base ≈ 2112 boards, others 1050 |
| 4591714..18 `rr-b2rt-<cfg>` | per-config retrains: policy COLD then value WARM-started (`jobs/patterns/b2_retrain_one.slurm` holds the exact warm-start ckpts, batch caps, low-lr rules) → runs in `scaling/runs/<cfg>/backward-{policy,value}-b2/` | each on its own label job | ~1–2 nh each |
| 4592278 `rr-fwdprobe-g24r8` | forward-only, budget 6000, 32-instance frontier subsample (seed-1 shuffle) → `scaling/results/g24r8/forward_probe_e6000.json` | none | ~3–4 nh |
| 4592279 `rr-fwdprobe-g32r4` | forward-only, budget 4800, 24 instances → `scaling/results/g32r4/forward_probe_e4800.json` | none | ~3–4 nh |

Watch pattern: poll `sacct -j <id> --format=State --noheader -X` in a
background until-loop; the per-job DONE markers are grep-able
(`B2LABELS <cfg> DONE`, `B2RETRAIN <cfg> DONE`, `FWDPROBE <cfg> ... DONE`).
The probe jobs are chunked and idempotent — a walltime kill loses nothing;
resubmit the same command to resume. If a job dies within seconds, check
`scontrol show reservations` (daily 10:00–18:00 cooling windows kill jobs
scheduled onto reserved nodes) and resubmit.

**When a label job lands (QC immediately — note the race):** the `afterok`
dependency releases the retrain on label-job COMPLETION, not on your QC.
The priority queue usually gives you hours of margin, but if a retrain
starts before QC and QC then fails, `scancel` the running retrain (its
node-hours are lost), fix, regenerate labels, and resubmit that config's
retrain manually (`sbatch -J rr-b2rt-<cfg> jobs/patterns/b2_retrain_one.slurm <cfg>`).
1. `wc -l scaling/data/<cfg>/backward_b2.rust.jsonl` (expect order 50k–150k
   records; base larger).
2. By-reference share: candidates whose `cand_helper` position is not any
   robot's start position — expect roughly 5–20% (base measured 13%).
3. Manifest sanity: `scaling/data/<cfg>/rust_work/backward_b2.manifest.json`
   has `engine: rust, vocab: b2`, plausible kept/attempts ratios.
4. If QC fails: `scancel` the config's retrain job, fix, regenerate.
   If QC passes: let the retrain run.

**When a retrain lands (banking — deliberately manual):**
1. Inspect `scaling/runs/<cfg>/backward-{policy,value}-b2/lightning_logs/`
   — identify runs by hparams+mtime, NEVER by version number (resubmissions
   accumulate version_N dirs). Mechanism: the ModelCheckpoint callback
   saves the top-1 checkpoint by `val_regret` (min), so the single
   `epoch=*-step=*.ckpt` under the chosen run's `checkpoints/` IS the best
   one; confirm the metric trajectory in that run's `metrics.csv`. Check
   `val_regret` trended down and the best value beats or approaches the
   config's old-vocab run.
2. Bank: **g16r4 (= base) → copy its best policy/value ckpts to
   `checkpoints_backward/policy_b2.ckpt` and `value_b2.ckpt`** (the
   Track 1 base-450 command consumes exactly these paths); the other four
   configs → ckpt files stay in place, referenced by full path in eval
   commands.
3. Value-net warm-start is load-bearing: if a value retrain looks unstable
   (val top-1 near random from epoch 1 — the known signature), do NOT bank;
   rerun with `--torch-seed` varied. Policy retrains are stable.
4. Copy banked ckpts + the new label datasets to `/mnt/proj1/open-37-42/`
   (scratch purges untouched files after 90 days).

## Track 1 — the definitive head-to-heads (the next real milestone)

With banked B2 nets, rerun the BACKWARD rows only (all forward control rows
stand). **Filenames are fixed by `eval/report_data.py` — use exactly:**
- base 450: `eval/results/final450_backward_b2_retrained.json`
  — flags: `--instances eval/data/bench450.jsonl --expansions 1200 --k 5
  --backward-policy checkpoints_backward/policy_b2.ckpt
  --backward-value checkpoints_backward/value_b2.ckpt
  --backward-anytime --backward-b2 --forward-ckpts "" --device cpu`
- per config graded: `scaling/results/<cfg>/comparison_b2retrained.json`
  (instances `scaling/data/<cfg>/bench.solved.jsonl`)
- per config frontier: `scaling/results/<cfg>/comparison_ungraded_b2retrained.json`
  (instances `bench.unsolved.jsonl`)
- configs: g16r6, g16r8, g24r8, g32r4 — each with ITS OWN retrained nets
  (this deliberately removes §17's nets-vs-language confound; the 16×16
  rungs previously borrowed base-trained nets).
Execution: **author a fresh Slurm script** — the historical row/bench
patterns (`lanes_one`, `b2_rows_one`, `b2_bench450_qgpu`) use the
pre-retrain nets and old output names and are NOT runnable for Track 1
as-is; copy only their structure: shard the instance file, small
idempotent chunks (skip-if-output-exists, tmp+mv atomic writes), 24 h
walltimes, merge with `eval/merge_compare_shards.py` (row-identity vs
unsharded proven; the merger refuses protocol mismatches, so keep flags
identical across chunks, and shards must be in benchmark order).
Backward-only rows are cheap: ~5–15 node-hours total. State the projection
in chat before submitting (house rule). Then: FINDINGS §21+ (append,
numbers only from the JSONs), rebuild the report (`-m eval.build_report` —
the pending rows fill automatically), commit, push.

**Fold into the same reruns (owner-approved compute-accounting directive):**
- The instrumentation branch (see UNCOMMITTED WORK below): run the rerun
  lanes with the new counters active and `--dump-moves`; write the per-rung
  aggregate counters to `eval/results/compute_accounting.json` (the report's
  hook renders it automatically); run `eval/replay_validate.py` over every
  result file produced with `--dump-moves` and record the pass rate in
  FINDINGS — this is the independent certification of claimed solves.
- The missing 2×2 cell (publishability objection 1.3): base-B1 nets with
  OLD-vocabulary search on the g16r6/g16r8 pinned sets (backward-only,
  <1 nh) — isolates language effect at fixed nets. Output names (my
  choice, FINDINGS-only — the report has NO slot for these, deliberately):
  `scaling/results/<cfg>/comparison_basenets_oldvocab.json` and
  `comparison_ungraded_basenets_oldvocab.json`.

## UNCOMMITTED WORK IN THE TREE — handle first

If `git status` shows modified `eval/compare.py`, `eval/realize.py` and a
new `eval/replay_validate.py`, that is the compute-accounting
instrumentation (counters for NN calls by head + physics-slide calls split
by prefix-check/realization/park-repair + free exact fixes + rejected pops;
`--dump-moves`; the standalone validator that replays move sequences
through ONLY `simulate.py:slide`). Verification status at handoff time:
**A/B proof IDENTICAL** (20-instance bench slice; after-JSON stripped of
only the new `accounting`/`moves` keys and wall-clock fields equals
before-JSON byte-level), replay validator 19/19 on solved rows, prefix-
check lane 10/10; the forward-lane check was finishing. **Commit rule: only
with the IDENTICAL verdict standing; to re-verify, the "before" side is
`git show <pre-instrumentation-commit>:supervised_valuenet/eval/compare.py`
(note: repo-root-relative path) or a fresh checkout.** If the tree is
already clean, the instrumentation was committed — check
`git log --oneline -5`. If verification cannot be reproduced and the edits
are uncommitted, `git checkout -- supervised_valuenet/eval/compare.py
supervised_valuenet/eval/realize.py` and re-implement from
publishability.md objections 1.1/1.4. Nothing queued depends on these
edits; all queued jobs use committed code only.

## Remaining work, in priority order, with justification

1. **Shepherd the queued chain** (labels → QC → retrains → banking). This
   is Track 0c/0d; everything else in Track 1 waits on it. Zero new
   decisions needed — recipes above.
2. **Resolve the instrumentation branch** (above), then **Track 1 reruns**
   with accounting + dump-moves + the 2×2 cell. Justification: the
   definitive rows are the mission's headline; the accounting column and
   independent replay certification close the two reject-level
   methodology objections (publishability 1.1, 1.4) at ~zero marginal cost.
3. **When the probe jobs land:** append a FINDINGS entry reading the
   result plainly (does forward's frontier solve rate climb at 4–5×
   budget?); rebuild the report (hooks auto-render). Justification: decides
   whether "the collapse is not a cap artifact" holds on both axes
   (currently proven only on the grid axis — FINDINGS §20).
4. **Publishability free tier not yet executed** (analysis-only,
   login-node CPU; publishability.md §2 Tier 0): paired McNemar +
   board-clustered bootstrap CIs for every cell (the "first graded win" is
   a one-puzzle margin — must be reframed); pooled graded+frontier union
   per rung; frontier hardness-stratification; d_star placeholder
   sentinel; train/test provenance statements + wall-layout dedup audit;
   two-sided tuning ledger; per-config data-budget table. Justification:
   cheapest referee-proofing steps, none needs owner input. One exception
   mis-bucketed as free in earlier notes: the **Rust 10× oracle probe**
   over the g24r8/g32r4 frontier sets (regret on the recoverable stratum
   + threshold re-slice) is real CPU work — run it inside a small qgpu job
   with the built engine (~minutes–hours; well under 1 nh).
5. **Report iteration 3** (resume or respawn the report agent) once
   retrained rows + probes land: the before/after-retraining dumbbell
   chart (hooks exist), browser-render QA, `scope` attrs on tables.
6. **Owner-gated / owner-decision items — do NOT start without input:**
   - **Krohn–Rhodes section** of publishability.md: blocked on the owner's
     notes/collaborator material (asked repeatedly, no answer yet). The
     drafting rule agreed: map states/generators/wreath-product levels
     explicitly; mark proven vs speculative; no fabricated connections.
   - Compute-tier experiments awaiting a go: seed study (~15–20 nh), lr
     sweep (~7), forward self-play control at g16r8 (~4), k-sweep (~2),
     g24r4 missing cells (~4), kSubS-style baseline (structural), second
     domain (structural). Recommendations and costs in publishability.md §5.
   - Deleting the retired `karolina_bundle/` (its launcher patterns are now
     committed in-repo at `jobs/launcher_patterns/`; pristine archive on
     `/mnt/proj1`). Owner has not answered — do not delete.

## Operational knowledge earned this session (trust it; each item cost time)

- **Walltime vs backfill:** the queue is saturated (~190 pending). 24 h
  requests sat ~2.5 days; the owner explicitly approved 2 h (smoke) and 3 h
  (gate) for provably-short validation jobs and both then started within
  minutes-to-hours via backfill. The 24/48 h floor REMAINS the standing
  order for real (chunked) runs — the trim exception was owner-granted for
  validation-class jobs only. Dependency-chain everything at submit time
  (`--dependency=afterok:<id>`) so queue priority accrues while
  prerequisites run; a cancelled prerequisite ID invalidates dependents —
  resubmit the whole chain if you must cancel a parent.
- **Git push:** authenticates via the VS Code askpass bridge; a stale
  socket env yields "Authentication failed". Fix:
  `VSCODE_GIT_IPC_HANDLE=$(ls -t /tmp/vscode-git-*.sock | while read s; do [ -O "$s" ] && echo "$s" && break; done) git push`.
  If the owner's VS Code is fully disconnected, pushing is impossible —
  queue commits locally and say so. (Also in the agent memory dir.)
- **B2 rollouts wander** (plateau/relocation loops): legitimate base
  rollouts finish in ~60–1,700 solver iterations; a wanderer's
  per-iteration cost explodes with plan size (one failed to exhaust 200k
  iters in 8 min). Hence per-config `--budget-iters` (50k/100k/200k) and
  the bridge's wave protocol (both committed). Never run plain
  `nn.generate --vocab b2` without a timeout wrapper.
- The Rust engine binary is NOT committed: `cd rust_datagen && cargo build
  --release` (system cargo 1.92 works; ~36 s) before any bridge run.
- The eval driver writes JSON only at lane end — killed lanes bank nothing;
  chunk everything (the probe/label patterns already do).
- `ml Python/3.11.5-GCCcore-13.2.0` BEFORE venv activation (libbz2), in
  EVERY shell including background ones; report generator also wants
  `ml bzip2/1.0.8-GCCcore-13.2.0`.
- Foreground `sleep` is blocked in this harness; the Bash tool default
  timeout (2 min) kills long foreground commands — background them.
- `merge_compare_shards.py` refuses protocol mismatches (expansions, k,
  device, checkpoints) and requires shards in benchmark order.
- Evals on CPU inside GPU jobs (`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8`,
  `PYTHONUNBUFFERED=1`); never mix torch builds within a comparison.
- `scancel` by job id, never `pkill -f`; kill local processes by explicit
  PID from `ps`.

## House rules (owner-set; unchanged; violations have been reverted before)

- Playable-moves scoring only; never compare abstract plan cost to move
  optima; beyond-oracle sets have placeholder optima (only solve
  rate/steps/time/length meaningful; `d_star: 0` is a live footgun until
  the sentinel fix lands).
- Subgoal methodology frozen: fixes/tuning/new candidate types inside
  propose allowed; redesigns and raw-move fallbacks are not.
- No exact solver at solve time. Vocabulary separation absolute — never mix
  vocabularies in one dataset; name datasets/runs/checkpoints by vocabulary.
- Slurm: account `open-37-42` (lowercase), `qgpu` only, 24 h walltimes for
  real runs (see backfill note for the validation-job exception the owner
  granted), state projected node-hours in chat before big submissions.
- FINDINGS.md is the single log; results append there first, sources named.
- Commit + push often; after milestones copy new checkpoints/results to
  `/mnt/proj1/open-37-42/` (90-day scratch purge).
- The report is local-only. Never publish it or any artifact anywhere.
- Session conduct: never block the CLI with waiting calls — arm background
  watchers, give a one-line status, end the turn; lead status answers with
  the status; report failures honestly the moment they happen.

## Owner-pending questions (ask again at the first natural opportunity)

1. **Krohn–Rhodes:** what notes/collaborator material exists? Who is the
   theory collaborator? Intended depth (motivating analogy / formal section
   / precise conjecture)? Blocks publishability.md §6 only.
2. **Compute-tier experiment go/no-go** (list + costs above and in
   publishability.md §5).
3. **`karolina_bundle/` deletion** (safe now, but owner must say yes).
