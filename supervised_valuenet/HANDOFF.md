# HANDOFF — continue this work (single self-sufficient document)

Written 2026-07-18. Give this document to the continuing agent as its prompt. Working
dir: `/mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet` (branch
`supervised-full-planner`), run modules with `PYTHONPATH=.`, conda env `ph_main`.

## 0. Read first, in this order
1. `PRIMER.md` — the whole project in plain language (what the puzzle is, the two
   planners, the honesty rules, the pipelines). Matches the required writing register.
2. `SOURCE_OF_TRUTH.md` — canonical audit; wins any disagreement (its §5 ceiling
   mechanism is now superseded by the B1 results below, but its protocol sections hold).
3. `FINDINGS.md` — THE single running log. §§1–13 = everything established, each claim
   with its source file. §9 is the ladder scoreboard with "–" for pending cells.
   **Every new result you produce appends here first**, then the web page.
4. `COMPARISON.md` — the measurement protocol (identical instances, 1200-step cap,
   top-5, playable-moves scoring).
5. `analysis/b1_extension_notes.md` + `analysis/b1_design.md` — the plan-language
   extension ("B1") that lifted the ceiling; `analysis/failure_families.md` — the
   7-family failure taxonomy with per-family B1 coverage.

## 1. The goal (owner's words, unchanged)
Prove — or honestly test — that the backward/subgoal planner is more efficient and
more useful than the forward/move-by-move planner, especially as puzzles scale (more
robots, bigger boards). No tricks; the subgoal methodology itself stays frozen (bug
fixes, tuning, evaluation, and new candidate types inside the propose step are allowed;
redesigns and raw-move fallbacks are not). Final deliverable: the comparison web page
(claude.ai artifact `785268c8-43ad-4f45-a5af-3e09c815d910`; regenerate content with
`PYTHONPATH=. python3 -m eval.build_report` → `eval/results/report.html`, then publish
the artifact to THAT SAME URL). A failure-example gallery also exists:
`eval/results/failure_gallery.html` (generator `analysis/failure_gallery/build_gallery.py`).

## 2. Standing rules (owner-set; violations were reverted before)
- **Two GPUs maximum, only ever free ones.** Check `nvidia-smi` before any GPU launch;
  claim via `/tmp/claude-1010/gpu_claims/` lock dirs (`claim_gpu.sh` pattern — global
  cap 2 enforced in the script; scratchpad path below). Evaluations run on CPU
  (`OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""`); GPUs are for training. Keep a GPU
  busy whenever meaningful work exists.
- **Playable-moves scoring only**: a puzzle is solved iff the plan plays out legally
  (`eval/realize.py:strict_moves`). Never compare abstract plan cost to move optima.
  Beyond-oracle result files carry placeholder optima — only solve rate / steps /
  time / plan length are meaningful there.
- **Plain language, neutral tone, no jargon** in chat and every document; define any
  technical term at first use. No attributions.
- **Vocabulary discipline**: B1-language labels (`nn/data/combined_b1.jsonl`, 108,902
  records) must NEVER be mixed with old-vocabulary labels (cost-to-go is
  vocabulary-relative). Name datasets/runs by vocabulary; keep `subgoal_selfplay/arms_index.json` current.
- Everything significant → `FINDINGS.md` with the source file named.

## 3. State: DONE (all verified, logged in FINDINGS)
- Backward planner repaired 53.1% → 89.1% playable at ~7 search steps (4 fixes +
  in-search playability check, all zero-regression A/B'd).
- Ceiling measured (90.7% base / 84.2% at 6 robots), proven cap-independent, then
  **lifted by the B1 language extension to 97.6% base / ≥96.9% at 6 robots**
  (structural failures 42→11 and 71→2; 249/249 realizer A/B identical; default paths
  byte-identical). B1 = wall-less "transient" stoppers + costed slide-aside "parks"
  (opt-in: `heuristics.propose_b1`, `--b1` on the probe).
- Failure taxonomy: 7 families, gallery + doc shipped; forward planner solved 42/42
  base structural failures (33/71 at 6 robots) — the ceiling was purely linguistic.
- Oracle-mortality curve: 0% → 29.8% (r6) → 40.9% (r8) → 48.4% (24×24) → 61.1%
  (32×32) at practical budget; at 10× budget 41% of r8 failures crack, 59% don't.
- Scaling rows measured (see FINDINGS §9 scoreboard): forward — properly trained,
  after control retrains for its recurring robot-axis training divergence — wins the
  oracle-gradable sets (98–99%); backward wins beyond the oracle at 6 robots
  (52.2% vs 48.5% at 2.8× fewer steps) and is 7×/35× cheaper at 24×24.
- Self-play for backward works (playable-by-construction generation; matches
  supervised solve rate, better regret, fewer steps).
- Data generation ported to Rust (`rust_datagen/`, adopted per its ADOPTION.md,
  50–180× faster). **The Rust crate does NOT know the B1 vocabulary yet** — B1 labels
  use the Python driver.

## 4. State: RUNNING right now (detached nohup jobs — they survive; check first)
Scratchpad (session-specific but files persist until reboot):
`SCRATCH=/tmp/claude-1010/-mnt-raid-data-Hyner-Petr-MCTS-evolution/6bc085eb-eb79-4a20-97d5-e670e4d4c89c/scratchpad`
- **B1 retraining chain** (`$SCRATCH/b1_retrain_chain.sh`, log
  `$SCRATCH/b1_retrain_chain.log`): waits for a free GPU claim, then trains the
  proposal net cold (25 epochs, log `$SCRATCH/b1_policy_train.log`) and the value net
  **warm-started from `checkpoints_backward/value_v2.ckpt`** (20 epochs, log
  `$SCRATCH/b1_value_train.log`) on `nn/data/combined_b1.jsonl`. Marker on finish:
  `B1 RETRAIN DONE`. (Warm start because cold value retrains proved seed-unstable —
  4 failed attempts documented; the policy trainer is stable cold.)
- **8-robot beyond-oracle eval** (CPU): log `$SCRATCH/g16r8_ctl_results.log`, writes
  `scaling/results/g16r8/comparison_ungraded.json`, marker `UNGRADED_CTL_DONE`.
  Graded row already done: forward control 261/266 (98.1%) vs backward 230/266 (86.5%).
- **32×32 forward training** (GPU, log `$SCRATCH/train_g32r4_fwd3.log`, val top1
  ~0.869, 4 epochs, chain `$SCRATCH/g32r4_fwd_chain.log`, marker `G32R4 FWD DONE`).
  Backward-value/policy for 32×32 already trained (`scaling/runs/g32r4/*`).
- **24×24/8 forward control** (GPU, log `$SCRATCH/train_g24r8_fwd_ctl.log`, val top1
  ~0.832, batch 32, chain `$SCRATCH/g24r8_ctl_chain.log`, marker `G24R8 CONTROL DONE`).
  Backward nets for 24×24/8 already trained (`scaling/runs/g24r8/*`).
- Best-checkpoint locations pattern: `scaling/runs/<cfg>/<system>/lightning_logs/
  version_*/checkpoints/epoch=*.ckpt`; control retrains land in the repo-root
  `lightning_logs/` (use `ls -dt lightning_logs/version_* | head -1`, verify by log).
- Two persistent subagents from the previous session (B1 implementer, gallery
  analyst) delivered everything to disk; you cannot message them — their artifacts
  are the handoff. Spawn fresh agents as needed.

## 5. NEXT STEPS, in order (the owner's directive: extended backward vs all others,
then the page)
1. **When `B1 RETRAIN DONE`**: bank best ckpts as
   `checkpoints_backward/policy_b1.ckpt` and `checkpoints_backward/value_b1.ckpt`
   (policy monitor = min val_regret; value = min val_regret; check per-version
   hparams/logs to identify which lightning version is which).
2. **Teach the benchmark driver the extension**: add `--backward-b1` to
   `eval/compare.py`'s backward driver: build candidates via
   `heuristics.propose_b1`, and inside the anytime/prefix loop add the deterministic
   park-repair step (`skeleton/astar.py::park_repairs` — proposals from strict-
   realization failure feedback, children re-enter the search ordered by plan cost;
   the B1 agent's guidance: it is a search-loop change, deterministic physics, no
   training required; keep it opt-in, defaults untouched, and A/B a 20-instance
   slice for no-regression vs the non-B1 path before trusting it).
3. **Run the extended system** ("backward B1" rows, clearly labeled) at matched
   budgets (k=5, 1200): `eval/data/bench450.jsonl` (plain + prefix-check modes),
   then g16r6 graded (`scaling/data/g16r6/bench.solved.jsonl`, RR env vars per
   `scaling.run_config --config g16r6` printout) and g16r6 beyond-oracle
   (`bench.unsolved.jsonl`). Compare against the FINDINGS §9 rows. Expected: solve
   rate approaching the 97.6%/96.9% ceilings if the nets learned the vocabulary;
   regret and steps are the open questions. Append everything to FINDINGS.
4. **Finish the ladder**: when the two GPU trainings finish, run each config's
   graded compare + beyond-oracle compare (the g16r8 command pattern in
   `$SCRATCH/g16r8_ctl_results.log` is the template; g24r8 uses RR_GRID=24
   RR_ROBOTS=8 RR_WALLS=108 RR_ENV_DIR=.../environments_g24r8; g32r4 uses
   RR_GRID=32 RR_ROBOTS=4 RR_WALLS=192 RR_ENV_DIR=.../environments_g32r4; use
   `bench.solved.jsonl`, `--backward-prefix-check`, CPU). Fill FINDINGS §9 cells.
5. **Fill the page**: spawn an HTML agent (persistent, so the owner can iterate) to
   fold in: the extended-backward rows, all new scaling cells, the gallery link, and
   an updated ceiling section (90.7 → 97.6 with B1). Generator contract: modify
   `eval/build_report.py` only; numbers from JSONs only; pending rows with "–";
   keep the 5-tab layout, plain language, theme-awareness, the build verification
   pass. Publish: strip the doc shell (title + style/script blocks + body inner) and
   publish via the Artifact tool **to the existing URL above** (never a new URL).
6. **Final report to the owner in chat**: the full arc — use FINDINGS' verdict
   section as the spine; plain language; concrete numbers; honest caveats (forward
   training fragility was rescuable each time; 12 unresolved 6-robot probes are
   memory-bound; B1-trained nets' benchmark = the new headline).
7. **Optional follow-ups, in priority order** (only after 1–6): generalize parks
   (pairwise + multi-slide — recovers 3 measured instances); "supports-by-reference"
   bookkeeping extension (B2) for the remaining 11 base failures — scoped in the B1
   agent's report as low-vocabulary-risk but concentrated in `_apply`'s invariants,
   same A/B burden; self-play on the B1 stack (`--prefix-check --gen-realize-check`,
   base ckpts = the B1 pair); B1 support in the Rust datagen crate (gates per its
   VERIFICATION.md pattern); `validate_plan.py` should learn the `park` node type.

## 6. Pitfalls that already bit this project (do not repeat)
1. `pkill -f` matches your own wrapper/command text — kill by explicit PID list from
   plain `ps`, and verify with `ps` not `pgrep` (which matches itself).
2. Glob traps: `gen_s*` matched `gen_stock.jsonl` and deleted it once.
3. Value-net cold training is seed-unstable on regenerated data — warm-start from
   `checkpoints_backward/value_v2.ckpt` (or the B1 successor once banked).
4. Forward training diverges at ≥6 robots with the stock learning rate — always use
   the lr-1e-4 launcher pattern WITH the config's SPLITS rebinding (never call
   `move_planner.net`/`train.looped_pc` mains directly for scaling configs; see
   `$SCRATCH/train_fwd_lowlr_g16r8.py` for the working pattern).
5. Batch sizes: 24×24 forward max ~32–64; 32×32 forward 16, backward value batch 2 /
   group 8 (bigger OOMs a 40GB card).
6. Per-config board dirs (`environments_<cfg>/`) — never cross configs
   (silent geometry mismatch cost a full training run once; see memory +
   FINDINGS history).
7. lightning_logs version numbering is shared — identify runs by mtime + log, not
   by assumption.
8. Long CPU evals look hung because Python buffers output — check CPU% before
   killing anything.
9. The 6-robot probe artifacts join by `(tag, idx)`, not `idx` alone (graded and
   beyond-oracle indices collide).

## 7. Key file map (beyond the docs in §0)
- Benchmarks: `eval/bench_instances.py`, `eval/compare.py` (head-to-head driver),
  `eval/realize.py` (plan→moves under physics, B1-aware), `eval/build_report.py`
  (page generator), `eval/results/*.json` (all measurements),
  `eval/data/bench450.jsonl` (+`.meta.json`, sha-pinned).
- Backward planner: `GridEnv.py`, `skeleton/astar.py`, `skeleton/heuristics.py`
  (`propose` = old vocabulary, `propose_b1` = extended), `partial_plan.py`,
  `train/looped_pc.py` (value), `train/policy_tf.py` (proposal),
  `checkpoints_backward/` (banked nets).
- Forward planner: `move_planner/` (net, oracle, evaluate), `move_planner_v2/`
  (its self-play).
- Self-play (backward): `subgoal_selfplay/` (+ `arms_index.json`).
- Scaling: `scaling/` (configs, harness), `scaling/data/<cfg>/`,
  `scaling/runs/<cfg>/`, `scaling/results/<cfg>/`.
- Rust engine: `rust_datagen/` (README/DESIGN/VERIFICATION/ADOPTION).
- Probes: `analysis/artifacts/ceiling_probe.py` (`--b1` flag), probe results under
  `analysis/artifacts/` and `scaling/results/g16r6/`.
- Data: `nn/data/combined_v3.jsonl` (old vocabulary, clean),
  `nn/data/combined_b1.jsonl` (B1 vocabulary — keep separate).

## 8. Reporting duties
The owner reads FINDINGS.md and the web page, and expects: results reported in chat
in plain language when batches complete; ETAs with a basis when asked; honest
negatives stated as findings; the single-log discipline maintained.
