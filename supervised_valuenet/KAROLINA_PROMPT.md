# KAROLINA_PROMPT — continuation prompt for the agent on Karolina

Paste this file to the agent as its first prompt. Written 2026-07-20, when the
project moved from a shared lab machine to the Karolina supercomputer
(IT4Innovations, Ostrava). Everything below is either verified against
docs.it4i.cz (link given) or marked "verify on-site".

## 0. What this project is, in one paragraph

We compare two ways of building a neural-network solver for Ricochet Robots (a
sliding-robots puzzle): a *forward* planner that picks one move at a time, and a
*backward* planner that reasons in larger units called subgoals ("park helper
robot H on cell S so the slider stops on cell B"). Both use a proposal network +
a value network + best-first search under identical step budgets, on identical
pinned puzzle sets, and a puzzle only counts as solved if the plan plays out
legally move by move. The study is essentially complete: the backward planner
was repaired (53.1% → 89.1% playable), its expressiveness ceiling was measured
(90.7%) and then lifted by a plan-language extension called B1 (97.6% ceiling;
the retrained "B1" networks reach 95.3% on the base benchmark and flip the
beyond-oracle frontier: 80.6% vs forward's 48.5% at 6 robots). What remains is
finishing/logging the last scaling rows, quality-focused self-play, two scoped
language follow-ups, and the report page.

**Read order (all in `supervised_valuenet/`, your working directory):**
1. `PRIMER.md` — the whole story in plain language.
2. `SOURCE_OF_TRUTH.md` — canonical audit; wins any disagreement.
3. `FINDINGS.md` — THE single running log; §§13–15 are the language-extension
   arc; every new result appends here first.
4. `HANDOFF.md` — operational detail. CAVEAT: it was written on the OLD
   machine. Its GPU rules, scratchpad paths (`/tmp/claude-1010/...`) and
   "running right now" section are superseded by THIS file and by
   `MANIFEST.md` at the bundle top level.

## 1. What arrived in the bundle (and what did not)

The bundle unpacks to `karolina_bundle/` containing:
- `MCTS_evolution/` — the git repo (branch `b1-language-extension-scaling-study`,
  `.git` included — you can pull/push; remote `github.com:Jan21/MCTS_evolution`)
  with all gitignored data already in place: board sets `environments*/`
  (**boards only — the solver `cache/` dirs were stripped to shrink the
  archive; they rebuild automatically on first `GridEnv` use, ~seconds per
  board**), label data `nn/data/`, `scaling/data/`, banked checkpoints
  `checkpoints_backward/` (incl. `policy_b1.ckpt`/`value_b1.ckpt`),
  `move_planner/` data+checkpoints, `move_planner_v2/` and `subgoal_selfplay/`
  run dirs, `scaling/runs/`, and the repo-root `lightning_logs/` versions the
  measured rows reference (0, 1, 42–49; v43=g16r6 fwd control, v44=g16r8,
  v45=g32r4 fwd, v47=g24r8 control, v48/49=B1 policy/value).
- `supervised_valuenet/jobs/*.slurm` — Slurm job templates (committed to the
  repo; see §4). Same two files are also copied to the bundle top level at
  `karolina_bundle/jobs/`.
- `karolina_bundle/launcher_patterns/` (bundle top level, one directory above
  the `MCTS_evolution` checkout) — the working launcher scripts from the origin
  machine's scratchpad that HANDOFF references (`train_fwd_lowlr*.py`,
  `b1_retrain_chain.sh`, `final_lanes.sh`, other chain scripts). HANDOFF's
  `$SCRATCH/...` paths map here.
- `karolina_bundle/MANIFEST.md` (bundle top level) — full inventory, sizes,
  cache-stripped note, and the status of the two deferred scaling lanes.

NOT included (all rebuildable): the `environments*/cache/` solver caches
(~67 GB — stripped; `GridEnv` rebuilds any missing/corrupt cache entry
automatically on first use, so a first eval touching a board is just slower —
the boards themselves ARE all included); `rust_datagen/target/` (run
`cargo build --release`); `rust_datagen/pyref/out/` and `pyref/cache/`
(regenerate with the pyref scripts if ever needed); `scaling/data/*/rust_work/`
shard intermediates (the merged label jsonls ARE included); old lightning_logs
versions nothing references; Python caches.

## 2. Karolina facts (researched 2026-07-20; each item cited or flagged)

- **Login:** `ssh username@karolina.it4i.cz` (or `login[1-4].karolina.it4i.cz`),
  **SSH key only (RSA or ED25519) — no passwords**; the account must be attached
  to a project (the PI/owner authorizes it).
  https://docs.it4i.cz/en/docs/general/access-services/accessing-the-clusters/shell-and-data-access
- **Scheduler is Slurm, NOT PBS** (the cluster migrated): submit `sbatch`,
  monitor `squeue --me`, interactive `salloc -A PROJECT-ID -p qgpu_exp --gpus 1`;
  every job needs `-A PROJECT-ID`.
  https://docs.it4i.cz/en/docs/general/run-jobs/job-sub-exec/job-submission-and-execution
- **Partitions** (default/max walltime): `qcpu` 24/48 h, `qcpu_long` 72/144 h,
  `qcpu_exp` 1/1 h (tests), `qcpu_free` 12/18 h (low priority); `qgpu` 24/48 h,
  `qgpu_exp` 1/1 h, `qgpu_free` 12/18 h.
  https://docs.it4i.cz/en/docs/general/run-jobs/resource-allocation/karolina-partitions
- **GPU nodes:** 72 nodes, each 8× A100-40GB + 128 EPYC cores + 1 TB RAM. One
  GPU = `--partition qgpu --gpus 1` (gives 16 cores + 1/8 RAM).
  https://docs.it4i.cz/en/docs/clusters/karolina/hardware-overview and
  https://docs.it4i.cz/en/docs/general/run-jobs/job-sub-exec/karolina-slurm
- **Accounting is node-hours:** GPU jobs cost `gpus × hours / 8` node-hours;
  CPU jobs are charged whole nodes (128 cores) per hour, used or not.
  https://docs.it4i.cz/en/docs/general/run-jobs/resource-allocation/resource-accounting
- **Storage:** `/home` = 25 GB quota (code-sized things only — the bundle does
  NOT fit there); `/scratch` = Lustre, 20 TB quota, **files unused for 90 days
  are auto-deleted**; PROJECT storage `/mnt/proj1..3` (20 TB, 7-day snapshots)
  for project-lifetime data, not for job I/O.
  https://docs.it4i.cz/en/docs/clusters/karolina/storage and
  https://docs.it4i.cz/en/docs/storage/project/project-storage
- **Internet:** login nodes have outbound 22/80/443/873 (pip works there);
  **compute nodes have no direct external access** — install everything and
  publish anything from login nodes. Same shell-and-data-access page as above.
- **Modules:** Lmod, `ml` alias. Python, Anaconda3, CUDA, PyTorch and Rust all
  exist as modules; exact versions: run `ml av Python CUDA PyTorch Rust`.
  https://docs.it4i.cz/en/docs/software/modules/modules-karolina
- **Apptainer** (Singularity successor) is available as the container fallback
  (`apptainer exec --nv ...`). https://docs.it4i.cz/en/docs/software/tools/apptainer
- **Verify on-site:** exact module versions; your project ID and its node-hour
  balance (`it4ifree`); the exact scratch path convention for your project;
  whether `*_free`/`*_exp` queues charge the allocation; whether any HTTP proxy
  exists for compute nodes (assume none).

## 3. First hour on the machine (setup checklist)

0. **Before you start**: confirm with the owner your SSH access is working and
   get the **PROJECT-ID** and the node-hour budget — nothing schedules without
   `-A PROJECT-ID` and every job spends the allocation (§4).
1. **Unpack on /scratch, run everything from /scratch** — NEVER /home. `/home`
   is only 25 GB; the bundle unpacks to ~66 GB. `/scratch` is Lustre, 20 TB.
   `tar -xzf karolina_bundle.tar.gz -C <your-scratch-dir>/` — one command, then
   `cd <your-scratch-dir>/karolina_bundle/MCTS_evolution/supervised_valuenet`.
   (You may keep the small git checkout/configs referenceable from /home if you
   like, but data + runs live on /scratch.)
2. **PURGE WARNING — copy results off /scratch.** Files on /scratch untouched
   for **90 days are auto-deleted**. Commit + push results often (`.git` is
   live), and after milestones copy `checkpoints_backward/`, new
   `scaling/results/`, and `eval/results/` to PROJECT storage (`/mnt/proj*`) or
   download them — before the 90-day window.
3. **Python env — build on a LOGIN node** (compute nodes have no internet;
   venv on /scratch). Prefer Karolina's own modules first: `ml av Python CUDA
   PyTorch` and load matching modules; only `pip install` what the modules
   don't provide. Fallback recipe: `ml Python` (≥3.10; origin machine ran
   3.11.13), then `python3 -m venv <scratch>/venv && source
   <scratch>/venv/bin/activate && pip install "torch>=2.8"
   "pytorch-lightning>=2.5" numpy networkx`. Origin machine: torch 2.8.0+cu126,
   lightning 2.5.1. The default torch wheel bundles the CUDA runtime and covers
   the A100; `requirements.txt` in `supervised_valuenet/` is the authoritative
   dependency list (nothing else is imported). If a module/pip torch mismatches
   the driver, `apptainer exec --nv <img>` is the container fallback.
4. **Rust toolchain — on a LOGIN node** (cargo fetches crates from the
   internet): `ml Rust` (or rustup on a login node), then `cd rust_datagen &&
   cargo build --release && make smoke`. `make smoke` passing is the acceptance
   test (golden corpora are included). Do the build on login, then use the
   binary from compute jobs.
5. **End-to-end verification** (CPU, ~minutes; run under `qcpu_exp` or a short
   `salloc -A PROJECT-ID -p qcpu_exp` — keep login nodes light). This exact
   recipe PASSED against the bundle before transfer (both planners solved 19/20
   on the origin machine). From `supervised_valuenet/`:
   ```
   head -20 eval/data/bench450.jsonl > /tmp/bench20.jsonl
   OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
   python3 -m eval.compare --instances /tmp/bench20.jsonl \
       --expansions 1200 --k 5 \
       --backward-policy checkpoints_backward/policy_b1.ckpt \
       --backward-value  checkpoints_backward/value_b1.ckpt \
       --backward-anytime --backward-b1 --device cpu \
       --out /tmp/smoke20.json --md /dev/null
   ```
   Pass = it completes, writes the JSON, and both planners post nonzero solve
   counts (forward uses its default checkpoint `move_planner/checkpoints/
   best.ckpt`, included). Only then start real work.

## 4. How GPU/CPU work maps to Karolina

- Slurm templates (submit with `sbatch`): `jobs/train_template.slurm` (one A100
  via `-p qgpu --gpus 1`) and `jobs/eval_template.slurm` (whole CPU node via
  `-p qcpu`; pack ~16 eight-thread eval lanes per node — CPU nodes charge whole
  node-hours regardless). Both are committed at `supervised_valuenet/jobs/` and
  copied to `karolina_bundle/jobs/`. Edit `PROJECT-ID`, the /scratch paths, the
  module loads, and the payload before first use. Interactive testing:
  `salloc -A PROJECT-ID -p qgpu_exp --gpus 1` (GPU) or `-p qcpu_exp` (CPU).
- **The origin machine's GPU rules (max one/two GPUs, claim locks, nvidia-smi
  first) are obsolete here.** On Karolina the binding constraint is the
  project's **node-hour** allocation: 1 GPU-hour = 0.125 node-hour, charged on
  the allocation whether the GPU is busy or idle; a CPU job bills the whole
  128-core node. **Before any big run, ask the owner for the PROJECT-ID and how
  many node-hours the project may spend**; report projected cost
  (jobs × walltime × rate) when asking. Smoke-test on `qgpu_exp`/`qcpu_exp`
  (1 h) first; consider `*_free` partitions for low-priority work. Request a
  whole node (`--gpus 8`) only when a run actually parallelizes across 8 GPUs.
- Evaluations stay CPU (`OMP_NUM_THREADS=8`, `CUDA_VISIBLE_DEVICES=""`), as
  before. Training uses one GPU per job unless the owner approves more.

## 5. Work queue, in order

1. **FIRST — rerun the two unfinished scaling comparisons.** They were too slow
   to finish on the old shared CPU box (g32r4 graded was at 150/161 after ~20 h);
   they run trivially here (one `eval.compare` per lane on a GPU node, or a
   packed CPU node). All data + checkpoints are in the bundle. Run from
   `supervised_valuenet/` with `PYTHONPATH=. CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8`
   (backward ckpt paths use a glob — resolve with `ls`, there is one epoch ckpt each):

   **g32r4** (32×32, 4 robots) — BOTH graded and beyond-oracle are missing:
   ```
   export RR_GRID=32 RR_ROBOTS=4 RR_WALLS=192
   export RR_ENV_DIR=$PWD/environments_g32r4
   BP=$(ls scaling/runs/g32r4/backward-policy/lightning_logs/*/checkpoints/epoch*.ckpt | head -1)
   BV=$(ls scaling/runs/g32r4/backward-value/lightning_logs/*/checkpoints/epoch*.ckpt | head -1)
   FWD=lightning_logs/version_45/checkpoints/epoch=3-step=158564.ckpt
   # graded:
   python3 -m eval.compare --instances scaling/data/g32r4/bench.solved.jsonl \
       --expansions 1200 --k 5 --backward-policy $BP --backward-value $BV \
       --forward-ckpts $FWD --backward-prefix-check --device cpu \
       --out scaling/results/g32r4/comparison.json --md scaling/results/g32r4/COMPARISON.md
   # beyond-oracle:
   python3 -m eval.compare --instances scaling/data/g32r4/bench.unsolved.jsonl \
       --expansions 1200 --k 5 --backward-policy $BP --backward-value $BV \
       --forward-ckpts $FWD --backward-prefix-check --device cpu \
       --out scaling/results/g32r4/comparison_ungraded.json --md /dev/null
   ```

   **g24r8** (24×24, 8 robots) — ONLY beyond-oracle is missing (graded already
   landed at `scaling/results/g24r8/comparison.json`: backward 144/161 = 89.4%
   at 44 steps, forward control 157/161 = 97.5% at 152 steps — do NOT rerun it):
   ```
   export RR_GRID=24 RR_ROBOTS=8 RR_WALLS=108
   export RR_ENV_DIR=$PWD/environments_g24r8
   BP=$(ls scaling/runs/g24r8/backward-policy/lightning_logs/*/checkpoints/epoch*.ckpt | head -1)
   BV=$(ls scaling/runs/g24r8/backward-value/lightning_logs/*/checkpoints/epoch*.ckpt | head -1)
   FWD=lightning_logs/version_47/checkpoints/epoch=3-step=115144.ckpt
   # bench.unsolved.jsonl already built and in the bundle:
   python3 -m eval.compare --instances scaling/data/g24r8/bench.unsolved.jsonl \
       --expansions 1200 --k 5 --backward-policy $BP --backward-value $BV \
       --forward-ckpts $FWD --backward-prefix-check --device cpu \
       --out scaling/results/g24r8/comparison_ungraded.json --md /dev/null
   ```
   `launcher_patterns/final_lanes.sh` is the original two-lane script if you want
   to run them together (fix its absolute paths, drop its scratch-cache scan).
2. **Log all landed results**: append every result JSON under
   `scaling/results/g24r8/` and `scaling/results/g32r4/` — the pre-existing ones
   AND the two just produced — that is absent from `FINDINGS.md` §9 to FINDINGS
   (numbers from the JSONs only), then commit the result files.
3. **Quality-focused B1 self-play** — the open weakness: frontier solutions run
   long (mean ~16.9 moves at 6-robot beyond-oracle). Self-play on the B1 stack
   (`subgoal_selfplay/`, flags `--prefix-check --gen-realize-check`, base
   checkpoints = the B1 pair) aiming to shorten solutions without giving back
   the 80.6% frontier solve rate. Name runs by vocabulary; update
   `subgoal_selfplay/arms_index.json`.
4. **Parks generalization** (pairwise + multi-slide parks — recovers 3 measured
   instances; see `analysis/b1_extension_notes.md`).
5. **B2 "supports-by-reference"** for the remaining 11 base structural
   failures — scoped as bookkeeping in `_apply`'s invariants, same A/B burden
   as B1 (probe re-runs + realizer A/B, zero regression on default paths).
6. **B1 support in `rust_datagen`** (currently Python-only for B1 labels);
   gate it per `rust_datagen/VERIFICATION.md`'s pattern.
7. **`validate_plan.py` learns the `park` node type.**
8. **Report page refresh**: `PYTHONPATH=. python3 -m eval.build_report` →
   `eval/results/report.html`; numbers only from result JSONs; pending cells
   show "–". Publishing: the page lives at claude.ai artifact
   `785268c8-43ad-4f45-a5af-3e09c815d910` — publish to THAT URL, never a new
   one. If artifact access is unavailable on Karolina, regenerate
   `report.html` and hand the file to the owner instead.

## 6. House rules (unchanged; violations were reverted before)

- **Playable-moves scoring only**: solved iff the plan plays out legally
  (`eval/realize.py`). Never compare abstract plan cost to move optima;
  beyond-oracle files carry placeholder optima — only solve rate / steps /
  time / plan length are meaningful there.
- **Vocabulary separation**: never mix B1-vocabulary labels
  (`nn/data/combined_b1.jsonl`) with old-vocabulary labels; cost-to-go is
  vocabulary-relative; name datasets/runs/checkpoints by vocabulary.
- **The subgoal methodology stays frozen**: bug fixes, tuning, and new
  candidate types inside the propose step are allowed; redesigns and raw-move
  fallbacks are not. No exact solver at solve time — networks + search only;
  the oracle exists only for training labels and reference optima.
- **Plain language, neutral tone** in chat and every document; define terms at
  first use; no attributions. Every significant result appends to
  `FINDINGS.md` with the source file named.

## 7. Pitfalls that already bit this project (do not repeat)

1. `pkill -f` matches your own command text — on Karolina prefer `scancel
   <jobid>`; inside a node kill by explicit PID from plain `ps`.
2. Glob traps: `gen_s*` once matched `gen_stock.jsonl` and deleted it.
3. Value-net COLD training is seed-unstable — warm-start from the current
   banked value checkpoint (`value_b1.ckpt`, else `value_v2.ckpt`).
4. Forward training diverges at ≥6 robots on the stock learning rate — use the
   `launcher_patterns/train_fwd_lowlr*.py` pattern (lr 1e-4 WITH the config's
   SPLITS rebinding); never call the trainer mains directly for scaling configs.
5. Batch caps on a 40 GB A100: 24×24 forward ≤32–64; 32×32 forward 16; 32×32
   backward value batch 2 / group 8.
6. Per-config board dirs (`environments_<cfg>/`) must never be crossed —
   silent geometry mismatch once wasted a full training run.
7. `lightning_logs` version numbers are shared across run types — identify a
   run by its log/hparams and mtime, never by assumption.
8. Long CPU evals look hung because Python buffers output — check CPU% (sstat/
   top) before killing anything.
9. The 6-robot probe artifacts join on `(tag, idx)`, not `idx` alone.

## 8. Reporting

The owner reads `FINDINGS.md` and the report page, and expects: batch results
reported in chat in plain language; ETAs with a stated basis; honest negatives
stated as findings; the single-log discipline maintained; and a question
BEFORE any run that spends meaningful node-hours (§4).
