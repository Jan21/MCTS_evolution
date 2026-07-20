# Pass 4 — Scaling-study readiness audit

Read-only audit of `scaling/` and its artifacts, 2026-07-10 ~12:30 CEST.
Working dir: `/mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet` (all paths below relative to it unless absolute).
No repo file was modified; no jobs were touched. Three live jobs were identified and left alone:
PID 1405327 (`subgoal_selfplay.train_iterate`, warm self-play loop, ~12 h elapsed), PID 25116
(`run_scratch_v6.py`, forward from-scratch self-play, ~24 h elapsed), and PID 3094365
(`eval.compare --backward-anytime` on bench450, GPU 0, started 12:32 today by a sibling task —
this is the addendum's L0 measurement, already running).

Terminology used throughout (plain language):
- **labeling / data generation** — running the exact solver on random puzzle instances to produce
  training answers (the solver is used only here and for reference answers on the test set, never
  inside the planner at test time).
- **bench** — the fixed file of test puzzles each configuration shares; `d_star` is the true
  optimal move count stored with each puzzle, when the exact solver could compute it.
- **strict realized moves** — a backward plan converted to an actual legal move sequence under
  full physics; the honest comparison metric (addendum.md §1).

---

## 1. Inventory per configuration

### g16r6 (robot axis, 16×16, 6 robots) — DATA COMPLETE, everything after data is TODO

| stage | status | evidence |
|---|---|---|
| boards | **DONE** — 1200 `env_*.pkl` | `environments_g16r6/` (env_0 21:48 Jul 9 → env_1199 22:25 Jul 9) |
| backward training data | **DONE** — 129,423 records, 67 MB | `scaling/data/g16r6/backward.jsonl`, merged from 8 shards (shard files kept) |
| forward training data | **DONE** — 1,038,827 records, 338 MB | `scaling/data/g16r6/forward.jsonl`, 8 shards, generated with `--score-candidates` |
| training (all 3 nets) | **TODO** — no checkpoints exist | `scaling/runs/g16r6/{backward-policy,backward-value}/lightning_logs/version_0/` holds only `hparams.yaml` + a ~1.1 KB tfevents spanning 09:18→09:20 today (2-minute wrapper verification runs by now-dead PIDs 2402011/2399332, no `checkpoints/` dir); `scaling/runs/g16r6/forward/` is empty |
| bench | **PARTIAL (smoke only)** — 6 instances, 3 with `d_star=null` | `scaling/data/g16r6/bench_smoke.jsonl` (+ `.solved.jsonl` with 3, `.meta.json`); no full `bench.jsonl` anywhere |
| comparison | **TODO** | `scaling/results/` does not exist |

Record schema samples (first line of each file):
- backward: `{"env_id": 0, "target": [11,9], "target_robot": [[5,12],"Red"], "helpers": [... 5 helpers ...], "seg_start": ..., "seg_end": ..., "cand_bottleneck": ..., "cand_support": ..., "cand_helper": ..., "cost_to_go": 15, "is_optimal": false, "depth": 0}` — same candidate-decision schema the legacy backward trainers consume.
- forward: `{"env_id": 80, "robots": [... 6 positions ...], "target": [11,0], "target_idx": 1, "cost_to_go": 7, "best_moves": [[1,0]], "legal_moves": [...], "depth": 0, "full": false}` — same MoveNet schema, robots list length 6.

The training run dirs' hparams already carry the config (`grid: 16, robots: 6` in both
`scaling/runs/g16r6/*/lightning_logs/version_0/hparams.yaml`), confirming the wrapper's env
plumbing works (`scaling/train.py:52-71`).

### g24r4 (grid axis, 24×24, 4 robots) — DATA COMPLETE, no runs dir at all

| stage | status | evidence |
|---|---|---|
| boards | **DONE** — 1200 pkls | `environments_g24r4/` (22:24→22:31 Jul 9) |
| backward training data | **DONE** — 109,595 records, 53 MB | `scaling/data/g24r4/backward.jsonl` (8 shards kept) |
| forward training data | **DONE** — 916,526 records, 242 MB | `scaling/data/g24r4/forward.jsonl` (8 shards, `--score-candidates`) |
| training | **TODO** — `scaling/runs/g24r4/` absent | |
| bench | **TODO** — no bench files, not even smoke | |
| comparison | **TODO**; and backward cannot be scored here yet (see §5) | |

Both backward files cover exactly boards 0–1049 (train+val+test splits; verified by scanning
`env_id` in every record: 1050 distinct boards, min 0, max 1049).

### g16r8, g24r8, g32r4 — NOT STARTED

No `environments_g16r8/`, `environments_g24r8/`, `environments_g32r4/` dirs; no
`scaling/data/<cfg>/` dirs; nothing else. Every stage is TODO from `gen_boards` onward.

---

## 2. Pipeline dry-runs (print-only), annotated

`PYTHONPATH=. python3 -m scaling.run_config --config g16r6` prints 8 stages; status from §1:

1. `python -m scaling.gen_boards --config g16r6` — **DONE**
2. `python -m scaling.gen_data --config g16r6 --system backward --per-graph 20` — **DONE** (was run sharded ×8, then concatenated, exactly as the printed note suggests)
3. `python -m scaling.gen_data --config g16r6 --system forward --per-graph 30 --score-candidates` — **DONE** (sharded ×8)
4. `CUDA_VISIBLE_DEVICES=<FREE_GPU> python -m scaling.train --config g16r6 --system backward-value -- --data scaling/data/g16r6/backward.jsonl --epochs 50 --batch-size 8 --max-per-group 32` — **TODO**
5. `CUDA_VISIBLE_DEVICES=<FREE_GPU> python -m scaling.train --config g16r6 --system backward-policy -- --data scaling/data/g16r6/backward.jsonl --epochs 50` — **TODO**
6. `CUDA_VISIBLE_DEVICES=<FREE_GPU> python -m scaling.train --config g16r6 --system forward -- --data scaling/data/g16r6/forward.jsonl --epochs 50 --patience 5` — **TODO**
7. `python -m scaling.bench --config g16r6 --per-board 3 --seed 1` — **TODO** (only the 6-instance smoke exists)
8. `CUDA_VISIBLE_DEVICES=<FREE_GPU> RR_GRID=16 RR_ROBOTS=6 RR_WALLS=48 RR_ENV_DIR=/mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet/environments_g16r6 PYTHONPATH=. python -m eval.compare --instances scaling/data/g16r6/bench.jsonl --expansions 1200 --k 5 --backward-policy <BACKWARD_POLICY_CKPT> --backward-value <BACKWARD_VALUE_CKPT> --forward-ckpts <FORWARD_CKPT> --device cuda --out scaling/results/g16r6/comparison.json --md scaling/results/g16r6/COMPARISON.md` — **TODO**

`... --config g24r4` prints the same sequence with g24r4 paths and
`--batch-size 4 --max-per-group 16` for the value net (stage 4); stages 1–3 **DONE**, 4–8 **TODO**.
Its compare line replaces the two backward checkpoint flags with `--skip-backward` and prints:
"backward system skipped: eval/realize.py is 16x16-only today, so backward plans cannot be
realized/scored at this grid size yet".

---

## 3. Labeling cost (measured from shard file birth/mtime + record counts)

Shard files are opened at labeler start and last written at completion, so birth→mtime is the
per-shard duration (`stat --format='%w %y'`; the filesystem stores creation time).

| config / system | shards | started | last finished | wall | Σ core-hours | records | records per core-hour |
|---|---|---|---|---|---|---|---|
| g16r6 backward | 8 × 1 process | Jul 9 22:25 | Jul 10 05:13 | **6 h 48 m** | ≈ 45.6 | 129,423 | ≈ 2,840 |
| g16r6 forward | 8 × (`--workers 16`) | Jul 10 05:13 | Jul 10 06:32 | **1 h 19 m** | ≤ 164 (128 workers) | 1,038,827 | ≥ 6,300 |
| g24r4 backward | 8 × 1 process | Jul 9 22:33 | Jul 10 08:49 | **10 h 16 m** | ≈ 76.6 | 109,595 | ≈ 1,430 |
| g24r4 forward | 8 × (`--workers 16`) | Jul 10 08:49 | Jul 10 09:40 | **51 m** | ≤ 109 | 916,526 | ≥ 8,400 |

Per planned instance (backward 1050 boards × 20 = 21,000; forward × 30 = 31,500):
backward ≈ **7.8 s·core/instance at G=16 → ≥ 13.1 s·core at G=24** (≈ 1.7× per grid step, and
records per board fell 123 → 104); forward ≤ 19 s·core/instance at g16r6, ≤ 12 at g24r4
(upper bounds — pool workers idle at the tail). Caveats: the two backward jobs ran concurrently
overnight and overlapped the forward jobs, so contention inflates the later g24r4 shards
somewhat; and the forward labeler is the exact-move-solver-per-candidate recipe
(`move_planner/generate.py:86-129`, cap 40,000 expansions per solve, `generate.py:149`).

The backward labeler's 120 s per-instance wall-time guard is mandatory in the harness
(`scaling/backward_label.py:45-46,66-76`; noted in `scaling/README.md:44-47`). Timeout and
attempt-failure counts are printed to stdout only (`backward_label.py:84-88`) — **no sidecar or
log file was kept**, so the realized instance/timeout counts per config are not recoverable from
disk (only record counts are).

Volume vs the legacy 16×16/4-robot training sets:
- backward: 129,423 (g16r6) and 109,595 (g24r4) vs 135,834 (`nn/data/combined_v2.jsonl`) — **0.95× / 0.81×, comparable**.
- forward: 1,038,827 and 916,526 vs 153,625 (`move_planner/data/moves.jsonl`, the naive set) — 6–6.8×; vs 512,752 (`move_planner/data/moves_combined.jsonl`, the candidate-scored set actually behind the best legacy forward model) — **≈ 2×, comparable-to-larger**. Composition shifted as expected with `--score-candidates`: 5% of g16r6 forward rows are `full=true` vs 85% in naive `moves.jsonl`.

---

## 4. Oracle strain (every bench meta found)

| bench file | config | oracle caps | failure handling | n_instances | n_oracle_failed |
|---|---|---|---|---|---|
| `eval/data/bench450.jsonl` | g16r4 (legacy) | 60,000 expansions, no time cap (`move_planner/evaluate.py:173`) | resample until solved — failures invisible | 450 | n/a (meta has no field) |
| `eval/data/smoke.jsonl` | g16r4 | same | same | 6 | n/a |
| `scaling/data/g16r6/bench_smoke.jsonl` | g16r6 | 200,000 expansions **and** 60 s wall cap | kept with `d_star=null` | 6 | **3 (50%)** |
| g24r4 / g16r8 / g24r8 / g32r4 | — | — | — | **no bench exists** | — |

No full bench exists for any scaled config; the only scaled meta is the 6-instance g16r6 smoke
(boards 0–2, per_board 2, generated Jul 9 21:50). Caps do **not** differ per config: they are
the module defaults `--max-expansions 200000 --oracle-timeout 60.0`
(`scaling/bench.py:71-74`) and nothing in `run_config.py` overrides them (`run_config.py:99-101`).
Note the scaled caps are more generous than the legacy sampler's (200k + 60 s vs 60k), yet half
the r6 smoke draws still failed — direct evidence of the exact solver straining with 6 robots.
The failure rate is recorded per run in the meta (`scaling/bench.py:146-149`) and is itself a
headline scaling number. `run_config --execute` automatically switches the comparison to
`bench.solved.jsonl` when `n_oracle_failed > 0` (`run_config.py:134-139`); solve rate can still
be reported over the full file (`scaling/README.md:95-98`).

---

## 5. Blockers and exact remaining steps

### Robot axis — g16r6 (nothing blocks it; this is the first scaled comparison)

Remaining commands, in order (from the dry-run; run from the repo root with `PYTHONPATH=.`;
pick an idle GPU with `nvidia-smi` first — GPU 0 currently carries the anytime benchmark):

```bash
# 1–3: training (three independent GPU jobs; flags per scaling/README.md:78-84 resource table)
CUDA_VISIBLE_DEVICES=<FREE_GPU> python -m scaling.train --config g16r6 --system backward-value -- \
  --data scaling/data/g16r6/backward.jsonl --epochs 50 --batch-size 8 --max-per-group 32
CUDA_VISIBLE_DEVICES=<FREE_GPU> python -m scaling.train --config g16r6 --system backward-policy -- \
  --data scaling/data/g16r6/backward.jsonl --epochs 50
CUDA_VISIBLE_DEVICES=<FREE_GPU> python -m scaling.train --config g16r6 --system forward -- \
  --data scaling/data/g16r6/forward.jsonl --epochs 50 --patience 5

# 4: full bench (CPU, single process — the sampler consumes one random.Random(seed)
#    sequentially over boards (scaling/bench.py:92-99), so the canonical file must
#    come from ONE run; sharding would change the instance set)
python -m scaling.bench --config g16r6 --per-board 3 --seed 1

# 5: head-to-head (exact env prefix as run_config prints; substitute the best checkpoints
#    from scaling/runs/g16r6/<system>/lightning_logs/, and use bench.solved.jsonl —
#    the meta will almost certainly show n_oracle_failed > 0 at r6)
CUDA_VISIBLE_DEVICES=<FREE_GPU> RR_GRID=16 RR_ROBOTS=6 RR_WALLS=48 \
  RR_ENV_DIR=/mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet/environments_g16r6 \
  PYTHONPATH=. python -m eval.compare --instances scaling/data/g16r6/bench.solved.jsonl \
  --expansions 1200 --k 5 --backward-policy <BACKWARD_POLICY_CKPT> \
  --backward-value <BACKWARD_VALUE_CKPT> --forward-ckpts <FORWARD_CKPT> --device cuda \
  --out scaling/results/g16r6/comparison.json --md scaling/results/g16r6/COMPARISON.md
```

Equivalent one-liner (resolves checkpoints and the solved-file swap itself):
`python -m scaling.run_config --config g16r6 --steps train-backward-value,train-backward-policy,train-forward,bench,compare --execute --gpu <idx>` (sequential on one GPU).

`eval/realize.py` is robot-count-agnostic at 16×16 (it reads robots from the state), so the
backward system is scoreable here today — confirmed by the addendum (`addendum.md:186-190`).

### Grid axis — g24r4 (one real blocker: 16×16-only realization)

Trainings and bench are runnable now with the same pattern (value net uses
`--batch-size 4 --max-per-group 16`; commands identical to §2's g24r4 dry-run). But the
comparison can only score the forward system: `run_config` hard-codes the skip at
`scaling/run_config.py:106-109` —

```python
skip_back = g != 16
backward_args = ("--skip-backward" if skip_back else
                 "--backward-policy <BACKWARD_POLICY_CKPT> "
                 "--backward-value <BACKWARD_VALUE_CKPT>")
```

with the explanatory note built at `run_config.py:116-118` (and `scaling/README.md:91-94`).

What realize-generalization must satisfy before that gate can be removed:
1. Every physics function must stop assuming a 16-cell side. Defaults `size=16` live at
   `eval/realize.py:37,65,163,182` and `simulate.py:20,40,74,102,132`; boundary checks use
   `size-1` (`simulate.py:55,63`). Either derive `size` from the board itself
   (`len(env.grid_data) == size**2`) inside `wall_sets` / `abstract_moves` / `strict_moves` /
   `verify_plan`, or read `RR_GRID` — the addendum recommends deriving from the board
   (`addendum.md:191-197`).
2. The callers that pass no `size` must then be correct unchanged: `eval/compare.py:209,226-229`
   and `subgoal_selfplay/selfplay.py:236-237`.
3. Regression bar: at G=16 results must be unchanged (re-run a bench450 slice and diff), and at
   G=24 a handful of strict realizations must be verified legal by replaying the moves.
4. Finally edit the `skip_back = g != 16` gate (`run_config.py:106`) so g24r4/g32r4 emit the
   backward flags. Until all four hold, no backward score exists at G≠16 — the g24r4 backward
   training data (already on disk) is waiting on exactly this. Note the backward *planner* side
   already works at G=24: the labeler's rollout+solver ran at 24×24 to produce that data; only
   plan→moves scoring is 16-only.

---

## 6. Sanity checks run (CPU only, no training launched)

1. **Backward records are trainer-compatible.** With the g16r6 env
   (`RR_GRID=16 RR_ROBOTS=6 RR_WALLS=48 RR_ENV_DIR=.../environments_g16r6`), the first 3,000
   lines of `scaling/data/g16r6/backward.jsonl` load via `nn.benchmark.load` and group into 251
   decision groups via `nn.benchmark.group_by_decision` (mean 12.0 candidates, max 14 —
   consistent with `--max-candidates 14`); **251/251 groups contain an `is_optimal` candidate**;
   every record has 5 helpers (6 robots total). The exact encode path the value trainer uses
   (`train.looped_pc:29-30,239-241`) also works: `train.encode` imports with `GRID=16 ROBOTS=6`,
   `_graph(0)` builds the slide graph (4,196 edges) and `_node_features(record)` returns the
   expected (256, 9) tensor.
2. **Boards load under the config env.** `GridEnv.from_env(0)` returns 6 robots on a 256-cell
   (16×16) board for g16r6, and 4 robots on a 576-cell (24×24) board for g24r4
   (`GridEnv.py:9-12` honors `RR_ENV_DIR`).

---

## 7. Cost estimate for the complete g16r6 row

Anchors: legacy backward-policy full training (`lightning_logs/version_24`, 135,834 records,
same architecture/hparams) ran 22:23→23:11 Jul 9 = **48 min** to its best epoch (13); a legacy
value-shaped run (`lightning_logs/version_0`, 12.4 MB checkpoint) ran ≈ **56 min** to epoch 11.
The g16r6 backward set (129k records) is the same size. The forward set is 1.04 M records at
batch 64 (`move_planner/net.py:196,200` defaults) ≈ 16,200 steps/epoch; no full legacy MoveNet
training log survives, so its per-epoch time is assumed 5–12 min on an idle A100-40GB
(small recurrent net, 16×16 inputs). Bench cost is dominated by oracle failures (60 s each);
smoke showed 3/6 failing. Compare cost is grounded on measured bench450 aggregates
(`eval/results/comparison_forward.json`: 1.0 s/instance for the best forward model,
11.4 s/instance for the weakest; the live anytime run is pacing ≈ 1.9–2.3 s/instance backward).

| step | resource | estimate | basis |
|---|---|---|---|
| train backward-value | 1 GPU | **1–2 h** | legacy ≈ 1 h at same data size; early stopping |
| train backward-policy | 1 GPU | **0.75–1.5 h** | legacy 48 min |
| train forward | 1 GPU | **2–5 h** (worst case 6–10 h if it runs all 50 epochs) | 16,200 steps/epoch × 5–12 min, `--patience 5` stops ~epoch 15–25 |
| bench (450 draws) | 1 CPU core, single process | **3–6 h** | ~40–60% failures × 60 s + solved draws 1–30 s; runs in parallel with training |
| compare (both systems) | 1 GPU | **0.5–1.5 h** | 450 × (≈1–3 s forward at r6 branching + ≈2 s backward) |
| **total** | | **≈ 4–10 h GPU on one card + overlappable CPU bench → one working day wall-clock**; ≈ 3.5–7 h if the two backward trainings share a second idle GPU | |

Assumptions: an idle A100-40GB per GPU job (shared machine — GPUs 0 busy now with the anytime
run, plus the two long-running jobs; check `nvidia-smi` at launch); the wrapper's early stopping
behaves as in the legacy runs; bench failure rate near the smoke's 50%. The g24r4 row, once
realize is generalized, adds ≈ 3–8 h forward training (2.25× larger boards), ≈ 2–4 h per
backward net (`--batch-size 4 --max-per-group 16` halves throughput), 2–8 h bench (failure rate
unknown), ≤ 1.5 h compare. Rows g16r8/g24r8/g32r4 additionally need boards (~0.5–1 h) and data
(~7–14 h wall backward, 1–2 h wall forward, on the same 8-shard / 128-worker pattern).

---

## 8. Bottom line

- **Robot axis (g16r6): ready now.** Boards, both training sets, smoke checks, and the harness
  all in place and verified; remaining work is pure compute (3 trainings + full bench + compare,
  ≈ one day, commands in §5). The oracle-failure rate (50% on smoke) will make regret reportable
  only on the `bench.solved.jsonl` subset — by design, and itself a result.
- **Grid axis (g24r4): data-ready, score-blocked.** Everything through data exists; forward-only
  results are obtainable today; the backward score waits on generalizing
  `eval/realize.py`/`simulate.py` beyond 16×16 (§5), the single piece of engineering (as opposed
  to queueing) left in the study.
- **g16r8, g24r8, g32r4: not started** (no boards, no data).
