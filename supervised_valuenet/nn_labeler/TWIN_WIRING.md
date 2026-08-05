# Twin retrain wiring — what the NN-labeled corpora must replay, and how the
# retrained planners get benched

Companion to `nn_labeler/jobs/twin_retrain.slurm`. Everything below was read out
of the repo on 2026-08-05; every claim carries a `file:line` or a measured
number. Unresolved items are marked **OPEN** and collected in section 6.

The experiment: `gate_v2_and_twins.slurm` (job 4618940) replays the exact
corpora's own instances with the banked NN labeler and writes
`nn_labeler/results/twin_<cfg>.jsonl` for `g24r4`, `g24r8`, `g32r4`. This
document is the recipe for retraining the two backward planners on those twins
and benching them so the twin row and the exact row
(`scaling/results/<cfg>/comparison.json`) differ in **one** variable: where the
labels came from.

Vocabulary is BASE throughout. No path, run dir or result file here touches a
`b2` artifact (house rule: never mix vocabularies in one dataset or run name).

---

## 1. Corpus provenance — which file each existing planner actually trained on

### 1a. The table

| config | system | checkpoint of record | corpus file it trained on | primary evidence |
|---|---|---|---|---|
| g24r4 | backward-value  | `scaling/runs/g24r4/backward-value/lightning_logs/version_0/checkpoints/epoch=22-step=38019.ckpt` | `scaling/data/g24r4/backward.jsonl` | `jobs/launcher_patterns/g24r4_chain.sh:18-20` |
| g24r4 | backward-policy | `scaling/runs/g24r4/backward-policy/lightning_logs/version_0/checkpoints/epoch=1-step=1654.ckpt` | `scaling/data/g24r4/backward.jsonl` | `jobs/launcher_patterns/g24r4_chain.sh:23-24` |
| g24r4 | forward | `scaling/runs/g24r4/forward/lightning_logs/version_0/checkpoints/epoch=7-step=76360.ckpt` | `scaling/data/g24r4/forward.jsonl` | `jobs/launcher_patterns/g24r4_chain.sh:13-15` |
| g24r8 | backward-value  | `scaling/runs/g24r8/backward-value/lightning_logs/version_0/checkpoints/epoch=10-step=21681.ckpt` | `scaling/data/g24r8/backward.jsonl` *(byte-identical to `backward.rust.jsonl`)* | `jobs/launcher_patterns/g24r8_chain.sh:10` |
| g24r8 | backward-policy | `scaling/runs/g24r8/backward-policy/lightning_logs/version_0/checkpoints/epoch=29-step=29580.ckpt` | `scaling/data/g24r8/backward.jsonl` *(idem)* | `jobs/launcher_patterns/g24r8_chain.sh:12` |
| g24r8 | forward | `lightning_logs/version_47/checkpoints/epoch=3-step=115144.ckpt` (repo-root logs) | `scaling/data/g24r8/forward.jsonl` *(byte-identical to `forward.rust.jsonl`)* | `jobs/launcher_patterns/train_fwd_lowlr_g24r8.py:22-24` |
| g32r4 | backward-value  | `scaling/runs/g32r4/backward-value/lightning_logs/version_1/checkpoints/epoch=15-step=52528.ckpt` | `scaling/data/g32r4/backward.jsonl` *(byte-identical to `backward.rust.jsonl`)* | `jobs/launcher_patterns/g32r4_retrain.sh:8-10` |
| g32r4 | backward-policy | `scaling/runs/g32r4/backward-policy/lightning_logs/version_0/checkpoints/epoch=1-step=1642.ckpt` | `scaling/data/g32r4/backward.jsonl` *(idem)* | `jobs/launcher_patterns/g32r4_chain.sh:12` |
| g32r4 | forward | `lightning_logs/version_45/checkpoints/epoch=3-step=158564.ckpt` (repo-root logs) | `scaling/data/g32r4/forward.jsonl` | `jobs/launcher_patterns/train_fwd_lowlr_g32r4.py:22-24` |

"Checkpoint of record" = the path in the `protocol.checkpoints` block of
`scaling/results/<cfg>/comparison.json`.

### 1b. Why `backward.jsonl` and `backward.rust.jsonl` are not a choice at all

At g24r8 and g32r4 the two files are **byte-identical** (measured 2026-08-05):

```
g24r8/backward.jsonl == g24r8/backward.rust.jsonl   sha256 99d45b215e15f2d0…
g32r4/backward.jsonl == g32r4/backward.rust.jsonl   sha256 8c85faa12a50cf56…
g24r8/forward.jsonl  == g24r8/forward.rust.jsonl    (cmp, identical)
g32r4/forward.jsonl  == g32r4/forward.rust.jsonl    (cmp, identical)
```

`scaling/gen_data.py:75` writes the Rust-engine output to `<system>.rust.jsonl`
and the Python-engine output to `<system>.jsonl`; at these two rungs the `.rust`
file was produced first (mtimes: g32r4 `.rust` 07-15 20:18 vs `backward.jsonl`
20:32) and copied onto the default name that `scaling/README.md`'s pipeline and
every launcher chain read. So the trainers read Rust-engine content under the
Python-engine filename.

g24r4 has **no** `backward.rust.jsonl` at all (`ls scaling/data/g24r4/`): its
`backward.jsonl` is the concatenation of the eight `backward.shard*of8.jsonl`
Python-labeler shards.

### 1c. Independent numeric corroboration (the decisive check)

Lightning's step counter pins the training file exactly. `train.looped_pc`
batches **groups** (`looped_pc.py:242-246`), `train.policy_tf` batches
**decisions** (`policy_tf.py:226-229`); both use the config's `train` split
(boards 0-699, `scaling/configs.py:93-94`). Counting the train-split decision
groups in each corpus and dividing by the batch size reproduces every
checkpoint's steps/epoch to the unit:

| config | train groups in `backward.jsonl` | value bs → steps/ep | ckpt says | policy bs → steps/ep | ckpt says |
|---|---|---|---|---|---|
| g24r4 | 6612 | 4 → 1653 | 38019/23 = **1653** | 8 → 827 (ceil) | 1654/2 = **827** |
| g24r8 | 7884 | 4 → 1971 | 21681/11 = **1971** | 8 → 986 (ceil) | 29580/30 = **986** |
| g32r4 | 6565 | 2 → 3283 (ceil) | 52528/16 = **3283** | 8 → 821 (ceil) | 1642/2 = **821** |

Six independent exact matches. No other corpus in the tree produces these
counts.

### 1d. Verdict on `nn_labeler/jobs/twin_map.env`

`twin_map.env:5-7` currently reads

```
TWIN_CORPUS_g24r4=scaling/data/g24r4/backward.jsonl
TWIN_CORPUS_g24r8=scaling/data/g24r8/backward.rust.jsonl
TWIN_CORPUS_g32r4=scaling/data/g32r4/backward.rust.jsonl
```

**Correct as written.** g24r4 has only one candidate; g24r8/g32r4 name the
`.rust` file, whose bytes are exactly what the planners trained on. See OPEN-7
for the one way this could rot.

### 1e. Full-corpus facts (for sizing and for the twin's expected shape)

| config | records | unique instances | decision groups | train recs / groups | val recs / groups | max `cost_to_go` |
|---|---|---|---|---|---|---|
| g24r4 | 53,789 | 6,300 | 9,930 | 35,792 / 6,612 | 10,183 / 1,886 | 72 |
| g24r8 | 109,180 | 6,300 | 11,790 | 72,594 / 7,884 | 20,971 / 2,197 | 80 |
| g32r4 | 51,353 | 6,300 | 9,808 | 34,236 / 6,565 | 9,701 / 1,856 | 91 |

6,300 instances = 1,050 boards × 6. `gate_v2_and_twins.slurm:114` replays
graphs `0-349,350-699,700-1049`, i.e. the whole train+val+test span.

---

## 2. Train invocation

### 2a. Does `scaling/train.py` take an explicit data path? Yes.

Everything after `--` is forwarded verbatim to the underlying trainer
(`scaling/train.py:15,73,153-154`), and a **relative `--data` is resolved
against the repo root before the chdir into the run dir**
(`scaling/train.py:74-77`). So

```
python -m scaling.train --config g24r4 --system backward-policy \
    --run-dir scaling/runs/g24r4/backward-policy-nntwin \
    -- --data nn_labeler/results/twin_g24r4.jsonl --epochs 30 --batch-size 8
```

reads `supervised_valuenet/nn_labeler/results/twin_g24r4.jsonl`. **No symlink,
no data-dir shadowing, and no existing file is touched.** `--run-dir`
(`scaling/train.py:43-44`) keeps `lightning_logs/` out of the original run dirs
— the same mechanism `jobs/patterns/b2_retrain_one.slurm:86,93` uses.

Verified by replaying `scaling/train.py`'s own argparse over the exact command
strings the job emits (all three parse, `--data` rewrites to the absolute
in-tree path, `rest` forwards intact).

### 2b. The six commands, hyperparameters verbatim from the originals

Run from `supervised_valuenet/` with `PYTHONPATH=.`:

```bash
# g24r4 — g24r4_chain.sh:18-24
python -m scaling.train --config g24r4 --system backward-policy \
  --run-dir scaling/runs/g24r4/backward-policy-nntwin \
  -- --data nn_labeler/results/twin_g24r4.jsonl --epochs 30 --batch-size 8
python -m scaling.train --config g24r4 --system backward-value \
  --run-dir scaling/runs/g24r4/backward-value-nntwin \
  -- --data nn_labeler/results/twin_g24r4.jsonl --epochs 30 --warmup 0 \
     --batch-size 4 --max-per-group 16

# g24r8 — g24r8_chain.sh:10-12
python -m scaling.train --config g24r8 --system backward-policy \
  --run-dir scaling/runs/g24r8/backward-policy-nntwin \
  -- --data nn_labeler/results/twin_g24r8.jsonl --epochs 30 --batch-size 8
python -m scaling.train --config g24r8 --system backward-value \
  --run-dir scaling/runs/g24r8/backward-value-nntwin \
  -- --data nn_labeler/results/twin_g24r8.jsonl --epochs 30 --warmup 0 \
     --batch-size 4 --max-per-group 16

# g32r4 — policy from g32r4_chain.sh:12, value from g32r4_retrain.sh:8-10
python -m scaling.train --config g32r4 --system backward-policy \
  --run-dir scaling/runs/g32r4/backward-policy-nntwin \
  -- --data nn_labeler/results/twin_g32r4.jsonl --epochs 30 --batch-size 8
python -m scaling.train --config g32r4 --system backward-value \
  --run-dir scaling/runs/g32r4/backward-value-nntwin \
  -- --data nn_labeler/results/twin_g32r4.jsonl --epochs 25 --warmup 0 \
     --batch-size 2 --max-per-group 8
```

Everything NOT passed matters as much as what is:

* **lr** — the trainer default 3e-4 (`looped_pc.py:139`, `policy_tf.py:109`) in
  all six. Confirmed against every `hparams.yaml` of record: `lr: 0.0003`. The
  `--lr 1e-4` low-lr launcher pattern was **not** used at these rungs (it
  applies at ≥6 robots in the *B2* retrains, `b2_retrain_one.slurm:61,67`, and
  those runs' hparams do show `lr: 0.0001`).
* **value bins** — the default 50 (`looped_pc.py:138`) in all three.
  `scaling/run_config.py:38` recommends 96 at G=32, but the g32r4 checkpoint of
  record is 50-class
  (`scaling/runs/g32r4/backward-value/lightning_logs/version_1/hparams.yaml`),
  so the twin must be 50-class too. See OPEN-5.
* **no `--warm-start`** — the exact-arm value nets are cold. (The B2 arm warm-
  started; that is a different experiment, `b2_retrain_one.slurm:50-54`.)
* **no `--torch-seed`** — the chains set none. See OPEN-2.
* **`--warmup 0`** on every value net → `Curriculum` holds frac at 1.0
  (`looped_pc.py:218-224`).
* g32r4's value net uses the **retrain** recipe (25 epochs, bs 2, mpg 8), not
  the chain's (30 epochs, bs 4, mpg 16), because `version_1` — the retrain — is
  the checkpoint `comparison.json` benched.

`RR_RESUME=1` (`looped_pc.py:258-262`, `policy_tf.py:235-239`) makes a
walltime-killed fit resume from its own `last.ckpt`; the job sets it whenever a
`last.ckpt` exists in that run dir and refuses to resume if the corpus changed
underneath.

### 2c. How long the originals took, and what the twins should cost

**Originals** (hparams.yaml mtime = fit start → tfevents mtime = fit end; the
"dgx" host, A100):

| run | epochs | wall |
|---|---|---|
| g24r4 backward-value  | 30 | 3 h 48 m |
| g24r4 backward-policy | 30 | 1 h 17 m |
| g24r8 backward-value  | 30 | 7 h 21 m |
| g24r8 backward-policy | 30 | 1 h 35 m |
| g32r4 backward-policy | 30 | 3 h 53 m |
| g32r4 backward-value  | 25 | 7 h 47 m |

**Karolina rates**, measured on the same trainers from the B2 cap-20000
retrains (fit start → best-checkpoint mtime, divided by that checkpoint's step
count):

| net | batch × mpg | s/step | source |
|---|---|---|---|
| value  | 4 × 16 | 0.399 | `g24r8/backward-value-b2-cap20000` v1: 07-29 19:09:48 → `epoch=13-step=57918.ckpt` 07-30 01:35:06 |
| value  | 2 × 8  | 0.218 | `g32r4/backward-value-b2-cap20000` v2: 07-31 15:07:49 → `epoch=28-step=206567.ckpt` 08-01 03:37:41 |
| policy | 8      | 0.159 | `g24r8/backward-policy-b2-cap20000` v0: 07-28 21:03:28 → `epoch=16-step=32045.ckpt` 22:28:14 |
| policy | 16     | 0.793 | `g32r4/backward-policy-b2-cap20000` v1: 07-29 21:36:11 → `epoch=15-step=13024.ckpt` 07-30 00:28:21 |

**Twin projection** (exact-corpus step counts × the Karolina rate; the g32r4
policy rate is halved for the halved batch):

| step | steps/epoch | epochs | projected |
|---|---|---|---|
| g24r4 policy | 827   | 30 | **1.4 h** |
| g24r4 value  | 1653  | 30 | **5.5 h** |
| g24r8 policy | 986   | 30 | **1.3 h** |
| g24r8 value  | 1971  | 30 | **6.6 h** |
| g32r4 policy | 821   | 30 | **3.5 h** |
| g32r4 value  | 3283  | 25 | **5.0 h** |
| | | | **≈ 23 h training** |

These are upper-ish bounds: the twin corpora will probably be *smaller* than
their exact counterparts (OPEN-3), which shrinks steps/epoch proportionally.

---

## 3. Bench invocation

### 3a. `scaling/bench.py` does **not** produce `comparison.json`

Worth stating plainly because it is easy to assume otherwise:
`scaling/bench.py` **materializes the benchmark instance file** — it samples
instances per board and labels each with the exact move oracle's `d_star`,
writing `scaling/data/<cfg>/bench.jsonl`, `bench.solved.jsonl` and a sidecar
`.meta.json` with the sha256 (`scaling/bench.py:117-160`). Those files are
already pinned and must not be regenerated.

`comparison.json` is produced by **`eval.compare`**
(`scaling/run_config.py:116-122`, and the `protocol.command` string inside each
existing `comparison.json`).

### 3b. `eval.compare` arguments (`eval/compare.py:608-659`)

`--expansions` **defaults to 1200** and `--k` **defaults to 5**
(`eval/compare.py:611-612`), and every row in the study passes them explicitly
anyway. `--out` (default `eval/results/comparison.json`) and `--md` are what
redirect output, so an alternative name needs nothing but a different `--out`
— no existing file is at risk.

Checkpoints are **not** discovered by `eval.compare`; they are given by
`--backward-policy` / `--backward-value` / `--forward-ckpts`, and the protocol
block's `checkpoints` dict is just `{path: mtime}` for whatever was passed
(`eval/compare.py:681-684`). `scaling/run_config.py:48-54,137-147` is the only
auto-resolver, and it picks the newest non-`last.ckpt` under
`scaling/runs/<cfg>/<system>/lightning_logs/version_*/checkpoints/`.

### 3c. Which flags reproduce the exact arm

The three base rows were produced with `--backward-prefix-check` and **without**
`--backward-anytime`: their system key is
`"backward subgoal planner (prefix-check)"`, and `eval/compare.py:709-712`
builds that name from exactly that flag combination. Their recorded commands
(inside `scaling/results/<cfg>/comparison.json`) are e.g.

```
PYTHONPATH=. python -m eval.compare --instances scaling/data/g24r4/bench.solved.jsonl \
  --expansions 1200 --k 5 \
  --backward-policy scaling/runs/g24r4/backward-policy/.../epoch=1-step=1654.ckpt \
  --backward-value  scaling/runs/g24r4/backward-value/.../epoch=22-step=38019.ckpt \
  --forward-ckpts   scaling/runs/g24r4/forward/.../epoch=7-step=76360.ckpt \
  --backward-prefix-check --device cpu \
  --out scaling/results/g24r4/comparison.json --md scaling/results/g24r4/COMPARISON.md
```

So the twin command is that, minus the forward system, plus `--dump-moves`:

```bash
RR_GRID=24 RR_ROBOTS=4 RR_WALLS=108 RR_ENV_DIR=$PWD/environments_g24r4 \
PYTHONPATH=. python -m eval.compare \
  --instances scaling/data/g24r4/bench.solved.jsonl --expansions 1200 --k 5 \
  --backward-policy <nntwin policy ckpt> --backward-value <nntwin value ckpt> \
  --backward-prefix-check --forward-ckpts "" --device cpu --dump-moves \
  --out scaling/results/g24r4/comparison_nntwin.json --md /dev/null
```

`--dump-moves` is inert on the search: it only threads a `moves_out` list into
`strict_moves` (`eval/compare.py:323-329,406-410`;
`eval/realize.py:468-480` "Observational only"). It buys independent replay
certification via `eval.replay_validate`, which is the house completion gate.
The twin protocol block will therefore carry `dump_moves: true` where the older
exact rows carry no such key at all — a compare.py version difference, not a
search difference.

The frontier ("ungraded") row uses `bench.unsolved.jsonl` and the
`comparison_ungraded_*` name; `d_star` there is a placeholder, which
`eval/compare.py:673-680` detects and which suppresses regret.

### 3d. How the `b2retrained_cap20000` rows were actually produced

`jobs/patterns/track1_rows.slurm`, one lane per invocation:

```
BANK_MANIFEST=scaling/runs/b2_banked_cap20000.json \
  sbatch -J rr-t1-g24r8-grd jobs/patterns/track1_rows.slurm g24r8 graded
```

Mechanics worth copying (and copied into `twin_retrain.slurm`):

* checkpoints come from a **manifest**, not a guessed `version_N`
  (`track1_rows.slurm:15-20,66-75`);
* an alternate manifest **suffixes the output name** so two corpora's rows
  cannot collide (`track1_rows.slurm:101-111`) — this is where
  `comparison_b2retrained_cap20000.json` / `comparison_ungraded_b2retrained_cap20000.json`
  get their names;
* the instance file is `split -d -a 3 -l 8` into 8-instance chunks, run 8-wide
  with `xargs -P 8`, each chunk skip-if-exists and tmp+mv
  (`track1_rows.slurm:136-156`);
* the chunk directory is keyed on a sha of the two checkpoint paths, so a
  rerun after re-banking cannot merge stale rows (`track1_rows.slurm:126-133`);
* merge with `eval/merge_compare_shards.py --shards … --instances … --out …`;
* then `eval.replay_validate --compare <out> --env-dir <envdir>`, and a failed
  certification **quarantines** the merged file instead of leaving it where the
  resume gate would accept it (`track1_rows.slurm:164-183`).

Their flags were `--backward-anytime --backward-b2 --dump-moves`, i.e. the
extended language. The twins are BASE, so they take neither `--backward-b2`
nor `--backward-anytime`.

### 3e. Bench cost

Measured CPU-seconds of the exact arm at the identical protocol (sum of the
`seconds` field over the backward rows):

| lane | instances | CPU-s |
|---|---|---|
| g24r4 graded   | 232 | 1,810 |
| g24r8 graded   | 161 | 5,483 |
| g32r4 graded   | 175 | 1,789 |
| g24r8 frontier | 289 | 58,080 |
| g32r4 frontier | 275 | 4,014 |
| g24r4 frontier | 218 | ~3,600 (estimated: its B2 frontier is 10,764 s and B2:base at g24r4 graded is 5,440:1,810 = 3.0×) |

≈ 21 CPU-hours total; at `xargs -P 8` inside the job's 16 cores, ≈ **2.7 h
wall**, call it 4 h with headroom for retrained nets that rank worse and burn
more of the 1,200-expansion budget. g24r8's frontier lane is 75 % of it.

---

## 4. Schema check — descent output vs the exact backward corpus

Compared `nn_labeler/results/capgate_g24r4.jsonl` (3,695 records, the same
descent-replay format the twins will have) against
`scaling/data/g24r4/backward.jsonl` (53,789 records).

**Field sets:** the descent record is a strict **superset**.

* Shared, all 18: `env_id, target, target_robot, helpers, seg_start, seg_end,
  seg_support, mover_color, ctx_bottlenecks, ctx_supports, ctx_open_endpoints,
  cand_bottleneck, cand_support, cand_helper, cand_parent_support, cost_to_go,
  is_optimal, depth`.
* Exact-only: **none**.
* Descent-only (provenance, ignored by every loader): `n, label_source,
  ctg_certified, label_model, label_model_sha`
  (`nn_labeler/descent.py:248-256`).

**Every field the training path reads is present and identically shaped:**

| reader | fields read | present? |
|---|---|---|
| `nn/benchmark.py:30-36` `load` / `by_split` | `env_id` | yes |
| `nn/benchmark.py:39-56` `group_by_decision` | `env_id, target, target_robot[0], seg_start, seg_end, depth` | yes |
| `train/encode.py:236-256` `_node_features` (value net) | `seg_start, seg_end, cand_bottleneck, cand_support, cand_helper[0], helpers, ctx_open_endpoints, ctx_bottlenecks, ctx_supports` | yes |
| `train/looped_pc.py:80-90` `DenseDataset` | `env_id, cost_to_go, is_optimal, cand_bottleneck, cand_support, cand_helper[0], seg_start, seg_end` | yes |
| `train/policy_common.py:25-41` `_features` (policy net) | `seg_start, seg_end, seg_support, target_robot[0], helpers, ctx_open_endpoints, ctx_bottlenecks, ctx_supports` | yes |
| `train/policy_common.py:44-73` `_meta` | `helpers, cand_helper, cand_bottleneck, cand_support, cost_to_go, is_optimal, env_id, seg_start, seg_end` | yes |

Shapes/types match record-for-record: `target_robot = [[x,y], color]`,
`helpers = [[[x,y], color], …]` (3 entries at 4 robots), `cand_helper =
[[x,y], color]`, `seg_support`/`cand_parent_support` are `null`-or-`[x,y]` in
both (measured over the first 5,000 records of each: 4664 null / 336 list in
the exact corpus, 3498 null / 197 list in the descent output).

**Structural properties the trainers depend on:**

| property | exact g24r4 | descent g24r4 |
|---|---|---|
| decision groups | 9,930 | 790 |
| candidates per group | 5.42 | 4.68 |
| groups with **zero** `is_optimal` | 0 | 0 |
| groups with >1 `is_optimal` | 2,922 | 186 |
| `is_optimal` share | 25.5 % | 27.8 % |
| `cost_to_go` min/max/mean | 1 / 72 / 17.37 | 2 / 70 / 16.18 |
| `cost_to_go` > 49 (the 50-bin ceiling) | 0.244 % | 0.081 % |
| depth histogram (0/1/2) | 41772 / 9715 / 2302 | 2830 / 754 / 111 |

Zero groups without an optimal matters: `looped_pc._rank`
(`looped_pc.py:175-183`) silently **drops** any group with no `is_optimal`
member from the ranking loss. Descent guarantees at least one per emitted
decision by construction (`descent.py:246-250` sets `is_optimal = ctg ==
best_ctg` over the decision's certified candidates).

**Verdict: no converter, no shim, no field surgery.** Point `--data` at
`nn_labeler/results/twin_<cfg>.jsonl` and both trainers work unchanged. The
board-id splits also line up: descent replays graphs 0-1049 and
`scaling/train.py:85-86` rebinds `nn.benchmark.SPLITS` to the config's ranges
(train 0-699, val 700-899), so the twin's bench-board records (900-1049) sit in
`test` and are never trained on — exactly as in the exact corpus.

One semantic difference that is the *point* of the experiment, not a defect:
descent's `cost_to_go` is a **certified upper bound** (a realized greedy
completion), not the exact optimum, and `is_optimal` is argmin over those upper
bounds (`descent.py:11-16`).

---

## 5. Forward twins — feasibility only, nothing implemented

**What a forward record is** (`scaling/data/<cfg>/forward.jsonl`, built by
`move_planner/generate.py:72-82`):

```json
{"env_id": 40, "robots": [[18,11],[3,3],[4,12],[22,19]], "target": [18,23],
 "target_idx": 2, "cost_to_go": 9, "best_moves": [[2,1]],
 "legal_moves": [[0,0],[0,1],[0,2],[1,1],[1,2],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]],
 "depth": 0, "full": false}
```

It is a **per-primitive-state** record: full robot tuple, `cost_to_go` = the
**exact** optimal move count from that state (the value target),
`best_moves` = the complete optimal-move set (the policy target),
`legal_moves` = every legal `[slot, direction]`, and `full` = whether the
optimal set is complete (`move_planner/generate.py:10-13`; consumed at
`move_planner/encode.py:90-98` and `move_planner/net.py:66,81,172`).

**What descent retains.** Nothing usable. `descent.py:239` calls

```python
if strict_moves(env, state, done, log=_silent) is None:
```

purely as a boolean certification gate. `strict_moves` *can* hand back the
realized `[color, direction]` slide sequence via its optional `moves_out`
argument (`eval/realize.py:468-480`), but descent does not pass it, so the
sequence is computed and thrown away. The emitted record carries only the
18-field subgoal schema plus provenance — no move list, no per-state rows.

**What a converter would need to do.**

1. **Change `descent.py` to keep the moves.** Pass `moves_out=mv` in the
   certification call and store `mv` on the *winning* candidate's record (the
   one that becomes `plan` at `descent.py:258`). One line plus a field; it is
   the only unavoidable upstream edit.
2. **Replay the sequence into states.** Walk the realized slides from the
   instance's start state with `move_planner.state.apply_move`, emitting one
   record per intermediate state (robots tuple, target, `target_idx`, depth).
   Physics only; no oracle.
3. **`legal_moves` is free** — `move_planner.state.legal_moves` on each
   replayed state (`move_planner/generate.py:132`).
4. **`cost_to_go` becomes an upper bound**, not the exact optimum: the number
   of slides remaining in the realized sequence. That is the honest NN-labeled
   analogue and it is what the twin experiment is about, but note it is a
   *plan-realization* length, not a move-optimal length, so it is upper-bounded
   twice over (the plan may be suboptimal *and* the realization of that plan
   may be).
5. **`best_moves` is the hard part.** The exact labeler knows the whole optimal
   move set because it has an optimal-distance oracle over successors. A
   realized sequence certifies exactly **one** move per state, so the twin's
   `best_moves` would be a singleton and `full` would have to be `False`
   everywhere. `move_planner/net.py:172` treats `full` specially, so a corpus
   of all-`full=False` records is a materially different training signal from
   the exact forward corpus — the forward twin would **not** be the clean
   one-variable experiment the backward twin is. Making it clean would require
   scoring every legal successor with the labeler and taking the argmin set,
   i.e. a forward-flavoured descent that does not exist today.

**Recommendation:** treat the forward arm as out of scope for the headline
claim, state the headline as "backward planner, label provenance", and keep
every forward row in the study as the untouched control it already is.

---

## 6. OPEN — risks and unresolved items

**OPEN-1 · g24r4 has no exact frontier row.** `scaling/results/g24r4/` holds
`comparison.json` but **no `comparison_ungraded.json`** (g24r8 and g32r4 have
both). The job still produces `comparison_ungraded_nntwin.json` at g24r4
(~1 CPU-h), but it has no exact counterpart to be compared against. Either
drop that row from the claim or produce the exact-arm frontier row separately
with the exact-arm checkpoints and the same flags.

**OPEN-2 · Neither arm is seed-controlled.** The chains set no
`--torch-seed`, so the twin arm reproduces the recipe faithfully by also not
setting one — but the study has documented cold value retrains as
*seed-unstable* (`jobs/patterns/b2_retrain_one.slurm:3-4,50-54`, which is why
the B2 arm warm-started and pinned seed 11). A twin-vs-exact delta smaller than
the known seed spread (`analysis/seed_spread.py`,
`scaling/runs/b2_banked_cap20000_seed{21,37,53}.json`) is not interpretable.
The job supports `TWIN_TAG=-seed21 TWIN_SEED=21` for replicates that cannot
collide with the production row; **budget for at least one replicate before
claiming equivalence.**

**OPEN-3 · The twin corpora do not exist yet, so their size is unmeasured.**
Every timing above assumes twin ≈ exact in decision count. The one available
sample says otherwise: `capgate_g24r4` kept 524/600 instances and labeled 4.68
candidates per group vs the exact corpus's 5.42, i.e. plausibly **10-25 %
fewer training decisions**. That makes training *faster* than projected, but it
also means the twin arm trains on less data — a second confound alongside label
provenance. If a size-matched arm is wanted, it must be designed in (e.g.
subsample the exact corpus to the twin's group count); this job does not do it.
**Re-derive the walltime from the actual `wc -l` once the twins land.**

**OPEN-4 · The twin rows contain only a backward system.** The exact
`comparison.json` contains backward *and* forward. Any table must compare the
backward rows only, and must not read `comparison_nntwin.json` expecting a
forward key.

**OPEN-5 · 50 value bins vs the study's own 96-bin advice at G=32.**
`scaling/run_config.py:38` says 96 bins at G=32; the g32r4 checkpoint of record
is 50-class, so the twin matches at 50. Truncation measured: exact g32r4 has
0.611 % of records above 49 (max 91); the descent sample `capgate_g32r4` has
0.190 % (max 58). Low and *lower* on the descent side, so the choice is safe —
but it is a shared handicap of both arms, not a neutral one, and the descent
sample only covers bench boards 900-1049, not the train split.

**OPEN-6 · Which labeler produced the twins is decided at runtime.**
`gate_v2_and_twins.slurm:67-105` picks v1 or v2 by argmin agreement at g32r4
and banks the winner. The twin records carry `label_model` /
`label_model_sha`, and `twin_<cfg>.jsonl.manifest.json` carries `ckpt` /
`ckpt_sha`, so provenance is recoverable — but this job neither pins nor
verifies it. **Read the manifest and record the labeler identity in FINDINGS
alongside the result.**

**OPEN-7 · `backward.jsonl` == `backward.rust.jsonl` is a fact about today's
bytes, not an invariant.** They are separate inodes with identical content. If
either is ever regenerated they diverge silently and `twin_map.env` would then
name a file the planners never trained on. The descent manifest records
`instances_from`, which is the audit trail; consider adding the corpus sha to
it.

**OPEN-8 · The exact rows were produced by an older `eval/compare.py`.**
`comparison.json` for these three configs dates 2026-07-14/19/23; `compare.py`
has changed since (`25d50f5` d_star placeholder + `--dump-moves`, `933b93b`
slide accounting, `ac27efb`, and two B2-only commits). `933b93b`'s own message
records the instrumentation A/B as *identical* on 3 lanes and the B2 commits
cannot touch a base-vocabulary prefix-check run, so the risk is low — but it is
not zero, and the fully honest control is to **re-bench the exact-arm
checkpoints with today's code** into a separate file, e.g.

```
--backward-policy scaling/runs/g24r4/backward-policy/lightning_logs/version_0/checkpoints/epoch=1-step=1654.ckpt \
--backward-value  scaling/runs/g24r4/backward-value/lightning_logs/version_0/checkpoints/epoch=22-step=38019.ckpt \
--out scaling/results/g24r4/comparison_exactrebench.json
```

at ~9,000 CPU-s (graded) + ~66,000 CPU-s (frontier) for all three configs.
Not done here (this job creates the twin arm only).

**OPEN-9 · Cost vs the 16 h cap.** ≈ 23 h training + ≈ 4 h bench ≈ **27 h** of
work in a job capped at 16 h (the cooling reservation blocks anything longer,
`b2_retrain_one.slurm:16-19`). This is handled, not hidden: steps are
per-config and skip-if-exists, trainings resume from `last.ckpt`, bench chunks
are individually idempotent, and the job exits `NNLAB TWINRT PARTIAL` (rc 1)
until all 12 steps are complete. **Expect 2 submissions** — the first should
land g24r4 complete (~7 h) and most of g24r8 (~11 h more).
