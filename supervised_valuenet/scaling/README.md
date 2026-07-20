# Scaling study: grid size x robot count

Harness for comparing the two planners -- the forward move planner
(`move_planner/`) and the backward subgoal planner (`skeleton/` + `train/`) --
across board configurations. Each configuration fixes a grid side length G, a
robot count R (target robot included) and an interior wall count W (stock
density scaled by area: `round(48 * (G/16)^2)`).

## Configurations

| name  | grid | robots | walls | boards dir           | notes |
|-------|------|--------|-------|----------------------|-------|
| g16r4 | 16   | 4      | 48    | `environments/`      | legacy baseline: the pre-existing 16x16 pipeline (boards, splits and `eval/data/bench450.jsonl` bench boards 2400-2549); read-only here |
| g16r6 | 16   | 6      | 48    | `environments_g16r6/`| robot axis |
| g16r8 | 16   | 8      | 48    | `environments_g16r8/`| robot axis |
| g24r4 | 24   | 4      | 108   | `environments_g24r4/`| grid axis |
| g24r8 | 24   | 8      | 108   | `environments_g24r8/`| grid + robot axis |
| g32r4 | 32   | 4      | 192   | `environments_g32r4/`| stretch |

Non-legacy configs own board ids 0-1199: train 0-699, val 700-899, test/bench
900-1049 (1050-1199 spare). The registry lives in `scaling/configs.py`
(`BoardConfig`, `CONFIGS`, `env(cfg)`).

## Configuration mechanism

One config per process. A config is selected purely by four environment
variables read by repo modules at import time: `RR_GRID`, `RR_ROBOTS`,
`RR_WALLS`, `RR_ENV_DIR`. With none set, every module behaves exactly as the
legacy 16x16 pipeline. The harness entrypoints set these themselves (in-proc
before importing, or in the child's environment when they spawn labelers);
only `eval.compare` needs the `RR_*` prefix written into its command line.

## Pipeline (per config)

`python -m scaling.run_config --config <name>` prints the full ordered
command sequence; `--execute [--gpu N]` runs it. Stages (run from the repo
root, `PYTHONPATH=.`):

1. **Boards** -- `scaling.gen_boards --config <name>`: full-schema boards via
   `nn.gen_grids.make_board`, seeded `random.Random(idx)` per id (byte-equal
   to `nn.gen_grids --seed 0`). Never overwrites; shardable.
2. **Data** -- `scaling.gen_data --config <name> --system {backward,forward}`:
   spawns the labeler in a subprocess carrying the config env. Backward runs
   `scaling.backward_label` (the `nn.generate` rollout, `--max-candidates 14`,
   plus a mandatory 120 s SIGALRM per-instance wall-time guard -- the capped
   rollout can wander on larger boards). Forward runs `move_planner.generate`
   unmodified (`--score-candidates` recommended). Boards must exist first.
3. **Training** -- `scaling.train --config <name> --system
   {backward-value,backward-policy,forward} -- <trainer args>`: wraps the
   unmodified trainers (`train.looped_pc`, `train.policy_tf`,
   `move_planner.net`), rebinding `nn.benchmark.SPLITS` to the config's board
   ranges (the stock constant encodes legacy ids) and scoping lightning
   outputs to `scaling/runs/<name>/<system>/`. Checkpoints self-describe
   grid/robots in hparams, so they reload correctly anywhere.
4. **Bench instances** -- `scaling.bench --config <name> --per-board 3 --seed
   1`: one shared instance file per config (`scaling/data/<name>/bench.jsonl`),
   sampled exactly like `eval/bench_instances.py`. d\* comes from the exact
   move oracle under an expansion cap and a wall-time cap; a failed oracle
   keeps the instance with `d_star=null` (the failure rate is itself a
   scaling datapoint, recorded in the sidecar meta). A `bench.solved.jsonl`
   companion holds the d\*-labelled subset.
5. **Comparison** -- `eval.compare` on the config's instance file with the
   config's env prefix, writing `scaling/results/<name>/`.

## Budget matching

The head-to-head follows the conventions documented in `../COMPARISON.md`:
both systems search the SAME instance file with the same per-instance
expansion cap (default 1200) and top-k proposal width (default 5); one
expansion costs one policy pass plus one batched value pass in either system;
quality is primitive moves vs the stored d\* (oracle used only for that
reference label and for training data, never at inference). Report solve
rate over all bench instances; regret only over instances with d\* known.

## Resource guidance (A100-40GB, measured)

Looped value net memory scales ~linearly with in-flight records
(`batch_size x max_per_group`), ~0.32 GB/record at G=24 and ~0.85 GB at G=32:

| grid | backward value              | backward policy | value bins |
|------|-----------------------------|-----------------|------------|
| 16   | defaults (bs 8 x mpg 32)    | defaults        | 50 (default) |
| 24   | `--batch-size 4 --max-per-group 16` | defaults | 50 |
| 32   | `--batch-size 2 --max-per-group 16` | bs <= 16 | 96 (wrapper `--num-classes`) |

GPU stages are printed with a `CUDA_VISIBLE_DEVICES=<FREE_GPU>` placeholder:
check `nvidia-smi` and pin an idle GPU at launch time (shared machine).

## Known limitations

- `eval/realize.py` realizes plans with 16x16 physics only, so the backward
  planner cannot yet be scored at G != 16; `run_config` emits `--skip-backward`
  for those configs until realization is generalized. Backward rows at G=16
  (all robot counts) are unaffected.
- `eval.compare` computes `moves - d_star` and cannot take `d_star=null`;
  run it on `bench.solved.jsonl` whenever the bench meta shows
  `n_oracle_failed > 0` (solve rate over the full set can still be computed
  from the raw rows of both files).
- Board dirs are not self-validating: keep one directory per config (the
  registry enforces distinct defaults) and never point `RR_ENV_DIR` across
  configs.
