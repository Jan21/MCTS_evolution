# Asset inventory (2026-08-17)

Checkpoint binaries in `assets/` are LOCAL COPIES, deliberately not
committed to git (repo convention: checkpoints live on /scratch + the /mnt
archive, git holds code/results/docs). If they are missing (fresh clone or
scratch purge), re-copy with the commands below or pull from the archive.

## assets/ (copied 2026-08-17, sources verified on disk)

| file | source (repo-root-relative) | what |
|---|---|---|
| `labeler_prod_v1_s11.ckpt` (+`.json`) | `supervised_valuenet/nn_labeler/banked/prod_v1_s11.ckpt` | THE size-free value net (labeler of record, sha dd0e7211b1cbd39f, FINDINGS 61). pe=none, works at any board size. |
| `g16r4_backward_policy_b1s21.ckpt` | `supervised_valuenet/scaling/runs/g16r4/backward-policy-b1-seed21/lightning_logs/version_0/checkpoints/epoch=24-step=10475.ckpt` | Supervised backward policy, g16r4, B1 base vocabulary, seed 21. |
| `g16r4_backward_value_b1s21.ckpt` | `supervised_valuenet/scaling/runs/g16r4/backward-value-b1-seed21/lightning_logs/version_0/checkpoints/epoch=12-step=10881.ckpt` | Matching value net. |
| `g24r4_backward_policy_exact.ckpt` | `supervised_valuenet/scaling/runs/g24r4/backward-policy/lightning_logs/version_0/checkpoints/epoch=1-step=1654.ckpt` | The g24r4 exact-arm headline pair (FINDINGS 67/74 baseline rows). |
| `g24r4_backward_value_exact.ckpt` | `supervised_valuenet/scaling/runs/g24r4/backward-value/lightning_logs/version_0/checkpoints/epoch=22-step=38019.ckpt` | Matching value net. |

More pairs (all configs, twin/deploy/seed arms, forward nets): under
`supervised_valuenet/scaling/runs/<cfg>/<system>[-<arm>]/lightning_logs/`.
Archive: `/mnt/proj1/open-37-42/petrhyner_archive/2026-08-16/` (runs,
corpora, banked ckpts; boards under `boards/`).

## Referenced in place (do NOT copy — import-coupled or huge)

- Code: `supervised_valuenet/{simulate.py, eval/, train/, scaling/,
  nn_labeler/}` — modules import each other and read env vars
  (`RR_GRID/RR_ROBOTS/RR_WALLS/RR_ENV_DIR`, see `scaling/configs.py`);
  run with `cd supervised_valuenet && PYTHONPATH=. python -m ...`.
- Rust engine: `supervised_valuenet/rust_datagen/` (build:
  `cargo build --release`; login node has internet, compute nodes do not).
- Benchmarks: `supervised_valuenet/scaling/data/<cfg>/bench.solved.jsonl`
  and `bench.unsolved.jsonl` — the pinned exams. NEVER regenerate.
- Corpora: `supervised_valuenet/scaling/data/<cfg>/backward.jsonl` (exact),
  `supervised_valuenet/nn_labeler/results/twin_*.jsonl` (NN twins),
  `deploy_g32r4.jsonl` (NN deployment), `beyond_UNVERIFIABLE_g{80,96}r4.jsonl`.
- Boards: `supervised_valuenet/environments_g*` (127 GB) — treat as
  read-only; make new boards with `nn_labeler/leanboard.py`.
- Result JSONs: `supervised_valuenet/scaling/results/<cfg>/comparison*.json`
  — the frozen baseline numbers. Read, never edit.

## Quick commands (from supervised_valuenet/, venv active)

```bash
# fresh lean boards at any size
python -m nn_labeler.leanboard --n 24 --robots 4 --walls 108 --ids 5000-5019 --out /tmp/boards_test
# label instances with the frozen labeler (greedy certified descent)
python -m nn_labeler.descent --config g24r4 --ckpt nn_labeler/banked/prod_v1_s11.ckpt \
    --graphs 900-905 --per-graph 5 --boards lean --device cuda --timeout 60 --out /tmp/lab.jsonl
# bench a planner pair on the pinned exam (chunked pattern: see
# nn_labeler/jobs/exact_replicate.slurm lines 56-86 for the full recipe)
python -m eval.compare --instances <chunk.jsonl> --expansions 1200 --k 5 \
    --backward-policy <BP.ckpt> --backward-value <BV.ckpt> \
    --backward-prefix-check --forward-ckpts "" --device cpu --dump-moves \
    --out out.json --md /dev/null
# validate solutions against physics
python -m eval.replay_validate --compare out.json --env-dir environments_g24r4
```
