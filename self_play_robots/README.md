# self_play_robots — module map and how to run

Brief: `PROBLEM.md` (authority). Assets: `ASSETS.md`. Results log:
`supervised_valuenet/FINDINGS.md` (max+1 numbering; this project's entries
start at §82). Local report: `report/gen_report.py` → `report/selfplay.html`
(reads result files only; never published). Status manifest the report and
this README refer to: `results/status.json`.

Everything imports the supervised stack in place (`PYTHONPATH=
supervised_valuenet:self_play_robots`, one board config per process); nothing
in `supervised_valuenet/` is modified.

## Package `spr/`

| module | what | calls (unchanged) |
|---|---|---|
| `arena.py` | the gating arena: chunked 8-wide bench → merge → replay-certify → parity vs recorded JSON; registered arms (M0) + ad-hoc arms; `--driver compare` (per-size nets) or `spr` (size-free nets / spr searches) | `eval.compare`, `eval/merge_compare_shards.py`, `eval.replay_validate` |
| `ceiling.py` | plan-language ceiling on the MOVES metric: exhaustive no-net subgoal search + strict realization vs d\* on a whole bench (`--vocab base|b1|b2`) | `skeleton.astar`, `eval.realize.strict_moves`, `park_repairs` |
| `nets.py` | `SizeFreePolicyNet` (PolicyTF minus `self.pos`, same pointer heads, `pe=none`), size-parametric policy featurization/meta, batching, `ValueAdapter`; re-exports `SizeFreeValueNet` | `nn_labeler.model.LoopedLayer/SizeFreeValueNet`, `nn_labeler.encode` |
| `train.py` | one trainer for both nets: mixed-size corpora (`--data CFG=PATH`), `--init` warm start, CollapseStop, ModelCheckpoint(min val_regret), save_last/RR_RESUME, `--splits` overrides, `--torch-seed` | `nn_labeler.dataset`, `nn_labeler.model.collate_groups` |
| `bench.py` | arena driver for size-free nets: `--search arena_astar` (imports `eval.compare._nn_astar_backward` — the arena loop itself), `spr_astar`, `mcts`, `greedy`; same row/payload schema as `eval.compare` | `eval.compare`, `eval.realize` |
| `search.py` | the searches: `expand` (1 expansion = policy pass + value pass over ≤k children), `greedy`, `astar` (f-mode child/parent, best-at-budget), `mcts` (PUCT, min-max normalized Q, min/mean backup, certified terminals, root Dirichlet noise), `Certifier` | `skeleton.astar`, `nn.generate._context/_fixed_g`, `eval.realize` |
| `selfplay.py` | generation: fresh lean boards → instances → MCTS → certified labels (18-field records + provenance) with worker pool; manifest + per-instance stats | `nn_labeler.leanboard`, `nn.generate.random_instance` |
| `gauge.py` | fidelity gauge: sample depth-0 decisions, exact-label them with the Rust engine, run `nn_labeler.audit_descent` → argmin agreement | `scaling.rust_bridge.EngineProc`, `nn_labeler.audit_descent` |
| `buffer.py` / `trend.py` / `tables.py` | replay-buffer window (`--tag _b2`), iteration trend table, markdown rows from payloads | — |
| `audit.py` | M5 zero-shot exact audit of a (policy, value) pair at ANY size: policy regret@k / recall@5, value argmin, the greedy pair decision, on exact-labeled decision corpora (32 test split; 40/48/56/64 `backward_audit.rust.jsonl`) | `nn_labeler.dataset/encode/model.collate_groups` |
| `gate.py` | milestone gates: M1 bars (3.5 solve / 6.6 optimality pts), paired A/B (McNemar on solved vectors, both-solved moves sign test) | — |
| `fwd/` | primitive-move arm (owner: both action spaces): `mcts.py` PUCT over slides on the forward MoveNet Guide (arena budget unit), `bench.py`/`arena.py` (run_forward row schema + parity), `selfplay.py` (move_planner_v2 record schema), `train.py` (warm-start MoveNet), `gate.py` shim; jobs `fwd_*.slurm` | `move_planner`, `move_planner_v2`, `eval.compare.run_forward` |

## Jobs (`jobs/`, all `-A open-37-42`, qgpu partitions, logs in `runs/spr/`)

- `m0_arena_parity.slurm` — M0 (job 4679719).
- `ceiling.slurm` — ceiling study, 4 arms (job 4679720).
- `m1_train.slurm <policy|value_warm|value_cold> [seed]` — M1 nets.
- `m1_bench.slurm <TAG> <POLICY_DIR> <VALUE_DIR> [device]` — M1 benches + gate.
- `selfplay_iter.slurm <cfg> <K> <P|auto> <V|auto>` — one loop iteration (env knobs; `VOCAB=b2`).
- `selfplay_mix_iter.slurm <K> <P|auto> <V|auto>` — M5 mixed-size B2 curriculum iteration
  (`CFGS="g24r4 g32r4 g24r8"`; per-config buffers, one size-free pair, `--batch-ref-n 24`).
- `bench_pair.slurm`, `gauge_only.slurm`, `audit_far.slurm <TAG> <P> <V> [--byref]`,
  `transfer_probe.slurm`, `*_ceiling.slurm`; `watch.sh` monitor.
- `fwd_iter.slurm <K> <CKPT|auto> [PREV_BENCH|auto] [CFG]` — one PRIMITIVE-MOVE
  loop iteration; `CFG` (default `g16r4`) picks the exam / anchor corpus /
  frozen forward reference row / base checkpoint and the `<cfg>_iter<k>` run and
  result dirs (`results/fwd_selfplay/`). Other `fwd_*.slurm`: the F-M0/F-M2 arms.

## Results (`results/`, small JSONs, committed)

`m0/<arm>.json`, `ceiling/<cfg>_<vocab>.json`, `m1/<tag>_<cfg>.json` (+ `.gate.json`),
`selfplay/<cfg>_iter<k>/...` (manifests, gauges, benches), `status.json`;
`fwd_selfplay/<cfg>_iter<k>/...` (the primitive-move loop's manifest, train
result, benches, gates), `fwd_m0/`, `fwd_m2/`, `fwd_g24/`.
Checkpoints/corpora: `runs/spr/` on scratch (+ `/mnt/proj1/.../petrhyner_archive/`), never git.
