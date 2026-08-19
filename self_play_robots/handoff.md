# self_play_robots — handoff (written 2026-08-19 ~07:40 CEST, session near context limit)

Read in this order: `PROBLEM.md` (brief) → this file → `FINDINGS.md` §1–§16 (all
results, sources, job ids) → `README.md` (module map) → `DESIGN.md` (as-built loop,
§4b = B2 vocabulary pivot) → `results/status.json`. Local report:
`report/selfplay.html` (regenerate with `python self_play_robots/report/gen_report.py`
from the repo root; LOCAL ONLY, never publish). Audit memos:
`results/audit_claims_2026-08-18.md`, `results/review_b2_pipeline_2026-08-18.md`,
`results/audit_findings_vs_files_2026-08-18.md`.

## Where the project stands (milestones per PROBLEM.md §8)

- **M0 PASS** (§2): arena wrapper reproduces recorded rows bit-exactly on Karolina;
  forward arm too (`spr/fwd`).
- **Ceiling study** (§3, §11, §14): exhaustive base-language moves/solve ceilings on
  every exam (`results/ceiling/*`); base subgoal language is far from d\* (regret
  ≥1.4/1.6) and the size-free planner SATURATES it everywhere (graded + frontier,
  16→32, 4 and 8 robots). B2 language ceilings also probed (frontier ones capped).
- **M1 PASS** (§4, §7): size-free pe=none policy (labeler-encoder init, lr 1e-4,
  grad-clip 1.0 — the plain recipe diverges) + value warm from the labeler; ONE mixed
  pair beats per-size supervised at 24×24 (215/232 vs 205, regret 2.43 vs 4.20), ties
  at 16×16, transfers zero-shot both directions; per-size family = supervised pairs.
- **M2 PASS** both families (§5, §8): MCTS (= certified enumeration of the top-5 tree,
  see §14 caveats) beats greedy and arena A\* on moves at identical solve sets.
- **M3 PASS / M4 on track in the B2 loop** (§10–§16): base-vocab loop is flat (saturated);
  B2 loop at g24r4: frontier 133→144→139→158→153 (it3/it4 vs it0 p≤3e-4), graded A\*
  223→228 solves at 53→26 expansions, MCTS regret 1.55→1.46; final nets beat the
  recorded supervised B2 rows (p<1e-4). Iteration 5 running (job 4698742 chained
  after 4698741). Zero-shot transfer of the loop's nets to 32×32/8 robots shown (§16).
- **M5 partial**: transfer shown; next = mixed-size B2 self-play curriculum (24+32,
  8 robots) — the specialization regret cost at 32/8r (§16b) is the thing to fix.
- **Forward arm** (§6 + `results/fwd_g24/`): F-M0 parity at 16 and 24; MCTS beats greedy,
  not A\* on quality; A\* is the efficient expert. Its self-play loop (`jobs/fwd_iter.slurm`)
  was never run — the subgoal loop had the headroom.
- Node-hours spent ≈ 10 (of ~783); everything is committed locally; `git push` fails
  intermittently (VS Code askpass socket, see memory) — retry `git push` first thing.

## Live/queued jobs at handoff
- 4698741 spr-b2-it4 (running; only the frontier-MCTS bench left) → 4698742 spr-b2-it5
  (chained, `auto` nets from iter4). Both write `results/selfplay/g24r4_b2_iter{4,5}/`.
- Nothing else queued. GPU nodes are reserved daily 10:00–18:00 (jobs must be ≤16 h;
  submit chains before 18:00; jobs starting exactly at a boundary sometimes die with
  signal 15 in <20 s — just resubmit).

## How to continue (concrete)
1. When 4698742 finishes: `PYTHONPATH=supervised_valuenet:self_play_robots python -m spr.trend --config g24r4 --tag _b2`,
   paired tests with `python -m spr.gate compare --a <it5 bench> --b <it0/it4 bench>`,
   append FINDINGS §17 (M4 verdict over 5 iterations), regenerate report, commit, push.
2. M5 curriculum: run the B2 loop with a mixed-size generation (needs a small change:
   `spr.selfplay` is one config per process → run two generation processes (g24r4 and
   g32r4 boards, ids ≥8000 each) and concatenate buffers with `spr.buffer --tag`; train
   with `--data g24r4=… --data g32r4=…` (spr.train handles mixed sizes; value batch at
   32×32 must stay ≤ 2 groups × 8 records — FINDINGS 63); bench at 24/32 graded+frontier
   and 8 robots with `jobs/bench_pair.slurm … "--vocab b2 --anytime"`.
3. Optional controls the audits asked for: per-size policy + constant value; a second
   seed of the mixed size-free pair (`jobs/m1_train.slurm mixed 37 …`).
4. Iteration job knobs: `VOCAB=b2 WORKERS=6 BOARDS=60 EXP=300 STOP=80 sbatch -J spr-b2-itK -t 06:00:00 jobs/selfplay_iter.slurm g24r4 K auto auto`
   (auto = nets/bench of iteration K−1). Bench-only of any pair: `jobs/bench_pair.slurm`.
   Fidelity gauge: `jobs/gauge_only.slurm` (exact engine budget 100k iters; for B2 the
   exact reference is weak — §14b — use certified benches as the instrument).

## Known rough edges
- Relative `--out` paths in `spr.gauge`/`spr.selfplay`/`spr.buffer` resolve against the
  repo root, `spr.arena` against cwd — pass absolute paths in jobs (jobs do).
- Transient CUDA OOM warnings in generation (root-all value passes × 6–8 workers) are
  caught per instance (2–5 instances/iteration lost).
- `eval.realize.prefix_key` cannot order park/by-reference plans → B2 runs without
  `--prefix-check` (matches the recorded B2 protocol).
- Frontier MCTS benches take ~1 h at 24×24 (budget binds); size the job walltime.
