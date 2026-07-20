# HANDOFF — move-based Ricochet-Robots planner (supervised + self-play)

> **STALE (2026-07-07).** Superseded by `supervised_valuenet/SOURCE_OF_TRUTH.md`. This
> handoff frames the *forward* self-play as "the goal"; the current goal is the
> forward-vs-backward comparison and the scaling study. Any running-job PIDs below are
> long dead.

Written 2026-07-07 evening. All work lives in **`supervised_valuenet/`** (run everything
from there with `PYTHONPATH=.`). Persistent memory (`~/.claude/.../memory/MEMORY.md`) also
carries the key findings. This file = how to reattach to running jobs and what's pending.

---

## 0. HOW TO REATTACH (do this first on relaunch)

Two background jobs were launched via `nohup` and survive session restarts. Check them:

```bash
cd /mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet
SCRATCH=/tmp/claude-1010/-mnt-raid-data-Hyner-Petr-MCTS-evolution/9a4cf68f-5c52-4ddc-ace6-e57504c2f6e9/scratchpad
ps -p 121010 -o pid=,etime=,cmd=   # (A) from-scratch full 450-inst benchmark  (GPU1)
ps -p 121012 -o pid=,etime=,cmd=   # (B) from-scratch LONG run K->20, 26 iters  (GPU2)
tail -20 $SCRATCH/scratch_fullbench_iter15.log   # (A) output
grep -E "iteration|NN A\* \(k=5\)|DONE" $SCRATCH/run_scratch_long.log | tail -30   # (B) trend
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

- If a PID is gone, the job finished (or died) — read its log for the result. Checkpoints
  are in the repo (persist regardless): `move_planner_v2/runs_scratch_long/iter*.ckpt`.
- If `$SCRATCH` was cleaned, the ckpts still exist — just re-run the benchmark (cmd in §4).
- The harness "waiters" from the old session are gone; re-poll with `ps`/`tail` or a new
  `while ps -p <pid>; do sleep 30; done` background wait.

---

## 1. TWO RUNNING JOBS (as of handoff)

**(A) Fair benchmark of the CURRENT from-scratch net** — PID 121010, GPU1.
`python -m move_planner.evaluate --ckpt move_planner_v2/runs_scratch/iter15.ckpt --boards 2400-2549 --per-board 3 --k 5 --astar-iters 1200 --seed 1`
→ the SAME 450-instance test set as supervised/warm, for an apples-to-apples number.
Log: `$SCRATCH/scratch_fullbench_iter15.log`. SLOW (weak net fails → long search; expect hours).

**(B) Longer from-scratch run to convergence** — PID 121012, GPU2.
`$SCRATCH/run_scratch_long.py`: `Config(from_scratch=True, K_start=1, K_max=20, n_iters=26, instances_per_iter=1200, epochs=3, batch_size=256, eval_boards=20, eval_per_board=2, eval_astar_iters=1200, eval_every=3, out_dir="move_planner_v2/runs_scratch_long")`.
Log: `$SCRATCH/run_scratch_long.log`. Ckpts: `move_planner_v2/runs_scratch_long/iter{0..25}.ckpt`.

---

## 2. IMMEDIATE PENDING TASKS (pick up here)

1. When (A) finishes: record the from-scratch **450-instance** NN-A* solved/regret/optimal.
2. When (B) finishes: benchmark its best/last ckpt on the SAME 450-inst set (cmd §4);
   this is the CONVERGED from-scratch number the user actually wants.
3. Update with the fair from-scratch number, in all three:
   - HTML self-play tab comparison (see §5 for the HTML rebuild recipe),
   - `move_planner_v2/README.md` (Results section),
   - memory `move-based-reformulation.md`.
4. USER'S FOCUS: **from-scratch (from nothing) is the goal**, warm-start is not interesting.
   If from-scratch underperforms, the likely cause is the curriculum's near-goal bias
   (see §6) — improving start-state coverage is the main lever.

---

## 3. PROJECT STATE (what exists, results)

**`supervised_valuenet/move_planner/`** — SUPERVISED baseline (DONE, documented).
Pure-NN planner: shared looped-transformer encoder + VALUE head (cost-to-go, HL-Gauss 64
bins) + POLICY head (robot×dir, reads each move's slide-DESTINATION cell = 1-step lookahead).
Data `move_planner/data/moves.jsonl` (153,625 records: train 92,316 / val 30,402 / test
30,907; split by env_id — train 1000-1599, val 1800-1999, test 2400-2599). Trained ckpt
`move_planner/checkpoints/best.ckpt` (val_mae 0.263, policy_top1 0.960).
**Benchmark (450 inst): NN A* 381/450 = 85% solved, regret 0.354, 277/381 optimal.**
Docs: `README.md`, `SUMMARY.md`, standalone `summary.html`. Files: state.py, oracle.py,
generate.py, encode.py, net.py, evaluate.py.

**`supervised_valuenet/move_planner_v2/`** — SELF-PLAY (Expert Iteration / DeepCubeA DAVI).
Reuses net/state/encoder/eval verbatim; only the LABEL SOURCE changes (net's own A* search,
no oracle in training — oracle ONLY for eval regret). Reverse-scramble CURRICULUM (start on
goal, apply K random reverse moves → start ≤K from goal; K grows when solve-rate ≥0.9) + HER
+ 1-step Bellman. Files: config.py, start_states.py, selfplay.py, train_iterate.py,
DESIGN.md, README.md. Both modes: warm-start (default) and `--from-scratch --k-start 1`.
Results so far (SAME 450-inst benchmark unless noted):
- WARM-start self-play: **439/450 = 97.6%, regret 0.055, 417/439 optimal** (BEATS supervised).
  Value-greedy 7%→51% (value ranking fixed). Policy-greedy 86%→69% (curriculum near-goal bias).
  ckpt `move_planner_v2/runs_warm/iter5.ckpt`.
- FROM-SCRATCH (random init): curriculum climbed K=1→16, gen solve_rate ~1.0/iter. Small-probe
  eval only so far (~60-80% on 30 inst, regret 0.28-0.71); NOT converged; full 450-bench = job (A).
  ckpt `move_planner_v2/runs_scratch/iter15.ckpt`. Longer run = job (B).

---

## 4. KEY COMMANDS (run from `supervised_valuenet/`, `PYTHONPATH=.`, empty GPU)

Canonical 450-instance benchmark (the comparison set — always these exact flags):
```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python3 -m move_planner.evaluate \
  --ckpt <CKPT> --boards 2400-2549 --per-board 3 --k 5 --astar-iters 1200 --seed 1
# reports NN A* / greedy(pol) / greedy(val): solved / mean_regret / optimal / fail
```
Supervised demo: `python -m move_planner.evaluate --ckpt move_planner/checkpoints/best.ckpt --demo --boards 2400`
Self-play from scratch: `python -m move_planner_v2.train_iterate --from-scratch --k-start 1 --n-iters 26 --instances-per-iter 1200 --device auto` (eval_boards/eval_every set via Config in a launcher script, not CLI).
Retrain supervised net: `python -m move_planner.net --data move_planner/data/moves.jsonl --epochs 20 --patience 4 --batch-size 128`

---

## 5. THE HTML (2-tab summary + interactive demo)

- Artifact URL (redeploys to same URL): **https://claude.ai/code/artifact/785268c8-43ad-4f45-a5af-3e09c815d910**
- Repo standalone: `supervised_valuenet/move_planner/summary.html`.
- Source (session scratchpad, may be gone on relaunch): `$SCRATCH/move_planner_summary.html`.
  If gone, rebuild from `supervised_valuenet/move_planner/summary.html` (it's the same content,
  just wrapped in a full <html> doc). Tab1 = supervised (+ interactive board demo playing real
  NN solutions, data in `$SCRATCH/demo.json`, exporter `$SCRATCH/export_demo.py`). Tab2 = self-play.
- To update: edit the HTML, then `Artifact(file_path=<scratchpad html>, url=<same URL>)`, and
  regenerate the repo standalone (strip to head/body, wrap in <!doctype html>...). To update the
  self-play tab comparison with the fair from-scratch number, edit the results table in the
  `id="tab-selfplay"` section.

---

## 6. CONSTRAINTS & KNOWN ISSUES (important)

- **NO ORACLE AT INFERENCE.** Planner must be pure NN. Oracle = supervised labels + eval
  reference ONLY. (User rejected a `--fallback` oracle crutch; it was removed.)
- **DGX is SHARED** — always pin an EMPTY gpu (`nvidia-smi` first). GPUs 0/4 often busy.
- **Solve-rate bottleneck = VALUE mis-ranking** (off-path blind spot), NOT k/completeness
  (k=16 was WORSE than k=5). More iters 1200→4000 → 90%. Admissible blend was a no-op.
  Self-play fixes it (warm 97.6%). This is the core empirical finding.
- **From-scratch curriculum near-goal bias**: reverse-scramble yields mean solved length ~1.2
  (~59% of scrambles leave target on/near goal), so deep far-from-goal states are under-trained.
  This likely caps from-scratch full-puzzle performance. FIX DIRECTION: better start-state
  coverage (ensure the target robot actually moves far; or mix in solvable random instances
  once the net is strong; tune scramble_prob<1.0). This is the key open research lever.
- Self-play fine-tuning hurts policy-greedy (near-goal bias) but helps value+A* a lot.
- Next upgrades documented in `move_planner_v2/DESIGN.md`: PUCT-on-cost if A*-expert plateaus.

---

## 7. ENV
Python 3.11 (`~/.conda/envs/ph_main`), torch 2.8+cu126, pytorch-lightning 2.5. 128 CPUs
(shared, often loaded). `environments/env_*.pkl` = board slide-graphs (gitignored; generated
on the fly by `move_planner.generate` / `nn.gen_grids`); the encoder needs a pkl per env_id.
