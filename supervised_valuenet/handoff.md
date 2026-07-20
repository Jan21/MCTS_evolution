# Handoff — move-based Ricochet-Robots planner (supervised + self-play)

> **STALE (2026-07-07).** Superseded by `SOURCE_OF_TRUTH.md`. Describes the *forward*
> self-play as the goal; the current goal is the forward-vs-backward comparison and the
> scaling study.

Working dir: `/mnt/raid/data/Hyner_Petr/MCTS_evolution` (branch `supervised-full-planner`).
All work under `supervised_valuenet/`. Run modules from there with `PYTHONPATH=.`.
Shared DGX — always pin an EMPTY gpu (`CUDA_VISIBLE_DEVICES=<idx>`); check `nvidia-smi` first.

## The task
Flip the subgoal-DAG value net into a **move-based** planner: policy predicts
`(robot, direction)` (a Ricochet slide), value predicts the **optimal move-count-to-go**.
Supervised baseline first, then **self-play** (the real goal = self-play **from nothing**).
HARD RULE: the planner must be **pure NN at inference** — NO oracle/classical-solver
fallback in search. Oracle allowed ONLY for supervised training labels + eval regret
reference. (I once added a `--fallback`; user rejected it; reverted.)

## Packages
- `move_planner/` — SUPERVISED baseline (done, documented, pure NN).
  - `state.py` (moves), `oracle.py` (exact heuristic-A\* labeler + eval ref),
    `generate.py` (parallel data gen), `encode.py` (features), `net.py` (MoveNet:
    shared looped-transformer encoder + value head [HL-Gauss 64 bins] + policy head
    [robot-cell ⊕ **dest-cell** → 4×4 masked]), `evaluate.py` (nn_astar / greedy / benchmark).
  - `data/moves.jsonl` (153,625 records), `checkpoints/best.ckpt` (trained).
  - Docs: `README.md`, `SUMMARY.md`, `summary.html` (2-tab; also a claude.ai Artifact).
- `move_planner_v2/` — SELF-PLAY (DeepCubeA-style Approx. Value Iteration via the net's
  own A\* as Expert-Iteration expert; oracle-free training; reuses `net.py`/MoveDataset
  verbatim). `config.py`, `start_states.py` (reverse-scramble curriculum + HER),
  `selfplay.py` (target gen), `train_iterate.py` (loop + CLI), `DESIGN.md`, `README.md`.
  - Modes: warm-start (default) and `--from-scratch --k-start 1`.
  - Ckpts in `runs_warm/`, `runs_scratch/`, `runs_scratch_long/`.

## Data / splits (split by BOARD id — no leakage)
train boards 1000–1599 (92,316 recs) · val 1800–1999 (30,402) · test 2400–2599 (30,907).
~25 instances/board, ~6 states/instance, cost-to-go 1–13 (mean 3.9). Boards are fresh
`gen_grids` geometry saved as `environments/env_*.pkl` (the encoder needs the pkl per env_id).

## Results so far
- **Supervised** (450-inst test benchmark, boards 2400-2549 per_board 3 k5 iters1200 seed1):
  NN A\* 381/450 (**85%**), regret **0.354**, optimal 277/381 (73%). Offline val_mae 0.263,
  policy top1 0.960.
- **Warm self-play** (same 450-inst benchmark, CONFIRMED): NN A\* 439/450 (**97.6%**),
  regret **0.055**, optimal 95%. value-greedy 7%→51% (value ranking fixed). policy-greedy
  86%→69% (curriculum skews near-goal). Beats supervised.
- **From-scratch** (random init, run `runs_scratch/`, 16 iters, small 30-inst probe only):
  curriculum climbed K=1→16, gen solve_rate ~0.97-1.0 each iter, full-puzzle eval regret
  0.71(K4)→0.30(K12)→0.46(K16), test 60% regret 0.278. NOT converged. NOT yet on the full
  450-inst benchmark → see RUNNING below.

## Key findings (verified)
- Solve-rate bottleneck = **value-net off-path mis-ranking**, NOT k/completeness (k=16 was
  WORSE 80% vs 86.7%@k5; all failures are iteration "cap" on medium d*=5-8). More iters
  1200→4000 → 90%. Admissible blend max(value,h) = no-op (value already tighter). Self-play
  fixes it at the source (value-greedy 7%→51%).
- Policy head MUST read each move's slide-**destination** cell embedding (1-step lookahead);
  a static per-robot readout can't even overfit (top1 ~0.1 → 1.0 with dest).
- Curriculum ("K") = reverse-scramble depth: goal state → K random reverse moves → start
  ≤K from goal. K grows +1 when gen solve_rate ≥ 0.9. from-scratch K_start=1.
- Blind BFS over 4-robot states intractable → heuristic-A\* (relaxed_target_dist admissible).

## SIMPLIFIED SELF-PLAY (current, after review workflow w2cqiw18w)
Review verdict: paradigm right, from-scratch was over-built + mis-targeted by 3 faults:
(1) value target `min(T-j, net_est)` = downward ratchet toward the weak heuristic;
(2) near-goal-only data (reverse-scramble local for RR); (3) fine-tune-grade under-training
of a random net. FIX = ONE generator + plain ExIt + supervised-grade training. DELETED:
reverse-scramble, HER, DAVI, admissible floor, K-ladder, target-tighten, mix_supervised.
NOW (move_planner_v2 rewritten, much simpler):
- selfplay.py: value = plain remaining length T-j; policy = committed move; UNSOLVED dropped.
- start_states.py: `forward_walk_relabel` (random S0 + k target-biased forward slides;
  goal = target's final cell; solvable, oracle-free, DEEP with bias). sample_start_state =
  rand_mix_prob random eval-instances + rest biased-walk.
- config.py: walk_k_max=16, walk_target_bias=0.6, rand_mix_prob=0.25; supervised-grade
  train (lr 3e-4, policy_weight 1.0, epochs 6, replay 5).
- train_iterate.py: dropped K-ladder / guide_target / _load_mix.
EMPIRICAL (measured): random net solves 20% of RANDOM instances -> mean_plan_len 4.1
(cold-starts + DEEP, matches eval 3.9); target-biased forward-walk k=16 -> mean 3.6,
ctg>=5 0.28 (deep + cheap/guaranteed-solvable). Old reverse-scramble was ~1.2/0.02 (the bug).

## RUN: runs_scratch_v3 (GPU0, the simplified-design from-scratch attempt)
`$SCRATCH/run_scratch_v3.py`: from_scratch, instances_per_iter=800, rand_mix_prob=0.2,
astar_iters=1500, k_top=8, n_iters=8, epochs=6, eval_every=2. Watch [gen] mean_plan_len
(should be ~3+) + ctg_ge5_frac (~0.2+) + eval solve/regret climbing toward supervised 85%/0.354.
GOAL: within 1% of supervised (>=84%).

## KEY INSIGHT (superseded — reverse-scramble; kept for history)
Reverse-scramble (DeepCubeA's curriculum) does NOT make hard Ricochet-Robots puzzles:
measured optimal solve length stays ~1.3 (max 2-4) even at K=16, because RR slides aren't
reversible so the reverse-reachable set from the goal stays LOCAL. So scramble-only
from-scratch under-trains deep states. FIX (implemented): MIXED curriculum in
`start_states.sample_start_state` — scramble (K-ladder + cold-start) + random full-difficulty
instances (deep training via net-solve + HER); K advances off SCRAMBLE solve-rate only
(`selfplay.generate_iteration` returns n_scramble/n_random/rand_solve_rate). New knobs:
`cfg.scramble_prob` (0.6 in mixed run), `cfg.scramble_target_bias`. sample_start_state now
returns a 4-tuple (…, is_scramble).

## RUNNING JOBS
- **(A)** scramble-only from-scratch 450-inst benchmark on `runs_scratch/iter15.ckpt` (GPU1,
  PID 121010) — the fair baseline vs supervised. Log at OLD scratchpad
  `9a4cf68f-…/scratchpad/scratch_fullbench_iter15.log`.
- **(B)** scramble-only longer run — KILLED (superseded by the mixed run).
- **(C) NEW: MIXED-curriculum from-scratch run** (GPU2, PID 181453) →
  `runs_scratch_mixed/` (from_scratch, K_start=1, K_max=20, n_iters=22, scramble_prob=0.6,
  instances_per_iter=1000, eval_every=3). Log at NEW scratchpad
  `6445c7a0-…/scratchpad/run_scratch_mixed.log`. This is the real from-scratch attempt.

## (old running-jobs list, superseded)
Logs in the previous session scratchpad:
`/tmp/claude-1010/-mnt-raid-data-Hyner-Petr-MCTS-evolution/9a4cf68f-5c52-4ddc-ace6-e57504c2f6e9/scratchpad/`
- **(A)** `scratch_fullbench_iter15.log` — full 450-inst benchmark on `runs_scratch/iter15.ckpt`
  (fair, same set as supervised). Command was:
  `CUDA_VISIBLE_DEVICES=1 python -m move_planner.evaluate --ckpt move_planner_v2/runs_scratch/iter15.ckpt --boards 2400-2549 --per-board 3 --k 5 --astar-iters 1200 --seed 1`
- **(B)** `run_scratch_long.log` — longer from-scratch run → `runs_scratch_long/`
  (from_scratch, K_start=1, K_max=20, n_iters=26, instances_per_iter=1200, eval_every=3).
Check: `ps -eo pid,etime,cmd | grep move_planner`. Both slow (weak scratch net = more search).

## NEXT STEPS (continuation)
1. When (A) finishes → the **fair from-scratch 450-inst number** vs supervised 85%/0.354.
2. When (B) finishes → the **converged from-scratch** trend (K→20); benchmark its final
   ckpt on the same 450-inst set.
3. Update the self-play tab in `summary.html` (scratchpad `move_planner_summary.html` →
   re-run the standalone regen → `move_planner/summary.html`; republish Artifact same URL
   `https://claude.ai/code/artifact/785268c8-43ad-4f45-a5af-3e09c815d910`) with the fair numbers.
4. Open tuning: from-scratch under-trains deep states (reverse-scramble mean solved len ~1.2
   → near-goal skew). Deeper-coverage start states (mix random solvable instances / ensure
   target robot moves far) is the lever to close the gap to supervised.

## How to reattach
`ps` for the two PIDs / grep move_planner; read the two logs above; re-set a waiter
(`while ps -p <pid> ...; do sleep 30; done` as run_in_background) or just tail the logs.
Memory (persists): `~/.claude/projects/-mnt-raid-data-Hyner-Petr-MCTS-evolution/memory/`
(move-based-reformulation, no-oracle-at-inference, dgx-gpu-usage).
