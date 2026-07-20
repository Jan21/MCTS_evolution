# move_planner_v2 — Self-Play (plain Expert Iteration) for the Move Planner

## 0. TL;DR

**The current net's own budgeted A\* is the expert; the net is the apprentice.** On a
generated puzzle the net's A\* (`evaluate.nn_astar`) either solves it or it doesn't. When
it solves, every decision state on the found path becomes a training record — value =
the plain **remaining path length**, policy = the **committed move**. When it fails, the
instance is **dropped**. Train the net on those records, repeat. That's the whole loop.

No oracle labels are ever used for training; `oracle.solve`/`optimal_cost` appear **only
at eval** to measure regret. This maximises reuse: **100% of `net.py` (both heads,
HL-Gauss value loss, masked policy loss) and `MoveDataset`/`collate` are reused
verbatim**, because self-play emits records in the *exact same JSONL schema* the
supervised generator emits. The only new code is (a) producing targets from the net's own
search instead of the oracle, (b) generating solvable start states without an oracle, and
(c) the outer iterate loop.

Two entry modes, same code: **from scratch** (`--from-scratch`, a random-init net, pure
self-play, no oracle ever) and **warm start** (fine-tune `move_planner/checkpoints/best.ckpt`).
The headline result is from-scratch: from random weights it reaches **87.6% solved /
regret 0.213** (16 iters), *beating* the supervised baseline (85% / 0.354), no oracle, no
tricks. See `../RESULTS.md`.

> **What this deliberately is NOT.** No MCTS/PUCT (A\*-as-expert dominates at branching
> ≤16 with a learned cost-to-go value). No reverse-scramble curriculum, no K-ladder, no
> HER, no DAVI/1-step-Bellman fallback, no target network, no admissible floor, no
> value-target tightening. An earlier design had all of those; they were removed because
> the plain loop is simpler *and* works better. The history is kept honest in §5.

## 1. MDP recap

- **State** `s = positions`: a hashable `tuple[(x,y), …]` of the 4 robot cells in
  `COLOR_ORDER = [Red, Blue, Green, Yellow]` slot order (Markov — the A\* closed-set key).
  A goal-conditioned instance also carries `target_idx ∈ {0..3}` (which robot must arrive)
  and `target = (x,y)` (the goal cell), plus an `env_id` selecting the board (walls).
- **Action** `a = (robot_slot, dir)`, `dir ∈ {0..3}` = up/down/left/right. The chosen robot
  slides until a wall or another robot stops it (`state.apply_move`); no-op slides are
  illegal. ≤16 actions, typically 6–10 legal (`state.legal_moves`).
- **Transition**: deterministic, known, cheap (`state.apply_move` / `legal_moves`).
- **Cost**: 1 per move. **Goal**: `state.is_goal(s, target_idx, target)`.
- **Cost-to-go** `J*(s)` = optimal number of remaining moves. `MoveNet`'s value head is a
  cost-to-go regressor: HL-Gauss over 64 bins, `net._value(logits) = E[bin]`.
- **env_id / board-pkl constraint**: the encoder reads the per-board slide-graph from
  `environments/env_{id}.pkl`. **Every self-play state must carry an env_id whose pkl
  exists.** Splits: train `1000–1599`, val `1800–1999`, test `2400–2599`. Self-play draws
  boards from the **train** range only; eval on val/test. `Config.train_ids()` filters to
  ids whose pkl is present.

## 2. Why this paradigm

**Expert Iteration / ExIt** (Anthony et al. 2017): the search is the expert, the net is the
apprentice. Our expert is a budgeted `evaluate.nn_astar` given a **larger** budget than the
eval planner (`k_top=8`, `astar_iters=4000` vs eval `k=5`, `iters≈1500`), so it returns a
plan *better* than the greedy net — that gap is the learning signal. The search also gives
an **on-policy** state distribution: the states the net actually meets at runtime, which is
exactly where its value estimates need to be correct — the value net learns at runtime, on
the states its own search visits.

Rejected: MCTS/PUCT (we already have an exact model and a cost-to-go value — A\* is the
right expert at this branching factor; MCTS collapses to A\* once value is in cost units);
MuZero (learning a model we already have); DAVI 1-step Bellman fallback and HER (extra
machinery that the plain drop-unsolved loop makes unnecessary — see §5).

## 3. The expert (search)

`evaluate.nn_astar(guide, env_id, start, tidx, target, wr, wd, k, max_iters)` is reused
**verbatim**: best-first on `f = g + value(child)`, policy top-k expansion, `best_g` closed
set; returns `(cost, path)` with `path = [(slot,dir), …]` or `(None, None)` if the budget
is exhausted. Self-play calls it with `k=cfg.k_top`, `max_iters=cfg.astar_iters`.

```
cost, path = nn_astar(guide, env_id, start, tidx, target, wr, wd,
                      k=cfg.k_top, max_iters=cfg.astar_iters)
if path is None:                      # unsolved -> drop the whole instance
    return [], False
states = path_to_states(start, path, wr, wd)   # replay plan -> [start, ..., goal]
return exit_records(env_id, states, path, tidx, target, wr, wd), True
```

## 4. Targets (`selfplay.exit_records`)

For a **solved** instance with plan length `T` and visited states `s_0..s_T` (`s_T` = goal):

- **Value = plain remaining length.** For each decision state `j = 0..T-1`:
  `cost_to_go(s_j) = clamp(T - j, 0, 63)`. A Monte-Carlo return of the length the search
  actually found. Early plans are loose upper bounds, which keeps the value *high* → the
  A\* stays optimistic and keeps finding solutions; as the net improves, `T` shrinks toward
  optimal, so the labels self-tighten without any explicit `min`/target-net bookkeeping.
  A **goal-anchor** record `s_T` with `cost_to_go = 0` pins `J(goal) = 0`.
- **Policy = the committed move.** `best_moves(s_j) = [path[j]]` (one-hot over the legal
  set, via `encode.policy_target`). The A\* move is the search-improved choice, so this is a
  valid policy-improvement distillation.

For an **unsolved** instance: emit **nothing** and drop it. No HER, no Bellman backup. The
solvable frontier then ratchets outward on its own (§5.2).

**Record schema (identical to the supervised generator — reused by `MoveDataset` verbatim):**
```
{ "env_id": int, "robots": [[x,y]×4 in COLOR_ORDER], "target": [x,y], "target_idx": int,
  "cost_to_go": float, "best_moves": [[slot,dir], …], "legal_moves": [[slot,dir], …],
  "depth": int, "full": false }
```
`full=false` for all self-play records (it only gates the offline `val_policy_top1` metric,
which needs a ground-truth optimal set we don't have — we track **regret vs oracle** instead).

**Loss (unchanged, `MoveNet.training_step`):**
```
L = HLGauss_CE(value_logits, cost_to_go)  +  policy_weight · masked_policy_CE(policy_logits, legal, ptar)
```
`policy_weight = 1.0` (supervised-grade — a random-init net needs a strong policy signal,
not the timid `0.4` an already-good warm net would use).

## 5. Start states — the oracle-free curriculum (`start_states.py`)

A random net can't solve a full 10-move puzzle: its search finds nothing, so there's
nothing to learn from. Two ingredients fix this with **no explicit difficulty schedule**.

### 5.1 Generator — target-biased forward-walk + relabel

Ricochet slides are **not reversible**, so the DeepCubeA reverse-scramble trick fails here
(reverse-reachable states stay local to the goal — measured optimal ~1.3 even at depth 16,
so the net never trains deep). The fix is a **forward** walk:

```
sample a random 4-robot board S0; apply k random forward slides (biased toward moving the
TARGET robot); set the GOAL = the cell the target robot ends up on.
```

`forward_walk_relabel(env_id, k, rng, wr, wd, target_bias=0.6)`: the walk is itself a
witness that a length-≤`k` solution exists (replaying it lands the target on the goal), so
`k ~ U(1, walk_k_max)` is an **oracle-free, genuinely non-local** difficulty dial — small
`k` = near-goal (bootstraps a random net), large `k` approaches full difficulty. The
`target_bias` matters: an unbiased walk moves the target only ~k/4 times and stays shallow
(mean optimal ~2.5); biasing toward the target makes the puzzle travel. Returns `None`
(dropped) if the target robot never displaced. The target always stops on the goal via a
slide, so the goal is guaranteed stoppable — no reachability filter needed.

### 5.2 Net-solvability = the implicit curriculum

The net's A\* tries each puzzle; **solved → train on it, unsolved → drop it**. As the net
gets stronger it solves deeper puzzles, so it trains on deeper puzzles next round — the
solvable frontier widens by itself. No K-ladder, no solve-rate gate. The falsification
signal is `stats["ctg_ge5_frac"]`: the deep-state (cost-to-go ≥ 5) share of the training
data must climb each iteration toward the supervised ~38% tail (measured 0.01 → 0.07 → …).

### 5.3 Random eval-distribution mix-in

With probability `cfg.rand_mix_prob` (0.25) an instance is drawn as a **raw random
solvable instance** (`sample_solvable_instance`, mirroring `evaluate.sample_instance`'s
cheap `relaxed_target_dist` reachability filter — **not** an oracle solve) instead of a
forward-walk. This covers the deep tail of the true test distribution that short walks
under-sample.

## 6. Modules (all import — never fork — `move_planner.*` and `train.*`)

```
move_planner_v2/
  DESIGN.md          # this file
  config.py          # Config dataclass — all hyperparameters in one place
  start_states.py    # oracle-free forward-walk start states + random mix-in
  selfplay.py        # net-guided A* expert -> ExIt records (the core new code)
  train_iterate.py   # outer loop: generate -> train -> eval regret; from_scratch/warm; main()
```
Reused verbatim (imported): `state.{legal_moves, apply_move, is_goal, COLOR_ORDER}`;
`encode.{MOVE_CHANNELS, policy_target, …}`; `net.{MoveNet, MoveDataset, collate}`;
`evaluate.{Guide, nn_astar, benchmark}`; `oracle.{relaxed_target_dist, INF}` (filter only —
never `solve`/`optimal_cost` for training); `train.encode.walls_for`; `nn.gen_grids.GRID`.
No `dataset.py` is needed — the self-play record schema equals the supervised schema.

### 6.1 `config.py`
`@dataclass Config` holding every hyperparameter (§8). `train_ids()/val_ids()/test_ids()`
return the pkl-backed env_id lists.

### 6.2 `start_states.py`
- `forward_walk_relabel(env_id, k, rng, wr, wd, target_bias=0.6) -> (positions, tidx, goal) | None`.
- `sample_solvable_instance(env_id, rng, wr, wd, max_try=200) -> (positions, tidx, target) | None`
  — raw eval-distribution instance, reachability-filtered (no oracle solve).
- `sample_start_state(env_id, cfg, rng) -> (positions, tidx, target) | None` — dispatch
  random-mix vs forward-walk by `cfg.rand_mix_prob`.
- `train_board_ids(cfg) -> list[int]`.

### 6.3 `selfplay.py`
- `path_to_states(start, path, wr, wd, size=GRID) -> list` — replay a plan to its states.
- `exit_records(env_id, states, path, tidx, target, wr, wd) -> list[dict]` — solved path →
  per-decision (value = remaining length, policy = committed move) + goal-anchor record.
- `play_instance(guide, env_id, start, tidx, target, wr, wd, cfg) -> (records, solved)` —
  run the expert; solved → ExIt records, unsolved → `([], False)`.
- `generate_iteration(guide, cfg, rng) -> (records, stats)` — loop `cfg.instances_per_iter`
  over sampled (board, start); `stats = {solve_rate, n_records, mean_plan_len, n_instances,
  ctg_ge5_frac}`.

### 6.4 `train_iterate.py`
- `build_fresh_ckpt(cfg, path) -> str` — save a random-init `MoveNet` as a ckpt so the loop
  can `Guide`/`load_from_checkpoint` it exactly like a warm ckpt (from-scratch mode).
- `train_on_records(base_ckpt, records, cfg, out_ckpt) -> str` — `MoveNet.load_from_checkpoint`,
  retarget `hparams.lr`/`policy_weight`, `pl.Trainer(max_epochs=cfg.epochs).fit(...)` on
  `MoveDataset(records)`, save `out_ckpt`.
- `eval_regret(ckpt, cfg, split="val") -> None` — `evaluate.benchmark` on a small board
  probe; prints solved / mean_regret / %optimal for NN A\*, greedy(pol), greedy(val).
- `iterate(cfg) -> None` — the loop (§7). `main()` — argparse → `Config` → `iterate`.

## 7. The iterate loop

```
rng = Random(cfg.seed);  buffer = deque(maxlen=cfg.replay_iters)
ckpt = build_fresh_ckpt(...)  if cfg.from_scratch  else cfg.base_ckpt   # + warm-start baseline eval
for it in range(cfg.n_iters):
    guide       = Guide(ckpt, device)                       # apprentice = A* prior + leaf value
    recs, stats = generate_iteration(guide, cfg, rng)       # net's own A* labels moves
    buffer.append(recs)
    train_recs  = concat(buffer)                            # replay window (last replay_iters)
    ckpt        = train_on_records(ckpt, train_recs, cfg, out=f"{out_dir}/iter{it}.ckpt")
    if (it+1) % cfg.eval_every == 0: eval_regret(ckpt, cfg, "val")   # regret vs oracle
eval_regret(ckpt, cfg, "test")                              # final report on 2400-2599
```

Fine-tune forward from the latest ckpt each iteration, with a `replay_iters`-deep window of
recent records to resist forgetting. No target network, no promote-on-improve gate — the
loop is deliberately plain.

## 8. Hyperparameters (defaults in `Config`)

| group | value |
|---|---|
| mode | `from_scratch=False` (pass `--from-scratch` for pure self-play); `base_ckpt=move_planner/checkpoints/best.ckpt` |
| boards | train `(1000,1599)`, val `(1800,1999)`, test `(2400,2599)` |
| iters | `n_iters=18`, `instances_per_iter=8000` |
| generator | `walk_k_max=16`, `walk_target_bias=0.6`, `rand_mix_prob=0.25` |
| expert search | `k_top=8`, `astar_iters=4000` (> eval budget) |
| value target | `value_clamp=63` (plain remaining length, no floor/tighten) |
| replay | `replay_iters=5` |
| training | `epochs=6`, `batch_size=256`, `lr=3e-4`, `policy_weight=1.0`, `num_workers=8` (all supervised-grade) |
| eval probe | `eval_boards=25`, `eval_per_board=2`, `eval_astar_iters=1500`, `eval_every=1` |
| value head | inherited from ckpt: `num_classes=64`, `d_model=192`, `recurrence=12`, `in_channels=14` |
| misc | `seed=0`, `device=cuda\|auto` |

## 9. What went wrong first (honest history)

The first design was over-built; each of these was measured to be unnecessary or harmful
and removed:

- **Reverse-scramble curriculum** — assumes reversible moves; Ricochet slides aren't, so
  scrambled starts stayed ~1 move from the goal (measured optimal ~1.3 even at depth 16).
  The net never trained on deep states → capped at 66%. Replaced by the **forward-walk**
  generator (§5.1).
- **Value-target tightening** `min(found_length, net_estimate)` — a downward ratchet toward
  the net's own wrong belief. Replaced by the plain remaining length.
- **HER + DAVI 1-step Bellman fallback on failures** — extra machinery to squeeze a signal
  from unsolved instances; simply **dropping** them (net-solvability curriculum) is simpler
  and works better.
- **K-ladder / `advance_solve_rate` gate / target network / admissible floor** — all
  removed; net-solvability is the implicit curriculum and the plain remaining-length target
  needs no target net or floor.
- **Under-training a random net** with fine-tune-grade settings (`lr=1e-4`, `policy_weight=0.4`,
  3 epochs) — fixed to supervised-grade (`lr=3e-4`, `policy_weight=1.0`, 6 epochs).
- **A rejected oracle fallback at inference** (guaranteed 100% solve) — removed; the planner
  is pure NN.

## 10. Eval (regret vs oracle)

`evaluate.benchmark` unchanged. Per instance on val/test: `regret = nn_astar_cost −
oracle.optimal_cost`. Report solved/N, mean_regret, %optimal (regret==0), fail count for
`NN A*`, `greedy(pol)`, `greedy(val)`. In-loop eval is a small 25-board probe (noisy);
final numbers come from the full 450-instance test sweep in `../RESULTS.md`. The oracle
(`oracle.solve`/`optimal_cost`) is **eval-only** — never a training label.
