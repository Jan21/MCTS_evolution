# self_play_robots — implementation design (as built, 2026-08-17)

Companion to `PROBLEM.md` (the brief) and `README.md` (module map). This
describes what the code actually does, so the FINDINGS entries and the local
report can refer to one place. Owner decisions of 2026-08-17: both action
spaces (subgoal main line + primitive-move arm at g16r4), both net families
(size-free pe=none rebuild AND per-size PolicyTF/LoopedValueNet) at two sizes
(g16r4, g24r4), own FINDINGS.md.

## 1. Subgoal action space (main line, `spr/`)

**State/action.** Node = partial plan (`partial_plan.PartialPlan`), root =
`skeleton.astar._initial_plan` after free exact fixes; a decision resolves the
plan's FIRST open edge (canonical order → the plan space is a tree, no
transpositions); actions = `heuristics.propose` candidates that `_apply`
accepts and whose helper stands at a robot start cell (`_hidx`, base
vocabulary, exactly the arena's filter). Terminal = complete plan.

**Expansion (the budget unit, = the arena's).** One expansion = candidates →
one policy pass (prior = softmax of the AR log-probs over the valid
candidates) → top-k by prior (k=5) → one batched value pass over those k →
children born with `ctg_hat`, `fixed_g`, `v_est`. Lever-A prefix filter
(`eval.realize.prefix_playable`) drops doomed children AFTER the value pass
(physics, free), forced exact fixes are free, certification is free — all as
in `eval/compare.py::_nn_astar_backward` (which `spr.bench --search
arena_astar` calls unmodified for gate rows).

**Value units.** `ctg` labels are "completed plan cost − fixed cost of the
decision plan" (nn/generate.py:104, descent.py:249): abstract plan-cost units.
Estimates `v_est = fixed_g(parent) + ctg_hat` (`--f-mode parent`) or the
arena's `fixed_g(child) + ctg_hat` (`child`, double-counts the candidate's own
fixed increment — kept as an arm). Certified terminals: search minimizes the
STRICT realized move count (the metric); their tree value is the plan's
abstract cost (unit-consistent with estimates); labels use abstract cost with
`strict_total` alongside.

**Searches (`spr/search.py`).**
- `greedy`: one expansion per decision, value argmin (the labeler's rule).
- `astar`: best-first on `v_est`; plain (first complete plan popped, = arena)
  or anytime (certify at pop, discard unplayable, continue) or best-at-budget
  (continue while the frontier's estimate < the best certified plan's
  abstract cost; return the cheapest certified plan).
- `mcts`: PUCT. Selection `argmax norm(Q) + c·P·√N(s)/(1+N(a))` with Q = cost,
  `norm` = min-max over the tree (MuZero style, clamped to [0,1]); backup
  `min` (default; deterministic single agent) or visit-weighted `mean`;
  unvisited children already carry their own `v_est` (no FPU hack);
  certified terminals are exact leaves (kept in Q, no longer selectable);
  unplayable terminals are dead (excluded, parents re-evaluated); a node
  whose children are all closed is closed; every iteration expands, certifies
  or closes something, so the loop terminates at the budget or when the root
  closes. Knobs: `c_puct` (1.5), root Dirichlet noise (generation), `root_k`
  (all candidates at the root for generation), `stop_after_certified`.

**Arena parity.** `spr.bench --search spr_astar --f-mode child` (plain)
reproduces `_nn_astar_backward` row-for-row (checked on 6 g24r4 instances,
both net families). Rows/payloads follow `eval.compare` so
`merge_compare_shards`, `replay_validate` and the report consume them.

## 2. Nets

- **Size-free** (`spr/nets.py`): `SizeFreeValueNet` (nn_labeler, pe=none, 96
  bins) and `SizeFreePolicyNet` (PolicyTF's 3-step pointer heads on the same
  LoopedLayer encoder, no `self.pos`, 7 channels + global row). Trainer
  `spr.train` (mixed sizes, size-bucketed batches, `--init` warm start,
  CollapseStop, min-val_regret checkpoint, save_last/RR_RESUME, `--splits`).
- **Per-size** (`--arch persize`): PolicyTF / LoopedValueNet unchanged, one
  config per process, warm-startable through the same trainer (self-play
  boards served through a symlinked combined env dir), driven in the searches
  by `PerSizePolicy` / `PerSizeValue` adapters.

## 3. Self-play generation (`spr/selfplay.py`) and labels

Fresh lean boards per iteration (ids 5000+120(k−1)…, never the pinned pool),
uniform random instances, MCTS best-at-budget with root noise + `root_k=all`
+ greedy sibling completion; records only where a complete plan through the
candidate was strictly realized: `cost_to_go = plan_cost(certified) −
fixed_g(decision)`, `is_optimal` = argmin, ≥2 labeled candidates per group,
principal-path decisions (`--emit path`; `all` namespaces depths). Provenance:
`label_source=spr_mcts`, `strict_total`, `abstract_total`, `visits`, `prior`,
`ctg_hat`, `iter`, `label_model`, `boards_dir`. Frozen 18-field schema →
`spr.train` (and the supervised trainers) read it unchanged.

## 4. Loop iteration (`jobs/selfplay_iter.slurm`)

A generate → B fidelity gauge (`spr.gauge`: 200 sampled depth-0 decisions
exact-labeled by the Rust engine, `nn_labeler.audit_descent` argmin agreement)
→ C buffer (last 3 iterations + exact anchors) → D/E warm-start policy+value
→ F bench (arena A\* and MCTS best-at-budget on the pinned exam) → G gate
(`spr.gate`: paired McNemar vs iteration k−1, M1 bars vs the supervised
pair). Idempotent phases (DONE markers), one queue wait per iteration.

## 5. Primitive-move arm (`spr/fwd/`, in progress)

PUCT over slides on the forward MoveNet Guide (`move_planner`), same budget
unit as `eval.compare.run_forward` (1 policy + 1 batched value call per
expansion), certification = the move path itself (+ `replay_validate` dump),
generation with `move_planner_v2`'s record schema and trainer.

## 6. Milestone instruments

M0 `spr.arena parity` (per-row); ceiling `spr.ceiling` (language optimum vs
d\*, base and B2 vocabularies); M1 `spr.gate m1` (3.5 solve / 6.6 optimality
points); M2 `spr.gate compare` (paired McNemar on solved vectors, sign test on
both-solved moves) of every search arm vs greedy and vs the arena A\*; M3+
paired vs the previous iteration + gauge trend + bench vs the frozen supervised
rows (`results/m0`, `eval/results`, `scaling/results`).
