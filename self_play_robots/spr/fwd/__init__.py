"""`spr.fwd` -- the PRIMITIVE-MOVE (forward) arm of the self-play loop.

The sibling package `spr` searches over SUBGOAL decisions (backward plan
language, horizon 2-6, branching up to ~50). This one searches over the raw
game: actions are primitive moves `(robot, direction)`, horizon 8-30, branching
<= 4R. PROBLEM.md section 6.1 calls it "the purer AlphaZero" -- it removes the
candidate-generator ceiling at the price of a deeper horizon.

Everything imports the supervised forward stack in place and never forks it:

    move_planner.state       positions tuple, legal_moves, apply_move, is_goal
    move_planner.evaluate    Guide (MoveNet policy+value), nn_astar, greedy
    move_planner.encode/net  featurization + the two-headed MoveNet
    move_planner_v2          the existing expert-iteration loop (record schema,
                             oracle-free start states, train_on_records)
    eval.compare             aggregate / load_instances / _CountingGuide / run_forward
    eval.replay_validate     independent certification of dumped move sequences

Conventions are mirrored from `spr.search` / `spr.bench` / `spr.selfplay` so the
two arms are gate-comparable: one SearchResult type, the arena's expansion unit
(1 policy pass + 1 batched value pass over <= k children), best-at-budget +
`stop_after_certified`, root Dirichlet noise, manifest/instances sidecars,
atomic writes, and a grep-able `SPR FWD ... DONE` last line.

ONE BOARD CONFIG PER PROCESS: the forward stack reads RR_GRID / RR_ROBOTS /
RR_ENV_DIR at IMPORT time (`train.encode`, `nn.gen_grids`, `GridEnv`). For the
base 16x16 pool those variables must be UNSET; self-play sets RR_ENV_DIR before
the first forward-stack import in each worker.
"""
from __future__ import annotations

from spr import ASSETS, REPO, RESULTS, SPR, SV  # noqa: F401

FWD = SPR / "spr" / "fwd"
# default landing zone for anything this package writes (results/ is the
# orchestrator's; jobs override with $OUT_ROOT)
RUNS = REPO / "runs" / "spr" / "fwd"
