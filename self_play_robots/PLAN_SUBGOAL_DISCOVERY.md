# Plan: learned subgoal discovery (the project's actual thesis)

Written 2026-08-28. This plan is for the agent that picks the project up next.
Read it fully before running anything.

## 1. What this is for

The project set out to build a self-play planner that **discovers subgoal states
by itself** and explores toward them, with no human or solver labels, and beats
both hand-built planners.

What exists today does not do that. The subgoals are hand-written:
`supervised_valuenet/heuristics.py::propose` enumerates candidate triples
(bottleneck cell, support cell, helper robot) from hand-coded geometry, and the
networks only **rank** what that proposer offers. The network has never proposed
a subgoal, never decided that a state is worth reaching.

We measured the price of this. An exhaustive, network-free search over the
hand-written language cannot get below about +1.0 extra moves over optimal
(`self_play_robots/results/ceiling/g24r4_b2.json`). The current headline result
(the hybrid) works around that by letting the planner play ordinary robot moves
outside the language. It is a good workaround for a bad vocabulary, not a
learned better vocabulary.

**This plan builds the missing thing: a planner whose subgoals are states it
chooses, scored by a network trained on its own certified experience.**

## 2. The one metric, and the one table

Everything reports on **16x16, 4 robots**, the pinned 450-puzzle benchmark
`supervised_valuenet/eval/data/bench450.jsonl`. Every instance there has a known
optimal move count, so quality is measurable on 100% of the set. Ignore every
other configuration until Stage 5.

**Headline metric: percent of the 450 puzzles solved with a provably optimal
number of moves.** One number. It cannot be gamed: an unsolved puzzle is not
optimal, and a solved-but-longer puzzle is not optimal. A solution counts only
after `eval/realize.py::strict_moves` replays it under the real rules and the
replayed length equals the instance's `d_star`.

Report two supporting columns, never as the headline:
- solved of 450
- mean extra moves over optimal, on the puzzles that model solved (state the count)

Search budget is fixed for every row: **1200 expansions, k = 5**, the arena
convention. Any row that cannot be run under that budget is marked, not adjusted.

The table, which is the deliverable of every stage from 0 onward:

| model | optimal % of 450 | solved / 450 | extra moves (on its solves) |
|---|---|---|---|
| forward (move-by-move, supervised) | | | |
| backward (subgoal, supervised) | | | |
| current self-play line | | | |
| **new: learned subgoal discovery** | | | |

The current self-play networks are size-free, so they run at 16x16 unchanged.
Use the same checkpoints the project's own 16x16 rows used; do not retrain them.

## 3. The fourth model

**Subgoal = a state you want to reach, expressed as "robot R comes to rest on
cell C".** Nothing else. No bottlenecks, no supports, no helper slots, no
hand-written geometry.

At 16x16 with 4 robots that space is 4 x 256 = 1024 candidates per decision.
That is small enough to score exhaustively with one batched network pass, which
removes the proposer bottleneck entirely: the network sees every possible
subgoal and picks.

Three components:

1. **Goal-conditioned cost network.** Input: the board, the robot positions, and
   a goal (robot R on cell C). Output: a distribution over the number of moves
   needed to reach that goal, reusing the existing 96-bin head and the size-free
   encoder in `self_play_robots/spr/nets.py`. This is the one real architecture
   change: today's cost network answers "how much more will this plan cost";
   the new one answers "what does it cost to get from here to there".
2. **Search over states.** Best-first. A node is (current board state, moves
   spent). Expanding a node scores all 1024 goals, keeps the top k = 5, and for
   each one asks physics whether that goal is reachable directly — the engine
   already answers this exactly and cheaply via
   `GridEnv.compute_exact_shortest_path_length`, which the existing code uses for
   its "free exact fixes". Reachable goals become children with their true move
   cost. Unreachable goals are dropped. The final goal (target robot on target
   cell) terminates the search.
3. **Self-play with certified labels.** Identical in spirit to the existing loop
   (`spr/selfplay.py`): fresh boards, search with noise, replay every finished
   plan, and label each chosen goal with the true replayed cost of the best
   certified plan beneath it. No oracle. The one new thing is that the label now
   attaches to a *state the network chose*, which is what makes discovery
   learnable.

## 4. Stages, each with a kill criterion

Do not skip a stage. Each one must answer its question before the next starts.
Report the table at every stage, even when the new row is empty.

### Stage 0 — the harness (minutes, no GPU)
Produce the three-row table for the existing models from payloads that already
exist. Verify every number by re-reading the payload and re-checking the replay
lengths against `d_star`; do not copy from any document.
**Gate:** the three rows reproduce the project's recorded 16x16 numbers. If they
do not, stop and report the discrepancy — that is a finding in itself.

### Stage 1 — is the new action space even better? (minutes, CPU)
No learning. Take 20 puzzles. Search the (robot, cell) subgoal space with a
trivial heuristic and a generous budget, and measure the best solution the space
can express, exactly as `spr/ceiling.py` does for the old language.
**Question:** can this space express solutions the hand-written language cannot?
**Gate:** its ceiling must be at least as good as the hand-written language's on
the same 20 puzzles, and should approach 0 extra moves. If the space cannot
express better solutions, the whole idea is dead and this plan ends here.
This is the cheapest possible test of the central premise. Run it first.

### Stage 2 — can a network learn goal-conditioned cost? (under an hour, 1 GPU)
Generate training data cheaply: sample states from bench450 boards, sample goals,
and compute the true cost with the exact engine, which is fast at 16x16. Train
the goal-conditioned cost network. Evaluate on held-out boards.
**Gate:** it must rank goals better than a distance-based heuristic baseline by a
clear margin, on boards it never trained on. State both numbers.

### Stage 3 — does it plan? (a few hours, 1 GPU)
**Revised 2026-08-28 after Stage 2 failed its gate.** Stage 2 asked the network to
predict the cost of reaching a subgoal, and that was the wrong object:
- exact physics computes that cost in 10.9 ms, cheaper than the network predicts
  it, and the search must compute it anyway to build the child state;
- ranking by cost at all is the binding constraint. With a hard top-5 prune, an
  optimal next subgoal survives at the root only 61% of the time even when
  ranked by the *exact* cost, which caps this design in the low sixties.
- ranking by cost plus a remaining-distance term raises root survival to 77% and
  whole-path survival from 28% to 55%.

So Stage 3 uses physics for edges and learning for the heuristic:
1. children come from `subgoal/space.py::rest_cells` with their true slide costs;
2. the beam ranks by `g + h`, where `h` estimates the moves still needed to
   finish from the child state — the same cost-to-go object the existing
   networks already learn, now over the state-subgoal space;
3. the network learns `h` only. Train it on exact-engine cost-to-go first.
Report the fourth row at the fixed 1200-expansion budget, and also at matched
network calls with a wider beam, because k=5 was chosen for a vocabulary with
about 50 candidates and this space offers 1024.
**Gate:** it must beat the backward planner's optimal percentage on all 450. If
it cannot beat a hand-written proposer while using exact-engine labels, the
search or the heuristic is wrong, and self-play will not rescue it.

### Stage 4 — does self-play work in this space? (hours, 1 GPU)
Replace the exact-engine labels with certified self-play labels. Run at least
three rounds from the Stage 3 checkpoint, and separately at least one run from
random initialisation, so we learn whether the supervised start is needed.
**Gate:** the optimal percentage must rise across rounds and beat Stage 3.
Report the per-round series with denominators.

### Stage 5 — scale (only after Stage 4 passes)
Then, and only then: 24x24, the frontier sets, transfer, seeds, and the
comparison against the hybrid search. Nothing here matters if Stage 1 or 3 fails.

## 5. Rules

- **The kill criteria are real.** A stage that fails its gate ends the plan; write
  up why and stop. A negative result at Stage 1 is worth more than a month of
  Stage 5.
- **Verify, never trust.** Every number in the table comes from a payload you
  read this session. The project's own documents have been wrong more than once,
  and the audit of 2026-08-28 (`FINDINGS.md` section 28) lists the cases.
- **Certification is not optional.** A plan counts only after physics replays it.
- **Never train on the 450 benchmark boards** or on any pinned exam id.
- Slurm: `-A open-37-42`, qgpu only, one leg per job, walltime under 8 hours,
  idempotent jobs, state projected node-hours before submitting anything.
- Keep the whole plan under 15 node-hours through Stage 4. Stages 0 and 1
  together should cost minutes and no GPU.
- Log results in `self_play_robots/FINDINGS.md` at the next free number, with the
  table and the file paths behind every number.

## 6. What success looks like

The fourth row beats the other three on the headline metric, and an inspection
of what it chose shows it aiming at intermediate states that the hand-written
proposer never offered. At that point the project's original claim is true:
the network found the subgoals itself.

If instead it matches but does not beat, that is still publishable as a negative
result with a measured explanation, provided Stage 1 showed the space was rich
enough. Say so plainly rather than tuning until something wins.
