# Primer — what this repository is, start to finish

Written 2026-07-17 as the orientation document for anyone (human or agent) joining the
project. Plain language throughout; terms defined at first use. The companion documents
are `SOURCE_OF_TRUTH.md` (canonical audit — wins any disagreement), `FINDINGS.md` (the
running log of results with sources), `COMPARISON.md` (measurement protocol),
`HANDOFF.md` (operations: what runs where, pitfalls).

## 1. The puzzle

Ricochet Robots: a square grid (16×16 in the base game) with walls, and a few robots
(4 in the base game). When you move a robot it slides in a straight line until it hits
a wall or another robot, stopping in the cell just before the obstacle — it cannot stop
mid-slide. One robot is the *target*; one cell is the *goal*. Solve the puzzle by
getting the target robot onto the goal cell in as few moves as possible. The tricky
part: robots often need other robots as bumpers, so solutions involve parking helper
robots at exact spots first.

## 2. The research question

Two ways to build a puzzle-solving program from neural networks:

- **Forward (move-by-move):** think exactly like the game — "which robot slides which
  way right now?" — one move at a time from the start position toward the goal.
- **Backward (subgoals):** think from the goal outward — "for the final slide to stop
  on the goal, a robot must first be parked HERE; getting one there needs THIS first" —
  in units called **subgoals** ("park helper H on support cell S so the slider stops on
  bottleneck cell B"), each worth several moves at once.

Both use the same machinery: a *proposal network* suggests what to try next, a *value
network* estimates how close each option is to done, and a best-first search (always
explore the most promising option first) ties them together, capped at a fixed number
of **search steps** (one step = examine one position/partial plan and generate its
candidate continuations; in both families a step costs one pass of each network, so
budgets are directly comparable).

**The question: which approach is more efficient and more useful — especially as
puzzles scale up (more robots, bigger boards)?** The project owner's hypothesis is that
subgoals win at scale.

## 3. The rules that keep the comparison honest

1. **No exact solver at solve time.** The planners are networks + search only. An
   exact solver (we call it "the oracle") is used ONLY to create training labels and
   reference optima for scoring.
2. **Same puzzles, same budgets.** All systems are scored on one pinned set of 450
   puzzles (16×16, 4 robots), then per-configuration pinned sets at larger scales, at
   the same step cap (1200) and proposal width (top 5).
3. **A puzzle counts as solved only if the plan actually plays out** — the backward
   planner's subgoal plans are converted to real move sequences under full physics
   (`eval/realize.py`); if the conversion fails, it does not count. This mattered
   enormously (see §5).
4. **Plain language and neutral tone in all documents**; every claimed number traces to
   a results file.

## 4. The training pipelines

- **Supervised:** the oracle solves sampled puzzles and every decision is labeled with
  exact costs; networks learn to imitate. (Forward labels: optimal move + cost-to-go
  per position, plus each child position's cost. Backward labels: every candidate
  subgoal at each decision is committed and solved to completion for its exact cost.)
- **Self-play:** no oracle. The planner solves puzzles with its own current networks;
  solved puzzles become training data; repeat. This is the only training that survives
  at scale, because the oracle progressively fails there (§6).
- Data generation was ported to **Rust** (`rust_datagen/`, 50–180× faster,
  verification in its VERIFICATION.md + ADOPTION.md); Python remains the reference
  implementation. Board/robot counts are runtime-generic (`RR_GRID`, `RR_ROBOTS` env
  vars; per-config board directories `environments_<cfg>/`).
- The scaling harness (`scaling/`) packages per-configuration pipelines: generate
  boards → label data → train three networks (backward value + proposal, forward) →
  build a pinned benchmark → run the head-to-head (`eval/compare.py`).

## 5. What happened at the base scale (16×16, 4 robots) — the short story

1. The backward planner's historical "99.6% solved" counted plans that could not be
   played. Forced through real physics: **53.1%**.
2. Four verified bugs/semantic fixes later (self-bounce proposals; one robot holding
   two plan roles; claimed-but-never-placed stoppers; execution order corrections):
   **80%**, no methodology change.
3. Checking playability *inside* the search (discard doomed partial plans immediately;
   zero false discards proven): **89.1% at ~7 search steps**.
4. The remainder is a measured **language ceiling**: for 42/450 puzzles (9.3%) NO
   playable plan exists in the subgoal vocabulary at all (22 cannot even be decomposed;
   20 decompose but no decomposition survives physics; only 7 are the networks'
   fault). Best possible ≈ **90.7%**. Fix path: extend the plan language (a stopper
   that may vacate/return after use — "Lever B1" in SOURCE_OF_TRUTH §5c). Training
   cannot fix this; vocabulary can. Probe + re-measurement scripts exist
   (`analysis/artifacts/ceiling_probe.py`).
5. The forward planner, properly trained, solves **100% at 0.067 extra moves** here —
   but needs 36–424 search steps vs backward's ~7. Efficiency vs quality is the trade.
6. Self-play works for both families. For backward, generation now trains only on
   plans that really play out.

## 6. What happens as puzzles scale — the evidence so far

- **The oracle dies:** its failure rate at practical budget is 0% (4 robots) → 29.8%
  (6 robots) → 40.9% (8 robots) → 48.4% (24×24) → 61.1% (32×32). Probed at 10× budget:
  41% of the 8-robot failures become solvable (at 10× the cost), 59% stay out of
  reach. Supervised forward training runs out of teacher; self-play remains.
- **Forward training is fragile on the robot axis:** its stock recipe diverged at 6
  and 8 robots (and 24×24/8), each time rescued by a lower learning rate — the tuning
  cost is itself evidence, honestly caveated since each point was rescuable so far.
- **Head-to-heads on the puzzles the oracle can grade:** forward (properly trained)
  still wins solve rate/quality at 6 robots and 24×24 — but its per-puzzle cost
  explodes with board size (24×24: 191 steps and 275 s vs backward's 27 steps and
  7.8 s).
- **Beyond the oracle's reach** (hardest puzzles, solving is self-certifying): the
  backward planner WINS at 6 robots — 52.2% vs 48.5% at 2.8× fewer steps. This is the
  regime the thesis is about; more such measurements are running.
- Current per-rung status: FINDINGS §9 scoreboard (measured cells + "–" pending).

## 7. The deliverable

A single self-contained HTML comparison page (generated by
`eval/build_report.py` → `eval/results/report.html`, published as claude.ai artifact
`785268c8-43ad-4f45-a5af-3e09c815d910`), built entirely from the machine-readable
results (`eval/results/*.json`, `scaling/results/*/*.json`,
`subgoal_selfplay/*/iter*_stats.json`) — numbers are never hardcoded. It shows: what
the puzzle is, both planner families, the headline table, efficiency and budget
curves, the playability story and the measured ceiling (with the honest "fixable by
vocabulary, not by training" verdict), self-play, the scaling scoreboard including
planned-but-pending rows, and full provenance. Its predecessor
(`move_planner/summary.html`) had a tabbed layout users liked; the current page is
single-scroll and is being redesigned toward tabs.

## 8. House rules for agents working here

- Two GPUs maximum, only ever free ones (`nvidia-smi` first; claim locks in
  `/tmp/claude-1010/gpu_claims/`); evaluations usually on CPU (8 threads).
- Never compare "abstract" backward plan cost to real move optima; only played-out
  move counts are comparable.
- Beyond-oracle result files carry placeholder optima: only solve rate / steps / time
  / plan length are meaningful there.
- Everything of significance appends to FINDINGS.md, with the source file named.
- Watch for `pkill -f` self-matches and glob traps (`gen_s*` matches `gen_stock`);
  verify process state with plain `ps` before concluding anything.
