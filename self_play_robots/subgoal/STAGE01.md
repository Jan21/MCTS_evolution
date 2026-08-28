# Learned subgoal discovery — Stages 0 and 1

`PLAN_SUBGOAL_DISCOVERY.md` §4, run 2026-08-28. Everything below is recomputed
from payload rows read this session; nothing is copied from a project document.
Regenerate the table with

    PYTHONPATH=supervised_valuenet:self_play_robots python -m subgoal.table
    PYTHONPATH=supervised_valuenet:self_play_robots python -m subgoal.space --n 20

**Verdict: Stage 0 gate PASS (with two findings), Stage 1 gate PASS. Stage 2 is
justified.**

---

## Stage 0 — the table

Exam: `supervised_valuenet/eval/data/bench450.jsonl`, 450 instances, 16×16, four
robots, every instance carrying an exact optimum `d*`. Budget for every row:
1200 expansions, k = 5. Headline metric: **percent of the 450 solved in a
provably optimal number of moves** — the row's own move dump replayed under the
real joint-game rules (`simulate.slide`, every other robot a blocker, a no-op
slide illegal — the physics of `eval/replay_validate.py` and of
`eval/realize.py::strict_moves`), with the replayed length equal to `d*`.

| model | optimal % of 450 | solved / 450 | extra moves (on its solves) |
|---|---|---|---|
| forward (move-by-move, supervised) | **94.2%** (424/450) | 450/450 | 0.067 (n=450) |
| backward (subgoal, supervised) | 53.3% (240/450) | 432/450 | 1.840 (n=432) |
| current self-play line (v14_stack nets) | 51.3% (231/450) | 439/450 | 1.995 (n=439) |
| *(second self-play arm: mix_b2mix_iter3)* | *52.4% (236/450)* | *438/450* | *2.011 (n=438)* |
| **new: learned subgoal discovery** | — | — | — |

### Gate: PASS — every row reproduces its payload exactly

For each row, all 450 move dumps were replayed. Checks that had to hold and did:
rows aligned with bench450 by `env_id` and `d*` (0 misaligned); every replay
legal and ending on the target (0 failures); every replayed length equal to the
payload's own `realized_strict` (0 disagreements); no replayed length shorter
than `d*` (0 — so no `d*` is contradicted); recomputed mean extra equal to the
payload's `mean_regret` to machine precision; recomputed optimal count equal to
the payload's `pct_optimal` under the payload's own denominator.

The two payloads that have recorded supervised references also match those
references row for row: `results/fwd_m0/astar_candidate_scored.json` vs the
`candidate_scored.ckpt` system of `eval/results/comparison_forward.json` — 0
differing rows on (moves, solved), and 424 rows with `regret == 0` on both
sides; `results/m0/g16r4_b1s21_b2flags.json` vs
`eval/results/final450_backward_b2_seed21.json` — 0 differing rows on (solved,
realized_strict, expansions), 240 rows with `regret == 0` on both sides.

### Finding 1 — the recorded "55.6% optimal" is a percentage of SOLVES, not of 450

`eval/compare.py::aggregate` (lines 507–509) computes

    pct_optimal = 100 * |{r in solved : r.regret == 0}| / len(solved)

The denominator is the solved subset, not `n`. So the project's recorded 16×16
backward number, 55.6%, is 240/432; the plan's headline number for the same
payload is 240/450 = **53.3%**. The forward row is unaffected only because it
solves all 450 (424/450 = 94.2% either way). This is a definitional gap, not a
data error: the payload rows reproduce exactly. Every optimal-% comparison in
this project between arms with different solve counts is therefore not directly
readable as the plan's headline metric, and the fourth row must be scored
against 53.3% / 51.3%, not against 55.6%.

### Finding 2 — no self-play checkpoint had ever been benched at 16×16

The plan says "the current self-play networks are size-free, so they run at
16×16 unchanged; use the same checkpoints the project's own 16×16 rows used".
**There are no such rows.** A scan of all 493 payloads under
`self_play_robots/results/` finds exactly 28 on `bench450`, and every one of
them is a supervised or M1/M2 arm (`m0/`, `m1/`, `m2/`, `fwd_m0/`, `fwd_m2/`).
Every payload under `results/selfplay/` and `results/variants/` runs on
`bench.solved` / `bench.unsolved` / `g24r4_unseen` at g24r4, g24r8 or g32r4.

Rather than leave the row blank or substitute a different exam, the size-free
self-play nets were run unchanged at g16r4 on bench450 under the same protocol
as the backward supervised row (arena A\*, 1200 expansions, k = 5, B2
vocabulary, anytime, replay-certified): `jobs/subgoal_stage0_selfplay16.slurm`,
job 4862862, qgpu_exp, 1 GPU, 13 min 22 s = **0.028 node-hours**. Two arms: the
planner-of-record's training pair (`v14_stack`, `results/status.json` M5) and
the last main-line curriculum pair (`mix_b2mix_iter3`). Both rows are new
numbers produced this session; the Stage 0 gate can only confirm that they
reproduce their own payloads, which they do.

**Reading of the third row.** At 16×16 the self-play line solves more than the
supervised backward planner (439 and 438 vs 432) but is *no better, and slightly
worse, on move quality* (231 and 236 optimal vs 240; extra moves 1.995 and 2.011
vs 1.840). That is the same pattern FINDINGS §19 recorded at 24×24 — the loop
distilled search into the nets and raised the solve rate; it never shortened
solutions. On the plan's single headline metric the self-play line does not beat
the supervised backward planner it grew out of, and all three subgoal rows sit
41 to 43 points below the forward move-by-move planner.

---

## Stage 1 — is the state subgoal space richer than the hand-written language?

No learning, no networks, no `d*` inside the search.

**The space.** A subgoal is "robot R comes to rest on cell C" — 4 × 256 = 1024
candidates per decision, against `heuristics.propose`'s geometric (bottleneck,
support, helper) triples.

**The probe** (`subgoal/space.py`), the same shape as `spr/ceiling.py`:

- a node is the joint robot positions;
- expanding it realizes **every** candidate against the real rules: for each
  robot, a BFS over `simulate.slide` with every other robot frozen at its
  current cell gives the exact set of cells that robot can come to rest on and
  the true slide count for each. Reachable cells become children at their true
  cost; unreachable ones are dropped;
- the frontier is ordered by f = g + h with h the "any-stop" relaxation distance
  of the target robot to the target cell (one move may end on any cell of the
  row/column segment, walls only, robots ignored). Every real move is a relaxed
  move, so h is an admissible and consistent lower bound — the trivial heuristic
  the plan asks for. It reads the board only;
- budget 500,000 pops / 300 s per instance, far above what any instance used;
- because h is consistent, the first goal state popped is **optimal in this
  space**, so the number reported is the space's ceiling, not a sample of it;
- every solution is then replay-certified move by move.

Because a macro edge moves one robot while the others stand still, the
concatenation of a path's edges is a legal primitive move sequence by
construction — which is why the certification never fails here, unlike in the
hand-written language where realization is the hard part.

*Implementation note.* The plan suggests
`GridEnv.compute_exact_shortest_path_length` for the reachability test. It is
the wrong primitive: it reads `reachability_matrix`, a blocker-free lone-robot
distance built from the plan DAG, so it cannot see a helper robot acting as a
stopper, and it both mis-reports reachability and mis-costs edges in a joint
state. The joint-state BFS used here is the exact answer and is what the
certifier requires.

Two deterministic 20-instance samples were run — the first 20 (as the plan
words it) and a stride-23 sample spanning the whole benchmark. No instance was
chosen for its result. The hand-written columns are `best_realizable_moves` per
instance from the project's own exhaustive language probes,
`results/ceiling/g16r4_base.json` and `results/ceiling/g16r4_b2.json`
(FINDINGS §3).

**Sample A - the first 20 instances of bench450** (`self_play_robots/results/subgoal/stage1_state_space_first20.json`)

| bench idx | env | d\* | state space (certified) | subgoals | proved optimal | hand-written base | hand-written B2 |
|---|---|---|---|---|---|---|---|
| 0 | 2400 | 12 | **12** | 3 | yes | 12 | 12 |
| 1 | 2400 | 4 | **4** | 2 | yes | 4 | 4 |
| 2 | 2400 | 7 | **7** | 2 | yes | 7 | 7 |
| 3 | 2401 | 1 | **1** | 1 | yes | 1 | 1 |
| 4 | 2401 | 4 | **4** | 1 | yes | 4 | 4 |
| 5 | 2401 | 6 | **6** | 3 | yes | 7 **+1** | 7 **+1** |
| 6 | 2402 | 2 | **2** | 1 | yes | 2 | 2 |
| 7 | 2402 | 4 | **4** | 1 | yes | 4 | 4 |
| 8 | 2402 | 10 | **10** | 3 | yes | 11 **+1** | 11 **+1** |
| 9 | 2403 | 6 | **6** | 2 | yes | 7 **+1** | 7 **+1** |
| 10 | 2403 | 1 | **1** | 1 | yes | 1 | 1 |
| 11 | 2403 | 7 | **7** | 3 | yes | 7 | 7 |
| 12 | 2404 | 8 | **8** | 2 | yes | 8 | 8 |
| 13 | 2404 | 5 | **5** | 2 | yes | 15 **+10** | 15 **+10** |
| 14 | 2404 | 6 | **6** | 2 | yes | 6 | 6 |
| 15 | 2405 | 7 | **7** | 3 | yes | 28 **+21** | inconclusive |
| 16 | 2405 | 3 | **3** | 1 | yes | 3 | 3 |
| 17 | 2405 | 7 | **7** | 2 | yes | 11 **+4** | 11 **+4** |
| 18 | 2406 | 9 | **9** | 4 | yes | 11 **+2** | 11 **+2** |
| 19 | 2406 | 6 | **6** | 1 | yes | 6 | 6 |

summary: state space mean extra **0.000** over 20, 20/20 optimal; hand-written base mean extra 2.000 over 20, 13/20 optimal; hand-written B2 mean extra 1.000 over 19, 13/20 optimal. mean d\* 5.75, mean subgoals 2.0, 169.0 s on one CPU core.

**Sample B - every 23rd instance (idx 0, 23, ... 437), spanning the whole benchmark** (`self_play_robots/results/subgoal/stage1_state_space_stride23.json`)

| bench idx | env | d\* | state space (certified) | subgoals | proved optimal | hand-written base | hand-written B2 |
|---|---|---|---|---|---|---|---|
| 0 | 2400 | 12 | **12** | 3 | yes | 12 | 12 |
| 23 | 2407 | 8 | **8** | 3 | yes | 12 **+4** | 8 |
| 46 | 2415 | 7 | **7** | 2 | yes | 7 | 7 |
| 69 | 2423 | 8 | **8** | 2 | yes | 8 | 8 |
| 92 | 2430 | 7 | **7** | 2 | yes | 14 **+7** | 12 **+5** |
| 115 | 2438 | 9 | **9** | 5 | yes | 11 **+2** | 11 **+2** |
| 138 | 2446 | 9 | **9** | 4 | yes | 9 | 9 |
| 161 | 2453 | 6 | **6** | 3 | yes | 8 **+2** | 6 |
| 184 | 2461 | 3 | **3** | 1 | yes | 22 **+19** | 3 |
| 207 | 2469 | 7 | **7** | 2 | yes | 7 | 7 |
| 230 | 2476 | 7 | **7** | 4 | yes | no plan | 10 **+3** |
| 253 | 2484 | 6 | **6** | 2 | yes | 6 | 6 |
| 276 | 2492 | 1 | **1** | 1 | yes | 1 | 1 |
| 299 | 2499 | 5 | **5** | 1 | yes | 5 | 5 |
| 322 | 2507 | 9 | **9** | 4 | yes | unplayable | inconclusive |
| 345 | 2515 | 6 | **6** | 3 | yes | no plan | 8 **+2** |
| 368 | 2522 | 6 | **6** | 1 | yes | 6 | 6 |
| 391 | 2530 | 8 | **8** | 2 | yes | 8 | 8 |
| 414 | 2538 | 7 | **7** | 3 | yes | 7 | 7 |
| 437 | 2545 | 3 | **3** | 1 | yes | 4 **+1** | 3 |

summary: state space mean extra **0.000** over 20, 20/20 optimal; hand-written base mean extra 2.059 over 17, 11/20 optimal; hand-written B2 mean extra 0.632 over 19, 15/20 optimal. mean d\* 6.7, mean subgoals 2.45, 243.2 s on one CPU core.

### Gate: PASS, at the ceiling

| sample (n = 20) | state space | hand-written base | hand-written B2 |
|---|---|---|---|
| first 20 — mean extra moves | **0.000** (20/20) | 2.000 over 20 (13/20 optimal) | 1.000 over 19 (13/20 optimal) |
| stride 23 — mean extra moves | **0.000** (20/20) | 2.059 over 17 (11/20 optimal) | 0.632 over 19 (15/20 optimal) |

On all 40 instances the state subgoal space expresses a certified solution of
exactly `d*`: 40/40 optimal, 0.000 extra moves, every one proved optimal by the
search itself and every one replay-certified. The hand-written language is
strictly worse on 7 of the first 20 and 9 of the stride 20 in its base
vocabulary (including 2 instances with no complete plan at all and 1 with no
playable one), and on 6 and 4 respectively in the extended B2 vocabulary (plus
1 inconclusive in each sample). **It is never better on any of the 40.**

Both gate clauses are met with room to spare: the new space is at least as good
everywhere, and it does not merely approach zero extra moves, it reaches zero.

### What it expresses that the hand-written language cannot

The gaps are large, not marginal — the base language needs 15 moves where the
optimum is 5 (idx 13), 28 where it is 7 (idx 15), 22 where it is 3 (idx 184).
Inspecting the chosen subgoals, two structures do the work and neither exists in
the (bottleneck, support, helper) vocabulary:

- **intermediate parking of the target robot itself.** idx 15 (d\* 7, base
  ceiling 28): the plan is *Green comes to rest at (13,7)* → *Red comes to rest
  at (7,14)* → *Green comes to rest at (8,14)*. The target robot is moved to a
  waiting position first, then a helper is placed as its stopper, then it
  finishes. The hand-written proposer names a bottleneck and a support for one
  final approach; it has no way to say "put the target robot somewhere else
  first". 3 of the first 20 optimal solutions have this shape.
- **helper journeys costed as one decision.** idx 13 (d\* 5, base ceiling 15):
  *Blue comes to rest at (14,8)* → *Red comes to rest at (10,2)*, five slides in
  two decisions. The helper's own multi-slide trip to its stopper cell is part
  of the subgoal, not a realization detail that may or may not survive.

The space is also cheap to search: the optimal solutions use **1 to 5** subgoals
(mean 2.0 / 2.45), so a planner in this space makes very few decisions per
puzzle, each over 1024 candidates that one batched network pass can score.

### Cost of the probe

169 s + 243 s = 412 s on one login-node CPU core for 40 instances; the most
expensive single instance (idx 0, d\* 12) took 48 s and 60,685 pops. No GPU.

---

## Verdicts and what follows

| gate | verdict |
|---|---|
| Stage 0 — the three rows reproduce the recorded 16×16 numbers | **PASS**, with two findings: the recorded `pct_optimal` is a percentage of solves, not of 450; and no 16×16 self-play row existed, so one was produced (job 4862862). |
| Stage 1 — the new space's ceiling is at least as good as the hand-written language's, and approaches zero extra moves | **PASS**. 0.000 extra moves, 40/40 optimal, never worse on any instance, against 2.0 / 0.6–1.0 for the hand-written language. |

The premise of `PLAN_SUBGOAL_DISCOVERY.md` survives its cheapest test. The
+1.42 (base) / +0.90 (B2) floor that FINDINGS §3 recorded for the hand-written
language is a property of that vocabulary alone, not of subgoal planning: in the
state subgoal space the floor is 0.00. Everything the three existing subgoal
rows lose on the headline metric — 46 to 49 points below optimal — is inside
what the new space can express.

Two cautions for Stage 2 and later, from these numbers:

1. **Expressiveness is not the whole story.** The forward move-by-move planner
   already searches a space with the same 0.00 floor (raw slides) and reaches
   94.2%. The state subgoal space's advantage over it must come from the *2.0
   to 2.5 decisions per puzzle* seen here versus its ~6.4 moves, not from
   expressiveness. Stage 3's gate (beat the backward planner, 53.3%) is the
   right bar; the forward planner's 94.2% is the honest ambition.
2. **The exhaustive search that found these optima is not the Stage 3 search.**
   Here it popped up to 60,685 nodes with a perfect admissible heuristic; the
   arena budget is 1200 expansions with k = 5. The gap between "the space
   contains the optimum" and "a k = 5 beam over network scores finds it" is
   exactly what the goal-conditioned cost network must close, and it is the
   real risk in Stage 2/3.

---

## Provenance appendix

Every number above, with the file it came from and how it was read.

### Stage 0 rows

| number | file | how it was read |
|---|---|---|
| forward: 424/450 optimal, 450/450 solved, 0.0667 extra | `self_play_robots/results/fwd_m0/astar_candidate_scored.json` | `subgoal/table.py::score` — replayed each row's `moves_seq` (450 dumps) with `simulate.slide`, counted length == `d_star`. Payload aggregate `solved` 450, `pct_optimal` 94.222, `mean_regret` 0.06667 for comparison only. Job 4680465. |
| forward reference cross-check: 0 differing rows, 424 with `regret == 0` | `supervised_valuenet/eval/results/comparison_forward.json`, system `forward move planner (move_planner/checkpoints/candidate_scored.ckpt)` | row-by-row diff on (`moves`, `solved`) against the payload above. Note this reference has three other forward systems; the one named here is the one `results/fwd_m0/astar_candidate_scored.parity.json` checks against. |
| backward: 240/450 optimal, 432/450 solved, 1.8403 extra | `self_play_robots/results/m0/g16r4_b1s21_b2flags.json` | same, from each row's `moves` dump. Payload aggregate `solved` 432, `pct_optimal` 55.5556, `mean_regret` 1.84028. Job 4679719. Checkpoints: `assets/g16r4_backward_{policy,value}_b1s21.ckpt`. |
| backward reference cross-check: 0 differing rows, 240 with `regret == 0` | `supervised_valuenet/eval/results/final450_backward_b2_seed21.json` | row-by-row diff on (`solved`, `realized_strict`, `expansions`). |
| the denominator finding | `supervised_valuenet/eval/compare.py` lines 507–509 | read directly; `240/432 = 0.5555556`, `240/450 = 0.5333`. |
| self-play v14_stack: 231/450 optimal, 439/450 solved, 1.9954 extra | `self_play_robots/results/subgoal/v14_stack_g16r4_bench450_astar.json` | produced this session, job 4862862; scored by the same replay. Nets: `runs/spr/variants/v14_stack/{policy/…epoch=4-step=1035.ckpt, value/…epoch=1-step=826.ckpt}` (= `results/variants/v14_stack/nets.txt`). |
| self-play mix_b2mix_iter3: 236/450 optimal, 438/450 solved, 2.0114 extra | `self_play_robots/results/subgoal/mix_b2mix_iter3_g16r4_bench450_astar.json` | same job. Nets: `runs/spr/selfplay/mix_b2mix_iter3/{policy,value}/…` (= `results/selfplay/mix_b2mix_iter3/nets.txt`). |
| "no 16×16 self-play row exists" | all 493 JSONs under `self_play_robots/results/` | scanned `protocol.instances_file`: `bench450.jsonl` 28, `bench.solved.jsonl` 102, `bench.unsolved.jsonl` 63, `g24r4_unseen.jsonl` 31. All 28 bench450 payloads are in `m0/`, `m1/`, `m2/`, `fwd_m0/`, `fwd_m2/`. |
| d\* for all 450 | `supervised_valuenet/eval/data/bench450.jsonl` | field `d_star`, one line per instance, rows matched by position (the arena emits one row per instance in file order) and cross-checked on `env_id` and `d_star`. |
| boards | `supervised_valuenet/environments/env_<id>.pkl` | field `grid_data` only, via `simulate.wall_sets` — the same read `eval/replay_validate.py` does. |

### Stage 1 rows

| number | file | how it was read |
|---|---|---|
| state space, first 20: 0.000 extra, 20/20 optimal, 2.0 subgoals, 169 s | `self_play_robots/results/subgoal/stage1_state_space_first20.json` | produced this session by `subgoal/space.py` (`--n 20 --time-cap 300 --max-pops 500000`), login node, one core, no GPU. `certified_moves` is the length of the replay-certified sequence; `proved_optimal` is true when the goal was popped inside budget. |
| state space, stride 23: 0.000 extra, 20/20 optimal, 2.45 subgoals, 243 s | `self_play_robots/results/subgoal/stage1_state_space_stride23.json` | same, `--n 20 --stride 23`. |
| hand-written base ceiling per instance | `self_play_robots/results/ceiling/g16r4_base.json` | field `best_realizable_moves` of the row with matching `idx`; `category` gives `NO_COMPLETE_PLAN` / `NO_REALIZABLE_PLAN` / `INCONCLUSIVE` for the blanks. Job 4679720, FINDINGS §3. |
| hand-written B2 ceiling per instance | `self_play_robots/results/ceiling/g16r4_b2.json` | same. |
| whole-benchmark context: base 1.419 extra over 408 realizable (252 optimal), B2 0.896 over 441 (295 optimal) | the two ceiling files above | recomputed over all 450 rows from `best_realizable_moves - d_star`; FINDINGS §3 records +1.42 and +0.90 (the small difference is that §3 averages over graded realizable rows the same way — 1.4191 vs "≈1.42"). |

### Compute

| what | job / host | resource | node-hours |
|---|---|---|---|
| self-play pairs at 16×16 on bench450 (2 arms × 450) | Slurm 4862862, qgpu_exp | 1 × A100, 13 min 22 s | **0.028** |
| Stage 1 probes (40 instances) | login node | 1 CPU core, 412 s | 0 |
| Stage 0 recomputation (4 payloads × 450 replays) | login node | 1 CPU core, ~6 s per run | 0 |

Total for Stages 0 and 1: **0.028 node-hours.**
