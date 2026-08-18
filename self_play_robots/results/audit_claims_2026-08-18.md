# Adversarial audit of self_play_robots/FINDINGS.md §1–§13 headline claims (2026-08-18)

Scope: read-only verification of the result files, training provenance, statistics
and the ceiling probe behind the headline claims. All numbers below were recomputed
from the payload rows with independent scripts (not spr.gate) unless marked
"file says". Paths are relative to `/scratch/project/open-37-42/petrhyner/MCTS_evolution/`
(`SV/` = `supervised_valuenet/`, `SPR/` = `self_play_robots/`).

## Verdicts per headline claim

| # | claim (FINDINGS §) | verdict |
|---|---|---|
| A | §5 MCTS beats greedy AND the arena A* on realized moves at identical solve sets, per-size nets, both sizes | SUPPORTED WITH CAVEAT (finding 6: the "MCTS" arm is in fact an exhaustive enumeration of the k=5 tree with ~5× more strict-realization calls; the win over first-solution A* is real and reproduces 57/0, 15/0) |
| B | §7 size-free pair beats the per-size supervised backward planner at g24r4: 215 vs 205, regret 2.43 vs 4.20, +10 solves p=0.006, 61/12 | SUPPORTED WITH CAVEAT (finding 3: vs the pinned "exact pair" reference the numbers reproduce exactly; vs the best recorded per-size BASE-vocab pair at g24r4 — the nntwin pair, 212/232, regret 3.05, `SV/scaling/results/g24r4/comparison_nntwin.json` — the solve gain is +3 (p=0.375, n.s.); the moves gain stays significant, 39/13 p=4e-4 (A*), 50/1 p=5e-14 (MCTS)) |
| C | §7 no loss at g16r4 (401/450 = per-size v2 pair) | SUPPORTED |
| D | §7(d) value net contributes little in the arena A*; the collapsed constant-value control scores the same | SUPPORTED (finding 5: cold value val_group_spread 4e-7; same 215 solve set; cold even 11/4 better on moves, n.s.) |
| E | §8 size-free MCTS "at the base-language moves ceiling" (0.06–0.12 above; further gain ≤ 0.1 moves) | SUPPORTED WITH CAVEAT (finding 7: the probe's `best_realizable_moves` is not a valid bound — planners already beat it on individual instances by up to 8–11 moves; a full-enumeration re-probe lowers the mean ceiling regret measurably on the subsets tested; the SOLVE ceiling is sound) |
| F | §9 zero-shot 32×32: 156/175 vs 147/175 (p=0.004; 20/4); 8 robots 147 vs 144 (p=0.25; 17/8); frontier +9/+14/+17 | SUPPORTED WITH CAVEAT (statistics reproduce exactly; the recorded 32×32/8-robot reference rows carry no move dumps so they are not replay-certifiable from the file; "every per-size supervised backward planner" holds only within the base vocabulary + prefix-check protocol — recorded B2-vocabulary per-size arms reach 154–171/175 at g32r4) |
| G | §3/§8/§11 base-language SOLVE ceilings (408/450, 216/232, 156/175, 148/161, 103/218, 142/275) | SUPPORTED (finding 7a: `capped=0` in every base arm; no base-vocab payload across 21–27 files per exam ever solves an instance the probe calls unrealizable; g16r4 408 = 22 no-plan + 20 unplayable reproduces supervised FINDINGS §4) |
| H | Like-for-like protocol / replay certification of every solved row in the new payloads | SUPPORTED (finding 1) |
| I | No train/bench leakage for the size-free nets | SUPPORTED (finding 2) |

## Numbered findings

### 1. Like-for-like protocol and certification — fine

Evidence (script over every payload under `SPR/results/{m0,m1,m2,transfer}` and the
recorded references):

- `instances_sha256` matches the reference on every headline pair: g16r4 bench450
  `1b384dfd375c…` (new m1/m2 vs `SV/eval/results/final450_backward_prefix.json`),
  g24r4 `0e8a5bada3a0…` (vs `SV/scaling/results/g24r4/comparison.json` and the M0
  same-machine re-run `SPR/results/m0/g24r4_exact_prefix.json`), g32r4 graded
  `da01708531ed…`, g32r4 frontier `3101e863e78a…`, g24r8 graded `e3e2fe34ec77…`,
  g24r8 frontier `90f16ec3ec05…`, g24r4 frontier `1d6747fcab49…` (vs
  `comparison_ungraded_nntwin.json`). Row order and `d_star` agree row-for-row
  (0 mismatches on graded exams).
- Every new payload: `protocol.expansions=1200, k=5`; `search_options.prefix_check=True`
  for every arena/A*/MCTS/greedy arm in §5/§7/§8/§9; references were all run with
  `--backward-prefix-check` (from `protocol.command`), base vocabulary. Device differs
  (new: cuda; references: cpu) — the M0 parity run shows this is float drift only
  (aggregates identical, `SPR/results/m0/g24r4_exact_prefix.json`; row diffs 34/232
  on expansions ±1–5, 3 rows realized_strict ±1). §7's win/loss counts (61/12,
  58/10, 59/11) were computed against the same-machine M0 re-run; against the recorded
  CPU file they are 62/11, 58/8, 60/10 — immaterial.
- `payload["spr"]["replay_certified"] == True` on all 46 new payloads (m0 2, m1 8, m2 28, transfer 8), and in every one
  the number of rows with a `moves` dump equals the number of solved rows (no solved
  row lacks a dump). Re-ran `python -m eval.replay_validate --compare … --env-dir …`
  on six files: `m1/mixed_value_warm_s21_g24r4.json` (215 passed, 0 failed),
  `m1/mixed_value_warm_s21_g16r4.json` (401/0), `m2/sizefree_mixed_warm_g24r4_mcts_min.json`
  (215/0), `m2/persize_exact_g24r4_mcts_min.json` (205/0), `transfer/g32r4_graded_astar.json`
  (156/0), `transfer/g24r8_graded_mcts.json` (147/0).
- Caveat (minor): the recorded per-size references at g32r4/g24r8/g24r4-frontier
  (`SV/scaling/results/*/comparison*.json`) have `moves` dumps for 0 backward rows, so
  their solve counts rest on `eval.compare`'s internal strict-realization criterion
  (`SV/eval/compare.py:467`), not on independent replay. The g24r4 exact reference was
  re-run and certified on this machine (M0), the others were not.

### 2. Leakage — fine

- Split code: `SPR/spr/train.py:198-215` filters each corpus by its config's board
  ranges via `nn_labeler.dataset.by_split` (`SV/nn_labeler/dataset.py:78-96`).
  Re-running load+split on the two corpora reproduces `runs/spr/m1/mixed_*/DATA.json`
  exactly (g16r4: 4008 train / 1975 val groups, max train env_id 1799; g24r4: 6612 /
  1886, max train env_id 699). Bench boards (2400–2549, 900–1049) are in the `test`
  range and are dropped from train and val.
- Instances: reconstructed (env_id, positions-by-color, target) for all 181,768 +
  53,789 corpus records; 0 of the 450 bench450 / 232 bench.solved / 218 bench.unsolved
  instances appear anywhere in either corpus (exact or order-insensitive match), let
  alone in the train split. (The corpora do contain other instances on bench boards —
  7,739 / 6,739 records — but those are test-range and excluded.)
- Encoder/value initialisation (`assets/labeler_prod_v1_s11.ckpt`) was trained on
  g8r4–g15r4, g16r6, g16r8 corpora only (`SV/nn_labeler/jobs/prod_train.slurm:31-40`;
  the legacy g16r4 corpus is explicitly excluded), i.e. on different board directories
  from every bench.
- Self-play: `runs/spr/selfplay/*/records.jsonl.manifest.json` board_ids 5000–5119,
  5120–5239, 8000–8059 on their own `runs/spr/boards/<iter>` directories; verified
  min/max env_id in the three `records.jsonl` and the iter-2 `buffer.jsonl` — no id
  < 3000. Buffer splits `train=5000-5101,…`, `val=5102-5119,…`.

### 3. Reference selection for §7/§9 — caveat

- §7's per-size opponent at g24r4 is the pinned "exact pair" (205/232, regret 4.22;
  `SPR/spr/gate.py:23` REFS). The repository also holds a per-size BASE-vocab
  prefix-check pair for g24r4 that is much stronger:
  `SV/scaling/results/g24r4/comparison_nntwin.json` (2026-08-07): 212/232, regret 3.05,
  45.8 % opt, 16.0 exp. Recomputed paired vs that file: mixed size-free A* 215 vs 212
  (A-only 4, B-only 1, McNemar p = 0.375; both-solved moves 9.79 vs 10.53, 39/13,
  p = 4e-4); size-free MCTS 9.13 vs 10.53 (50/1, p = 5e-14); g24-only pair 216 vs 212
  (p = 0.125). So the +10-solve / p = 0.006 headline is specific to the exact-pair
  reference; against the best per-size base pair the solve-rate gain is not
  significant while the moves gain is. §7's text ("beats the per-size supervised
  planners at 24×24 (+4–5 solve pts …)") should name the reference. The nntwin arm is
  not mentioned anywhere in `SPR/FINDINGS.md`.
- §9 says the pair "matches or beats every per-size supervised backward planner at
  32×32 and at 8 robots". Within base vocab + prefix-check that holds against all
  recorded arms (g32r4: exact 147, nntwin 148, nndeploy 146 → size-free 156, McNemar
  p = 0.0039/0.0078/0.002; g24r8: exact 144, nntwin 110/118 → 147). It does NOT hold
  against the recorded B2-vocabulary per-size arms (`SV/scaling/results/g32r4/
  comparison_b2*.json`: 154, 159, 163, 167, 171/175; g24r8 `comparison_b2retrained.json`
  154/161), which are a different plan language. The sentence needs the "base
  vocabulary" qualifier (the table's row label already implies it).
- Single seed (21) for every size-free net; no seed replicate of the size-free pair
  exists. Partial mitigation: three differently-trained size-free pairs (mixed,
  g16-only, g24-only) land at 213–216/232 at g24r4.

### 4. Statistics — fine (all quoted numbers reproduce)

Independent recomputation (exact two-sided McNemar on discordant solves; sign test on
both-solved strict moves), file → result:

- §5 g16r4 per-size MCTS vs greedy: 401 vs 366, A-only 35/B-only 0, p = 5.8e-11;
  moves 7.44 vs 8.03 on 366 both, 58/0, p = 6.9e-18. MCTS vs A*(child): 401 = 401,
  7.92 vs 8.38, 57/0, p = 1.4e-17. g24r4: vs greedy +70, p = 1.7e-21, 29/0 (p = 3.7e-9);
  vs A* 15/0 (p = 6.1e-5). A*(child) reproduces the CPU record row-for-row (0/0 at g16r4;
  1/1 vs the recorded g24r4 file, 0/3 vs the M0 re-run = the float-drift rows).
- §7 mixed vs exact pair at g24r4: 215 vs 205, 11/1, p = 0.0063; 61/12, p = 4.8e-9
  (vs M0 re-run) — matches. mixed vs v2 pair at g16r4: 3/3, p = 1; 44/37, p = 0.51.
  g16-only at g24r4: 10/2, p = 0.039; g24-only: 11/0, p = 0.00098; g24-only at g16r4
  39/21, p = 0.027.
- §8: sf-MCTS vs greedy g16r4 +53, 80/0 (p = 1.7e-24); g24r4 +10, 24/0 (p = 1.2e-7);
  vs A*(child) 59/0 (p = 3.5e-18), 27/0 (p = 1.5e-8); sf-MCTS vs per-size MCTS g16r4
  42/27, p = 0.091.
- §9: g32r4 graded A* 156 vs 147, 9/0, p = 0.0039, 20/4 (p = 0.0015); g32r4 frontier
  141 vs 127, 14/0, p = 1.2e-4; g24r8 graded 147 vs 144, 3/0, p = 0.25, 17/8 (p = 0.11);
  g24r8 MCTS 22/5, p = 0.0015; g24r8 frontier 171 vs 154, 21/4, p = 9.1e-4, 46/29
  (p = 0.064, correctly not claimed); g24r4 frontier 99 vs 90 (twin), 11/2, p = 0.022.

### 5. Value-net contribution (§7d) — fine, and it sharpens the interpretation

- `runs/spr/m1/mixed_value_cold_s21/lightning_logs/version_{0,1}/metrics.csv`:
  `val_group_spread` = 1.1e-6, 7.2e-7, 4.9e-7, 5.5e-7 (v0), 4.5e-7 … 4.4e-7 (v1) at
  every epoch; `val_top1_optimal` 0.19–0.27 (chance-level for ~5 candidates); RESULT.json
  best_ckpt = v1 epoch 2 (val_regret 3.39, spread 3.7e-7). This is a constant-output net.
- With it, the arena A* gives 215/232 (identical solve set to the warm value: 0/0
  discordant), regret 2.23 vs 2.43, moves 11/4 in the cold net's favour (p = 0.12), at
  10.7 vs 5.3 expansions; at g16r4 401 = 401, 25/14 for cold (p = 0.11).
- Implication: in `_nn_astar_backward` the value term is inert for the size-free pair —
  the ordering is effectively uniform-cost over the policy's top-5, so the entire §7
  gain over the per-size pair is a POLICY effect (candidate ranking/pruning), as the
  text says. Two consequences a reviewer will raise: (i) the per-size comparison mixes
  policy and value changes — no "per-size policy + constant value" control was run, so
  it is unknown whether the per-size value net is helping or hurting its own pair;
  (ii) the value net is not exercised by any headline number in §5–§9 (see finding 6:
  MCTS closes its tree, so leaf values only order the enumeration).

### 6. What "MCTS" measures in §5/§8/§9 — caveat

- In every graded M2/transfer MCTS payload the root CLOSED on 100 % of instances
  (`row["search"]["root_closed"]`: 450/450, 232/232, 450/450, 232/232, 175/175; g24r8
  132/161 with 29 at the 1200 cap) — the k=5-pruned plan tree is enumerated
  exhaustively and every complete leaf is certified (`n_certified` mean 3.6–6.0 per
  instance vs 0.9–1.1 strict-realization calls for the A* arms; wall time 3–5×). So
  "MCTS best-at-budget" here == exhaustive certified enumeration of the top-5 tree; min
  vs mean backup being identical is a tautology, and the value net's PUCT role is limited
  to visit order.
- The comparison to the arena A* (first complete plan, one strict realization) therefore
  measures the benefit of certifying every complete plan in the pruned tree and keeping
  the shortest — a real and reproducible gain, but the "same budget" statement holds for
  the 1200-expansion CAP only: MCTS spends 3–4× the expansions and ~5× the physics calls.
  The A* best-at-budget arm that also certifies at pop was run only with `f_mode=parent`
  (`SPR/results/m2/*_astar_best.json`), which is the worse f at g24r4 (§5b); no
  `f=child + best-at-budget` arm exists, so the strongest A* variant was not benched.
  MCTS still beats the parent-mode anytime A* (g16r4 1.68 vs 2.03; g24r4 sf 1.78 vs
  2.33), so the qualitative claim stands.

### 7. Ceiling probe (`SPR/spr/ceiling.py`) — solve ceiling sound; moves ceiling is not a bound

7a. Solve ceiling: sound.
- Every base arm has `capped=0` (`results/ceiling/{g16r4,g24r4,g32r4,g24r8}_base.json`,
  `{g24r4,g32r4}_frontier_base.json`), i.e. the search ran to frontier exhaustion; the
  frontier is only pruned by the same `max_open = 2*(helpers+2)` guard the arena's
  `skeleton.astar.AStar._search` uses (`SV/skeleton/astar.py:64,76`) and by a plan-key
  dedup that the arena lacks. Dedup sensitivity test on 40 bench450 instances (coarse
  key vs endpoint-aware key vs no dedup): identical category/first/best/iters on all 40.
- g16r4 base: `categories = {REALIZABLE_EXISTS: 408, NO_COMPLETE_PLAN: 22,
  NO_REALIZABLE_PLAN: 20}` — reproduces supervised FINDINGS §4 exactly.
- Cross-check against everything ever benched: over 21 (g16r4) / 27 (g24r4) / 7 / 7
  base-vocabulary payloads (spr + recorded), no planner solves any instance the probe
  marks unrealizable. (B1/B2-vocab arms solve 27–31 of the 42 g16r4 "unrealizable"
  instances, as expected for a larger language.)
- The B2 probes are NOT exhaustive: `capped` = 12 (g16r4_b2), 7 (g24r4_b2), 115/124/133
  on the frontier B2 arms (`max_frontier` lowered to 100k there); §12 correctly reports
  those as "≥".

7b. Moves ceiling (`best_realizable_moves`): not a valid upper bound on the language
optimum, and it is already contradicted by the planners' own rows.
- The bound relies on strict ≥ abstract (`ceiling.py:97-99`: stop when popped abstract
  cost ≥ best strict + slack). Strict undercuts abstract by large amounts in practice:
  size-free MCTS rows with abstract 22 → strict 11 (env 977, d* 10), 28 → 20 (env 983),
  14 → 11 (908), 15 → 7 (g16r4 env 2405), 12 → 7 (2537). Result: certified planner
  plans strictly SHORTER than the probe's "best" exist on 4/215 g24r4 instances
  (`sizefree_mixed_warm_g24r4_mcts_min.json`: 908 11<14, 977 11<19, 983 20<26,
  1045 11<12), 3/401 at g16r4, 3/205 for the per-size MCTS, 2/147 at g24r8; the
  slack-4 re-run still leaves 977 (17 vs planner 11), 983 (24 vs 20), 2405 (11 vs 7).
- Magnitude: taking min(probe slack-4 best, best strict of ANY base-vocab planner row)
  per instance moves the mean ceiling regret 1.397 → 1.387 (g16r4), 1.685 → 1.639
  (g24r4), 1.581 → 1.547 (g24r8), 1.558 → 1.558 (g32r4). A full-enumeration re-probe
  (slack = ∞, 30 s cap) on subsets from the login node: see 7c. So the quoted "0.06–0.12
  above the ceiling" (§8b) is "0.06–0.15 above the best plan anyone has found"; the true
  language optimum is unknown and lower. §8's "any further gain … ≤ 0.1 moves" is not
  established for the moves metric (it is established for solves).
- The docstring (`ceiling.py:26-31`) states this caveat; §3/§8/§9 prose does not.
- Recommendation (cheap, no GPU): re-run the base probes with a very large `--slack`
  (full enumeration) — the frontier is small (mean 1.6–7 k pops, 2–7 s per instance in
  the test below); use the exhaustive value as the moves ceiling.

7c. Full-enumeration re-probe (login node, same `probe()` with `slack=1e9`, 30 s cap,
first 103 of 232 g24r4 bench.solved instances and first 80 of 450 bench450 instances,
scratch script `dedup_check.py`/`slack_check.py`, not committed):
- g24r4 (n = 103): full enumeration hit the 30 s cap on 9 instances (their optimum is
  therefore still unbounded); it found shorter certified plans than the bounded probe on
  2 instances (env 908 14 → 11, env 912 13 → 12); mean best-plan regret on the subset
  2.234 → 2.191 (−0.043); 0 new solves; mean 3.0 k pops / 4.9 s per instance.
- g16r4 (n = 80): capped 1; improved 1 (env 2405 11 → 7); mean regret 1.694 → 1.639
  (−0.056); 0 new solves; mean 2.3 k pops / 2.0 s per instance.
- Reading: on ~40 % of each bench the exhaustive language optimum is ≈ 0.05 moves below
  the probe's "best"; extrapolated, the ceiling regrets in §3/§8 (1.42 / 1.72) are
  ≈ 0.05 too high, comparable to the whole reported "gap to ceiling" (0.06–0.12), so
  the honest statement is "MCTS is within ~0.1–0.2 moves of the exhaustive optimum on
  the top-5 tree", not "at the ceiling"; and the true optimum on the capped instances
  is still unbounded. The solve ceilings are unaffected (0 new solves).

### 8. Miscellaneous checks — fine

- Checkpoint selection for the size-free nets is by validation regret
  (`runs/spr/m1/*/RESULT.json best_ckpt` = ModelCheckpoint(min val_regret)); nets.txt in
  `SPR/results/m1/*.nets.txt` cite exactly those checkpoints; no bench-based selection.
- M1 gate files (`SPR/results/m1/*.gate.json`) agree with the recomputation (deltas,
  paired counts) and use the pinned references (`gate.py` REFS).
- Payload aggregates match FINDINGS tables (spot-checked all rows in §5, §7, §8, §9).
- Recorded g16r4 v2 reference (`final450_backward_prefix.json`, 2026-07-13, cpu) and the
  new per-size A*(child) arm are row-identical (0 discordant, 0/0 moves), which validates
  the spr search stack against `eval.compare` at g16r4.
