# M6 — "beyond the oracle": self-play at 80×80 / 96×96 (design, Phase 0)

PROBLEM.md §8/M6, owner-approved 2026-08-26. At these sizes **no exact solver
output exists anywhere in the project** (the exact corpora stop at 64); physics
certification (strict replay of every solution) is the only ground truth. This
document fixes the protocol BEFORE compute is spent (house rule). Results log:
`M6_FINDINGS.md`. Budget: ≤20 nh.

## 1. What exists already (verified 2026-08-26)

- **Configs**: `g80r4`/`g96r4` are registered in `scaling.configs` (walls 1200 /
  1728, env dirs `environments_g{80,96}r4`, marked UNVERIFIABLE) with 20 lean
  boards each (ids 0–19, from the 2026-08-05 coarse-ladder prep). Lean-board
  construction is cheap (≈2–4 s/board; FINDINGS 62).
- **Nets**: the mixed-curriculum pair (`mix_b2mix_iter3`, trained on 24/32/8r
  self-play, pe=none, 96 cost bins) runs at any size by construction; exact
  audits certify it (and its lineage) at 32–64 (§17: value top1 0.838–0.863,
  beats the labeler at every rung).
- **Search**: the flagship hybrid (root slides, depth 2) is size-free — its
  ceiling break transferred zero-shot 24→32 and 24→8r (§26).
- **Certification**: `eval.realize.strict_moves` + `eval.replay_validate` are
  size-agnostic; the arena's chunked bench takes `--boards lean` + RR_* env.

## 2. Feasibility measurements (Phase 0)

### 2a. The 96-bin value clamp (login-node, exact/descent corpora)
Abstract cost-to-go distributions vs the head's 0–95 range:

| corpus | n | max | p99 | p95 | ≥90 |
|---|---|---|---|---|---|
| g32r4 exact | 51,353 | 91 | 46 | 36 | 3 |
| g48r4 exact | 11,394 | 89 | 54 | 41 | 0 |
| g64r4 exact | 11,321 | 93 | 57 | 47 | 5 |
| g56r4 descent-gen | 1,382 | **103** | 52 | 44 | 2 |
| g64r4 descent-gen | 1,377 | 72 | 53 | 43 | 0 |

The tail already touches the clamp at 56–64; at 80/96 some certified labels
WILL exceed 95. `collate_groups` clamps silently (with a counter) — the §4.5
lesson says never train on silently clamped tails. **Decision: drop (not
clamp) records with cost_to_go > 95 at buffer time and report the drop rate;
escalate to a 128-bin head-widening surgery only if the drop exceeds 2% of
records** (projected <1%: p99 at 64 is 57; the smoke measures the real rate
at 80).

### 2b. The Rust envelope is conventional, not structural — a 72×72 rung is cheap
A live probe (login node, 1 instance, 100k iters): the engine labels a 72×72
lean-board instance in **2.6 s, status ok, exact records produced**. Nothing
in `rust_datagen` hard-caps n at 64 — the "envelope" was where corpus-building
stopped. **Decision: build a g72r4 exact audit corpus** (test-split boards,
~150×10 rollouts, one qgpu job, ≈0.3–0.5 nh) and add the 72 rung to the
fidelity curve (`spr.audit`). This moves the last verified rung from 64 to 72
— 78% of the way to 80 by area — and is the strongest cheap improvement to
the warranty argument available. (A g72r4 config shim lives in M6 code only;
`supervised_valuenet` stays untouched.)

### 2c. GPU numbers (job 4829842, qgpu_exp; results/m6/smoke.jsonl)
<!-- SMOKE -->

## 3. Protocol

- **Exam (pinned, never trainable)**: fresh lean boards ids **30000–30049**
  (80×80) and **30100–30124** (96×96), 4 uniform instances per board → 200
  instances at 80, 100 at 96; built by an M6 copy of the variants exam
  builder into `results/m6/exam/`; committed as instance JSONLs (d_star=0
  placeholders — no optima exist, and none can).
- **Metrics**: certified solve rate, realized strict moves (both-solved
  pairings), expansions; replay certification on every row (unchanged
  arena machinery). NO optimality claims of any kind at 80/96.
- **Arms at 80** (1200/k5 budget, B2 vocabulary): (a) labeler-descent greedy
  (the supervised bootstrap planner — the only supervised system that runs
  here); (b) mixed nets, arena A\*; (c) mixed nets, MCTS best-at-budget;
  (d) mixed nets, hybrid-d2 (the flagship search). At 96: (b) and (d) only
  (inference fits; see 2c).
- **Self-play round at 80** (Phase 1): generation on fresh boards ids
  31000+ (never the exam), v14 recipe (emit-all + strict-value), clamp-drop
  rule from 2a, retrain warm from the mixed pair, re-bench all arms.
- **Gates**: (i) retrained vs zero-shot on the 80-exam — paired McNemar on
  solves + sign test on both-solved strict moves; (ii) **warranty**: the
  retrained nets re-audited with `spr.audit` at 32–72 — training at 80 must
  not degrade exact-checkable fidelity by more than the §18 seed bar
  (matched-instrument comparison vs the §17 rows + the new 72 rung);
  (iii) the 96 exam re-benched zero-shot (does 80-training transfer up?).

## 4. Cost projection (filled after the smoke)
<!-- COST -->

## 5. Honesty rules

Every 80/96 number is a certified LOWER bound on solvability and an UPPER
bound on required moves; no statement about optimality is possible and none
will be made. The warranty argument is explicitly an extrapolation: exact
fidelity holds at every checkable size (32–72) and the certified benches are
exact at all sizes; what cannot be checked is whether the value net's
RANKING at 80/96 is as good as at 72 — stated as such wherever M6 results
are reported.
