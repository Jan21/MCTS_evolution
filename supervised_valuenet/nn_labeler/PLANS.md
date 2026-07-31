# NN labeler — three plans for replacing the exact solver in data labeling across grid sizes

Written 2026-07-31 after a three-way research pass over the repo (labeling pipeline,
network architecture, board/config infrastructure). Goal set by the owner: the exact
solver does not scale as a *labeler*; train a neural network that generalizes across
grid types (training data possibly mixing several grids), then walk up the size ladder
(17×17, 18×18, … 32×32) generating labeled data at each rung. Debug the whole process
on a small setup (8×8) before committing to larger grids. Deliverable: an HTML document
describing the process. Constraints: one GPU at a time, small experiments before big
runs, ≤16 h walltime (cooling-reservation rule in the MOTD), `qgpu` only, projected
node-hours stated in chat before big submissions. This file holds the three candidate
plans; the chosen plan's execution log goes to FINDINGS.md as usual.

---

## 0. Why — the measured wall (all numbers from FINDINGS.md / BENCH.md / meta sidecars)

Labels, not search, are the scaling bottleneck. Two independent walls:

**Backward (subgoal) labels.** Each candidate at each decision costs one full
plan-space A* solve (`nn/generate.py:96-107`). With the extended B2 vocabulary the
rollouts "wander": an iteration budget does not bound wall-time because per-iteration
cost grows with plan size (FINDINGS §30). The original B2 label campaign projected
**~441 node-hours against ~879 then remaining** — with the Rust engine at 16 threads,
i.e. no faster implementation to switch to. Recalibrating the cap to 5,000 iterations
made the campaign land in ~5 nh but **depleted exactly the candidate type the labels
exist to teach** (by-reference share 13.5% → 4.8%; the cap-5000 retrain then lost
−24.6 points on the frontier, FINDINGS §32/§34). Cap 20,000 is the current compromise
and still costs 16–53 s/board and rises along the grid axis: median legitimate-rollout
iterations 170–253 (16×16) → 465 (24×24) → 1,062 (32×32), with per-iteration cost
growing on top. The exact labeler is a dead end above 32×32, and already painful there.

**Forward labels and bench grading.** The move-space A* oracle
(`move_planner/oracle.py`) dies on both axes at fixed caps (200k expansions / 60 s):
failure 0% (16×16/4r) → 29.8% (6r) → 40.9% (8r) → 48.4% (24×24/4r) → **64.2%**
(24×24/8r) → 61.1% (32×32/4r). State space is (G²)^R. Supervised forward training
"runs out of teacher"; bench sets above the base rung carry placeholder optima.

**The opening the domain gives us:** every backward plan can be realized to a real
move sequence and replayed through `simulate.py` alone (`eval/replay_validate.py`).
A solution found by *any* means is self-certifying, and its cost is a true upper
bound. NN-produced labels therefore never need to be trusted — they can be certified
playable, and their optimality gap can be audited wherever the exact solver still
works. All three plans below exploit this.

---

## 1. Facts the plans build on (established in this session's research pass)

1. **The nets are one tensor away from size-invariance.** All three nets (backward
   value `train/looped_pc.py`, backward policy `train/policy_tf.py`, forward
   `move_planner/net.py`) are the same weight-tied looped transformer over per-cell
   tokens with graph-masked attention. The **only** grid-size-dependent weight is the
   learned absolute positional embedding `self.pos: [G²+1, 192]`. Everything else —
   encoder linear, the ~741k-parameter looped block, readout heads — is size-agnostic.
2. **Cross-robot-count zero-shot transfer already worked.** The g16r4-trained backward
   pair ran unchanged on 6- and 8-robot boards and was *slightly better* than
   per-config nets (net-provenance effect +2.2/+7.5/+0.8/+3.8, every sign favoring the
   base nets — FINDINGS §29). Cross-grid transfer was never attempted (blocked by
   `pos`), but the analogous bet has already paid out once.
3. **Retraining on same-size data repeatedly *hurt*** (FINDINGS §34, §47: "zero-shot
   remains the method's best configuration at every rung measured"). This weighs
   against plans that lean on per-rung fine-tuning and for plans that train once, well.
4. **Board generation is size-generic today**: `nn/gen_grids.py` reads `RR_GRID` /
   `RR_WALLS` / `RR_ROBOTS`; walls scale as `round(48·(G/16)²)` (8×8 → 12,
   17×17 → 54). Adding a rung is one `_std()` line in `scaling/configs.py`. The Rust
   engine is fully size-parametric (asserted envelope n ≤ 64, R ≤ 10) and its work-item
   wire format already carries `"n"` + inline `grid_data` — the size-agnostic format
   exists. The "lean board" path (`rust_datagen/pyref/dump_decisions.py`) skips the
   O(G⁴) all-pairs pickles that make big-board dirs huge (37 GB at g32r4).
5. **Data plumbing blockers for mixed-size training**: records carry no grid field and
   `env_id` is reused 0–1199 by every config (silent aliasing if merged); the encoders
   freeze `GRID` at import from env vars (one config per process); `@lru_cache` keys on
   `env_id` alone. All fixable in a new module without touching frozen study code.
6. **Costs of the pieces we reuse**: exact base-vocab labeling at ≤16×16 via Rust is
   cheap (order of ms–100ms per board·instance batch at r4–r8, 16 threads); value-net
   training at 16×16 is ~8 min/epoch (bs 8) and at 32×32 ~45 min/epoch (bs 2) on one
   A100 — the O(G⁴) dense attention mask is the driver. Value bins: new nets get
   `num_classes` sized for 32×32 costs from day one (no warm-start constraint).
7. **House rules that bind here**: never mix vocabularies in one dataset (cost-to-go is
   vocabulary-relative); playable-moves scoring only; FINDINGS.md is the single log;
   the HTML deliverable stays local (never published); `qgpu` only; ≤16 h walltime.

---

## 2. Phase 0 — shared prerequisites (identical for all three plans)

**S0.1 Size-invariant encoding.** Replace the learned `pos` with a size-parametric
position signal. Candidates, ablated at 8×8 during debug: (a) 2-D sinusoidal absolute
PE computed from (x, y) at runtime; (b) normalized coordinate input channels
(x/G, y/G, plus wall-distance features); (c) both. The slide-graph structure already
enters through the attention masks, so absolute position is plausibly secondary —
measure, don't assume.

**S0.2 Per-record size plumbing.** New dataset/loader in `nn_labeler/` where every
record carries `n` (grid side) and a corpus namespace (`config` string); adjacency and
features built per-sample from `n`; batches bucketed by size (dense [N+1, N+1] masks
make cross-size padding wasteful). Existing 18-field schema kept as a subset —
NN-labeled records stay drop-in for the existing trainers, with added provenance
fields (`n`, `label_source`, `ctg_certified`, `label_model_sha`).

**S0.3 Registry entries** for debug/ladder rungs (g8r4, g10r4, g12r4, …, g17r4 …)
via `_std()` lines; lean boards (no all-pairs pickles) for rungs used only by the NN
labeler.

**S0.4 New code lives in `supervised_valuenet/nn_labeler/`**; frozen study code is
imported, not modified. Anything generally useful (e.g. a `--grid/--robots` flag on
board gen) goes in as additive, default-off changes.

**S0.5 The audit harness (the honesty instrument).** At any rung: draw a subsample of
decisions, run the *exact* solver with a generous budget on CPU, and report (i)
exact-solver coverage, (ii) label regret of NN cost-to-go vs exact on the covered
stratum, (iii) certified-playability rate of emitted plans, (iv) is_optimal agreement.
This is what "the labels are good" means everywhere below; thresholds get calibrated
at 8×8–12×12 where exact coverage is 100%.

**8×8 debug protocol (all plans).** g8r4 (12 walls): generate ~300 boards + exact
base-vocab labels (Rust, minutes); same at 9×9/10×10/12×12 for transfer audits; train
the size-invariant net on 8×8 only, then on {8,10,12} mixed; verify (1) pipeline runs
end-to-end, (2) mixed-size training matches per-size training at each size (no
interference), (3) zero-shot label quality at 12×12 from {8,10}-training, and at
16×16 against the existing exact g16r4 corpus (free audit — labels already exist),
(4) the plan-specific mechanism (A: guided-solver speedup; B: certified descent
labeling; C: extrapolation curve) works at toy scale. Cost: ~1–2 node-hours total.

---

## 3. Plan A — "Learned-heuristic oracle": the NN accelerates the exact solver; labels stay exact-or-audited

**Idea.** Don't replace the solver — amortize it. The wanderer pathology *is* the cost
wall (FINDINGS §30: budget-exhausted rollouts burn 65–91% of all iterations), and
wandering is exactly what a trained value function kills: hopeless
plateau/relocation branches get scored to the bottom of the frontier and are never
expanded. The exact labeler machinery, budgets, and record semantics stay unchanged.

**Mechanism.**
- Train size-invariant value (+ policy) nets on exact labels from cheap sizes
  (8–16 mixed), per S0.
- Inside `AStar._search`, use the net two ways, both optional and flagged:
  (i) *candidate ordering* — replace/augment `heuristics.score` for expansion order
  (pure tie-breaking: first-completed-plan-is-optimal argument survives untouched);
  (ii) *frontier priority* — weighted-A*-style `g + w·h_NN`, which abandons the
  optimality guarantee for a bounded-suboptimality label (record `w` and audit the
  realized gap).
- Walk the ladder 17 → 32: at each rung, label with the guided solver; the audit
  (S0.5) measures how often guided search completes within budget vs the unguided
  baseline, and label regret in mode (ii).
- Optionally fine-tune the guide on each rung's fresh labels before the next rung
  (the guide only steers *search order*, so even a mediocre guide cannot corrupt
  mode-(i) labels — the safest possible use of a net).

**Where it runs.** Guidance calls happen inside the Python solver first (correctness),
then — only if the Python path is too slow to be useful — via a batched NN sidecar to
the Rust engine (a real integration project; decision point, not default).

**Cost projection** (estimate; re-sized from smoke runs per the §30 operating rule):
Phase 0 + debug ~2–3 nh; guide training ~2–4 nh; per rung: labeling ~0.5–2 nh
(the whole point is that this collapses vs the exact baseline) + audit ~0.5 nh;
16 rungs ≈ 20–45 nh total.

**Risks / kill criteria.**
- Mode (i) may not save enough: ordering only helps if completions exist within
  budget. Kill if 8×8–16×16 A/B shows <3× reduction in budget-exhausted share.
- Python-side guidance may be too slow (NN call per expansion on CPU); the Rust
  sidecar is the expensive fix. Decide at the 16×16 A/B.
- Mode (ii) labels need the same audit machinery as Plan B — at which point B does
  the same thing without carrying the solver.

---

## 4. Plan B — "Certified descent": per-rung expert iteration; the NN planner *is* the labeler

**Idea.** The expensive call in labeling is `solve_plan` per candidate (exact optimal
completion). Replace it with a *greedy/beam descent* using the trained policy+value
nets — the same machinery the eval-time backward planner already uses — then
**realize the finished plan to moves and replay it through the physics** for
certification. Committing to *each* candidate at a decision and descending to
completion yields a **certified upper-bound cost-to-go per candidate** at pure
NN+physics cost (no A* anywhere). `is_optimal` becomes argmin of certified costs;
loose upper bounds are measurable as label regret via the audit.

**Mechanism.**
- Bootstrap: exact labels at 8–16 (cheap), train size-invariant nets (S0).
- Rung loop for G = 17 … 32:
  1. Generate lean boards at G (walls formula), sample instances.
  2. For each instance: NN rollout as in `nn/generate.py` but with `solve_plan`
     replaced by candidate-wise NN descent + realization; emit 18-field records with
     `ctg_certified: true` where the completion realized, plus hindsight-relabeled
     records along the executed trajectory (suffix costs of the realized plan).
  3. Audit (S0.5) against the exact solver on a subsample — at 17–24 coverage is
     still decent; report the recoverable stratum honestly above that.
  4. Fine-tune the nets on the cumulative mixed-size corpus (**decision point per
     rung**: fact 3 above says fine-tuning may hurt — B carries C inside it as the
     null hypothesis: only accept a fine-tune that beats the zero-shot net on the
     next rung's audit).
- Unlabelable instances (descent never completes a plan) are dropped and counted,
  mirroring the current wanderer-drop accounting.

**Cost projection.** Phase 0 + debug ~2–3 nh; bootstrap corpus + training ~3–5 nh;
per rung: generation is GPU-light (a descent is ~plan-depth × (1 policy + k value)
passes; even ×14 candidates it is minutes-to-an-hour per rung on one A100) + audit
~0.5 nh + optional fine-tune ~0.5–2 nh; 16 rungs ≈ **15–35 nh total**.

**Risks / kill criteria.**
- Upper-bound looseness where the net is weak → systematically inflated cost-to-go
  for exactly the hard candidates. The audit measures this; kill a rung's corpus if
  mean certified-label regret at the audit stratum exceeds the threshold calibrated
  at debug (provisional: ≤0.5 moves mean, ≤10% is_optimal disagreement).
- Distribution drift compounding up the ladder (each rung trained on the last rung's
  biases). Mitigation: the audit gate per rung + keeping exact-labeled ≤16×16 data
  in every fine-tune mix.
- Vocabulary: start base (no wandering, cheap exact audits), extend to B2 only after
  the base ladder lands; never mix (house rule).

---

## 5. Plan C — "Train once, label everywhere": one size-invariant net, zero-shot ladder

**Idea.** The repo's own evidence (facts 2–3) says: nets transfer zero-shot across the
one axis that was tried, and retraining is where quality went to die. Plan C makes the
strongest architectural bet: train ONE size-invariant net pair on a size-and-robot
mixed exact-labeled corpus at cheap scales (8, 10, 12, 14, 16 × r4/r6/r8), and apply
it **zero-shot** at every target size — no per-rung fine-tuning, no ladder
dependency; 17×17 through 32×32 can even be labeled in parallel once the net is
frozen. Labeling mechanics are identical to Plan B's certified descent; the plans
differ only in whether the net evolves along the ladder.

**Mechanism.**
- Extrapolation curve first, then commitment: after training, measure label quality
  at 18, 20, 24, 28, 32 (audit harness + certified-playability) *before* generating
  bulk data anywhere. The curve tells us where zero-shot dies — perhaps it doesn't
  (the looped, weight-tied, graph-masked architecture with recurrence 12 is exactly
  the kind that extrapolates; recurrence may need to scale with G — a free knob at
  inference, worth one ablation).
- Bulk-label all rungs where the audit passes; hand the rungs where it fails to
  Plan B's fine-tune loop (C degrades gracefully into B).
- Interpolation control: hold out 13×13 and 15×15 from training; if the net is bad
  *between* training sizes it will never extrapolate — cheapest possible early kill.

**Cost projection.** Phase 0 + debug ~2–3 nh; mixed corpus (15 config cells ≤16×16,
Rust-labeled) ~2–4 nh; one big training run ~4–8 nh (mixed-size batches, one A100,
fits a 16 h job with resume enabled); extrapolation audit ~2–3 nh; bulk labeling of
all passing rungs ~3–6 nh. **Total ≈ 13–25 nh — the cheapest plan**, and the most
informative per node-hour (the extrapolation curve is a publishable finding on its
own, win or lose).

**Risks / kill criteria.**
- Zero-shot may genuinely fail at 2× the training diagonal (absolute-position
  artifacts, value-range shift, longer plans). That outcome IS the deliverable
  insight, and the fallback (B) reuses everything.
- Mixed-size batching + per-sample adjacency is the most invasive Phase 0 variant
  (C needs it fully general; A/B could limp along per-size). Budgeted in S0.2.
- Value-bin range must cover 32×32 costs from day one (choose ~96–128 bins; the
  0.036% truncation measured at g32r4/50 bins shows the failure mode).

---

## 6. Comparison and recommendation

| | A: guided exact solver | B: per-rung expert iteration | C: train once, zero-shot |
|---|---|---|---|
| label semantics | exact (mode i) / bounded (mode ii) | certified upper bounds | certified upper bounds |
| per-rung retraining | optional | yes (gated) | no |
| ladder coupling | sequential | sequential | parallel after training |
| biggest risk | speedup insufficient | compounding drift | extrapolation cliff |
| est. total nh | 20–45 | 15–35 | 13–25 |
| what failure teaches | little | drift dynamics | the extrapolation curve |

**Recommendation: run C's spine, keep B armed, hold A in reserve.** Phase 0 and the
8×8 debug are identical for all three; the certified-descent labeler is shared by B
and C. So: build once, train the mixed-size net, measure the extrapolation curve
(C); wherever zero-shot fails the audit, drop into B's gated fine-tune loop for
those rungs; only if certified-descent labels are systematically too loose does A's
solver-in-the-loop mode become the tool. This ordering front-loads the cheapest,
most informative experiment and matches the owner's step-by-step framing (the audit
walks 17→32 rung by rung even when the net is frozen).

---

## 7. Operational rules for execution (binding, from MOTD + house rules + this goal)

- One GPU at a time (`--gpus 1`, 0.125 nh/h); smoke first; state projected nh in chat
  before big submissions; never size a job from its opening boards (FINDINGS §30 rule).
- ≤16 h walltime on everything (MOTD: >16 h jobs do not start while the daily
  10:00–18:00 cooling reservation stands); resume/checkpoint enabled on any job that
  could hit the cap; chunked, idempotent, atomic writes (tmp+mv); grep-able DONE
  markers; one job = one log.
- `qgpu` only, account `open-37-42`; CPU-side work (Rust labeling, audits) runs inside
  the GPU job's 16 cores (`CUDA_VISIBLE_DEVICES=""` for CPU evals, `OMP_NUM_THREADS`
  pinned, `PYTHONUNBUFFERED=1`).
- Results append to FINDINGS.md first, numbers only from result JSONs; commit + push
  often; milestone artifacts copied to `/mnt/proj1/open-37-42/` (90-day purge).
- Vocabulary separation absolute; datasets/nets named by vocabulary AND by label
  source (`exact` vs `nnlab`); NN-labeled and exact-labeled records never mixed
  silently (provenance fields, S0.2).
- The HTML process report is a local file only, never published anywhere.
- Implementation labor: parallel Opus agents for well-specified small chunks; Fable
  for design, integration, and anything touching label semantics.
