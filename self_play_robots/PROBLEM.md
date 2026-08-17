# self_play_robots — problem definition & implementation handoff

**Created 2026-08-17** by the NN-labeler track session, as the successor
project to `supervised_valuenet/`. Everything below is written for the agent
(or human) who implements this: it defines the problem, inventories what
already exists, transplants the expensive lessons, and proposes the loop —
with the design decisions that are still open clearly marked as such.

---

## 1. North star

Build a complete **AlphaZero-style self-play loop** for Ricochet Robots
planning that ends up **beating every supervised planner we have** — the
forward move-level planner and the backward subgoal planner — on the one
metric that matters:

> **Realized primitive moves from initial state to terminal state**, on a
> fixed benchmark, under a fixed search budget. Fewer moves wins.
> Secondary: solve rate (a planner that solves fewer puzzles doesn't get to
> brag about the ones it solved), and compute per puzzle as the tiebreaker.

The supervised campaign (see §4) proved the pieces exist: a size-free value
net that writes near-optimal labels, planners that train to useful strength
from those labels, and a full NN-only data pipeline that matches the
exact-solver pipeline. What no one has done is **close the loop**: let the
network search, learn from its own search, and iterate past the ceiling of
its teacher.

## 2. The game, formally

**Ricochet Robots.** An n×n grid with interior walls. R robots occupy
distinct cells. A *primitive move* selects a robot and a direction; the
robot slides in a straight line until blocked by a wall or another robot
(it must slide at least one cell; a move that wouldn't slide is illegal).
A puzzle instance fixes a board, robot start positions, a designated
target robot, and a target cell. Terminal state: target robot on target
cell. Cost = number of primitive moves executed. Non-target ("helper")
robots matter: optimal play routinely parks helpers as blockers.

**As an MDP:** state = (board walls, robot positions, target spec); actions
as defined by the chosen action space (§6.1); deterministic transitions;
episode cost = moves to terminal; discount 1; solved/unsolved absorbing at
a search-budget horizon.

**Two action-space views exist in this repo** (both with trained nets):

- **Forward / move-level:** actions are primitive moves (≤ 4R). Deep
  horizon (optimal solutions run 8–30+ moves at the sizes we care about),
  small branching. This is the natural AlphaZero action space.
- **Backward / subgoal:** actions are *subgoal decisions* — pick the next
  segment endpoint / helper placement from a generated candidate set,
  reasoning from the goal backwards; a realization step converts the
  subgoal plan to primitive moves and physics-verifies it. Shallow horizon
  (2–6 decisions), branching = candidate-set size (up to ~50). This is the
  action space the strongest supervised planner uses.

**Scale:** primary configs are the studied grid/robot combos —
`g16r4` (16×16, 4 robots), `g24r4`, `g24r8`, `g32r4`; naming convention
`g<N>r<R>` everywhere. Ground truth (exact solver) exists **only for
n ≤ 64** (hard engine assert). Solution *validity* is checkable at ANY
size by physics replay; solution *optimality* only ≤ 64.

## 3. What you inherit (assets)

All paths relative to the repo root `/scratch/project/open-37-42/petrhyner/MCTS_evolution/`
unless absolute. See `ASSETS.md` in this folder for the full inventory,
sizes, shas, and re-copy commands. Highlights:

| asset | where | why it matters |
|---|---|---|
| **Exact Rust solver** | `supervised_valuenet/rust_datagen/` (binary: `target/release/datagen`) | Optimal solutions + labels, n ≤ 64. Deterministic, iteration-budget semantics (`budget.solver_iters`), NO wall-clock timeout by design. The referee for everything gradeable. |
| **Size-free labeler net** | `self_play_robots/assets/labeler_prod_v1_s11.ckpt` (copy of `supervised_valuenet/nn_labeler/banked/prod_v1_s11.ckpt`, sha dd0e7211b1cbd39f) | LoopedValueNet, d_model=192, recurrence=12, **pe=none — no size-dependent parameters**. Trained on 8–16 boards only; labels near-optimally to 64×64 (argmin agreement 86–93%) and runs at any size. Your bootstrap value function AND the architectural template (§6.2). |
| **Backward planner pairs** | `assets/g16r4_backward_{policy,value}_b1s21.ckpt`, `assets/g24r4_backward_{policy,value}_exact.ckpt` (+ ~40 more pairs under `supervised_valuenet/scaling/runs/`) | The bootstrap policy/value for the subgoal action space, and the baselines to beat. NOTE: these carry an n²-sized learned positional table — usable only at their training size (§4.5). |
| **Lean board generator** | `supervised_valuenet/nn_labeler/leanboard.py` | Boards at any size in seconds (3.8 s at 96×96 vs 11.9 h for the eager path); layout-identical to eager boards (parity-proven); lazy distance oracle. `python -m nn_labeler.leanboard --n N --robots R --walls W --ids A-B --out DIR`. |
| **Certified-descent labeler** | `supervised_valuenet/nn_labeler/descent.py` | Greedy NN plan completion + strict physics certification → labels. The template for your self-play *realization + verification* step. |
| **Benchmark machinery** | `supervised_valuenet/eval/compare.py`, `eval/merge_compare_shards.py`, `eval/replay_validate.py` | The pinned-protocol arena: playable-moves scoring, 1200 expansions, k=5, chunked CPU eval, replay certification. Reuse it as the gating arena (§7). |
| **Pinned benchmarks** | `supervised_valuenet/scaling/data/<cfg>/bench.solved.jsonl` (+ `bench.unsolved.jsonl` frontier sets) | The fixed exams every supervised arm was graded on. Self-play planners MUST be graded on these same files for comparability. |
| **Exact corpora** | `supervised_valuenet/scaling/data/<cfg>/backward.jsonl` etc. | Supervised bootstrap data; also the source of exact optima per instance. |
| **Boards** | `supervised_valuenet/environments_g*/` (127 GB, also archived to `/mnt/proj1/open-37-42/petrhyner_archive/2026-08-16/boards/`) | Pinned board pools ids 0–1199 per config (+ lean 2000–3049 at g32r4). Bench boards are sacred — never train on ids 900–1049 (test) or the bench instance boards. |
| **The experiment log** | `supervised_valuenet/FINDINGS.md` (items 48–81 are the labeler track) | Where every number above comes from. Read §74, §79, §80, §81 minimum. |

**Checkpoint format note:** planner ckpts are PyTorch-Lightning; the nets
are defined in `supervised_valuenet/train/policy_tf.py` (PolicyTF) and
`train/looped_pc.py` (LoopedValueNet-with-pos); the labeler's size-free
variant is `supervised_valuenet/nn_labeler/model.py`. Loading requires the
repo's encode path (`train/encode.py`, env vars `RR_GRID`/`RR_ROBOTS`/
`RR_WALLS`/`RR_ENV_DIR` — see `scaling/configs.py::apply_env`).

## 4. Lessons you must not relearn (they were expensive)

These are the supervised campaign's conclusions, each with its FINDINGS
citation. They are constraints and free wins, not suggestions.

**4.1 Label/value fidelity gates downstream utility (FINDINGS 74).**
Planners trained on labels with ~91% best-candidate agreement match
exact-trained planners; ~89% keeps solve rate but loses optimality; ~82%
collapses. For self-play this transfers directly: **your value targets are
self-generated labels.** Monitor their argmin agreement against exact
optima (cheap ≤ 64) every iteration; a fidelity slide below ~90% predicts
a utility slide *before* the bench shows it. This is your early-warning
gauge — the supervised track gifted you a calibrated instrument.

**4.2 The full NN pipeline is proven end-to-end (FINDINGS 79).** NN-made
boards + instances + labels trained a planner indistinguishable from the
exact pipeline (83.4% vs 84.0% solve at g32r4). You are not gambling on
whether NN-generated data can teach — that's settled. The open question is
only whether *search-improved* self-play data teaches better.

**4.3 Distilling from damaged data reproduces the damage (FINDINGS 68).**
The B2 experiment: a net trained on iteration-capped corpora reproduced
the cap's pathology in its own outputs. Self-play corollary: **whatever
your search does poorly, your data does poorly, and the next net learns
it.** Data-quality control (certified replay, fidelity gates, diversity
checks) is not hygiene, it is the difference between improvement and a
feedback loop of degradation.

**4.4 Value-net training is bistable (FINDINGS 44/50/61).** Cold value-net
training falls into a constant-output mode — sometimes mid-run (the
production labeler collapsed at epoch 2; its banked ckpt is the epoch-0
snapshot). Mitigations that exist and work: `val_group_spread` monitoring,
the `CollapseStop` callback (3 epochs spread < 0.05 → stop),
`ModelCheckpoint(min val_regret)` so a collapse never destroys the best
epoch, warm-starting from the previous iteration's net (cold retrains are
seed-unstable), and low-lr rescue (1e-4) for ≥6-robot configs. **In a
self-play loop where training runs unattended every iteration, wire ALL of
these in from day one.** Also: 8-robot cold value retrains are a
documented bistable-collapse regime (FINDINGS 44/56) — treat g24r8 as the
hard mode it is.

**4.5 The supervised planner nets are NOT size-free (FINDINGS 81).** Both
`policy_tf.py` and `looped_pc.py` carry `self.pos = Parameter(randn(n²+1,
d))` — a learned per-cell table (1.23M params at 80×80). Consequences:
(a) bootstrap checkpoints only work at their training size; (b)
cross-size transfer/curriculum requires either the labeler's pe=none
recipe (proven to generalize 4× in size) or per-size nets. §6.2 makes
this the project's first architectural decision. Memory scaling is
attention-O(T²) with T = n²+1 tokens; at batch 8×32 groups and 32×32 that
was already a 13 GB/step OOM (FINDINGS 63) — the working config at 32×32
is batch 2 / max-per-group 8. Also inherited quirks: value-net bins
default to 50 (`--num-classes`; 32×32 needed 96 — labels silently clamp
otherwise); `--max-per-group 1` silently disables the ranking loss;
`scaling.train --splits` overrides board-range splits for corpora on
non-standard board ids (FINDINGS 79c).

**4.6 Certification is non-negotiable (whole track).** Nothing a network
reports counts until replayed move-by-move against `simulate.py` physics.
The eval driver (`eval/compare.py --dump-moves` + `replay_validate.py`)
already enforces this; the descent labeler drops uncertifiable candidates
instead of guessing. Keep this property in the MCTS: **a backup value from
an unrealized plan is a hypothesis; only realized, replayed plans update
the record.** This is also your defense against the self-play analogue of
reward hacking (the net convincing itself illegal shortcuts work).

**4.7 Seed noise is large enough to fool you (FINDINGS 67→71).** A
3-point solve-rate "win" evaporated on a seed replicate; measured same-arm
spread is ~3.4 solve / ~6.6 optimality points at g24r4. **Promotion
decisions inside the loop must clear the seed-noise bar** — an
iteration-(k+1) net that beats iteration k by 1 point has shown nothing.
Either use large benches, or paired-instance tests (McNemar over the
per-instance solved/unsolved vector — machinery exists in the concurrent
track's by-reference A/B, FINDINGS 78), or both.

**4.8 Costs you can plan with (measured).** Lean board: 0.24 s at 32×32,
3.8 s at 96. Descent labeling: ~0.36 s/inst depth-0 at 32×32, 4.3 s full
descent; 16.5 s at 80. Planner pair train+bench: 0.54 nh at g24r4, 1.6 nh
at g32r4 (batch sizes above). Bench: 175 graded + 275 frontier instances,
~8-wide CPU chunks, 1.5–20 h wall depending on net quality (bad nets burn
the full expansion budget — bench cost is itself a quality signal). GPU
eval port: `eval/end2end.py` has NO `torch.no_grad()` (fine on CPU where
autograd memory was never hit at ≤32; add it before any GPU eval — see
FINDINGS 81's feasibility brief, `supervised_valuenet/analysis/
followup_2026-08-17/01_8096_planner_feasibility.md`). Budget context:
~755 of 1000 node-hours remain (check `it4ifree`); the whole supervised
campaign cost ~27 nh. A self-play loop is more expensive by nature —
budget estimates per milestone in §8, and state projected nh in chat
before big submissions (house rule).

## 5. The loop (proposed shape)

AlphaZero mapping, adapted to what we have:

| AlphaZero piece | here |
|---|---|
| Game rules | `simulate.py` physics + lean boards (self-generated positions at will) |
| Network f(s) → (p, v) | policy over the chosen action space + value = cost-to-go estimate (§6.2) |
| MCTS with PUCT | over subgoal decisions (recommended) or primitive moves (§6.1); "win" is replaced by realized move count → minimize cost, so back up **negative realized cost** (or a cost-bucket distributional value like the existing HL-Gauss value nets) |
| Self-play games | solve generated instances with search; **certify every solution by replay**; realized primitive count is the episode's ground truth |
| Training targets | policy ← visit distribution at each decision; value ← realized cost-to-go from each visited state (certified labels only — exactly the descent labeler's contract, now with search instead of greed) |
| Replay buffer | rolling window over recent iterations' certified episodes; keep board/instance provenance for leakage control |
| Evaluator / gating | pinned bench arena (§7) + fidelity gauge (§4.1) + seed-noise-aware promotion (§4.7) |

**Iteration k:** generate fresh instances on fresh lean boards → search
with net k under a fixed expansion budget → certify → append to buffer →
train net k+1 (warm-start from k, CollapseStop armed) → gate: bench vs
net k AND vs the frozen supervised baselines → promote or diagnose.

**The bootstrap (iteration 0) is free:** initialize from the supervised
backward planner pair (copied in `assets/`) — or, if §6.2 decides on the
size-free rebuild, train the new architecture on the existing exact
corpora first and verify it reproduces supervised-level bench numbers
BEFORE any self-play. Never debug architecture and loop dynamics at the
same time.

**The improvement hypothesis, stated honestly:** greedy descent under the
labeler already labels at 86–93% argmin agreement. Search should beat
greed — MCTS at inference found solutions greedy descent misses (that gap
is exactly the 100−86…93% plus the unsolved tail). If search-labeled data
trains a net that searches better, the loop climbs. If it merely matches
the supervised ceiling, that is a publishable negative result about
self-play in deterministic single-agent planning — cf. §4.3: the loop can
also go DOWN, and the fidelity gauge is what tells you which is happening
before you've spent 100 nh.

## 6. Open design decisions (with recommendations)

**6.1 Action space: subgoal MCTS (recommended) vs primitive-move MCTS.**
Recommendation: **subgoal decisions**, because (a) horizon 2–6 vs 8–30
makes search tractable per expansion budget; (b) the strongest baseline
lives there, so bootstrap is warm; (c) realization+certification machinery
exists (`descent.py`, `eval/realize.py`); (d) the supervised track showed
the backward planner beats forward by growing margins with scale
(+47.8 pts solve at 32×32). Primitive-move MCTS is the purer AlphaZero
and removes the candidate-generator ceiling — keep it as a comparison arm
at g16r4 where it's cheap, not as the main line. NOTE the candidate
generator bounds the reachable policy: if optimal play requires a subgoal
the generator never proposes, no amount of search finds it. Measure the
generator's ceiling early (exact-solution subgoals ⊆ candidate sets? —
answerable offline against exact corpora ≤ 64).

**6.2 Network: rebuild size-free (recommended) vs keep per-size nets.**
Recommendation: **rebuild policy+value on the labeler's pe=none recipe**
(`nn_labeler/model.py` is the template; add a policy head over candidate
encodings). Reasons: one net across the curriculum, cross-size transfer
(train small, deploy large — the project's most novel angle, §9), no n²
table to relearn per size, and the labeler proves the recipe holds 4× in
size. Cost: you must first show the rebuilt nets match the per-size
supervised planners on the bench (that's milestone M1, and it is NOT
optional). Fallback if the policy head underperforms: per-size nets at
g16r4/g24r4 only, decide again after M3.

**6.3 Value target form.** The existing nets are distributional
(HL-Gauss over cost buckets) and that survived the whole campaign — keep
it. Bucket count must cover realized costs at the largest curriculum size
(§4.5 clamp bug).

**6.4 Exploration.** Deterministic game, single agent → self-play
"diversity" comes from instance generation (fresh boards each iteration —
lean boards are ~free) plus Dirichlet noise at the root and temperature
on visit counts early in training. Instance difficulty curriculum: reuse
the descent labeler's depth-0 vs deeper split; FINDINGS 81's brief showed
80×80 corpora are 81% depth-0 — flat difficulty is a real failure mode
worth actively steering against (sample instances the current net finds
hard; the frontier bench files are a ready-made hard-instance model).

**6.5 Search budget accounting.** The metric is moves, but comparisons are
only fair under equal search budget. Adopt the bench's convention (1200
expansions, k=5) for gating. Inside self-play generation you may spend
more (data quality scales with search depth — but see §4.3: don't cap it
in a way that biases what the data contains).

## 7. Evaluation protocol (the arena)

Non-negotiables, inherited: **playable-moves scoring** only; the pinned
per-config bench files; replay certification of every solve; report solve
rate, mean realized moves (headline), % optimal + mean regret where exact
ground truth exists (≤ 64); frontier sets for the hard tail. Baselines to
beat, frozen forever (from `supervised_valuenet/scaling/results/<cfg>/
comparison*.json` — regenerate the table from files, don't copy numbers):

- `backward supervised (exact-taught)` — the real opponent.
- `forward supervised` — slower but near-optimal when it solves
  (e.g. g32r4: 76.0% solve / 88.7% optimal / 1379 s per puzzle vs
  backward's 84.0% / 51.7% / 10 s). Beating backward on moves while
  matching its solve rate ≈ closing toward forward's quality at
  backward's speed — that's the headline chart.
- `exact optimum` — the ceiling (≤ 64), from the bench files' `d_star`.

Success statement to aim the whole project at: *"self-play planner X
solves ≥ backward-supervised solve rate with mean realized moves closer to
optimal than forward-supervised, at backward-supervised compute."* Partial
orderings short of that are still results; define per-milestone gates
below so progress is measurable.

## 8. Milestones (each with a hard gate; est. node-hours)

- **M0 — arena parity (≈1 nh).** Re-run the supervised g16r4 backward pair
  through the bench from this folder's harness; numbers must match the
  recorded comparison JSONs. Proves your wiring before anything novel.
- **M1 — size-free rebuild matches supervised (≈3–6 nh).** Train pe=none
  policy+value on existing exact corpora at g16r4 + g24r4; bench within
  seed noise of the per-size supervised pair. Gate: solve rate within
  3.5 pts, optimality within 6.6 pts (the measured seed bars, §4.7).
- **M2 — search beats greed at inference (≈2–4 nh).** MCTS with the M1
  nets (no retraining yet) vs greedy descent on the same instances/budget.
  Gate: strictly better mean moves at equal-or-better solve rate.
- **M3 — one self-play iteration helps (≈5–10 nh).** Generate, certify,
  retrain once, bench. Gate: net k+1 beats net k beyond seed noise on the
  paired-instance test, AND fidelity gauge (§4.1) has not slipped.
- **M4 — the loop climbs (≈15–30 nh).** 3–5 iterations at g16r4/g24r4.
  Gate: monotone-ish bench improvement; final net beats the supervised
  backward baseline beyond seed noise.
- **M5 — scale (≈10–25 nh).** Curriculum to g32r4 (and the g24r8 hard
  mode); if §6.2's size-free bet holds, evaluate transfer: net trained via
  self-play at ≤32 benched zero-shot at 40–64 against exact ground truth.
- **M6 — beyond the oracle (stretch).** Self-play at 80/96 where only
  certification exists; the ≤64 fidelity curve is the warranty argument
  (same structure as FINDINGS 72c).

Log every result in `supervised_valuenet/FINDINGS.md` (max+1 numbering —
two sessions share it; check before appending) or a FINDINGS.md in this
folder if the owner prefers separation — ask once, then be consistent.

## 9. What is scientifically new here (why this is worth compute)

1. **Self-play for single-agent deterministic planning with certified
   labels** — AlphaZero's loop minus the opponent, plus a physics
   certifier that makes label corruption structurally impossible (only
   *suboptimality* can creep in, and §4.1's gauge measures it against an
   exact oracle up to 64×64 — a luxury Go never had).
2. **Size-free self-play**: if §6.2 holds, train the loop small and
   evaluate zero-shot large against exact ground truth — connecting the
   labeler track's headline (fidelity flat to 64) to policy improvement.
3. **A calibrated dose-response instrument** (FINDINGS 74/80) for
   predicting when self-play data degradation will hurt, before it does.

## 10. House rules (inherited, binding)

Karolina: account `open-37-42` (lowercase), **qgpu partitions only**,
everything via Slurm from `/scratch`, smoke-test on `qgpu_exp` (1 h) first,
state projected node-hours in chat before big submissions, `scancel` by
job id only (never `pkill`), one job = one log with a grep-able DONE
marker, atomic writes (tmp + `mv`), idempotent/resumable jobs (walltime is
a hard kill), `--exclude` known-bad nodes, positional args not
`--export=ALL` (scheduler env-hold bug), `ml Python/3.11.5-GCCcore-13.2.0`
then `source /scratch/project/open-37-42/petrhyner/venv/bin/activate` in
every job and every shell. Results: append to FINDINGS first, name
sources (files + job ids), commit and push often — git is the backup that
survives the 90-day scratch purge; checkpoints/corpora go to
`/mnt/proj1/open-37-42/petrhyner_archive/`, not git. Reports are LOCAL
FILES ONLY — never published anywhere. Vocabulary separation is absolute
(base vs B2 datasets never mix; this project is base-vocabulary unless
the owner says otherwise). Never train on bench boards or test ranges
(ids 900–1049 per config).

## 11. Suggested first week (concrete)

1. Read: `supervised_valuenet/FINDINGS.md` §§48–81, `PRIMER.md`,
   `TWIN_WIRING.md`, `analysis/followup_2026-08-17/*` (esp. the
   feasibility brief), this file, `ASSETS.md`.
2. M0: wire the arena from this folder (thin wrappers; do NOT fork
   `eval/compare.py` — call it), reproduce the g16r4 supervised bench row.
3. Answer the candidate-generator-ceiling question offline (§6.1) against
   the g16r4/g24r4 exact corpora — it's free and it bounds the project.
4. Draft the pe=none policy head design; get the owner's sign-off on §6.1
   + §6.2 recommendations (one message, both decisions).
5. M1 training run at g16r4 (`qgpu_exp` smoke first).
6. Only then think about MCTS internals.

*Everything in this document that states a number cites FINDINGS or a
result file; when in doubt, the file wins. Good hunting.*
