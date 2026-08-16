# NN-labeler track — handoff, 2026-08-05

You are taking over the "replace the exact solver as a data labeler with a
neural network that generalizes across board sizes" track. Working directory
`/scratch/project/open-37-42/petrhyner/MCTS_evolution`, branch
`b1-language-extension-scaling-study`, everything committed and pushed.
Machine-level rules auto-load from `~/.claude/CLAUDE.md` (qgpu only, account
`open-37-42`, /scratch, scancel by id). The parent study's rules
(`prompt.md`, `PRIMER.md`, `FINDINGS.md`, house rules) all still apply. This
document covers only this track.

**Read in this order:** `nn_labeler/PLANS.md` (the three candidate paths and
why C+B was chosen) → `FINDINGS.md` items **48, 50, 52, 53, 54, 55, 57, 58, 59**
(this track's log; 49/51/56 belong to a concurrent Track-1 session — see the
numbering note in FINDINGS) → `nn_labeler/report/process.html` (technical) and
`nn_labeler/report/scaling_story.html` (plain-English, figures) → this file.

---

## 1. LIVE STATE — what is running and queued

**Updated 2026-08-05 (evening): the campaign below was re-planned with the
owner** — headline = downstream equivalence; see FINDINGS **61 + 62** for the
day's results (v2 net COMPLETED healthy and beats v1 everywhere; lean-board
path landed, record-identical, unlocking 40–96). prod2 (4616677) is done;
4610118 (eager lprep) was CANCELLED and replaced by the lean gates job. The
single-GPU dependency chain, in order:

**CAMPAIGN COMPLETE (2026-08-17).** Read FINDINGS **61–74 and 79** for the
full arc (75–78 are the concurrent Track-1 session's). The controlled
suite closed as a three-cell dose-response (§74: fidelity gates downstream
utility — 91% argmin → full equivalence on a 2×2, 89% → solve-rate
equivalence, 82% → collapse); B2 payoff was NEGATIVE (§68, distillation of
capped data); the verifiable ladder runs 17–64 flat with the calibration
endpoint at 90.1% (§70); UNVERIFIABLE labels exist at 80/96 (§72c). The
deployment run LANDED (§79, job 4670119): fresh NN-everything pipeline at
g32r4 → 83.4 solve / 47.9 opt / 3.10 regret, between the exact and twin
arms — equivalence holds with no exact solver anywhere in data
production. (`scaling.train` gained `--splits` for corpora on
non-standard board ids en route.) NO jobs queued. Open next steps: 80/96
downstream validation (planner-side architecture above 32 UNPROBED —
feasibility first), B2 rescue via a small uncapped exact seed corpus
(owner decision), more seeds for g32r4/g24r8 twin cells. Total campaign
≈27 nh.
The table below is the historical lane plan it ran under:

| lane | jobs | what it does |
|---|---|---|
| A (headline) | 4618940 `gtwin` → 4618960/61 `twinrt` ×2 | gates v2 vs v1 (capstone protocol), banks winner (`jobs/pick_winner.py`), builds the THREE TWIN CORPORA; then retrains backward-policy/value on them (hyperparams verbatim, `TWIN_WIRING.md`) and benches with the exact rows' protocol → `comparison_nntwin.json` (~27 h idempotent, two submissions) |
| B (payoff) | 4620064 → 4620065 `b2` ×2 | B2 net on the ≤24 B2 corpora, UNCAPPED descent labeling at g16r6 + g32r4 (zero-shot), by-reference share + gate |
| C (curve) | 4620484 `intgates` → 4620485–88 `coarse-g{40,48,56,64}` | per-integer rungs 17–31 as GATES ONLY on lean boards, then the coarse ladder to the verifiable endpoint; **pinned to v1 via `NNLAB_CKPT`** so the curve is one net end-to-end (its 24/32 anchors are v1) |

After these: seed replicate of the twin arm (`TWIN_TAG=-seed21`), the 32×32
deployment run (fresh NN-everything corpus → train → bench), 80/96
unverifiable rungs (configs exist, marked UNVERIFIABLE), optional exact-rebench
control (`TWIN_WIRING.md` OPEN 8). Results page:
`nn_labeler/report/suite.html` — regenerate with
`python nn_labeler/report/gen_suite.py` after any landing; cells auto-fill.

Monitors from the previous session are dead; arm your own
(`sacct -j <id> --format=State --noheader -X` in a background until-loop).

---

## 2. What this track established (the short version)

1. **The nets were one tensor away from size-invariance.** The backward value
   net is a weight-tied looped transformer over per-cell tokens with
   slide-graph-masked attention; the only grid-dependent weight was a learned
   positional table `[G²+1, 192]`. `nn_labeler/model.py::SizeFreeValueNet`
   removes it. **The best replacement is nothing at all** (`--pe none`):
   sinusoidal PE is not merely worse, it is *untrainable* under this recipe
   (both arms collapsed even with the curriculum). §53.
2. **No extrapolation cliff.** A net trained only on 8–10 boards labels
   32×32 (3.2×) with mean label error 0.71 and 81% top-1; the production net
   (≤16×16) reaches 0.425 / 88.3% at 24×24. §54, §57.
3. **Certified descent works as a labeler.** Candidates are enumerated exactly
   as the exact labeler does, priced by greedy NN completion instead of an A*
   solve, and every completed plan is realized and replayed under physics —
   only plans that actually play out become labels. At 32×32: **87.7% of
   labels are exactly the exact solver's optimum, argmin agreement 91.5%,
   zero impossible labels**, and ranking agreement is *better* on big boards
   than small ones. §55, §59.
4. **But it saves no compute in base vocabulary.** Measured head-to-head on
   identical boards: the Rust exact labeler does 200 instances in 6 s (24×24)
   / 15 s (32×32) on 4 CPU cores; descent costs 5.1 s / 18.6 s **per
   instance** on an A100 — 170× / 248×; even replaying known-labelable
   instances it is 9× slower at 32×32. §58, §59.
5. **Therefore the claim moved.** The method's value is not speed on easy
   settings. It is (a) the **extended (B2) vocabulary**, whose exact campaign
   projected **441 node-hours** and whose iteration-cap rescue destroyed the
   labels it existed to make (by-reference share 13.5% → 4.8%; the retrain
   lost 24.6 points), and (b) **beyond-oracle regimes** where no exact labels
   exist at any price. **Descent has no iteration budget**, so the wandering
   that forced the cap structurally cannot occur — that is the real claim, and
   job 4618888 tests it.

---

## 3. Assets — what exists and where

**Code** (all new, under `supervised_valuenet/nn_labeler/`):

| file | what |
|---|---|
| `encode.py` | size-parametric featurization; **verified `np.array_equal`-identical** to the frozen pipeline on real data. Pure functions, no import-time env reads. |
| `dataset.py` | multi-corpus loader with `_config`/`_n`/`_env_dir` stamping (fixes `env_id` aliasing across configs), ported grouping/subsampling, `SizeBucketBatchSampler`. |
| `model.py` | `SizeFreeValueNet` + `collate_groups` (dependency-injected featurizers). Metrics include **`val_group_spread`** — the collapse detector. |
| `train.py` | multi-corpus trainer; two-phase curriculum; `CollapseStop`; resume; post-fit validation guard. |
| `audit.py` | value-quality audit (regret / top-1 / MAE / bias / spread) of any checkpoint on any corpus+split. |
| `descent.py` | **the labeler**: certified descent, physics certification, RNG + instance-replay modes, drop accounting in a sidecar manifest. |
| `audit_descent.py` | label-vs-exact comparison (per-decision, per-candidate gaps, argmin agreement); passes an identity self-audit. |
| `tests/` | `test_encode_dataset.py` (incl. subprocess parity vs frozen code), `test_model_smoke.py`. **Run both after any edit.** |
| `jobs/*.slurm` | `debug_battery{,2}`, `descent_gate`, `prod_train{,_v2}`, `capstone_g32`, `ladder_rung`, `ladder_prep`, `b2_payoff`. |

**Data** (gitignored, in-tree): exact base-vocab corpora at **grids 8–15 (r4)**,
~100k records each, `scaling/data/g<N>r4/backward.rust.jsonl`; 1200 boards each
in `environments_g<N>r4/`. Registry entries for all of these plus ladder rungs
g17–g23, g25–g31 are in `scaling/configs.py` (additive `_std` lines).

**Banked model:** `nn_labeler/banked/prod_v1_s11.ckpt` (+ `.json` provenance,
sha `dd0e7211b1cbd39f`) — the labeler of record. pe=none, 96 bins, trained on
grids 8–16 / robots 4,6,8. Audits: 0.178@8, 0.254@12, 0.432@16r6, 0.425@24.

**Results:** `nn_labeler/results/` — battery audits, gate JSONs, capstone
(`capgen_*`, `capgate_*`). **Reports:** `nn_labeler/report/` — `process.html`
(technical), `scaling_story.html` (plain-English + 7 figures),
`fragments/` (figure sources). Both are **local-only, never publish** (standing
owner order).

---

## 4. THE NEXT GOAL — 64×64, and an honest feasibility read

The owner's next goal: keep going to **64×64**, with the model itself
generating data upward from 16×16.

### 4a. Terminology — it is not self-play, and the difference matters

What exists is **zero-shot amortized labeling with physics certification**: a
frozen net prices candidates by greedy completion, physics certifies, exact
spot-sets gate each rung. There is no opponent and no game outcome, so it is
not self-play in the RL sense (the repo's actual self-play lives in
`subgoal_selfplay/`). The owner's *intent* — bootstrap upward without the exact
solver — is exactly what this does.

The live design choice is **frozen vs. bootstrapped**:
- **Frozen (current, chosen):** train once ≤16×16, apply zero-shot at every
  larger size. Chosen because §34/§47/§55 show retraining historically *hurt*,
  and §54/§59 show frozen extrapolation holds to 3.2× with ranking improving.
- **Bootstrapped ("self-play"-flavoured):** relabel at size N, retrain, move to
  N+1. This is the risky variant — drift compounds and each rung inherits the
  last rung's biases. **Do not adopt it by default.** If tried, gate every rung
  against exact ground truth and keep the ≤16 exact data in every mix.

### 4b. Can we reach 64×64? Yes — but one engineering blocker stands in the way

**Settled in favour (measured 2026-08-05):**
- **The architecture runs at 64×64.** Verified: 4,097 tokens, forward pass
  clean, one attention mask 67.1 MB, peak RSS 1.3 GB at batch 1 on CPU. On an
  A100 use batch ≤ 4–8 (attention scores are `[B, heads, 4097, 4097]` ≈ 268 MB
  per head-family per record; three families).
- **64 is exactly the last verifiable size.** The Rust engine's hard envelope
  is **n ≤ 64, R ≤ 10** (`rust_datagen/src/move_oracle.rs::assert_envelope`,
  asserted in release too), and its 64×64 cell is already exercised in the
  cross-engine smoke matrix. So ground truth is obtainable at 64 and *nowhere
  above it*. **This makes 64 the calibration endpoint**: proving NN labels
  match exact labels at 64 is what licenses trusting them beyond 64, where
  nothing can check them. That is the scientific case for the goal.
- Extrapolation demand is 4× (16→64) vs the 3.2× already demonstrated with
  *improving* ranking agreement — plausible, unverified.

**The blocker — board infrastructure, not the network:**
`nn/gen_grids.make_board` computes all-pairs Dijkstra, O(G⁴ log G), and stores
two `(G²)²` distance dicts per board. Measured cold `GridEnv.from_env`: 8.2 s
(16) → 60 s (24) → 222 s (32); pickles 1.6 → 8.8 → 28.6 MB. Extrapolating to
64: **tens of minutes and several hundred MB per board** — a 1200-board config
is impossible (BENCH.md's own estimate for 64×64 sidecars is ~230 GB) and even
100 boards is days. `descent.py` inherits this because it uses
`GridEnv.from_env` and `compute_exact_shortest_path_length`.

**Therefore task #1 for 64×64 is a lean-board / lazy-distance path**, not more
compute. Three viable routes, cheapest first:
1. **Lean boards + memoized distances.** `encode.adjacency` needs only
   `grid_graph` (cheap, O(G²) slides) — *not* the all-pairs tables. Generate
   boards without `independent_paths`/`all_paths` (the pattern already exists:
   `rust_datagen/pyref/dump_decisions.py::_resolve_boards`, which is how the
   repo's existing 64×64 smoke cell works), and replace
   `compute_exact_shortest_path_length` with a memoized bounded BFS/Dijkstra
   computed on demand per (start, end, support). Most of the ladder's queries
   repeat, so a per-board LRU should make this cheap.
2. **Rust-side distances.** Have the Rust engine emit the per-board distance
   information the descent needs (it compiles a 64×64 board in 2.81 s and its
   `SubgoalEnv` is lazy with memoization). Costs a bridge protocol addition.
3. **Fewer, bigger-value boards.** Even 30–50 boards at 64×64 is enough for a
   *calibration* claim (the capstone used 20 for generation and 600 instances
   for the gate). This alone may make route 1 unnecessary at first — but the
   per-board cost still has to be paid, so measure one board before committing.

**Also true, and worth stating plainly:** at 64×64 in *base* vocabulary the
Rust exact labeler probably still wins on cost (it did at 32×32 by 9–250×). So
64×64 base is a **calibration/validation** exercise, not a compute saving. The
compute saving lives in B2 and beyond-oracle regimes at *any* size.

### 4c. Recommended plan to 64×64

| phase | work | est. |
|---|---|---|
| P0 | Let 4618888 (B2 payoff) land and read it. **It, not the ladder, decides whether this method is worth scaling at all.** | queued |
| P1 | Measure one 64×64 board end-to-end: `make_board` time/size, `GridEnv.from_env` time, one descent instance. **Do not size anything from estimates** (the §30 rule: an early projection was wrong by 12–42×). | ~1 h login CPU |
| P2 | Build the lean-board/lazy-distance path (route 1 above); prove it record-identical to the current path at 16×16 (an A/B like the encode parity test). | code |
| P3 | Ladder in *coarse* steps — 40, 48, 56, 64 — not every integer. The 8→32 curve is already smooth and dense; intermediate rungs add little. Gate each against a Rust-exact spot set (still possible ≤64). | ~2–6 nh |
| P4 | The 64×64 calibration statement, then (if it holds) the first *unverifiable* labels above 64 — clearly marked as such, since the Rust envelope forbids checking them. | — |

Rungs 17–31 (job 4610118 and `ladder_rung.slurm`) are now **low value** —
the capstone already anchors 24 and 32, and their labels are cheaper to
obtain exactly. Keep or drop at the owner's preference; they were the
originally stated goal, so they are queued, but P1–P3 above matter more.

---

## 5. Pitfalls earned the hard way (each cost real time)

1. **Value nets collapse into a constant-value mode, cold *and* mid-training.**
   Signature: `val_group_spread` → 0.00, top-1 → chance, and — the giveaway —
   *byte-identical* best scores across different architectures. Battery 1 lost
   7/8 runs to it; the production run lost 24 epochs to it at epoch 2. Fixes
   in place: two-phase curriculum (`--warmup`) and `CollapseStop`. **Always
   check `val_group_spread` before trusting any new run.**
2. **A batch sampler that yields fewer batches than its `len()` silently
   disables Lightning validation** — no metrics, no checkpoints, no error.
   This Lightning neither re-reads dataloader length per epoch nor calls
   `set_epoch` on custom batch samplers; that is why the curriculum is two
   static fits rather than a per-epoch ramp. §52.
3. **A resume whose checkpoint is already at `max_epochs` runs no epoch and
   logs no metrics** — "finished", not "failed". `train.py` handles this now;
   don't re-add a naive guard.
4. **Never size a job from its opening boards** (§30): a projection from 8
   graphs was wrong by 12× at base and 42× at 24×24/8.
5. **Environment**: `ml Python/3.11.5-GCCcore-13.2.0` **before**
   `source /scratch/project/open-37-42/petrhyner/venv/bin/activate`, in every
   shell including background ones — otherwise `libbz2`/`libffi` import errors.
   Fresh Bash tool calls do NOT inherit it.
6. **Git push** authenticates through the VS Code askpass bridge and dies when
   the owner's editor disconnects; retry with an owned socket
   (`VSCODE_GIT_IPC_HANDLE=/tmp/vscode-git-<id>.sock git push`, see the agent
   memory dir). Queue commits locally and say so rather than looping.
7. **FINDINGS numbering collides** — a second Claude session appends to the
   same file. Take `max+1` from
   `grep -E '^[0-9]+\. ' FINDINGS.md`, and expect to renumber.
8. **The daily cooling reservation now runs 10:00–20:00** (longer than the
   MOTD's stated 18:00), removing 48+ GPU nodes; long jobs wait days on
   priority. Chunk everything, keep walltimes ≤16 h, make every job idempotent.
9. `descent.py` writes its output file **once, at the end** — bound long legs
   with `--limit-instances` so a walltime kill cannot destroy hours of work.

---

## 6. Open questions for the owner

1. **Rungs 17–31**: complete the originally requested per-integer ladder, or
   skip to the coarse 40/48/56/64 plan? (My recommendation: skip; the
   fine-grained curve is already established and those labels are cheaper
   exact.)
2. **Beyond 64**: the Rust envelope forbids verification above 64. Is
   unverifiable labeled data wanted at all, and under what marking?
3. **B2 ladder**: if 4618888 succeeds, the natural follow-up is B2 labeling at
   sizes the exact campaign could never reach — likely the highest-value work
   in the whole track, and a bigger prize than 64×64 base.
