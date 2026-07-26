# Two-sided tuning ledger

Closes `analysis/publishability.md` objection 0.8 ("tuning-effort asymmetry
unreported — forward got per-scale lr rescues; backward got warm-starts, bin
changes, batch tuning"). Written 2026-07-26 from the job recipes and result
JSONs, not from memory; every row names its source.

The point of the ledger is not to show the effort was equal — it was not — but
to state what each side received so a reader can discount accordingly. Where
the asymmetry favours the forward planner it is said so plainly.

## 1. Forward (move-level) planner

| config | what it received | source |
|---|---|---|
| 16×16 · 4r (base) | **Four separately trained systems evaluated, best selected**: `best.ckpt`, `candidate_scored.ckpt`, `move_planner_v2/runs_warm/iter5`, `runs_scratch_v5/iter15`. Solve rates 84.7 / **100.0** / 97.6 / 87.6%. The reported base cell is the best of the four. | `eval/results/comparison_forward.json`; selection rule in `eval/report_data.py` (`entry["graded"]["fwd"] = best`) |
| 16×16 · 6r | Stock recipe destabilised after one pass; **learning-rate rescue retrain** (validation top-1 0.71 → 0.878). n = 1 original, n = 1 rescue. | FINDINGS §6; `scaling/results/g16r6/comparison_forward_control.json` |
| 16×16 · 8r | Stock recipe **collapsed to near-random**; stability-controlled retrain. The collapsed run is loaded by the report only to assert it is never rendered as a comparable result (`fwd_withheld`). | `eval/report_data.py:104`; `scaling/results/g16r8/comparison_forward_control.json` |
| 24×24 · 4r | Stock recipe, stable. No rescue. | `eval/report_data.py` fwd_note |
| 24×24 · 8r | Stock recipe collapsed; stability-controlled retrain. | `eval/report_data.py` fwd_note |
| 32×32 · 4r | Stock recipe collapsed; stability-controlled retrain. | `eval/report_data.py` fwd_note |

## 2. Backward (subgoal) planner

Per-config knobs, all from `jobs/patterns/b2_retrain_one.slurm`:

| config | learning rate | value warm-start | batch caps | value bins | epochs (policy/value) |
|---|---|---|---|---|---|
| 16×16 · 4r | default | `checkpoints_backward/value_b1.ckpt` | default | default | 25 / 20 |
| 16×16 · 6r | **1e-4** | banked g16r6 value ckpt | default | default | 30 / 30 |
| 16×16 · 8r | **1e-4** | banked g16r8 value ckpt | default | default | 30 / 30 |
| 24×24 · 8r | **1e-4** | banked g24r8 value ckpt | value 4 / max-per-group 16; policy 8 | default | 30 / 30 |
| 32×32 · 4r | default | banked g32r4 value ckpt | value 2 / max-per-group 8; policy 16 | **96** (`--num-classes 96`) | 30 / 30 |

Plus two standing mitigations, both documented as responses to observed
instability rather than to a search over values:

* **Value nets are always warm-started** and pinned to `--torch-seed 11`,
  because cold value retrains are recorded as seed-unstable (the known
  signature is validation top-1 near random from epoch 1). Policy nets train
  cold and are stable.
* **Batch caps are memory-driven**, set from the 40 GB A100 tables, not tuned
  for accuracy.

## 3. The honest comparison

* **Neither side received a learning-rate sweep.** Forward's "properly trained"
  means *after a per-scale learning-rate rescue triggered by observed
  validation collapse* — one value, applied by pattern, n = 1 per rescue.
  Backward's 1e-4 at ≥6 robots is the same kind of move. Objection 2.2 (a
  3-point sweep) is unexecuted on both sides. Any sentence claiming either
  system is "properly trained" must carry this definition.
* **The asymmetry that favours FORWARD, and is rarely stated:** at base scale
  the forward cell is the best of four independently trained systems, while
  the backward cell is a single run. A best-of-four selection is worth real
  points, and the base row is exactly where forward's advantage is quoted
  (100% vs 95.6%). The scaling rungs do not have this asymmetry — there the
  forward control is n = 1 (after rescue), as is backward.
* **The asymmetry that favours BACKWARD:** its value nets are warm-started
  from a banked predecessor at every configuration, which is a form of
  transferred training the forward side never receives. It is a stability
  mitigation, not a performance trick — but it is unmatched, and it should be
  named.
* **Both sides are single-seed everywhere** (objection 2.1). Backward's value
  seed is explicitly pinned; forward's is not recorded per run. Neither system
  has a min/median/max across seeds, so no reported difference smaller than
  the seed spread — which is unmeasured — can be defended. This is the
  strongest remaining statistical caveat after the significance testing of
  FINDINGS §22.
* **The B2 retraining adds one more backward-side intervention**: the label
  set is generated under a 5,000-iteration rollout cap that biases it toward
  instances whose rollout succeeds quickly (FINDINGS §30). That is a data
  intervention with no forward counterpart, and it belongs in this ledger the
  moment the retrained rows are reported.
