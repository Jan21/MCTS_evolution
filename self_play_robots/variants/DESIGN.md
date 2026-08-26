# Variants lab — design, protocol, and survey

Owner directive (2026-08-20): *"the architecture/design of the self-play should
be something where we can try many different approaches and compare them ...
thoroughly document the approaches ... think outside the box ... the neural
network should not require any human in the loop; goal is it should generalize
to unseen examples where it finds better solutions faster / amount of moves is
lower than with backward/forward baselines."* Follow-ups: bootstrapping FROM the
supervised stack is allowed (the LOOP must run label-free; a cold-start control
measures the prior's worth), and the budget goes to **structurally different
mechanisms, not knob tweaking** (at most one knob arm as sanity control).

## 1. What a variant is

One module `variants/vNN_<slug>.py` exporting a `Variant` (see
`variants/__init__.py`): id, axis, falsifiable hypothesis, mechanism, expected
failure mode, and its delta vs the control expressed as (a) runner knobs and/or
(b) in-process hooks (`apply_selfplay/train/bench`) that monkeypatch the spr
stack inside the phase's own process (generation workers included — `spr.
selfplay` passes its worker state to hooks because under `python -m` the live
module is `__main__`). Variants may ship whole modified search functions
(v06 carries its own `mcts_gumbel`); `supervised_valuenet/` is never modified.

## 2. The matched protocol (what "comparable" means)

Fixed for every arm:
- **Warm start**: the frozen `mix_b2mix_iter2` pair (`results/selfplay/
  mix_b2mix_iter2/nets.txt`) — the best all-around nets when the lab opened.
- **Generation**: g24r4, B2 vocabulary, 30 fresh lean boards × 8 instances,
  MCTS 300 expansions / stop-after 80 / root noise 0.25 / root-all / sibling
  completion; per-variant board ids `40000 + 300·index` (disjoint across arms;
  never the pinned 0–1199, never the exam's 20000+).
- **Retrain**: policy + value warm from the seed pair, 6 epochs, lr 1e-4,
  `--byref`, on THIS iteration's records only (window 1 — one-shot arms).
- **Benches** (arena A\*, B2 anytime, 1200 expansions, k=5, replay-certified):
  pinned graded (232), pinned frontier (218), **unseen exam** (200).
- **Gates**: `spr.gate compare` (McNemar on solved vectors, sign test on
  both-solved moves) vs `v00_control`'s bench of the same exam.
- **Seeds**: default seed 7; any arm claimed as a WIN is re-run at a second
  seed before the claim enters FINDINGS.

A variant changes exactly the piece its hypothesis names. Budget matching:
generation budget is matched in EXPANSIONS (the arena's unit). v12's probe
spends ≤15% extra value/policy calls on instance screening — reported, and its
per-instance search budget is unchanged.

## 3. The unseen exam (the generalization instrument)

`variants/exam.py` → `results/variants/exam/g24r4_unseen.jsonl` + boards.
50 fresh lean boards (ids 20000–20049, seed 777) × 4 uniform random instances,
pinned at first generation (the file is never regenerated). No exact labels
exist and none are computed — scoring is frontier-style: solve rate, paired
moves, expansions. Baselines benched on the SAME instances
(`jobs/variant_baselines.slurm`): the frozen seed nets (what every arm warm
starts from) and the supervised per-size backward pair under its own recorded
protocol (base vocab + prefix-check). The forward MoveNet baseline needs a
lean-board loader in `spr/fwd` (wave 2). **Success criterion** (owner): a
variant wins when, on unseen boards at equal search budget, it solves ≥ the
backward-supervised baseline AND realizes fewer mean moves than the current
best loop nets; stretch = moves competitive with the forward planner (requires
the action-space axis).

## 4. Literature survey (what the wave-1 portfolio is built on)

- **Gumbel AlphaZero/MuZero** (Danihelka et al., ICLR 2022: [Policy improvement
  by planning with Gumbel](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf);
  [Danihelka thesis](https://discovery.ucl.ac.uk/id/eprint/10167022/2/ivo_danihelka_thesis.pdf)):
  root actions sampled without replacement by Gumbel-top-k on `g + logits`,
  budget spent by **sequential halving**, action chosen by `g + logits + σ(q)`;
  policy targets from **completed Q-values** (backed-up Q for visited children,
  the value net for unvisited) instead of visit counts — guaranteed policy
  improvement at ANY simulation budget. Recent transfers of the same recipe to
  LLM tree search ([arXiv 2603.21162](https://arxiv.org/pdf/2603.21162)) and
  multi-agent variants ([Multiagent Gumbel MuZero](https://scispace.com/pdf/multiagent-gumbel-muzero-efficient-planning-in-combinatorial-42qjy1a1jj.pdf))
  report the biggest gains exactly at small budgets (ours: 300 expansions).
  → **v06_gumbel_root**. Note: our control's policy target (softmax over
  certified candidate costs) is already closer to completed-Q than to
  AlphaZero visit counts — v01 tests the visit-count alternative, so the
  target axis is covered from both sides.
- **EfficientZero / value-target engineering** ([EfficientZero V2](https://arxiv.org/pdf/2403.00564)):
  off-policy value targets mixing observed returns with bootstrapped estimates
  dominate pure-outcome targets in low-data regimes → v02 (parked as
  incremental) and, structurally, **v09_strict_value** — our unique twist:
  self-play certification exposes the REALIZED STRICT MOVES of every plan,
  i.e. the benchmark metric itself, which the supervised exact corpora never
  contained. Nobody in this project has trained on the metric; v09 does.
- **Self-competition for single-agent planning** ([Policy-Based Self-Competition,
  arXiv 2306.04403](https://arxiv.org/html/2306.04403)): single-agent AlphaZero
  loops stall when the value net saturates; competing against past selves
  restores a gradient. Our analogue: the loop's plateau (FINDINGS §19d) and
  the curriculum answer — **v12_frontier_curriculum** mines the CURRENT net's
  failure frontier (probe-and-reject) rather than uniform boards; PROBLEM.md
  §6.4 asked for exactly this.
- **Hindsight relabeling in planning** ([lifted/propositional HER for planning,
  arXiv 2605.25720](https://arxiv.org/html/2605.25720v1); [GBER,
  arXiv 2412.15525](https://arxiv.org/pdf/2412.15525)): failed episodes
  relabeled toward achieved sub-states. In our stack, failed certifications
  already partially recycle through B2 park repairs; full hindsight (relabel
  the reached configuration as the goal) is wave-2 (needs goal-conditioned
  featurization — the target cell is an input channel, so relabeling is
  representable). Parked pending wave-1 results.
- **Subtree reuse across moves** (MiniZero, [arXiv 2310.11305](https://arxiv.org/pdf/2310.11305)):
  inapplicable as-is — our generation runs ONE tree per instance (decisions
  are extracted from a single root), so there is no across-move reuse to win.
- **Action-space surgery**: FINDINGS §3 proves the moves headline is
  unreachable in ANY pure-subgoal language (ceiling +0.9/+1.2 regret vs the
  forward planner's 0.04–0.07), so the breakthrough axis is the action space:
  **v07_hybrid_actions** (subgoal macros + primitive-move escapes, design
  stub with the State→PartialPlan re-projection blocker documented) and a
  learned PROPOSAL net (breaking the `propose()` bottleneck PROBLEM.md §6.1
  warns bounds the reachable policy) are wave-2 candidates; v07's sketch
  names the forward MoveNet Guide as the natural slide-head.

## 5. Wave 1 (submitted 2026-08-20)

| arm | axis | delta | why it could be a breakthrough |
|---|---|---|---|
| v00_control | control | none | the bar |
| v01_visit_policy | targets | π ∝ N over candidates | tests whether the search's own preference beats certified-cost softmax |
| v04_deep_emit | data (knob sanity control) | --emit all | the one permitted knob arm: is data volume even the binding constraint? |
| v06_gumbel_root | search | Gumbel root + sequential halving | policy-improvement guarantee at small budgets; the strongest known AZ upgrade |
| v08_cold_start | bootstrap control | no --init | bounds the supervised prior's contribution |
| v09_strict_value | targets | train value on realized strict moves | trains on the benchmark metric itself — never done anywhere in this project |
| v12_frontier_curriculum | data | probe-and-reject instance mining | attacks the diagnosed saturation mechanism (§19d) |

Parked (documented, not scheduled): v02_td_blend, v03_hard_mining,
v05_mean_backup (incremental). Stub: v07_hybrid_actions (wave 2).

## 5b. Mandatory card structure (owner 2026-08-21)

Every variant — future ones included — carries plain-English card text a
non-expert can follow: `plain_what` / `plain_why` on the `Variant` (the
registry REFUSES variants without them), and `result` / `conclusion` sentences
in `results/variants/<vid>/VERDICT.json` (a definite conclusion even when
pending, flat, parked or killed). The HTML card renders exactly four parts —
What we tested / Why it might help / Result / Conclusion — with the technical
hypothesis/mechanism folded into a details block, and the tab opens with a
glossary translating expansions, frontier/graded/unseen, regret, B2, warm
start, seed and the "fluke chance" phrasing of p-values. FINDINGS entries end
with a `Conclusion (plain):` line.

## 6. How to read the results

HTML: the **Variants lab** tab of `report/selfplay.html` (per-arm cards:
hypothesis, mechanism, matched table vs control, paired p-values, verdict
chip). Files: `results/variants/<vid>/bench_{graded,frontier,unseen}_astar.json`
+ `gate_*_vs_control.json` + `generation.manifest.json` + `nets.txt`. Log:
`variants/FINDINGS.md` (result files + job ids + projected/actual nh). Verdict
rule (also in the report): WIN = a paired p<0.05 in the variant's favour on any
exam with no significant loss elsewhere (then replicated on a second seed);
LOSS = the reverse; everything else FLAT. Flat results are results.

## 7. Consolidated verdict table (program closeout, 2026-08-26)

| arm | axis | verdict | one-line evidence |
|---|---|---|---|
| v00 control | control | the bar | 228 / 160 / 172 (graded/frontier/unseen); seed pair bounds noise at ~4-5 solves |
| v01 visit-policy | targets | **KILLED** | every exam collapses, p<=1e-7 |
| v02/v03/v05 | — | parked | judged too incremental; never run |
| v04 emit-all | data | **ADOPTED** | frontier +11/+21 at two seeds (Fisher 3.4e-5); r8 extraction 6x cost caveat |
| v06 Gumbel root | search | not replicated | seed-7 moves win reversed at seed 8 |
| v07 root-slides | action-space | **FLAGSHIP** | wins all exams vs matched controls; regret 0.86 < pure-subgoal floor 1.17; transfers zero-shot (17/0, 23/0); depth saturates ~3 |
| v08 cold start | control | prior-worth | label-free-from-zero ties the supervised baseline on unseen (132 vs 134) |
| v09 strict-value | targets | **ADOPTED** | 2-seed frontier win (Fisher 0.0015), best regret, best unseen |
| v12 curriculum (+x3) | data | **KILLED** | flat at 1 and at 3 chained iterations |
| v13 three-way combo | combo | superseded | solves interfere, moves compose |
| v14 v04+v09 stack | combo | **ADOPTED** (recipe) | unseen win both seeds (Fisher 0.018); does not compound over iterations |
| v15/v16 slide training | action-space | closed (double negative) | uniform and search-ranked nudges both flat at 2 seeds; nets already generalize to slid states |
| adopt_mainline | transfer | flat | mature main-line iteration unchanged under the recipe |

**The program's three durable lessons (plain):**
1. **Train on the number you are graded on.** Switching the value network's
   target from abstract plan cost to real move counts was the strongest
   single network change (v09) — and every attempt to be cleverer about
   targets (v01, v02) lost to it.
2. **Keep everything the search examined.** Training on every certified
   decision, not just the winning line, was the cheapest reliable win (v04)
   — data volume beat data curation (v03, v12) every time they competed.
3. **Search in subgoals PLUS ordinary moves.** The only change that broke
   the proven quality ceiling was widening the action space (v07); no
   amount of better training inside the old space (v15, v16) could match
   it, and the fix transferred unchanged to board sizes it never saw.
