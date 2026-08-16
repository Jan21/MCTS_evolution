# The by-reference step, and why the planner never uses it

## 1. What it is; why the learned planner is blind

Every subgoal needs a *helper* — a robot standing somewhere to stop the mover.
Old/B1 vocabulary offers helpers only **where they stand now**. The by-reference
step (B2) lets a subgoal name a robot **at a cell the plan itself will put it
on**: reuse a placement already paid for. `skeleton/astar.py::_reference_helpers`
builds those phantom helpers.

Three independent blocks (FINDINGS 40):

- **Proposal.** `eval/compare.py::_nn_astar_backward` calls `propose(...)`
  without injecting reference helpers and `_apply(...)` without
  `by_reference=True`. `nn/generate.py` (labeling) does both correctly.
- **Featurization.** `eval/end2end.py::_hidx` matches a helper by **start
  position**; a by-reference helper sits at a planned cell, so it returns None
  and the candidate is dropped.
- **Training.** `train/policy_common.py::_meta` keys on start positions too and
  silently skips such records — the policy net never saw one. (The value net,
  on raw cell indices, did.)

The 99.6% ceiling probe is honest — correctly wired solver — but measures a
language the planner cannot speak. The gap is **integration, not ranking**.

## 2. Honest size and payoff

**Code small; experiment medium.** Wiring ~5 lines. `_hidx` can key on robot
**colour** (already in `cand_helper`), plus one feature channel naming the
referenced cell. Then a policy retrain per config (~2 node-hours), filter removed.

The real cost is **data**: FINDINGS 32/68 show affordable corpora are depleted of
by-reference labels (13.5% uncapped exact → 4.8% capped; NN labeler 0.3–1.3%).
Wiring onto depleted corpora measures the depletion, not the language.

Payoff: **modest at base, plausibly real at the frontier**. B1→B2 at fixed nets
is +2/0 (FINDINGS 39) — B1 carries the executable gain. But measured supply is
~53 by-reference candidates per expansion in 97% of expansions on 134/134
frontier instances (job 4599947), and headroom widens with scale. Even a null
result closes the study's clearest known hole.

## 3. Next steps on the plan language

1. **Zero-shot supply A/B, no retrain.** Wire driver + colour-keyed `_hidx`; run
   g16r6 frontier by-reference on vs off, same nets. Separates supply from
   ranking. ~1 day, 2–3 node-hours.
2. **Cap the candidate pool.** By-reference inflates candidates ~1.5×; keep
   top-m by heuristic score. Protects the step-efficiency headline. Hours.
3. **Small uncapped-exact B2 seed corpus**, cheapest config — FINDINGS 68 calls
   this the only rescue. Pilot ~50 boards for yield/hour (~2 nh); full run
   unknown, so measure before promising.
4. **Retrain the policy without the filter** once a healthy corpus exists — the
   actual test of whether the ceiling is reachable. ~2 nh per config.
5. **Attribute ceiling headroom by candidate family** (FINDINGS 12
   decomposition): how much of 95.6→99.6 by-reference can even claim. Zero
   compute, half a day.
