# 07 — Generality: the method, and where else it applies

## 1. The method, domain stripped away

**(a) Subgoal language = typed DAG of goal regressions.** `partial_plan.py` is 78
lines: nodes are goal / precondition / target-state / current-state; edges carry
open|fixed status and a cost; a plan is complete when no edge is open, its cost is
the edge sum. Search runs in plan space, so one step commits a macro worth several
moves. Applies wherever reaching a state first needs another object to establish a
precondition.

**(b) Realization gate.** A plan counts only if it replays through the raw simulator
(`eval/realize.py`, `replay_validate.py`). Solutions become self-certifying — found
by any means, still a true upper bound. This turned a paper 99.6% into a real 53.1%.

**(c) Self-labeling whole decisions.** Self-play labels *every* candidate at a
decision by finishing siblings with the current nets (`subgoal_selfplay/DESIGN.md`
§1), so the supervised ranking loss is reused unchanged and labels self-correct. No
oracle at training time.

**(d) Size-free scorer.** The nets were one tensor from size-invariance: per-cell
tokens, graph-masked attention, only the position embedding grid-locked. Delete it,
one net serves 8×8 to 96×96 (§48/§50).

Reusable law (§74): **label fidelity gates downstream utility with a threshold** —
91% argmin agreement gives equivalence, 89% solve-rate parity, 82% collapse. Two
cautions: a vocabulary extension is dead unless proposal, featurization *and* policy
training admit it (§40); expressibility ceilings are measurable apart from training
error.

## 2. Three cheap transfer targets

- **Rush Hour** — closest twin. Subgoal = "vacate C so piece P slides to S". Minimum:
  200 pinned puzzles, exact BFS labels, both planners at matched steps with the
  realization gate. ~1 nh.
- **Sokoban (Boxoban 8×8)** — subgoal = "push box B to S, pusher at C"; preconditions
  are ordered and partly irreversible, so the gate does real work. Minimum: 500
  levels, both planners, pooled solve rate + replay pass rate. ~3–5 nh.
- **Pushable-obstacle warehouse planning** — best test of (d): train one scorer at
  12×12, audit argmin agreement zero-shot at 16/24/32 against BFS, check the ~89%
  knee reproduces.

## 3. Next steps

1. **Rush Hour port (small, ~1 nh).** Reuses DAG, realizer and self-play loop
   verbatim; turns a Ricochet result into a method result. Best impact per hour.
2. **Fidelity-threshold law in domain two (small).** Re-run §74's dose-response; a
   quotable constant for when a learned labeler is good enough.
3. **Ceiling probe as reusable diagnostic (small).** Measure expressible ∧ realizable
   share before training. Here it proved 9.3% was unfixable by training.
4. **Wire by-reference end-to-end (small–medium).** §40: ~5 driver lines, non-start
   helper featurization, policy retrain without the silent filter. Closes the clearest
   gap; yields a "propose × featurize × train" checklist.
5. **Size-free scorer as standalone result (medium).** One net 8×8→96×96 with the
   flat-then-cliff curve (§50/§54) — useful anywhere a looped graph transformer is
   grid-locked.
