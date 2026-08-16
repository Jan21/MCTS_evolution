# Memo — the thesis: is "backward beats forward" demonstrated?

## 1. Demonstrated? Partly, and honestly so.

**Strongest.** On the whole pinned pool at every rung above base, the subgoal
planner wins by +8.0, +15.8, +19.8, +24.0, +47.8 points, monotone along both
hardness axes, all p<0.0005 (§27, §46). The pooled view needs no selection
argument, every solve is replay-certified, and the efficiency edge survives a
matched physics unit (64 vs 40,448 slide calls, §25). The teacher-death curve
is independent, well-measured support: the forward pipeline's supervision
fails on up to 61-64% of instances at scale.

**Weakest.** At base scale forward simply wins (450/450 vs 430) and keeps a
significant edge on most oracle-gradable sets. The dramatic "forward collapses"
numbers are budget-censored: at 4x budget forward gains +21.9/+31.2 points at
16x16 and the backward lead loses significance (§28) — and backward was never
given the same extra budget, so there is no like-for-like row. The opponent is
one seed of an oracle-supervised forward planner; forward self-play at scale is
unmeasured. Internally, the vocabulary-lift story is partly hollow (§40: the
by-reference machinery never reached the learned planner), and retraining is
bistable (§44/45), so all headline backward rows are zero-shot nets.

## 2. What a tough reviewer attacks first
1. **Budget.** Forward is saturated at the cap; the collapse holds only on the
   grid axis. Everything else follows from this.
2. **Selection.** Frontier sets are defined by a move-level search failing —
   adversarial to forward by construction.
3. **What is actually being compared:** a hand-designed subgoal vocabulary plus
   exact distance tables vs generic flat search, with no learned-subgoal
   (kSubS-style) baseline. This attributes the win to engineering, not to
   subgoals.
4. **Single seed and asymmetric tuning** (base forward = best of four runs).
5. Single domain.

## 3. Next steps
1. **Symmetric budget sweep** — both planners, 1x-10x budget, same subsamples,
   every rung. Turns the top objection into a measured curve. *Medium.*
2. **Forward self-play at scale** as the at-scale opponent. Removes the
   "teacher death" confound that makes the strong claim rejectable. *Medium.*
3. **Cost-matched headline** — report wins per unit physics/wall-clock rather
   than expansions. Data largely exists. *Small.*
4. **Learned-subgoal baseline** at base plus one hard rung. The only way to
   claim subgoals-as-such rather than this vocabulary. *Large.*
5. **Wire by-reference into the learned planner** and re-measure. Closes the
   ceiling-vs-achieved gap the paper advertises. *Small-medium.*
