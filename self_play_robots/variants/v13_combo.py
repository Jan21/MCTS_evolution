"""Combination arm: the three wave-1 positive signals stacked (v04 + v06 + v09).

Wave 1 (variants/FINDINGS.md 4) found three arms better than control on
DIFFERENT axes with little mechanistic overlap:
  * v04_deep_emit  (data):    --emit all -> 2.2x records, frontier 171 vs 160
    (p=0.013);
  * v06_gumbel_root (search): Gumbel root + sequential halving -> graded moves
    19/6 wins (p=0.015), regret 2.05 vs 2.39, frontier 169 (p=0.049);
  * v09_strict_value (targets): strict-moves value targets -> graded moves
    19/8 (p=0.052), regret 2.06, frontier 168 (p=0.057).
Search improves WHAT is explored, emit-all improves HOW MUCH of it is kept,
strict targets improve WHAT the value learns from it -- orthogonal by
construction, so the combination is the natural next arm. Failure mode to
watch: emit-all's off-path decisions receive Gumbel-shaped root visits only at
the root; deeper decisions keep PUCT statistics, so the v09 rescale must not
double-count (it does not: it reads certified totals only).
"""
from variants import Variant
from variants.v06_gumbel_root import apply_selfplay          # noqa: F401 (re-export)
from variants.v09_strict_value import apply_train            # noqa: F401 (re-export)

VARIANT = Variant(
    vid="v13_combo",
    axis="combo",
    title="v04 + v06 + v09 stacked (emit-all, Gumbel root, strict targets)",
    hypothesis="The three wave-1 positive deltas are orthogonal (data volume / "
               "root exploration / value metric) and stack to a arm that beats "
               "control on frontier solves AND graded moves simultaneously.",
    mechanism="env EMIT=all + v06's generation hook + v09's train hook, "
              "otherwise the matched protocol.",
    expected_failure="Interaction: Gumbel-shaped trees emit different off-path "
                     "decision distributions; if v04's gain depended on "
                     "PUCT-shaped trees the stack underperforms v04 alone.",
    env={"EMIT": "all"},
    hooks=("selfplay", "train"),
)
