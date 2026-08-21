"""The wave-2 winners stacked: emit-all data (v04) + strict-moves value targets
(v09), WITHOUT the non-replicating Gumbel root (v06).

v13 showed v04+v06+v09 interfere on frontier solves (v04's +11/+21 gain
vanished under Gumbel-shaped trees) while composing on moves. This arm drops
the interfering component: PUCT trees exactly as control (so v04's off-path
emission sees the tree shape its win was measured on), doubled record volume,
value trained on the benchmark metric. If it holds at both seeds it becomes
the recipe the main loop adopts (see results/variants/adopt_mainline/).
"""
from variants import Variant
from variants.v09_strict_value import apply_train            # noqa: F401 (re-export)

VARIANT = Variant(
    vid="v14_stack",
    axis="combo",
    title="v04 + v09 stacked (emit-all + strict targets, no Gumbel)",
    hypothesis="The two replicated/2-seed wins compose: frontier solves from "
               "emit-all AND the regret/moves gains from strict targets, "
               "with no interference because the tree shape is control's.",
    mechanism="env EMIT=all + v09's train hook; everything else the matched "
              "protocol.",
    expected_failure="Off-path decisions rescaled to strict units double the "
                     "attribution noise (off-path strict totals come from "
                     "sibling completions) -> value regresses, frontier gain "
                     "survives but regret worsens.",
    env={"EMIT": "all"},
    hooks=("train",),
)
