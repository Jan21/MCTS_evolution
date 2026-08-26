"""Control arm -- the current loop recipe, unchanged, under the matched protocol.

Every other variant is judged against THIS arm's paired results (same warm-start
nets, same board-id budget shape, same benches). Its own numbers also measure
"one more standard iteration from the mix_b2mix_iter2 nets at g24r4", which is
the honest no-op baseline: if a variant does not beat v00, it did nothing.
"""
from variants import Variant

VARIANT = Variant(
    vid="v00_control",
    axis="control",
    title="The standard recipe, unchanged",
    hypothesis="Baseline: one more standard iteration from the frozen seed nets "
               "moves nothing beyond noise (the 24-only loop is saturated, §19).",
    mechanism="spr.selfplay MCTS generation (300 exp, stop 80, root noise 0.25, "
              "root-all, sibling completion, B2) -> warm retrain (6 ep, lr 1e-4, "
              "byref) -> arena A* benches (graded / frontier / unseen).",
    expected_failure="n/a (control).",
)
