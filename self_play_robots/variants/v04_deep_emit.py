"""Train on ALL expanded decisions, not only the principal path.

The control extracts labeled groups along the principal path only (--emit path).
The tree expands and certifies many off-path decisions whose labels are equally
physics-certified; AlphaZero trains on every visited state. --emit all
(namespaced depths) roughly doubles records per expansion -- data efficiency for
free if off-path decisions are not systematically junk (they are the states the
CURRENT policy considers plausible, exactly where its ranking needs sharpening).
"""
from variants import Variant

VARIANT = Variant(
    vid="v04_deep_emit",
    axis="data",
    title="Train on every examined decision (sanity check: small tweak)",
    hypothesis="Off-principal-path certified decisions are cheap extra signal: "
               "same generation budget, ~2x records, better generalization on "
               "unseen boards.",
    mechanism="Generation knob only: spr.selfplay --emit all.",
    expected_failure="Off-path labels are biased toward states a WEAK ranking "
                     "visits; training on them entrenches the bias (flat or "
                     "worse unseen-exam row).",
    env={"EMIT": "all"},
)
