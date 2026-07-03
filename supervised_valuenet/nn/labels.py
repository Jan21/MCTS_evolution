"""Turn raw integer cost-to-go into value-network targets.

The value head is a classifier over discrete cost-to-go bins, not a regressor.
Records store the raw integer cost so the binning and smoothing scheme stays a
free knob: sweep `num_classes`, `clip`, and `sigma` without regenerating data.

Smoothing uses the HL-Gauss histogram-loss trick (Imani & White; "Stop
Regressing"): place a Gaussian centered on the true value and bin it, so nearby
costs share probability mass. A scalar prediction is recovered as the expected
bin, which is what the search uses and what MAE is measured against.
"""
from __future__ import annotations

import math


def to_class(cost_to_go: int, num_classes: int, clip: bool = True) -> int:
    """Hard class index for a cost (clipped into range)."""
    c = int(round(cost_to_go))
    if clip:
        return max(0, min(c, num_classes - 1))
    return c


def soft_label(cost_to_go: float, num_classes: int, sigma: float = 1.0) -> list[float]:
    """HL-Gauss soft target: Gaussian over bins centered at `cost_to_go`."""
    if sigma <= 0:
        out = [0.0] * num_classes
        out[to_class(cost_to_go, num_classes)] = 1.0
        return out
    w = [math.exp(-0.5 * ((i - cost_to_go) / sigma) ** 2) for i in range(num_classes)]
    z = sum(w)
    return [x / z for x in w] if z > 0 else [1.0 / num_classes] * num_classes


def expected_value(probs) -> float:
    """Recover a scalar cost from a class distribution (its mean bin)."""
    return sum(i * p for i, p in enumerate(probs))
