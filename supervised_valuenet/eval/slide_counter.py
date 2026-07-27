"""Matched physics-work accounting: count `simulate.slide` invocations.

`analysis/publishability.md` objection 1.1 asks for a work unit commensurable
across the two planners, because the headline "5.3 vs 423 expansions" counts
only high-level decisions and hides the backward planner's realization
physics. `simulate.slide` is the one primitive both stacks bottom out in — the
forward planner calls it once per (robot, direction) when generating
successors (`move_planner/state.py`) and inside its NN featurization
(`move_planner/encode.py`), the backward planner calls it inside every
realization / prefix-check / park-repair BFS (`eval/realize.py`). The backward
SEARCH itself — `skeleton/astar.py`'s proposal step — performs NO slides: it
plans over the board's precomputed slide-graph and exact-distance tables, and
its bucket (`backward_search`) measured literally zero in the pilot
(`eval/results/instrumentation_ab/v2_slidepilot_backward5.json`). Counting
`slide` on both sides gives one number in one unit.

**The unit is well-defined but NOT physics-only — read before quoting a
ratio.** Roughly 112 slides per forward expansion decompose as 16 from
`legal_moves` successor generation (search physics, `move_planner/state.py`)
plus ~96 from `dest_cells` one-step-lookahead featurization of the states fed
to the nets (`move_planner/encode.py`), plus a once-per-board 1024-slide
`_slide_fields` fill (`train/encode.py`, lru-cached). The backward planner's
counterpart featurization reads precomputed graph/distance tables and never
calls `slide`, so it contributes zero to its buckets. A cross-system ratio of
raw totals therefore conflates physics work with how each planner happens to
featurize states. Under `--count-slides`, `eval/compare.py` splits the forward
attribution: slides made inside `Guide.eval_states` land in `forward_encode`
(featurization), the rest of the search window in `forward_search` (physics).
Quote `forward_search` for a physics-only comparison, or the total WITH the
featurization caveat — never the bare total. Runs recorded before the split
lump both into `forward_search`.

**Nothing is silently dropped.** A `slide` call made while no bucket is
active is counted under the sentinel bucket `UNBUCKETED` ("unbucketed")
rather than discarded, so any unlabeled call path shows up in the per-row
map and the aggregate instead of vanishing from a number presented as a
total. A nonzero sentinel inside a measured window means an unattributed
call path — investigate before publishing.

Installation rebinds the name everywhere it was captured at import time
(`from simulate import slide` binds a module-level reference, so patching
`simulate.slide` alone would miss those). It is therefore done by sweeping
`sys.modules` for any module whose `slide` attribute IS the original function,
which needs no hardcoded module list and picks up lazily-imported modules as
long as `install()` runs after them.

**Off by default and never installed unless asked.** The wrapper adds a Python
call to the hottest function in the codebase, so a counted run's wall-clock is
NOT comparable to an uncounted one — accounting passes and timing passes must
be separate runs. `eval.compare --count-slides` opts in.

    from eval import slide_counter
    slide_counter.install()
    with slide_counter.bucket("strict_realize"):
        ...                       # every slide() in here lands in that bucket
    slide_counter.counts()        # {"strict_realize": 1234, ...}
"""
from __future__ import annotations

import contextlib
import sys

import simulate

#: Sentinel bucket for calls made while no bucket is active. Counting them
#: (instead of dropping them, as before 2026-07-27) keeps `slide_calls`
#: honest as a TOTAL: an unlabeled call path becomes a visible nonzero
#: sentinel, not a silent hole in the accounting.
UNBUCKETED = "unbucketed"

_counts: dict[str, int] = {}
_bucket: str | None = None
_original = None
_installed = False


def install():
    """Wrap `simulate.slide` and rebind every import-time capture of it.

    Idempotent. Returns the number of module bindings rewritten (the original
    `simulate` module included), which callers may record as evidence that the
    instrument actually attached.
    """
    global _installed, _original
    if _installed:
        return 0
    _original = simulate.slide

    def counting_slide(*args, **kwargs):
        b = _bucket if _bucket is not None else UNBUCKETED
        _counts[b] = _counts.get(b, 0) + 1
        return _original(*args, **kwargs)

    counting_slide.__wrapped__ = _original
    rebound = 0
    for module in list(sys.modules.values()):
        if module is None:
            continue
        try:
            if getattr(module, "slide", None) is _original:
                module.slide = counting_slide
                rebound += 1
        except Exception:                      # exotic module proxies
            continue
    simulate.slide = counting_slide            # for later `from simulate import`
    _installed = True
    return rebound


def installed():
    return _installed


@contextlib.contextmanager
def bucket(name):
    """Attribute every `slide` call made inside the block to `name`.

    Nests: an inner bucket restores the outer one on exit, so wrapping the
    whole backward search in "backward_search" and the realization hooks in
    their own buckets yields a disjoint split, not double counting.
    """
    global _bucket
    previous = _bucket
    _bucket = name
    try:
        yield
    finally:
        _bucket = previous


def counts():
    return dict(_counts)


def delta(before):
    """Per-bucket increase since a `counts()` snapshot.

    Zeros are omitted EXCEPT the `UNBUCKETED` sentinel, which is always
    present (0 when nothing leaked) so every per-row map affirmatively
    states that no call was dropped, rather than leaving absence ambiguous.
    """
    now = counts()
    out = {k: now[k] - before.get(k, 0) for k in now}
    out = {k: v for k, v in out.items() if v or k == UNBUCKETED}
    out.setdefault(UNBUCKETED, 0)
    return out
