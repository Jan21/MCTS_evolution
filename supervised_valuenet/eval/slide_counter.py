"""Matched physics-work accounting: count `simulate.slide` invocations.

`analysis/publishability.md` objection 1.1 asks for a work unit commensurable
across the two planners, because the headline "5.3 vs 423 expansions" counts
only high-level decisions and hides the backward planner's realization
physics. `simulate.slide` is the one primitive both stacks bottom out in — the
forward planner calls it once per (robot, direction) when generating
successors (`move_planner/state.py`), the backward planner calls it inside
every realization / prefix-check / park-repair BFS (`eval/realize.py`) and in
`skeleton/astar.py`'s proposal step. Counting it on both sides gives one
number in one unit.

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
        if _bucket is not None:
            _counts[_bucket] = _counts.get(_bucket, 0) + 1
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
    """Per-bucket increase since a `counts()` snapshot, zeros omitted."""
    now = counts()
    out = {k: now[k] - before.get(k, 0) for k in now}
    return {k: v for k, v in out.items() if v}
