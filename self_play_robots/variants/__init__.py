"""Variants lab -- a registry of self-play design variants run under ONE matched
protocol and compared against a common control arm (owner directive 2026-08-20:
"many modified .py architectural files/design variants", results as a separate
HTML tab, thoroughly documented, no human labels in the loop).

A variant = one module `variants/vNN_<slug>.py` exporting `VARIANT = Variant(...)`.
It can change the loop through two channels:
  1. KNOBS -- env/CLI overrides the runner job reads (`python -m variants describe
     <vid> --format env`): generation knobs (BOARDS/EXP/STOP/MINEXP/EMIT/BACKUP...),
     extra flags for spr.selfplay / spr.train / spr.bench.
  2. HOOKS -- `apply_<phase>()` functions (phase in selfplay|train|bench) that
     monkeypatch spr modules INSIDE the process that runs the phase (workers
     included: spr.selfplay applies the hook in `_init_worker`). This is how a
     variant ships modified architecture (its own search function, target
     transform, ...) without forking the stack for everyone.

Matched protocol (DESIGN.md): every arm = ONE self-play iteration at g24r4 in the
B2 vocabulary from the SAME frozen warm-start nets (mix_b2mix_iter2, the current
best pair), same seed boards ids (40000+300*index), same budget (30 boards x 8
instances, 300 expansions, stop 80), same retrain recipe (6 ep, lr 1e-4, byref)
unless the variant IS a training change -- then benched with the arena A* (B2
convention) on the pinned graded exam, the pinned frontier exam, and the UNSEEN
exam (fresh boards ids 20000+, `variants/exam.py`, never trained on by anything).
Verdicts come from `spr.gate compare` paired tests vs v00_control.

Boards-id budget: generation ids 40000..49999 (300 per variant by index), exam
ids 20000..20999. Nothing may ever train on 0-1199 (pinned) or 20000+ (exam).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path

VARIANTS_DIR = Path(__file__).resolve().parent
REPO = VARIANTS_DIR.parent.parent


@dataclass
class Variant:
    vid: str                       # module name stem, e.g. "v01_visit_policy"
    axis: str                      # search | targets | data | action-space | control
    title: str
    hypothesis: str                # one sentence, falsifiable
    mechanism: str                 # what exactly changes, and where
    expected_failure: str          # how we would see it NOT work
    cost_note: str = "~2 h on 1 GPU (0.25 nh)"
    # knob channel (runner job): env-style overrides + extra per-phase CLI flags
    env: dict = field(default_factory=dict)          # BOARDS/EXP/STOP/EMIT/...
    selfplay_flags: str = ""
    train_flags: str = ""
    bench_flags: str = ""
    # hook channel: names of callables in the variant module, applied in-process
    hooks: tuple = ()              # subset of ("selfplay", "train", "bench")
    status: str = "wave1"          # wave1 | wave2 | stub


_ORDER = sorted(p.stem for p in VARIANTS_DIR.glob("v[0-9][0-9]_*.py"))


def ids():
    return list(_ORDER)


def get(vid: str) -> Variant:
    mod = importlib.import_module(f"variants.{vid}")
    v = mod.VARIANT
    assert v.vid == vid, f"{vid}: VARIANT.vid mismatch ({v.vid})"
    return v


def index_of(vid: str) -> int:
    return _ORDER.index(vid)


def gen_ids(vid: str, boards: int = 30) -> str:
    """Deterministic per-variant fresh-board id range (never exam/pinned ids)."""
    lo = 40000 + 300 * index_of(vid)
    return f"{lo}-{lo + boards - 1}"


def apply_phase(vid: str, phase: str, ctx=None):
    """Called inside the process running a phase; a no-op unless the variant
    declares a hook for it. Import happens here so worker processes (spawn)
    pick the hook up after their own imports. `ctx` hands the caller's state
    to hooks that accept it (spr.selfplay passes its worker dict `_W`: under
    `python -m spr.selfplay` the live module is __main__, so a hook importing
    spr.selfplay would see a SECOND, empty module instance)."""
    import inspect
    v = get(vid)
    if phase in v.hooks:
        mod = importlib.import_module(f"variants.{vid}")
        fn = getattr(mod, f"apply_{phase}")
        if inspect.signature(fn).parameters:
            fn(ctx)
        else:
            fn()
        print(f"[variants] {vid}: {phase} hook applied", flush=True)
    return v


def main(argv=None):
    import argparse, json
    p = argparse.ArgumentParser(description="variants registry CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("describe")
    d.add_argument("vid")
    d.add_argument("--format", choices=["env", "json"], default="json")
    sub.add_parser("list")
    a = p.parse_args(argv)
    if a.cmd == "list":
        for vid in ids():
            v = get(vid)
            print(f"{vid:24s} [{v.axis:12s}] {v.status:6s} {v.title}")
        return
    v = get(a.vid)
    if a.format == "json":
        print(json.dumps({k: getattr(v, k) for k in (
            "vid", "axis", "title", "hypothesis", "mechanism", "expected_failure",
            "cost_note", "env", "selfplay_flags", "train_flags", "bench_flags",
            "hooks", "status")}, indent=1, default=list))
    else:                                    # eval-able KEY=VAL lines for bash
        base = {"BOARDS": 30, "PER_BOARD": 8, "EXP": 300, "STOP": 80, "NOISE": 0.25,
                "EP": 6, "LR": "1e-4", "WORKERS": 6, "MINEXP": 0, "EMIT": "path",
                "BACKUP": "min", "SIB": 1}
        base.update(v.env)
        for k, val in base.items():
            print(f"{k}={val}")
        print(f"GEN_IDS={gen_ids(a.vid, int(base['BOARDS']))}")
        print(f"SELFPLAY_FLAGS='{v.selfplay_flags}'")
        print(f"TRAIN_FLAGS='{v.train_flags}'")
        print(f"BENCH_FLAGS='{v.bench_flags}'")
        print(f"HOOKS='{','.join(v.hooks)}'")


if __name__ == "__main__":
    main()
