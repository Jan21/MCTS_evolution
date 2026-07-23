"""Shared utilities for the report generator (eval/build_report.py).

Three registries thread through the whole build:

* SOURCES   — every file the build touched, with status ("ok" / "missing" /
              "invalid") and a note; rendered as the provenance table.
* CHECKS    — every headline number rendered on the page, registered at render
              time via ck(); after the page is written, the verifier re-parses
              the HTML and compares every tagged span against its source value.
              Any mismatch fails the build.
* NEEDLES   — substring assertions on the final HTML (a string that must, or
              must never, appear), for facts that are not single numbers.

Nothing in this module knows about the page's content.
"""

import html as _html
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "eval", "results", "report.html")


def rp(*parts):
    return os.path.join(ROOT, *parts)


def esc(x):
    return _html.escape(str(x), quote=True)


# ---------------------------------------------------------------------------
# Source registry (provenance)
# ---------------------------------------------------------------------------

SOURCES = {}   # relpath -> {"status": ..., "note": ...}


def _record(relpath, status, note=""):
    e = SOURCES.setdefault(relpath, {"status": status, "note": note})
    if status != "ok" or e["status"] == "ok":
        e["status"], e["note"] = status, note or e["note"]


def load_json(relpath, required_keys=()):
    """Load a JSON file; on absence/corruption record it and return None."""
    path = rp(relpath)
    if not os.path.exists(path):
        _record(relpath, "missing")
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        _record(relpath, "invalid", str(e)[:120])
        return None
    for k in required_keys:
        if isinstance(data, dict) and k not in data:
            _record(relpath, "invalid", f"missing key {k!r}")
            return None
    _record(relpath, "ok")
    return data


def load_jsonl(relpath, limit=None):
    path = rp(relpath)
    if not os.path.exists(path):
        _record(relpath, "missing")
        return None
    rows = []
    try:
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except (json.JSONDecodeError, OSError) as e:
        _record(relpath, "invalid", str(e)[:120])
        return None
    _record(relpath, "ok")
    return rows


def source_note(relpath, note):
    if relpath in SOURCES:
        SOURCES[relpath]["note"] = note


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

def fnum(x, dec=1):
    """Fixed-decimal format; em-dash for missing."""
    if x is None:
        return "—"
    return f"{x:.{dec}f}"


def fpct(x, dec=1):
    """x is a FRACTION (0..1); rendered as a percentage."""
    if x is None:
        return "—"
    return f"{x * 100:.{dec}f}%"


def fpct100(x, dec=1):
    """x is already in percent units."""
    if x is None:
        return "—"
    return f"{x:.{dec}f}%"


def ffrac(k, n):
    if k is None or n is None:
        return "—"
    return f"{k}/{n}"


def fint(x):
    if x is None:
        return "—"
    return f"{x:,}"


# ---------------------------------------------------------------------------
# Check registry — the self-verification backbone
# ---------------------------------------------------------------------------

CHECKS = []    # dicts: id, rendered, source, desc, raw


def ck(rendered, source, desc, raw=None):
    """Register a rendered headline value and return the tagged span.

    rendered : the exact string shown on the page
    source   : the relpath of the JSON the value came from
    desc     : human description ("6-robot frontier, backward solved")
    raw      : the raw source value (for the check table's 'source value' col)
    """
    cid = len(CHECKS)
    CHECKS.append({"id": cid, "rendered": str(rendered), "source": source,
                   "desc": desc, "raw": raw})
    return f'<span class="ck" data-ck="{cid}">{esc(rendered)}</span>'


NEEDLES = []   # dicts: desc, needle, present


def need(desc, needle, present=True):
    """Assert that `needle` appears (or never appears) in the final HTML."""
    NEEDLES.append({"desc": desc, "needle": needle, "present": present})


def fact(desc, ok):
    """Assert a build-time boolean (rendered into the check log)."""
    NEEDLES.append({"desc": desc, "needle": None, "present": bool(ok)})


# ---------------------------------------------------------------------------
# Small HTML helpers
# ---------------------------------------------------------------------------

def scroll(inner):
    return f'<div class="scroll">{inner}</div>'


def pending(msg):
    return (f'<p class="pending"><span class="pendtag">pending</span> '
            f'{esc(msg)}</p>')


def progress_tag(msg):
    return (f'<p class="pending"><span class="pendtag run">in progress</span> '
            f'{esc(msg)}</p>')


import re as _re


def kicker_h2(kicker, title, lede="", sec_id=None):
    if sec_id is None:
        sec_id = _re.sub(r"[^a-z0-9]+", "-", kicker.lower()).strip("-")[:40]
    lede_html = f'\n  <p class="lede">{lede}</p>' if lede else ""
    return (f'<section id="{sec_id}">\n'
            f'<div class="sechead"><div class="kicker">{esc(kicker)}</div>\n'
            f'<h2>{title}</h2>{lede_html}\n</div>')


def dot(family):
    """Colored series key dot for a planner family."""
    return f'<span class="dot dot-{family}" aria-hidden="true"></span>'


def chip(text, kind=""):
    return f'<span class="chip {kind}">{esc(text)}</span>'
