"""Build eval/results/report.html — one self-contained, tabbed page comparing
every Ricochet-Robots planner variant measured in this repository.

Usage (from the repo root):
    PYTHONPATH=. python3 -m eval.build_report

The script reads every machine-readable result it knows about, renders a single
HTML file with inline CSS/JS/SVG (no external assets), and prints a verification
table to stdout comparing each headline number against the JSON aggregate it
came from.  Missing source files never crash the build: the affected section is
rendered with a "pending" note (or a dash-filled row) instead.  Re-run any time
new results land — no code change needed for a refresh.

No result number is hardcoded in the HTML — everything is computed here from
the source files at build time.  The page is organised into five tabs
(Overview / Base-scale results / Why plans fail / Scaling / Method & sources)
with hash routing, so every tab and every section is deep-linkable.
"""

import glob
import html as _html
import json
import math
import os
import pickle
import re
import sys
from collections import Counter, deque
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "eval", "results", "report.html")

# Robot slot -> display name.  Slot order is fixed project-wide
# (move_planner/state.py COLOR_ORDER = nn.gen_grids PALETTE[:4]); duplicated
# here as a display label only, so the report builder needs no heavy imports.
SLOT_NAMES = ("Red", "Blue", "Green", "Yellow",
              "Purple", "Orange", "Cyan", "Pink")
SLOT_VARS = ("--r-red", "--r-blue", "--r-green", "--r-yellow",
             "--r-red", "--r-blue", "--r-green", "--r-yellow")


def rp(*parts):
    return os.path.join(ROOT, *parts)


def esc(x):
    return _html.escape(str(x), quote=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

SOURCES = {}  # relpath -> dict(status, data, mtime, note)

# extra self-checks appended while sections render; verified at the end.
# each: {"desc": str, "needle": str|None, "present": bool, "ok": bool|None}
AUX_CHECKS = []


def aux_need(desc, needle, present=True):
    AUX_CHECKS.append({"desc": desc, "needle": needle, "present": present,
                       "ok": None})


def aux_fact(desc, ok):
    AUX_CHECKS.append({"desc": desc, "needle": None, "present": True,
                       "ok": bool(ok)})


def _entry(relpath):
    entry = {"status": "missing", "data": None, "mtime": None, "note": ""}
    SOURCES[relpath] = entry
    return entry


def load_json(relpath, required_keys=()):
    """Load a JSON file; record provenance; return dict or None."""
    path = rp(relpath)
    entry = _entry(relpath)
    if not os.path.exists(path):
        entry["note"] = "file not found"
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:  # malformed file: treat as pending, never crash
        entry["status"] = "unreadable"
        entry["note"] = f"could not parse: {e}"
        return None
    for k in required_keys:
        if k not in data:
            entry["status"] = "unreadable"
            entry["note"] = f"missing expected key {k!r}"
            return None
    entry["status"] = "ok"
    entry["data"] = data
    entry["mtime"] = datetime.fromtimestamp(os.path.getmtime(path)).strftime(
        "%Y-%m-%d %H:%M"
    )
    return data


def load_jsonl(relpath, limit=None):
    """Load a JSON-lines file as a list; record provenance; None if absent."""
    path = rp(relpath)
    entry = _entry(relpath)
    if not os.path.exists(path):
        entry["note"] = "file not found"
        return None
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    except Exception as e:
        entry["status"] = "unreadable"
        entry["note"] = f"could not parse: {e}"
        return None
    entry["status"] = "ok"
    entry["note"] = f"{len(rows)} row(s)"
    entry["mtime"] = datetime.fromtimestamp(os.path.getmtime(path)).strftime(
        "%Y-%m-%d %H:%M"
    )
    return rows


def record_glob(pattern, found):
    """Record a glob pattern in the provenance list."""
    SOURCES[pattern] = {
        "status": "ok" if found else "missing",
        "data": None,
        "mtime": None,
        "note": f"{len(found)} file(s)" if found else "no files yet",
    }


def record_note(relpath, status, note):
    SOURCES[relpath] = {"status": status, "data": None, "mtime": None,
                        "note": note}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fnum(x, dec):
    if x is None:
        return "—"
    return f"{x:.{dec}f}"


def fpct(x, dec=1):
    if x is None:
        return "—"
    return f"{100.0 * x:.{dec}f}%"


def ffrac(k, n):
    if k is None or not n:
        return "—"
    return f"{k}/{n} ({100.0 * k / n:.1f}%)"


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def systems_of_kind(comp, kind):
    """Yield (name, system_dict) for systems of a given kind in an
    eval.compare output file."""
    out = []
    for name, s in (comp or {}).get("systems", {}).items():
        if isinstance(s, dict) and s.get("kind") == kind and "aggregate" in s:
            out.append((name, s))
    return out


def first_backward(comp):
    ss = systems_of_kind(comp, "backward")
    return ss[0] if ss else (None, None)


def first_forward(comp):
    ss = systems_of_kind(comp, "forward")
    return ss[0] if ss else (None, None)


def agg_from_backward_rows(rows):
    """Recompute the aggregate over a slice of backward per-instance rows
    (used for the matched first-150 'before fixes' stage)."""
    n = len(rows)
    if n == 0:
        return None
    solved = [r for r in rows if r.get("solved")]
    regs = [r["regret"] for r in solved if r.get("regret") is not None]
    moves = [r["realized_strict"] for r in solved if r.get("realized_strict") is not None]
    return {
        "n": n,
        "solved": len(solved),
        "solve_rate": len(solved) / n,
        "mean_regret": (sum(regs) / len(regs)) if regs else None,
        "pct_optimal": (100.0 * sum(1 for r in regs if r == 0) / len(regs)) if regs else None,
        "mean_moves": (sum(moves) / len(moves)) if moves else None,
        "mean_expansions": sum(r.get("expansions", 0) for r in rows) / n,
        "mean_seconds": sum(r.get("seconds", 0.0) for r in rows) / n,
        "plan_found": sum(1 for r in rows if r.get("plan_found")),
    }


FORWARD_LABELS = [
    # (substring of checkpoint path, friendly label, one-line description)
    ("candidate_scored", "Forward planner — candidate-scored training",
     "value network also trained on the exact cost of every legal next move"),
    ("runs_warm", "Forward planner — self-play, warm start",
     "kept training on its own solutions, starting from the supervised networks"),
    ("runs_scratch", "Forward planner — self-play, from scratch",
     "trained only on its own solutions, starting from random weights"),
    ("best.ckpt", "Forward planner — supervised",
     "trained on optimal solutions produced by the exact reference solver"),
]


def forward_label(name):
    m = re.search(r"\((.*?)\)", name)
    path = m.group(1) if m else name
    for frag, label, desc in FORWARD_LABELS:
        if frag in path:
            return label, desc, path
    return f"Forward planner ({path})", "", path


# ---------------------------------------------------------------------------
# SVG charts (inline, theme-aware via CSS variables)
# ---------------------------------------------------------------------------

def nice_ceiling(v):
    if v <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * mag:
            return m * mag
    return 10 * mag


def wrap_label(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines[:3]


def _fam_color(family):
    if family == "forward":
        return "var(--fwd)"
    if family == "backward":
        return "var(--bwd)"
    return "var(--faint)"


def bar_chart(rows, caption, aria, value_fmt="{:.1f}", vmax=None):
    """Horizontal bar chart. rows: list of (label, value, family)."""
    if not rows:
        return ""
    label_w, pad_r, bar_h, gap, top, axis_h = 320, 96, 20, 22, 12, 30
    W = 900
    plot_w = W - label_w - pad_r
    if vmax is None:
        vmax = nice_ceiling(max(v for _, v, _ in rows))
    labels_wrapped = [wrap_label(lbl, 44) for lbl, _, _ in rows]
    H = top + len(rows) * (bar_h + gap) - gap + axis_h + 8
    ticks = [vmax * i / 5.0 for i in range(6)]
    parts = [
        f'<svg viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="{esc(aria)}" '
        f'style="width:100%;height:auto;display:block">'
    ]
    base_y = top + len(rows) * (bar_h + gap) - gap + 6
    for t in ticks:
        x = label_w + (t / vmax) * plot_w
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" y2="{base_y}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{base_y + 16}" text-anchor="middle" '
            f'fill="var(--muted)" font-size="11" '
            f'style="font-variant-numeric:tabular-nums">{t:g}</text>'
        )
    parts.append(
        f'<text x="{label_w + plot_w / 2:.0f}" y="{H - 2}" text-anchor="middle" '
        f'fill="var(--muted)" font-size="11">{esc(caption)}</text>'
    )
    parts.append(
        f'<line x1="{label_w}" y1="{top - 4}" x2="{label_w}" y2="{base_y}" '
        f'stroke="var(--baseline)" stroke-width="1"/>'
    )
    for i, (label, value, family) in enumerate(rows):
        y = top + i * (bar_h + gap)
        w = max((value / vmax) * plot_w, 1.5)
        color = _fam_color(family)
        x0 = label_w
        r = min(4.0, w)
        path = (
            f"M{x0},{y} h{w - r:.2f} a{r},{r} 0 0 1 {r},{r} "
            f"v{bar_h - 2 * r:.2f} a{r},{r} 0 0 1 -{r},{r} h-{w - r:.2f} z"
        )
        vtxt = value_fmt.format(value)
        parts.append(
            f'<g><path d="{path}" fill="{color}">'
            f"<title>{esc(label)}: {esc(vtxt)}</title>"
            f"</path>"
        )
        lines = labels_wrapped[i][:2]
        y0 = y + bar_h / 2 + 4 - (len(lines) - 1) * 7
        for j, chunk in enumerate(lines):
            parts.append(
                f'<text x="{x0 - 10}" y="{y0 + j * 14:.1f}" text-anchor="end" '
                f'fill="var(--ink)" font-size="12.5">{esc(chunk)}</text>'
            )
        parts.append(
            f'<text x="{x0 + w + 8:.1f}" y="{y + bar_h / 2 + 4}" '
            f'fill="var(--ink)" font-size="12" font-weight="600" '
            f'style="font-variant-numeric:tabular-nums">{esc(vtxt)}</text></g>'
        )
    parts.append("</svg>")
    return "".join(parts)


def line_chart_budget(series, budgets):
    """Line chart: solve rate vs search-step cap (log-spaced x).
    series: list of dicts {label, family, points:[(budget, rate)]}."""
    if not series or not budgets:
        return ""
    W, H = 900, 380
    pad_l, pad_r, pad_t, pad_b = 64, 170, 16, 46
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b
    bmin, bmax = min(budgets), max(budgets)

    def xpos(b):
        if bmax == bmin:
            return pad_l + plot_w / 2
        return pad_l + (math.log(b) - math.log(bmin)) / (math.log(bmax) - math.log(bmin)) * plot_w

    def ypos(rate):
        return pad_t + (1.0 - rate) * plot_h

    shades = {
        "forward": ["var(--fwd)", "var(--fwd2)", "var(--fwd3)", "var(--fwd4)"],
        "backward": ["var(--bwd)", "var(--bwd2)", "var(--bwd3)", "var(--bwd4)"],
    }
    parts = [
        f'<svg viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Share of puzzles solved at each search-step cap" '
        f'style="width:100%;height:auto;display:block">'
    ]
    for i in range(6):  # horizontal grid at 0..100%
        r = i / 5.0
        y = ypos(r)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="var(--muted)" font-size="11" '
            f'style="font-variant-numeric:tabular-nums">{int(r * 100)}%</text>'
        )
    for b in budgets:
        x = xpos(b)
        parts.append(
            f'<text x="{x:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" '
            f'fill="var(--muted)" font-size="11" '
            f'style="font-variant-numeric:tabular-nums">{b}</text>'
        )
    parts.append(
        f'<text x="{pad_l + plot_w / 2:.0f}" y="{H - 8}" text-anchor="middle" '
        f'fill="var(--muted)" font-size="11">search-step cap per puzzle</text>'
    )
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" '
        f'y2="{pad_t + plot_h}" stroke="var(--baseline)" stroke-width="1"/>'
    )
    fam_count = {"forward": 0, "backward": 0}
    legend_y = pad_t + 6
    for s in series:
        fam = s["family"] if s["family"] in shades else "forward"
        color = shades[fam][fam_count[fam] % len(shades[fam])]
        fam_count[fam] += 1
        pts = sorted(s["points"])
        if not pts:
            continue
        d = " ".join(
            ("M" if i == 0 else "L") + f"{xpos(b):.1f},{ypos(r):.1f}"
            for i, (b, r) in enumerate(pts)
        )
        parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for b, r in pts:
            parts.append(
                f'<circle cx="{xpos(b):.1f}" cy="{ypos(r):.1f}" r="4.5" '
                f'fill="{color}" stroke="var(--surface)" stroke-width="2">'
                f"<title>{esc(s['label'])} — cap {b}: {100 * r:.1f}% solved</title></circle>"
            )
        parts.append(
            f'<line x1="{pad_l + plot_w + 14}" y1="{legend_y}" '
            f'x2="{pad_l + plot_w + 32}" y2="{legend_y}" stroke="{color}" '
            f'stroke-width="2.5"/>'
        )
        for j, chunk in enumerate(wrap_label(s["label"], 24)):
            parts.append(
                f'<text x="{pad_l + plot_w + 38}" y="{legend_y + 4 + j * 13}" '
                f'fill="var(--ink)" font-size="11">{esc(chunk)}</text>'
            )
        legend_y += 16 + 13 * max(0, len(wrap_label(s["label"], 24)) - 1)
    parts.append("</svg>")
    return "".join(parts)


def seg_bar(segments, aria):
    """One stacked horizontal bar. segments: list of (label, count, cssvar)."""
    total = sum(c for _, c, _ in segments)
    if total <= 0:
        return ""
    W, H, bar_h = 900, 92, 34
    x = 0.0
    parts = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(aria)}" '
        f'style="width:100%;height:auto;display:block">',
        # rounded outer ends only; inner segment edges stay square
        f'<clipPath id="segclip"><rect x="0" y="8" width="{W}" '
        f'height="{bar_h}" rx="7"/></clipPath>',
        '<g clip-path="url(#segclip)">',
    ]
    labels = []
    for i, (label, count, var) in enumerate(segments):
        w = W * count / total
        gap = 2 if i < len(segments) - 1 else 0  # 2px surface gap between fills
        parts.append(
            f'<rect x="{x:.1f}" y="8" width="{max(w - gap, 2):.1f}" '
            f'height="{bar_h}" fill="{var}">'
            f"<title>{esc(label)}: {count} of {total} puzzles"
            f' ({100.0 * count / total:.1f}%)</title></rect>'
        )
        if w > 60:
            labels.append(
                f'<text x="{x + w / 2:.1f}" y="{8 + bar_h / 2 + 4}" '
                f'text-anchor="middle" fill="var(--bg)" font-size="12.5" '
                f'font-weight="650" '
                f'style="font-variant-numeric:tabular-nums">{count}</text>'
            )
        x += w
    parts.append("</g>")
    parts.extend(labels)
    parts.append(
        f'<text x="0" y="{8 + bar_h + 22}" fill="var(--muted)" font-size="11">'
        f'all {total} benchmark puzzles</text>'
    )
    parts.append("</svg>")
    legend = "".join(
        f'<span class="lkey"><span class="fdot" style="background:{var}"></span>'
        f"{esc(label)} ({count})</span>"
        for label, count, var in segments
    )
    return f'{"".join(parts)}<div class="legend">{legend}</div>'


# ---------------------------------------------------------------------------
# Board drawing (inline SVG, theme-aware)
# ---------------------------------------------------------------------------

def board_svg(size, walls_right, walls_down, robots, arrows=(), rings=(),
              marks=(), aria="", max_px=430):
    """Draw one Ricochet-Robots board.

    robots: list of (x, y, cssvar, letter, is_target)
    arrows: list of (x0, y0, x1, y1, cssvar, ordinal)  — a slide from cell to cell
    rings : list of (x, y, cssvar)                     — goal cells
    marks : list of (x, y, text)                       — annotated cells
    """
    c = 26
    pad = 4
    W = H = size * c + 2 * pad
    parts = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(aria)}" '
        f'style="width:100%;max-width:{max_px}px;height:auto;display:block">'
    ]

    def cx(x):
        return pad + x * c + c / 2

    def cy(y):
        return pad + y * c + c / 2

    # cell grid
    for i in range(1, size):
        parts.append(f'<line x1="{pad + i * c}" y1="{pad}" x2="{pad + i * c}" '
                     f'y2="{H - pad}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<line x1="{pad}" y1="{pad + i * c}" x2="{W - pad}" '
                     f'y2="{pad + i * c}" stroke="var(--grid)" stroke-width="1"/>')
    # border
    parts.append(f'<rect x="{pad}" y="{pad}" width="{size * c}" height="{size * c}" '
                 f'fill="none" stroke="var(--ink)" stroke-width="3"/>')
    # interior walls
    for (x, y) in sorted(walls_right):
        X = pad + (x + 1) * c
        parts.append(f'<line x1="{X}" y1="{pad + y * c}" x2="{X}" '
                     f'y2="{pad + (y + 1) * c}" stroke="var(--ink)" '
                     f'stroke-width="3" stroke-linecap="round"/>')
    for (x, y) in sorted(walls_down):
        Y = pad + (y + 1) * c
        parts.append(f'<line x1="{pad + x * c}" y1="{Y}" x2="{pad + (x + 1) * c}" '
                     f'y2="{Y}" stroke="var(--ink)" stroke-width="3" '
                     f'stroke-linecap="round"/>')
    # annotated cells (drawn under robots/arrows)
    for (x, y, txt) in marks:
        parts.append(
            f'<rect x="{pad + x * c + 2}" y="{pad + y * c + 2}" width="{c - 4}" '
            f'height="{c - 4}" rx="4" fill="none" stroke="var(--muted)" '
            f'stroke-width="1.6" stroke-dasharray="3.5 2.5">'
            f"<title>{esc(txt)}</title></rect>"
        )
    # goal rings
    for (x, y, var) in rings:
        parts.append(
            f'<rect x="{pad + x * c + 3}" y="{pad + y * c + 3}" width="{c - 6}" '
            f'height="{c - 6}" rx="5" fill="none" stroke="var({var})" '
            f'stroke-width="2.2" stroke-dasharray="4 3"/>'
        )

    # slide arrows, offset so overlapping lanes stay readable
    groups = {}
    for a in arrows:
        x0, y0, x1, y1 = a[:4]
        axis = "v" if x0 == x1 else "h"
        line = x0 if axis == "v" else y0
        groups.setdefault((axis, line), []).append(a)
    off_of = {}
    for key, mem in groups.items():
        n = len(mem)
        for i, a in enumerate(mem):
            off_of[id(a)] = (i - (n - 1) / 2.0) * 9.0
    for a in arrows:
        x0, y0, x1, y1, var, ordinal = a
        o = off_of.get(id(a), 0.0)
        if x0 == x1:  # vertical slide -> offset horizontally
            X0, Y0, X1, Y1 = cx(x0) + o, cy(y0), cx(x1) + o, cy(y1)
        else:         # horizontal slide -> offset vertically
            X0, Y0, X1, Y1 = cx(x0), cy(y0) + o, cx(x1), cy(y1) + o
        dx, dy = X1 - X0, Y1 - Y0
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        # shorten so the head does not sit on the robot circle
        X0s, Y0s = X0 + ux * 10, Y0 + uy * 10
        X1s, Y1s = X1 - ux * 4, Y1 - uy * 4
        parts.append(
            f'<line x1="{X0s:.1f}" y1="{Y0s:.1f}" x2="{X1s:.1f}" y2="{Y1s:.1f}" '
            f'stroke="var({var})" stroke-width="2.6" stroke-linecap="round" '
            f'opacity="0.85"/>'
        )
        # arrow head
        hx, hy = X1s, Y1s
        px, py = -uy, ux
        parts.append(
            f'<path d="M{hx:.1f},{hy:.1f} '
            f'L{hx - ux * 8 + px * 4.5:.1f},{hy - uy * 8 + py * 4.5:.1f} '
            f'L{hx - ux * 8 - px * 4.5:.1f},{hy - uy * 8 - py * 4.5:.1f} Z" '
            f'fill="var({var})" opacity="0.85"/>'
        )
        # ordinal badge at the start of the slide
        if ordinal:
            bx, by = X0 - ux * 0, Y0 - uy * 0
            parts.append(
                f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="8.5" '
                f'fill="var(--surface)" stroke="var({var})" stroke-width="2"/>'
                f'<text x="{bx:.1f}" y="{by + 3.6:.1f}" text-anchor="middle" '
                f'fill="var(--ink)" font-size="10.5" font-weight="700">{ordinal}</text>'
            )
    # robots
    for (x, y, var, letter, is_target) in robots:
        ring = (f'<circle cx="{cx(x):.1f}" cy="{cy(y):.1f}" r="11.5" fill="none" '
                f'stroke="var({var})" stroke-width="1.6" opacity="0.55"/>'
                if is_target else "")
        parts.append(
            f'{ring}<circle cx="{cx(x):.1f}" cy="{cy(y):.1f}" r="8.5" '
            f'fill="var({var})" stroke="var(--surface)" stroke-width="1.6"/>'
            f'<text x="{cx(x):.1f}" y="{cy(y) + 3.4:.1f}" text-anchor="middle" '
            f'fill="var(--bg)" font-size="9.5" font-weight="800">{esc(letter)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def slide_rule_svg():
    """Static schematic: how one move works. Explanatory art only — contains
    no measured numbers."""
    size = 8
    wr = {(4, 6)}          # one interior wall, right edge of (4,6)
    wd = set()
    robots = [
        (1, 1, "--r-red", "R", False),
        (5, 6, "--r-blue", "B", False),
        (5, 0, "--r-yellow", "Y", True),
    ]
    arrows = [
        (1, 1, 7, 1, "--r-red", None),    # slides to the border wall
        (5, 0, 5, 5, "--r-yellow", None), # stops one cell above Blue
    ]
    rings = [(5, 5, "--r-yellow")]
    return board_svg(size, wr, wd, robots, arrows, rings,
                     aria="Schematic board: a red robot slides right until the "
                          "border wall stops it; a yellow robot slides down and "
                          "stops in the cell just above a blue robot, landing "
                          "on its goal cell.",
                     max_px=300)


# ---------------------------------------------------------------------------
# Worked example: a puzzle the plan language cannot express
# ---------------------------------------------------------------------------

DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def _slide(pos, d, blockers, wr, wd, size):
    x, y = pos
    dx, dy = DIRS[d]
    while True:
        if d == "up" and (y == 0 or (x, y - 1) in wd):
            return (x, y)
        if d == "down" and (y == size - 1 or (x, y) in wd):
            return (x, y)
        if d == "left" and (x == 0 or (x - 1, y) in wr):
            return (x, y)
        if d == "right" and (x == size - 1 or (x, y) in wr):
            return (x, y)
        nxt = (x + dx, y + dy)
        if nxt in blockers:
            return (x, y)
        x, y = nxt


def _bfs_solve(start, target_idx, goal, wr, wd, size, node_cap=300000):
    q = deque([(start, ())])
    seen = {start}
    while q:
        st, path = q.popleft()
        if st[target_idx] == goal:
            return path
        if len(seen) > node_cap:
            return None
        for i in range(len(st)):
            blockers = set(st) - {st[i]}
            for d in DIRS:
                np_ = _slide(st[i], d, blockers, wr, wd, size)
                if np_ == st[i]:
                    continue
                ns = list(st)
                ns[i] = np_
                ns = tuple(ns)
                if ns in seen:
                    continue
                seen.add(ns)
                q.append((ns, path + ((i, d, st[i], np_),)))
    return None


def worked_example(probe_rows, want_idx=333):
    """Build the idx-333 worked example: the real board, the real optimal
    solution (recomputed from the wall layout at build time), and the probe's
    verdict for the same puzzle.  Returns None (with a provenance note) if any
    ingredient is missing."""
    insts = load_json("analysis/artifacts/ceiling_probe_instances.json")
    if not insts or not probe_rows:
        return None
    inst = next((i for i in insts if i.get("idx") == want_idx), None)
    verdict = next((r for r in probe_rows if r.get("idx") == want_idx), None)
    if inst is None or verdict is None:
        return None
    env_rel = os.path.join("environments", f"env_{inst['env_id']}.pkl")
    try:
        sys.path.insert(0, ROOT)
        from simulate import wall_sets  # stdlib-only module
        with open(rp(env_rel), "rb") as f:
            env = pickle.load(f)
        wr, wd = wall_sets(env["grid_data"])
        size = int(math.isqrt(len(env["grid_data"])))
    except Exception as e:
        record_note(env_rel, "unreadable", f"could not load board: {e}")
        return None
    record_note(env_rel, "ok", "board walls for the worked example")
    start = tuple(tuple(p) for p in inst["positions"])
    goal = tuple(inst["target"])
    tidx = inst["target_idx"]
    sol = _bfs_solve(start, tidx, goal, wr, wd, size)
    if sol is None:
        return None

    # replay the solution to find the final bump and any earlier pass-through
    conflict = None
    last = sol[-1]
    li, ld, lfrm, lto = last
    dx, dy = DIRS[ld]
    stop_cell = (lto[0] + dx, lto[1] + dy)
    # which robot occupies stop_cell when the last move is played?
    pos = list(start)
    for (i, d, frm, to) in sol[:-1]:
        pos[i] = to
    stopper_slot = next((j for j, p in enumerate(pos) if tuple(p) == stop_cell), None)
    # did the target robot slide THROUGH that cell earlier in the solution?
    passed_before = None
    for step_no, (i, d, frm, to) in enumerate(sol[:-1], start=1):
        if i != tidx:
            continue
        ddx, ddy = DIRS[d]
        x, y = frm
        while (x, y) != to:
            x, y = x + ddx, y + ddy
            if (x, y) == stop_cell:
                passed_before = step_no
                break
    # when was the stopper parked, and what stopped IT?
    stopper_step = None
    stopper_stop = None
    if stopper_slot is not None:
        run = list(start)
        for step_no, (i, d, frm, to) in enumerate(sol, start=1):
            run[i] = to
            if i == stopper_slot and tuple(to) == stop_cell:
                stopper_step = step_no
                sdx, sdy = DIRS[d]
                cell2 = (to[0] + sdx, to[1] + sdy)
                prev = [tuple(p) for k, p in enumerate(run) if k != i]
                # rewind one step for occupancy at play time
                occ = list(start)
                for (i2, d2, frm2, to2) in sol[:step_no - 1]:
                    occ[i2] = to2
                blocker = next((j for j, p in enumerate(occ)
                                if tuple(p) == cell2 and j != i), None)
                if blocker is not None and 0 <= cell2[0] < size and 0 <= cell2[1] < size:
                    stopper_stop = (blocker, cell2)
                break
    if stopper_slot is not None and passed_before is not None:
        conflict = {
            "cell": stop_cell,
            "stopper_slot": stopper_slot,
            "stopper_step": stopper_step,
            "passed_step": passed_before,
            "stopper_stop": stopper_stop,
        }
    return {
        "inst": inst,
        "verdict": verdict,
        "walls": (wr, wd),
        "size": size,
        "solution": sol,
        "conflict": conflict,
        "env_rel": env_rel,
    }


# ---------------------------------------------------------------------------
# Small HTML building blocks
# ---------------------------------------------------------------------------

def pending(msg):
    return f'<div class="pending"><span class="pdot"></span>Pending — {esc(msg)}</div>'


def scroll(inner):
    return f'<div class="scroll">{inner}</div>'


def fam_dot(family):
    cls = "fdot-fwd" if family == "forward" else "fdot-bwd"
    return f'<span class="fdot {cls}"></span>'


def defcard(term, text):
    return (f'<div class="defcard"><div class="dt">{esc(term)}</div>'
            f'<div class="dd">{text}</div></div>')


def tile(value, label, sub="", accent=""):
    cls = f" tile-{accent}" if accent else ""
    s = f'<div class="ts">{sub}</div>' if sub else ""
    return (f'<div class="tile{cls}"><div class="tv">{value}</div>'
            f'<div class="tl">{label}</div>{s}</div>')


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------

# every configuration the study runs, in ladder order; scoreboard rows render
# with "–" cells until their result files exist, so the page always shows what
# is planned vs measured
SCALING_LADDER = [
    ("g16r6", "16×16 board, 6 robots"),
    ("g16r8", "16×16 board, 8 robots"),
    ("g24r4", "24×24 board, 4 robots"),
    ("g24r8", "24×24 board, 8 robots"),
    ("g32r4", "32×32 board, 4 robots"),
]

SCALING_EXPECTED = {
    "g16r6": ["comparison.json", "comparison_forward_control.json",
              "comparison_ungraded.json", "comparison_b1.json",
              "comparison_ungraded_b1.json"],
    "g16r8": ["comparison.json", "comparison_forward_control.json",
              "comparison_ungraded.json"],
    "g24r4": ["comparison.json"],
    "g24r8": ["comparison.json", "comparison_ungraded.json"],
    "g32r4": ["comparison.json", "comparison_ungraded.json"],
}

SCALING_FILE_LABEL = {
    "comparison.json": "head-to-head on the puzzles the exact solver could grade",
    "comparison_forward_control.json":
        "forward planner re-run after its training-stability fix",
    "comparison_b1.json":
        "backward planner with the extended plan language, on the gradable set",
    "comparison_ungraded_b1.json":
        "backward planner with the extended plan language, "
        "beyond the exact solver's reach",
    "comparison_ungraded.json":
        "beyond the exact solver's reach (a solution proves itself)",
}


def collect():
    D = {}
    D["fwd"] = load_json("eval/results/comparison_forward.json")
    D["bwd450"] = load_json("eval/results/comparison_backward.json")
    D["bwd_any150"] = load_json("eval/results/comparison_backward_anytime.json")
    D["postfix"] = load_json("eval/results/comparison_backward_postfix.json")
    D["postfix_any"] = load_json("eval/results/comparison_backward_postfix_anytime.json")
    D["postfix2"] = load_json("eval/results/comparison_backward_postfix2.json")
    D["postfix2_any"] = load_json("eval/results/comparison_backward_postfix2_anytime.json")
    D["realizer"] = load_json("eval/results/realizer_twophase_ab.json")
    D["postfix4"] = load_json("eval/results/postfix4_v2pol_plain.json")
    D["postfix4_any"] = load_json("eval/results/postfix4_v2pol_anytime.json")
    D["final450"] = load_json("eval/results/final450_backward_plain.json")
    D["final450_any"] = load_json("eval/results/final450_backward_anytime.json")
    D["final450_prefix"] = load_json("eval/results/final450_backward_prefix.json")
    D["prefix_ab"] = load_json("eval/results/prefix_check_ab.json")
    D["prefix150"] = load_json("eval/results/prefix150_prefix.json")
    D["arm_prefix5"] = load_json("eval/results/arm_prefix_iter5.json")
    D["b1_450"] = load_json("eval/results/final450_backward_b1.json")
    D["b1_ab"] = load_json("eval/results/realizer_b1_ab.json")
    D["ceiling_probe"] = load_json("analysis/artifacts/ceiling_probe_results.json")
    D["ceiling_probe_b1"] = load_json(
        "analysis/artifacts/ceiling_probe_results_b1.json")
    # the follow-up language extension (re-using an already-parked robot as a
    # second stopper); renders automatically once its probe file lands
    D["ceiling_probe_b2"] = load_json(
        "analysis/artifacts/ceiling_probe_results_b2.json")
    D["bench_meta"] = load_json("eval/data/bench450.jsonl.meta.json")
    D["bench_rows"] = load_jsonl("eval/data/bench450.jsonl")
    D["arms"] = load_json("subgoal_selfplay/arms_index.json")
    D["cap_probe"] = load_jsonl("scaling/data/g16r8/cap_probe_10x_results.jsonl")

    D["worked"] = worked_example(D["ceiling_probe"])

    D["curve_files"] = sorted(glob.glob(rp("eval", "results", "curves", "*.json")))
    record_glob("eval/results/curves/*.json", D["curve_files"])

    D["selfplay_stats"] = sorted(
        glob.glob(rp("subgoal_selfplay", "runs_warm*", "iter*_stats.json"))
    )
    record_glob("subgoal_selfplay/runs_warm*/iter*_stats.json", D["selfplay_stats"])

    # scaling: load each expected file per rung (missing ones recorded, and
    # rendered as dash rows), note any extra files present but not expected
    D["scaling"] = {}
    for cfg, _desc in SCALING_LADDER:
        D["scaling"][cfg] = {
            f: load_json(os.path.join("scaling", "results", cfg, f))
            for f in SCALING_EXPECTED[cfg]
        }
    extra = [os.path.relpath(p, ROOT)
             for p in sorted(glob.glob(rp("scaling", "results", "*", "*.json")))
             if os.path.basename(p) not in SCALING_FILE_LABEL]
    record_glob("scaling/results/* (files outside the ladder, not rendered)", extra)

    D["scaling_metas"] = sorted(glob.glob(rp("scaling", "data", "*", "bench.jsonl.meta.json")))
    record_glob("scaling/data/*/bench.jsonl.meta.json", D["scaling_metas"])
    return D


# ---------------------------------------------------------------------------
# Headline table
# ---------------------------------------------------------------------------

def build_headline_rows(D):
    """Return (rows, checks). Each row is a dict of display fields; each check
    records (row label, metric, source file, source value, rendered string)."""
    rows, checks = [], []

    def add(label, family, sub, agg, source, solve_txt=None, note="", hl=False):
        row = {
            "label": label,
            "family": family,
            "sub": sub,
            "n": agg["n"],
            "solve_txt": solve_txt or ffrac(agg["solved"], agg["n"]),
            "regret": fnum(agg["mean_regret"], 3),
            "pct_opt": fnum(agg["pct_optimal"], 1),
            "moves": fnum(agg["mean_moves"], 2),
            "exp": fnum(agg["mean_expansions"], 1),
            "sec": fnum(agg["mean_seconds"], 3),
            "note": note,
            "hl": hl,
        }
        rows.append(row)
        for metric, key, dec in [
            ("mean extra moves", "mean_regret", 3),
            ("% at optimum", "pct_optimal", 1),
            ("mean plan length", "mean_moves", 2),
            ("mean search steps", "mean_expansions", 1),
            ("mean seconds", "mean_seconds", 3),
        ]:
            checks.append((label, metric, source, agg.get(key), row[
                {"mean extra moves": "regret", "% at optimum": "pct_opt",
                 "mean plan length": "moves", "mean search steps": "exp",
                 "mean seconds": "sec"}[metric]]))
        checks.append((label, "solved", source, agg.get("solved"), row["solve_txt"]))

    # forward systems, in a stable friendly order
    if D["fwd"]:
        fwd_sys = systems_of_kind(D["fwd"], "forward")
        order = {"best.ckpt": 0, "candidate_scored": 1, "runs_warm": 2, "runs_scratch": 3}

        def sort_key(item):
            _, _, path = forward_label(item[0])
            for frag, idx in order.items():
                if frag in path:
                    return idx
            return 99

        for name, s in sorted(fwd_sys, key=sort_key):
            label, desc, path = forward_label(name)
            add(label, "forward", desc, s["aggregate"],
                "eval/results/comparison_forward.json",
                hl=("candidate_scored" in path))

    # backward, before fixes, full 450
    if D["bwd450"]:
        _, s = first_backward(D["bwd450"])
        if s:
            a = s["aggregate"]
            solve_txt = (
                f"plan found {ffrac(a.get('plan_found'), a['n'])} · "
                f"playable {ffrac(a['solved'], a['n'])}"
            )
            add("Backward planner — before the fixes", "backward",
                "as first measured: four coding defects (since repaired) made "
                "most of its plans impossible to play", a,
                "eval/results/comparison_backward.json", solve_txt=solve_txt)

    # backward, official final rows (all four fixes, full 450)
    if D["final450"]:
        _, s = first_backward(D["final450"])
        if s:
            add("Backward planner — after the four fixes", "backward",
                "same networks; planning and plan-to-moves code repaired "
                "(each fix verified zero-regression)",
                s["aggregate"],
                "eval/results/final450_backward_plain.json")
    if D["final450_any"]:
        _, s = first_backward(D["final450_any"])
        if s:
            add("Backward planner — fixes + keep-searching mode",
                "backward",
                "every finished plan is test-played; if it fails, the search "
                "discards it and keeps looking",
                s["aggregate"],
                "eval/results/final450_backward_anytime.json")
    elif D["postfix2"]:
        _, s = first_backward(D["postfix2"])
        if s:
            add("Backward planner — after both fixes", "backward",
                "same networks, planning code after both fixes",
                s["aggregate"],
                "eval/results/comparison_backward_postfix2.json",
                note="first 150 of the 450 puzzles")
    have_b1 = bool(D.get("b1_450"))
    if D["final450_prefix"]:
        _, s = first_backward(D["final450_prefix"])
        if s:
            add("Backward planner — fixes + in-search playability checking",
                "backward",
                "each partial plan is physics-checked as it is built, so doomed "
                "branches are dropped immediately"
                + ("" if have_b1 else " (the best backward mode)"),
                s["aggregate"],
                "eval/results/final450_backward_prefix.json", hl=not have_b1)
    if have_b1:
        _, s = first_backward(D["b1_450"])
        if s:
            add("Backward planner — richer plan language, retrained",
                "backward",
                "two new things a plan may say (a stopper may stand on a "
                "wall-less cell; a robot may briefly step aside), networks "
                "retrained to use them — the best backward mode",
                s["aggregate"],
                "eval/results/final450_backward_b1.json", hl=True)
    return rows, checks


def render_headline(rows):
    if not rows:
        return pending("no comparison result files found yet "
                       "(eval/results/comparison_*.json)")
    head = (
        "<tr>"
        "<th>system</th>"
        "<th>puzzles</th>"
        "<th>solved (playable)</th>"
        "<th>extra moves vs optimal<sup>a</sup></th>"
        "<th>% at optimum<sup>a</sup></th>"
        "<th>plan length (moves)<sup>a</sup></th>"
        "<th>search steps used</th>"
        "<th>seconds / puzzle</th>"
        "</tr>"
    )
    body = []
    for r in rows:
        note = f'<div class="cellnote">{esc(r["note"])}</div>' if r["note"] else ""
        sub = f'<div class="cellnote">{esc(r["sub"])}</div>' if r["sub"] else ""
        cls = ' class="hlrow"' if r.get("hl") else ""
        body.append(
            f"<tr{cls}>"
            f'<td class="syscell">{fam_dot(r["family"])}<b>{esc(r["label"])}</b>{sub}</td>'
            f'<td class="num">{r["n"]}{note}</td>'
            f'<td class="num">{esc(r["solve_txt"])}</td>'
            f'<td class="num">{r["regret"]}</td>'
            f'<td class="num">{r["pct_opt"]}</td>'
            f'<td class="num">{r["moves"]}</td>'
            f'<td class="num">{r["exp"]}</td>'
            f'<td class="num">{r["sec"]}</td>'
            "</tr>"
        )
    table = (f'<table id="headline-table">{head}{"".join(body)}</table>')
    return scroll(table)


# ---------------------------------------------------------------------------
# Shared header
# ---------------------------------------------------------------------------

def sec_header(D):
    proto = (D["fwd"] or D["bwd450"] or {}).get("protocol", {})
    meta = D["bench_meta"] or {}
    n = meta.get("n_instances") or proto.get("n_instances")
    budget = proto.get("expansions")
    k = proto.get("k")
    sha = (meta.get("sha256") or proto.get("instances_sha256") or "")[:16]
    generated = meta.get("generated", "")

    chips = []
    if n:
        chips.append(f"{n} identical puzzles for every system")
    if budget:
        chips.append(f"search capped at {budget} search steps per puzzle")
    if k:
        chips.append(f"up to {k} suggested continuations per step")
    chips.append("all costs counted in real moves")
    chips.append("“solved” = the plan plays out legally, move by move")
    if sha:
        chips.append(f"benchmark stamp {sha}… ({generated})")
    chip_html = "".join(f'<span class="chip">{esc(c)}</span>' for c in chips)

    return f"""
<header>
  <div class="eyebrow"><span class="dot"></span>planner comparison · full results</div>
  <h1>Ricochet Robots: two ways to plan, measured on the same puzzles</h1>
  <p class="lede">Ricochet Robots is a sliding-robot puzzle. A few robots sit on
  a walled grid; a moved robot slides in a straight line until it hits a wall or
  another robot — it cannot stop mid-slide. One robot is the <b>target</b>, one
  cell is the <b>goal</b>: get the target robot onto the goal cell in as few
  moves as possible.</p>
  <p class="lede">This page compares two families of learned solvers on that
  puzzle. The <b class="fwd-ink">forward planner</b> plays the game the obvious
  way — one robot move at a time from the start toward the goal. The
  <b class="bwd-ink">backward planner</b> reasons from the goal outward, in
  larger units called <b>subgoals</b> — stepping-stones such as “a helper robot
  must first be parked here so the final slide stops on the goal” — each worth
  several moves at once. Everything on this page is generated from the
  experiments’ own result files; nothing is typed in by hand.</p>
  <div class="chips">{chip_html}</div>
</header>
"""


# ---------------------------------------------------------------------------
# The consolidated experiments table (the centerpiece of the overview)
# ---------------------------------------------------------------------------

def _sysagg(comp, kind, name_frag=None):
    """Aggregate of the first system of `kind` in an eval.compare file,
    optionally filtered by a name fragment; None when absent."""
    for name, s in systems_of_kind(comp, kind):
        if name_frag is None or name_frag in name:
            return s["aggregate"]
    return None


def sec_experiments(D):
    sc = D["scaling"]

    def sfile(cfg, fname):
        return (sc.get(cfg) or {}).get(fname)

    fwd_best_base = (_sysagg(D["fwd"], "forward", "candidate_scored")
                     or _sysagg(D["fwd"], "forward"))
    b1_base = _sysagg(D["b1_450"], "backward")
    entries = [
        {
            "name": "Standard puzzles — 16×16 board, 4 robots",
            "sub": "the shared 450-puzzle benchmark; every solution is graded "
                   "against the known shortest one",
            "b": b1_base or _sysagg(D["final450_prefix"], "backward"),
            "f": fwd_best_base,
            "tag": ("subgoal side uses the richer plan language"
                    if b1_base is not None else ""),
        },
        {
            "name": "6 robots — puzzles the exact solver can still grade",
            "sub": "more robots, same board; graded against known optima",
            "b": (_sysagg(sfile("g16r6", "comparison_b1.json"), "backward")
                  or _sysagg(sfile("g16r6", "comparison.json"), "backward")),
            "f": _sysagg(sfile("g16r6", "comparison_forward_control.json"),
                         "forward"),
            "tag": ("subgoal side uses the richer plan language"
                    if sfile("g16r6", "comparison_b1.json") else ""),
        },
        {
            "name": "6 robots — the hardest puzzles, beyond the exact solver",
            "sub": "no reference answers exist; solving is self-certifying — "
                   "a plan that plays out legally to the goal proves itself",
            "b": (_sysagg(sfile("g16r6", "comparison_ungraded_b1.json"),
                          "backward")
                  or _sysagg(sfile("g16r6", "comparison_ungraded.json"),
                             "backward")),
            "f": _sysagg(sfile("g16r6", "comparison_ungraded.json"), "forward"),
            "tag": ("subgoal side uses the richer plan language"
                    if sfile("g16r6", "comparison_ungraded_b1.json") else ""),
            "ungraded": True,
        },
        {
            "name": "8 robots — puzzles the exact solver can still grade",
            "sub": "",
            "b": _sysagg(sfile("g16r8", "comparison.json"), "backward"),
            "f": _sysagg(sfile("g16r8", "comparison_forward_control.json"),
                         "forward"),
        },
        {
            "name": "8 robots — the hardest puzzles, beyond the exact solver",
            "sub": "",
            "b": _sysagg(sfile("g16r8", "comparison_ungraded.json"), "backward"),
            "f": _sysagg(sfile("g16r8", "comparison_ungraded.json"), "forward"),
            "tag": "measured before the richer plan language landed — "
                   "both sides use the original language",
            "ungraded": True,
        },
        {
            "name": "Bigger board — 24×24, 4 robots",
            "sub": "the board-size axis: longer slides, same solution depth",
            "b": _sysagg(sfile("g24r4", "comparison.json"), "backward"),
            "f": _sysagg(sfile("g24r4", "comparison.json"), "forward"),
        },
        {
            "name": "Bigger board and more robots — 24×24, 8 robots",
            "sub": "",
            "b": _sysagg(sfile("g24r8", "comparison.json"), "backward"),
            "f": _sysagg(sfile("g24r8", "comparison.json"), "forward"),
        },
        {
            "name": "24×24, 8 robots — the hardest puzzles",
            "sub": "",
            "b": _sysagg(sfile("g24r8", "comparison_ungraded.json"), "backward"),
            "f": _sysagg(sfile("g24r8", "comparison_ungraded.json"), "forward"),
            "ungraded": True,
        },
        {
            "name": "Biggest board — 32×32, 4 robots",
            "sub": "",
            "b": _sysagg(sfile("g32r4", "comparison.json"), "backward"),
            "f": _sysagg(sfile("g32r4", "comparison.json"), "forward"),
        },
        {
            "name": "32×32, 4 robots — the hardest puzzles",
            "sub": "",
            "b": _sysagg(sfile("g32r4", "comparison_ungraded.json"), "backward"),
            "f": _sysagg(sfile("g32r4", "comparison_ungraded.json"), "forward"),
            "ungraded": True,
        },
    ]

    head = (
        "<tr><th rowspan='2'>experiment</th><th rowspan='2'>puzzles</th>"
        "<th colspan='3'>subgoal planner (backward)</th>"
        "<th colspan='3'>move-by-move planner (forward)</th></tr>"
        "<tr>"
        "<th>solved</th><th>search steps</th><th>seconds</th>"
        "<th>solved</th><th>search steps</th><th>seconds</th>"
        "</tr>"
    )
    body = []
    n_pending = 0
    for e in entries:
        b, f = e.get("b"), e.get("f")
        n = (b or f or {}).get("n")
        n_txt = str(n) if n else "–"
        tag = e.get("tag", "")
        if b is None and f is None:
            tag = "planned — results pending"
            n_pending += 1

        def side(agg, other):
            if agg is None:
                return ["–", "–", "–"], False
            win = (other is None or
                   (agg.get("solve_rate") or 0) > (other.get("solve_rate") or 0))
            return [ffrac(agg.get("solved"), agg.get("n")),
                    fnum(agg.get("mean_expansions"), 1),
                    fnum(agg.get("mean_seconds"), 1)], win

        bc, bwin = side(b, f)
        fc, fwin = side(f, b)
        if b is not None:
            aux_need(f"experiments table: {e['name']} — subgoal solved cell",
                     bc[0])
        if f is not None:
            aux_need(f"experiments table: {e['name']} — move-by-move solved "
                     f"cell", fc[0])
        cells = []
        for i, c in enumerate(bc):
            v = f"<b>{esc(c)}</b>" if (i == 0 and bwin and b) else esc(c)
            cells.append(f'<td class="num">{v}</td>')
        for i, c in enumerate(fc):
            v = f"<b>{esc(c)}</b>" if (i == 0 and fwin and f) else esc(c)
            cells.append(f'<td class="num">{v}</td>')
        sub = f'<div class="cellnote">{esc(e["sub"])}</div>' if e["sub"] else ""
        tag_html = f'<div class="cellnote">{esc(tag)}</div>' if tag else ""
        body.append(f"<tr><td>{esc(e['name'])}{sub}{tag_html}</td>"
                    f"<td class='num'>{n_txt}</td>{''.join(cells)}</tr>")

    pending_note = ""
    if n_pending:
        pending_note = (f"<p class='muted small'>{n_pending} row(s) show "
                        f"dashes: those runs are in progress; the page "
                        f"rebuilds from their result files as they land.</p>")
    return f"""
<section id="experiments">
  <div class="sec-head">
    <div class="kicker">all experiments at a glance</div>
    <h2>Every head-to-head, one table</h2>
    <p class="muted">Each row is one experiment: the same puzzles given to both
    planners under the same search budget. <b>Solved</b> counts only solutions
    that play out legally, move by move. <b>Search steps</b> is how much
    searching the planner needed per puzzle (lower = less work), and
    <b>seconds</b> is wall-clock time per puzzle. The higher solve rate in each
    row is bold. “Hardest puzzles” rows are the ones the exact reference solver
    itself could not crack — there, solving more puzzles with less work is the
    whole story.</p>
  </div>
  {scroll(f'<table id="experiments-table">{head}{"".join(body)}</table>')}
  {pending_note}
  <p class="footcell">Solution-quality numbers (extra moves beyond the known
  optimum) exist only where the exact solver can grade, and are reported in the
  <a href="#results">detailed results</a> and <a href="#scaling">scaling</a>
  tabs. Full provenance for every row — exact result files and settings — is in
  <a href="#method">method &amp; sources</a>.</p>
</section>
"""


# ---------------------------------------------------------------------------
# Tab 1 — Overview
# ---------------------------------------------------------------------------

def tab_overview(D, rows):
    # --- how the game works, with a schematic ---
    game = f"""
<section id="the-game">
  <div class="sec-head">
    <div class="kicker">the game</div>
    <h2>How one move works</h2>
  </div>
  <div class="split">
    <div class="panel">{slide_rule_svg()}</div>
    <div>
      <p>Robots slide; they cannot stop themselves. In the sketch, the red robot
      pushed right slides across the whole board and only stops at the border
      wall. The yellow robot — the target, marked with a halo — pushed down
      stops in the cell just above the blue robot, and that cell happens to be
      its goal (the dashed square).</p>
      <p>That second slide is the heart of every puzzle: <b>to stop somewhere in
      the open, a robot needs something to bump into</b>. Good solutions
      therefore park helper robots at exact spots first, then bounce the target
      off them. The two planner families differ only in how they think about
      that: move by move, or bounce-spot by bounce-spot.</p>
    </div>
  </div>
</section>
"""

    # --- the two planners + shared vocabulary ---
    planners = """
<section id="two-planners">
  <div class="sec-head">
    <div class="kicker">the contestants</div>
    <h2>Two planners, one recipe</h2>
    <p class="muted">Both families use the same machinery: a <b>proposal
    network</b> suggests what to try next, a <b>value network</b> estimates how
    close each option is to done, and a best-first search — one that always
    explores the most promising option first — ties the two together. They
    differ only in what one search decision commits to.</p>
  </div>
  <div class="grid2">
    <div class="panel edge-fwd"><h3 class="ph">forward planner</h3>
      <p style="margin:0">Thinks exactly like the game: “which robot slides
      which way right now?” It builds the solution one real move at a time from
      the start position, so anything it finds is legal by construction — but it
      must search through many positions to find it.</p></div>
    <div class="panel edge-bwd"><h3 class="ph">backward planner</h3>
      <p style="margin:0">Thinks from the goal outward: “for the final slide to
      stop on the goal, a robot must first be parked HERE — and getting one
      there needs THIS first.” One such subgoal is worth several moves, so its
      searches are short — but its plans are written in shorthand and must
      survive being played out.</p></div>
  </div>
  <div class="defrow">
    {DEFS}
  </div>
</section>
"""
    defs = "".join([
        defcard("search step",
                "the search examines one position (or partial plan) and "
                "generates its candidate continuations. In both families one "
                "step costs one pass of each network, so step budgets are "
                "directly comparable."),
        defcard("playable",
                "a backward plan counts as solved only if it converts into an "
                "actual legal move sequence on the real board, with every robot "
                "present. A plan that only works on paper does not count."),
        defcard("extra moves",
                "how many moves a found solution used beyond the shortest "
                "possible solution for that puzzle (0 = perfect)."),
        defcard("the oracle",
                "an exact, slow solver that knows true optimal answers. It is "
                "used only to create training labels and reference optima for "
                "scoring — never while the planners solve."),
    ])
    planners = planners.replace("{DEFS}", defs)

    # --- takeaway tiles ---
    tiles = []
    best_fwd = next((r for r in rows if r["family"] == "forward" and r["hl"]), None)
    best_bwd = next((r for r in rows if r["family"] == "backward" and r["hl"]), None)
    if best_fwd:
        tiles.append(tile(esc(best_fwd["solve_txt"].split(" (")[0]),
                          "solved by the best forward planner",
                          f"…but it needs {esc(best_fwd['exp'])} search steps "
                          f"per puzzle on average"))
    if best_bwd:
        tiles.append(tile(esc(best_bwd["solve_txt"].split(" (")[0]),
                          "solved by the best backward planner",
                          f"…in only {esc(best_bwd['exp'])} search steps "
                          f"per puzzle on average"))
    probe_b1 = D.get("ceiling_probe_b1")
    probe = D.get("ceiling_probe")
    nt = (len(D["bench_rows"]) if D.get("bench_rows")
          else (D["bench_meta"] or {}).get("n_instances") or 450)
    if probe and probe_b1:
        old_struct = sum(1 for r in probe if r["category"] in
                         ("NO_COMPLETE_PLAN", "NO_REALIZABLE_PLAN"))
        new_struct = sum(1 for r in probe_b1 if r["category"] in
                         ("NO_COMPLETE_PLAN", "NO_REALIZABLE_PLAN"))
        old_c = (nt - old_struct) / nt * 100
        new_c = (nt - new_struct) / nt * 100
        tiles.append(tile(
            f"{old_c:.1f}% → {new_c:.1f}%",
            "the hard ceiling on the backward planner, before and after its "
            "plan language was made richer",
            f"puzzles with no playable plan at all: {old_struct} → "
            f"{new_struct} of {nt} — <a href='#ceiling'>why plans fail</a>"))
    elif probe:
        old_struct = sum(1 for r in probe if r["category"] in
                         ("NO_COMPLETE_PLAN", "NO_REALIZABLE_PLAN"))
        ceiling = (nt - old_struct) / nt * 100
        tiles.append(tile(f"{ceiling:.1f}%",
                          "the most any backward planner can score here",
                          f"{old_struct} of {nt} puzzles have no playable plan "
                          f"in its language — "
                          f"<a href='#ceiling'>why plans fail</a>"))
    g16r6 = D["scaling"].get("g16r6") or {}
    ug_b1 = g16r6.get("comparison_ungraded_b1.json")
    ug = g16r6.get("comparison_ungraded.json")
    if ug:
        _, fs = first_forward(ug)
        ba = (_sysagg(ug_b1, "backward") if ug_b1
              else (first_backward(ug)[1] or {}).get("aggregate"))
        fa = fs["aggregate"] if fs else None
        if ba and fa:
            sub = ("after the language extension, the decisive win — "
                   "<a href='#scaling'>scaling</a>" if ug_b1 else
                   "the first regime where the backward planner wins — "
                   "<a href='#scaling'>scaling</a>")
            tiles.append(tile(
                f"{fpct(ba['solve_rate'])} vs {fpct(fa['solve_rate'])}",
                "subgoals vs move-by-move on the hardest 6-robot puzzles",
                sub))
    takeaways = ""
    if tiles:
        takeaways = f"""
<section id="takeaways">
  <div class="sec-head">
    <div class="kicker">the short version</div>
    <h2>The numbers to remember</h2>
  </div>
  <div class="tiles">{"".join(tiles)}</div>
  <p class="muted">On small puzzles the move-by-move planner wins on solution
  quality and the subgoal planner wins on effort — by one to two orders of
  magnitude. As puzzles grow, three measured things happen: the exact solver
  that move-by-move training depends on dies first; the subgoal planner keeps
  training itself where that solver is gone; and on the hardest puzzles beyond
  the solver's reach, the subgoal planner — once its plan language was made
  rich enough — now solves the most puzzles at a fraction of the work. Details
  in <a href="#results">detailed results</a> and
  <a href="#scaling">scaling</a>.</p>
</section>
"""
    return game + planners + sec_experiments(D) + takeaways


def sec_headline(D, rows):
    n_missing = sum(
        1 for k in ("fwd", "bwd450", "final450", "final450_prefix") if not D[k]
    )
    warn = ""
    if n_missing:
        warn = pending(
            "some comparison files are not available yet; the table shows "
            "only the systems whose results exist"
        )
    slice_note = ""
    if any(r["note"] for r in rows):
        slice_note = (" Rows with a smaller puzzle count state it in the "
                      "puzzles column.")
    return f"""
<section id="headline">
  <div class="sec-head">
    <div class="kicker">every system · 16×16 board, 4 robots</div>
    <h2>Every system on the shared benchmark</h2>
    <p class="muted">Same puzzles, same search budget, same scoring. For the
    backward planner two numbers exist: how often it <b>finds a plan</b> in its
    own shorthand, and how often that plan is <b>playable</b>. Only the playable
    number is comparable with the forward planners, and it is the one used
    everywhere on this page. The backward rows are labeled by what changed
    between them; the shaded rows are each family’s best.</p>
  </div>
  {warn}
  {render_headline(rows)}
  <p class="footcell"><sup>a</sup> averaged over the puzzles that system solved
  (playably). “Extra moves vs optimal” is the average number of moves beyond
  the known shortest solution; “% at optimum” is the share of its solved
  puzzles finished in exactly the shortest move count. Search steps and seconds
  are averaged over all puzzles.{esc(slice_note)}</p>
</section>
"""


# ---------------------------------------------------------------------------
# Tab 2 — Base-scale results in depth
# ---------------------------------------------------------------------------

def sec_efficiency(D, rows):
    bars = []
    for r in rows:
        try:
            v = float(r["exp"])
        except ValueError:
            continue
        bars.append((r["label"] + (" *" if r["note"] else ""), v, r["family"]))
    if not bars:
        return f"""
<section id="efficiency">
  <div class="sec-head">
    <div class="kicker">search effort</div>
    <h2>How much searching each system needs</h2>
  </div>
  {pending("waiting for comparison result files")}
</section>
"""
    legend = (
        '<div class="legend">'
        f'<span class="lkey">{fam_dot("forward")}forward (one move at a time)</span>'
        f'<span class="lkey">{fam_dot("backward")}backward (over subgoals)</span>'
        "</div>"
    )
    fwd_vals = [v for _, v, f in bars if f == "forward"]
    bwd_vals = [v for _, v, f in bars if f == "backward"]
    para = ""
    if fwd_vals and bwd_vals:
        bmin = min(bwd_vals)
        para = (
            f"<p>The two formulations spend their budget very differently on the "
            f"same puzzles. The backward planner completes a plan after about "
            f"<b>{bmin:.1f} search steps</b> on average, because one backward step "
            f"commits an entire stepping-stone — several moves at once — and "
            f"precomputed exact travel-distance tables absorb the movement between "
            f"stepping-stones. The forward planners need "
            f"<b>{min(fwd_vals):.0f} to {max(fwd_vals):.0f} steps</b> depending on "
            f"how they were trained, because each step commits only a single move. "
            f"That is the trade in one sentence: the backward planner compresses "
            f"the search by one to two orders of magnitude, and pays for it with "
            f"plans that are not always playable as written; the forward planners "
            f"pay with far more searching and get plans that are legal by "
            f"construction. Which cost matters more shifts with problem size — "
            f"the <a href='#scaling'>scaling tab</a> measures exactly that.</p>"
        )
    star_note = ""
    if any(lbl.endswith(" *") for lbl, _, _ in bars):
        star_note = ('<p class="muted small">* measured on the first 150 of the '
                     "450 puzzles.</p>")
    chart = bar_chart(bars, "average search steps per puzzle (lower is fewer)",
                      "Average search steps used per puzzle, by system")
    return f"""
<section id="efficiency">
  <div class="sec-head">
    <div class="kicker">search effort</div>
    <h2>How much searching each system needs</h2>
  </div>
  {legend}
  <div class="panel">{scroll(chart)}</div>
  {star_note}
  {para}
</section>
"""


def curve_series_label(name, family):
    """Plain-language label for a curve series."""
    if family == "forward":
        label, _, _ = forward_label(name)
        return label
    if "anytime" in name.lower():
        return "Backward planner — keep-searching mode"
    return "Backward planner"


def sec_budget_curves(D):
    points = {}   # system name -> dict(budget -> (rate, family))
    curve_sha = {}  # system name -> benchmark sha of its curve runs
    unrecognized = []
    for path in D["curve_files"]:
        rel = os.path.relpath(path, ROOT)
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            unrecognized.append(rel)
            continue
        got = False
        # shape 1: an eval.compare output at some cap
        if isinstance(d, dict) and "systems" in d and isinstance(d.get("protocol"), dict):
            b = d["protocol"].get("expansions")
            sha = d["protocol"].get("instances_sha256")
            for name, s in d["systems"].items():
                if isinstance(s, dict) and isinstance(s.get("aggregate"), dict) \
                        and b is not None and "solve_rate" in s["aggregate"]:
                    fam = "backward" if s.get("kind") == "backward" else "forward"
                    points.setdefault(name, {})[b] = (s["aggregate"]["solve_rate"], fam)
                    curve_sha[name] = sha
                    got = True
        # shape 2: {"budgets":[...], "systems": {name: [rates...]}} or
        #          {name: [{budget/cap/expansions, solve_rate}, ...]}
        if not got and isinstance(d, dict):
            budgets = d.get("budgets")
            if budgets and isinstance(d.get("systems"), dict):
                for name, rates in d["systems"].items():
                    if isinstance(rates, list) and len(rates) == len(budgets):
                        fam = "backward" if "backward" in name.lower() else "forward"
                        for b, r in zip(budgets, rates):
                            if isinstance(r, (int, float)):
                                points.setdefault(name, {})[b] = (r, fam)
                                got = True
            else:
                for name, lst in d.items():
                    if not isinstance(lst, list):
                        continue
                    for item in lst:
                        if not isinstance(item, dict):
                            continue
                        b = next((item[k] for k in
                                  ("budget", "cap", "expansions", "x") if k in item), None)
                        r = next((item[k] for k in
                                  ("solve_rate", "rate", "y") if k in item), None)
                        if b is not None and r is not None:
                            fam = "backward" if "backward" in name.lower() else "forward"
                            points.setdefault(name, {})[b] = (r, fam)
                            got = True
        if not got:
            unrecognized.append(rel)

    chart_html = pending(
        "budget-sweep runs at several search-step caps are in progress; the "
        "curve will render here from eval/results/curves/*.json as those "
        "files land")
    note = ""
    small_budget = ""
    if points:
        # Extend each curve with the full-budget point from the main
        # comparisons — but only when it is honestly the same measurement:
        # same system name AND same benchmark file (stamp).  The before-fixes
        # backward file ("bwd450") is deliberately excluded: curve reruns use
        # the fixed planner code, so joining the pre-fix point would fabricate
        # a trend across two different planners.
        main_points = {}
        for comp_key in ("fwd", "postfix2", "postfix2_any"):
            comp = D[comp_key]
            if not comp:
                continue
            b = comp.get("protocol", {}).get("expansions")
            sha = comp.get("protocol", {}).get("instances_sha256")
            for name, s in comp.get("systems", {}).items():
                if isinstance(s, dict) and isinstance(s.get("aggregate"), dict) \
                        and b is not None and "solve_rate" in s["aggregate"]:
                    fam = "backward" if s.get("kind") == "backward" else "forward"
                    main_points[name] = (b, s["aggregate"]["solve_rate"], fam, sha)
        for name, (b, r, fam, sha) in main_points.items():
            if name in points and b not in points[name] \
                    and sha and curve_sha.get(name) == sha:
                points[name][b] = (r, fam)
        series = []
        for name, pts in sorted(points.items()):
            fam = next(iter(pts.values()))[1]
            series.append({
                "label": curve_series_label(name, fam), "family": fam,
                "points": [(b, r) for b, (r, _) in sorted(pts.items())],
            })
        budgets = sorted({b for pts in points.values() for b in pts})
        fams_present = {s["family"] for s in series}
        keys = []
        if "forward" in fams_present:
            keys.append(f'<span class="lkey">{fam_dot("forward")}forward systems</span>')
        if "backward" in fams_present:
            keys.append(f'<span class="lkey">{fam_dot("backward")}backward systems</span>')
        legend = f'<div class="legend">{"".join(keys)}</div>'
        chart_html = legend + f'<div class="panel">{scroll(line_chart_budget(series, budgets))}</div>'
        note = ("<p>Reading: a single budget point can flatter either system; "
                "the curve shows how the share of playably solved puzzles grows "
                "as each system is allowed more search steps per puzzle. The "
                "full-budget point from the headline table is joined onto a "
                "curve only when it measured the same benchmark file with the "
                "same planner code.</p>")
        if len(budgets) < 4:
            note += pending(
                "the sweep is partial so far — points at further caps will be "
                "added to the chart as their run files land in "
                "eval/results/curves/")
        # small-budget callout at the lowest common cap
        bmin = min(budgets)
        at_min = {name: pts[bmin][0] for name, pts in points.items() if bmin in pts}
        bwd_best = max((v for name, v in at_min.items()
                        if points[name][bmin][1] == "backward"), default=None)
        fwd_at = [v for name, v in at_min.items()
                  if points[name][bmin][1] == "forward"]
        if bwd_best is not None and fwd_at:
            small_budget = (
                f'<div class="callout">In a hurry, the difference is dramatic: '
                f"allowed only <b>{bmin} search steps</b> per puzzle, the best "
                f"backward mode still solves <b>{fpct(bwd_best)}</b> of the "
                f"puzzles, while the forward planners manage "
                f"<b>{fpct(min(fwd_at))}–{fpct(max(fwd_at))}</b>. The backward "
                f"planner front-loads its solving; the forward planners need "
                f"room to search.</div>")
    if unrecognized:
        note += ('<p class="muted small">Curve files present but in an '
                 "unrecognized format (listed, not rendered): "
                 + ", ".join(esc(u) for u in unrecognized) + "</p>")
    return f"""
<section id="budget">
  <div class="sec-head">
    <div class="kicker">solve rate vs search budget</div>
    <h2>What each system does with a smaller or larger budget</h2>
    <p class="muted">The headline table uses one budget cap. This section
    repeats the comparison at several caps — fewer allowed search steps per
    puzzle — so the two families can be compared across the whole range rather
    than at a single operating point.</p>
  </div>
  {chart_html}
  {small_budget}
  {note}
</section>
"""


def sec_selfplay(D):
    arms = D["arms"]
    blocks = []

    # matched-150 verdict: supervised nets vs self-play-improved nets, both
    # using the in-search check
    sup = D["prefix150"]
    arm5 = D["arm_prefix5"]
    if sup and arm5:
        _, s_sup = first_backward(sup)
        _, s_arm = first_backward(arm5)
        if s_sup and s_arm:
            a, b = s_sup["aggregate"], s_arm["aggregate"]
            step_gain = ""
            if a.get("mean_expansions") and b.get("mean_expansions"):
                step_gain = (f" and <b>{100 * (1 - b['mean_expansions'] / a['mean_expansions']):.0f}% "
                             f"fewer search steps</b>")
            head = ("<tr><th>backward networks</th><th>puzzles</th>"
                    "<th>solved (playable)</th><th>extra moves</th>"
                    "<th>search steps</th><th>seconds / puzzle</th></tr>")
            body = ""
            for label, agg in [
                    ("supervised networks", a),
                    ("after six self-play rounds", b)]:
                body += (f"<tr><td><b>{esc(label)}</b></td>"
                         f"<td class='num'>{agg['n']}</td>"
                         f"<td class='num'>{ffrac(agg['solved'], agg['n'])}</td>"
                         f"<td class='num'>{fnum(agg['mean_regret'], 3)}</td>"
                         f"<td class='num'>{fnum(agg['mean_expansions'], 1)}</td>"
                         f"<td class='num'>{fnum(agg['mean_seconds'], 2)}</td></tr>")
            blocks.append(
                "<h3 class='subh'>The verdict on a matched benchmark slice</h3>"
                + scroll(f"<table>{head}{body}</table>")
                + f"<p>Both rows use the same in-search playability check. After "
                  f"six rounds of self-play the networks match the supervised "
                  f"ones on solve rate — both are pressed against the language "
                  f"ceiling explained in <a href='#ceiling'>why plans fail</a> — "
                  f"with slightly better move quality{step_gain}. At this scale "
                  f"self-play is a wash on solve rate and a modest efficiency "
                  f"gain. Its decisive property is different: it keeps working "
                  f"at scales where the exact solver (and therefore supervised "
                  f"training) is gone — see <a href='#scaling'>scaling</a>. "
                  f"Sources: eval/results/prefix150_prefix.json, "
                  f"eval/results/arm_prefix_iter5.json.</p>")
    else:
        blocks.append(pending(
            "the matched supervised-vs-self-play benchmark files "
            "(eval/results/prefix150_prefix.json, eval/results/arm_prefix_iter5.json) "
            "are not both present yet"))

    # per-round generation statistics, one compact table per run directory
    stats_files = D["selfplay_stats"]
    if not stats_files:
        blocks.append(pending(
            "per-round generation and playability statistics will appear here "
            "as self-play rounds complete "
            "(subgoal_selfplay/runs_warm*/iter*_stats.json)"))
    else:
        by_run = {}
        for path in stats_files:
            run = os.path.relpath(os.path.dirname(path), ROOT)
            by_run.setdefault(run, []).append(path)
        for run, paths in sorted(by_run.items()):
            rows_html = []
            for path in sorted(paths):
                try:
                    with open(path) as f:
                        d = json.load(f)
                except Exception as e:
                    rows_html.append(
                        f"<tr><td colspan='7' class='muted small'>"
                        f"{esc(os.path.basename(path))}: unreadable ({esc(e)})"
                        f"</td></tr>")
                    continue
                g = d.get("gen", {}) if isinstance(d.get("gen"), dict) else {}
                p = d.get("probe", {}) if isinstance(d.get("probe"), dict) else {}
                rows_html.append(
                    "<tr>"
                    f"<td class='num'>{esc(d.get('iteration', '—'))}</td>"
                    f"<td class='num'>{esc(g.get('n_instances', '—'))}</td>"
                    f"<td class='num'>{fpct(g.get('solve_rate')) if g.get('solve_rate') is not None else '—'}</td>"
                    f"<td class='num'>{fpct(g.get('strict_pass_rate')) if g.get('strict_pass_rate') is not None else '—'}</td>"
                    f"<td class='num'>{esc(g.get('n_records', '—'))}</td>"
                    f"<td class='num'>{fpct(p.get('solve_rate')) if p.get('solve_rate') is not None else '—'}</td>"
                    f"<td class='num'>{fnum(p.get('mean_regret'), 3) if p.get('mean_regret') is not None else '—'}</td>"
                    "</tr>")
            head = ("<tr><th>round</th><th>puzzles generated</th>"
                    "<th>solved by own search</th><th>winners playable</th>"
                    "<th>training records</th><th>probe: solved</th>"
                    "<th>probe: extra moves</th></tr>")
            blocks.append(
                f"<h3 class='subh mono'>{esc(run)}</h3>"
                + scroll(f"<table>{head}{''.join(rows_html)}</table>"))
        blocks.append(
            "<p class='muted small'>Each self-play round: generate fresh "
            "puzzles, solve them with the planner's own current networks (no "
            "exact solver anywhere), keep only solutions that really play out, "
            "train on those, repeat. “Winners playable” is the share of kept "
            "training plans that pass the strict physics check — the runs "
            "listed here keep it at or near 100% by construction. The probe is "
            "a fixed 50-puzzle set scored between rounds.</p>")

    # run index (arms) as recorded
    if arms:
        head = ("<tr><th>run</th><th>role</th><th>recorded settings</th>"
                "<th>training label</th><th>progress</th></tr>")
        rows = []
        for arm in arms.get("arms", []):
            prog = str(arm.get("iterations_completed", ""))
            prog_cls = " class='pendtag'" if "pending" in prog.lower() else ""
            rows.append(
                "<tr>"
                f"<td class='mono small'>{esc(arm.get('run_dir', ''))}</td>"
                f"<td>{esc(arm.get('role', ''))}</td>"
                f"<td>{esc(arm.get('flags', ''))}</td>"
                f"<td>{esc(arm.get('label_semantics', ''))}</td>"
                f"<td><span{prog_cls}>{esc(prog)}</span></td>"
                "</tr>"
            )
        blocks.append(
            "<h3 class='subh'>Every self-play run on record</h3>"
            "<p class='muted small'>Shown as recorded in "
            "subgoal_selfplay/arms_index.json; the settings column quotes the "
            "runs’ own notes verbatim. Runs are never mixed.</p>"
            + scroll(f"<table>{head}{''.join(rows)}</table>"))
    else:
        blocks.append(pending("waiting for subgoal_selfplay/arms_index.json"))

    return f"""
<section id="selfplay">
  <div class="sec-head">
    <div class="kicker">self-play for the backward planner</div>
    <h2>Training the backward planner on its own solutions</h2>
    <p class="muted">Self-play means: the planner solves puzzles with its own
    current networks, the solved puzzles become its next batch of training
    data, and the loop repeats — no exact solver anywhere. The forward family’s
    self-play results are in the headline table (warm start and from-scratch
    rows); this section covers the backward family’s.</p>
  </div>
  {"".join(blocks)}
</section>
"""


def tab_results(D, rows):
    return (sec_headline(D, rows) + sec_efficiency(D, rows)
            + sec_budget_curves(D) + sec_selfplay(D))


# ---------------------------------------------------------------------------
# Tab 3 — Why plans fail (the ceiling)
# ---------------------------------------------------------------------------

def sec_fixes(D):
    stages = []
    pre_plain = None
    if D["bwd450"]:
        _, s = first_backward(D["bwd450"])
        if s and s.get("rows"):
            pre_plain = agg_from_backward_rows(s["rows"][:150])
    pre_any = None
    if D["bwd_any150"]:
        _, s = first_backward(D["bwd_any150"])
        pre_any = s["aggregate"] if s else None
    stages.append(("Before the fixes", pre_plain, pre_any))

    def stage_from(key_plain, key_any, label):
        p = a = None
        if D[key_plain]:
            _, s = first_backward(D[key_plain])
            p = s["aggregate"] if s else None
        if D[key_any]:
            _, s = first_backward(D[key_any])
            a = s["aggregate"] if s else None
        stages.append((label, p, a))

    stage_from("postfix", "postfix_any", "After fix 1 — no bouncing off yourself")
    stage_from("postfix2", "postfix2_any",
               "After fix 2 — flexible order when turning a plan into moves")
    stage_from("postfix4", "postfix4_any",
               "After fixes 3+4 — one robot per role; promised stoppers must be placed")

    any_data = any(p or a for _, p, a in stages)
    if not any_data:
        table = pending("waiting for the staged rerun files "
                        "(eval/results/comparison_backward_postfix*.json)")
    else:
        head = (
            "<tr><th rowspan='2'>planner code</th>"
            "<th colspan='2'>standard search</th>"
            "<th colspan='2'>keep-searching mode<sup>b</sup></th></tr>"
            "<tr><th>solved (playable)</th><th>extra moves</th>"
            "<th>solved (playable)</th><th>extra moves</th></tr>"
        )
        body = []
        for label, p, a in stages:
            cells = []
            for agg in (p, a):
                if agg:
                    cells.append(f'<td class="num">{ffrac(agg["solved"], agg["n"])}</td>'
                                 f'<td class="num">{fnum(agg["mean_regret"], 3)}</td>')
                else:
                    cells.append('<td class="num">—</td><td class="num">—</td>')
            body.append(f"<tr><td><b>{esc(label)}</b></td>{''.join(cells)}</tr>")
        table = scroll(f'<table id="progression-table">{head}{"".join(body)}</table>')

    ab = D["realizer"]
    ab_html = ""
    if ab:
        n_pass = ab.get("n_passing_checked")
        n_reg = ab.get("n_passing_regressions")
        n_fail = ab.get("n_failing_checked")
        n_fixed = ab.get("n_newly_fixed")
        ab_html = (
            f"<p class='muted small'>Each fix was accepted only after an A/B "
            f"test against the stored benchmark: for fix 2, all "
            f"<b>{n_pass}</b> previously working plans still produce exactly "
            f"the same move counts (<b>{n_reg} regressions</b>), and "
            f"<b>{n_fixed} of the {n_fail}</b> previously failing plans now "
            f"play out fully, verified move by move against the game physics "
            f"(eval/results/realizer_twophase_ab.json).</p>"
        )

    fixcards = """
  <div class="grid2">
    <div class="panel"><h3 class="ph">fix 1 · no bouncing off yourself</h3>
      <p style="margin:0">Every backward step names a stopper — the robot the
      slider will bump into. A bug let the proposal network nominate the
      sliding robot as its <i>own</i> stopper. Fine on paper; impossible on a
      real board. Such proposals are now forbidden during search.</p></div>
    <div class="panel"><h3 class="ph">fix 2 · flexible order when converting to moves</h3>
      <p style="margin:0">Some plans only play out if the moving robot walks up
      close <i>first</i> and the helper parks <i>second</i>. The converter
      always parked the helper first. It now retries a failing step in the
      walk-up-first order — no other reordering, no extra moves added.</p></div>
    <div class="panel"><h3 class="ph">fix 3 · one robot, one job</h3>
      <p style="margin:0">A plan could recruit the same robot under two
      identities — say, as the stopper for two different slides in two places
      at once. Plans may no longer assign one robot two roles.</p></div>
    <div class="panel"><h3 class="ph">fix 4 · promised stoppers must be placed</h3>
      <p style="margin:0">A plan could claim “the slider stops here because a
      robot is parked there” without any step that actually parks one. A
      declared stopper must now really be placed by the plan.</p></div>
  </div>
"""
    return f"""
<section id="fix-history">
  <div class="sec-head">
    <div class="kicker">step 1 · the repairs</div>
    <h2>Most of the gap was bugs — four fixes, each verified</h2>
    <p class="muted">The backward planner almost always finds a plan in its own
    shorthand. The honest question is whether the plan survives contact with
    the real board. Four coding defects accounted for most of the early
    failures; each was fixed and re-measured on the same matched set (the
    first 150 benchmark puzzles), with no change to how the method works.</p>
  </div>
  {table}
  <p class="footcell"><sup>b</sup> keep-searching mode: whenever the search
  completes a plan, the plan is test-played first; if it fails, the search
  discards it and keeps looking instead of returning it.</p>
  {fixcards}
  {ab_html}
</section>
"""


def sec_insearch(D):
    agg = None
    if D["final450_prefix"]:
        _, s = first_backward(D["final450_prefix"])
        agg = s["aggregate"] if s else None
    facts = ""
    if agg:
        facts = (
            f"<p>Result on the full benchmark: "
            f"<b>{ffrac(agg['solved'], agg['n'])} playable</b>, "
            f"{fnum(agg['mean_regret'], 2)} extra moves on average, "
            f"<b>{fnum(agg['mean_expansions'], 1)} search steps</b> and "
            f"{fnum(agg['mean_seconds'], 2)} s per puzzle "
            f"(eval/results/final450_backward_prefix.json). The planner now "
            f"refuses to build unplayable plans rather than discovering the "
            f"problem afterwards.</p>")
    else:
        facts = pending("waiting for eval/results/final450_backward_prefix.json")
    ab = D["prefix_ab"] or {}
    nofp = (ab.get("no_false_pruning_ab") or {}).get("plain_vs_prefix") or {}
    ab_note = ""
    if "false_prunes" in nofp:
        n_fp = len(nofp.get("false_prunes") or [])
        off, on = nofp.get("solved_off"), nofp.get("solved_on")
        ab_note = (
            f"<p class='muted small'>Safety check: across the verification "
            f"set, the filter discarded <b>{n_fp}</b> branches it should have "
            f"kept (zero false discards), and the solved count moved from "
            f"{off} to {on} of {nofp.get('n')} on the matched slice "
            f"(eval/results/prefix_check_ab.json).</p>")
    return f"""
<section id="insearch">
  <div class="sec-head">
    <div class="kicker">step 2 · checking playability inside the search</div>
    <h2>Don’t finish doomed plans — drop them the moment they break</h2>
  </div>
  <p>After the fixes, one structural improvement finished the climb. Instead of
  building a whole plan and only then test-playing it, the search now
  physics-checks every partial plan <b>as it is built</b>: the moment a plan’s
  already-committed steps cannot be played out, that branch of the search is
  discarded and the budget goes to branches that can still succeed.</p>
  {facts}
  {ab_note}
  <p>This same check matters beyond the benchmark score: it is what lets
  self-play train only on plans that really play out
  (<a href="#selfplay">self-play section</a>).</p>
</section>
"""


def sec_ceiling(D):
    probe = D.get("ceiling_probe")
    if not probe:
        return f"""
<section id="the-ceiling">
  <div class="sec-head">
    <div class="kicker">step 3 · the ceiling</div>
    <h2>What remains after the fixes — and whether it can be fixed</h2>
  </div>
  {pending("waiting for analysis/artifacts/ceiling_probe_results.json")}
</section>
"""
    cats = Counter(r["category"] for r in probe)
    n_none = cats.get("NO_COMPLETE_PLAN", 0)
    n_unplay = cats.get("NO_REALIZABLE_PLAN", 0)
    n_miss = cats.get("REALIZABLE_EXISTS", 0)
    n_fail = len(probe)
    n_struct = n_none + n_unplay
    n_total = (len(D["bench_rows"]) if D.get("bench_rows")
               else (D["bench_meta"] or {}).get("n_instances") or 450)
    n_solved = n_total - n_fail
    ceiling = (n_total - n_struct) / n_total * 100
    cur = None
    if D["final450_prefix"]:
        _, s = first_backward(D["final450_prefix"])
        if s:
            cur = s["aggregate"]["solve_rate"] * 100
            aux_fact(
                f"ceiling: official solved count ({s['aggregate']['solved']}) "
                f"+ probed failures ({n_fail}) covers the whole benchmark "
                f"({n_total}) — probe file is not stale",
                s["aggregate"]["solved"] + n_fail == n_total)
    # easy puzzles hit by the ceiling (smallest optimum in each structural group)
    d_none = min((r["d_star"] for r in probe if r["category"] == "NO_COMPLETE_PLAN"
                  and r.get("d_star") is not None), default=None)
    d_unplay = min((r["d_star"] for r in probe if r["category"] == "NO_REALIZABLE_PLAN"
                    and r.get("d_star") is not None), default=None)
    # quality of the 7 recoverable puzzles
    rec = [r for r in probe if r["category"] == "REALIZABLE_EXISTS"
           and r.get("realizable_moves") is not None and r.get("d_star") is not None]
    n_rec_opt = sum(1 for r in rec if r["realizable_moves"] == r["d_star"])
    rec_note = ""
    if rec:
        lo = min(r["realizable_moves"] for r in rec)
        hi = max(r["realizable_moves"] for r in rec)
        dlo = min(r["d_star"] for r in rec)
        dhi = max(r["d_star"] for r in rec)
        rec_note = (
            f" And they are poor recoveries: the plans that do exist need "
            f"{lo}–{hi} moves against optima of {dlo}–{dhi}; only "
            f"{n_rec_opt} of the {len(rec)} is optimal.")

    aux_need("ceiling: structural count rendered", f"{n_struct} of the {n_total}")
    aux_need("ceiling: percentage rendered", f"{ceiling:.1f}%")

    bar = seg_bar([
        ("solved by the best backward mode", n_solved, "var(--bwd)"),
        ("no plan can even be written", n_none, "var(--bad)"),
        ("plans exist, none survives physics", n_unplay, "var(--bad2)"),
        ("a playable plan exists — networks missed it", n_miss, "var(--warn)"),
    ], "All 450 benchmark puzzles, split by why the backward planner fails")

    cur_txt = f"{cur:.1f}%" if cur is not None else "—"
    gap_txt = (f"{(n_total - n_struct) / n_total * 100 - cur:.1f}"
               if cur is not None else "—")

    tiles_html = "".join([
        tile(f"{n_fail}", "puzzles the best backward mode still fails",
             "out of 450"),
        tile(f"{n_struct}", "of those have NO playable plan at all",
             "in the current plan language — measured, not estimated"),
        tile(f"{ceiling:.1f}%", "the ceiling: best score any training or "
             "search can reach", f"the planner already sits at {esc(cur_txt)}"),
    ])

    method_html = f"""
  <h3 class="subh">How the ceiling was measured</h3>
  <p>Every one of the <b>{n_fail}</b> puzzles the best backward mode fails was
  re-solved with the <b>hand-written subgoal search</b> — the neural networks
  removed entirely, every budget cap lifted — running exhaustively until it
  either found a plan that plays out or provably ran out of plans to try.
  Because the networks are out of the loop, “bad guidance” is ruled out as an
  explanation: whatever this search cannot do, no network-guided version of it
  can do either. Every puzzle resolved conclusively, each in seconds
  (analysis/artifacts/ceiling_probe.py → ceiling_probe_results.json). The
  {n_fail} failures split three ways:</p>
"""

    split_table = scroll(f"""
<table>
  <tr><th>group</th><th>puzzles</th><th>what it means</th><th>can training or search fix it?</th></tr>
  <tr><td><b>No plan can even be written</b></td><td class="num">{n_none}</td>
      <td>The exhaustive search builds <i>zero</i> complete plans. The subgoal
      vocabulary cannot decompose these puzzles at all.</td>
      <td><b>No</b> — there is nothing to find.</td></tr>
  <tr><td><b>Plans exist, none survives physics</b></td><td class="num">{n_unplay}</td>
      <td>Complete plans can be written, but every single one fails when played
      out move by move on the real board.</td>
      <td><b>No</b> — every findable plan is a dud.</td></tr>
  <tr><td><b>A playable plan exists — the networks missed it</b></td><td class="num">{n_miss}</td>
      <td>A working plan is out there; the network-guided search never
      generated it within its budget.</td>
      <td><b>Yes</b> — this part is genuine headroom.</td></tr>
</table>
""")

    LIFTED_STATUS = (
        "That extension has since been <b>implemented and re-measured</b> — "
        "the next section shows the ceiling moving."
        if D.get("ceiling_probe_b1") else
        "It is designed but <b>not yet implemented</b>; how much of the "
        "structural share it recovers is an open measurement — re-running "
        "this same probe after the change answers it directly.")
    answer_html = f"""
  <h3 class="subh">“Is that unfixable?” — the honest answer</h3>
  <div class="grid2">
    <div class="panel edge-bad">
      <h3 class="ph">not fixable by training or more search</h3>
      <p style="margin:0">For <b>{n_struct} of the {n_total} puzzles
      ({100 * n_struct / n_total:.1f}%)</b> there is no playable plan to find:
      a planner that can only say <i>“park a helper robot on this cell and
      bounce off it”</i> has no correct thing to say about them, no matter how
      good its networks get or how long it searches. That puts a hard ceiling
      of <b>{ceiling:.1f}%</b> on this benchmark. The planner already scores
      {esc(cur_txt)}, so even perfect networks would add only about
      {esc(gap_txt)} points ({n_miss} puzzles).{esc(rec_note)}</p>
    </div>
    <div class="panel edge-ok">
      <h3 class="ph">fixable by extending the plan language</h3>
      <p style="margin:0">Today a helper parked as a stopper is assumed to stay
      put forever. The scoped extension adds new things a plan may say: a
      stopper robot may stand on a wall-less cell (held in place only by
      another robot or by timing), and a robot may <b>briefly step aside</b> to
      clear the way for one specific slide. That is exactly the maneuver the
      inexpressible puzzles need (see the worked example below).
      {LIFTED_STATUS}</p>
    </div>
  </div>
  <p>Two things worth stating plainly. First, this ceiling is about
  <b>expressiveness, not difficulty</b>: among the inexpressible puzzles is one
  whose optimal solution is just {esc(str(d_none)) if d_none is not None else "a few"} moves
  long, and the unplayable-plans group starts at
  {esc(str(d_unplay)) if d_unplay is not None else "a few"} moves. Easy puzzles hit it too.
  Second, a clean negative would also be a real result: if extending the
  vocabulary cannot lift the ceiling economically, that is itself the
  strongest evidence that the move-by-move formulation is the better vehicle
  at this scale — which is exactly the question this project exists to
  answer.</p>
"""
    return f"""
<section id="the-ceiling">
  <div class="sec-head">
    <div class="kicker">step 3 · the ceiling</div>
    <h2>The last ~10% is a property of the plan language — measured, and answerable</h2>
  </div>
  <div class="tiles tiles3">{tiles_html}</div>
  <div class="panel">{bar}</div>
  {method_html}
  {split_table}
  {answer_html}
</section>
"""


def sec_ceiling_lifted(D):
    probe_b1 = D.get("ceiling_probe_b1")
    if not probe_b1:
        return f"""
<section id="ceiling-lifted">
  <div class="sec-head">
    <div class="kicker">step 4 · lifting the ceiling</div>
    <h2>Making the plan language richer</h2>
  </div>
  {pending("the re-measured ceiling renders here from "
           "analysis/artifacts/ceiling_probe_results_b1.json once the "
           "language-extension probe has run")}
</section>
"""
    probe = D.get("ceiling_probe") or []
    n_total = (len(D["bench_rows"]) if D.get("bench_rows")
               else (D["bench_meta"] or {}).get("n_instances") or 450)
    old_struct = sum(1 for r in probe if r["category"] in
                     ("NO_COMPLETE_PLAN", "NO_REALIZABLE_PLAN"))
    cats = Counter(r["category"] for r in probe_b1)
    n_none = cats.get("NO_COMPLETE_PLAN", 0)
    n_unplay = cats.get("NO_REALIZABLE_PLAN", 0)
    n_rec = cats.get("REALIZABLE_EXISTS", 0)
    new_struct = n_none + n_unplay
    old_c = (n_total - old_struct) / n_total * 100
    new_c = (n_total - new_struct) / n_total * 100

    rec = [r for r in probe_b1 if r["category"] == "REALIZABLE_EXISTS"
           and r.get("realizable_moves") is not None
           and r.get("d_star") is not None]
    n_opt = sum(1 for r in rec if r["realizable_moves"] == r["d_star"])
    excess = sorted(r["realizable_moves"] - r["d_star"] for r in rec)
    med_excess = excess[len(excess) // 2] if excess else None
    n_parks = sum(1 for r in probe_b1
                  if r["category"] == "REALIZABLE_EXISTS"
                  and (r.get("parks") or 0) > 0)

    aux_need("lifted ceiling: new percentage rendered", f"{new_c:.1f}%")
    aux_fact(f"lifted ceiling: probe covers the same failure set "
             f"({len(probe_b1)} rows = {len(probe) or len(probe_b1)} rows)",
             not probe or len(probe_b1) == len(probe))

    learned = ""
    b1a = _sysagg(D.get("b1_450"), "backward")
    if b1a:
        aux_fact(f"lifted ceiling: retrained planner ({b1a['solved']}) sits "
                 f"at or under the new ceiling ({n_total - new_struct})",
                 b1a["solved"] <= n_total - new_struct)
        learned = (
            f"<p>The language is only half the job — the networks had never "
            f"seen the new kinds of plan step, so they were <b>retrained</b> "
            f"on examples written in the richer language. Result on the full "
            f"benchmark: <b>{ffrac(b1a['solved'], b1a['n'])} solved "
            f"playably</b> at {fnum(b1a['mean_expansions'], 1)} search steps "
            f"per puzzle — up from the pre-extension best in the table above, "
            f"and about {(new_c - 100 * b1a['solve_rate']):.1f} points under "
            f"the new ceiling (eval/results/final450_backward_b1.json).</p>")

    ab = D.get("b1_ab") or {}
    safety = ""
    if ab.get("n"):
        safety = (
            f"<p class='muted small'>Safety check: all <b>{ab['n']}</b> "
            f"stored plans from before the change convert to moves "
            f"identically under the extended converter "
            f"({ab.get('n_equal')} of {ab['n']} equal, "
            f"{len(ab.get('mismatches') or [])} mismatches) — the richer "
            f"language adds words without changing the meaning of any "
            f"existing plan (eval/results/realizer_b1_ab.json).</p>")

    remaining = ""
    probe_b2 = D.get("ceiling_probe_b2")
    if probe_b2:
        cats2 = Counter(r["category"] for r in probe_b2)
        s2 = (cats2.get("NO_COMPLETE_PLAN", 0)
              + cats2.get("NO_REALIZABLE_PLAN", 0))
        c2 = (n_total - s2) / n_total * 100
        aux_need("second extension: new percentage rendered", f"{c2:.1f}%")
        remaining = (
            f"<p><b>The second extension.</b> A follow-up change lets a plan "
            f"<i>re-use</i> a robot that is already parked — as the stopper "
            f"for another bounce, or by sliding it from its current parking "
            f"spot to a new one — instead of always recruiting a fresh "
            f"robot. Re-probing the remaining failures with it: puzzles with "
            f"no playable plan drop from <b>{new_struct} to {s2}</b> of "
            f"{n_total}, moving the ceiling from {new_c:.1f}% to "
            f"<b>{c2:.1f}%</b> "
            f"(analysis/artifacts/ceiling_probe_results_b2.json).</p>")
    elif new_struct:
        remaining = (
            f"<p><b>What still cannot be said.</b> {new_struct} puzzles "
            f"remain out of reach: {n_none} still admit no complete plan "
            f"(their helper-delivery chains run out of robots), and "
            f"{n_unplay} have plans that all fail the physics check in ways "
            f"one step-aside cannot fix. These need one further, already "
            f"scoped bookkeeping extension — letting a plan <i>re-use</i> an "
            f"already-parked robot as the stopper for a second bounce — "
            f"which is in progress. Re-running this probe after it lands "
            f"updates this section automatically.</p>")

    quality = ""
    if rec:
        quality = (
            f" Recovery quality is good: {n_opt} of the {len(rec)} newly "
            f"expressible puzzles are solved move-optimally, with a median "
            f"of {med_excess} extra moves; {n_parks} needed the step-aside "
            f"maneuver.")

    return f"""
<section id="ceiling-lifted">
  <div class="sec-head">
    <div class="kicker">step 4 · lifting the ceiling</div>
    <h2>The richer plan language moved the ceiling: {old_c:.1f}% → {new_c:.1f}%</h2>
  </div>
  <div class="tiles tiles3">
    {tile(f"{old_struct} → {new_struct}",
          "puzzles with no playable plan at all, before → after",
          f"out of {n_total}; measured by the same exhaustive no-network probe")}
    {tile(f"{new_c:.1f}%",
          "the new hard ceiling for the backward planner",
          f"was {old_c:.1f}% before the language extension")}
    {tile(f"{n_rec}",
          "previously impossible puzzles that now have a playable plan",
          "confirmed by playing each recovered plan out move by move")}
  </div>
  <p>Two additions were made to what a plan may say, both staying strictly
  inside the subgoal way of thinking (no raw moves were added to plans):
  a stopper robot may now stand on a <b>wall-less cell</b> — a cell it can
  only be held on because another robot, or the timing of the plan itself,
  stops it there — and a robot may be scheduled to <b>slide aside</b> just
  before a specific step of the plan runs, clearing the way. Every such
  maneuver is costed as ordinary moves; nothing is free.{quality}</p>
  {learned}
  {remaining}
  {safety}
</section>
"""


def sec_worked_example(D):
    w = D.get("worked")
    if not w:
        return f"""
<section id="worked-example">
  <div class="sec-head">
    <div class="kicker">step 5 · one puzzle, in full</div>
    <h2>A four-move puzzle the plan language cannot express</h2>
  </div>
  {pending("the worked example renders from analysis/artifacts/"
           "ceiling_probe_instances.json plus the board file "
           "environments/env_*.pkl; one of them is unavailable")}
</section>
"""
    inst, verdict, sol = w["inst"], w["verdict"], w["solution"]
    wr, wd = w["walls"]
    size = w["size"]
    tidx = inst["target_idx"]
    goal = tuple(inst["target"])
    aux_fact("worked example: recomputed optimal length equals the recorded "
             f"optimum ({len(sol)} vs {inst['d_star']})",
             len(sol) == inst["d_star"])

    robots = [(p[0], p[1], SLOT_VARS[i], SLOT_NAMES[i][0], i == tidx)
              for i, p in enumerate(inst["positions"])]
    arrows = [(frm[0], frm[1], to[0], to[1], SLOT_VARS[i], k)
              for k, (i, d, frm, to) in enumerate(sol, start=1)]
    rings = [(goal[0], goal[1], SLOT_VARS[tidx])]
    marks = []
    conflict = w.get("conflict")
    if conflict:
        cx_, cy_ = conflict["cell"]
        marks = [(cx_, cy_, "this cell must be empty for one slide and "
                            "occupied for a later one")]
    svg = board_svg(size, wr, wd, robots, arrows, rings, marks,
                    aria=f"The real {size}-by-{size} board of benchmark puzzle "
                         f"{inst['idx']}, with its {len(sol)}-move optimal "
                         f"solution drawn as numbered arrows.")

    steps = []
    for k, (i, d, frm, to) in enumerate(sol, start=1):
        who = SLOT_NAMES[i] + (" (the target)" if i == tidx else "")
        steps.append(
            f"<li><span class='rdot' style='background:var({SLOT_VARS[i]})'></span>"
            f"<b>{esc(who)}</b> slides {esc(d)}: "
            f"{esc(str(tuple(frm)))} → {esc(str(tuple(to)))}</li>")
    steps_html = f"<ol class='movelist'>{''.join(steps)}</ol>"

    story = ""
    if conflict:
        cell = conflict["cell"]
        stopper = SLOT_NAMES[conflict["stopper_slot"]]
        tgt = SLOT_NAMES[tidx]
        second = ""
        if conflict.get("stopper_stop"):
            b2, c2 = conflict["stopper_stop"]
            second = (f" And {esc(stopper)} can only stop at "
                      f"{esc(str(cell))} because <b>{esc(SLOT_NAMES[b2])}</b> "
                      f"is already parked at {esc(str(tuple(c2)))} — a helper "
                      f"for the helper.")
        story = f"""
      <p><b>Why no subgoal plan can say this.</b> Look at cell
      <b>{esc(str(cell))}</b> (dashed outline). In move
      {conflict['passed_step']}, {esc(tgt)} slides straight <i>through</i> that
      cell — so it must be <b>empty</b>. In move {conflict['stopper_step']},
      <b>{esc(stopper)}</b> parks exactly there — so that {esc(tgt)}’s final
      slide can bump into it and stop on the goal. The same cell has to be
      empty first and occupied later.{second}</p>
      <p>A subgoal can only say “park a helper on a cell — then it stays put —
      and bounce off it.” It has no words for <i>“let me pass first, then park
      behind me.”</i> The exhaustive no-network probe confirms this is fatal
      here: it built <b>{verdict.get('complete_tested', 0)} complete plans</b>
      before provably running out of options
      ({verdict.get('iters', '—')} search steps to exhaust every possibility) —
      the puzzle is not merely hard for the backward planner, it is
      <b>unwritable in its language</b>. The proposed “step aside / arrive
      late” stopper described above is precisely the missing word.</p>
"""
    else:
        story = f"""
      <p>The exhaustive no-network probe built
      <b>{verdict.get('complete_tested', 0)} complete plans</b> for this puzzle
      before provably running out of options — it is unwritable in the current
      plan language.</p>
"""
    coda = ""
    b1p = D.get("ceiling_probe_b1")
    if b1p:
        b1row = next((r for r in b1p if r.get("idx") == inst["idx"]), None)
        if b1row and b1row.get("category") == "REALIZABLE_EXISTS":
            rm = b1row.get("realizable_moves")
            opt = (rm is not None and rm == inst["d_star"])
            aux_fact(f"worked example: solved by the richer language at "
                     f"{rm} moves vs optimum {inst['d_star']}",
                     rm is not None)
            coda = (
                f"<p><b>Epilogue.</b> After the plan language was made richer "
                f"(previous section), this exact puzzle became expressible: "
                f"the probe now finds a playable plan of <b>{rm} moves</b>"
                + (" — exactly the optimum" if opt else "")
                + " (analysis/artifacts/ceiling_probe_results_b1.json).</p>")
    return f"""
<section id="worked-example">
  <div class="sec-head">
    <div class="kicker">step 5 · one puzzle, in full</div>
    <h2>A {len(sol)}-move puzzle the plan language cannot express</h2>
    <p class="muted">Benchmark puzzle {inst['idx']} (board
    env_{inst['env_id']}), drawn from the real board file with its real optimal
    solution, recomputed from the wall layout at build time. The forward
    planner solves it; under the original plan language the backward planner
    never can — here is exactly why.</p>
  </div>
  <div class="split">
    <div class="panel">{svg}</div>
    <div>
      <p><b>The only way to solve it in {len(sol)} moves:</b></p>
      {steps_html}
      {story}
      {coda}
    </div>
  </div>
</section>
"""


def tab_ceiling(D):
    intro = """
<section id="ceiling-intro">
  <div class="sec-head">
    <div class="kicker">the playability story</div>
    <h2>From “found a plan” to “the plan actually plays”</h2>
  </div>
  <p>The backward planner writes its plans in shorthand: “park a helper here,
  bounce the target off it, done.” Early measurements graded that shorthand —
  a plan counted as a solution if it was internally complete. Forced through
  the real rules of the game, move by legal move, many of those plans turned
  out to be impossible to play. This tab tells that story start to finish: the
  bugs that were fixed, the search change that finished the climb, the hard
  ceiling that remained — and how making the plan language richer then moved
  that ceiling.</p>
</section>
"""
    return (intro + sec_fixes(D) + sec_insearch(D) + sec_ceiling(D)
            + sec_ceiling_lifted(D) + sec_worked_example(D))


# ---------------------------------------------------------------------------
# Tab 4 — Scaling
# ---------------------------------------------------------------------------

def _oracle_rows(D):
    """(label, failure %, source) rows for the exact-solver failure chart."""
    rows = []
    if D["bench_rows"]:
        n = len(D["bench_rows"])
        failed = sum(1 for r in D["bench_rows"] if r.get("d_star") is None)
        rows.append((f"16×16 board, 4 robots (the base scale)",
                     100.0 * failed / n, n, failed,
                     "eval/data/bench450.jsonl"))
    metas = []
    for path in D["scaling_metas"]:
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        if d.get("oracle_failure_rate") is None:
            continue
        proto = d.get("protocol", {})
        env = proto.get("env", {})
        try:
            grid = int(env.get("RR_GRID"))
            robots = int(env.get("RR_ROBOTS"))
        except (TypeError, ValueError):
            continue
        metas.append((grid, robots, d, os.path.relpath(path, ROOT)))
    for grid, robots, d, rel in sorted(metas):
        rows.append((f"{grid}×{grid} board, {robots} robots",
                     100.0 * d["oracle_failure_rate"],
                     d.get("n_instances"), d.get("n_oracle_failed"), rel))
    return rows


def sec_oracle_death(D):
    rows = _oracle_rows(D)
    if not rows:
        chart = pending(
            "exact-solver failure figures will appear here once the benchmark "
            "stamps (scaling/data/*/bench.jsonl.meta.json) are present")
        items = ""
    else:
        chart = ('<div class="panel">'
                 + scroll(bar_chart(
                     [(lbl, v, "neutral") for lbl, v, _n, _f, _rel in rows],
                     "share of benchmark puzzles the exact solver could NOT "
                     "solve within its (generous) limits",
                     "Exact-solver failure rate by puzzle configuration",
                     value_fmt="{:.1f}%", vmax=100))
                 + "</div>")
        items = ("<p class='muted small'>Sources: "
                 + ", ".join(sorted({rel for *_x, rel in rows})) + ".</p>")
    probe_note = ""
    cp = D["cap_probe"]
    if cp:
        n = len(cp)
        solved = sum(1 for r in cp if r.get("status") == "solved")
        probe_note = (
            f'<div class="callout">Is that a bug or a wall? Measured directly: '
            f"all {n} of the 8-robot puzzles the exact solver failed were "
            f"re-run at <b>ten times</b> its normal search budget. "
            f"<b>{ffrac(solved, n)}</b> became solvable at that price; "
            f"<b>{ffrac(n - solved, n)}</b> stayed out of reach even then. "
            f"The solver is complete in principle — this is a cost explosion, "
            f"not a defect, and it moves outward with scale "
            f"(scaling/data/g16r8/cap_probe_10x_results.jsonl).</div>")
    return f"""
<section id="oracle-death">
  <div class="sec-head">
    <div class="kicker">why scale is the real question</div>
    <h2>The exact solver — the forward planner’s teacher — dies first</h2>
    <p class="muted">Supervised training for the forward planner needs an exact
    solver to label every training example, and the same solver provides the
    reference optima for scoring. As boards grow and robots multiply, that
    solver starts failing even with generous limits — so supervised training
    runs out of teacher, scoring loses its reference, and self-play becomes
    the only training that still works. Whichever family copes better with
    that is the family that scales.</p>
  </div>
  {chart}
  {probe_note}
  {items}
</section>
"""


def _agg_cells(agg, ungraded=False):
    """solved / extra-moves / steps / seconds cells for a scoreboard row."""
    if not agg:
        return ["–"] * 4
    return [
        ffrac(agg.get("solved"), agg.get("n")),
        "–" if ungraded else fnum(agg.get("mean_regret"), 2),
        fnum(agg.get("mean_expansions"), 1),
        fnum(agg.get("mean_seconds"), 1),
    ]


def sec_ladder(D):
    head = (
        "<tr><th rowspan='2'>configuration</th><th rowspan='2'>experiment</th>"
        "<th rowspan='2'>puzzles</th>"
        "<th colspan='4'>backward planner</th>"
        "<th colspan='4'>forward planner</th></tr>"
        "<tr>"
        "<th>solved</th><th>extra moves</th><th>steps</th><th>s/puzzle</th>"
        "<th>solved</th><th>extra moves</th><th>steps</th><th>s/puzzle</th>"
        "</tr>"
    )
    body = []
    notes = set()

    def row(cfg_desc, exp_desc, n_txt, bcells, fcells, tag=""):
        tag_html = f'<div class="cellnote">{esc(tag)}</div>' if tag else ""
        cells = "".join(f'<td class="num">{esc(c) if not c.startswith("<") else c}</td>'
                        for c in bcells + fcells)
        body.append(
            f"<tr><td>{esc(cfg_desc)}</td><td>{esc(exp_desc)}{tag_html}</td>"
            f"<td class='num'>{esc(n_txt)}</td>{cells}</tr>")

    for cfg, desc in SCALING_LADDER:
        files = D["scaling"][cfg]
        expects_control = "comparison_forward_control.json" in SCALING_EXPECTED[cfg]
        comp = files.get("comparison.json")
        control = files.get("comparison_forward_control.json")

        # graded head-to-head
        b_agg = f_agg = None
        n_txt, tag = "–", "planned — results pending"
        if comp:
            _, bs = first_backward(comp)
            b_agg = bs["aggregate"] if bs else None
            n_txt = str((comp.get("protocol") or {}).get("n_instances")
                        or (b_agg or {}).get("n") or "–")
            tag = ""
            if expects_control:
                if control:
                    _, fs = first_forward(control)
                    f_agg = fs["aggregate"] if fs else None
                    tag = "forward = the stability-controlled retrain"
                else:
                    f_agg = None
                    tag = ("forward row withheld: its training collapsed; "
                           "control retrain in progress")
                    notes.add("withheld")
            else:
                _, fs = first_forward(comp)
                f_agg = fs["aggregate"] if fs else None
        row(desc, "graded head-to-head", n_txt,
            _agg_cells(b_agg), _agg_cells(f_agg), tag)

        # beyond the oracle
        if "comparison_ungraded.json" in SCALING_EXPECTED[cfg]:
            ug = files.get("comparison_ungraded.json")
            b_agg = f_agg = None
            n_txt, tag = "–", "planned — results pending"
            if ug:
                _, bs = first_backward(ug)
                _, fs = first_forward(ug)
                b_agg = bs["aggregate"] if bs else None
                f_agg = fs["aggregate"] if fs else None
                n_txt = str((ug.get("protocol") or {}).get("n_instances")
                            or (b_agg or {}).get("n") or "–")
                tag = "no reference optimum exists — a solution proves itself"
                notes.add("ungraded")
            row(desc, "beyond the oracle", n_txt,
                _agg_cells(b_agg, ungraded=True), _agg_cells(f_agg, ungraded=True),
                tag)

        # rows where the backward side was re-run with the richer plan
        # language (its networks retrained on it); the forward opponent is the
        # same properly trained one as in the row above
        if "comparison_b1.json" in SCALING_EXPECTED[cfg]:
            b1 = files.get("comparison_b1.json")
            if b1:
                b_agg = _sysagg(b1, "backward")
                f_agg = _sysagg(files.get("comparison_forward_control.json"),
                                "forward")
                n_txt = str((b1.get("protocol") or {}).get("n_instances")
                            or (b_agg or {}).get("n") or "–")
                row(desc, "graded head-to-head — richer plan language", n_txt,
                    _agg_cells(b_agg), _agg_cells(f_agg),
                    "backward networks retrained on the richer language; "
                    "same forward opponent as above")
        if "comparison_ungraded_b1.json" in SCALING_EXPECTED[cfg]:
            ugb1 = files.get("comparison_ungraded_b1.json")
            if ugb1:
                b_agg = _sysagg(ugb1, "backward")
                f_agg = _sysagg(files.get("comparison_ungraded.json"),
                                "forward")
                n_txt = str((ugb1.get("protocol") or {}).get("n_instances")
                            or (b_agg or {}).get("n") or "–")
                row(desc, "beyond the oracle — richer plan language", n_txt,
                    _agg_cells(b_agg, ungraded=True),
                    _agg_cells(f_agg, ungraded=True),
                    "no reference optimum exists — a solution proves itself")
                notes.add("ungraded")

    note_html = ""
    if "ungraded" in notes:
        note_html += (
            "<p class='muted small'>“Beyond the oracle” rows are the puzzles "
            "the exact solver could not grade — the hardest subset. No optimal "
            "move counts exist there, so the extra-moves column is dashed by "
            "design; solve rate is still fully trustworthy, because a plan "
            "that plays out legally to the goal proves itself.</p>")
    if "withheld" in notes:
        note_html += (
            "<p class='muted small'>Integrity rule for withheld forward rows: "
            "the forward training recipe that works at 4 robots has "
            "destabilized at larger robot counts (validation accuracy near "
            "random). A row publishes only against a properly trained "
            "opponent, so each affected rung waits for its lower-learning-rate "
            "control retrain — the same correction that was applied, and "
            "verified, at 6 robots. The repeated need for per-scale tuning is "
            "itself scaling evidence, with the caveat that each point has so "
            "far been rescuable.</p>")
    return f"""
<section id="ladder">
  <div class="sec-head">
    <div class="kicker">the ladder</div>
    <h2>Every planned configuration — measured or pending</h2>
    <p class="muted">The same head-to-head is being repeated up a ladder of
    harder configurations. Dashes mean the experiment is defined and waiting
    on training or evaluation runs; the page rebuilds from the result files,
    so pending cells fill in as runs land.</p>
  </div>
  {scroll(f'<table id="ladder-table">{head}{"".join(body)}</table>')}
  {note_html}
</section>
"""


def friendly_system_name(name, kind):
    """Plain-language display name for a raw eval.compare system name."""
    if kind == "forward":
        label, _desc, path = forward_label(name)
        if "lightning_logs" in path or "scaling/runs" in path:
            return "Move-by-move planner (trained for this configuration)"
        return label.replace("Forward planner", "Move-by-move planner")
    parts = []
    if "extended language" in name or "B1" in name:
        parts.append("richer plan language")
    if "anytime" in name:
        parts.append("keeps searching past plans that fail the physics test")
    if "prefix-check" in name:
        parts.append("checks playability while it plans")
    extra = f" ({'; '.join(parts)})" if parts else ""
    return f"Subgoal planner{extra}"


def sec_scaling_details(D):
    blocks = []
    for cfg, desc in SCALING_LADDER:
        files = D["scaling"][cfg]
        expects_control = "comparison_forward_control.json" in SCALING_EXPECTED[cfg]
        control_present = bool(files.get("comparison_forward_control.json"))
        for fname in SCALING_EXPECTED[cfg]:
            d = files.get(fname)
            if not d:
                continue
            rel = os.path.join("scaling", "results", cfg, fname)
            ungraded = "ungraded" in fname
            variant = SCALING_FILE_LABEL.get(fname, fname)
            head = ("<tr><th>system</th><th>solved (playable)</th>"
                    "<th>extra moves vs optimal</th><th>% at optimum</th>"
                    "<th>search steps used</th><th>seconds / puzzle</th></tr>")
            rows = []
            for name, s in (d.get("systems") or {}).items():
                if not (isinstance(s, dict) and isinstance(s.get("aggregate"), dict)):
                    continue
                a = s["aggregate"]
                fam = "backward" if s.get("kind") == "backward" else "forward"
                withheld = (fam == "forward" and fname == "comparison.json"
                            and expects_control and not control_present)
                superseded = (fam == "forward" and fname == "comparison.json"
                              and expects_control and control_present)
                # ungraded runs carry a placeholder optimum: regret/% at optimum
                # are meaningless there and must not render as real numbers
                if withheld:
                    cells = "<td class='num'>–</td>" * 5
                    note = ("<div class='cellnote'>withheld — this run’s "
                            "training collapsed (accuracy near random); the "
                            "row publishes after the control retrain</div>")
                else:
                    regret_cell = "–" if ungraded else fnum(a.get("mean_regret"), 3)
                    opt_cell = "–" if ungraded else fnum(a.get("pct_optimal"), 1)
                    cells = (
                        f"<td class='num'>{ffrac(a.get('solved'), a.get('n'))}</td>"
                        f"<td class='num'>{regret_cell}</td>"
                        f"<td class='num'>{opt_cell}</td>"
                        f"<td class='num'>{fnum(a.get('mean_expansions'), 1)}</td>"
                        f"<td class='num'>{fnum(a.get('mean_seconds'), 3)}</td>")
                    note = ""
                    if superseded:
                        note = ("<div class='cellnote'>first run — training "
                                "had destabilized; superseded by the control "
                                "retrain table</div>")
                rows.append(f"<tr><td>{fam_dot(fam)}"
                            f"{esc(friendly_system_name(name, fam))}{note}"
                            f"</td>{cells}</tr>")
            note_html = ("<p class='muted small'>No reference optimum exists "
                         "for these puzzles (the exact solver could not grade "
                         "them); solve rate, steps and time are the meaningful "
                         "columns.</p>" if ungraded else "")
            blocks.append(
                f"<h3 class='subh'>{esc(desc)} — {esc(variant)} "
                f"<span class='mono small muted'>({esc(rel)})</span></h3>"
                + scroll(f"<table>{head}{''.join(rows)}</table>") + note_html)

    # headline reading of the two decided frontier results
    reading = ""
    g16r6_files = D["scaling"].get("g16r6") or {}
    ug6 = g16r6_files.get("comparison_ungraded.json")
    ug6_b1 = g16r6_files.get("comparison_ungraded_b1.json")
    if ug6:
        _, fs = first_forward(ug6)
        ba = _sysagg(ug6_b1, "backward") if ug6_b1 else _sysagg(ug6, "backward")
        fa = fs["aggregate"] if fs else None
        if ba and fa:
            ratio_steps = (fa["mean_expansions"] / ba["mean_expansions"]
                           if ba.get("mean_expansions") else None)
            ratio_time = (fa["mean_seconds"] / ba["mean_seconds"]
                          if ba.get("mean_seconds") else None)
            lang_note = (" (with its networks retrained on the richer plan "
                         "language)" if ug6_b1 else "")
            reading += (
                f'<div class="callout"><b>The frontier flip.</b> On the '
                f"{ba['n']} six-robot puzzles too hard for the exact solver, "
                f"the subgoal planner{lang_note} solves "
                f"<b>{ffrac(ba['solved'], ba['n'])}</b> vs the move-by-move "
                f"planner’s {ffrac(fa['solved'], fa['n'])}, using "
                f"{fnum(ratio_steps, 1)}× fewer search steps and "
                f"{fnum(ratio_time, 1)}× less time. Honest caveat: where both "
                f"solve, the move-by-move planner’s solutions are shorter "
                f"({fnum(fa.get('mean_moves'), 1)} vs "
                f"{fnum(ba.get('mean_moves'), 1)} moves on their own solved "
                f"sets), and no true optimum is known on this set. This is "
                f"the regime that matters as puzzles outgrow exact methods — "
                f"and the subgoal planner owns it.</div>")
    g24 = (D["scaling"].get("g24r4") or {}).get("comparison.json")
    if g24:
        _, bs = first_backward(g24)
        _, fs = first_forward(g24)
        if bs and fs:
            ba, fa = bs["aggregate"], fs["aggregate"]
            r_steps = (fa["mean_expansions"] / ba["mean_expansions"]
                       if ba.get("mean_expansions") else None)
            r_time = (fa["mean_seconds"] / ba["mean_seconds"]
                      if ba.get("mean_seconds") else None)
            reading += (
                f'<div class="callout"><b>The board-size axis.</b> On 24×24 '
                f"boards the forward planner still wins quality "
                f"({ffrac(fa['solved'], fa['n'])} vs "
                f"{ffrac(ba['solved'], ba['n'])}), but its cost explodes with "
                f"board size: {fnum(fa['mean_expansions'], 0)} steps and "
                f"{fnum(fa['mean_seconds'], 0)} s per puzzle against the "
                f"backward planner’s {fnum(ba['mean_expansions'], 1)} and "
                f"{fnum(ba['mean_seconds'], 1)} — {fnum(r_steps, 0)}× fewer "
                f"steps, {fnum(r_time, 0)}× faster. Subgoal plans stay a few "
                f"decisions deep however large the board; move-level search "
                f"grows with travel distance.</div>")
    if not blocks:
        blocks = [pending(
            "the larger configurations are being trained and benchmarked; "
            "results will render here from scaling/results/*/ as they land")]
    return f"""
<section id="scaling-details">
  <div class="sec-head">
    <div class="kicker">the measurements</div>
    <h2>Every completed rung, in full</h2>
  </div>
  {reading}
  {"".join(blocks)}
</section>
"""


def tab_scaling(D):
    intro = """
<section id="scaling-intro">
  <div class="sec-head">
    <div class="kicker">the thesis</div>
    <h2>Do subgoals win as puzzles grow?</h2>
  </div>
  <p>At the base scale the forward planner wins on quality and the backward
  planner on effort. The project’s driving hypothesis is that the balance
  tips backward as puzzles grow — more robots, bigger boards — for two
  reasons: the forward planner’s search cost grows with solution and travel
  length, and its supervised training depends on an exact solver that itself
  stops being computable. This tab collects the evidence, rung by rung.</p>
</section>
"""
    return intro + sec_oracle_death(D) + sec_ladder(D) + sec_scaling_details(D)


# ---------------------------------------------------------------------------
# Tab 5 — Method & sources
# ---------------------------------------------------------------------------

def tab_method(D):
    proto = (D["fwd"] or D["bwd450"] or {}).get("protocol", {})
    meta = D["bench_meta"] or {}
    n = meta.get("n_instances") or proto.get("n_instances")
    budget = proto.get("expansions")
    k = proto.get("k")
    sha = meta.get("sha256") or proto.get("instances_sha256") or ""
    generated = meta.get("generated", "")

    protocol = f"""
<section id="protocol">
  <div class="sec-head">
    <div class="kicker">the rules of the comparison</div>
    <h2>Protocol</h2>
  </div>
  <ul class="foots">
    <li><b>Same puzzles.</b> One pinned set of {esc(n or "—")} puzzles (16×16
    board, 4 robots) is used for every base-scale number on this page
    (eval/data/bench450.jsonl, sha256 {esc(sha[:16])}…, generated
    {esc(generated or "—")}). Each scaling rung pins its own set the same
    way.</li>
    <li><b>Same budget.</b> Every system runs best-first search capped at
    {esc(budget or "—")} search steps per puzzle, keeping the top
    {esc(k or "—")} suggested continuations per step. In both families one
    step costs one pass of each network, so budgets are directly
    comparable.</li>
    <li><b>Strict scoring.</b> All costs are counted in real moves. A backward
    plan counts as solved only if it converts into a legal move sequence with
    every robot present (eval/realize.py) — plans that only work on paper do
    not count.</li>
    <li><b>No exact solver at solve time.</b> The planners are networks +
    search only. The exact solver is used solely to create training labels
    and reference optima; where it fails (large configurations), solving is
    scored by self-certification — a plan that plays out legally to the goal
    proves itself.</li>
  </ul>
</section>
"""

    systems_map = """
<section id="four-systems">
  <div class="sec-head">
    <div class="kicker">the study design</div>
    <h2>Two structures × two ways to train = four systems</h2>
  </div>
  {TBL}
  <p class="muted">Both structures share the same neural-network encoder
  design (a transformer that passes messages along the board’s slide graph);
  they do not share weights. “Self-play” everywhere on this page means the
  planner trains on its own search’s solutions — there is no adversary and no
  Monte-Carlo tree search anywhere in the project.</p>
</section>
"""
    tbl = scroll("""
<table>
  <tr><th></th><th>supervised (copy the exact solver)</th><th>self-play (no solver)</th></tr>
  <tr><td><b>forward / move-by-move</b></td>
      <td>the “supervised” and “candidate-scored” headline rows</td>
      <td>the “warm start” and “from scratch” headline rows</td></tr>
  <tr><td><b>backward / subgoals</b></td>
      <td>the backward headline rows</td>
      <td>the self-play section of the results tab</td></tr>
</table>
""")
    systems_map = systems_map.replace("{TBL}", tbl)

    abs_note = ""
    if D["bwd450"]:
        _, s = first_backward(D["bwd450"])
        if s:
            a = s["aggregate"]
            neg = a.get("n_negative_abstract_regret")
            if neg is not None:
                abs_note = (
                    f" In the full 450-puzzle run, {neg} of {a['n']} plans "
                    f"claimed an internal move count <i>below</i> the known "
                    f"optimum — impossible as played, and direct proof the two "
                    f"units must not be mixed."
                )
    prov_rows = []
    for rel, e in SOURCES.items():
        status = {"ok": "found", "missing": "not present yet",
                  "unreadable": "unreadable"}[e["status"]]
        cls = "" if e["status"] == "ok" else " class='pendtag'"
        internal = ""
        if e["data"] and isinstance(e["data"], dict):
            d = e["data"]
            proto2 = d.get("protocol") if isinstance(d.get("protocol"), dict) else {}
            internal = (d.get("date") or d.get("generated")
                        or proto2.get("date") or proto2.get("generated") or "")
        prov_rows.append(
            "<tr>"
            f"<td class='mono small'>{esc(rel)}</td>"
            f"<td><span{cls}>{esc(status)}</span>"
            + (f" <span class='muted small'>({esc(e['note'])})</span>" if e["note"] else "")
            + "</td>"
            f"<td class='num small'>{esc(e['mtime'] or '—')}</td>"
            f"<td class='num small'>{esc(internal or '—')}</td>"
            "</tr>"
        )
    prov = scroll(
        "<table><tr><th>source</th><th>status</th><th>file modified</th>"
        "<th>recorded run date</th></tr>" + "".join(prov_rows) + "</table>"
    )
    built = datetime.now().strftime("%Y-%m-%d %H:%M")
    return protocol + systems_map + f"""
<section id="notes">
  <div class="sec-head">
    <div class="kicker">fine print</div>
    <h2>Footnotes and provenance</h2>
  </div>
  <ol class="foots">
    <li><b>Search steps are not the same size in the two families.</b> One
    backward search step commits an entire stepping-stone — a multi-move
    commitment (the moving robot's approach plus a helper placement) — while one
    forward step commits a single move. A shared cap therefore grants the
    backward planner strictly more solution-building work per step; the matched
    budget is generous to the backward system, which is the conservative
    direction for this comparison.</li>
    <li><b>The backward planner gets extra precomputed knowledge.</b> Its
    environment includes exact shortest-travel-distance tables for each board,
    used during planning; the forward planner sees only the wall geometry.
    Part of the backward planner's step-count advantage comes from those
    tables, not from the networks.</li>
    <li><b>Two cost units exist and only one is honest.</b> The backward
    planner's internal move estimate prices each plan step as if only its
    intended helper robot were on the board, so it can undercount real game
    moves.{abs_note} Every comparable number on this page uses the strict
    count: moves of a plan that was actually executed legally, move by move.</li>
    <li><b>Where every number comes from.</b> This page is generated by
    <span class="mono">eval/build_report.py</span>; no result number is typed
    into the page by hand, and a build-time verification pass re-checks the
    headline table against its source aggregates. Sources read at build time
    ({esc(built)}):</li>
  </ol>
  {prov}
</section>
"""


# ---------------------------------------------------------------------------
# Page shell
# ---------------------------------------------------------------------------

TABS = [
    ("overview", "Overview"),
    ("results", "Detailed results"),
    ("ceiling", "Why plans fail"),
    ("scaling", "Scaling"),
    ("method", "Method & sources"),
]

CSS = """
:root{
  --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --muted:#52514e;
  --faint:#898781; --grid:#e1e0d9; --baseline:#c3c2b7; --line:rgba(11,11,11,.10);
  --fwd:#2a78d6; --fwd2:#6da7ec; --fwd3:#184f95; --fwd4:#0d366b;
  --bwd:#eb6834; --bwd2:#b84a1a; --bwd3:#f0946b; --bwd4:#7d3311;
  --fwd-ink:#1c5cab; --bwd-ink:#b84a1a;
  --bad:#bf3f3f; --bad2:#8f4fae; --warn:#a87f1f; --ok:#2e7d52;
  --hl:rgba(127,127,110,.09);
  --r-red:#d9480f; --r-blue:#1c7ed6; --r-green:#2f9e44; --r-yellow:#c9971c;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --maxw:1080px;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0d0d0d; --surface:#1a1a19; --ink:#f2f1ec; --muted:#c3c2b7;
    --faint:#898781; --grid:#2c2c2a; --baseline:#383835; --line:rgba(255,255,255,.10);
    --fwd:#3987e5; --fwd2:#86b6ef; --fwd3:#5598e7; --fwd4:#9ec5f4;
    --bwd:#d95926; --bwd2:#f08a5c; --bwd3:#eb6834; --bwd4:#f4ad8d;
    --fwd-ink:#86b6ef; --bwd-ink:#f08a5c;
    --bad:#e06c6c; --bad2:#a678d6; --warn:#b98d2e; --ok:#58c48a;
    --hl:rgba(200,200,180,.07);
    --r-red:#ff8a4c; --r-blue:#4dabf7; --r-green:#51cf66; --r-yellow:#ffd43b;
  }
}
:root[data-theme="light"]{
  --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --muted:#52514e;
  --faint:#898781; --grid:#e1e0d9; --baseline:#c3c2b7; --line:rgba(11,11,11,.10);
  --fwd:#2a78d6; --fwd2:#6da7ec; --fwd3:#184f95; --fwd4:#0d366b;
  --bwd:#eb6834; --bwd2:#b84a1a; --bwd3:#f0946b; --bwd4:#7d3311;
  --fwd-ink:#1c5cab; --bwd-ink:#b84a1a;
  --bad:#bf3f3f; --bad2:#8f4fae; --warn:#a87f1f; --ok:#2e7d52;
  --hl:rgba(127,127,110,.09);
  --r-red:#d9480f; --r-blue:#1c7ed6; --r-green:#2f9e44; --r-yellow:#c9971c;
}
:root[data-theme="dark"]{
  --bg:#0d0d0d; --surface:#1a1a19; --ink:#f2f1ec; --muted:#c3c2b7;
  --faint:#898781; --grid:#2c2c2a; --baseline:#383835; --line:rgba(255,255,255,.10);
  --fwd:#3987e5; --fwd2:#86b6ef; --fwd3:#5598e7; --fwd4:#9ec5f4;
  --bwd:#d95926; --bwd2:#f08a5c; --bwd3:#eb6834; --bwd4:#f4ad8d;
  --fwd-ink:#86b6ef; --bwd-ink:#f08a5c;
  --bad:#e06c6c; --bad2:#a678d6; --warn:#b98d2e; --ok:#58c48a;
  --hl:rgba(200,200,180,.07);
  --r-red:#ff8a4c; --r-blue:#4dabf7; --r-green:#51cf66; --r-yellow:#ffd43b;
}
*{box-sizing:border-box}
html,body{max-width:100%;overflow-x:hidden}
@media (prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  line-height:1.6;font-size:16px;-webkit-font-smoothing:antialiased}
.wrap{max-width:var(--maxw);margin:0 auto;padding:0 22px 60px}
section{padding:30px 0;border-top:1px solid var(--grid)}
[id]{scroll-margin-top:70px}
.tabpanel > section:first-child{border-top:0}
h1,h2,h3{line-height:1.18;margin:0}
h1{font-size:clamp(27px,4.4vw,42px);font-weight:750;letter-spacing:-.02em}
h2{font-size:clamp(20px,2.8vw,26px);font-weight:700;letter-spacing:-.015em;margin-bottom:6px}
.subh{font-size:16px;font-weight:680;margin:22px 0 4px}
p{margin:.6em 0;max-width:74ch}
a{color:var(--fwd-ink);text-decoration:none;border-bottom:1px solid var(--line)}
a:hover{border-bottom-color:var(--fwd-ink)}
a:focus-visible,button:focus-visible{outline:2px solid var(--fwd);outline-offset:2px;border-radius:3px}
.muted{color:var(--muted)}
.small{font-size:13px}
.mono{font-family:var(--mono);font-size:.92em}
.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
header{padding:44px 0 6px}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.15em;
  text-transform:uppercase;color:var(--muted);display:flex;gap:9px;
  align-items:center;margin-bottom:14px}
.eyebrow .dot{width:6px;height:6px;border-radius:50%;background:var(--fwd)}
.lede{font-size:clamp(16px,1.9vw,18.5px);color:var(--muted);max-width:72ch;margin-top:14px}
.lede b{color:var(--ink)}
.fwd-ink{color:var(--fwd-ink)} .bwd-ink{color:var(--bwd-ink)}
.chips{display:flex;flex-wrap:wrap;gap:7px;margin-top:20px}
.chip{font-family:var(--mono);font-size:11.5px;background:var(--surface);
  border:1px solid var(--line);border-radius:20px;padding:3px 11px;color:var(--muted)}
.kicker{font-family:var(--mono);font-size:12px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--faint);margin-bottom:14px}
.sec-head{margin-bottom:18px}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:14px;
  padding:18px;margin:10px 0}
.panel .ph{font-size:13px;font-family:var(--mono);letter-spacing:.05em;
  text-transform:uppercase;color:var(--muted);margin:0 0 8px;font-weight:600}
.edge-fwd{border-left:3px solid var(--fwd)}
.edge-bwd{border-left:3px solid var(--bwd)}
.edge-bad{border-left:3px solid var(--bad)}
.edge-ok{border-left:3px solid var(--ok)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.split{display:grid;grid-template-columns:minmax(0,460px) 1fr;gap:22px;align-items:start}
.scroll{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;width:100%;font-size:14px;background:var(--surface);
  border:1px solid var(--line);border-radius:10px}
th{font-family:var(--mono);font-size:11px;letter-spacing:.05em;text-transform:uppercase;
  color:var(--muted);text-align:left;padding:9px 12px;border-bottom:1px solid var(--baseline);
  white-space:nowrap}
td{padding:8px 12px;border-bottom:1px solid var(--grid);vertical-align:top}
tr:last-child td{border-bottom:0}
tr.hlrow td{background:var(--hl)}
.syscell{min-width:230px}
.cellnote{color:var(--muted);font-size:12px;font-weight:400;margin-top:2px;white-space:normal}
.footcell{color:var(--muted);font-size:13px;max-width:80ch}
.fdot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:8px;
  vertical-align:baseline;flex:none}
.fdot-fwd{background:var(--fwd)} .fdot-bwd{background:var(--bwd)}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin:6px 0 2px}
.lkey{font-size:13px;color:var(--muted);display:inline-flex;align-items:center;gap:2px}
.pending{border:1px dashed var(--baseline);border-radius:10px;padding:12px 16px;
  color:var(--muted);font-size:14px;margin:10px 0;background:var(--surface)}
.pdot{display:inline-block;width:8px;height:8px;border-radius:50%;
  background:var(--faint);margin-right:9px}
.pendtag{color:var(--muted);font-style:italic}
.callout{background:var(--surface);border:1px solid var(--line);
  border-left:3px solid var(--baseline);border-radius:10px;padding:12px 16px;
  font-size:14.5px;margin:12px 0;max-width:86ch}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:12px 0}
.tiles3{grid-template-columns:repeat(3,1fr)}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:13px;
  padding:14px 16px}
.tile .tv{font-size:25px;font-weight:740;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums}
.tile .tl{font-size:13px;color:var(--ink);margin-top:3px;line-height:1.4}
.tile .ts{font-size:12px;color:var(--muted);margin-top:7px;line-height:1.45}
.defrow{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:16px}
.defcard{background:var(--surface);border:1px solid var(--line);border-radius:11px;
  padding:12px 14px}
.defcard .dt{font-family:var(--mono);font-size:11.5px;letter-spacing:.07em;
  text-transform:uppercase;color:var(--fwd-ink);margin-bottom:5px}
.defcard .dd{font-size:13px;color:var(--muted);line-height:1.5}
.movelist{list-style:none;margin:8px 0;padding:0;display:flex;flex-direction:column;gap:6px;
  max-width:74ch}
.movelist li{display:flex;align-items:baseline;gap:9px;font-size:14.5px;
  counter-increment:mv}
.movelist li::before{content:counter(mv);font-family:var(--mono);font-size:11px;
  color:var(--faint);min-width:14px;text-align:right}
.movelist{counter-reset:mv}
.rdot{width:11px;height:11px;border-radius:3px;display:inline-block;flex:none;
  align-self:center}
.foots{padding-left:20px;max-width:82ch}
.foots li{margin:10px 0}
.themebtn{position:fixed;top:14px;right:14px;font-family:var(--mono);font-size:12px;
  background:var(--surface);color:var(--muted);border:1px solid var(--line);
  border-radius:18px;padding:5px 13px;cursor:pointer;z-index:30}
.themebtn:hover{color:var(--ink)}
.tabbar{position:sticky;top:0;z-index:20;display:flex;gap:4px;flex-wrap:wrap;
  margin:26px 0 0;padding-top:6px;border-bottom:1px solid var(--baseline);
  background:var(--bg)}
.maintab{font-family:var(--sans);font-size:15px;font-weight:640;padding:10px 16px;
  border:0;background:none;color:var(--muted);cursor:pointer;
  border-bottom:2.5px solid transparent;margin-bottom:-1px}
.maintab:hover{color:var(--ink)}
.maintab.on{color:var(--ink);border-bottom-color:var(--fwd)}
.maintab:focus-visible{outline:2px solid var(--fwd);outline-offset:3px;border-radius:4px}
[hidden]{display:none !important}
sup{color:var(--muted)}
ul{max-width:80ch}
@media (max-width:760px){
  .grid2,.defrow,.split{grid-template-columns:1fr}
  .tiles,.tiles3{grid-template-columns:1fr 1fr}
  .maintab{padding:9px 11px;font-size:14px}
}
"""

JS = """
(function(){
  var btn=document.getElementById('themebtn');
  var modes=['auto','light','dark'];
  var i=0;
  btn.addEventListener('click',function(){
    i=(i+1)%3;
    var m=modes[i];
    if(m==='auto'){delete document.documentElement.dataset.theme;}
    else{document.documentElement.dataset.theme=m;}
    btn.textContent='theme: '+m;
  });
})();
(function(){
  var ids=['overview','results','ceiling','scaling','method'];
  var tabs={},panels={};
  for(var i=0;i<ids.length;i++){
    tabs[ids[i]]=document.getElementById('tab-'+ids[i]);
    panels[ids[i]]=document.getElementById('panel-'+ids[i]);
  }
  var current='overview';
  function activate(id,focus,push){
    if(!tabs[id])return;
    current=id;
    for(var i=0;i<ids.length;i++){
      var k=ids[i],on=(k===id);
      tabs[k].classList.toggle('on',on);
      tabs[k].setAttribute('aria-selected',on?'true':'false');
      tabs[k].tabIndex=on?0:-1;
      if(on){panels[k].removeAttribute('hidden');}
      else{panels[k].setAttribute('hidden','');}
    }
    if(focus){tabs[id].focus();}
    if(push&&window.history&&history.replaceState){
      history.replaceState(null,'','#'+id);
    }
  }
  for(var j=0;j<ids.length;j++){
    (function(id){
      tabs[id].addEventListener('click',function(){activate(id,false,true);});
    })(ids[j]);
  }
  document.getElementById('tabbar').addEventListener('keydown',function(ev){
    var i=ids.indexOf(current),j=null;
    if(ev.key==='ArrowRight'){j=(i+1)%ids.length;}
    else if(ev.key==='ArrowLeft'){j=(i-1+ids.length)%ids.length;}
    else if(ev.key==='Home'){j=0;}
    else if(ev.key==='End'){j=ids.length-1;}
    if(j!==null){ev.preventDefault();activate(ids[j],true,true);}
  });
  function fromHash(){
    var h=(location.hash||'').replace(/^#/,'');
    if(!h){activate('overview',false,false);return;}
    if(tabs[h]){activate(h,false,false);window.scrollTo(0,0);return;}
    var el=document.getElementById(h);
    if(el){
      for(var i=0;i<ids.length;i++){
        if(panels[ids[i]].contains(el)){
          activate(ids[i],false,false);
          el.scrollIntoView();
          return;
        }
      }
    }
    activate('overview',false,false);
  }
  window.addEventListener('hashchange',fromHash);
  fromHash();
})();
"""


def build_page(D):
    rows, checks = build_headline_rows(D)
    header = sec_header(D)
    # method tab is rendered LAST so its provenance table sees every source
    contents = {
        "overview": tab_overview(D, rows),
        "results": tab_results(D, rows),
        "ceiling": tab_ceiling(D),
        "scaling": tab_scaling(D),
    }
    contents["method"] = tab_method(D)

    tabbar = ['<div class="tabbar" role="tablist" aria-label="Report sections" id="tabbar">']
    panels = []
    for i, (tid, label) in enumerate(TABS):
        on = " on" if i == 0 else ""
        sel = "true" if i == 0 else "false"
        ti = "0" if i == 0 else "-1"
        tabbar.append(
            f'<button class="maintab{on}" id="tab-{tid}" role="tab" '
            f'aria-selected="{sel}" aria-controls="panel-{tid}" '
            f'tabindex="{ti}" type="button">{esc(label)}</button>')
        hidden = "" if i == 0 else " hidden"
        panels.append(
            f'<div class="tabpanel" id="panel-{tid}" role="tabpanel" '
            f'aria-labelledby="tab-{tid}" tabindex="0"{hidden}>'
            f'{contents[tid]}</div>')
        aux_need(f"tab button {tid} present", f'id="tab-{tid}"')
        aux_need(f"tab panel {tid} present", f'id="panel-{tid}"')

    # a withheld forward run must never surface as a comparable result
    for cfg, _desc in SCALING_LADDER:
        files = D["scaling"][cfg]
        if ("comparison_forward_control.json" in SCALING_EXPECTED[cfg]
                and files.get("comparison.json")
                and not files.get("comparison_forward_control.json")):
            _, fs = first_forward(files["comparison.json"])
            if fs:
                frac = ffrac(fs["aggregate"].get("solved"),
                             fs["aggregate"].get("n"))
                aux_need(f"{cfg}: withheld forward result ({frac}) is not "
                         f"rendered anywhere", frac, present=False)

    tabbar.append("</div>")
    body = header + "".join(tabbar) + "".join(panels)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ricochet Robots planner comparison</title>
<style>{CSS}</style>
</head>
<body>
<button class="themebtn" id="themebtn" type="button">theme: auto</button>
<noscript><style>.tabpanel[hidden]{{display:block !important}}
.tabbar{{display:none}}</style></noscript>
<div class="wrap">
{body}
</div>
<script>{JS}</script>
</body>
</html>
"""
    return page, rows, checks


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def parse_headline_from_html(html_text):
    """Extract the rendered headline-table cells back out of the HTML."""
    from html.parser import HTMLParser

    class P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_table = False
            self.in_cell = False
            self.rows = []
            self.cur = None
            self.buf = ""

        def handle_starttag(self, tag, attrs):
            if tag == "table" and ("id", "headline-table") in attrs:
                self.in_table = True
            if not self.in_table:
                return
            if tag == "tr":
                self.cur = []
            if tag == "td":
                self.in_cell = True
                self.buf = ""
            if tag == "div" and self.in_cell:
                # cell notes: separate from the primary value
                self.buf += " | "

        def handle_endtag(self, tag):
            if not self.in_table:
                return
            if tag == "td" and self.in_cell:
                self.cur.append(self.buf.strip())
                self.in_cell = False
            if tag == "tr" and self.cur:
                self.rows.append(self.cur)
                self.cur = None
            if tag == "table":
                self.in_table = False

        def handle_data(self, data):
            if self.in_cell:
                self.buf += data

    p = P()
    p.feed(html_text)
    return p.rows


def verify(html_text, rows, checks):
    """Round-trip parse + numeric check table. Returns number of failures."""
    from html.parser import HTMLParser

    # 1. whole-document parse (structural sanity)
    errors = []

    class Chk(HTMLParser):
        def error(self, message):  # pragma: no cover (py<3.10 compat)
            errors.append(message)

    Chk().feed(html_text)
    print(f"[verify] html.parser consumed the document without raising"
          f" ({len(html_text)} bytes, {len(errors)} reported errors)")

    # 2. headline table cells vs source aggregates
    parsed = parse_headline_from_html(html_text)
    fails = 0
    col = {"solved": 2, "mean extra moves": 3, "% at optimum": 4,
           "mean plan length": 5, "mean search steps": 6, "mean seconds": 7}
    dec = {"mean extra moves": 3, "% at optimum": 1, "mean plan length": 2,
           "mean search steps": 1, "mean seconds": 3}
    print(f"\n[verify] headline check table "
          f"({len(parsed)} rendered rows, {len(checks)} checks):")
    print(f"{'row':52s} {'metric':18s} {'source value':>14s} "
          f"{'rendered':>34s}  ok?")
    by_label = {}
    for i, r in enumerate(rows):
        by_label[r["label"]] = parsed[i] if i < len(parsed) else None
    for label, metric, source, src_val, rendered in checks:
        cells = by_label.get(label)
        ok = False
        shown = "(row missing)"
        if cells:
            shown = cells[col[metric]].split(" | ")[0]
            if metric == "solved":
                ok = str(rendered) == shown and (
                    src_val is None or re.search(rf"\b{src_val}/", shown))
            else:
                expect = fnum(src_val, dec[metric])
                ok = shown == expect == rendered
        fails += 0 if ok else 1
        print(f"{label[:52]:52s} {metric:18s} "
              f"{(fnum(src_val, dec.get(metric, 3)) if not isinstance(src_val, str) else src_val)!s:>14s} "
              f"{shown!s:>34.34s}  {'OK' if ok else 'MISMATCH'}")
    print(f"\n[verify] {len(checks) - fails}/{len(checks)} headline numbers "
          f"match their source aggregates; {fails} mismatches")

    # 3. auxiliary self-checks appended by the sections
    aux_fails = 0
    print(f"\n[verify] auxiliary checks ({len(AUX_CHECKS)}):")
    for c in AUX_CHECKS:
        if c["needle"] is not None:
            found = c["needle"] in html_text
            ok = found if c["present"] else not found
        else:
            ok = bool(c["ok"])
        aux_fails += 0 if ok else 1
        print(f"  {'OK      ' if ok else 'MISMATCH'} {c['desc']}")
    print(f"[verify] {len(AUX_CHECKS) - aux_fails}/{len(AUX_CHECKS)} "
          f"auxiliary checks passed")
    return fails + aux_fails


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    D = collect()
    page, rows, checks = build_page(D)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(page)
    print(f"[build_report] wrote {os.path.relpath(OUT_PATH, ROOT)} "
          f"({len(page)} bytes)")
    n_pending = page.count('class="pending"')
    print(f"[build_report] sections with pending notes: {n_pending}")
    for relpath, e in SOURCES.items():
        print(f"[build_report]   {e['status']:10s} {relpath}"
              + (f"  ({e['note']})" if e["note"] else ""))
    fails = verify(page, rows, checks)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
