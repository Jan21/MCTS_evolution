"""Overlap/overflow linter for the report's inline SVG figures.

Parses every <svg> in a generated HTML page (or in a fragment passed as a
string), estimates a bounding box for every <text> element, and fails on:

  T1  two text boxes that overlap,
  T2  a text box that leaves the viewBox,
  T3  a text box that starts inside a <rect> but overflows it (container
      overflow),
  T4  a text element whose effective font size is below MIN_FONT px.

Text width is estimated as chars * font_size * CHAR_W and height as
font_size * 1.2 — deliberately conservative (CHAR_W 0.58 for the system-ui
stack), so a clean pass means real whitespace. Font sizes come from the CLASS
MAP below, which mirrors the page CSS; unknown classes assume 12 px.

Usage:
    python report/svg_lint.py report/selfplay.html      # lint a page
    from svg_lint import lint_html; errs = lint_html(html_string)

Run by gen_report.py after every page build so figure regressions fail loudly.
"""
from __future__ import annotations

import re
import sys

CHAR_W = 0.58
LINE_H = 1.2
MIN_FONT = 12.0
# class -> (font-size px, bold?)  — keep in sync with the page CSS
FONT_OF = {
    "nt": (13, True), "ns": (12, False), "nb": (12, False), "lbl": (12, True),
    "cap": (12, False), "axl": (12, False), "good-t": (12, True),
    "bad-t": (12, True), "ptl": (11, False), "refl-t": (10, False),
}
DEFAULT_FONT = 12.0


def _attrs(tag: str) -> dict:
    return dict(re.findall(r'([a-zA-Z-]+)="([^"]*)"', tag))


def _text_boxes(svg: str):
    """[(x0, y0, x1, y1, snippet, font)] for every <text> element."""
    boxes = []
    for m in re.finditer(r'<text\b([^>]*)>(.*?)</text>', svg, re.S):
        a = _attrs("<text" + m.group(1) + ">")
        raw = re.sub(r"<[^>]+>", "", m.group(2))
        txt = re.sub(r"&[#a-zA-Z0-9]+;", "0", raw).strip()
        if not txt:
            continue
        try:
            x, y = float(a.get("x", 0)), float(a.get("y", 0))
        except ValueError:
            continue
        classes = a.get("class", "").split()
        font = DEFAULT_FONT
        for c in classes:
            if c in FONT_OF:
                font = float(FONT_OF[c][0])
                break
        w = len(txt) * font * CHAR_W
        anchor = a.get("text-anchor", "")
        if anchor == "middle" or "middle" in m.group(1):
            x0 = x - w / 2
        elif anchor == "end":
            x0 = x - w
        else:
            x0 = x
        # SVG text y is the baseline: box rises ~0.8*font above, ~0.25 below
        y0, y1 = y - font * 0.95, y + font * 0.25
        boxes.append((x0, y0, x0 + w, y1, txt[:44], font))
    return boxes


def _rects(svg: str):
    out = []
    for m in re.finditer(r"<rect\b[^>]*>", svg):
        a = _attrs(m.group(0))
        try:
            out.append((float(a["x"]), float(a["y"]),
                        float(a["x"]) + float(a["width"]),
                        float(a["y"]) + float(a["height"]),
                        a.get("class", "")))
        except (KeyError, ValueError):
            continue
    return out


def _overlap(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return min(w, h) if (w > 0 and h > 0) else 0.0


def lint_svg(svg: str, name: str) -> list[str]:
    errs = []
    vb = re.search(r'viewBox="([\d.\s-]+)"', svg)
    if not vb:
        return [f"{name}: missing viewBox"]
    vx, vy, vw, vh = (float(v) for v in vb.group(1).split())
    texts = _text_boxes(svg)
    rects = _rects(svg)
    for i, t in enumerate(texts):
        if t[5] < MIN_FONT:
            errs.append(f"{name}: T4 font {t[5]:.0f}px < {MIN_FONT:.0f}px: "
                        f"'{t[4]}'")
        if t[0] < vx - 1 or t[2] > vx + vw + 1 or t[1] < vy - 1 \
                or t[3] > vy + vh + 1:
            errs.append(f"{name}: T2 text leaves viewBox: '{t[4]}' "
                        f"[{t[0]:.0f},{t[1]:.0f},{t[2]:.0f},{t[3]:.0f}]")
        for r in rects:
            # container overflow: anchor midpoint inside the rect, box outside
            cx, cy = (t[0] + t[2]) / 2, (t[1] + t[3]) / 2
            if r[0] < cx < r[2] and r[1] < cy < r[3]:
                if t[0] < r[0] - 1 or t[2] > r[2] + 1 or t[1] < r[1] - 1 \
                        or t[3] > r[3] + 1:
                    errs.append(f"{name}: T3 text overflows its box "
                                f"[{r[0]:.0f},{r[1]:.0f}..{r[2]:.0f},{r[3]:.0f}]: "
                                f"'{t[4]}'")
                break
        for j in range(i + 1, len(texts)):
            u = texts[j]
            if _overlap(t, u) > 1.0:
                errs.append(f"{name}: T1 text overlap: '{t[4]}' vs '{u[4]}'")
    return errs


def lint_html(src: str, only_story: bool = False) -> list[str]:
    errs = []
    if only_story:
        m = re.search(r'<section id="story".*?</section>', src, re.S)
        src = m.group(0) if m else src
    for k, m in enumerate(re.finditer(r"<svg\b.*?</svg>", src, re.S)):
        svg = m.group(0)
        label = re.search(r'aria-label="([^"]{0,38})', svg)
        name = f"svg#{k}({label.group(1)[:30] if label else '?'}...)"
        errs.extend(lint_svg(svg, name))
    return errs


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "selfplay.html"
    only_story = "--story" in sys.argv
    errs = lint_html(open(path).read(), only_story=only_story)
    for e in errs:
        print("SVGLINT", e)
    print(f"svg_lint: {'CLEAN' if not errs else f'{len(errs)} issue(s)'}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
