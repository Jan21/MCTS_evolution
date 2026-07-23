"""Board drawings (inline SVG, theme-aware via CSS variables).

Ported from the previous generator with the same coordinate conventions:
walls_right = set of (x, y) cells with a wall on their right edge;
walls_down = wall under the cell. Robots use the project-wide slot order.
"""

import math

from eval.report_util import esc

SLOT_NAMES = ("Red", "Blue", "Green", "Yellow",
              "Purple", "Orange", "Cyan", "Pink")
SLOT_VARS = ("--r-red", "--r-blue", "--r-green", "--r-yellow",
             "--r-red", "--r-blue", "--r-green", "--r-yellow")


def board_svg(size, walls_right, walls_down, robots, arrows=(), rings=(),
              marks=(), aria="", max_px=430):
    """robots: (x, y, cssvar, letter, is_target); arrows: (x0,y0,x1,y1,var,
    ordinal); rings: (x,y,var) goal cells; marks: (x,y,text) annotations."""
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

    for i in range(1, size):
        parts.append(f'<line x1="{pad + i * c}" y1="{pad}" x2="{pad + i * c}" '
                     f'y2="{H - pad}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<line x1="{pad}" y1="{pad + i * c}" x2="{W - pad}" '
                     f'y2="{pad + i * c}" stroke="var(--grid)" stroke-width="1"/>')
    parts.append(f'<rect x="{pad}" y="{pad}" width="{size * c}" '
                 f'height="{size * c}" fill="none" stroke="var(--ink)" '
                 f'stroke-width="3"/>')
    for (x, y) in sorted(walls_right):
        X = pad + (x + 1) * c
        parts.append(f'<line x1="{X}" y1="{pad + y * c}" x2="{X}" '
                     f'y2="{pad + (y + 1) * c}" stroke="var(--ink)" '
                     f'stroke-width="3" stroke-linecap="round"/>')
    for (x, y) in sorted(walls_down):
        Y = pad + (y + 1) * c
        parts.append(f'<line x1="{pad + x * c}" y1="{Y}" '
                     f'x2="{pad + (x + 1) * c}" y2="{Y}" stroke="var(--ink)" '
                     f'stroke-width="3" stroke-linecap="round"/>')
    for (x, y, txt) in marks:
        parts.append(
            f'<rect x="{pad + x * c + 2}" y="{pad + y * c + 2}" '
            f'width="{c - 4}" height="{c - 4}" rx="4" fill="none" '
            f'stroke="var(--muted)" stroke-width="1.6" '
            f'stroke-dasharray="3.5 2.5"><title>{esc(txt)}</title></rect>')
    for (x, y, var) in rings:
        parts.append(
            f'<rect x="{pad + x * c + 3}" y="{pad + y * c + 3}" '
            f'width="{c - 6}" height="{c - 6}" rx="5" fill="none" '
            f'stroke="var({var})" stroke-width="2.2" stroke-dasharray="4 3"/>')

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
        if x0 == x1:
            X0, Y0, X1, Y1 = cx(x0) + o, cy(y0), cx(x1) + o, cy(y1)
        else:
            X0, Y0, X1, Y1 = cx(x0), cy(y0) + o, cx(x1), cy(y1) + o
        dx, dy = X1 - X0, Y1 - Y0
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        X0s, Y0s = X0 + ux * 10, Y0 + uy * 10
        X1s, Y1s = X1 - ux * 4, Y1 - uy * 4
        parts.append(
            f'<line x1="{X0s:.1f}" y1="{Y0s:.1f}" x2="{X1s:.1f}" '
            f'y2="{Y1s:.1f}" stroke="var({var})" stroke-width="2.6" '
            f'stroke-linecap="round" opacity="0.85"/>')
        hx, hy = X1s, Y1s
        px, py = -uy, ux
        parts.append(
            f'<path d="M{hx:.1f},{hy:.1f} '
            f'L{hx - ux * 8 + px * 4.5:.1f},{hy - uy * 8 + py * 4.5:.1f} '
            f'L{hx - ux * 8 - px * 4.5:.1f},{hy - uy * 8 - py * 4.5:.1f} Z" '
            f'fill="var({var})" opacity="0.85"/>')
        if ordinal:
            bx, by = X0, Y0
            parts.append(
                f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="8.5" '
                f'fill="var(--surface)" stroke="var({var})" stroke-width="2"/>'
                f'<text x="{bx:.1f}" y="{by + 3.6:.1f}" text-anchor="middle" '
                f'fill="var(--ink)" font-size="10.5" '
                f'font-weight="700">{ordinal}</text>')
    for (x, y, var, letter, is_target) in robots:
        ring = (f'<circle cx="{cx(x):.1f}" cy="{cy(y):.1f}" r="11.5" '
                f'fill="none" stroke="var({var})" stroke-width="1.6" '
                f'opacity="0.55"/>' if is_target else "")
        parts.append(
            f'{ring}<circle cx="{cx(x):.1f}" cy="{cy(y):.1f}" r="8.5" '
            f'fill="var({var})" stroke="var(--surface)" stroke-width="1.6"/>'
            f'<text x="{cx(x):.1f}" y="{cy(y) + 3.4:.1f}" '
            f'text-anchor="middle" fill="var(--bg)" font-size="9.5" '
            f'font-weight="800">{esc(letter)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def slide_rule_svg():
    """Static schematic of how one move works — explanatory art only,
    contains no measured numbers."""
    size = 8
    wr = {(4, 6)}
    wd = set()
    robots = [
        (1, 1, "--r-red", "R", False),
        (5, 6, "--r-blue", "B", False),
        (5, 0, "--r-yellow", "Y", True),
    ]
    arrows = [
        (1, 1, 7, 1, "--r-red", None),
        (5, 0, 5, 5, "--r-yellow", None),
    ]
    rings = [(5, 5, "--r-yellow")]
    return board_svg(size, wr, wd, robots, arrows, rings,
                     aria="Schematic board: a red robot slides right until "
                          "the border wall stops it; a yellow robot slides "
                          "down and stops in the cell just above a blue "
                          "robot, landing on its goal cell.",
                     max_px=300)
