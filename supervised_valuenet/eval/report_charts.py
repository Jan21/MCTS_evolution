"""Inline-SVG chart primitives for the report.

Design rules (dataviz skill, applied throughout):
  * thin marks (columns <= 24px), 4px rounded data-end, square at baseline
  * hairline solid gridlines/axes in --grid / --baseline, recessive
  * direct value labels in text tokens (never in the series color)
  * legend for >= 2 series; single series charts carry no legend box
  * every chart ships a table twin (<details>) so no value is color-gated
  * hover tooltips via data-tt payloads (report_theme.JS renders them)

All colors are CSS variables so both themes work from one SVG.
"""

from eval.report_util import esc

FAMILY_VAR = {"bwd": "var(--c-bwd)", "bwd-old": "var(--c-bwd-old)",
              "fwd": "var(--c-fwd)", "seq": "var(--seq4)"}
FAMILY_LABEL = {"bwd": "subgoal planner — full plan language",
                "bwd-old": "subgoal planner — original plan language",
                "fwd": "move-by-move planner"}


def _nice_ceiling(v):
    if v <= 0:
        return 1.0
    for c in (1, 2, 2.5, 4, 5, 8, 10, 12.5, 20, 25, 40, 50, 80, 100,
              125, 200, 250, 400, 500, 800, 1000, 1250, 2000, 2500):
        if v <= c:
            return float(c)
    import math
    m = 10 ** math.floor(math.log10(v))
    for k in (1, 2, 2.5, 5, 10):
        if v <= k * m:
            return k * m
    return v


def _ticks(vmax, n=4):
    step = vmax / n
    return [round(step * i, 6) for i in range(n + 1)]


def _fmt_tick(v):
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:g}"


def col_path(x, y, w, h, r=4):
    """Column with rounded TOP corners only (data end), square baseline."""
    if h <= r:
        r = max(0.0, h * 0.5)
    return (f'M{x:.1f},{y + h:.1f} L{x:.1f},{y + r:.1f} '
            f'Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} '
            f'L{x + w - r:.1f},{y:.1f} '
            f'Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} '
            f'L{x + w:.1f},{y + h:.1f} Z')


def bar_path(x, y, w, h, r=4):
    """Horizontal bar with rounded RIGHT end (data end), square at baseline."""
    if w <= r:
        r = max(0.0, w * 0.5)
    return (f'M{x:.1f},{y:.1f} L{x + w - r:.1f},{y:.1f} '
            f'Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} '
            f'L{x + w:.1f},{y + h - r:.1f} '
            f'Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} '
            f'L{x:.1f},{y + h:.1f} Z')


def legend(series):
    """series: list of (family, label). Rendered above the plot."""
    keys = []
    for fam, label in series:
        keys.append(
            f'<span class="key"><span class="swatch" '
            f'style="background:{FAMILY_VAR[fam]}"></span>{esc(label)}</span>')
    return f'<div class="legend">{"".join(keys)}</div>'


def line_legend(series):
    keys = []
    for fam, label in series:
        keys.append(
            f'<span class="key"><span class="lswatch" '
            f'style="background:{FAMILY_VAR[fam]}"></span>{esc(label)}</span>')
    return f'<div class="legend">{"".join(keys)}</div>'


def table_twin(head, rows, summary="chart data as a table"):
    """The WCAG-clean twin of a chart. head: list of column names;
    rows: list of lists (already formatted strings)."""
    h = "".join(f'<th class="num">{esc(c)}</th>' if i else f"<th>{esc(c)}</th>"
                for i, c in enumerate(head))
    body = []
    for r in rows:
        tds = [f"<td>{r[0]}</td>"] + [
            f'<td class="num">{c}</td>' for c in r[1:]]
        body.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<details class="tabletwin"><summary class="small muted">'
            f'{esc(summary)}</summary><div class="scroll"><table>'
            f'<thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody>'
            f"</table></div></details>")


def figure(title, subtitle, legend_html, svg, caption, source, twin=""):
    src = f'<span class="src">source: {esc(source)}</span>' if source else ""
    sub = f'<p class="chartsub">{subtitle}</p>' if subtitle else ""
    return (f'<figure class="chart"><p class="charthead">{title}</p>{sub}'
            f'{legend_html}{svg}'
            f"<figcaption>{caption}{src}</figcaption>{twin}</figure>")


# ---------------------------------------------------------------------------
# Grouped column chart
# ---------------------------------------------------------------------------

def grouped_columns(groups, series, unit="%", vmax=None, height=250,
                    aria="", gap_note="not measured", pct=True,
                    label_dec=1):
    """groups: list of dicts {label, sub, values: {series_key: float|None},
    tips: optional {series_key: str} extra tooltip line}.
    series: list of (key, family, label).
    Values of None render as a small 'not measured' marker.
    Returns svg string.
    """
    n_g = len(groups)
    n_s = len(series)
    colw = min(24, max(14, 150 // n_s))
    gap_in = 6           # gap between columns inside a group
    pad_l, pad_r, pad_t, pad_b = 46, 10, 18, 40
    group_w = n_s * colw + (n_s - 1) * gap_in
    gap_out = max(26, group_w // 2)
    # spread groups out toward a comfortable width (extra space goes to the
    # gaps, never to the marks)
    target_w = 720
    if n_g > 1:
        need_gap = (target_w - pad_l - pad_r - n_g * group_w) // (n_g - 1)
        gap_out = max(gap_out, need_gap)
    W = pad_l + pad_r + n_g * group_w + (n_g - 1) * gap_out
    H = height + pad_t + pad_b
    plot_h = height
    allv = [v for g in groups for v in g["values"].values() if v is not None]
    if vmax is None:
        vmax = _nice_ceiling(max(allv) * (1.12 if not pct else 1.0)) \
            if allv else 1.0
    if pct:
        vmax = 100.0
    parts = [f'<svg viewBox="0 0 {W} {H}" style="max-width:{min(int(W * 1.3), 1000)}px" role="img" aria-label="{esc(aria)}" '
             f'preserveAspectRatio="xMidYMid meet">']
    # gridlines + y ticks
    for tv in _ticks(vmax):
        y = pad_t + plot_h * (1 - tv / vmax)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W - pad_r}" '
                     f'y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 7}" y="{y + 3.5:.1f}" '
                     f'text-anchor="end" font-size="10.5" '
                     f'fill="var(--faint)">{_fmt_tick(tv)}{unit if pct else ""}'
                     f"</text>")
    # baseline
    ybase = pad_t + plot_h
    parts.append(f'<line x1="{pad_l}" y1="{ybase}" x2="{W - pad_r}" '
                 f'y2="{ybase}" stroke="var(--baseline)" stroke-width="1"/>')
    for gi, g in enumerate(groups):
        gx = pad_l + gi * (group_w + gap_out)
        # tooltip payload for the whole group
        tip_lines = [g["label"] + (f' — {g["sub"]}' if g.get("sub") else "")]
        for key, fam, slabel in series:
            v = g["values"].get(key)
            extra = (g.get("tips") or {}).get(key, "")
            val = (f"{v:.{label_dec}f}{unit}" if v is not None else gap_note)
            tip_lines.append(f"{slabel}\t{val}" + (f" {extra}" if extra else ""))
        tip = esc("\n".join(tip_lines))
        for si, (key, fam, slabel) in enumerate(series):
            v = g["values"].get(key)
            x = gx + si * (colw + gap_in)
            if v is None:
                # honest gap: a dashed placeholder on the baseline
                parts.append(
                    f'<line x1="{x + 2}" y1="{ybase - 1.5}" '
                    f'x2="{x + colw - 2}" y2="{ybase - 1.5}" '
                    f'stroke="var(--faint)" stroke-width="2.5" '
                    f'stroke-dasharray="3 3"/>')
                parts.append(
                    f'<text x="{x + colw / 2 + 3:.1f}" y="{ybase - 10:.1f}" '
                    f'font-size="9" fill="var(--faint)" '
                    f'transform="rotate(-90 {x + colw / 2 + 3:.1f} '
                    f'{ybase - 10:.1f})">{esc(gap_note)}</text>')
                continue
            h = plot_h * (v / vmax)
            y = pad_t + plot_h - h
            parts.append(f'<path d="{col_path(x, y, colw, h)}" '
                         f'fill="{FAMILY_VAR[fam]}"/>')
            # value on the cap (text token, never series color)
            parts.append(f'<text x="{x + colw / 2:.1f}" y="{y - 4:.1f}" '
                         f'text-anchor="middle" font-size="10.5" '
                         f'font-weight="600" fill="var(--ink)">'
                         f"{v:.{label_dec}f}</text>")
        # group hit target + labels (clamped to the canvas)
        hx0 = max(0.0, gx - gap_out / 2)
        hx1 = min(float(W), gx + group_w + gap_out / 2)
        parts.append(f'<rect x="{hx0:.1f}" y="{pad_t}" '
                     f'width="{hx1 - hx0:.1f}" height="{plot_h + 30}" '
                     f'fill="transparent" data-tt="{tip}" tabindex="0" '
                     f'role="img" aria-label="{tip}"/>')
        parts.append(f'<text x="{gx + group_w / 2:.1f}" y="{ybase + 16}" '
                     f'text-anchor="middle" font-size="11" '
                     f'fill="var(--muted)">{esc(g["label"])}</text>')
        if g.get("sub"):
            parts.append(f'<text x="{gx + group_w / 2:.1f}" y="{ybase + 29}" '
                         f'text-anchor="middle" font-size="9.5" '
                         f'fill="var(--faint)">{esc(g["sub"])}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Horizontal grouped bars (for wide-range magnitudes like search steps)
# ---------------------------------------------------------------------------

def grouped_hbars(groups, series, unit="", vmax=None, aria="",
                  label_fmt="{:.0f}", tick_unit=""):
    """groups: {label, sub, values {key: float|None}}; series: (key,fam,label)."""
    n_s = len(series)
    barh = 16
    gap_in = 4
    group_h = n_s * barh + (n_s - 1) * gap_in
    gap_out = 22
    pad_l, pad_r, pad_t, pad_b = 118, 78, 8, 26
    W = 880
    H = pad_t + pad_b + len(groups) * group_h + (len(groups) - 1) * gap_out
    plot_w = W - pad_l - pad_r
    allv = [v for g in groups for v in g["values"].values() if v is not None]
    if vmax is None:
        vmax = _nice_ceiling(max(allv)) if allv else 1.0
    parts = [f'<svg viewBox="0 0 {W} {H}" style="max-width:{min(int(W * 1.3), 1000)}px" role="img" aria-label="{esc(aria)}" '
             f'preserveAspectRatio="xMidYMid meet">']
    for tv in _ticks(vmax):
        x = pad_l + plot_w * tv / vmax
        parts.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" '
                     f'y2="{H - pad_b}" stroke="var(--grid)" '
                     f'stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{H - pad_b + 14}" '
                     f'text-anchor="middle" font-size="10.5" '
                     f'fill="var(--faint)">{_fmt_tick(tv)}{tick_unit}</text>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" '
                 f'y2="{H - pad_b}" stroke="var(--baseline)" '
                 f'stroke-width="1"/>')
    y = pad_t
    for g in groups:
        tip_lines = [g["label"] + (f' — {g["sub"]}' if g.get("sub") else "")]
        for key, fam, slabel in series:
            v = g["values"].get(key)
            tip_lines.append(
                f"{slabel}\t" + (label_fmt.format(v) + unit
                                 if v is not None else "not measured"))
        tip = esc("\n".join(tip_lines))
        hy0 = max(0.0, y - gap_out / 2)
        hy1 = min(float(H), y + group_h + gap_out / 2)
        parts.append(f'<rect x="0" y="{hy0:.1f}" width="{W}" '
                     f'height="{hy1 - hy0:.1f}" fill="transparent" '
                     f'data-tt="{tip}" tabindex="0" role="img" '
                     f'aria-label="{tip}"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{y + group_h / 2 + 4:.1f}" '
                     f'text-anchor="end" font-size="11.5" '
                     f'fill="var(--ink)">{esc(g["label"])}</text>')
        for si, (key, fam, slabel) in enumerate(series):
            v = g["values"].get(key)
            by = y + si * (barh + gap_in)
            if v is None:
                parts.append(f'<text x="{pad_l + 6}" y="{by + barh - 4}" '
                             f'font-size="10" fill="var(--faint)">'
                             f"not measured</text>")
                continue
            w = plot_w * v / vmax
            parts.append(f'<path d="{bar_path(pad_l, by, w, barh)}" '
                         f'fill="{FAMILY_VAR[fam]}"/>')
            parts.append(f'<text x="{pad_l + w + 6:.1f}" '
                         f'y="{by + barh - 4:.1f}" font-size="10.5" '
                         f'font-weight="600" fill="var(--ink)">'
                         f"{label_fmt.format(v)}{unit}</text>")
        y += group_h + gap_out
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Single-series columns (oracle-death curve etc.)
# ---------------------------------------------------------------------------

def columns_single(points, unit="%", vmax=100.0, aria="", fam="bwd",
                   height=210, label_dec=1, sub_key="sub"):
    """points: list of {label, sub, value, tip}."""
    groups = [{"label": p["label"], "sub": p.get(sub_key),
               "values": {"v": p["value"]},
               "tips": {"v": p.get("tip", "")}} for p in points]
    return grouped_columns(groups, [("v", fam, "value")], unit=unit,
                           vmax=vmax, height=height, aria=aria,
                           pct=(unit == "%"), label_dec=label_dec)


# ---------------------------------------------------------------------------
# Line chart on ordinal x positions (budget curves, self-play iterations)
# ---------------------------------------------------------------------------

def line_chart(xlabels, series, unit="%", aria="", height=240,
               ymax=100.0, xtitle="", label_dec=1, end_labels=True):
    """xlabels: list of x tick labels (ordinal positions, evenly spaced).
    series: list of dicts {key, fam, label, values: [float|None per x]}.
    """
    pad_l, pad_r, pad_t, pad_b = 46, 150 if end_labels else 16, 14, 44
    W = 880
    H = height + pad_t + pad_b
    plot_w = W - pad_l - pad_r
    plot_h = height
    n = len(xlabels)
    xs = [pad_l + plot_w * (i / (n - 1) if n > 1 else 0.5) for i in range(n)]

    def Y(v):
        return pad_t + plot_h * (1 - v / ymax)

    parts = [f'<svg viewBox="0 0 {W} {H}" style="max-width:{min(int(W * 1.3), 1000)}px" role="img" aria-label="{esc(aria)}" '
             f'preserveAspectRatio="xMidYMid meet">']
    for tv in _ticks(ymax):
        y = Y(tv)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W - pad_r}" '
                     f'y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 7}" y="{y + 3.5:.1f}" '
                     f'text-anchor="end" font-size="10.5" '
                     f'fill="var(--faint)">{_fmt_tick(tv)}{unit}</text>')
    ybase = pad_t + plot_h
    parts.append(f'<line x1="{pad_l}" y1="{ybase}" x2="{W - pad_r}" '
                 f'y2="{ybase}" stroke="var(--baseline)" stroke-width="1"/>')
    for i, xl in enumerate(xlabels):
        parts.append(f'<text x="{xs[i]:.1f}" y="{ybase + 17}" '
                     f'text-anchor="middle" font-size="10.5" '
                     f'fill="var(--muted)">{esc(str(xl))}</text>')
    if xtitle:
        parts.append(f'<text x="{pad_l + plot_w / 2:.1f}" y="{H - 8}" '
                     f'text-anchor="middle" font-size="10.5" '
                     f'fill="var(--faint)">{esc(xtitle)}</text>')
    # lines + markers (2px line, >=8px marker with 2px surface ring)
    for s in series:
        pts = [(xs[i], Y(v)) for i, v in enumerate(s["values"])
               if v is not None]
        if len(pts) >= 2:
            d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            parts.append(f'<path d="{d}" fill="none" '
                         f'stroke="{FAMILY_VAR[s["fam"]]}" stroke-width="2" '
                         f'stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" '
                         f'fill="{FAMILY_VAR[s["fam"]]}" '
                         f'stroke="var(--surface)" stroke-width="2"/>')
        if end_labels and pts:
            ex, ey = pts[-1]
            lastv = [v for v in s["values"] if v is not None][-1]
            parts.append(f'<text x="{ex + 10:.1f}" y="{ey + 4:.1f}" '
                         f'font-size="10.5" fill="var(--muted)">'
                         f'<tspan font-weight="650" fill="var(--ink)">'
                         f"{lastv:.{label_dec}f}{unit}</tspan> "
                         f'{esc(s["label"])}</text>')
    # hover hit columns: one tooltip listing every series at that x
    col_w = plot_w / max(1, n - 1)
    for i, xl in enumerate(xlabels):
        lines = [str(xl) + (f" {xtitle}" if xtitle else "")]
        for s in series:
            v = s["values"][i]
            lines.append(f'{s["label"]}\t'
                         + (f"{v:.{label_dec}f}{unit}" if v is not None
                            else "—"))
        tip = esc("\n".join(lines))
        x0 = xs[i] - col_w / 2 if i else pad_l - 6
        x1 = xs[i] + col_w / 2 if i < n - 1 else xs[i] + 8
        parts.append(f'<rect x="{x0:.1f}" y="{pad_t}" '
                     f'width="{x1 - x0:.1f}" height="{plot_h + 20}" '
                     f'fill="transparent" data-tt="{tip}" tabindex="0" '
                     f'role="img" aria-label="{tip}"/>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Dense budget curve (solve rate vs budget, log-x) — one small-multiple panel
# ---------------------------------------------------------------------------

def budget_curve_panel(series, aria="", width=430, height=170,
                       xmax=1200, hover_budgets=(10, 30, 100, 300, 1200)):
    """series: list of {fam, label, points: [(budget, rate_pct), ...]}.
    Log-x from 1 to xmax; dense step curves are downsampled for the path;
    hover columns at the canonical budgets list every series."""
    import math
    pad_l, pad_r, pad_t, pad_b = 40, 12, 8, 30
    W, H = width, height + pad_t + pad_b
    plot_w, plot_h = W - pad_l - pad_r, height

    def X(b):
        b = max(1.0, float(b))
        return pad_l + plot_w * math.log(b) / math.log(xmax)

    def Y(r):
        return pad_t + plot_h * (1 - r / 100.0)

    parts = [f'<svg viewBox="0 0 {W} {H}" '
             f'style="max-width:{int(W * 1.15)}px" role="img" '
             f'aria-label="{esc(aria)}" preserveAspectRatio="xMidYMid meet">']
    for tv in (0, 50, 100):
        y = Y(tv)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W - pad_r}" '
                     f'y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{y + 3.5:.1f}" '
                     f'text-anchor="end" font-size="9.5" '
                     f'fill="var(--faint)">{tv}%</text>')
    for tb in (1, 10, 100, 1200):
        x = X(tb)
        parts.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" '
                     f'y2="{pad_t + plot_h}" stroke="var(--grid)" '
                     f'stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{pad_t + plot_h + 14}" '
                     f'text-anchor="middle" font-size="9.5" '
                     f'fill="var(--faint)">{tb}</text>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t + plot_h}" '
                 f'x2="{W - pad_r}" y2="{pad_t + plot_h}" '
                 f'stroke="var(--baseline)" stroke-width="1"/>')
    parts.append(f'<text x="{pad_l + plot_w / 2:.1f}" y="{H - 4}" '
                 f'text-anchor="middle" font-size="9.5" '
                 f'fill="var(--faint)">search-step budget (log scale)</text>')
    for s in series:
        pts = sorted((b, r) for b, r in s["points"] if b and b <= xmax)
        if not pts:
            continue
        # downsample: keep points where the x-position or rate moves visibly
        kept, last = [], None
        for b, r in pts:
            if last is None or X(b) - X(last[0]) > 2.5 \
                    or abs(r - last[1]) > 0.4:
                kept.append((b, r))
                last = (b, r)
        if kept[-1] != pts[-1]:
            kept.append(pts[-1])
        # step curve: horizontal-then-up segments
        d = [f"M{X(kept[0][0]):.1f},{Y(kept[0][1]):.1f}"]
        for (b0, r0), (b1, r1) in zip(kept, kept[1:]):
            d.append(f"L{X(b1):.1f},{Y(r0):.1f}")
            d.append(f"L{X(b1):.1f},{Y(r1):.1f}")
        parts.append(f'<path d="{" ".join(d)}" fill="none" '
                     f'stroke="{FAMILY_VAR[s["fam"]]}" stroke-width="2" '
                     f'stroke-linejoin="round"/>')
        bE, rE = kept[-1]
        parts.append(f'<circle cx="{X(bE):.1f}" cy="{Y(rE):.1f}" r="3.5" '
                     f'fill="{FAMILY_VAR[s["fam"]]}" '
                     f'stroke="var(--surface)" stroke-width="2"/>')
    # hover columns at canonical budgets
    hb = [b for b in hover_budgets if b <= xmax]
    for i, b in enumerate(hb):
        lines = [f"budget {b} steps"]
        for s in series:
            r = _rate_at(s["points"], b)
            lines.append(f'{s["label"]}\t'
                         + (f"{r:.1f}%" if r is not None else "—"))
        tip = esc("\n".join(lines))
        x0 = X(hb[i - 1]) if i else pad_l
        x1 = X(hb[i + 1]) if i < len(hb) - 1 else W - pad_r
        xm0, xm1 = (x0 + X(b)) / 2 if i else x0, (X(b) + x1) / 2 \
            if i < len(hb) - 1 else x1
        parts.append(f'<rect x="{xm0:.1f}" y="{pad_t}" '
                     f'width="{xm1 - xm0:.1f}" height="{plot_h + 16}" '
                     f'fill="transparent" data-tt="{tip}" tabindex="0" '
                     f'role="img" aria-label="{tip}"/>')
    parts.append("</svg>")
    return "".join(parts)


def _rate_at(points, budget):
    """Solve rate at a given budget from a (budget, rate) step curve."""
    best = None
    for b, r in sorted(points):
        if b <= budget:
            best = r
        else:
            break
    return best


# ---------------------------------------------------------------------------
# Ceiling meter: achieved fill vs permitted-ceiling track
# ---------------------------------------------------------------------------

def ceiling_meters(rows, aria=""):
    """rows: list of {label, sub, achieved, ceiling, ach_txt, ceil_txt,
    ceil_note}. Track = lighter step of the same (blue) ramp per the meter
    spec; fill = achieved; tick = ceiling."""
    barh, gap = 30, 44
    pad_l, pad_r, pad_t, pad_b = 218, 190, 8, 26
    W = 880
    H = pad_t + pad_b + len(rows) * (barh + gap) - gap
    plot_w = W - pad_l - pad_r
    parts = [f'<svg viewBox="0 0 {W} {H}" style="max-width:{min(int(W * 1.3), 1000)}px" role="img" '
             f'aria-label="{esc(aria)}" preserveAspectRatio="xMidYMid meet">']
    for tv in (0, 25, 50, 75, 100):
        x = pad_l + plot_w * tv / 100
        parts.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" '
                     f'y2="{H - pad_b}" stroke="var(--grid)" '
                     f'stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{H - pad_b + 14}" '
                     f'text-anchor="middle" font-size="10.5" '
                     f'fill="var(--faint)">{tv}%</text>')
    y = pad_t
    for r in rows:
        tip = esc(f'{r["label"]}\n'
                  f'networks achieve\t{r["ach_txt"]}\n'
                  f'language permits\t{r["ceil_txt"]}')
        parts.append(f'<rect x="0" y="{y - 6}" width="{W}" '
                     f'height="{barh + 12}" fill="transparent" '
                     f'data-tt="{tip}" tabindex="0" role="img" '
                     f'aria-label="{tip}"/>')
        parts.append(f'<text x="{pad_l - 10}" y="{y + 13}" text-anchor="end" '
                     f'font-size="11.5" font-weight="600" fill="var(--ink)">'
                     f'{esc(r["label"])}</text>')
        if r.get("sub"):
            parts.append(f'<text x="{pad_l - 10}" y="{y + 26}" '
                         f'text-anchor="end" font-size="10" '
                         f'fill="var(--faint)">{esc(r["sub"])}</text>')
        # track to the ceiling (lighter step of the same ramp)
        cw = plot_w * r["ceiling"] / 100
        parts.append(f'<path d="{bar_path(pad_l, y, cw, barh, 5)}" '
                     f'fill="var(--seq2)"/>')
        # achieved fill
        aw = plot_w * r["achieved"] / 100
        parts.append(f'<path d="{bar_path(pad_l, y + 4, aw, barh - 8, 4)}" '
                     f'fill="var(--c-bwd)"/>')
        # ceiling tick + label (clamped so nothing leaves the viewBox)
        parts.append(f'<line x1="{pad_l + cw:.1f}" y1="{y - 4}" '
                     f'x2="{pad_l + cw:.1f}" y2="{y + barh + 4}" '
                     f'stroke="var(--ink)" stroke-width="2"/>')
        lx = min(pad_l + cw + 7, W - 10 - len(r["ceil_txt"]) * 7.0)
        parts.append(f'<text x="{lx:.1f}" y="{y + 12:.1f}" '
                     f'font-size="10.5" font-weight="600" fill="var(--ink)">'
                     f'{esc(r["ceil_txt"])}</text>')
        note = r.get("ceil_note", "language permits")
        nx = min(pad_l + cw + 7, W - 10 - len(note) * 6.0)
        ny = y + 24
        parts.append(f'<text x="{nx:.1f}" y="{ny:.1f}" '
                     f'font-size="9.5" fill="var(--faint)">'
                     f"{esc(note)}</text>")
        # achieved label inside the fill if it fits, else outside
        atxt = r["ach_txt"]
        est_w = len(atxt) * 6.5 + 14
        if aw > est_w:
            parts.append(f'<text x="{pad_l + aw - 7:.1f}" y="{y + 19:.1f}" '
                         f'text-anchor="end" font-size="10.5" '
                         f'font-weight="650" fill="#ffffff">{esc(atxt)}</text>')
        else:
            parts.append(f'<text x="{pad_l + 7:.1f}" y="{y + 19:.1f}" '
                         f'font-size="10.5" font-weight="650" '
                         f'fill="var(--ink)">{esc(atxt)}</text>')
        y += barh + gap
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# One stacked horizontal bar (part-to-whole, e.g. the 10x oracle probe)
# ---------------------------------------------------------------------------

def stacked_bar(segments, total_label, aria=""):
    """segments: list of {label, value, var} (value in absolute units)."""
    total = sum(s["value"] for s in segments)
    barh = 34
    pad_l, pad_r, pad_t, pad_b = 8, 8, 6, 46
    W = 880
    H = barh + pad_t + pad_b
    plot_w = W - pad_l - pad_r
    parts = [f'<svg viewBox="0 0 {W} {H}" style="max-width:{min(int(W * 1.3), 1000)}px" role="img" '
             f'aria-label="{esc(aria)}" preserveAspectRatio="xMidYMid meet">']
    x = pad_l
    for i, s in enumerate(segments):
        w = plot_w * s["value"] / total - (2 if i < len(segments) - 1 else 0)
        tip = esc(f'{total_label}\n{s["label"]}\t{s["value"]} of {total}')
        parts.append(f'<rect x="{x:.1f}" y="{pad_t}" width="{w:.1f}" '
                     f'height="{barh}" rx="4" fill="{s["var"]}" '
                     f'data-tt="{tip}" tabindex="0" role="img" '
                     f'aria-label="{tip}"/>')
        # label under the segment
        parts.append(f'<text x="{x + w / 2:.1f}" y="{pad_t + barh + 16}" '
                     f'text-anchor="middle" font-size="10.5" '
                     f'font-weight="600" fill="var(--ink)">{s["value"]}</text>')
        parts.append(f'<text x="{x + w / 2:.1f}" y="{pad_t + barh + 30}" '
                     f'text-anchor="middle" font-size="9.5" '
                     f'fill="var(--faint)">{esc(s["label"])}</text>')
        x += w + 2
    parts.append("</svg>")
    return "".join(parts)
