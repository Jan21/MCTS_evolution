"""Result-table builders. Every numeric cell is registered with the check
registry (report_util.ck) so the verifier can parse it back out of the final
HTML and compare it against the source aggregate."""

from eval.report_util import esc, ck, fnum, ffrac, dot, scroll

MACHINE_SUP = {"karolina": "K", "origin": "O"}


def solve_cell(agg, src, desc, pct_dec=1):
    txt = f"{agg['solved']}/{agg['n']} ({agg['solve_rate'] * 100:.{pct_dec}f}%)"
    return ck(txt, src, desc, raw=agg["solved"])


def num_cell(agg, key, dec, src, desc):
    v = agg.get(key)
    return ck(fnum(v, dec), src, desc, raw=v)


def machine_sup(row, mixed):
    if not mixed or not row.get("machine"):
        return ""
    return (f'<sup class="mnote" title="measured on the '
            f'{"Karolina cluster" if row["machine"] == "karolina" else "origin machine"}">'
            f'{MACHINE_SUP[row["machine"]]}</sup>')


def machine_legend(mixed):
    if not mixed:
        return ""
    return ('<p class="cellnote">Seconds per puzzle carry a machine tag — '
            '<sup>O</sup> origin machine, <sup>K</sup> Karolina cluster '
            '(the study moved on 2026-07-20). Wall-clock times are comparable '
            'only within the same tag; solve rates, search steps and move '
            'counts are machine-independent.</p>')


def sys_table(rows, frontier=False, table_id=None):
    """rows: list of dicts with keys
       label, family ('bwd'|'bwd-old'|'fwd'), sub, agg, src, hl(bool),
       note (cell note under n), machine ('karolina'|'origin'),
       pending_msg (renders a dash row when agg is None).
    frontier=True drops the optimality columns (placeholder optima)."""
    mixed = len({r.get("machine") for r in rows if r.get("machine")}) > 1
    if frontier:
        head = ("<tr><th>system</th><th class='num'>puzzles</th>"
                "<th class='num'>solved (playable)</th>"
                "<th class='num'>solution length (moves)"
                "<a class='fnref' href='#footnotes'>a</a></th>"
                "<th class='num'>search steps / puzzle</th>"
                "<th class='num'>seconds / puzzle</th></tr>")
    else:
        head = ("<tr><th>system</th><th class='num'>puzzles</th>"
                "<th class='num'>solved (playable)</th>"
                "<th class='num'>extra moves vs optimal</th>"
                "<th class='num'>% at optimum</th>"
                "<th class='num'>solution length (moves)</th>"
                "<th class='num'>search steps / puzzle</th>"
                "<th class='num'>seconds / puzzle</th></tr>")
    body = []
    ncols = 6 if frontier else 8
    for r in rows:
        sub = (f'<div class="cellnote">{r["sub"]}</div>') if r.get("sub") else ""
        syscell = (f'<td class="syscell">{dot(r["family"])}'
                   f'<b>{esc(r["label"])}</b>{sub}</td>')
        if r.get("agg") is None:
            body.append(
                f'<tr>{syscell}<td class="num" colspan="{ncols - 1}">'
                f'<span class="chip run">in progress</span> '
                f'<span class="small muted">{esc(r.get("pending_msg", ""))}'
                f"</span></td></tr>")
            continue
        a, src = r["agg"], r["src"]
        d = r["label"]
        note = (f'<div class="cellnote">{esc(r["note"])}</div>'
                if r.get("note") else "")
        hl = ' class="hl"' if r.get("hl") else ""
        cells = [syscell,
                 f'<td class="num">{a["n"]}{note}</td>',
                 f'<td class="num">{solve_cell(a, src, d + " — solved")}</td>']
        if not frontier:
            cells.append(f'<td class="num">'
                         f'{num_cell(a, "mean_regret", 3, src, d + " — extra moves")}</td>')
            cells.append(f'<td class="num">'
                         f'{num_cell(a, "pct_optimal", 1, src, d + " — % optimal")}</td>')
        cells.append(f'<td class="num">'
                     f'{num_cell(a, "mean_moves", 2, src, d + " — moves")}</td>')
        cells.append(f'<td class="num">'
                     f'{num_cell(a, "mean_expansions", 1, src, d + " — steps")}</td>')
        cells.append(f'<td class="num">'
                     f'{num_cell(a, "mean_seconds", 2, src, d + " — seconds")}'
                     f"{machine_sup(r, mixed)}</td>")
        body.append(f"<tr{hl}>" + "".join(cells) + "</tr>")
    tid = f' id="{table_id}"' if table_id else ""
    return (scroll(f"<table{tid}><thead>{head}</thead>"
                   f'<tbody>{"".join(body)}</tbody></table>')
            + machine_legend(mixed))
