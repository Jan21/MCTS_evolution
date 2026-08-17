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
            chip_txt = r.get("pending_chip", "in progress")
            chip_cls = "chip run" if chip_txt == "in progress" else "chip warn"
            body.append(
                f'<tr>{syscell}<td class="num" colspan="{ncols - 1}">'
                f'<span class="{chip_cls}">{esc(chip_txt)}</span> '
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


# ---------------------------------------------------------------------------
# The retraining story, rendered from report_data.retrain_verdict()
#
# Three places on the page used to assert, in fixed prose, that retraining
# "did not improve on" the zero-shot rows and had "landed at four of six
# configurations".  Both were false against the files.  Every one of those
# places now calls one of the two renderers below, so the claim is derived
# per rung and cannot drift again.
# ---------------------------------------------------------------------------

def _rung_list(rungs):
    names = [esc(r["short"]) for r in rungs]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _improved_clause(r):
    """'32×32 · 4r (97.7% gradable / 77.8% beyond-oracle vs 88.0% / 71.3%)'."""
    got, was = [], []
    for s in r["sets"]:
        d = f'retrain verdict {r["key"]} {s["set"]}'
        got.append(ck(f'{s["retrained"]["agg"]["solve_rate"] * 100:.1f}%',
                      s["retrained"]["src"], d + " — retrained",
                      raw=s["retrained"]["agg"]["solved"])
                   + f' {esc(s["set_label"])}')
        was.append(ck(f'{s["zeroshot"]["agg"]["solve_rate"] * 100:.1f}%',
                      s["zeroshot"]["src"], d + " — zero-shot",
                      raw=s["zeroshot"]["agg"]["solved"]))
    return (f'<b>{esc(r["short"])}</b> (' + " / ".join(got)
            + " solved, against " + " / ".join(was) + " zero-shot)")


def retrain_story_html(D):
    """The full per-rung retraining paragraph (tabs 3 and 4)."""
    rv = D.get("retrain")
    if not rv:
        return ""
    parts = []
    if rv["improved"]:
        parts.append("it clearly <b>improves</b> "
                     + "; ".join(_improved_clause(r) for r in rv["improved"]))
    if rv["matched"]:
        parts.append("it is a <b>wash</b> at " + _rung_list(rv["matched"]))
    if rv["regressed"]:
        parts.append("and it <b>regresses</b> at " + _rung_list(rv["regressed"])
                     + ", where a collapsed value-network training run is the "
                     "cause — measured in the fairness tab's seed-robustness "
                     "section, not a property of the puzzles")
    not_run = (" The " + _rung_list(rv["not_run"]) + " retrain was never run."
               if rv["not_run"] else "")
    return (
        "<p class='small'>Retraining on the full-vocabulary corpus has since "
        f"landed at <b>{rv['n_landed']} of the {rv['n_total']}</b> "
        f"configurations.{not_run} The outcome is <b>not uniform</b>, so the "
        "page states it per rung: " + "; ".join(parts) + ". The zero-shot "
        "rows are therefore <i>not</i> the best measured configuration "
        "everywhere — they are the one configuration measured identically at "
        "every rung, which is why the headline table uses them. The newest "
        "step type never reaches the learned planner in either arm (the "
        "integration gap), which the flag experiment below quantifies.</p>")


def retrain_story_short(D):
    """One-sentence version for captions and footnotes."""
    rv = D.get("retrain")
    if not rv:
        return ""
    bits = []
    if rv["improved"]:
        bits.append("improves at " + _rung_list(rv["improved"]))
    if rv["matched"]:
        bits.append("is a wash at " + _rung_list(rv["matched"]))
    if rv["regressed"]:
        bits.append("regresses at " + _rung_list(rv["regressed"]))
    not_run = (" the " + _rung_list(rv["not_run"]) + " retrain was never run;"
               if rv["not_run"] else "")
    return ("Retraining on the regenerated corpus " + ", ".join(bits) + ";"
            + not_run + " it is not a uniform upgrade in either direction. "
            "The zero-shot rows are the one configuration measured "
            "identically at every rung — that, not superiority everywhere, "
            "is why the headline uses them.")
