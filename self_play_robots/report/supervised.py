"""The "Supervised campaign" tab — the backward-vs-forward results that the
self-play track inherited as its frozen baselines, REGENERATED from the same
result files the supervised suite reads (nothing copied, nothing screenshotted;
`supervised_valuenet/` itself is never modified).

Data sources (all read at page build time; a missing file renders as pending):
  supervised_valuenet/scaling/results/<cfg>/comparison.json           graded
  supervised_valuenet/scaling/results/<cfg>/comparison_ungraded.json  frontier
  supervised_valuenet/scaling/results/<cfg>/comparison_b2*.json       B2 arms + seeds
  supervised_valuenet/scaling/results/g16r8/comparison_forward_{control,rescue}.json
  supervised_valuenet/eval/results/final450_backward_prefix.json      g16r4 backward of record
  supervised_valuenet/eval/results/final450_backward_b2.json          g16r4 B2 arm
  supervised_valuenet/eval/results/comparison_forward.json            g16r4 forward of record

Kept in its own module so edits here never touch the data-tab regions of
`gen_report.py`.
"""
from __future__ import annotations

import html as _html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SV = HERE.parent.parent / "supervised_valuenet"

_READ: list[str] = []
_MISS: list[str] = []


def _esc(s) -> str:
    return _html.escape(str(s))


def _load(rel: str):
    p = SV / rel
    try:
        with open(p) as fh:
            d = json.load(fh)
        _READ.append(rel)
        return d
    except (OSError, json.JSONDecodeError):
        _MISS.append(rel)
        return None


def _sys(payload, kind: str, name_has: str | None = None):
    """First system of the given kind (or whose name contains name_has) with a
    non-empty aggregate."""
    if not payload:
        return None
    for name, sysd in payload.get("systems", {}).items():
        a = sysd.get("aggregate") or {}
        if not a:
            continue
        if name_has is not None:
            if name_has in name:
                return a
            continue
        if sysd.get("kind") == kind or (kind in name and "kind" not in sysd):
            return a
    return None


def _f(v, nd=2):
    return "&mdash;" if v is None else f"{v:.{nd}f}"


def _solved(a):
    return f"{a['solved']}/{a['n']}" if a else '<span class="miss">pending</span>'


CFG_LABEL = {
    "g16r4": "16&times;16 &middot; 4 robots (base rung)",
    "g16r6": "16&times;16 &middot; 6 robots",
    "g16r8": "16&times;16 &middot; 8 robots",
    "g24r4": "24&times;24 &middot; 4 robots",
    "g24r8": "24&times;24 &middot; 8 robots",
    "g32r4": "32&times;32 &middot; 4 robots",
}


def _ladder_rows():
    """(cfg, backward graded agg, forward graded agg, backward frontier agg,
    forward frontier agg) per rung, from the files of record."""
    rows = []
    # g16r4: the base rung lives under eval/results (450-puzzle standard exam)
    b16 = _sys(_load("eval/results/final450_backward_prefix.json"), "backward")
    f16 = _sys(_load("eval/results/comparison_forward.json"), "forward",
               name_has="candidate_scored")
    rows.append(("g16r4", b16, f16, None, None))
    for cfg in ("g16r6", "g16r8", "g24r4", "g24r8", "g32r4"):
        g = _load(f"scaling/results/{cfg}/comparison.json")
        u = _load(f"scaling/results/{cfg}/comparison_ungraded.json") \
            if cfg != "g24r4" else None
        rows.append((cfg, _sys(g, "backward"), _sys(g, "forward"),
                     _sys(u, "backward"), _sys(u, "forward")))
    return rows


def _pooled(bg, fg, bu, fu):
    """Pooled solve counts and the backward-minus-forward margin in points."""
    if not bg or not fg:
        return None
    bs, fs, n = bg["solved"], fg["solved"], bg["n"]
    if bu and fu:
        bs, fs, n = bs + bu["solved"], fs + fu["solved"], n + bu["n"]
    return bs, fs, n, 100.0 * (bs - fs) / n


def _margin_cell(m):
    if m is None:
        return '<td class="dash">&mdash;</td>'
    cls = "g" if m > 0 else ""
    sign = "+" if m > 0 else ""
    w = min(abs(m) * 1.6, 80)
    side = "var(--good-ink)" if m > 0 else "var(--s2)"
    return (f'<td class="{cls}">{sign}{m:.1f}'
            f'<span style="display:inline-block;height:.55em;width:{w:.0f}px;'
            f'background:{side};opacity:.55;margin-left:.4rem;border-radius:2px;">'
            f'</span></td>')


def sec_supervised() -> str:
    _READ.clear()
    _MISS.clear()
    out = ['<section id="supervised">',
           "<h2>The supervised campaign &mdash; backward vs forward</h2>"]
    out.append(
        '<p>Before any self-play, a supervised campaign trained two planner '
        'families on exact-solver answers. The <strong>backward sub-goal '
        'planner</strong> is cheap and plans in the sub-goal language. The '
        '<strong>forward move-by-move planner</strong> is near-perfect but '
        'slow, and its teacher &mdash; the exact solver whose answers it learns '
        'from &mdash; fails on big boards. The campaign&rsquo;s '
        'results are locked in as the numbers to beat. Its value network gave '
        'the self-play loop its starting point. This tab rebuilds the key '
        'results from the campaign&rsquo;s own result files. The full log is '
        '<code>supervised_valuenet/FINDINGS.md</code>.</p>')

    # ---- the scale ladder ---------------------------------------------------
    out.append("<h3>The scale ladder: who wins as boards grow</h3>")
    out.append(
        '<p>Per rung: the standard exam (perfect play known) and the hard '
        'exam (frontier). Frontier = the puzzles the exact solver could not '
        'crack when the exams were frozen &mdash; an optimum exists but is '
        'not known. So the hard exam compares solve counts only. '
        'The last column pools both exams: it is backward&rsquo;s solve-rate '
        'lead over forward, in percentage points.</p>')
    hdr = ('<div class="tw"><table class="t wide"><thead><tr>'
           '<th>rung</th>'
           '<th>backward: solved</th><th>extra moves</th><th>seconds per puzzle</th>'
           '<th>forward: solved</th><th>extra moves</th><th>seconds per puzzle</th>'
           '<th>hard exam (frontier): backward / forward solved</th>'
           '<th>both exams combined: backward&rsquo;s lead in points</th>'
           '</tr></thead><tbody>')
    body = []
    for cfg, bg, fg, bu, fu in _ladder_rows():
        pool = _pooled(bg, fg, bu, fu)
        dag = '&thinsp;&dagger;' if not (bu and fu) else ''
        front = (f"{bu['solved']} / {fu['solved']} of {bu['n']}"
                 if bu and fu else f'<span class="dash">&mdash;{dag}</span>')
        body.append(
            "<tr>"
            f"<td>{CFG_LABEL[cfg]}</td>"
            f"<td>{_solved(bg)}</td>"
            f"<td>{_f(bg['mean_regret'] if bg else None)}</td>"
            f"<td>{_f(bg['mean_seconds'] if bg else None, 1)}</td>"
            f"<td>{_solved(fg)}</td>"
            f"<td>{_f(fg['mean_regret'] if fg else None)}</td>"
            f"<td>{_f(fg['mean_seconds'] if fg else None, 1)}</td>"
            f"<td>{front}</td>"
            + _margin_cell(pool[3] if pool else None).replace('</td>', dag + '</td>')
            + "</tr>")
    out.append(hdr + "\n".join(body) + "</tbody></table></div>")
    out.append(
        '<p class="note">&dagger; No frontier exam was tested for this '
        'pair at the daggered rungs. Those margins cover the standard exam '
        'only. At 24&times;24 with 4 robots the recorded frontier rows use '
        'the extended vocabulary and sit in the next table. That board is '
        'the forward planner&rsquo;s strongest big board: 220 of 232, at '
        '4.6 minutes per puzzle against backward&rsquo;s 8 seconds.</p>')
    out.append(
        '<p class="note">Reading, bottom to top. At the base rung the forward '
        'planner is essentially perfect (450/450, +0.07 moves) and the '
        'backward planner trails. As boards and robot counts grow, the '
        'forward planner&rsquo;s search cost explodes: 23 minutes per '
        '32&times;32 puzzle, and 2 of 275 hard-exam puzzles solved there. The '
        'backward planner stays at seconds per puzzle. So the combined lead '
        'swings hard to backward on the crowded and large boards (+44.4, '
        '+21.6, +30.9 points). At 24&times;24 with 4 robots the graded-only '
        'margin&dagger; still favours forward on solves. The 16&times;16 '
        '8-robot board is the clearest case of the teacher dying: the exact '
        'solver failed on most of the forward planner&rsquo;s training set, '
        'and the planner inherited the failure (25 of 266). That asymmetry is '
        'the whole reason the self-play track exists.</p>')

    # ---- the B2 arms --------------------------------------------------------
    out.append("<h3>The extended vocabulary (B2), under supervision</h3>")
    out.append(
        '<p>The campaign also trained backward planners on the extended '
        'sub-goal vocabulary, called B2 in the logs (see the '
        '<a href="#glossary">Glossary</a>): blockers that hold only for a '
        'moment, reuse of robots already placed, and shoving a robot aside '
        'first. '
        'The extra vocabulary was available. Supervised training never '
        'learned to rank it:</p>')
    b2rows = []
    b2_16 = _sys(_load("eval/results/final450_backward_b2.json"), "backward")
    base_16 = _sys(_load("eval/results/final450_backward_anytime.json"), "backward")
    if b2_16 or base_16:
        b2rows.append(("g16r4",
                       f"{base_16['solved']}/{base_16['n']}" if base_16 else "&mdash;",
                       _f(base_16['mean_regret'] if base_16 else None),
                       f"{b2_16['solved']}/{b2_16['n']}" if b2_16 else "&mdash;",
                       _f(b2_16['mean_regret'] if b2_16 else None),
                       "&mdash;"))
    for cfg in ("g16r6", "g16r8", "g24r4", "g24r8", "g32r4"):
        base = _sys(_load(f"scaling/results/{cfg}/comparison.json"), "backward")
        b2 = _sys(_load(f"scaling/results/{cfg}/comparison_b2.json"), "backward")
        b2u = _sys(_load(f"scaling/results/{cfg}/comparison_ungraded_b2.json"),
                   "backward")
        b2rows.append((cfg,
                       f"{base['solved']}/{base['n']}" if base else "&mdash;",
                       _f(base['mean_regret'] if base else None),
                       f"{b2['solved']}/{b2['n']}" if b2 else "&mdash;",
                       _f(b2['mean_regret'] if b2 else None),
                       f"{b2u['solved']}/{b2u['n']}" if b2u else "&mdash;"))
    out.append('<div class="tw"><table class="t"><thead><tr>'
               '<th>rung</th><th>base vocabulary: solved</th><th>extra moves</th>'
               '<th>extended vocabulary: solved</th><th>extra moves</th>'
               '<th>extended vocabulary: hard-exam solved</th></tr></thead><tbody>'
               + "\n".join(
                   f"<tr><td>{CFG_LABEL[c]}</td><td>{a}</td><td>{b}</td>"
                   f"<td>{d}</td><td>{e}</td><td>{f}</td></tr>"
                   for c, a, b, d, e, f in b2rows)
               + "</tbody></table></div>")
    out.append(
        '<p class="note">A number-matching note first: this table&rsquo;s '
        '16&times;16 base column (2.15 extra moves) and the ladder&rsquo;s '
        '16&times;16 cell (2.14) are the same network pair. They differ '
        'only in the search rule. The ladder uses the prefix-check rule &mdash; '
        'abandon plans whose first moves already fail. This column uses the '
        'anytime rule &mdash; keep searching after the first answer, return '
        'the best found. The <a href="#baselines">Baselines tab</a> lists '
        'all four recorded 16&times;16 runs side by side.</p>'
        '<p class="note">At 24&times;24 the B2-trained planner did worse '
        'than the same planner trained on the plain vocabulary (199 vs 205), '
        'despite the richer language. The problem was the training data, not '
        'the vocabulary: supervised training never saw examples that taught '
        'it to rank the new steps. Making those examples is exactly what the '
        'self-play loop later did (<a href="#res-loops">'
        'milestone results</a>: solving 225&ndash;228 of 232 standard puzzles '
        'and 139&ndash;158 of the frontier from the same vocabulary).</p>')

    # ---- seed replication ---------------------------------------------------
    out.append("<h3>Seed replication of the headline rungs</h3>")
    out.append(
        '<p>The campaign&rsquo;s reviewers asked whether its headline margins '
        'were single-seed flukes. Three fresh training seeds at each of the '
        'two headline rungs:</p>')
    seed_rows = []
    for cfg, seeds in (("g16r8", ("21", "37", "53")),
                       ("g32r4", ("21", "37", "53"))):
        for s in seeds:
            g = _sys(_load(f"scaling/results/{cfg}/comparison_b2_seed{s}.json"),
                     "backward")
            u = _sys(_load(f"scaling/results/{cfg}/"
                           f"comparison_ungraded_b2_seed{s}.json"), "backward")
            pooled = (f"{g['solved'] + u['solved']}/{g['n'] + u['n']}"
                      if g and u else "&mdash;")
            seed_rows.append(
                f"<tr><td>{CFG_LABEL[cfg]} &middot; seed {s}</td>"
                f"<td>{_solved(g)}</td>"
                f"<td>{_f(g['mean_regret'] if g else None)}</td>"
                f"<td>{_solved(u)}</td><td>{pooled}</td></tr>")
    out.append('<div class="tw"><table class="t"><thead><tr>'
               '<th>training run</th><th>standard-exam solved</th><th>extra moves</th>'
               '<th>hard-exam solved</th>'
               '<th>both exams combined: solved of 450</th></tr></thead><tbody>'
               + "\n".join(seed_rows) + "</tbody></table></div>")
    out.append(
        '<p class="note">32&times;32 replicates tightly. The 16&times;16 '
        '8-robot runs split into two groups: seed 21 trained into a much '
        'worse network &mdash; training sometimes lands in a bad solution and '
        'stays there &mdash; with 108 of 184 '
        'hard-exam solves against 157&ndash;165 for the other two. The '
        'campaign reports the middle result and shows the bad run instead of '
        'hiding it.</p>')
    out.append(
        '<p class="note">These three runs are not the ladder&rsquo;s run. The '
        'ladder row for the 16&times;16 8-robot board is the base-vocabulary '
        'pair, searched with the prefix-check rule. That row reads 230 of '
        '266, and 88 of 184 on the hard exam, which pools to 318 of 450. The '
        'three seeds here retrain the extended-vocabulary pair and search '
        'with the anytime rule. That is why none of them repeats the ladder '
        'row. The next section scores the forward planner against the middle '
        'seed run above (418 of 450), not against the ladder row.</p>')

    # ---- forward rescue -----------------------------------------------------
    out.append("<h3>A fair second chance for the forward planner</h3>")
    out.append(
        '<p>The strongest objection to the ladder was &ldquo;weak '
        'opponent&rdquo;: maybe the forward planner just needed tuning. The '
        'campaign answered with a nine-way tuning sweep (3 seeds &times; 3 '
        'learning rates &mdash; a learning rate sets how big each training '
        'adjustment step is) at the 16&times;16 8-robot board. The winner was '
        'picked on held-out practice puzzles (set aside from training, never '
        'the exam):</p>')
    fc = _sys(_load("scaling/results/g16r8/comparison_forward_control.json"),
              "forward", name_has="forward")
    fr = _sys(_load("scaling/results/g16r8/comparison_forward_rescue.json"),
              "forward", name_has="forward")
    fru = _sys(_load("scaling/results/g16r8/"
                     "comparison_ungraded_forward_rescue.json"),
               "forward", name_has="forward")
    fo = _sys(_load("scaling/results/g16r8/comparison.json"), "forward")
    fou = _sys(_load("scaling/results/g16r8/comparison_ungraded.json"),
               "forward")
    out.append('<div class="tw"><table class="t"><thead><tr>'
               '<th>forward training run (16&times;16, 8 robots)</th>'
               '<th>standard-exam solved</th><th>extra moves</th>'
               '<th>hard-exam solved</th></tr></thead><tbody>'
               f"<tr><td>original (its teacher failed on most training examples)</td>"
               f"<td>{_solved(fo)}</td><td>{_f(fo['mean_regret'] if fo else None)}</td>"
               f"<td>{_solved(fou)}</td></tr>"
               f"<tr><td>re-trained control</td>"
               f"<td>{_solved(fc)}</td><td>{_f(fc['mean_regret'] if fc else None, 3)}</td>"
               f"<td>&mdash;</td></tr>"
               f"<tr class=\"hl\"><td>best-of-9, selected by validation</td>"
               f"<td>{_solved(fr)}</td><td>{_f(fr['mean_regret'] if fr else None, 3)}</td>"
               f"<td>{_solved(fru)}</td></tr>"
               "</tbody></table></div>")
    out.append(
        '<p class="note">The rescue is real but small. The tuned forward '
        'planner takes the standard exam perfectly (266/266 &mdash; no '
        'backward run matches that). But it gains only 8 hard-exam puzzles '
        '(93 &rarr; 101). The pre-registered failure rule said: about 150 '
        'hard-exam solves would kill the scale claim (that the backward '
        'planner wins as boards grow). It reached 101. Pooled over '
        'both exams against the backward planner&rsquo;s middle-of-three seed '
        'run in the table above (418/450 vs 367/450), the backward lead '
        'stands at +11.3 points. The '
        '&ldquo;weak opponent&rdquo; objection was answered with measurement, '
        'and the scale claim survived.</p>')

    # ---- what it handed over ------------------------------------------------
    out.append("<h3>What this campaign handed the self-play track</h3>")
    out.append(
        '<ul>'
        '<li><strong>The frozen opponents</strong> &mdash; every baseline row in '
        'the <a href="#baselines">Baselines</a> and '
        '<a href="#variants">Variants lab</a> tabs is one of the result files '
        'above, never re-run.</li>'
        '<li><strong>The starting network</strong> &mdash; a small value network '
        '(the &ldquo;labeller&rdquo;) that graded candidates almost as well '
        'as the exact solver, up to 64&times;64. The self-play networks '
        'began from its weights.</li>'
        '<li><strong>The instruments</strong> &mdash; the replay-checking test '
        'harness, the fixed (&ldquo;pinned&rdquo;) exams, the search-budget '
        'accounting, and the '
        'measured run-to-run variation limits. Run-to-run randomness is '
        'bigger than many effects, so only puzzle-by-puzzle pairing '
        'separates real wins from luck.</li>'
        '<li><strong>The open problem</strong> &mdash; a planner that is cheap '
        '<em>and</em> near-optimal <em>and</em> survives scale. The '
        '<a href="#story">Story tab</a> is what happened next.</li>'
        '</ul>')

    n_read, n_miss = len(set(_READ)), len(set(_MISS))
    miss_txt = ""
    if n_miss:
        miss_txt = (' Missing (rendered as pending): '
                    + ", ".join(f"<code>{_esc(m)}</code>"
                                for m in sorted(set(_MISS))) + ".")
    out.append(f'<p class="meta">This tab read {n_read} supervised result '
               f'file(s) at generation time; nothing under '
               f'<code>supervised_valuenet/</code> was modified.{miss_txt}</p>')
    out.append("</section>")
    return "\n".join(out)
