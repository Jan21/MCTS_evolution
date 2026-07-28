"""The budget sweep: forward vs backward solve rate across the whole range.

Every head-to-head table in this report is one budget point (1,200 search
steps). That point flatters neither system consistently -- it is the LEAST
favourable point for the subgoal planner on gradable sets (which it saturates
by ~100 steps) and the most favourable for the move-by-move planner (still
climbing at the cap). The trend is the argument, so it gets a section.

Reconstructed at zero compute: both searches are deterministic and the budget
only truncates, so an instance solved after e steps is solved at every budget
>= e (`eval/budget_table.py`).
"""

from eval.report_util import esc, ck, scroll, progress_tag, kicker_h2, chip

SHOW = [5, 25, 50, 100, 200, 400, 800, 1200]


def sec_budget_table(D):
    bt = D.get("budget_table")
    if not bt or not bt.get("rows"):
        return kicker_h2(
            "the whole budget range",
            "How each planner improves as it is allowed to search longer",
            "") + progress_tag(
            "eval/results/budget_table.json not generated yet -- run "
            "PYTHONPATH=. python -m eval.budget_table") + "</section>"

    budgets = bt["budgets"]
    idx = [budgets.index(b) for b in SHOW if b in budgets]
    src = "eval/results/budget_table.json"
    head = ("<tr><th>configuration</th><th>puzzle set</th><th>planner</th>"
            + "".join(f'<th class="num">{budgets[i]}</th>' for i in idx)
            + "</tr>")
    body = []
    for r in bt["rows"]:
        n = r["backward"]["n"]
        for arm, label, fam in (("backward", "subgoal", "bwd"),
                                ("forward", "move-by-move", "fwd")):
            cells = []
            for i in idx:
                v = r[arm]["solve_rate"][i] * 100
                cells.append('<td class="num">%s</td>' % ck(
                    "%.1f%%" % v, src,
                    f"budget {budgets[i]} {r['rung']} {r['set']} {arm}",
                    raw=round(v, 4)))
            first = (f'<td rowspan="3"><b>{esc(r["rung_label"])}</b></td>'
                     f'<td rowspan="3">{esc(r["set"])}'
                     f'<div class="cellnote">n={n}</div></td>'
                     if arm == "backward" else "")
            body.append(f"<tr>{first}<td>{esc(label)}</td>"
                        + "".join(cells) + "</tr>")
        dcells = "".join('<td class="num">%+.1f</td>' % (r["diff"][i] * 100)
                         for i in idx)
        body.append(f'<tr class="hl"><td><b>difference</b></td>{dcells}</tr>')
    table = scroll(f"<table><thead>{head}</thead>"
                   f"<tbody>{''.join(body)}</tbody></table>")

    ext = ""
    if bt.get("extended"):
        erows = []
        for e in bt["extended"]:
            if e.get("status") != "ok":
                continue
            erows.append(
                f"<tr><td><b>{esc(e['rung'])}</b></td>"
                f'<td class="num">{e["n"]}</td>'
                f'<td class="num">{e["forward_1200"]["rate"]*100:.1f}%</td>'
                f'<td class="num">{e["forward_probe"]["rate"]*100:.1f}%</td>'
                f'<td class="num">{e["forward_climb"]*100:+.1f}</td>'
                f'<td class="num">{e["backward_1200"]["rate"]*100:.1f}%</td></tr>')
        if erows:
            ext = ("<h3 class=\"ph\">Above the cap: what more search buys the "
                   "move-by-move planner</h3>"
                   "<p class=\"small\">Real runs at 4–5× the standard budget on "
                   "a seeded subsample of each beyond-oracle set — <b>not</b> "
                   "the same instances as the table above, so they are reported "
                   "separately. The subgoal planner stays at 1,200 throughout, "
                   "so this is a deliberately unfair robustness check.</p>"
                   + scroll("<table><thead><tr><th>configuration</th>"
                            '<th class="num">n</th>'
                            '<th class="num">move-by-move @1,200</th>'
                            '<th class="num">@4–5×</th><th class="num">gain</th>'
                            '<th class="num">subgoal @1,200</th></tr></thead>'
                            f"<tbody>{''.join(erows)}</tbody></table>"))

    return kicker_h2(
        "the whole budget range",
        "How each planner improves as it is allowed to search longer",
        "Every other scoreboard here reports a single budget. That one point "
        "is not neutral: the subgoal planner has essentially finished "
        "improving by 100 search steps, while the move-by-move planner is "
        "still climbing at the cap — so a single-point comparison understates "
        "the efficiency gap and overstates the move-level planner's quality "
        "edge.") + f"""
  <p>Solve rate at each budget, both planners, every configuration and puzzle
  set. These are not separate runs: because both searches are deterministic
  and the budget only truncates, a puzzle solved after <i>e</i> steps is
  solved at every budget at least <i>e</i>, so the whole curve is recoverable
  from the per-instance step counts already recorded.</p>
  {table}
  <p class="small">Two things visible only in the sweep. <b>The subgoal
  planner saturates early</b> — it is within a point or two of its final rate
  by 100 steps almost everywhere, while the move-by-move planner needs the
  full budget. <b>The crossover moves with scale</b>: on gradable sets the
  move-by-move planner catches up by roughly 400 steps at 16×16 and 24×24,
  but at 32×32 it never does — 57.7% against 88.0% at 400 steps, still twelve
  points behind at the cap.</p>
  {ext}
</section>"""
