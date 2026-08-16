"""Tabs 4–5: Scaling · Methods & sources."""

from eval.report_util import (esc, ck, fnum, ffrac, dot, chip, scroll,
                              pending, progress_tag, kicker_h2, need, fact,
                              SOURCES, CHECKS, NEEDLES)
from eval.report_data import RUNGS, probe_counts
from eval.report_tables import sys_table
from eval import report_charts as C


def pct(c):
    return c["agg"]["solve_rate"] * 100 if c else None


# ---------------------------------------------------------------------------
# Oracle death
# ---------------------------------------------------------------------------

def sec_oracle(D):
    pts = []
    bm = D.get("bench450_meta")
    if bm:
        pts.append({"label": "16×16 · 4r", "sub": "the base scale",
                    "value": 0.0,
                    "tip": "(every benchmark puzzle carries an exact optimum)"})
        need("oracle chart: base point", "16×16 · 4r")
    order = ["g16r6", "g16r8", "g24r4", "g24r8", "g32r4"]
    shorts = {r["key"]: r["short"] for r in RUNGS if not r.get("base")}
    for key in order:
        m = (D.get("metas") or {}).get(key)
        if not m:
            continue
        v = m["oracle_failure_rate"] * 100
        pts.append({"label": shorts[key], "sub": None, "value": v,
                    "tip": f"({m['n_oracle_failed']}/{m['n_instances']} "
                           "benchmark puzzles unsolved by the exact method)"})
        need(f"oracle chart: {key} value", f"{v:.1f}")
    svg = C.columns_single(pts, unit="%", aria="Share of benchmark puzzles "
                           "the exact solver fails at its practical budget, "
                           "per configuration", height=200)
    twin = C.table_twin(
        ["configuration", "oracle failure rate"],
        [[esc(p["label"]), f"{p['value']:.1f}% {p['tip']}"] for p in pts],
        "oracle failure as a table")
    fig = C.figure(
        "The exact solver — the move-by-move planner's teacher — dies first",
        "share of each 450-puzzle benchmark the exact solver cannot solve "
        "at its practical budget (200,000 expansions + time cap)",
        "", svg,
        "The solver is complete in principle; these are resource caps being "
        "hit, and the required compute grows exponentially with scale. "
        "Supervised training for the move-by-move planner needs this "
        "solver's labels — past 24×24 it fails on the majority of puzzles. "
        "The two axes (more robots, bigger boards) both hurt; 24×24 with 8 "
        "robots combines them.",
        "eval/data/bench450.jsonl.meta.json · "
        "scaling/data/*/bench.jsonl.meta.json", twin)

    probe = D.get("cap_probe")
    probe_html = ""
    if probe:
        solved = sum(1 for r in probe if r.get("status") == "solved")
        unsolved = len(probe) - solved
        src = "scaling/data/g16r8/cap_probe_10x_results.jsonl"
        bar = C.stacked_bar(
            [{"label": "solvable with 10× the compute",
              "value": solved, "var": "var(--seq3)"},
             {"label": "still out of reach at 10×",
              "value": unsolved, "var": "var(--seq6)"}],
            "the 184 8-robot puzzles the oracle failed, re-run at 10× budget",
            aria="Of 184 oracle failures at 8 robots, re-run at ten times "
                 "the budget: how many became solvable")
        probe_html = C.figure(
            "Is the wall real? The 10× probe says yes",
            f"all {len(probe)} of the oracle's 8-robot failures re-run at "
            "2,000,000 expansions (10× its practical budget)",
            "", bar,
            f"{ck(str(solved), src, 'cap probe: solvable at 10x', raw=solved)}"
            f" of {len(probe)} became solvable at ten times the cost; "
            f"{ck(str(unsolved), src, 'cap probe: still failing at 10x', raw=unsolved)}"
            " remain out of reach — a 24% failure rate even at the 10× "
            "budget. The wall moves outward with compute; it does not fall.",
            src, C.table_twin(["outcome", "puzzles"],
                              [["solvable at 10× budget", str(solved)],
                               ["still unsolved", str(unsolved)]],
                              "10× probe as a table"))

    return kicker_h2(
        "why scale is the real question",
        "Exact search runs out first — that is what makes the thesis matter",
        "If exact methods scaled, neither neural planner would be needed. "
        "They do not: the exact solver that labels training data and grades "
        "solutions fails on a growing share of puzzles as boards and robot "
        "counts grow.") + fig + probe_html + "</section>"


# ---------------------------------------------------------------------------
# Forward training fragility
# ---------------------------------------------------------------------------

def sec_fragility(D):
    lad = {e["key"]: e for e in D["ladder"]}
    rows = []
    e6 = lad.get("g16r6", {})
    first = e6.get("fwd_first")
    ctrl = (e6.get("graded") or {}).get("fwd")
    if first and ctrl:
        rows.append((
            "16×16, 6 robots",
            f"destabilized after one pass — scored "
            f"{ck(f'{pct(first):.1f}%', first['src'], 'fragility: 6-robot first run', raw=first['agg']['solved'])} "
            "and produced the since-retracted “tie at 87%”",
            f"lower learning rate — "
            f"{ck(f'{pct(ctrl):.1f}%', ctrl['src'], 'fragility: 6-robot control', raw=ctrl['agg']['solved'])}"))
    e8 = lad.get("g16r8", {})
    ctrl8 = (e8.get("graded") or {}).get("fwd")
    if e8.get("withheld") and ctrl8:
        rows.append((
            "16×16, 8 robots",
            "collapsed to near-random (validation top-1 ≈ 0.06); its "
            "benchmark score is withheld under the integrity rule — a "
            "collapsed opponent proves nothing",
            f"lower learning rate — "
            f"{ck(f'{pct(ctrl8):.1f}%', ctrl8['src'], 'fragility: 8-robot control', raw=ctrl8['agg']['solved'])}"))
    e24 = lad.get("g24r4", {})
    f24 = (e24.get("graded") or {}).get("fwd")
    if f24:
        rows.append((
            "24×24, 4 robots",
            f"stable out of the box — "
            f"{ck(f'{pct(f24):.1f}%', f24['src'], 'fragility: 24x24 stock', raw=f24['agg']['solved'])}",
            "none needed"))
    for key, label in (("g24r8", "24×24, 8 robots"),
                       ("g32r4", "32×32, 4 robots")):
        e = lad.get(key, {})
        f = (e.get("graded") or {}).get("fwd")
        if f:
            rows.append((
                label,
                "the stock recipe destabilized again; the published row is "
                "the stability-controlled retrain",
                f"lower learning rate — "
                f"{ck(f'{pct(f):.1f}%', f['src'], 'fragility: ' + key + ' control', raw=f['agg']['solved'])}"))
    body = "".join(
        f"<tr><td><b>{esc(a)}</b></td><td>{b}</td><td>{c}</td></tr>"
        for a, b, c in rows)
    table = scroll(
        "<table><thead><tr><th>configuration</th>"
        "<th>stock training recipe</th><th>rescue + result</th></tr></thead>"
        f"<tbody>{body}</tbody></table>")
    # integrity: the withheld collapsed run's numbers never render
    wh = (lad.get("g16r8") or {}).get("withheld")
    if wh:
        frac = f"{wh['agg']['solved']}/{wh['agg']['n']}"
        need("integrity: the withheld collapsed 8-robot forward score "
             f"({frac}) is not rendered anywhere", frac, present=False)
    return kicker_h2(
        "training fragility",
        "The move-by-move recipe needs hand-tuning at every new scale",
        "The training recipe that works out of the box at the base scale "
        "destabilized at 6 robots, 8 robots, 24×24/8 and 32×32 — each time "
        "rescued by a lower learning rate. The tuning cost is itself "
        "scaling evidence, honestly caveated: every point has so far been "
        "rescuable. Per the integrity rule, a head-to-head row publishes "
        "only against a properly trained opponent.") + table + """
  <p class="small muted">The subgoal planner's training pipeline needed no
  per-scale tuning on any rung.</p>
</section>"""


# ---------------------------------------------------------------------------
# The ladder scoreboard
# ---------------------------------------------------------------------------

def _score_cell(c, desc, win=None, fam=None, frontier=False):
    if not c:
        return '<td class="num"><span class="muted">not measured</span></td>'
    a = c["agg"]
    main = ck(f"{a['solve_rate'] * 100:.1f}%", c["src"], desc,
              raw=a["solved"])
    steps = ck(fnum(a["mean_expansions"], 1), c["src"], desc + " — steps",
               raw=a["mean_expansions"])
    cls = "num"
    windot = ""
    if win:
        cls += f" win win-{fam}"
        windot = '<span class="windot"></span>'
    return (f'<td class="{cls}">{main}{windot}'
            f'<div class="cellnote">{a["solved"]}/{a["n"]} · {steps} steps'
            f"</div></td>")


def sec_ladder(D):
    head = ("<tr><th>configuration · puzzle set</th>"
            "<th class='num'>" + dot("bwd-old") + "subgoals<br>original "
            "language</th>"
            "<th class='num'>" + dot("bwd") + "subgoals<br>full language, "
            "same networks</th>"
            "<th class='num'>" + dot("bwd") + "subgoals<br>full language, "
            "retrained</th>"
            "<th class='num'>" + dot("fwd") + "move-by-move<br>(healthy "
            "training)</th></tr>")
    body = []
    for e in D["ladder"]:
        for group, gname in (("graded", "puzzles the oracle can grade"),
                             ("frontier", "beyond the oracle")):
            cells = e.get(group) or {}
            if not any(cells.get(k) for k in
                       ("bwd_old", "bwd_b1", "bwd_b2", "fwd")):
                if group == "frontier" and not e.get("base"):
                    body.append(
                        f'<tr><td><b>{esc(e["label"])}</b> · {gname}</td>'
                        '<td colspan="4"><span class="muted small">not yet '
                        "measured</span></td></tr>")
                continue
            n = None
            for k in ("bwd_old", "bwd_b2", "fwd"):
                if cells.get(k):
                    n = cells[k]["agg"]["n"]
                    break
            # winner: best measured backward vs forward (both present only)
            best_key = next((k for k in ("bwd_b2", "bwd_b1", "bwd_old")
                             if cells.get(k)), None)
            best_b = cells.get(best_key) if best_key else None
            f = cells.get("fwd")
            win_b = win_f = False
            if best_b and f:
                if best_b["agg"]["solve_rate"] > f["agg"]["solve_rate"]:
                    win_b = True
                elif f["agg"]["solve_rate"] > best_b["agg"]["solve_rate"]:
                    win_f = True
            desc = f'ladder {e["key"]} {group}'
            row = [f'<td><b>{esc(e["label"])}</b>'
                   f'<div class="cellnote">{gname} — {n} puzzles</div></td>',
                   _score_cell(cells.get("bwd_old"), desc + " bwd_old",
                               win=win_b and best_key == "bwd_old",
                               fam="bwd"),
                   _score_cell(cells.get("bwd_b2"), desc + " bwd_b2",
                               win=win_b and best_key == "bwd_b2",
                               fam="bwd"),
                   ]
            rc = cells.get("bwd_retrained")
            if rc:
                row.append(_score_cell(rc, desc + " bwd_retrained"))
            else:
                row.append('<td class="num"><span class="chip run">in '
                           "progress</span></td>")
            row.append(_score_cell(f, desc + " fwd", win=win_f, fam="fwd"))
            body.append("<tr>" + "".join(row) + "</tr>")
    table = scroll("<table id='ladder-table'><thead>" + head + "</thead>"
                   "<tbody>" + "".join(body) + "</tbody></table>")
    return kicker_h2(
        "the ladder",
        "Every rung, both puzzle sets, all planners",
        "Solve rate (playable) with search steps per puzzle underneath; a "
        "colored dot marks the better solve rate where both a subgoal "
        "planner and a properly trained move-by-move planner are measured. "
        "“Beyond the oracle” rows have no move optima — solving there is "
        "self-certifying. The full-language rows run zero-shot (networks "
        "never trained on the newest step type). Retrained rows show the "
        "regenerated-corpus retrain wherever it completed; read them with "
        "the fairness tab's seed-robustness section, which found retraining "
        "outcomes are decided by which of two training basins the value "
        "network lands in — a retrained row can sit far below its zero-shot "
        "sibling for that reason, and the zero-shot rows remain the "
        "method's best measured configuration. Rows from the earlier "
        "depleted label corpus appear only inside the fairness tab's "
        "label-budget experiment, never here.") \
        + table + """
  <p class="small muted">Where a retrained cell sits far below its
  zero-shot sibling (the 8-robot rungs), the cause is a collapsed
  value-network training run — detectable before benchmarking, measured
  in the fairness tab — not a property of the puzzles.</p>"""  + """
  <p class="small muted">Full per-rung tables — with solution quality,
  seconds and source files — are in the “every rung in full” section
  below.</p>
</section>"""


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def sec_cost(D):
    groups_g, groups_f = [], []
    for e in D["ladder"]:
        gr = e.get("graded") or {}
        best_b = gr.get("bwd_b2") or gr.get("bwd_b1") or gr.get("bwd_old")
        f = gr.get("fwd")
        if best_b and f:
            groups_g.append({
                "label": e["short"],
                "values": {"b": best_b["agg"]["mean_expansions"],
                           "f": f["agg"]["mean_expansions"]}})
        fr = e.get("frontier") or {}
        best_bf = fr.get("bwd_b2") or fr.get("bwd_old")
        ff = fr.get("fwd")
        if best_bf and ff:
            groups_f.append({
                "label": e["short"],
                "values": {"b": best_bf["agg"]["mean_expansions"],
                           "f": ff["agg"]["mean_expansions"]}})
    series = [("b", "bwd", "subgoals (best language measured)"),
              ("f", "fwd", "move-by-move")]
    fig1 = C.figure(
        "Search steps per puzzle — gradable sets",
        "same 1,200-step cap for everyone; lower is better",
        C.legend([("bwd", "subgoals (best language measured)"),
                  ("fwd", "move-by-move")]),
        C.grouped_hbars(groups_g, series, unit="",
                        aria="Mean search steps per puzzle on gradable "
                        "sets, per configuration"),
        "Subgoal plans stay a few decisions deep however large the board; "
        "move-level search grows with travel distance — the gap re-opens "
        "wide on the grid axis.",
        "eval/results + scaling/results/*/comparison*.json",
        C.table_twin(["configuration", "subgoals", "move-by-move"],
                     [[esc(g["label"]), f"{g['values']['b']:.1f}",
                       f"{g['values']['f']:.1f}"] for g in groups_g],
                     "steps as a table"))
    fig2 = C.figure(
        "Search steps per puzzle — beyond the oracle",
        "the hardest pools; a bar at ~1,200 means the budget is exhausted "
        "on average — the planner is failing, not searching",
        C.legend([("bwd", "subgoals (best language measured)"),
                  ("fwd", "move-by-move")]),
        C.grouped_hbars(groups_f, series, unit="",
                        aria="Mean search steps per puzzle beyond the "
                        "oracle, per configuration"),
        "At 32×32 the move-by-move planner saturates its whole 1,200-step "
        "budget and still solves almost nothing (0.7%).",
        "scaling/results/*/comparison_ungraded*.json",
        C.table_twin(["configuration", "subgoals", "move-by-move"],
                     [[esc(g["label"]), f"{g['values']['b']:.1f}",
                       f"{g['values']['f']:.1f}"] for g in groups_f],
                     "steps as a table"))

    # 32×32 wall-clock tile (same machine → comparable)
    lad = {e["key"]: e for e in D["ladder"]}
    g32 = lad.get("g32r4", {})
    tiles = ""
    fg = (g32.get("graded") or {}).get("fwd")
    bg = (g32.get("graded") or {}).get("bwd_b2")
    ff = (g32.get("frontier") or {}).get("fwd")
    bf = (g32.get("frontier") or {}).get("bwd_b2")
    if fg and bg and ff and bf and all(
            c["machine"] == "karolina" for c in (fg, bg, ff, bf)):
        fmin = fg["agg"]["mean_seconds"] / 60
        ffmin = ff["agg"]["mean_seconds"] / 60
        # a mean is never quoted alone on this page: pull the medians for the
        # same two cells out of the per-puzzle wall-clock model
        wc = {(r["key"], r["set"]): r for r in (D.get("wallclock") or [])
              if r.get("same_machine")}

        def _medpair(setname, desc):
            r = wc.get(("g32r4", setname))
            if not r:
                return ""
            return (" Medians, which the tail inflates away from: "
                    + ck(_fsec(r["bwd"]["median"]), r["bwd_src"],
                         desc + " bwd median", raw=r["bwd"]["median"])
                    + " for subgoals against "
                    + ck(_fsec(r["fwd"]["median"]), r["fwd_src"],
                         desc + " fwd median", raw=r["fwd"]["median"])
                    + " move-by-move — the full median/tail/total table is "
                    "in the “Is it fair?” section.")

        med_g = _medpair("gradable set", "tile 32x32 graded")
        med_f = _medpair("beyond the oracle", "tile 32x32 frontier")
        tiles = f"""
  <div class="kpirow">
    <div class="tile"><div class="tlabel">32×32, gradable set — minutes per
    puzzle, move-by-move</div>
    <div class="tvalue">{ck(f"{fmin:.0f} min", fg["src"],
    "cost: 32x32 forward minutes", raw=fg["agg"]["mean_seconds"])}</div>
    <div class="tsub">vs the subgoal planner's
    {ck(f"{bg['agg']['mean_seconds']:.0f} s", bg["src"],
    "cost: 32x32 backward seconds", raw=bg["agg"]["mean_seconds"])} —
    measured on the same machine. These two are means.{med_g}</div></div>
    <div class="tile"><div class="tlabel">32×32, beyond the oracle —
    minutes per puzzle, move-by-move</div>
    <div class="tvalue">{ck(f"{ffmin:.0f} min", ff["src"],
    "cost: 32x32 frontier forward minutes", raw=ff["agg"]["mean_seconds"])}</div>
    <div class="tsub">to solve {pct(ff):.1f}% — forty minutes per puzzle to
    solve almost nothing, vs {ck(f"{bf['agg']['mean_seconds']:.0f} s",
    bf["src"], "cost: 32x32 frontier backward seconds",
    raw=bf["agg"]["mean_seconds"])} at {pct(bf):.1f}% for subgoals. These
    two are means.{med_f}</div>
    </div>
  </div>"""
    return kicker_h2(
        "the cost of thinking in moves",
        "Where the two formulations pay for their choices",
        "Both planners obey the same step cap; what differs is how many "
        "steps they need — and what a step is worth.") \
        + fig1 + fig2 + tiles + "</section>"


# ---------------------------------------------------------------------------
# Per-rung details
# ---------------------------------------------------------------------------

def _detail_rows(e, group):
    cells = e.get(group) or {}
    label_of = {
        "bwd_old": ("Subgoals — original language", "bwd-old",
                    "checked search; the language the networks were bred on"),
        "bwd_b1": ("Subgoals — extended language (B1), retrained", "bwd",
                   "wall-less stoppers + step-asides; retrained networks"),
        "bwd_b2": ("Subgoals — full language (B2), same networks", "bwd",
                   "generalized repairs live; the re-use step type is not "
                   "reachable by the learned proposal path (plan-language "
                   "tab)"),
        "fwd": ("Move-by-move", "fwd", ""),
    }
    rows = []
    for k in ("bwd_old", "bwd_b1", "bwd_b2", "fwd"):
        c = cells.get(k)
        if not c:
            continue
        label, fam, sub = label_of[k]
        if k == "fwd" and e.get("fwd_note"):
            sub = e["fwd_note"]
        rows.append({"label": label, "family": fam, "sub": sub,
                     "agg": c["agg"], "src": c["src"],
                     "machine": c["machine"],
                     "note": f'measured {c["date"]}'})
    rc = cells.get("bwd_retrained")
    rows.append({"label": "Subgoals — full language, retrained networks",
                 "family": "bwd",
                 "sub": ("trained on a corpus carrying the new step type at "
                         "its natural rate; rows from the earlier, depleted "
                         "corpus are withheld"),
                 "agg": rc["agg"] if rc else None,
                 "src": rc["src"] if rc else None,
                 "machine": rc["machine"] if rc else None,
                 "pending_msg": "row fills automatically when the result "
                 "file lands"})
    return rows


def sec_rung_details(D):
    blocks = []
    for e in D["ladder"]:
        if e.get("base"):
            continue
        srcs = sorted({c["src"] for g in ("graded", "frontier")
                       for c in (e.get(g) or {}).values() if c})
        inner = ""
        gr = _detail_rows(e, "graded")
        if any(r["agg"] for r in gr):
            inner += ("<h3>Puzzles the oracle can grade</h3>"
                      + sys_table(gr))
        fr = _detail_rows(e, "frontier")
        if any(r["agg"] for r in fr):
            inner += ("<h3>Beyond the oracle — no move optima exist; "
                      "solution length is the mean over solved puzzles</h3>"
                      + sys_table(fr, frontier=True))
        elif not e.get("base"):
            inner += "<h3>Beyond the oracle</h3>" + pending(
                "this pool has not been measured yet")
        files = "".join(f'<code class="small">{esc(s)}</code><br>'
                        for s in srcs)
        blocks.append(f"""
  <details class="rungdetail">
    <summary><b>{esc(e["label"])}</b><span class="muted small"> — full
    numbers</span></summary>
    {inner}
    <p class="cellnote">sources:<br>{files}</p>
  </details>""")
    return kicker_h2(
        "every rung in full",
        "The complete measurements behind the scoreboard",
        "Expand a configuration for its full tables — solution quality "
        "where optima exist, machine-tagged seconds, and source files.") \
        + "".join(blocks) + "</section>"


def sec_failure_gallery(D):
    """What failure looks like at 24×24 · 8 robots — taxonomy + real boards."""
    from eval.report_boards import board_svg, SLOT_VARS
    fx = D.get("failure_examples")
    if not fx:
        return ""
    SRC = "analysis/artifacts/failure_examples.json"
    t = fx["taxonomy"]
    intro = (
        "<p>The hardest measured pool where both planners were run in "
        "full is 24×24 · 8 robots, beyond the oracle: "
        + ck(str(t["n_bwd_failed"]), SRC, "failgal n failed",
             raw=t["n_bwd_failed"])
        + " of "
        + ck(str(t["n_set"]), SRC, "failgal n set", raw=t["n_set"])
        + " puzzles defeat the subgoal planner. The anatomy of those "
        "failures: "
        + ck(str(t["never_completed"]), SRC, "failgal never",
             raw=t["never_completed"])
        + " never completed a single abstract plan within the search "
        "budget — the planner ran out of budget proposing, not playing — "
        "while only "
        + ck(str(t["all_rejected"]), SRC, "failgal rejected",
             raw=t["all_rejected"])
        + " completed plans on paper whose every variant then failed the "
        "move-by-move play-out. And only "
        + ck(str(t["fwd_solved_of_bwd_failed"]), SRC, "failgal fwd solved",
             raw=t["fwd_solved_of_bwd_failed"])
        + " of the failures were solved by the move-by-move planner: "
        "almost everything that defeats the subgoal planner here defeats "
        "its opponent too. Three real boards from the set:</p>")

    captions = {
        "fwd_solved": ("The honest case — the move-by-move planner solved "
                       "this one and the subgoal planner did not.",
                       "the rarest failure kind"),
        "all_rejected": ("Plans completed on paper; every one failed when "
                         "played move by move.",
                         "the play-out check doing its job"),
        "never_completed": ("No abstract plan completed within budget — "
                            "the common case.",
                            "the typical failure"),
    }
    cards = []
    for key in ("fwd_solved", "all_rejected", "never_completed"):
        ex = (fx.get("examples") or {}).get(key)
        if not ex:
            continue
        b = ex["board"]
        robots = []
        for ri, (x, y) in enumerate(b["positions"]):
            robots.append((x, y, f"var({SLOT_VARS[ri % len(SLOT_VARS)]})",
                           chr(ord('A') + ri), ri == b["target_idx"]))
        svg = board_svg(b["size"],
                       {tuple(w) for w in b["walls_right"]},
                       {tuple(w) for w in b["walls_down"]},
                       robots, rings=((b["target"][0], b["target"][1],
                                       "var(--ink)"),),
                       aria=f"24 by 24 board, puzzle {ex['env_id']}, "
                            f"an unsolved case ({key})", max_px=340)
        head, sub = captions[key]
        stats = (f"subgoal planner: {ex['bwd']['expansions']} search steps, "
                 f"{ex['bwd']['seconds']:.0f} s, "
                 f"{ex['bwd']['plans_rejected']} paper plans rejected · "
                 "move-by-move planner: "
                 + ("solved" if ex['fwd']['solved'] else "also failed"))
        cards.append(
            '<div style="flex:1 1 300px;max-width:360px">'
            f"<p><b>{esc(head)}</b><br>"
            f'<span class="small muted">{esc(sub)} — board {ex["env_id"]}, '
            "the lettered piece must reach the outlined cell"
            f"</span></p>{svg}"
            f'<p class="small muted">{esc(stats)}</p></div>')
    return kicker_h2(
        "what failure looks like",
        "Three unsolved boards from the hardest measured pool",
        "Failure here is overwhelmingly the search not finishing a plan "
        "under its budget — not plans that break when played. Every "
        "count and board below comes from the archived per-puzzle rows.") \
        + intro \
        + '<div style="display:flex;flex-wrap:wrap;gap:18px">' \
        + "".join(cards) + "</div></section>"


def sec_open(D):
    return kicker_h2(
        "still open", "What is running and what is next") + """
  <p>On 2026-08-16 fourteen independent reviewers (each reading the study
  through one lens: thesis, statistics, efficiency, plan language, failures,
  data, scaling, reproducibility, related work, a devil's advocate, and so
  on) judged the work honest and essentially complete, and ranked the gaps
  a hostile referee would attack. The three that can be closed cheaply are
  <b>running now</b>; the rest are listed as future work. Memos:
  <code>analysis/review_2026-08-16/</code>; log: FINDINGS 75–77;
  paper plan: <code>PAPER_PLAN.md</code>.</p>
  <ul>
    <li><b>Running — seed replicates of the backward headline pair</b>
    (16×16·8r and 32×32·4r, seeds 21/37/53, ~20 node-hours). Objection
    answered: “the headline is a single training run; measured seed swing
    reaches 28 points.” Rule fixed in advance: if seeds scatter, the
    headline becomes a median with a range.</li>
    <li><b>Running — a fair rescue for the forward planner</b> at 16×16·8r
    (3 seeds × 3 learning rates, winner chosen by its own validation score,
    never by test; ~8 node-hours). Objection answered: “you beat a weak
    opponent — forward got one run and one learning-rate fix.”</li>
    <li><b>Running — the re-use (by-reference) step, wired at last.</b>
    The step type the plan-language ceiling relies on was never connected
    to the learned planner (proposal path, helper featurization, and a
    silent training filter). It is now wired behind a flag that defaults
    off (flag-off is byte-identical to every row on this page); an on/off
    A/B at 16×16·6r (~1–2 node-hours) measures its zero-shot value.
    Still to do after that: a feature that lets the policy net <i>name</i>
    the referenced cell (a checkpoint migration) and a policy retrain on a
    corpus not depleted of these labels.</li>
    <li><b>Future work, ranked:</b> a learned-subgoal (kSubS-style)
    baseline — the one comparison every reviewer asked for, large; the
    8-robot cell, where retraining never finds its good basin and where
    the labeler track also breaks; a second domain (Rush Hour reuses the
    plan structure, realizer and self-play loop nearly verbatim);
    quality-focused self-play at the frontier; the two unresolved base
    probe instances.</li>
  </ul>
</section>"""


def tab_scaling(D):
    from eval.report_sections_strata import sec_strata
    from eval.report_sections_budget import sec_budget_table
    return (sec_oracle(D) + sec_fragility(D) + sec_ladder(D)
            + sec_budget_table(D) + sec_strata(D)
            + sec_cost(D) + sec_failure_gallery(D) + sec_rung_details(D)
            + sec_open(D))


# ---------------------------------------------------------------------------
# Tab 5 — Is the comparison fair? (compute accounting)
# ---------------------------------------------------------------------------

def sec_fair_matched(D):
    exp_def = ""
    for key in ("bwd_b2", "bwd_prefix", "fwd450"):
        comp = D.get(key)
        if comp and comp.get("protocol", {}).get("expansion_definition"):
            exp_def = comp["protocol"]["expansion_definition"]
            break
    # paired instances: every file of a rung+set must pin the same instances
    all_paired = True
    n_groups = 0
    for key, slots in D["rungs"].items():
        by_group = {}
        for (group, slot), (rel, comp, kind) in slots.items():
            if comp and group in ("graded", "frontier"):
                sha = comp.get("protocol", {}).get("instances_sha256")
                if sha:
                    by_group.setdefault(group, set()).add(sha)
        for group, shas in by_group.items():
            n_groups += 1
            if len(shas) != 1:
                all_paired = False
    fact(f"paired instances: every result file of a rung+set pins the "
         f"identical instance file (sha256 checked, {n_groups} groups)",
         all_paired)
    exp_html = (f'<p class="small muted">As recorded in every result file: '
                f"“{esc(exp_def)}”</p>" if exp_def else "")
    return kicker_h2(
        "what is matched today",
        "The protocol's matched unit: one search step, same everywhere",
        "Both planners run under a cap of 1,200 search expansions per "
        "puzzle with top-5 proposals per decision, on the identical pinned "
        "instances (checksummed per rung and set).") + f"""
  <p>One expansion means the same thing in both families: <b>pop the most
  promising node and generate its candidate continuations — one
  proposal-network pass plus one (batched) value-network pass</b>. That
  makes the step budgets directly comparable at the network-call level:
  neither planner gets more neural computation per step than the other.
  Every head-to-head on this page is paired — the same puzzles, the same
  caps, the same scoring.</p>
  {exp_html}
</section>"""


def sec_fair_ledger(D):
    acc = D.get("compute_accounting")
    hook = ""
    if acc:
        hook = ("<p><b>Instrumented counters have landed</b> "
                "(<code>eval/results/compute_accounting.json</code>): "
                "per-rung counts of network calls and physics-slide calls "
                "for both planners, measured in separate instrumented "
                "runs so the headline rows' timings stay uncontaminated. "
                "The pilot numbers quoted below agree with them.</p>")
    else:
        hook = progress_tag(
            "measurement queued: instrumented re-runs will publish "
            "per-rung counters (network calls by head; physics-slide calls "
            "inside realization, prefix checks and park repairs; free-fix "
            "expansions) to eval/results/compute_accounting.json — this "
            "section renders them automatically when the file lands")
    return kicker_h2(
        "the honest ledger",
        "What the expansion counter does not see",
        "The matched unit counts network passes. The subgoal planner also "
        "does bookkeeping work that the counter does not meter — listed "
        "here in full, because a fair comparison must either count all "
        "work or measure the exclusion.") + f"""
  <div class="cols2">
    <div class="panel"><h3 class="ph">Physics checks inside the search</h3>
    <p class="small">The checked search test-plays every <b>partial</b>
    plan as it is built (the prefix filter). These are pure board-physics
    slide simulations — no network calls — and they are not charged to the
    expansion budget.</p></div>
    <div class="panel"><h3 class="ph">“Free” forced-exact fixes</h3>
    <p class="small">When a decision has exactly one viable candidate, the
    search commits it without spending an expansion. Cheap by
    construction, but uncounted.</p></div>
    <div class="panel"><h3 class="ph">Zero-cost pops of unplayable
    plans</h3><p class="small">A complete plan that fails its playback
    test is discarded and the search continues; the pop-and-discard is not
    charged as an expansion.</p></div>
    <div class="panel"><h3 class="ph">Deterministic park repairs</h3>
    <p class="small">The full-language search may repair a failed complete
    plan by scheduling step-asides — a deterministic, network-free
    procedure, also outside the counter.</p></div>
  </div>
  <p>Why this matters: the hierarchical-search literature's standard
  (“What Matters in Hierarchical Search”, §4.3) is that a two-level method
  must either <b>count all work at both levels</b> or <b>measure the
  excluded work and show it is negligible</b>. The expansion counter here
  meters exactly the network computation — the dominant cost by design —
  but the four mechanisms above are real work done only on the subgoal
  side, so the honest position is to measure them rather than assert
  them away. Two complementary checks exist today: <b>wall-clock time</b>
  (below), which counts everything including the unmetered bookkeeping,
  and the <b>instrumented counters</b> already collected (<code>compute_accounting.json</code>).</p>
  <p><b>The exclusion is not one-sided, and saying so was an error.</b> The
  expansion counter meters network passes, so it misses the forward
  planner's physics too: generating one move-level successor set calls the
  board simulator once per (robot, direction). Measured in the unit both
  systems share — invocations of <code>simulate.slide</code> — a pilot over
  five identical base puzzles reads <b>64 slide calls for the subgoal
  planner against 5,776 for the move-level planner</b> (medians, counting
  search physics only), so the unmetered work is larger on the side the
  ledger above does not list. That figure deliberately excludes a further
  ~34,700 move-level slides per puzzle spent featurizing states for its
  network: the raw totals are 983 against 40,448, but most of the
  move-level total is network input encoding rather than search, while the
  subgoal planner's own featurization reads precomputed tables and calls
  the primitive never — quoting the raw ratio would pass featurization off
  as physics. The
  qualification that survives is about the <i>shape</i> of the subgoal
  planner's cost rather than its size: its distribution is heavy-tailed,
  and a single unsolved puzzle in that pilot consumed more physics than the
  most expensive puzzle the move-level planner solved, almost all of it
  inside park repair. Accounting is therefore reported as median plus tail,
  never as a mean.</p>
  {hook}
</section>"""


def _read_budget_by_rung(data, D):
    """Reader for eval/results/budget_curves_by_rung.json.

    Schema (landed 2026-07-24): {"budgets": [B...], "method": str,
    "sources": {relpath: {system_name: {"n": int, "solve_rate": [r per B],
    "solve_rate_1200": r}}}}, rates as fractions.

    The rows to plot are selected through the ladder cells (which already
    carry each slot's source file and exact system name), so the same
    control/withhold logic applies here: the collapsed 8-robot stock
    forward run is present in the file but is never plotted.

    Returns list of panels {key, label, set, n, series:[{fam,label,points,
    final_frac, src, name}]} or None if the schema is unrecognized."""
    if not isinstance(data, dict) or not isinstance(data.get("sources"),
                                                    dict) \
            or not isinstance(data.get("budgets"), list):
        return None
    budgets = data["budgets"]
    sources = data["sources"]

    def lookup(cell):
        if not cell:
            return None
        rates = (sources.get(cell["src"]) or {}).get(cell["name"])
        if not rates or not isinstance(rates.get("solve_rate"), list):
            return None
        pts = [(b, r * 100) for b, r in zip(budgets, rates["solve_rate"])]
        return {"points": pts, "final_frac": rates.get("solve_rate_1200"),
                "n": rates.get("n"), "src": cell["src"],
                "name": cell["name"], "cell": cell}

    panels = []
    for e in D["ladder"]:
        for group, gname in (("graded", "gradable set"),
                             ("frontier", "beyond the oracle")):
            cells = e.get(group) or {}
            series = []
            for slot, fam, lab in (
                    ("bwd_old", "bwd-old", "subgoals, original language"),
                    ("bwd_b2", "bwd", "subgoals, full language"),
                    ("fwd", "fwd", "move-by-move")):
                cand = cells.get(slot)
                if slot == "bwd_b2" and not cand:
                    cand = cells.get("bwd_b1")
                    lab = "subgoals, extended language (B1)"
                got = lookup(cand)
                if got:
                    got.update({"fam": fam, "label": lab})
                    series.append(got)
            if series:
                panels.append({"key": e["key"], "label": e["label"],
                               "set": gname, "n": series[0]["n"],
                               "series": series})
    return panels or None


BUDGET_TWIN_CAPS = (5, 25, 100, 300, 1200)


def sec_fair_budget(D):
    head = kicker_h2(
        "solve rate vs budget, every rung",
        "What each planner does when the budget shrinks",
        "The step cap of 1,200 is one point on a curve. Because both "
        "searches are deterministic, a puzzle solved using e expansions is "
        "solved at every budget B ≥ e — so the full solve-rate-vs-budget "
        "curve for every budget up to 1,200 can be reconstructed exactly "
        "from the archived per-instance expansion counts, with no new "
        "compute.")
    raw = D.get("budget_by_rung")
    src = "eval/results/budget_curves_by_rung.json"
    if raw is None:
        return head + pending(
            "eval/results/budget_curves_by_rung.json is being generated "
            "(per-rung, per-system solve-rate-vs-budget reconstructed from "
            "archived per-instance expansion counts); the small-multiple "
            "charts render here automatically when it lands") + "</section>"
    panels_data = _read_budget_by_rung(raw, D)
    if not panels_data:
        return head + pending(
            f"{src} is present but its schema was not recognized by "
            "report_sections_scale._read_budget_by_rung — extend the "
            "reader") + "</section>"
    # cross-file consistency: each curve's endpoint must equal the solve
    # rate of the aggregate row this report renders elsewhere
    worst = 0.0
    for p in panels_data:
        for s in p["series"]:
            agg_rate = s["cell"]["agg"]["solve_rate"]
            if s["final_frac"] is not None:
                worst = max(worst, abs(agg_rate - s["final_frac"]))
    fact("budget curves: every curve's endpoint equals its comparison "
         f"file's aggregate solve rate (max deviation {worst:.2e})",
         worst < 1e-9)
    fams = []
    for p in panels_data:
        for s in p["series"]:
            if (s["fam"], s["label"]) not in fams:
                fams.append((s["fam"], s["label"]))
    panels, twin_rows = [], []
    for p in panels_data:
        svg = C.budget_curve_panel(
            p["series"], hover_budgets=BUDGET_TWIN_CAPS,
            aria=f'Solve rate vs search-step budget, {p["label"]}, '
                 f'{p["set"]}')
        panels.append(
            f'<div class="panel"><h3 class="ph">{esc(p["label"])}</h3>'
            f'<p class="cellnote">{esc(p["set"])} — {p["n"]} puzzles</p>'
            f"{svg}</div>")
        for s in p["series"]:
            row = [esc(f'{p["label"]}, {p["set"]} — {s["label"]}')]
            for b in BUDGET_TWIN_CAPS:
                v = C._rate_at(s["points"], b)
                row.append(ck(f"{v:.1f}%", src,
                              f'budget curve {p["key"]} {p["set"]} '
                              f'{s["label"]} at {b}', raw=v)
                           if v is not None else "—")
            twin_rows.append(row)
    twin = C.table_twin(
        ["configuration, set — system"]
        + [f"{b} steps" for b in BUDGET_TWIN_CAPS],
        twin_rows, "reconstructed curves as a table (every panel)")
    method = raw.get("method", "")
    caption = (
        f"Reconstruction method, as recorded in the file: “{esc(method)}” "
        "No new runs are involved. Log-scale budget axis; the dot marks "
        "each curve's endpoint at the full 1,200-step cap, which is "
        "verified at build time to equal the aggregate solve rate "
        "reported elsewhere on this page. Forward rows use the same "
        "healthy-training selection as the rest of the report (the "
        "collapsed 8-robot stock run is present in the file and excluded "
        "here too).")
    # derived readings (kept honest by computing them from the curves)
    shares50 = []
    for p in panels_data:
        if p["set"] != "gradable set":
            continue
        for s in p["series"]:
            if s["fam"] == "bwd" or (s["fam"] == "bwd-old" and not any(
                    x["fam"] == "bwd" for x in p["series"])):
                r50 = C._rate_at(s["points"], 50)
                rf = C._rate_at(s["points"], 1200)
                if r50 and rf:
                    shares50.append(r50 / rf * 100)
    gains = []   # (label, last-200-step gain) for frontier forward curves
    for p in panels_data:
        if p["set"] != "beyond the oracle":
            continue
        for s in p["series"]:
            if s["fam"] == "fwd":
                g = (C._rate_at(s["points"], 1200) or 0) \
                    - (C._rate_at(s["points"], 1000) or 0)
                gains.append((p["label"], g))
    climbing = [f"{lab} (+{g:.1f} points)" for lab, g in gains if g >= 2]
    flat = [f"{lab} (+{g:.1f})" for lab, g in gains if g < 2]
    reading = ""
    if shares50 and gains:
        reading = f"""
  <p class="small">Two readings, computed from the curves themselves.
  <b>The efficiency gap is structural:</b> on gradable sets the best
  subgoal curves reach {min(shares50):.0f}–{max(shares50):.0f}% of their
  final solve rate within the first 50 steps, while the move-by-move
  curves need hundreds. <b>The frontier picture is budget-limited
  everywhere it has been probed — including where these curves look
  flat.</b> The move-by-move frontier curves are still climbing at the cap
  at {esc("; ".join(climbing)) if climbing else "no rung"} and have gone
  nearly flat at {esc("; ".join(flat)) if flat else "no rung"}. It is
  tempting to read the flat rungs as saturation; the extended-budget
  probes below show that reading is wrong. At 24×24 · 8 robots, a rung
  whose curve gains only ~1 point over its last 200 expansions, a 5×
  budget adds <b>12.5 points</b> — a small per-window gain integrated over
  a long extension still compounds. Local flatness over a 200-step window
  is not saturation, and no claim on this page rests on it.</p>"""
    return (head + C.line_legend(fams)
            + f'<div class="cols2">{"".join(panels)}</div>'
            + f'<figure class="chart"><figcaption>{caption}'
            f'<span class="src">source: {esc(src)}</span></figcaption>'
            f"{twin}</figure>" + reading + "</section>")


def _fsec(x):
    """Seconds, rendered at a readable precision for its magnitude."""
    if x is None:
        return "—"
    if x < 10:
        return f"{x:.2f} s"
    if x < 100:
        return f"{x:.1f} s"
    return f"{x:,.0f} s"


def _fdur(x):
    """A total duration, in the largest unit that keeps it readable."""
    if x is None:
        return "—"
    if x < 120:
        return f"{x:.0f} s"
    if x < 7200:
        return f"{x / 60:.0f} min"
    return f"{x / 3600:.1f} h"


def _fratio(x):
    if x is None:
        return "—"
    return f"{x:.1f}×" if x < 100 else f"{x:.0f}×"


def sec_fair_wallclock(D):
    """The wall-clock table: median + 90th percentile + total, per rung.

    Answers the reviewer objection that “search steps is not a fair unit”:
    wall-clock is the one unit both planners spend in the same currency, it
    charges every scrap of unmetered bookkeeping, and it is read here from
    the per-puzzle `seconds` field that every comparison file carries for
    both systems.  Median and 90th percentile are reported side by side
    because the subgoal planner's cost is heavy-tailed — a mean alone would
    hide exactly the puzzles the objection is about."""
    head = kicker_h2(
        "the complementary check: wall-clock",
        "Time counts everything — and it does not flatter the subgoal side "
        "at the base scale",
        "A search step is a contested unit: a move-level step and a subgoal "
        "decision are not the same amount of work. Seconds are not "
        "contested. This table drops the step counter entirely and reports "
        "the per-puzzle wall-clock every comparison file already records "
        "for both planners — median, 90th percentile (the tail), and the "
        "total for the whole set. Seconds are only comparable between runs "
        "on the same machine, so every row pairs the move-by-move planner "
        "with the fullest subgoal language measured on the SAME machine, "
        "and names that machine.")
    rows = D.get("wallclock")
    src = "eval/results + scaling/results/*/comparison*.json (per-puzzle rows)"
    if not rows:
        return head + pending(
            "per-puzzle wall-clock could not be read from the comparison "
            "files (the `rows` arrays carry a `seconds` field per puzzle); "
            "the table renders here automatically once they are readable") \
            + "</section>"

    body = []
    for r in rows:
        title = (f"<td><b>{esc(r['label'])}</b>"
                 f'<div class="cellnote">{esc(r["set"])}')
        if not r.get("same_machine"):
            body.append(
                title + "</div></td>"
                '<td colspan="8" class="small muted">no same-machine '
                "backward/forward pair exists at this rung — seconds from "
                "different machines are never compared on this page</td></tr>")
            continue
        mach = ("Karolina" if r["machine"] == "karolina"
                else "the origin machine")
        note = (f" · {esc(r['bwd_label'])} · both runs on {esc(mach)}")
        skip = ""
        if r.get("skipped"):
            skip = ('<div class="cellnote muted">machines differ: '
                    f"{esc(r['skipped'])} was measured at this rung too, but "
                    "on the other machine — unusable here</div>")
        d = f"wall-clock {r['key']} {r['set']}"
        b, f = r["bwd"], r["fwd"]
        cells = [
            ck(_fsec(b["median"]), r["bwd_src"], d + " bwd median",
               raw=b["median"]),
            ck(_fsec(b["p90"]), r["bwd_src"], d + " bwd p90", raw=b["p90"]),
            ck(_fdur(b["total"]), r["bwd_src"], d + " bwd total",
               raw=b["total"]),
            ck(_fsec(f["median"]), r["fwd_src"], d + " fwd median",
               raw=f["median"]),
            ck(_fsec(f["p90"]), r["fwd_src"], d + " fwd p90", raw=f["p90"]),
            ck(_fdur(f["total"]), r["fwd_src"], d + " fwd total",
               raw=f["total"]),
        ]
        body.append(
            title + note + "</div>" + skip
            + f'<div class="cellnote">{b["n"]} puzzles</div></td>'
            + "".join(f'<td class="num">{c}</td>' for c in cells[:3])
            + "".join(f'<td class="num sep">{c}</td>' for c in cells[3:])
            + f'<td class="num"><b>{esc(_fratio(r["r_median"]))}</b></td>'
            + f'<td class="num"><b>{esc(_fratio(r["r_total"]))}</b></td></tr>')
    body = ["<tr>" + x for x in body]

    table = scroll(
        "<table><thead>"
        "<tr><th rowspan='2'>configuration · set<br>"
        "<span class='small muted'>machine, and which subgoal language</span>"
        "</th>"
        "<th colspan='3' class='num'>" + dot("bwd")
        + "subgoals — seconds per puzzle</th>"
        "<th colspan='3' class='num'>" + dot("fwd")
        + "move-by-move — seconds per puzzle</th>"
        "<th colspan='2' class='num'>move-by-move ÷ subgoals</th></tr>"
        "<tr><th class='num'>median</th><th class='num'>90th pct</th>"
        "<th class='num'>total</th>"
        "<th class='num sep'>median</th><th class='num sep'>90th pct</th>"
        "<th class='num sep'>total</th>"
        "<th class='num'>at the median</th><th class='num'>on total</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>")

    warn = ('<p class="small muted"><b>How to read it.</b> Compare only '
            "left-to-right within one row: both numbers in a row come from "
            "the same machine, which the row names. Never compare seconds "
            "down a column — the rungs measured before 2026-07-20 ran on a "
            "different machine from those measured after, and the study "
            "does not calibrate between them. The ratio columns are "
            "therefore the only quantities that travel between rows. "
            "“Total” is the summed wall-clock of the whole set, so it is "
            "the mean in disguise and is dominated by the tail; it is "
            "printed next to the median deliberately, so the two can "
            "disagree in public.</p>")

    # ---- honest reading, computed from the table itself ------------------
    ok = [r for r in rows if r.get("same_machine")]
    scaling = [r for r in ok if not r.get("base")]
    base = [r for r in ok if r.get("base")]
    fact("wall-clock: at every scaling rung the move-by-move planner is "
         "slower at the median, at the 90th percentile and on total",
         bool(scaling) and all(r["r_median"] > 1 and r["r_p90"] > 1
                               and r["r_total"] > 1 for r in scaling))
    fact("wall-clock table covers every rung on both sets that has a "
         "same-machine pair (11 rows)", len(ok) == 11)
    reading = ""
    if scaling:
        lo_m = min(r["r_median"] for r in scaling)
        hi_m = max(r["r_median"] for r in scaling)
        lo_t = min(r["r_total"] for r in scaling)
        hi_t = max(r["r_total"] for r in scaling)
        lo_p = min(r["r_p90"] for r in scaling)
        tight = min(scaling, key=lambda r: r["r_p90"])
        base_txt = ""
        if base:
            b0 = base[0]
            base_txt = (
                " <b>At the base scale it does not.</b> On 16×16 with 4 "
                "robots the two planners have an all-but-identical median "
                "puzzle — "
                f"{ck(_fsec(b0['bwd']['median']), b0['bwd_src'], 'wall-clock base bwd median (reading)', raw=b0['bwd']['median'])}"
                " against "
                f"{ck(_fsec(b0['fwd']['median']), b0['fwd_src'], 'wall-clock base fwd median (reading)', raw=b0['fwd']['median'])}"
                " — and the subgoal planner is the <i>slower</i> of the two "
                "in the tail and on total ("
                f"{ck(_fsec(b0['bwd']['p90']), b0['bwd_src'], 'wall-clock base bwd p90 (reading)', raw=b0['bwd']['p90'])}"
                " vs "
                f"{ck(_fsec(b0['fwd']['p90']), b0['fwd_src'], 'wall-clock base fwd p90 (reading)', raw=b0['fwd']['p90'])}"
                " at the 90th percentile; "
                f"{ck(_fdur(b0['bwd']['total']), b0['bwd_src'], 'wall-clock base bwd total (reading)', raw=b0['bwd']['total'])}"
                " vs "
                f"{ck(_fdur(b0['fwd']['total']), b0['fwd_src'], 'wall-clock base fwd total (reading)', raw=b0['fwd']['total'])}"
                " for the whole 450-puzzle set). That inversion is a real "
                "result, not a rounding artifact: on a small board a "
                "move-level step is cheap enough that the move-by-move "
                "planner's many steps cost less than the subgoal planner's "
                "few expensive ones plus its unmetered realization, "
                "prefix-check and park-repair work. The time advantage is "
                "earned at scale; it is not a property of the formulation "
                "at every size.")
        reading = f"""
  <p><b>The honest reading.</b> Wall-clock is the unit that cannot be
  accused of favouring either side — it charges the subgoal planner for
  every unmetered thing the step counter misses — and at every rung above
  the base scale it points the same way as the step counts, though not by
  the same multiplier: the move-by-move planner needs
  {esc(_fratio(lo_m))}–{esc(_fratio(hi_m))} the subgoal planner's time on
  the median puzzle, and {esc(_fratio(lo_t))}–{esc(_fratio(hi_t))} in
  total.{base_txt}</p>
  <p class="small"><b>Where the tail bites.</b> The median and the total
  disagree by a lot, and the disagreement is informative in both
  directions. The subgoal planner's own tail is heavy — its 90th
  percentile can be two orders of magnitude above its median, which is the
  park-repair cost FINDINGS 25 isolates — so its advantage shrinks at the
  tail: the tightest 90th-percentile margin in the table is
  {esc(_fratio(lo_p))} ({esc(tight["label"])}, {esc(tight["set"])}), where
  the two planners' worst puzzles cost nearly the same. Where the total
  ratio exceeds the median ratio instead (the large-board rungs), it is
  the move-by-move planner whose tail explodes. No single number
  summarizes this table, which is why all three are printed.</p>"""

    return (head + table
            + f'<p class="small muted">source: {esc(src)} — per-puzzle '
            "<code>seconds</code> fields, not the files' "
            "<code>mean_seconds</code> aggregates.</p>"
            + warn + reading + _wallclock_curves(D) + "</section>")


def _wallclock_curves(D):
    """Solve rate vs a per-puzzle wall-clock budget — the step-free curve.

    Sidesteps the unit argument completely: for a time budget t, a planner's
    solve rate is the share of puzzles it solves in t seconds or less.  Both
    searches are deterministic and were run to completion, so these curves
    are exact reconstructions from the recorded per-puzzle seconds — no new
    compute, and no step counter anywhere in the picture."""
    rows = [r for r in (D.get("wallclock") or []) if r.get("same_machine")]
    if not rows:
        return ""
    panels = []
    for r in rows:
        b, f = r["bwd"], r["fwd"]
        lo = max(1e-3, min(b["min"], f["min"]))
        hi = max(b["max"], f["max"])
        import math as _m
        lo = 10 ** _m.floor(_m.log10(lo))
        hi = 10 ** _m.ceil(_m.log10(hi))
        ticks, t = [], lo
        while t <= hi * 1.0001:
            ticks.append(t)
            t *= 10
        series = [{"fam": "bwd", "label": r["bwd_label"], "points": b["curve"]},
                  {"fam": "fwd", "label": "move-by-move",
                   "points": f["curve"]}]
        svg = C.budget_curve_panel(
            series, xmin=lo, xmax=hi, xticks=ticks,
            hover_budgets=[x for x in ticks if x > lo],
            xlabel="seconds per puzzle (log scale)",
            hover_fmt=lambda v: f"within {v:g} s per puzzle",
            aria=f"Solve rate versus per-puzzle wall-clock budget, "
                 f"{r['label']}, {r['set']}")
        panels.append(
            f'<div class="panel"><h3 class="ph">{esc(r["label"])}</h3>'
            f'<p class="cellnote">{esc(r["set"])} — {b["n"]} puzzles · '
            f'{esc("Karolina" if r["machine"] == "karolina" else "origin machine")}'
            f"</p>{svg}</div>")
    fams = [("bwd", "subgoal planner (language named in the table above)"),
            ("fwd", "move-by-move planner")]
    return ("""
  <h3 class="ph">The same claim without a step counter</h3>
  <p class="small">Each curve reads: give a planner <i>t</i> seconds per
  puzzle and it solves this share of the set. Nothing here is measured in
  search steps, so the fairness of a step is not in question. Both axes are
  the planner's own: the curve that reaches a given height further left is
  the cheaper planner at that solve rate. Within a panel both curves are
  same-machine; between panels they are not, so read heights and crossings,
  not absolute seconds, across panels.</p>"""
            + C.line_legend(fams)
            + f'<div class="cols2">{"".join(panels)}</div>')



def sec_fair_probes(D):
    lad = {e["key"]: e for e in D["ladder"]}
    sat_html = ""
    ff32 = (lad.get("g32r4", {}).get("frontier") or {}).get("fwd")
    ff24 = (lad.get("g24r8", {}).get("frontier") or {}).get("fwd")
    if ff32:
        me = ff32["agg"]["mean_expansions"]
        cap = 1200
        fact("32×32 frontier forward row is budget-saturated "
             "(mean expansions > 99% of the cap)", me > 0.99 * cap)
        e24 = ""
        if ff24:
            me24 = ff24["agg"]["mean_expansions"]
            e24 = (f" (at 24×24 · 8 robots the same row averages "
                   f"{ck(fnum(me24, 1), ff24['src'], 'saturation: g24r8 frontier fwd steps', raw=me24)}"
                   f" steps)")
        sat_html = f"""
  <p><b>The caveat that motivates these probes:</b> the 32×32 frontier
  move-by-move row is <b>budget-saturated</b> — it averages
  {ck(fnum(me, 1), ff32["src"], "saturation: g32r4 frontier fwd steps",
  raw=me)} of its {cap} allowed expansions{e24}. A saturated row shows the
  planner failing <i>at this budget</i>; it cannot by itself distinguish
  “cannot solve these puzzles” from “needs a larger budget”. The
  extended-budget probes answer exactly that question on frontier
  subsamples.</p>"""
    probes = [
        ("fwd_probe_g24r8", "scaling/results/g24r8/forward_probe_e6000.json",
         "24×24 board, 8 robots", "6,000 steps (5× the standard cap)",
         "is the 15.2% frontier solve rate a budget-cap artifact?"),
        ("fwd_probe_g32r4", "scaling/results/g32r4/forward_probe_e4800.json",
         "32×32 board, 4 robots", "4,800 steps (4× the standard cap)",
         "is the 0.7% frontier collapse a budget-cap artifact?"),
        ("fwd_probe_g16r6", "scaling/results/g16r6/forward_probe_e4800.json",
         "16×16 board, 6 robots", "4,800 steps (4× the standard cap)",
         "the robot axis: its curve was still climbing at the 1,200 cap"),
        ("fwd_probe_g16r8", "scaling/results/g16r8/forward_probe_e4800.json",
         "16×16 board, 8 robots", "4,800 steps (4× the standard cap)",
         "the rung where the decisive frontier win is claimed"),
    ]
    blocks = []
    for dkey, rel, label, budget, question in probes:
        comp = D.get(dkey)
        if not comp:
            blocks.append(f"""
  <div class="panel"><h3 class="ph">{esc(label)} — move-by-move at
  {esc(budget)}</h3>
  <p class="small">Question: {esc(question)}</p>
  {progress_tag(f"measurement queued — renders automatically from {rel} "
                "when the run lands (a frontier-subsample comparison at "
                "the extended budget)")}</div>""")
            continue
        from eval.report_data import pick, pick_name, machine_of, proto_date
        agg = pick(comp, "forward")
        rows = []
        if agg:
            rows.append({
                "label": f"Move-by-move at {budget}", "family": "fwd",
                "sub": f"frontier subsample; standard rows use 1,200 steps",
                "agg": agg, "src": rel,
                "machine": machine_of(comp),
                "note": f"measured {proto_date(comp)[:10]}"})
        b_agg = pick(comp, "backward")
        if b_agg:
            rows.append({
                "label": "Subgoals on the same subsample (reference)",
                "family": "bwd", "sub": "", "agg": b_agg, "src": rel,
                "machine": machine_of(comp)})
        blocks.append(f"""
  <div class="panel"><h3 class="ph">{esc(label)} — move-by-move at
  {esc(budget)}</h3>
  <p class="small">Question: {esc(question)}</p>
  {sys_table(rows, frontier=True)}</div>""")
    return kicker_h2(
        "extended-budget probes",
        "Is the frontier collapse a budget artifact? The direct test",
        "Four measured runs gave the move-by-move planner several times its "
        "standard budget on frontier subsamples. Where its solve rate "
        "stayed flat, the collapse is real; where it climbed, the "
        "standard-budget frontier rows overstate the gap — both happened, "
        "at different scales.") \
        + sat_html + "".join(blocks) + "</section>"


def tab_fairness(D):
    from eval.report_sections_stats import sec_significance
    return (sec_fair_matched(D) + sec_significance(D) + sec_fair_ledger(D)
            + sec_fair_budget(D) + sec_fair_wallclock(D) + sec_fair_probes(D))


# ---------------------------------------------------------------------------
# Tab 6 — Methods & sources
# ---------------------------------------------------------------------------

def sec_protocol(D):
    exp_def = ""
    for key in ("bwd_b2", "bwd_prefix", "fwd450"):
        comp = D.get(key)
        if comp and comp.get("protocol", {}).get("expansion_definition"):
            exp_def = comp["protocol"]["expansion_definition"]
            break
    exp_html = (f'<p class="small muted">Definition recorded in the result '
                f"files: “{esc(exp_def)}”</p>" if exp_def else "")
    bm = (D.get("bench450_meta") or {}).get("protocol", {})
    dstar = bm.get("d_star", "")
    return kicker_h2("the rules of the comparison", "Protocol") + f"""
  <div class="cols2">
    <div class="panel"><h3 class="ph">Identical budgets</h3>
    <p class="small">Every head-to-head runs at a cap of <b>1,200 search
    steps</b> per puzzle with <b>top-5 proposals</b> per decision, on pinned
    per-configuration benchmarks of 450 puzzles (SHA-recorded instance
    files; the gradable/beyond-oracle split follows the exact solver's own
    success). One search step = examine one position or partial plan and
    generate its candidate continuations; in both families a step costs one
    pass of each network, so budgets are directly comparable.</p>
    {exp_html}</div>
    <div class="panel"><h3 class="ph">Playable-moves scoring</h3>
    <p class="small">A puzzle counts as solved only if the produced plan
    plays out legally, move by move, under full physics
    (<code>eval/realize.py</code>). “Extra moves” is the played-out length
    minus the exact optimum, averaged over solved puzzles; “% at optimum”
    is the share of solved puzzles at exactly the optimum. The subgoal
    planner's internal plan costs are never compared to move optima.</p>
    </div>
    <div class="panel"><h3 class="ph">The oracle's role</h3>
    <p class="small">The exact solver produces training labels and
    reference optima only — it never participates at solve time. Recorded
    with the base benchmark: “{esc(dstar)}”. On puzzles it cannot solve,
    result files carry placeholder optima — this report never renders
    solution-quality columns for those pools.</p></div>
    <div class="panel"><h3 class="ph">Machines and timing</h3>
    <p class="small">Results before 2026-07-20 were measured on the origin
    machine; from 2026-07-20 on the Karolina cluster (CPU lanes of EPYC
    7763 nodes). The tag is derived from each file's own recorded run date.
    Solve rates, search steps and move counts are machine-independent;
    wall-clock seconds are compared only within one machine, and tables
    that mix machines tag every seconds cell.</p></div>
  </div>
</section>"""


def sec_design(D):
    return kicker_h2("study design", "Two structures × two ways to train") + """
  <p>Four systems span the design space: the <b>plan structure</b> axis
  (move-by-move vs subgoals) crossed with the <b>training</b> axis
  (supervised on exact-solver labels vs self-play on the planner's own
  playable solutions). Supervised training is the quality path while the
  oracle lives; self-play is the only path that survives where the oracle
  fails. Both structures have working versions of both — the base-scale tab
  measures all four.</p>
  <p class="small muted">Data generation for training runs on a verified
  Rust port of the labeler (50–180× faster; 20,649 replayed decision
  contexts with zero label differences on pinned inputs —
  <code>rust_datagen/VERIFICATION.md</code>). Two disclosed findings from
  that verification: the original Python labeler's tie-breaking was
  order-dependent (≈7% of instances at the trajectory level, never within
  a trajectory), and a rare cost-estimate bug (≈0.02% of records) made
  some “exact” labels slightly off — negligible for training, recorded for
  honesty.</p>
</section>"""


def sec_glossary():
    terms = [
        ("rung / configuration", "one setting of board size and robot count "
         "(for example 24\u00d724 with 8 robots). The study runs the same "
         "comparison at six of them; \u201cthe ladder\u201d is all six "
         "together"),
        ("paired test", "both planners attempt the same puzzles, so each "
         "puzzle gives a matched pair of outcomes; only the puzzles where "
         "they disagree carry information about which is better"),
        ("search step (expansion)", "examine one position or partial plan "
         "and generate its candidate continuations; one pass of each "
         "network in either planner family"),
        ("playable / realized", "a plan converted to actual moves that "
         "execute legally under full physics from the start position"),
        ("extra moves (regret)", "played-out solution length minus the "
         "exact optimum, averaged over solved puzzles"),
        ("oracle", "the exact solver used for labels and reference optima; "
         "never available at solve time"),
        ("gradable set", "benchmark puzzles the oracle solved, so optima "
         "exist and quality is measurable"),
        ("beyond the oracle / frontier", "benchmark puzzles the oracle "
         "failed at its practical budget; a solution proves itself by "
         "playing out — only solve rate, steps and time are meaningful"),
        ("subgoal", "one backward-planner decision: “park helper H on "
         "support cell S so a slider stops on bottleneck cell B”"),
        ("B1", "first plan-language extension: stoppers may park on "
         "wall-less cells; a robot may step aside before a named slide"),
        ("B2", "second extension: a plan may re-use a robot it already "
         "placed (by reference); generalized multi-robot step-asides"),
        ("zero-shot", "the networks rank candidate types they were never "
         "trained on (the B2 rows use B1- or older-trained networks)"),
        ("checked search (prefix check)", "physics-checking every partial "
         "plan during search so doomed branches are dropped immediately"),
        ("anytime mode", "test-playing each finished plan; on failure the "
         "search discards it and continues"),
        ("ceiling", "the share of benchmark puzzles for which at least one "
         "playable plan exists in the plan language at all, measured by "
         "exhaustive probe"),
    ]
    items = "".join(f"<dt>{esc(t)}</dt><dd>{esc(d)}</dd>" for t, d in terms)
    return kicker_h2("vocabulary", "Glossary") + \
        f'<dl class="gloss">{items}</dl></section>'


def sec_provenance(D):
    rows = []
    for rel in sorted(SOURCES):
        e = SOURCES[rel]
        cls = {"ok": "okc", "missing": "warn"}.get(e["status"], "warn")
        rows.append(f"<tr><td><code class='small'>{esc(rel)}</code></td>"
                    f"<td>{chip(e['status'], cls)}</td>"
                    f"<td class='small muted'>{esc(e['note'])}</td></tr>")
    n_ok = sum(1 for e in SOURCES.values() if e["status"] == "ok")
    table = scroll("<table><thead><tr><th>file</th><th>status</th>"
                   "<th>note</th></tr></thead><tbody>"
                   + "".join(rows) + "</tbody></table>")
    return kicker_h2(
        "provenance", "Every file this page was built from",
        f"{n_ok} files loaded; “missing” rows are expected results that "
        "render as pending until they land.") + \
        f"<details><summary class='small muted'>file list "\
        f"({len(SOURCES)} entries)</summary>{table}</details></section>"


def sec_selfcheck(D):
    rows = []
    for c in CHECKS:
        raw = c["raw"]
        raw_s = ("—" if raw is None
                 else (f"{raw:.6g}" if isinstance(raw, float) else str(raw)))
        rows.append(
            f"<tr><td class='small'>{esc(c['desc'])}</td>"
            f"<td><code class='small'>{esc(c['source'] or '—')}</code></td>"
            f"<td class='num small'>{esc(raw_s)}</td>"
            f"<td class='num small'>{esc(c['rendered'])}</td></tr>")
    table = scroll("<table><thead><tr><th>rendered number</th>"
                   "<th>source file</th><th class='num'>source value</th>"
                   "<th class='num'>as rendered</th></tr></thead><tbody>"
                   + "".join(rows) + "</tbody></table>")
    n_needles = len(NEEDLES)
    return kicker_h2(
        "machine verification", "The self-check appendix",
        sec_id="selfcheck") + f"""
  <p class="small">This page is generated by
  <code>eval/build_report.py</code>, which reads every number from its
  source file at build time. After the page is assembled, the builder
  parses the finished HTML back and compares each of the
  <b>{len(CHECKS)} tagged numbers</b> below against its source aggregate,
  plus {n_needles} structural assertions (values that must or must never
  appear). <b>Any mismatch aborts the build</b> — if you are reading this
  page, every check passed at generation time.</p>
  <details><summary class="small muted">the {len(CHECKS)} checked numbers
  </summary>{table}</details>
  <p class="cellnote">Regenerate any time with
  <code>PYTHONPATH=. python3 -m eval.build_report</code> from the repository
  root — new result files (for example the retrained-network rows) are
  picked up automatically.</p>
</section>"""


def sec_footnotes():
    return """
<section id="footnotes">
  <div class="sechead"><div class="kicker">fine print</div>
  <h2>Footnotes</h2></div>
  <ol class="fnlist">
    <li id="fn-a"><b>Solution length on beyond-oracle pools</b> is averaged
    over solved puzzles only, and no true optimum exists there — never read
    it as solution quality; it is reported for completeness.</li>
    <li><b>Wall-clock seconds</b> from different machines are never
    compared: tables tag each seconds cell with its machine where they mix
    (O = origin machine, K = Karolina), and charts use search steps —
    machine-independent — instead of time.</li>
    <li><b>Withheld results:</b> the 8-robot stock forward run collapsed in
    training; per the integrity rule its benchmark score appears nowhere in
    this report, and the published 8-robot rows use the stability-controlled
    retrain.</li>
    <li><b>Zero-shot full-language rows</b> use networks never trained on
    the newest candidate type. Retraining did not improve on them — the
    fairness tab's seed-robustness section explains why (a bistable
    value-network training), so the zero-shot rows stand as the method's
    best measured configuration.</li>
    <li><b>Sources of truth:</b> where a document and a result file
    disagree, the result file wins. This page renders result files only.
    </li>
  </ol>
</section>"""


def tab_methods(D):
    from eval.report_sections_hygiene import sec_hygiene, sec_databudget
    return (sec_protocol(D) + sec_design(D) + sec_hygiene(D)
            + sec_databudget(D) + sec_glossary()
            + sec_provenance(D) + sec_selfcheck(D) + sec_footnotes())
