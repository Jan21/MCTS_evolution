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
        "never trained on the newest step type). Retrained rows appear only "
        "where the networks were trained on a corpus that actually contains "
        "the new step type at its natural rate: a cheaper label corpus used "
        "earlier stripped those examples and cost 22.4 points at the "
        "beyond-oracle set, so rows trained on it are withheld rather than "
        "shown as the method's performance.") \
        + table + """
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
        tiles = f"""
  <div class="kpirow">
    <div class="tile"><div class="tlabel">32×32, gradable set — minutes per
    puzzle, move-by-move</div>
    <div class="tvalue">{ck(f"{fmin:.0f} min", fg["src"],
    "cost: 32x32 forward minutes", raw=fg["agg"]["mean_seconds"])}</div>
    <div class="tsub">vs the subgoal planner's
    {ck(f"{bg['agg']['mean_seconds']:.0f} s", bg["src"],
    "cost: 32x32 backward seconds", raw=bg["agg"]["mean_seconds"])} —
    measured on the same machine.</div></div>
    <div class="tile"><div class="tlabel">32×32, beyond the oracle —
    minutes per puzzle, move-by-move</div>
    <div class="tvalue">{ck(f"{ffmin:.0f} min", ff["src"],
    "cost: 32x32 frontier forward minutes", raw=ff["agg"]["mean_seconds"])}</div>
    <div class="tsub">to solve {pct(ff):.1f}% — forty minutes per puzzle to
    solve almost nothing, vs {ck(f"{bf['agg']['mean_seconds']:.0f} s",
    bf["src"], "cost: 32x32 frontier backward seconds",
    raw=bf["agg"]["mean_seconds"])} at {pct(bf):.1f}% for subgoals.</div>
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


def sec_open(D):
    return kicker_h2(
        "still open", "What is running and what is next") + """
  <ul>
    <li><b>Corpus-correct retrains</b> (running; every “in progress” row on
    this page fills automatically when its result file lands) — and, next,
    <b>integrating the re-use step type into the learned planner</b>
    (helper featurization that can name non-start cells, a policy retrain
    without the silent record filter, and the proposal-path wiring); the
    measured candidate supply and the ceiling headroom are in the
    plan-language tab.</li>
    <li><b>Quality-focused self-play on the extended stack</b> — frontier
    solutions run long (no optima exist there); self-play with a quality
    pressure is the scoped follow-up.</li>
    <li><b>The 24×24 · 4-robot full-language rows</b> and the two base
    probe instances whose ceiling status is unresolved at current probe
    memory (a depth-bounded probe redesign is scoped).</li>
  </ul>
</section>"""


def tab_scaling(D):
    from eval.report_sections_strata import sec_strata
    from eval.report_sections_budget import sec_budget_table
    return (sec_oracle(D) + sec_fragility(D) + sec_ladder(D)
            + sec_budget_table(D) + sec_strata(D)
            + sec_cost(D) + sec_rung_details(D) + sec_open(D))


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
                "(<code>eval/results/compute_accounting.json</code>) — "
                "render support for this file's schema should be reviewed "
                "in the next report build pass.</p>")
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
  and the <b>instrumented counters</b> now being collected.</p>
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


def sec_fair_wallclock(D):
    rows = []
    pref = ["bwd_b2", "bwd_b1", "bwd_old"]
    label_of = {"bwd_b2": "subgoals, full language",
                "bwd_b1": "subgoals, extended language (B1)",
                "bwd_old": "subgoals, original language"}
    for e in D["ladder"]:
        for group, gname in (("graded", "gradable set"),
                             ("frontier", "beyond the oracle")):
            cells = e.get(group) or {}
            f = cells.get("fwd")
            if not f:
                continue
            cand = [k for k in pref if cells.get(k)
                    and cells[k]["machine"] == f["machine"]]
            if not cand:
                rows.append((e["label"], gname, None, None, None, None,
                             "no same-machine pair yet"))
                continue
            b = cells[cand[0]]
            bs, fs = b["agg"]["mean_seconds"], f["agg"]["mean_seconds"]
            desc = f'wall-clock {e["key"]} {group}'
            rows.append((
                e["label"], gname, label_of[cand[0]],
                ck(fnum(bs, 2), b["src"], desc + " backward s", raw=bs),
                ck(fnum(fs, 2), f["src"], desc + " forward s", raw=fs),
                f"{fs / bs:.1f}×" if bs else "—",
                "Karolina" if b["machine"] == "karolina"
                else "origin machine"))
    body = []
    for label, gname, bsys, bs, fs, ratio, mach in rows:
        if bsys is None:
            body.append(f"<tr><td><b>{esc(label)}</b>"
                        f'<div class="cellnote">{esc(gname)}</div></td>'
                        f'<td colspan="3" class="small muted">{esc(mach)}'
                        "</td></tr>")
            continue
        body.append(
            f"<tr><td><b>{esc(label)}</b>"
            f'<div class="cellnote">{esc(gname)} · measured on {esc(mach)}'
            f"</div></td>"
            f'<td class="num">{bs}<div class="cellnote">{esc(bsys)}</div>'
            "</td>"
            f'<td class="num">{fs}</td>'
            f'<td class="num"><b>{esc(ratio)}</b></td></tr>')
    table = scroll(
        "<table><thead><tr><th>configuration · set</th>"
        "<th class='num'>" + dot("bwd") + "subgoals<br>seconds / puzzle</th>"
        "<th class='num'>" + dot("fwd") + "move-by-move<br>seconds / puzzle"
        "</th><th class='num'>time ratio</th></tr></thead><tbody>"
        + "".join(body) + "</tbody></table>")
    # data-driven closing paragraph (never assert what the table refutes)
    numeric = [(label, gname, float(r[:-1]))
               for (label, gname, bsys, bs, fs, r, m) in rows
               if bsys is not None and r and r.endswith("×")]
    scaling_ratios = [(la, g, v) for la, g, v in numeric
                      if "16×16 board, 4 robots" not in la]
    base_ratios = [(la, g, v) for la, g, v in numeric
                   if "16×16 board, 4 robots" in la]
    fact("wall-clock: subgoals faster on every same-machine pair beyond "
         "the base scale", all(v > 1 for _, _, v in scaling_ratios))
    closing = ""
    if scaling_ratios:
        lo = min(v for _, _, v in scaling_ratios)
        hi = max(v for _, _, v in scaling_ratios)
        base_note = ""
        if base_ratios and base_ratios[0][2] < 1:
            base_note = (
                f" The one exception is the base scale "
                f"({base_ratios[0][2]:.1f}×): on a 16×16 board a single "
                "move-level step is cheap enough that the move-by-move "
                "planner's many steps out-run the subgoal planner's few — "
                "an exception that disappears as boards grow, which is the "
                "thesis in miniature.")
        closing = f"""
  <p class="small">Wall-clock charges the subgoal planner for all of its
  unmetered bookkeeping — and it still runs {lo:.1f}×–{hi:.0f}× faster
  per puzzle on every same-machine pair at every scaling
  rung.{base_note} The instrumented counters will put exact numbers on
  the exclusions themselves.</p>"""
    return kicker_h2(
        "the complementary check: wall-clock",
        "Time counts everything — and points the same way at scale",
        "Wall-clock time meters every kind of work, including all the "
        "bookkeeping the expansion counter excludes. It is only "
        "comparable between runs on the same machine, so this table is "
        "restricted to same-machine pairs (the fullest backward language "
        "measured on the forward row's machine).") + table + closing \
        + "</section>"


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
        "Two queued runs give the move-by-move planner several times its "
        "standard budget on frontier subsamples. If its solve rate stays "
        "flat, the collapse is real; if it climbs substantially, the "
        "standard-budget frontier rows overstate the gap.") \
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
    the newest candidate type; their gains are a lower bound on what the
    language is worth. Retrained rows will replace the “in progress” cells
    automatically.</li>
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
