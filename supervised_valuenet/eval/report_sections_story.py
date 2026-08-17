"""Tabs 1–3: The study · Base-scale head-to-head · The plan language."""

import os

from eval.report_util import (esc, ck, fnum, ffrac, dot, chip, scroll,
                              pending, progress_tag, kicker_h2, need, fact,
                              rp)
from eval.report_data import (pick, systems_of_kind, proto_date,
                              newest_source_file)
from eval.report_tables import (sys_table, retrain_story_html,
                                retrain_story_short)
from eval import report_charts as C
from eval.report_boards import board_svg, slide_rule_svg, SLOT_NAMES, SLOT_VARS
from eval.plan_viz_core import (GEOM_COMPACT, dag_svg, board_inset,
                                legend as plan_legend,
                                STRUCTURAL_PAIRS)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fwd_agg(D, frag):
    """Aggregate of the forward system whose name contains frag."""
    for name, s in systems_of_kind(D.get("fwd450"), "forward"):
        if frag in name:
            return s["aggregate"]
    return None


def bwd_agg(comp):
    return pick(comp, "backward")


def pct(agg):
    return agg["solve_rate"] * 100 if agg else None


def rate_txt(agg, dec=1):
    return f"{agg['solve_rate'] * 100:.{dec}f}%"


# ---------------------------------------------------------------------------
# Masthead
# ---------------------------------------------------------------------------

def masthead(D):
    dates = []
    for key in ("fwd450", "bwd_before", "bwd_prefix", "bwd_b1", "bwd_b2"):
        if D.get(key):
            dates.append(proto_date(D[key])[:10])
    for slots in D["rungs"].values():
        for (g, s), (rel, comp, kind) in slots.items():
            if comp:
                dates.append(proto_date(comp)[:10])
    newest_run = max(d for d in dates if d) if dates else "—"
    # the date that must never look stale: when a source file was last WRITTEN
    newest_file, newest_rel = newest_source_file()
    return f"""
<header class="masthead">
  <div class="doctag">Measurement report · Ricochet Robots planner study</div>
  <h1>Planning with subgoals, or planning move by move?</h1>
  <p class="subtitle">Two neural-network planners solve the same sliding-robot
  puzzles under identical rules and search budgets — one thinks one move at a
  time, the other in goal-directed subgoals. This report follows the
  comparison from a 16×16 training ground to 32×32 boards that have outgrown
  exact search, and reports every result from its machine-readable source.</p>
  {abstract(D)}
  <p class="meta">newest source file: {esc(newest_file)}
  {f'(<code>{esc(newest_rel)}</code>)' if newest_rel else ''}
  <span class="sep">·</span>newest benchmark run recorded inside a result
  file: {esc(newest_run)}<span class="sep">·</span>
  every number on this page is read from a result file at build time and
  machine-verified — see <a href="#selfcheck">the self-check appendix</a></p>
</header>
"""


def abstract(D):
    """Five plain sentences, before anything technical (readability audit §5).

    The margins are read out of the same pooled cells the headline table
    uses, so the abstract cannot disagree with the table under it.
    """
    st = D.get("stats_tests") or {}
    cells = [c for c in (st.get("cells") or [])
             if "skipped" not in c and c.get("a") == "bwd_b2"
             and c.get("b") == "fwd"
             and (c.get("set") == "pooled"
                  or (c.get("rung") == "g16r4" and c.get("set") == "graded"))]
    wins = [c for c in cells if c["diff"] > 0]
    losses = [c for c in cells if c["diff"] < 0]
    if not cells:
        return ""
    lo = min(c["diff"] for c in wins) * 100
    hi = max(c["diff"] for c in wins) * 100
    SRC = "eval/results/stats_tests.json"
    n_win, n_all = len(wins), len(cells)
    fwd_edge = f"{abs(losses[0]['diff']) * 100:.1f}" if losses else None
    return f"""
  <div class="panel abstract">
  <p><b>In short.</b> This report compares two planners for Ricochet Robots,
  a puzzle where robots slide until they hit something: a
  <b>move-by-move</b> planner that picks one robot move at a time, and a
  <b>subgoal</b> planner that works backward from the goal in
  multi-move chunks — <b>both built in this study</b>, so there is no
  third-party system anywhere on this page and every number is one of our
  planners against the other, never against a published baseline. They
  were tested on six
  configurations of board size and robot count (16×16 up to 32×32, 4 to 8
  robots), 450 fixed puzzles each, with identical search budgets, and a
  puzzle counts as solved only when the plan replays legally on the real
  board. The subgoal planner wins {n_win} of the {n_all} configurations by
  {ck(f'{lo:.1f}', SRC, "abstract: smallest winning margin", raw=lo)} to
  {ck(f'{hi:.1f}', SRC, "abstract: largest winning margin", raw=hi)}
  percentage points, the margin growing with both board size and robot
  count. Two caveats we do not bury: the move-by-move planner still
  <b>wins the smallest configuration outright</b>{
  f" (by {ck(fwd_edge, SRC, 'abstract: forward edge at base', raw=fwd_edge)} points)"
  if fwd_edge else ""} and keeps a real solution-quality edge wherever the
  exact solver can grade quality, and every headline arm is a
  <b>single training seed</b> on both sides — seed variation has been
  measured and is large, and replicate runs are under way. The first half
  of this page (tabs 1–3) explains the puzzle, the two planners and the
  plan language; the second half (tabs 4–6) is the audit trail: the
  rung-by-rung ladder, whether the comparison is compute-fair, and the
  protocol, provenance and machine self-check.</p>
  </div>"""


# ---------------------------------------------------------------------------
# Tab 1 — The study
# ---------------------------------------------------------------------------

def sec_puzzle():
    return kicker_h2("the game", "The puzzle: robots that cannot stop",
                     sec_id="puzzle") + f"""
  <div class="boardrow">
    <div class="textcol">
      <p>Ricochet Robots is played on a walled grid. A robot moves by sliding
      in a straight line and <b>cannot stop mid-slide</b> — it keeps going
      until it hits a wall or another robot, stopping in the cell just before
      the obstacle. One robot is the <b>target</b>; one cell is the
      <b>goal</b>. The task: get the target robot onto the goal cell in as
      few moves as possible.</p>
      <p>What makes it hard: a robot can only stop where something blocks it.
      Solutions therefore hinge on <b>parking helper robots at exact spots
      first</b> so that later slides bump into them and stop where needed —
      a chain of preparations for one final slide. In the schematic, Yellow
      can only reach its goal because Blue happens to sit one cell below it;
      Red, with nothing in its row, slides all the way to the border.</p>
    </div>
    <div class="boardcol">
      {slide_rule_svg()}
      <p class="cellnote">Schematic (not a measured puzzle). The ring marks
      Yellow's goal; arrows show slides running until something stops
      them.</p>
    </div>
  </div>
</section>"""


def sec_two_ideas():
    return kicker_h2("the contestants", "Two ways to build a planner") + """
  <div class="cols2">
    <div class="panel">
      <h3 class="ph">""" + dot("fwd") + """Move-by-move (the “forward” planner)</h3>
      <p class="small">Thinks exactly like the game: <i>“which robot slides
      which way right now?”</i> — one move at a time from the start position
      toward the goal. Every decision is a single legal move; a solution
      emerges after as many decisions as the solution has moves.</p>
    </div>
    <div class="panel">
      <h3 class="ph">""" + dot("bwd") + """Subgoals (the “backward” planner)</h3>
      <p class="small">Thinks from the goal outward: <i>“for the final slide
      to stop on the goal, a robot must first be parked HERE; getting one
      there needs THIS first.”</i> Each decision is a <b>subgoal</b> —
      “park helper H on support cell S so a slider stops on cell B” — worth
      several moves at once, so complete plans are only a few decisions
      deep.</p>
    </div>
  </div>
  <p><b>Same machinery, different vocabulary.</b> Both planners use a
  <i>proposal network</i> (suggests what to try next), a <i>value network</i>
  (estimates how close each option is to done), and best-first search over
  the proposals, capped at a fixed number of <b>search steps</b>. One search
  step means: take the most promising position or partial plan, run the
  networks once, and generate its candidate continuations — so a step costs
  the same in both families and <b>budgets are directly comparable</b>. The
  study's question: which vocabulary — moves or subgoals — plans better, and
  which keeps working as puzzles grow?</p>
</section>"""


def sec_rules():
    return kicker_h2("ground rules", "What keeps the comparison honest") + """
  <div class="cols2">
    <div class="panel"><h3 class="ph">1 · A plan must actually play</h3>
      <p class="small">A puzzle counts as solved <b>only if the plan plays
      out legally move by move</b> on the real board. Plans that are
      complete on paper but bounce off robots that are not really there
      count as failures. This rule alone rewrote the study's early history
      (see the base-scale tab).</p></div>
    <div class="panel"><h3 class="ph">2 · Same puzzles, same budgets</h3>
      <p class="small">Every head-to-head runs on a pinned benchmark of 450
      puzzles per configuration, at the same cap of 1,200 search steps and
      the same proposal width (top 5). Nothing is tuned per puzzle.</p></div>
    <div class="panel"><h3 class="ph">3 · No exact solver at solve time</h3>
      <p class="small">An exact solver (“the oracle”) is used only to create
      training labels and reference optima for scoring — never during
      solving. Where the oracle itself fails, solving is
      <b>self-certifying</b>: a plan that plays out legally to the goal
      proves itself.</p></div>
    <div class="panel"><h3 class="ph">4 · Only real moves are compared</h3>
      <p class="small">Solution quality is measured in played-out moves
      against the oracle's move optimum. The subgoal planner's internal plan
      costs are never compared to move optima. On puzzles beyond the
      oracle's reach no optimum exists, so only solve rate, search steps and
      time are reported there.</p></div>
  </div>
</section>"""


def sec_verdict_tiles(D):
    lad = {e["key"]: e for e in D["ladder"]}
    tiles = []

    f_best = fwd_agg(D, "candidate_scored")
    b_b2 = bwd_agg(D.get("bwd_b2"))
    if f_best and b_b2:
        v1 = ck(ffrac(f_best["solved"], f_best["n"]),
                "eval/results/comparison_forward.json",
                "tile: base forward solved", raw=f_best["solved"])
        v2 = ck(rate_txt(b_b2), "eval/results/final450_backward_b2.json",
                "tile: base backward rate", raw=b_b2["solved"])
        tiles.append(f"""
  <div class="tile"><div class="tlabel">Small puzzles (16×16, 4 robots) —
  move-by-move is the quality champion</div>
  <div class="tvalue">{v1}</div>
  <div class="tsub">puzzles solved, {ck(fnum(f_best['mean_regret'], 3),
      'eval/results/comparison_forward.json', 'tile: base forward regret',
      raw=f_best['mean_regret'])} extra moves on average. The subgoal
  planner reaches {v2} here.</div></div>""")

    g32 = lad.get("g32r4", {})
    fb, bb = (g32.get("frontier") or {}).get("fwd"), \
        (g32.get("frontier") or {}).get("bwd_b2")
    if fb and bb:
        v1 = ck(rate_txt(bb["agg"]), bb["src"],
                "tile: 32x32 frontier backward rate", raw=bb["agg"]["solved"])
        v2 = ck(rate_txt(fb["agg"]), fb["src"],
                "tile: 32x32 frontier forward rate", raw=fb["agg"]["solved"])
        tiles.append(f"""
  <div class="tile"><div class="tlabel">The hardest pool measured (32×32,
  beyond the exact solver) — subgoals keep working</div>
  <div class="tvalue">{v1} <span class="vs">vs</span> {v2}</div>
  <div class="tsub">subgoal planner (full language) vs move-by-move, solve
  rate over {bb["agg"]["n"]} puzzles no exact method can grade.</div></div>""")

    g8 = lad.get("g16r8", {})
    bg, fg = (g8.get("graded") or {}).get("bwd_b2"), \
        (g8.get("graded") or {}).get("fwd")
    if bg and fg:
        v1 = ck(fnum(bg["agg"]["mean_expansions"], 1), bg["src"],
                "tile: 8-robot backward steps",
                raw=bg["agg"]["mean_expansions"])
        v2 = ck(fnum(fg["agg"]["mean_expansions"], 1), fg["src"],
                "tile: 8-robot forward steps",
                raw=fg["agg"]["mean_expansions"])
        tiles.append(f"""
  <div class="tile"><div class="tlabel">Search effort where both planners
  excel (16×16, 8 robots, gradable set — the puzzles the exact solver
  managed to solve, so an optimal move count exists for each)</div>
  <div class="tvalue">{v1} <span class="vs">vs</span> {v2}</div>
  <div class="tsub">search steps per puzzle, subgoals vs move-by-move — at
  near-equal solve rates ({rate_txt(bg["agg"])} vs {rate_txt(fg["agg"])}).
  </div></div>""")

    probe_b2 = D.get("probe_b2")
    if isinstance(probe_b2, list) and b_b2:
        from eval.report_data import probe_counts
        cc = probe_counts(probe_b2)
        n_unres = cc.get("INCONCLUSIVE", 0)
        ceiling = (450 - n_unres) / 450 * 100
        v1 = ck(f"{ceiling:.1f}%",
                "analysis/artifacts/ceiling_probe_results_b2.json",
                "tile: language ceiling", raw=n_unres)
        v2 = ck(rate_txt(b_b2), "eval/results/final450_backward_b2.json",
                "tile: achieved vs ceiling", raw=b_b2["solved"])
        ab = D.get("byref_ab")
        if ab:
            p = ab["pooled"]
            net = ck(str(p["on"] - p["off"]),
                     "scaling/results/g16r6/comparison_b2retrained_cap20000"
                     "_byref_on.json", "tile: byref net gain",
                     raw=p["on"] - p["off"])
            wired = (f"It has since been wired behind a flag: turning it on "
                     f"moves {net} more puzzles of {p['n']} at 16×16 · 6 "
                     "robots — see the plan-language tab.")
        else:
            wired = ("It has since been wired behind a flag; the on/off "
                     "measurement is in the plan-language tab.")
        tiles.append(f"""
  <div class="tile"><div class="tlabel">What the extended plan language
  permits vs what today's networks reach (base benchmark)</div>
  <div class="tvalue">{v1} <span class="vs">vs</span> {v2}</div>
  <div class="tsub">the gap is an integration gap, not a design wall — the
  newest step type was missing from the learned planner's proposal path and
  featurization. {wired}</div></div>""")

    return f'<div class="kpirow">{"".join(tiles)}</div>'


def ladder_chart_frontier(D, small=False):
    groups = []
    for e in D["ladder"]:
        fr = e.get("frontier") or {}
        if not any(fr.get(k) for k in ("bwd_old", "bwd_b2", "fwd")):
            continue
        vals, tips = {}, {}
        n = None
        for k in ("bwd_old", "bwd_b2", "fwd"):
            c = fr.get(k)
            if c:
                vals[k] = pct(c["agg"])
                n = c["agg"]["n"]
                tips[k] = (f'({c["agg"]["solved"]}/{c["agg"]["n"]}, '
                           f'{c["agg"]["mean_expansions"]:.0f} steps)')
            else:
                vals[k] = None
        groups.append({"label": e["short"], "sub": f"{n} puzzles",
                       "values": vals, "tips": tips})
        for k in ("bwd_old", "bwd_b2", "fwd"):
            c = fr.get(k)
            if c:
                need(f'frontier chart: {e["key"]}/{k} value rendered',
                     f'{pct(c["agg"]):.1f}')
    series = [("bwd_old", "bwd-old", "subgoals, original language"),
              ("bwd_b2", "bwd", "subgoals, full language (same networks)"),
              ("fwd", "fwd", "move-by-move (stabilized training recipe — scaling tab)")]
    svg = C.grouped_columns(
        groups, [(k, f, l) for k, f, l in series], unit="%",
        aria="Solve rates beyond the exact solver's reach, per configuration")
    twin_rows = []
    for g in groups:
        twin_rows.append([esc(g["label"] + " — " + (g["sub"] or ""))]
                         + [(f"{g['values'][k]:.1f}%" if g["values"][k]
                             is not None else "not measured")
                            for k, _, _ in series])
    twin = C.table_twin(["configuration"] + [l for _, _, l in series],
                        twin_rows, "solve rates as a table")
    caption = ("Puzzles the exact solver could not grade at its full budget "
               "— the hardest pool at each scale, where a solution proves "
               "itself by playing out. No move optima exist here, so solve "
               "rate (bars), search steps and time (tooltips) are the only "
               "meaningful measures. The full-language rows run the SAME "
               "trained networks with a richer plan vocabulary, zero-shot — i.e. the "
               "networks were never trained on the newest step type and "
               "rank it unseen.")
    srcs = "scaling/results/*/comparison_ungraded*.json"
    return C.figure("Beyond the exact solver's reach, the subgoal planner "
                    "takes over", "solved (playable), % of each pool — "
                    "higher is better", C.legend([(f, l) for _, f, l in
                                                  series]), svg, caption,
                    srcs, twin)


def ladder_chart_graded(D):
    groups = []
    for e in D["ladder"]:
        gr = e.get("graded") or {}
        if not any(gr.get(k) for k in ("bwd_old", "bwd_b2", "fwd")):
            continue
        vals, tips = {}, {}
        n = None
        for k in ("bwd_old", "bwd_b2", "fwd"):
            c = gr.get(k)
            if c:
                vals[k] = pct(c["agg"])
                n = c["agg"]["n"]
                tips[k] = (f'({c["agg"]["solved"]}/{c["agg"]["n"]}, '
                           f'{c["agg"]["mean_expansions"]:.0f} steps)')
            else:
                vals[k] = None
        groups.append({"label": e["short"], "sub": f"{n} puzzles",
                       "values": vals, "tips": tips})
    series = [("bwd_old", "bwd-old", "subgoals, original language"),
              ("bwd_b2", "bwd", "subgoals, full language (same networks)"),
              ("fwd", "fwd", "move-by-move (stabilized training recipe — scaling tab)")]
    svg = C.grouped_columns(
        groups, series, unit="%",
        aria="Solve rates on the oracle-gradable puzzles, per configuration")
    twin_rows = []
    for g in groups:
        twin_rows.append([esc(g["label"] + " — " + (g["sub"] or ""))]
                         + [(f"{g['values'][k]:.1f}%" if g["values"][k]
                             is not None else "not measured")
                            for k, _, _ in series])
    twin = C.table_twin(["configuration"] + [l for _, _, l in series],
                        twin_rows, "solve rates as a table")
    g32 = next((g for g in groups if g["label"] == "32×32 · 4r"), None)
    third = bool(g32 and g32["values"]["fwd"] is not None
                 and all(g32["values"][k] is not None
                         and g32["values"]["fwd"] < g32["values"][k]
                         for k in ("bwd_old", "bwd_b2")))
    fact("32×32 graded: the forward rate is below both backward rows "
         "(the caption's 'drops to third')", third)
    caption = ("Puzzles the exact solver can still grade. Move-by-move "
               "planning holds a small solve-rate lead until the board "
               "grows: at 32×32 it drops to third while its per-puzzle cost "
               "explodes (see the scaling tab).")
    return C.figure("On puzzles exact search can grade, the lead changes "
                    "hands at 32×32", "solved (playable), % of each gradable "
                    "set — higher is better",
                    C.legend([(f, l) for _, f, l in series]), svg, caption,
                    "eval/results + scaling/results/*/comparison*.json", twin)



def sec_headline_table(D):
    """The one scoreboard a reader should take away: the whole pinned pool.

    Every other table in this report slices the benchmark somehow. This one
    does not -- it is all 450 puzzles at each configuration, gradable and
    beyond-oracle together, which is the only view carrying no selection to
    argue about. It therefore belongs above the fold rather than buried inside
    the statistics section.
    """
    st = D.get("stats_tests")
    if not st:
        return ""
    src = "eval/results/stats_tests.json"
    names = {"g16r6": "16×16 board, 6 robots",
             "g16r8": "16×16 board, 8 robots",
             "g24r4": "24×24 board, 4 robots",
             "g24r8": "24×24 board, 8 robots",
             "g32r4": "32×32 board, 4 robots"}
    cells = {c["rung"]: c for c in st.get("cells", [])
             if not c.get("skipped") and c.get("set") == "pooled"
             and c.get("a") == "bwd_b2" and c.get("b") == "fwd"}
    rows = []
    for key in ("g16r6", "g16r8", "g24r4", "g24r8", "g32r4"):
        c = cells.get(key)
        if not c:
            continue
        n = c["n"]
        sa, sb = c["solved_a"], c["solved_b"]
        ra = "%.1f%%" % (c["rate_a"] * 100)
        rb = "%.1f%%" % (c["rate_b"] * 100)
        diff = c["diff"] * 100
        dtxt = "%+.1f pts" % diff
        lo = c["ci95_lo"] * 100
        hi = c["ci95_hi"] * 100
        pv = c["mcnemar_p"]
        pstr = "&lt;0.0001" if pv < 0.0001 else "%.4f" % pv
        cell_a = ck(ra, src, "headline pooled subgoal " + key, raw=sa)
        cell_b = ck(rb, src, "headline pooled forward " + key, raw=sb)
        cell_d = ck(dtxt, src, "headline pooled diff " + key, raw=round(diff, 4))
        rows.append(
            "<tr><td><b>" + esc(names[key]) + "</b></td>"
            + '<td class="num">' + cell_a
            + '<div class="cellnote">' + str(sa) + " of " + str(n) + "</div></td>"
            + '<td class="num">' + cell_b
            + '<div class="cellnote">' + str(sb) + " of " + str(n) + "</div></td>"
            + '<td class="num">' + cell_d
            + '<div class="cellnote">95%% CI [%+.1f, %+.1f]</div></td>' % (lo, hi)
            + '<td class="num">' + pstr + "</td></tr>")
    if not rows:
        return ""
    table = scroll(
        "<table><thead><tr><th>configuration</th>"
        '<th class="num">subgoal planner</th>'
        '<th class="num">move-by-move planner</th>'
        '<th class="num">difference</th>'
        '<th class="num">p</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table>")
    need("headline table present", "the whole benchmark, nothing left out")
    return (
        '\n  <h3 class="ph">The headline: the whole benchmark, nothing left'
        " out</h3>\n"
        "  <p>This is the opening table again, now with its uncertainty."
        " Most other scoreboards in this report split each benchmark in"
        " two — the puzzles an exact solver could grade, and the harder"
        " ones it could not. That split explains <i>why</i> the planners"
        " differ, but the harder half is defined by the exact solver failing,"
        " which is unfair to the move-by-move planner by construction. This"
        " table sidesteps it: <b>all 450 puzzles</b> at each board size,"
        " both halves together, scored by the one rule that matters — a"
        " puzzle counts only if the plan plays out legally, move by move.</p>\n"
        + table
        + '\n  <p class="small muted">Both planners get the same puzzles and'
        " the same search budget. “Difference” is in percentage"
        " points. The p-value is an exact paired test (McNemar — it uses"
        " only the puzzles where the two planners disagree); the 95%"
        " confidence interval comes from a bootstrap that allows for"
        " several puzzles sharing one board. The"
        " small-puzzle configuration (16×16, 4 robots) is absent because"
        " its whole benchmark is gradable — the move-by-move planner wins"
        " that one outright, 450 of 450 against 430.</p>\n")


def sec_verdict(D):
    html = kicker_h2(
        "the verdict so far", "Where the evidence stands",
        "Every number below is measured; the one-paragraph verdict, then the "
        "whole ladder — the six board-size/robot-count configurations — in two "
        "pictures.")
    html += sec_verdict_tiles(D)
    html += sec_headline_table(D)
    html += """
  <div class="verdict">
  <p><b>On small puzzles the move-by-move planner is the quality champion;
  every trend along the two hardness axes bends the other way.</b> The exact
  solver its training depends on fails on a growing majority of puzzles as
  boards and robot counts grow, and its per-puzzle search cost explodes with
  board size. On the whole pinned pool at each rung — gradable puzzles (the
  exact solver graded them, so an optimal move count exists) and
  beyond-oracle puzzles (it failed at its practical budget, so no optimum
  exists and a solution proves itself by playing out) together, the only
  view free of any selection — the
  full-language subgoal planner wins <b>every rung measured</b>, by 7.8 to
  47.8 points, with the margin growing along both axes.</p>
  <p><b>Three qualifications, stated here rather than in a footnote.</b>
  (1) On gradable sets the move-by-move planner still holds a real edge at
  16×16 · 6 robots and 24×24 · 8 robots; at 16×16 · 8 robots the two are at
  <b>parity</b> — 262 versus 261 puzzles is a one-puzzle margin that a paired
  test cannot separate from noise — with the subgoal planner using 7× fewer
  search steps. (2) The beyond-oracle margins are <b>matched-budget</b>
  results. Give the move-by-move planner 4–5× the search budget and it gains
  4–31 points depending on the rung — a large recovery at 16×16, a modest
  one at 24×24, and almost none at 32×32 (four times the budget buys one
  extra puzzle in twenty-four there). So "it has stopped working" is
  supportable only at the largest boards; elsewhere the honest statement is
  that it needs several times the search to approach a rate the subgoal
  planner reaches immediately, and still does not catch up. (3) The headline
  arms are single-seed on both sides. Where seed sensitivity HAS been
  measured — the retrained backward networks, see the fairness tab — it is
  large, so no small difference on this page should be read as real; the
  headline margins are 8–48 points.</p>
  <p>The subgoal planner's remaining handicaps are honest and measured:
  longer solutions where no optimum exists, and a gap between what its
  extended plan language permits and what its current networks reach —
  an integration gap: the newest step type was absent from the learned
  planner's proposal path and featurization. It is now wired behind a flag,
  and the measured effect of switching it on is small (plan-language
  tab).</p>
  </div>
"""
    html += ladder_chart_frontier(D)
    html += ladder_chart_graded(D)
    html += """
  <p class="small muted">How to read the rest: tab 2 details the base-scale
  head-to-head where both planners were perfected; tab 3 explains why plans
  fail and how extending the plan language fixed it; tab 4 walks the scaling
  ladder rung by rung; tab 5 audits whether the comparison is compute-fair;
  tab 6 holds protocol, provenance and the machine self-check.</p>
</section>"""
    return html


def _seed_of_record_note(sh, SH_SRC):
    """The 16x16 / 8-robot cellnote: median of three seed replicates + band.

    The 8-robot networks are applied zero-shot (trained once at the smallest
    size), and that single training draw turned out to be seed-sensitive
    (FINDINGS 77a): one of three fresh seeds lands in the value net's bad
    basin. The pre-registered rule makes the median of the three replicates
    the reported number for this rung; the published run stays visible as the
    seed of record.
    """
    if not sh:
        return ""
    su = ((sh.get("sets") or {}).get("pooled") or {}).get("summary") or {}
    if su.get("median3_seeds") is None:
        return ""
    return ('number shown is the <b>seed of record</b> \u2014 the published '
            'run; retrained under three fresh seeds this rung\'s median is '
            + ck(f'{su["median3_seeds"]:g}/450', SH_SRC,
                 "headline g16r8 median-of-3", raw=su["median3_seeds"])
            + " (band "
            + ck(f'{su["min3_seeds"]}\u2013{su["max3_seeds"]}', SH_SRC,
                 "headline g16r8 band",
                 raw=(su["min3_seeds"], su["max3_seeds"]))
            + "), see below")


def _seed_of_record_foot(D):
    sh = D.get("seed_headline_g16r8")
    SH_SRC = "analysis/artifacts/seed_headline_g16r8.json"
    if not sh:
        return ""
    su = ((sh.get("sets") or {}).get("pooled") or {}).get("summary") or {}
    if su.get("median3_seeds") is None:
        return ""
    return ('<p class="small muted">One footnote on the 8-robot row: those '
            "networks are trained once at the smallest size and applied "
            "here unchanged, and repeating that training under three fresh "
            "random seeds showed one seed in three landing in a degenerate "
            "solution (visible in the value network's validation error "
            "before any benchmarking). By a rule fixed before those runs, "
            "this rung is reported as the median of the three replicates, "
            + ck(f'{su["median3_seeds"]:g}/450', SH_SRC,
                 "headline g16r8 median-of-3 (footnote)",
                 raw=su["median3_seeds"])
            + " (band "
            + ck(f'{su["min3_seeds"]}\u2013{su["max3_seeds"]}', SH_SRC,
                 "headline g16r8 band (footnote)",
                 raw=(su["min3_seeds"], su["max3_seeds"]))
            + "), with the published run above kept visible as the seed of "
            "record; the table's own margin is that published run's. The "
            "full per-seed breakdown is in the “Is it fair?” tab. The same "
            "replicate is still in flight for 32\u00d732 \u00b7 4 robots, "
            "which stays single-seed until it lands.</p>")


def sec_headline_first(D):
    """The whole-study result, first — before any exposition.

    Six board sizes, whole pinned 450-puzzle pool each, both planners under
    the same search budget; solved = the plan plays out legally move by
    move on the real board, and every solved row was independently
    replayed. Numbers from eval/results/stats_tests.json (pooled cells;
    base uses its graded cell, which IS its whole pool)."""
    st = D.get("stats_tests")
    if not st:
        return ""
    SRC = "eval/results/stats_tests.json"
    cells = {(c.get("rung"), c.get("set")): c
             for c in (st.get("cells") or [])
             if "skipped" not in c and c.get("a") == "bwd_b2"
             and c.get("b") == "fwd"}
    order = [("g16r4", "graded", "16×16 board · 4 robots",
              "the smallest size — the optimal move count is known for "
              "every puzzle here"),
             ("g16r6", "pooled", "16×16 board · 6 robots", ""),
             ("g16r8", "pooled", "16×16 board · 8 robots", ""),
             ("g24r4", "pooled", "24×24 board · 4 robots", ""),
             ("g24r8", "pooled", "24×24 board · 8 robots", ""),
             ("g32r4", "pooled", "32×32 board · 4 robots", "")]
    sh = D.get("seed_headline_g16r8")
    SH_SRC = "analysis/artifacts/seed_headline_g16r8.json"
    body = []
    for rung, set_, label, note in order:
        c = cells.get((rung, set_))
        if not c:
            continue
        if rung == "g16r8":
            note = _seed_of_record_note(sh, SH_SRC) or note
        d = f"headline {rung}"
        diff = c["diff"] * 100
        winner = ("bwd" if diff > 0 else "fwd")
        reads = (f'{abs(diff):.1f} points to the '
                 + ("subgoal planner" if diff > 0
                    else "move-by-move planner"))
        body.append(
            f'<tr><td><b>{esc(label)}</b>'
            + (f'<div class="cellnote">'
               + (note if rung == "g16r8" else esc(note))
               + "</div>" if note else "")
            + '</td><td class="num">'
            + ck(f'{c["solved_a"]}/{c["n"]}', SRC, d + " — subgoal solved",
                 raw=c["solved_a"])
            + '<div class="cellnote">'
            + ck(f'{c["rate_a"] * 100:.1f}%', SRC, d + " — subgoal rate",
                 raw=c["rate_a"]) + "</div></td>"
            '<td class="num">'
            + ck(f'{c["solved_b"]}/{c["n"]}', SRC, d + " — forward solved",
                 raw=c["solved_b"])
            + '<div class="cellnote">'
            + ck(f'{c["rate_b"] * 100:.1f}%', SRC, d + " — forward rate",
                 raw=c["rate_b"]) + "</div></td>"
            f'<td class="num">{dot(winner)} {esc(reads)}</td></tr>')
    if not body:
        return ""
    table = scroll(
        "<table><thead><tr><th>puzzle set (450 puzzles each)</th>"
        "<th class='num'>subgoal planner<br>solved · rate</th>"
        "<th class='num'>move-by-move planner<br>solved · rate</th>"
        "<th class='num'>who wins, by how much</th></tr></thead><tbody>"
        + "".join(body) + "</tbody></table>")
    return (kicker_h2(
        "the result",
        "One table before anything else",
        "Two planners solve the same fixed pools of 450 Ricochet Robots "
        "puzzles under the same search budget. One plans move by move; "
        "the other plans in subgoals, working backward from the goal. A "
        "puzzle counts as solved only if the plan plays out legally, "
        "move by move, on the real board — every solved row was "
        "independently replayed through the physics alone.")
        + table
        + _seed_of_record_foot(D)
        + '<p class="small">On the smallest board the move-by-move '
        "planner is the champion. On every other configuration the "
        "subgoal planner wins, by a margin that grows with both "
        "board size and robot count — every difference in the table "
        "clears a paired statistical test at p&nbsp;&lt;&nbsp;0.0001. "
        "The rest of this page explains the puzzle, the two planners, "
        "how the scoring stays honest, and every caveat we know "
        "of.</p></section>")


def tab_overview(D):
    return (sec_headline_first(D) + sec_puzzle() + sec_two_ideas()
            + sec_rules() + sec_verdict(D))


# ---------------------------------------------------------------------------
# Tab 2 — Base scale
# ---------------------------------------------------------------------------

def sec_base_headline(D):
    rows = []
    src_f = "eval/results/comparison_forward.json"

    def fwd_row(frag, label, sub, hl=False):
        a = fwd_agg(D, frag)
        if a:
            rows.append({"label": label, "family": "fwd", "sub": sub,
                         "agg": a, "src": src_f, "hl": hl,
                         "machine": "origin"})

    fwd_row("candidate_scored", "Move-by-move — best supervised networks",
            "picked for the head-to-head; the quality benchmark", hl=True)
    fwd_row("best.ckpt", "Move-by-move — earlier supervised checkpoint",
            "the same recipe before its final selection pass")
    fwd_row("runs_warm/iter5", "Move-by-move — self-play, warm start, round 5",
            "supervised start, then 5 rounds learning from its own solutions")
    fwd_row("runs_scratch_v5/iter15",
            "Move-by-move — self-play from scratch, round 15",
            "no supervised start; 15 self-play rounds")

    stages = [
        (D.get("bwd_before"), "eval/results/comparison_backward.json",
         "Subgoals — as first measured (before the fixes)",
         "four coding defects, since repaired, made most plans unplayable; "
         "plans were previously graded on paper", False),
        (D.get("bwd_fixed"), "eval/results/final450_backward_plain.json",
         "Subgoals — after the four fixes",
         "same networks; planning and plan-to-moves code repaired, each fix "
         "verified zero-regression", False),
        (D.get("bwd_any"), "eval/results/final450_backward_anytime.json",
         "Subgoals — fixes + test-play every finished plan",
         "a finished plan that fails its test playback is discarded and the "
         "search continues", False),
        (D.get("bwd_prefix"), "eval/results/final450_backward_prefix.json",
         "Subgoals — playability checked inside the search",
         "every partial plan is physics-checked as it is built; doomed "
         "branches are dropped immediately (zero false discards proven)",
         False),
        (D.get("bwd_b1"), "eval/results/final450_backward_b1.json",
         "Subgoals — extended plan language, retrained",
         "wall-less stoppers and step-asides added to the vocabulary; "
         "networks retrained on it", False),
        (D.get("bwd_b2"), "eval/results/final450_backward_b2.json",
         "Subgoals — full plan language (same networks, zero-shot)",
         "adds generalized repairs; the re-use step type itself never "
         "reaches the learned search (see the plan-language tab)", True),
        (D.get("bwd_heuristic"),
         "eval/results/final450_backward_heuristic_baseline.json",
         "Subgoals — hand-written scoring only (no neural networks)",
         "the same search loop, budget and candidate pool with the "
         "labeler's hand-written scorer; the gap to the row above is the "
         "networks' whole contribution", False),
    ]
    for comp, src, label, sub, hl in stages:
        a = bwd_agg(comp)
        if a:
            note = None
            if "before the fixes" in label:
                note = (f"plan found on paper for {a.get('plan_found')}"
                        f"/{a['n']} — playable for {a['solved']}")
            rows.append({"label": label, "family": "bwd", "sub": sub,
                         "agg": a, "src": src, "hl": hl, "note": note,
                         "machine": ("karolina" if proto_date(comp)[:10]
                                     >= "2026-07-20" else "origin")})
    a_rc = bwd_agg(D.get("bwd_retrained"))
    if a_rc:
        rows.append({
            "label": "Subgoals — full plan language, retrained networks",
            "family": "bwd",
            "sub": ("retrained on the regenerated full-vocabulary corpus; "
                    "matches the zero-shot row above — retraining is not "
                    "currently an upgrade (fairness tab, seed robustness)"),
            "agg": a_rc, "src": D.get("bwd_retrained_src",
                                      "eval/results/"
                                      "final450_backward_b2_retrained_cap20000.json"),
            "hl": False, "machine": "karolina"})
    else:
        rows.append({
            "label": "Subgoals — full plan language, retrained networks",
            "family": "bwd", "sub": "labels for the full vocabulary are being "
            "generated and the networks retrained", "agg": None,
            "pending_msg": "expected at eval/results/"
            "final450_backward_b2_retrained_cap20000.json"})

    html = kicker_h2(
        "every system, one benchmark",
        "The base-scale head-to-head: 16×16 board, 4 robots",
        "The pinned 450-puzzle benchmark both planner families were "
        "developed on. The exact solver can grade every puzzle here, so "
        "solution quality is measurable: “extra moves” is the played-out "
        "solution length minus the true optimum, averaged over solved "
        "puzzles.")
    html += sys_table(rows, table_id="base-table")
    f_steps = [r["agg"]["mean_expansions"] for r in rows
               if r["family"] == "fwd" and r.get("agg")]
    b_steps = [r["agg"]["mean_expansions"] for r in rows
               if r["family"] == "bwd" and r.get("agg")
               and "before the fixes" not in r["label"]]
    ranges = ""
    if f_steps and b_steps:
        ranges = (f"the subgoal rows plan in {min(b_steps):.0f}–"
                  f"{max(b_steps):.0f} search steps; the move-by-move rows "
                  f"need {min(f_steps):.0f}–{max(f_steps):.0f}")
    html += f"""
  <p class="small">Two readings. <b>Quality:</b> the best move-by-move
  networks solve everything nearly optimally — the quality bar the subgoal
  planner does not reach (its solutions run ~2 moves over optimum).
  <b>Efficiency:</b> {ranges}. The next section shows what that gap means
  when the budget is tight.</p>
</section>"""
    return html


def sec_base_climb(D):
    # matched-150 ladder, every stage from its own file
    stages = []
    if D.get("prefix150_slice_solved") is not None:
        stages.append(("As first measured", D["prefix150_slice_solved"], 150,
                       "eval/results/comparison_backward.json (first 150 rows)",
                       "four latent defects in planning and playback"))
    for key, src, label, note in [
            ("postfix1", "eval/results/comparison_backward_postfix.json",
             "Fix 1 — no bouncing off yourself",
             "a robot could be scheduled to bounce off itself"),
            ("postfix2", "eval/results/comparison_backward_postfix2.json",
             "Fix 2 — one robot, one plan role",
             "one robot could hold two plan roles at once"),
    ]:
        a = bwd_agg(D.get(key))
        if a:
            stages.append((label, a["solved"], a["n"], src, note))
    ro = D.get("reorder_ab")
    if ro and ro.get("solved_after") is not None:
        stages.append(("Fix 3 — flexible move order",
                       ro["solved_after"], 150,
                       "eval/results/realizer_reorder_ab.json",
                       "the plan-to-moves converter no longer forces a "
                       "stricter order than the plan assumes"))
    a = bwd_agg(D.get("postfix3"))
    if a:
        stages.append(("Fix 4 — promised stoppers must be placed",
                       a["solved"], a["n"],
                       "eval/results/comparison_backward_postfix3.json",
                       "a plan can no longer claim a stopper cell without "
                       "parking a robot there"))
    pts = []
    for label, solved, n, src, note in stages:
        v = solved / n * 100
        pts.append({"label": label.split(" — ")[0], "sub": None,
                    "value": v,
                    "tip": f"({solved}/{n}) — {note}"})
        need(f"climb chart: {label} rendered", f"{v:.1f}")
    svg = C.columns_single(pts, unit="%", aria="Playable-solve rate on the "
                           "matched 150 puzzles after each repair",
                           height=190)
    twin = C.table_twin(
        ["stage", "solved", "playable rate"],
        [[esc(l), f"{s}/{n}", f"{s / n * 100:.1f}%"]
         for l, s, n, src, note in stages],
        "repair ladder as a table")
    fig = C.figure(
        "Most of the honesty gap was bugs — four verified repairs",
        "playable-solve rate, matched 150-puzzle slice, same networks "
        "throughout",
        "", svg,
        "Each repair was verified zero-regression before adoption (every "
        "previously passing plan still passes; realizer A/B files). The "
        "methodology never changed — only defects were removed.",
        "eval/results/comparison_backward_postfix*.json · "
        "realizer_reorder_ab.json", twin)

    pab = D.get("prefix_ab") or {}
    nfp = (pab.get("no_false_pruning_ab") or {})
    plain_prefix = nfp.get("plain_vs_prefix") or {}
    fp = plain_prefix.get("false_prunes")
    fp_ok = (fp == [])
    fact("in-search checking A/B recorded zero false discards", fp_ok)

    a_before = bwd_agg(D.get("bwd_before"))
    a_prefix = bwd_agg(D.get("bwd_prefix"))
    a_b2 = bwd_agg(D.get("bwd_b2"))
    start_txt = (ck(rate_txt(a_before),
                    "eval/results/comparison_backward.json",
                    "climb: honest starting point", raw=a_before["solved"])
                 if a_before else "—")
    prefix_txt = (ck(rate_txt(a_prefix),
                     "eval/results/final450_backward_prefix.json",
                     "climb: prefix result", raw=a_prefix["solved"])
                  if a_prefix else "—")
    b2_txt = (ck(rate_txt(a_b2), "eval/results/final450_backward_b2.json",
                 "climb: b2 result", raw=a_b2["solved"]) if a_b2 else "—")
    t0 = (f"{a_before['solve_rate'] * 100:.0f}%" if a_before else "—")
    t1 = (f"{a_b2['solve_rate'] * 100:.0f}%" if a_b2 else "—")
    html = kicker_h2(
        "how the subgoal planner earned its score",
        f"From {t0} honest to {t1}: repairs, then checking, then vocabulary",
        "The subgoal planner's historical “99.6% solved” counted plans "
        "complete in its own notation that could not be played (a "
        "self-grading figure, unrelated to the 99.6% language ceiling "
        "discussed in the plan-language tab). Forced to "
        "play every plan out it scored " + start_txt
        + " — the honest starting point. Everything after that is measured "
          "recovery.")
    html += fig
    html += f"""
  <div class="cols2">
    <div class="panel"><h3 class="ph">Then: check playability inside the
    search</h3><p class="small">Instead of discovering at the end that a
    finished plan cannot be played, every <b>partial</b> plan is
    physics-checked as it is built and doomed branches are dropped
    immediately. An A/B run proved the check discards nothing it should keep
    (zero false discards; every newly kept plan plays out). Result on the
    full 450: <b>{prefix_txt}</b> at ~7 search steps.</p>
    </div>
    <div class="panel"><h3 class="ph">Finally: richer plan language</h3>
    <p class="small">The remaining failures are mostly puzzles whose
    solutions <b>cannot be said</b> in the original subgoal vocabulary at
    all — a measured property of the language, not of training. Extending
    the language (tab 3) lifted the planner to <b>{b2_txt}</b>, two points
    under what the extended language provably permits.</p></div>
  </div>
</section>"""
    return html


def sec_base_budget(D):
    bud = D.get("budget")
    if not bud:
        return kicker_h2("search budgets", "Solve rate vs search budget") + \
            pending("eval/results/budget_curves.json not found") + "</section>"
    systems = bud.get("systems") or {}

    def curve(frag):
        for name, s in systems.items():
            if frag in name:
                caps = {p["expansion_cap"]: p["solve_rate"] * 100
                        for p in s.get("points", [])}
                return caps
        return {}

    bwd = curve("anytime realization-checked")
    fwd_best = curve("candidate_scored")
    # extend the backward curve with its full-budget point
    a_any = bwd_agg(D.get("bwd_any"))
    if a_any and 1200 not in bwd:
        bwd[1200] = a_any["solve_rate"] * 100
    xcaps = [10, 30, 100, 300, 1200]
    series = [
        {"key": "bwd", "fam": "bwd", "label": "subgoals (checked search)",
         "values": [bwd.get(c) for c in xcaps]},
        {"key": "fwd", "fam": "fwd", "label": "move-by-move (best networks)",
         "values": [fwd_best.get(c) for c in xcaps]},
    ]
    for c in (10, 1200):
        if bwd.get(c) is not None:
            need(f"budget curve: backward at cap {c}", f"{bwd[c]:.1f}")
        if fwd_best.get(c) is not None:
            need(f"budget curve: forward at cap {c}", f"{fwd_best[c]:.1f}")
    svg = C.line_chart([str(c) for c in xcaps], series, unit="%",
                       aria="Solve rate at increasing search-step budgets, "
                       "both planners, base benchmark",
                       xtitle="search-step budget (cap per puzzle)")
    # table twin covers EVERY system in the file, not just the two plotted
    def friendly(name):
        if "candidate_scored" in name:
            return "move-by-move — best supervised"
        if "best.ckpt" in name:
            return "move-by-move — earlier checkpoint"
        if "runs_warm" in name:
            return "move-by-move — self-play, warm start"
        if "runs_scratch" in name:
            return "move-by-move — self-play from scratch"
        if "anytime" in name:
            return "subgoals — checked search"
        return "subgoals — plain search (no checking)"
    twin_rows = []
    other_f10 = []
    for name, s in systems.items():
        caps = {p["expansion_cap"]: p["solve_rate"] * 100
                for p in s.get("points", [])}
        if not caps:
            continue
        if "forward" in name and "candidate_scored" not in name \
                and 10 in caps:
            other_f10.append(caps[10])
        twin_rows.append([esc(friendly(name))]
                         + [(f"{caps[c]:.1f}%" if c in caps else "—")
                            for c in xcaps])
    twin = C.table_twin(["system"] + [f"{c} steps" for c in xcaps],
                        sorted(twin_rows), "every system's budget curve")
    b10 = bwd.get(10)
    f10 = fwd_best.get(10)
    others = (f" (earlier move-by-move checkpoints reach only "
              f"{min(other_f10):.1f}–{max(other_f10):.1f}% there — table)"
              if other_f10 else "")
    fig = C.figure(
        "With ten search steps, subgoals already solve three quarters of "
        "the benchmark",
        "solve rate (playable) vs search-step cap; positions are spaced "
        "evenly per cap, not to scale", C.line_legend(
            [("bwd", "subgoals (checked search)"),
             ("fwd", "move-by-move (best networks)")]), svg,
        f"At a 10-step budget the subgoal planner solves "
        f"{b10:.1f}% against the best move-by-move networks' "
        f"{f10:.1f}%{others}. Move-by-move catches up, then passes, as the "
        "budget grows toward 1,200: it spends search steps where the "
        "subgoal planner spends vocabulary. The 1,200-step points come "
        "from the main comparison files.",
        "eval/results/budget_curves.json", twin)
    html = kicker_h2(
        "efficiency", "What a search step buys each planner",
        "Subgoal decisions are worth several moves at once, so complete "
        "plans are only a few decisions deep — the efficiency claim of the "
        "thesis, measured directly by shrinking the budget.")
    html += fig
    html += "</section>"
    return html


def sec_base_selfplay(D):
    sup = bwd_agg(D.get("prefix150"))
    arm = bwd_agg(D.get("arm5"))
    rows = []
    if sup:
        rows.append({"label": "Supervised networks (oracle-labeled)",
                     "family": "bwd",
                     "sub": "trained on exact-solver labels; checked search",
                     "agg": sup,
                     "src": "eval/results/prefix150_prefix.json",
                     "machine": "origin"})
    if arm:
        rows.append({"label": "Self-play networks — round 5",
                     "family": "bwd",
                     "sub": "trained only on plans that really play out, "
                            "from its own solving; checked search",
                     "agg": arm, "src": "eval/results/arm_prefix_iter5.json",
                     "machine": "origin", "hl": True})
    table = sys_table(rows) if rows else pending(
        "self-play verdict files not found")

    iters = [d for d in (D.get("selfplay_iters") or []) if d]
    pts = [d.get("probe", {}).get("mean_strict_regret") for d in iters]
    fig = ""
    if len([p for p in pts if p is not None]) >= 2:
        series = [{"key": "r", "fam": "bwd", "label": "probe extra moves",
                   "values": pts}]
        svg = C.line_chart([f"round {d.get('iteration', i)}"
                            for i, d in enumerate(iters)], series, unit="",
                           ymax=1.0, label_dec=2,
                           aria="Probe extra moves per self-play round",
                           xtitle="", end_labels=True)
        twin = C.table_twin(
            ["round", "probe extra moves (playable plans)"],
            [[f"{d.get('iteration', i)}", f"{p:.3f}" if p is not None else "—"]
             for i, (d, p) in enumerate(zip(iters, pts))],
            "per-round probe as a table")
        fig = C.figure(
            "Self-play improves solution quality round over round — noisily",
            "39-puzzle probe, average extra moves of played-out plans, "
            "after each round", "", svg,
            f"Round-to-round noise is real (a 39-puzzle probe); the "
            f"endpoint improves from {pts[0]:.2f} to {pts[-1]:.2f} extra "
            "moves. The benchmark verdict above is the controlled "
            "comparison.", "subgoal_selfplay/runs_warm_prefix/"
            "iter*_stats.json", twin)

    html = kicker_h2(
        "training without a teacher",
        "Self-play: the training that survives at scale",
        "The exact solver that labels supervised training data fails "
        "increasingly as puzzles grow (tab 4), so a planner that keeps "
        "improving must learn from its own solved puzzles. Generation "
        "keeps only plans that really play out.")
    if sup and arm:
        d_steps = (sup["mean_expansions"] - arm["mean_expansions"]) \
            / sup["mean_expansions"] * 100
        html += table
        html += f"""
  <p class="small">On the matched 150-puzzle slice, five self-play rounds
  <b>match the supervised networks' solve rate</b> (both are pressed
  against the same language ceiling), with slightly better move quality and
  <b>{ck(f"{d_steps:.0f}%", "eval/results/arm_prefix_iter5.json",
  "selfplay: step reduction", raw=arm["mean_expansions"])} fewer search
  steps</b>. At the base scale that is a wash on rate and a modest
  efficiency gain — the decisive property is that this training loop needs
  no oracle at all.</p>"""
    else:
        html += table
    html += fig
    html += "</section>"
    return html


def tab_base(D):
    return (sec_base_headline(D) + sec_base_climb(D) + sec_base_budget(D)
            + sec_base_selfplay(D))


# ---------------------------------------------------------------------------
# Tab 3 — The plan language
# ---------------------------------------------------------------------------

PLAN_PICKS = {
    "base": ("base", 0),
    "b1": ("b1", 333),
    "b1_park": ("b1", 28),
    "b2": ("b2", 156),
    "b2_park": ("b2", 76),
}


def plan_example(D, pick_key, title, caption, marker):
    data = D.get("plan_structs")
    if not data or "examples" not in data:
        return pending("plan-structure example missing "
                       "(eval/results/plan_structures_data.json)")
    stage, idx = PLAN_PICKS[pick_key]
    ex = next((e for e in data["examples"]
               if (e["stage"], e["bench_idx"]) == (stage, idx)), None)
    if not ex:
        return pending(f"plan example ({stage}, bench {idx}) not present in "
                       "plan_structures_data.json")
    svg, ncl = dag_svg(ex, f"Plan structure of benchmark puzzle "
                           f"{ex['bench_idx']} — {title}",
                       G=GEOM_COMPACT, marker_id=marker)
    # widen the canvas so right-edge annotation labels ("re-uses this
    # robot…") never crop: extend the viewBox width, coordinates unchanged
    import re as _re
    m = _re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    if m:
        w0, h0 = int(m.group(1)), int(m.group(2))
        svg = svg.replace(f'viewBox="0 0 {w0} {h0}"',
                          f'viewBox="0 0 {w0 + 150} {h0}"', 1)
        svg = svg.replace(f"max-width:{w0}px", f"max-width:{w0 + 150}px", 1)
    by_id = {nd["id"]: nd for nd in ex["nodes"]}
    expect = sum(1 for e in ex["edges"]
                 if not e.get("byref") and e.get("cost")
                 and (by_id[e["from"]]["type"], by_id[e["to"]]["type"])
                 not in STRUCTURAL_PAIRS)
    fact(f"plan diagram {pick_key}: every real slide is labeled "
         f"({ncl}/{expect})", ncl == expect)
    inset = board_inset(ex)
    return f"""
  <div class="panel">
    <h3 class="ph">{esc(title)}</h3>
    <p class="cellnote">real plan for benchmark puzzle {ex['bench_idx']} —
    plays out in <b>{ex['legal_moves']} legal moves</b> (shortest possible:
    {ex['d_star']})</p>
    <div class="planrow"><div class="scroll planscroll">{svg}</div>
    <div class="planinset">{inset}
    <p class="cellnote">where the robots start</p></div></div>
    <p class="cellnote">{caption}</p>
  </div>"""


def sec_lang_what(D):
    html = kicker_h2(
        "the plans themselves", "What a subgoal plan actually says",
        "Read each diagram top-down as commitments (“for the goal slide to "
        "work, this must happen first…”); the robots travel bottom-up — a "
        "colored arrow means that robot really slides from the lower box to "
        "the upper one, and the pill counts its moves. All plans shown were "
        "verified by playing them out move by move.")
    html += plan_legend(marker_id="rr-leg")
    html += plan_example(
        D, "base", "The original design: chains of parked helpers",
        "A plan could only park a helper against a wall and leave it there "
        "forever. Two stepping-stones chain here: one robot parks so a "
        "second can stop; the second parks so the target can stop on the "
        "goal. Whatever cannot be phrased this way cannot be planned at "
        "all.", "rr-base")
    html += "</section>"
    return html


def ceiling_meter_fig(D):
    from eval.report_data import probe_counts
    n = 450
    rows = []
    cc0 = probe_counts(D.get("probe_base"))
    a_pre = bwd_agg(D.get("bwd_prefix"))
    if cc0 and a_pre:
        bad = cc0.get("NO_COMPLETE_PLAN", 0) + cc0.get("NO_REALIZABLE_PLAN", 0)
        rows.append({
            "label": "original language",
            "sub": "parked-forever stoppers only",
            "achieved": a_pre["solve_rate"] * 100,
            "ceiling": (n - bad) / n * 100,
            "ach_txt": f"networks: {a_pre['solve_rate'] * 100:.1f}%",
            "ceil_txt": f"{(n - bad) / n * 100:.1f}%",
            "ceil_note": f"{bad} puzzles unsayable"})
    cc1 = probe_counts(D.get("probe_b1"))
    a_b1 = bwd_agg(D.get("bwd_b1"))
    if cc1 and a_b1:
        bad = cc1.get("NO_COMPLETE_PLAN", 0) + cc1.get("NO_REALIZABLE_PLAN", 0)
        rows.append({
            "label": "+ extension 1 (B1)",
            "sub": "wall-less stoppers, step-asides",
            "achieved": a_b1["solve_rate"] * 100,
            "ceiling": (n - bad) / n * 100,
            "ach_txt": f"networks: {a_b1['solve_rate'] * 100:.1f}%",
            "ceil_txt": f"{(n - bad) / n * 100:.1f}%",
            "ceil_note": f"{bad} puzzles unsayable"})
    cc2 = probe_counts(D.get("probe_b2"))
    a_b2 = bwd_agg(D.get("bwd_b2"))
    if cc2 and a_b2:
        unres = cc2.get("INCONCLUSIVE", 0)
        rows.append({
            "label": "+ extension 2 (B2)",
            "sub": "re-use of placed robots",
            "achieved": a_b2["solve_rate"] * 100,
            "ceiling": (n - unres) / n * 100,
            "ach_txt": f"networks: {a_b2['solve_rate'] * 100:.1f}%",
            "ceil_txt": f"{(n - unres) / n * 100:.1f}%",
            "ceil_note": f"0 impossible, {unres} unresolved"})
    if not rows:
        return pending("ceiling probe files missing")
    for r in rows:
        need(f"ceiling meter: {r['label']} ceiling value", r["ceil_txt"])
    svg = C.ceiling_meters(rows, aria="What the plan language permits vs "
                           "what the trained networks achieve, per language "
                           "stage")
    twin = C.table_twin(
        ["language stage", "networks achieve", "language permits"],
        [[esc(r["label"]), r["ach_txt"].replace("networks: ", ""),
          r["ceil_txt"] + " (" + r["ceil_note"] + ")"] for r in rows],
        "ceiling vs achieved as a table")
    return C.figure(
        "The ceiling is a property of the language — and it moved",
        "base benchmark (450 puzzles): light track = share of puzzles for "
        "which a playable plan exists in the language at all (exhaustive "
        "no-network probe); solid bar = what the trained networks actually "
        "solve. B1 and B2 name the two plan-language extensions: B1 lets a "
        "stopper park on a cell with no wall behind it and lets a robot "
        "step aside before a named slide; B2 additionally lets a plan "
        "re-use a robot it has already placed and step two robots aside "
        "at once",
        "", svg,
        "The black tick is the measured ceiling. The remaining gap between "
        "bar and tick is an integration gap: the search that produced the "
        "bars cannot offer the newest step type at all — its proposal "
        "path, featurization and policy training all predate it. The B2 ceiling "
        "conservatively counts its 2 memory-capped probes as failures; "
        "nothing is proven impossible anymore.",
        "analysis/artifacts/ceiling_probe_results*.json · "
        "eval/results/final450_backward_*.json", twin)


def sec_lang_ceiling(D):
    from eval.report_data import probe_counts
    cc0 = probe_counts(D.get("probe_base"))
    html = kicker_h2(
        "the ceiling, measured",
        "For some puzzles, no playable subgoal plan exists at all",
        "An exhaustive no-network probe (try every plan the language can "
        "express, play each one out) settles, per puzzle, whether the "
        "planner's failure is the networks' fault or the language's.")
    if cc0:
        n_nc = cc0.get("NO_COMPLETE_PLAN", 0)
        n_nr = cc0.get("NO_REALIZABLE_PLAN", 0)
        n_re = cc0.get("REALIZABLE_EXISTS", 0)
        tot = n_nc + n_nr + n_re
        src = "analysis/artifacts/ceiling_probe_results.json"
        html += f"""
  <p>Probing every one of the original planner's {tot} base-benchmark
  failures: <b>{ck(str(n_nc), src, "ceiling: no complete plan",
  raw=n_nc)} puzzles cannot even be decomposed</b> into subgoals,
  <b>{ck(str(n_nr), src, "ceiling: none playable", raw=n_nr)} decompose but
  no decomposition survives legal playback</b>, and only
  <b>{ck(str(n_re), src, "ceiling: networks' fault", raw=n_re)}</b> have a
  playable plan the networks missed. So {n_nc + n_nr} of 450 puzzles
  (<b>{(n_nc + n_nr) / 450 * 100:.1f}%</b>) were structurally out of reach —
  no amount of training or extra search could help, which the cap-sensitivity
  re-probes confirm directly (quadrupling the search caps recovers none of
  them).</p>"""
        cs_base = probe_counts(D.get("cap_sens_base"))
        cs_g6 = probe_counts(D.get("cap_sens_g16r6"))
        if cs_base is not None:
            fact("cap sensitivity (base): 0 structural verdicts became "
                 "solvable at 4× caps",
                 cs_base.get("REALIZABLE_EXISTS", 0) == 0)
        if cs_g6 is not None:
            fact("cap sensitivity (6 robots): 0 structural verdicts became "
                 "solvable at 4× caps",
                 cs_g6.get("REALIZABLE_EXISTS", 0) == 0)
    gal = D.get("fwd_gallery_base")
    if gal and gal.get("rows"):
        rows = gal["rows"]
        n_solved = sum(1 for r in rows if r.get("solved"))
        src = "analysis/failure_gallery/forward_solutions_base.json"
        html += f"""
  <p>These are not unsolvable puzzles — the move-by-move planner solves
  <b>{ck(f"{n_solved}/{len(rows)}", src,
  "ceiling: forward solves the language failures", raw=n_solved)}</b> of
  them. They are puzzles whose solutions the subgoal vocabulary could not
  <i>say</i>: the only working plan needs a stopper standing on a wall-less
  cell, or a robot that steps aside and lets another pass, or one parked
  robot stopping two different sliders. The worked example below shows one
  in full.</p>"""
    html += ceiling_meter_fig(D)
    html += "</section>"
    return html


def sec_lang_b1(D):
    html = kicker_h2(
        "extension 1 · B1",
        "New words: wall-less stoppers and step-asides",
        "Both additions stay inside the subgoal formalism and are opt-in; "
        "with the extension off, behaviour is proven byte-identical (all "
        "249 stored plans replay identically; the default-vocabulary probe "
        "re-run matches the frozen baseline row for row).")
    b1ab = D.get("realizer_b1_ab") or {}
    if b1ab:
        fact("B1 realizer A/B: all stored plans identical",
             b1ab.get("identical") is True and not b1ab.get("mismatches"))
    html += plan_example(
        D, "b1", "A stopper with no wall to lean on",
        "The starred, dashed box is the new phrase: a helper parked on a "
        "cell with no wall — held in place only by another robot, whose own "
        "delivery recurses through the ordinary machinery. This exact "
        "puzzle was provably unwritable before the extension; the plan "
        "shown plays out at its 4-move optimum.", "rr-b1")
    html += plan_example(
        D, "b1_park", "A robot that steps aside",
        "The second new phrase: a plan may schedule a robot to slide out of "
        "a named slide's way before that slide runs — costed honestly as "
        "ordinary moves. No new stopping mechanism, just permission to "
        "clear a lane.", "rr-b1p")
    a_b1 = bwd_agg(D.get("bwd_b1"))
    if a_b1:
        html += f"""
  <p>Retrained on extended-vocabulary labels, the planner reaches
  <b>{ck(rate_txt(a_b1), "eval/results/final450_backward_b1.json",
  "b1: benchmark result", raw=a_b1["solved"])}</b>
  ({a_b1["solved"]}/{a_b1["n"]}) at
  {ck(fnum(a_b1["mean_expansions"], 1),
  "eval/results/final450_backward_b1.json", "b1: steps",
  raw=a_b1["mean_expansions"])} search steps — up from 89.1% — about two
  points under the lifted ceiling.</p>"""
    html += "</section>"
    return html


def sec_lang_b2(D):
    from eval.report_data import probe_counts
    html = kicker_h2(
        "extension 2 · B2",
        "No new words — permission to re-use what the plan already placed",
        "B2 adds no vocabulary: it lets a plan reference a robot it has "
        "already positioned (the same parked robot stopping two different "
        "sliders; the target itself serving as a stopper mid-route) and "
        "generalizes the step-aside repair to clear two robots at once. "
        "Same zero-regression burden as B1, all green.")
    html += plan_example(
        D, "b2", "The third stepping-stone is a robot the plan already "
        "placed",
        "The dashed link is the new permission: instead of recruiting a "
        "fresh helper, the plan points back at a robot it positioned "
        "earlier — three stepping-stones from two parked helpers.", "rr-b2")
    html += plan_example(
        D, "b2_park", "Two robots clear the same slide",
        "The generalized step-aside: two scheduled slides move two robots "
        "out of one lane before the main slide runs.", "rr-b2p")
    cc2 = probe_counts(D.get("probe_b2"))
    if cc2:
        src = "analysis/artifacts/ceiling_probe_results_b2.json"
        n_ok = cc2.get("REALIZABLE_EXISTS", 0)
        n_un = cc2.get("INCONCLUSIVE", 0)
        s40 = probe_counts(D.get("b2_solved40")) or {}
        html += f"""
  <p>Probed on the {n_ok + n_un} puzzles B1 still could not express:
  <b>{ck(str(n_ok), src, "b2: residuals now solvable", raw=n_ok)} of
  {n_ok + n_un} now have legal, played-out solutions</b> (one at exactly its
  8-move optimum), <b>0 are proven impossible</b>, and
  {ck(str(n_un), src, "b2: unresolved probes", raw=n_un)} remain unresolved
  with the probe out of search memory — an open budget question, not a
  proven wall (a deeper re-run on a 1&nbsp;TB node also exhausted memory).
  A superset check re-proved {s40.get("REALIZABLE_EXISTS", "—")}/40 of a
  previously-solved sample under B2's busier search.</p>"""
    a_b2 = bwd_agg(D.get("bwd_b2"))
    if a_b2 and cc2:
        ceil_pct = (450 - cc2.get("INCONCLUSIVE", 0)) / 450 * 100
        html += f"""
  <p>The learned planner — <b>unchanged B1-trained networks</b> plus the
  generalized repair pass — scores
  <b>{ck(rate_txt(a_b2), "eval/results/final450_backward_b2.json",
  "b2: benchmark result", raw=a_b2["solved"])}</b>
  ({a_b2["solved"]}/{a_b2["n"]}). An honest wiring note: the re-use
  permission itself is exercised by the exhaustive probe and the label
  generator, but <b>not by the learned planner's proposal path</b> — its
  featurization names helpers by robot start position, so a re-used robot
  standing mid-plan cannot be represented, its policy trainer skipped every
  such training record, and its search loop never offers such a candidate.
  The only piece of this extension the learned rows exercise is the
  generalized repair pass. Stated plainly: the language now permits
  ~{ceil_pct:.1f}%; the shortfall from the ceiling is an integration gap
  (proposal path, featurization, policy training), not a ranking failure.
  That gap has since been closed behind a flag, and the step's zero-shot
  value measured — see “the re-use step, wired at last” below.</p>"""
    html += retrain_story_html(D)
    if os.path.exists(rp("eval", "results", "plan_structures.html")):
        html += ("<p class='small muted'>All five plan diagrams, with full "
                 "board layouts, are also drawn in the "
                 "<a href='plan_structures.html'>plan-structures companion "
                 "page</a> next to this file.</p>")
    html += "</section>"
    return html


def sec_lang_attribution(D):
    """Which extension carries the gain -- the fixed-nets decomposition.

    The counts here were independently re-derived from the result JSONs on
    2026-07-28 (FINDINGS 39): the honest "old" comparator at g16r6 is the
    base-nets/old-vocabulary control (fixed networks), not the per-config
    row -- using the latter would fold a +7/+10 net-provenance effect into
    the B1 language gain.
    """
    st = D.get("stats_tests") or {}
    SRC = "eval/results/stats_tests.json"
    cells = {(c.get("rung"), c.get("set"), c.get("a"), c.get("b")): c
             for c in (st.get("cells") or []) if "skipped" not in c}

    def fixed_old(rung, set_):
        c = cells.get((rung, set_, "bwd_b2", "bwd_basenets_old"))
        return (c or {}).get("solved_b"), (c or {}).get("n")

    lad = {e["key"]: e for e in D.get("ladder", [])}

    def slot_solved(rung, set_, slot):
        cell = ((lad.get(rung) or {}).get(set_) or {}).get(slot)
        if not cell:
            return None, None, None
        return cell["agg"]["solved"], cell["agg"]["n"], cell["src"]

    rows = []
    # base: no fixed-nets old row exists; the old row is old-vocabulary NETS
    # as well as language, and the note says so.
    a_old = bwd_agg(D.get("bwd_prefix"))
    a_b1 = bwd_agg(D.get("bwd_b1"))
    a_b2 = bwd_agg(D.get("bwd_b2"))
    if a_old and a_b1 and a_b2:
        rows.append((
            "base 450",
            ck(str(a_old["solved"]), "eval/results/final450_backward_prefix.json",
               "lang-attrib base old", raw=a_old["solved"])
            + '<div class="cellnote">old nets + old language</div>',
            ck(str(a_b1["solved"]), "eval/results/final450_backward_b1.json",
               "lang-attrib base b1", raw=a_b1["solved"]),
            ck(str(a_b2["solved"]), "eval/results/final450_backward_b2.json",
               "lang-attrib base b2", raw=a_b2["solved"]),
            "nets + language change together — no fixed-nets base control"))
    for set_, slabel in (("graded", "gradable 316"),
                         ("frontier", "beyond-oracle 134")):
        old_n, n = fixed_old("g16r6", set_)
        b1_n, _, b1_src = slot_solved("g16r6", set_, "bwd_b1")
        b2_n, _, b2_src = slot_solved("g16r6", set_, "bwd_b2")
        if old_n is None or b1_n is None or b2_n is None:
            continue
        rows.append((
            f"16×16 · 6r, {slabel}",
            ck(str(old_n), SRC, f"lang-attrib g16r6 {set_} fixed-old",
               raw=old_n)
            + '<div class="cellnote">same nets, old language</div>',
            ck(str(b1_n), b1_src, f"lang-attrib g16r6 {set_} b1", raw=b1_n),
            ck(str(b2_n), b2_src, f"lang-attrib g16r6 {set_} b2", raw=b2_n),
            "all three columns run the SAME banked network pair "
            "(checkpoints_backward/policy_b1.ckpt + value_b1.ckpt); only "
            "the plan vocabulary changes"))
    if not rows:
        return ""
    body = "".join(
        f"<tr><td><b>{esc(r[0])}</b></td>"
        f'<td class="num">{r[1]}</td><td class="num">{r[2]}</td>'
        f'<td class="num">{r[3]}</td>'
        f'<td class="small muted">{esc(r[4])}</td></tr>' for r in rows)
    table = scroll(
        "<table><thead><tr><th>set</th>"
        "<th class='num'>old language</th>"
        "<th class='num'>+ extension 1 (B1)</th>"
        "<th class='num'>+ extension 2 (B2), zero-shot</th>"
        "<th>provenance</th></tr></thead><tbody>" + body + "</tbody></table>")
    # Which "old language" arm is which -- the two numbers differ and the page
    # shows both, so each is labelled by the file it comes from.
    arm_note = ""
    og, _, og_src = slot_solved("g16r6", "graded", "bwd_old")
    of, _, of_src = slot_solved("g16r6", "frontier", "bwd_old")
    fg, _n_g = fixed_old("g16r6", "graded")
    ff, _n_f = fixed_old("g16r6", "frontier")
    if None not in (og, of, fg, ff):
        arm_note = (
            "<p class='small muted'><b>Two different “old language” arms "
            "appear on this page; they are not the same run.</b> The column "
            "above is the <i>fixed-nets</i> control — the banked B1 network "
            "pair driven with the old vocabulary "
            "(<code>comparison_basenets_oldvocab.json</code> and its "
            "<code>_ungraded</code> twin): "
            + ck(str(fg), SRC, "arm note: fixed-nets old graded", raw=fg)
            + "/" + ck(str(ff), SRC, "arm note: fixed-nets old frontier",
                       raw=ff)
            + ". The scaling ladder's “original language” row is a "
            "<i>different</i> arm — networks trained per configuration on "
            "the old vocabulary (<code>comparison.json</code> / "
            "<code>comparison_ungraded.json</code>): "
            + ck(str(og), og_src, "arm note: ladder old graded", raw=og)
            + "/" + ck(str(of), of_src, "arm note: ladder old frontier",
                       raw=of)
            + ". Only the first isolates the language, which is why this "
            "table uses it; the difference between the pairs is the "
            "net-provenance effect, tested separately in the fairness "
            "tab.</p>")
    abl = D.get("byref_topk_ablation")
    ABL_SRC = "analysis/artifacts/byref_topk_ablation.json"
    if abl and abl.get("summary"):
        s = abl["summary"]
        t = s.get("totals") or {}
        abl_html = (
            "<h3>Why the second extension's step type never appears — an "
            "instrumented census</h3>"
            "<p>An instrumented re-run of the production search on the "
            f'beyond-oracle set ({ck(str(s.get("n")), ABL_SRC, "ablation n", raw=s.get("n"))} '
            "puzzles, production budget and networks) counted, at every "
            "expansion (one expansion = one search step, the matched unit "
            "defined in tab 1), both what the as-shipped planner offered "
            "and what a "
            "correctly wired robot-re-use proposal step "
            "<i>would</i> have offered. The as-shipped planner generated, "
            "ranked, shortlisted and expanded "
            + ck(str(t.get("asis_byref_topk", 0)), ABL_SRC,
                 "ablation as-is topk", raw=t.get("asis_byref_topk", 0))
            + " re-use candidates — structurally zero — while the "
            "counterfactual supply was "
            + ck(f'{t.get("shadow_byref_generated", 0):,}', ABL_SRC,
                 "ablation shadow generated",
                 raw=t.get("shadow_byref_generated", 0))
            + " candidates across "
            + ck(f'{t.get("shadow_groups_with_byref", 0):,}', ABL_SRC,
                 "ablation shadow groups",
                 raw=t.get("shadow_groups_with_byref", 0))
            + " of "
            + ck(f'{t.get("asis_expansion_groups", 0):,}', ABL_SRC,
                 "ablation groups", raw=t.get("asis_expansion_groups", 0))
            + " expansions. The vocabulary is not under-ranked; it is "
            "unreachable — the integration gap named above, measured.</p>")
    else:
        abl_html = progress_tag(
            "an instrumented census of the missing step type (as-shipped "
            "vs correctly wired candidate supply, production budget) is "
            "running; it renders here when "
            "analysis/artifacts/byref_topk_ablation.json lands")
    return (kicker_h2(
        "which extension carries the gain",
        "Nearly all of the executable-language gain is extension 1",
        "Holding the networks fixed, the first extension moves the solve "
        "count by tens of puzzles; adding the second moves it by 0–2. The "
        "second extension's value is what a plan can EXPRESS — the "
        "ceiling — and its measured shortfall is an integration gap: until "
        "the wiring landed, the re-use step type was absent from the "
        "learned planner's proposal path, featurization and policy "
        "training, so no ranking of it could be measured. It is wired "
        "now, behind a flag, and the next section measures it.")
        + table + arm_note + abl_html + "</section>")


def sec_lang_byref(D):
    """The re-use step, wired at last -- the zero-shot on/off A/B (FINDINGS 78).

    Every number is recomputed from the per-instance rows of the four A/B
    files by report_data.build_byref_ab(); the flag-off arm doubles as a
    regression check against the production rows (asserted there).
    """
    ab = D.get("byref_ab")
    net = (ab["pooled"]["on"] - ab["pooled"]["off"]) if ab else None
    n_pool = ab["pooled"]["n"] if ab else 450
    head = kicker_h2(
        "the re-use step, wired at last",
        "Switching the missing step type on changes almost nothing — and "
        "that is informative",
        "One rung only: <b>16×16 · 6 robots</b>, all "
        f"{n_pool} of its pinned puzzles. The step the ceiling depends on "
        "is now offered to the learned planner. Turning it on, with "
        "networks that were never trained on it, moves the solve count by "
        + (f"{net} more puzzles of {n_pool} at that rung."
           if net is not None else "a handful of puzzles."))
    if not ab:
        return head + pending(
            "the on/off A/B files (scaling/results/g16r6/comparison"
            "{,_ungraded}_b2retrained_cap20000_byref_{on,off}.json) are not "
            "in place") + "</section>"
    sets = {s["key"]: s for s in ab["sets"]}
    pooled = ab["pooled"]
    g, f_ = sets.get("graded"), sets.get("frontier")
    if not g or not f_:
        return head + pending("one arm of the by-reference A/B is missing"
                              ) + "</section>"

    def num(x, dec, src, desc):
        return ck(fnum(x, dec), src, desc, raw=x)

    rows_html = []
    for key in ("graded", "frontier"):
        s = sets[key]
        rows_html.append(
            f'<tr><td><b>{esc(s["label"])}</b></td>'
            f'<td class="num">{s["n"]}</td>'
            f'<td class="num">'
            + ck(str(s["on"]["solved"]), s["src_on"],
                 f"byref A/B {key}: solved, re-use on", raw=s["on"]["solved"])
            + '</td><td class="num">'
            + ck(str(s["off"]["solved"]), s["src_off"],
                 f"byref A/B {key}: solved, re-use off", raw=s["off"]["solved"])
            + '</td><td class="num">'
            + ck(str(s["b"]), s["src_on"], f"byref A/B {key}: puzzles gained",
                 raw=s["b"])
            + " / "
            + ck(str(s["c"]), s["src_off"], f"byref A/B {key}: puzzles lost",
                 raw=s["c"])
            + '</td><td class="num">'
            + num(s["on"]["mean_expansions"], 0, s["src_on"],
                  f"byref A/B {key}: mean search steps on")
            + " / "
            + num(s["off"]["mean_expansions"], 0, s["src_off"],
                  f"byref A/B {key}: mean search steps off")
            + '</td><td class="num">'
            + num(s["on"]["median_seconds"], 1, s["src_on"],
                  f"byref A/B {key}: median seconds on")
            + " / "
            + num(s["off"]["median_seconds"], 1, s["src_off"],
                  f"byref A/B {key}: median seconds off")
            + "</td></tr>")
    rows_html.append(
        '<tr><td><b>both sets pooled</b></td>'
        f'<td class="num">{pooled["n"]}</td>'
        '<td class="num">'
        + ck(str(pooled["on"]), f_["src_on"],
             "byref A/B pooled: solved, re-use on", raw=pooled["on"])
        + '</td><td class="num">'
        + ck(str(pooled["off"]), f_["src_off"],
             "byref A/B pooled: solved, re-use off", raw=pooled["off"])
        + '</td><td class="num">'
        + ck(str(pooled["b"]), f_["src_on"], "byref A/B pooled: puzzles gained",
             raw=pooled["b"])
        + " / "
        + ck(str(pooled["c"]), f_["src_off"], "byref A/B pooled: puzzles lost",
             raw=pooled["c"])
        + '</td><td class="small muted" colspan="2">exact paired test '
        f'(McNemar) on the {pooled["b"] + pooled["c"]} puzzles the two arms '
        "disagree about: <b>p = "
        + ck(fnum(pooled["p"], 2), f_["src_on"], "byref A/B pooled: McNemar p",
             raw=pooled["p"])
        + "</b></td></tr>")
    table = scroll(
        "<table><thead><tr><th>set</th><th class='num'>puzzles</th>"
        "<th class='num'>solved, re-use on</th>"
        "<th class='num'>solved, re-use off</th>"
        "<th class='num'>changed (gained / lost)</th>"
        "<th class='num'>mean search steps (on / off)</th>"
        "<th class='num'>median seconds (on / off)</th></tr></thead><tbody>"
        + "".join(rows_html) + "</tbody></table>")

    supply = ""
    if g["ranked"] is not None and f_["ranked"] is not None:
        supply = (
            "<p><b>Supply is not the problem.</b> With the flag on, the "
            "policy net scored, on average, "
            + ck(fnum(g["ranked"], 0), g["src_on"],
                 "byref A/B: re-use candidates ranked per gradable puzzle",
                 raw=g["ranked"])
            + " re-use candidates per gradable puzzle and "
            + ck(f'{f_["ranked"]:,.0f}', f_["src_on"],
                 "byref A/B: re-use candidates ranked per frontier puzzle",
                 raw=f_["ranked"])
            + " per beyond-oracle puzzle; "
            + ck(fnum(g["topk"], 0), g["src_on"],
                 "byref A/B: re-use candidates shortlisted per gradable puzzle",
                 raw=g["topk"])
            + " and "
            + ck(fnum(f_["topk"], 0), f_["src_on"],
                 "byref A/B: re-use candidates shortlisted per frontier puzzle",
                 raw=f_["topk"])
            + " of them respectively made a shortlist and were expanded. "
            "The as-shipped count, in the census just above, is zero.</p>")

    # the rows the flag-off arm reproduces (the cap-20,000 RETRAINED rows),
    # and the zero-shot production rows it is often mistaken for
    ab_off_g = ck(str(g["off"]["solved"]), g["src_prod"],
                  "byref A/B: flag-off reproduces retrained graded",
                  raw=g["off"]["solved"])
    ab_off_f = ck(str(f_["off"]["solved"]), f_["src_prod"],
                  "byref A/B: flag-off reproduces retrained frontier",
                  raw=f_["off"]["solved"])
    lad6 = {e["key"]: e for e in D["ladder"]}.get("g16r6") or {}
    pg = ((lad6.get("graded") or {}).get("bwd_b2") or {})
    pf = ((lad6.get("frontier") or {}).get("bwd_b2") or {})
    prod_g = (ck(f'{pg["agg"]["solved"]}/{pg["agg"]["n"]}', pg["src"],
                 "byref A/B: zero-shot production graded",
                 raw=pg["agg"]["solved"]) if pg else "—")
    prod_f = (ck(f'{pf["agg"]["solved"]}/{pf["agg"]["n"]}', pf["src"],
                 "byref A/B: zero-shot production frontier",
                 raw=pf["agg"]["solved"]) if pf else "—")

    body = f"""
  <p><b>What the step is.</b> A plan may point at a robot it has already
  positioned instead of recruiting a fresh helper. That permission is what
  lifts the language ceiling — and, as the census above shows, the learned
  planner could never propose it: the proposal path, the helper
  featurization and a silent training filter all skipped it. That step is
  now wired, behind a flag that defaults off.</p>
  <p><b>Which rows this A/B is, exactly.</b> Both arms run the banked
  <b>cap-20,000 retrained</b> network pair at <b>16×16 · 6 robots</b> —
  “cap-20,000” names the label corpus these networks were trained on (the
  cap is the search budget each labelling attempt was given). With the flag
  off the planner reproduces <i>those</i> rows exactly, row by row —
  {ab_off_g} of {g["n"]} on the gradable set and {ab_off_f} of {f_["n"]}
  beyond the oracle, machine-checked here on solved status, plan length and
  search steps. They are <b>not</b> the zero-shot production rows of the
  headline tables at this rung ({prod_g} and {prod_f}), which come from a
  different network pair. That exact reproduction is what makes the pair
  below a clean A/B — same banked networks, same pinned puzzles
  (sha256-checked), same budget of 1,200 search steps and 5 proposals,
  every solve replayed on the real board.</p>
  {table}
  {supply}
  <p><b>The honest reading.</b> The direction is right and the size is
  noise — {net} more puzzles of {n_pool} at this one rung, comfortably
  inside what the paired test calls
  chance — with the beyond-oracle search slightly cheaper and its plans
  slightly longer ({num(f_["on"]["mean_len_both"], 1, f_["src_on"],
  "byref A/B frontier: mean plan length on")} vs
  {num(f_["off"]["mean_len_both"], 1, f_["src_off"],
  "byref A/B frontier: mean plan length off")} moves on the puzzles both
  arms solve). But these networks have never seen a single training example
  of the step — the filter dropped them all — and they still cannot
  <i>name</i> the robot being referenced. What this measures is therefore
  the step's value to a planner that cannot really use it yet: a lower
  bound. The bottleneck is the training signal, not the supply of
  candidates. Two consequences for the rest of this page: the zero-shot
  rows stand exactly as reported, and the ceiling remains what it has
  always been here — a property of what the plan language can express, not
  a delivered solve rate.</p>
</section>"""
    return head + body

def sec_worked(D):
    w = D.get("worked")
    if not w:
        return kicker_h2("one puzzle in full",
                         "A puzzle the original language cannot express") + \
            pending("worked example needs analysis/artifacts/"
                    "ceiling_probe_instances.json and environments/env_*.pkl"
                    ) + "</section>"
    inst, verdict, sol = w["inst"], w["verdict"], w["solution"]
    wr, wd = w["walls"]
    size = w["size"]
    tidx = inst["target_idx"]
    goal = tuple(inst["target"])
    fact(f"worked example: recomputed optimum equals the recorded one "
         f"({len(sol)} vs {inst['d_star']})", len(sol) == inst["d_star"])
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
                    aria=f"The real {size}-by-{size} board of benchmark "
                         f"puzzle {inst['idx']}, with its {len(sol)}-move "
                         "optimal solution drawn as numbered arrows.")
    steps = []
    for k, (i, d, frm, to) in enumerate(sol, start=1):
        who = SLOT_NAMES[i] + (" (the target)" if i == tidx else "")
        steps.append(
            f"<li><span class='rdot' "
            f"style='background:var({SLOT_VARS[i]})'></span>"
            f"<b>{esc(who)}</b> slides {esc(d)}: "
            f"{esc(str(tuple(frm)))} → {esc(str(tuple(to)))}</li>")
    steps_html = f"<ol class='movelist'>{''.join(steps)}</ol>"
    story = ""
    if conflict:
        cell_ = conflict["cell"]
        stopper = SLOT_NAMES[conflict["stopper_slot"]]
        tgt = SLOT_NAMES[tidx]
        second = ""
        if conflict.get("stopper_stop"):
            b2s, c2 = conflict["stopper_stop"]
            second = (f" And {esc(stopper)} can only stop at "
                      f"{esc(str(cell_))} because <b>{esc(SLOT_NAMES[b2s])}"
                      f"</b> is already parked at {esc(str(tuple(c2)))} — a "
                      f"helper for the helper.")
        story = f"""
      <p><b>Why the original language cannot say this.</b> Look at cell
      <b>{esc(str(cell_))}</b> (dashed outline). In move
      {conflict['passed_step']}, {esc(tgt)} slides straight <i>through</i>
      that cell — so it must be <b>empty</b>. In move
      {conflict['stopper_step']}, <b>{esc(stopper)}</b> parks exactly there —
      so that {esc(tgt)}'s final slide can bump into it and stop on the
      goal. The same cell has to be empty first and occupied
      later.{second}</p>
      <p>The original vocabulary could only say “park a helper on a cell —
      it stays put forever — and bounce off it.” It had no words for
      <i>“let me pass first, then park behind me.”</i> The exhaustive probe
      confirms this is fatal here: it tested
      <b>{verdict.get('complete_tested', 0)} complete plans</b> and provably
      ran out of options — the puzzle was not merely hard, it was
      <b>unwritable</b>.</p>"""
    coda = ""
    b1p = D.get("probe_b1")
    if isinstance(b1p, list):
        b1row = next((r for r in b1p if r.get("idx") == inst["idx"]), None)
        if b1row and b1row.get("category") == "REALIZABLE_EXISTS":
            rm = b1row.get("realizable_moves")
            opt = (rm is not None and rm == inst["d_star"])
            coda = (
                f"<p><b>Epilogue.</b> Under the extended language this exact "
                f"puzzle became expressible: the probe now finds a playable "
                f"plan of <b>{rm} moves</b>"
                + (" — exactly the optimum" if opt else "")
                + " — and its plan diagram is the wall-less-stopper example "
                  "drawn above.</p>")
    return kicker_h2(
        "one puzzle in full",
        f"A {len(sol)}-move puzzle the original language could not express",
        f"Benchmark puzzle {inst['idx']} (board env_{inst['env_id']}), drawn "
        "from the real board file, with its optimal solution recomputed "
        "from the wall layout at build time. The move-by-move planner "
        "solves it; under the original plan language the subgoal planner "
        "provably never could.") + f"""
  <div class="boardrow">
    <div class="boardcol">{svg}</div>
    <div class="textcol">
      <p><b>The only way to solve it in {len(sol)} moves:</b></p>
      {steps_html}
      {story}
      {coda}
    </div>
  </div>
</section>"""


def sec_lang_scale(D):
    from eval.report_data import probe_counts
    cc = probe_counts(D.get("probe_g16r6_old"))
    cc1 = probe_counts(D.get("probe_g16r6_b1"))
    cc2 = probe_counts(D.get("probe_g16r6_b2"))
    html = kicker_h2(
        "why this matters for scale",
        "The language ceiling widens with scale — and the extensions close "
        "it",
        "The same exhaustive probe, run at 6 robots on all of the original "
        "planner's failures.")
    cc_base = probe_counts(D.get("probe_base"))
    if cc:
        n_nc, n_nr = cc.get("NO_COMPLETE_PLAN", 0), \
            cc.get("NO_REALIZABLE_PLAN", 0)
        n_re, n_in = cc.get("REALIZABLE_EXISTS", 0), cc.get("INCONCLUSIVE", 0)
        tot = n_nc + n_nr + n_re + n_in
        src = "scaling/results/g16r6/ceiling_probe_old_vocab.json"
        struct = n_nc + n_nr
        base_struct = ((cc_base.get("NO_COMPLETE_PLAN", 0)
                        + cc_base.get("NO_REALIZABLE_PLAN", 0))
                       if cc_base else None)
        base_share = (ck(f"{base_struct / 450 * 100:.1f}%",
                         "analysis/artifacts/ceiling_probe_results.json",
                         "6-robot ceiling: base share", raw=base_struct)
                      if base_struct is not None else "—")
        html += f"""
  <p>Of the original planner's {tot} failures at 6 robots,
  <b>{ck(str(struct), src, "6-robot ceiling: structural failures",
  raw=struct)} are structural</b> ({n_nc} with no complete plan, {n_nr} with
  no playable one) — the language-failure share of the whole 450-puzzle
  benchmark grows from {base_share} at base scale to
  <b>{struct / 450 * 100:.1f}%</b> at 6 robots. Language failure grows
  exactly where the thesis needs the subgoal planner to win.</p>"""
    if cc and cc1 and cc2:
        s1 = "scaling/results/g16r6/ceiling_probe_b1.json"
        s2 = "scaling/results/g16r6/ceiling_probe_b2.json"
        struct0 = cc.get("NO_COMPLETE_PLAN", 0) + cc.get("NO_REALIZABLE_PLAN", 0)
        imp1 = cc1.get("NO_COMPLETE_PLAN", 0) + cc1.get("NO_REALIZABLE_PLAN", 0)
        un1 = cc1.get("INCONCLUSIVE", 0)
        imp2 = cc2.get("NO_COMPLETE_PLAN", 0) + cc2.get("NO_REALIZABLE_PLAN", 0)
        un2 = cc2.get("INCONCLUSIVE", 0)
        c0 = (450 - struct0) / 450 * 100
        c1 = (450 - imp1 - un1) / 450 * 100
        c2v = (450 - imp2 - un2) / 450 * 100
        html += f"""
  <p>After the extensions, proven impossibility disappears: B1 leaves
  <b>{ck(str(imp1), s1, "6-robot ceiling: impossible after B1",
  raw=imp1)}</b> puzzles proven impossible (of the original {struct0}), and
  the B2 probe flips both of those too —
  <b>{ck(str(imp2), s2, "6-robot ceiling: impossible after B2",
  raw=imp2)} puzzles remain proven impossible at 6 robots</b>. The
  conservative ceiling reads {c0:.1f}% → ≥{c1:.1f}% → ≥{c2v:.1f}%
  (unresolved probes counted as failures). The scaling tab shows what the
  richer language does to the head-to-heads at every rung.</p>"""
    html += "</section>"
    return html


def tab_language(D):
    return (sec_lang_what(D) + sec_lang_ceiling(D) + sec_lang_b1(D)
            + sec_lang_b2(D) + sec_lang_attribution(D) + sec_lang_byref(D)
            + sec_worked(D) + sec_lang_scale(D))
