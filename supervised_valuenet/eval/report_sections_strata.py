"""Section: the frontier margin, stratified by oracle-independent hardness.

The beyond-oracle ("frontier") puzzle sets are defined as the puzzles the
exact move-level solver failed to grade — a selection that is adversarial
to the move-by-move planner by construction. This section answers the
referee question that selection invites: does the subgoal planner's
frontier margin survive when each set is split by hardness proxies that
never consult the solver? Data: analysis/artifacts/frontier_strata.json.
"""

from eval.report_util import (esc, ck, fnum, scroll, progress_tag,
                              kicker_h2, chip)

SRC = "analysis/artifacts/frontier_strata.json"

# single-factor strata, in reading order (walled first: the classic
# "easy" class, then the helper-dependent OPEN class, then travel)
_SINGLE_ORDER = ("goal walled", "goal OPEN",
                 "travel low", "travel mid", "travel high")


def _rung_label(key):
    """'g16r6' -> '16×16 · 6 robots'."""
    try:
        g, r = key[1:].split("r")
        return f"{g}×{g} · {r} robots"
    except (ValueError, IndexError):
        return key


def _fp(p):
    """p-value: 4 decimals; below-resolution values as '<0.0001'."""
    if p is None:
        return "—"
    if p < 0.0001:
        return "<0.0001"
    return fnum(p, 4)


def sec_strata(D):
    head = kicker_h2(
        "selection, tested",
        "Does the frontier margin survive its own selection?",
        "(“Frontier” is this page's shorthand for a beyond-oracle set.) "
        "The beyond-oracle sets are defined by a failure: they hold exactly "
        "the puzzles the exact move-level solver could not grade at its "
        "budget. That selection is adversarial to the move-by-move planner "
        "by construction, so its frontier deficit could in principle be an "
        "artifact of how the sets were chosen. The test here splits every "
        "frontier set by two hardness proxies computed from the puzzle and "
        "its board alone — never from any solver: whether the goal square "
        "has a wall on any side (a goal with no wall behind it can only be "
        "reached by first parking a helper robot to stop on — the domain's "
        "classic difficulty signal), and how far the target robot must "
        "travel (Manhattan distance, split at each set's own terciles). If "
        "the margin were a selection artifact, some stratum should show it "
        "evaporating.")

    data = D.get("frontier_strata")
    if not data or not data.get("strata"):
        return head + progress_tag(
            "analysis/artifacts/frontier_strata.json has not landed yet — "
            "the stratified frontier tables (goal-wall and travel-distance "
            "strata per rung, with per-stratum McNemar tests) render here "
            "automatically when analysis/frontier_strata.py produces it"
        ) + "</section>"

    strata = data["strata"]
    single = [s for s in strata if "/" not in s["stratum"]]
    crossed = [s for s in strata if "/" in s["stratum"]]
    leads = sum(1 for s in strata if s["diff"] > 0)
    ties = [s for s in strata if s["diff"] == 0]
    revs = [s for s in strata if s["diff"] < 0]
    sig = sum(1 for s in strata if s["mcnemar_p"] < 0.05)
    single_sig = sum(1 for s in single if s["mcnemar_p"] < 0.05)
    nonsig = [s for s in strata if s["mcnemar_p"] >= 0.05]

    # ---- summary line, computed from the file ------------------------
    parts = [
        "<p><b>Summary, computed from the source file:</b> ",
        ck(str(len(strata)), SRC, "strata: total stratum count",
           raw=len(strata)),
        " strata in all (",
        ck(str(len(single)), SRC, "strata: single-factor stratum count",
           raw=len(single)),
        " single-factor, ",
        ck(str(len(crossed)), SRC, "strata: crossed stratum count",
           raw=len(crossed)),
        " crossed). The subgoal planner leads in ",
        ck(str(leads), SRC, "strata: strata where subgoals lead",
           raw=leads),
    ]
    if not revs and len(ties) == 1:
        parts.append(
            " and ties in 1 (a seven-puzzle crossed cell where both "
            "planners solve everything — see below); it trails in none. ")
        parts.append(chip("no reversal in any stratum", "okc"))
        parts.append(" ")
    elif not revs:
        parts.append(f" and ties in {len(ties)}; it trails in none. ")
        parts.append(chip("no reversal in any stratum", "okc"))
        parts.append(" ")
    else:
        # honesty guard: if the data ever contradicts the no-reversal
        # claim, name the offending strata instead of asserting it
        where = "; ".join(f"{_rung_label(s['rung'])}: {s['stratum']}"
                          for s in revs)
        parts.append(
            f", ties in {len(ties)}, and trails in {len(revs)} "
            f"({esc(where)}) — the margin does NOT hold everywhere. ")
    parts.append(
        ck(str(sig), SRC, "strata: significant at the 0.05 level",
           raw=sig))
    parts.append(f" of {len(strata)} per-stratum differences are "
                 "significant at the 0.05 level (paired McNemar)")
    if single and single_sig == len(single):
        parts.append(", including every single-factor stratum")
    elif single:
        parts.append(f", including {single_sig} of {len(single)} "
                     "single-factor strata")
    parts.append(".</p>")
    summary = "".join(parts)

    # ---- one table, single-factor strata only, grouped by rung -------
    by = {}
    rung_keys = []
    for s in strata:
        if s["rung"] not in by:
            rung_keys.append(s["rung"])
        by.setdefault(s["rung"], {})[s["stratum"]] = s
    thead = ("<tr><th>stratum</th><th class='num'>n</th>"
             "<th class='num'>subgoals</th>"
             "<th class='num'>move-by-move</th>"
             "<th class='num'>diff (pts)</th>"
             "<th class='num'>McNemar p</th></tr>")
    body = []
    for rk in rung_keys:
        cells = by[rk]
        first = next(iter(cells.values()))
        terc = first.get("travel_terciles") or []
        tnote = (f' <span class="muted small">— travel terciles at '
                 f"{esc(terc[0])} / {esc(terc[1])}</span>"
                 if len(terc) == 2 else "")
        body.append(f'<tr><td colspan="6"><b>{esc(_rung_label(rk))}</b>'
                    f"{tnote}</td></tr>")
        for name in _SINGLE_ORDER:
            s = cells.get(name)
            if not s:
                continue
            d = f"strata {rk} {name}"
            n_ck = ck(str(s["n"]), SRC, d + " — n", raw=s["n"])
            b_ck = ck(f"{s['backward_rate'] * 100:.1f}%", SRC,
                      d + " — subgoal solve rate",
                      raw=s["backward_b2_solved"])
            f_ck = ck(f"{s['forward_rate'] * 100:.1f}%", SRC,
                      d + " — move-by-move solve rate",
                      raw=s["forward_solved"])
            di_ck = ck(f"{s['diff'] * 100:+.1f}", SRC,
                       d + " — margin in points", raw=s["diff"])
            p_ck = ck(_fp(s["mcnemar_p"]), SRC, d + " — McNemar p",
                      raw=s["mcnemar_p"])
            body.append(
                f"<tr><td>{esc(name)}</td>"
                f'<td class="num">{n_ck}</td>'
                f'<td class="num">{b_ck}<div class="cellnote">'
                f'{s["backward_b2_solved"]}/{s["n"]}</div></td>'
                f'<td class="num">{f_ck}<div class="cellnote">'
                f'{s["forward_solved"]}/{s["n"]}</div></td>'
                f'<td class="num"><b>{di_ck}</b></td>'
                f'<td class="num">{p_ck}</td></tr>')
    table = scroll("<table><thead>" + thead + "</thead><tbody>"
                   + "".join(body) + "</tbody></table>")
    systems = data.get("systems", "")
    proxy = data.get("proxy", "")
    prov = ""
    if systems or proxy:
        prov = ('<p class="small muted">As recorded in the file — systems: '
                f"“{esc(systems)}” Proxies: “{esc(proxy)}”</p>")

    # ---- crossed strata: where the only non-significant cells sit ----
    nonsig_all_crossed = bool(nonsig) and all("/" in s["stratum"]
                                             for s in nonsig)
    if nonsig_all_crossed:
        nmax = max(s["n"] for s in nonsig)
        cross_p = (
            "<p>The crossed goal-by-travel breakdown (both factors at "
            "once) is in the source file as well, but its cells get small "
            "and it is kept out of the table above. The small crossed "
            "cells are exactly where the only non-significant results "
            "sit: all "
            + ck(str(len(nonsig)), SRC,
                 "strata: non-significant cell count", raw=len(nonsig))
            + " differences that miss the 0.05 level are crossed cells "
            "with n of at most "
            + ck(str(nmax), SRC,
                 "strata: largest non-significant cell size", raw=nmax)
            + " — too few pairs for a paired test to resolve, not "
            "evidence of a weak stratum. Every single-factor stratum is "
            "significant on its own.</p>")
    elif nonsig:
        where = "; ".join(f"{_rung_label(s['rung'])}: {s['stratum']} "
                          f"(n={s['n']}, p={_fp(s['mcnemar_p'])})"
                          for s in nonsig)
        cross_p = (
            "<p>The crossed goal-by-travel breakdown is in the source "
            "file but kept out of the table above. The non-significant "
            f"results sit at: {esc(where)}.</p>")
    else:
        cross_p = ("<p>The crossed goal-by-travel breakdown is in the "
                   "source file but kept out of the table above; every "
                   "stratum, single-factor and crossed, is significant "
                   "at the 0.05 level.</p>")

    # ---- mechanism: the margin is largest where the vocabulary helps -
    mech = ""
    o6 = by.get("g16r6", {}).get("goal OPEN")
    w6 = by.get("g16r6", {}).get("goal walled")
    o8 = by.get("g16r8", {}).get("goal OPEN")
    w8 = by.get("g16r8", {}).get("goal walled")
    if o6 and w6 and o8 and w8:
        if o6["diff"] > w6["diff"] and o8["diff"] > w8["diff"]:
            mech = (
                "<p><b>The stratification is not just a robustness check "
                "— it localizes the mechanism.</b> A goal square with no "
                "wall on any side cannot stop a slider, so it can only be "
                "reached by first parking a helper robot for the target "
                "to collide with — the very maneuver the subgoal "
                "vocabulary names as a single decision. At both 16×16 "
                "rungs the margin is bigger in that goal-OPEN class than "
                "behind walled goals: "
                + ck(f"{o6['diff'] * 100:+.1f}", SRC,
                     "mechanism: g16r6 goal-OPEN margin", raw=o6["diff"])
                + " vs "
                + ck(f"{w6['diff'] * 100:+.1f}", SRC,
                     "mechanism: g16r6 goal-walled margin", raw=w6["diff"])
                + " points at 6 robots (a slight edge), "
                + ck(f"{o8['diff'] * 100:+.1f}", SRC,
                     "mechanism: g16r8 goal-OPEN margin", raw=o8["diff"])
                + " vs "
                + ck(f"{w8['diff'] * 100:+.1f}", SRC,
                     "mechanism: g16r8 goal-walled margin", raw=w8["diff"])
                + " at 8 robots (unambiguous).")
        else:
            mech = (
                "<p>Comparing goal-OPEN against goal-walled margins at "
                "the 16×16 rungs: "
                f"{o6['diff'] * 100:+.1f} vs {w6['diff'] * 100:+.1f} "
                "points at 6 robots and "
                f"{o8['diff'] * 100:+.1f} vs {w8['diff'] * 100:+.1f} "
                "at 8 robots — the helper-parking class does not show a "
                "uniformly larger margin here, so no mechanism claim is "
                "made from this split.")
        par = by.get("g16r6", {}).get("goal walled / travel low")
        if (par and par["n"] <= 13 and par["diff"] == 0
                and par["backward_rate"] == 1.0):
            mech += (
                " And where the proxies say the puzzles are easy, the "
                "planners agree: the easiest crossed cell (16×16 · 6 "
                "robots, goal walled / travel low, n="
                + ck(str(par["n"]), SRC, "parity cell: n", raw=par["n"])
                + ") is "
                + ck(f"{par['backward_rate'] * 100:.1f}%", SRC,
                     "parity cell: subgoal solve rate",
                     raw=par["backward_b2_solved"])
                + " vs "
                + ck(f"{par['forward_rate'] * 100:.1f}%", SRC,
                     "parity cell: move-by-move solve rate",
                     raw=par["forward_solved"])
                + " — parity, not a manufactured win.")
        mech += "</p>"

    return (head + summary + table + prov + cross_p + mech
            + "</section>")
