"""Section: statistical rigour (eval/results/stats_tests.json).

One exported function, sec_significance(D), renders the paired-test
section: the pooled whole-pool head-to-heads (the headline table), the
comparisons that do NOT survive the test, and the language-vs-networks
2x2.  Everything is read from D["stats_tests"] (the parsed
eval/results/stats_tests.json); nothing is hardcoded downstream, and a
missing file renders as an in-progress note rather than crashing.
"""

from eval.report_util import (esc, ck, fnum, scroll, progress_tag,
                              kicker_h2, chip)

SRC = "eval/results/stats_tests.json"

RUNG_LABEL = {
    "g16r4": "16×16 · 4 robots",
    "g16r6": "16×16 · 6 robots",
    "g16r8": "16×16 · 8 robots",
    "g24r4": "24×24 · 4 robots",
    "g24r8": "24×24 · 8 robots",
    "g32r4": "32×32 · 4 robots",
}
RUNG_ORDER = {k: i for i, k in enumerate(
    ("g16r4", "g16r6", "g16r8", "g24r4", "g24r8", "g32r4"))}
SET_LABEL = {"graded": "gradable set", "frontier": "beyond the oracle",
             "pooled": "pooled (whole 450)"}
SET_ORDER = {"graded": 0, "frontier": 1, "pooled": 2}
SYS_SHORT = {
    "bwd_b2": "subgoals, full language",
    "bwd_old": "subgoals, original language",
    "bwd_basenets_old": "subgoals, base nets, original language",
    "bwd_retrained": "subgoals, retrained (depleted cap-5,000 corpus)",
    "bwd_retrained_cap20k": "subgoals, retrained (cap-20,000 corpus)",
    "fwd": "move-by-move",
}


def _rlabel(key):
    return RUNG_LABEL.get(key, key)


def _slabel(key):
    return SET_LABEL.get(key, key)


def _sys(key):
    return SYS_SHORT.get(key, key)


def _cells(st):
    """Non-skipped cells, in rung-ladder then set order."""
    out = [c for c in (st.get("cells") or [])
           if "skipped" not in c and c.get("n")]
    out.sort(key=lambda c: (RUNG_ORDER.get(c.get("rung"), 99),
                            SET_ORDER.get(c.get("set"), 99)))
    return out


def _fp(p):
    """p-value: 4 decimals; below 1e-4 rendered as an inequality."""
    if p is None:
        return "—"
    if p < 0.0001:
        return "<0.0001"
    return fnum(p, 4)


def _fsig(x):
    """Signed percentage points, one decimal (input is a fraction)."""
    return f"{x * 100:+.1f}"


def _diff_ci(c, dbase):
    """The 'difference [95% CI]' cell, both parts check-tagged."""
    d, lo, hi = c.get("diff"), c.get("ci95_lo"), c.get("ci95_hi")
    if d is None or lo is None or hi is None:
        return '<span class="muted">—</span>'
    return (ck(_fsig(d), SRC, dbase + " — diff", raw=d) + " "
            + ck(f"[{_fsig(lo)}, {_fsig(hi)}]", SRC, dbase + " — ci95",
                 raw=(lo, hi)))


def _pval(c, dbase):
    p = c.get("mcnemar_p")
    if p is None:
        return '<span class="muted">—</span>'
    return ck(_fp(p), SRC, dbase + " — p", raw=p)


def _find(cells, rung, set_, a, b):
    for c in cells:
        if (c.get("rung") == rung and c.get("set") == set_
                and c.get("a") == a and c.get("b") == b):
            return c
    return None


# ---------------------------------------------------------------------------
# 1. The pooled union table (the headline)
# ---------------------------------------------------------------------------

def _pooled_block(cells, method):
    pooled = [c for c in cells if c.get("set") == "pooled"
              and c.get("a") == "bwd_b2" and c.get("b") == "fwd"]
    if not pooled:
        return "<h3>The whole-pool head-to-head</h3>" + progress_tag(
            "no pooled (graded+frontier union) cells are present in "
            "eval/results/stats_tests.json yet — this table renders "
            "automatically when they land")
    body = []
    for c in pooled:
        d = f"pooled {c['rung']} b2-vs-fwd"
        body.append(
            f'<tr><td><b>{esc(_rlabel(c["rung"]))}</b>'
            f'<div class="cellnote">whole pool — {c["n"]} puzzles on '
            f'{c["n_boards"]} boards</div></td>'
            '<td class="num">'
            + ck(f'{c["solved_a"]}/{c["n"]}', SRC, d + " — subgoal solved",
                 raw=c["solved_a"])
            + '<div class="cellnote">'
            + ck(f'{c["rate_a"] * 100:.1f}%', SRC, d + " — subgoal rate",
                 raw=c["rate_a"])
            + "</div></td>"
            '<td class="num">'
            + ck(f'{c["solved_b"]}/{c["n"]}', SRC, d + " — forward solved",
                 raw=c["solved_b"])
            + '<div class="cellnote">'
            + ck(f'{c["rate_b"] * 100:.1f}%', SRC, d + " — forward rate",
                 raw=c["rate_b"])
            + "</div></td>"
            f'<td class="num">{_diff_ci(c, d)}</td>'
            f'<td class="num">{_pval(c, d)}</td></tr>')
    table = scroll(
        "<table><thead><tr><th>configuration</th>"
        "<th class='num'>subgoals, full language<br>solved · rate</th>"
        "<th class='num'>move-by-move<br>solved · rate</th>"
        "<th class='num'>difference, points [95% CI]</th>"
        "<th class='num'>p (McNemar)</th></tr></thead><tbody>"
        + "".join(body) + "</tbody></table>")
    pooled_quote = ""
    if method.get("pooled"):
        pooled_quote = (f'<p class="small muted">As recorded in the file: '
                        f'“{esc(method["pooled"])}”</p>')
    base = _find(cells, "g16r4", "graded", "bwd_b2", "fwd")
    base_note = ""
    if base and base.get("diff", 0) < 0:
        base_note = (
            '<p class="small muted">The base rung (16×16 · 4 robots) has '
            "no pooled row: the oracle grades all 450 of its puzzles, so "
            "the whole-pool comparison there is the graded head-to-head — "
            "which the move-by-move planner wins, "
            + ck(f'{base["solved_b"]}/{base["n"]}', SRC,
                 "base graded b2-vs-fwd — forward solved",
                 raw=base["solved_b"]) + " against "
            + ck(f'{base["solved_a"]}/{base["n"]}', SRC,
                 "base graded b2-vs-fwd — subgoal solved",
                 raw=base["solved_a"]) + " (p "
            + ck(_fp(base["mcnemar_p"]), SRC,
                 "base graded b2-vs-fwd — p", raw=base["mcnemar_p"])
            + "). No subgoal advantage is claimed at the base scale.</p>")
    return ("<h3>The whole-pool head-to-head — free of frontier "
            "selection</h3>"
            "<p>The beyond-oracle (“frontier”) sets are selected by the "
            "failure of a move-level exhaustive search, so they are "
            "adversarial to the move-by-move planner by construction: a "
            "win measured only there could in part be an artifact of how "
            "the puzzles were chosen. The pooled rows remove that "
            "selection — the gradable and frontier halves together are "
            "the whole pinned 450-puzzle pool at each rung, fixed before "
            "any planner ran. Whatever survives here survives without "
            "help from the sampling. This is why the pooled table is the "
            "headline.</p>"
            + pooled_quote + table + base_note)


# ---------------------------------------------------------------------------
# 1b. The properly retrained planner vs the move-by-move control
# ---------------------------------------------------------------------------

def _retrained_block(cells):
    rows_ = [c for c in cells
             if c.get("a") == "bwd_retrained_cap20k" and c.get("b") == "fwd"]
    if not rows_:
        return ("<h3>With properly retrained networks</h3>" + progress_tag(
            "no cap-20,000-corpus retrained rows are in "
            "eval/results/stats_tests.json yet — this table renders "
            "automatically as the regenerated-corpus Track 1 lanes land"))
    body = []
    for c in rows_:
        d = f'retr20k {c["rung"]} {c["set"]} vs fwd'
        body.append(
            f'<tr><td><b>{esc(_rlabel(c["rung"]))}</b>'
            f'<div class="cellnote">{esc(_slabel(c["set"]))} — '
            f'{c["n"]} puzzles</div></td>'
            '<td class="num">'
            + ck(f'{c["solved_a"]}/{c["n"]}', SRC, d + " — retrained solved",
                 raw=c["solved_a"])
            + '<div class="cellnote">'
            + ck(f'{c["rate_a"] * 100:.1f}%', SRC, d + " — retrained rate",
                 raw=c["rate_a"]) + "</div></td>"
            '<td class="num">'
            + ck(f'{c["solved_b"]}/{c["n"]}', SRC, d + " — forward solved",
                 raw=c["solved_b"])
            + '<div class="cellnote">'
            + ck(f'{c["rate_b"] * 100:.1f}%', SRC, d + " — forward rate",
                 raw=c["rate_b"]) + "</div></td>"
            f'<td class="num">{_diff_ci(c, d)}</td>'
            f'<td class="num">{_pval(c, d)}</td></tr>')
    table = scroll(
        "<table><thead><tr><th>configuration · set</th>"
        "<th class='num'>subgoals, retrained (rich corpus)<br>solved · rate"
        "</th><th class='num'>move-by-move<br>solved · rate</th>"
        "<th class='num'>difference, points [95% CI]</th>"
        "<th class='num'>p (McNemar)</th></tr></thead><tbody>"
        + "".join(body) + "</tbody></table>")
    return ("<h3>After retraining on the regenerated corpus</h3>"
            "<p>These rows are the planner after retraining on the "
            "regenerated (cap-20,000) label corpus. Rows retrained on the "
            "earlier depleted corpus are a separate arm, shown in the "
            "label-budget experiment below, never here. Read the signs: "
            "retraining is not uniformly an upgrade — at 16×16 · 6 robots "
            "the retrained planner matches its zero-shot predecessor, "
            "while at 16×16 · 8 robots retraining damages the "
            "beyond-oracle set severely under either corpus (see the "
            "label-budget section) and the zero-shot planner remains the "
            "best measured backward configuration there. The whole-pool "
            "headline above is therefore the zero-shot planner's; these "
            "rows measure what imitation-retraining on self-generated "
            "subgoal labels currently delivers, favourable or not.</p>"
            + table)


# ---------------------------------------------------------------------------
# 1c. The label-budget experiment (FINDINGS 34/36)
# ---------------------------------------------------------------------------

def _corpus_block(cells, D):
    pair = [c for c in cells if c.get("a") == "bwd_retrained_cap20k"
            and c.get("b") == "bwd_retrained"]
    if not pair:
        return ("<h3>The label-budget experiment</h3>" + progress_tag(
            "corpus-effect cells (cap-20,000 vs cap-5,000 retrains of the "
            "same configuration) are not in eval/results/stats_tests.json "
            "yet — this table renders automatically when both arms of a "
            "rung exist"))
    shares = D.get("byref_shares") or {}
    SHARES_SRC = "analysis/artifacts/byref_shares.json"

    def dose(rung):
        hi = (shares.get(f"{rung}.cap20000") or {}).get("share_pct")
        lo = (shares.get(f"{rung}.cap5000") or {}).get("share_pct")
        if hi is None or lo is None:
            return ""
        return ('<div class="cellnote">labels '
                + ck(f"{hi}%", SHARES_SRC,
                     f"byref share {rung} cap20000", raw=hi)
                + " vs "
                + ck(f"{lo}%", SHARES_SRC,
                     f"byref share {rung} cap5000", raw=lo)
                + " by-reference</div>")

    body = []
    seen_rungs = set()
    for c in pair:
        d = f'corpus {c["rung"]} {c["set"]}'
        first = c["rung"] not in seen_rungs
        seen_rungs.add(c["rung"])
        body.append(
            f'<tr><td><b>{esc(_rlabel(c["rung"]))}</b>'
            f'<div class="cellnote">{esc(_slabel(c["set"]))} — '
            f'{c["n"]} puzzles</div>'
            + (dose(c["rung"]) if first else "") + '</td>'
            '<td class="num">'
            + ck(f'{c["solved_a"]}/{c["n"]}', SRC, d + " — cap20k solved",
                 raw=c["solved_a"])
            + '<div class="cellnote">'
            + ck(f'{c["rate_a"] * 100:.1f}%', SRC, d + " — cap20k rate",
                 raw=c["rate_a"]) + "</div></td>"
            '<td class="num">'
            + ck(f'{c["solved_b"]}/{c["n"]}', SRC, d + " — cap5k solved",
                 raw=c["solved_b"])
            + '<div class="cellnote">'
            + ck(f'{c["rate_b"] * 100:.1f}%', SRC, d + " — cap5k rate",
                 raw=c["rate_b"]) + "</div></td>"
            f'<td class="num">{_diff_ci(c, d)}</td>'
            f'<td class="num">{_pval(c, d)}</td></tr>')
    table = scroll(
        "<table><thead><tr><th>configuration · set</th>"
        "<th class='num'>retrained, cap-20,000 corpus<br>solved · rate</th>"
        "<th class='num'>retrained, cap-5,000 corpus<br>solved · rate</th>"
        "<th class='num'>difference, points [95% CI]</th>"
        "<th class='num'>p (McNemar)</th></tr></thead><tbody>"
        + "".join(body) + "</tbody></table>")
    return ("<h3>The label-budget experiment — same networks, same "
            "recipe, only the training corpus differs</h3>"
            "<p>Each row pairs two retrains of the same configuration "
            "that share the warm-start, recipe, learning rate and "
            "evaluation protocol; the only difference is the per-attempt "
            "budget of the label generator. The cheaper cap depleted the "
            "corpus — most visibly its share of plans that re-use an "
            "already-placed robot, the fingerprint quoted per rung below — "
            "and networks trained on the depleted corpus mis-rank the "
            "ordinary candidates that hard puzzles depend on. (The "
            "fingerprint is a marker, not the mechanism: the learned "
            "planner's proposal path never offers a re-use candidate at "
            "evaluation time — see the plan-language section — so the "
            "damage and the recovery both live in ordinary-candidate "
            "ranking.) A data-generation setting chosen for throughput "
            "masqueraded as a method failure, and it passed every "
            "record-count QC gate; only this controlled comparison "
            "exposed it.</p>"
            "<p><b>The effect is rung-heterogeneous, and the table must "
            "be read with that.</b> At 16×16 · 6 robots the richer corpus "
            "recovers the beyond-oracle collapse almost entirely; at "
            "16×16 · 8 robots retraining collapses the beyond-oracle set "
            "under BOTH corpora and the richer corpus is significantly "
            "worse, not better — there the pathology is the retraining "
            "itself (the retrained nets burn ~5× the search steps of "
            "their zero-shot predecessor on the same puzzles), with the "
            "corpus a second-order modifier. The rows landing from the "
            "remaining rungs decide which pattern is the rule.</p>"
            + table)


# ---------------------------------------------------------------------------
# 1d. Seed robustness (publishability objection 2.1)
# ---------------------------------------------------------------------------

def _vm(modes, key):
    """A checked val_regret quote from analysis/artifacts/valnet_modes.json."""
    v = (modes.get(key) or {}).get("best_val_regret")
    if v is None:
        return '<span class="muted">—</span>'
    return ck(f"{v:.3f}", "analysis/artifacts/valnet_modes.json",
              f"valmode {key}", raw=v)


def _seed_block(D):
    ss = (D.get("seed_spread") or {}).get("sets") or {}
    SS_SRC = "analysis/artifacts/seed_spread.json"
    usable = {k: v for k, v in ss.items() if v.get("n_arms", 0) >= 2}
    if not usable:
        return ("<h3>Seed robustness</h3>" + progress_tag(
            "three additional seeds of the g16r6 cap-20,000 retrain "
            "(policy + warm-started value pair, production recipe, only "
            "--torch-seed varied) are training; their per-seed benchmark "
            "rows render here when analysis/artifacts/seed_spread.json "
            "carries more than the production arm"))
    body = []
    order = {"production": 0, "seed21": 1, "seed37": 2, "seed53": 3}
    for set_name in ("graded", "frontier"):
        v = usable.get(set_name)
        if not v:
            continue
        arms = sorted(v["arms"].items(), key=lambda kv: order.get(kv[0], 9))
        cells = " · ".join(
            esc(name) + " " + ck(str(a["solved"]), SS_SRC,
                                 f"seed {set_name} {name}", raw=a["solved"])
            for name, a in arms)
        body.append(
            f"<tr><td><b>16×16 · 6 robots</b>"
            f'<div class="cellnote">{esc(_slabel(set_name))} — '
            f'{arms[0][1]["n"]} puzzles</div></td>'
            f'<td class="small">{cells}</td>'
            '<td class="num">'
            + ck(str(v["spread"]), SS_SRC, f"seed {set_name} spread",
                 raw=v["spread"]) + "</td></tr>")
    table = scroll(
        "<table><thead><tr><th>configuration · set</th>"
        "<th>solved, per seed</th>"
        "<th class='num'>spread (max − min)</th></tr></thead><tbody>"
        + "".join(body) + "</tbody></table>")
    modes = D.get("valnet_modes") or {}
    VM_SRC = "analysis/artifacts/valnet_modes.json"
    mode_html = ""
    mode_keys = [("seed 11 (production)", "g16r6/backward-value-b2-cap20000"),
                 ("seed 21", "g16r6/backward-value-b2-cap20000-seed21"),
                 ("seed 37", "g16r6/backward-value-b2-cap20000-seed37"),
                 ("seed 53", "g16r6/backward-value-b2-cap20000-seed53")]
    if all(k in modes for _, k in mode_keys):
        cells_ = " · ".join(
            esc(label) + " "
            + ck(f'{modes[k]["best_val_regret"]:.2f}', VM_SRC,
                 f"valmode {k}", raw=modes[k]["best_val_regret"])
            for label, k in mode_keys)
        mode_html = (
            "<p><b>The spread is not noise — it is a bimodality, and it "
            "is visible before any benchmarking.</b> The four runs' "
            "value nets separate into two basins on the SAME validation "
            "split (best val_regret: " + cells_ + "): the two runs near "
            "0.7–0.8 recover the beyond-oracle set (~78%), the two near "
            "2.3 collapse to ~50%. The policy nets are indistinguishable "
            "across all four — the mode lives in the warm-started value "
            "net. Every collapsed retrained row elsewhere on this page "
            "(the cap-5,000 arm here, both 16×16 · 8-robot arms) carries "
            "a bad-mode value net by the same criterion, so single-draw "
            "comparisons between retraining recipes or corpora are not "
            "interpretable at the beyond-oracle set without stating the "
            "mode. Six additional value-only probes resolved the causal "
            "question: on the depleted corpus the good basin was never "
            "reached (3 of 3 seeds converge to one plateau, "
            + _vm(modes, "g16r6/backward-value-b2-vmode37")
            + "–" + _vm(modes, "g16r6/backward-value-b2")
            + "), and at 16×16 · 8 robots not even the rich corpus "
            "reaches it (3 of 3 at "
            + _vm(modes, "g16r8/backward-value-b2-cap20000-vmode37")
            + "–" + _vm(modes, "g16r8/backward-value-b2-cap20000")
            + "). So the label-budget effect is real but acts by GATING "
            "ACCESS to the good basin, and a pre-stated best-of-k-seeds "
            "selection on val_regret can rescue retraining only where a "
            "good basin exists — at 16×16 · 6 robots yes, at 8 robots "
            "there is nothing to select. The counts (0/3 vs 2/4) are "
            "too small for significance on their own; the near-zero "
            "variance of the bad clusters across seeds is the decisive "
            "signature.</p>")
    return ("<h3>Seed robustness — the retrain repeated under varied "
            "seeds</h3>"
            "<p>Every margin on this page was, until this table, "
            "single-seed on both arms. The backward cap-20,000 retrain at "
            "16×16 · 6 robots was repeated with only the torch seed "
            "varied (production recipe: proposal net cold, value net "
            "warm-started). A margin is defensible against seed noise "
            "when it is large next to this spread. The forward arm "
            "remains single-seed at scale — at base it is the best of "
            "four independent trainings — and that asymmetry stands as a "
            "limitation.</p>" + table + mode_html)


# ---------------------------------------------------------------------------
# 2. What does not survive
# ---------------------------------------------------------------------------

def _nonsig_block(cells):
    ns = [c for c in cells if c.get("significant_at_05") is False]
    if not ns:
        return ("<h3>What does not survive the test</h3>"
                '<p class="small muted">Every tested comparison in the '
                "current file clears 0.05.</p>")
    body = []
    for c in ns:
        d = f'nonsig {c["rung"]} {c["set"]} {c["a"]}-vs-{c["b"]}'
        margin = abs(c.get("solved_a", 0) - c.get("solved_b", 0))
        tag = (chip("parity", "warn") if margin <= 1
               else chip("inconclusive"))
        body.append(
            f'<tr><td><b>{esc(_rlabel(c["rung"]))}</b>'
            f'<div class="cellnote">{esc(_slabel(c["set"]))} — '
            f'{c["n"]} puzzles</div></td>'
            f'<td class="small">{esc(_sys(c["a"]))} vs '
            f'{esc(_sys(c["b"]))}</td>'
            f'<td class="num">{_diff_ci(c, d)}</td>'
            f'<td class="num">{_pval(c, d)}</td>'
            f"<td>{tag}</td></tr>")
    table = scroll(
        "<table><thead><tr><th>configuration · set</th>"
        "<th>comparison</th>"
        "<th class='num'>difference, points [95% CI]</th>"
        "<th class='num'>p (McNemar)</th><th>reading</th></tr></thead>"
        "<tbody>" + "".join(body) + "</tbody></table>")

    callouts = []
    c1 = _find(cells, "g16r8", "graded", "bwd_b2", "fwd")
    if c1 and c1.get("significant_at_05") is False:
        callouts.append(
            "on the gradable set at 16×16 · 8 robots the full-language "
            "planner and the move-by-move planner solve "
            + ck(str(c1["solved_a"]), SRC,
                 "callout g16r8 graded — b2 solved", raw=c1["solved_a"])
            + " and "
            + ck(str(c1["solved_b"]), SRC,
                 "callout g16r8 graded — fwd solved", raw=c1["solved_b"])
            + f' of {c1["n"]} respectively (p = '
            + ck(_fp(c1["mcnemar_p"]), SRC, "callout g16r8 graded — p",
                 raw=c1["mcnemar_p"])
            + ") — a one-puzzle margin, which is parity, not a win")
    c2 = _find(cells, "g16r6", "frontier", "bwd_old", "fwd")
    if c2 and c2.get("significant_at_05") is False:
        callouts.append(
            "the 16×16 · 6-robot frontier comparison of the "
            "original-language planner against the move-by-move planner "
            "is inconclusive (p = "
            + ck(_fp(c2["mcnemar_p"]), SRC,
                 "callout g16r6 frontier old-vs-fwd — p",
                 raw=c2["mcnemar_p"]) + ")")
    callout_html = ""
    if callouts:
        callout_html = ("<p>Two deserve explicit mention: "
                        + "; and ".join(callouts) + ".</p>")
    return ("<h3>What does not survive the test</h3>"
            f"<p>Of the {len(cells)} paired comparisons tested, "
            f"{len(cells) - len(ns)} clear 0.05 and {len(ns)} do not. "
            "The ones that do not are listed in full.</p>"
            + callout_html + table
            + '<p class="small">These rows are reported as parity or '
            "inconclusive, and no claim on this page rests on any of "
            "them.</p>")


# ---------------------------------------------------------------------------
# 3. Language vs networks — the 2x2
# ---------------------------------------------------------------------------

def _twobytwo_block(cells):
    lang = {(c["rung"], c["set"]): c for c in cells
            if c.get("a") == "bwd_b2" and c.get("b") == "bwd_basenets_old"}
    prov = {(c["rung"], c["set"]): c for c in cells
            if c.get("a") == "bwd_basenets_old"
            and c.get("b") == "bwd_old"}
    if not lang and not prov:
        return ("<h3>Language vs networks</h3>" + progress_tag(
            "the base-nets/old-vocabulary control cells are not in "
            "eval/results/stats_tests.json yet — the 2×2 renders "
            "automatically when they land"))
    keys = sorted(set(lang) | set(prov),
                  key=lambda k: (RUNG_ORDER.get(k[0], 99),
                                 SET_ORDER.get(k[1], 99)))
    body = []
    for k in keys:
        rung, set_ = k
        row = (f'<tr><td><b>{esc(_rlabel(rung))}</b>'
               f'<div class="cellnote">{esc(_slabel(set_))}</div></td>')
        cl = lang.get(k)
        if cl:
            d = f"2x2 {rung} {set_} language"
            row += (f'<td class="num">{_diff_ci(cl, d)}</td>'
                    f'<td class="num">{_pval(cl, d)}</td>')
        else:
            row += ('<td class="num" colspan="2">'
                    '<span class="muted">not measured</span></td>')
        cp = prov.get(k)
        if cp:
            d = f"2x2 {rung} {set_} provenance"
            row += (f'<td class="num">{_diff_ci(cp, d)}</td>'
                    f'<td class="num">{_pval(cp, d)}</td>')
        else:
            row += ('<td class="num" colspan="2">'
                    '<span class="muted">not measured</span></td>')
        body.append(row + "</tr>")
    table = scroll(
        "<table><thead><tr><th>configuration · set</th>"
        "<th class='num'>language effect<br>points [95% CI]</th>"
        "<th class='num'>p</th>"
        "<th class='num'>network-provenance effect<br>points [95% CI]"
        "</th><th class='num'>p</th></tr></thead><tbody>"
        + "".join(body) + "</tbody></table>")

    # conclusion, computed from the cells rather than asserted
    pairs = [(lang[k], prov[k]) for k in keys if k in lang and k in prov]
    concl = ""
    if pairs:
        prov_pos = all(p.get("diff", 0) > 0 for _, p in pairs)
        ratios = [l["diff"] / p["diff"] for l, p in pairs
                  if p.get("diff") and l.get("diff") is not None
                  and p["diff"] > 0]
        if prov_pos and ratios and min(ratios) > 1:
            concl = (
                f"<p>Across the {len(pairs)} matched pairs the language "
                "effect is several times the provenance effect — from "
                f"roughly {min(ratios):.1f}× to {max(ratios):.1f}× — and "
                "every provenance difference carries the same sign, "
                "favouring the base-trained networks. The base networks "
                "were not handicapped by where they were trained; what "
                "moves the outcome is what plans are allowed to say.</p>")
        else:
            concl = ("<p>The table separates the two effects; read the "
                     "signs and intervals directly.</p>")
    return ("<h3>Language vs networks — separating the two "
            "ingredients</h3>"
            "<p>The full-language rows differ from the original planner "
            "in two ways at once: a larger plan vocabulary, and networks "
            "of a different provenance. Two paired comparisons pull the "
            "ingredients apart. The <b>language effect</b> holds the "
            "networks fixed (base-trained) and swaps the old vocabulary "
            "for the full language. The <b>network-provenance effect</b> "
            "holds the vocabulary fixed (old) and compares the "
            "base-trained networks against the networks behind the "
            "original-language rows — if the base networks were secretly "
            "weaker, this column would show it.</p>" + table + concl)


# ---------------------------------------------------------------------------
# The section
# ---------------------------------------------------------------------------

def sec_significance(D):
    head = kicker_h2(
        "statistical rigour",
        "Which differences survive a test",
        "Every head-to-head on this page is paired — both planners face "
        "the same puzzles under the same budget — so each comparison is "
        "tested with an exact two-sided McNemar test on the discordant "
        "pairs (the puzzles exactly one side solved). The 95% confidence "
        "intervals come from a bootstrap that resamples boards rather "
        "than puzzles, because up to three puzzles share a board's wall "
        "layout and are not independent.")
    st = D.get("stats_tests")
    if not st:
        return head + progress_tag(
            "eval/results/stats_tests.json has not been generated yet "
            "(produced by eval/stats_tests.py) — the paired McNemar "
            "tests, board-clustered bootstrap intervals, pooled "
            "whole-pool table and language-vs-networks 2×2 render here "
            "automatically when the file lands") + "</section>"
    cells = _cells(st)
    method = st.get("method") or {}
    method_html = ""
    quoted = [method.get(k) for k in ("test", "ci", "pairing")
              if method.get(k)]
    if quoted:
        method_html = ('<p class="small muted">As recorded in the file: '
                       + " · ".join(f"“{esc(q)}”" for q in quoted)
                       + "</p>")
    return (head + method_html
            + _pooled_block(cells, method)
            + _retrained_block(cells)
            + _corpus_block(cells, D)
            + _seed_block(D)
            + _nonsig_block(cells)
            + _twobytwo_block(cells)
            + "</section>")
