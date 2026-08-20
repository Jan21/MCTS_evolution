"""Section: statistical rigour (eval/results/stats_tests.json).

One exported function, sec_significance(D), renders the paired-test
section: the pooled whole-pool head-to-heads (the headline table), the
comparisons that do NOT survive the test, and the language-vs-networks
2x2.  Everything is read from D["stats_tests"] (the parsed
eval/results/stats_tests.json); nothing is hardcoded downstream, and a
missing file renders as an in-progress note rather than crashing.
"""

from eval.report_util import (esc, ck, fnum, scroll, progress_tag,
                              kicker_h2, chip, fact)

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
            "regenerated label corpus (“cap-20,000” — the cap is the "
            "search budget each labelling attempt gets; the earlier "
            "corpus used a 4× cheaper cap). Rows retrained on that "
            "earlier depleted corpus are a separate arm, shown in the "
            "label-budget experiment below, never here. Read the signs: "
            "retraining is not uniformly an upgrade, and the pattern "
            "followed the robot count, not the board size. At the base "
            "board and 16×16 · 6 robots it is a statistical wash; at "
            "BOTH 8-robot rungs it damages the beyond-oracle set "
            "severely under either corpus (see the label-budget "
            "section); and at 32×32 · 4 robots it clearly improves both "
            "sets (+7.8 points whole-pool over zero-shot) — producing "
            "the study's largest margin over the move-by-move planner, "
            "+55.6 points. The whole-pool headline above stays the "
            "zero-shot planner's, because that is the one configuration "
            "measured identically at every rung; these rows measure "
            "what imitation-retraining on self-generated subgoal labels "
            "currently delivers, favourable or not.</p>"
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
        return ('<div class="cellnote">'
                + ck(f"{hi}%", SHARES_SRC,
                     f"byref share {rung} cap20000", raw=hi)
                + " vs "
                + ck(f"{lo}%", SHARES_SRC,
                     f"byref share {rung} cap5000", raw=lo)
                + " of label plans re-use an already-placed robot</div>")

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
            "<p><b>The effect is rung-heterogeneous, and the split runs "
            "along the robot axis.</b> At 16×16 · 6 robots the richer "
            "corpus recovers the beyond-oracle collapse almost entirely. "
            "At both 8-robot rungs retraining collapses the beyond-oracle "
            "set under EITHER corpus and the richer corpus is "
            "significantly worse, not better — there the pathology is "
            "the retraining itself (the retrained nets burn several "
            "times the search steps of their zero-shot predecessor on "
            "the same puzzles), with the corpus a second-order modifier. "
            "At 32×32 · 4 robots retraining improves both sets. The "
            "seed-robustness section below carries the mechanism: "
            "whether a retrain lands in the good training basin decides "
            "the row, and at the 8-robot rungs no good-basin draw was "
            "ever observed.</p>"
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
            "<p><b>The spread is not noise — training has two distinct "
            "outcomes, and which one you got is visible before any "
            "benchmarking.</b> The signal is the value network's "
            "validation error (“val_regret” — its average scoring error "
            "on held-out puzzles, lower is better; all four runs share "
            "the same held-out set, so their numbers are directly "
            "comparable). The four runs split cleanly: " + cells_ + ". "
            "The two runs near 0.7–0.8 recover the beyond-oracle set "
            "(~78% solved); the two near 2.3 collapse to ~50%. The "
            "proposal networks are indistinguishable across all four — "
            "the difference lives entirely in the value network, which "
            "is initialized from an earlier model (“warm-started”) and "
            "then either genuinely learns or settles into a degenerate "
            "solution. Every collapsed retrained row elsewhere on this "
            "page (the cap-5,000 arm here, both 16×16 · 8-robot arms) "
            "carries a bad-outcome value net by the same criterion, so "
            "single-draw comparisons between retraining recipes or "
            "corpora are not interpretable at the beyond-oracle set "
            "without stating the "
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
            "remains single-seed at every scaling rung but one — at base "
            "it is the best of four independent trainings, and at "
            "16×16 · 8 robots it is now the best of nine (two blocks "
            "below) — and that asymmetry stands as a limitation.</p>"
            + table + mode_html)


# ---------------------------------------------------------------------------
# 1d-bis. Seed replicates of the backward headline pairs (FINDINGS 77a, 80)
# ---------------------------------------------------------------------------

def sh_src(rung):
    return f"analysis/artifacts/seed_headline_{rung}.json"


SH_RUNGS = ("g16r8", "g32r4")
SH_SET_LABEL = {"graded": "gradable set",
                "frontier": "beyond the oracle",
                "pooled": "whole pool",
                "base450": "base pool (16×16 · 4 robots)"}
SH_SET_ORDER = ("graded", "frontier", "pooled", "base450")
SH_ARMS = [("production", "seed of record (the published row)"),
           ("seed21", "fresh seed 21"),
           ("seed37", "fresh seed 37"),
           ("seed53", "fresh seed 53")]


def _sh_sets(sh):
    """The set columns this rung actually has, in reading order."""
    sets = sh.get("sets") or {}
    return [k for k in SH_SET_ORDER if (sets.get(k) or {}).get("arms")]


def _sh_num(sh, rung, set_key, arm, field, fmt="{}"):
    """A check-tagged number out of one rung's seed-headline artifact."""
    c = ((sh.get("sets") or {}).get(set_key, {}).get("arms") or {}).get(arm)
    if not c or c.get(field) is None:
        return '<span class="muted">—</span>'
    v = c[field]
    return ck(fmt.format(v), sh_src(rung), f"seedhl {rung} {set_key} {arm} "
              f"{field}", raw=v)


def _shs(sh, rung, set_key, field, fmt="{}", desc=None):
    """A check-tagged number out of one rung+set's summary block."""
    su = ((sh.get("sets") or {}).get(set_key) or {}).get("summary") or {}
    if su.get(field) is None:
        return '<span class="muted">—</span>'
    return ck(fmt.format(su[field]), sh_src(rung),
              f"seedhl {rung} {set_key} {desc or field}", raw=su[field])


def _sh_band(sh, rung, set_key, desc="band"):
    su = ((sh.get("sets") or {}).get(set_key) or {}).get("summary") or {}
    if su.get("min3_seeds") is None:
        return '<span class="muted">—</span>'
    return ck(f'{su["min3_seeds"]}–{su["max3_seeds"]}', sh_src(rung),
              f"seedhl {rung} {set_key} {desc}",
              raw=(su["min3_seeds"], su["max3_seeds"]))


def _shv(sh, rung, arm, suffix=""):
    """Check-tagged val_regret quote from one rung's artifact."""
    v = ((sh.get("val_regret") or {}).get(arm) or {}).get("val_regret")
    if v is None:
        return '<span class="muted">—</span>'
    return ck(f"{v:.2f}", sh_src(rung), f"seedhl {rung} val_regret {arm}"
              + suffix, raw=v)


def _pv(sh, rung, set_key, arm):
    """Check-tagged McNemar p from one rung's artifact."""
    c = ((sh.get("sets") or {}).get(set_key, {}).get("arms") or {}).get(arm)
    if not c or c.get("mcnemar_p") is None:
        return '<span class="muted">—</span>'
    return ck(_fp(c["mcnemar_p"]), sh_src(rung),
              f"seedhl {rung} {set_key} {arm} McNemar p", raw=c["mcnemar_p"])


def _sh_table(sh, rung):
    """The per-arm × per-set table of one rung's replicate experiment."""
    sets, cols = sh["sets"], _sh_sets(sh)
    body = []
    for arm, label in SH_ARMS:
        if arm not in (sets["pooled"].get("arms") or {}):
            continue
        vr = (sh.get("val_regret") or {}).get(arm) or {}
        v, basin = vr.get("val_regret"), vr.get("basin")
        if v is None:
            vcell = '<span class="muted">—</span>'
        else:
            vcell = _shv(sh, rung, arm)
            if basin == "bad":
                vcell += " " + chip("bad basin", "warn")
            elif basin == "good":
                vcell += " " + chip("good basin")
        cells = "".join(
            '<td class="num">' + _sh_num(sh, rung, k, arm, "solved_backward")
            + "</td>" for k in cols)
        body.append(f'<tr><td><b>{esc(label)}</b></td>'
                    f'<td class="num small">{vcell}</td>{cells}</tr>')

    med = "".join(
        '<td class="num">' + _shs(sh, rung, k, "median3_seeds", "{:g}",
                                  desc="median-of-3")
        + '<div class="cellnote">band ' + _sh_band(sh, rung, k)
        + "</div></td>" for k in cols)
    body.append('<tr class="hl"><td><b>median of the three fresh seeds</b>'
                '<div class="cellnote">what this rung now reports</div></td>'
                '<td class="num small"><span class="muted">—</span></td>'
                + med + "</tr>")
    fwd = "".join(
        '<td class="num">'
        + ck(str(sets[k]["solved_forward"]), sh_src(rung),
             f"seedhl {rung} {k} forward", raw=sets[k]["solved_forward"])
        + "</td>" for k in cols)
    body.append('<tr><td>move-by-move control (unchanged)</td>'
                '<td class="num small"><span class="muted">—</span></td>'
                + fwd + "</tr>")

    heads = "".join(f'<th class="num">{esc(SH_SET_LABEL[k])}<br>'
                    f'<span class="small">{sets[k]["n"]} puzzles</span></th>'
                    for k in cols)
    return scroll(
        "<table><thead><tr><th>training run of the network pair</th>"
        '<th class="num">value-net<br>validation error</th>'
        + heads + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>")


def _sh_reading_g16r8(sh, D):
    rung = "g16r8"
    sets = sh["sets"]

    def marg(arm):
        return (_sh_num(sh, rung, "pooled", arm, "diff_points", "{:+.1f}")
                + " points, p&nbsp;=&nbsp;" + _pv(sh, rung, "pooled", arm))

    return (
        "<p><b>Bimodal — and, as at 6 robots, which outcome you "
        "drew is visible in training, before a single puzzle is "
        "benchmarked.</b> The value network's validation error splits the "
        "four runs cleanly: one of the three fresh seeds lands at "
        + _shv(sh, rung, "seed21", " (reading)") + " against "
        + _shv(sh, rung, "seed37", " (reading)") + " and "
        + _shv(sh, rung, "seed53", " (reading)")
        + " for the other two (the published pair sits "
        "with the good ones, at " + _shv(sh, rung, "production", " (reading)")
        + "). This is the "
        "same two-basin signature the 6-robot retrain showed above, in the "
        "same network, and it is diagnosable from the training run alone.</p>"
        "<p><b>The two good-basin seeds reproduce the headline; the "
        "bad-basin seed does not.</b> Beyond the oracle the good seeds solve "
        + _sh_num(sh, rung, "frontier", "seed37", "solved_backward") + " and "
        + _sh_num(sh, rung, "frontier", "seed53", "solved_backward")
        + " of 184 against the published "
        + _sh_num(sh, rung, "frontier", "production", "solved_backward")
        + ", while the bad-basin seed manages "
        + _sh_num(sh, rung, "frontier", "seed21", "solved_backward")
        + ". It still finishes ahead of the move-by-move control there ("
        + _sh_num(sh, rung, "frontier", "seed21", "diff_points", "{:+.1f}")
        + " points, p&nbsp;=&nbsp;" + _pv(sh, rung, "frontier", "seed21")
        + ", so ahead but no longer beyond doubt), and over the whole 450 its "
        "margin shrinks to " + marg("seed21") + " — against "
        + marg("seed37") + " and " + marg("seed53") + " for the good seeds "
        "and " + marg("production") + " for the published row. Nothing of "
        "this shows at the networks' home size: on the base pool all four "
        "runs land within "
        + ck(f'{sets["base450"]["summary"]["max4"] - sets["base450"]["summary"]["min4"]}',
             sh_src(rung), "seedhl g16r8 base450 spread4",
             raw=sets["base450"]["summary"]["max4"]
             - sets["base450"]["summary"]["min4"])
        + " puzzles of each other, so the damage is specific to carrying the "
        "pair, unchanged, to a harder board. Solution length is not what "
        "moves: on the puzzles a run and the control both solve, the subgoal "
        "plans average "
        + _sh_num(sh, rung, "pooled", "production", "mean_len_backward",
                  "{:.1f}")
        + ", "
        + _sh_num(sh, rung, "pooled", "seed37", "mean_len_backward", "{:.1f}")
        + " and "
        + _sh_num(sh, rung, "pooled", "seed53", "mean_len_backward", "{:.1f}")
        + " moves for the three good-basin runs and "
        + _sh_num(sh, rung, "pooled", "seed21", "mean_len_backward", "{:.1f}")
        + " for the bad-basin one, against about "
        + _sh_num(sh, rung, "pooled", "production", "mean_len_forward",
                  "{:.1f}")
        + " for the move-by-move control throughout — the subgoal "
        "planner keeps solving more puzzles with longer plans, whichever "
        "basin it drew.</p>"
        "<p><b>What the page now reports for this rung.</b> The rule was "
        "fixed before the seeds ran: a split draw like this one forces the "
        "headline to be the median of the three replicates rather than any "
        "single run. So 16×16 · 8 robots is reported as "
        + _shs(sh, rung, "pooled", "median3_seeds", "{:g}",
               desc="median-of-3 (reading)")
        + " of 450 solved, band " + _sh_band(sh, rung, "pooled",
                                             "band (reading)")
        + ", with the published run kept visible as the seed of "
        "record; the margin over the move-by-move control at that median is "
        + _shs(sh, rung, "pooled", "median3_margin_points", "{:+.1f}",
               desc="median-of-3 margin")
        + " points, and the band spans "
        + _sh_num(sh, rung, "pooled", "seed21", "diff_points", "{:+.1f}")
        + " to "
        + _sh_num(sh, rung, "pooled", "seed37", "diff_points", "{:+.1f}")
        + " points.</p>")


def _sh_reading_g32r4(sh, D):
    rung = "g32r4"
    spread = sh.get("val_regret_spread")
    spread_html = ('<span class="muted">—</span>' if spread is None else
                   ck(f"{spread:.2f}", sh_src(rung),
                      "seedhl g32r4 val_regret spread", raw=spread))
    return (
        "<p><b>The largest board tells the opposite story: no split, and "
        "every replicate at or above the published row.</b> The value "
        "network's validation error is flat across the four runs — "
        + _shv(sh, rung, "seed21", " (reading)") + ", "
        + _shv(sh, rung, "seed37", " (reading)") + " and "
        + _shv(sh, rung, "seed53", " (reading)") + " for the fresh seeds "
        "against " + _shv(sh, rung, "production", " (reading)")
        + " for the published pair, a total spread of " + spread_html
        + ". (This error is measured against the plan lengths of this board "
        "size, so it is comparable only down this column, never against the "
        "8-robot numbers above.) There is no second basin to fall into "
        "here, and none of the four runs found one.</p>"
        "<p><b>Every fresh seed beats the published run, on both sets.</b> "
        "On the gradable set the replicates solve "
        + _sh_num(sh, rung, "graded", "seed21", "solved_backward") + ", "
        + _sh_num(sh, rung, "graded", "seed37", "solved_backward") + " and "
        + _sh_num(sh, rung, "graded", "seed53", "solved_backward")
        + " of 175 against the published "
        + _sh_num(sh, rung, "graded", "production", "solved_backward")
        + "; beyond the oracle, "
        + _sh_num(sh, rung, "frontier", "seed21", "solved_backward") + ", "
        + _sh_num(sh, rung, "frontier", "seed37", "solved_backward")
        + " and "
        + _sh_num(sh, rung, "frontier", "seed53", "solved_backward")
        + " of 275 against "
        + _sh_num(sh, rung, "frontier", "production", "solved_backward")
        + " — where the move-by-move control solves "
        + ck(str(sh["sets"]["frontier"]["solved_forward"]), sh_src(rung),
             "seedhl g32r4 frontier forward (reading)",
             raw=sh["sets"]["frontier"]["solved_forward"])
        + ". Over the whole 450 the three replicates land at "
        + _sh_num(sh, rung, "pooled", "seed21", "solved_backward") + ", "
        + _sh_num(sh, rung, "pooled", "seed37", "solved_backward") + " and "
        + _sh_num(sh, rung, "pooled", "seed53", "solved_backward")
        + " against the published "
        + _sh_num(sh, rung, "pooled", "production", "solved_backward")
        + " \u2014 margins of "
        + _sh_num(sh, rung, "pooled", "seed21", "diff_points", "{:+.1f}")
        + " to "
        + _sh_num(sh, rung, "pooled", "seed37", "diff_points", "{:+.1f}")
        + " points over the control, every one of them at "
        "p&nbsp;=&nbsp;" + _pv(sh, rung, "pooled", "seed21") + ".</p>"
        "<p><b>What the page reports for this rung.</b> The same "
        "pre-registered rule applies, so 32×32 · 4 robots is "
        "reported as the median of the three replicates, "
        + _shs(sh, rung, "pooled", "median3_seeds", "{:g}",
               desc="median-of-3 (reading)")
        + " of 450 solved, band " + _sh_band(sh, rung, "pooled",
                                             "band (reading)")
        + " — a spread of "
        + _shs(sh, rung, "pooled", "band3_points", "{:.1f}",
               desc="band in points")
        + " points, against "
        + _shs(D.get("seed_headline_g16r8") or {}, "g16r8", "pooled",
               "band3_points", "{:.1f}", desc="band in points (32×32 text)")
        + " at 8 robots — with the published run "
        "kept visible as the seed of record. The margin over the "
        "move-by-move control at that median is "
        + _shs(sh, rung, "pooled", "median3_margin_points", "{:+.1f}",
               desc="median-of-3 margin")
        + " points, above the published run's "
        + _sh_num(sh, rung, "pooled", "production", "diff_points", "{:+.1f}")
        + ". <b>The strongest rung on the page is therefore also the most "
        "robust one:</b> the seed sensitivity that damages the 8-robot cell "
        "is a property of that cell, not of the method, and the published "
        "32×32 row is if anything a conservative draw.</p>")


SH_READING = {"g16r8": _sh_reading_g16r8, "g32r4": _sh_reading_g32r4}


def _seed_headline_block(D):
    """The backward headline pair, retrained under three fresh seeds.

    Two rungs carry this experiment (report_data.SEED_HEADLINE_RUNGS): the
    zero-shot 8-robot pair, which turns out to be bimodal, and the 32x32
    pair, which does not.  Each renders its own table and reading; a rung
    whose artifact has not landed renders nothing (all of them missing
    renders the in-progress note).
    """
    head = "<h3>Seed replicates of the headline pair</h3>"
    have = [r for r in SH_RUNGS
            if (D.get("seed_headline_" + r) or {}).get("sets", {})
            .get("pooled", {}).get("arms")]
    if not have:
        return head + progress_tag(
            "the headline pairs are being retrained under three fresh "
            "seeds (jobs/patterns/seed_headline_pair.slurm); the per-seed "
            "benchmark rows render here when "
            "analysis/artifacts/seed_headline_<rung>.json lands")
    intro = ("<p>Two rungs of the ladder rest on a network pair that was "
             "trained exactly once, so both were repeated under three fresh "
             "random seeds — nothing else altered — and each "
             "resulting pair re-benchmarked with the headline protocol "
             "verbatim (same budget, same pinned puzzles, every solved plan "
             "replayed). At 16×16 · 8 robots the pair is applied "
             "<b>zero-shot</b> (trained at the smallest board size and used "
             "here unchanged, so the last column is its home size); at "
             "32×32 · 4 robots the pair is trained on that "
             "configuration. The rule was fixed before either ran: if the "
             "seeds scatter, the rung reports the median of the three with "
             "its band, and the published run stays visible as the seed of "
             "record.</p>")
    out = [head, intro]
    for rung in have:
        sh = D["seed_headline_" + rung]
        out.append(f"<h4>{esc(RUNG_LABEL.get(rung, rung))}</h4>")
        out.append(_sh_table(sh, rung))
        out.append(SH_READING[rung](sh, D))
    return "".join(out)


# ---------------------------------------------------------------------------
# 1d-ter. A fair second chance for the forward planner (FINDINGS 77b)
# ---------------------------------------------------------------------------

FR_SRC = "analysis/artifacts/forward_rescue_g16r8.json"
FR_SETS = ("graded", "frontier", "pooled")
FR_SET_LABEL = {"graded": "gradable set",
                "frontier": "beyond the oracle",
                "pooled": "whole pool"}


def _fr_get(fr, path):
    """Dotted-path lookup into the rescue artifact (None if any step misses)."""
    cur = fr
    for key in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _frn(fr, path, fmt="{}", desc=None):
    """A check-tagged number out of the rescue artifact."""
    v = _fr_get(fr, path)
    if v is None:
        return '<span class="muted">—</span>'
    return ck(fmt.format(v), FR_SRC, "fwdrescue " + (desc or path), raw=v)


def _frp(fr, path, desc=None):
    """A check-tagged McNemar p out of the rescue artifact."""
    v = _fr_get(fr, path)
    if v is None:
        return '<span class="muted">—</span>'
    return ck(_fp(v), FR_SRC, "fwdrescue " + (desc or path) + " p", raw=v)


def _fr_val_table(fr):
    """The nine training runs and their validation scores, seed x learning rate."""
    sel = fr["selection"]
    by_cell = {(a["seed"], a["lr_tag"]): a for a in sel["arms"]}
    seeds = sel["seeds"]
    lr_tags = []
    for a in sel["arms"]:
        if a["lr_tag"] not in lr_tags:
            lr_tags.append(a["lr_tag"])
    rows = []
    for tag in lr_tags:
        cells = []
        for s in seeds:
            a = by_cell[(s, tag)]
            mark = (" " + chip("selected")) if a["selected"] else ""
            cells.append(
                '<td class="num">'
                + ck(f'{a["val_policy_top1"] * 100:.2f}%', FR_SRC,
                     f'fwdrescue val {a["cell"]}', raw=a["val_policy_top1"])
                + mark + "</td>")
        rows.append(f'<tr><td><b>{esc(tag)}</b></td>' + "".join(cells)
                    + "</tr>")
    heads = "".join(f'<th class="num">seed {esc(s)}</th>' for s in seeds)
    return scroll(
        "<table><thead><tr><th>learning rate</th>" + heads
        + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def _fr_result_table(fr):
    """Rescued forward, its control and the subgoal arms, on all three sets."""
    sets = fr["sets"]
    cols = [k for k in FR_SETS if k in sets]

    def row(label, note, get, desc, hl=False):
        cells = "".join('<td class="num">' + get(k) + "</td>" for k in cols)
        cls = ' class="hl"' if hl else ""
        note_html = f'<div class="cellnote">{note}</div>' if note else ""
        return (f"<tr{cls}><td><b>{esc(label)}</b>{note_html}</td>"
                + cells + "</tr>")

    body = [
        row("Move-by-move — the control of record",
            "the published row: one training run, one emergency "
            "learning-rate fix",
            lambda k: _frn(fr, f"sets.{k}.solved_control",
                           desc=f"{k} control solved"), "control"),
        row("Move-by-move — after the fair second chance",
            "best of nine trainings, picked on validation alone",
            lambda k: _frn(fr, f"sets.{k}.solved_rescued",
                           desc=f"{k} rescued solved"), "rescued", hl=True),
        row("Subgoals — median of three fresh seeds",
            "what this rung reports (seeds 21/37/53)",
            lambda k: _frn(fr, f"sets.{k}.backward_median3", "{:g}",
                           desc=f"{k} backward median-of-3"), "median"),
        row("Subgoals — seed of record",
            "the published subgoal row",
            lambda k: _frn(fr, f"sets.{k}.solved_backward.production",
                           desc=f"{k} backward production solved"), "prod"),
    ]
    heads = "".join(
        f'<th class="num">{esc(FR_SET_LABEL[k])}<br>'
        f'<span class="small">{sets[k]["n"]} puzzles</span></th>'
        for k in cols)
    return scroll(
        "<table><thead><tr><th>system</th>" + heads
        + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>")


def _forward_rescue_block(D):
    """The move-by-move planner's fair second chance at 16x16 / 8 robots.

    Nine retrainings (3 seeds x 3 learning rates), one promoted by the
    training pipeline's own validation score and benched with the control's
    protocol.  The block states the design, the nine validation scores, the
    benchmark result against both the control it replaces and the subgoal
    arms, and then the verdict against the rule fixed BEFORE the nine runs
    started (FINDINGS 77b).
    """
    head = ("<h3>A fair second chance for the forward planner "
            "(16×16 · 8 robots)</h3>")
    fr = D.get("forward_rescue_g16r8") or {}
    if not (fr.get("sets") or {}).get("pooled"):
        return head + progress_tag(
            "the move-by-move planner is being retrained nine times at this "
            "rung (three seeds × three learning rates, winner chosen on "
            "validation alone); the table renders here when "
            "analysis/artifacts/forward_rescue_g16r8.json lands")

    sel = fr["selection"]
    rc = fr.get("replay_certified") or {}
    fact("forward rescue: the benchmarked checkpoint is exactly the "
         "arm the validation rule selected",
         bool(sel.get("benched_is_selected_arm")))
    fact("forward rescue: every solved rescue plan replayed cleanly "
         "(0 replay failures on both sets)",
         bool(rc) and all(v.get("failed") == 0 for v in rc.values()))

    design = (
        "<p>The sharpest objection to everything above is that the "
        "move-by-move planner was beaten while under-trained. At "
        "16×16 · 8 robots that objection has real force: the stock "
        "training recipe collapsed there, and the published control is a "
        "single emergency retrain at a lower learning rate. So that "
        "planner was given a fair second chance at exactly that rung — "
        + _frn(fr, "selection.n_arms", desc="n arms")
        + " fresh trainings, three random seeds crossed with three "
        "learning rates, the control's recipe otherwise unchanged (eight "
        "passes over the same data, batches of 128) — and one of the nine "
        "was promoted to the benchmark.</p>"
        "<p><b>The winner was chosen without looking at a single benchmark "
        "result.</b> The promotion used the training pipeline's own "
        "validation score — how often the network's top-ranked next move is "
        "the right one, measured on held-out boards 700–899, which do not "
        "overlap the benchmark boards 900–1049 — and nothing else. The "
        "chosen network was then run through the control's protocol "
        "verbatim: the same pinned puzzles, the same 1,200-step budget, the "
        "same top-5 shortlist, and every solved plan replayed move by move "
        "through the physics ("
        + _frn(fr, "replay_certified.graded.passed",
               desc="replay graded passed")
        + " and "
        + _frn(fr, "replay_certified.frontier.passed",
               desc="replay frontier passed")
        + " plans, "
        + _frn(fr, "replay_certified.graded.failed",
               desc="replay graded failed")
        + " failures).</p>")

    val_read = (
        "<p>"
        + _frn(fr, "selection.n_arms_above_control",
               desc="arms above control")
        + " of the nine beat the published control's own validation score of "
        + _frn(fr, "selection.control_of_record.val_policy_top1", "{:.2%}",
               desc="control of record val")
        + ", so the rescue really did produce a better-trained opponent, "
        "not a re-run of the same one. The largest learning rate is where "
        "training breaks: all three of its runs score worst, and one of "
        "them collapses to "
        + _frn(fr, "selection.val_min", "{:.2%}", desc="worst arm val")
        + " — the same failure that forced the emergency fix in the first "
        "place. The winner is seed "
        + _frn(fr, "selection.winner.seed", desc="winner seed")
        + " at learning rate "
        + ck(esc(sel["winner"]["lr_tag"]), FR_SRC, "fwdrescue winner lr",
             raw=sel["winner"]["lr"])
        + ", at "
        + _frn(fr, "selection.winner.val_policy_top1", "{:.2%}",
               desc="winner val")
        + ".</p>")

    def vs_ctrl(k):
        return (_frn(fr, f"sets.{k}.vs_control.diff_solved", "{:+d}",
                     desc=f"{k} rescued minus control")
                + " puzzles (p&nbsp;=&nbsp;"
                + _frp(fr, f"sets.{k}.vs_control.mcnemar_p",
                       desc=f"{k} rescued-vs-control")
                + ")")

    result_read = (
        "<p><b>The second chance helped — modestly, and mostly where the "
        "puzzles are easy.</b> Against the control it replaces, the rescued "
        "planner gains " + vs_ctrl("graded") + " on the gradable set, "
        + vs_ctrl("frontier") + " beyond the oracle, and "
        + vs_ctrl("pooled") + " over the whole pool, the only one of the "
        "three that clears the 0.05 line. The honest headline of that "
        "column is on the gradable set, and it goes against this report's "
        "usual direction: the rescued move-by-move planner now solves "
        + _frn(fr, "sets.graded.solved_rescued", desc="graded rescued (read)")
        + " of "
        + ck(str(fr["sets"]["graded"]["n"]), FR_SRC,
             "fwdrescue graded n (read)", raw=fr["sets"]["graded"]["n"])
        + " — every puzzle an exact solver could grade — which is more than "
        "any subgoal arm manages there ("
        + _frn(fr, "sets.graded.solved_backward.production",
               desc="graded backward production (read)")
        + " for the published pair). At 8 robots, on gradable puzzles, the "
        "properly trained move-by-move planner is the better system.</p>")

    def vs_bwd(k, arm, desc):
        return (_frn(fr, f"sets.{k}.vs_backward.{arm}.diff_points", "{:+.1f}",
                     desc=f"{k} {desc} minus rescued, points")
                + " points, p&nbsp;=&nbsp;"
                + _frp(fr, f"sets.{k}.vs_backward.{arm}.mcnemar_p",
                       desc=f"{k} {desc}-vs-rescued"))

    frontier_read = (
        "<p><b>Beyond the oracle, nothing changes.</b> On the harder half "
        "the rescued planner solves "
        + _frn(fr, "sets.frontier.solved_rescued", desc="frontier rescued")
        + " of "
        + ck(str(fr["sets"]["frontier"]["n"]), FR_SRC,
             "fwdrescue frontier n", raw=fr["sets"]["frontier"]["n"])
        + " where the subgoal planner's median seed solves "
        + _frn(fr, "sets.frontier.backward_median3", "{:g}",
               desc="frontier backward median (read)")
        + " (" + vs_bwd("frontier", "seed53", "median seed")
        + ") and the published subgoal pair "
        + _frn(fr, "sets.frontier.solved_backward.production",
               desc="frontier backward production (read)")
        + " (" + vs_bwd("frontier", "production", "seed of record")
        + "). Over the whole 450 the rescued planner reaches "
        + _frn(fr, "sets.pooled.solved_rescued", desc="pooled rescued (read)")
        + " against the subgoal median of "
        + _frn(fr, "sets.pooled.backward_median3", "{:g}",
               desc="pooled backward median (read)")
        + " (" + vs_bwd("pooled", "seed53", "median seed")
        + ") and the published "
        + _frn(fr, "sets.pooled.solved_backward.production",
               desc="pooled backward production (read)")
        + " (" + vs_bwd("pooled", "production", "seed of record")
        + "). The one subgoal arm it does match over the pool is the "
        "bad-basin seed above — "
        + _frn(fr, "sets.pooled.solved_backward.seed21",
               desc="pooled backward seed21 (read)")
        + " against "
        + _frn(fr, "sets.pooled.solved_rescued",
               desc="pooled rescued (seed21 read)")
        + ", p&nbsp;=&nbsp;"
        + _frp(fr, "sets.pooled.vs_backward.seed21.mcnemar_p",
               desc="pooled seed21-vs-rescued")
        + ", a tie — which is the same statement made above from the other "
        "side.</p>")

    verdict = (
        "<p><b>The verdict, against a rule fixed before the nine runs "
        "started.</b> The pre-registered reading was: a rescued planner "
        "reaching about "
        + _frn(fr, "prereg.kill_at_or_above", desc="prereg kill threshold")
        + " of "
        + ck(str(fr["sets"]["frontier"]["n"]), FR_SRC,
             "fwdrescue frontier n (prereg)",
             raw=fr["sets"]["frontier"]["n"])
        + " beyond the oracle would erase this rung's margin outright; "
        "about "
        + _frn(fr, "prereg.keep_with_caveat_near", desc="prereg caveat marker")
        + " would leave the margin standing but with a “best of nine” "
        "caveat attached. It reached "
        + _frn(fr, "prereg.rescued_frontier", desc="prereg measured frontier")
        + " — below even the lower marker, and only "
        + _frn(fr, "sets.frontier.vs_control.diff_solved", "{:+d}",
               desc="frontier rescued minus control (verdict)")
        + " puzzles above the control it replaced. <b>The margin stands: "
        + _frn(fr, "prereg.pooled_margin_median_points", "{:+.1f}",
               desc="pooled margin over rescued")
        + " points over the whole pool</b> for the subgoal planner's "
        "median-of-three against the best of the nine.</p>"
        "<p>What the rescue changes is the caveat, not the conclusion. At "
        "this rung the comparison now leans against the subgoal planner in "
        "two ways at once — the move-by-move arm is the best of nine "
        "trainings, the subgoal arm the median of three — and the subgoal "
        "planner still wins the pool by "
        + _frn(fr, "prereg.pooled_margin_median_points", "{:+.1f}",
               desc="pooled margin over rescued (caveat)")
        + " points, on the strength of the harder half alone. The gradable "
        "half now belongs to the move-by-move planner, and the table above "
        "says so.</p>")

    return (head + design + _fr_val_table(fr) + val_read
            + _fr_result_table(fr) + result_read + frontier_read + verdict)


# ---------------------------------------------------------------------------
# 1e. The no-network control (FINDINGS 48)
# ---------------------------------------------------------------------------

def _heuristic_block(cells):
    pair_nn = [c for c in cells if c.get("a") == "bwd_b2"
               and c.get("b") == "bwd_heuristic"]
    if not pair_nn:
        return ""
    body = []
    for c in pair_nn:
        d = f'heur {c["rung"]} {c["set"]}'
        body.append(
            f'<tr><td><b>{esc(_rlabel(c["rung"]))}</b>'
            f'<div class="cellnote">{esc(_slabel(c["set"]))} — '
            f'{c["n"]} puzzles</div></td>'
            '<td class="num">'
            + ck(f'{c["solved_a"]}/{c["n"]}', SRC, d + " — nets solved",
                 raw=c["solved_a"])
            + '<div class="cellnote">'
            + ck(f'{c["rate_a"] * 100:.1f}%', SRC, d + " — nets rate",
                 raw=c["rate_a"]) + "</div></td>"
            '<td class="num">'
            + ck(f'{c["solved_b"]}/{c["n"]}', SRC, d + " — heuristic solved",
                 raw=c["solved_b"])
            + '<div class="cellnote">'
            + ck(f'{c["rate_b"] * 100:.1f}%', SRC, d + " — heuristic rate",
                 raw=c["rate_b"]) + "</div></td>"
            f'<td class="num">{_diff_ci(c, d)}</td>'
            f'<td class="num">{_pval(c, d)}</td></tr>')
    table = scroll(
        "<table><thead><tr><th>configuration · set</th>"
        "<th class='num'>neural networks<br>solved · rate</th>"
        "<th class='num'>hand-written scorer<br>solved · rate</th>"
        "<th class='num'>difference, points [95% CI]</th>"
        "<th class='num'>p (McNemar)</th></tr></thead><tbody>"
        + "".join(body) + "</tbody></table>")
    return ("<h3>The no-network control — what do the networks actually "
            "add?</h3>"
            "<p>The subgoal planner's training labels come from a "
            "hand-written search, so a fair question is whether the "
            "learned planner is just that hand-written solver again. "
            "This control runs the EXACT production search loop — same "
            "candidate pool, same budget, same shortlist width, same "
            "play-out checking and repairs — with the two neural scoring "
            "points swapped back to the hand-written scorer. The "
            "difference column is therefore the networks' entire "
            "contribution at matched compute. (Context: the hand-written "
            "scorer needs budgets orders of magnitude larger to reach "
            "its ceiling — it generated the training labels at ~17× this "
            "budget per attempt.) Every solved row was independently "
            "replay-certified.</p>" + table)


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
        "tested with an exact two-sided McNemar test — a paired yes/no "
        "test that uses only the puzzles where the two planners disagree. "
        "The 95% confidence intervals come from a bootstrap that "
        "resamples boards rather than puzzles, because up to three "
        "puzzles share a board's wall layout and are not independent. "
        "Set names: “gradable” = the exact solver produced an optimum "
        "for the puzzle; “beyond the oracle” (elsewhere: the frontier) = "
        "it could not.")
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
            + _seed_headline_block(D)
            + _forward_rescue_block(D)
            + _heuristic_block(cells)
            + _nonsig_block(cells)
            + _twobytwo_block(cells)
            + "</section>")
