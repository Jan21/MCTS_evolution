"""Sections: train/test hygiene · supervision budget.

Renders analysis/artifacts/dedup_audit.json (wall-layout near-duplicate
audit across train/val/bench for every configuration) and
analysis/artifacts/data_budget.json (training records per system per
configuration), plus a prose summary of analysis/tuning_ledger.md.
Both functions render a progress tag when their input has not landed.
"""

from eval.report_util import (esc, ck, fnum, scroll, progress_tag,
                              kicker_h2, chip)
from eval.report_data import RUNGS

DEDUP_SRC = "analysis/artifacts/dedup_audit.json"
BUDGET_SRC = "analysis/artifacts/data_budget.json"

_LABELS = {r["key"]: r["label"] for r in RUNGS}
_BASE_KEYS = {r["key"] for r in RUNGS if r.get("base")}

_PAIR_NAMES = {
    "train_vs_val": "train-vs-validation",
    "train_vs_bench": "train-vs-bench",
    "val_vs_bench": "validation-vs-bench",
}


def _label(cfg):
    return _LABELS.get(cfg, cfg)


def _ge_count(pair):
    """The 'n_jaccard_ge_<threshold>' counter, whatever the threshold is."""
    for k, v in pair.items():
        if k.startswith("n_jaccard_ge"):
            return v
    return None


# ---------------------------------------------------------------------------
# Train/test hygiene — the wall-layout near-duplicate audit
# ---------------------------------------------------------------------------

def sec_hygiene(D):
    head = kicker_h2(
        "train/test hygiene",
        "Could the benchmark boards leak into training?",
        "Benchmark board IDs are disjoint from training board IDs, but "
        "that alone rules out nothing: the networks never see an ID — "
        "they see wall geometry. A benchmark board whose wall layout is a "
        "near-copy of a training board would leak the test set regardless "
        "of its ID. The audit below compares the interior wall-segment "
        "sets of every train/val/bench board pair by Jaccard similarity, "
        "for every configuration in the study.")
    aud = D.get("dedup_audit")
    if not aud or not aud.get("configs"):
        return head + progress_tag(
            "analysis/artifacts/dedup_audit.json has not landed yet — the "
            "wall-layout near-duplicate audit renders here automatically "
            "when analysis/dedup_audit.py writes it") + "</section>"

    configs = aud["configs"]
    body = []
    tot_pairs = tot_exact = tot_ge = 0
    global_max, global_where = None, None
    bench_max = None
    for c in configs:
        cfg = c.get("config", "?")
        n = c.get("n") or {}
        pairs = c.get("pairs") or {}
        exact = sum(int(p.get("n_exact_duplicate") or 0)
                    for p in pairs.values())
        ge = sum(int(_ge_count(p) or 0) for p in pairs.values())
        npairs = sum(int(p.get("n_pairs") or 0) for p in pairs.values())
        tot_exact += exact
        tot_ge += ge
        tot_pairs += npairs
        cfg_max = None
        for pk, p in pairs.items():
            mj = p.get("max_jaccard")
            if mj is None:
                continue
            if cfg_max is None or mj > cfg_max:
                cfg_max = mj
            if global_max is None or mj > global_max:
                global_max, global_where = mj, (cfg, pk)
            if "bench" in pk and (bench_max is None or mj > bench_max):
                bench_max = mj

        def bc(split, cfg=cfg, n=n):
            v = n.get(split)
            if v is None:
                return "—"
            return ck(f"{v:,}", DEDUP_SRC,
                      f"hygiene {cfg} {split} boards", raw=v)

        mx_html = (ck(fnum(cfg_max, 3), DEDUP_SRC,
                      f"hygiene {cfg} max Jaccard", raw=cfg_max)
                   if cfg_max is not None else "—")
        body.append(
            f"<tr><td><b>{esc(_label(cfg))}</b>"
            f'<div class="cellnote">'
            + ck(f"{npairs:,}", DEDUP_SRC,
                 f"hygiene {cfg} pairs compared", raw=npairs)
            + " pairs compared</div></td>"
            f'<td class="num">{bc("train")} / {bc("val")} / {bc("bench")}'
            "</td>"
            f'<td class="num">'
            + ck(f"{exact:,}", DEDUP_SRC,
                 f"hygiene {cfg} exact-duplicate pairs", raw=exact)
            + "</td>"
            f'<td class="num">'
            + ck(f"{ge:,}", DEDUP_SRC,
                 f"hygiene {cfg} pairs at or above threshold", raw=ge)
            + "</td>"
            f'<td class="num">{mx_html}</td></tr>')
    table = scroll(
        "<table><thead><tr><th>configuration</th>"
        "<th class='num'>boards<br>train / val / bench</th>"
        "<th class='num'>exact duplicate pairs</th>"
        "<th class='num'>pairs at Jaccard ≥ threshold</th>"
        "<th class='num'>max Jaccard observed</th></tr></thead><tbody>"
        + "".join(body) + "</tbody></table>")

    # verdict — every figure computed from the file, none hardcoded
    thr = aud.get("threshold")
    thr_html = (ck(fnum(thr, 1), DEDUP_SRC,
                   "hygiene near-duplicate threshold", raw=thr)
                if thr is not None else "near-duplicate")
    if tot_exact == 0 and tot_ge == 0:
        opening = "<p><b>Nothing leaks.</b> Across "
    else:
        opening = "<p><b>The audit found matches.</b> Across "
    verdict = (
        opening
        + ck(f"{tot_pairs:,}", DEDUP_SRC,
             "hygiene total pairs compared", raw=tot_pairs)
        + " board pairs compared there are "
        + ck(f"{tot_exact:,}", DEDUP_SRC,
             "hygiene total exact duplicates", raw=tot_exact)
        + " exact duplicates and "
        + ck(f"{tot_ge:,}", DEDUP_SRC,
             "hygiene total pairs at or above threshold", raw=tot_ge)
        + f" pairs at or above the {thr_html} threshold.")
    if global_max is not None:
        where = ""
        if global_where:
            wcfg, wpk = global_where
            where = (f" — a {_PAIR_NAMES.get(wpk, wpk.replace('_', ' '))} "
                     f"pair at {esc(_label(wcfg))}")
        verdict += (
            " The most similar pair anywhere in the audit reaches Jaccard "
            + ck(fnum(global_max, 3), DEDUP_SRC,
                 "hygiene max Jaccard anywhere", raw=global_max)
            + where)
        if bench_max is not None:
            verdict += (
                ", and the most similar pair involving any benchmark "
                "board reaches "
                + ck(fnum(bench_max, 3), DEDUP_SRC,
                     "hygiene max Jaccard involving a bench board",
                     raw=bench_max))
        if thr is not None and global_max < thr:
            verdict += " — both far below the threshold"
        verdict += "."
    n_cfg = len(configs)
    ids_clean = all(not (c.get("id_overlap") or {}) for c in configs)
    if ids_clean:
        verdict += (
            " The audit also re-verified the ID split directly: the "
            "board-ID sets of train, validation and benchmark are "
            "pairwise disjoint in all "
            + ck(str(n_cfg), DEDUP_SRC,
                 "hygiene configurations audited", raw=n_cfg)
            + " configurations (every <code>id_overlap</code> field is "
            "empty).")
    else:
        verdict += (
            " Note: the audit records a non-empty <code>id_overlap</code> "
            "for at least one configuration — inspect the file before "
            "trusting the split.")
    verdict += "</p>"

    # shared board pools along the robot-count axis — verified, not assumed
    by_cfg = {c.get("config"): c for c in configs}

    def _same_pool(a, b):
        ca, cb = by_cfg.get(a), by_cfg.get(b)
        return bool(ca and cb and ca.get("n") == cb.get("n")
                    and ca.get("pairs") == cb.get("pairs"))

    identical = _same_pool("g16r6", "g16r8") and _same_pool("g24r4", "g24r8")
    pool = ("<p>One design fact is worth surfacing rather than leaving to "
            "look like a copy-paste error: scaling rungs that differ only "
            "in robot count draw on exactly the same board pool, so the "
            "robot-count axis is a controlled comparison on identical "
            "walls — only the number of robots changes. ")
    if identical:
        pool += ("That is why the 6- and 8-robot 16×16 rungs report "
                 "figure-for-figure identical audit rows above, and "
                 "likewise the two 24×24 rungs. ")
    pool += ("The base 16×16 · 4-robot configuration keeps its own larger "
             "split and shares its pool with no scaling rung.</p>")

    method = aud.get("method") or ""
    meth = (f'<p class="small muted">Method, as recorded in the audit '
            f"file: “{esc(method)}” Source: "
            f'<code class="small">{DEDUP_SRC}</code></p>' if method else "")
    return head + table + verdict + pool + meth + "</section>"


# ---------------------------------------------------------------------------
# Supervision budget — training records per system per configuration
# ---------------------------------------------------------------------------

def sec_databudget(D):
    head = kicker_h2(
        "supervision budget",
        "How much training data each planner received",
        "Both planners learn supervised from the same exact solver, but "
        "they consume different kinds of labels — the subgoal planner "
        "learns to rank candidates at search decisions, the move-by-move "
        "planner learns from states along optimal trajectories. This "
        "table is the full accounting of what each side was trained on.")
    bud = D.get("data_budget")
    rows = (bud or {}).get("rows")
    if not rows:
        return head + progress_tag(
            "analysis/artifacts/data_budget.json has not landed yet — the "
            "per-system training-data accounting renders here "
            "automatically when analysis/data_budget.py writes it") \
            + "</section>"

    order, by_cfg = [], {}
    for r in rows:
        cfg = r.get("config", "?")
        if cfg not in by_cfg:
            by_cfg[cfg] = []
            order.append(cfg)
        by_cfg[cfg].append(r)

    body = []
    for cfg in order:
        first_tb = None
        for i, r in enumerate(by_cfg[cfg]):
            sysname = r.get("system", "?")
            unit = r.get("unit", "?")
            rec = r.get("records")
            tb = r.get("train_boards")
            if rec is None:
                rec_html = chip(r.get("status") or "not generated", "warn")
            else:
                rec_html = ck(f"{rec:,}", BUDGET_SRC,
                              f"budget {cfg} {sysname} records", raw=rec)
            if tb is None:
                tb_html = "—"
            elif first_tb is None:
                first_tb = tb
                tb_html = ck(f"{tb:,}", BUDGET_SRC,
                             f"budget {cfg} training boards", raw=tb)
            elif tb == first_tb:
                tb_html = f"{tb:,}"
            else:  # differs within the config — tag it separately
                tb_html = ck(f"{tb:,}", BUDGET_SRC,
                             f"budget {cfg} {sysname} training boards",
                             raw=tb)
            cfg_cell = (f"<td><b>{esc(_label(cfg))}</b></td>" if i == 0
                        else "<td></td>")
            fnote = (f'<div class="cellnote"><code class="small">'
                     f'{esc(r["file"])}</code></div>' if r.get("file")
                     else "")
            body.append(
                f"<tr>{cfg_cell}<td>{esc(sysname)}{fnote}</td>"
                f"<td>{esc(unit)}</td>"
                f'<td class="num">{rec_html}</td>'
                f'<td class="num">{tb_html}</td></tr>')
    table = scroll(
        "<table><thead><tr><th>configuration</th><th>system</th>"
        "<th>unit</th><th class='num'>records</th>"
        "<th class='num'>training boards</th></tr></thead><tbody>"
        + "".join(body) + "</tbody></table>")

    note = (bud or {}).get("note") or ""
    caveat = (
        "<p><b>These record counts must not be compared across "
        "systems.</b> The file (<code class=\"small\">" + BUDGET_SRC
        + "</code>) carries its own caveat: “" + esc(note) + "” One "
        "backward record is one candidate subgoal at one search decision; "
        "one forward record is one board state on an optimal move "
        "trajectory. The units measure different things, so only "
        "within-system comparisons across configurations are meaningful — "
        "and any quoted count must carry its unit.</p>")

    # the asymmetry that IS meaningful: same boards, many more forward
    # records — computed per scaling configuration from the table itself
    ratios = []
    boards_equal = True
    board_vals = set()
    for cfg in order:
        if cfg in _BASE_KEYS:
            continue
        grp = by_cfg[cfg]
        tbs = {r.get("train_boards") for r in grp
               if r.get("train_boards") is not None}
        if len(tbs) == 1:
            board_vals |= tbs
        else:
            boards_equal = False
        fwd = [r for r in grp if r.get("unit") == "moves"
               and r.get("records")]
        bwd = [r for r in grp if r.get("unit") == "decisions"
               and r.get("records")]
        if len(fwd) == 1 and len(bwd) == 1:
            ratios.append((cfg, fwd[0]["records"] / bwd[0]["records"]))
    ratio_html = ""
    if ratios:
        lo_cfg, lo = min(ratios, key=lambda t: t[1])
        hi_cfg, hi = max(ratios, key=lambda t: t[1])
        if boards_equal and len(board_vals) == 1:
            b = next(iter(board_vals))
            same_boards = ("the same "
                           + ck(f"{b:,}", BUDGET_SRC,
                                "budget shared non-base training boards",
                                raw=b)
                           + " boards")
        else:
            same_boards = "the same boards"
        ratio_html = (
            "<p>One asymmetry <i>is</i> meaningful, because it holds the "
            f"boards fixed: at every scaling configuration both systems "
            f"train on {same_boards}, and the forward planner receives "
            "many times more supervision records from them — between "
            + ck(f"{lo:.1f}×", BUDGET_SRC,
                 "budget records ratio, smallest "
                 "(forward moves per backward decision)", raw=lo)
            + " and "
            + ck(f"{hi:.1f}×", BUDGET_SRC,
                 "budget records ratio, largest "
                 "(forward moves per backward decision)", raw=hi)
            + " the backward planner's count (smallest at "
            f"{esc(_label(lo_cfg))}, largest at {esc(_label(hi_cfg))}). "
            "That is inherent to the formulations: every optimal "
            "trajectory contributes all of its states, while the backward "
            "corpus grows only with search decisions taken. It is not a "
            "tuning choice, but it means the forward networks see far "
            "more gradient signal per board.</p>")

    ledger = (
        "<p>Data volume is half of the training story; the other half — "
        "who got tuned, and how — is recorded in the two-sided tuning "
        'ledger at <code class="small">analysis/tuning_ledger.md</code>. '
        "Three of its facts belong next to this table. First, the "
        "move-by-move planner needed a per-scale learning-rate rescue at "
        "four of the six configurations, each time after observed "
        "validation collapse; its stock recipe survived unchanged only at "
        "the base scale and at 24×24 with 4 robots. Second, the subgoal "
        "planner's value networks are warm-started from a banked "
        "predecessor checkpoint at every configuration — a stability "
        "mitigation with no forward counterpart — plus a lower learning "
        "rate at six or more robots and memory-driven batch caps. Third, "
        "neither side ever received a learning-rate sweep and both are "
        "single-seed everywhere, so no small difference between the "
        "systems is defensible against seed variation. The ledger also "
        "names the asymmetry that favours the move-by-move planner: its "
        "reported base-scale cell is the best of four independently "
        "trained systems, judged against a single subgoal run.</p>")

    return head + table + caveat + ratio_html + ledger + "</section>"
