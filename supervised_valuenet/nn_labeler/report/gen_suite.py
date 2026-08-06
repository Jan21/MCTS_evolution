"""Single-source-of-truth suite page for the NN-labeler track.

Reads ONLY result JSONs already on disk and emits suite.html next to itself.
Every cell traces to a file; a missing file renders as a "pending" cell, never
a crash and never a hand-typed number. Regenerate any time:

    python nn_labeler/report/gen_suite.py       (from supervised_valuenet/)

LOCAL FILE ONLY — never published anywhere (owner's standing order).
"""
import html
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SV = Path(__file__).resolve().parents[2]  # supervised_valuenet/
OUT = Path(__file__).with_name("suite.html")

CFGS = ["g24r4", "g24r8", "g32r4"]
BWD = "backward subgoal planner (prefix-check)"

# ---------------------------------------------------------------- loaders ---

def load(rel):
    p = SV / rel
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception as e:  # truncated mid-write etc. — show, don't crash
        return {"_load_error": f"{rel}: {e}"}


def agg(rel, system=BWD, prefix=False):
    """aggregate block of one system in a comparison JSON (or None)."""
    d = load(rel)
    if not d or "_load_error" in d or "systems" not in d:
        return None
    if prefix:  # first system whose name starts with `system`
        for name, s in d["systems"].items():
            if name.startswith(system):
                return s.get("aggregate")
        return None
    s = d["systems"].get(system)
    return s.get("aggregate") if s else None


def gate_summary(rel):
    d = load(rel)
    if not d or "_load_error" in d or "summary" not in d:
        return None
    return d["summary"]

# ------------------------------------------------------------ html helpers --

def esc(x):
    return html.escape(str(x))


def fmt(v, spec=".3f"):
    if v is None:
        return '<td class="pend">pending</td>'
    if isinstance(v, float):
        return f"<td>{v:{spec}}</td>"
    return f"<td>{esc(v)}</td>"


def pct(v):
    return '<td class="pend">pending</td>' if v is None else f"<td>{100*v:.1f}%</td>"


def row(cells, cls=""):
    return f'<tr class="{cls}">' + "".join(cells) + "</tr>"


def table(headers, rows_, note=""):
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    n = f'<p class="note">{note}</p>' if note else ""
    return (f'<div class="tw"><table><thead><tr>{h}</tr></thead>'
            f'<tbody>{"".join(rows_)}</tbody></table></div>{n}')

# ------------------------------------------------------------- sections -----

def sec_downstream():
    out = ["<h2>1 &middot; Downstream equivalence — the headline table</h2>",
           "<p>Planners retrained on NN-generated labels vs the exact-solver-"
           "trained originals, on identical sha-pinned benchmark instances "
           "(playable-moves scoring, 1200 expansions, k=5). The twin corpora "
           "replay the exact corpora's own boards and instances — the only "
           "variable is where the labels came from. All three cells are "
           "leakage-free: the labeler never trained on any of these configs. "
           "The forward planner is an untouched control (descent keeps no "
           "primitive-move sequences, so a forward twin would change the "
           "training signal, not just the label source).</p>"]
    for cfg in CFGS:
        rows_ = []
        specs = [
            ("backward — exact labels", f"scaling/results/{cfg}/comparison.json", BWD, False, ""),
            ("backward — NN twin labels", f"scaling/results/{cfg}/comparison_nntwin.json", BWD, False, "hl"),
            ("forward — exact (untouched control)", f"scaling/results/{cfg}/comparison.json", "forward", True, "ctl"),
        ]
        for label, rel, system, pfx, cls in specs:
            a = agg(rel, system, prefix=pfx)
            if a is None:
                rows_.append(row([f"<td>{esc(label)}</td>",
                                  '<td class="pend" colspan="6">pending '
                                  f'({esc(rel.split("/")[-1])})</td>'], cls))
            else:
                rows_.append(row([f"<td>{esc(label)}</td>",
                                  pct(a.get("solve_rate")),
                                  fmt(a.get("pct_optimal"), ".1f"),
                                  fmt(a.get("mean_regret")),
                                  fmt(a.get("mean_moves"), ".2f"),
                                  fmt(a.get("mean_seconds"), ".1f"),
                                  fmt(a.get("n"), "d")], cls))
        # frontier (previously-unsolved instances)
        for label, rel, cls in [
                ("backward — exact, frontier", f"scaling/results/{cfg}/comparison_ungraded.json", "sub"),
                ("backward — NN twin, frontier", f"scaling/results/{cfg}/comparison_ungraded_nntwin.json", "sub hl")]:
            a = agg(rel)
            if a is None:
                note = "no exact frontier row exists" if cfg == "g24r4" and "ungraded.json" in rel \
                    else f'pending ({rel.split("/")[-1]})'
                rows_.append(row([f"<td>{esc(label)}</td>",
                                  f'<td class="pend" colspan="6">{esc(note)}</td>'], cls))
            else:
                rows_.append(row([f"<td>{esc(label)}</td>", pct(a.get("solve_rate")),
                                  "<td>—</td>", "<td>—</td>",
                                  fmt(a.get("mean_moves"), ".2f"),
                                  fmt(a.get("mean_seconds"), ".1f"),
                                  fmt(a.get("n"), "d")], cls))
        out.append(f"<h3>{cfg}</h3>")
        out.append(table(["system / label source", "solve rate", "% optimal",
                          "mean regret", "mean moves", "mean s", "n"], rows_))
    return "\n".join(out)


def sec_label_quality():
    out = ["<h2>2 &middot; Label quality vs the exact solver</h2>",
           "<p>How close the NN labels are to exact optima where ground truth "
           "exists. <em>argmin agreement</em> (does the label pick the same "
           "best candidate?) is the load-bearing metric for label use; "
           "absolute calibration degrades with size while ordering holds "
           "(FINDINGS 59).</p>"]
    rows_ = []
    for label, rel in [
            ("v1 gate 24&times;24 (600 test inst)", "nn_labeler/results/capgate_g24r4.json"),
            ("v1 gate 32&times;32 (600 test inst)", "nn_labeler/results/capgate_g32r4.json"),
            ("v2 gate 24&times;24 (600 test inst)", "nn_labeler/results/v2gate_g24r4.json"),
            ("v2 gate 32&times;32 (600 test inst)", "nn_labeler/results/v2gate_g32r4.json"),
            ("twin corpus g24r4 (full)", "nn_labeler/results/twin_gate_g24r4.json"),
            ("twin corpus g24r8 (full)", "nn_labeler/results/twin_gate_g24r8.json"),
            ("twin corpus g32r4 (full)", "nn_labeler/results/twin_gate_g32r4.json")]:
        s = gate_summary(rel)
        if s is None:
            rows_.append(row([f"<td>{label}</td>",
                              '<td class="pend" colspan="6">pending</td>']))
        else:
            rows_.append(row([f"<td>{label}</td>",
                              pct(s.get("argmin_agreement")),
                              pct(s.get("share_gap_zero")),
                              fmt(s.get("gap_mean")),
                              fmt(s.get("gap_p90"), ".0f"),
                              fmt(s.get("negative_gaps"), "d"),
                              fmt(s.get("n_candidate_matches"), "d")]))
    out.append(table(["gate", "argmin agree", "labels exactly optimal",
                      "mean gap", "p90 gap", "negative gaps", "n candidates"],
                     rows_))
    return "\n".join(out)


def sec_audit_curve():
    out = ["<h2>3 &middot; Value-audit curve (test split, both banked nets)</h2>"]
    v1 = load("nn_labeler/results/audit_prod_v1_s11_full.json")
    v2 = load("nn_labeler/results/audit_prod2_s11_lr1e-4.json")
    rows_ = []
    cfgs = ["g8r4", "g12r4", "g16r6", "g24r4", "g32r4"]
    for c in cfgs:
        a = (v1 or {}).get("configs", {}).get(c)
        b = (v2 or {}).get("configs", {}).get(c)
        rows_.append(row([f"<td>{c}</td>",
                          fmt(a and a.get("top1_optimal")), fmt(a and a.get("regret")),
                          fmt(b and b.get("top1_optimal")), fmt(b and b.get("regret")),
                          fmt(b and b.get("bias"), "+.2f")]))
    out.append(table(["config", "v1 top-1", "v1 regret", "v2 top-1",
                      "v2 regret", "v2 bias"], rows_,
                     note="v1 = banked capstone net; v2 = prod2 (job 4616677). "
                          "The descent gate above, not this audit, decides "
                          "which net labels the twins."))
    return "\n".join(out)


def sec_b2():
    out = ["<h2>4 &middot; B2 vocabulary — the compute-saving regime</h2>",
           "<p>Base-vocabulary labeling is cheaper exact (170&ndash;250&times;, "
           "FINDINGS 58/59); the NN labeler's economic case is the extended B2 "
           "vocabulary, where the exact campaign projected ~441 node-hours and "
           "its iteration cap destroyed the by-reference labels it existed to "
           "produce (13.5% uncapped &rarr; 4.8% capped). Descent has no "
           "iteration budget, so that trade-off cannot arise. Kept strictly "
           "separate from the base tables (vocabulary house rule).</p>"]
    rows_ = []
    for cfg in ["g16r6", "g32r4"]:
        lab = SV / f"nn_labeler/results/b2lab_{cfg}.jsonl"
        byref = tot = None
        if lab.exists():
            tot = byref = 0
            for line in open(lab):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                starts = {tuple(h[0]) for h in r["helpers"]}
                tot += 1
                byref += tuple(r["cand_helper"][0]) not in starts
        g = gate_summary(f"nn_labeler/results/b2gate_{cfg}.json")
        rows_.append(row([
            f"<td>{cfg}</td>",
            (f"<td>{100*byref/tot:.1f}% ({byref}/{tot})</td>" if tot
             else '<td class="pend">pending</td>'),
            pct(g.get("argmin_agreement") if g else None),
            fmt(g.get("gap_mean") if g else None)]))
    out.append(table(["config", "by-reference share (uncapped exact: 13.5%; "
                      "capped: 4.8–12.7%)", "argmin agree vs exact B2",
                      "mean gap"], rows_,
                     note="Source: job 4618888 (b2_payoff.slurm) — B2 net on "
                          "the five existing B2 corpora, then UNCAPPED descent "
                          "labeling."))
    return "\n".join(out)


def sec_ladder():
    out = ["<h2>5 &middot; Quality vs board size (the ladder)</h2>"]
    specs = [("8&times;8 (mixed net)", "nn_labeler/results/dgate_mix_none_s11_g8r4.json"),
             ("10&times;10 (mixed net)", "nn_labeler/results/dgate_mix_none_s11_g10r4.json"),
             ("12&times;12 (mixed net)", "nn_labeler/results/dgate_mix_none_s11_g12r4.json"),
             ("16&times;16 (mixed net)", "nn_labeler/results/dgate_mix_none_s11_g16r4.json")]
    for g in range(17, 32):
        if g == 24:
            specs.append(("24&times;24 (v1, capstone)",
                          "nn_labeler/results/capgate_g24r4.json"))
            continue
        specs.append((f"{g}&times;{g} (v1)",
                      f"nn_labeler/results/dgate_ladder_g{g}r4.json"))
    specs.append(("32&times;32 (v1, capstone)",
                  "nn_labeler/results/capgate_g32r4.json"))
    for g in (40, 48, 56, 64):
        specs.append((f"{g}&times;{g} (v1, coarse)",
                      f"nn_labeler/results/coarsegate_g{g}r4.json"))
    rows_ = []
    for label, rel in specs:
        s = gate_summary(rel)
        if s is None:
            rows_.append(row([f"<td>{label}</td>",
                              '<td class="pend" colspan="4">pending</td>']))
        else:
            rows_.append(row([f"<td>{label}</td>", pct(s.get("argmin_agreement")),
                              pct(s.get("share_gap_zero")), fmt(s.get("gap_mean")),
                              fmt(s.get("negative_gaps"), "d")]))
    out.append(table(["board", "argmin agree", "labels exactly optimal",
                      "mean gap", "negative gaps"], rows_,
                     note="8–16 from the debug-battery gates (mixed net); "
                          "everything from 17 up is the banked v1 net, 600 "
                          "replayed test instances per rung. 64 is the last "
                          "size with exact ground truth (Rust engine hard "
                          "limit); 80/96 have no gate by definition and "
                          "report certification stats + downstream solve "
                          "rate instead."))
    return "\n".join(out)


def main():
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=SV,
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        rev = "?"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = "\n".join([sec_downstream(), sec_label_quality(), sec_audit_curve(),
                      sec_b2(), sec_ladder()])
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NN-labeler suite — single source of truth</title>
<style>
:root {{ --ink:#1a1a1a; --mut:#6a6a6a; --line:#d8d8d8; --hl:#f3f7ee;
        --ctl:#f5f5f5; --bg:#fff; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --ink:#e8e8e8; --mut:#9a9a9a; --line:#3a3a3a; --hl:#20281c;
          --ctl:#242424; --bg:#161616; }} }}
body {{ font: 15px/1.55 system-ui, sans-serif; color: var(--ink);
       background: var(--bg); max-width: 62rem; margin: 2rem auto;
       padding: 0 1rem; }}
h1 {{ font-size: 1.5rem; }} h2 {{ font-size: 1.2rem; margin-top: 2.2rem; }}
h3 {{ font-size: 1rem; margin: 1.2rem 0 .3rem; }}
.tw {{ overflow-x: auto; }}
table {{ border-collapse: collapse; margin: .4rem 0; min-width: 40rem; }}
th, td {{ border: 1px solid var(--line); padding: .3rem .6rem;
         text-align: right; font-variant-numeric: tabular-nums; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ font-weight: 600; }}
tr.hl td {{ background: var(--hl); }}
tr.ctl td {{ background: var(--ctl); color: var(--mut); }}
tr.sub td:first-child {{ padding-left: 1.6rem; }}
td.pend {{ color: var(--mut); font-style: italic; text-align: left; }}
.note, .meta {{ color: var(--mut); font-size: .85rem; }}
.banner {{ border: 1px solid var(--line); padding: .5rem .8rem;
          font-size: .85rem; color: var(--mut); }}
</style></head><body>
<h1>NN-labeler track — results, single source of truth</h1>
<p class="banner">Auto-generated by <code>gen_suite.py</code> from result
JSONs on disk — no hand-typed numbers. Regenerate after any job lands.
Generated {stamp} at git {esc(rev)}. LOCAL FILE — not published.</p>
{body}
<p class="meta">Companion narratives: <code>process.html</code> (technical),
<code>scaling_story.html</code> (plain-English). Provenance:
<code>FINDINGS.md</code>, <code>TWIN_WIRING.md</code>.</p>
</body></html>"""
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
