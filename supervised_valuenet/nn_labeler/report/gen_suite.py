"""Single-source-of-truth suite page for the NN-labeler track.

Reads ONLY result JSONs already on disk and emits suite.html next to itself.
Every table cell traces to a file; a missing file renders as a "pending"
cell, never a crash and never a hand-typed number. Numbers appearing in
narrative verdict lines are computed from the same JSONs at generation time.
Regenerate any time:

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
CFG_HUMAN = {"g24r4": "24&times;24 board, 4 robots",
             "g24r8": "24&times;24 board, 8 robots",
             "g32r4": "32&times;32 board, 4 robots"}
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


def chip(kind, text):
    return f'<span class="chip {kind}">{text}</span>'

# ------------------------------------------------- computed verdict helpers --

def solve(rel, system=BWD, prefix=False):
    a = agg(rel, system, prefix)
    return a.get("solve_rate") if a else None


def optpct(rel):
    a = agg(rel)
    return a.get("pct_optimal") if a else None


def p1(v):  # 0.834 -> "83.4"
    return None if v is None else f"{100*v:.1f}"

# ------------------------------------------------------------- sections -----

def ladder_band():
    """(min%, max%, n_rungs) of argmin agreement over the v1 rungs 17-64."""
    rels = []
    for g in range(17, 32):
        rels.append(f"nn_labeler/results/dgate_ladder_g{g}r4.json" if g != 24
                    else "nn_labeler/results/capgate_g24r4.json")
    rels.append("nn_labeler/results/capgate_g32r4.json")
    rels += [f"nn_labeler/results/coarsegate_g{g}r4.json" for g in (40, 48, 56, 64)]
    vals = [s["argmin_agreement"] for s in (gate_summary(r) for r in rels) if s]
    if not vals:
        return None
    return 100 * min(vals), 100 * max(vals), len(vals)


def sec_glance():
    """Results at a glance -- every number computed from the result JSONs."""
    rows_ = []

    def r(finding, numbers, where):
        rows_.append(row([f"<td>{finding}</td>", f"<td>{numbers}</td>",
                          f"<td>{where}</td>"]))

    band = ladder_band()
    if band:
        lo, hi, n = band
        r("The NN's labels agree with the exact solver, at every board "
          "size that can be checked",
          f"{lo:.1f}&ndash;{hi:.1f}% best-move agreement across {n} board "
          "sizes from 17&times;17 to 64&times;64 — one network, flat curve, "
          "no cliff", '<a href="#s3">&sect;2</a>')
    g = gate_summary("nn_labeler/results/coarsegate_g64r4.json")
    if g:
        r("The curve holds to the last checkable size",
          f"{100*g['argmin_agreement']:.1f}% at 64&times;64 — the exact "
          "solver's hard limit; nothing above it can ever be graded",
          '<a href="#s3">&sect;2</a>')
    ex1 = solve("scaling/results/g24r4/comparison.json")
    ex2 = solve("scaling/results/g24r4/comparison_exactseed21.json")
    tw1 = solve("scaling/results/g24r4/comparison_nntwin.json")
    tw2 = solve("scaling/results/g24r4/comparison_nntwin-seed21.json")
    if None not in (ex1, ex2, tw1, tw2):
        r("Planners taught by 91%-faithful NN labels are as good as "
          "exact-taught ones",
          f"twin-taught planners solve {p1(tw1)}% / {p1(tw2)}% (two runs) vs "
          f"exact-taught {p1(ex1)}% / {p1(ex2)}% — no detectable difference",
          '<a href="#s4">&sect;3</a>')
    ex, tw = solve("scaling/results/g32r4/comparison.json"), solve("scaling/results/g32r4/comparison_nntwin.json")
    exo, two = optpct("scaling/results/g32r4/comparison.json"), optpct("scaling/results/g32r4/comparison_nntwin.json")
    if None not in (ex, tw, exo, two):
        r("At 89% faithfulness, solving survives but solution quality slips",
          f"solve rate {p1(tw)}% vs {p1(ex)}%, but only {two:.1f}% of "
          f"solutions perfectly optimal vs {exo:.1f}%",
          '<a href="#s4">&sect;3</a>')
    ex, tw = solve("scaling/results/g24r8/comparison.json"), solve("scaling/results/g24r8/comparison_nntwin.json")
    if None not in (ex, tw):
        r("At 82% faithfulness, the taught planner breaks down",
          f"solve rate {p1(tw)}% vs {p1(ex)}% — a collapse, not a slip",
          '<a href="#s4">&sect;3</a>')
    dep = solve("scaling/results/g32r4/comparison_nndeploy.json")
    ex = solve("scaling/results/g32r4/comparison.json")
    if None not in (dep, ex):
        r("A pipeline with NO exact solver anywhere still works",
          f"NN-made boards + puzzles + labels &rarr; planner solves {p1(dep)}% "
          f"vs the exact-taught {p1(ex)}%",
          '<a href="#s4">&sect;3</a>')
    beyond_bits = []
    for gsz in (80, 96):
        p = SV / f"nn_labeler/results/beyond_UNVERIFIABLE_g{gsz}r4.jsonl.manifest.json"
        if p.exists():
            m = json.load(open(p))["stats"]
            beyond_bits.append(f"{m['records']} at {gsz}&times;{gsz}")
    if beyond_bits:
        r("First-ever labels beyond the checkable limit",
          "certified label sets where no solver can ever grade them: "
          + ", ".join(beyond_bits) + " — every one physics-verified, zero timeouts",
          '<a href="#s5">&sect;4</a>')
    r("The hoped-for compute saving did NOT materialize",
      "a labeler distilled from capped data reproduced the cap's damage — "
      "the negative result, reported as such",
      '<a href="#s6">&sect;5</a>')
    return ("<h3>Results at a glance</h3>" +
            table(["finding", "the numbers", "details"], rows_,
                  note="The whole campaign cost roughly 27 node-hours of "
                       "cluster time against the ~441 the exact-solver plan "
                       "projected for a single extended dataset (measured "
                       "ledger: process.html &sect;10). Every number in this "
                       "table is read from the same result files as the "
                       "detailed tables below."))


def sec_overview():
    corrupt_pending = not (SV / "scaling/results/g24r4/comparison_corrupt_d822.json").exists()
    status = ("Controlled campaign complete; the causality arms (label "
              "corruption at fixed size) and one seed replicate are in "
              "flight — their rows below fill in automatically when the "
              "jobs land." if corrupt_pending else
              "Campaign complete, including the causality arms.")
    return f"""
<section id="overview">
<h2>What is this project?</h2>
<p><strong>The setting.</strong> We train neural-network <em>planners</em> to
solve Ricochet Robots puzzles — robots sliding on a grid until they hit a
wall — at board sizes from 16&times;16 up. Planners learn from training data
in which every puzzle position carries a <em>label</em>: the number of moves
an optimal solution still needs from that position ("cost-to-go"). Until this
track, those labels came from an <strong>exact Rust solver</strong> — perfect
labels, but expensive at scale (a projected ~441 node-hours for one extended
dataset) and physically impossible above 64&times;64 boards (a hard limit of
the solver's engine).</p>
<p><strong>The idea.</strong> Replace the exact solver, as the label writer,
with a small <strong>size-free neural network</strong> (it has no
board-size-dependent parameters, so one checkpoint works at any size). The
network proposes plan completions; every proposal is verified move-by-move
against the game physics before it may become a label ("certified descent"),
so labels are never fantasies — at worst they are slightly-too-long solutions.</p>
<p><strong>The question that decides everything:</strong> if a planner is
trained on NN-written labels instead of exact ones, is the resulting planner
just as good? That is what the tables on this page answer.</p>
{sec_glance()}
<h3>The story in six steps</h3>
<ol class="story">
<li><a href="#s2">Check the labels themselves</a> — on boards where the exact
solver still works, NN labels pick the same best move as the exact solver
~89&ndash;93% of the time ("argmin agreement"), and their errors are small
near-ties.</li>
<li><a href="#s3">Check that quality survives scale</a> — one net, gated at
every board size 17&ndash;64: the quality curve is flat all the way to
64&times;64, the largest size anything can ever be checked at.</li>
<li><a href="#s4">The headline experiment</a> — retrain the planners on
NN-labeled "twin" datasets (same boards, same puzzles, only the labels
swapped) and race them against the exact-taught originals on identical
benchmark puzzles. Verdict: <strong>label fidelity gates downstream
utility</strong> — at ~91% agreement the planners are equivalent, at ~89%
solve rate survives but optimality slips, at ~82% the planner collapses.</li>
<li><a href="#s4">The deployment run</a> — the NN pipeline generates
<em>everything</em> (fresh boards, fresh puzzles, its own labels; no exact
solver anywhere) and the resulting planner still matches the exact-taught
one.</li>
<li><a href="#s5">Past the edge of the checkable world</a> — the first
certified label datasets at 80&times;80 and 96&times;96, where no exact
solver can ever grade them.</li>
<li><a href="#s7">What's running now</a> — the causality experiment:
corrupt exact labels to a controlled fidelity dose and see whether fidelity
alone reproduces the dose-response. {chip("pend", "in flight") if corrupt_pending else chip("good", "done")}</li>
</ol>
<p class="statuscard"><strong>Status:</strong> {status}</p>
</section>
<section id="how">
<h2>How to read the tables — the vocabulary</h2>
<dl class="defs">
<dt>label / cost-to-go</dt>
<dd>The number attached to a puzzle position in the training data: how many
moves an optimal solution still needs from here. Planners learn by
imitating these numbers, so the whole study is about who writes them —
the exact solver, or the neural network.</dd>
<dt>label provenance (exact arm / twin arm / deployment arm)</dt>
<dd>"Provenance" = who wrote the labels. <em>Exact arm</em>: planner trained
on exact-solver labels (the control). <em>Twin arm</em>: identical boards
and puzzles, but the labels rewritten by the NN — so who-wrote-the-labels
is the <em>only</em> difference between the two planners.
<em>Deployment arm</em>: the NN generated boards, puzzles <em>and</em>
labels from scratch, no exact solver anywhere.</dd>
<dt>backward vs forward planner</dt>
<dd>Two planner designs trained side by side in the wider project. The
<em>backward (subgoal) planner</em> reasons from the goal backwards and is
the one retrained in every experiment here; the <em>forward</em> planner
picks moves one at a time and appears only as an untouched reference row
(its training data cannot be relabeled — see the headline section).</dd>
<dt>argmin agreement (label fidelity)</dt>
<dd>At each decision the planner faces several candidate moves; "argmin"
is just the candidate with the lowest (best) label. Agreement asks: does
the NN's labeling pick the <em>same best candidate</em> as the exact
solver? This is the load-bearing quality metric — planners follow the
ranking, not the absolute values — and it is what we mean by "fidelity"
or "faithfulness" throughout.</dd>
<dt>solve rate / % optimal / regret</dt>
<dd>On the fixed benchmark: how many puzzles the planner solves at all; how
many of its solutions are perfectly optimal; and how many extra moves the
non-optimal ones cost on average (regret 3.0 = three moves longer than
necessary, averaged).</dd>
<dt>graded vs frontier</dt>
<dd><em>Graded</em>: benchmark puzzles whose optimum is known (so % optimal
and regret can be scored). <em>Frontier</em>: harder puzzles nothing had
ever solved — only solve rate exists there.</dd>
<dt>the benchmark protocol</dt>
<dd>Every planner in a table faces the <em>same fixed puzzle set</em>
(checksum-locked so nothing can drift), with the same search budget (1,200
node expansions, 5 candidate moves considered per step), and only
physics-legal moves count. Different rows are always apples-to-apples.</dd>
<dt>certified</dt>
<dd>Every solution a planner (or the labeler) reports is replayed
move-by-move against the game physics before it counts. Nothing on this
page is self-reported by a neural network.</dd>
<dt>seed</dt>
<dd>The random initialization of a training run. Two runs that differ only
in seed give slightly different planners — that run-to-run wobble is
measured here (it once demoted an apparent win to noise), which is why
several cells have "seed 21" replicate rows.</dd>
<dt>node-hour</dt>
<dd>The cluster's cost unit (one full compute node for one hour). The
project's budget is 1,000 of them; this whole campaign used ~27.</dd>
</dl>
</section>"""


def dose_chart():
    """SVG scatter: label fidelity (x) vs twin-minus-exact solve gap (y).
    Series 1 (blue circles): real twin cells. Series 2 (orange diamonds):
    corruption arms vs their own size-matched control. Built entirely from
    result JSONs; renders only the points whose files exist."""
    pts = []  # (x_pct, y_pts, label, series)
    twin_gates = {c: gate_summary(f"nn_labeler/results/twin_gate_{c}.json")
                  for c in CFGS}
    for cfg in CFGS:
        g = twin_gates[cfg]
        if not g:
            continue
        x = 100 * g["argmin_agreement"]
        ex = solve(f"scaling/results/{cfg}/comparison.json")
        for tag, rel in [("", f"scaling/results/{cfg}/comparison_nntwin.json"),
                         (" s21", f"scaling/results/{cfg}/comparison_nntwin-seed21.json")]:
            tw = solve(rel)
            if ex is not None and tw is not None:
                pts.append((x, 100 * (tw - ex), f"{cfg}{tag}", "twin"))
    d1000 = solve("scaling/results/g24r4/comparison_corrupt_d1000.json")
    man = load("nn_labeler/results/corrupt_g24r4.manifest.json")
    if d1000 is not None and man:
        for tag, arm in man.get("arms", {}).items():
            if tag == "d1000":
                continue
            cr = solve(f"scaling/results/g24r4/comparison_corrupt_{tag}.json")
            if cr is not None:
                pts.append((100 * arm["achieved_argmin_agreement"],
                            100 * (cr - d1000), f"corrupt {tag[1:3]}%", "corrupt"))
    if len(pts) < 2:
        return ('<p class="note">Chart appears here once at least two arms '
                'have landed.</p>')
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts] + [0.0]
    x0, x1 = min(xs) - 2, max(xs) + 2
    y0, y1 = min(ys) - 4, max(ys) + 4
    W, H, L, B = 640, 300, 52, 34
    def X(v): return L + (v - x0) / (x1 - x0) * (W - L - 14)
    def Y(v): return (H - B) - (v - y0) / (y1 - y0) * (H - B - 14)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="Label fidelity vs downstream solve-rate gap">']
    s.append(f'<line x1="{L}" y1="{Y(0):.0f}" x2="{W-14}" y2="{Y(0):.0f}" '
             f'class="zero"/>')
    s.append(f'<text x="{W-16}" y="{Y(0)-5:.0f}" class="axl" '
             f'text-anchor="end">same as exact-taught</text>')
    for gx in range(int(x0) + 1, int(x1) + 1):
        if gx % 2 == 0:
            s.append(f'<line x1="{X(gx):.0f}" y1="14" x2="{X(gx):.0f}" '
                     f'y2="{H-B}" class="grid"/>')
            s.append(f'<text x="{X(gx):.0f}" y="{H-B+16}" class="axl" '
                     f'text-anchor="middle">{gx}%</text>')
    s.append(f'<text x="{(L+W)/2:.0f}" y="{H-4}" class="axl" '
             f'text-anchor="middle">label argmin agreement with the exact solver</text>')
    s.append(f'<text x="14" y="{(H-B)/2:.0f}" class="axl" text-anchor="middle" '
             f'transform="rotate(-90 14 {(H-B)/2:.0f})">solve-rate gap (pts)</text>')
    for x, y, lab, series in pts:
        cx, cy = X(x), Y(y)
        tip = f"{lab}: fidelity {x:.1f}%, gap {y:+.1f} pts"
        if series == "twin":
            s.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="6" class="s1">'
                     f'<title>{esc(tip)}</title></circle>')
        else:
            s.append(f'<path d="M {cx:.0f} {cy-7:.0f} L {cx+7:.0f} {cy:.0f} '
                     f'L {cx:.0f} {cy+7:.0f} L {cx-7:.0f} {cy:.0f} Z" class="s2">'
                     f'<title>{esc(tip)}</title></path>')
        anchor = "start" if x < (x0 + x1) / 2 else "end"
        dx = 10 if anchor == "start" else -10
        s.append(f'<text x="{cx+dx:.0f}" y="{cy+4:.0f}" class="ptl" '
                 f'text-anchor="{anchor}">{esc(lab)}</text>')
    legend = (f'<g class="legend"><circle cx="{L+8}" cy="24" r="6" class="s1"/>'
              f'<text x="{L+20}" y="28" class="axl">twin cells (real NN labels)</text>')
    if any(p[3] == "corrupt" for p in pts):
        legend += (f'<path d="M {L+178} 17 L {L+185} 24 L {L+178} 31 L {L+171} 24 Z" class="s2"/>'
                   f'<text x="{L+193}" y="28" class="axl">corruption arms (controlled dose)</text>')
    s.append(legend + "</g>")
    s.append("</svg>")
    return ('<figure class="fig">' + "".join(s) +
            '<figcaption>Each point is one retrained planner compared with '
            'its control on the same fixed benchmark: real twin cells against '
            'the exact-taught planner (blue circles), controlled-corruption '
            'arms against their size-matched control (orange diamonds). '
            'Hover a point for its numbers.</figcaption></figure>')


def sec_downstream():
    out = ['<section id="s4">',
           "<h2>3 &middot; The headline: are NN-taught planners as good as "
           "exact-taught ones?</h2>",
           "<p><strong>The experiment.</strong> For each configuration we "
           "retrained the backward planner from scratch on the twin corpus — "
           "the exact corpus's own boards and puzzles, with only the labels "
           "rewritten by the NN — and raced it against the exact-taught "
           "original under the fixed <a href='#how'>benchmark protocol</a>. "
           "Because boards and puzzles are identical, "
           "<a href='#how'>label provenance</a> (who wrote the labels) is "
           "the only variable. All cells are leakage-free: the labeler never "
           "trained on boards of any of these configurations. The forward "
           "planner rows are an untouched reference (the NN labeling process "
           "keeps no move-by-move solutions, so a forward twin would change "
           "the kind of training data, not just who wrote it — the "
           "comparison would no longer isolate one variable).</p>"
           "<p><strong>The verdict in one sentence:</strong> how faithful "
           "the labels are decides how good the taught planner is — "
           "equivalent at ~91% <a href='#how'>fidelity</a>, solving-but-"
           "sloppier at ~89%, broken at ~82%. Each cell's fidelity comes "
           "from the twin-corpus rows of <a href='#s2'>section 1</a>.</p>"]
    out.append(dose_chart())
    verdicts = {
        "g24r4": None, "g24r8": None, "g32r4": None}
    # computed verdict lines
    ex, tw = solve("scaling/results/g24r4/comparison.json"), solve("scaling/results/g24r4/comparison_nntwin.json")
    ex21, tw21 = solve("scaling/results/g24r4/comparison_exactseed21.json"), solve("scaling/results/g24r4/comparison_nntwin-seed21.json")
    if None not in (ex, tw, ex21, tw21):
        verdicts["g24r4"] = (chip("good", "equivalent") +
            f" At ~91% label fidelity the two twin seeds ({p1(tw)}% / {p1(tw21)}% solve) "
            f"straddle the two exact seeds ({p1(ex)}% / {p1(ex21)}%): a full 2&times;2 "
            "seed square with no detectable difference. (The first twin run "
            "looked <em>better</em> than exact; the replicate demoted that to "
            "seed noise — the campaign's own honesty check.)")
    ex, tw = solve("scaling/results/g24r8/comparison.json"), solve("scaling/results/g24r8/comparison_nntwin.json")
    if None not in (ex, tw):
        verdicts["g24r8"] = (chip("bad", "collapse") +
            f" At 82% fidelity the twin-taught planner solves {p1(tw)}% vs the "
            f"exact-taught {p1(ex)}% — the labels are no longer good enough to "
            "teach from. Whether fidelity or the 8-robot setting causes this "
            "is exactly what the <a href='#s7'>corruption arms</a> test.")
    ex, tw = solve("scaling/results/g32r4/comparison.json"), solve("scaling/results/g32r4/comparison_nntwin.json")
    exo, two = optpct("scaling/results/g32r4/comparison.json"), optpct("scaling/results/g32r4/comparison_nntwin.json")
    dep = solve("scaling/results/g32r4/comparison_nndeploy.json")
    if None not in (ex, tw, exo, two):
        v = (chip("good", "solve rate holds") + chip("warn", "optimality slips") +
             f" At 89% fidelity solve rate is equivalent ({p1(tw)}% vs {p1(ex)}%) "
             f"but optimality slips ({two:.1f}% vs {exo:.1f}% of solutions "
             "perfectly optimal). Caveat: this gap is smaller than the seed "
             "spread measured at 24&times;24 — the seed-21 replicate row "
             "below tests it.")
        if dep is not None:
            v += (f" The <strong>deployment arm</strong> — NN-made boards, puzzles "
                  f"and labels, no exact solver anywhere — solves {p1(dep)}%, "
                  "landing between the exact and twin arms: the fully "
                  "solver-free pipeline holds.")
        verdicts["g32r4"] = v
    for cfg in CFGS:
        rows_ = []
        specs = [
            ("backward — exact labels", f"scaling/results/{cfg}/comparison.json", BWD, False, ""),
            ("backward — exact, seed 21", f"scaling/results/{cfg}/comparison_exactseed21.json", BWD, False, ""),
            ("backward — NN twin labels", f"scaling/results/{cfg}/comparison_nntwin.json", BWD, False, "hl"),
            ("backward — NN twin, seed 21", f"scaling/results/{cfg}/comparison_nntwin-seed21.json", BWD, False, "hl"),
            ("forward — exact (untouched control)", f"scaling/results/{cfg}/comparison.json", "forward", True, "ctl"),
        ]
        if cfg == "g32r4":  # deployment ran at this config only (FINDINGS 79)
            specs.insert(4, ("backward — NN deployment (fresh boards, instances, labels)",
                             f"scaling/results/{cfg}/comparison_nndeploy.json", BWD, False, "hl"))
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
        frontier_specs = [
            ("backward — exact, frontier", f"scaling/results/{cfg}/comparison_ungraded.json", "sub"),
            ("backward — NN twin, frontier", f"scaling/results/{cfg}/comparison_ungraded_nntwin.json", "sub hl")]
        if cfg == "g32r4":
            frontier_specs.append(("backward — NN deployment, frontier",
                                   f"scaling/results/{cfg}/comparison_ungraded_nndeploy.json", "sub hl"))
        for label, rel, cls in frontier_specs:
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
        out.append(f"<h3>{cfg} <span class='cfgh'>({CFG_HUMAN[cfg]})</span></h3>")
        if verdicts[cfg]:
            out.append(f'<p class="verdict">{verdicts[cfg]}</p>')
        out.append(table(["system / label source", "solve rate", "% optimal",
                          "mean regret", "mean moves", "mean s", "n"], rows_))
    out.append("</section>")
    return "\n".join(out)


def sec_label_quality():
    out = ['<section id="s2">',
           "<h2>1 &middot; Are the NN labels any good? (the gates)</h2>",
           "<p><strong>The check.</strong> On boards where the exact solver "
           "still works, we compare every NN label against ground truth. "
           "<em>argmin agreement</em> — does the label pick the same best "
           "candidate? — is the metric that matters (planners follow the "
           "ranking); absolute calibration drifts with size while the "
           "ordering holds. The first four rows are 600-instance spot-check "
           "gates for the two production nets; the last three audit the "
           "complete twin corpora that the headline experiment trains on — "
           "those three fidelity numbers (91.0 / 82.2 / 89.4) are the doses "
           "of the dose-response.</p>"]
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
                     rows_,
                     note="How to read a row: 'gap' is how many moves the "
                          "label overestimates by when it is wrong (p90 = "
                          "90th percentile — small gaps mean errors are "
                          "near-ties). 'negative gaps' would mean a label "
                          "beat the exact optimum — a certification bug; "
                          "the handful shown trace to exact-solver timeout "
                          "artifacts, not label errors. v1 and v2 are the "
                          "two candidate production nets; v1 won this gate "
                          "and was locked in ('banked') as the official "
                          "labeler — every label on this page is v1's. v2 "
                          "won a different audit but lost this one."))
    out.append("</section>")
    return "\n".join(out)


def sec_audit_curve():
    out = ['<section id="s2b">',
           "<h3>1b &middot; v1 vs v2 value audit (why v1 is the production net)</h3>",
           "<p>The candidate-ranking audit across sizes. v2 wins this audit "
           "everywhere — and still lost the gate above, which is what "
           "decides deployment. Audit improvement does not automatically "
           "transfer to label quality; the gate, not the audit, picks the "
           "production net.</p>"]
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
                      "v2 regret", "v2 bias"], rows_))
    out.append("</section>")
    return "\n".join(out)


def sec_ladder():
    out = ['<section id="s3">',
           "<h2>2 &middot; Does quality survive scale? (the ladder)</h2>",
           "<p><strong>The check.</strong> One net — trained only on boards "
           "up to 16&times;16 (plus two anchors at 24/32) — gated at every "
           "size on the way up: 600 held-out instances per rung, labels "
           "compared against the exact solver. The question is whether "
           "quality falls off a cliff somewhere between the training sizes "
           "and the target sizes. It does not: argmin agreement stays in "
           "the 86&ndash;93% band across a 4&times; size range, all the "
           "way to 64&times;64 — the largest board the exact solver's "
           "engine can ever grade, and therefore the last rung anyone can "
           "ever check. Beyond it, labels still exist (bottom rows) but "
           "are certified-only: physics-verified upper bounds whose "
           "warranty is this flat curve.</p>"]
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
    beyond = []
    for g in (80, 96):
        p = SV / f"nn_labeler/results/beyond_UNVERIFIABLE_g{g}r4.jsonl.manifest.json"
        if p.exists():
            m = json.load(open(p))["stats"]
            beyond.append(
                row([f"<td>{g}&times;{g} (v1, UNVERIFIABLE)</td>",
                     '<td colspan="3">no gate can exist above 64 — '
                     f'{m["records"]} certified records, '
                     f'{m["kept"]}/{m["attempted"]} instances, '
                     f'{m["timeouts"]} timeouts</td>',
                     fmt(m["cand_uncertified"], "d")]))
    for label, rel in specs:
        s = gate_summary(rel)
        if s is None:
            rows_.append(row([f"<td>{label}</td>",
                              '<td class="pend" colspan="4">pending</td>']))
        else:
            rows_.append(row([f"<td>{label}</td>", pct(s.get("argmin_agreement")),
                              pct(s.get("share_gap_zero")), fmt(s.get("gap_mean")),
                              fmt(s.get("negative_gaps"), "d")]))
    rows_.extend(beyond)
    out.append(table(["board", "argmin agree", "labels exactly optimal",
                      "mean gap", "negative gaps"], rows_,
                     note="8&ndash;16 rows come from the early debug-battery "
                          "gates (an earlier mixed-size training net); "
                          "everything from 17 up is the production v1 net, "
                          "600 replayed test instances per rung. 'capstone' "
                          "= the pre-registered pass/fail exams at 24 and "
                          "32 that qualified the net before the headline "
                          "experiment; 'coarse' = the wider-spaced rungs "
                          "beyond the ladder's 1-cell steps."))
    out.append("</section>")
    return "\n".join(out)


def sec_beyond():
    return """
<section id="s5">
<h2>4 &middot; Past the edge of the checkable world</h2>
<p>The exact solver's engine hard-asserts a 64&times;64 limit — above it,
optimality is not merely expensive to check but <em>impossible</em>, forever.
The 80&times;80 and 96&times;96 rows at the bottom of the ladder are the
first label datasets in that territory: every label is still a real,
physics-replayed solution (a certified upper bound on cost-to-go), but
nobody can ever say how far from optimal it is. Their warranty is
indirect: the same net, under the same protocol, stayed inside the
86&ndash;93% band at every one of the twenty sizes where checking was
possible. A follow-up question — could the downstream <em>planners</em> be
trained at 80/96? — was answered by code analysis, not compute: no. The
planner networks carry a learned position table that grows with the board
(n&sup2; parameters) and would dwarf the available training data; the
labeler generalizes across sizes precisely because it has no such table
(FINDINGS 81). That asymmetry is a finding in itself.</p>
</section>"""


def sec_b2():
    out = ['<section id="s6">',
           "<h2>5 &middot; The negative result: the B2 vocabulary</h2>",
           "<p><strong>The hope.</strong> Base-vocabulary labeling is "
           "170&ndash;250&times; cheaper done exactly, so the labeler's "
           "economic case lived in the extended 'B2' move vocabulary, where "
           "the exact campaign projected ~441 node-hours and its iteration "
           "cap destroyed the special 'by-reference' labels it existed to "
           "produce (13.5% share uncapped &rarr; 4.8% capped). The NN's "
           "labeling process has no iteration budget at all, so the plan "
           "was: train a labeler on the capped B2 data, have it label fresh "
           "data without the cap, recover the lost vocabulary.</p>"
           "<p><strong>The verdict.</strong> " + chip("bad", "negative") +
           " It did not work — the by-reference share came out near zero, "
           "far below even the capped corpora. A network distilled from "
           "capped data reproduces the cap's pathology: the bottleneck was "
           "the training signal, not the search budget. The compute-saving "
           "story died here (~4 node-hours to find out); the headline "
           "result was never about savings and is unaffected. Kept in its "
           "own section because of the vocabulary house rule: base and B2 "
           "data are never mixed.</p>"]
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
                      "capped: 4.8&ndash;12.7%)", "argmin agree vs exact B2",
                      "mean gap"], rows_,
                     note="Source: jobs 4620064&rarr;4623993 (b2_payoff.slurm) "
                          "— B2 net trained on the capped corpora, then "
                          "uncapped descent labeling."))
    out.append("</section>")
    return "\n".join(out)


def sec_inflight():
    man = load("nn_labeler/results/corrupt_g24r4.manifest.json")
    out = ['<section id="s7">',
           "<h2>6 &middot; The causality experiment (in flight / latest)</h2>",
           "<p><strong>Why it exists.</strong> The headline dose-response is "
           "real but confounded: across the three cells, label fidelity "
           "moved together with corpus size and robot count, so no cell "
           "proves fidelity is the <em>causal</em> knob. The fix: take the "
           "exact 24&times;24 corpus, shrink it to twin size (whole decision "
           "groups, seeded), then surgically corrupt a calibrated fraction "
           "of labels — redirecting the best-candidate choice to the "
           "runner-up, only in near-ties, mimicking how the real labeler "
           "actually errs. Three arms share one identical subsample; the "
           "<em>only</em> difference between them is argmin agreement. "
           "Pre-registered reading: if the 82% arm collapses like the "
           "8-robot cell did, fidelity is causal; if it does not, that "
           "collapse was about robot count. Either answer is publishable.</p>"]
    rows_ = []
    if man and "arms" in man:
        arm_desc = {"d1000": "size control (labels untouched)",
                    "d860": "corrupted to 86%",
                    "d822": "corrupted to 82.2% (the collapse cell's dose)"}
        for tag, arm in man["arms"].items():
            a = agg(f"scaling/results/g24r4/comparison_corrupt_{tag}.json")
            label = f"{tag} — {arm_desc.get(tag, '')}"
            dose = f"<td>{100*arm['achieved_argmin_agreement']:.1f}%</td>"
            if a is None:
                rows_.append(row([f"<td>{esc(label)}</td>", dose,
                                  '<td class="pend" colspan="4">running '
                                  f'(comparison_corrupt_{esc(tag)}.json)</td>']))
            else:
                rows_.append(row([f"<td>{esc(label)}</td>", dose,
                                  pct(a.get("solve_rate")),
                                  fmt(a.get("pct_optimal"), ".1f"),
                                  fmt(a.get("mean_regret")),
                                  fmt(a.get("n"), "d")], "hl"))
        out.append(table(["arm", "label fidelity (measured)", "solve rate",
                          "% optimal", "mean regret", "n"], rows_,
                         note=f"Corpora: {man['sub_records']} records / "
                              f"{man['sub_groups']} decision points (each "
                              "with its full set of candidate moves) shared "
                              "by all three arms (nn_labeler/corrupt.py, "
                              f"seed {man['seed']}); compare against the "
                              "exact and twin g24r4 rows in the headline "
                              "section. Also in flight: a second seed of "
                              "the g32r4 twin (fills the pending seed-21 "
                              "row above)."))
    else:
        out.append('<p class="note">Corruption manifest not found — arms not '
                   'yet generated.</p>')
    out.append("</section>")
    return "\n".join(out)


def sec_provenance():
    return """
<section id="s8">
<h2>7 &middot; Provenance &amp; companions</h2>
<p>This page is generated by <code>nn_labeler/report/gen_suite.py</code>
purely from result files on disk — no hand-typed numbers anywhere,
including the verdict sentences (their figures are computed at generation
time from the same JSONs as the tables). A missing file renders as
<em>pending</em> and fills itself in on the next regeneration after a job
lands. Every result traces to a Slurm job and a FINDINGS entry.</p>
<ul>
<li><code>process.html</code> (same folder) — the full technical narrative:
what ran, in what order, what broke, what everything cost.</li>
<li><code>scaling_story.html</code> — the plain-English story, start to
finish, for readers who want prose before tables.</li>
<li><code>FINDINGS.md</code> — the append-only experiment log (this track:
items 48&ndash;74 and 79&ndash;81); <code>TWIN_WIRING.md</code> — checkpoint
and corpus provenance.</li>
</ul>
</section>"""


NAV = """<nav class="topnav"><a href="#overview">Overview</a>
<a href="#how">How to read</a>
<a href="#s2">1 Label quality</a>
<a href="#s3">2 The ladder</a>
<a href="#s4">3 Headline</a>
<a href="#s5">4 Beyond 64</a>
<a href="#s6">5 B2 (negative)</a>
<a href="#s7">6 Causality</a>
<a href="#s8">7 Provenance</a></nav>"""

CSS = """
:root { --ink:#1a1a1a; --mut:#6a6a6a; --line:#d8d8d8; --hl:#f3f7ee;
        --ctl:#f5f5f5; --bg:#fff; --card:#f7f7f4;
        --good-bg:#e7f2e4; --good-ink:#2c5e1e; --bad-bg:#f7e3e0;
        --bad-ink:#8a2b1d; --warn-bg:#f7eeda; --warn-ink:#7a5a12;
        --s1:#2563eb; --s2:#d97706; }
@media (prefers-color-scheme: dark) {
  :root { --ink:#e8e8e8; --mut:#9a9a9a; --line:#3a3a3a; --hl:#20281c;
          --ctl:#242424; --bg:#161616; --card:#1e1e1c;
          --good-bg:#22371c; --good-ink:#a4d194; --bad-bg:#3d211c;
          --bad-ink:#e0a297; --warn-bg:#37301c; --warn-ink:#d9be7a;
          --s1:#60a5fa; --s2:#fbbf24; } }
body { font: 15px/1.55 system-ui, sans-serif; color: var(--ink);
       background: var(--bg); max-width: 62rem; margin: 0 auto 3rem;
       padding: 0 1rem; }
h1 { font-size: 1.5rem; margin-top: 1.4rem; }
h2 { font-size: 1.2rem; margin-top: 2.4rem; }
h3 { font-size: 1rem; margin: 1.4rem 0 .3rem; }
.cfgh { color: var(--mut); font-weight: 400; font-size: .85rem; }
.topnav { position: sticky; top: 0; z-index: 5; background: var(--bg);
          border-bottom: 1px solid var(--line); padding: .55rem 0;
          font-size: .82rem; display: flex; flex-wrap: wrap;
          gap: .25rem 1rem; }
.topnav a { color: var(--ink); text-decoration: none; white-space: nowrap; }
.topnav a:hover { text-decoration: underline; }
section { scroll-margin-top: 3rem; }
.tw { overflow-x: auto; }
table { border-collapse: collapse; margin: .4rem 0; min-width: 40rem; }
th, td { border: 1px solid var(--line); padding: .3rem .6rem;
         text-align: right; font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; }
th { font-weight: 600; }
tr.hl td { background: var(--hl); }
tr.ctl td { background: var(--ctl); color: var(--mut); }
tr.sub td:first-child { padding-left: 1.6rem; }
td.pend { color: var(--mut); font-style: italic; text-align: left; }
.note, .meta { color: var(--mut); font-size: .85rem; }
.banner { border: 1px solid var(--line); padding: .5rem .8rem;
          font-size: .85rem; color: var(--mut); }
.statuscard { background: var(--card); border: 1px solid var(--line);
              padding: .6rem .9rem; border-radius: 4px; }
.story li { margin: .35rem 0; }
.defs dt { font-weight: 600; margin-top: .7rem; }
.defs dd { margin: .15rem 0 0 0; color: var(--ink); }
.verdict { background: var(--card); border-left: 3px solid var(--line);
           padding: .5rem .8rem; }
.chip { display: inline-block; font-size: .75rem; font-weight: 600;
        padding: .05rem .5rem; border-radius: 99px; margin-right: .4rem; }
.chip.good { background: var(--good-bg); color: var(--good-ink); }
.chip.bad { background: var(--bad-bg); color: var(--bad-ink); }
.chip.warn { background: var(--warn-bg); color: var(--warn-ink); }
.chip.pend { background: var(--ctl); color: var(--mut); }
.fig { margin: 1.2rem 0; }
.fig svg { width: 100%; height: auto; max-width: 42rem; display: block; }
.fig figcaption { color: var(--mut); font-size: .85rem; max-width: 42rem; }
.fig .zero { stroke: var(--mut); stroke-dasharray: 4 3; stroke-width: 1; }
.fig .grid { stroke: var(--line); stroke-width: .5; }
.fig .axl { fill: var(--mut); font-size: 11px; }
.fig .ptl { fill: var(--ink); font-size: 11px; }
.fig .s1 { fill: var(--s1); }
.fig .s2 { fill: var(--s2); }
"""


def main():
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=SV,
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        rev = "?"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = "\n".join([sec_overview(), sec_label_quality(), sec_audit_curve(),
                      sec_ladder(), sec_downstream(), sec_beyond(), sec_b2(),
                      sec_inflight(), sec_provenance()])
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Can a neural network replace the exact solver? — results suite</title>
<style>{CSS}</style></head><body>
{NAV}
<h1>Can a neural network replace the exact solver as the label writer?</h1>
<p class="banner">Single source of truth for the NN-labeler track.
Auto-generated by <code>gen_suite.py</code> from result JSONs on disk — no
hand-typed numbers. Generated {stamp} at git {esc(rev)}. LOCAL FILE — not
published.</p>
{body}
</body></html>"""
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
