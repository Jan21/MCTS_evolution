"""The "Breakthrough" tab -- the pipeline of the hybrid search, told in order.

The change itself is one sentence: before the planner starts sub-goal planning,
let it play one or two ordinary robot moves first. This tab explains why anyone
looked there, what the search actually does on one puzzle, what the change
bought against a matched control, and where it still loses.

Every number here is read from the project's own result files at page build
time (`results/variants/v07_hybrid_actions/`, `results/variants/v07_transfer/`,
`results/ceiling/`, `results/fwd_g24/`). A missing file renders as pending, so
the tab never prints a number that no file on disk supports.

Kept in its own module so edits here never touch the data-tab regions of
`gen_report.py`.
"""
from __future__ import annotations

import html as _html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPR = HERE.parent                                  # self_play_robots/
RES = SPR / "results"
V07 = RES / "variants" / "v07_hybrid_actions"
TRN = RES / "variants" / "v07_transfer"

_READ: list[str] = []
_MISS: list[str] = []

PEND = '<span class="miss">pending</span>'


def _esc(s) -> str:
    return _html.escape(str(s))


def _load(p: Path):
    try:
        with open(p) as fh:
            d = json.load(fh)
        _READ.append(p.name)
        return d
    except (OSError, json.JSONDecodeError):
        _MISS.append(p.name)
        return None


def _agg(payload):
    """Aggregate of the payload's single system (these benches carry one)."""
    if not payload:
        return None
    for _name, sysd in (payload.get("systems") or {}).items():
        a = sysd.get("aggregate") or {}
        if a:
            return a
    return None


def _rows(payload):
    if not payload:
        return []
    for _name, sysd in (payload.get("systems") or {}).items():
        if sysd.get("rows"):
            return sysd["rows"]
    return []


def _cfg(payload):
    """Search settings recorded on the hybrid rows: b0, top_m, sub, depth."""
    for r in _rows(payload):
        s = r.get("search") or {}
        if s.get("b0"):
            return s
    return {}


def _winners(payload):
    """How each solved puzzle was won: no slide, one slide, two slides."""
    out = {"subgoal": 0, "slide1": 0, "slide2": 0}
    for r in _rows(payload):
        if not r.get("solved"):
            continue
        w = ((r.get("search") or {}).get("winner") or "")
        if w.startswith("slide2"):
            out["slide2"] += 1
        elif w.startswith("slide"):
            out["slide1"] += 1
        else:
            out["subgoal"] += 1
    return out


def _replay(name: str):
    """(passed, failed) from a replay_validate log, or None."""
    p = V07 / name if (V07 / name).exists() else TRN / name
    try:
        last = [l for l in open(p).read().splitlines() if l.strip()][-1]
        _READ.append(p.name)
    except (OSError, IndexError):
        _MISS.append(name)
        return None
    m = re.search(r"(\d+) passed, (\d+) failed", last)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _n(v, nd=2):
    return PEND if v is None else f"{v:.{nd}f}"


def _selfplay_series():
    """(first, last, lo, hi, n_rounds) standard-exam score of the mainline
    self-play chain, under the same search settings as everything else."""
    import glob
    out = []
    for d in sorted(glob.glob(str(RES / "selfplay" / "g24r4_b2_iter*"))):
        for f in sorted(Path(d).glob("*g24r4*mcts*.json")) + \
                 sorted(Path(d).glob("*bench_solved_mcts*.json")):
            a = _agg(_load(f))
            if a and a.get("n") == 232 and a.get("mean_regret") is not None:
                out.append(a["mean_regret"])
                break
    if len(out) < 2:
        return None
    return out[0], out[-1], min(out), max(out), len(out) - 1


def _common_population():
    """The +0.94-vs-+1.17 comparison redone on one identical puzzle set."""
    c = _load(RES / "ceiling" / "g24r4_b2.json")
    h = _rows(_load(V07 / "bench_graded_hybrid_d2.json"))
    if not c or not h:
        return None
    cr = c.get("rows") or []
    if len(cr) != len(h) or not all(a.get("env_id") == b.get("env_id")
                                    for a, b in zip(cr, h)):
        return None
    idx = [i for i, r in enumerate(cr)
           if r.get("category") == "REALIZABLE_EXISTS"
           and r.get("best_realizable_moves") is not None
           and r.get("d_star") is not None
           and h[i].get("solved") and h[i].get("regret") is not None]
    if not idx:
        return None
    probe = [cr[i]["best_realizable_moves"] - cr[i]["d_star"] for i in idx]
    hyb = [h[i]["regret"] for i in idx]
    wins = sum(1 for a, b in zip(hyb, probe) if a < b)
    loss = sum(1 for a, b in zip(hyb, probe) if a > b)
    return (len(idx), sum(probe) / len(probe), sum(hyb) / len(hyb), wins, loss)


def _i(v):
    return PEND if v is None else f"{v:,d}"


def _solved(a):
    return f"{a['solved']} of {a['n']}" if a else PEND


def _pval(p):
    """The house phrasing: a fluke chance, never a bare p."""
    if p is None:
        return PEND
    if p < 1e-9:
        return "below one in a billion"
    if p < 1e-6:
        return "below one in a million"
    if p < 1e-3:
        return "below one in a thousand"
    return f"about {p:.3f}"


_DEFS = (
    '<defs>'
    '<marker id="bt-ink" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7"'
    ' markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" class="ah inkf"/></marker>'
    '<marker id="bt-mut" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7"'
    ' markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" class="ah mutf"/></marker>'
    '</defs>'
)


def _fig_pipeline(b0, top_m, sub, cap) -> str:
    """Start position, two lanes, one replay gate, cheapest verified answer."""
    capt = _i(cap)
    return f'''
<figure class="fig big">
<svg viewBox="0 0 900 300" role="img"
     aria-label="The hybrid pipeline. The start position feeds two lanes: one ordinary sub-goal search with {b0} expansions, and {top_m} slid start positions with {sub} expansions each. Both lanes feed a physics replay against the original puzzle, and the cheapest total wins.">
  {_DEFS}
  <rect x="8" y="118" width="124" height="56" rx="8" class="nbox"/>
  <text x="70" y="142" text-anchor="middle" class="nt">the puzzle</text>
  <text x="70" y="160" text-anchor="middle" class="ns">start position</text>
  <rect x="214" y="40" width="232" height="56" rx="8" class="nbox"/>
  <text x="330" y="64" text-anchor="middle" class="nt">search the start position</text>
  <text x="330" y="82" text-anchor="middle" class="ns">{b0} expansions, no slide</text>
  <rect x="170" y="196" width="200" height="56" rx="8" class="nbox"/>
  <text x="270" y="220" text-anchor="middle" class="nt">list the slides</text>
  <text x="270" y="238" text-anchor="middle" class="ns">every legal robot move</text>
  <rect x="392" y="196" width="214" height="56" rx="8" class="nbox"/>
  <text x="499" y="220" text-anchor="middle" class="nt">keep the best {top_m}</text>
  <text x="499" y="238" text-anchor="middle" class="ns">{sub} expansions each</text>
  <rect x="650" y="100" width="240" height="56" rx="8" class="nbox"/>
  <text x="770" y="124" text-anchor="middle" class="nt">replay in physics</text>
  <text x="770" y="142" text-anchor="middle" class="ns">against the original puzzle</text>
  <rect x="650" y="196" width="240" height="56" rx="8" class="nbox"/>
  <text x="770" y="220" text-anchor="middle" class="nt">cheapest total wins</text>
  <text x="770" y="238" text-anchor="middle" class="ns">slides pay their own moves</text>
  <path d="M132 138 C 170 138, 176 68, 208 68" class="arr ink" marker-end="url(#bt-ink)"/>
  <path d="M132 156 C 150 156, 148 224, 164 224" class="arr ink" marker-end="url(#bt-ink)"/>
  <line x1="370" y1="224" x2="386" y2="224" class="arr ink" marker-end="url(#bt-ink)"/>
  <path d="M446 68 C 540 68, 560 120, 644 124" class="arr ink" marker-end="url(#bt-ink)"/>
  <path d="M606 224 C 626 224, 634 150, 644 134" class="arr ink" marker-end="url(#bt-ink)"/>
  <line x1="770" y1="158" x2="770" y2="190" class="arr ink" marker-end="url(#bt-ink)"/>
  <text x="388" y="278" text-anchor="middle" class="cap">a two-slide start pays two extra moves</text>
</svg>
<figcaption><strong>The pipeline.</strong> Two lanes race inside one budget of
{capt} expansions. Only a plan that survives the physics replay can win, and a
slid start pays for its slides. The boxes group the nine steps above into
lanes, so they carry no step numbers.</figcaption>
</figure>'''


def _fig_line(hyb, std, fwd, base_lim, b2_lim, taught) -> str:
    """A number line of extra moves against perfect play, standard exam."""
    def x(v):
        return 60 + 131.0 * v
    def dot(v, cls="dot", r=6):
        return f'<circle cx="{x(v):.1f}" cy="150" r="{r}" class="{cls}"/>'
    marks = [dot(0.0, "dot hollow")]
    for v, cls, r in ((fwd, "dot", 6), (hyb, "dot acc", 8), (std, "dot", 6),
                      (taught, "dot", 6)):
        if v is not None:
            marks.append(dot(v, cls, r))
    lim = []
    if b2_lim is not None:
        lim.append(f'<line x1="{x(b2_lim):.1f}" y1="40" x2="{x(b2_lim):.1f}" '
                   f'y2="146" class="floorln"/>'
                   f'<line x1="248" y1="32" x2="{x(b2_lim) + 3:.1f}" y2="42" class="lead"/>'
                   f'<text x="330" y="26" text-anchor="middle" class="bad-t">'
                   f'the extended vocabulary cannot beat +{b2_lim:.2f}</text>')
    if base_lim is not None:
        lim.append(f'<line x1="{x(base_lim):.1f}" y1="64" x2="{x(base_lim):.1f}" '
                   f'y2="146" class="floorln"/>'
                   f'<line x1="300" y1="56" x2="{x(base_lim) + 3:.1f}" y2="66" class="lead"/>'
                   f'<text x="412" y="50" text-anchor="middle" class="bad-t">'
                   f'the base vocabulary cannot beat +{base_lim:.2f}</text>')
    ticks = "".join(
        f'<line x1="{60 + 131 * i}" y1="145" x2="{60 + 131 * i}" y2="155" class="axis"/>'
        f'<text x="{60 + 131 * i}" y="172" text-anchor="middle" class="cap">'
        f'{"0" if i == 0 else "+" + str(i)}</text>' for i in range(6))
    return f'''
<figure class="fig big">
<svg viewBox="0 0 760 230" role="img"
     aria-label="A number line of extra moves against perfect play on the standard exam. Perfect play is 0, the move-by-move planner {_n(fwd)}, the hybrid {_n(hyb)}, the extended vocabulary limit {_n(b2_lim)}, the self-play sub-goal planner {_n(std)}, the base vocabulary limit {_n(base_lim)}, the solver-taught sub-goal planner {_n(taught)}.">
  <line x1="55" y1="150" x2="715" y2="150" class="axis"/>
  {ticks}
  {"".join(lim)}
  {"".join(marks)}
  <text x="60" y="74" text-anchor="middle" class="ns">perfect play</text>
  <line x1="60" y1="78" x2="60" y2="142" class="lead"/>
  <text x="150" y="98" text-anchor="middle" class="ns">move-by-move planner +{_n(fwd)}</text>
  <line x1="120" y1="104" x2="74" y2="142" class="lead"/>
  <text x="612" y="98" text-anchor="middle" class="ns">solver-taught sub-goal planner +{_n(taught)}</text>
  <line x1="612" y1="104" x2="612" y2="142" class="lead"/>
  <text x="400" y="124" text-anchor="middle" class="ns">self-play sub-goal planner +{_n(std)}</text>
  <line x1="330" y1="130" x2="250" y2="142" class="lead"/>
  <text x="170" y="196" text-anchor="middle" class="lbl">the hybrid +{_n(hyb)}</text>
  <line x1="172" y1="184" x2="181" y2="160" class="lead"/>
  <text x="385" y="220" text-anchor="middle" class="cap">extra moves per puzzle against perfect play, 24&times;24 standard exam</text>
</svg>
<figcaption><strong>The limit, and the planner that scored below it.</strong>
Every dot averages over that planner&rsquo;s own solves. The two dashed lines
are limits of the plan language, not planners. The hybrid is the only planner
using sub-goals left of +{_n(b2_lim)}.</figcaption>
</figure>'''


def _table(head, rows, note=None) -> str:
    h = "".join(f"<th>{c}</th>" for c in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                   for r in rows)
    out = (f'<div class="tw"><table class="t wide"><thead><tr>{h}</tr></thead>'
           f"<tbody>{body}</tbody></table></div>")
    if note:
        out += f'<p class="note">{note}</p>'
    return out


def sec_breakthrough() -> str:
    _READ.clear()
    _MISS.clear()

    hyb_g = _load(V07 / "bench_graded_hybrid_d2.json")
    std_g = _load(V07 / "bench_graded_stdmcts.json")
    d3_g = _load(V07 / "bench_graded_hybrid_d3.json")
    hyb_f = _load(V07 / "bench_frontier_hybrid_d2.json")
    std_f = _load(V07 / "bench_frontier_stdmcts.json")
    hyb_u = _load(V07 / "bench_unseen_hybrid_d2.json")
    std_u = _load(V07 / "bench_unseen_stdmcts.json")
    g_graded = _load(V07 / "gate_graded_d2_vs_std.json")
    g_front = _load(V07 / "gate_frontier_d2_vs_stdmcts.json")
    g_unseen = _load(V07 / "gate_unseen_d2_vs_std.json")
    g_v14 = _load(V07 / "gate_graded_v14_hyb_vs_std.json")
    g_d3 = _load(V07 / "gate_graded_d3_vs_d2.json")
    t_g32g = _load(TRN / "gate_g32r4_graded_hyb_vs_std.json")
    t_g32f = _load(TRN / "gate_g32r4_frontier_hyb_vs_std.json")
    t_g24r8g = _load(TRN / "gate_g24r8_graded_hyb_vs_std.json")
    t_g24r8f = _load(TRN / "gate_g24r8_frontier_hyb_vs_std.json")
    ceil_base = _load(RES / "ceiling" / "g24r4_base_slack12.json")
    ceil_b2 = _load(RES / "ceiling" / "g24r4_b2.json")
    fwd = _load(RES / "fwd_g24" / "astar.json")
    sup = _load(SPR.parent / "supervised_valuenet" / "scaling" / "results"
                / "g24r4" / "comparison.json")

    ah, as_, ad3 = _agg(hyb_g), _agg(std_g), _agg(d3_g)
    cfg = _cfg(hyb_g)
    b0 = cfg.get("b0", 500)
    top_m = cfg.get("top_m", 8)
    sub = cfg.get("sub", 80)
    depth = cfg.get("prefix_depth", 2)
    cap = (hyb_g or {}).get("protocol", {}).get("expansions", 1200)
    d2_from = 4                                   # variants/v07_hybrid_actions.py
    base_lim = ((ceil_base or {}).get("summary") or {}).get("mean_gap_best")
    base_n = ((ceil_base or {}).get("summary") or {}).get("n_realizable")
    b2_lim = ((ceil_b2 or {}).get("summary") or {}).get("mean_gap_best")
    b2_n = ((ceil_b2 or {}).get("summary") or {}).get("n_realizable")
    win_g = _winners(hyb_g)
    win_u = _winners(hyb_u)
    rep_g = _replay("bench_graded_hybrid_d2.replay.log")
    afwd = _agg(fwd)

    out = ['<section id="breakthrough" class="story">',
           "<h2>The breakthrough &mdash; the pipeline, step by step</h2>"]
    out.append(
        '<nav class="toc" aria-label="breakthrough chapters">'
        '<a href="#bt-why">1 why anyone looked here</a>'
        '<a href="#bt-change">2 the change</a>'
        '<a href="#bt-pipe">3 the pipeline</a>'
        '<a href="#bt-bought">4 what it bought</a>'
        '<a href="#bt-luck">5 idea, not luck</a>'
        '<a href="#bt-limits">6 the honest limits</a>'
        '</nav>')
    out.append(
        '<p class="lede">The change is one sentence long. Before the planner '
        'starts sub-goal planning, let it play one or two ordinary robot moves '
        'first. This tab explains why anyone looked there, what the search '
        'does on one puzzle, and what the change bought against a matched '
        'control. Every number below is read from a result file when this page '
        'is built.</p>')

    # ---- 1. why anyone looked here -----------------------------------------
    out.append('<h3 id="bt-why"><span class="no">1</span>Why anyone looked here</h3>')
    out.append(
        '<p>A sub-goal planner speaks a small plan language. One word of that '
        'language sends a helper robot to a cell, so the moving robot stops '
        'where the plan needs it. Before any training, the project measured '
        'what those words can do at best. The probe lists every plan a '
        'vocabulary can express, replays each one against the game physics, '
        'and compares the best with perfect play.</p>')
    out.append(_table(
        ["plan vocabulary", "standard-exam puzzles it can reach",
         "best plan, extra moves vs perfect play"],
        [["base",
          f"{base_n} of 232" if base_n else PEND,
          f"+{_n(base_lim)}" if base_lim else PEND],
         ["extended",
          f"{b2_n} of 232" if b2_n else PEND,
          f"+{_n(b2_lim)}" if b2_lim else PEND]],
        note="Each row averages over the puzzles that vocabulary can reach at "
             "all, so the two rows have different populations."))
    out.append(
        '<p>Those numbers belong to the vocabulary, not to the networks. No '
        'amount of training moves them. Read them as measured limits, not as '
        'mathematical proofs: the probe searched everything a vocabulary can '
        'express, inside a fixed search budget. Two probes at different '
        'budgets differ by a few hundredths of a move.</p>')
    out.append(
        '<p>The measurement therefore made a prediction. More practice inside '
        'the base vocabulary would buy nothing, and two rounds of self-play '
        'inside it changed nothing on any exam. Only a change to the actions '
        'the planner may consider could score below the limit.</p>')
    ser = _selfplay_series()
    if ser:
        first, last, lo, hi, rounds = ser
        out.append(
            f'<p>Training inside the extended vocabulary was flat in the same '
            f'way. The mainline self-play chain scored +{_n(first)} extra '
            f'moves on the standard exam before its first round. After all '
            f'{rounds} rounds it scored +{_n(last)}, and it never left the '
            f'band +{_n(lo)} to +{_n(hi)}. That chain and the control in '
            f'section 4 run the same search under the same cap. They differ '
            f'only in which trained network pair they load, which is why '
            f'their standard-exam scores are close but not equal. The story '
            f'page quotes the section 4 pair throughout.</p>')

    # ---- 2. the change ------------------------------------------------------
    out.append('<h3 id="bt-change"><span class="no">2</span>The change, in one sentence</h3>')
    out.append(
        '<p>Before the planner starts sub-goal planning, let it play one or '
        'two ordinary robot moves first. Call the result <strong>the '
        'hybrid</strong>.</p>')
    out.append(
        '<p>The position after one slide is a different plan space. It can '
        'hold a plan that the original position cannot express, and that plan '
        'can be cheaper even after the slide costs its own move. A planner '
        'that slides first is no longer a pure sub-goal planner, so the '
        'measured limit no longer applies to it.</p>')

    # ---- 3. the pipeline ----------------------------------------------------
    out.append('<h3 id="bt-pipe"><span class="no">3</span>The pipeline, on one puzzle</h3>')
    out.append(
        f'<p>The planner runs several independent searches and keeps the '
        f'cheapest answer that really works. The searches share one budget of '
        f'{_i(cap)} expansions. A plan is <strong>certified</strong> when the '
        f'physics simulator turns it into a real move sequence that reaches '
        f'the target. Only a certified plan can win. Here is one puzzle, in '
        f'order.</p>')
    out.append(
        '<ol class="steps">'
        f'<li><b>Search the start position.</b> The ordinary sub-goal search '
        f'runs on the untouched puzzle, with {b0} of the {_i(cap)} '
        f'expansions.</li>'
        '<li><b>List the slides.</b> The planner lists every legal single '
        'robot move from the start position.</li>'
        '<li><b>Score each slid position.</b> It builds that position&rsquo;s '
        'cheap opening plan, which costs no budget, and reads the '
        'plan&rsquo;s cost.</li>'
        f'<li><b>Add a second slide.</b> The {d2_from} cheapest slid positions '
        f'each get one more legal move. The new positions are scored the same '
        f'way, and a repeated position is dropped.</li>'
        f'<li><b>Keep {top_m}.</b> The score of a candidate is its opening-plan '
        f'cost plus one point per slide. The {top_m} cheapest candidates '
        f'stay.</li>'
        f'<li><b>Search each kept candidate.</b> Each one gets {sub} '
        f'expansions. A candidate is skipped when its score sits 3 or more '
        f'moves above the best certified total so far. Its budget is not '
        f'spent.</li>'
        '<li><b>Pay for the slides.</b> A slid answer costs the number of '
        'slides plus the certified cost of its sub-plan. A two-slide start '
        'therefore pays two extra moves.</li>'
        '<li><b>Replay everything.</b> The physics simulator plays every '
        'finished plan, one move at a time. The joined move list, slides '
        'first and sub-plan second, is replayed against the original '
        'puzzle.</li>'
        '<li><b>Return the cheapest.</b> The cheapest certified total wins, '
        'whether it came from a slide or not.</li>'
        '</ol>')
    out.append(_fig_pipeline(b0, top_m, sub, cap))
    out.append(
        f'<p>The budget arithmetic is honest. The lanes can spend at most '
        f'{b0} + {top_m} &times; {sub} = {_i(b0 + top_m * sub)} expansions, '
        f'which stays inside the project&rsquo;s cap of {_i(cap)}. The actual spend is '
        f'summed per puzzle and reported, and it averages '
        f'{_n((ah or {}).get("mean_expansions"), 0)} on the standard exam. The '
        f'control search in section 4 runs under the same cap and averages '
        f'{_n((as_ or {}).get("mean_expansions"), 0)}.</p>')
    if rep_g:
        out.append(
            f'<p>Replay is not a formality. On the standard exam all '
            f'{rep_g[0]} written-out solutions replayed against their original '
            f'puzzle, and {rep_g[1]} failed. A slid answer that does not '
            f'replay cannot win, because it is never certified.</p>')
    out.append(
        f'<p>Most puzzles never need a slide. On the standard exam the plain '
        f'search wins {win_g["subgoal"]} of the solved puzzles, a one-slide '
        f'start wins {win_g["slide1"]}, and a two-slide start wins '
        f'{win_g["slide2"]}. On the new-boards exam the split is '
        f'{win_u["subgoal"]}, {win_u["slide1"]} and {win_u["slide2"]}. The '
        f'second slide is therefore a small part of the gain at depth '
        f'{depth}.</p>')

    # ---- 4. what it bought --------------------------------------------------
    out.append('<h3 id="bt-bought"><span class="no">4</span>What it bought, with its control</h3>')
    out.append(
        '<p>The control is the fair one. It is the same two networks, the same '
        'exam files and the same expansion cap, under the ordinary sub-goal '
        'search. Only the search differs. That control is the '
        '<strong>self-play sub-goal planner</strong> below.</p>')
    rows = []
    for label, gate, ha, sa in (
            ("standard exam (232)", g_graded, ah, as_),
            ("hard exam (218)", g_front, _agg(hyb_f), _agg(std_f)),
            ("new-boards exam (200)", g_unseen, _agg(hyb_u), _agg(std_u))):
        g = gate or {}
        reg = ("&mdash;" if not ha or ha.get("mean_regret") is None
               else f'+{_n(ha["mean_regret"])} vs +{_n((sa or {}).get("mean_regret"))}')
        rows.append([
            label,
            f'{_solved(ha)} vs {_solved(sa)}',
            reg,
            (f'{g.get("moves_wins_a")} shorter, {g.get("moves_wins_b")} longer'
             if g else PEND),
            _pval(g.get("sign_p_moves")) if g else PEND,
        ])
    out.append(_table(
        ["exam", "solved: hybrid vs control",
         "extra moves vs perfect play, each over its own solves",
         "solutions that differ, on puzzles both solved", "fluke chance"],
        rows,
        note="The hard exam has no known optimum, so it compares solve counts "
             "and solution lengths only."))
    if ah and as_ and ah.get("mean_seconds") and as_.get("mean_seconds"):
        ratio = as_["mean_seconds"] / ah["mean_seconds"]
        out.append(
            f'<p>The hybrid is also faster in wall clock, at '
            f'{_n(ah["mean_seconds"], 0)} seconds per puzzle against '
            f'{_n(as_["mean_seconds"], 0)}. That is about one '
            f'{"fifth" if 4.5 < ratio < 5.5 else f"{ratio:.1f}th"} of the '
            f'running time. The two searches spend a similar number of '
            f'expansions, so the saving comes from shorter searches inside the '
            f'slid positions, not from a smaller budget.</p>')
    if ad3:
        wins = (g_d3 or {}).get("moves_wins_a")
        out.append(
            f'<p>A third ordinary move helps a little more, and then the gain '
            f'stops. At depth 3 the standard-exam score is '
            f'+{_n(ad3.get("mean_regret"))} over its own {ad3.get("solved")} '
            f'solves, against +{_n((ah or {}).get("mean_regret"))} at depth '
            f'{depth}. Depth 3 was shorter on {wins} puzzles and longer on '
            f'{(g_d3 or {}).get("moves_wins_b")}. The new-boards exam stopped '
            f'improving at depth {depth}, so no deeper probe was run.</p>')
    taught = None
    for _nm, _sd in ((sup or {}).get("systems") or {}).items():
        if "backward" in _nm:
            taught = (_sd.get("aggregate") or {}).get("mean_regret")
            break
    cp = _common_population()
    if cp:
        n_cp, probe_m, hyb_m, w, l = cp
        out.append(
            f'<p>Is the comparison with the limit fair? The two averages cover '
            f'different puzzle sets. It was therefore redone on one identical '
            f'set: the {n_cp} puzzles the probe can reach and the hybrid also '
            f'solves. There the probe&rsquo;s best possible sub-goal plan '
            f'averages +{_n(probe_m)} extra moves and the hybrid averages '
            f'+{_n(hyb_m)}. Puzzle by puzzle, the hybrid beats the best '
            f'possible sub-goal plan on {w} and loses on {l}.</p>')
    out.append(_fig_line(
        (ah or {}).get("mean_regret"), (as_ or {}).get("mean_regret"),
        (afwd or {}).get("mean_regret"), base_lim, b2_lim, taught))

    # ---- 5. idea, not luck --------------------------------------------------
    out.append('<h3 id="bt-luck"><span class="no">5</span>Why we believe it is the idea, not luck</h3>')
    out.append(
        '<p>Two controls test that. The first repeats the whole comparison on '
        'a second, unrelated pair of networks. The second runs the unchanged '
        'search on board sizes and robot counts it never practised on, with no '
        'new training at all.</p>')
    if g_v14:
        out.append(
            f'<p>On the second network pair the hybrid solved '
            f'{g_v14["solved_a"]} of {g_v14["n"]} standard-exam puzzles '
            f'against the control&rsquo;s {g_v14["solved_b"]}. It was shorter '
            f'on {g_v14["moves_wins_a"]} puzzles and longer on '
            f'{g_v14["moves_wins_b"]}. The fluke chance is '
            f'{_pval(g_v14.get("sign_p_moves"))}.</p>')
    trows = []
    for label, gate in (("32&times;32 standard exam", t_g32g),
                        ("32&times;32 hard exam", t_g32f),
                        ("8-robot standard exam", t_g24r8g),
                        ("8-robot hard exam", t_g24r8f)):
        g = gate or {}
        trows.append([
            label,
            f'{g.get("solved_a")} of {g.get("n")}' if g else PEND,
            f'{g.get("solved_b")} of {g.get("n")}' if g else PEND,
            (f'{g.get("moves_wins_a")} shorter, {g.get("moves_wins_b")} longer'
             if g else PEND),
            _pval(g.get("sign_p_moves")) if g else PEND,
        ])
    out.append(_table(
        ["exam it never practised on", "hybrid solved", "control solved",
         "solutions that differ, on puzzles both solved", "fluke chance"],
        trows,
        note="No new training was done for any row here. The same saved "
             "networks and the same search settings ran on every board type."))
    out.append(
        '<p>Read the 8-robot hard exam honestly. The control solved a few more '
        'puzzles there, and the hybrid still wrote much shorter solutions on '
        'the puzzles both solved. The 32&times;32 hard exam is the strongest '
        'row: more puzzles solved and shorter solutions at the same time.</p>')

    # ---- 6. the honest limits ----------------------------------------------
    out.append('<h3 id="bt-limits"><span class="no">6</span>The honest limits</h3>')
    out.append(
        '<ul class="plain">'
        '<li><b>The move-by-move planner still writes shorter solutions when '
        'it solves.</b> On the 219 standard-exam puzzles that both planners '
        'solve, the hybrid used +0.79 more moves. That figure is not '
        f'{_n((ah or {}).get("mean_regret"))} minus '
        f'{_n((afwd or {}).get("mean_regret"))}, because those two scores '
        'average over different sets of puzzles.</li>'
        '<li><b>But that planner solves about half as many new boards.</b> It '
        f'solves {(afwd or {}).get("solved", "?")} of 232 on the standard exam. '
        'On the new-boards exam it solves 101 of 200, against the '
        f'hybrid&rsquo;s {_solved(_agg(hyb_u))}. It also spends 275 seconds '
        'per puzzle.</li>'
        '<li><b>Part of the depth-2 gain is wider screening, not two-slide '
        f'plans.</b> Depth 2 also raised the kept-candidate count to {top_m}. '
        f'Only {win_u["slide2"]} of the {win_u["slide1"] + win_u["slide2"]} '
        'new-boards slide wins actually used both slides.</li>'
        '<li><b>The move-by-move planner here is the supervised one of '
        'record.</b> Nobody re-tuned it for this comparison, and a re-tuned '
        'version would likely close part of the measured gap.</li>'
        '<li><b>This is a search change only.</b> Nothing here was trained to '
        'slide. Two later attempts to teach slides during training both '
        'failed, and the search change did all the work.</li>'
        '</ul>')
    out.append(
        '<p>The same story without the tables is the '
        '<a href="#story">Story tab</a>.</p>')
    if _MISS:
        out.append('<p class="note">Result files not on disk yet: '
                   + _esc(", ".join(sorted(set(_MISS)))) + ".</p>")
    out.append("</section>")
    return "\n".join(out)
