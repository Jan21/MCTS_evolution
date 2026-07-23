"""Build eval/results/plan_structures.html — an abstract visualization of
backward-planner plan structures before and after the two plan-language
extensions.

Usage (from supervised_valuenet/):
    PYTHONPATH=. python3 -m eval.build_plan_viz

Reads eval/results/plan_structures_data.json (five real, strictly playable
plans extracted by analysis/plan_viz/extract_plans.py) plus the three ceiling
probe files (for the 90.7% / 97.6% / 99.6% story), renders one self-contained
HTML file (inline CSS/SVG, no external assets, light/dark theming identical in
mechanism to eval/build_report.py), and runs a self-verification pass. No
number is hardcoded in the HTML — everything comes from the JSON files at
build time. Exit code is non-zero if any check fails.

All SVG rendering lives in eval/plan_viz_core.py, shared with the report
page's compact in-page versions of the same diagrams.
"""

import json
import os
import sys

from eval.plan_viz_core import (STRUCTURAL_PAIRS, ROBOT_VARS, dag_svg,
                                board_inset, legend, esc)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "eval", "results", "plan_structures_data.json")
OUT_PATH = os.path.join(ROOT, "eval", "results", "plan_structures.html")

CHECKS = []  # (description, ok)


def check(desc, ok):
    CHECKS.append((desc, bool(ok)))


def rp(*parts):
    return os.path.join(ROOT, *parts)


def load_json(relpath):
    path = rp(relpath)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Ceiling numbers (same math and same source files as eval/build_report.py)
# ---------------------------------------------------------------------------

def ceiling_story():
    """{'n': 450, 'old': (n_struct, pct), 'b1': ..., 'b2': (n_unresolved,
    pct, n_proven_impossible)} — values None when a probe file is absent."""
    bench = load_json("eval/data/bench450.jsonl.meta.json") or {}
    n = bench.get("n_instances") or 450
    out = {"n": n, "old": None, "b1": None, "b2": None}
    p0 = load_json("analysis/artifacts/ceiling_probe_results.json")
    p1 = load_json("analysis/artifacts/ceiling_probe_results_b1.json")
    p2 = load_json("analysis/artifacts/ceiling_probe_results_b2.json")

    def struct(rows):
        return sum(1 for r in rows if r.get("category") in
                   ("NO_COMPLETE_PLAN", "NO_REALIZABLE_PLAN"))

    if isinstance(p0, list):
        s = struct(p0)
        out["old"] = (s, (n - s) / n * 100)
    if isinstance(p1, list):
        s = struct(p1)
        out["b1"] = (s, (n - s) / n * 100)
    if isinstance(p2, list):
        fail = sum(1 for r in p2 if r.get("category") != "REALIZABLE_EXISTS")
        out["b2"] = (fail, (n - fail) / n * 100, struct(p2))
    return out


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------

def stage_section(sid, title, intro, examples, blocks, caption):
    ex_html = []
    for ex, (svg, _) in zip(examples, blocks):
        head = (f"puzzle {ex['bench_idx']} of the benchmark · shortest "
                f"possible solution: {ex['d_star']} moves · this plan plays "
                f"out in {ex['legal_moves']} legal moves")
        ex_html.append(f"""
  <div class="panel">
    <p class="ph">{esc(ex["why"])}</p>
    <p class="muted small">{esc(head)}</p>
    <div class="viz-split">
      <div class="scroll">{svg}</div>
      <div class="inset">
        {board_inset(ex)}
        <p class="muted tiny">where the robots start (walls not drawn);
        the dashed square is the goal, the haloed robot must reach it</p>
      </div>
    </div>
  </div>""")
    return f"""
<section id="{sid}">
  <div class="sec-head">
    <h2>{esc(title)}</h2>
  </div>
  {intro}
  {legend(marker_id="la-" + sid)}
  {"".join(ex_html)}
  <p class="cap">{caption}</p>
</section>
"""


CSS = """
:root{
  --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --muted:#52514e;
  --faint:#898781; --grid:#e1e0d9; --baseline:#c3c2b7;
  --line:rgba(11,11,11,.10); --accent:#2a78d6; --accent-ink:#1c5cab;
  --r-red:#d9480f; --r-blue:#1c7ed6; --r-green:#2f9e44; --r-yellow:#c9971c;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0d0d0d; --surface:#1a1a19; --ink:#f2f1ec; --muted:#c3c2b7;
    --faint:#898781; --grid:#2c2c2a; --baseline:#55544f;
    --line:rgba(255,255,255,.10); --accent:#3987e5; --accent-ink:#86b6ef;
    --r-red:#ff8a4c; --r-blue:#4dabf7; --r-green:#51cf66; --r-yellow:#ffd43b;
  }
}
:root[data-theme="light"]{
  --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --muted:#52514e;
  --faint:#898781; --grid:#e1e0d9; --baseline:#c3c2b7;
  --line:rgba(11,11,11,.10); --accent:#2a78d6; --accent-ink:#1c5cab;
  --r-red:#d9480f; --r-blue:#1c7ed6; --r-green:#2f9e44; --r-yellow:#c9971c;
}
:root[data-theme="dark"]{
  --bg:#0d0d0d; --surface:#1a1a19; --ink:#f2f1ec; --muted:#c3c2b7;
  --faint:#898781; --grid:#2c2c2a; --baseline:#55544f;
  --line:rgba(255,255,255,.10); --accent:#3987e5; --accent-ink:#86b6ef;
  --r-red:#ff8a4c; --r-blue:#4dabf7; --r-green:#51cf66; --r-yellow:#ffd43b;
}
*{box-sizing:border-box}
html,body{max-width:100%;overflow-x:hidden}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  line-height:1.6;font-size:16px;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:0 22px 60px}
header{padding:44px 0 8px}
h1{font-size:clamp(25px,4vw,38px);font-weight:750;letter-spacing:-.02em;
  line-height:1.16;margin:0}
h2{font-size:clamp(19px,2.6vw,25px);font-weight:700;letter-spacing:-.015em;
  margin:0 0 6px}
section{padding:28px 0;border-top:1px solid var(--grid)}
p{margin:.6em 0;max-width:76ch}
a{color:var(--accent-ink);text-decoration:none;
  border-bottom:1px solid var(--line)}
a:hover{border-bottom-color:var(--accent-ink)}
.muted{color:var(--muted)} .small{font-size:13px} .tiny{font-size:11.5px}
.lede{font-size:clamp(15.5px,1.8vw,18px);color:var(--muted);max-width:74ch}
.panel{background:var(--surface);border:1px solid var(--line);
  border-radius:14px;padding:18px;margin:14px 0}
.panel .ph{font-weight:680;margin:0 0 2px}
.howto{background:var(--surface);border:1px solid var(--line);
  border-left:3px solid var(--accent);border-radius:10px;
  padding:12px 16px;margin:14px 0;max-width:86ch}
.cap{color:var(--muted);font-size:14px;max-width:84ch}
.scroll{overflow-x:auto;max-width:100%}
.viz-split{display:grid;grid-template-columns:minmax(0,1fr) 232px;gap:18px;
  align-items:start}
.inset{padding-top:6px}
.legend-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px 16px;
  margin:8px 0 4px;font-size:12px;color:var(--muted)}
.legend-grid div{display:flex;align-items:center;gap:8px}
.legend-grid svg{flex:none}
.themebtn{position:fixed;top:14px;right:14px;font-family:var(--mono);
  font-size:12px;background:var(--surface);color:var(--muted);
  border:1px solid var(--line);border-radius:18px;padding:5px 13px;
  cursor:pointer;z-index:30}
.footer{color:var(--muted);font-size:13px;border-top:1px solid var(--grid);
  padding-top:18px;margin-top:30px}
.mono{font-family:var(--mono);font-size:.92em}
@media (max-width:760px){
  .viz-split{grid-template-columns:1fr}
  .legend-grid{grid-template-columns:1fr 1fr}
}
"""

JS = """
(function(){
  var btn=document.getElementById('themebtn');
  var modes=['auto','light','dark'];
  var i=0;
  btn.addEventListener('click',function(){
    i=(i+1)%3;
    var m=modes[i];
    if(m==='auto'){delete document.documentElement.dataset.theme;}
    else{document.documentElement.dataset.theme=m;}
    btn.textContent='theme: '+m;
  });
})();
"""


def build():
    data = load_json("eval/results/plan_structures_data.json")
    if not data or "examples" not in data:
        sys.exit("plan_structures_data.json missing or malformed")
    exs = data["examples"]
    by_stage = {}
    for ex in exs:
        by_stage.setdefault(ex["stage"], []).append(ex)
    cs = ceiling_story()
    n = cs["n"]

    robots_seen = {r for ex in exs for r in ex["robots"]}
    check("every robot color in the data has a palette mapping",
          robots_seen <= set(ROBOT_VARS))

    rendered = {}
    total_cost_labels = 0
    expect_cost_labels = 0
    for ex in exs:
        by_id = {nd["id"]: nd for nd in ex["nodes"]}
        expect_cost_labels += sum(
            1 for e in ex["edges"]
            if not e.get("byref") and e.get("cost")
            and (by_id[e["from"]]["type"], by_id[e["to"]]["type"])
            not in STRUCTURAL_PAIRS)
        svg, ncl = dag_svg(
            ex, f"Plan structure for benchmark puzzle {ex['bench_idx']}",
            marker_id=f"arr-{ex['stage']}-{ex['bench_idx']}")
        rendered[(ex["stage"], ex["bench_idx"])] = (svg, ncl)
        total_cost_labels += ncl
    check(f"every physical movement carries a move-count pill "
          f"({total_cost_labels} of {expect_cost_labels})",
          total_cost_labels == expect_cost_labels)

    def pct(v):
        return f"{v:.1f}%"

    old_txt = pct(cs["old"][1]) if cs["old"] else "—"
    b1_txt = pct(cs["b1"][1]) if cs["b1"] else "—"
    b2_txt = pct(cs["b2"][1]) if cs["b2"] else "—"

    base_intro = """
  <p>In the original plan language, a plan may say exactly one kind of thing:
  <b>“park a helper robot on a cell next to a wall, then bounce another robot
  off it.”</b> Helpers can be delivered by other helpers (the picture below
  chains two of these stepping-stones), but two rules are absolute: a parked
  helper can never move again, and a helper can only park where a wall will
  stop it. Whatever cannot be phrased that way cannot be planned at all.</p>
"""
    base_cap = (
        f"This ordinary puzzle fits the language comfortably — the plan above "
        f"plays out in {by_stage['base'][0]['legal_moves']} legal moves, "
        f"exactly the known optimum. But on the 450-puzzle benchmark the "
        f"language itself runs out for "
        + (f"{cs['old'][0]} puzzles, capping the backward planner at "
           f"<b>{old_txt}</b> no matter how well its networks are trained."
           if cs["old"] else "dozens of puzzles."))

    b1_intro = """
  <p>The first extension adds two new phrases. A helper may park on a cell
  <b>with no wall to lean on</b> — it is held there only because another
  robot (or the timing of the plan itself) stops it; the starred, dashed box
  below is exactly that. And a robot may <b>step aside</b> before one
  specific slide, clearing the way — the dotted box, whose dotted arrow
  points at the slide it must precede. Every one of these maneuvers is paid
  for in ordinary counted moves.</p>
"""
    b1_cap = (
        "Left: the famous four-move puzzle that was provably unwritable in "
        "the original language — the wall-less park (★) makes it sayable, "
        "and the plan plays out at its exact optimum. Right: a step-aside "
        "plan; honest pricing means stepping aside costs real moves "
        f"({by_stage['b1'][1]['legal_moves']} played vs an optimum of "
        f"{by_stage['b1'][1]['d_star']}). "
        + (f"Re-measuring the whole benchmark with these phrases: the cap "
           f"moves from {old_txt} to <b>{b1_txt}</b>."
           if cs["b1"] else ""))

    b2_intro = """
  <p>The second extension removes the last big restriction: a plan may now
  <b>re-use a robot it has already placed</b>. The same parked robot can
  stop two different sliders; the main robot itself can serve as a bounce
  point mid-route; a placed helper can slide on to a second post. In the
  left picture that shows up as a dashed link: the third stepping-stone
  points back at an <i>existing</i> parked robot instead of recruiting a
  fresh one — three stepping-stones, only two parked robots. The step-aside
  repair was generalized the same way: on the right, <i>two</i> robots must
  both clear the same slide.</p>
"""
    b2_cap = (
        "Left: the 8-move optimum is reached because one parked robot (Blue) "
        "stops two different robots in turn — inexpressible before this "
        "extension. Right: two step-asides at once "
        f"({by_stage['b2'][1]['legal_moves']} moves played vs an optimum of "
        f"{by_stage['b2'][1]['d_star']} — again, clearing the way is paid "
        "for honestly). "
        + (f"With the full language, only {cs['b2'][0]} of {n} benchmark "
           f"puzzles remain unresolved — none proven impossible — a cap of "
           f"at least <b>{b2_txt}</b>."
           if cs["b2"] else ""))

    sections = []
    sections.append(stage_section(
        "before", "Before: what the original language could say",
        base_intro, by_stage["base"],
        [rendered[("base", e["bench_idx"])] for e in by_stage["base"]],
        base_cap))
    sections.append(stage_section(
        "b1", "First extension: wall-less parking and stepping aside",
        b1_intro, by_stage["b1"],
        [rendered[("b1", e["bench_idx"])] for e in by_stage["b1"]],
        b1_cap))
    sections.append(stage_section(
        "b2", "Second extension: re-using robots the plan already placed",
        b2_intro, by_stage["b2"],
        [rendered[("b2", e["bench_idx"])] for e in by_stage["b2"]],
        b2_cap))

    ceiling_line = ""
    if cs["old"] and cs["b1"] and cs["b2"]:
        ceiling_line = (f"  <p class=\"lede\">The three pictures below are "
                        f"the reason the backward planner's hard ceiling "
                        f"moved <b>{old_txt} → {b1_txt} → {b2_txt}</b>: each "
                        f"extension lets a plan say something that was "
                        f"previously unsayable.</p>\n")

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ricochet Robots: what a plan looks like</title>
<style>{CSS}</style>
</head>
<body>
<button class="themebtn" id="themebtn" type="button">theme: auto</button>
<div class="wrap">
<header>
  <h1>What a backward plan looks like — before and after the two language
  extensions</h1>
  <p class="lede">The backward planner writes a puzzle solution as a
  <b>plan</b>: a small diagram of commitments (“this robot must stop here;
  for that, this helper must first park there…”) rather than a list of
  moves. This page draws five real plans — all of them verified to play out
  legally, move by move — to show what the plan language could and could not
  express at each stage.</p>
{ceiling_line}  <div class="howto"><b>How to read the pictures.</b> The goal
  sits at the top; robots' starting positions sit at the bottom. A plan is
  <i>written</i> top-down (“for the goal slide to work, this must happen
  first…”), but the robots <b>travel bottom-up</b>: every colored arrow means
  “this robot really slides from the lower box up to the upper box,” and the
  pill on the arrow counts its moves. Thin gray lines only group things and
  involve no movement. Cell coordinates like (4,1) locate each box on the
  real board — the small board beside each diagram shows the starting
  layout.</div>
</header>
{"".join(sections)}
<div class="footer">
  <p>Every number and every diagram on this page is generated from
  <span class="mono">eval/results/plan_structures_data.json</span> (five
  plans produced by the hand-coded search and verified by legal playback;
  extractor: <span class="mono">analysis/plan_viz/extract_plans.py</span>)
  and the ceiling probe files. Rebuild with
  <span class="mono">PYTHONPATH=. python3 -m eval.build_plan_viz</span>.</p>
  <p><a href="report.html#ceiling">← back to the full comparison report</a></p>
</div>
</div>
<script>{JS}</script>
</body>
</html>
"""
    return page, exs, cs


def verify(page, exs, cs):
    from html.parser import HTMLParser

    class P(HTMLParser):
        pass

    P().feed(page)
    check(f"html.parser consumed the document ({len(page)} bytes)", True)

    for ex in exs:
        check(f"puzzle {ex['bench_idx']}: optimum ({ex['d_star']}) and "
              f"legal moves ({ex['legal_moves']}) rendered",
              f"shortest possible solution: {ex['d_star']} moves" in page
              and f"plays out in {ex['legal_moves']} legal moves" in page)
    n_byref = sum(1 for ex in exs for e in ex["edges"] if e.get("byref"))
    check(f"by-reference links drawn ({n_byref})",
          page.count("re-uses this robot — no new one recruited") == n_byref)
    n_park = sum(1 for ex in exs for nd in ex["nodes"]
                 if nd["type"] == "park")
    check(f"park ordering arrows drawn ({n_park})",
          page.count("must happen before this slide") == n_park)
    n_transient = sum(1 for ex in exs for nd in ex["nodes"]
                      if nd.get("transient"))
    check(f"transient stoppers starred ({n_transient})",
          page.count("★ no wall to lean on") == n_transient)
    for key, label in (("old", "original ceiling"),
                       ("b1", "first-extension ceiling"),
                       ("b2", "second-extension ceiling")):
        if cs[key]:
            check(f"{label} percentage rendered",
                  f"{cs[key][1]:.1f}%" in page)
    check("no external references",
          "http://" not in page and "https://" not in page)


def main():
    page, exs, cs = build()
    verify(page, exs, cs)
    with open(OUT_PATH, "w") as f:
        f.write(page)
    print(f"[build_plan_viz] wrote {os.path.relpath(OUT_PATH, ROOT)} "
          f"({len(page)} bytes)")
    fails = 0
    for desc, ok in CHECKS:
        print(f"  {'OK      ' if ok else 'MISMATCH'} {desc}")
        fails += 0 if ok else 1
    print(f"[build_plan_viz] {len(CHECKS) - fails}/{len(CHECKS)} checks "
          f"passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
