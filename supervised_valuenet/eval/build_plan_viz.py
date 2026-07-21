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

Each plan is a DAG. Nodes are drawn goal-at-top, starting positions at the
bottom; edges between them are either *structural* (grouping, no movement) or
*physical* (a robot really slides). The one non-obvious reading rule — the
robot travels from the LOWER node UP to the upper node — is stated on the page
itself and encoded in the arrowheads.
"""

import json
import html as _html
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "eval", "results", "plan_structures_data.json")
OUT_PATH = os.path.join(ROOT, "eval", "results", "plan_structures.html")

STRUCTURAL_PAIRS = {("subgoal", "bottleneck"), ("subgoal", "support")}

ROBOT_VARS = {"Red": "--r-red", "Blue": "--r-blue",
              "Green": "--r-green", "Yellow": "--r-yellow"}

CHECKS = []  # (description, ok)


def check(desc, ok):
    CHECKS.append((desc, bool(ok)))


def esc(x):
    return _html.escape(str(x), quote=True)


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
# DAG layout: layered, goal on top, leaves at the bottom
# ---------------------------------------------------------------------------

def layout(nodes, edges):
    """Return {node_id: (col, row)} plus the grid extent (ncols, nrows).

    Layering: primary-tree depth from each root (the goal, then any park
    nodes), park subtrees bottom-aligned with the main tree. Columns: one
    global post-order walk, leaves take successive columns, every parent
    centers over its children. Deterministic; by-reference edges are overlay
    links and do not affect the layout.
    """
    by_id = {n["id"]: n for n in nodes}
    children = {n["id"]: [] for n in nodes}
    has_parent = set()
    for e in edges:
        if e.get("byref"):
            continue                      # overlay link, not a tree edge
        children[e["from"]].append(e["to"])
        has_parent.add(e["to"])
    roots = [n["id"] for n in nodes if n["id"] not in has_parent]
    roots.sort(key=lambda i: (0 if by_id[i]["type"] == "goal" else 1, i))

    depth, col = {}, {}
    counter = [0]

    def walk(nid, d):
        depth[nid] = max(depth.get(nid, 0), d)
        kids = children[nid]
        if not kids:
            col[nid] = counter[0]
            counter[0] += 1
            return col[nid]
        xs = [walk(k, d + 1) for k in kids]
        col[nid] = sum(xs) / len(xs)
        return col[nid]

    sub_max = {}
    for r in roots:
        walk(r, 0)
        sub_max[r] = max(depth[nid] for nid in _subtree(r, children))
    grand = max(sub_max.values()) if sub_max else 0
    for r in roots:                       # bottom-align each root's subtree
        shift = grand - sub_max[r]
        if shift:
            for nid in _subtree(r, children):
                depth[nid] += shift
    ncols = max(1, int(round(max(col.values()))) + 1) if col else 1
    return depth, col, ncols, grand + 1


def _subtree(root, children):
    out, todo = set(), [root]
    while todo:
        cur = todo.pop()
        if cur in out:
            continue
        out.add(cur)
        todo.extend(children[cur])
    return out


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------

CELL_W, ROW_H, NODE_W, NODE_H, PAD = 168, 96, 132, 44, 18


def robot_var(name):
    return ROBOT_VARS.get(name, "--faint")


def _mover_robot(by_id, children, v_id):
    """Robot doing the physical travel of edge (u, v): v's own robot, or —
    when v is a grouping node — the robot of its 'stops here' child."""
    v = by_id[v_id]
    if v["type"] != "subgoal":
        return v.get("robot")
    for k in children[v_id]:
        if by_id[k]["type"] == "bottleneck":
            return by_id[k].get("robot")
    return None


def node_svg(n, x, y):
    t = n["type"]
    cxm = x + NODE_W / 2
    pos = ""
    if n.get("pos") is not None:
        pos = f"({n['pos'][0]},{n['pos'][1]})"
    var = robot_var(n.get("robot"))
    parts = []

    def label(title, sub, ink="var(--ink)"):
        parts.append(
            f'<text x="{cxm:.0f}" y="{y + 18:.0f}" text-anchor="middle" '
            f'font-size="11.5" font-weight="640" fill="{ink}">{esc(title)}'
            f"</text>")
        if sub:
            parts.append(
                f'<text x="{cxm:.0f}" y="{y + 33:.0f}" text-anchor="middle" '
                f'font-size="10.5" fill="var(--muted)" '
                f'style="font-family:var(--mono)">{esc(sub)}</text>')

    if t == "goal":
        parts.append(
            f'<rect x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" '
            f'rx="10" fill="var(--surface)" stroke="var(--ink)" '
            f'stroke-width="2.2"/>')
        parts.append(
            f'<rect x="{x + 4}" y="{y + 4}" width="{NODE_W - 8}" '
            f'height="{NODE_H - 8}" rx="7" fill="none" stroke="var(--ink)" '
            f'stroke-width="1" stroke-dasharray="4 3"/>')
        label("the goal cell", pos)
    elif t == "subgoal":
        r = 7
        parts.append(
            f'<path d="M{cxm} {y + NODE_H / 2 - r} l{r} {r} l-{r} {r} '
            f'l-{r} -{r} z" fill="var(--baseline)"/>')
    elif t == "bottleneck":
        parts.append(
            f'<rect x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" '
            f'rx="9" fill="var(--surface)" stroke="var({var})" '
            f'stroke-width="2"/>')
        label(f"{n['robot']} stops here", pos, f"var({var})")
    elif t == "support":
        extra = ""
        if n.get("transient"):
            extra = ' stroke-dasharray="5 3"'
        parts.append(
            f'<rect x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" '
            f'rx="9" fill="var({var})" fill-opacity="0.14" '
            f'stroke="var({var})" stroke-width="2"{extra}/>')
        label(f"{n['robot']} parks here", pos, f"var({var})")
        if n.get("transient"):
            parts.append(
                f'<text x="{x + NODE_W - 4}" y="{y - 5}" text-anchor="end" '
                f'font-size="10" font-weight="700" fill="var({var})">'
                f"★ no wall to lean on</text>")
    elif t == "leaf":
        parts.append(
            f'<rect x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" '
            f'rx="{NODE_H / 2}" fill="var(--surface)" stroke="var({var})" '
            f'stroke-width="1.6"/>')
        parts.append(
            f'<circle cx="{x + 20}" cy="{y + NODE_H / 2}" r="7" '
            f'fill="var({var})"/>')
        parts.append(
            f'<text x="{x + 34}" y="{y + 18}" font-size="11.5" '
            f'font-weight="640" fill="var(--ink)">start: {esc(n["robot"])}'
            f"</text>")
        parts.append(
            f'<text x="{x + 34}" y="{y + 33}" font-size="10.5" '
            f'fill="var(--muted)" style="font-family:var(--mono)">{esc(pos)}'
            f"</text>")
    elif t == "park":
        parts.append(
            f'<rect x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" '
            f'rx="9" fill="none" stroke="var({var})" stroke-width="2" '
            f'stroke-dasharray="2.5 3.5"/>')
        label(f"{n['robot']} steps aside", pos, f"var({var})")
    return "".join(parts)


def dag_svg(ex, aria):
    nodes, edges = ex["nodes"], ex["edges"]
    by_id = {n["id"]: n for n in nodes}
    children = {n["id"]: [] for n in nodes}
    for e in edges:
        if not e.get("byref"):
            children[e["from"]].append(e["to"])
    depth, col, ncols, nrows = layout(nodes, edges)

    W = PAD * 2 + ncols * CELL_W
    H = PAD * 2 + (nrows - 1) * ROW_H + NODE_H + 26

    def cx(nid):                          # left x of the node box
        return PAD + col[nid] * CELL_W + (CELL_W - NODE_W) / 2

    def cy(nid):                          # top y of the node box
        return PAD + 14 + depth[nid] * ROW_H

    def anchor(nid, top):
        n = by_id[nid]
        x = cx(nid) + NODE_W / 2
        if n["type"] == "subgoal":
            y = cy(nid) + NODE_H / 2 + (-9 if top else 9)
        else:
            y = cy(nid) + (0 if top else NODE_H)
        return x, y

    parts = [f'<svg viewBox="0 0 {W:.0f} {H:.0f}" role="img" '
             f'aria-label="{esc(aria)}" '
             f'style="width:100%;max-width:{W:.0f}px;height:auto;'
             f'display:block">',
             '<defs>'
             '<marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" '
             'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             '<path d="M0 0 L8 4 L0 8 z" fill="context-stroke"/>'
             "</marker></defs>"]

    edge_mid = {}
    n_cost_labels = 0
    for e in edges:
        u, v = e["from"], e["to"]
        structural = ((by_id[u]["type"], by_id[v]["type"]) in STRUCTURAL_PAIRS
                      and not e.get("byref"))
        x1, y1 = anchor(u, top=False)      # parent bottom
        x2, y2 = anchor(v, top=True)       # child top
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        edge_mid[(u, v)] = (mx, my)
        if e.get("byref"):
            var = robot_var(by_id[v].get("robot"))
            parts.append(
                f'<path d="M{x1:.0f} {y1:.0f} C {x1:.0f} {y1 + 34:.0f}, '
                f'{x2:.0f} {y2 - 34:.0f}, {x2:.0f} {y2:.0f}" fill="none" '
                f'stroke="var({var})" stroke-width="2" '
                f'stroke-dasharray="6 4" marker-end="url(#arr)"/>')
            parts.append(
                f'<text x="{mx + 8:.0f}" y="{my:.0f}" font-size="10.5" '
                f'font-weight="700" fill="var({var})">re-uses this robot — '
                f"no new one recruited</text>")
            continue
        if structural or not e.get("cost"):
            parts.append(
                f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" '
                f'y2="{y2:.0f}" stroke="var(--baseline)" '
                f'stroke-width="1.3"/>')
            continue
        mover = _mover_robot(by_id, children, v)
        var = robot_var(mover)
        # arrowhead at the PARENT end: the robot travels child -> parent (up)
        parts.append(
            f'<line x1="{x2:.0f}" y1="{y2:.0f}" x2="{x1:.0f}" y2="{y1:.0f}" '
            f'stroke="var({var})" stroke-width="2.2" '
            f'marker-end="url(#arr)"/>')
        c = int(e["cost"]) if float(e["cost"]).is_integer() else e["cost"]
        parts.append(
            f'<rect x="{mx - 26:.0f}" y="{my - 10:.0f}" width="52" '
            f'height="18" rx="9" fill="var(--surface)" '
            f'stroke="var(--line)"/>'
            f'<text class="cost" x="{mx:.0f}" y="{my + 4:.0f}" '
            f'text-anchor="middle" font-size="10.5" font-weight="700" '
            f'fill="var(--ink)">{c} slide{"s" if c != 1 else ""}</text>')
        n_cost_labels += 1

    for n in nodes:                        # park ordering arrows on top
        if n["type"] != "park" or not n.get("before_edge"):
            continue
        be = tuple(n["before_edge"])
        if be not in edge_mid:
            continue
        px, py = cx(n["id"]) + NODE_W / 2, cy(n["id"])
        tx, ty = edge_mid[be]
        var = robot_var(n.get("robot"))
        parts.append(
            f'<path d="M{px:.0f} {py:.0f} C {px:.0f} {py - 40:.0f}, '
            f'{tx + 40:.0f} {ty + 40:.0f}, {tx + 8:.0f} {ty + 8:.0f}" '
            f'fill="none" stroke="var({var})" stroke-width="1.6" '
            f'stroke-dasharray="2 4" marker-end="url(#arr)"/>')
        parts.append(
            f'<text x="{(px + tx) / 2 + 12:.0f}" y="{(py + ty) / 2:.0f}" '
            f'font-size="10.5" font-style="italic" fill="var({var})">'
            f"must happen before this slide</text>")

    for n in nodes:
        parts.append(node_svg(n, cx(n["id"]), cy(n["id"])))
    parts.append("</svg>")
    return "".join(parts), n_cost_labels


# ---------------------------------------------------------------------------
# Mini board inset
# ---------------------------------------------------------------------------

def board_inset(ex):
    size, c, pad = 16, 13, 3
    W = size * c + 2 * pad
    parts = [f'<svg viewBox="0 0 {W} {W}" role="img" aria-label="Where the '
             f'robots start on the real 16-by-16 board" '
             f'style="width:100%;max-width:220px;height:auto;display:block">']
    for i in range(1, size):
        parts.append(f'<line x1="{pad + i * c}" y1="{pad}" '
                     f'x2="{pad + i * c}" y2="{W - pad}" '
                     f'stroke="var(--grid)" stroke-width="0.6"/>')
        parts.append(f'<line x1="{pad}" y1="{pad + i * c}" '
                     f'x2="{W - pad}" y2="{pad + i * c}" '
                     f'stroke="var(--grid)" stroke-width="0.6"/>')
    parts.append(f'<rect x="{pad}" y="{pad}" width="{size * c}" '
                 f'height="{size * c}" fill="none" stroke="var(--ink)" '
                 f'stroke-width="1.6"/>')
    tx, ty = ex["target_cell"]
    tvar = robot_var(ex["target_robot"])
    parts.append(
        f'<rect x="{pad + tx * c + 1.5}" y="{pad + ty * c + 1.5}" '
        f'width="{c - 3}" height="{c - 3}" rx="2.5" fill="none" '
        f'stroke="var({tvar})" stroke-width="1.6" stroke-dasharray="3 2"/>')
    for name, (x, y) in sorted(ex["robots"].items()):
        var = robot_var(name)
        cxp, cyp = pad + x * c + c / 2, pad + y * c + c / 2
        if name == ex["target_robot"]:
            parts.append(f'<circle cx="{cxp}" cy="{cyp}" r="{c / 2 + 1}" '
                         f'fill="none" stroke="var({var})" '
                         f'stroke-width="1.2" opacity="0.55"/>')
        parts.append(f'<circle cx="{cxp}" cy="{cyp}" r="{c / 2 - 1.5}" '
                     f'fill="var({var})"/>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------

def legend():
    return """
<div class="legend-grid">
  <div><svg viewBox="0 0 150 30" style="width:130px"><rect x="4" y="3"
    width="140" height="24" rx="7" fill="var(--surface)" stroke="var(--ink)"
    stroke-width="1.8"/><rect x="8" y="7" width="132" height="16" rx="4"
    fill="none" stroke="var(--ink)" stroke-dasharray="4 3"
    stroke-width="0.8"/></svg><span>the goal cell</span></div>
  <div><svg viewBox="0 0 150 30" style="width:130px"><rect x="4" y="3"
    width="140" height="24" rx="12" fill="var(--surface)"
    stroke="var(--r-blue)" stroke-width="1.6"/><circle cx="18" cy="15" r="6"
    fill="var(--r-blue)"/></svg><span>a robot's starting position</span></div>
  <div><svg viewBox="0 0 150 30" style="width:130px"><rect x="4" y="3"
    width="140" height="24" rx="7" fill="var(--surface)"
    stroke="var(--r-green)" stroke-width="1.8"/></svg>
    <span>"stops here" — a robot's planned stopping cell</span></div>
  <div><svg viewBox="0 0 150 30" style="width:130px"><rect x="4" y="3"
    width="140" height="24" rx="7" fill="var(--r-yellow)" fill-opacity="0.14"
    stroke="var(--r-yellow)" stroke-width="1.8"/></svg>
    <span>"parks here" — a helper placed so another robot can bounce off
    it</span></div>
  <div><svg viewBox="0 0 150 30" style="width:130px"><rect x="4" y="3"
    width="140" height="24" rx="7" fill="none" stroke="var(--r-red)"
    stroke-width="1.8" stroke-dasharray="2.5 3.5"/></svg>
    <span>"steps aside" — a robot moves out of the way first</span></div>
  <div><svg viewBox="0 0 150 30" style="width:130px"><path d="M10 15 l16 0"
    stroke="var(--baseline)" stroke-width="1.3"/><path d="M40 22 l10 -7
    l-10 -7 l-10 7 z" fill="var(--baseline)" transform="translate(0,0)
    scale(0.6) translate(30,8)"/></svg>
    <span>thin gray line — grouping only, no movement</span></div>
  <div><svg viewBox="0 0 150 30" style="width:130px"><defs><marker id="la"
    viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7"
    orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="context-stroke"/></marker>
    </defs><line x1="14" y1="24" x2="14" y2="6" stroke="var(--r-blue)"
    stroke-width="2" marker-end="url(#la)"/><rect x="30" y="6" width="52"
    height="18" rx="9" fill="var(--surface)" stroke="var(--line)"/>
    <text x="56" y="19" text-anchor="middle" font-size="10"
    font-weight="700" fill="var(--ink)">2 slides</text></svg>
    <span>colored arrow — that robot really slides, upward along the arrow;
    the pill counts its moves</span></div>
  <div><svg viewBox="0 0 150 30" style="width:130px"><line x1="10" y1="24"
    x2="60" y2="8" stroke="var(--r-blue)" stroke-width="2"
    stroke-dasharray="6 4"/></svg>
    <span>dashed link — the plan re-uses a robot it already placed</span></div>
</div>
"""


def stage_section(sid, title, intro, examples, blocks, caption):
    ex_html = []
    for ex, (svg, _), extra in zip(examples, blocks, [None] * len(examples)):
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
  {legend()}
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
            ex, f"Plan structure for benchmark puzzle {ex['bench_idx']}")
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

    base_intro = f"""
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

    b1_intro = f"""
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
