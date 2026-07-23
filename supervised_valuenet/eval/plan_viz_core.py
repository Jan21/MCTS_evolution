"""Shared renderer for abstract plan-structure diagrams.

Used by eval/build_plan_viz.py (the deep-dive companion page) and
eval/build_report.py (the compact in-report versions), so the SVG code lives
exactly once. Stdlib only. All CSS custom properties referenced here
(--surface, --ink, --muted, --baseline, --line, --grid, --r-red, --r-blue,
--r-green, --r-yellow) are defined by both host pages in both themes.

Input: one example dict from eval/results/plan_structures_data.json —
nodes {id, type: goal|subgoal|bottleneck|support|leaf|park, pos, robot,
transient?, before_edge?}; edges {from, to, status, cost, byref}. Edges point
goal→leaf; the robot physically travels child→parent, which the arrowheads
encode (they point up, at the parent end).
"""

import html as _html

STRUCTURAL_PAIRS = {("subgoal", "bottleneck"), ("subgoal", "support")}

ROBOT_VARS = {"Red": "--r-red", "Blue": "--r-blue",
              "Green": "--r-green", "Yellow": "--r-yellow"}

GEOM_FULL = {"cell_w": 168, "row_h": 96, "node_w": 132, "node_h": 44,
             "pad": 18}
GEOM_COMPACT = {"cell_w": 142, "row_h": 82, "node_w": 116, "node_h": 40,
                "pad": 14}


def esc(x):
    return _html.escape(str(x), quote=True)


def robot_var(name):
    return ROBOT_VARS.get(name, "--faint")


# ---------------------------------------------------------------------------
# Layout: layered, goal on top, leaves at the bottom
# ---------------------------------------------------------------------------

def layout(nodes, edges):
    """{node_id: depth}, {node_id: column}, ncols, nrows.

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
            continue
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
    for r in roots:
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


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def node_svg(n, x, y, G):
    t = n["type"]
    nw, nh = G["node_w"], G["node_h"]
    cxm = x + nw / 2
    pos = ""
    if n.get("pos") is not None:
        pos = f"({n['pos'][0]},{n['pos'][1]})"
    var = robot_var(n.get("robot"))
    parts = []

    def label(title, sub, ink="var(--ink)"):
        parts.append(
            f'<text x="{cxm:.0f}" y="{y + nh * 0.41:.0f}" '
            f'text-anchor="middle" font-size="11" font-weight="640" '
            f'fill="{ink}">{esc(title)}</text>')
        if sub:
            parts.append(
                f'<text x="{cxm:.0f}" y="{y + nh * 0.75:.0f}" '
                f'text-anchor="middle" font-size="10" fill="var(--muted)" '
                f'style="font-family:var(--mono)">{esc(sub)}</text>')

    if t == "goal":
        parts.append(
            f'<rect x="{x}" y="{y}" width="{nw}" height="{nh}" rx="10" '
            f'fill="var(--surface)" stroke="var(--ink)" stroke-width="2.2"/>')
        parts.append(
            f'<rect x="{x + 4}" y="{y + 4}" width="{nw - 8}" '
            f'height="{nh - 8}" rx="7" fill="none" stroke="var(--ink)" '
            f'stroke-width="1" stroke-dasharray="4 3"/>')
        label("the goal cell", pos)
    elif t == "subgoal":
        r = 7
        parts.append(
            f'<path d="M{cxm} {y + nh / 2 - r} l{r} {r} l-{r} {r} '
            f'l-{r} -{r} z" fill="var(--baseline)"/>')
    elif t == "bottleneck":
        parts.append(
            f'<rect x="{x}" y="{y}" width="{nw}" height="{nh}" rx="9" '
            f'fill="var(--surface)" stroke="var({var})" stroke-width="2"/>')
        label(f"{n['robot']} stops here", pos, f"var({var})")
    elif t == "support":
        extra = ' stroke-dasharray="5 3"' if n.get("transient") else ""
        parts.append(
            f'<rect x="{x}" y="{y}" width="{nw}" height="{nh}" rx="9" '
            f'fill="var({var})" fill-opacity="0.14" stroke="var({var})" '
            f'stroke-width="2"{extra}/>')
        label(f"{n['robot']} parks here", pos, f"var({var})")
        if n.get("transient"):
            parts.append(
                f'<text x="{x + nw - 4}" y="{y - 5}" text-anchor="end" '
                f'font-size="10" font-weight="700" fill="var({var})">'
                f"★ no wall to lean on</text>")
    elif t == "leaf":
        parts.append(
            f'<rect x="{x}" y="{y}" width="{nw}" height="{nh}" '
            f'rx="{nh / 2}" fill="var(--surface)" stroke="var({var})" '
            f'stroke-width="1.6"/>')
        parts.append(
            f'<circle cx="{x + 18}" cy="{y + nh / 2}" r="6.5" '
            f'fill="var({var})"/>')
        parts.append(
            f'<text x="{x + 31}" y="{y + nh * 0.41:.0f}" font-size="11" '
            f'font-weight="640" fill="var(--ink)">start: {esc(n["robot"])}'
            f"</text>")
        parts.append(
            f'<text x="{x + 31}" y="{y + nh * 0.75:.0f}" font-size="10" '
            f'fill="var(--muted)" style="font-family:var(--mono)">{esc(pos)}'
            f"</text>")
    elif t == "park":
        parts.append(
            f'<rect x="{x}" y="{y}" width="{nw}" height="{nh}" rx="9" '
            f'fill="none" stroke="var({var})" stroke-width="2" '
            f'stroke-dasharray="2.5 3.5"/>')
        label(f"{n['robot']} steps aside", pos, f"var({var})")
    return "".join(parts)


# ---------------------------------------------------------------------------
# The full diagram
# ---------------------------------------------------------------------------

def dag_svg(ex, aria, G=None, marker_id="arr"):
    """Render one plan DAG. Returns (svg, n_cost_labels)."""
    G = G or GEOM_FULL
    cell_w, row_h = G["cell_w"], G["row_h"]
    nw, nh, pad = G["node_w"], G["node_h"], G["pad"]
    nodes, edges = ex["nodes"], ex["edges"]
    by_id = {n["id"]: n for n in nodes}
    children = {n["id"]: [] for n in nodes}
    for e in edges:
        if not e.get("byref"):
            children[e["from"]].append(e["to"])
    depth, col, ncols, nrows = layout(nodes, edges)

    W = pad * 2 + ncols * cell_w
    H = pad * 2 + (nrows - 1) * row_h + nh + 26

    def cx(nid):
        return pad + col[nid] * cell_w + (cell_w - nw) / 2

    def cy(nid):
        return pad + 14 + depth[nid] * row_h

    def anchor(nid, top):
        n = by_id[nid]
        x = cx(nid) + nw / 2
        if n["type"] == "subgoal":
            y = cy(nid) + nh / 2 + (-9 if top else 9)
        else:
            y = cy(nid) + (0 if top else nh)
        return x, y

    parts = [f'<svg viewBox="0 0 {W:.0f} {H:.0f}" role="img" '
             f'aria-label="{esc(aria)}" '
             f'style="width:100%;max-width:{W:.0f}px;height:auto;'
             f'display:block">',
             f'<defs><marker id="{marker_id}" viewBox="0 0 8 8" refX="7" '
             f'refY="4" markerWidth="7" markerHeight="7" '
             f'orient="auto-start-reverse">'
             f'<path d="M0 0 L8 4 L0 8 z" fill="context-stroke"/>'
             f"</marker></defs>"]

    edge_mid = {}
    n_cost_labels = 0
    for e in edges:
        u, v = e["from"], e["to"]
        structural = ((by_id[u]["type"], by_id[v]["type"]) in STRUCTURAL_PAIRS
                      and not e.get("byref"))
        x1, y1 = anchor(u, top=False)
        x2, y2 = anchor(v, top=True)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        edge_mid[(u, v)] = (mx, my)
        if e.get("byref"):
            var = robot_var(by_id[v].get("robot"))
            parts.append(
                f'<path d="M{x1:.0f} {y1:.0f} C {x1:.0f} {y1 + 34:.0f}, '
                f'{x2:.0f} {y2 - 34:.0f}, {x2:.0f} {y2:.0f}" fill="none" '
                f'stroke="var({var})" stroke-width="2" '
                f'stroke-dasharray="6 4" marker-end="url(#{marker_id})"/>')
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
            f'marker-end="url(#{marker_id})"/>')
        c = int(e["cost"]) if float(e["cost"]).is_integer() else e["cost"]
        parts.append(
            f'<rect x="{mx - 26:.0f}" y="{my - 10:.0f}" width="52" '
            f'height="18" rx="9" fill="var(--surface)" '
            f'stroke="var(--line)"/>'
            f'<text class="cost" x="{mx:.0f}" y="{my + 4:.0f}" '
            f'text-anchor="middle" font-size="10.5" font-weight="700" '
            f'fill="var(--ink)">{c} slide{"s" if c != 1 else ""}</text>')
        n_cost_labels += 1

    for n in nodes:
        if n["type"] != "park" or not n.get("before_edge"):
            continue
        be = tuple(n["before_edge"])
        if be not in edge_mid:
            continue
        px, py = cx(n["id"]) + nw / 2, cy(n["id"])
        tx, ty = edge_mid[be]
        var = robot_var(n.get("robot"))
        parts.append(
            f'<path d="M{px:.0f} {py:.0f} C {px:.0f} {py - 40:.0f}, '
            f'{tx + 40:.0f} {ty + 40:.0f}, {tx + 8:.0f} {ty + 8:.0f}" '
            f'fill="none" stroke="var({var})" stroke-width="1.6" '
            f'stroke-dasharray="2 4" marker-end="url(#{marker_id})"/>')
        parts.append(
            f'<text x="{(px + tx) / 2 + 12:.0f}" y="{(py + ty) / 2:.0f}" '
            f'font-size="10.5" font-style="italic" fill="var({var})">'
            f"must happen before this slide</text>")

    for n in nodes:
        parts.append(node_svg(n, cx(n["id"]), cy(n["id"]), G))
    parts.append("</svg>")
    return "".join(parts), n_cost_labels


# ---------------------------------------------------------------------------
# Mini board inset + legend
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


def legend(marker_id="la"):
    return f"""
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
  <div><svg viewBox="0 0 150 30" style="width:130px"><path d="M10 15 l30 0"
    stroke="var(--baseline)" stroke-width="1.3"/></svg>
    <span>thin gray line — grouping only, no movement</span></div>
  <div><svg viewBox="0 0 150 30" style="width:130px"><defs><marker
    id="{marker_id}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7"
    markerHeight="7" orient="auto"><path d="M0 0 L8 4 L0 8 z"
    fill="context-stroke"/></marker></defs><line x1="14" y1="24" x2="14"
    y2="6" stroke="var(--r-blue)" stroke-width="2"
    marker-end="url(#{marker_id})"/><rect x="30" y="6" width="52"
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
