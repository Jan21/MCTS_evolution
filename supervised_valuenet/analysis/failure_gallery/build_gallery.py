"""Build eval/results/failure_gallery.html — the structural-failure gallery.

One self-contained page: for every family of puzzles the backward (subgoal)
planner provably cannot express, show one real puzzle, the forward planner's
actual solution stepped move by move, and the exact moment the subgoal
language runs out of words.

Usage (from the repo root):
    PYTHONPATH=. python3 analysis/failure_gallery/build_gallery.py

Inputs (all machine-readable; no number on the page is typed by hand):
  analysis/artifacts/ceiling_probe_results.json        base probe verdicts
  analysis/artifacts/ceiling_probe_instances.json      base probed puzzles
  scaling/results/g16r6/ceiling_probe_old_vocab.json   6-robot probe verdicts
  scaling/results/g16r6/ceiling_probe_instances.json   6-robot probed puzzles
  analysis/artifacts/cap_sensitivity_base.json         4x-cap re-probe (base)
  scaling/results/g16r6/cap_sensitivity.json           4x-cap re-probe (6r)
  analysis/failure_gallery/forward_solutions_base.json forward solves (base)
  analysis/failure_gallery/forward_solutions_g16r6.json forward solves (6r)
  eval/data/bench450.jsonl                             benchmark size (base)
  scaling/results/g16r6/comparison.json + _ungraded    benchmark size (6r)

Every representative's move sequence is replayed move-by-move under
simulate.slide at build time; an illegal replay aborts the build. A
verification table (page numbers vs artifact numbers) is printed to stdout.
"""
from __future__ import annotations

import html as _html
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from diagnose import (analyze_row, board_walls, wall_holdable_cells,  # noqa: E402
                      family_of, b1_coverage, FAMILY_ORDER, FAMILY_LABEL)
from simulate import slide  # noqa: E402

OUT_PATH = ROOT / "eval" / "results" / "failure_gallery.html"

STRUCTURAL = ("NO_COMPLETE_PLAN", "NO_REALIZABLE_PLAN")

SLOT_NAMES = ("Red", "Blue", "Green", "Yellow", "Purple", "Orange")
SLOT_VARS = ("--r-red", "--r-blue", "--r-green", "--r-yellow",
             "--r-purple", "--r-orange")

CHECKS = []  # (description, page_value, artifact_value)
REP_VALIDATIONS = []  # replay-legality confirmations, printed at the end


def esc(x):
    return _html.escape(str(x), quote=True)


def check(desc, page, artifact):
    CHECKS.append((desc, page, artifact))
    return page


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

def jload(rel):
    return json.load(open(ROOT / rel))


SCALES = {
    "base": {
        "label": "base scale (16x16, 4 robots)",
        "short": "base",
        "env_dir": str(ROOT / "environments"),
        "grid": 16,
        "probe": "analysis/artifacts/ceiling_probe_results.json",
        "instances": "analysis/artifacts/ceiling_probe_instances.json",
        "cap": "analysis/artifacts/cap_sensitivity_base.json",
        "solutions": "analysis/failure_gallery/forward_solutions_base.json",
    },
    "g16r6": {
        "label": "6 robots (16x16)",
        "short": "6-robot",
        "env_dir": str(ROOT / "environments_g16r6"),
        "grid": 16,
        "probe": "scaling/results/g16r6/ceiling_probe_old_vocab.json",
        "instances": "scaling/results/g16r6/ceiling_probe_instances.json",
        "cap": "scaling/results/g16r6/cap_sensitivity.json",
        "solutions": "analysis/failure_gallery/forward_solutions_g16r6.json",
    },
}


def bench_size(scale):
    if scale == "base":
        n = sum(1 for line in open(ROOT / "eval/data/bench450.jsonl")
                if line.strip())
        return n
    tot = 0
    for f in ("scaling/results/g16r6/comparison.json",
              "scaling/results/g16r6/comparison_ungraded.json"):
        d = jload(f)
        ns = {s["aggregate"]["n"] for s in d["systems"].values()
              if "aggregate" in s}
        assert len(ns) == 1
        tot += ns.pop()
    return tot


def load_scale_data(key):
    sc = SCALES[key]
    probe = jload(sc["probe"])
    cats = Counter(r["category"] for r in probe)
    sols = jload(sc["solutions"])
    rows = sols["rows"]
    # structural set must equal the probe's structural rows
    n_struct = cats["NO_COMPLETE_PLAN"] + cats["NO_REALIZABLE_PLAN"]
    assert len(rows) == n_struct, (key, len(rows), n_struct)
    recs = []
    for row in rows:
        rec = {"row": row, "scale": key}
        key_path = ("path" if row.get("solved")
                    else "fallback_path" if row.get("fallback_solved") else None)
        if key_path:
            rec["path_key"] = key_path
            rec["analysis"] = analyze_row(row, sc["env_dir"], sc["grid"],
                                          key_path)
            rec["family"] = family_of(rec["analysis"]["gates"])
            rec["b1"] = b1_coverage(rec["analysis"]["gates"])
        else:
            rec["family"] = "unsolved"
            rec["b1"] = "unknown (no forward solution to read)"
        recs.append(rec)
    cap = jload(sc["cap"])
    cap_still = sum(1 for r in cap if r["category"] in STRUCTURAL)
    return {
        "key": key, "cfg": sc, "probe": probe, "cats": cats, "recs": recs,
        "protocol": sols["protocol"], "bench_n": bench_size(key),
        "cap_n": len(cap), "cap_still_structural": cap_still,
    }


# ---------------------------------------------------------------------------
# board SVG (precedent: eval/build_report.py board_svg; adapted standalone)
# ---------------------------------------------------------------------------

def board_svg(size, wr, wd, robots, arrow=None, ring=None, marks=(),
              aria="", max_px=430):
    """robots: [(x, y, cssvar, letter, is_target)]; arrow: (x0,y0,x1,y1,var);
    ring: (x, y, var); marks: [(x, y, cssvar, title)]."""
    import math
    c = 26
    pad = 4
    W = H = size * c + 2 * pad
    parts = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(aria)}" '
        f'style="width:100%;max-width:{max_px}px;height:auto;display:block">'
    ]

    def cx(x):
        return pad + x * c + c / 2

    def cy(y):
        return pad + y * c + c / 2

    for i in range(1, size):
        parts.append(f'<line x1="{pad + i * c}" y1="{pad}" x2="{pad + i * c}" '
                     f'y2="{H - pad}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<line x1="{pad}" y1="{pad + i * c}" x2="{W - pad}" '
                     f'y2="{pad + i * c}" stroke="var(--grid)" stroke-width="1"/>')
    parts.append(f'<rect x="{pad}" y="{pad}" width="{size * c}" height="{size * c}" '
                 f'fill="none" stroke="var(--ink)" stroke-width="3"/>')
    for (x, y) in sorted(wr):
        X = pad + (x + 1) * c
        parts.append(f'<line x1="{X}" y1="{pad + y * c}" x2="{X}" '
                     f'y2="{pad + (y + 1) * c}" stroke="var(--ink)" '
                     f'stroke-width="3" stroke-linecap="round"/>')
    for (x, y) in sorted(wd):
        Y = pad + (y + 1) * c
        parts.append(f'<line x1="{pad + x * c}" y1="{Y}" x2="{pad + (x + 1) * c}" '
                     f'y2="{Y}" stroke="var(--ink)" stroke-width="3" '
                     f'stroke-linecap="round"/>')
    for (x, y, var, title) in marks:
        parts.append(
            f'<rect x="{pad + x * c + 2}" y="{pad + y * c + 2}" width="{c - 4}" '
            f'height="{c - 4}" rx="4" fill="none" stroke="var({var})" '
            f'stroke-width="2" stroke-dasharray="3.5 2.5">'
            f"<title>{esc(title)}</title></rect>")
    if ring is not None:
        x, y, var = ring
        parts.append(
            f'<rect x="{pad + x * c + 3}" y="{pad + y * c + 3}" width="{c - 6}" '
            f'height="{c - 6}" rx="5" fill="none" stroke="var({var})" '
            f'stroke-width="2.2" stroke-dasharray="4 3"/>')
    if arrow is not None:
        x0, y0, x1, y1, var = arrow
        X0, Y0, X1, Y1 = cx(x0), cy(y0), cx(x1), cy(y1)
        dx, dy = X1 - X0, Y1 - Y0
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        X0s, Y0s = X0 + ux * 10, Y0 + uy * 10
        X1s, Y1s = X1 - ux * 4, Y1 - uy * 4
        parts.append(
            f'<line x1="{X0s:.1f}" y1="{Y0s:.1f}" x2="{X1s:.1f}" y2="{Y1s:.1f}" '
            f'stroke="var({var})" stroke-width="2.6" stroke-linecap="round" '
            f'opacity="0.85"/>')
        hx, hy = X1s, Y1s
        px, py = -uy, ux
        parts.append(
            f'<path d="M{hx:.1f},{hy:.1f} '
            f'L{hx - ux * 8 + px * 4.5:.1f},{hy - uy * 8 + py * 4.5:.1f} '
            f'L{hx - ux * 8 - px * 4.5:.1f},{hy - uy * 8 - py * 4.5:.1f} Z" '
            f'fill="var({var})" opacity="0.85"/>')
    for (x, y, var, letter, is_target) in robots:
        ring_svg = (f'<circle cx="{cx(x):.1f}" cy="{cy(y):.1f}" r="11.5" '
                    f'fill="none" stroke="var({var})" stroke-width="1.6" '
                    f'opacity="0.55"/>' if is_target else "")
        parts.append(
            f'{ring_svg}<circle cx="{cx(x):.1f}" cy="{cy(y):.1f}" r="8.5" '
            f'fill="var({var})" stroke="var(--surface)" stroke-width="1.6"/>'
            f'<text x="{cx(x):.1f}" y="{cy(y) + 3.4:.1f}" text-anchor="middle" '
            f'fill="var(--bg)" font-size="9.5" font-weight="800">{esc(letter)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# representative rendering
# ---------------------------------------------------------------------------

def replay_states(row, env_dir, grid, path_key):
    """Legal replay under simulate.slide; returns positions after each move.
    Raises on any illegal move — this is the build-time validity gate."""
    wr, wd = board_walls(env_dir, row["env_id"], grid)
    pos = [tuple(p) for p in row["positions"]]
    states = [list(pos)]
    for i, (slot, d) in enumerate(row[path_key]):
        blockers = frozenset(p for j, p in enumerate(pos) if j != slot)
        new = slide(pos[slot], d, blockers, wr, wd, grid)
        if new == pos[slot]:
            raise ValueError(f"illegal move {i} in idx {row['idx']}")
        pos[slot] = new
        states.append(list(pos))
    if tuple(pos[row["target_idx"]]) != tuple(row["target"]):
        raise ValueError(f"idx {row['idx']}: target not on goal after replay")
    return states


def gate_annotations(family, analysis, row, env_dir, grid):
    """move index -> extra plain-language annotation for the family's gate."""
    notes = {}

    def add(t, txt):
        notes.setdefault(int(t), []).append(txt)

    if family == "blocked_direct":
        from diagnose import paper_walk, walk_blockers, board_walls
        wr, wd = board_walls(env_dir, row["env_id"], grid)
        walk = paper_walk(tuple(row["positions"][row["target_idx"]]),
                          tuple(row["target"]), wr, wd, grid)
        e = next(x for x in analysis["events"]
                 if x["gate"] == "blocked_direct")
        blockers = walk_blockers(walk, row["positions"], row["target_idx"])
        via = " → ".join(str(tuple(c)) for c in walk)
        who = ", ".join(SLOT_NAMES[j] for j in blockers) or "a robot"
        add(0, (f"On paper the target could walk to the goal in "
                f"{e['paper_moves']} walls-only slides ({via}), so the plan "
                f"builder pins exactly that walk and proposes nothing else. "
                f"But {who} stands on the walk. The language has no "
                "“step aside” word and no way to ask for a longer "
                "route: it has already run out of words before the first "
                "move."))

    for e in analysis["events"]:
        g = e["gate"]
        if g != family:
            continue
        if g == "transient_support":
            add(e["t"], (
                f"The stopper stands on {tuple(e['cell'])} — a cell with no "
                "adjacent wall. The subgoal vocabulary only lets a plan park "
                "a helper where a wall can hold it, so this parking spot "
                "cannot be named: the plan language runs out of words here."))
        elif g == "vacate":
            add(e["served_t"], (
                f"Here the robot serves as a stopper on {tuple(e['cell'])}. "
                "In the plan language a placed stopper is pinned to its cell "
                "forever."))
            add(e["vacated_t"], (
                f"Now the same robot must leave {tuple(e['cell'])} again, "
                "because the cell is needed later (move "
                f"{e['needed_t'] + 1}). “Step away after the bounce” "
                "does not exist in the vocabulary: this is the moment the "
                "plan language runs out of words."))
        elif g == "relocate":
            t1, c1 = e["first"]
            t2, c2 = e["again"]
            add(t1, (f"First stopper job at {tuple(c1)}. The plan language "
                     "allows one placement per helper, ever."))
            add(t2, (f"The same robot serves as a stopper again at "
                     f"{tuple(c2)}. Re-recruiting a used helper cannot be "
                     "said in the vocabulary: this is where the plan "
                     "language runs out of words."))
        elif g == "shared_support":
            t1, c1 = e["first"]
            t2, c2 = e["again"]
            add(t1, (f"First customer: the parked robot stops a slide at "
                     f"{tuple(c1)}. In the plan language this uses up its "
                     "one support role."))
            add(t2, (f"Second customer: the same parked robot, never having "
                     "moved, stops another mover. One parking spot, two "
                     "support roles — the one-robot-one-job rule has no "
                     "words for this: here the plan language runs out."))
        elif g == "target_clears":
            add(e["cleared_t"], (
                f"The target robot steps aside, vacating its start cell "
                f"{tuple(e['cell'])} — move {e['needed_t'] + 1} needs to "
                "pass through it. A step-aside by the target has no plan "
                "word (route legs are shortest paths; this detour cannot "
                "be a plan node): the plan language runs out of words "
                "here."))
            add(e["needed_t"], (
                f"Here is why the target had to move: this slide uses the "
                f"target's start cell {tuple(e['cell'])}."))
        elif g == "target_support":
            add(e["t"], (
                f"The slide is stopped by the target robot itself at "
                f"{tuple(e['cell'])}. The vocabulary reserves the target "
                "robot for reaching the goal — it can never be scheduled as "
                "a stopper: the plan language runs out of words here."))
        elif g == "idle_clearing":
            add(e["moved_t"], (
                f"This robot is simply in the way: it steps aside so that "
                f"cell {tuple(e['cell'])} is free for move "
                f"{e['needed_t'] + 1}. The subgoal cost model assumes "
                "blockers dissolve for free, and the vocabulary has no "
                "“move aside” word — here the plan language runs "
                "out of words."))
    return notes


def rep_viewer_html(rec, uid):
    """Step-through viewer for one representative instance."""
    row = rec["row"]
    sc = SCALES[rec["scale"]]
    env_dir, grid = sc["env_dir"], sc["grid"]
    path_key = rec["path_key"]
    states = replay_states(row, env_dir, grid, path_key)
    analysis = rec["analysis"]
    moves = analysis["moves"]
    holdable = wall_holdable_cells(env_dir, row["env_id"], grid)
    wr, wd = board_walls(env_dir, row["env_id"], grid)
    tidx = row["target_idx"]
    notes = gate_annotations(rec["family"], analysis, row, env_dir, grid)
    tgt = tuple(row["target"])
    n = len(moves)

    steps = []
    for t in range(n + 1):
        pos = states[t]
        robots = [(x, y, SLOT_VARS[j], SLOT_NAMES[j][0], j == tidx)
                  for j, (x, y) in enumerate(pos)]
        arrow = None
        marks = []
        if t < n:
            m = moves[t]
            arrow = (m["from"][0], m["from"][1], m["to"][0], m["to"][1],
                     SLOT_VARS[m["slot"]])
            if m["stopper_slot"] is not None:
                cell = m["support_cell"]
                var = "--bad" if cell not in holdable else "--baseline"
                what = ("stopper cell — no adjacent wall"
                        if cell not in holdable else "stopper cell")
                marks.append((cell[0], cell[1], var, what))
        svg = board_svg(grid, wr, wd, robots, arrow=arrow,
                        ring=(tgt[0], tgt[1], SLOT_VARS[tidx]),
                        marks=marks,
                        aria=f"board after {t} of {n} moves")
        if t < n:
            m = moves[t]
            who = SLOT_NAMES[m["slot"]]
            if m["stopper_slot"] is not None:
                stop_txt = (f"stops against {SLOT_NAMES[m['stopper_slot']]} "
                            f"at {tuple(m['to'])}")
            else:
                stop_txt = f"stops at the wall at {tuple(m['to'])}"
            cap = (f"Move {t + 1} of {n}: {who} slides {m['dir']} from "
                   f"{tuple(m['from'])} and {stop_txt}.")
        else:
            cap = (f"Done in {n} moves: {SLOT_NAMES[tidx]} (the target robot) "
                   f"stands on the goal cell {tgt}.")
        note_html = "".join(
            f'<p class="gatenote">{esc(x)}</p>' for x in notes.get(t, []))
        gate_cls = " gatestep" if t in notes else ""
        steps.append(
            f'<div class="step{gate_cls}" data-step="{t}" '
            f'{"hidden" if t else ""}>{svg}'
            f'<p class="stepcap">{esc(cap)}</p>{note_html}</div>')

    gate_steps = ",".join(str(t) for t in sorted(notes))
    return f"""
<div class="viewer" id="{uid}" data-steps="{n + 1}" data-gates="{gate_steps}"
     tabindex="0" role="group"
     aria-label="move-by-move replay, {n} moves; use the buttons or arrow keys">
  {''.join(steps)}
  <div class="vbar">
    <button type="button" class="vbtn" data-act="prev" aria-label="previous move">&#8592; Prev</button>
    <span class="vpos mono" aria-live="polite">start</span>
    <button type="button" class="vbtn" data-act="next" aria-label="next move">Next &#8594;</button>
    <span class="vhint small muted">arrow keys work too; highlighted steps carry the annotation</span>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# family sections
# ---------------------------------------------------------------------------

FAMILY_MECHANISM = {
    "transient_support": (
        "The winning line needs a robot to act as a stopper on a cell that "
        "has no adjacent wall. Such a robot can only hold that cell because "
        "something else stops it there — another robot, placed first. The "
        "subgoal vocabulary admits only wall-holdable cells as parking "
        "spots (a support must survive on its own), so no plan can even be "
        "written down. This is the mechanism of the worked example in the "
        "B1 design note: pass through the cell first, then park a robot "
        "behind — the cell is empty at one moment and occupied later."),
    "vacate": (
        "A stopper does its job — a slide bounces off it — and then has to "
        "move away, because the cell it occupies lies on a route (or is a "
        "destination) that a later move needs. The vocabulary pins every "
        "placed helper to its support cell forever; there is no way to say "
        "“stop the slide, then step aside.”"),
    "relocate": (
        "One robot has to serve as a stopper twice, at two different "
        "places, one after the other, moving in between. The plan language "
        "gives each robot at most one job (one leaf, one placement), so the "
        "second recruitment cannot be expressed."),
    "shared_support": (
        "One robot parks once and, without ever moving again, stops the "
        "slides of two different movers. Physically it is one parking spot "
        "doing double duty; in the plan language it is two support roles, "
        "and the one-robot-one-job rule means the second role cannot be "
        "assigned. (Sharing is expressible only in the narrow case where "
        "the second use rides an already-scheduled support as a route "
        "claim, which these puzzles provably cannot arrange.)"),
    "target_support": (
        "The solution uses the target robot itself as a stopper for a "
        "helper's slide, before the target goes on to the goal. In the "
        "vocabulary the target robot always owns the plan's root role and "
        "can never be scheduled as a support."),
    "target_clears": (
        "Another robot's delivery route runs through the target robot's "
        "own starting cell, so the target must first step aside and later "
        "come back through where it started. The extra step-aside move has "
        "no plan word: route legs are always shortest paths, and a detour "
        "stop can only enter a plan as a supported bottleneck — which this "
        "detour is not."),
    "idle_clearing": (
        "A robot that never serves as a stopper is standing in the way and "
        "must first move off a route the plan needs. The subgoal cost model "
        "assumes such blockers slide away for free (“blocker "
        "clearing”), but the vocabulary has no move that actually "
        "clears them — so every complete plan the search builds dies in "
        "strict playback."),
    "blocked_direct": (
        "On paper the target robot can reach the goal by walls-only slides, "
        "so the plan builder immediately pins that walk as the whole plan — "
        "a segment with an exact route is never decomposed further, and "
        "nothing else is ever proposed. In the real game a robot stands on "
        "the walk. The plan is one sentence long and wrong, and the "
        "language cannot utter a second one: no “step aside” word, "
        "no way to ask for a longer route. The frozen probes confirm the "
        "mechanism at both scales: this family is exactly the set of probe "
        "rows that built one complete plan and exhausted, and no other "
        "instance — solved or unsolved — has that signature."),
    "none": (
        "Every individual maneuver in the winning line is nameable: all "
        "stoppers stand on wall-holdable cells, nobody vacates, nobody "
        "serves twice, and no paper walk short-circuits the search. The "
        "obstruction sits one level deeper, in how plans are assembled: a "
        "claimed route may only name a stopper that is already part of the "
        "plan (the declared-stopper invariant), and the assembly order this "
        "puzzle needs cannot be reached. The exhaustive probe still proves "
        "no expressible plan plays out; this is reported as a residual "
        "rather than forced under a headline it does not match."),
}

FAMILY_B1_NOTE = {
    "transient_support": (
        "Covered by B1 Layer 1 (transient support cells): the wall filter "
        "is lifted for a new candidate type, and the existing recursion "
        "already delivers the robot that props up the wall-less cell."),
    "vacate": (
        "Not in Layer 1. Covered by the park (vacate/step-aside) "
        "augmentation, which schedules a costed post-bounce departure for "
        "a placed helper."),
    "relocate": (
        "Not covered by Layer 1 and outside the park augmentation as "
        "designed: re-recruiting a used helper is the deferred "
        "“relocate” half of Layer 2."),
    "shared_support": (
        "Not covered by B1 as designed: neither transient cells nor parks "
        "let one robot hold two support roles — that is the one-leaf "
        "invariant, untouched by the extension."),
    "target_support": (
        "Not covered by B1 as designed: parks refuse mover robots, and the "
        "one-leaf invariant keeps the target robot out of support roles."),
    "target_clears": (
        "Not covered by B1 as designed: parks refuse mover robots, so the "
        "target's step-aside cannot be scheduled."),
    "idle_clearing": (
        "Not in Layer 1. Covered by the park (step-aside) augmentation "
        "when one park suffices; plans needing several parks are outside "
        "the bounded repair as designed."),
    "blocked_direct": (
        "Covered by the park (step-aside) augmentation when moving one "
        "robot off the pinned walk unblocks it — exactly the repair rule "
        "the augmentation implements (find the robot whose one-robot "
        "removal unblocks the failing slide; park it aside at full move "
        "cost)."),
    "none": (
        "Unclear: B1 does not change plan-assembly order, so coverage "
        "cannot be claimed from the design; the B1 re-probe measures it."),
}


def pick_representative(recs_by_family, family):
    """Deterministic: prefer base scale, standard-budget solve, a pure
    single-gate signature, then the shortest solution, then lowest idx."""
    cands = recs_by_family.get(family, [])
    cands = [r for r in cands if "analysis" in r]

    def rank(r):
        row = r["row"]
        pure = 0 if r["analysis"]["gates"] == [family] else 1
        return (0 if r["scale"] == "base" else 1,
                0 if row.get("solved") else 1,
                pure,
                len(row[r["path_key"]]),
                row["idx"])
    return min(cands, key=rank) if cands else None


# ---------------------------------------------------------------------------
# page assembly
# ---------------------------------------------------------------------------

CSS = """
:root{
  --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --muted:#52514e;
  --faint:#898781; --grid:#e1e0d9; --baseline:#c3c2b7; --line:rgba(11,11,11,.10);
  --fwd:#2a78d6; --bwd:#eb6834; --fwd-ink:#1c5cab; --bwd-ink:#b84a1a;
  --bad:#bf3f3f; --warn:#a87f1f; --ok:#2e7d52; --hl:rgba(127,127,110,.09);
  --r-red:#d9480f; --r-blue:#1c7ed6; --r-green:#2f9e44; --r-yellow:#c9971c;
  --r-purple:#9c36b5; --r-orange:#e8590c;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --maxw:1080px;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0d0d0d; --surface:#1a1a19; --ink:#f2f1ec; --muted:#c3c2b7;
    --faint:#898781; --grid:#2c2c2a; --baseline:#383835; --line:rgba(255,255,255,.10);
    --fwd:#3987e5; --bwd:#d95926; --fwd-ink:#86b6ef; --bwd-ink:#f08a5c;
    --bad:#e06c6c; --warn:#b98d2e; --ok:#58c48a; --hl:rgba(200,200,180,.07);
    --r-red:#ff8a4c; --r-blue:#4dabf7; --r-green:#51cf66; --r-yellow:#ffd43b;
    --r-purple:#da77f2; --r-orange:#ffa94d;
  }
}
:root[data-theme="light"]{
  --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --muted:#52514e;
  --faint:#898781; --grid:#e1e0d9; --baseline:#c3c2b7; --line:rgba(11,11,11,.10);
  --fwd:#2a78d6; --bwd:#eb6834; --fwd-ink:#1c5cab; --bwd-ink:#b84a1a;
  --bad:#bf3f3f; --warn:#a87f1f; --ok:#2e7d52; --hl:rgba(127,127,110,.09);
  --r-red:#d9480f; --r-blue:#1c7ed6; --r-green:#2f9e44; --r-yellow:#c9971c;
  --r-purple:#9c36b5; --r-orange:#e8590c;
}
:root[data-theme="dark"]{
  --bg:#0d0d0d; --surface:#1a1a19; --ink:#f2f1ec; --muted:#c3c2b7;
  --faint:#898781; --grid:#2c2c2a; --baseline:#383835; --line:rgba(255,255,255,.10);
  --fwd:#3987e5; --bwd:#d95926; --fwd-ink:#86b6ef; --bwd-ink:#f08a5c;
  --bad:#e06c6c; --warn:#b98d2e; --ok:#58c48a; --hl:rgba(200,200,180,.07);
  --r-red:#ff8a4c; --r-blue:#4dabf7; --r-green:#51cf66; --r-yellow:#ffd43b;
  --r-purple:#da77f2; --r-orange:#ffa94d;
}
*{box-sizing:border-box}
html,body{max-width:100%;overflow-x:hidden}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  line-height:1.6;font-size:16px;-webkit-font-smoothing:antialiased}
.wrap{max-width:var(--maxw);margin:0 auto;padding:0 22px 60px}
section{padding:30px 0;border-top:1px solid var(--grid)}
[id]{scroll-margin-top:20px}
h1,h2,h3{line-height:1.18;margin:0}
h1{font-size:clamp(26px,4.2vw,40px);font-weight:750;letter-spacing:-.02em}
h2{font-size:clamp(20px,2.8vw,26px);font-weight:700;letter-spacing:-.015em;margin-bottom:6px}
h3{font-size:17px;font-weight:680;margin:18px 0 4px}
p{margin:.6em 0;max-width:74ch}
a{color:var(--fwd-ink);text-decoration:none;border-bottom:1px solid var(--line)}
a:hover{border-bottom-color:var(--fwd-ink)}
a:focus-visible,button:focus-visible,.viewer:focus-visible{outline:2px solid var(--fwd);outline-offset:2px;border-radius:3px}
.muted{color:var(--muted)} .small{font-size:13px}
.mono{font-family:var(--mono);font-size:.92em}
.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
header{padding:44px 0 6px}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.15em;
  text-transform:uppercase;color:var(--muted);display:flex;gap:9px;
  align-items:center;margin-bottom:14px}
.eyebrow .dot{width:6px;height:6px;border-radius:50%;background:var(--bwd)}
.lede{font-size:clamp(16px,1.9vw,18.5px);color:var(--muted);max-width:72ch;margin-top:14px}
.lede b{color:var(--ink)}
.kicker{font-family:var(--mono);font-size:12px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--faint);margin-bottom:14px}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:14px;
  padding:18px;margin:10px 0}
.callout{background:var(--surface);border:1px solid var(--line);
  border-left:3px solid var(--baseline);border-radius:10px;padding:12px 16px;
  margin:12px 0;font-size:14.5px;color:var(--muted);max-width:80ch}
.callout.warn{border-left-color:var(--warn)}
.callout b{color:var(--ink)}
.scroll{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;width:100%;font-size:14px;background:var(--surface);
  border:1px solid var(--line);border-radius:10px}
th{font-family:var(--mono);font-size:11px;letter-spacing:.05em;text-transform:uppercase;
  color:var(--muted);text-align:left;padding:9px 12px;border-bottom:1px solid var(--baseline);
  white-space:nowrap}
td{padding:8px 12px;border-bottom:1px solid var(--grid);vertical-align:top}
tr:last-child td{border-bottom:0}
.chips{display:flex;flex-wrap:wrap;gap:7px;margin:10px 0}
.chip{font-family:var(--mono);font-size:11.5px;background:var(--surface);
  border:1px solid var(--line);border-radius:20px;padding:3px 11px;color:var(--muted)}
.viewer{background:var(--surface);border:1px solid var(--line);border-radius:14px;
  padding:16px;margin:14px 0;max-width:480px}
.step .stepcap{font-size:14px;margin:.5em 0 0}
.gatenote{font-size:14px;color:var(--ink);background:var(--hl);
  border-left:3px solid var(--bad);border-radius:6px;padding:8px 12px;margin:.5em 0 0}
.vbar{display:flex;align-items:center;gap:10px;margin-top:12px;flex-wrap:wrap}
.vbtn{font:inherit;font-size:14px;padding:5px 14px;border-radius:8px;
  border:1px solid var(--baseline);background:var(--bg);color:var(--ink);cursor:pointer}
.vbtn:disabled{opacity:.4;cursor:default}
.vpos{font-size:12.5px;color:var(--muted);min-width:64px;text-align:center}
.vhint{flex-basis:100%;color:var(--faint)}
.famgrid{display:grid;grid-template-columns:minmax(0,480px) minmax(260px,1fr);
  gap:20px;align-items:start}
@media (max-width:840px){.famgrid{grid-template-columns:1fr}}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin:8px 0 2px}
.lkey{font-size:13px;color:var(--muted);display:inline-flex;align-items:center;gap:6px}
.ldot{display:inline-block;width:10px;height:10px;border-radius:50%}
footer{padding:26px 0;color:var(--faint);font-size:13px;border-top:1px solid var(--grid)}
ul{max-width:78ch}
li{margin:.35em 0}
"""

JS = """
document.querySelectorAll('.viewer').forEach(function (v) {
  var steps = Array.prototype.slice.call(v.querySelectorAll('.step'));
  var pos = v.querySelector('.vpos');
  var prev = v.querySelector('[data-act="prev"]');
  var next = v.querySelector('[data-act="next"]');
  var cur = 0;
  function show(i) {
    cur = Math.max(0, Math.min(steps.length - 1, i));
    steps.forEach(function (s, j) { s.hidden = (j !== cur); });
    pos.textContent = cur === 0 ? 'start' : cur + ' / ' + (steps.length - 1);
    prev.disabled = (cur === 0);
    next.disabled = (cur === steps.length - 1);
  }
  prev.addEventListener('click', function () { show(cur - 1); });
  next.addEventListener('click', function () { show(cur + 1); });
  v.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { show(cur + 1); e.preventDefault(); }
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { show(cur - 1); e.preventDefault(); }
    if (e.key === 'Home') { show(0); e.preventDefault(); }
    if (e.key === 'End') { show(steps.length - 1); e.preventDefault(); }
  });
  show(0);
});
"""


def fmt_pct(a, b):
    return f"{100.0 * a / b:.1f}%"


def build():
    data = {k: load_scale_data(k) for k in SCALES}
    base, six = data["base"], data["g16r6"]

    # ---- headline numbers (all computed) ----
    n_base_struct = check("base structural count",
                          base["cats"]["NO_COMPLETE_PLAN"]
                          + base["cats"]["NO_REALIZABLE_PLAN"],
                          len(base["recs"]))
    n_six_struct = check("6-robot structural count",
                         six["cats"]["NO_COMPLETE_PLAN"]
                         + six["cats"]["NO_REALIZABLE_PLAN"],
                         len(six["recs"]))
    base_bench = base["bench_n"]
    six_bench = six["bench_n"]
    base_ceiling = fmt_pct(base_bench - n_base_struct, base_bench)
    six_ceiling = fmt_pct(six_bench - n_six_struct, six_bench)

    # ---- families ----
    recs_by_family = {}
    for d in data.values():
        for r in d["recs"]:
            recs_by_family.setdefault(r["family"], []).append(r)

    fam_counts = {}
    for fam, rs in recs_by_family.items():
        fam_counts[fam] = {
            "base": sum(1 for r in rs if r["scale"] == "base"),
            "g16r6": sum(1 for r in rs if r["scale"] == "g16r6"),
        }

    families = [f for f in FAMILY_ORDER if f in recs_by_family]
    extra = [f for f in recs_by_family
             if f not in FAMILY_ORDER and f not in ("unsolved",)]
    families += sorted(extra)

    unsolved = recs_by_family.get("unsolved", [])

    # ---- forward-planner scoreboard on the structural set ----
    def fwd_stats(d):
        rows = [r["row"] for r in d["recs"]]
        s1 = sum(1 for r in rows if r.get("solved"))
        s4 = sum(1 for r in rows if r.get("fallback_solved"))
        graded = [r for r in rows if r.get("solved")
                  and r.get("mode", "graded") == "graded"]
        reg = ([r["moves"] - r["d_star"] for r in graded]
               if graded else [])
        exp = [r["expansions"] for r in rows if r.get("solved")]
        return {
            "n": len(rows), "solved": s1, "fallback_extra": s4,
            "mean_regret": (sum(reg) / len(reg)) if reg else None,
            "n_graded_solved": len(graded),
            "mean_exp": (sum(exp) / len(exp)) if exp else None,
        }

    fb, fs = fwd_stats(base), fwd_stats(six)
    check("base forward solved (page vs rows)", fb["solved"],
          sum(1 for r in jload(SCALES['base']['solutions'])["rows"]
              if r.get("solved")))
    check("6-robot forward solved (page vs rows)", fs["solved"],
          sum(1 for r in jload(SCALES['g16r6']['solutions'])["rows"]
              if r.get("solved")))

    # ---- sections ----
    built = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = []
    parts.append(f"""
<header class="wrap">
  <div class="eyebrow"><span class="dot"></span>failure gallery — the backward planner's structural ceiling</div>
  <h1>The puzzles no subgoal plan can say</h1>
  <p class="lede">The backward planner writes its solutions in a small vocabulary:
  “park helper H on support cell S so the slider stops on bottleneck cell B.”
  An exhaustive, network-free probe proved that for
  <b>{n_base_struct} of {base_bench}</b> benchmark puzzles at the base scale
  ({fmt_pct(n_base_struct, base_bench)}) and
  <b>{n_six_struct} of {six_bench}</b> at 6 robots
  ({fmt_pct(n_six_struct, six_bench)}) <b>no playable plan exists in that
  vocabulary at all</b> — capping the backward planner at
  <b>{base_ceiling}</b> and <b>{six_ceiling}</b> respectively, no matter how
  well its networks are trained. This page shows what those puzzles actually
  need: the forward (move-by-move) planner solved
  {fb['solved'] + fs['solved']} of the {n_base_struct + n_six_struct} for
  real, and each family below steps through one such solution up to the
  exact moment the subgoal language runs out of words.</p>
</header>""")

    # intro / method
    parts.append(f"""
<section class="wrap" id="method">
  <div class="kicker">how to read this page</div>
  <h2>Where the verdicts and the solutions come from</h2>
  <p>Two independent measurements meet here.</p>
  <ul>
    <li><b>The impossibility verdicts</b> come from the ceiling probe: the
    hand-coded subgoal search run exhaustively (no neural network, so guidance
    is not the bottleneck), which either finds a strictly-playable plan or
    proves none exists. Verdicts split into <span class="mono">NO_COMPLETE_PLAN</span>
    (no complete subgoal plan can even be built:
    {base['cats']['NO_COMPLETE_PLAN']} base / {six['cats']['NO_COMPLETE_PLAN']}
    six-robot) and <span class="mono">NO_REALIZABLE_PLAN</span> (plans exist
    but none survives legal playback: {base['cats']['NO_REALIZABLE_PLAN']} /
    {six['cats']['NO_REALIZABLE_PLAN']}). The verdicts are cap-independent:
    re-probed at 4&times; the plan-width cap, {base['cap_still_structural']} of
    {base['cap_n']} base and {six['cap_still_structural']} of {six['cap_n']}
    six-robot <span class="mono">NO_COMPLETE_PLAN</span> instances stayed
    structural failures. Probed failures that are NOT structural are excluded
    from this page: {base['cats'].get('REALIZABLE_EXISTS', 0)} base and
    {six['cats'].get('REALIZABLE_EXISTS', 0)} six-robot puzzles have a playable
    plan the networks missed (guidance headroom, not a language limit), and
    {six['cats'].get('INCONCLUSIVE', 0)} six-robot probes hit their time cap
    without a verdict.</li>
    <li><b>The solutions</b> come from the forward move planner — a pure
    network + search system (no oracle) — run on exactly these puzzles at its
    standard operating point (top-5 proposals, 1200-expansion budget). Every
    move sequence shown replays legally, move by move, under the ground-truth
    slide simulator; the replay is re-checked when this page is built.</li>
  </ul>
  <p>The subgoal vocabulary has four hard rules, and each failure family below
  is a real maneuver that breaks one of them:</p>
  <ul>
    <li><b>Rule 1 — supports need walls.</b> A helper may only be parked on a
    cell where a wall can hold it (a cell reachable by a walls-only slide).</li>
    <li><b>Rule 2 — supports are forever.</b> Once parked, a helper never
    moves again.</li>
    <li><b>Rule 3 — one robot, one job.</b> Each robot fills at most one plan
    role; the target robot always fills the “reach the goal” role.</li>
    <li><b>Rule 4 — blockers are assumed to dissolve.</b> Route costs are
    computed as if robots in the way slide off for free; the vocabulary has no
    actual “step aside” move.</li>
  </ul>
  <div class="callout"><b>One honest caveat.</b> The impossibility verdicts are
  solution-independent (exhaustive proof). The family assignment is not: it is
  read off the one solution the forward planner found, so it names the missing
  word <i>that solution</i> needed. Different solutions of the same puzzle
  could stress different missing words; the families are evidence about what
  the vocabulary lacks, not unique labels per puzzle.</div>
</section>""")

    # families overview table
    fam_rows = []
    for fam in families:
        c = fam_counts.get(fam, {"base": 0, "g16r6": 0})
        fam_rows.append(
            f"<tr><td><b>{esc(FAMILY_LABEL[fam])}</b></td>"
            f"<td class='num'>{c['base']}</td>"
            f"<td class='num'>{c['g16r6']}</td>"
            f"<td>{esc(FAMILY_B1_NOTE.get(fam, ''))}</td></tr>")
    c_un = fam_counts.get("unsolved", {"base": 0, "g16r6": 0})
    if unsolved:
        fam_rows.append(
            f"<tr><td class='muted'>{esc(FAMILY_LABEL['unsolved'])}</td>"
            f"<td class='num'>{c_un['base']}</td>"
            f"<td class='num'>{c_un['g16r6']}</td>"
            f"<td class='muted'>counted honestly below; no family assigned</td></tr>")
    parts.append(f"""
<section class="wrap" id="families">
  <div class="kicker">the taxonomy</div>
  <h2>Failure families, counted at both scales</h2>
  <p>Each structural instance is assigned to the hardest missing word its
  forward solution needed (an instance whose solution needs several missing
  words is counted once, under the one that decides whether the planned
  vocabulary extension covers it). Columns sum to {n_base_struct} (base) and
  {n_six_struct} (6 robots).</p>
  <div class="scroll"><table>
    <thead><tr><th>family</th><th>base</th><th>6 robots</th>
    <th>does the B1 vocabulary extension cover it?</th></tr></thead>
    <tbody>{''.join(fam_rows)}</tbody>
  </table></div>
  <p class="small muted">B1 (design: analysis/b1_design.md) extends the plan
  vocabulary in two layers: Layer 1 admits wall-less “transient”
  support cells; Layer 2 adds bounded “park” moves (a placed helper or
  idle robot steps aside, at full move cost). Re-recruiting a used helper and
  scheduling the target robot as a support are outside B1 as designed.</p>
</section>""")

    # per-family sections
    uid = 0
    for fam in families:
        rep = pick_representative(recs_by_family, fam)
        c = fam_counts.get(fam, {"base": 0, "g16r6": 0})
        if rep is None:
            parts.append(f"""
<section class="wrap" id="fam-{fam}">
  <h2>{esc(FAMILY_LABEL[fam])}</h2>
  <p>{esc(FAMILY_MECHANISM.get(fam, ''))}</p>
  <div class="callout warn"><b>No representative to show.</b> The forward
  planner solved none of this family's instances, so there is no move
  sequence to step through. Counts: {c['base']} base, {c['g16r6']}
  six-robot.</div>
</section>""")
            continue
        uid += 1
        row = rep["row"]
        sc = SCALES[rep["scale"]]
        viewer = rep_viewer_html(rep, f"viewer{uid}")   # raises on any
        # illegal replay (replay_states validates under simulate.slide)
        REP_VALIDATIONS.append(
            f"{fam}: idx {row['idx']} ({rep['scale']}) replays "
            f"{len(row[rep['path_key']])} moves legally under simulate.slide")
        d_star_txt = ""
        if row.get("mode", "graded") == "graded" and row.get("d_star"):
            moves_n = len(row[rep["path_key"]])
            d_star_txt = (f" The exact optimum for this puzzle is "
                          f"{row['d_star']} moves"
                          + (" — the forward solution shown is optimal."
                             if moves_n == row["d_star"] else
                             f"; the solution shown takes {moves_n}."))
        budget_note = ""
        if rep["path_key"] == "fallback_path":
            budget_note = (f" The forward planner needed the 4&times; fallback "
                           f"budget ({row['fallback_budget']} expansions) for "
                           f"this one — recorded honestly.")
        other_gates = [g for g in rep["analysis"]["gates"] if g != fam]
        other_txt = ""
        if other_gates:
            other_txt = (" This instance's solution also touches: "
                         + ", ".join(FAMILY_LABEL[g].lower()
                                     for g in other_gates) + ".")
        parts.append(f"""
<section class="wrap" id="fam-{fam}">
  <div class="kicker">family — {esc(FAMILY_LABEL[fam]).lower()}</div>
  <h2>{esc(FAMILY_LABEL[fam])}</h2>
  <div class="chips">
    <span class="chip">base: {c['base']} of {n_base_struct}</span>
    <span class="chip">6 robots: {c['g16r6']} of {n_six_struct}</span>
    <span class="chip">B1: {esc(rep['b1'])}</span>
  </div>
  <div class="famgrid">
    <div>{viewer}</div>
    <div>
      <p>{esc(FAMILY_MECHANISM.get(fam, ''))}</p>
      <h3>The example</h3>
      <p>Benchmark instance <span class="mono">idx {row['idx']}</span>
      (board <span class="mono">env {row['env_id']}</span>,
      {esc(sc['label'])}), probe verdict
      <span class="mono">{esc(row['category'])}</span>.
      The forward planner solved it in {len(row[rep['path_key']])} moves.
      {esc(d_star_txt)}{budget_note}{esc(other_txt)}</p>
      <p class="small muted">{esc(FAMILY_B1_NOTE.get(fam, ''))}</p>
    </div>
  </div>
</section>""")

    # forward performance + honest unsolved note
    def perf_row(d, st):
        proto = d["protocol"]
        reg = "--" if st["mean_regret"] is None else format(st["mean_regret"], ".2f")
        mexp = "--" if st["mean_exp"] is None else format(st["mean_exp"], ".0f")
        return (f"<tr><td>{esc(d['cfg']['label'])}</td>"
                f"<td class='num'>{st['solved']}/{st['n']}</td>"
                f"<td class='num'>{fmt_pct(st['solved'], st['n'])}</td>"
                f"<td class='num'>{st['fallback_extra']}</td>"
                f"<td class='num'>{reg}"
                f"<div class='small muted'>over {st['n_graded_solved']} graded solves</div></td>"
                f"<td class='num'>{mexp}</td>"
                f"<td class='mono small'>{esc(Path(proto['ckpt']).name)}</td></tr>")

    unsolved_note = ""
    if unsolved:
        groups = {}   # (scale, fallback-status) -> rows
        for r in unsolved:
            row = r["row"]
            if row.get("fallback_solved"):
                st = f"solved at the 4x budget"
            elif "fallback_budget" in row:
                st = "also unsolved at the 4x budget"
            else:
                st = ("4x fallback not attempted — none is a gallery "
                      "representative (every family has a solved example)")
            groups.setdefault((r["scale"], st), []).append(row)
        items = []
        for (scale, st), rws in sorted(groups.items()):
            cats = Counter(r["category"] for r in rws)
            cat_txt = ", ".join(f"{v} {k}" for k, v in sorted(cats.items()))
            ids = ", ".join(
                f"{r['idx']}" + (f"/{r['tag']}" if r.get("tag") else "")
                for r in rws)
            items.append(
                f"<li><b>{len(rws)}</b> at {esc(SCALES[scale]['label'])} "
                f"({esc(cat_txt)}); {esc(st)}. Instance idx: "
                f"<span class='mono small'>{esc(ids)}</span></li>")
        unsolved_note = (
            "<div class='callout warn'><b>Honest note — instances the forward "
            "planner also failed to solve.</b> These are structural failures "
            "of the backward vocabulary where the forward planner, at the "
            "budgets tried here, found no solution either; they are counted "
            "in the tables above as “forward planner also failed” and "
            "no mechanism is claimed for them.<ul class='small'>"
            + "".join(items) + "</ul></div>")

    parts.append(f"""
<section class="wrap" id="forward">
  <div class="kicker">the other planner's scorecard</div>
  <h2>How the forward planner did on these puzzles</h2>
  <p>These {n_base_struct + n_six_struct} puzzles are provably outside the
  backward planner's vocabulary, so its solve rate on them is 0% by
  construction. The forward planner, at its standard operating point
  (top-5, 1200 expansions), solved:</p>
  <div class="scroll"><table>
    <thead><tr><th>scale</th><th>solved</th><th>rate</th>
    <th>extra at 4&times; budget</th><th>mean extra moves</th>
    <th>mean expansions (solved)</th><th>checkpoint</th></tr></thead>
    <tbody>{perf_row(base, fb)}{perf_row(six, fs)}</tbody>
  </table></div>
  <p class="small muted">“Mean extra moves” compares against the exact
  optimum d*, which exists only for oracle-graded instances; beyond-oracle
  rows (6-robot set) carry placeholder optima and are excluded. Expansion
  counts use the same definition as the head-to-head protocol (one popped
  node whose children are generated).</p>
  {unsolved_note}
</section>""")

    # provenance
    prov_files = []
    for d in data.values():
        for k in ("probe", "instances", "cap", "solutions"):
            prov_files.append(d["cfg"][k])
    prov_files += ["eval/data/bench450.jsonl",
                   "scaling/results/g16r6/comparison.json",
                   "scaling/results/g16r6/comparison_ungraded.json",
                   "analysis/b1_design.md"]
    prov_lis = []
    for f in prov_files:
        p = ROOT / f
        stamp = (datetime.fromtimestamp(p.stat().st_mtime)
                 .strftime("%Y-%m-%d %H:%M") if p.exists() else "missing")
        prov_lis.append(f"<li><span class='mono'>{esc(f)}</span> "
                        f"<span class='muted small'>({stamp})</span></li>")
    fwd_protos = []
    for d in data.values():
        pr = d["protocol"]
        fwd_protos.append(
            f"<li>{esc(d['cfg']['label'])}: checkpoint "
            f"<span class='mono'>{esc(pr['ckpt'])}</span>, top-{pr['k']}, "
            f"budget {pr['budget']} (fallback {pr['fallback_budget']}), run "
            f"{esc(pr['date'])}.</li>")
    parts.append(f"""
<section class="wrap" id="provenance">
  <div class="kicker">provenance</div>
  <h2>Sources</h2>
  <p>This page is generated by
  <span class="mono">analysis/failure_gallery/build_gallery.py</span>. Every
  number is computed at build time from the files below; every stepped-through
  solution is replay-validated under <span class="mono">simulate.slide</span>
  during the build. Built {esc(built)}.</p>
  <ul class="small">{''.join(prov_lis)}</ul>
  <h3>Forward-planner runs</h3>
  <ul class="small">{''.join(fwd_protos)}</ul>
  <p class="small muted">Robot letters in the boards abbreviate the fixed slot
  order (Red, Blue, Green, Yellow, then Purple, Orange at 6 robots). The
  ringed robot is the target; the dashed square of its color is the goal
  cell. Dashed red squares mark stopper cells that have no adjacent wall.</p>
</section>
<footer class="wrap">Structural-failure gallery — generated from the pinned
probe artifacts and forward-planner runs; regenerate with
<span class="mono">PYTHONPATH=. python3 analysis/failure_gallery/build_gallery.py</span>.</footer>
""")

    page = ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Failure gallery — puzzles outside the subgoal vocabulary</title>"
            f"<style>{CSS}</style></head><body>"
            + "".join(parts)
            + f"<script>{JS}</script></body></html>")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(page)

    # ---- verification pass ----
    print(f"[gallery] wrote {OUT_PATH} ({len(page) / 1024:.0f} KiB)")
    print("[gallery] verification: page value vs artifact value")
    ok = True
    for desc, pv, av in CHECKS:
        good = pv == av
        ok &= good
        print(f"  {'ok ' if good else 'FAIL'} {desc}: {pv} vs {av}")
    # counts equal artifacts by construction; assert family totals
    tot_b = sum(v["base"] for v in fam_counts.values())
    tot_s = sum(v["g16r6"] for v in fam_counts.values())
    print(f"  {'ok ' if tot_b == n_base_struct else 'FAIL'} family totals base: "
          f"{tot_b} vs {n_base_struct}")
    print(f"  {'ok ' if tot_s == n_six_struct else 'FAIL'} family totals 6r: "
          f"{tot_s} vs {n_six_struct}")
    ok &= tot_b == n_base_struct and tot_s == n_six_struct
    # html parse check
    from html.parser import HTMLParser

    class _P(HTMLParser):
        def error(self, msg):  # pragma: no cover
            raise ValueError(msg)
    _P().feed(page)
    print("  ok  html parses (html.parser)")
    for v in REP_VALIDATIONS:
        print(f"  ok  {v}")
    if not ok:
        sys.exit("[gallery] VERIFICATION FAILED")
    print("[gallery] all checks passed")


if __name__ == "__main__":
    build()
