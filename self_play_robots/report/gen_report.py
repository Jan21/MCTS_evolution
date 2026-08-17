"""Single-source-of-truth report page for the self-play track (self_play_robots).

Reads ONLY files already on disk — the project's own result JSONs, the frozen
supervised comparison JSONs, the pinned benchmark JSONLs and the status
manifest `results/status.json` — and writes `selfplay.html` next to itself.
Every number traces to a file read at generation time; a missing or partial
file renders as a "pending" cell, never a crash and never a hand-typed number.
Hand-typed text is limited to definitions and gate thresholds quoted from
PROBLEM.md, each with its section citation.

Regenerate any time (from anywhere — paths resolve from __file__):

    python self_play_robots/report/gen_report.py

LOCAL FILE ONLY — never published anywhere (owner's standing order,
PROBLEM.md section 10).
"""
from __future__ import annotations

import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent          # self_play_robots/report
SPR = HERE.parent                               # self_play_robots
REPO = SPR.parent                               # MCTS_evolution
SV = REPO / "supervised_valuenet"               # the supervised stack
RESULTS = SPR / "results"
OUT = HERE / "selfplay.html"

STATS = {"read": [], "missing": [], "error": []}
_CACHE: dict = {}

MILESTONE_KEYS = ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]

# Gate text quoted from PROBLEM.md section 8 (the only hand-typed numbers on
# the page besides the glossary definitions; each row cites its source).
GATES = {
    "M0": ("&asymp;1 nh",
           "Re-run the supervised g16r4 backward pair through the bench from "
           "this folder's harness; numbers must match the recorded comparison "
           "JSONs. Proves your wiring before anything novel."),
    "M1": ("&asymp;3&ndash;6 nh",
           "Train pe=none policy+value on existing exact corpora at g16r4 + "
           "g24r4; bench within seed noise of the per-size supervised pair. "
           "Gate: solve rate within 3.5 pts, optimality within 6.6 pts (the "
           "measured seed bars, &sect;4.7)."),
    "M2": ("&asymp;2&ndash;4 nh",
           "MCTS with the M1 nets (no retraining yet) vs greedy descent on the "
           "same instances/budget. Gate: strictly better mean moves at "
           "equal-or-better solve rate."),
    "M3": ("&asymp;5&ndash;10 nh",
           "Generate, certify, retrain once, bench. Gate: net k+1 beats net k "
           "beyond seed noise on the paired-instance test, AND the fidelity "
           "gauge (&sect;4.1) has not slipped."),
    "M4": ("&asymp;15&ndash;30 nh",
           "3&ndash;5 iterations at g16r4/g24r4. Gate: monotone-ish bench "
           "improvement; final net beats the supervised backward baseline "
           "beyond seed noise."),
    "M5": ("&asymp;10&ndash;25 nh",
           "Curriculum to g32r4 (and the g24r8 hard mode); if &sect;6.2's "
           "size-free bet holds, evaluate transfer: net trained via self-play "
           "at &le;32 benched zero-shot at 40&ndash;64 against exact ground "
           "truth."),
    "M6": ("stretch",
           "Self-play at 80/96 where only certification exists; the &le;64 "
           "fidelity curve is the warranty argument (same structure as "
           "FINDINGS 72c)."),
}

# PROBLEM.md section 5, the AlphaZero mapping table (quoted).
MAPPING = [
    ("Game rules",
     "<code>simulate.py</code> physics + lean boards (self-generated "
     "positions at will)"),
    ("Network f(s) &rarr; (p, v)",
     "policy over the chosen action space + value = cost-to-go estimate "
     "(&sect;6.2)"),
    ("MCTS with PUCT",
     "over subgoal decisions (recommended) or primitive moves (&sect;6.1); "
     "\"win\" is replaced by realized move count &rarr; minimize cost, so "
     "back up <em>negative realized cost</em> (or a cost-bucket "
     "distributional value like the existing HL-Gauss value nets)"),
    ("Self-play games",
     "solve generated instances with search; <strong>certify every solution "
     "by replay</strong>; realized primitive count is the episode's ground "
     "truth"),
    ("Training targets",
     "policy &larr; visit distribution at each decision; value &larr; "
     "realized cost-to-go from each visited state (certified labels only — "
     "exactly the descent labeler's contract, now with search instead of "
     "greed)"),
    ("Replay buffer",
     "rolling window over recent iterations' certified episodes; keep "
     "board/instance provenance for leakage control"),
    ("Evaluator / gating",
     "pinned bench arena (&sect;7) + fidelity gauge (&sect;4.1) + "
     "seed-noise-aware promotion (&sect;4.7)"),
]

# PROBLEM.md section 4 — "lessons you must not relearn".
LESSONS = [
    ("4.1 Label/value fidelity gates downstream utility", "FINDINGS 74",
     "The value targets are self-generated labels. Monitor their argmin "
     "agreement against exact optima (cheap &le;64) <em>every iteration</em>; "
     "a fidelity slide below ~90% predicts a utility slide before the bench "
     "shows it. Calibrated early-warning instrument, inherited free."),
    ("4.2 The full NN pipeline is proven end-to-end", "FINDINGS 79",
     "NN-made boards + instances + labels already trained a planner "
     "indistinguishable from the exact pipeline (83.4% vs 84.0% solve at "
     "g32r4). Do not re-litigate whether NN data can teach — the only open "
     "question is whether <em>search-improved</em> data teaches better."),
    ("4.3 Distilling from damaged data reproduces the damage", "FINDINGS 68",
     "Whatever the search does poorly, the data does poorly, and the next net "
     "learns it. Data-quality control (certified replay, fidelity gates, "
     "diversity checks) is the difference between improvement and a feedback "
     "loop of degradation — not hygiene."),
    ("4.4 Value-net training is bistable", "FINDINGS 44/50/61",
     "Wire in from day one: <code>val_group_spread</code> monitoring, "
     "<code>CollapseStop</code> (3 epochs spread &lt; 0.05 &rarr; stop), "
     "<code>ModelCheckpoint(min val_regret)</code>, warm-start from the "
     "previous iteration's net, low-lr rescue (1e-4) for &ge;6-robot "
     "configs. Training runs unattended every iteration. g24r8 is hard mode."),
    ("4.5 The supervised planner nets are NOT size-free", "FINDINGS 81",
     "<code>policy_tf.py</code> / <code>looped_pc.py</code> carry a learned "
     "per-cell table (1.23M params at 80&times;80): bootstrap checkpoints "
     "work only at their training size. This is what makes &sect;6.2 the "
     "project's first architectural decision. Also: value bins default to 50 "
     "(labels silently clamp), <code>--max-per-group 1</code> silently "
     "disables the ranking loss."),
    ("4.6 Certification is non-negotiable", "whole supervised track",
     "A backup value from an unrealized plan is a hypothesis; only realized, "
     "replayed plans update the record. This is also the defense against the "
     "self-play analogue of reward hacking (the net convincing itself illegal "
     "shortcuts work)."),
    ("4.7 Seed noise is large enough to fool you", "FINDINGS 67&rarr;71",
     "Measured same-arm spread ~3.4 solve / ~6.6 optimality points at g24r4. "
     "Promotion decisions inside the loop must clear the seed-noise bar — use "
     "large benches, paired-instance tests (McNemar over the per-instance "
     "solved vector, FINDINGS 78), or both."),
    ("4.8 Costs you can plan with", "measured, FINDINGS 81 brief",
     "Lean board 0.24 s at 32&times;32; descent labeling ~0.36 s/inst "
     "depth-0 at 32&times;32; planner pair train+bench 0.54 nh at g24r4, 1.6 "
     "nh at g32r4. Bad nets burn the full expansion budget — bench cost is "
     "itself a quality signal. <code>eval/end2end.py</code> has no "
     "<code>torch.no_grad()</code>: add it before any GPU eval."),
]

# The frozen opponents named in PROBLEM.md section 7, per configuration.
# (file selection only — every number comes from the file.)
OPPONENT_FILES = {
    "g16r4": ["eval/results/final450_backward_prefix.json",
              "eval/results/final450_backward_b2_seed21.json",
              "eval/results/comparison_forward.json"],
    "g24r4": ["scaling/results/g24r4/comparison.json",
              "scaling/results/g24r4/comparison_b2.json"],
    "g24r8": ["scaling/results/g24r8/comparison.json",
              "scaling/results/g24r8/comparison_ungraded.json"],
    "g32r4": ["scaling/results/g32r4/comparison.json",
              "scaling/results/g32r4/comparison_ungraded.json"],
}
GLOB_DIRS = {"g24r4": "scaling/results/g24r4",
             "g24r8": "scaling/results/g24r8",
             "g32r4": "scaling/results/g32r4"}
CFG_HUMAN = {"g16r4": "16&times;16 board, 4 robots &mdash; the legacy base450 exam",
             "g24r4": "24&times;24 board, 4 robots",
             "g24r8": "24&times;24 board, 8 robots (the hard mode)",
             "g32r4": "32&times;32 board, 4 robots"}
BENCH_FILES = ["eval/data/bench450.jsonl",
               "scaling/data/g24r4/bench.solved.jsonl",
               "scaling/data/g24r8/bench.solved.jsonl",
               "scaling/data/g32r4/bench.solved.jsonl"]


# ------------------------------------------------------------------ loaders --

def disp(p) -> str:
    """A path as shown on the page: repo-root-relative when possible."""
    p = Path(p)
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except Exception:
        return str(p)


def load(p) -> dict | None:
    """A JSON file, or None if absent. Never raises; records what it saw."""
    p = Path(p)
    key = str(p)
    if key in _CACHE:
        return _CACHE[key]
    if not p.is_file():
        STATS["missing"].append(disp(p))
        _CACHE[key] = None
        return None
    try:
        d = json.loads(p.read_text())
    except Exception as e:            # truncated mid-write etc. — show, don't die
        STATS["error"].append(f"{disp(p)}: {e}")
        d = {"_load_error": f"{disp(p)}: {e}"}
    else:
        STATS["read"].append(disp(p))
    if not isinstance(d, dict):
        d = {"_load_error": f"{disp(p)}: top-level JSON is not an object"}
    _CACHE[key] = d
    return d


def load_sv(rel) -> dict | None:
    """A JSON under supervised_valuenet/ (accepts absolute paths too)."""
    p = Path(rel)
    return load(p if p.is_absolute() else SV / rel)


def ok(d) -> bool:
    return bool(d) and "_load_error" not in d


def bench_stats(rel) -> dict | None:
    """n / n_graded / mean d* of a pinned benchmark JSONL (or None)."""
    if not rel:
        return None
    p = Path(rel)
    if not p.is_absolute():
        p = SV / rel
    key = "jsonl:" + str(p)
    if key in _CACHE:
        return _CACHE[key]
    if not p.is_file():
        STATS["missing"].append(disp(p))
        _CACHE[key] = None
        return None
    n = 0
    ds = []
    try:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                v = r.get("d_star")
                if v not in (None, 0):
                    ds.append(v)
    except Exception as e:
        STATS["error"].append(f"{disp(p)}: {e}")
        _CACHE[key] = None
        return None
    STATS["read"].append(disp(p))
    out = {"path": disp(p), "n": n, "n_graded": len(ds),
           "mean_d_star": (sum(ds) / len(ds)) if ds else None}
    _CACHE[key] = out
    return out


def status() -> dict:
    d = load(RESULTS / "status.json")
    return d if ok(d) else {}


def arena_arms():
    """spr.arena's ARMS registry (the single source of truth for M0 arms).

    Imported, not copied — arena.py is stdlib-only at import time. If the
    import fails the page still renders, from whatever landed on disk.
    """
    try:
        if str(SPR) not in sys.path:
            sys.path.insert(0, str(SPR))
        from spr.arena import ARMS            # noqa: PLC0415  (deliberate)
        return ARMS, None
    except Exception as e:
        return {}, f"{type(e).__name__}: {e}"


# ------------------------------------------------------------- html helpers --

def esc(x) -> str:
    return html.escape(str(x))


def td(v, spec=".3f", cls="") -> str:
    c = f' class="{cls}"' if cls else ""
    if v is None:
        return '<td class="pend">pending</td>'
    if isinstance(v, float):
        return f"<td{c}>{v:{spec}}</td>"
    return f"<td{c}>{esc(v)}</td>"


def td_txt(x, cls="") -> str:
    c = f' class="{cls}"' if cls else ""
    return f"<td{c}>{x}</td>"


def td_pct(v) -> str:
    """A 0..1 fraction as a percentage."""
    return '<td class="pend">pending</td>' if v is None else f"<td>{100 * v:.1f}%</td>"


def td_pct100(v) -> str:
    """A value already expressed in percent."""
    return '<td class="pend">pending</td>' if v is None else f"<td>{v:.1f}%</td>"


DASH = '<td class="dash">&mdash;</td>'


def row(cells, cls="") -> str:
    c = f' class="{cls}"' if cls else ""
    return f"<tr{c}>" + "".join(cells) + "</tr>"


def table(headers, rows_, note="", cls="") -> str:
    h = "".join(f"<th>{x}</th>" for x in headers)
    n = f'<p class="note">{note}</p>' if note else ""
    body = "".join(rows_) or row([f'<td class="pend" colspan="{len(headers)}">'
                                  "nothing on disk yet</td>"])
    k = f" {cls}" if cls else ""
    return (f'<div class="tw"><table class="t{k}"><thead><tr>{h}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>{n}")


CHIP_CLASS = {"pass": "good", "ok": "good", "done": "good", "fail": "bad",
              "failed": "bad", "running": "run", "queued": "run",
              "pending": "pend", "awaiting owner": "pend"}


def chip(kind, text=None) -> str:
    k = str(kind).strip().lower()
    cls = CHIP_CLASS.get(k, "pend")
    return f'<span class="chip {cls}">{esc(text if text is not None else kind)}</span>'


def pend_note(what) -> str:
    return (f'<p class="pendbox">Pending &mdash; no results on disk yet '
            f'({what}). This section fills in automatically on the next '
            f'regeneration after the job lands.</p>')


def src(relpath) -> str:
    return f'<code class="src">{esc(relpath)}</code>'


# --------------------------------------------------- comparison-payload bits --

def systems_of(payload):
    """[(name, kind, aggregate, n_rows)] of a comparison payload."""
    out = []
    if not ok(payload):
        return out
    sysd = payload.get("systems")
    if not isinstance(sysd, dict):
        return out
    for name, s in sysd.items():
        if not isinstance(s, dict):
            continue
        out.append((name, s.get("kind", "?"), s.get("aggregate") or {},
                    len(s.get("rows") or [])))
    return out


def backward_system(payload):
    """(name, system) of the first backward system carrying rows, or (None, None).

    Mirrors spr.arena._backward_system, minus the SystemExit.
    """
    if not ok(payload):
        return None, None
    for name, s in (payload.get("systems") or {}).items():
        if isinstance(s, dict) and s.get("kind") == "backward" and s.get("rows"):
            return name, s
    return None, None


def is_frontier(payload, aggregate) -> bool:
    """A frontier (beyond-oracle) set: no exact optimum exists for it."""
    if aggregate.get("d_star_placeholder"):
        return True
    b = bench_stats((payload.get("protocol") or {}).get("instances_file"))
    if b is not None and b["n"] and b["n_graded"] == 0:
        return True
    return False


def protocol_bits(payload) -> str:
    p = (payload.get("protocol") or {}) if ok(payload) else {}
    bits = []
    if p.get("expansions") is not None:
        bits.append(f"{p['expansions']} expansions")
    if p.get("k") is not None:
        bits.append(f"k={p['k']}")
    if p.get("n_instances") is not None:
        bits.append(f"n={p['n_instances']}")
    sha = p.get("instances_sha256")
    if sha:
        bits.append(f"exam sha {esc(str(sha)[:8])}")
    if p.get("device"):
        bits.append(esc(p["device"]))
    if p.get("date"):
        bits.append(esc(p["date"]))
    return " &middot; ".join(bits)


BASE_HEADERS = ["config", "set", "system", "solve rate", "mean realized moves",
                "mean regret", "% optimal", "mean expansions", "mean s/inst",
                "source file"]


def baseline_rows(cfg, rel, cls=""):
    """One table row per system inside one comparison payload."""
    d = load_sv(rel)
    if d is None:
        return [row([td_txt(esc(cfg)),
                     f'<td class="pend" colspan="8">file not on disk</td>',
                     td_txt(src(disp(SV / rel)))], cls)]
    if not ok(d):
        return [row([td_txt(esc(cfg)),
                     f'<td class="pend" colspan="8">unreadable: '
                     f'{esc(d["_load_error"])}</td>',
                     td_txt(src(disp(SV / rel)))], "bad")]
    out = []
    for name, kind, a, nrows in systems_of(d):
        if kind == "pending" or not a:
            out.append(row([td_txt(esc(cfg)), td_txt("&mdash;"),
                            td_txt(esc(name)),
                            '<td class="pend" colspan="6">'
                            f'no aggregate in file (kind={esc(kind)})</td>'],
                           cls))
            continue
        front = is_frontier(d, a)
        n, solved = a.get("n"), a.get("solved")
        sr = (f"{solved}/{n} &middot; {100 * a['solve_rate']:.1f}%"
              if None not in (n, solved) and a.get("solve_rate") is not None
              else None)
        out.append(row([
            td_txt(esc(cfg)),
            td_txt('<span class="tag front">frontier</span>' if front
                   else '<span class="tag graded">graded</span>'),
            td_txt(esc(name)),
            td_txt(sr) if sr else '<td class="pend">pending</td>',
            td(a.get("mean_moves"), ".2f", cls="hlnum"),
            DASH if front else td(a.get("mean_regret"), ".2f"),
            DASH if front else td_pct100(a.get("pct_optimal")),
            td(a.get("mean_expansions"), ".1f"),
            td(a.get("mean_seconds"), ".1f"),
            td_txt(src(disp(SV / rel))),
        ], cls))
    return out


def ceiling_row(cfg, bench_rel):
    """The exact-optimum row: mean d* over the graded bench file itself."""
    b = bench_stats(bench_rel)
    if b is None:
        return row([td_txt(esc(cfg)), td_txt('<span class="tag graded">graded</span>'),
                    td_txt("<strong>exact optimum d* (the ceiling)</strong>"),
                    '<td class="pend" colspan="6">bench file not on disk</td>'],
                   "ceil")
    return row([
        td_txt(esc(cfg)),
        td_txt('<span class="tag graded">graded</span>'),
        td_txt("<strong>exact optimum d* (the ceiling)</strong>"),
        td_txt(f"{b['n_graded']}/{b['n']} &middot; 100.0%"),
        td(b["mean_d_star"], ".2f", cls="hlnum"),
        td_txt("0.00"), td_txt("100.0%"), DASH, DASH,
        td_txt(src(b["path"])),
    ], "ceil")


# ------------------------------------------------------------ status helpers --

def milestone(key) -> dict:
    m = (status().get("milestones") or {}).get(key)
    return m if isinstance(m, dict) else {}


def side_study(key) -> dict:
    s = (status().get("side_studies") or {}).get(key)
    return s if isinstance(s, dict) else {}


def jobs_txt(entry) -> str:
    js = entry.get("jobs") or []
    return ", ".join(str(j) for j in js) if js else "&mdash;"


def sources_txt(entry) -> str:
    ss = entry.get("sources") or []
    if not ss:
        return "&mdash;"
    bits = []
    for s in ss:
        p = SPR / s if not Path(s).is_absolute() else Path(s)
        mark = "" if p.is_file() else ' <span class="miss">(not yet)</span>'
        bits.append(src(s) + mark)
    return " ".join(bits)


# ------------------------------------------------------------------ results ---

def scan_results() -> dict:
    """Classify every JSON under results/ — generic, so later milestones land
    on the page without touching this file."""
    out = {"comparison": {}, "summary": {}, "ceiling": [], "other": [],
           "errors": []}
    if not RESULTS.is_dir():
        return out
    for p in sorted(RESULTS.rglob("*.json")):
        if p.name == "status.json" and p.parent == RESULTS:
            continue
        d = load(p)
        if d is None:
            continue
        if not ok(d):
            out["errors"].append((p, d["_load_error"]))
            continue
        group = p.parent.name if p.parent != RESULTS else "(top level)"
        if "systems" in d and "protocol" in d:
            out["comparison"].setdefault(group, []).append((p, d))
        elif p.name == "summary.json":
            out["summary"].setdefault(group, []).append((p, d))
        elif "summary" in d and "vocab" in d and "rows" in d:
            out["ceiling"].append((p, d))
        else:
            out["other"].append(p)
    return out


SCAN = None


# ------------------------------------------------------------------- charts ---

def svg_loop() -> str:
    """The iteration-k loop of PROBLEM.md section 5, hand-drawn (no libraries)."""
    W, H = 920, 372
    BW, BH = 250, 68
    boxes = [
        (20, 40, "1 &middot; Generate",
         ["fresh lean boards + instances,", "every iteration (boards are ~free)"]),
        (335, 40, "2 &middot; Search with net k",
         ["MCTS over the chosen action space,", "fixed expansion budget"]),
        (650, 40, "3 &middot; Certify",
         ["replay every solution against physics;", "uncertified plans dropped, not guessed"]),
        (650, 200, "4 &middot; Replay buffer",
         ["rolling window of certified episodes,", "board/instance provenance kept"]),
        (335, 200, "5 &middot; Train net k+1",
         ["warm-start from net k,", "CollapseStop armed, best-epoch ckpt"]),
        (20, 200, "6 &middot; Gate",
         ["bench vs net k AND the frozen", "supervised baselines + fidelity gauge"]),
    ]
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="The self-play '
         f'iteration loop: generate, search, certify, buffer, train, gate, '
         f'promote">',
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" '
         'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
         '<path d="M 0 0 L 10 5 L 0 10 z" class="ahead"/></marker></defs>']
    for x, y, title, lines in boxes:
        s.append(f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="8" '
                 f'class="nbox"/>')
        s.append(f'<text x="{x + 14}" y="{y + 22}" class="nt">{title}</text>')
        for i, ln in enumerate(lines):
            s.append(f'<text x="{x + 14}" y="{y + 40 + 14 * i}" class="ns">{ln}</text>')

    def arrow(x1, y1, x2, y2, cls="arr"):
        s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}" '
                 f'marker-end="url(#ah)"/>')

    arrow(270, 74, 331, 74)                       # 1 -> 2
    arrow(585, 74, 646, 74)                       # 2 -> 3
    arrow(775, 108, 775, 196)                     # 3 -> 4
    arrow(650, 234, 589, 234)                     # 4 -> 5
    arrow(335, 234, 274, 234)                     # 5 -> 6
    arrow(145, 200, 145, 112)                     # 6 -> 1  (promote)
    arrow(145, 268, 145, 300, "arr dashed")       # 6 -> diagnose
    s.append('<rect x="20" y="300" width="250" height="52" rx="8" '
             'class="nbox alt"/>')
    s.append('<text x="34" y="322" class="nt">Diagnose, do not promote</text>')
    s.append('<text x="34" y="340" class="ns">fidelity slide? damaged data? '
             '(&sect;4.1/&sect;4.3)</text>')
    s.append('<text x="156" y="158" class="lbl">promote: net k+1 becomes net k</text>')
    s.append('<text x="156" y="288" class="lbl">gate not cleared</text>')
    s.append(f'<text x="{W - 8}" y="18" class="cap" text-anchor="end">'
             'PROBLEM.md &sect;5 &mdash; iteration k</text>')
    s.append("</svg>")
    return "".join(s)


def svg_gap_hist(hist, title) -> str:
    """Bar chart of the gap_best histogram of one ceiling arm."""
    if not hist:
        return ""
    try:
        items = sorted(((int(k), int(v)) for k, v in hist.items()))
    except Exception:
        return ""
    if not items:
        return ""
    W, H, L, B, T, R = 440, 170, 34, 34, 18, 10
    vmax = max(v for _, v in items) or 1
    n = len(items)
    pitch = (W - L - R) / n
    bw = min(30.0, pitch - 6)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">']
    for i in range(0, 5):
        gv = vmax * i / 4
        y = (H - B) - (gv / vmax) * (H - B - T)
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" class="grid"/>')
        s.append(f'<text x="{L - 5}" y="{y + 4:.1f}" class="axl" '
                 f'text-anchor="end">{gv:.0f}</text>')
    for i, (g, v) in enumerate(items):
        x = L + i * pitch + (pitch - bw) / 2
        h = (v / vmax) * (H - B - T)
        y = (H - B) - h
        cls = "s1" if g <= 0 else "s2"
        s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                 f'height="{h:.1f}" rx="2" class="{cls}"><title>gap {g}: '
                 f'{v} instances</title></rect>')
        s.append(f'<text x="{x + bw / 2:.1f}" y="{H - B + 14}" class="axl" '
                 f'text-anchor="middle">{g:+d}</text>')
    s.append(f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" class="axis"/>')
    s.append(f'<text x="{(L + W) / 2:.0f}" y="{H - 4}" class="axl" '
             f'text-anchor="middle">best expressible plan &minus; d* (moves)'
             f'</text>')
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------------ sections --

def sec_overview() -> str:
    st = status()
    out = ['<section id="overview">', "<h2>Overview</h2>"]
    out.append(
        "<p><strong>The north star.</strong> Build a complete "
        "<strong>AlphaZero-style self-play loop</strong> for Ricochet Robots "
        "planning that ends up beating every supervised planner in this "
        "repository — the forward move-level planner and the backward subgoal "
        "planner — on the one metric that matters: <strong>realized primitive "
        "moves from the initial state to the terminal state</strong>, on a "
        "fixed benchmark, under a fixed search budget. Fewer moves wins. "
        "Secondary: solve rate (a planner that solves fewer puzzles does not "
        "get to brag about the ones it solved); compute per puzzle is the "
        "tiebreaker. <span class='cite'>PROBLEM.md &sect;1</span></p>")
    out.append(
        "<p>The supervised campaign proved the pieces exist: a size-free value "
        "net that writes near-optimal labels, planners that train to useful "
        "strength from those labels, and a full NN-only data pipeline that "
        "matches the exact-solver pipeline. What no one has done is "
        "<em>close the loop</em>: let the network search, learn from its own "
        "search, and iterate past the ceiling of its teacher.</p>")
    out.append("<h3>The game, in five lines</h3><ul class='tight'>"
               "<li>An n&times;n grid with interior walls; R robots on distinct "
               "cells.</li>"
               "<li>A <em>primitive move</em> picks a robot and a direction; it "
               "slides until blocked by a wall or another robot (it must slide "
               "at least one cell).</li>"
               "<li>An instance fixes a board, robot starts, a target robot and "
               "a target cell; terminal = target robot on target cell.</li>"
               "<li>Cost = number of primitive moves. Non-target "
               "(&ldquo;helper&rdquo;) robots matter: optimal play routinely "
               "parks them as blockers.</li>"
               "<li>Deterministic MDP, discount 1; ground truth (exact solver) "
               "exists only for n &le; 64 — <em>validity</em> is checkable at "
               "any size by physics replay, <em>optimality</em> only &le; 64."
               "</li></ul>"
               "<p class='cite'>PROBLEM.md &sect;2</p>")
    out.append(
        "<h3>Two action-space views (both have trained nets)</h3>"
        + table(["view", "actions", "horizon", "branching", "status here"],
                [row([td_txt("<strong>Forward / move-level</strong>"),
                      td_txt("primitive moves (&le; 4R)"),
                      td_txt("deep: 8&ndash;30+ moves"),
                      td_txt("small"),
                      td_txt("the natural AlphaZero action space; kept as a "
                             "cheap comparison arm at g16r4")]),
                 row([td_txt("<strong>Backward / subgoal</strong>"),
                      td_txt("subgoal decisions from a generated candidate set, "
                             "reasoning from the goal backwards; a realization "
                             "step converts the plan to primitive moves and "
                             "physics-verifies it"),
                      td_txt("shallow: 2&ndash;6 decisions"),
                      td_txt("candidate-set size, up to ~50"),
                      td_txt("where the strongest supervised planner lives — "
                             "the recommended main line (&sect;6.1)")])],
                note="Definitions quoted from PROBLEM.md &sect;2; the "
                     "recommendation and its current status live in "
                     "<a href='#decisions'>The loop &rarr; open decisions</a>."))
    out.append(
        "<h3>The success statement</h3>"
        "<blockquote>&ldquo;Self-play planner X solves &ge; "
        "backward-supervised solve rate with mean realized moves closer to "
        "optimal than forward-supervised, at backward-supervised "
        "compute.&rdquo;<span class='cite'> PROBLEM.md &sect;7</span>"
        "</blockquote>"
        "<p class='note'>Partial orderings short of that are still results — "
        "which is why every milestone below carries its own hard gate. "
        "Non-negotiables inherited by the arena: playable-moves scoring only, "
        "the pinned per-config bench files, replay certification of every "
        "solve, and % optimal / mean regret reported only where exact ground "
        "truth exists (&le; 64).</p>")

    # status strip
    out.append('<h3 id="strip">Milestone status</h3><div class="strip">')
    for k in MILESTONE_KEYS:
        m = milestone(k)
        stt = m.get("status", "unknown")
        title = m.get("title", "")
        out.append(f'<a class="stripitem" href="#milestones"><span class="mk">'
                   f'{esc(k)}</span>{chip(stt, stt)}<span class="mt">'
                   f'{esc(title)}</span></a>')
    out.append("</div>")
    ss = status().get("side_studies") or {}
    if ss:
        out.append('<div class="strip">')
        for k, v in ss.items():
            stt = (v or {}).get("status", "unknown")
            out.append(f'<a class="stripitem" href="#ceiling"><span class="mk">'
                       f'{esc(k)}</span>{chip(stt, stt)}<span class="mt">'
                       f'{esc((v or {}).get("title", ""))}</span></a>')
        out.append("</div>")
    upd = st.get("updated")
    nh = st.get("node_hours") or {}
    meta = []
    if upd:
        meta.append(f"status manifest last updated {esc(upd)}")
    if nh:
        meta.append(f"node-hours spent {esc(nh.get('spent'))}, projected next "
                    f"{esc(nh.get('projected_next'))}"
                    + (f" ({esc(nh.get('note'))})" if nh.get("note") else ""))
    if not st:
        out.append(f'<p class="pendbox">{src(disp(RESULTS / "status.json"))} '
                   "not found or unreadable — every chip above reads "
                   "&ldquo;unknown&rdquo;.</p>")
    elif meta:
        out.append('<p class="note">Chips read from '
                   + src(disp(RESULTS / "status.json")) + " &mdash; "
                   + "; ".join(meta) + ".</p>")

    # what happened so far
    out.append("<h3>What happened so far</h3>")
    bullets = []
    for k in MILESTONE_KEYS:
        m = milestone(k)
        if not m:
            continue
        note = m.get("note")
        stt = m.get("status", "unknown")
        if stt == "pending" and not note:
            continue
        j = jobs_txt(m)
        bullets.append(
            f"<li>{chip(stt, stt)} <strong>{esc(k)} &mdash; "
            f"{esc(m.get('title', ''))}</strong>: {esc(note) if note else ''}"
            + (f' <span class="note">job(s) {j}</span>' if j != "&mdash;" else "")
            + f'<br><span class="note">sources: {sources_txt(m)}</span></li>')
    for k, v in (status().get("side_studies") or {}).items():
        v = v or {}
        j = jobs_txt(v)
        bullets.append(
            f"<li>{chip(v.get('status', 'unknown'), v.get('status', 'unknown'))} "
            f"<strong>side study &mdash; {esc(v.get('title', k))}</strong>"
            + (f' <span class="note">job(s) {j}</span>' if j != "&mdash;" else "")
            + f'<br><span class="note">sources: {sources_txt(v)}</span></li>')
    if bullets:
        out.append("<ul class='log'>" + "".join(bullets) + "</ul>")
    else:
        out.append('<p class="pendbox">No milestone has a note or a non-pending '
                   "status in the manifest yet.</p>")
    scan = SCAN or {}
    nfiles = sum(len(v) for v in scan.get("comparison", {}).values()) \
        + sum(len(v) for v in scan.get("summary", {}).values()) \
        + len(scan.get("ceiling", [])) + len(scan.get("other", []))
    out.append(f'<p class="note">Result files found under '
               f'{src(disp(RESULTS))}: <strong>{nfiles}</strong> '
               f"(comparison payloads, ceiling payloads, summaries and "
               f"anything else). Everything on this page is read from disk at "
               f"generation time.</p>")
    out.append("</section>")
    return "\n".join(out)


def sec_loop() -> str:
    out = ['<section id="loop">', "<h2>The loop</h2>",
           "<p><strong>Iteration k</strong> (PROBLEM.md &sect;5): generate "
           "fresh instances on fresh lean boards &rarr; search with net k under "
           "a fixed expansion budget &rarr; certify &rarr; append to buffer "
           "&rarr; train net k+1 (warm-start from k, CollapseStop armed) &rarr; "
           "gate: bench vs net k AND vs the frozen supervised baselines &rarr; "
           "promote or diagnose.</p>",
           '<figure class="fig">' + svg_loop() +
           "<figcaption>The bootstrap (iteration 0) is free: initialize from "
           "the supervised backward planner pair copied into "
           + src("self_play_robots/assets/") +
           " — or, if &sect;6.2 decides on the size-free rebuild, train the "
           "new architecture on the existing exact corpora first and verify it "
           "reproduces supervised-level bench numbers <em>before</em> any "
           "self-play. Never debug architecture and loop dynamics at the same "
           "time.</figcaption></figure>"]
    out.append("<h3>The AlphaZero mapping</h3>")
    out.append(table(["AlphaZero piece", "here"],
                     [row([td_txt(f"<strong>{a}</strong>"), td_txt(b)])
                      for a, b in MAPPING],
                     note="Quoted from PROBLEM.md &sect;5. This is the only "
                          "table on the page that is not read from a result "
                          "file — it is the design, not a measurement.",
                     cls="wide"))
    out.append(
        "<p class='note'><strong>The improvement hypothesis, stated "
        "honestly:</strong> greedy descent under the labeler already labels at "
        "86&ndash;93% argmin agreement, and MCTS at inference found solutions "
        "greedy descent misses. If search-labeled data trains a net that "
        "searches better, the loop climbs. If it merely matches the supervised "
        "ceiling, that is a publishable negative result about self-play in "
        "deterministic single-agent planning — and per &sect;4.3 the loop can "
        "also go <em>down</em>, which is what the fidelity gauge is for. "
        "<span class='cite'>PROBLEM.md &sect;5</span></p>")

    out.append('<h3 id="lessons">Lessons you must not relearn</h3>')
    out.append(table(["lesson", "citation", "what it forces in the loop"],
                     [row([td_txt(f"<strong>{a}</strong>"), td_txt(b), td_txt(c)])
                      for a, b, c in LESSONS],
                     note="Quoted / condensed from PROBLEM.md &sect;4; each row "
                          "carries the FINDINGS entry it came from (the log "
                          "lives at " + src("supervised_valuenet/FINDINGS.md")
                          + "). These are constraints and free wins, not "
                            "suggestions.",
                     cls="wide"))

    out.append('<h3 id="decisions">Open design decisions</h3>')
    dec = status().get("decisions") or {}
    dspecs = [
        ("action_space", "&sect;6.1 &mdash; action space",
         "Subgoal MCTS (recommended) vs primitive-move MCTS.",
         "Recommendation: <strong>subgoal decisions</strong> — (a) horizon "
         "2&ndash;6 vs 8&ndash;30 makes search tractable per expansion budget; "
         "(b) the strongest baseline lives there, so the bootstrap is warm; "
         "(c) realization + certification machinery already exists "
         "(<code>descent.py</code>, <code>eval/realize.py</code>); (d) the "
         "backward planner beats forward by growing margins with scale (+47.8 "
         "pts solve at 32&times;32). Primitive-move MCTS is the purer "
         "AlphaZero and removes the candidate-generator ceiling — keep it as a "
         "comparison arm at g16r4, not as the main line. <em>The candidate "
         "generator bounds the reachable policy</em>: measured by the "
         "<a href='#ceiling'>ceiling study</a>."),
        ("network", "&sect;6.2 &mdash; network",
         "Rebuild size-free (recommended) vs keep per-size nets.",
         "Recommendation: <strong>rebuild policy+value on the labeler's "
         "pe=none recipe</strong> (<code>nn_labeler/model.py</code> is the "
         "template; add a policy head over candidate encodings). One net "
         "across the curriculum, cross-size transfer (train small, deploy "
         "large — the project's most novel angle), no n&sup2; table to relearn "
         "per size. Cost: the rebuilt nets must first match the per-size "
         "supervised planners on the bench — that is milestone M1, and it is "
         "NOT optional. Fallback if the policy head underperforms: per-size "
         "nets at g16r4/g24r4 only, decide again after M3."),
    ]
    rows_ = []
    for key, name, short, rec in dspecs:
        d = dec.get(key) or {}
        stt = d.get("status", "not in manifest")
        manifest_rec = d.get("recommendation")
        rows_.append(row([
            td_txt(f"<strong>{name}</strong><br><span class='note'>{short}</span>"),
            td_txt(rec + (f"<br><span class='note'>manifest records the "
                          f"recommendation as: {esc(manifest_rec)}</span>"
                          if manifest_rec else "")),
            td_txt(chip(stt, stt)),
        ]))
    out.append(table(["decision", "recommendation (PROBLEM.md &sect;6)",
                      "status"], rows_,
                     note="Recommendation text quoted from PROBLEM.md "
                          "&sect;6.1/&sect;6.2; the status column is read from "
                          "the <code>decisions</code> block of "
                          + src(disp(RESULTS / "status.json")) + ".",
                     cls="wide"))
    out.append(
        "<p class='note'>Two further decisions are already settled in "
        "PROBLEM.md and need no owner sign-off: <strong>&sect;6.3 value target "
        "form</strong> — keep the distributional HL-Gauss head, bucket count "
        "must cover realized costs at the largest curriculum size (the "
        "&sect;4.5 clamp bug); <strong>&sect;6.5 search budget accounting</strong> "
        "— gate under the bench convention (1200 expansions, k=5), spend more "
        "inside self-play generation but never in a way that biases what the "
        "data contains.</p>")
    out.append("</section>")
    return "\n".join(out)


def seed_spread_line() -> str:
    """The measured same-arm seed spread, computed from the g24r4 files."""
    pairs = [("exact-taught pair",
              "scaling/results/g24r4/comparison.json",
              "scaling/results/g24r4/comparison_exactseed21.json"),
             ("NN-twin pair",
              "scaling/results/g24r4/comparison_nntwin.json",
              "scaling/results/g24r4/comparison_nntwin-seed21.json")]
    bits = []
    for label, a_rel, b_rel in pairs:
        _, sa = backward_system(load_sv(a_rel))
        _, sb = backward_system(load_sv(b_rel))
        if not (sa and sb):
            continue
        aa, ab = sa.get("aggregate") or {}, sb.get("aggregate") or {}
        if aa.get("solve_rate") is None or ab.get("solve_rate") is None:
            continue
        ds = abs(aa["solve_rate"] - ab["solve_rate"]) * 100
        do = (abs(aa.get("pct_optimal", 0) - ab.get("pct_optimal", 0))
              if None not in (aa.get("pct_optimal"), ab.get("pct_optimal"))
              else None)
        bits.append(f"{label}: {ds:.1f} pts solve"
                    + (f" / {do:.1f} pts optimality" if do is not None else ""))
    if not bits:
        return ""
    return ("<p class='note'><strong>The yardstick, measured from these very "
            "files.</strong> Two runs of the <em>same</em> arm at g24r4 that "
            "differ only in random seed — " + "; ".join(bits) +
            ". PROBLEM.md &sect;4.7 quotes ~3.4 solve / ~6.6 optimality points "
            "and &sect;8 turns that into the M1 gate (3.5 / 6.6). Any "
            "self-play win smaller than this bar has shown nothing.</p>")


def sec_baselines() -> str:
    out = ['<section id="baselines">', "<h2>Baselines &mdash; the frozen opponents</h2>",
           "<p>Every row is regenerated from a result JSON on disk; nothing "
           "here is hand-typed. These are the numbers the self-play planner has "
           "to beat, frozen forever (PROBLEM.md &sect;7). The headline column "
           "is <strong>mean realized moves</strong> — solve rate is the "
           "qualifier, compute is the tiebreaker.</p>"]
    # protocol, read from a file rather than asserted
    d = load_sv("scaling/results/g24r4/comparison.json")
    p = (d or {}).get("protocol") or {}
    if p:
        out.append("<p class='note'>Bench protocol as recorded in "
                   + src("supervised_valuenet/scaling/results/g24r4/comparison.json")
                   + f": <strong>{esc(p.get('expansions'))} expansions</strong>, "
                     f"<strong>k={esc(p.get('k'))}</strong>, device "
                     f"{esc(p.get('device'))}, exam "
                   + src(p.get("instances_file"))
                   + f" (sha {esc(str(p.get('instances_sha256'))[:12])}, "
                     f"n={esc(p.get('n_instances'))}). Expansion definition: "
                     f"{esc(p.get('expansion_definition'))}.</p>")
    out.append(seed_spread_line())

    rows_ = []
    for cfg, rels in OPPONENT_FILES.items():
        bench = {"g16r4": "eval/data/bench450.jsonl",
                 "g24r4": "scaling/data/g24r4/bench.solved.jsonl",
                 "g24r8": "scaling/data/g24r8/bench.solved.jsonl",
                 "g32r4": "scaling/data/g32r4/bench.solved.jsonl"}[cfg]
        rows_.append(row([f'<td class="grp" colspan="10"><strong>{esc(cfg)}</strong> '
                          f'&mdash; {CFG_HUMAN[cfg]}</td>']))
        for rel in rels:
            rows_.extend(baseline_rows(cfg, rel, cls="op"))
        rows_.append(ceiling_row(cfg, bench))
    out.append(table(BASE_HEADERS, rows_,
                     note="Columns: <em>solve rate</em> = "
                          "<code>aggregate.solved</code>/<code>aggregate.n</code> "
                          "and <code>aggregate.solve_rate</code>; <em>mean "
                          "realized moves</em> = <code>aggregate.mean_moves</code> "
                          "(identical to <code>mean_realized_strict</code> for "
                          "backward systems — playable moves after physics "
                          "replay); <em>mean regret</em> = "
                          "<code>aggregate.mean_regret</code>; <em>% optimal</em> "
                          "= <code>aggregate.pct_optimal</code>; <em>mean "
                          "expansions</em> / <em>mean s/inst</em> = "
                          "<code>aggregate.mean_expansions</code> / "
                          "<code>mean_seconds</code>; <em>set</em> is "
                          "<em>frontier</em> when the aggregate carries "
                          "<code>d_star_placeholder</code> or the protocol's "
                          "instances file has no <code>d_star</code> at all "
                          "(regret and optimality are then suppressed, because "
                          "no optimum exists there). The <em>exact optimum</em> "
                          "row is the mean <code>d_star</code> of the bench "
                          "JSONL itself — the ceiling nothing can beat. "
                          "<strong>Careful:</strong> a system's mean moves and "
                          "mean regret average over the instances "
                          "<em>it solved</em>, while the exact-optimum row "
                          "averages the whole file, so a system that solves "
                          "only the easy subset can print a mean below the "
                          "file's mean d* without being better than optimal — "
                          "always read mean moves next to solve rate, which is "
                          "exactly why PROBLEM.md &sect;1 makes solve rate the "
                          "qualifier on the moves metric.",
                     cls="wide"))
    out.append(
        "<p class='note'>Reading the g32r4 pair the way PROBLEM.md &sect;7 "
        "does: the forward planner is slower but near-optimal when it solves, "
        "the backward planner solves more and faster but further from optimal. "
        "<strong>Beating backward on moves while matching its solve rate "
        "&asymp; closing toward forward's quality at backward's speed — that "
        "is the headline chart this project is aiming at.</strong></p>")

    # the exact-optimum ceiling table
    brows = []
    for b in BENCH_FILES:
        s = bench_stats(b)
        if s is None:
            brows.append(row([td_txt(src(b)),
                              '<td class="pend" colspan="3">not on disk</td>']))
        else:
            brows.append(row([td_txt(src(s["path"])), td(s["n"], "d"),
                              td(s["n_graded"], "d"),
                              td(s["mean_d_star"], ".3f", cls="hlnum")]))
    for b in ["scaling/data/g24r4/bench.unsolved.jsonl",
              "scaling/data/g32r4/bench.unsolved.jsonl"]:
        s = bench_stats(b)
        if s is not None:
            brows.append(row([td_txt(src(s["path"]) + ' <span class="tag front">'
                                     "frontier</span>"), td(s["n"], "d"),
                              td(s["n_graded"], "d"),
                              td_txt("&mdash; (no optimum exists)")], "sub"))
    out.append("<h3>The exact-optimum ceiling, per pinned bench file</h3>")
    out.append(table(["bench file (the pinned exam)", "instances",
                      "with an exact optimum", "mean d*"], brows,
                     note="Read line by line from the JSONL itself: <code>n</code> "
                          "= lines, <code>with an exact optimum</code> = lines "
                          "whose <code>d_star</code> is neither null nor 0, "
                          "<code>mean d*</code> = their mean. Frontier files "
                          "carry no optimum by construction — that is what "
                          "makes them the frontier."))

    # everything else on disk
    shown = {r for rels in OPPONENT_FILES.values() for r in rels}
    extra_rows = []
    for cfg, d in GLOB_DIRS.items():
        p = SV / d
        if not p.is_dir():
            continue
        for f in sorted(p.glob("comparison*.json")):
            rel = f.relative_to(SV).as_posix()
            if rel in shown:
                continue
            extra_rows.extend(baseline_rows(cfg, rel))
    out.append("<h3>Every other comparison file on disk</h3>")
    out.append("<details><summary>Full inventory of "
               + ", ".join(f"<code>{esc(v)}/comparison*.json</code>"
                           for v in GLOB_DIRS.values())
               + f" ({len(extra_rows)} system rows) &mdash; the labeler-track "
                 "arms, seed replicates, B2 arms and frontier runs</summary>"
               + table(BASE_HEADERS, extra_rows,
                       note="Globbed, not enumerated: any file whose name starts "
                            "with <code>comparison</code> in those directories "
                            "appears here automatically. Same column sources as "
                            "the table above.",
                       cls="wide")
               + "</details>")
    out.append("</section>")
    return "\n".join(out)


def sec_m0() -> str:
    out = ['<section id="m0">', "<h2>M0 &mdash; arena parity</h2>",
           "<p><strong>Gate</strong> (PROBLEM.md &sect;8): "
           f"{GATES['M0'][1]} Estimated cost {GATES['M0'][0]}.</p>",
           "<p>The harness is "
           + src("self_play_robots/spr/arena.py")
           + " — a thin wrapper that <em>calls</em> "
           + src("supervised_valuenet/eval/compare.py")
           + " (never forks it): chunked 8-wide CPU evaluation, shard merge, "
             "independent replay certification with <code>eval/replay_validate.py</code>, "
             "then a per-row parity check against the recorded comparison JSON. "
             "A run that fails certification is quarantined, not reported.</p>"]
    m = milestone("M0")
    if m:
        out.append(f'<p class="verdict">{chip(m.get("status", "unknown"), m.get("status", "unknown"))} '
                   f'{esc(m.get("note", ""))} <span class="note">job(s) '
                   f'{jobs_txt(m)}; sources: {sources_txt(m)}</span></p>')

    ARMS, err = arena_arms()
    if err:
        out.append(f'<p class="pendbox">Could not import the ARMS registry from '
                   f'<code>spr/arena.py</code> ({esc(err)}); falling back to '
                   f'whatever landed under {src(disp(RESULTS / "m0"))}.</p>')
    arm_items = []
    if ARMS:
        for name in sorted(ARMS):
            a = ARMS[name]
            arm_items.append({
                "name": name, "config": a.config, "instances": a.instances,
                "flags": " ".join(a.flags) or "(none)", "ref": a.ref,
                "expansions": a.expansions, "k": a.k, "notes": a.notes,
                "policy": disp(a.policy), "value": disp(a.value)})
    else:
        m0dir = RESULTS / "m0"
        for f in (sorted(m0dir.glob("*.json")) if m0dir.is_dir() else []):
            d = load(f)
            sp = (d or {}).get("spr") or {}
            arm_items.append({
                "name": sp.get("arm", f.stem), "config": sp.get("config", "?"),
                "instances": sp.get("instances", "?"),
                "flags": " ".join(sp.get("flags") or []) or "(none)",
                "ref": None, "expansions": None, "k": None,
                "notes": "(registry unavailable — arm reconstructed from the "
                         "result file's spr block)",
                "policy": disp(sp.get("policy", "?")),
                "value": disp(sp.get("value", "?"))})

    if not arm_items:
        out.append(pend_note("no arms registered and nothing under results/m0/"))
        out.append("</section>")
        return "\n".join(out)

    # registry table
    rrows = []
    for a in arm_items:
        rrows.append(row([
            td_txt(f"<code>{esc(a['name'])}</code>"
                   f'<br><span class="note">policy {src(a["policy"])}'
                   f'<br>value {src(a["value"])}</span>'),
            td_txt(esc(a["config"])),
            td_txt(src(a["instances"])),
            td_txt(f"<code>{esc(a['flags'])}</code>"),
            td_txt(f"{esc(a['expansions'])} / k={esc(a['k'])}"
                   if a["expansions"] else "&mdash;"),
            td_txt(src(a["ref"]) if a["ref"] else
                   '<span class="note">no recorded reference (frontier arm)</span>'),
            td_txt(f'<span class="note">{esc(a["notes"])}</span>'),
        ]))
    out.append("<h3>The arm registry</h3>")
    out.append(table(["arm", "config", "instances", "eval.compare flags",
                      "expansions / k", "recorded reference", "notes"], rrows,
                     note="Read live from the <code>ARMS</code> registry in "
                          + src("self_play_robots/spr/arena.py")
                          + " — the same object the job script drives, so this "
                            "table cannot drift from what actually runs.",
                     cls="wide"))

    # per-arm parity
    any_result = False
    for a in arm_items:
        name = a["name"]
        out.append(f'<h3>{esc(name)}</h3>')
        newp = RESULTS / "m0" / f"{name}.json"
        new = load(newp)
        refd = load_sv(a["ref"]) if a["ref"] else None
        if not a["ref"]:
            out.append('<p class="note">No recorded reference for this arm '
                       "(frontier set: solve rate and mean moves only), so "
                       "there is no parity verdict — the row below is a plain "
                       "result.</p>")
        if new is None:
            out.append(pend_note(f"{disp(newp)} not on disk"))
            continue
        if not ok(new):
            out.append(f'<p class="pendbox bad">Result file unreadable: '
                       f'{esc(new["_load_error"])}</p>')
            continue
        any_result = True
        sp = new.get("spr") or {}
        nn, ns = backward_system(new)
        rn, rs = backward_system(refd) if refd else (None, None)
        na = (ns or {}).get("aggregate") or {}
        ra = (rs or {}).get("aggregate") or {}

        def arow(label, agg_, sysname, cls):
            if not agg_:
                return row([td_txt(label),
                            '<td class="pend" colspan="8">not available</td>'], cls)
            return row([
                td_txt(label), td_txt(esc(sysname or "&mdash;")),
                td(agg_.get("n"), "d"), td(agg_.get("solved"), "d"),
                td_pct(agg_.get("solve_rate")),
                td(agg_.get("mean_moves"), ".4f", cls="hlnum"),
                td(agg_.get("mean_regret"), ".4f"),
                td_pct100(agg_.get("pct_optimal")),
                td(agg_.get("mean_expansions"), ".2f"),
            ], cls)

        prows = [arow("<strong>new</strong> (this folder's harness)", na, nn, "hl")]
        if a["ref"]:
            prows.append(arow("recorded reference", ra, rn, "ctl"))
        out.append(table(["run", "system", "n", "solved", "solve rate",
                          "mean realized moves", "mean regret", "% optimal",
                          "mean expansions"], prows,
                         note="New row: " + src(disp(newp)) + "; reference row: "
                              + (src(a["ref"]) if a["ref"] else "&mdash;")
                              + ". Both read from <code>systems[&hellip;].aggregate</code> "
                                "of the first backward system carrying rows — "
                                "exactly what <code>spr.arena.summarize()</code> "
                                "reports.",
                         cls="wide"))

        # per-row difference count, mirroring spr.arena.parity
        if a["ref"] and rs and ns:
            nrows_, rrows_ = ns.get("rows") or [], rs.get("rows") or []
            keys = ("solved", "realized_strict", "expansions", "plan_found")
            diffs = []
            for i, (x, y) in enumerate(zip(nrows_, rrows_)):
                dd = {k: (x.get(k), y.get(k)) for k in keys if x.get(k) != y.get(k)}
                if dd:
                    diffs.append((i, x.get("env_id"), dd))
            sha_new = (new.get("protocol") or {}).get("instances_sha256")
            sha_ref = (refd.get("protocol") or {}).get("instances_sha256")
            same_exam = sha_new == sha_ref
            same_n = len(nrows_) == len(rrows_)
            passed = (not diffs and same_exam and same_n
                      and na.get("solved") == ra.get("solved")
                      and na.get("mean_regret") == ra.get("mean_regret"))
            vcls = "good" if passed else "bad"
            bits = [f"per-row differences on (solved, realized_strict, "
                    f"expansions, plan_found): <strong>{len(diffs)}"
                    f"/{len(nrows_)}</strong>",
                    ("same exam (instances sha matches)" if same_exam else
                     "<strong>instance sha DIFFERS &mdash; not the same "
                     "exam</strong>"),
                    (f"row counts {len(nrows_)} vs {len(rrows_)}"
                     if not same_n else f"{len(nrows_)} rows on both sides"),
                    f"solved {na.get('solved')} vs {ra.get('solved')}"]
            out.append(f'<p class="verdict {vcls}">{chip("pass" if passed else "fail", "M0 PARITY " + ("PASS" if passed else "FAIL"))} '
                       + "; ".join(bits) + ".</p>")
            if diffs:
                drows = [row([td(i, "d"), td(e, "d"),
                              td_txt("; ".join(f"<code>{esc(k)}</code>: "
                                               f"{esc(v[0])} &rarr; {esc(v[1])}"
                                               for k, v in dd.items()))])
                         for i, e, dd in diffs[:40]]
                out.append(table(["row", "env_id", "new &rarr; recorded"], drows,
                                 note="First 40 differing rows; the same "
                                      "comparison <code>spr.arena parity</code> "
                                      "prints."))
        # provenance
        if sp:
            prov = [(k, sp.get(k)) for k in
                    ("arm", "config", "flags", "policy", "value", "instances",
                     "run_dir", "slurm_job_id", "wall_seconds",
                     "replay_certified", "width", "omp_threads", "date")
                    if k in sp]
            out.append(table(["field", "value"],
                             [row([td_txt(f"<code>{esc(k)}</code>"),
                                   td_txt(esc(v))]) for k, v in prov],
                             note="The <code>spr</code> provenance block written "
                                  "by <code>spr.arena.bench()</code> into "
                                  + src(disp(newp)) + "."))
    if not any_result:
        out.append('<p class="note">No M0 result file has landed yet, so no '
                   "parity verdict exists. The registry table above still shows "
                   "exactly which checkpoints, exam and flags will be run.</p>")
    out.append("</section>")
    return "\n".join(out)


CEIL_EXPECTED = [("g16r4", "base"), ("g24r4", "base"),
                 ("g16r4", "b2"), ("g24r4", "b2")]


def sec_ceiling() -> str:
    out = ['<section id="ceiling">',
           "<h2>Ceiling study &mdash; what the subgoal language can express</h2>",
           "<p><strong>Why.</strong> PROBLEM.md &sect;6.1: <em>&ldquo;the "
           "candidate generator bounds the reachable policy: if optimal play "
           "requires a subgoal the generator never proposes, no amount of "
           "search finds it. Measure the generator's ceiling early.&rdquo;</em> "
           "&sect;11 adds why it comes first: <em>&ldquo;it's free and it "
           "bounds the project.&rdquo;</em> The supervised track's probe "
           "answered the "
           "<em>solve-rate</em> question on failing instances only. This one "
           "answers the question that matters for the self-play metric, on the "
           "whole pinned bench: <strong>how many primitive moves does the best "
           "plan the language can express cost, versus the exact optimum "
           "d*?</strong></p>",
           "<p><strong>How.</strong> "
           + src("self_play_robots/spr/ceiling.py")
           + " runs an exhaustive best-first search over partial plans in "
             "abstract plan-cost order (the solver's own admissible ordering — "
             "<em>no network anywhere</em>), strictly realizes every complete "
             "plan it pops (<code>eval.realize.strict_moves</code>, the arena's "
             "own certified move count) and records the first and the best "
             "realizable plan. <code>best_realizable_moves</code> is a tight "
             "<em>upper bound</em> on the language optimum, not a proof: a plan "
             "whose strict count undercuts its abstract cost (an incidental "
             "robot serving as a stopper) can in principle sit beyond the "
             "search bound.</p>"]
    ss = side_study("ceiling")
    if ss:
        out.append(f'<p class="verdict">{chip(ss.get("status", "unknown"), ss.get("status", "unknown"))} '
                   f'{esc(ss.get("title", ""))} <span class="note">job(s) '
                   f'{jobs_txt(ss)}; sources: {sources_txt(ss)}</span></p>')

    found = []
    cdir = RESULTS / "ceiling"
    seen = set()
    if cdir.is_dir():
        for f in sorted(cdir.glob("*.json")):
            d = load(f)
            if ok(d) and isinstance(d.get("summary"), dict):
                found.append((f, d))
                seen.add((d.get("config"), d.get("vocab")))
    expected_missing = [(c, v) for c, v in CEIL_EXPECTED if (c, v) not in seen]

    rows_ = []
    for f, d in found:
        s = d["summary"]
        rows_.append(row([
            td_txt(f"<strong>{esc(d.get('config'))}</strong> / "
                   f"{esc(d.get('vocab'))}"),
            td(s.get("n"), "d"),
            td_pct(s.get("solve_ceiling")),
            td(s.get("n_graded_realizable"), "d"),
            td(s.get("mean_d_star"), ".2f"),
            td(s.get("mean_best_moves"), ".2f", cls="hlnum"),
            td(s.get("mean_gap_best"), ".2f"),
            td_pct100(s.get("pct_best_optimal")),
            td(s.get("mean_first_moves"), ".2f"),
            td(s.get("mean_gap_first"), ".2f"),
            td_pct100(s.get("pct_first_optimal")),
            td(s.get("capped"), "d"),
            td_txt(src(disp(f))),
        ], "hl"))
    for c, v in expected_missing:
        rows_.append(row([td_txt(f"<strong>{esc(c)}</strong> / {esc(v)}"),
                          '<td class="pend" colspan="11">pending</td>',
                          td_txt(src(f"self_play_robots/results/ceiling/{c}_{v}.json"))]))
    out.append(table(["config / vocab", "n", "solve ceiling", "graded &amp; "
                      "realizable", "mean d*", "mean best-plan moves",
                      "mean gap (best)", "% best = optimal",
                      "mean first-plan moves", "mean gap (first)",
                      "% first = optimal", "capped", "source file"], rows_,
                     note="All columns read from <code>summary</code> of "
                          "<code>results/ceiling/&lt;cfg&gt;_&lt;vocab&gt;.json</code>: "
                          "<code>n</code>, <code>solve_ceiling</code>, "
                          "<code>n_graded_realizable</code>, "
                          "<code>mean_d_star</code>, <code>mean_best_moves</code>, "
                          "<code>mean_gap_best</code>, <code>pct_best_optimal</code>, "
                          "<code>mean_first_moves</code>, <code>mean_gap_first</code>, "
                          "<code>pct_first_optimal</code>, <code>capped</code>. "
                          "<em>first</em> = the first realizable plan popped (the "
                          "supervised probe's definition, i.e. what a "
                          "cost-ordered planner would take); <em>best</em> = the "
                          "cheapest realizable plan found before the abstract "
                          "cost order proves nothing cheaper remains.",
                     cls="wide"))

    if not found:
        out.append(pend_note("no file under results/ceiling/ yet; jobs listed in "
                             "the status manifest above"))
        out.append("<h3>What this section will say</h3>"
                   "<p class='note'>Once the arms land, each one gets a "
                   "plain-language reading of the form &ldquo;the best plan the "
                   "subgoal language can express costs <em>X</em> moves against "
                   "an exact optimum of <em>Y</em>, and reaches the optimum on "
                   "<em>Z</em>% of the bench&rdquo;, plus the histogram of the "
                   "per-instance gap. That number is a hard bound on the "
                   "self-play planner in the subgoal action space: <strong>no "
                   "amount of search can beat a plan the language cannot "
                   "write.</strong></p>")
    for f, d in found:
        s = d["summary"]
        cfg, voc = d.get("config"), d.get("vocab")
        out.append(f'<h3>{esc(cfg)} &middot; {esc(voc)} vocabulary</h3>')
        bits = []
        if s.get("mean_best_moves") is not None and s.get("mean_d_star") is not None:
            gapb = s.get("mean_gap_best")
            gaptxt = (f" — a mean gap of <strong>{gapb:.2f} moves</strong>"
                      if gapb is not None else "")
            bits.append(
                f"On the {esc(s.get('n_graded_realizable'))} graded instances it "
                f"can express a plan for, the best plan the subgoal language can "
                f"write costs <strong>{s['mean_best_moves']:.2f} moves</strong> "
                f"against an exact optimum of <strong>{s['mean_d_star']:.2f} "
                f"moves</strong>" + gaptxt)
        if s.get("pct_best_optimal") is not None:
            bits.append(f"the language reaches the exact optimum on "
                        f"<strong>{s['pct_best_optimal']:.1f}%</strong> of them")
        if s.get("solve_ceiling") is not None:
            bits.append(f"and it can express <em>any</em> realizable plan for "
                        f"<strong>{100 * s['solve_ceiling']:.1f}%</strong> of the "
                        f"{esc(s.get('n'))} bench instances (the solve-rate "
                        f"ceiling)")
        if bits:
            out.append('<p class="verdict">' + "; ".join(bits) + ".</p>")
        if s.get("mean_gap_first") is not None and s.get("mean_gap_best") is not None:
            out.append(
                f"<p>First-vs-best: a planner that simply takes the first "
                f"realizable plan in abstract-cost order lands "
                f"{s['mean_gap_first']:.2f} moves above d*, while the best plan "
                f"in the language lands {s['mean_gap_best']:.2f} above. The "
                f"difference — <strong>{s['mean_gap_first'] - s['mean_gap_best']:.2f} "
                f"moves</strong> — is the room <em>search inside the existing "
                f"language</em> has to work with. The "
                f"{s['mean_gap_best']:.2f}-move residue is the part search can "
                f"never recover: it needs a richer candidate generator or the "
                f"primitive-move action space (&sect;6.1).</p>")
        cats = s.get("categories") or {}
        if cats:
            out.append(table(["category", "instances"],
                             [row([td_txt(f"<code>{esc(k)}</code>"), td(v, "d")])
                              for k, v in sorted(cats.items())],
                             note="<code>summary.categories</code>: "
                                  "REALIZABLE_EXISTS = the language can write a "
                                  "physics-valid plan; NO_REALIZABLE_PLAN = "
                                  "complete abstract plans exist but none "
                                  "realizes; NO_COMPLETE_PLAN = the generator "
                                  "never completed a plan; INCONCLUSIVE = the "
                                  "search hit a cap; ERROR = the instance threw "
                                  "(never lost, always recorded)."))
        svg = svg_gap_hist(s.get("gap_best_hist"),
                           f"{cfg} {voc}: distribution of best-plan gap to d*")
        if svg:
            out.append('<figure class="fig">' + svg +
                       "<figcaption>Per-instance gap between the best plan the "
                       "language can express and the exact optimum, from "
                       "<code>summary.gap_best_hist</code> of "
                       + src(disp(f)) + ". Bars at or below zero (left, accent "
                       "colour) are instances where the language reaches — or "
                       "appears to undercut — d*. "
                       + (f"<code>n_best_below_dstar</code> = "
                          f"{esc(s.get('n_best_below_dstar'))}, "
                          f"<code>n_abstract_below_dstar</code> = "
                          f"{esc(s.get('n_abstract_below_dstar'))}: a strict "
                          "count below d* means an incidental stopper made the "
                          "realization cheaper than its abstract cost, or the "
                          "recorded d* is a solver artifact — either way, worth "
                          "an eyeball before it is quoted."
                          if s.get("n_best_below_dstar") else "")
                       + "</figcaption></figure>")
        caps = d.get("caps") or {}
        meta = [("instances", src(disp(d.get("instances", "?")))),
                ("caps", f"<code>{esc(json.dumps(caps))}</code>"),
                ("workers", esc(d.get("workers"))),
                ("slurm job", esc(d.get("slurm_job_id"))),
                ("wall seconds", esc(d.get("wall_seconds"))),
                ("date", esc(d.get("date"))),
                ("capped instances", esc(s.get("capped"))),
                ("best plans proven bounded", esc(s.get("best_bounded")))]
        out.append(table(["field", "value"],
                         [row([td_txt(k), td_txt(v)]) for k, v in meta],
                         note="Provenance block of " + src(disp(f)) + "."))
    out.append(
        "<p class='note'><strong>What this bounds.</strong> Everything above is "
        "measured with <em>no network anywhere</em> — it is a property of the "
        "plan language and its candidate generator, not of any planner. In the "
        "subgoal action space the self-play planner can never beat the "
        "<em>best-plan</em> column on the moves metric, however good its search "
        "or its net becomes. The distance between the supervised planner's mean "
        "realized moves (<a href='#baselines'>Baselines</a>) and this ceiling "
        "is the room the whole project has to play in; the distance between the "
        "ceiling and mean d* is what would require the primitive-move action "
        "space or a richer generator (PROBLEM.md &sect;6.1).</p>")
    out.append("</section>")
    return "\n".join(out)


def comparison_table(files, note):
    """Generic renderer: bench rows out of any comparison payload."""
    rows_ = []
    for p, d in files:
        proto = d.get("protocol") or {}
        sp = d.get("spr") or {}
        for name, kind, a, nrows in systems_of(d):
            if not a:
                rows_.append(row([td_txt(src(disp(p))), td_txt(esc(name)),
                                  '<td class="pend" colspan="7">no aggregate '
                                  f"(kind={esc(kind)})</td>"]))
                continue
            front = is_frontier(d, a)
            rows_.append(row([
                td_txt(src(disp(p))
                       + (f'<br><span class="note">{esc(sp.get("arm"))} '
                          f'&middot; {esc(sp.get("config"))}'
                          + (" &middot; replay-certified"
                             if sp.get("replay_certified") else "")
                          + "</span>" if sp else "")),
                td_txt(esc(name) + (f' <span class="tag front">frontier</span>'
                                    if front else "")),
                td(a.get("n"), "d"),
                td_pct(a.get("solve_rate")),
                td(a.get("mean_moves"), ".2f", cls="hlnum"),
                DASH if front else td(a.get("mean_regret"), ".2f"),
                DASH if front else td_pct100(a.get("pct_optimal")),
                td(a.get("mean_expansions"), ".1f"),
                td(a.get("mean_seconds"), ".1f"),
            ]))
        if not systems_of(d):
            rows_.append(row([td_txt(src(disp(p))),
                              '<td class="pend" colspan="8">payload has no '
                              "systems block</td>"]))
        rows_.append(row([f'<td class="protorow" colspan="9">protocol: '
                          f"{protocol_bits(d)}</td>"]))
    return table(["source file", "system", "n", "solve rate",
                  "mean realized moves", "mean regret", "% optimal",
                  "mean expansions", "mean s/inst"], rows_, note=note,
                 cls="wide")


def kv_table(p, d):
    def flat(prefix, obj, out_):
        if isinstance(obj, dict):
            for k, v in obj.items():
                flat(f"{prefix}.{k}" if prefix else str(k), v, out_)
        elif isinstance(obj, list) and len(obj) > 12:
            out_.append((prefix, f"[{len(obj)} items]"))
        else:
            out_.append((prefix, json.dumps(obj) if not isinstance(obj, str) else obj))
        return out_
    items = flat("", d, [])
    return table(["key", "value"],
                 [row([td_txt(f"<code>{esc(k)}</code>"), td_txt(esc(v))])
                  for k, v in items[:200]],
                 note="Flattened key/value view of " + src(disp(p))
                      + (" (first 200 keys)" if len(items) > 200 else "") + ".")


def sec_milestones() -> str:
    scan = SCAN or {}
    comp = scan.get("comparison", {})
    summ = scan.get("summary", {})
    claimed = set()
    out = ['<section id="milestones">', "<h2>Milestones M0&ndash;M6</h2>",
           "<p>Gate text and estimated node-hours are quoted from PROBLEM.md "
           "&sect;8; status chips, sources and job ids come from "
           + src(disp(RESULTS / "status.json"))
           + ". Result tables are generic: any JSON under "
           + src(disp(RESULTS))
           + " whose payload carries <code>systems</code> + <code>protocol</code> "
             "is rendered as bench rows in the milestone whose directory it "
             "sits in, and any <code>summary.json</code> is rendered as a "
             "key/value table — so later milestones appear here without "
             "touching the generator.</p>"]
    for k in MILESTONE_KEYS:
        m = milestone(k)
        cost, gate = GATES[k]
        stt = m.get("status", "unknown")
        out.append(f'<h3 id="ms-{k.lower()}">{esc(k)} &mdash; '
                   f'{esc(m.get("title", ""))} {chip(stt, stt)}</h3>')
        out.append(f'<p class="gate"><strong>Gate</strong> (PROBLEM.md &sect;8, '
                   f'est. {cost}): {gate}</p>')
        if m.get("note"):
            out.append(f'<p>{esc(m["note"])}</p>')
        out.append(f'<p class="note">status manifest &mdash; sources: '
                   f'{sources_txt(m)}; job(s): {jobs_txt(m)}</p>')
        # generic result pickup: directories named m0, m1, ... (or m1_something)
        dirs = [g for g in sorted(set(list(comp) + list(summ)))
                if g == k.lower() or g.startswith(k.lower() + "_")]
        rendered = False
        for g in dirs:
            claimed.add(g)
            if comp.get(g):
                rendered = True
                out.append(comparison_table(
                    comp[g],
                    "Every comparison payload under "
                    + src(disp(RESULTS / g))
                    + ", read generically: <code>systems[&hellip;].aggregate</code> "
                      "for the numbers, <code>protocol</code> for the exam line "
                      "under each file, and the <code>spr</code> block (when "
                      "present) for arm/config/certification provenance."))
            for p, d in summ.get(g, []):
                rendered = True
                out.append(kv_table(p, d))
        if not rendered:
            if k == "M0":
                out.append('<p class="note">Detailed parity tables for this '
                           'milestone live in the <a href="#m0">M0 arena '
                           "parity</a> tab.</p>")
            else:
                out.append(pend_note(f"nothing under {disp(RESULTS)}/"
                                     f"{k.lower()}/"))
    # anything not claimed by a milestone
    leftovers = [g for g in sorted(set(list(comp) + list(summ)))
                 if g not in claimed and g != "ceiling"]
    if leftovers:
        out.append('<h3 id="ms-other">Other result directories</h3>')
        for g in leftovers:
            out.append(f"<h4>{esc(g)}</h4>")
            if comp.get(g):
                out.append(comparison_table(
                    comp[g], "Comparison payloads under "
                    + src(disp(RESULTS / g)) + " (picked up generically)."))
            for p, d in summ.get(g, []):
                out.append(kv_table(p, d))
    other = scan.get("other", [])
    errs = scan.get("errors", [])
    if other or errs:
        bits = []
        if other:
            bits.append("unclassified JSON files (neither a comparison payload "
                        "nor a summary.json): "
                        + ", ".join(src(disp(p)) for p in other[:30]))
        if errs:
            bits.append("unreadable files: "
                        + ", ".join(f"{src(disp(p))} ({esc(e)})"
                                    for p, e in errs[:10]))
        out.append('<p class="note">' + "; ".join(bits) + ".</p>")
    out.append("</section>")
    return "\n".join(out)


def sec_glossary() -> str:
    d = load_sv("scaling/results/g24r4/comparison.json")
    p = (d or {}).get("protocol") or {}
    exp = p.get("expansions")
    kk = p.get("k")
    budget = (f"{exp} node expansions with k={kk} candidate children per "
              f"expansion, as recorded in the <code>protocol</code> block of "
              + src("supervised_valuenet/scaling/results/g24r4/comparison.json")
              if exp is not None else
              "1200 node expansions, k=5 (PROBLEM.md &sect;6.5; no protocol "
              "file on disk to quote)")
    expdef = p.get("expansion_definition")
    b450 = bench_stats("eval/data/bench450.jsonl")
    b24 = bench_stats("scaling/data/g24r4/bench.solved.jsonl")
    dstar_bits = []
    for b in (b450, b24):
        if b:
            short = "/".join(b["path"].split("/")[-2:])
            dstar_bits.append(f"<code>{esc(short)}</code> mean d* = "
                              f"{b['mean_d_star']:.2f} over {b['n_graded']} "
                              f"graded instances")
    items = [
        ("playable-moves scoring",
         "The only scoring the arena accepts: a solution counts as the number "
         "of <em>primitive moves actually executed</em> against the physics, "
         "after replay — not the abstract plan cost the planner thought it had. "
         "In the result JSONs this is <code>mean_realized_strict</code>, and "
         "<code>mean_moves</code> is set to it. PROBLEM.md &sect;7."),
        ("expansion budget",
         "The fixed search budget every arm gets, so comparisons are "
         "apples-to-apples: " + budget + "."
         + (f" Expansion definition recorded in the same file: "
            f"<em>{esc(expdef)}</em>." if expdef else "")
         + " Inside self-play <em>generation</em> you may spend more — but "
           "never in a way that biases what the data contains (&sect;4.3, "
           "&sect;6.5)."),
        ("certification / replay",
         "Nothing a network reports counts until it is replayed move-by-move "
         "against <code>simulate.py</code> physics "
         "(<code>eval/compare.py --dump-moves</code> + "
         "<code>eval/replay_validate.py</code>; "
         "<code>spr.arena</code> quarantines a run that fails). In the loop: "
         "<em>a backup value from an unrealized plan is a hypothesis; only "
         "realized, replayed plans update the record.</em> This is also the "
         "defense against the self-play analogue of reward hacking. "
         "PROBLEM.md &sect;4.6."),
        ("subgoal / bottleneck / support / helper",
         "The backward planner does not choose moves; it chooses "
         "<em>subgoals</em> — segment endpoints and helper placements — from a "
         "generated candidate set, reasoning from the goal backwards. A "
         "<em>helper</em> is any non-target robot; optimal play routinely parks "
         "helpers as blockers, which is exactly why a plan step may be "
         "&ldquo;put the blue robot <em>there</em>&rdquo; rather than a move. A "
         "<em>realization</em> step turns the subgoal plan into primitive moves "
         "and physics-verifies it. PROBLEM.md &sect;2, &sect;6.1."),
        ("backward vs forward planner",
         "Two action-space views with separately trained nets. "
         "<em>Backward / subgoal</em>: shallow horizon (2&ndash;6 decisions), "
         "branching = candidate-set size; the strongest supervised planner and "
         "the recommended self-play main line. <em>Forward / move-level</em>: "
         "actions are primitive moves, deep horizon (8&ndash;30+), small "
         "branching; slower but near-optimal when it solves. Both appear in "
         "the <a href='#baselines'>Baselines</a> tables, read from the same "
         "comparison files."),
        ("base / B1 / B2 vocabulary",
         "The language a plan is written in. <em>Base</em> names plan steps by "
         "absolute board cells. <em>B1</em> adds park repairs. <em>B2</em> adds "
         "<em>by-reference</em> steps (&ldquo;park blue where red currently "
         "stands&rdquo;) — more expressive, far more expensive for the exact "
         "solver to label, and the source of the supervised track's negative "
         "result (a net distilled from iteration-capped B2 data reproduced the "
         "cap's pathology, FINDINGS 68). Vocabulary separation is absolute: "
         "base and B2 datasets never mix. <strong>This project is "
         "base-vocabulary unless the owner says otherwise</strong> "
         "(PROBLEM.md &sect;10); the B2 rows on this page are baselines and "
         "ceiling arms, not training data."),
        ("seed noise bars",
         "Two runs of the same arm differing only in random seed differ by "
         "~3.4 solve / ~6.6 optimality points at g24r4 (FINDINGS 67&rarr;71); "
         "PROBLEM.md &sect;8 turns that into the M1 gate of <strong>3.5 solve "
         "points / 6.6 optimality points</strong>. A promotion decision inside "
         "the loop must clear this bar — or use a paired-instance test "
         "(McNemar over the per-instance solved vector). The measured spread "
         "for this repository's own files is computed live in the "
         "<a href='#baselines'>Baselines</a> tab."),
        ("fidelity gauge",
         "Argmin agreement of self-generated value targets against exact "
         "optima: at each decision, does the label pick the <em>same best "
         "candidate</em> as the exact solver? Cheap to measure &le; 64. "
         "Calibrated by the supervised track: ~91% agreement &rarr; planners "
         "match exact-taught ones, ~89% keeps solve rate but loses optimality, "
         "~82% collapses (FINDINGS 74). A slide below ~90% predicts a utility "
         "slide <em>before</em> the bench shows it — the loop's early-warning "
         "instrument. PROBLEM.md &sect;4.1."),
        ("lean boards",
         "Boards generated at any size in seconds by "
         "<code>nn_labeler/leanboard.py</code> (3.8 s at 96&times;96 vs 11.9 h "
         "for the eager path), layout-identical to eager boards "
         "(parity-proven), with a lazy distance oracle. They make &ldquo;fresh "
         "instances every iteration&rdquo; essentially free — which is where "
         "self-play diversity comes from in a deterministic single-agent game "
         "(&sect;6.4)."),
        ("d*",
         "The exact move-optimal cost of an instance, from the Rust solver, "
         "stored per line in the pinned bench JSONL. Exists only for n &le; 64 "
         "(hard engine assert), and only on the graded sets."
         + (" Measured here: " + "; ".join(dstar_bits) + "."
            if dstar_bits else "")),
        ("frontier set",
         "The hard tail: <code>bench.unsolved.jsonl</code> instances that "
         "nothing had solved when the bench was pinned, so no optimum exists "
         "(<code>d_star</code> null or 0, and the aggregate may carry "
         "<code>d_star_placeholder</code>). Only solve rate and mean realized "
         "moves are meaningful there — this page suppresses regret and "
         "optimality on frontier rows rather than printing a meaningless "
         "number."),
        ("graded set",
         "The complement: <code>bench.solved.jsonl</code> (and "
         "<code>eval/data/bench450.jsonl</code> at g16r4), where every instance "
         "carries d*, so % optimal and mean regret are defined."),
        ("node-hour (nh)",
         "The cluster's cost unit — one full node for one hour; 1 GPU for 1 h = "
         "0.125 nh. The whole supervised campaign cost ~27 nh; a self-play loop "
         "is more expensive by nature, which is why every milestone in "
         "PROBLEM.md &sect;8 carries an estimate."),
    ]
    body = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in items)
    return ('<section id="glossary"><h2>Glossary</h2>'
            "<p class='note'>Definitions quoted or condensed from PROBLEM.md "
            "and the inherited FINDINGS log; where a number can be read from a "
            "file instead of asserted, it is.</p>"
            f'<dl class="defs">{body}</dl></section>')


# --------------------------------------------------------------- page shell ---

PANELS = [
    ("Overview", "overview", sec_overview),
    ("The loop", "loop", sec_loop),
    ("Baselines", "baselines", sec_baselines),
    ("M0 arena parity", "m0", sec_m0),
    ("Ceiling study", "ceiling", sec_ceiling),
    ("Milestones", "milestones", sec_milestones),
    ("Glossary", "glossary", sec_glossary),
]

JS = """
(function () {
  var D = document, tabs = [].slice.call(D.querySelectorAll(".tabs a"));
  function panelOf(el) {
    while (el && el.nodeType === 1 && !el.classList.contains("panel")) {
      el = el.parentNode;
    }
    return el && el.nodeType === 1 ? el : null;
  }
  function show(panel, target) {
    if (!panel) { return; }
    [].forEach.call(D.querySelectorAll(".panel"), function (p) {
      p.classList.toggle("on", p === panel);
    });
    tabs.forEach(function (a) {
      var cur = a.getAttribute("data-panel") === panel.id;
      a.classList.toggle("cur", cur);
      a.setAttribute("aria-current", cur ? "true" : "false");
    });
    if (target && target !== panel && target.scrollIntoView) {
      target.scrollIntoView();
    } else {
      window.scrollTo(0, 0);
    }
  }
  function route() {
    var id = "", el = null;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch (e) { id = ""; }
    if (id) { el = D.getElementById(id); }
    if (el) { show(panelOf(el), el); } else { show(D.querySelector(".panel"), null); }
  }
  window.addEventListener("hashchange", route);
  route();
})();
"""

CSS = """
:root { --ink:#1a1a1a; --mut:#6a6a6a; --line:#d8d8d8; --hl:#f4f1fb;
        --ctl:#f5f5f5; --bg:#fff; --card:#f7f6fa; --grp:#efecf7;
        --good-bg:#e7f2e4; --good-ink:#2c5e1e; --bad-bg:#f7e3e0;
        --bad-ink:#8a2b1d; --warn-bg:#f7eeda; --warn-ink:#7a5a12;
        --run-bg:#e6eef9; --run-ink:#1d4b86;
        --acc:#6d3fc4; --s1:#6d3fc4; --s2:#c2410c; --s3:#0f766e; }
@media (prefers-color-scheme: dark) {
  :root { --ink:#e8e8e8; --mut:#9a9a9a; --line:#3a3a3a; --hl:#241f31;
          --ctl:#242424; --bg:#161616; --card:#1d1b22; --grp:#272233;
          --good-bg:#22371c; --good-ink:#a4d194; --bad-bg:#3d211c;
          --bad-ink:#e0a297; --warn-bg:#37301c; --warn-ink:#d9be7a;
          --run-bg:#1c2a3d; --run-ink:#9dc2ee;
          --acc:#b18cf0; --s1:#b18cf0; --s2:#fb923c; --s3:#2dd4bf; } }
* { box-sizing: border-box; }
body { font: 15px/1.55 system-ui, -apple-system, sans-serif; color: var(--ink);
       background: var(--bg); max-width: 66rem; margin: 0 auto 3rem;
       padding: 0 1rem; }
h1 { font-size: 1.5rem; margin: 1.2rem 0 .4rem; }
h2 { font-size: 1.2rem; margin-top: 2.2rem; border-bottom: 2px solid var(--acc);
     display: inline-block; padding-bottom: .15rem; }
h3 { font-size: 1rem; margin: 1.5rem 0 .3rem; }
h4 { font-size: .92rem; margin: 1rem 0 .2rem; color: var(--mut); }
a { color: var(--acc); }
blockquote { margin: .6rem 0; padding: .5rem .9rem; border-left: 3px solid var(--acc);
             background: var(--card); }
.tabs { position: sticky; top: 0; z-index: 6; background: var(--bg);
        border-bottom: 1px solid var(--line); padding: .5rem 0 0;
        font-size: .82rem; display: flex; flex-wrap: wrap; gap: .15rem .3rem; }
.tabs a { color: var(--mut); text-decoration: none; white-space: nowrap;
          padding: .3rem .6rem; border-radius: 5px 5px 0 0;
          border-bottom: 3px solid transparent; }
.tabs a:hover { color: var(--ink); background: var(--card); }
.tabs a.cur { color: var(--ink); font-weight: 600; background: var(--card);
              border-bottom-color: var(--acc); }
html.js .panel { display: none; }
html.js .panel.on { display: block; }
section, figure[id] { scroll-margin-top: 3.4rem; }
.tw { overflow-x: auto; max-width: 100%; }
table.t { border-collapse: separate; border-spacing: 0; margin: .4rem 0;
          min-width: 38rem; font-size: .88rem; }
table.t.wide { min-width: 48rem; }
th, td { border-bottom: 1px solid var(--line); border-right: 1px solid var(--line);
         padding: .3rem .55rem; text-align: right;
         font-variant-numeric: tabular-nums; vertical-align: top; }
th:first-child, td:first-child { text-align: left; border-left: 1px solid var(--line); }
thead th { position: sticky; top: 2.55rem; z-index: 3; background: var(--bg);
           border-top: 1px solid var(--line); font-weight: 600;
           text-align: left; }
tbody td { text-align: right; }
tbody td:first-child, tbody td.pend { text-align: left; }
tr.hl td { background: var(--hl); }
tr.op td { background: var(--card); }
tr.ceil td { background: var(--grp); font-weight: 600; }
tr.ctl td { background: var(--ctl); color: var(--mut); }
tr.sub td:first-child { padding-left: 1.4rem; }
tr.bad td { background: var(--bad-bg); }
td.grp { background: var(--grp); font-size: .92rem; }
td.protorow { color: var(--mut); font-size: .78rem; text-align: left; }
td.pend { color: var(--mut); font-style: italic; }
td.dash { color: var(--mut); }
td.hlnum { font-weight: 600; }
.note, .meta { color: var(--mut); font-size: .85rem; }
.cite { color: var(--mut); font-size: .8rem; }
code, .src { font: .82em/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
             word-break: break-all; }
.src { color: var(--mut); }
.miss { color: var(--mut); font-style: italic; font-size: .8em; }
.banner { border: 1px solid var(--line); border-left: 3px solid var(--acc);
          padding: .5rem .8rem; font-size: .85rem; color: var(--mut);
          background: var(--card); }
.pendbox { background: var(--card); border: 1px dashed var(--line);
           padding: .5rem .8rem; color: var(--mut); font-size: .88rem; }
.pendbox.bad { border-color: var(--bad-ink); color: var(--bad-ink); }
.verdict { background: var(--card); border-left: 3px solid var(--acc);
           padding: .5rem .8rem; }
.verdict.good { border-left-color: var(--good-ink); }
.verdict.bad { border-left-color: var(--bad-ink); }
.gate { background: var(--card); border-left: 3px solid var(--line);
        padding: .4rem .8rem; font-size: .9rem; }
.chip { display: inline-block; font-size: .74rem; font-weight: 600;
        padding: .05rem .5rem; border-radius: 99px; margin-right: .35rem;
        white-space: nowrap; }
.chip.good { background: var(--good-bg); color: var(--good-ink); }
.chip.bad { background: var(--bad-bg); color: var(--bad-ink); }
.chip.warn { background: var(--warn-bg); color: var(--warn-ink); }
.chip.run { background: var(--run-bg); color: var(--run-ink); }
.chip.pend { background: var(--ctl); color: var(--mut); }
.tag { display: inline-block; font-size: .72rem; padding: 0 .35rem;
       border-radius: 3px; border: 1px solid var(--line); color: var(--mut); }
.tag.front { border-color: var(--s2); color: var(--s2); }
.strip { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0 .8rem; }
.stripitem { display: flex; flex-direction: column; gap: .15rem;
             border: 1px solid var(--line); border-radius: 6px;
             padding: .4rem .6rem; min-width: 8.5rem; flex: 1 1 8.5rem;
             text-decoration: none; color: var(--ink); background: var(--card); }
.stripitem:hover { border-color: var(--acc); }
.stripitem .mk { font-weight: 700; font-size: .85rem; }
.stripitem .mt { color: var(--mut); font-size: .78rem; }
ul.log { list-style: none; padding-left: 0; }
ul.log li { border-left: 2px solid var(--line); padding: .25rem 0 .35rem .7rem;
            margin: .35rem 0; }
ul.tight li { margin: .15rem 0; }
.defs dt { font-weight: 600; margin-top: .8rem; }
.defs dd { margin: .15rem 0 0 0; }
details { margin: .5rem 0; }
summary { cursor: pointer; color: var(--acc); font-size: .9rem; }
.fig { margin: 1rem 0; }
.fig svg { width: 100%; height: auto; max-width: 46rem; display: block; }
.fig figcaption { color: var(--mut); font-size: .85rem; max-width: 46rem;
                  margin-top: .3rem; }
.fig .nbox { fill: var(--card); stroke: var(--acc); stroke-width: 1.4; }
.fig .nbox.alt { stroke: var(--mut); stroke-dasharray: 5 4; }
.fig .nt { fill: var(--ink); font: 600 13px system-ui, sans-serif; }
.fig .ns { fill: var(--mut); font: 11px system-ui, sans-serif; }
.fig .lbl { fill: var(--acc); font: 600 11px system-ui, sans-serif; }
.fig .cap { fill: var(--mut); font: 11px system-ui, sans-serif; }
.fig .arr { stroke: var(--mut); stroke-width: 1.6; fill: none; }
.fig .arr.dashed { stroke-dasharray: 5 4; }
.fig .ahead { fill: var(--mut); }
.fig .grid { stroke: var(--line); stroke-width: .5; }
.fig .axis { stroke: var(--line); stroke-width: 1; }
.fig .axl { fill: var(--mut); font: 11px system-ui, sans-serif; }
.fig .s1 { fill: var(--s1); }
.fig .s2 { fill: var(--s2); }
.fig .s3 { fill: var(--s3); }
footer { margin-top: 2.5rem; border-top: 1px solid var(--line);
         padding-top: .6rem; color: var(--mut); font-size: .82rem; }
"""


def main() -> int:
    global SCAN
    SCAN = scan_results()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tabs, panels = [], []
    for i, (label, anchor, fn) in enumerate(PANELS, 1):
        pid = f"p{i}"
        try:
            body = fn()
        except Exception as e:              # a broken section must not kill the page
            body = (f'<section id="{anchor}"><h2>{label}</h2>'
                    f'<p class="pendbox bad">This section failed to render: '
                    f'{esc(type(e).__name__)}: {esc(e)}</p></section>')
            STATS["error"].append(f"section {label}: {type(e).__name__}: {e}")
        tabs.append(f'<a href="#{anchor}" data-panel="{pid}">{label}</a>')
        panels.append(f'<div class="panel" id="{pid}">\n{body}\n</div>')
    nav = '<nav class="tabs" aria-label="sections">' + "\n".join(tabs) + "</nav>"
    body_html = "\n".join(panels)
    st = status()
    upd = st.get("updated")
    upd_txt = f"; status manifest of {esc(upd)}" if upd else ""
    n_read, n_missing = len(set(STATS["read"])), len(set(STATS["missing"]))
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Self-play planner suite (self_play_robots)</title>\n"
        f"<style>{CSS}</style>\n"
        '<script>document.documentElement.className += " js";</script>'
        "</head><body>\n"
        f"{nav}\n"
        "<h1>Self-play planner suite "
        '<span class="cite">(self_play_robots)</span></h1>\n'
        '<p class="banner">Single source of truth for the AlphaZero-style '
        "self-play track. Auto-generated by <code>gen_report.py</code> from "
        "files on disk &mdash; no hand-typed numbers; a missing file renders as "
        f"<em>pending</em>. Generated {stamp}; {n_read} file(s) read, "
        f"{n_missing} expected file(s) not yet on disk" + upd_txt + ".</p>\n"
        "<noscript><p class='banner'>JavaScript is off, so every tab's content "
        "is shown stacked below and the tab strip acts as plain jump links."
        "</p></noscript>\n"
        + body_html + "\n"
        "<footer>LOCAL FILE ONLY &mdash; generated " + stamp +
        " by <code>self_play_robots/report/gen_report.py</code> from files on "
        "disk; never published.</footer>\n"
        f"<script>{JS}</script>\n"
        "</body></html>\n")
    tmp = OUT.with_suffix(".html.tmp")
    tmp.write_text(page)
    tmp.replace(OUT)

    scan = SCAN
    m0_files = len(scan["comparison"].get("m0", []))
    arms, _ = arena_arms()
    n_arms = len([a for a in arms.values() if a.ref]) if arms else 0
    ceil_files = len(scan["ceiling"])
    print(f"gen_report: wrote {OUT} ({len(page)} bytes) | read {n_read} file(s), "
          f"{n_missing} pending, {len(STATS['error'])} error(s) | "
          f"M0 {m0_files}/{n_arms} arm result(s), ceiling "
          f"{ceil_files}/{len(CEIL_EXPECTED)} arm(s), "
          f"{sum(len(v) for v in scan['comparison'].values())} comparison "
          f"payload(s) + {sum(len(v) for v in scan['summary'].values())} "
          f"summary.json under results/")
    for e in STATS["error"]:
        print(f"gen_report: WARN {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
