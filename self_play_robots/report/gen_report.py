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

import glob
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent          # self_play_robots/report
SPR = HERE.parent                               # self_play_robots
REPO = SPR.parent                               # MCTS_evolution
SV = REPO / "supervised_valuenet"               # the supervised stack
RESULTS = SPR / "results"
OUT = HERE / "selfplay.html"

sys.path.insert(0, str(HERE))                   # report/compare.py: the
from compare import Entry as CmpEntry           # noqa: E402  apples-to-apples
from compare import compare as cmp_compare      # noqa: E402  comparison engine
from story import STORY_CSS, sec_story          # noqa: E402  the Story tab
from supervised import sec_supervised           # noqa: E402  supervised-campaign tab

STATS = {"read": [], "missing": [], "error": []}
_CACHE: dict = {}

MILESTONE_KEYS = ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]
# where a status-strip chip jumps to (milestone/side-study key -> anchor)
STRIP_LINKS = {"M0": "#m0", "M1": "#res-m1", "M2": "#res-m2", "M3": "#res-loops",
               "M4": "#res-loops", "M5": "#res-mix", "M6": "#milestones",
               "ceiling": "#ceiling", "forward_mcts": "#res-fwd",
               "transfer": "#res-transfer"}

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

# plain-language reading of the dense gate quotes (jargon audit MS3)
PLAIN_GATE = {
    "M0": "pass if the new harness reproduces the old recorded results.",
    "M2": "pass if tree search beats the one-shot chooser (greedy descent = "
          "always take the value network&rsquo;s top pick) at the same solve "
          "rate.",
    "M6": "self-play on boards so big that no exact checker exists; trust "
          "comes from the accuracy curve measured up to 64&times;64.",
    "M1": "pass if it matches the hand-built pair within normal run-to-run "
          "variation.",
    "M3": "pass if the new network beats the previous one beyond run-to-run "
          "noise, and label quality holds.",
    "M4": "pass if the benchmarks keep improving and the final network "
          "clearly beats the supervised baseline.",
    "M5": "pass if the networks, trained at small sizes, still score well on "
          "boards up to 64&times;64 where exact answers exist.",
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


def chip_class(kind) -> str:
    """Chip colour from a status string: exact table first, then keywords
    (status.json carries free text such as "F-M0 pass; F-M2 done" or
    "B2 loop iteration 1 done (...); iterations 2-3 chained")."""
    k = str(kind).strip().lower()
    if k in CHIP_CLASS:
        return CHIP_CLASS[k]
    if "fail" in k:
        return "bad"
    if any(w in k for w in ("running", "queued", "chained", "in progress", "resumed")):
        return "run"
    if any(w in k for w in ("pass", "done", "complete")):
        return "good"
    if "partial" in k:
        return "warn"
    return "pend"


def chip_word(status) -> str:
    """One plain word for a chip label; the full status string belongs in a
    hover, not on the chip (naive-reader audit)."""
    s = str(status).strip()
    w = re.split(r"[:(;,]", s, 1)[0].strip().lower()
    first = w.split()[0] if w else ""
    return {"pass": "pass", "done": "done", "running": "running",
            "started": "running", "queued": "queued", "pending": "pending",
            "partial": "partial", "complete": "done", "on": "on track",
            "waves": "done", "drafting": "in progress", "curriculum": "done",
            "in": "in progress"}.get(first, (w[:14] or "unknown"))


def chip(kind, text=None) -> str:
    cls = chip_class(kind)
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


def is_unseen(payload) -> bool:
    """The 200-puzzle unseen exam (fresh boards, ids 20000+): not the
    frontier set, even though it also lacks a full optimum."""
    f = str((payload.get("protocol") or {}).get("instances_file") or "")
    return "unseen" in f or "/exam/" in f


def is_frontier(payload, aggregate) -> bool:
    """A frontier (beyond-oracle) set: no exact optimum exists for it."""
    if is_unseen(payload):
        return False
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
        bits.append(f'<span title="instances_sha256 {esc(str(sha))}">exam '
                    f"fingerprint {esc(str(sha)[:8])}&hellip;</span>")
    if p.get("device"):
        bits.append(esc(p["device"]))
    if p.get("date"):
        bits.append(esc(p["date"]))
    return " &middot; ".join(bits)


def _pw_cell(pw):
    """Render one pairwise-delta cell from compare.py's pairwise dict."""
    if not pw:
        return DASH
    if pw["delta"] is None:
        return '<td class="pend">no shared solves</td>'
    good = pw["delta"] < 0 and pw["sign_p"] < 0.05
    return td_txt(f"{pw['delta']:+.2f} moves &middot; {pw['wins']}/{pw['losses']} "
                  f'&middot; <span title="exact sign test, p={pw["sign_p"]:.2g}, '
                  f'on the {pw["both_n"]} puzzles both solved">'
                  f"{fluke(pw['sign_p'])}</span>",
                  cls="hlnum" if good else "")


_H2H_WHY_SHOWN = False


def h2h_table(entries, ref_labels, note_extra="", dstar=None):
    """One apples-to-apples table: common-subset moves + paired deltas.

    `entries` are compare.CmpEntry objects (label + payload + system filter);
    `ref_labels` name which entries the delta columns compare against.
    `dstar`: optional per-position exact optima (else read off graded rows);
    when known, a "vs perfect play" column appears.
    """
    r = cmp_compare(entries, ref_labels=ref_labels, dstar=dstar)
    if r.get("error"):
        return (f'<p class="note"><span class="chip pend">not renderable</span> '
                f"head-to-head skipped: {esc(r['error'])}</p>")
    n_ok = sum(1 for e in r["entries"] if e["ok"])
    perfect = bool(r.get("perfect_n"))
    headers = (["system", "solved",
                f"moves, on the {r['common_n']} puzzles all {n_ok} systems solved"]
               + [f"&Delta; moves vs {esc(rl)} <span class=\"note\">(both-solved; "
                  f"&minus; = fewer = better)</span>" for rl in ref_labels]
               + ([f"&Delta; vs perfect play <span class=\"note\">(the "
                   f"{r['perfect_n']} common puzzles with a known optimum; "
                   f"mean optimum {r['perfect_mean']:.2f})</span>"]
                  if perfect else [])
               + ["search effort (expansions)"])
    tech_of = {en.label: getattr(en, "tech", None) for en in entries}

    def _lab(label):
        t = tech_of.get(label)
        return (f'<span title="{esc(t)}">{esc(label)}</span>' if t
                else esc(label))

    rows_ = []
    for e in r["entries"]:
        if not e["ok"]:
            rows_.append(row([td_txt(_lab(e["label"])),
                              f'<td class="pend" colspan="{len(headers) - 1}">'
                              "pending (job not finished)</td>"]))
            continue
        cells = [td_txt(_lab(e["label"])),
                 td_txt(f"{e['solved']}/{e['n']}"),
                 td(e["moves_common"], ".2f", cls="hlnum")]
        for rl in ref_labels:
            cells.append(td_txt("<em>reference</em>") if e["label"] == rl
                         else _pw_cell(e["pairwise"].get(rl)))
        if perfect:
            cells.append(td_txt(f"+{e['delta_perfect']:.2f} moves"
                                if e["delta_perfect"] is not None else "&mdash;",
                                cls="hlnum" if e["delta_perfect"] is not None
                                and e["delta_perfect"] < 1.0 else ""))
        cells.append(td(e["mean_exp"], ".0f"))
        rows_.append(row(cells))
    global _H2H_WHY_SHOWN
    if _H2H_WHY_SHOWN:
        why = ("Same fair-comparison rule as the first head-to-head table "
               "above. ")
    else:
        why = ("Why these columns: a system&rsquo;s own &ldquo;mean moves&rdquo; averages over "
               "the puzzles <em>it</em> solved, so raw means from two systems are NOT comparable "
               "&mdash; the one that solves more hard puzzles looks worse. The "
               "<strong>common-subset moves</strong> column scores every system on the exact same "
               "puzzles. Each <strong>&Delta;</strong> column is paired on the puzzles both "
               "systems solved, with win/loss counts and the fluke chance (p, exact sign test). ")
        _H2H_WHY_SHOWN = True
    note = why + note_extra
    return table(headers, rows_, note=note, cls="wide")


BASE_HEADERS = ["config", "set", "system", "solve rate", "mean realized moves",
                "extra moves vs optimum", "% optimal",
                "search effort (expansions)", "seconds per puzzle",
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
                            f'<td class="pend" colspan="6" title="no aggregate '
                            f'in file (kind={esc(kind)})">pending</td>'],
                           cls))
            continue
        front = is_frontier(d, a)
        n, solved = a.get("n"), a.get("solved")
        sr = (f"{solved}/{n} &middot; {100 * a['solve_rate']:.1f}%"
              if None not in (n, solved) and a.get("solve_rate") is not None
              else None)
        out.append(row([
            td_txt(esc(cfg)),
            td_txt('<span class="tag">unseen exam</span>' if is_unseen(d)
                   else ('<span class="tag front">frontier</span>' if front
                         else '<span class="tag graded">graded</span>')),
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


# ---------------------------------------------------------------------------
# shared plain-language helpers (jargon audit REVIEW_jargon.md, systemic fixes
# R1/R2/R3): use these in every section so slugs, p-values and status notes
# render for a human reader. Other section owners: import/reuse, do not fork.
# ---------------------------------------------------------------------------

CFG_WORDS = {}          # slug -> "24&times;24, 4 robots"


def cfg_label(slug: str, suffix: str = "") -> str:
    """Plain words for a config slug, slug kept as a hover (systemic fix R2)."""
    lab = CFG_WORDS.get(slug)
    if lab is None:
        m = re.fullmatch(r"g(\d+)r(\d+)", slug)
        if m:
            n, r = m.group(1), m.group(2)
            lab = f"{n}&times;{n}, {r} robot" + ("s" if r != "1" else "")
            CFG_WORDS[slug] = lab
    if lab is None:
        return esc(slug)
    tail = f" {suffix}" if suffix else ""
    return f'<span title="{esc(slug)}">{lab}{tail}</span>'


def fluke(p) -> str:
    """A p-value as plain language for use inside a sentence (systemic fix R3).
    Raw p's belong in table cells or details blocks, not prose."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if p > 0.05:
        return f"could be a fluke (p&nbsp;=&nbsp;{p:.2g})"
    for bound, words in ((1e-12, "one in a trillion"), (1e-9, "one in a billion"),
                         (1e-6, "one in a million"), (1e-3, "one in a thousand"),
                         (0.05, "one in twenty")):
        if p <= bound:
            return f"fluke chance below {words}"
    return f"fluke chance below one in twenty"


def note_html(entry: dict) -> str:
    """Milestone/side-study note for display (systemic fix R1): the plain_note
    is the reading surface; the raw FINDINGS-register note collapses into a
    details block for auditors."""
    plain, raw = entry.get("plain_note"), entry.get("note")
    if plain and raw:
        return (esc(plain) + ' <details class="inl"><summary>log text</summary>'
                f'<span class="note">{esc(raw)}</span></details>')
    return esc(plain or raw or "")


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
        if not p.exists() and not Path(s).is_absolute() and (REPO / s).exists():
            p = REPO / s
        # sources may be dirs, globs or brace patterns -- expand before judging
        cand = str(p)
        found = p.exists()
        if not found and "{" in cand:
            import itertools
            parts = re.split(r"\{([^}]*)\}", cand)
            opts = [parts[0]]
            for i in range(1, len(parts), 2):
                alts = parts[i].split(",")
                tail = parts[i + 1] if i + 1 < len(parts) else ""
                opts = [o + a + tail for o in opts for a in alts]
            found = any(Path(o).exists() or glob.glob(o + "*") for o in opts)
        if not found:
            found = bool(glob.glob(cand + "*") or glob.glob(cand))
        mark = "" if found else ' <span class="miss">(not yet)</span>'
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
    BW, BH = 270, 68
    boxes = [
        (20, 40, "1 &middot; Generate",
         ["fresh lean boards + instances,", "every iteration (boards ~free)"]),
        (325, 40, "2 &middot; Search with net k",
         ["MCTS over the chosen action", "space, fixed expansion budget"]),
        (630, 40, "3 &middot; Certify",
         ["replay each solution vs physics;", "uncertified plans are dropped"]),
        (630, 200, "4 &middot; Keep recent data",
         ["a rolling window of verified", "solutions (origin recorded)"]),
        (325, 200, "5 &middot; Train the next nets",
         ["start from the last ones;", "collapse alarm; keep best pass"]),
        (20, 200, "6 &middot; Examine",
         ["test vs the previous nets and", "the fixed baselines"]),
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

    arrow(290, 74, 321, 74)                       # 1 -> 2
    arrow(595, 74, 626, 74)                       # 2 -> 3
    arrow(765, 108, 765, 196)                     # 3 -> 4
    arrow(630, 234, 599, 234)                     # 4 -> 5
    arrow(325, 234, 294, 234)                     # 5 -> 6
    arrow(145, 200, 145, 112)                     # 6 -> 1  (promote)
    arrow(145, 268, 145, 300, "arr dashed")       # 6 -> diagnose
    s.append('<rect x="20" y="300" width="270" height="52" rx="8" '
             'class="nbox alt"/>')
    s.append('<text x="34" y="322" class="nt">Worse? Investigate first</text>')
    s.append('<text x="34" y="340" class="ns">bad labels? damaged data?</text>')
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
        "<p><strong>The goal, in one line:</strong> beat both hand-taught "
        "planners on real moves used, on a fixed exam, with the same search "
        "budget.</p>")
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
               "<li>The game has no randomness. An exact solver exists only up to "
               "64&times;64 boards. Physics replay checks <em>validity</em> at "
               "any size, so only <em>optimality</em> is capped at 64."
               "</li></ul>"
               "<p class='cite'>PROBLEM.md &sect;2</p>")
    out.append(
        "<h3>Two action-space views (both have trained nets)</h3>"
        + table(["view", "actions", "horizon (how many decisions deep)",
                 "branching (choices per decision)", "status here"],
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
                note="Definitions quoted from PROBLEM.md &sect;2. The "
                     "recommendation and its current status live on the "
                     "<a href='#loop'>Loop tab</a>, inside the collapsed "
                     "engineering-plan block."))
    out.append(
        "<h3>The success statement</h3>"
        "<blockquote>&ldquo;Self-play planner X solves &ge; "
        "backward-supervised solve rate with mean realized moves closer to "
        "optimal than forward-supervised, at backward-supervised "
        "compute.&rdquo;<span class='cite'> PROBLEM.md &sect;7</span>"
        "</blockquote>"
        "<p class='note'>Results that fall short of the full goal still count "
        "&mdash; every milestone below has its own pass bar. Fixed rules for "
        "every test: score only replayed moves, use only the pinned exams, "
        "certify every solve. Optimality is reported only where the exact "
        "solver works (boards up to 64&times;64).</p>")

    # status strip
    out.append('<h3 id="strip">Milestone status</h3><div class="strip">')
    for k in MILESTONE_KEYS:
        m = milestone(k)
        stt = m.get("status", "unknown")
        title = m.get("title", "")
        href = STRIP_LINKS.get(k, "#milestones")
        tip_txt = "; ".join(x for x in (stt, m.get("plain_note") or m.get("note")) if x)
        tip = f' title="{esc(tip_txt)}"' if tip_txt else ""
        out.append(f'<a class="stripitem" href="{href}"{tip}><span class="mk">'
                   f'{esc(k)}</span>{chip(stt, chip_word(stt))}<span class="mt">'
                   f'{esc(title)}</span></a>')
    out.append("</div>")
    ss = status().get("side_studies") or {}
    if ss:
        out.append('<div class="strip">')
        for k, v in ss.items():
            stt = (v or {}).get("status", "unknown")
            href = STRIP_LINKS.get(k, "#results")
            tip_txt = "; ".join(x for x in (stt, (v or {}).get("plain_note") or (v or {}).get("note")) if x)
            tip = f' title="{esc(tip_txt)}"' if tip_txt else ""
            out.append(f'<a class="stripitem" href="{href}"{tip}><span class="mk">'
                       f'{esc(k)}</span>{chip(stt, chip_word(stt))}<span class="mt">'
                       f'{esc((v or {}).get("title", ""))}</span></a>')
        out.append("</div>")
    upd = st.get("updated")
    nh = st.get("node_hours") or {}
    meta = []
    if upd:
        meta.append(f"status manifest last updated {esc(upd)}")
    if nh:
        proj = nh.get("projected_next")
        meta.append(
            f"about {esc(nh.get('spent'))} node-hours of compute used so far"
            + (f", roughly {esc(proj)} planned next" if proj else "")
            + (f' <details class="inl"><summary>detail</summary>'
               f'<span class="note">{esc(nh.get("note"))}</span></details>'
               if nh.get("note") else ""))
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
            f"<li>{chip(stt, chip_word(stt))} <strong>{esc(k)} &mdash; "
            f"{esc(m.get('title', ''))}</strong>: {note_html(m)}"
            + (f' <span class="note">job(s) {j}</span>' if j != "&mdash;" else "")
            + f'<br><span class="note">sources: {sources_txt(m)}</span></li>')
    for k, v in (status().get("side_studies") or {}).items():
        v = v or {}
        j = jobs_txt(v)
        bullets.append(
            f"<li>{chip(v.get('status', 'unknown'), chip_word(v.get('status', 'unknown')))} "
            f"<strong>side study &mdash; {esc(v.get('title', k))}</strong>"
            + (f": {note_html(v)}" if (v.get("plain_note") or v.get("note")) else "")
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
           "<p><strong>One round</strong> (PROBLEM.md &sect;5): make fresh puzzles "
           "on fresh boards &rarr; search them with the current networks under "
           "a fixed budget &rarr; keep only replay-verified solutions &rarr; "
           "train the next networks starting from the last ones, with the "
           "collapse alarm on &rarr; examine them against the previous round "
           "and the fixed baselines &rarr; keep or investigate.</p>",
           '<figure class="fig">' + svg_loop() +
           "<figcaption>The bootstrap (iteration 0) is free: initialize from "
           "the supervised backward planner pair copied into "
           + src("self_play_robots/assets/") +
           " — or, if &sect;6.2 decides on the size-free rebuild, train the "
           "new architecture on the existing exact corpora first and verify it "
           "reproduces supervised-level bench numbers <em>before</em> any "
           "self-play. Never debug architecture and loop dynamics at the same "
           "time.</figcaption></figure>"]
    out.append(
        '<p>That is the whole idea. The rest of this tab is the original '
        'engineering plan the loop was built from &mdash; kept for auditors, '
        'collapsed below. The measured results live in the '
        '<a href="#results">Milestone results tab</a>.</p>')
    out.append('<details><summary>The original engineering plan '
               '(design quotes, risk register, open decisions &mdash; for '
               'auditors)</summary>')
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
        "honestly:</strong> the labeling network already picks the same best "
        "candidate as the exact solver 86&ndash;93% of the time, and the tree "
        "search finds solutions the one-shot chooser misses. If search-made "
        "data trains a network that searches better, the loop climbs. If it "
        "merely matches the supervised "
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
        "PROBLEM.md and need no owner sign-off. <strong>&sect;6.3 value target "
        "form:</strong> keep the 96-bin cost output, and make sure the bins "
        "cover the biggest real costs (a past bug silently capped them). "
        "<strong>&sect;6.5 search budget:</strong> compare planners at 1,200 "
        "expansions with 5 candidates per step. Self-play generation may "
        "spend more, but never in a way that biases the data.</p>")
    out.append("</details>")
    out.append("</section>")
    return "\n".join(out)


def capfirst(x: str) -> str:
    return x[:1].upper() + x[1:] if x else x


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
        bits.append((label, ds, do))
    if not bits:
        return ""
    sent = []
    for i, (label, ds, do) in enumerate(bits):
        who = ("the exact-taught pair" if "exact" in label
               else "the NN-taught twin (the same recipe retrained on "
                    "network-written labels)" if "twin" in label.lower()
               else label)
        upto = "up to " if i else ""
        f1 = lambda v: f"{v:.1f}".rstrip("0").rstrip(".")
        if do is None:
            sent.append(f"{who} disagrees by {upto}{f1(ds)} solved puzzles")
        elif i == 0:
            sent.append(f"{who} disagrees by {f1(ds)} solved puzzles and "
                        f"{f1(do)} optimality points")
        else:
            sent.append(f"{who} disagrees by up to {f1(ds)} and {f1(do)}")
    return ("<p class='note'><strong>The yardstick, measured from these very "
            "files.</strong> Two identical runs that differ only in their "
            "random start: " + sent[0] + ". "
            + ". ".join(capfirst(x) for x in sent[1:])
            + (". " if len(sent) > 1 else "")
            + "(Optimality points = percentage-point difference in puzzles "
            "solved with a perfect-length answer.) The gate is set just above "
            "the measured spread, at 3.5 solve and 6.6 optimality points. Any "
            "win smaller than this bar is noise.</p>")


def g16_runs_line() -> str:
    """Disambiguate the four recorded 16x16 runs (naive-reader #2, item 1):
    two of them differ only in the realization rule and two are different B2
    arms -- nearby tabs quote different ones, and side by side they can read
    as the same run disagreeing. Numbers computed from the payloads."""
    runs = [("eval/results/final450_backward_prefix.json",
             "the pair under the prefix-check rule (quoted here and in M0)"),
            ("eval/results/final450_backward_anytime.json",
             "the SAME pair under the anytime rule (quoted on the supervised "
             "tab's ladder)"),
            ("eval/results/final450_backward_b2.json",
             "the original B2-vocabulary arm (supervised tab)"),
            ("eval/results/final450_backward_b2_seed21.json",
             "the seed-21 B2 replicate (this project's M0 parity target)")]
    bits = []
    for rel, what in runs:
        _, sy = backward_system(load_sv(rel))
        a = (sy or {}).get("aggregate") or {}
        if a.get("solved") is None:
            continue
        r = a.get("mean_regret")
        bits.append(f"{a['solved']}/{a['n']} &middot; "
                    + (f"{r:.2f}" if r is not None else "&mdash;")
                    + f" extra moves &mdash; {what}")
    if len(bits) < 2:
        return ""
    return ("<p class='note'><strong>Reading 16&times;16 numbers across "
            "tabs:</strong> four distinct recorded runs exist and they are "
            "different runs, not one run disagreeing: "
            + "; ".join(bits) + ".</p>")


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
                     f"<strong>k={esc(p.get('k'))}</strong> (keep the top "
                     f"{esc(p.get('k'))} candidates at each step), device "
                     f"{esc(p.get('device'))}, exam "
                   + src(p.get("instances_file"))
                   + f' (<span title="instances_sha256 '
                     f'{esc(str(p.get("instances_sha256")))}">exam '
                     f"fingerprint {esc(str(p.get('instances_sha256'))[:8])}"
                     f"&hellip;</span>, n={esc(p.get('n_instances'))}). "
                     "Every system MAY use up to that many expansions per "
                     "puzzle; the search-effort column shows how many it "
                     "actually needed &mdash; using fewer is better. "
                     "<details><summary>Expansion definition</summary>"
                     f"{esc(p.get('expansion_definition'))}</details></p>")
    out.append(seed_spread_line())
    out.append(g16_runs_line())

    # ---- head-to-head: ONE comparable table per exam (owner 2026-08-21) ----
    out.append("<h3>Head-to-head on the shared exam &mdash; forward vs backward "
               "vs this project, same puzzles, same columns</h3>")
    out.append("<p class='note'>The inventory table further down keeps each "
               "system&rsquo;s own aggregates (useful as provenance, but its moves "
               "columns are computed over different solve sets and must not be "
               "compared across rows). The tables here fix that: all systems on one "
               "exam are scored on the <em>same</em> puzzles. The &ldquo;this "
               "project&rdquo; rows use the final self-play networks (the run "
               "name sits in each row label). At 16&times;16 they use this "
               "project&rsquo;s size-free network pair instead, because no "
               "self-play loop ran at that size.</p>")
    h2h_specs = [
        ("g16r4", "16&times;16, 4 robots &mdash; legacy 450-puzzle exam", [
            CmpEntry("backward supervised",
                     load_sv("eval/results/final450_backward_prefix.json"),
                     prefer_kind="backward"),
            CmpEntry("forward supervised",
                     load_sv("eval/results/comparison_forward.json"),
                     prefer_kind="forward", name_contains="candidate_scored"),
            CmpEntry("size-free pair (this project, M1)",
                     load(RESULTS / "m1" / "mixed_value_warm_s21_g16r4.json"),
                     tech="M1 size-free pair mixed_value_warm_s21"),
        ]),
        ("g24r4", "24&times;24, 4 robots &mdash; pinned graded exam", [
            CmpEntry("backward supervised",
                     load_sv("scaling/results/g24r4/comparison.json"),
                     prefer_kind="backward"),
            CmpEntry("forward supervised",
                     load_sv("scaling/results/g24r4/comparison.json"),
                     prefer_kind="forward"),
            CmpEntry("this project's self-play planner",
                     load(RESULTS / "selfplay" / "mix_b2mix_iter3" /
                          "bench_g24r4_astar.json"),
                     tech="mix_b2mix_iter3 nets, B2 anytime A*"),
            CmpEntry("current best planner (flagship)",
                     load(RESULTS / "variants" / "v07_hybrid_actions" /
                          "bench_graded_hybrid_d2.json"),
                     tech="v09 strict-value nets + depth-2 slide-prefix hybrid search"),
        ]),
        ("g24r8", "24&times;24, 8 robots &mdash; pinned graded exam", [
            CmpEntry("backward supervised",
                     load_sv("scaling/results/g24r8/comparison.json"),
                     prefer_kind="backward"),
            CmpEntry("forward supervised",
                     load_sv("scaling/results/g24r8/comparison.json"),
                     prefer_kind="forward"),
            CmpEntry("this project's self-play planner",
                     load(RESULTS / "selfplay" / "mix_b2mix_iter3" /
                          "bench_g24r8_astar.json"),
                     tech="mix_b2mix_iter3 nets, B2 anytime A*"),
            CmpEntry("the flagship, never trained at this size",
                     load(RESULTS / "variants" / "v07_transfer" /
                          "bench_g24r8_graded_hybrid_d2.json"),
                     tech="v09 nets + depth-2 hybrid, zero-shot transfer"),
            CmpEntry("same nets, standard search (transfer control)",
                     load(RESULTS / "variants" / "v07_transfer" /
                          "bench_g24r8_graded_stdmcts.json")),
        ]),
        ("g32r4", "32&times;32, 4 robots &mdash; pinned graded exam", [
            CmpEntry("backward supervised",
                     load_sv("scaling/results/g32r4/comparison.json"),
                     prefer_kind="backward"),
            CmpEntry("forward supervised",
                     load_sv("scaling/results/g32r4/comparison.json"),
                     prefer_kind="forward"),
            CmpEntry("this project's self-play planner",
                     load(RESULTS / "selfplay" / "mix_b2mix_iter3" /
                          "bench_g32r4_astar.json"),
                     tech="mix_b2mix_iter3 nets, B2 anytime A*"),
            CmpEntry("the flagship, never trained at this size",
                     load(RESULTS / "variants" / "v07_transfer" /
                          "bench_g32r4_graded_hybrid_d2.json"),
                     tech="v09 nets + depth-2 hybrid, zero-shot transfer"),
            CmpEntry("same nets, standard search (transfer control)",
                     load(RESULTS / "variants" / "v07_transfer" /
                          "bench_g32r4_graded_stdmcts.json")),
        ]),
    ]
    for cfg, human, ents in h2h_specs:
        out.append(f"<h4><code>{esc(cfg)}</code> &mdash; {human}</h4>")
        out.append(h2h_table(ents, ("backward supervised", "forward supervised")))
        if cfg == "g24r4":
            out.append("<p class='note'>Letting the search try up to three "
                       "single robot moves before subgoal planning "
                       "(&ldquo;depth-3 slide prefixes&rdquo;) pushes the "
                       "flagship further still: 0.861 extra moves over the "
                       "known optimum, against depth-2&rsquo;s 0.944, with the "
                       "same 231/232 solves. The depth study lives on the "
                       "Variants tab (<code>v07</code>).</p>")
    out.append("<h4><code>g24r8</code> frontier &mdash; 24&times;24, 8 robots, "
               "the 289 puzzles no supervised solver fully cracked</h4>")
    out.append(h2h_table(
        [CmpEntry("backward supervised",
                  load_sv("scaling/results/g24r8/comparison_ungraded.json"),
                  prefer_kind="backward"),
         CmpEntry("forward supervised",
                  load_sv("scaling/results/g24r8/comparison_ungraded.json"),
                  prefer_kind="forward"),
         CmpEntry("the flagship, never trained at this size",
                  load(RESULTS / "variants" / "v07_transfer" /
                       "bench_g24r8_frontier_hybrid_d2.json"),
                  tech="v09 nets + depth-2 hybrid, zero-shot transfer"),
         CmpEntry("same nets, standard search (transfer control)",
                  load(RESULTS / "variants" / "v07_transfer" /
                       "bench_g24r8_frontier_stdmcts.json"))],
        ("backward supervised", "forward supervised")))

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
                     note="Every number is read straight off that row&rsquo;s result "
                          "file. A row&rsquo;s moves average covers only the puzzles "
                          "that system solved, so rows here are provenance, not a "
                          "fair race &mdash; compare systems in the head-to-head "
                          "tables above. &ldquo;frontier&rdquo; marks exams with no "
                          "known optimum, and the <em>exact optimum</em> row is the "
                          "ceiling nothing can beat. "
                          "<details><summary>Column definitions, field by field "
                          "(for auditors)</summary>Columns: <em>solve rate</em> = "
                          "<code>aggregate.solved</code>/<code>aggregate.n</code> "
                          "and <code>aggregate.solve_rate</code>; <em>mean "
                          "realized moves</em> = <code>aggregate.mean_moves</code> "
                          "(identical to <code>mean_realized_strict</code> for "
                          "backward systems — playable moves after physics "
                          "replay); <em>extra moves vs optimum</em> = "
                          "<code>aggregate.mean_regret</code>; <em>% optimal</em> "
                          "= <code>aggregate.pct_optimal</code>; <em>search "
                          "effort</em> / <em>seconds per puzzle</em> = "
                          "<code>aggregate.mean_expansions</code> / "
                          "<code>mean_seconds</code>; <em>set</em> is "
                          "<em>frontier</em> when the aggregate carries "
                          "<code>d_star_placeholder</code> or the protocol's "
                          "instances file has no <code>d_star</code> at all "
                          "(regret and optimality are then suppressed, because "
                          "no optimum exists there). The exact-optimum row "
                          "averages the whole file while a system averages its "
                          "own solves, so a system solving only the easy subset "
                          "can print a mean below the file's mean d* without "
                          "being better than optimal — always read mean moves "
                          "next to solve rate (PROBLEM.md &sect;1).</details>",
                     cls="wide"))
    out.append(
        "<p class='note'>Reading the 32&times;32 pair the way the project "
        "brief does: the forward planner is slower but near-optimal when it solves, "
        "the backward planner solves more and faster but further from optimal. "
        "<strong>Beating backward on moves while matching its solve rate "
        "comes close to forward's quality at backward's speed — that "
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
                      "with a known optimum", "mean known optimum"], brows,
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
           f"{GATES['M0'][1]} Estimated cost {GATES['M0'][0].replace('nh', 'node-hour(s)')}.</p>",
           "<p>The harness is "
           + src("self_play_robots/spr/arena.py")
           + " — a thin wrapper that <em>calls</em> the original benchmark code "
           "directly, "
           + src("supervised_valuenet/eval/compare.py")
           + " (it keeps no copy of its own). It runs the exam in 8 parallel parts and merges "
             "them. An independent replay check verifies every solution. "
             "Then every row is compared against the recorded result. A "
             "run that fails the replay check is set aside, never "
             "reported.</p>"]
    m = milestone("M0")
    if m:
        out.append(f'<p class="verdict">{chip(m.get("status", "unknown"), chip_word(m.get("status", "unknown")))} '
                   f'{note_html(m)} <span class="note">job(s) '
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
    out.append("<h3>The registered test setups</h3>")
    out.append(table(["setup", "config", "instances", "eval.compare flags",
                      "expansions / k", "recorded reference", "notes"], rrows,
                     note="Read live from the harness&rsquo;s own list of "
                          "test setups ("
                          + src("self_play_robots/spr/arena.py")
                          + "), so this table always matches what actually "
                            "runs — the same list the job script drives, so this "
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
            bits = [f"same exam: {'yes' if same_exam else '<strong>NO</strong>'}",
                    (f"row counts {len(nrows_)} vs {len(rrows_)}"
                     if not same_n else f"{len(nrows_)} rows on both sides"),
                    f"solved {na.get('solved')} vs {ra.get('solved')}"]
            # pass criterion (FINDINGS 2): the TOTALS must match on the same
            # exam; per-row float drift across machines is explained, not a
            # failure. Row-exact is reported as the stronger result when true.
            agg_ok = (same_exam and same_n
                      and na.get("solved") == ra.get("solved")
                      and na.get("pct_optimal") == ra.get("pct_optimal"))
            passed = agg_ok
            vcls = "good" if passed else "bad"
            if not diffs and agg_ok:
                plainv = f"exact match &mdash; 0 of {len(nrows_)} rows differ"
            elif agg_ok:
                try:
                    dreg = abs((na.get("mean_regret") or 0)
                               - (ra.get("mean_regret") or 0))
                    dtxt = (f", shifting mean extra-moves by {dreg:.3f}"
                            if dreg > 1e-9 else "")
                except TypeError:
                    dtxt = ""
                plainv = (f"totals match ({na.get('solved')} solved, "
                          f"{(na.get('pct_optimal') or 0):.1f}% optimal on both "
                          f"sides); {len(diffs)} of {len(nrows_)} rows differ "
                          f"only by floating-point rounding across machines "
                          f"(re-running with a different number of CPU threads "
                          f"reproduces exactly these rounding differences)"
                          f"{dtxt}")
            else:
                plainv = f"{len(diffs)} of {len(nrows_)} rows differ"
            out.append(f'<p class="verdict {vcls}">{chip("pass" if passed else "fail", "parity " + ("PASS" if passed else "FAIL"))} '
                       f'{plainv}. <details class="inl"><summary>checked fields</summary>'
                       f'<span class="note">compared per row: solved, '
                       f'realized_strict, expansions, plan_found. '
                       + "; ".join(bits) + '.</span></details></p>')
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
           "<p><strong>Why (quoted from the brief).</strong> PROBLEM.md &sect;6.1: <em>&ldquo;the "
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
           "<p><strong>How the probe works.</strong> "
           + src("self_play_robots/spr/ceiling.py")
           + " searches every expressible plan in cheapest-plan-first order "
             "(the solver's own safe ordering &mdash; <em>no network "
             "anywhere</em>). It replays every complete plan in the physics "
             "and records the first and the best plan that works. The result "
             "is a tight <em>upper bound</em> on the language optimum, not a "
             "proof. A plan can occasionally cost fewer real moves than its "
             "plan-step estimate (a robot happens to stand in a useful spot), "
             "and such a plan could hide beyond where the search stopped.</p>",
           '<p class="note">Outcome categories used in the per-run tables '
           'below: REALIZABLE_EXISTS = the language can write a plan that '
           'works. NO_REALIZABLE_PLAN = plans exist on paper but none works. '
           'NO_COMPLETE_PLAN = the generator never finished a plan. '
           'INCONCLUSIVE = the search hit its cap. ERROR = the puzzle threw '
           'an error (recorded, never lost). &ldquo;Slack&rdquo; in a '
           'heading = how much further past the first answer the probe '
           'keeps searching. More slack = a deeper probe = a tighter '
           'proven limit.</p>']
    ss = side_study("ceiling")
    if ss:
        out.append(f'<p class="verdict">{chip(ss.get("status", "unknown"), chip_word(ss.get("status", "unknown")))} '
                   f'{note_html(ss)} <span class="note">job(s) '
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
        caps = d.get("caps") or {}
        cats = s.get("categories") or {}
        inst = str(d.get("instances") or "")
        front = (s.get("n_graded_realizable") in (0, None)
                 and bool(s.get("n_realizable"))) or "unsolved" in inst
        capped = s.get("capped") or 0
        inconclusive = cats.get("INCONCLUSIVE", 0)
        variant = f.stem
        vbits = []
        if caps.get("slack"):
            vbits.append(f"slack {caps['slack']:g}")
        if caps.get("time_cap"):
            vbits.append(f"{caps['time_cap']:g} s")
        if caps.get("max_frontier"):
            vbits.append(f"search queue cap {caps['max_frontier'] // 1000}k")
        if capped:
            reading = (f'<span class="tag front">lower bound</span> probe hit a cap on '
                       f'{capped} instance(s), {inconclusive} inconclusive: the solve '
                       f'ceiling is &ge; {s.get("n_realizable")}/{s.get("n")}'
                       + (" and the best-plan columns are upper bounds on the language optimum"
                          if not front else ""))
        else:
            reading = "no cap hit: solve ceiling exact for these caps"
        sr = s.get("solve_ceiling")
        rows_.append(row([
            td_txt(f"<strong>{esc(d.get('config'))}</strong> / "
                   f"{esc(d.get('vocab'))}<br><span class='note'><code>{esc(variant)}</code>"
                   + (f" &middot; {' &middot; '.join(vbits)}" if vbits else "") + "</span>"),
            td_txt('<span class="tag">unseen exam</span>' if is_unseen(d)
                   else ('<span class="tag front">frontier</span>' if front
                         else '<span class="tag graded">graded</span>')),
            td(s.get("n"), "d"),
            td_txt((f"{s.get('n_realizable')}/{s.get('n')} &middot; {100 * sr:.1f}%"
                    + (" <strong>&ge;</strong>" if capped else ""))
                   if sr is not None else '<span class="pend">pending</span>'),
            td(s.get("n_graded_realizable"), "d"),
            td(s.get("mean_d_star"), ".2f") if not front else DASH,
            td(s.get("mean_best_moves"), ".2f", cls="hlnum") if not front else DASH,
            td(s.get("mean_gap_best"), ".2f") if not front else DASH,
            td_pct100(s.get("pct_best_optimal")) if not front else DASH,
            td(s.get("mean_first_moves"), ".2f") if not front else DASH,
            td(s.get("mean_gap_first"), ".2f") if not front else DASH,
            td_pct100(s.get("pct_first_optimal")) if not front else DASH,
            td(capped, "d"),
            td_txt(f'<span class="note">{reading}</span>'),
            td_txt(src(disp(f))),
        ], "hl" if not capped else ""))
    for c, v in expected_missing:
        rows_.append(row([td_txt(f"<strong>{esc(c)}</strong> / {esc(v)}"),
                          '<td class="pend" colspan="13">pending</td>',
                          td_txt(src(f"self_play_robots/results/ceiling/{c}_{v}.json"))]))
    out.append(table(["config / vocab (file, caps)", "set", "n", "solve ceiling",
                      "graded &amp; realizable", "mean d*", "mean best-plan moves",
                      "mean gap (best)", "% best = optimal",
                      "mean first-plan moves", "mean gap (first)",
                      "% first = optimal", "capped", "reading", "source file"], rows_,
                     note="Every <code>results/ceiling/*.json</code> carrying a "
                          "<code>summary</code> (base, B2, slack re-runs, graded and "
                          "frontier sets). Columns from <code>summary</code>: "
                          "<code>n</code>, <code>n_realizable</code>/<code>solve_ceiling</code>, "
                          "<code>n_graded_realizable</code>, "
                          "<code>mean_d_star</code>, <code>mean_best_moves</code>, "
                          "<code>mean_gap_best</code>, <code>pct_best_optimal</code>, "
                          "<code>mean_first_moves</code>, <code>mean_gap_first</code>, "
                          "<code>pct_first_optimal</code>, <code>capped</code>; the "
                          "<em>reading</em> column turns <code>capped</code> and "
                          "<code>categories.INCONCLUSIVE</code> into words: a capped "
                          "probe proves only a lower bound on the solve ceiling "
                          "(&ge;), which is why the frontier B2 arms must be read as "
                          "&ldquo;at least&rdquo;. Frontier sets carry no d*, so the "
                          "moves/gap/optimal columns are dashed there. "
                          "<em>first</em> = the first working plan the search "
                          "meets in cheapest-first order (what a simple planner "
                          "would take). <em>best</em> = the cheapest working "
                          "plan found before the ordering proves nothing "
                          "cheaper remains. Caps "
                          "(<code>caps</code> block: time cap, frontier size, slack) "
                          "are printed under the arm name.",
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
                   "write &mdash; while it speaks only this language.</strong></p>")
    for f, d in found:
        s = d["summary"]
        cfg, voc = d.get("config"), d.get("vocab")
        voc_words = {"base": "base vocabulary", "b1": "extended vocabulary (B1)",
                     "b2": "extended vocabulary (B2)"}.get(voc, f"{voc} vocabulary")
        mslack = re.search(r"slack(\d+)", f.stem)
        knob = (f"deeper probe (slack {mslack.group(1)}: searches more, "
                f"proves a tighter limit)" if mslack
                else ("frontier exam" if "frontier" in f.stem
                      else "standard probe"))
        out.append(f'<h3><span title="{esc(f.stem)}">{cfg_label(cfg)} &mdash; '
                   f'{esc(voc_words)}, {knob}</span></h3>')
        bits = []
        if s.get("capped"):
            cats = s.get("categories") or {}
            bits.append(
                f"The probe hit a cap on <strong>{esc(s.get('capped'))}</strong> "
                f"instance(s) ({esc(cats.get('INCONCLUSIVE', 0))} left inconclusive), "
                f"so its solve ceiling of {esc(s.get('n_realizable'))}/{esc(s.get('n'))} "
                f"is a <strong>lower bound</strong>"
                + (" and its best-plan moves an upper bound on the language optimum"
                   if s.get("mean_best_moves") is not None else ""))
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
            bits.append(f"The language reaches the exact optimum on "
                        f"<strong>{s['pct_best_optimal']:.1f}%</strong> of them")
        if s.get("solve_ceiling") is not None:
            bits.append(f"It can express a working plan for "
                        f"<strong>{100 * s['solve_ceiling']:.1f}%</strong> of the "
                        f"{esc(s.get('n'))} exam puzzles (the solve-rate "
                        f"ceiling)")
        if bits:
            out.append('<p class="verdict">' + ". ".join(bits) + ".</p>")
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
                f"never recover <em>while the planner may only use "
                f"sub-goals</em>: closing it needs a richer vocabulary or "
                f"ordinary moves &mdash; which is what the hybrid search later "
                f"did (see the <a href='#story'>Story tab</a>).</p>")
        cats = s.get("categories") or {}
        if cats:
            out.append(table(["category", "instances"],
                             [row([td_txt(f"<code>{esc(k)}</code>"), td(v, "d")])
                              for k, v in sorted(cats.items())],
                             note="Outcome categories &mdash; defined once in "
                                  "the note at the top of this tab."))
        svg = svg_gap_hist(s.get("gap_best_hist"),
                           f"{cfg} {voc}: distribution of best-plan gap to d*")
        if svg:
            out.append('<figure class="fig">' + svg +
                       "<figcaption>Per-instance gap between the best plan the "
                       "language can express and the exact optimum, from "
                       "<code>summary.gap_best_hist</code> of "
                       + src(disp(f)) + ". "
                       + (f"{esc(s.get('n_best_below_dstar'))} puzzle(s) land "
                          "at or below zero: a lucky robot position made the "
                          "real move count cheaper than the plan-step estimate "
                          "— worth an eyeball before it is quoted."
                          if s.get("n_best_below_dstar") else
                          "All bars sit at or above zero: the language matches "
                          "but never beats the known optimum here.")
                       + "</figcaption></figure>")
        caps = d.get("caps") or {}
        meta = [("instances", src(disp(d.get("instances", "?")))),
                ("search caps",
                 f'<span title="{esc(json.dumps(caps))}">'
                 f"{esc(len(caps))} cap setting(s) (hover)</span>"),
                ("workers", esc(d.get("workers"))),
                ("cluster job", esc(d.get("slurm_job_id"))),
                ("wall seconds", esc(d.get("wall_seconds"))),
                ("date", esc(d.get("date"))),
                ("puzzles that hit a cap", esc(s.get("capped"))),
                ("best plans proven to be the language optimum",
                 esc(s.get("best_bounded")))]
        meta = [kv for kv in meta if kv[1] not in (None, "None", "", None)]
        out.append('<details><summary>run provenance (for auditors)</summary>'
                   + table(["field", "value"],
                           [row([td_txt(k), td_txt(v)]) for k, v in meta],
                           note="Provenance block of " + src(disp(f)) + ".")
                   + "</details>")
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
           '<p class="banner">This tab is the raw results ledger, kept '
           'for auditors. Every headline number here appears, explained '
           'in plain language, in the <a href="#story">Story</a> and '
           '<a href="#variants">Variants</a> tabs.</p>',
           "<p>Gate text and estimated node-hours are quoted from PROBLEM.md "
           "&sect;8; status chips, sources and job ids come from "
           + src(disp(RESULTS / "status.json"))
           + ". Result tables are generic: any JSON under "
           + src(disp(RESULTS))
           + " whose payload carries <code>systems</code> + <code>protocol</code> "
             "is rendered as bench rows in the milestone whose directory it "
             "sits in, and any <code>summary.json</code> is rendered as a "
             "key/value table. In plain words: new result files show up on "
             "this page automatically.</p>",
           '<p class="note"><strong>Terms used in every table:</strong> '
           'mean regret = extra moves over the proven optimum. '
           'mean expansions = search effort per puzzle. '
           'Full definitions: the <a href="#glossary">Glossary tab</a>. '
           '<details class="inl"><summary>how these tables are read</summary>'
           '<span class="note">Each result file is read generically: '
           '<code>systems[&hellip;].aggregate</code> for the numbers, '
           '<code>protocol</code> for the exam line under each file, and the '
           '<code>spr</code> block (when present) for setup, config and '
           'certification provenance.</span></details></p>']
    for k in MILESTONE_KEYS:
        m = milestone(k)
        cost, gate = GATES[k]
        stt = m.get("status", "unknown")
        out.append(f'<h3 id="ms-{k.lower()}">{esc(k)} &mdash; '
                   f'{esc(m.get("title", ""))} '
                   f'<span title="{esc(stt)}">{chip(stt, chip_word(stt))}</span></h3>')
        out.append(f'<p class="gate"><strong>Gate, quoted from the brief</strong> '
                   f'(PROBLEM.md &sect;8, est. {cost.replace("nh", "node-hours")}): &ldquo;{gate}&rdquo;'
                   + (f'<br><span class="note">In plain terms: '
                      f'{PLAIN_GATE[k]}</span>' if k in PLAIN_GATE else "")
                   + '</p>')
        if m.get("note") or m.get("plain_note"):
            out.append(f'<p>{note_html(m)}</p>')
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
                    "Result files under " + src(disp(RESULTS / g))
                    + ". Columns follow the same rules everywhere &mdash; see "
                      "the definitions note at the top of this tab."))
            for p, d in summ.get(g, []):
                rendered = True
                out.append(kv_table(p, d))
        if not rendered:
            HOMES = {"M3": ('the <a href="#res-loops">Milestone results tab</a> '
                            '(files under <code>results/selfplay/</code>)'),
                     "M4": ('the <a href="#res-loops">Milestone results tab</a> '
                            '(files under <code>results/selfplay/</code>)'),
                     "M5": ('the <a href="#res-mix">Milestone results tab</a> and '
                            'the <a href="#variants">Variants lab tab</a> (files '
                            'under <code>results/selfplay/</code> and '
                            '<code>results/variants/</code>)'),
                     "M6": ('the M6 design log (<code>M6_DESIGN.md</code>) while '
                            'its first runs are in flight')}
            if k == "M0":
                out.append('<p class="note">Detailed parity tables for this '
                           'milestone live in the <a href="#m0">M0 arena '
                           "parity</a> tab.</p>")
            elif k in HOMES:
                out.append(f'<p class="note">This milestone&rsquo;s results '
                           f'live in {HOMES[k]}.</p>')
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
            bits.append("<details><summary>"
                        f"{len(other)} other JSON file(s) this page does not "
                        "render (for auditors)</summary>"
                        + ", ".join(src(disp(p)) for p in other[:30])
                        + "</details>")
        if errs:
            bits.append("unreadable files: "
                        + ", ".join(f"{src(disp(p))} ({esc(e)})"
                                    for p, e in errs[:10]))
        out.append('<p class="note">' + "; ".join(bits) + ".</p>")
    out.append("</section>")
    return "\n".join(out)


# ------------------------------------------------------- milestone results ---
# Everything below reads results/m1, results/m2, results/transfer,
# results/fwd_*, results/selfplay/<loop>_iter<k>/ generically; a missing file
# is a "pending" cell, a directory that does not exist yet is a pending box.

import re as _re

RESULT_AGG_KEYS = ("n", "solved", "solve_rate", "mean_moves", "mean_regret",
                   "pct_optimal", "mean_expansions", "mean_seconds")


def agg_of(path):
    """(payload, aggregate-of-first-backward-system-with-rows) or (None, {})."""
    d = load(path)
    if not ok(d):
        return d, {}
    _, s = backward_system(d)
    if s is None:                       # forward payloads: first system with rows
        for _, sys_ in (d.get("systems") or {}).items():
            if isinstance(sys_, dict) and sys_.get("rows"):
                s = sys_
                break
    return d, ((s or {}).get("aggregate") or {})


def frontier_agg(d, a) -> bool:
    return bool(a) and (a.get("d_star_placeholder") or is_frontier(d, a))


def agg_cells(d, a, moves_spec=".2f"):
    """solved/n · rate, moves, regret, %opt, exp, s/inst — regret/opt dashed on frontier."""
    if not a:
        return ['<td class="pend" colspan="6">pending</td>']
    front = frontier_agg(d, a)
    n, solved = a.get("n"), a.get("solved")
    sr = (f"{solved}/{n} &middot; {100 * a['solve_rate']:.1f}%"
          if None not in (n, solved) and a.get("solve_rate") is not None else None)
    return [td_txt(sr) if sr else '<td class="pend">pending</td>',
            td(a.get("mean_moves"), moves_spec, cls="hlnum"),
            DASH if front else td(a.get("mean_regret"), ".2f"),
            DASH if front else td_pct100(a.get("pct_optimal")),
            td(a.get("mean_expansions"), ".1f"),
            td(a.get("mean_seconds"), ".2f")]


AGG_HEADERS = ["solved (count &middot; rate)", "moves (that run's solves)",
               "extra moves vs optimum", "% optimal",
               "search effort (expansions)", "seconds/puzzle"]
AGG_NOTE = ("<em>solved / rate</em> = <code>aggregate.solved</code>/<code>n</code> "
            "and <code>solve_rate</code>; <em>mean moves</em> = "
            "<code>aggregate.mean_moves</code> (over solved rows; = "
            "<code>mean_realized_strict</code> for backward systems); <em>mean "
            "regret</em> / <em>% optimal</em> = <code>mean_regret</code> / "
            "<code>pct_optimal</code> over solved rows with a d* (dashed on "
            "frontier sets, where <code>d_star_placeholder</code> is set); "
            "<em>mean exp</em> / <em>s/inst</em> = <code>mean_expansions</code> / "
            "<code>mean_seconds</code> over all rows — the same fields "
            "<code>spr.arena.summarize()</code> reports.")


def paired_cells(pr):
    """[solves a-only/b-only + McNemar p, both-solved moves, wins/losses + sign p]."""
    if not pr:
        return ['<td class="pend" colspan="3">pending</td>']
    def pf(x):
        return "&mdash;" if x is None else (f"{x:.2g}" if x < 0.01 else f"{x:.3f}")
    ao, bo = pr.get("a_only"), pr.get("b_only")
    mp = pr.get("mcnemar_p")
    solves = (f"+{ao}/&minus;{bo}" if None not in (ao, bo) else "&mdash;")
    if mp is not None:
        solves += (f' &middot; <span title="exact McNemar test, p={pf(mp)}">'
                   f"{fluke(mp)}</span>")
    ma, mb = pr.get("mean_moves_a_both"), pr.get("mean_moves_b_both")
    moves = (f"{ma:.2f} vs {mb:.2f} on {pr.get('both_solved')}"
             if None not in (ma, mb) else "&mdash;")
    sp_ = pr.get("sign_p_moves")
    wins = f"{pr.get('moves_wins_a')}/{pr.get('moves_wins_b')}"
    if sp_ is not None:
        wins += (f' &middot; <span title="exact sign test, p={pf(sp_)}">'
                 f"{fluke(sp_)}</span>")
    return [td_txt(solves), td_txt(moves), td_txt(wins)]


PAIRED_HEADERS = ["solves: only-A / only-B (fluke chance)",
                  "moves on puzzles both solved (A vs B)",
                  "move wins A/B (fluke chance)"]
PAIRED_NOTE = ("Paired columns come from the <code>paired</code> block "
               "(<code>spr.gate.paired</code>): <code>a_only</code>/"
               "<code>b_only</code> = instances solved by only one side, "
               "<code>mcnemar_p</code> = two-sided exact McNemar on those "
               "discordant counts; <code>mean_moves_a_both</code>/"
               "<code>mean_moves_b_both</code> over <code>both_solved</code>; "
               "<code>moves_wins_a</code>/<code>moves_wins_b</code> = both-solved "
               "instances where A needs strictly fewer / more realized moves, "
               "<code>sign_p_moves</code> = exact sign test on them.")


def _pf(x):
    return "&mdash;" if x is None else (f"{x:.2g}" if x < 0.01 else f"{x:.3f}")


# ---- M1 -------------------------------------------------------------------

M1_PAIRS = [("mixed_value_warm_s21", "mixed (g16r4+g24r4 corpora), value warm from labeler"),
            ("mixed_value_cold_s21", "mixed, value cold (collapse control)"),
            ("g16_value_warm_s21", "g16r4-only corpus (zero-shot at g24r4)"),
            ("g24_value_warm_s21", "g24r4-only corpus (zero-shot at g16r4)")]
M1_EXAMS = ["g16r4", "g24r4"]


def m1_pairs():
    """Pair tags found on disk (the registry list first, then anything else)."""
    d = RESULTS / "m1"
    tags = [t for t, _ in M1_PAIRS]
    if d.is_dir():
        for f in sorted(d.glob("*_g*r*.json")):
            if f.name.endswith(".gate.json"):
                continue
            m = _re.match(r"^(.*)_(g\d+r\d+)\.json$", f.name)
            if m and m.group(1) not in tags:
                tags.append(m.group(1))
    return tags


def sub_m1() -> str:
    out = ['<h3 id="res-m1">M1 &mdash; size-free rebuild vs the per-size supervised pairs</h3>',
           "<p class='note'>Every pair is benched by the standard harness on "
           "both pinned exams, with every solution replay-checked. The gate "
           "compares it with the recorded per-size supervised pair on the same "
           "exam. PASS means solve rate within 3.5 points and optimality "
           "within 6.6 points. <details><summary>Exact command and gate file"
           "</summary><code>spr.bench --search arena_astar --prefix-check"
           "</code>, 1200 expansions, k=5; gate = <code>spr.gate m1</code> "
           "(PROBLEM.md &sect;8).</details></p>"]
    labels = dict(M1_PAIRS)
    rows_ = []
    any_ = False
    for tag in m1_pairs():
        for cfg in M1_EXAMS:
            f = RESULTS / "m1" / f"{tag}_{cfg}.json"
            g = RESULTS / "m1" / f"{tag}_{cfg}.gate.json"
            d, a = agg_of(f)
            gd = load(g)
            gd = gd if ok(gd) else {}
            if d is None and not gd:
                rows_.append(row([td_txt(f"<strong>{esc(tag)}</strong><br><span class='note'>{esc(labels.get(tag, ''))}</span>"),
                                  td_txt(esc(cfg)),
                                  '<td class="pend" colspan="12">pending</td>',
                                  td_txt(src(disp(f)))]))
                continue
            any_ = True
            ref = gd.get("ref") or {}
            refcells = ([td_txt(f"{ref.get('solved')}/{ref.get('n')}"),
                         td(ref.get("mean_moves"), ".2f"),
                         td(ref.get("mean_regret"), ".2f"),
                         td_pct100(ref.get("pct_optimal"))]
                        if ref else ['<td class="pend" colspan="4">no gate file</td>'])
            verdict = ('<td class="pend">pending</td>' if not gd else
                       td_txt(chip("pass" if gd.get("pass") else "fail",
                                   "PASS" if gd.get("pass") else "FAIL")
                              + f"<br><span class='note'>&Delta;solve {gd.get('delta_solve_pts', 0):+.1f} pts, "
                                f"&Delta;opt {gd.get('delta_opt_pts', 0):+.1f} pts</span>"))
            rows_.append(row([
                td_txt(f"<strong>{esc(tag)}</strong><br><span class='note'>{esc(labels.get(tag, ''))}</span>"),
                td_txt(esc(cfg)),
                *agg_cells(d, a),
                *refcells,
                verdict,
                *paired_cells(gd.get("paired")),
                td_txt(src(disp(f)) + (f"<br>{src(disp(g))}" if gd else "")),
            ], "hl" if tag == "mixed_value_warm_s21" else ""))
    out.append(table(["pair (nets)", "exam", *AGG_HEADERS,
                      "ref solved", "ref moves", "ref regret", "ref % opt",
                      "M1 gate", *PAIRED_HEADERS, "source"], rows_,
                     note="Same column rules as defined once at the top of this "
                          "tab. The reference columns quote the recorded "
                          "per-size supervised pair the gate compares against. "
                          "A = the size-free pair, B = the supervised reference.",
                     cls="wide"))
    if not any_:
        out.append(pend_note(f"nothing under {disp(RESULTS / 'm1')}"))
    return "\n".join(out)


# ---- M2 -------------------------------------------------------------------

M2_VARIANTS = [("greedy", "always take the top candidate (greedy)"),
               ("astar_child", "A*, scoring by child estimate (the benchmark's own search)"),
               ("astar_parent", "A*, scoring by parent estimate, first solution"),
               ("astar_parent_any", "A*, parent estimate, keep going until a solution replays"),
               ("astar_best", "A*, parent estimate, best answer within the budget"),
               ("mcts_min", "tree search, best answer within the budget (min backup)"),
               ("mcts_mean", "tree search, best answer within the budget (mean backup)")]


def sub_m2() -> str:
    out = ['<h3 id="res-m2">M2 &mdash; search beats greed at inference</h3>',
           "<p class='note'>Same nets, same exam, same 1200-expansion / k=5 "
           "budget, replay-certified; each variant's <code>.vs_greedy.json</code> "
           "is the paired test against the greedy row (gate: strictly better mean "
           "moves at equal-or-better solve rate, PROBLEM.md &sect;8). Files are "
           "grouped by <code>&lt;family&gt;_&lt;nets&gt;_&lt;exam&gt;_&lt;variant&gt;.json</code>.</p>"]
    d2 = RESULTS / "m2"
    groups = {}
    if d2.is_dir():
        for f in sorted(d2.glob("*.json")):
            if ".vs_" in f.name:
                continue
            m = _re.match(r"^(persize|sizefree)_(.+)_(g\d+r\d+)_(.+)\.json$", f.name)
            if not m:
                continue
            fam, nets, cfg, var = m.groups()
            groups.setdefault((fam, nets, cfg), {})[var] = f
    if not groups:
        out.append(pend_note(f"nothing under {disp(d2)}"))
        return "\n".join(out)
    labels = dict(M2_VARIANTS)
    order = [v for v, _ in M2_VARIANTS]
    rows_ = []
    for (fam, nets, cfg), vars_ in sorted(groups.items()):
        rows_.append(row([f'<td class="grp" colspan="12"><strong>{esc(fam)}</strong> '
                          f'family &middot; nets <code>{esc(nets)}</code> &middot; '
                          f'exam <strong>{esc(cfg)}</strong></td>']))
        for var in sorted(vars_, key=lambda v: (order.index(v) if v in order else 99, v)):
            f = vars_[var]
            d, a = agg_of(f)
            g = f.with_name(f.stem + ".vs_greedy.json")
            pr = load(g) if g.is_file() else None
            pr = pr if ok(pr) else None
            hl = "hl" if var.startswith("mcts") else ""
            flags = " ".join((d or {}).get("spr", {}).get("flags") or []) if ok(d) else ""
            rows_.append(row([
                td_txt(f"<code>{esc(var)}</code><br><span class='note'>{esc(labels.get(var, ''))}"
                       + (f"<br><code>{esc(flags)}</code>" if flags else "") + "</span>"),
                *agg_cells(d, a),
                *(paired_cells(pr) if var != "greedy" else
                  ['<td class="dash" colspan="3">&mdash; (the B side)</td>']),
                td_txt(src(disp(f)) + (f"<br>{src(disp(g))}" if pr else "")),
            ], hl))
    out.append(table(["search variant", *AGG_HEADERS, *PAIRED_HEADERS, "source"],
                     rows_, note="Same column rules as defined once at the top of this tab. "
                     + " A = the variant, B = greedy on the same exam.",
                     cls="wide"))
    return "\n".join(out)


# ---- transfer / headroom ---------------------------------------------------

TRANSFER_SETS = [(RESULTS / "transfer", "M1 mixed pair (zero-shot)"),
                 (RESULTS / "selfplay" / "g24r4_iter1" / "transfer",
                  "base loop, iteration-1 nets")]
# recorded per-size supervised rows per (cfg, set) — file selection only
TRANSFER_REF = {("g24r4", "frontier"): "scaling/results/g24r4/comparison_ungraded_nntwin.json",
                ("g32r4", "graded"): "scaling/results/g32r4/comparison.json",
                ("g32r4", "frontier"): "scaling/results/g32r4/comparison_ungraded.json",
                ("g24r8", "graded"): "scaling/results/g24r8/comparison.json",
                ("g24r8", "frontier"): "scaling/results/g24r8/comparison_ungraded.json",
                ("g24r4", "graded"): "scaling/results/g24r4/comparison.json"}


def ceiling_solve(cfg, set_, vocab="base"):
    """(solved, n, capped, file) of the matching ceiling probe, or None."""
    name = (f"{cfg}_{vocab}.json" if set_ == "graded" and vocab == "base"
            else f"{cfg}_{set_}_{vocab}.json")
    f = RESULTS / "ceiling" / name
    if not f.is_file() and set_ == "graded":
        f = RESULTS / "ceiling" / f"{cfg}_graded_{vocab}.json"
    d = load(f) if f.is_file() else None
    if not ok(d) or not isinstance(d.get("summary"), dict):
        return None
    s = d["summary"]
    return {"solved": s.get("n_realizable"), "n": s.get("n"),
            "capped": s.get("capped") or 0, "file": f,
            "inconclusive": (s.get("categories") or {}).get("INCONCLUSIVE", 0),
            "gap_best": s.get("mean_gap_best"), "pct_best": s.get("pct_best_optimal")}


def sub_transfer() -> str:
    out = ['<h3 id="res-transfer">Transfer / headroom &mdash; zero-shot at 32&times;32, 8 robots and on the frontier sets</h3>',
           "<p class='note'>The size-free pair, trained on 16&times;16 + "
           "24&times;24 with 4 robots only, benched under the arena protocol on "
           "exams it never saw; the recorded per-size supervised row for the "
           "same exam and the base-language solve ceiling (<code>spr.ceiling</code>) "
           "bracket it. Frontier sets have no d*: solve rate and mean moves only.</p>"]
    found = {}
    for d_, label in TRANSFER_SETS:
        if not d_.is_dir():
            continue
        for f in sorted(d_.glob("*.json")):
            m = _re.match(r"^(g\d+r\d+)_(graded|frontier)_(astar|mcts)\.json$", f.name)
            if not m:
                continue
            cfg, set_, search = m.groups()
            found.setdefault((cfg, set_), []).append((label, search, f))
    if not found:
        out.append(pend_note("nothing under results/transfer/ or "
                             "results/selfplay/*/transfer/ yet"))
        return "\n".join(out)
    rows_ = []
    for (cfg, set_) in sorted(found):
        rows_.append(row([f'<td class="grp" colspan="9"><strong>{esc(cfg)}</strong> '
                          f'&middot; {esc(set_)} set</td>']))
        ref = TRANSFER_REF.get((cfg, set_))
        if ref:
            rd = load_sv(ref)
            _, rs = backward_system(rd)
            ra = (rs or {}).get("aggregate") or {}
            rows_.append(row([td_txt("recorded per-size supervised pair"),
                              td_txt("arena A*"),
                              *agg_cells(rd, ra),
                              td_txt(src(ref))], "ctl"))
        for label, search, f in found[(cfg, set_)]:
            d, a = agg_of(f)
            rows_.append(row([td_txt(esc(label)),
                              td_txt("arena A*" if search == "astar" else "MCTS"),
                              *agg_cells(d, a),
                              td_txt(src(disp(f)))],
                             "hl" if label.startswith("M1") else ""))
        c = ceiling_solve(cfg, set_)
        if c:
            note = ""
            if c["capped"]:
                note = (f" <span class='note'>(probe capped on {c['capped']} "
                        f"instances, {c['inconclusive']} inconclusive: lower bound)</span>")
            rows_.append(row([td_txt("<strong>base-language solve ceiling</strong>" + note),
                              td_txt("exhaustive, no net"),
                              td_txt(f"{c['solved']}/{c['n']} &middot; "
                                     f"{100 * c['solved'] / c['n']:.1f}%"
                                     if c["n"] else "&mdash;"),
                              DASH,
                              td(c["gap_best"], ".2f") if c["gap_best"] is not None else DASH,
                              td_pct100(c["pct_best"]) if c["pct_best"] is not None else DASH,
                              DASH, DASH,
                              td_txt(src(disp(c["file"])))], "ceil"))
    out.append(table(["nets", "search", *AGG_HEADERS, "source"], rows_,
                     note="Same column rules as defined once at the top of this tab."
                          " Ceiling row: <code>summary.n_realizable</code>/"
                          "<code>n</code> of the matching <code>results/ceiling/</code> "
                          "file; its regret column is <code>mean_gap_best</code> "
                          "(the best expressible plan's gap to d*), its % optimal "
                          "<code>pct_best_optimal</code>.",
                     cls="wide"))
    return "\n".join(out)


# ---- M5: the mixed-size B2 curriculum --------------------------------------

MIX_CFGS = ["g24r4", "g32r4", "g24r8"]
MIX_CFG_HUMAN = {"g24r4": "24&times;24, 4 robots &mdash; the loop's home size",
                 "g32r4": "32&times;32, 4 robots",
                 "g24r8": "24&times;24, 8 robots &mdash; hard mode"}
# Iteration-0 rows = the nets the curriculum was seeded from (the g24r4 B2
# loop's iteration-4 pair), benched before any mixed-size self-play. Only
# g24r4 has that pair's own benches; at g32r4/g24r8 the newest transfer
# benches on disk are the iteration-3 nets', so those stand in and are
# labelled as a different (earlier) net pair. (label, graded, frontier)
MIX_SEED = {
    "g24r4": ("iteration-4 nets of the g24r4 B2 loop &mdash; the seed of this "
              "curriculum, on its own exam",
              RESULTS / "selfplay" / "g24r4_b2_iter4" / "bench_g24r4_astar.json",
              RESULTS / "selfplay" / "g24r4_b2_iter4" / "bench_g24r4_frontier_astar.json"),
    "g32r4": ("<strong>iteration-3</strong> nets, zero-shot &mdash; no iteration-4 "
              "transfer bench exists at this size",
              RESULTS / "selfplay" / "g24r4_b2_iter3" / "transfer" / "b2it3_g32r4_bench_solved_astar.json",
              RESULTS / "selfplay" / "g24r4_b2_iter3" / "transfer" / "b2it3_g32r4_bench_unsolved_astar.json"),
    "g24r8": ("<strong>iteration-3</strong> nets, zero-shot &mdash; no iteration-4 "
              "transfer bench exists at this size",
              RESULTS / "selfplay" / "g24r4_b2_iter3" / "transfer" / "b2it3_g24r8_bench_solved_astar.json",
              None),
}
MIX_HEADERS = ["iteration", "instances", "certified", "records",
               "graded A* solved / rate", "moves", "regret", "% optimal",
               "exp", "s/inst",
               "frontier A* solved / rate", "moves", "exp",
               "paired vs previous mixed iteration", "sources"]


def mix_dirs():
    """[(k, dir)] for results/selfplay/mix_b2mix_iter<k>/ (the M5 curriculum)."""
    root = RESULTS / "selfplay"
    out = []
    if not root.is_dir():
        return out
    for p in sorted(root.iterdir()):
        m = _re.match(r"^mix_b2mix_iter(\d+)$", p.name)
        if p.is_dir() and m:
            out.append((int(m.group(1)), p))
    out.sort()
    return out


def front_cells(d, a):
    """solved/n · rate, mean moves, mean expansions — a frontier set has no d*."""
    if not a:
        return ['<td class="pend" colspan="3">pending</td>']
    n, solved = a.get("n"), a.get("solved")
    sr = (f"{solved}/{n} &middot; {100 * a['solve_rate']:.1f}%"
          if None not in (n, solved) and a.get("solve_rate") is not None else None)
    return [td_txt(sr) if sr else '<td class="pend">pending</td>',
            td(a.get("mean_moves"), ".2f", cls="hlnum"),
            td(a.get("mean_expansions"), ".1f")]


def gate_txt(pr, label):
    """One paired-gate line (spr.gate compare payload), or None."""
    if not ok(pr):
        return None
    return (f"{esc(label)}: solves +{pr.get('a_only')}/&minus;{pr.get('b_only')} "
            f'&middot; <span title="exact McNemar test, p={_pf(pr.get("mcnemar_p"))}">'
            f"{fluke(pr.get('mcnemar_p'))}</span>; moves "
            f"{pr.get('moves_wins_a')}/{pr.get('moves_wins_b')} "
            f'&middot; <span title="exact sign test, p={_pf(pr.get("sign_p_moves"))}">'
            f"{fluke(pr.get('sign_p_moves'))}</span> on "
            f"{pr.get('both_solved')} both-solved")


def nets_txt(p):
    """{'policy': ..., 'value': ...} of a nets.txt, or {}."""
    p = Path(p)
    if not p.is_file():
        STATS["missing"].append(disp(p))
        return {}
    try:
        txt = p.read_text()
    except Exception as e:                      # noqa: BLE001
        STATS["error"].append(f"{disp(p)}: {e}")
        return {}
    STATS["read"].append(disp(p))
    out = {}
    for line in txt.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def short_ckpt(p, keep=3) -> str:
    parts = Path(str(p)).parts
    return "/".join(parts[-keep:]) if len(parts) > keep else str(p)


def sub_mix() -> str:
    out = ['<h3 id="res-mix">M5 &mdash; the mixed-size B2 curriculum '
           '(24&times;24 + 32&times;32 + 24&times;24&times;8 robots)</h3>',
           "<p class='note'>One size-free pair, trained every iteration on the "
           "union of three per-config replay buffers and benched on all three "
           "pinned exams (<code>jobs/selfplay_mix_iter.slurm</code>, DESIGN.md "
           "&sect;4c: generation per config from board ids 9000, "
           "<code>--batch-ref-n 24</code>, arena A* under the B2 convention, "
           "gate paired against the previous mixed iteration). The curriculum "
           "is seeded from the g24r4 B2 loop's iteration-4 pair, so its "
           "iteration-0 row is that pair's own bench &mdash; the question the "
           "table answers is whether mixing sizes buys anything the 24&times;24 "
           "loop did not already have (FINDINGS &sect;16b: the 24&times;24-only "
           "loop pays a small regret price at 32&times;32 and 8 robots). "
           "MCTS rows are not run in-loop (frontier MCTS costs 1&ndash;2 h per "
           "exam); A* is the in-loop instrument here.</p>"]
    dirs = mix_dirs()
    if not dirs:
        out.append(pend_note("no results/selfplay/mix_b2mix_iter&lt;k&gt;/ "
                             "directory yet &mdash; the reference rows below "
                             "are the bar the curriculum has to clear"))
    for cfg in MIX_CFGS:
        out.append(f"<h4>{esc(cfg)} &mdash; {MIX_CFG_HUMAN[cfg]}</h4>")
        rows_ = []
        # the frozen per-size supervised B2 opponent at this size
        grel = f"scaling/results/{cfg}/comparison_b2.json"
        frel = f"scaling/results/{cfg}/comparison_ungraded_b2.json"
        gd = load_sv(grel)
        _, gs = backward_system(gd)
        fd = load_sv(frel)
        _, fs = backward_system(fd)
        rows_.append(row([
            td_txt("recorded per-size supervised B2 pair"
                   "<br><span class='note'>the frozen opponent (PROBLEM.md "
                   "&sect;7), same exams, same budget</span>"),
            DASH, DASH, DASH,
            *agg_cells(gd, (gs or {}).get("aggregate") or {}),
            *front_cells(fd, (fs or {}).get("aggregate") or {}),
            DASH,
            td_txt(src(disp(SV / grel)) + "<br>" + src(disp(SV / frel))),
        ], "ctl"))
        # iteration 0 = the seed nets
        lab, gp, fp = MIX_SEED.get(cfg, (None, None, None))
        if lab:
            gdd, gaa = agg_of(gp) if gp else (None, {})
            fdd, faa = agg_of(fp) if fp else (None, {})
            srcs = [src(disp(p)) for p in (gp, fp) if p and Path(p).is_file()]
            rows_.append(row([
                td_txt(f"<strong>0</strong> <span class='note'>({lab})</span>"),
                DASH, DASH, DASH,
                *agg_cells(gdd, gaa),
                *(front_cells(fdd, faa) if fp else [DASH, DASH, DASH]),
                DASH,
                td_txt(" ".join(srcs) if srcs else "&mdash;"),
            ], "ctl"))
        ks = [k for k, _ in dirs]
        for k, d_ in dirs:
            man = load(d_ / f"generation_{cfg}.manifest.json")
            man = man if ok(man) else {}
            gpath = d_ / f"bench_{cfg}_astar.json"
            fpath = d_ / f"bench_{cfg}_frontier_astar.json"
            gdd, gaa = agg_of(gpath)
            fdd, faa = agg_of(fpath)
            inst, sol = man.get("instances"), man.get("solved")
            cert = (f"{sol} ({100 * sol / inst:.1f}%)"
                    if None not in (inst, sol) and inst else None)
            bits = []
            if man.get("board_ids"):
                bits.append(f"boards {esc(man['board_ids'])}")
            s_ = man.get("search") or {}
            if s_:
                bits.append(f"MCTS {s_.get('expansions')} exp / stop "
                            f"{s_.get('stop_after')}, vocab {s_.get('vocab')}")
            if man.get("mean_expansions") is not None:
                bits.append(f"{man['mean_expansions']:.1f} exp / "
                            f"{man.get('mean_seconds', 0):.1f} s per instance")
            if man.get("slurm_job_id"):
                bits.append(f"job {esc(man['slurm_job_id'])}")
            nets = nets_txt(d_ / "nets.txt")
            if nets.get("policy"):
                bits.append("nets " + esc(short_ckpt(nets["policy"], 1)) + " / "
                            + esc(short_ckpt(nets.get("value", ""), 1)))
            if k > 1:
                lines = [gate_txt(load(d_ / f"gate_{cfg}_astar_vs_prev.json"), "graded A*"),
                         gate_txt(load(d_ / f"gate_{cfg}_frontier_astar_vs_prev.json"),
                                  "frontier A*")]
                lines = [x for x in lines if x]
                pv = "<br>".join(lines) if lines else None
            else:
                pv = ("&mdash; <span class='note'>(first mixed iteration; the "
                      "reference rows above are its baseline)</span>")
            srcs = [src(disp(p)) for p in
                    (d_ / f"generation_{cfg}.manifest.json", gpath, fpath)
                    if p.is_file()]
            rows_.append(row([
                td_txt(f"<strong>{k}</strong>"
                       + (f"<br><span class='note'>{'; '.join(bits)}</span>"
                          if bits else "")),
                td(inst, "d") if inst is not None else '<td class="pend">pending</td>',
                td_txt(cert) if cert else '<td class="pend">pending</td>',
                td(man.get("records"), "d") if man.get("records") is not None
                else '<td class="pend">pending</td>',
                *agg_cells(gdd, gaa),
                *front_cells(fdd, faa),
                td_txt(pv) if pv else '<td class="pend">pending</td>',
                td_txt(" ".join(srcs) if srcs else "&mdash;"),
            ], "hl" if k == max(ks) else ""))
        out.append(table(MIX_HEADERS, rows_,
                         note="Same column rules as defined once at the top of this tab."
                              " The frontier columns come from "
                              "<code>bench_&lt;cfg&gt;_frontier_astar.json</code> "
                              "(<code>bench.unsolved.jsonl</code>: no d* exists "
                              "there, so no regret column); generation columns "
                              "from <code>generation_&lt;cfg&gt;.manifest.json</code>; "
                              "the paired column from "
                              "<code>gate_&lt;cfg&gt;_{astar,frontier_astar}_vs_prev.json</code> "
                              "(A = this iteration, B = the previous mixed "
                              "iteration), which the job writes from iteration 2 "
                              "on. Iteration-0 and supervised rows are read from "
                              "the files named in the last column.",
                         cls="wide"))
    return "\n".join(out)


# ---- M5: far-size zero-shot exact audits -----------------------------------

AUDIT_DIR = RESULTS / "audit"
AUDIT_CFGS = ["g32r4", "g40r4", "g48r4", "g56r4", "g64r4"]
AUDIT_PAIR_LABEL = {
    "b2it0_m1mixed": "M1 mixed size-free pair &mdash; loop iteration 0 "
                     "(supervised corpora only, never self-played)",
    "b2it4": "B2 loop iteration-4 nets &mdash; four self-play iterations at "
             "24&times;24 in the B2 vocabulary",
}
# The labeler's own recorded numbers on the same corpora. Two instruments:
# "value" = nn_labeler.audit of the production value net (directly comparable
# to this audit's value columns); "descent" = the descent-audit's argmin
# agreement (a different instrument — see the note under the table).
LABELER_REF = {
    "g32r4": ("nn_labeler/results/audit_prod_v1_s11_full.json", "value", "g32r4"),
    "g40r4": ("nn_labeler/results/audit_coarse_g40r4.json", "value", "g40r4"),
    "g48r4": ("nn_labeler/results/audit_coarse_g48r4.json", "value", "g48r4"),
    # 56/64: the same-instrument audit ran later and lives in this project's
    # results tree (res: prefix); the coarsegate descent numbers stay below as
    # an explicitly-separated different-method row.
    "g56r4": ("res:audit/labeler_prod_v1_s11_g56g64.json", "value", "g56r4"),
    "g64r4": ("res:audit/labeler_prod_v1_s11_g56g64.json", "value", "g64r4"),
}
LABELER_REF_OTHER = {
    "g56r4": ("nn_labeler/results/coarsegate_g56r4.json", "descent", None),
    "g64r4": ("nn_labeler/results/coarsegate_g64r4.json", "descent", None),
}
AUDIT_METRICS = [
    ("value", "top1_optimal", "value top1-optimal", "pct"),
    ("value", "regret", "value regret (moves)", "num"),
    ("policy", "top1_optimal", "policy top1-optimal", "pct"),
    ("policy", "regret@1", "policy regret@1 (moves)", "num"),
    ("policy", "recall@5", "policy recall@5", "pct"),
    ("pair", "top1_optimal", "pair top1-optimal", "pct"),
    ("pair", "regret", "pair regret (moves)", "num"),
]
def _h(words, tech):
    return f'<span title="{tech}">{words}</span>'


AUDIT_HEADERS = [
    "exam size", _h("decisions scored", "n_groups (n_policy_groups in brackets)"),
    _h("best pick is optimal &mdash; value net", "value.top1_optimal"),
    _h("extra moves, value's pick", "value.regret"),
    _h("best pick is optimal &mdash; policy net", "policy.top1_optimal"),
    _h("extra moves, policy's top choice", "policy.regret@1"),
    _h("extra moves, best of policy's top 5", "policy.regret@5"),
    _h("an optimal candidate is in its top 5", "policy.recall@5"),
    _h("best pick is optimal &mdash; the planner's actual decision", "pair.top1_optimal"),
    _h("extra moves, planner's decision", "pair.regret"),
    _h("planner beats value alone (wins/losses)", "paired.pair_vs_value"),
    _h("planner beats policy alone (wins/losses)", "paired.pair_vs_policy")]


def labeler_ref(cfg, table=None):
    """The labeler's recorded row for one config, or None."""
    spec = (table if table is not None else LABELER_REF).get(cfg)
    if not spec:
        return None
    rel, kind, key = spec
    res_local = rel.startswith("res:")
    d = load(RESULTS / rel[4:]) if res_local else load_sv(rel)
    if not ok(d):
        return None
    if kind == "value":
        c = ((d.get("configs") or {}).get(key)) or {}
        if not c:
            return None
        return {"kind": "value", "top1": c.get("top1_optimal"),
                "regret": c.get("regret"), "n": c.get("n_groups"),
                "file": disp(RESULTS / rel[4:]) if res_local else disp(SV / rel),
                "what": "labeler value net <code>prod_v1_s11</code> on the same "
                        "decision corpus, argmin vs the exact optimum "
                        "(<code>nn_labeler.audit</code>) &mdash; directly "
                        "comparable to the value columns"}
    dec = d.get("decisions") or {}
    if dec.get("argmin_agreement") is None:
        return None
    return {"kind": "descent", "top1": dec.get("argmin_agreement"),
            "regret": None, "n": dec.get("n_groups_scored"),
            "jaccard": dec.get("optimal_set_jaccard"), "file": disp(SV / rel),
            "what": "<strong>a different instrument</strong>: the descent audit's "
                    "<code>decisions.argmin_agreement</code> &mdash; how often the "
                    "labeler's greedy descent picks the exact engine's argmin over "
                    "the <em>matched</em> candidates of its own corpus, not a "
                    "value-net top-1 over this corpus' groups"}


def _audit_key(tag):
    m = _re.match(r"^b2it(\d+)", tag)
    if m:
        return (0, int(m.group(1)), tag)
    m = _re.match(r"^mix_?it(\d+)", tag)
    if m:
        return (1, int(m.group(1)), tag)
    return (2, 0, tag)


def audit_reports():
    """[(tag, report, path)] of results/audit/*.json, in iteration order."""
    out = []
    if not AUDIT_DIR.is_dir():
        return out
    for f in sorted(AUDIT_DIR.glob("*.json")):
        d = load(f)
        if not ok(d) or not isinstance(d.get("configs"), dict):
            continue
        out.append((f.stem, d, f))
    out.sort(key=lambda t: _audit_key(t[0]))
    return out


def _audit_metric(cfg_block, block, key):
    b = (cfg_block or {}).get(block) or {}
    return b.get(key)


def audit_rows(rep):
    """One row per config of one audit report, each followed by the labeler row."""
    rows_ = []
    cfgs = [c for c in AUDIT_CFGS if c in (rep.get("configs") or {})]
    for c in (rep.get("configs") or {}):
        if c not in cfgs:
            cfgs.append(c)
    for cfg in (cfgs or AUDIT_CFGS):
        c = (rep.get("configs") or {}).get(cfg)
        if not c:
            rows_.append(row([td_txt(f"<strong>{esc(cfg)}</strong>"),
                              f'<td class="pend" colspan="11">pending &mdash; '
                              f'not in this report</td>']))
        elif not c.get("n_groups"):
            rows_.append(row([td_txt(f"<strong>{esc(cfg)}</strong>"),
                              td(c.get("n_groups"), "d"),
                              '<td class="pend" colspan="10">no decision groups '
                              'in this split of the corpus</td>']))
        else:
            v, p, pr = c.get("value") or {}, c.get("policy") or {}, c.get("pair") or {}
            pd_ = c.get("paired") or {}
            npol = c.get("n_policy_groups")
            grp = f"{c.get('n_groups')}"
            if npol is not None and npol != c.get("n_groups"):
                grp += f" <span class='note'>({npol} scored for policy)</span>"
            pv = pd_.get("pair_vs_value") or []
            pp = pd_.get("pair_vs_policy") or []
            rows_.append(row([
                td_txt(f"<strong>{esc(cfg)}</strong>"),
                td_txt(grp),
                td_pct(v.get("top1_optimal")), td(v.get("regret"), ".3f"),
                td_pct(p.get("top1_optimal")), td(p.get("regret@1"), ".3f"),
                td(p.get("regret@5"), ".3f"), td_pct(p.get("recall@5")),
                td_pct(pr.get("top1_optimal")),
                td(pr.get("regret"), ".3f", cls="hlnum"),
                td_txt(f"{pv[0]}/{pv[1]}" if len(pv) == 2 else "&mdash;"),
                td_txt(f"{pp[0]}/{pp[1]}" if len(pp) == 2 else "&mdash;"),
            ]))
        lr = labeler_ref(cfg)
        if lr:
            rows_.append(row([
                td_txt("&#8627; the labeler (teacher), <strong>same measuring "
                       "method</strong> &mdash; the comparison row"
                       "<br><span class='note'>" + src(lr["file"]) + "</span>"),
                td_txt(esc(lr["n"]) if lr["n"] is not None else "&mdash;"),
                td_pct(lr["top1"]),
                td(lr["regret"], ".3f") if lr["regret"] is not None else DASH,
                DASH, DASH, DASH, DASH, DASH, DASH, DASH, DASH,
            ], "ctl"))
        lo = labeler_ref(cfg, LABELER_REF_OTHER)
        if lo:
            rows_.append(row([
                td_txt("&#8627; <span title=\"descent-gate argmin agreement over "
                       "matched candidates of its own generated corpus\">labeler, "
                       "<strong>different measuring method &mdash; not "
                       "comparable</strong> &Dagger;</span>"
                       "<br><span class='note'>" + src(lo["file"]) + "</span>"),
                td_txt(esc(lo["n"]) if lo["n"] is not None else "&mdash;"),
                td_pct(lo["top1"]), DASH,
                DASH, DASH, DASH, DASH, DASH, DASH, DASH, DASH,
            ], "dim"))
    return rows_


def audit_depth_table(rep) -> str:
    rows_ = []
    for cfg, c in (rep.get("configs") or {}).items():
        for d_, b in sorted((c.get("by_depth") or {}).items(),
                            key=lambda kv: int(kv[0])):
            rows_.append(row([td_txt(esc(cfg)), td_txt(esc(d_)),
                              td(b.get("n"), "d"),
                              td_pct(b.get("value_top1")),
                              td_pct(b.get("pair_top1"))]))
    if not rows_:
        return ""
    return ("<details><summary>per-decision-depth breakdown "
            "(<code>by_depth</code>)</summary>"
            + table(["config", "depth", "decisions scored", "value top1", "pair top1"],
                    rows_,
                    note="<code>by_depth[d]</code> of the same report: "
                         "<code>n</code> groups at that decision depth, "
                         "<code>value_top1</code> over all of them, "
                         "<code>pair_top1</code> over the ones the policy could "
                         "score. Depth 0 is the first decision from the goal — "
                         "the hardest one and the one the labeler's fidelity "
                         "curve also reports.")
            + "</details>")


def sub_audit() -> str:
    out = ['<h3 id="res-audit">M5 &mdash; far-size zero-shot exact audits '
           '(32 &rarr; 64)</h3>',
           "<p class='note'>PROBLEM.md &sect;8/M5 asks for &ldquo;net trained "
           "via self-play at &le;32 benched zero-shot at 40&ndash;64 against "
           "exact ground truth&rdquo;. No pinned exam and no d* bench exists "
           "above 32&times;32, so the instrument is decision-level "
           "(<code>spr.audit</code>, DESIGN.md &sect;4c): every net pair is "
           "asked to rank the exact-labeled candidate sets of the labeler's own "
           "fidelity corpora &mdash; the g32r4 <em>test</em> split of "
           "<code>scaling/data/g32r4/backward.jsonl</code> and "
           "<code>scaling/data/g&lt;n&gt;r4/backward_audit.rust.jsonl</code> at "
           "40/48/56/64. <em>value</em> = argmin of the value estimate; "
           "<em>policy</em> = its top-k; <em>pair</em> = the planner's actual "
           "greedy decision (value argmin over the policy's top-5 — one arena "
           "expansion). Regret is in moves above the group's exact optimum, so "
           "lower is better and 0.000 means the decision was optimal.</p>"]
    reps = audit_reports()
    # plain-language conclusion, computed from the files themselves
    it0 = next((rep for tag, rep, _ in reps if tag.startswith("b2it0")), None)
    if it0:
        wins, sizes = [], []
        for cfg_, c_ in (it0.get("configs") or {}).items():
            lr_ = labeler_ref(cfg_)
            v_ = (c_.get("value") or {}).get("top1_optimal")
            if lr_ and lr_.get("top1") is not None and v_ is not None:
                sizes.append(cfg_)
                wins.append(v_ >= lr_["top1"])
        if sizes and all(wins):
            out.append("<p><strong>The one-line conclusion: with the same "
                       "measuring method, the loop&rsquo;s starting network "
                       "(iteration 0) scores higher than the labeler that "
                       "taught it at every size tested.</strong> Rows marked "
                       "&Dagger; use a different measuring method and must not "
                       "be compared with the rest of their table.</p>")
        out.append("<p class='note'>Two claims live in this table and must not "
                   "be conflated. The Story&rsquo;s &ldquo;beats its own "
                   "teacher&rdquo; compares the <em>iteration-0</em> network "
                   "with the labeler, same method, and holds at every size. "
                   "The <em>iteration-4</em> rows sit below iteration 0 at "
                   "every size because four self-play rounds on 24&times;24 "
                   "boards only specialize the networks toward that setting "
                   "&mdash; a documented cost (FINDINGS &sect;17c), not a "
                   "contradiction.</p>")
    if not reps:
        out.append(pend_note("nothing under results/audit/ yet &mdash; "
                             "<code>jobs/audit_far.slurm</code> writes "
                             "<code>results/audit/&lt;TAG&gt;.json</code>"))
        return "\n".join(out)
    for tag, rep, f in reps:
        meta = []
        if rep.get("split"):
            meta.append(f"split <code>{esc(rep['split'])}</code>")
        meta.append("by-reference helper matching "
                    + ("on" if rep.get("byref") else "off"))
        for k in ("policy", "value"):
            if rep.get(k):
                meta.append(f"{k} <code>{esc(short_ckpt(rep[k]))}</code>")
        out.append(f'<h4>{esc(tag)} &mdash; '
                   f'{AUDIT_PAIR_LABEL.get(tag, "net pair " + esc(tag))}</h4>')
        out.append(f"<p class='note'>{' &middot; '.join(meta)} &middot; "
                   f"{src(disp(f))}</p>")
        out.append(table(AUDIT_HEADERS, audit_rows(rep),
                         note="Hover any column header for the exact field "
                              "name. Lower &ldquo;extra moves&rdquo; is better; "
                              "0 means the choice was optimal. "
                              "<details><summary>Field-by-field definitions "
                              "(for auditors)</summary>"
                              "<em>groups</em> = <code>n_groups</code> exact "
                              "decision groups (&ge;2 candidates) in the split, "
                              "<code>n_policy_groups</code> in brackets when the "
                              "policy could not be scored on all of them (a "
                              "candidate's helper slot has no match in the "
                              "policy's encoding); <em>value</em> = "
                              "<code>value.top1_optimal</code> / "
                              "<code>value.regret</code>; <em>policy</em> = "
                              "<code>policy.top1_optimal</code>, "
                              "<code>regret@1</code>, <code>regret@5</code>, "
                              "<code>recall@5</code> (an optimal candidate is "
                              "inside the top-5 &mdash; the arena's k=5 filter "
                              "never loses the optimum); <em>pair</em> = "
                              "<code>pair.top1_optimal</code> / "
                              "<code>pair.regret</code>; the last two columns are "
                              "<code>paired.pair_vs_value</code> / "
                              "<code>paired.pair_vs_policy</code> (groups where "
                              "the pair's regret is strictly lower / strictly "
                              "higher than that side alone). Labeler rows: "
                              "recorded numbers from "
                              "<code>nn_labeler/results/</code> &mdash; at 32/40/48 "
                              "the labeler's own value-net audit (same "
                              "instrument as the value columns); at 56/64 no "
                              "such audit exists, so the row shows the "
                              "descent-audit's <code>argmin_agreement</code>, "
                              "<strong>a different instrument</strong> (greedy "
                              "descent vs the exact engine over matched "
                              "candidates) &mdash; read it as a scale marker, not "
                              "as a like-for-like row.</details>",
                         cls="wide"))
        out.append(audit_depth_table(rep))
    # cross-pair comparison
    cfgs = []
    for _, rep, _ in reps:
        for c in (rep.get("configs") or {}):
            if c not in cfgs:
                cfgs.append(c)
    cfgs = [c for c in AUDIT_CFGS if c in cfgs] + [c for c in cfgs if c not in AUDIT_CFGS]
    rows_ = []
    for cfg in cfgs:
        lr = labeler_ref(cfg)
        ns = " &middot; ".join(
            f"{tag}: {((rep.get('configs') or {}).get(cfg) or {}).get('n_groups', '&mdash;')} groups"
            for tag, rep, _ in reps)
        rows_.append(row([f'<td class="grp" colspan="{2 + len(reps) + 1}">'
                          f'<strong>{esc(cfg)}</strong> &middot; {ns}</td>']))
        for block, key, label, kind in AUDIT_METRICS:
            cells = [td_txt(label)]
            vals = []
            for tag, rep, _ in reps:
                v = _audit_metric((rep.get("configs") or {}).get(cfg), block, key)
                vals.append(v)
                cells.append(td_pct(v) if kind == "pct" else td(v, ".3f"))
            if len(vals) >= 2 and None not in (vals[0], vals[-1]):
                dv = vals[-1] - vals[0]
                better = (dv > 0) if kind == "pct" else (dv < 0)
                arrow = "&uarr;" if dv > 0 else ("&darr;" if dv < 0 else "&rarr;")
                txt = (f"{100 * dv:+.1f} pts" if kind == "pct" else f"{dv:+.3f}")
                cells.append(td_txt(f"<span class='tag {'front' if not better and dv else ''}'>"
                                    f"{arrow} {txt}</span>"))
            else:
                cells.append('<td class="pend">pending</td>')
            if lr and block == "value" and (
                    (lr["kind"] == "value" and key in ("top1_optimal", "regret"))
                    or (lr["kind"] == "descent" and key == "top1_optimal")):
                v = lr["top1"] if key == "top1_optimal" else lr["regret"]
                cell = td_pct(v) if key == "top1_optimal" else td(v, ".3f")
                if lr["kind"] == "descent" and v is not None:
                    cell = td_txt(f"{100 * v:.1f}% &Dagger;")
                cells.append(cell)
            else:
                cells.append(DASH)
            rows_.append(row(cells))
    out.append("<h4>The same numbers, one metric per row</h4>")
    out.append(table(["metric", *[esc(t) for t, _, _ in reps],
                      "last &minus; first", "labeler reference"], rows_,
                     note="The comparison view of the tables above: every metric "
                          "of every net pair on the same corpus, so the "
                          "curriculum's effect at a size it never trained on is "
                          "one row. <em>last &minus; first</em> is the newest "
                          "pair minus the oldest (percentage points for the "
                          "rates, moves for the regrets; an arrow is direction "
                          "only &mdash; for regret rows down is better). "
                          "&Dagger; = the descent-audit argmin agreement, a "
                          "different instrument (see the note above).",
                     cls="wide"))
    return "\n".join(out)


# ---- forward arm -----------------------------------------------------------

def sub_forward() -> str:
    out = ['<h3 id="res-fwd">Forward (primitive-move) arm &mdash; <code>spr/fwd</code></h3>',
           "<p class='note'>The comparison arm in the natural AlphaZero action "
           "space: AlphaZero-style tree search over single robot moves, guided "
           "by the move network. F-M0 checks the harness reproduces the "
           "recorded forward result. F-M2 tries the search variants on the "
           "450-puzzle exam. F-g24 repeats that at 24&times;24.</p>"]
    dirs = [("fwd_m0", "F-M0 &mdash; parity"), ("fwd_m2", "F-M2 &mdash; search variants at g16r4"),
            ("fwd_g24", "F-g24 &mdash; 24&times;24")]
    seen = {n for n, _ in dirs}
    if RESULTS.is_dir():
        for p in sorted(RESULTS.iterdir()):
            if p.is_dir() and p.name.startswith("fwd_") and p.name not in seen:
                dirs.append((p.name, p.name))
    any_ = False
    for name, title in dirs:
        d_ = RESULTS / name
        out.append(f"<h4>{title} <span class='note'>{src(disp(d_))}</span></h4>")
        if not d_.is_dir():
            out.append(pend_note(f"{disp(d_)} not on disk"))
            continue
        files = [f for f in sorted(d_.glob("*.json")) if ".vs_" not in f.name]
        rows_ = []
        for f in files:
            d, a = agg_of(f)
            if not a:
                continue
            any_ = True
            sysname = ""
            for n_, k_, a_, nr in systems_of(d):
                if a_ is a:
                    sysname = n_
            vs = sorted(d_.glob(f.stem + ".vs_*.json"))
            vs_bits = []
            for g in vs:
                pr = load(g)
                if not ok(pr):
                    continue
                other = g.name[len(f.stem) + 4:-5]
                vs_bits.append(f"vs <code>{esc(other)}</code>: solves "
                               f"+{pr.get('a_only')}/&minus;{pr.get('b_only')} "
                               f'&middot; <span title="exact McNemar test, '
                               f'p={_pf(pr.get("mcnemar_p"))}">'
                               f"{fluke(pr.get('mcnemar_p'))}</span>, moves "
                               f"{pr.get('moves_wins_a')}/{pr.get('moves_wins_b')} "
                               f'&middot; <span title="exact sign test, '
                               f'p={_pf(pr.get("sign_p_moves"))}">'
                               f"{fluke(pr.get('sign_p_moves'))}</span>")
            rows_.append(row([td_txt(f"<code>{esc(f.stem)}</code><br><span class='note'>{esc(sysname[:90])}</span>"),
                              *agg_cells(d, a, ".3f"),
                              td_txt("<br>".join(vs_bits) if vs_bits else "&mdash;"),
                              td_txt(src(disp(f)))]))
        if rows_:
            out.append(table(["setup", *AGG_HEADERS, "paired (.vs_*.json)", "source"], rows_,
                             note="Same column rules as defined once at the top of this tab."
                                  " Paired column: every "
                                  "<code>&lt;arm&gt;.vs_&lt;other&gt;.json</code> next to the "
                                  "arm file, A = the arm, B = the other.",
                             cls="wide"))
        else:
            out.append(pend_note(f"no comparison payload under {disp(d_)}"))
    return "\n".join(out)


# ---- loop iterations -------------------------------------------------------

def loop_dirs():
    """{loop_key: [(k, dir)]} for results/selfplay/<cfg>[_b2]_iter<k>/."""
    root = RESULTS / "selfplay"
    loops = {}
    if not root.is_dir():
        return loops
    for p in sorted(root.iterdir()):
        m = _re.match(r"^(g\d+r\d+)(_b2)?_iter(\d+)$", p.name)
        if p.is_dir() and m:
            key = m.group(1) + (m.group(2) or "")
            loops.setdefault(key, []).append((int(m.group(3)), p))
    for k in loops:
        loops[k].sort()
    return loops


def _pick(dir_, patterns):
    """First existing file among glob patterns (checked in dir_ and dir_/transfer)."""
    for pat in patterns:
        for base in (dir_, dir_ / "transfer"):
            hits = sorted(base.glob(pat)) if base.is_dir() else []
            if hits:
                return hits[0]
    return None


def loop_iter_files(cfg, dir_):
    return {
        "manifest": dir_ / "generation.manifest.json",
        "gauge": dir_ / "gauge.json",
        "astar": _pick(dir_, [f"bench_{cfg}_astar.json", f"*{cfg}_bench_solved_astar.json",
                              f"*{cfg}_graded_astar.json"]),
        "mcts": _pick(dir_, [f"bench_{cfg}_mcts.json", f"*{cfg}_bench_solved_mcts.json",
                             f"*{cfg}_graded_mcts.json"]),
        "frontier": _pick(dir_, [f"bench_{cfg}_frontier_astar.json",
                                 f"*{cfg}_bench_unsolved_astar.json",
                                 f"{cfg}_frontier_astar.json"]),
        "frontier_mcts": _pick(dir_, [f"bench_{cfg}_frontier_mcts.json",
                                      f"*{cfg}_bench_unsolved_mcts.json",
                                      f"{cfg}_frontier_mcts.json"]),
        "vs_prev": dir_ / "gate_vs_prev.json",
        "vs_sup": dir_ / "gate_vs_supervised.json",
        "nets": dir_ / "nets.txt",
    }


# iteration-0 reference rows (the M1 nets before any self-play data)
LOOP_ITER0 = {
    "g24r4": {"astar": RESULTS / "m1" / "mixed_value_warm_s21_g24r4.json",
              "mcts": RESULTS / "m2" / "sizefree_mixed_warm_g24r4_mcts_min.json",
              "frontier": RESULTS / "transfer" / "g24r4_frontier_astar.json"},
}


def sub_loops() -> str:
    out = ['<h3 id="res-loops">M3/M4 &mdash; the self-play loop, iteration by iteration</h3>',
           "<p class='note'>One row per <code>results/selfplay/&lt;cfg&gt;[_b2]_iter&lt;k&gt;/</code>: "
           "generation manifest (fresh instances searched, certified, records "
           "written), fidelity gauge (on a sample of decisions: does the loop's "
           "own label pick the same best candidate as the exact solver?), the "
           "arena benches of the retrained pair on the graded exam (A* and MCTS) "
           "and on the frontier set, and the paired gate against the previous "
           "iteration's nets. Iteration 0 = the M1 nets before any self-play "
           "data.</p>"]
    loops = loop_dirs()
    if not loops:
        out.append(pend_note("nothing under results/selfplay/ yet"))
        return "\n".join(out)
    hdr = ["iteration", "instances", "certified", "records", "gauge (argmin)",
           "A* solved", "A* moves", "A* regret", "A* exp",
           "MCTS solved", "MCTS moves", "MCTS regret", "MCTS exp",
           "frontier A* solved", "frontier A* moves", "frontier MCTS solved",
           "frontier MCTS moves", "paired vs previous iteration", "vs supervised",
           "sources"]
    for key, iters in sorted(loops.items()):
        cfg = key.split("_")[0]
        b2 = key.endswith("_b2")
        out.append(f'<h4>loop <code>{esc(key)}</code> &mdash; {esc(cfg)}, '
                   f'{"B2 (extended) vocabulary" if b2 else "base vocabulary"}</h4>')
        rows_ = []
        ks = [k for k, _ in iters]
        if 0 not in ks and not b2 and cfg in LOOP_ITER0:
            f0 = LOOP_ITER0[cfg]
            _, aa = agg_of(f0["astar"]) if f0["astar"].is_file() else (None, {})
            _, ma = agg_of(f0["mcts"]) if f0["mcts"].is_file() else (None, {})
            _, fa = agg_of(f0["frontier"]) if f0["frontier"].is_file() else (None, {})
            rows_.append(row([
                td_txt("0 <span class='note'>(M1 nets)</span>"),
                DASH, DASH, DASH, DASH,
                td(aa.get("solved"), "d"), td(aa.get("mean_moves"), ".2f"),
                td(aa.get("mean_regret"), ".2f"), td(aa.get("mean_expansions"), ".1f"),
                td(ma.get("solved"), "d"), td(ma.get("mean_moves"), ".2f"),
                td(ma.get("mean_regret"), ".2f"), td(ma.get("mean_expansions"), ".1f"),
                td(fa.get("solved"), "d"), td(fa.get("mean_moves"), ".2f"),
                DASH, DASH,
                DASH, DASH,
                td_txt(" ".join(src(disp(p)) for p in f0.values() if p.is_file())),
            ], "ctl"))
        for k, dir_ in iters:
            F = loop_iter_files(cfg, dir_)
            man = load(F["manifest"]) if F["manifest"].is_file() else None
            man = man if ok(man) else {}
            gg = load(F["gauge"]) if F["gauge"].is_file() else None
            gg = gg if ok(gg) else {}
            au = gg.get("audit_summary") or {}
            _, aa = agg_of(F["astar"]) if F["astar"] else (None, {})
            _, ma = agg_of(F["mcts"]) if F["mcts"] else (None, {})
            _, fa = agg_of(F["frontier"]) if F["frontier"] else (None, {})
            _, fm = agg_of(F["frontier_mcts"]) if F["frontier_mcts"] else (None, {})
            vp = load(F["vs_prev"]) if F["vs_prev"].is_file() else None
            vp = vp if ok(vp) else None
            vs = load(F["vs_sup"]) if F["vs_sup"].is_file() else None
            vs = vs if ok(vs) else None
            inst = man.get("instances")
            solved = man.get("solved")
            cert = (f"{solved} ({100 * solved / inst:.1f}%)"
                    if None not in (inst, solved) and inst else None)
            gauge = None
            if au.get("argmin_agreement") is not None:
                hov = (f"argmin agreement {au['argmin_agreement']:.3f}"
                       + (f", n={gg.get('sample')}, Jaccard "
                          f"{au.get('optimal_set_jaccard', 0):.2f}"
                          if gg.get("sample") else ""))
                gauge = (f'<span title="{hov}">label check: '
                         f"{100 * au['argmin_agreement']:.1f}% match the exact "
                         f"solver's best choice</span>")
            elif k == 0:
                gauge = "&mdash;"
            if vp:
                pv = (f"solves +{vp.get('a_only')}/&minus;{vp.get('b_only')} "
                      f'&middot; <span title="exact McNemar test, '
                      f'p={_pf(vp.get("mcnemar_p"))}">{fluke(vp.get("mcnemar_p"))}'
                      f"</span><br>moves "
                      f"{vp.get('moves_wins_a')}/{vp.get('moves_wins_b')} "
                      f'&middot; <span title="exact sign test, '
                      f'p={_pf(vp.get("sign_p_moves"))}">{fluke(vp.get("sign_p_moves"))}'
                      f"</span> on {vp.get('both_solved')} both-solved")
            elif k == 0:
                pv = "&mdash; (reference iteration)"
            else:
                pv = None
            if vs:
                sv = (chip("pass" if vs.get("pass") else "fail",
                           "PASS" if vs.get("pass") else "FAIL")
                      + f" &Delta;solve {vs.get('delta_solve_pts', 0):+.1f}, "
                        f"&Delta;opt {vs.get('delta_opt_pts', 0):+.1f} pts")
            else:
                sv = None
            man_bits = []
            if man.get("board_ids"):
                man_bits.append(f"boards {esc(man['board_ids'])}")
            s_ = man.get("search") or {}
            if s_:
                man_bits.append(f"MCTS {s_.get('expansions')} exp / stop {s_.get('stop_after')}"
                                + (f", min-exp {s_.get('min_expansions')}" if s_.get("min_expansions") else "")
                                + (f", vocab {s_.get('vocab')}" if s_.get("vocab") else ""))
            if man.get("mean_expansions") is not None:
                man_bits.append(f"{man['mean_expansions']:.1f} exp / {man.get('mean_seconds', 0):.1f} s per instance")
            if man.get("slurm_job_id"):
                man_bits.append(f"job {esc(man['slurm_job_id'])}")
            srcs = [src(disp(p)) for p in (F["manifest"], F["gauge"], F["astar"], F["mcts"],
                                             F["frontier"], F["frontier_mcts"],
                                             F["vs_prev"], F["vs_sup"])
                    if p and Path(p).is_file()]
            rows_.append(row([
                td_txt(f"<strong>{k}</strong>" + (" <span class='note'>(M1 nets, zero-shot in B2)</span>" if k == 0 and b2 else "")
                       + (f"<br><details><summary class='note'>run details</summary>"
                          f"<span class='note'>{'; '.join(man_bits)}</span></details>" if man_bits else "")),
                td(inst, "d") if inst is not None else (DASH if k == 0 else '<td class="pend">pending</td>'),
                td_txt(cert) if cert else (DASH if k == 0 else '<td class="pend">pending</td>'),
                td(man.get("records"), "d") if man.get("records") is not None else (DASH if k == 0 else '<td class="pend">pending</td>'),
                td_txt(gauge) if gauge else '<td class="pend">pending</td>',
                td(aa.get("solved"), "d"), td(aa.get("mean_moves"), ".2f"),
                td(aa.get("mean_regret"), ".2f"), td(aa.get("mean_expansions"), ".1f"),
                td(ma.get("solved"), "d"), td(ma.get("mean_moves"), ".2f"),
                td(ma.get("mean_regret"), ".2f"), td(ma.get("mean_expansions"), ".1f"),
                td(fa.get("solved"), "d"), td(fa.get("mean_moves"), ".2f"),
                td(fm.get("solved"), "d"), td(fm.get("mean_moves"), ".2f"),
                td_txt(pv) if pv else '<td class="pend">pending</td>',
                td_txt(sv) if sv else (DASH if k == 0 else '<td class="pend">pending</td>'),
                td_txt(" ".join(srcs) if srcs else "&mdash;"),
            ], "hl" if k == max(ks) else ""))
        out.append(table(hdr, rows_,
                         note="<em>instances</em> / <em>certified</em> / <em>records</em> = "
                              "<code>generation.manifest.json</code> <code>instances</code>, "
                              "<code>solved</code>, <code>records</code> (plus its "
                              "<code>search</code> block and per-instance means in the first "
                              "column); <em>gauge</em> = <code>gauge.json</code> "
                              "<code>audit_summary.argmin_agreement</code> over "
                              "<code>sample</code> exact-labeled depth-0 decisions "
                              "(<code>optimal_set_jaccard</code> in brackets); A* / MCTS "
                              "columns = <code>aggregate</code> of the graded-exam bench "
                              "payloads (<code>bench_&lt;cfg&gt;_astar.json</code> / "
                              "<code>_mcts.json</code>, or the iteration-0 file names); "
                              "<em>frontier</em> columns = the frontier bench payloads "
                              "(<code>bench_&lt;cfg&gt;_frontier_{astar,mcts}.json</code>, "
                              "<code>transfer/&lt;cfg&gt;_frontier_&lt;search&gt;.json</code> or "
                              "<code>*_bench_unsolved_&lt;search&gt;.json</code>) on "
                              "<code>bench.unsolved.jsonl</code>, which has no d* — solve "
                              "count and mean realized moves only, and the two searches "
                              "solve different instance subsets, so their moves are not "
                              "comparable to each other; paired columns = "
                              "<code>gate_vs_prev.json</code> (A = this iteration, B = the "
                              "previous nets) and <code>gate_vs_supervised.json</code> "
                              "(<code>spr.gate m1</code> against the per-size supervised pair).",
                         cls="wide"))
    return "\n".join(out)


# ---- chart: regret vs solve rate on the g24r4 graded exam -------------------

CHART_POINTS = [
    # (series, label, file)
    ("supervised per-size", "supervised A* (exact pair)", RESULTS / "m0" / "g24r4_exact_prefix.json"),
    ("size-free, base vocab", "size-free A*", RESULTS / "m1" / "mixed_value_warm_s21_g24r4.json"),
    ("size-free, base vocab", "size-free MCTS", RESULTS / "m2" / "sizefree_mixed_warm_g24r4_mcts_min.json"),
    ("B2 vocab (loop)", "B2 A* it0", RESULTS / "selfplay" / "g24r4_b2_iter0" / "m1mixed_b2_g24r4_bench_solved_astar.json"),
    ("B2 vocab (loop)", "B2 MCTS it0", RESULTS / "selfplay" / "g24r4_b2_iter0" / "m1mixed_b2_g24r4_bench_solved_mcts.json"),
]
CHART_SERIES = ["supervised per-size", "size-free, base vocab", "B2 vocab (loop)",
                "B2 mixed-size curriculum"]
CHART_CLASS = {"supervised per-size": "c1", "size-free, base vocab": "c2",
               "B2 vocab (loop)": "c3", "B2 mixed-size curriculum": "c4"}


def chart_points():
    pts = list(CHART_POINTS)
    for k, d_ in loop_dirs().get("g24r4_b2", []):
        if k == 0:
            continue
        for search, lab in (("astar", "A*"), ("mcts", "MCTS")):
            f = d_ / f"bench_g24r4_{search}.json"
            pts.append(("B2 vocab (loop)", f"B2 {lab} it{k}", f))
    for k, d_ in mix_dirs():
        pts.append(("B2 mixed-size curriculum", f"mix A* it{k}",
                    d_ / "bench_g24r4_astar.json"))
    out = []
    for series, label, f in pts:
        d, a = agg_of(f) if Path(f).is_file() else (None, {})
        if a and a.get("solve_rate") is not None and a.get("mean_regret") is not None:
            out.append((series, label, 100 * a["solve_rate"], a["mean_regret"], disp(f)))
        else:
            out.append((series, label, None, None, disp(f)))
    return out


def svg_regret_vs_solve() -> str:
    pts = chart_points()
    have = [p for p in pts if p[2] is not None]
    refs = []
    for vocab, lab in (("base", "base-language ceiling"), ("b2", "B2-language ceiling")):
        f = RESULTS / "ceiling" / f"g24r4_{vocab}.json"
        d = load(f) if f.is_file() else None
        s = (d or {}).get("summary") if ok(d) else None
        if s and s.get("solve_ceiling") is not None and s.get("mean_gap_best") is not None:
            refs.append((lab, 100 * s["solve_ceiling"], s["mean_gap_best"], disp(f), vocab))
    if not have:
        return ""
    W, H, L, R, T, B = 640, 360, 56, 20, 22, 46
    xs = [p[2] for p in have] + [r[1] for r in refs]
    ys = [p[3] for p in have] + [r[2] for r in refs]
    x0, x1 = min(xs) - 3, min(100.0, max(xs) + 2)
    y0, y1 = 0.0, max(ys) + 0.5
    def X(v): return L + (v - x0) / (x1 - x0) * (W - L - R)
    def Y(v): return (H - B) - (v - y0) / (y1 - y0) * (H - B - T)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Mean regret against solve rate on the g24r4 graded exam, one point per planner, ceilings as dashed reference lines">']
    # grid
    step_y = 1.0 if y1 <= 6 else 2.0
    v = 0.0
    while v <= y1:
        s.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W - R}" y2="{Y(v):.1f}" class="grid"/>')
        s.append(f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" class="axl" text-anchor="end">{v:.0f}</text>')
        v += step_y
    step_x = 2 if (x1 - x0) <= 16 else 5
    v = int(x0) // step_x * step_x + step_x
    while v <= x1:
        s.append(f'<line x1="{X(v):.1f}" y1="{T}" x2="{X(v):.1f}" y2="{H - B}" class="grid"/>')
        s.append(f'<text x="{X(v):.1f}" y="{H - B + 14}" class="axl" text-anchor="middle">{v}%</text>')
        v += step_x
    s.append(f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" class="axis"/>')
    s.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H - B}" class="axis"/>')
    s.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 6}" class="axl" text-anchor="middle">solve rate on bench.solved (232 graded instances) &rarr; better</text>')
    s.append(f'<text x="{L}" y="13" class="axl">extra moves vs perfect (lower is better)</text>')
    # reference lines (short staggered labels; full detail in the tooltips)
    for ri, (lab, xr, yr, f, vocab) in enumerate(refs):
        cls = "ref1" if vocab == "base" else "ref2"
        short = "base ceiling" if vocab == "base" else "B2 ceiling"
        s.append(f'<line x1="{X(xr):.1f}" y1="{T}" x2="{X(xr):.1f}" y2="{H - B}" class="refl {cls}"><title>{esc(lab)}: solve ceiling {xr:.1f}% ({esc(f)})</title></line>')
        s.append(f'<line x1="{L}" y1="{Y(yr):.1f}" x2="{W - R}" y2="{Y(yr):.1f}" class="refl {cls}"><title>{esc(lab)}: best-plan regret {yr:.2f} ({esc(f)})</title></line>')
        s.append(f'<text x="{X(xr) - 4:.1f}" y="{T + 13 + 16 * ri}" class="refl-t {cls}" text-anchor="end">{esc(short)} {xr:.1f}%</text>')
        s.append(f'<text x="{W - R - 4}" y="{Y(yr) - 5:.1f}" class="refl-t {cls}" text-anchor="end">{esc(short)} regret {yr:.2f}</text>')
    # points: dots + tooltips only — per-point names live in the legend and in
    # the data table right under the figure (labels overlapped when 20+ points
    # landed in the same corner; owner rejected the overplot)
    for series, label, xv, yv, f in have:
        cls = CHART_CLASS.get(series, "c1")
        s.append(f'<circle cx="{X(xv):.1f}" cy="{Y(yv):.1f}" r="6" class="pt {cls}"><title>{esc(label)}: {xv:.1f}% solved, regret {yv:.2f} ({esc(f)})</title></circle>')
    # legend (only the series that actually have a point on the plot)
    shown = [x for x in CHART_SERIES if any(p[0] == x for p in have)]
    lx, ly = L + 8, H - B - 14 - 16 * len(shown)
    for j, series in enumerate(shown):
        yy = ly + 16 * j
        s.append(f'<circle cx="{lx}" cy="{yy}" r="5" class="pt {CHART_CLASS[series]}"/>')
        s.append(f'<text x="{lx + 10}" y="{yy + 4}" class="axl">{esc(series)}</text>')
    s.append("</svg>")
    return "".join(s)


def sub_chart() -> str:
    pts = chart_points()
    svg = svg_regret_vs_solve()
    out = ['<h3 id="res-chart">Regret vs solve rate on the g24r4 graded exam</h3>']
    if not svg:
        out.append(pend_note("none of the chart's source files is on disk yet"))
    else:
        out.append('<figure class="fig"><div class="chart">' + svg + "</div>"
                   "<figcaption>Each dot is one benchmark run: puzzles solved "
                   "(across) against extra moves used (down). Dashed lines are "
                   "the proven language limits. Down and to the right is "
                   "better. Hover any dot for its name and exact numbers. The "
                   "table below lists every dot with its source file. "
                   "<details><summary>Provenance</summary>x = "
                   "<code>aggregate.solve_rate</code>, y = "
                   "<code>aggregate.mean_regret</code> of one bench payload on "
                   "<code>scaling/data/g24r4/bench.solved.jsonl</code> under the "
                   "1200-expansion, k=5 protocol. Dashed lines: "
                   "<code>summary.solve_ceiling</code> and "
                   "<code>summary.mean_gap_best</code> of "
                   "<code>results/ceiling/g24r4_base.json</code> / "
                   "<code>g24r4_b2.json</code>. A planner cannot sit right of its "
                   "language's vertical line; its regret is bounded below by the "
                   "horizontal one only on the same instance set. Later "
                   "iterations are picked up automatically from "
                   "<code>results/selfplay/&hellip;/bench_g24r4_*.json</code>."
                   "</details></figcaption></figure>")
    rows_ = [row([td_txt(esc(series)), td_txt(esc(label)),
                  td_pct100(xv) if xv is not None else '<td class="pend">pending</td>',
                  td(yv, ".2f"), td_txt(src(f))])
             for series, label, xv, yv, f in pts]
    out.append(table(["series", "planner", "solve rate", "mean regret", "source"], rows_,
                     note="The chart's data, one row per point (the table view of the "
                          "figure). Missing files render as pending and are left off "
                          "the plot."))
    return "\n".join(out)


def sec_results() -> str:
    out = ['<section id="results">', "<h2>Milestone results</h2>",
           '<p class="banner">This tab is the raw results ledger, kept '
           'for auditors. Every headline number here appears, explained '
           'in plain language, in the <a href="#story">Story</a> and '
           '<a href="#variants">Variants</a> tabs.</p>',
           "<p>Every table here is regenerated from result files under "
           + src(disp(RESULTS))
           + " at generation time — nothing is hand-typed. Where a milestone's "
             "files have not landed the cells read <em>pending</em>. Definitions "
             "of the columns are under each table; the reading of the numbers "
             "lives in " + src("self_play_robots/FINDINGS.md") + ".</p>",
           "<p class='note'>Columns are defined once here &mdash; every table "
           "in this tab uses the same rules. <details><summary>Column "
           "definitions, field by field (for auditors)</summary>" + AGG_NOTE
           + " " + PAIRED_NOTE + "</details></p>",
           "<p class='note'>Jump to: <a href='#res-m1'>M1</a> &middot; "
           "<a href='#res-m2'>M2</a> &middot; <a href='#res-transfer'>transfer / headroom</a> "
           "&middot; <a href='#res-fwd'>forward arm</a> &middot; <a href='#res-loops'>loop "
           "iterations</a> &middot; <a href='#res-mix'>M5 mixed-size curriculum</a> "
           "&middot; <a href='#res-audit'>M5 far-size audits</a> &middot; "
           "<a href='#res-chart'>regret vs solve chart</a>.</p>"]
    for fn in (sub_m1, sub_m2, sub_transfer, sub_forward, sub_loops, sub_mix,
               sub_audit, sub_chart):
        try:
            out.append(fn())
        except Exception as e:          # one broken subsection must not kill the tab
            out.append(f'<p class="pendbox bad">{esc(fn.__name__)} failed to render: '
                       f'{esc(type(e).__name__)}: {esc(e)}</p>')
            STATS["error"].append(f"{fn.__name__}: {type(e).__name__}: {e}")
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
         "The fixed search budget every planner gets, so comparisons are "
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
         "defense against the loop grading its own homework (a network cannot "
         "train on a solution it merely imagined). "
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
         "absolute board cells. <em>B1</em> adds park repairs (shove a robot "
         "aside first). <em>B2</em> adds <em>by-reference</em> steps "
         "(&ldquo;park blue where red currently stands&rdquo;) — more "
         "expressive, far more expensive for the exact solver to label. A "
         "network taught from artificially limited B2 data inherited the "
         "limitation (main log, entry 68). Vocabulary separation is absolute: "
         "base and B2 datasets never mix. <strong>The self-play loop runs the "
         "extended (B2) vocabulary</strong>, with owner approval: the base "
         "vocabulary was measured as already saturated (see the "
         "<a href='#ceiling'>Ceiling tab</a>), so B2 self-play data is the "
         "training data on this page."),
        ("seed noise bars",
         "Two runs that differ only in their random start can differ by ~3.4 "
         "solve and ~6.6 optimality points at 24&times;24 (main log, entries "
         "67&ndash;71). The M1 gate uses those limits: <strong>3.5 solve "
         "points / 6.6 optimality points</strong>. Any claimed win must clear "
         "this bar, or compare puzzle-by-puzzle instead (the McNemar test). "
         "The measured spread for this repository's own files is computed "
         "live in the <a href='#baselines'>Baselines</a> tab."),
        ("fidelity gauge",
         "At each decision: does the loop's own label pick the same best "
         "candidate as the exact solver would? Cheap to measure on boards up "
         "to 64&times;64. "
         "Calibrated by the supervised track: ~91% agreement &rarr; planners "
         "match exact-taught ones, ~89% keeps solve rate but loses optimality, "
         "~82% collapses (FINDINGS 74). A slide below ~90% predicts a utility "
         "slide <em>before</em> the bench shows it — the loop's early-warning "
         "instrument. PROBLEM.md &sect;4.1."),
        ("lean boards",
         "Boards generated at any size in seconds by "
         "<code>nn_labeler/leanboard.py</code> (3.8 s at 96&times;96 vs 11.9 h "
         "for the eager path), layout-identical to eager boards "
         "(checked cell for cell against the slow generator), computing "
         "distances only when asked. They make &ldquo;fresh "
         "instances every iteration&rdquo; essentially free — which is where "
         "self-play diversity comes from in a deterministic single-agent game "
         "(&sect;6.4)."),
        ("d*",
         "The exact move-optimal cost of an instance, from the Rust solver, "
         "stored per line in the pinned bench file. Exists only for boards up to "
         "64&times;64 (the solver refuses larger boards), and only on the "
         "graded sets."
         + (" Measured here: " + "; ".join(dstar_bits) + "."
            if dstar_bits else "")),
        ("frontier set",
         "The hard tail: the puzzles nothing had solved when the exam was "
         "fixed, so no optimum is known. Only solve rate and real move counts "
         "mean anything there — this page hides regret and optimality on "
         "frontier rows rather than print a meaningless number."),
        ("graded set",
         "The complement: <code>bench.solved.jsonl</code> (and "
         "<code>eval/data/bench450.jsonl</code> at g16r4), where every instance "
         "carries d*, so % optimal and mean regret are defined."),
        ("supervised",
         "Trained from provided answers. Here: planners taught from the "
         "exact solver&rsquo;s solutions. The opposite of self-play, which "
         "makes its own training data."),
        ("neural network (NN, &ldquo;net&rdquo;)",
         "A trainable function. This project uses two small ones: the policy "
         "network and the value network."),
        ("policy network / value network",
         "The two halves of the planner. The policy network proposes which "
         "sub-goal to try. The value network estimates how many steps a plan "
         "still needs."),
        ("A*",
         "A classic search method: always continue from the partial plan "
         "that looks cheapest overall. The project&rsquo;s fast first-answer "
         "search."),
        ("MCTS",
         "Monte-Carlo Tree Search &mdash; the tree search AlphaZero uses. It "
         "balances trying what looks good against checking what is "
         "unexplored."),
        ("greedy descent",
         "No search at all: at every decision, take the value "
         "network&rsquo;s top pick and never look back."),
        ("zero-shot",
         "Used on a task or board size it never trained on, with no "
         "adjustment."),
        ("epoch",
         "One full pass over the training data."),
        ("arena",
         "This project&rsquo;s benchmarking harness: fixed exams, fixed "
         "search budget, replay-checked scoring."),
        ("checkpoint",
         "A saved copy of a network&rsquo;s weights &mdash; one file that can "
         "be loaded and run."),
        ("warm start",
         "Start training from an existing checkpoint instead of random "
         "weights."),
        ("flagship",
         "The project&rsquo;s best planner: the strict-move-trained networks "
         "plus the hybrid search that may make up to two ordinary moves "
         "before sub-goal planning."),
        ("McNemar test",
         "A standard statistical test for paired yes/no outcomes &mdash; "
         "here: which puzzles each of two planners solved, puzzle by "
         "puzzle."),
        ("regret / extra moves",
         "How many moves a solution uses beyond the proven optimum, "
         "averaged over solved puzzles with a known optimum."),
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


# ------------------------------------------------------------- variants lab --

def _variants_meta():
    """Registry metadata via the variants package (no torch at import time)."""
    import sys as _sys
    sp = str(SPR)
    if sp not in _sys.path:
        _sys.path.insert(0, sp)
    try:
        import variants as V
        return {vid: V.get(vid) for vid in V.ids()}
    except Exception as e:
        STATS["error"].append(f"variants registry: {type(e).__name__}: {e}")
        return {}


def _var_agg(payload):
    for _name, _kind, agg, _n in systems_of(payload):
        return agg
    return {}


def _var_verdict(vid, gates):
    """(chip class, label) from the gate files vs control."""
    if vid == "v00_control":
        return "run", "control"
    if vid == "v08_cold_start":
        # a control arm: the GAP is the measurement, not a defeat
        have = [g for g in gates.values() if ok(g)]
        return ("run", "control (prior-worth)") if have else ("pend", "pending")
    have = [g for g in gates.values() if ok(g)]
    if not have:
        return "pend", "pending"
    win = loss = False
    for g in have:
        sa, sb = g.get("solved_a"), g.get("solved_b")
        mp, sp_ = g.get("mcnemar_p"), g.get("sign_p_moves")
        wa, wb = g.get("moves_wins_a", 0), g.get("moves_wins_b", 0)
        if sa is not None and sb is not None and mp is not None and mp < 0.05:
            win, loss = win or sa > sb, loss or sa < sb
        if sp_ is not None and sp_ < 0.05 and sa is not None and sb is not None and sa >= sb:
            win, loss = win or wa > wb, loss or wa < wb
    if win and not loss:
        rep = RESULTS / "variants" / f"{vid}_s8" / "gate_graded_vs_control.json"
        return ("good", "win") if ok(load(rep)) else ("good", "win (1 seed)")
    if loss and not win:
        return "bad", "loss"
    if win and loss:
        return "run", "mixed"
    return "pend", "flat (n.s.)" if len(have) == 3 else "flat so far"


def sec_variants() -> str:
    meta = _variants_meta()
    vd = RESULTS / "variants"
    out = ['<section id="variants"><h2>Variants lab</h2>']
    out.append(
        '<p>This tab is the experiment lab. Each experiment changes exactly one '
        'thing in the self-play design and is measured against an unchanged '
        'control run. Every experiment starts from the same saved networks, '
        'gets the same practice budget, and takes the same three exams. '
        'Every verdict is a puzzle-by-puzzle comparison against the control '
        'run. Move counts are compared only on puzzles both runs solved, '
        'because averages over different puzzle sets are not comparable. '
        'A minus number in a moves column means the experiment used fewer '
        'moves. Category chips on the cards: targets = what the networks '
        'learn to predict. data = what they practice on. search = how the '
        'planner explores. action-space = which moves it may use. '
        'bootstrap = where training starts from. combo = combined changes. '
        'Full protocol: <code>variants/DESIGN.md</code>. Results log: '
        '<code>variants/FINDINGS.md</code>.</p>')
    out.append(
        '<details open><summary><strong>Plain-English glossary</strong> (terms used '
        'on every card)</summary><ul class="note">'
        '<li><strong>standard (graded) exam</strong> &mdash; 232 puzzles where the '
        'true shortest solution is known, so we can measure wasted moves.</li>'
        '<li><strong>frontier exam</strong> &mdash; 218 harder puzzles no solver has '
        'fully cracked; only solve counts and move counts can be compared.</li>'
        '<li><strong>unseen exam</strong> &mdash; 200 puzzles on 50 boards no network '
        'ever trained on; measures generalization, the project goal.</li>'
        '<li><strong>expansions</strong> &mdash; the unit of search effort; every arm '
        'gets the same budget (1200 per puzzle) so comparisons are fair.</li>'
        '<li><strong>moves</strong> &mdash; actual robot moves in the final, replayed '
        'solution; the headline metric. <strong>regret</strong> = extra moves beyond '
        'the known optimum (standard exam only).</li>'
        '<li><strong>why moves columns say &ldquo;on puzzles both/all solved&rdquo;'
        '</strong> &mdash; a system&rsquo;s own average covers only the puzzles '
        '<em>it</em> solved, so two systems&rsquo; raw averages are not comparable '
        '(solving more hard puzzles makes the average look worse). Every moves '
        'comparison on this page is therefore restricted to the same shared set of '
        'puzzles, stated in the column header.</li>'
        '<li><strong>B2</strong> &mdash; the extended subgoal vocabulary the loop '
        'plans in; <strong>warm start</strong> &mdash; initializing training from the '
        'previous networks instead of from scratch.</li>'
        '<li><strong>experiment (also called an &ldquo;arm&rdquo;)</strong> '
        '&mdash; one changed version of the loop in the comparison; '
        '&ldquo;arm&rdquo; is the medical-trial term some tables use.</li>'
        '<li><strong>seed</strong> &mdash; the run&rsquo;s random-number '
        'initialization; a result that holds across two seeds is unlikely to be a '
        'fluke.</li>'
        '<li><strong>&ldquo;fluke chance&rdquo;</strong> &mdash; the p-value of the '
        'paired statistical test (McNemar for solve counts, sign test for move '
        'counts): the probability of seeing a difference this large if the change '
        'did nothing.</li>'
        '</ul></details>')

    # the unseen-exam headline: every system on the same 200 fresh puzzles
    out.append("<h3>The unseen-exam headline &mdash; every system, same 200 fresh "
               "puzzles, same columns</h3>")
    unseen_dstar = None                  # exact optima sidecar (oracle, offline)
    ds_path = vd / "exam" / "g24r4_unseen.dstar.jsonl"
    if ds_path.is_file():
        recs = [json.loads(l) for l in ds_path.read_text().splitlines() if l.strip()]
        unseen_dstar = [None] * (max(r["i"] for r in recs) + 1)
        for r_ in recs:
            unseen_dstar[r_["i"]] = r_.get("d_star") or None
        STATS["read"].append(str(ds_path))
    out.append(h2h_table(
        [CmpEntry("forward baseline (MoveNet A*)",
                  load(vd / "baselines" / "forward_movenet_unseen.json")),
         CmpEntry("backward baseline (supervised per-size)",
                  load(vd / "baselines" / "supervised_persize_unseen.json")),
         CmpEntry("the loop's networks before the lab started",
                  load(vd / "baselines" / "seed_nets_unseen.json")),
         CmpEntry("control: one more standard training round",
                  load(vd / "v00_control" / "bench_unseen_astar.json")),
         CmpEntry("networks trained on real move counts (v09)",
                  load(vd / "v09_strict_value" / "bench_unseen_astar.json")),
         CmpEntry("both winning changes combined (v14)",
                  load(vd / "v14_stack" / "bench_unseen_astar.json")),
         CmpEntry("one ordinary move first, then subgoals (v07)",
                  load(vd / "v07_hybrid_actions" / "bench_unseen_hybrid.json")),
         CmpEntry("flagship: two ordinary moves first (v07 + v09 networks)",
                  load(vd / "v07_hybrid_actions" / "bench_unseen_hybrid_d2.json")),
         CmpEntry("three ordinary moves first (limit check)",
                  load(vd / "v07_hybrid_actions" / "bench_unseen_hybrid_d3.json"))],
        ("backward baseline (supervised per-size)",
         "forward baseline (MoveNet A*)"),
        dstar=unseen_dstar,
        note_extra="This is the generalization bar of the whole project: fresh "
                   "boards no network ever saw, no exact labels anywhere. The "
                   "project goal reads directly off the two &Delta; columns: beat "
                   "the backward baseline (done, with fewer moves AND more solves) "
                   "and close the moves gap to the forward baseline (v07 cuts it "
                   "roughly in half). Caveat: the forward baseline is the original "
                   "network; a carefully re-tuned forward planner would likely "
                   "solve a few more puzzles and could narrow these gaps slightly "
                   "&mdash; at the size where re-tuning was tried (smaller boards, "
                   "8 robots) it gained 8 hard puzzles of 184 (see the project log, entry 24)."))

    # per-variant cards
    DISPLAY = {                       # plain card titles (naive-reader gate)
        "v00_control": "The control run",
        "v01_visit_policy": "AlphaZero's own scoring rule",
        "v02_td_blend": "Softened training targets",
        "v03_hard_mining": "Keep only hard practice puzzles",
        "v04_deep_emit": "Train on every examined decision",
        "v05_mean_backup": "Average the branch summaries",
        "v06_gumbel_root": "Lottery-based first-decision exploration",
        "v07_hybrid_actions": "Ordinary moves first, then subgoals",
        "v08_cold_start": "Start from a blank slate",
        "v09_strict_value": "Predict real move counts",
        "v12_frontier_curriculum": "Practice only on failed puzzles",
        "v13_combo": "All three promising changes combined",
        "v14_stack": "Both winning changes combined",
        "v15_slide_training": "Practice on randomly nudged puzzles",
        "v16_ranked_slide_training": "Practice on search-chosen nudges",
    }
    exams = [("graded", "pinned graded (232)"), ("frontier", "pinned frontier (218)"),
             ("unseen", "unseen boards (200)")]
    for vid, v in meta.items():
        res = vd / vid
        gates = {t: load(res / f"gate_{t}_vs_control.json") for t, _ in exams}
        cls, verdict = _var_verdict(vid, gates)
        note = ""
        vj = load(res / "VERDICT.json")
        if ok(vj):                       # curated verdict (replication-aware)
            cls, verdict = vj.get("chip", cls), vj.get("label", verdict)
            note = vj.get("note", "")
        if v.status in ("parked", "stub"):
            cls, verdict = "pend", v.status
        out.append(f'<h3 id="var-{esc(vid)}"><code>{esc(vid)}</code> &mdash; '
                   f'{esc(DISPLAY.get(vid, v.title))} '
                   f'<span class="chip {cls}">{esc(verdict)}</span> '
                   f'<span class="chip">{esc(v.axis)}</span></h3>')
        result_txt = (vj or {}).get("result") or "Still running &mdash; no results on disk yet."
        concl_txt = (vj or {}).get("conclusion") or "Still running &mdash; conclusion pending."
        out.append(f'<p><strong>What we tested:</strong> {esc(v.plain_what)}<br>'
                   f'<strong>Why it might help:</strong> {esc(v.plain_why)}<br>'
                   f'<strong>Result:</strong> {esc(result_txt)}<br>'
                   f'<strong>Conclusion:</strong> {esc(concl_txt)}</p>')
        man = load(res / "generation.manifest.json")
        detail = [f'<p class="note"><em>{esc(v.title)}.</em><br>'
                  f'{esc(v.hypothesis)}<br>{esc(v.mechanism)}<br>'
                  f'{esc(v.expected_failure)}</p>']
        if note:
            detail.append(f'<p class="note"><strong>Statistics:</strong> {esc(note)}</p>')
        if ok(man):
            detail.append(f'<p class="note">Generation: {man.get("instances", "?")} '
                          f'instances, {man.get("solved", "?")} solved, '
                          f'{man.get("records", "?")} certified records, '
                          f'{man.get("seconds", 0):.0f}s '
                          f'(job {esc(man.get("slurm_job_id"))}).</p>')
        out.append('<details><summary class="note">Technical detail (hypothesis / '
                   'mechanism / statistics / provenance)</summary>'
                   + "".join(detail) + '</details>')
        if v.status in ("stub", "parked"):
            continue                      # the four-part card already says it all
        rows = []
        res8 = vd / f"{vid}_s8"

        def _arm_cells(arm_path, ctl_path, label_html, agg, g, cls=""):
            """One table row: shared-puzzle moves + paired delta vs control."""
            ctl_agg = _var_agg(load(ctl_path)) if ctl_path else {}
            pw, both_txt = None, DASH
            if ctl_path:
                r2 = cmp_compare([CmpEntry("arm", load(arm_path)),
                                  CmpEntry("ctl", load(ctl_path))],
                                 ref_labels=("ctl",))
                if not r2.get("error") and r2["entries"][0]["ok"] \
                        and r2["entries"][1]["ok"]:
                    pw = r2["entries"][0]["pairwise"].get("ctl")
            if pw and pw["delta"] is not None:
                both_txt = td_txt(f"{pw['mean_self']:.2f} vs {pw['mean_ref']:.2f} "
                                  f"<span class=\"note\">(n={pw['both_n']})</span>")
            gtxt = "&mdash;"
            if ok(g):
                gtxt = f"p={g.get('mcnemar_p'):.3g}"
            return row([
                td_txt(label_html),
                td(f"{agg['solved']}/{agg['n']}" if agg.get("n") else None),
                td(f"{ctl_agg['solved']}/{ctl_agg['n']}" if ctl_agg.get("n") else None)
                if ctl_path else DASH,
                both_txt,
                _pw_cell(pw) if ctl_path else DASH,
                td(agg.get("mean_regret")) if agg.get("mean_regret") is not None
                else DASH,
                td(agg.get("mean_expansions"), ".1f"),
                td_txt(gtxt)], cls=cls)

        for t, label in exams:
            if vid == "v07_hybrid_actions":
                # search variant: its results are its OWN payloads (hybrid) and
                # its control is the same nets under the standard search
                arm_p = res / f"bench_{t}_hybrid_d2.json"
                if not arm_p.is_file():
                    arm_p = res / f"bench_{t}_hybrid.json"
                ctl_p = res / f"bench_{t}_stdmcts.json"
                g = (load(res / f"gate_{t}_d2_vs_std.json")
                     or load(res / f"gate_{t}_hybrid_vs_std.json")
                     or load(res / f"gate_frontier_d2_vs_std.json" if t == "frontier"
                             else res / f"gate_{t}_hybrid_vs_std.json"))
                rows.append(_arm_cells(
                    arm_p, ctl_p,
                    esc(label) + ' <span class="note">(two ordinary moves first '
                    'vs same-budget standard search)</span>',
                    _var_agg(load(arm_p)), g))
                continue
            arm_p = res / f"bench_{t}_astar.json"
            ctl_p = (vd / "v00_control" / f"bench_{t}_astar.json") \
                if vid != "v00_control" else None
            rows.append(_arm_cells(arm_p, ctl_p, esc(label),
                                   _var_agg(load(arm_p)), gates.get(t)))
            if res8.is_dir():
                a8_p = res8 / f"bench_{t}_astar.json"
                c8_p = vd / "v00_control_s8" / f"bench_{t}_astar.json"
                rows.append(_arm_cells(
                    a8_p, c8_p,
                    f"&nbsp;&nbsp;&#8627; {esc(label)} <em>(seed-8 replicate)</em>",
                    _var_agg(load(a8_p)), load(res8 / f"gate_{t}_vs_control.json"),
                    cls="dim"))
        out.append(table(
            ["exam", "solved", "control solved",
             "moves, on puzzles BOTH solved (experiment vs control)",
             "&Delta; moves (win/loss, fluke chance)", "extra moves vs perfect (graded exam only)",
             "search effort", "solves vs control (fluke chance)"], rows,
            note="How to read this table: see the two comparison rules at the "
                 "top of the tab."))
    out.append("</section>")
    return "\n".join(out)

PANELS = [
    ("The story &mdash; start here", "story", sec_story),
    ("Supervised campaign", "supervised", sec_supervised),
    ("Overview", "overview", sec_overview),
    ("The loop", "loop", sec_loop),
    ("Baselines", "baselines", sec_baselines),
    ("M0 arena parity", "m0", sec_m0),
    ("Ceiling study", "ceiling", sec_ceiling),
    ("Milestone results", "results", sec_results),
    ("Variants lab", "variants", sec_variants),
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
thead th { background: var(--bg);
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
.fig .ns { fill: var(--mut); font: 12px system-ui, sans-serif; }
.fig .lbl { fill: var(--acc); font: 600 11px system-ui, sans-serif; }
.fig .cap { fill: var(--mut); font: 11px system-ui, sans-serif; }
.fig .arr { stroke: var(--mut); stroke-width: 1.6; fill: none; }
.fig .arr.dashed { stroke-dasharray: 5 4; }
.fig .ahead { fill: var(--mut); }
.fig .grid { stroke: var(--line); stroke-width: .5; }
.fig .axis { stroke: var(--line); stroke-width: 1; }
.fig .axl { fill: var(--mut); font: 12px system-ui, sans-serif; }
.fig .s1 { fill: var(--s1); }
.fig .s2 { fill: var(--s2); }
.fig .s3 { fill: var(--s3); }
.fig .chart svg { max-width: 40rem; }
.fig .pt { stroke: var(--bg); stroke-width: 2; }
.fig .pt.c1 { fill: var(--mut); }
.fig .pt.c2 { fill: var(--s1); }
.fig .pt.c3 { fill: var(--s2); }
.fig .pt.c4 { fill: var(--s3); }
.fig .ptl { fill: var(--ink); font: 12px system-ui, sans-serif; }
.fig .refl { stroke: var(--mut); stroke-width: 1.2; stroke-dasharray: 5 4; fill: none; }
.fig .refl.ref2 { stroke: var(--s3); }
.fig .refl-t { fill: var(--mut); font: 12px system-ui, sans-serif; }
.fig .refl-t.ref2 { fill: var(--s3); }
.chip.warn { background: var(--warn-bg); color: var(--warn-ink); }
.pth { border-bottom: 1px dotted var(--mut); cursor: help; }
details.inl { display: inline; margin: 0; }
details.inl summary { display: inline; font-size: .78rem; }
details.inl[open] { display: block; }
footer { margin-top: 2.5rem; border-top: 1px solid var(--line);
         padding-top: .6rem; color: var(--mut); font-size: .82rem; }
"""


# ---------------------------------------------------------------------------
# machine-junk scrubber (owner 2026-08-26: full model paths etc. are provenance,
# not reading material). Applied centrally to the assembled page so every
# section -- including ones rendered from payload-embedded system names --
# is covered by one fix. Visible text only: markup, title="..." hovers and
# <details> blocks (the sanctioned provenance locations) are left alone.
# ---------------------------------------------------------------------------

_REPO_PREFIX = re.compile(r"/scratch/project/open-37-42/petrhyner/MCTS_evolution/")
_ABS_PATH = re.compile(r"/scratch/project/open-37-42/[\w./=+-]+")
_REL_CKPT = re.compile(r"[\w./=+-]*?(?:lightning_logs/[\w./=+-]*?)?[\w=+-]+\.ckpt")


def _fmt_one_path(full: str) -> str:
    """Short, human display for one path; the full path moves to a hover."""
    tail = _REPO_PREFIX.sub("", full)
    if tail.endswith(".ckpt"):
        m = re.search(r"(?:runs/spr/)?(.+?)/lightning_logs/", tail)
        if m:
            disp = m.group(1)
        else:
            disp = re.sub(r"\.ckpt$", "", tail.rsplit("/", 1)[-1])
        disp = re.sub(r"^(?:self_play_robots/)?assets/", "", disp)
        e = re.fullmatch(r"epoch=(\d+)-step=\d+", disp)
        if e:
            disp = f"epoch {e.group(1)}"
    else:
        comps = tail.split("/")
        disp = "/".join(comps[-3:]) if len(comps) > 3 else tail
    # truncate run-dir hash suffixes (literal ellipsis; escaped below)
    disp = re.sub(r"\.([0-9a-f]{8,})", lambda m: "." + m.group(1)[:6] + "\u2026", disp)
    if disp == tail == full:                      # nothing shortened
        return html.escape(full)
    return f'<span class="pth" title="{html.escape(full)}">{html.escape(disp)}</span>'


# one alternation, one pass: inserted markup is never re-scanned, so spans
# can never nest inside each other's title attributes.
_SCRUB_RX = re.compile(
    r"/scratch/project/open-37-42/[\w./=+-]+"        # absolute paths
    r"|[\w./=+-]{18,}\.ckpt"                         # long relative ckpt paths
    r"|epoch=\d+-step=\d+\.ckpt"                     # bare ckpt basenames
)


def _scrub_one(m: "re.Match") -> str:
    tok = m.group(0)
    b = re.fullmatch(r"epoch=(\d+)-step=\d+\.ckpt", tok)
    if b:
        return (f'<span class="pth" title="{tok}">epoch {b.group(1)}</span>')
    return _fmt_one_path(tok)


def _scrub_text(seg: str) -> str:
    # path fragments cut off by upstream text truncation (never node-initial):
    # swallow to an ellipsis before the main pass
    seg = re.sub(r"(.)\(?/scratch/project/open-37-42[\w./=+-]*$",
                 "\\1\u2026", seg)
    return _SCRUB_RX.sub(_scrub_one, seg)


def shorten_paths(page: str) -> str:
    """Scrub machine paths from visible text, leaving <details> blocks intact."""
    out, pos = [], 0
    for m in re.finditer(r"<details\b.*?</details>", page, re.S):
        out.append(page[pos:m.start()])
        out.append(None)                          # placeholder marker
        out.append(m.group(0))
        pos = m.end()
    out.append(page[pos:])
    scrubbed = []
    for part in out:
        if part is None:
            continue
        if part.startswith("<details"):
            scrubbed.append(part)                 # sanctioned location
            continue
        segs = re.split(r"(<[^>]+>)", part)       # tags vs text nodes
        segs = [s if s.startswith("<") else _scrub_text(s)
                for s in segs if s != ""]
        scrubbed.append("".join(segs))
    return "".join(scrubbed)


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
    upd_txt = (f"; status manifest last updated {esc(upd)} "
               f"(cluster local time)" if upd else "")
    n_read, n_missing = len(set(STATS["read"])), len(set(STATS["missing"]))
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Self-play planner suite (self_play_robots)</title>\n"
        f"<style>{CSS}{STORY_CSS}</style>\n"
        '<script>document.documentElement.className += " js";</script>'
        "</head><body>\n"
        f"{nav}\n"
        "<h1>Self-play planner suite "
        '<span class="cite">(self_play_robots)</span></h1>\n'
        '<p class="banner">Everything on this page is built automatically '
        "from the project&rsquo;s result files. Numbers are read from disk, "
        'never typed in &mdash; except the <a href="#story">Story tab</a>, '
        "which is written by hand and checked against the same files. A "
        "missing result shows as <em>pending</em>. "
        f"Generated {stamp} from {n_read} file(s); {n_missing} expected "
        f"file(s) not on disk yet" + upd_txt + ".</p>\n"
        "<noscript><p class='banner'>JavaScript is off, so every tab's content "
        "is shown stacked below and the tab strip acts as plain jump links."
        "</p></noscript>\n"
        + body_html + "\n"
        "<footer>LOCAL FILE ONLY &mdash; generated " + stamp +
        " by <code>self_play_robots/report/gen_report.py</code> from files on "
        "disk; never published.</footer>\n"
        f"<script>{JS}</script>\n"
        "</body></html>\n")
    page = shorten_paths(page)                 # owner: no machine paths in view
    tmp = OUT.with_suffix(".html.tmp")
    tmp.write_text(page)
    tmp.replace(OUT)

    # figure linter (report/svg_lint.py): the Story tab's figures must be
    # overlap-free -- fail loudly so regressions cannot return silently.
    try:
        from svg_lint import lint_html as _svg_lint
        _errs = _svg_lint(page, only_story=True)
        for _e in _errs:
            print(f"gen_report: SVGLINT {_e}")
            STATS["error"].append(f"svglint: {_e}")
        if not _errs:
            print("gen_report: svg_lint story figures CLEAN")
    except Exception as _e:                       # never kill the build
        print(f"gen_report: WARN svg_lint failed: {_e}")

    scan = SCAN
    m0_files = len(scan["comparison"].get("m0", []))
    arms, _ = arena_arms()
    n_arms = len([a for a in arms.values() if a.ref]) if arms else 0
    ceil_files = len(scan["ceiling"])
    loops = loop_dirs()
    loop_txt = ", ".join(f"{k}: iter {'/'.join(str(i) for i, _ in v)}"
                         for k, v in sorted(loops.items())) or "none"
    print(f"gen_report: wrote {OUT} ({len(page)} bytes) | read {n_read} file(s), "
          f"{n_missing} pending, {len(STATS['error'])} error(s) | "
          f"M0 {m0_files}/{n_arms} arm result(s), ceiling "
          f"{ceil_files} arm file(s), "
          f"{sum(len(v) for v in scan['comparison'].values())} comparison "
          f"payload(s) + {sum(len(v) for v in scan['summary'].values())} "
          f"summary.json under results/ | M1 {len(scan['comparison'].get('m1', []))}, "
          f"M2 {len(scan['comparison'].get('m2', []))}, transfer "
          f"{len(scan['comparison'].get('transfer', []))} payload(s) | loops: {loop_txt}")
    for e in STATS["error"]:
        print(f"gen_report: WARN {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
