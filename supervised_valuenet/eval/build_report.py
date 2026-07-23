"""Build eval/results/report.html — the study's self-contained report page.

Usage (from the repo root):
    PYTHONPATH=. python3 -m eval.build_report

On Karolina, the full environment for a build is:
    ml purge && ml Python/3.11.5-GCCcore-13.2.0 bzip2/1.0.8-GCCcore-13.2.0
    source /scratch/project/open-37-42/petrhyner/venv/bin/activate
(bzip2 is needed because the worked example unpickles a board file whose
graph object imports networkx -> bz2; without it that one section renders
as pending and everything else still builds.)

Architecture (all under eval/):
    report_util.py       shared helpers + the three verification registries
    report_theme.py      design tokens, CSS, JS (tabs / theme / tooltips)
    report_charts.py     inline-SVG chart primitives (dataviz-skill specs)
    report_boards.py     Ricochet-Robots board drawings
    report_data.py       loads every result file into one data model
    report_tables.py     result-table builders (check-registered cells)
    report_sections_story.py   tabs 1–3 (study / base scale / plan language)
    report_sections_scale.py   tabs 4–5 (scaling / methods & sources)

Self-verification: every headline number is rendered through ck(), which
tags it with a data-ck id. After the page is assembled, this builder parses
the finished HTML back and compares every tagged span against its
registered source value, checks every needle assertion (strings that must
or must never appear), verifies tag balance, and confirms the page is
fully self-contained (no external references). ANY failure aborts the
build without touching the previous report.html.

No result number is hardcoded anywhere — missing files render as pending
rows, and future results (e.g. the B2-retrained networks) fill in
automatically once their files land (see report_data.RUNGS / BASE_FUTURE).
"""

import os
import re
import sys
from html.parser import HTMLParser

from eval.report_util import (OUT_PATH, ROOT, esc, CHECKS, NEEDLES, SOURCES)
from eval.report_theme import CSS, JS
from eval.report_data import collect
from eval import report_sections_story as story
from eval import report_sections_scale as scale

TABS = [
    ("study", "The study"),
    ("base", "Base scale"),
    ("language", "The plan language"),
    ("scaling", "Scaling"),
    ("methods", "Methods & sources"),
]


def build_page(D):
    contents = {
        "study": story.tab_overview(D),
        "base": story.tab_base(D),
        "language": story.tab_language(D),
        "scaling": scale.tab_scaling(D),
    }
    # methods LAST so its provenance + self-check tables see everything
    contents["methods"] = scale.tab_methods(D)

    head = story.masthead(D)
    tabbar = ['<div class="tabwrap"><div class="tabbar" role="tablist" '
              'aria-label="Report sections">']
    panels = []
    for i, (tid, label) in enumerate(TABS):
        on = " on" if i == 0 else ""
        sel = "true" if i == 0 else "false"
        ti = "0" if i == 0 else "-1"
        tabbar.append(
            f'<button class="maintab{on}" id="tab-{tid}" role="tab" '
            f'aria-selected="{sel}" aria-controls="panel-{tid}" '
            f'tabindex="{ti}" type="button">'
            f'<span class="tabnum">{i + 1}</span>{esc(label)}</button>')
        hidden = "" if i == 0 else " hidden"
        panels.append(
            f'<div class="tabpanel" id="panel-{tid}" role="tabpanel" '
            f'aria-labelledby="tab-{tid}" tabindex="0"{hidden}>'
            f'{contents[tid]}</div>')
    tabbar.append("</div></div>")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Planning with subgoals vs move by move — Ricochet Robots planner study</title>
<style>{CSS}</style>
</head>
<body>
<button class="themebtn" id="themebtn" type="button">theme: auto</button>
<noscript><style>.tabpanel[hidden]{{display:block !important}}
.tabwrap{{display:none}}</style></noscript>
<div class="wrap">
{head}
{"".join(tabbar)}
{"".join(panels)}
</div>
<script>{JS}</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr"}


class PageParser(HTMLParser):
    """Collects data-ck span texts and checks tag balance."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.balance_errors = []
        self.ck_texts = {}
        self._ck_open = []   # (id, depth)
        self.buf = {}

    def handle_starttag(self, tag, attrs):
        if tag in VOID:
            return
        self.stack.append(tag)
        d = dict(attrs)
        if "data-ck" in d:
            cid = int(d["data-ck"])
            self._ck_open.append((cid, len(self.stack)))
            self.buf[cid] = ""

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack:
            self.balance_errors.append(f"closing </{tag}> with empty stack")
            return
        if self.stack[-1] != tag:
            self.balance_errors.append(
                f"expected </{self.stack[-1]}>, found </{tag}>")
        else:
            depth = len(self.stack)
            self.stack.pop()
            while self._ck_open and self._ck_open[-1][1] == depth:
                cid, _ = self._ck_open.pop()
                self.ck_texts[cid] = self.buf[cid]

    def handle_data(self, data):
        for cid, _ in self._ck_open:
            self.buf[cid] += data


def verify(page):
    fails = []

    # 1. structural parse + tag balance
    p = PageParser()
    p.feed(page)
    p.close()
    if p.stack:
        p.balance_errors.append(f"unclosed tags at EOF: {p.stack[-8:]}")
    for e in p.balance_errors[:20]:
        fails.append(f"tag balance: {e}")
    print(f"[verify] parsed {len(page):,} bytes; "
          f"{len(p.balance_errors)} tag-balance problems")

    # 2. every registered check parses back to its registered text
    n_ok = 0
    print(f"[verify] {len(CHECKS)} tagged numbers:")
    for c in CHECKS:
        got = p.ck_texts.get(c["id"])
        ok = (got is not None and got.strip() == c["rendered"].strip())
        n_ok += ok
        status = "OK      " if ok else "MISMATCH"
        if not ok:
            fails.append(f"check #{c['id']} ({c['desc']}): "
                         f"expected {c['rendered']!r}, page has {got!r}")
        print(f"  {status} {c['desc'][:58]:58s} "
              f"{c['rendered'][:20]:>20s}  <- {c['source'] or '-'}")
    print(f"[verify] {n_ok}/{len(CHECKS)} tagged numbers match their source")

    # 3. needle assertions
    n_ok = 0
    for nd in NEEDLES:
        if nd["needle"] is None:
            ok = bool(nd["present"])
        else:
            found = nd["needle"] in page
            ok = found if nd["present"] else not found
        n_ok += ok
        if not ok:
            fails.append(f"needle: {nd['desc']}")
        print(f"  {'OK      ' if ok else 'MISMATCH'} {nd['desc']}")
    print(f"[verify] {n_ok}/{len(NEEDLES)} assertions hold")

    # 4. self-containment: no external references at all
    ext = re.findall(r'(?:src|href)\s*=\s*"(?!#)[^"]*"', page)
    bad_refs = [m for m in ext if re.search(r"^(?:src|href)\s*=\s*\"(?:https?:)?//",
                                            m)]
    if "http://" in page or "https://" in page:
        # allow nothing: even citation text stays offline in this report
        bad_refs.append("literal http(s):// found in page text")
    for m in bad_refs:
        fails.append(f"external reference: {m}")
    print(f"[verify] external references: {len(bad_refs)}")

    # 5. every tab panel non-trivial
    for tid, _ in TABS:
        if f'id="panel-{tid}"' not in page:
            fails.append(f"missing panel {tid}")

    # 6. no duplicate element ids
    ids = re.findall(r'\bid="([^"]+)"', page)
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    for d in dupes:
        fails.append(f"duplicate id: {d}")
    print(f"[verify] {len(ids)} element ids, {len(dupes)} duplicates")
    return fails


def main():
    D = collect()
    page = build_page(D)
    fails = verify(page)
    for rel, e in sorted(SOURCES.items()):
        if e["status"] != "ok":
            print(f"[sources] {e['status']:8s} {rel}"
                  + (f"  ({e['note']})" if e["note"] else ""))
    if fails:
        failed_path = OUT_PATH.replace(".html", ".failed.html")
        with open(failed_path, "w") as f:
            f.write(page)
        print(f"\n[build_report] BUILD FAILED — {len(fails)} problems; "
              f"page written to {os.path.relpath(failed_path, ROOT)} for "
              "inspection; report.html left untouched:")
        for m in fails[:40]:
            print(f"  FAIL {m}")
        return 1
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(page)
    print(f"\n[build_report] wrote {os.path.relpath(OUT_PATH, ROOT)} "
          f"({len(page):,} bytes); {len(CHECKS)} checks + {len(NEEDLES)} "
          "assertions all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
