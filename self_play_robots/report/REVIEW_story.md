# Independent validation — Story tab + Supervised tab (selfplay.html @ cd7f486)

Validator: independent agent, 2026-08-26. Method: full read of `story.py`/`supervised.py`,
25+ numbers recomputed from primary payloads/checkpoints/code, DOM parse + anchor +
CSS audit of the generated page. **Verdict: NEEDS-FIXES** — the page is close (the
story arc, diagrams and honesty are genuinely strong), but seven concrete items keep
it off 10/10. Everything not listed below checked out exactly.

## Report card

| axis | score | summary |
|---|---|---|
| 1. Human-reader test | 8.5/10 | Clear arc, right chapter order, honest captions; stumbles listed below (items 1, 5, 7, 8, 10) |
| 2. Factual verification | 9/10 | 25+ numbers recomputed — one wrong (item 4), two loose (items 5, 6); everything else exact |
| 3. Mechanics | 10/10 | 0 parse errors; all 23 anchors resolve through the hash router; markers contained in panel p1; all SVG/CSS classes defined; `.tw` scroll containers; SVGs scale at 380 px; noscript fallback; dark theme via variables |
| 4. Consistency with the rest of the report | 8/10 | Banner contradiction (item 2), unexplained −6.5 pooled cell (item 3), 1.63-vs-1.72 cross-tab pointer missing (item 9) |

## Verified correct (spot-checks, primary sources)

Parameter counts by loading checkpoints: value **982,944**, policy **1,223,232**,
sum **2,206,176**; repeated block **740,928** (= enc-less LoopedLayer; ×12 untied
= 8,891,136 ✓); d_model 192, recurrence 12, pe=none, 96 bins; three attention
families g/a/i, concat→proj, MLP 192→768→192; 5-cell readout, distribution-mean
value, HL-Gauss + rank loss (all vs `nn_labeler/model.py`, `spr/nets.py`,
`encode.key_indices`). `expand()` matches the expansion figure. Loop figure
numbers = mix-iter1 manifests (941/864/11,915 records; 5.2 h job). Unseen
scoreboard table: **all 24 cells of all six rows recomputed exactly** from the
payloads + d\* sidecar (137 known, mean 8.71). Paired wins recomputed from rows:
graded d2-vs-std **40/0** (230 both-solved), g32r4 **17/0**, g24r8 **23/0**;
g24r8 frontier **276/289**. Floors: +1.63 is real (`g24r4_base_slack12.json`,
mean_gap_best 1.6296, capped 0) and +1.17 (`g24r4_b2.json`, 1.1711). Supervised
tab: 25/266, 93→101, control 261, rescue 266/266 (+0.098), g32r4 forward 1378.6 s
and 2/275 frontier, g16r4 401/450 vs 450/450 (+0.067), B2 199 vs base 205
(g24r4), all six seed rows incl. the 108/184 bad basin and 157–165 siblings; all
38 files read, none pending. Audit claims 86.3%/83.8% = payload 0.863/0.838.
M1 215-vs-205, B2 loop 133→158 and 26-vs-500 expansions, curriculum 170/235/269,
cold start 132-vs-134 — all match FINDINGS/payloads.

## Fix list (severity order)

1. **Architecture SVG double-counts visually** (`story.py::_fig_arch`). One trunk
   is drawn feeding both heads, and the head boxes are labeled "…982,944 params
   total" / "…1,223,232 params total" — those are FULL-network counts, so a
   reader summing the picture gets 740,928 + 982,944 + 1,223,232 ≈ 2.95 M ≠ the
   stated 2,206,176. The nets do NOT share weights (verified: value = 1,920 enc
   + 740,928 block + head; policy = 1,536 enc + 740,928 block + heads). Fix:
   annotate the trunk box "each network trains its OWN copy of this block" and
   relabel the head boxes "value network — 982,944 params in all" / "policy
   network — 1,223,232 params in all" (or draw two thin trunks).
2. **Page banner now false**: "Auto-generated … no hand-typed numbers" — the
   first tab is an explicitly hand-written narrative. Fix (gen_report.py
   banner): "no hand-typed numbers outside the hand-written Story tab, whose
   figures are verified against EXPLAINER.md".
3. **Supervised ladder, g24r4 row**: the frontier is silently excluded
   (`supervised.py::_ladder_rows`, `cfg != "g24r4"`), so the rendered pooled
   margin is **−6.5** — directly under a note claiming "above base scale the
   pooled margin flips decisively to backward". A careful reader sees the table
   contradict the prose. Fix: footnote the g24r4 cell (why its frontier payload
   is excluded — different net family — and that the cell is graded-only), and
   qualify the note.
4. **Wrong number**: both ceiling figures label the supervised sub-goal planner
   "+4.20"; the payload (`scaling/results/g24r4/comparison.json` backward
   mean_regret) is **4.22**. Fix both figure labels (and keep the dot at
   x≈613.0).
5. **Count mismatch**: journey tree says "21 matched-protocol arms"; the
   Variants lab tab it links to shows 16 cards. Fix: use the tab's own count, or
   "11 distinct ideas, every claimed win replicated on a second seed".
6. **Loose stat**: v01 bullet "every exam collapsed, p ≤ 1e-7" — graded was
   p = 2e-7. Fix: "p ≤ 2e-7" or "p < 1e-6 on every exam".
7. **Unseen table header**: "extra moves vs perfect" is computed over each
   system's OWN solved∩known set (101–134 puzzles, different per row); only the
   note hints at it. Fix: append "(over the puzzles that system solved)" to the
   column header — the flagship-vs-forward comparison is otherwise misread.
8. Nit: ch. 1 "solutions are … 8–30 moves long" — graded optima average 7.59;
   say "up to ~30 moves" or "often 8–30".
9. Nit: ch. 3 cite the floor as the tightened probe ("+1.63, tightened probe;
   the Ceiling tab also shows the earlier +1.72 run") so the cross-tab reader
   isn't left reconciling 1.63 vs 1.72 alone.
10. Nit: lede "a one-line change to the search" oversells v07 (a portfolio
    search module); "a small change to the search" is the honest phrase.

## Not required, noted

- Robot circles use hard-coded `#c94436`/`#3a6fc9` in both themes — acceptable
  (semantic colors), contrast fine on both backgrounds.
- "fluke odds below one in a trillion" for 40/0: one-sided p = 0.5⁴⁰ ≈ 9.1e-13 —
  defensible as written.
- The M1 mixed pair does have a seed-37 replicate (FINDINGS §18); limits ch.
  slightly undersells replication. Conservative direction — leave or soften.

PASS criteria: items 1–7 fixed and the page regenerated → I would hand this to
the owner as 10/10.

---

# Re-grade after the builder's fix round (selfplay.html @ 4ecd602)

Method: every one of the ten fixes verified in the RENDERED page (not the fix
report), plus a full re-run of the mechanics checks and a re-visit of round-1
borderline items.

| # | item | rendered as claimed? |
|---|---|---|
| 1 | arch SVG double-count | ✓ trunk annotated "each network trains its OWN copy of this block"; head boxes "value network — 982,944 params in all" / "policy network — 1,223,232 params in all"; caption spells out 982,944 + 1,223,232 = 2,206,176 — the picture now sums |
| 2 | banner | ✓ "no hand-typed numbers outside the hand-written Story tab (whose figures are verified against EXPLAINER.md)", with a #story link |
| 3 | g24r4 ladder | ✓ dagger on the cells + a footnote that goes beyond the asked fix: it explains the frontier rows at that rung are B2-family arms (shown in the next table), so the cell is graded-only; the note now says the margin "flips decisively at the crowded and large rungs (+44.4, +21.6, +30.9)" with the graded-only −6.5 explicitly qualified |
| 4 | +4.20 → +4.22 | ✓ both figures (labels and captions, 3 occurrences, zero residual "4.20") and the dots moved to x=612.8 = 60 + 4.22×131 exactly |
| 5 | arm count | ✓ "11 distinct ideas, one change each" |
| 6 | v01 p-value | ✓ "p ≤ 2e-7" in the story |
| 7 | unseen header | ✓ "extra moves vs perfect (over that system's own solved set)" |
| 8 | move-length claim | ✓ "up to ~30 moves long" (the 8–30 that remains refers to move-tree DEPTH, which is correct) |
| 9 | floor provenance | ✓ "(+1.63 is the tightened probe; …)" with 1.72 mentioned alongside |
| 10 | lede | ✓ "a small change to the search" |

Mechanics re-run: 0 parse errors, 0 leftover stack; all cross-tab anchors
resolve; page 593,943 bytes. Round-1 borderline items unchanged and accepted
("one in a trillion" for 40/0 is defensible one-sided; hard-coded robot colors
fine; the limits chapter's single-seed caution is conservative given FINDINGS
§18 — acceptable direction).

**Known/routed residual (not counted against the verdict, per coordinator):**
the Variants tab's v01 card renders "All exams collapse at p<=1e-7" from the lab
agent's VERDICT.json — same loose stat the story fixed (graded was 2e-7). It is
worth routing: a one-line VERDICT.json edit ("p<=2e-7"), else the two tabs
disagree by that hair. Location: v01 card, "Verdict note" line, panel 9.

## Final verdict: **PASS — 10/10**

Human-reader 10/10 · factual 10/10 (25+ numbers verified across both rounds,
all now payload-exact) · mechanics 10/10 · consistency 10/10 (with the one
routed residual noted above). I would hand this to the owner as the single-file
knowledgebase it was commissioned to be.

---

# Re-grade round 3 — after the owner's rejection and the STE rework (selfplay.html @ dafe7b0)

Recalibrated: graded with tools against the RENDERED page, not source-reading.

## Builder claims, verified
- **svg_lint, story-scoped: 0 issues** — I ran `python3 svg_lint.py selfplay.html --story` logic myself (`lint_html(only_story=True)`): **0**. gen_report.py wires the story-scoped lint into every build (line ~3745). Claim TRUE.
- **Whole page is NOT clean**: 93 issues — all in two figures OUTSIDE the story tab (see "routed" below).
- **Word count**: 1,407 by my extraction (claimed 1,399; ≈50% of 2,802). TRUE.
- **Prose mechanics** (137 sentences): 0 real sentences >25 words (my single hit is an artifact of concatenating the journey-tree list item), **0 semicolons**, 0 perfect tenses, 4 passives all in STE-permitted descriptive positions ("were used", "is known", "is met", "is parked" as participle-adjective), no idiomatic phrasal verbs ("go below" is literal). Claim TRUE.
- **Sticky thead removed page-wide**: `thead th` is now static — rows can no longer hide under the wrapped tab strip. TRUE.
- **TOC added**: 10 chapter links. TRUE (but see finding A).
- **Facts**: the 29-probe set still holds — every number in the rewritten tab re-checked against my round-1/2 recomputations (unseen table 6 rows, transfer rows incl. 17/0 & 23/0, walls +1.17/+1.63-tightened/+1.72 noted, +4.22, +0.94/+1.42/40:0, params, 133→158, 26-vs-500, 170/235/269, 132-vs-134, 232/218/200/137, 7.59, 8.71). The cuts lost no required fact; the v01 card now renders p≤2e-7 (c838b1e) — last round's residual closed.
- **Read as the owner**: the story prose is now genuinely clean, clear, easy English. Chapter 3 ("First, measure the limit. … Replay each one.") is exemplary.

## New findings

**A. BLOCKER — the sticky story TOC repeats the exact bug class the owner rejected.**
The 11-tab strip needs ≈1,090–1,170 px (141 label chars at 13 px + 11×19 px padding + gaps) but page content maxes at 1,024 px (66rem − 2rem), so the strip wraps to TWO rows at every viewport width. `.story .toc` sticks at `top: 2.32rem` (a one-row assumption) with z-index 4 under the strip's z-index 6 — on scroll, the TOC slides UNDER the strip's second row, exactly as the thead did. The `@media (max-width: 64rem)` de-sticky is on the wrong side: between 64rem and ∞ the strip still wraps (content ≤1,024 px < 1,090 px). Fix: make `.story .toc` `position: static` at all widths (calm, zero risk), or set its `top` from the strip's measured height.

**B. Heads figure: the 96-bin histogram floats outside its box.** Bars at x 742–812; the value-head box ends at x = 722. It reads as stray decoration — unanchored, unconnected. Move it inside the box or add a short connector arrow. (The linter cannot see rect-vs-rect placement; this is exactly the eyeball class the round asked for.)

**C. Nits (non-blocking):** flagship leader line in the break figure clears the "+1" tick label by 0.05 px under the linter's own char model — real fonts may touch (shift label/leader to x≈170); "extended" vocabulary never gets its one-clause definition in the story (add: "extended = the planner may also use temporary blockers and shove robots aside first"); "warm start" and "MCTS" appear in the loop figure unglossed (Glossary tab exists — acceptable); the +1.63 wall label sits 24 px right of its line with no leader.

**D. Routed items (other agents' regions — will trigger the owner's next rejection if left):** the p8 regret-vs-solve chart has **84** linter issues (overlapping point labels, sub-12px text) — the single worst-looking figure on the page now that the story is clean; the p4 loop diagram has **9** (text overflows boxes / leaves viewBox). Both predate this round and belong to the data-tab owners. Route them; the page is graded as ONE artifact by the owner.

## Verdict: **NEEDS-FIXES** (short list)

Scores: story figures 9/10 (lint-clean; B + nits) · prose 9.5/10 (STE-compliant, genuinely clear) · facts 10/10 · mechanics 7/10 (thead fixed, but A re-introduces the same geometry bug on the story's own TOC).
The prose bar the owner set is met. I will not bet the owner reads the page as clean while A (their exact complaint, relocated) and D (84 overlaps one tab over) stand. Fix A + B, route D (and ideally sweep C) → PASS.

---

# Round 4 — final sign-off grade (selfplay.html @ 3592cff)

Graded as ONE artifact, tools first, then a human read.

1. **svg_lint whole-page: CLEAN — 0 issues** (ran it myself). Eyeball of the two
   rebuilt figures: the p8 chart is now proper form — axis captions, ceiling
   reference lines labeled, an in-SVG series legend, 22 dots each with a
   provenance tooltip ("base-language ceiling: solve ceiling 93.1% (…/g24r4_base.json)"),
   and a data table below replacing the 84 overlapping point labels; the p4
   loop got 270-wide boxes and a taller viewBox, texts fit. Story figures:
   nothing new found.
2. **Round-3 items closed, verified in coordinates**: exactly ONE sticky rule
   remains (the tab strip itself) — the story TOC is static at all widths, the
   bug class is dead; the 96-bin histogram now sits at x 646–708 inside its box
   (330–722); the flagship label moved to x=170 and its leader (170,151→181,129)
   clears the "+1" tick label with real margin; the +1.63 wall label has a
   leader.
3. **Junk sweep verified**: 0 visible `/scratch/`, 0 visible 64-char hex; all 23
   remaining `/scratch/` strings live in title-hovers; the single visible
   `.ckpt` is inside a `<details>` provenance block — the sweep's sanctioned
   verbatim location. Short names read as names ("g16r4_backward_policy_b1s21",
   "candidate_scored", "iter5"); 0 hover-less `pth` spans, 0 nested-span
   artifacts, 0 truncation garbage, `epoch=N-step` gone from the surface.
4. **STE spot-read**: the new glosses land ("MCTS is the tree search that
   AlphaZero uses. Warm start: training continues from the last weights."), the
   "extended" clause is in the wall caption, chapters 4/5/8 read clean as the
   owner. One ADVISORY, not a blocker: the p8 chart caption is data-register
   prose (2 sentences >25 words, 2 semicolons, `aggregate.solve_rate` field
   names) — consistent with the data tabs' longstanding provenance style the
   owner has accepted for weeks, but it is the densest paragraph left on the
   page if a future sweep wants it.
5. **Fact integrity: 21/21 story probes + 4 chart ceiling references hold**
   (unseen table all six rows cell-exact, transfer rows, +0.94/+1.42/40-0,
   walls, +4.22, params, 133→158, 215-vs-205, 232/218/200/137, 7.59, 8.71,
   93.1/1.72/98.3/1.17). The sweeps changed display only.
6. **Mechanics**: 0 parse errors; all anchors resolve; **all 134 tables** sit in
   `.tw` overflow containers; dark-mode variables cover every figure class (the
   only hard-coded fills are the two intentional robot colors); body max-width
   + static thead: no hidden rows, no horizontal page scroll.

## Final verdict: **PASS — sign-off given**

I stake the "done" on this: the owner's three rejection reasons (overlapping
figure labels, rows hidden under the tab strip, incomprehensible prose) are
each fixed, regression-gated (svg_lint runs on every build), and verified in
the rendered artifact; the machine-junk surface is clean with provenance
preserved in hovers/details; every number re-checked is payload-exact. One
advisory noted in item 4 for a future polish pass.
