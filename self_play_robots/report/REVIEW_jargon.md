# EXHAUSTIVE jargon audit — every user-visible sentence, every tab (selfplay.html @ 55f4877, regenerated before audit)

Owner escalation: zero tolerance — "if there is a single sentence that I will not
understand…". Method: extracted EVERY visible sentence from the rendered page
(p, li, figcaption, summary, dd/dt, blockquote, h2–h4, **table headers**, captions,
SVG tooltips; `<details>` bodies excluded as the sanctioned expert region — but
note several "Verdict note"/"Generation"/stat lines render OUTSIDE details and are
audited). Each sentence judged: clear on first read for a smart reader with zero
ML/project background, else FAIL with the failing construct named + a plain
rewrite. Repeated identical sentences = one item, counted ×N in totals.

**Adjacency ruling applied** (stated so the final naive-reader gate can check it):
a term counts as glossed if its gloss appears earlier ON THE SAME TAB in normal
reading order (story ch.1 exam table; the variants tab's glossary card; a
caption's own parenthetical). The Glossary tab does NOT gloss other tabs.

## Totals

| region (owner) | audited | passed | FAILED sentences | distinct items |
|---|---|---|---|---|
| Story (builder) | 172 | 157 | 15 | 11 |
| Supervised (builder) | 62 | 43 | 19 | 18 |
| Overview (builder/shared) | 61 | 33 | 28 | 7 |
| The loop (builder/shared) | 25 | 20 | 5 | 4 |
| Baselines (compare agent) | 90 | 76 | 14 | 7 |
| M0 (builder/shared) | 58 | 50 | 8 | 5 |
| Ceiling (builder/shared) | 244 | 204 | 40 | 6 |
| Milestone results (compare agent) | 305 | 260 | 45 | 5 |
| Variants (lab agent) | 413 | 373 | 40 | 9 |
| Milestones (builder/shared) | 481 | 436 | 45 | 4 |
| Glossary (builder/shared) | 58 | 48 | 10 | 7 |
| **TOTAL** | **1,969** | **1,700** | **269** | **83** |

---

## STORY (builder) — 11 items / 15 sentences

S1. "170 / 235 / 269 hard-set solves." — compressed triple + "hard-set" (ch.1
    named these exams "frontier"). → "Frontier solves: 170, 235 and 269 on the
    three board types."
S2. "One round broke the plateau:" — metaphor. → "One round ended the stall:"
S3. [journey v01, 32 words, ambiguous] "v01 removed the old scoring rule:
    candidates are scored by the certified cost of the finished plans that used
    them." — reads as if the colon-clause is the NEW rule. → "v01 replaced our
    scoring rule (score a candidate by the verified cost of the plans that used
    it) with AlphaZero's (prefer what the search visited most)."
S4. "Training collapsed on every exam (p ≤ 2e-7)." — naked p. → "Training
    collapsed on every exam (fluke chance below one in a million)."
S5. [journey v09 label] "train on the graded number" — cryptic. → label
    "predict real move counts"; why-line "the number the project is graded on".
S6. "…predicts real move counts, not abstract cost." — "abstract cost" never
    introduced. → "not an internal plan-step count".
S7. [flagship line] "the depth-2 hybrid search" — "depth-2" unexplained. →
    "the hybrid search, two ordinary moves allowed first".
S8. "…the same accounting as the frozen benchmark." — metaphor. → "counted the
    same way as the fixed benchmark everyone is scored on."
S9. "Paired on the same puzzles: 40/0 move wins." — compressed. → "On the same
    puzzles, 40 solutions got shorter and none got longer."
S10. "The lever was the search, not the training." — metaphor. → "The search
    change did the work; the training change did not."
S11. "seed" ×5 (journey/failures: "seed 7", "second seed", "two-seed rule",
    "both seeds") — unglossed on this tab. → gloss at first use: "seed — the
    run's random starting conditions"; then the other four sentences pass.

## SUPERVISED (builder) — 18 items / 19 sentences

V1. [intro, 46 words] "Before any self-play, a full supervised campaign…" →
    split into three sentences (planners taught from exact-solver answers; the
    backward one is cheap; the forward one is near-perfect but its teacher fails
    on big boards).
V2. "Its frozen result rows are the baselines… and its value-labelling network
    is what bootstrapped the loop." — "frozen rows"/"bootstrapped". → "Its
    results are locked in as the numbers to beat, and its value network gave the
    self-play loop its starting point."
V3. [26 words + semicolon] "This tab regenerates… reads (…); the full log is…"
    → two sentences, no semicolon.
V4. [th] "frontier b / f" — cryptic header. → "frontier: backward / forward
    solved".
V5. [dagger footnote, 40 words] → split; keep the content.
V6. [65-word sentence] "As boards and robot counts grow… still favours forward
    on solves." → three sentences.
V7. "the oracle-mortality showcase" → "the clearest case of the teacher dying".
V8. [34 words] "…the supervised B2 arm was actually worse… than its
    base-vocabulary sibling…" — "arm"/"sibling". → "the B2-trained planner did
    worse than the same planner trained on the plain vocabulary (199 vs 205)".
V9. "Supply existed; supervised training never learned to rank it:" — semicolon
    + cryptic "Supply". → "The extra vocabulary was available. Supervised
    training never learned to rank it:"
V10. "227–228/232 and 153–158 frontier from the same vocabulary" — compressed.
    → "solving 227–228 of 232 standard puzzles and 153–158 of the frontier".
V11. [th] "arm" (seed table) → "training run".
V12. [42 words] "…bimodal — seed 21 fell into a bad basin…" → "The three runs
    split into two groups: seed 21 trained into a much worse network — a known
    training hazard — with 108 of 184 frontier solves against its siblings'
    157–165. The campaign reports the middle result and shows the bad run."
V13. "a 9-arm tune" → "a nine-way tuning sweep".
V14. "selecting by validation only, never by test" → "picking the winner on
    held-out practice puzzles, never on the exam".
V15. [34 words] "…far below the pre-registered kill line." → "…far below the
    failure threshold written down before the test ran." (and split).
V16. "Pooling both exams against the backward median seed…" — "median seed". →
    "against the backward planner's middle-of-three run".
V17. "…the measured seed-noise bars that make every comparison… paired rather
    than aggregate." → "…the measured run-to-run variation limits, which force
    every comparison to be puzzle-by-puzzle."
V18. "…is one of the payloads above, never re-run." → "…is one of the result
    files above, never re-run." (+ [th] "base vocab" → "base vocabulary").

## OVERVIEW (builder/shared) — 7 items / 28 sentences

O1. **Milestone chips render raw FINDINGS strings** (~10 sentences), e.g.
    "g24r4 arm aggregate-exact (205/232, 38.5% opt) with 34/232 float-drift rows
    vs the origin-machine reference (thread-count flip demonstrated)" and
    "frontier A* 133->144->139->158->153->153 (it5 vs it0 p=8e-4)… rg
    0.944/0.861". → systemic: add a one-line plain summary per milestone
    (status.json `plain_note`); move log text into `<details>`.
O2. **STALE, misleads the owner**: "node-hours spent 10.0, projected next 2.5
    (… next: M1 ~1.5-2, M2 ~1)" — real spend ≈50 nh, M1 long done. Regenerate
    from current status or drop the sentence.
O3. **STALE**: "(not yet)" markers beside ~14 sources that exist on disk (the
    same page reads 551 files) — the marker logic is broken; reads as missing
    results.
O4. [north-star quote, 50+ words] "…realized primitive moves from the initial
    state to the terminal state…" → keep the quote but add one plain sentence
    above it: "The goal: beat both hand-taught planners on real moves used, on a
    fixed exam, with the same search budget."
O5. "Deterministic MDP, discount 1; ground truth (exact solver) exists only for
    n ≤ 64…" — MDP/discount jargon + semicolon. → "The game has no randomness.
    An exact solver exists only up to 64×64 boards; physics replay checks
    validity at any size."
O6. "Partial orderings short of that are still results…" → "Results that fall
    short of the full goal still count — every milestone has its own pass bar."
O7. "Non-negotiables inherited by the arena: playable-moves scoring only, the
    pinned per-config bench files, replay certification…" [30 words, terms] →
    "Fixed rules for every test: score only replayed moves, use only the pinned
    exams, certify every solve."

## THE LOOP (builder/shared) — 4 items / 5 sentences

L1. "generate fresh instances… → certify → append to buffer → train net k+1
    (warm-start from k, CollapseStop armed) → gate: bench vs net k AND vs the
    frozen supervised baselines → promote or diagnose." — internal names. →
    plain chain: "…train the next network starting from the last one (with the
    collapse alarm on) → examine it against the previous network and the fixed
    baselines → keep it or investigate."
L2. "greedy descent under the labeler already labels at 86–93% argmin
    agreement" → "the labeling network already picks the same best candidate as
    the exact solver 86–93% of the time".
L3. "keep the distributional HL-Gauss head, bucket count must cover realized
    costs… (the §4.5 clamp bug)" → mark visually as a design quote, or: "keep
    the 96-bin cost output and make sure the bins cover the biggest real costs
    (a past bug silently capped them)."
L4. "MCTS at inference found solutions greedy descent misses." — two unglossed
    names in one sentence. → "the tree search finds solutions the one-shot
    chooser misses."

## BASELINES (compare agent) — 7 items / 14 sentences

B1. "Columns: solve rate = aggregate.solved / aggregate.n and
    aggregate.solve_rate; … d_star_placeholder…" [~6 sentences of field names]
    → collapse into `<details>Column definitions (for auditors)</details>`; one
    plain sentence stays visible.
B2. "Two runs of the same arm at g24r4 that differ only in random seed —
    exact-taught pair: 0.0 pts solve / 1.5 pts optimality; NN-twin pair: 3.4 pts
    solve / 6.5 pts optimality." — compressed + "NN-twin" + semicolons. → "Two
    identical runs differing only in their random start: the exact-taught pair
    differs by 0 solves and 1.5 optimality points; the NN-taught twin by up to
    3.4 and 6.5. Any win smaller than this is noise."
B3. "'This project' rows: the mixed-size self-play curriculum's final nets
    (mix_b2mix_iter3, B2 anytime) at 24/32/8r…" — run names + shorthand. →
    "the final self-play networks (run name in the hover) on the 24×24, 32×32
    and 8-robot exams."
B4. [th] "mean expansions" / "exp" with no local gloss. → "search effort
    (expansions)" once per tab, or a one-line terms strip up top.
B5. "Why these columns" paragraph repeated verbatim ×5 → print once at top;
    afterwards "Same fair-comparison rule as above."
B6. "set is frontier when the aggregate carries d_star_placeholder or the
    protocol's instances file has no d_star at all" → belongs in B1's details.
B7. "Reading the g32r4 pair the way PROBLEM.md §7 does…" → "Reading the 32×32
    pair the way the project brief does…".

## M0 (builder/shared) — 5 items / 8 sentences

M1. PASS chip = FINDINGS string ("bit-exact… float-drift rows… thread-count
    flip demonstrated") → plain: "The harness reproduces the recorded results
    exactly at 16×16; at 24×24 the totals match exactly and 34 rows differ only
    by floating-point rounding (cause demonstrated). Forward arm: exact match."
    Log text → details.
M2. h3 "The arm registry" + [th] "arm" → "The registered test setups" / "setup".
M3. "M0 PARITY PASS per-row differences on (solved, realized_strict,
    expansions, plan_found): 0/450" — machine tuple. → "Parity: PASS — 0 of 450
    rows differ." (tuple into details.)
M4. [36-word harness sentence] → split in two.
M5. "aggregate-exact" coinage → "totals match exactly".

## CEILING (builder/shared) — 6 items / 40 sentences

C1. Headings "g16r4 · b2 vocabulary (g16r4_b2)" ×17 → reuse the supervised
    tab's plain labels ("16×16 · 4 robots — extended vocabulary"), slug in
    hover. (Same fix owed to Results/Milestones headings — R2 below.)
C2. "summary.categories: REALIZABLE_EXISTS = … INCONCLUSIVE = …" enum block
    repeated per arm (~15 sentences) → print once, collapse to details.
C3. Columns-definition paragraph (machine fields) → details, one plain line
    visible (same pattern as B1).
C4. "…abstract plan-cost order (the solver's own admissible ordering — no
    network anywhere)" — "admissible". → "cheapest-plan-first order (the
    solver's own safe ordering — no network anywhere)".
C5. "a plan whose strict count undercuts its abstract cost (an incidental robot
    serving as a stopper) can in principle sit beyond the search bound" → "a
    plan can occasionally cost fewer real moves than its plan-step estimate (a
    robot happens to stand in a useful spot), and such a plan could hide beyond
    where the search stopped".
6. "Why. PROBLEM.md §6.1: …" / "How. …" intros — mark clearly as quotes from
    the brief (they read as unexplained voice shifts).

## MILESTONE RESULTS (compare agent) — 5 items / 45 sentences

R1. The per-table column-definition block ("solved / rate = aggregate.solved …
    spr.gate.paired … mcnemar_p …") repeated ~5× (~30 sentences) → ONE details
    block per tab; visible text: "Columns are defined once here — every table
    uses the same rules."
R2. Section intros carry raw CLI ("spr.bench --search arena_astar
    --prefix-check, 1200 expansions, k=5") → "benched by the standard harness
    (command in the details block)".
R3. Chart caption: two 31/35-word sentences with semicolons and
    "aggregate.solve_rate" → "Each dot is one benchmark run: puzzles solved
    (across) against extra moves used (down). Dashed lines are the proven
    language limits. Down and to the right is better." (provenance → details).
R4. "F-M0 = parity vs the recorded forward row; F-M2 = search variants on
    bench450; F-g24 = the same at 24×24 (when it lands)." — semicolons +
    shorthand. → three short sentences, names spelled out.
R5. "PUCT over slides with the MoveNet guide" → "AlphaZero-style tree search
    over single moves, guided by the move network".

## VARIANTS (lab agent) — 9 items / 40 sentences

VL1. [intro, 89 words] "Every arm runs ONE iteration at g24r4 in the B2
    vocabulary from the same frozen warm-start nets (mix_b2mix_iter2)…" → break
    into 4 sentences; machine names into hovers; "Every experiment starts from
    the same saved networks, gets the same practice budget, and takes the same
    three exams."
VL2. "Verdicts are paired tests (spr.gate compare) vs the control arm." → "Every
    verdict is a puzzle-by-puzzle comparison against the control run."
VL3. v00 title "(the FINDINGS §15 recipe)" → "(the standard recipe from the
    main log)".
VL4. **The 9 "Verdict note" stat runs render OUTSIDE the details blocks**, e.g.
    "Frontier +11/+21 solves at seeds 7/8 (p=0.013 / 0.0002; Fisher 3.4e-5)",
    "Gates: unseen moves 31/1 p=1.5e-8; graded moves 30/0 p=1.9e-9", "Seed-7
    graded-moves win (19/6, p=0.015) reversed at seed 8 (14/16, p=0.86)…" —
    they duplicate the plain Result sentence. → move them (and the
    "Generation: 320 instances…" provenance lines) INSIDE the details block.
VL5. "Standard-exam solve comparison: p = 2.16e-7." / "The other exams show p
    at or below 1e-7." → "The chance these drops are flukes is below one in a
    million on every exam." (numbers → details).
VL6. Boilerplate ×22: "Moves are only ever compared on the puzzles the arm AND
    the control both solved (raw per-arm means cover different puzzle sets and
    are not comparable); −Δ = the arm needs fewer moves." — 33 words, semicolon,
    "arm", "−Δ". → "Move counts are compared only on puzzles both runs solved.
    A minus number means this experiment used fewer moves." (…and print once
    per card group, not under every table.)
VL7. "arm" unglossed across the tab (~50 uses) → add one glossary line ("arm —
    one experiment in the comparison; a term from medical trials") or say
    "experiment" in prose.
VL8. [th] "exp" ×22 → "search effort"; title-chip strings like "knob
    sanity-control" → "sanity check (small tweak)".
VL9. v09 says the same thing three times in a row ("— the number the whole
    project is graded on." / "That number is what the project is graded on." /
    "Train on the number you are graded on.") — sweep artifact; keep one.

## MILESTONES (builder/shared) — 4 items / 45 sentences

MS1. Status notes = raw FINDINGS strings on ~10 milestones (same as O1; the
    biggest single source of FINDINGS-register text on the page) → plain_note +
    details, as in O1.
MS2. "read generically: systems[…].aggregate for the numbers, protocol for the
    exam line…" ×~10 → one details block.
MS3. Gate quotes from PROBLEM.md ("bench within seed noise of the per-size
    supervised pair") — mark as quotes; add plain line where the quote is dense
    ("pass if it matches the hand-built pair within normal run-to-run
    variation").
MS4. [th] "mean regret" / "mean expansions" with no tab-local gloss → terms
    strip at top of the tab (one line, links to Glossary).

## GLOSSARY (builder/shared) — 7 items / 10 sentences

G1. "fidelity gauge" entry opens with jargon ("Argmin agreement of
    self-generated value targets against exact optima") — and "argmin" is never
    defined anywhere on the page. → "At each decision: does the loop's own label
    pick the same best candidate as the exact solver would? Measured cheaply up
    to 64×64."
G2. **STALE/WRONG**: "This project is base-vocabulary unless the owner says
    otherwise (PROBLEM.md §10); the B2 rows on this page are baselines and
    ceiling arms, not training data." — the loop pivoted to B2 with owner
    approval; B2 self-play data IS the training data. Rewrite to current truth.
G3. "a net distilled from iteration-capped B2 data reproduced the cap's
    pathology, FINDINGS 68" → "a network taught from artificially limited B2
    data inherited the limitation (main log, entry 68)".
G4. seed-noise-bars entry [40 words + "FINDINGS 67→71"] → split; "main log,
    entries 67–71".
G5. "…or use a paired-instance test (McNemar over the per-instance solved
    vector)" → "…or compare puzzle-by-puzzle (the McNemar test)".
G6. "Exists only for n ≤ 64 (hard engine assert)…" → "(the solver refuses
    larger boards)".
G7. "The fixed search budget every arm gets…" → "…every planner gets".

## Systemic rollups (fix once, applies page-wide)

- **R1 — FINDINGS-register text as display**: milestone chips/notes (O1, MS1,
  M1) are the top offenders by volume. plain_note + details.
- **R2 — config slugs in headings/prose** (g24r4, g16r8, mix_b2mix_iter3):
  plain labels exist (CFG_LABEL); slugs → hovers. Ceiling/Results/Milestones.
- **R3 — stats-without-words**: apply the variants glossary's "fluke chance"
  idiom to every p-value in a SENTENCE; raw p's live in table cells/details.
- **R4 — semicolons in prose** (STE): V3, V9, O5, R4, VL6 named above; sweep
  the remainder mechanically.

## Not flagged

Bare-number table CELLS; `<code>` provenance paths; `<details>` bodies; glosses
themselves ("MCTS is the tree search that AlphaZero uses"); the ceiling tab's
auto-generated per-arm reading sentences (good plain form — the model to copy).

---

# ROUND 6 — mechanical two-class sweep (selfplay.html, built 2026-08-27 07:53 UTC, 666,933 B)

Auditor: fresh mechanical pass. Scope as briefed: (CLASS 1) every chip label,
table column header, figure/SVG text label and `<summary>` line in a
NON-exempt region, graded individually; (CLASS 2) every quantity that appears
in more than one place; plus six named fix verifications, a parse check and
`svg_lint`.

**Exempt regions used** (computed, not assumed): whole `#results` (L843–903)
and `#milestones` (L971–1091) sections — both open with the "raw results
ledger, kept for auditors" banner; the post-banner tails of `#baselines`
(L702–709), `#m0` (L716–734), `#ceiling` (L745–840); and the body of every
`<details>` element page-wide (`<summary>` lines themselves treated as
NON-exempt). Extractor: 332 distinct micro-copy strings in plain regions.

## VERDICTS

| check | verdict |
|---|---|
| CLASS 1 — micro-copy | **FAIL** — 26 items (4 hard, 12 medium, 10 soft) |
| CLASS 2 — cross-site numbers | **FAIL** — 9 flags, of which 1 is a flat contradiction |
| six named fixes | 5 pass (one with a caveat), **1 fails** (#4, frontier definition) |
| HTML parses cleanly | **PASS** — 0 unbalanced tags (html.parser strict walk), lxml parses to `<html>` |
| `python self_play_robots/report/svg_lint.py …` | **PASS** — `svg_lint: CLEAN`, exit 0 |

---

## CLASS 1 — micro-copy failures (verbatim | location | suggested plain fix)

### Hard failures

**C1-1. `f-m0 pass`** — Overview tab, status strip, L651, `<span class="chip
good">f-m0 pass</span>`. "f-m0" is an internal milestone code (forward-track
M0) that is never expanded anywhere on the page. Every neighbouring chip reads
`pass` / `done` / `running`.
→ `parity checked`

**C1-2. `forward_loop_g24r4`** — Overview strip card name, L653,
`<span class="mk">forward_loop_g24r4</span>`. Raw directory slug *plus* the
named-past-failure config slug `g24r4`.
→ `Move-by-move self-play (24x24)`

**C1-3. `forward_mcts`** — Overview strip card name, L651,
`<span class="mk">forward_mcts</span>`. Raw directory slug; "MCTS" unexplained
on this tab.
→ `Move-by-move planner`

**C1-4. `title="MoveNet A*"`** — Variants tab, unseen-exam headline table,
L910, on the row `forward baseline (move-by-move planner)`. This is the exact
string named in the brief as a past failure, still live as hover micro-copy in
a non-exempt region.
→ `the move-by-move planner's network, run with the quick shortest-path search`

### Medium failures

**C1-5. `variants_lab`** — Overview strip card name, L654. Underscore slug.
→ `Experiment lab`

**C1-6. `B2 frontier`** — Supervised tab, extended-vocabulary table column
header, L606. Code name + project term, and the header states no unit (cells
read `108/134`, `—`).
→ `extended vocabulary: hard-exam solved`

**C1-7. `B2 vocab: solved`** — Supervised tab, same table, L606. Abbreviation
"vocab" plus the code name.
→ `extended vocabulary: solved`

**C1-8. `pooled`** — Supervised tab, seed-replication table column header,
L615. A bare statistics word; cells hold `361/450` (both exams summed), while
the *previous* table's similarly-named header `pooled margin (pts)` means
something entirely different (a lead measured in percentage points). Same word,
two meanings, two tables apart.
→ `both exams combined (solved of 450)`

**C1-9. `Δ moves vs forward supervised (both-solved; − = fewer = better)`** —
Baselines tab, column header, repeated at L690, L692, L695, L697, L700 (5
tables), and as body row label `forward supervised`. "forward supervised" is
result-file jargon; every other surface calls this planner "the move-by-move
planner". The Variants tab's matching header already reads
`Δ moves vs forward baseline (move-by-move planner)`.
→ adopt the Variants wording verbatim, header and row label.

**C1-10. Ceiling table first-column micro-copy** — Ceiling tab, main table,
L744, non-exempt: `g16r4 / b2`, `g16r4 / base`, `g24r4 / b2`, `g24r8 / base`,
`g32r4 / b2`, and probe names `g24r4_frontier_b2 · 120 s · search queue cap
100k`, `g16r4_base_slack12 · slack 12 · 60 s · search queue cap 400k`, … (18
rows). Raw config slugs and raw result-file stems used as the reader-facing row
identity.
→ `24x24, 4 robots — extended vocabulary` / `24x24 hard exam, extended
vocabulary · 120 s cap · …`; keep the file stem in the existing `source file`
column only.

**C1-11. `known optimum d* (mean)`** — Ceiling tab column header, L744.
`d*` is notation. (Mitigating: the "How to read the table" paragraph two lines
above writes "whose exact optimum d* is known", so it is glossed in place.)
→ `shortest possible solution, mean moves`

**C1-12. `MCTS over the chosen action space` / `fixed expansion budget`** —
Loop tab, figure text labels, L667. Three jargon terms in two labels. The Story
tab's fig-4 caption glosses MCTS, but a reader who opens the Loop tab directly
never sees that gloss, and the Loop tab's own caption glosses only "k" and the
collapse alarm.
→ `tree search over the chosen kind of move` / `same search budget every time`

**C1-13. `gate not cleared`** — Loop tab, figure text label, L667. "gate" is
defined only in the Glossary.
→ `did not pass the exam`

**C1-14. `PROBLEM.md §5 — iteration k`** — Loop tab, figure text label, L667.
A source-file citation rendered inside the diagram.
→ move to the figcaption / source note.

**C1-15. `output: scores 0–95 plan-steps`** (Story fig 6 label, L422) and the
matching table cell `cost still needed (scores 0–95 plan-steps — enough for the
largest boards)` (Story networks table, L451). "plan-steps" is glossed only
obliquely, and the label additionally contradicts §7 (see CLASS 2 flag N4).
→ `output: how many more steps it thinks the plan needs (0–95)`, plus a
half-sentence noting that the flagship's value network was later retrained to
predict real robot moves instead.

**C1-16. `log text`** — `<summary>` line, Overview L659 (×11 milestone
entries) and Ceiling L743. Gives the reader no warning that what unfolds is raw
engineering shorthand (`PASS: g16r4 arm bit-exact (0/450 row diffs)…`).
→ `raw log entry (engineering shorthand)`

### Soft failures / consistency nits

**C1-17. Overview strip tooltips** — L646 (M5):
`done: curriculum loop (§20/§22), far-size audits (§17), variants lab waves 1-5
(§23/§25/§26); planner-of-record = v14 training + hybrid d2/d3 search
(flagship: graded 231/232 rg 0.944/0.861, frontier 177/218, unseen 188/200
+1.47 vs perfect; transfers zero-shot to 32/8r with paired 17/0, 23/0); …`
Contains `rg`, `d2/d3`, `32/8r`, `paired 17/0`, `planner-of-record`, bare
`§` numbers. Same pattern at L654 (`adopted v09/v04/v14/v07-d2; killed
v01/v12/v15/v16; … ~28/50 nh`), L651 (`F-M0 pass; F-M2 done (MCTS ≈ A* only at
17x expansions)`), L647 (`M6 lead agent in Phase 0 feasibility (M6_DESIGN.md to
come); budget <=20 nh`), L655 (`drafting: paper/DRAFT.md + paper/CLAIMS.md
(local only)`). These are the "(owner: 2026-08-20: …)" failure class relocated
into `title=` attributes.
→ keep only the already-plain second half of each tooltip; move the register
line into the collapsed log-text block below.

**C1-18. Baselines row tooltips** — L690 `M1 size-free pair
mixed_value_warm_s21`; L692 `mix_b2mix_iter3 nets, B2 anytime A*`,
`v09 strict-value nets + depth-2 slide-prefix hybrid search`; L695 `v09 nets +
depth-2 hybrid, zero-shot transfer`. Run slugs + jargon as hover text.

**C1-19. `title="g16r4"`** on the visible text "16×16, 4 robots" — Overview
L638. Harmless but it is a raw slug surfaced to the reader.

**C1-20. `detail`** — `<summary>`, Overview L657. Unhelpfully generic for a
node-hour breakdown. → `how the 50 node-hours break down`

**C1-21. Baselines section sub-headings** `g16r4 — 16×16, 4 robots — legacy
450-puzzle exam`, `g24r4 — 24×24, 4 robots — pinned graded exam`, `g24r8 …`,
`g32r4 …`, `g24r8 frontier …` (L689/691/694/696/698). Slug-first headings; the
expansion follows immediately, so soft. → drop the leading slug or demote it to
the source note. ("pinned" is also unglossed on this tab.)

**C1-22. `search effort (expansions)`** — Baselines L690ff, Variants L910.
"expansions" is glossed on the Variants tab's own glossary list and in the main
Glossary, never on Baselines.

**C1-23. `solves vs control (fluke chance)` column cells** — Variants tab, all
15 card tables (L914–L967), 51 cells reading `p=2.16e-07`, `p=6.31e-15`,
`p=0.0703`, `p=1`. The header promises words; the neighbouring "Δ moves" column
in the *same row* renders the identical statistic in plain English ("fluke
chance below one in a million"), so a reader sees both idioms side by side.
*Noted: round-5 rollup R3 explicitly ruled raw p-values acceptable inside table
cells. Flagging anyway because the header wording and the sibling column set a
different expectation — owner to confirm the ruling still holds.*
→ render `fluke chance about 1 in 5 million` etc., raw p in the tooltip.

**C1-24. `(n=201)`** — Variants moves cells, L914ff. Statistics shorthand;
"on N puzzles" is used everywhere else on the page.

**C1-25. `depth-2` / `depth-3 slide prefixes`** — Baselines L693 (`against
depth-2's 0.944`), Variants L937 (`same solve count as depth-2`). Prose, not
micro-copy, but the same failure class and adjacent to the flagged headers.
→ `the two-move version`.

**C1-26. `the move-by-move planner arm at 16x16`** — Overview strip subtitle,
L651. "arm" is glossed at L638 ("a cheap comparison arm (a separate test
line)") and in the Glossary, but not here. Also uses ASCII `16x16` where the
rest of the page uses `16×16` (same in L653 `24x24`, L647 `80x80`).

### Micro-copy that PASSED (spot-record, so a later round need not re-grade)
All 11 Story journey chips (`✓ kept`, `✗ flat`, `✓ step`, `✓ jump`, `✗ killed`,
`✓ adopted`, `~ did not repeat`, `from-scratch run`, `✗ killed twice`,
`✓ the break`, `✓ flagship`); all Variants verdict + category chips (`loss
(decisive)`, `targets`, `data`, `search`, `action-space`, `bootstrap`, `combo`,
`win (replicated)`, `not replicated`, `mixed (interference)`, `parked`,
`flagship: wins all three exams`, `control: measures the value of the
supervised head start`) — all defined in the tab intro; all Story SVG labels in
figs 1, 2, 4, 5, 7 and 8 (each glossed by its own figcaption); Story, Overview,
Supervised(ladder) and Baselines(head-to-head Δ-vs-perfect) column headers; the
ceiling `standard` / `hard` / `lower bound` tags; `<summary>` lines "the
brief's own words", "column sources (for auditors)", "Plain-English glossary
(terms used on every card)", "Technical detail (hypothesis / mechanism /
statistics / provenance)", "The original engineering plan …", "the exact
tools". Confirmed ABSENT from every non-exempt region: `MoveNet` (as visible
text), `d_star`, `top1`, `argmin`, `gauge (argmin)`, `owner:` / `owner 2026-`,
`exp` as a bare column, `PUCT`, `Dirichlet`, `val_regret`.

---

## CLASS 2 — cross-site number consistency

### Flags

| # | quantity | site A (value + stated population/anchor) | site B (value + anchor) | verdict |
|---|---|---|---|---|
| N1 | cost of the whole supervised campaign | Glossary, `node-hour (nh)`, L1094: **"~27 nh"** — anchor: "the whole supervised campaign" | Overview note, L657 (details body, exempt): **"supervised campaign ~37"**, feeding the visible "about **50.0** node-hours of compute used so far" (37+13=50) | **CONTRADICTION.** 27 ≠ 37 for the same named quantity; the ledger figure is the one that reconciles with the plain 50.0 total, so the Glossary is wrong. Exempt text must not contradict plain text — it does. |
| N2 | definition of the hard exam / frontier set | Story L256 + Supervised L595 + Glossary L1094: **"the puzzles the exact solver could not crack when the exam(s) were frozen"** | Baselines L699 (plain region): **"the puzzles no planner had solved when the exam was frozen"**; Variants L908: **"218 puzzles where the exact solver failed within its budget"** | **DIFFERENT CRITERIA, no bridge.** "no planner had solved" and "the exact solver could not crack" are not the same set. The Glossary papers over it by asserting both in consecutive sentences; Baselines and Variants each state only one, and different ones. Also fails named fix #4. |
| N3 | run-to-run noise of two identical runs | Baselines yardstick L685 + Glossary L1094: **"up to 3.4 solve points and 6.5 optimality points"**, gate **3.5 / 6.6** — anchor: network-taught twin, 24×24 | Variants control card L916: **"normal run-to-run variation is about 2 to 5 puzzles, depending on the exam"** — anchor: two seeds of the same loop over 232 / 218 / 200 puzzles (≈1.7–2.3 solve points) | **Agree in direction, populations differ, no bridging note.** A reader cannot tell whether the lab's ±5 puzzles is inside or outside the 3.5-point gate the rest of the page is judged by. Add one clause. |
| N4 | what the value network outputs | Story §5, fig-6 label L422 + networks table L451: **"scores 0–95 plan-steps"**, "cost still needed" | Story §7 journey L514 + Glossary `flagship`: v09, adopted and part of the flagship, makes the value net **"predict real move counts … not an internal plan-step count"** | **Unbridged contradiction inside one tab.** Figure describes the pre-v09 target as if current. Add "(the flagship's networks were later retrained to score real robot moves — see §7)". |
| N5 | solver-taught planner's extra moves at 24×24 | Story fig 3 L309 and fig 8 L568: **"solver-taught +4.22"** — *no population stated on either chart*; Supervised ladder L599: **4.22** over its own 205 solves | Baselines g24r4 head-to-head L692, `Δ vs perfect play`: **+3.92** on "the 198 common puzzles with a known optimum" | **Same defect class as the already-fixed +0.94/+0.90/+1.47 case, still open for the baseline planner.** 4.22 vs 4.20 *is* bridged (fig-8 caption, machine rounding); 4.22 vs 3.92 is not, and neither chart says +4.22 is an over-its-own-solves figure the way the caption does for the flagship. |
| N6 | "flagship, +0.90 extra moves vs perfect play" | Baselines g24r4 L692: **+0.90** on "the 198 common puzzles" of the **standard** exam; bridged at L693 only to 0.944/0.861 | Variants unseen headline L910: **+0.90** on "the 93 common puzzles" of the **new-boards** exam; bridged at Story L578 only to +1.47 | **Two different measurements print the same number, each with its own separate bridging note, neither aware of the other.** (A third +0.90 sits in the Ceiling table L744 as the 16×16 extended-vocabulary floor.) Carrying "+0.90" between tabs mis-attributes it. Add a cross-reference to each bridge. |
| N7 | supervised backward planner, 16×16 · 8 robots | Ladder L598: **230/266** standard, **88/184** hard (pools to 318/450) | Seed-replication table L615–617: **253 / 259 / 261** and **108 / 165 / 157**; body text L622: "the backward planner's **middle-of-three run (418/450)**" | **Values differ, populations identical, no bridging note.** The headline ladder run is none of the three replication seeds and its own pooled total never appears; the +11.3-point conclusion is computed from 418, not from the ladder row. Say in one clause why the ladder row is a different run. |
| N8 | Story fig 3 axis anchor | Fig 3 axis label L296: **"extra moves per puzzle, compared with perfect play"** — no exam named, yet all four plotted values (0.07, 1.17, 1.63, 4.22) are 24×24 numbers | Fig 8 axis label L548, otherwise identical chart: **"…compared with perfect play (24×24 standard exam)"** | **Number appears with no anchor.** Copy fig 8's parenthetical onto fig 3. |
| N9 | flagship-vs-move-by-move gap population | Story limits L586: **"+0.79 … on the 219 puzzles both solved. (That is the paired number in Baselines.)"** | Baselines g24r4 L692: visible cell **`+0.79 moves · 6/64`**; "on the 219 puzzles both solved" exists only inside the `title=` tooltip | **Soft.** The Story sends the reader to a site where the anchor is hover-only. Put "219 both-solved" in the visible cell or the column header. |

### Pairs checked and found CONSISTENT (with their bridging notes)

| quantity | sites | verdict |
|---|---|---|
| 16×16 backward, 2.14 vs 2.15 vs 2.04 vs 1.84 | Supervised ladder L596 / Supervised B2 table L606 / Baselines L686 | **OK** — Baselines L686 and Supervised L612 both spell out that these are four distinct recorded runs (prefix-check vs anytime vs extended-vocabulary vs seed-21 replicate), each "average over its own N solves". The past 2.14/2.15 defect is fixed. |
| 24×24 base-vocabulary ceiling, 1.63 vs 1.72 vs 1.69 | Story fig 3 L300 / fig-3 caption L312 / Ceiling prose L741 / Ceiling table L744 | **OK** — "(An earlier, shallower probe gave +1.72 — see the Ceiling tab.)" |
| 16×16 deepest probes, 1.39 vs 1.40 | Ceiling table L744 / Ceiling prose L741 | **OK** — "The two deepest 16×16 probes differ by 0.01 moves." Correctly quoted (not rounded away). |
| noise 3.4 / 6.5, gate 3.5 / 6.6 | Baselines L685 / Glossary L1094 | **OK** — identical in both, and the measured spread is quoted as 6.5 (not 6.6). Past defect fixed. |
| flagship 0.944 ↔ +0.94; 0.861 ↔ +0.86 | Baselines L693 / Variants L939 & L937 / Story L558, L567, L577 / Overview tooltip L646 | **OK** — same run, consistent rounding, "average over its own 231 solves" stated at Story L567 and Variants L937. |
| 231/232, 230/232, 177/218, 174/218, 188/200, 182/200 | Story L578/L582 / Baselines L692 / Variants L939 / Overview L646 | **OK** |
| 158 vs 133 hard-exam solves; 133→144→139→158→153→153 | Overview L644 & L659 / Story L498 / Milestones L993 (exempt) | **OK** — ledger agrees with plain text. |
| 25.9 and 30.3 expansions (rounds 4 and 5) | Story L498 / Results ledger L873, L1031, L1033 (exempt) | **OK** |
| 688 / 30 / 798 expansions, "23 times" | Story L579 / Variants L910 | **OK** — 688/30 = 22.9; all three are new-boards-exam figures and the paragraph says so. |
| mean optimum 7.69 over 232 | Story L256 / Glossary L1094 | **OK** |
| 132 vs 134 of 200 (cold start vs solver-taught) | Story L513 / Variants L941 & L910 | **OK** |
| transfer 174/175 & 172/175 & 17/0; 159/161 & 23/0; 276/289 & 279/289 & 82/2 | Story L582 / Baselines L697, L695, L700 / Overview L646 | **OK** — Story's `**` note states the 172 / 159 / 274 both-solved populations. |
| 418/450 vs 367/450, +11.3 points | Supervised L622 (arithmetic checks: 266+101=367; 418−367=51; 51/450=11.3 pts) | **OK** |
| 1378.6 s ≈ "23 minutes"; 274.9 s ≈ "4.6 minutes"; 7.8 s ≈ "8 seconds" | Supervised L596–603 | **OK** |
| 1,200 expansions / k=5 budget | Story L478 / Variants L908 / Glossary L1094 | **OK** |
| 740,928 / 982,944 / 1,223,232 parameters | Story fig-5 label L394 + caption L396 / networks table L451 | **OK** |

---

## Verification of the six named fixes (checked against the built page, not the changelog)

1. **Two-part project-goal verdict on Overview AND Story — PASS.**
   Overview L644…L645 region, `<p><strong>Where it ended.</strong> The original
   goal had two halves. The first — at least as many solves as the solver-taught
   baseline, with fewer moves — is met. The second — matching the move-by-move
   planner's solution quality — is not. The hybrid halved the gap.</p>`
   Story §8 scoreboard, L580: the identical four sentences (minus the
   "Where it ended." lead-in). Byte-identical wording; both halves present in
   both places; "halved the gap" present in both.

2. **No single-round statistical-improvement claim — PASS (with note).**
   The only significance-bearing loop claim is Overview L644/L659: *"Three
   self-play rounds beat the starting networks on the hard exam: 158 vs 133
   solves (p=5e-6: fluke chance about 1 in 200,000). **One round alone was not
   separable from run-to-run noise.**"* The exempt ledger agrees ("iteration 3
   vs 0 frontier 158 vs 133 (p=5e-6)"). Searched all plain regions for
   `round 1`, `one iteration`, `single iteration`, `iteration 1 vs` — no hits.
   *Note (not a failure):* Story L501 says "One round ended the stall" for the
   mixed-size jump, and Variants L957 says "The gain comes in one round" for
   v14 — both are change-vs-control comparisons, neither claims a statistically
   checked round-over-round loop gain, and Story L501 immediately caveats
   ("Later rounds gave a little back"). The forward-loop tooltip (L653) reports
   two rounds with no p-value; its p=0.04 stays inside the collapsed block.

3. **+0.79 on 219 both-solved; no unlabelled 0.87 — PASS (caveat).**
   Story L586: "the flagship uses +0.79 more moves than the move-by-move
   planner, on the 219 puzzles both solved." Baselines L692 flagship row,
   `Δ moves vs forward supervised` = `+0.79 moves · 6/64`, tooltip "exact sign
   test, p=2.4e-13, on the 219 puzzles both solved". Grep for `0.87`: only two
   hits, both unrelated ledger cells (`10.87` mean-moves at Baselines L708,
   exempt; a `0.87` seconds/instance cell at Results L854, exempt). The old
   figure is gone. **Caveat:** see CLASS 2 flag N9 — "219" is hover-only on the
   Baselines side the Story points at.

4. **One single hard-exam/frontier definition on four surfaces — FAIL.**
   Story L256: "the exact solver could not crack these when the exam was
   frozen." Supervised L595: "puzzles the exact solver could not crack when the
   exams were frozen." Glossary L1094: "the puzzles the exact solver could not
   crack when the exams were frozen. … No planner had solved them then."
   **Baselines L699 (plain region): "Frontier = the puzzles *no planner had
   solved* when the exam was frozen"** — a different criterion, not the solver
   criterion, and it is the only sentence a Baselines reader gets. A fourth
   wording sits on Variants L908: "218 puzzles where the exact solver failed
   *within its budget*." Fix: paste the Story sentence verbatim into Baselines
   L699 and Variants L908.

5. **Baselines bridging clause for +0.861/+0.944 vs +0.90 — PASS.**
   Baselines L693, verbatim: *"Letting the search try up to three single robot
   moves before subgoal planning ("depth-3 slide prefixes") pushes the flagship
   further still: 0.861 extra moves over the known optimum, against depth-2's
   0.944, with the same 231/232 solves. **Those two figures average over each
   planner's own solves; the table's +0.90 counts only the shared puzzles.**
   The depth study lives on the Variants tab (v07)."* Present and correct.
   (Two unrelated nits carried above: the clause uses "depth-2"/"depth-3 slide
   prefixes" — C1-25 — and the "+0.90" it names collides with the Variants
   "+0.90" — N6.)

6. **Noise story 3.4 / 6.5 measured, gate 3.5 / 6.6 — PASS.**
   Baselines L685: "…disagrees by up to **3.4** solve points and **6.5**
   optimality points. … The gate is set just above the measured spread, at
   **3.5** solve and **6.6** optimality points."
   Glossary L1094 `seed noise bars`: "…can differ by up to **3.4** solve points
   and **6.5** optimality points … The gate sits just above the measured
   spread, at **3.5 solve points / 6.6 optimality points**."
   Consistent, and the 6.5-quoted-as-6.6 defect is gone. (See N3 for the
   Variants tab's separate, unbridged "2 to 5 puzzles" noise statement.)

## Mechanical checks

- `python self_play_robots/report/svg_lint.py self_play_robots/report/selfplay.html`
  → `svg_lint: CLEAN`, exit code 0. **0 issues.**
- Strict tag-balance walk (`html.parser`, void/SVG-void aware): **0 errors** —
  no stray closers, no never-closed elements. `lxml.html` parses the file to a
  single `<html>` root without recovery warnings.

## Priority for routing

1. **N1** (27 vs 37 node-hours) — a flat numeric contradiction, one-word fix.
2. **Fix #4 / N2** (frontier definition on Baselines and Variants) — the named
   fix did not land on two of the four surfaces.
3. **C1-1 … C1-4** (`f-m0 pass`, `forward_loop_g24r4`, `forward_mcts`,
   `MoveNet A*`) — the four hard micro-copy failures, all one-line edits, and
   `MoveNet` is a re-appearance of a previously routed defect.
4. **N4** (value-net output label contradicts the adopted v09 change).
5. **C1-6 … C1-16** (column headers and figure labels), then **N3, N5–N9**.
6. **C1-17 … C1-26** (tooltips, summaries, nits) — bulk sweep.

---

# ROUND 6 — FIXES APPLIED (2026-08-27, one editor for all files)

Every item routed above is now fixed in the generators, not in the built HTML.
Rebuild with `python self_play_robots/report/gen_report.py`; `svg_lint` is
CLEAN and a strict tag-balance walk reports 0 errors after the change.

## Numbers (CLASS 2)

- **N1** node-hours. The status manifest's ledger (`~37` supervised + `~13`
  self-play = the visible `50.0`) is the figure of record. The Glossary
  entry now reads "about 37 nh … about 13 nh more … about 50 nh in total".
- **N5** solver-taught +4.22. Both Story number lines carry a second label,
  "over its own 205 solves", and both captions bridge to the Baselines
  +3.92 (the same run on the 198 shared puzzles).
- **N6** the two +0.90s. The "Δ vs perfect play" header now names its exam
  (`h2h_table(..., exam=...)`), and each bridging note points at the other
  site: Baselines = standard exam / 198 shared, Variants = new-boards exam /
  93 shared, "the two match by coincidence".
- **N4** value-net output. Story fig 6 label is now "output: steps still
  needed (0–95)", and the caption plus a new note under the networks table
  say the flagship's value network was retrained (v09) to score real robot
  moves.
- **N3** noise units. The control card (`results/variants/v00_control/
  VERDICT.json`) converts its "2 to 5 puzzles" to "1.0 to 2.3 percentage
  points" and states that Baselines reports up to 3.4 points for the same
  kind of noise.
- **N7** the 16×16 8-robot ladder row. A new note names the runs: the ladder
  row is the base-vocabulary pair under the prefix-check rule (318/450); the
  three seeds retrain the extended-vocabulary pair under the anytime rule;
  the +11.3-point conclusion uses the middle seed run (418/450).
- **N8** Story fig 3 axis now carries "(24×24 standard exam)".
- **N9** every paired Δ cell prints its population: "+0.79 moves, on the 219
  puzzles both solved · 6/64 · fluke chance …".

## The frontier definition (named fix #4 / N2)

`FRONTIER_DEF` in `gen_report.py` holds the one agreed sentence. Story,
Supervised, Baselines (both places), Variants and the Glossary now render it
verbatim: *the puzzles the exact solver could not crack when the exams were
frozen — an optimum exists but is not known*.

## Micro-copy (CLASS 1)

Hard: `f-m0 pass` → `parity checked`; `forward_mcts` → `Move-by-move
planner`; `forward_loop_g24r4` → `Move-by-move self-play (24×24)`;
`title="MoveNet A*"` → "the move-by-move planner's network, run with the
quick shortest-path search". Slug-to-plain mapping lives in
`SIDE_STUDY_NAME` / `SIDE_STUDY_SUB` / `SIDE_STUDY_CHIP`.

Medium: `variants_lab` → `Experiment lab`; `B2 frontier` / `B2 vocab:
solved` → "extended vocabulary: hard-exam solved" / "extended vocabulary:
solved"; both `pooled` headers disambiguated ("both exams combined:
backward's lead in points" and "both exams combined: solved of 450");
"Δ moves vs forward supervised" → "Δ moves vs forward baseline
(move-by-move planner)" (header and row label, all five tables); the Ceiling
first column now reads "24×24, 4 robots — extended vocabulary" with the
probe budgets underneath and the file stem only in the source column;
"known optimum d* (mean)" → "shortest possible solution, mean moves"; the
four Loop figure labels rewritten ("tree search over the chosen kind of
move", "same budget every time", "did not pass the exam", the PROBLEM.md
citation moved into the figcaption); all twelve `log text` summaries →
"raw log entry (engineering shorthand)".

Soft: strip tooltips now carry the plain sentence only (the register line
stays in the collapsed log block); Baselines run-slug hovers replaced with
plain descriptions; `title="g16r4"` and the variant-card slug hovers
removed; `detail` → "how the 50.0 node-hours break down"; the slug-first
`<h4>` headings dropped their slug and "pinned graded exam" wording;
"expansions" glossed in the head-to-head note; the 51 raw `p=…` cells in the
Variants card tables now print the page's "fluke chance …" idiom with the
raw p in the hover; `(n=201)` → "(on 201 puzzles)"; `depth-2`/`depth-3`
prose → "the two-move version" / "the three-move version"; ASCII `16x16`
sizes normalised to `16×16` in status-manifest text.

## Judged NOT to change

- Raw p-values inside `title=` hovers (statistical provenance) stay: the
  visible cell now always carries the plain-English form.
- `results/…json` source paths stay as visible `<code class="src">`
  provenance — they are file names, not prose.
- `story.html` was not touched (owned by another editor at the time).
