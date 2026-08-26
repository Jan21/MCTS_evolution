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
