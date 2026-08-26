# IDEAS.md — what this work is actually about

The conceptual core of the project, stripped of numbers. Written 2026-08-26
against the frozen `paper/DRAFT.md` (evidence pointers below refer to its
sections and claim numbers in `paper/CLAIMS.md`). Each idea: the statement,
why it is novel or contrarian, what the frozen draft demonstrates, and what
would falsify it. Demonstrated vs suggested is marked honestly — this is the
document to think with, not to advertise with.

---

## 1. Certification replaces the answer key

**Statement.** Learned planners are built on answer keys — solved examples
from an exact solver — and exact solvers die of cost as problems grow. The
replacement is not a better teacher but a *referee*: a cheap, sound checker
(here, the game physics replaying a plan move by move) that admits any
solution it cannot refute. Training on referee-admitted, self-generated
solutions changes the failure model fundamentally: a label can be
*suboptimal* (the plan works but wastes moves) but never *wrong* (a plan that
doesn't work produces no label). Wrongness is eliminated by construction;
what remains is a one-sided, measurable error.

**Why novel/contrarian.** Self-training literature mostly fights label noise
statistically — confidence thresholds, filtering, ensembling — treating bad
labels as inevitable and managing their rate. Here the damaging error class
is *structurally impossible*, and the remaining class (suboptimality) is
audited against an exact oracle wherever one still exists. The asymmetry
between solving and checking does the work that human supervision used to do.

**Evidence (demonstrated).** The label-free loop beats the label-trained
planner on its own pinned exams and on never-seen boards (DRAFT §3.1, §3.3;
claims 10–11); a run from *random initialization* reaches the supervised
baseline with zero labeled examples (claim 12). Sharpest supporting result:
the causality experiment (claim 8) showed that even deliberately corrupted
*rankings* don't break training at 4 robots — the thing certification
prevents (invalid plans) is precisely the thing that would.

**Falsified by.** A regime where certified-but-suboptimal data compounds —
each generation training on its own waste until quality ratchets *down*.
The loop's observed plateaus (DRAFT §3.1) are the mild form; a demonstrated
downward spiral, or M6 failing at 80×80 for data-quality (not capacity)
reasons, would bound the idea's reach.

---

## 2. A plan language is a measurable object with a provable ceiling

**Statement.** A planner's action vocabulary — what its plans *can say* — has
a quality ceiling that is a property of the language, not of the model, the
data, or the search budget. And it is measurable: an exhaustive, network-free
probe over the vocabulary yields a floor ("no plan expressible in this
language does better than X") that no amount of training can pass. This
converts the eternal "train harder vs change the approach" argument into a
decidable question: measure the ceiling; if the planner sits on it, training
is done and only language surgery can help.

**Why novel/contrarian.** The field's default reading of a plateau is an
optimization or data problem — tune, scale, augment. Measured expressiveness
ceilings make that reading *refutable per benchmark*. The closest prior
(downward-refinement probability) treats this as theory; here it is an
instrument you run.

**Evidence (demonstrated).** The base-language ceiling was measured, the
trained planner climbed to within 1–2 points of it and stopped (DRAFT §2.2);
five self-play iterations pinned the exam at the ceiling without passing it
(§3.1); vocabulary surgery moved the measured ceiling itself in controlled
steps (claim 6); and the hybrid search finally passed the floor — which is
exactly what the ceiling concept predicts only a language change could do
(claim 15).

**Falsified by.** A training-only method beating a measured floor (would mean
the probe's caps lied — the two probes' 99.6%-vs-98.0% budget discrepancy,
DRAFT Limitation 7, is the small crack to watch). Honest scope limit: an
exhaustive probe is a domain luxury; in richer domains the ceiling may be
real but unmeasurable, and then the decidable question degrades back to
judgment.

---

## 3. Action-space surgery is the lever; tuning is not

**Statement.** Once a planner sits at its language ceiling, the biggest
available gains come from widening *what the search may consider* — not from
better targets, better data, better exploration, or more compute. The recipe
is general: (1) measure the ceiling, (2) extend the vocabulary *minimally*
(one new plan capability at a time, full cost accounting, certification
unchanged), (3) verify at matched budget against the same nets. In this
project a one-to-three-move "slide first, then plan" extension passed the
proven floor; every training-side attempt to achieve the same thing failed.

**Why novel/contrarian.** The community's reflex ordering is: architecture,
then loss, then data, then — rarely — the action space. The evidence here
inverts it: the action space was worth more than every network change
combined, and the two direct attempts to *train* the same capability into
the nets (slide-aware training, twice, sharpened the second time) were flat.
The search can consider what the nets never learned to propose.

**Evidence (demonstrated).** Matched-budget controls (same nets, same
expansions) stay above the floor while the hybrid passes it, 30/0 on paired
moves; the result is net-independent and transfers zero-shot across board
size and robot count (claims 15–17); the training-substitution attempts are
2-seed flat (claim 21).

**Falsified by.** A matched-budget control catching up with more/longer
training (tested twice, failed twice — but only nudge-style training was
tried; full hybrid-*generation* self-play is the untested corner, DRAFT
claim 21 caveat). Scope limit: in domains where widening the action space
explodes branching, the recipe's step 2 ("minimally, with full cost
accounting") is the entire game, and this project only demonstrates one
successful instance of it.

---

## 4. Train on the graded metric; keep everything explored

**Statement.** Two deliberately boring data lessons beat every clever
alternative tried. First: make the value network predict the exact quantity
the benchmark grades (realized move count), not a convenient proxy (abstract
plan cost) — you get what you train for. Second: train on every decision the
search examined and certified, not a curated subset — volume beat curation
every time they competed. And one anti-lesson with teeth: the canonical
AlphaZero policy target (visit-count distributions) was not merely useless
here but *harmful*; the certified-cost target it replaced was load-bearing.

**Why novel/contrarian.** Proxy targets survive because they are
differentiable, smooth, or traditional; visit-count targets survive because
AlphaZero used them. Both defaults lost to direct measurement. The
contrarian content is not "these tricks work" but "the defaults were never
checked in this setting, and checking took one matched arm each."

**Evidence (demonstrated).** Strict-moves value: two-seed win, best quality
of any single change (claim 20). Emit-all: two-seed frontier win (DRAFT §5).
Visit-count targets: decisive multi-exam collapse (claim 19). Curricula and
curation: flat at one and three iterations (claim 22 caveat).

**Falsified by.** Domains where the graded metric is too sparse or noisy to
regress on directly (long-horizon sparse reward), or where explored-but-bad
branches poison the value estimate; the claims are domain findings elevated
to lessons, and they should travel only with their matched-control test
harness attached.

---

## 5. The variants lab: single-delta arms, replication gates, honest kills

**Statement.** Run design exploration as a *program*, not a pile of runs: one
structural change per arm, a frozen matched protocol (same boards, budget,
warm start), a common control, paired statistics, a two-seed gate before any
"win" may be claimed, pre-declared expected failure modes, and kill decisions
recorded with the same care as wins. Track the *ideas'* lineage — which idea
descends from, composes, or replicates which — as a first-class artifact
alongside the runs.

**Why novel/contrarian.** Most ablation tables are post-hoc justifications of
a chosen design. Here the lab *was* the research, and its discipline is what
made the negative space visible: of seven apparent wave-1 wins, replication
killed or downgraded three. An idea-lineage graph (with statuses: adopted /
refuted / superseded) turned out to be the most legible artifact the project
produced — the thing a newcomer can actually read.

**Evidence (demonstrated).** A seed-reversed "win" caught by the gate (v06);
a failed idea sharpened and re-killed rather than quietly dropped (v15→v16);
sixteen verdicts, four adopted, at a cost small against the training runs
they informed (claim 22); the tracker's lineage view as shipped artifact.

**Falsified by.** Not falsifiable — it is a practice, not a claim. Its
measure is error-catch rate per node-hour, and the honest caveat is that the
whole apparatus leans on a cheap, trusted exam; where evaluation itself is
expensive or contested, the gates get slow or soft.

---

## 6. Size-free nets and the warranty curve: verify small, deploy large

**Statement.** Build the network with no size-locked parameters, train it
where exact verification is affordable, and audit its fidelity at every size
verification can still reach. If that fidelity curve is *flat* to the last
verifiable size, the flatness itself is the warranty for deploying beyond it
— not proof, but the only license that can exist past the oracle's horizon,
and an explicit, quantified one. The scaling story is then: supervision
lives at small scale, competence is exported to large scale, and the
warranty curve is the export document.

**Why novel/contrarian.** The standard scaling story extrapolates *loss*;
this extrapolates *audited agreement with ground truth* and is honest about
the exact point where checking becomes impossible (labels past it are named
UNVERIFIABLE in the data itself). It replaces "trust the trend" with "here
is the last measurable rung, and nothing bent before it."

**Evidence (demonstrated / suggested).** Demonstrated: a net trained ≤16×16
holds 86–93% argmin agreement over twenty rungs to 64×64 with no cliff; the
self-play student then *beats its teacher* at every audited size including
64; the action-space fix transfers zero-shot across sizes (claims 7, 13,
17). Suggested only: that this licenses useful *planning* at 80–96 — the
80/96 labels exist with zero timeouts (claim 9), but the downstream test is
M6, in progress and explicitly not claimed.

**Falsified by.** A fidelity cliff just past the verifiable horizon — which
is exactly what M6 is designed to expose if it exists. A softer failure:
flat ordering with drifting calibration (already observed) turning out to
matter downstream after all at extreme sizes.

---

## 7. Beyond this game: wherever a referee is cheaper than a teacher

**Statement.** The whole construction — certification instead of labels,
ceilings instead of plateaus, warranty curves instead of trust — applies
wherever solutions are expensive to produce but cheap to *check soundly*:
program synthesis against test suites and type checkers, theorem proving
against kernel checkers, query/serialization round-trips, and (with a
weaker, noisier referee) robotics in simulation. In each, the same bind
exists (reference solutions run out before problems do) and the same escape
is available (the checker admits self-generated training data that cannot be
wrong, only weak).

**Why novel/contrarian.** The ingredients are known individually (ExIt,
test-driven synthesis, verifier-guided proving). The transferable content is
the *package*: referee-only supervision + a measured expressiveness ceiling
for the system's "plan language" (a synthesizer's DSL, a prover's tactic
set) + minimal language surgery verified at matched budget + a fidelity
warranty for operating past the reference-solution horizon. The prediction
is specific: in those domains too, the language ceiling — not the model —
will be the binding constraint, and widening the action space will beat
tuning.

**Evidence (suggested only).** One domain demonstrated end-to-end. The
cheapest second data point is the Rush Hour port (the supervised plan lists
it at ~1 nh of setup); until a second domain runs, this is an argued
analogy, not a result — and the paper should say so in exactly those words.

**Falsified by.** A second domain where the recipe's diagnosis works but the
lever fails — ceiling measured, planner at ceiling, minimal language
extension found, and matched-budget gains *don't* appear. That would leave
the instruments standing but demote the central lesson to a domain fact.

---

*Cross-cutting honesty note: ideas 1–5 are demonstrated within one domain
and one architecture family; idea 6 is demonstrated up to the verifiable
horizon and suggested past it; idea 7 is suggested only. The falsifiers are
not rhetorical — M6 (idea 6), full hybrid-generation self-play (idea 3), and
the Rush Hour port (idea 7) are the three cheapest experiments that could
break something written here, and all three are known to the project's
future-work list.*
