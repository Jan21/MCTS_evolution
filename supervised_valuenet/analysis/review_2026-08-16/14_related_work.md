# Novelty and positioning — memo

**1. Closest prior work, and what this repo adds.**
- **kSubS/AdaSubS** (Kozakowski et al.) — the nearest architectural relative: transformer subgoal generator + best-first search, same imitation-then-generalize recipe. This repo's addition: a hand-crafted, physics-grounded subgoal vocabulary (park-helper-then-slide, extended B1/B2) rather than kSubS's generic k-step-ahead state; a stricter whole-plan playability check (99.6%→53.1%) vs kSubS's per-edge reachability; and the genuinely new piece — an exhaustive-probe *ceiling methodology* that partitions failures into no-plan / no-realizable-plan / realizable-but-missed and then does controlled vocabulary surgery with pre-registered per-family predictions. kSubS never measures instance-level expressiveness this way.
- **"What Matters in Hierarchical Search"** (arXiv 2406.03361) — supplies the matched-budget standard this repo explicitly targets (§4.3 compute accounting); not itself a baseline, but the framing yardstick.
- **TAMP/HTN, expert iteration/AlphaZero** — the in-search pruning, anytime reject-continue, and self-play loop are standard; correctly flagged in-repo as "cite, don't claim."
- **Learned heuristics for A*** (DeepCubeA-style value nets) — this repo's value net plays that role for backward search; nothing novel there per se.
- **Ricochet Robots solvers** — literature is complexity-theoretic only (PSPACE-complete, W[SAT]-hard in robot count; Hesterberg & Kopinsky; Balanza-Martinez et al. 2024). No prior learned planner on this domain — a real, underused positioning asset.
- **Size-generalizing GNN/transformer planners, learned labelers/distillation** — the NN-labeler-as-oracle-substitute (§48–74, argmin-fidelity ladder gating downstream equivalence) is closer to weak-to-strong/self-distillation and neural-surrogate-solver lines than to planning papers; currently framed only as an internal engineering fix, underselling it as a second contribution.

**2. Comparison reviewers would demand.** A kSubS-style *generic* subgoal baseline (learned k-step subgoals, no hand-coded vocabulary) at base + one frontier rung — without it, wins can't be attributed to subgoals-as-such vs. the domain-engineered vocabulary. The repo's own publishability.md already flags this (3.1) as reviewer-conditional; it is the single biggest gap.

**3. Next steps (positioning/baselines):**
1. Consolidate the existing lit-mapping (pass5/publishability.md) into an explicit related-work table in the paper. Payoff: forecloses "not novel" cheaply. Cost: small (writing only).
2. Build the kSubS-style generic-subgoal baseline. Payoff: isolates structured-subgoals-as-such from hand-crafted vocabulary — likely acceptance-conditional. Cost: large (~10–20 nh + engineering, already scoped).
3. State the domain-novelty claim plainly (no prior learned planner for Ricochet Robots, only complexity proofs) with citations. Payoff: real, nearly free reframe — "opens a hard domain," not "replays kSubS." Cost: small.
4. Reframe the NN-labeler ladder explicitly against self-distillation/weak-to-strong literature as a second contribution (a calibration study of learned-oracle-substitutes), not just infrastructure. Payoff: opens a second reviewer audience/venue angle. Cost: small–medium (mostly writing, maybe one clarifying probe).
5. A cheap second domain (Lunar Lockout/Atomix before Sokoban) to test the vocabulary-surgery methodology's portability. Payoff: converts "single domain" objection into a demonstrated-general methodology. Cost: large (project-scale).
