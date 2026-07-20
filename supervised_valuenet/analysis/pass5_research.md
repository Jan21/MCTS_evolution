# Research pass 5 — literature grounding for "hierarchical subgoal planner vs flat move planner"

Scope: five questions from the task brief, each with key sources and plain-language takeaways, then
implications mapped to the addendum's levers (L0–L4) and the Phase B scaling protocol. Every technical
term is defined at first use. Sources were read where feasible (full PDFs for kSubS and the Ricochet
Robots PSPACE paper; full-text extractions for AdaSubS, the "What Matters" study, and the HTN-repair
comparison; abstracts/summaries elsewhere).

Terminology used throughout:
- **Abstract plan / subgoal plan**: a plan written as a chain of intermediate states ("subgoals")
  rather than primitive moves. **Realization / refinement**: turning that chain into an actual legal
  move sequence. Our "strict realization" is a refinement check under full game physics.
- **Flat / low-level search**: search directly over primitive moves (our forward planner).
- **Value function (value net)**: a learned estimate of how far a state is from the goal.
- **Expansion**: one step of a search algorithm where a state is taken from the queue and its
  successors are generated.

---

## Q1. Abstraction executability: abstract plans that fail at the concrete level

**The problem has a name and a 30-year literature.** The failure mode in the addendum (plans found at
the subgoal level that cannot be played as moves) is the classic breakdown of the *downward refinement
property*, and the field's standard responses are exactly the levers already planned.

1. **"Downward Refinement and the Efficiency of Hierarchical Problem Solving" — Fahiem Bacchus &
   Qiang Yang, Artificial Intelligence 71(1):43–100, 1994** (conference version: "The Downward
   Refinement Property", IJCAI 1991).
   [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/0004370294900620) ·
   [1991 version](https://bibbase.org/network/publication/bacchus-yang-thedownwardrefinementproperty-1991)
   Defines the **downward refinement property (DRP)**: an abstraction has the DRP when every abstract
   solution can be turned into a concrete solution without ever having to go back and redo the abstract
   plan. They show hierarchical planning is most effective when the DRP holds, that the DRP is a strong
   requirement rarely met in practice, and — most useful for us — they build an analytical model of
   search cost parameterized by the *probability that an abstract solution is refinable*. When that
   probability drops, the hierarchical advantage shrinks and can invert. Our measured 53.1% strict-solve
   rate is precisely this refinement probability, measured empirically; the addendum's Phase A is, in
   their terms, "raise the refinement probability toward 1".

2. **"Angelic Semantics for High-Level Actions" — Bhaskara Marthi, Stuart Russell, Jason Wolfe,
   ICAPS 2007.** [AAAI page](https://aaai.org/papers/icaps-07-030-angelic-semantics-for-high-level-actions/) ·
   [PDF](https://people.eecs.berkeley.edu/~russell/papers/icaps07-hla.pdf)
   The principled endpoint of the "fix the abstraction" road: give each high-level step a meaning as the
   *set of concrete states it can actually reach* (with computable upper and lower bounds). A plan
   declared solved under the lower bound is *guaranteed* refinable — the DRP holds by construction.
   Relevant as the citation for why our abstraction (which assumes blockers can be moved aside) is
   "optimistic": it uses, in their vocabulary, an upper-bound reachability model, and optimistic-only
   abstractions are exactly the ones that produce unexecutable plans.

3. **"HTN Plan Repair Algorithms Compared: Strengths and Weaknesses of Different Methods" — ICAPS 2025**
   (arXiv 2504.16209; builds on Höller et al., "HTN Plan Repair via Model Transformation", 2020).
   [arXiv](https://arxiv.org/html/2504.16209v1) ·
   [Höller 2020](https://bercher.net/publications/2020/Hoeller2020HTNRepair.pdf)
   **HTN (hierarchical task network) planning** = planning by decomposing high-level tasks into
   subtasks. When a hierarchical plan fails, the field's three families of response are: (a) **full
   replanning** (throw the plan away, plan again), (b) **plan repair** (minimally change the failing
   plan, prized for "stability"), and (c) **backtracking/backjumping** through the decomposition tree
   (undo only the decisions that caused the failure). The 2025 comparison finds no universally superior
   strategy, but repair is typically faster than replanning from scratch. Our "anytime" lever is family
   (a)/(c) — discard the failing plan and continue the same search — which is a recognized, defensible
   strategy, not a hack.

4. **Task and motion planning (TAMP) — e.g. "Policy-Guided Lazy Search with Feedback for Task and
   Motion Planning", Khodeir et al., 2022** ([arXiv](https://arxiv.org/pdf/2210.14055)) and **"A
   Meta-Engine Framework for Interleaved Task and Motion Planning using Topological Refinements",
   2024** ([arXiv](https://arxiv.org/pdf/2408.05795)); overview: [Robohub/TAMP](https://robohub.org/integrated-task-and-motion-planning-tamp-in-robotics/).
   **TAMP** = robotics planning where a symbolic task plan (pick A, place on B) must be refined into
   continuous motions. The robotics community explicitly calls our failure "downward-refinement
   failure", and its *standard* architecture is: propose a task plan, attempt refinement with the real
   physics/geometry, and on failure **reject the plan and keep searching, feeding the failure reason
   back into the next proposal**. This is the strongest precedent for L0 (reject-and-research) and L4
   (failure-type feedback into proposals): a whole subfield operates this way.

5. **"Hybrid Search for Efficient Planning with Completeness Guarantees" — Kalle Kujanpää, Joni
   Pajarinen, Alexander Ilin, NeurIPS 2023.** [arXiv](https://arxiv.org/abs/2310.12819)
   Documents that learned subgoal searches "may fail to find a solution even if one exists" and fixes it
   by mixing primitive-move edges into the high-level search ("complete subgoal search"): if subgoals
   fail, the search can always fall back to single moves, restoring **completeness** (the guarantee of
   finding a solution when one exists) while keeping most of the speed. A ready-made citation — and a
   candidate extra lever — if L0–L4 plateau.

**Bottom line for Q1.** "Keep searching until a refinable plan is found" has clear precedent in three
separate literatures (Bacchus & Yang's refinement-probability model, HTN repair-vs-replan, TAMP's
reject-with-feedback loop). The literature's *preferred* placement of the check, however, is per step
during search (TAMP; and see kSubS below, which validates every subgoal edge as it is created), not
once at the end — which is the addendum's L3.

---

## Q2. Learned subgoal search beating flat search as the horizon grows

1. **"Subgoal Search For Complex Reasoning Tasks" (kSubS) — Konrad Czechowski, Tomasz Odrzygóźdź,
   Marek Zbysiński, Michał Zawalski, Krzysztof Olejnik, Yuhuai Wu, Łukasz Kuciński, Piotr Miłoś,
   NeurIPS 2021.** [arXiv](https://arxiv.org/abs/2108.11204) ·
   [NeurIPS PDF](https://proceedings.neurips.cc/paper/2021/file/05d8cccb5f47e5072f0a05b5f514941a-Paper.pdf) (read in full)
   Architecture: a learned generator proposes states k moves ahead; a low-level policy (or plain
   breadth-first search) connects consecutive subgoals; a value net guides a best-first search over the
   subgoal graph. Two details matter for us: (i) **every subgoal edge is realized before it enters the
   search graph** — if the low-level policy cannot reach the proposed subgoal within a step limit it
   returns an empty path and the subgoal is discarded ("used as a pruning mechanism"); in Rubik's Cube
   and the INT theorem-proving benchmark about **50% of generated subgoals are discarded** this way, and
   on Sokoban ~17% of subgoals move *away* from the goal and ~5% lead into dead ends. (ii) The headline
   evidence is **solve-rate-versus-budget curves** (success on 1000 instances as a function of search
   graph size), plus difficulty scaling: Rubik's Cube at budget 6000 nodes, flat best-first search
   solves <10% vs kSubS 99.2%; Sokoban 16×16 at budget 50 nodes, 4% vs 42%; INT proofs of length 15 at
   budget 50, 9% vs 38%. They also propose *why* subgoals win: value-net errors are relative — jumping k
   steps gives a larger true value change per evaluation, so noisy values still rank states correctly
   (probability the value decreases along a solution path: 0.32 stepping 1 move vs 0.02 stepping k=4).

2. **"Fast and Precise: Adjusting Planning Horizon with Adaptive Subgoal Search" (AdaSubS) — Michał
   Zawalski et al., ICLR 2023 (notable top-5%).** [arXiv](https://arxiv.org/abs/2206.00702)
   Adds generators at several distances plus a **learned verifier** — a binary classifier trained on
   reachability labels harvested from the search itself — that cheaply rejects unreachable subgoals
   (uncertain cases fall back to the expensive low-level check). When a subgoal fails verification the
   search retracts and tries closer, more conservative subgoals. Their difficulty-scaling design:
   train on INT proofs of length 15, test up to length 28 — AdaSubS keeps >50% of its performance while
   the fixed-k version halves by length 21. Precedent both for L3 (validated subgoals inside search) and
   for the L4 option "learn a feasibility signal".

3. **"What Matters in Hierarchical Search for Combinatorial Reasoning Problems?" — Michał Zawalski,
   Gracjan Góral, Michał Tyrolski, Emilia Wiśnios, Franciszek Budrowski, Marek Cygan, Łukasz Kuciński,
   Piotr Miłoś, 2024.** [arXiv](https://arxiv.org/abs/2406.03361)
   The field's own audit of when subgoal methods actually beat flat search. Findings: subgoal methods
   win when **value functions are hard to learn or noisy** (kSubS reaches 40% success with completely
   random values), when the **action space is large** (with a 100×-inflated action space flat search is
   stuck below 30% while subgoal methods hold up), when the environment has **dead ends**, and when
   training data comes from diverse/suboptimal experts. They *lose their edge* on clean, homogeneous,
   short-horizon data — on tidy Rubik's Cube data AdaSubS and flat best-first search become "nearly
   identical". This is the single most useful external reference for predicting where our crossover
   will and won't appear.

4. **"Divide-and-Conquer Monte Carlo Tree Search For Goal-Directed Planning" — Giambattista
   Parascandolo, Lars Buesing, et al. (DeepMind), 2020.** [arXiv](https://arxiv.org/abs/2004.11410)
   **MCTS (Monte Carlo tree search)** = search that grows a tree by sampling promising branches. Here
   the planner proposes a *midpoint* subgoal and recursively solves both halves, rather than planning
   moves in execution order. With a learned subgoal proposer it beats sequential planning on
   long-horizon grid navigation and continuous control — independent (non-Warsaw) evidence that learned
   subgoal decomposition is what defeats depth.

5. **"Solving the Rubik's cube with deep reinforcement learning and search" (DeepCubeA) — Forest
   Agostinelli, Stephen McAleer, Alexander Shmakov, Pierre Baldi, Nature Machine Intelligence, 2019.**
   [Paper](https://deepcube.igb.uci.edu/static/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf)
   The canonical *flat* learned planner: a value net trained by approximate value iteration plus
   weighted A* solves 100% of Rubik's Cube instances and generalizes to the 15/24/35/48-puzzle, Lights
   Out and Sokoban. Important as the fair strong-flat-baseline citation: flat learned search *does*
   reach 100% — but only with large searches (their solves expand thousands-to-millions of nodes),
   which is exactly the budget-explosion axis our forward planner is on.

6. **"Data-Efficient Hierarchical Reinforcement Learning" (HIRO) — Ofir Nachum, Shixiang Gu, Honglak
   Lee, Sergey Levine, NeurIPS 2018.** [arXiv](https://arxiv.org/pdf/1805.08296)
   Reinforcement-learning-side corroboration: a two-level agent where the top level emits subgoal
   states and the bottom level chases them solves long-horizon control tasks that flat agents fail,
   with far fewer environment samples. (Conceptual ancestor: the **options framework** — Sutton,
   Precup & Singh 1999 — "temporally extended actions" as the general formalism for acting at a
   coarser time scale.)

**Experimental designs that made the claim credible** (recurring pattern across 1–4): fixed instance
sets of 500–1000 problems with confidence intervals; *curves* of solve rate against search budget, not
single budget points; a difficulty axis (board size 12→20, proof length 5→15, out-of-distribution
scramble/proof lengths) showing the gap *widening*; shared components between compared systems
wherever possible; and budget accounting that includes the low-level work (next section).

---

## Q3. Matched-budget evaluation across different action granularities

1. **"What Matters in Hierarchical Search…" (as above), Section 4.3.** [arXiv](https://arxiv.org/abs/2406.03361)
   The explicit methodological ruling: **count all visited states, at both levels** — "reporting only
   the high-level nodes excessively enhances subgoal methods' scores". Budgets must include both
   subgoal generation and the low-level path-filling between subgoals; components (e.g. the value net)
   should be shared across compared algorithms where possible; the headline metric is solve rate vs
   total budget. This is the paper to cite for the comparison protocol — and the warning our "2.9
   expansions" number must be defended against.

2. **kSubS (as above), Section 4.3.** The worked example of granularity-fair accounting: for INT and
   Rubik they *include* the nodes visited by the low-level connector in the reported "graph size"
   because it costs real network calls; for Sokoban they *exclude* the breadth-first connector after
   measuring that it is <1% of runtime — i.e., the rule is "include low-level work unless measured
   negligible, and say so". For MCTS they switch the budget unit to MCTS passes, showing budget units
   are method-relative and must be justified per method.

3. **AdaSubS (as above), appendix tables.** Goes one step further: reports **network calls broken out
   by head** (generator / verifier / low-level policy / value calls) alongside node counts. Since a
   subgoal expansion can cost several network calls while a move expansion costs one, per-head call
   counts are the cleanest common currency for "one subgoal = many moves" comparisons.

4. **"Implementing Fast Heuristic Search Code" — Ethan Burns, Matthew Hatem, Michael Leighton,
   Wheeler Ruml, SoCS 2012.** [PDF](https://www.cs.unh.edu/~ruml/papers/implementation-socs-12.pdf)
   The classical-search critique of expansion counting: node expansions ignore per-node cost, and
   implementations expanding the *same* number of nodes differ by up to 28× in wall-clock time. Their
   position — report solving time, because theoretical effort measures don't predict practice — is the
   standard citation for why expansions alone are not enough. (In learned planning the per-node cost
   asymmetry is even larger: a physics check vs a neural-net forward pass.)

**Bottom line for Q3.** The defensible triple is: (i) solve rate vs *total* budget curves counting both
levels, (ii) neural-net calls per instance broken out by network, and (iii) wall-clock — with any
excluded cost (like a cheap physics-only realization check) explicitly measured and declared
negligible, kSubS-style. Expansion-matched single-point comparisons are exactly what the literature
warns against.

---

## Q4. Ricochet Robots: complexity, solvers, and how depth scales

1. **"A Simple Proof that Ricochet Robots is PSPACE-complete" — Jose Balanza-Martinez, Angel A.
   Cantu, Robert Schweller, Tim Wylie, 2024.** [arXiv](https://arxiv.org/pdf/2402.11440) (read pp. 1–5)
   **PSPACE-complete** = as hard as any problem solvable with polynomial memory; believed strictly
   harder than NP-complete, and it implies optimal solutions need not have short (polynomial-length)
   certificates — plan lengths can in general blow up. Their Table 1: with full-step sliding and
   per-robot moves (our exact rules), relocation was proven PSPACE-complete in 2003 (via the
   equivalent game Lunar Lockout with fixed geometry); this paper gives a simpler proof. NP-hardness
   dates to 2001. All four "tilt" variants except unit-step-per-move are PSPACE-complete.

2. **"Randolph's Robot Game is NP-hard!" — Birgit Engels, Tom Kamphans, Electronic Notes in Discrete
   Mathematics, 2006 (report 2005).** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1571065306000631)
   Earliest hardness result phrased on Ricochet Robots itself: deciding whether a given cell is
   reachable is NP-hard for arbitrary board layouts. Useful as the "even reachability is hard" citation
   — relevant to why our realization check (reachability under blockers) is nontrivial.

3. **"The Parameterized Complexity of Ricochet Robots" — Adam Hesterberg, Justin Kopinsky, Journal of
   Information Processing 25:716–723, 2017.** [Semantic Scholar](https://www.semanticscholar.org/paper/The-Parameterized-Complexity-of-Ricochet-Robots-Hesterberg-Kopinsky/4653295f8183513d23f8484e187eabc958ee16c4)
   **Parameterized complexity** asks whether hardness can be confined to a parameter (here: robot
   count) so that instances with few robots are easy regardless of board size. Result: the puzzle is
   W[SAT]-hard in the number of robots — meaning it is very unlikely to be "fixed-parameter tractable",
   i.e., **the robot count is provably the hardness knob**, not the board size. Directly supports the
   addendum's claim that the robot axis and grid axis stress different things, and predicts the robot
   axis is the fundamentally harder one.

4. **"Ricochet Robots: A Transverse ASP Benchmark" — Martin Gebser, Holger Jost, Roland Kaminski,
   Philipp Obermeier, Orkunt Sabuncu, Torsten Schaub, Marius Schneider, LPNMR 2013** (and "Ricochet
   Robots Reloaded", 2015). [Springer](https://link.springer.com/chapter/10.1007/978-3-642-40564-8_35)
   The published-solver reference: 256 instances on an authentic 16×16 board (four robots in the
   corners, target at each cell), solved with answer-set programming (a logic-solving technology) under
   a bounded move horizon; the optima in this benchmark run up to roughly 20 moves, most far shorter.
   Establishes that published exact solving is done at 16×16/4 robots — nobody has published exact
   benchmarks at substantially larger scales, consistent with our oracle straining at g16r6.

5. **Michael Fogleman's solver analyses (2011–2017).**
   [Project page](https://www.michaelfogleman.com/projects/ricochet-robot/) ·
   [histogram post](https://fogleman.tumblr.com/post/10962311432/ricochet-robot)
   The best empirical depth-distribution data: over millions of random 16×16/4-robot positions, most
   optima are under 10 moves; the hardest found was 21 moves, taking minutes of optimized iterative
   deepening (a memory-light exhaustive search) — i.e., exact-search time visibly explodes with depth
   even on the standard board. Community solvers (e.g. Rust/A* implementations on GitHub) confirm the
   same regime.

6. **Derived scaling observation (ours, not a citation).** With R robots on an n×n board there are at
   most (n²)^R distinct configurations, so optimal solutions at *fixed robot count* are polynomially
   bounded in board size, while the bound grows *exponentially with robot count* — matching the
   W[SAT]-hardness picture (robots = hardness) and implying mean optimal depth d* grows along both of
   Phase B's axes, with the robot axis the steeper one in the worst case. (Curiosity supporting the
   depth-can-explode point: on an infinite board the game is Turing-complete — JIP 2023,
   ["Ricochet Robots with Infinite Horizontal Board is Turing-complete"](https://www.jstage.jst.go.jp/article/ipsjjip/31/0/31_413/_pdf).)

---

## Q5. Oracle-free self-play labeling vs exact-solver labeling

1. **"Thinking Fast and Slow with Deep Learning and Tree Search" (Expert Iteration) — Thomas Anthony,
   Zheng Tian, David Barber, NeurIPS 2017.** [Abstract](https://ui.adsabs.harvard.edu/abs/2017arXiv170508439A/abstract)
   The template our self-play loop instantiates: the "expert" is the learner's own search (tree search
   guided by the current net), the "apprentice" net is trained on the expert's outputs, and the
   improved net makes the expert stronger next round. Trained from scratch with no external teacher, it
   beat MoHex, the reigning computer-Olympiad Hex champion. Canonical citation for "the labeler is the
   search itself, so labeling never needs an outside solver".

2. **"Mastering the game of Go without human knowledge" (AlphaGo Zero) — David Silver et al., Nature
   550:354–359, 2017.** [Nature](https://www.nature.com/articles/nature24270)
   The best-known instance of the labeling asymmetry: the self-play-only system surpassed the versions
   trained on expert (human) data. Standard citation for "self-generated labels scale past curated
   labels" — the qualitative claim; it does not by itself prove the claim for puzzle domains.

3. **DeepCubeA (as above, 2019).** Its value net is trained by **approximate value iteration** on
   states generated by scrambling *backward from the solved state* — no solver ever labels anything,
   yet the flat planner reaches 100% solve rate. Precedent that backward-from-goal data generation
   fully replaces an oracle even in a flat pipeline; relevant because our forward planner's *best*
   recipe (candidate-scoring) is the one chained to the oracle, and DeepCubeA marks the oracle-free
   alternative for the forward side of the fair "oracle-free vs oracle-free" pairing.

4. **"Learning heuristic functions for large state spaces" (Bootstrap learning) — Shahab Jabbari
   Arfaee, Sandra Zilles, Robert C. Holte, Artificial Intelligence 175:2075–2098, 2011.**
   [PDF](https://webdocs.cs.ualberta.ca/~holte/Publications/AIJ2011LearningHeuristics.pdf)
   The classical-search precedent for oracle-free labeling: start from a weak (even all-zero)
   heuristic, solve whatever instances you can within a time cap, train on your own solutions, repeat
   on harder instances — bootstrapped with random walks backward from the goal when too weak to solve
   anything. Solved 24-puzzle, 35-pancake, Rubik's Cube and blocks-world instances where no practical
   exact labeler existed. Directly supports "self-play labeling works where exact-solver labeling
   does not", a decade before deep nets.

5. **"Solving Hard AI Planning Instances Using Curriculum-Driven Deep Reinforcement Learning" —
   Dieqiao Feng, Carla Gomes, Bart Selman, IJCAI 2020** (companion NeurIPS 2020 paper on automated
   curricula). [arXiv](https://arxiv.org/abs/2006.02689)
   On hard Sokoban instances, their self-labeled, curriculum-driven learner solves within a day
   instances that specialized exact solvers cannot solve in any reasonable time — a published,
   quantified case of "the exact solver fails exactly where the learning loop keeps working", the
   same asymmetry the addendum wants to report (oracle cap-outs at g16r6 vs self-play continuing).

6. **kSubS training-data note (as above).** Even the subgoal-search papers avoided exact labels:
   Sokoban training data came from a suboptimal RL agent's trajectories, Rubik data from random
   backward walks ("highly suboptimal"), and they note the INT proof engine "can easily generate
   multiple proofs of random statements, but *cannot* prove a given theorem" — i.e., label generation
   was designed around what the labeler can produce at scale, not around optimality. Supports labeling
   with suboptimal/self-generated plans (our strict-filtered winners) rather than optimal oracle plans.

---

## Implications for this project

**L0 — anytime reject-and-research (eval-side).**
Do it; it is standard practice under several names: TAMP's reject-and-replan-with-feedback loop,
HTN repair's "backtrack/replan on refinement failure", and Bacchus & Yang's framing (retrying abstract
solutions until one refines is exactly search under refinement probability p ≈ 0.53, with ~400×
expansion slack to spend). Cite Bacchus & Yang for the problem, TAMP (e.g. lazy-search-with-feedback)
and HTN repair for the strategy. One caution from the same literature: their model also says that if p
decays with depth (ours does: 95%→20%), retry-only fixes stall on deep instances because *all* nearby
abstract plans share the flawed assumption — which is why L0 is a measurement, not the destination.

**L1 — make the loop log the strict metric.**
Directly mirrors published practice: kSubS reports the fraction of generated subgoals that are
reachable (~50%) and the distribution of subgoal quality; AdaSubS logs per-head verification traffic.
"What Matters" makes the sharper methodological point: any number that hides low-level infeasibility
inflates subgoal methods. Until the loop logs strict-pass fraction and strict cost, its numbers are
the kind that paper tells reviewers to discount.

**L2 — strict filter / strict-cost labels.**
Precedented twice: AdaSubS's verifier is *trained on exactly the labels the strict filter produces*
(reachable / not-reachable outcomes harvested from search), and the bootstrap-learning line (Arfaee;
ExIt; kSubS's suboptimal data) shows training on self-generated, feasible-but-suboptimal solutions is
the norm, not a compromise. The addendum's caution on mixed units (abstract g + strict-trained value)
has an echo in kSubS's finding that ranking quality, not absolute calibration, is what drives search —
watch ranking metrics, not just loss scale.

**L3 — realization check inside generation/search.**
This is where the literature is loudest: kSubS validates *every* subgoal edge with the low-level
policy before it enters the search graph (unrealizable proposals never contaminate the plan), and
AdaSubS only adds a learned shortcut to make that check cheap. Deferring realization to the end of
the whole plan (our current design) has no direct precedent in the learned-subgoal-search line; L3
converts our planner to the standard architecture. Expect the addendum's "two halves reinforce each
other" claim to hold: in kSubS the check is simultaneously a search filter and (via AdaSubS) a source
of training labels.

**L4 — fix the abstraction at the source.**
Three graded precedents matching the addendum's three options: (i) a learned feasibility feature =
AdaSubS's verifier network; (ii) failure-type feedback shaping proposals ("a robot sits on the path" →
propose moving it) = TAMP's failed-refinement feedback into the task planner; (iii) the principled
ceiling = angelic semantics (make the abstraction's reachability claims sound, so plans are refinable
by construction). And if L0–L4 all plateau, Kujanpää et al.'s hybrid search (mix primitive-move edges
into the subgoal search) is the published completeness-restoring fallback — a candidate L5.

**Phase B protocol — what to keep, add, and avoid.**
- *Keep*: same instance file for all systems, strict moves only, regret vs oracle-computed d*,
  reporting labeling cost and oracle-failure rate (Feng et al. and the kSubS data-generation notes are
  the precedents that make labeling cost a legitimate first-class result).
- *Add*: solve-rate-vs-budget **curves** (several budgets, not just 1200) — every credible subgoal-vs-
  flat claim in the literature is a curve plus a difficulty axis; per-network NN-call counts and
  wall-clock alongside expansions (AdaSubS-style accounting; Burns et al. for why expansions alone
  mislead); and an explicit statement of what the realization check costs and why it is (or is not)
  excluded from the budget, kSubS-style. The backward planner's "2.9 expansions" headline *must* be
  accompanied by total accounting (realization physics steps + NN calls), or it is exactly the
  high-level-only number "What Matters" warns inflates subgoal methods.
- *Expectations to state up front* (citing "What Matters"): the subgoal advantage should appear where
  value functions are noisy, horizons long, action spaces large — i.e., grow along both of our axes —
  and may vanish at small scale on clean data; parity at 16×16/r4 with a widening gap at scale is the
  literature-consistent success shape, so Phase A parity (not victory) at the small scale is what the
  external evidence predicts. The robot axis carries the stronger theory (W[SAT]-hardness in robot
  count; the inflated-action-space experiment) while the grid axis has the kSubS board-size precedent;
  running both is well grounded.
- *Avoid*: single-budget-point comparisons; counting only subgoal-level expansions; quoting abstract
  plan cost against d* (the addendum already forbids this — "What Matters" §4.3.1 is the citable
  justification); and claiming novelty for reject-and-research itself (it is standard; see below).

**Novel vs standard, honestly stated.**
Standard (cite, don't claim): rejecting unrealizable subgoals and continuing search; per-edge
realization checks; learned feasibility verifiers; self-play/bootstrap labeling without an oracle;
budget-matched solve-rate curves. Genuinely less-precedented in our setup and worth emphasizing:
(1) *backward* (goal-to-start) subgoal chaining under sliding physics, where most learned-subgoal work
generates subgoals forward; (2) end-to-end strict executability *by difficulty band* as the headline
metric — kSubS reports subgoal-level reachability but not whole-plan legality curves; (3) an
oracle-free subgoal self-play loop evaluated in a PSPACE-complete tilt puzzle with a published
complexity/solver literature to anchor the depth-scaling story; (4) the explicit two-axis crossover
design (state-space axis vs action-space axis on one domain) with labeling cost reported as a
first-class outcome.
