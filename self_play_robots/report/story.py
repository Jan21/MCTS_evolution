"""The Story tab — a short, plain-English account of the self-play project.

Prose follows STE-flavored Simplified Technical English: short sentences, one
idea per sentence, active voice, simple tenses, no semicolons, vertical lists.
Depth lives in `self_play_robots/EXPLAINER.md` (every number source-cited);
the live tables in the other tabs are the authoritative numbers. Figures are
inline SVG checked by `report/svg_lint.py` (no overlapping labels, no
overflow, min 12 px text) — run by gen_report.py after every build.

Kept in its own module so edits here never touch the data-tab regions of
`gen_report.py`.
"""
from __future__ import annotations

STORY_CSS = """
.story p, .story ul.plain { max-width: 44rem; }
.story .lede { font-size: 1.02rem; }
.story h3 { margin-top: 2.6rem; font-size: 1.08rem; }
.story h3 .no { color: var(--mut); font-weight: 500; margin-right: .35rem; }
.story .toc { border-bottom: 1px solid var(--line); padding: .4rem 0;
              font-size: .82rem; color: var(--mut); }
.story .toc a { white-space: nowrap; margin-right: .7rem; }
.story .key { background: var(--card); border-left: 3px solid var(--acc);
              padding: .6rem .9rem; margin: 1.1rem 0; max-width: 42rem; }
.story .fig { margin: 1.4rem 0; }
.story .fig.big svg { max-width: 56rem; }
.story .fig figcaption { max-width: 56rem; }
/* story figures use 12px minimum text */
.story .ns { font: 12px system-ui, sans-serif; }
.story .cap { font: 12px system-ui, sans-serif; }
.story .lbl { font: 600 12px system-ui, sans-serif; }
.story .nb { fill: var(--ink); font: 12px system-ui, sans-serif; }
.story .arr.ink { stroke: var(--ink); }
.story .arr.acc { stroke: var(--acc); }
.story .ah.inkf { fill: var(--ink); }
.story .ah.accf { fill: var(--acc); }
.story .ah.mutf { fill: var(--mut); }
.story .bgrid { stroke: var(--line); stroke-width: 1; fill: none; }
.story .wallln { stroke: var(--ink); stroke-width: 4; }
.story .cellmark { fill: var(--hl); }
.story .floorln { stroke: var(--bad-ink); stroke-width: 1.6; stroke-dasharray: 6 4; }
.story .lead { stroke: var(--line); stroke-width: 1; }
.story .good-t { fill: var(--good-ink); font: 600 12px system-ui, sans-serif; }
.story .bad-t { fill: var(--bad-ink); font: 600 12px system-ui, sans-serif; }
.story .dot { stroke: var(--bg); stroke-width: 2; fill: var(--mut); }
.story .dot.acc { fill: var(--acc); }
.story .dot.hollow { fill: var(--bg); stroke: var(--ink); }
/* story tables: readable rows */
.story table.t { min-width: 0; width: auto; max-width: 100%; }
.story table.t th, .story table.t td { white-space: nowrap; padding: .38rem .7rem; }
.story table.t td:first-child, .story table.t th:first-child {
  white-space: normal; min-width: 11rem; }
.story tbody tr:nth-child(even) td { background: var(--card); }
.story td.g { color: var(--good-ink); font-weight: 600; }
.story ul.tree { list-style: none; padding-left: 0; margin: 1rem 0; }
.story ul.tree ul { list-style: none; padding-left: 1.3rem;
                    border-left: 2px solid var(--line);
                    margin: .25rem 0 .25rem .45rem; }
.story ul.tree li { margin: .6rem 0; padding-left: .2rem; }
.story ul.tree .what { font-weight: 600; }
.story ul.tree .why { color: var(--mut); font-size: .88rem; display: block;
                      max-width: 42rem; }
.story .chip.ctl { background: var(--ctl); color: var(--mut); }
.story dl.gloss { max-width: 44rem; }
.story dl.gloss dt { font-weight: 600; margin-top: .55rem; }
.story dl.gloss dd { margin: .1rem 0 0 0; color: var(--mut); font-size: .92rem; }
"""

_DEFS = (
    '<defs>'
    '<marker id="sa-ink" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7"'
    ' markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" class="ah inkf"/></marker>'
    '<marker id="sa-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7"'
    ' markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" class="ah accf"/></marker>'
    '<marker id="sa-mut" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7"'
    ' markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" class="ah mutf"/></marker>'
    '</defs>'
)


def _fig_slide() -> str:
    return f'''
<figure class="fig big">
<svg viewBox="0 0 600 150" role="img"
     aria-label="Two mini-boards. Left: no blocker, so the red robot slides past the target cell. Right: the blue robot is parked one cell beyond the target, so the red robot stops on the target.">
  {_DEFS}
  <g>
    <text x="121" y="24" text-anchor="middle" class="ns">Red slides past the target</text>
    <rect x="30" y="38" width="182" height="78" class="bgrid"/>
    <path d="M56 38V116 M82 38V116 M108 38V116 M134 38V116 M160 38V116 M186 38V116 M30 64H212 M30 90H212" class="bgrid"/>
    <line x1="212" y1="34" x2="212" y2="120" class="wallln"/>
    <rect x="108.5" y="64.5" width="25" height="25" class="cellmark"/>
    <text x="121" y="83" text-anchor="middle" class="nb">&#9678;</text>
    <circle cx="43" cy="77" r="8" fill="#c94436"/>
    <line x1="56" y1="77" x2="202" y2="77" class="arr ink" marker-end="url(#sa-ink)"/>
  </g>
  <g>
    <text x="461" y="24" text-anchor="middle" class="ns">park Blue first &mdash; Red stops</text>
    <rect x="370" y="38" width="182" height="78" class="bgrid"/>
    <path d="M396 38V116 M422 38V116 M448 38V116 M474 38V116 M500 38V116 M526 38V116 M370 64H552 M370 90H552" class="bgrid"/>
    <line x1="552" y1="34" x2="552" y2="120" class="wallln"/>
    <rect x="448.5" y="64.5" width="25" height="25" class="cellmark"/>
    <text x="461" y="83" text-anchor="middle" class="nb">&#9678;</text>
    <circle cx="487" cy="77" r="8" fill="#3a6fc9"/>
    <circle cx="383" cy="77" r="8" fill="#c94436"/>
    <line x1="396" y1="77" x2="468" y2="77" class="arr ink" marker-end="url(#sa-ink)"/>
  </g>
</svg>
<figcaption><strong>The core mechanic.</strong> Left: nothing stops Red, so
it slides past the target (&#9678;). Right: Blue is parked one cell past the
target, so Red stops on it. The other robots are the walls you
build.</figcaption>
</figure>'''


def _fig_subgoal() -> str:
    return '''
<figure class="fig big">
<svg viewBox="0 0 600 216" role="img"
     aria-label="A mini-board with one sub-goal. Arrow 1: send helper robot H to the support cell S. Arrow 2: the mover then stops at the bottleneck cell B, because H blocks it.">
  <rect x="40" y="30" width="182" height="130" class="bgrid"/>
  <path d="M66 30V160 M92 30V160 M118 30V160 M144 30V160 M170 30V160 M196 30V160 M40 56H222 M40 82H222 M40 108H222 M40 134H222" class="bgrid"/>
  <rect x="92.5" y="108.5" width="25" height="25" class="cellmark"/>
  <text x="105" y="127" text-anchor="middle" class="nt">B</text>
  <rect x="92.5" y="82.5" width="25" height="25" fill="none" stroke="var(--acc)" stroke-width="1.6" stroke-dasharray="4 3"/>
  <text x="105" y="100" text-anchor="middle" class="lbl">S</text>
  <circle cx="183" cy="69" r="8" fill="#3a6fc9"/>
  <text x="183" y="49" text-anchor="middle" class="ns">helper H</text>
  <circle cx="183" cy="147" r="8" fill="#c94436"/>
  <text x="183" y="176" text-anchor="middle" class="ns">mover</text>
  <path d="M171 69 C 140 62, 116 72, 107 82" class="arr acc dashed" marker-end="url(#sa-acc)"/>
  <path d="M171 147 C 145 150, 122 140, 110 126" class="arr ink" marker-end="url(#sa-ink)"/>
  <text x="280" y="70" class="lbl">1 &middot; send H to the support cell S</text>
  <text x="280" y="118" class="nb">2 &middot; the mover stops at B,</text>
  <text x="280" y="136" class="nb">because H blocks it there</text>
</svg>
<figcaption><strong>One sub-goal</strong> = a triple (B, S, H). The
bottleneck B: the cell where the mover needs something to crash into. The
support S: the cell the helper H must stand on to provide that crash. So
the sub-goal reads: send H to S, and the mover stops at B. The choice creates
two new journeys: mover to B, and H to S. That recursion is the search tree
&mdash; wide, but only 2&ndash;6 decisions deep. A full worked decision:
<code>EXPLAINER.md</code> &sect;2.3.</figcaption>
</figure>'''


def _fig_ceiling() -> str:
    """Number line, chapter 3: perfect / forward / the two walls / supervised."""
    ticks = "".join(
        f'<line x1="{60 + i * 131}" y1="115" x2="{60 + i * 131}" y2="125" class="axis"/>'
        f'<text x="{60 + i * 131}" y="142" text-anchor="middle" class="cap">'
        f'{"0" if i == 0 else f"+{i}"}</text>' for i in range(6))
    return f'''
<figure class="fig big">
<svg viewBox="0 0 760 178" role="img"
     aria-label="A number line of extra moves versus perfect play. Perfect play is at 0. The move-by-move planner is at +0.07. Two dashed walls mark the sub-goal language limits: +1.17 extended and +1.63 base. The solver-taught sub-goal planner is at +4.22.">
  <line x1="55" y1="120" x2="715" y2="120" class="axis"/>
  {ticks}
  <text x="385" y="166" text-anchor="middle" class="cap">extra moves per puzzle, compared with perfect play</text>
  <line x1="213.3" y1="40" x2="213.3" y2="120" class="floorln"/>
  <text x="213" y="30" text-anchor="middle" class="bad-t">wall +1.17 (extended)</text>
  <line x1="273.5" y1="68" x2="273.5" y2="120" class="floorln"/>
  <text x="298" y="58" text-anchor="middle" class="bad-t">wall +1.63 (base)</text>
  <line x1="286" y1="62" x2="276" y2="68" class="lead"/>
  <circle cx="60" cy="120" r="6" class="dot hollow"/>
  <text x="60" y="62" text-anchor="middle" class="ns">perfect play</text>
  <line x1="60" y1="66" x2="60" y2="112" class="lead"/>
  <circle cx="69.2" cy="120" r="6" class="dot"/>
  <text x="120" y="90" text-anchor="middle" class="ns">move-by-move +0.07</text>
  <line x1="72" y1="94" x2="70" y2="112" class="lead"/>
  <circle cx="612.8" cy="120" r="6" class="dot"/>
  <text x="612" y="90" text-anchor="middle" class="ns">solver-taught +4.22</text>
  <line x1="612" y1="94" x2="612" y2="112" class="lead"/>
</svg>
<figcaption><strong>The measured wall.</strong> As measured, no sub-goal
planner can average less than +1.63 extra moves (base vocabulary). The
extended vocabulary&rsquo;s floor is about +1.17 (extra tricks, such as
parking a robot aside first). Training cannot change this. It is a limit of
the vocabulary itself, measured to within a few hundredths of a move. (An
earlier, shallower probe gave +1.72 &mdash; see the
<a href="#ceiling">Ceiling tab</a>.)</figcaption>
</figure>'''


def _fig_break() -> str:
    """Number line, chapter 8: the flagship crosses the wall."""
    ticks = "".join(
        f'<line x1="{60 + i * 131}" y1="115" x2="{60 + i * 131}" y2="125" class="axis"/>'
        f'<text x="{60 + i * 131}" y="142" text-anchor="middle" class="cap">'
        f'{"0" if i == 0 else f"+{i}"}</text>' for i in range(6))
    return f'''
<figure class="fig big">
<svg viewBox="0 0 760 200" role="img"
     aria-label="The same number line. The flagship is at +0.94 extra moves, on the left side of the +1.17 wall. The same networks with the standard search are at +1.42, on the right side of the wall.">
  <line x1="55" y1="120" x2="715" y2="120" class="axis"/>
  {ticks}
  <text x="385" y="188" text-anchor="middle" class="cap">extra moves per puzzle, compared with perfect play (24&times;24 standard exam)</text>
  <line x1="213.3" y1="40" x2="213.3" y2="120" class="floorln"/>
  <text x="213" y="30" text-anchor="middle" class="bad-t">wall +1.17</text>
  <circle cx="60" cy="120" r="6" class="dot hollow"/>
  <text x="60" y="62" text-anchor="middle" class="ns">perfect</text>
  <line x1="60" y1="66" x2="60" y2="112" class="lead"/>
  <circle cx="69.2" cy="120" r="6" class="dot"/>
  <text x="120" y="90" text-anchor="middle" class="ns">move-by-move +0.07</text>
  <line x1="72" y1="94" x2="70" y2="112" class="lead"/>
  <circle cx="183.7" cy="120" r="8" class="dot acc"/>
  <text x="170" y="164" text-anchor="middle" class="lbl">flagship +0.94</text>
  <line x1="170" y1="151" x2="181" y2="129" class="lead"/>
  <circle cx="246.1" cy="120" r="6" class="dot"/>
  <text x="300" y="90" text-anchor="middle" class="ns">unchanged search +1.42</text>
  <line x1="266" y1="94" x2="248" y2="112" class="lead"/>
  <circle cx="612.8" cy="120" r="6" class="dot"/>
  <text x="612" y="90" text-anchor="middle" class="ns">solver-taught +4.22</text>
  <line x1="612" y1="94" x2="612" y2="112" class="lead"/>
</svg>
<figcaption><strong>The wall, crossed.</strong> Same networks, same budget.
The unchanged search stays right of the wall (+1.42). The hybrid is the only
planner left of it (+0.94, average over its own 231 solves; the unchanged
search: +1.42 over its 230). On the same puzzles, 40 solutions got shorter
and none got longer. (The solver-taught dot uses its recorded
file: +4.22. A re-run on this machine gives +4.20. Computers on different
machines round tiny numbers differently. When two plans score as equal, the
rounding decides which one the search meets first &mdash; see the M0
tab.)</figcaption>
</figure>'''


def _fig_loop() -> str:
    return '''
<figure class="fig big">
<svg viewBox="0 0 920 296" role="img"
     aria-label="The self-play loop. Fresh boards go to a noisy search. The physics replays every finished plan. Plans that work become training records. Plans that fail are discarded, so a wrong label cannot exist. The networks retrain, and the cycle repeats on new boards.">
  <rect x="14" y="40" width="180" height="64" rx="8" class="nbox alt"/>
  <text x="104" y="66" text-anchor="middle" class="nt">fresh random boards</text>
  <text x="104" y="86" text-anchor="middle" class="ns">new ones every round</text>
  <rect x="234" y="40" width="180" height="64" rx="8" class="nbox"/>
  <text x="324" y="66" text-anchor="middle" class="nt">search, with noise</text>
  <text x="324" y="86" text-anchor="middle" class="ns">two networks + MCTS</text>
  <rect x="454" y="40" width="200" height="64" rx="8" class="nbox alt"/>
  <text x="554" y="66" text-anchor="middle" class="nt">replay every plan</text>
  <text x="554" y="86" text-anchor="middle" class="ns">the physics simulator</text>
  <rect x="694" y="40" width="212" height="64" rx="8" class="nbox"/>
  <text x="800" y="66" text-anchor="middle" class="nt">training records</text>
  <text x="800" y="86" text-anchor="middle" class="ns">label = certified real cost</text>
  <rect x="694" y="200" width="212" height="58" rx="8" class="nbox"/>
  <text x="800" y="224" text-anchor="middle" class="nt">retrain the networks</text>
  <text x="800" y="243" text-anchor="middle" class="ns">warm start, 6 epochs</text>
  <rect x="234" y="200" width="220" height="58" rx="8" class="nbox alt"/>
  <text x="344" y="224" text-anchor="middle" class="nt">exams + paired tests</text>
  <text x="344" y="243" text-anchor="middle" class="ns">keep only real gains</text>
  <line x1="194" y1="72" x2="226" y2="72" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="414" y1="72" x2="446" y2="72" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="654" y1="62" x2="686" y2="62" class="arr ink" marker-end="url(#sa-ink)"/>
  <text x="670" y="48" text-anchor="middle" class="good-t">works</text>
  <line x1="554" y1="104" x2="554" y2="148" class="arr" marker-end="url(#sa-mut)"/>
  <text x="566" y="128" class="bad-t">fails &rarr; no record</text>
  <text x="566" y="148" class="cap">a wrong label cannot exist</text>
  <line x1="800" y1="104" x2="800" y2="192" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="686" y1="229" x2="462" y2="229" class="arr" marker-end="url(#sa-mut)"/>
  <path d="M694 210 C 470 140, 400 130, 340 108" class="arr acc dashed" marker-end="url(#sa-acc)"/>
  <text x="380" y="166" class="lbl">next round: better networks, new boards</text>
</svg>
<figcaption><strong>The loop&rsquo;s contract.</strong> A label = the answer a network trains toward. An illegal shortcut
fails the replay and leaves nothing. Only too-long plans can slip in. In the
diagram: MCTS is the tree search that AlphaZero uses. Noise = small random
nudges at the search&rsquo;s first choice, so rounds explore differently.
Warm start = training continues from the last weights. An epoch = one pass
over the training data. A paired test compares puzzle by puzzle, on the same
puzzles. One mixed-size round (all three board types together): ~940
puzzles, ~12,000 records, 5&ndash;6 GPU-hours.</figcaption>
</figure>'''


def _fig_encoder() -> str:
    return '''
<figure class="fig big">
<svg viewBox="0 0 900 262" role="img"
     aria-label="The shared encoder. The board becomes tokens, one per cell. The walls become two attention masks. One transformer block reads the tokens through three attention channels and runs twelve times with the same weights.">
  <rect x="14" y="46" width="200" height="78" rx="8" class="nbox alt"/>
  <text x="114" y="72" text-anchor="middle" class="nt">the board</text>
  <text x="114" y="92" text-anchor="middle" class="ns">one token per cell</text>
  <text x="114" y="110" text-anchor="middle" class="ns">+ 1 scratchpad token</text>
  <rect x="14" y="158" width="200" height="60" rx="8" class="nbox alt"/>
  <text x="114" y="182" text-anchor="middle" class="nt">walls &rarr; two masks</text>
  <text x="114" y="202" text-anchor="middle" class="ns">what can slide to what</text>
  <rect x="300" y="34" width="310" height="196" rx="10" class="nbox"/>
  <text x="455" y="58" text-anchor="middle" class="nt">one transformer block</text>
  <rect x="322" y="72" width="266" height="28" rx="5" class="nbox alt"/>
  <text x="455" y="91" text-anchor="middle" class="ns">attention: whole board</text>
  <rect x="322" y="106" width="266" height="28" rx="5" class="nbox alt"/>
  <text x="455" y="125" text-anchor="middle" class="ns">attention: legal slides only</text>
  <rect x="322" y="140" width="266" height="28" rx="5" class="nbox alt"/>
  <text x="455" y="159" text-anchor="middle" class="ns">attention: helper-free slides</text>
  <rect x="322" y="174" width="266" height="28" rx="5" class="nbox alt"/>
  <text x="455" y="193" text-anchor="middle" class="ns">+ a small mixing layer</text>
  <text x="455" y="220" text-anchor="middle" class="lbl">runs 12 times, same weights</text>
  <path d="M610 100 C 650 116, 650 160, 610 176" class="arr acc dashed" marker-end="url(#sa-acc)"/>
  <text x="656" y="144" class="lbl">&times;12</text>
  <line x1="214" y1="85" x2="292" y2="105" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="214" y1="188" x2="292" y2="160" class="arr dashed" marker-end="url(#sa-mut)"/>
  <text x="450" y="252" text-anchor="middle" class="cap">740,928 parameters per network &mdash; each network trains its own copy of the block</text>
</svg>
<figcaption><strong>The encoder</strong> (the part that turns the board
into numbers). A transformer block is a standard neural-network building
block that lets every cell look at every other cell. The board enters as
tokens: one short list of numbers per cell. Walls become attention masks &mdash; filters
that say which cells may exchange information &mdash; not input channels, so
information moves only along legal slides. There are no positional embeddings
(no cell-number table baked into the weights), so nothing depends on board
size. One saved network file runs on any board. Twelve passes of the same
block let information travel twelve slides. In the diagram: the scratchpad
token is one extra cell-less slot the network may use as working memory;
helper-free slides are moves that need no blocker; the mixing layer blends
each cell&rsquo;s information after the attention step. The block is 740,928 of each
network&rsquo;s parameters (trainable numbers). With its own head (output
part) on top, the value network totals 982,944 and the policy network
1,223,232.</figcaption>
</figure>'''


def _fig_heads() -> str:
    return '''
<figure class="fig big">
<svg viewBox="0 0 900 240" role="img"
     aria-label="The two heads. The encoder output goes to a value head and a policy head. The value head reads five marked cells and outputs a 96-bin cost distribution. The policy head asks three chained questions: which bottleneck, which support, which helper.">
  <rect x="14" y="86" width="220" height="64" rx="8" class="nbox alt"/>
  <text x="124" y="112" text-anchor="middle" class="nt">encoder output</text>
  <text x="124" y="132" text-anchor="middle" class="ns">one vector per cell</text>
  <rect x="330" y="16" width="392" height="92" rx="8" class="nbox"/>
  <text x="526" y="40" text-anchor="middle" class="nt">value network head</text>
  <text x="480" y="60" text-anchor="middle" class="ns">reads 5 marked cells</text>
  <text x="480" y="78" text-anchor="middle" class="ns">output: scores 0&ndash;95 plan-steps</text>
  <g>
    <rect x="646" y="76" width="10" height="12" fill="var(--mut)"/>
    <rect x="659" y="70" width="10" height="18" fill="var(--mut)"/>
    <rect x="672" y="58" width="10" height="30" fill="var(--acc)"/>
    <rect x="685" y="66" width="10" height="22" fill="var(--mut)"/>
    <rect x="698" y="76" width="10" height="12" fill="var(--mut)"/>
    <line x1="640" y1="88" x2="714" y2="88" class="axis"/>
  </g>
  <rect x="330" y="132" width="392" height="92" rx="8" class="nbox"/>
  <text x="526" y="156" text-anchor="middle" class="nt">policy network head</text>
  <rect x="348" y="170" width="108" height="34" rx="5" class="nbox alt"/>
  <text x="402" y="192" text-anchor="middle" class="ns">bottleneck?</text>
  <rect x="472" y="170" width="108" height="34" rx="5" class="nbox alt"/>
  <text x="526" y="192" text-anchor="middle" class="ns">support?</text>
  <rect x="596" y="170" width="108" height="34" rx="5" class="nbox alt"/>
  <text x="650" y="192" text-anchor="middle" class="ns">helper?</text>
  <line x1="456" y1="187" x2="464" y2="187" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="580" y1="187" x2="588" y2="187" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="234" y1="102" x2="322" y2="62" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="234" y1="134" x2="322" y2="174" class="arr ink" marker-end="url(#sa-ink)"/>
</svg>
<figcaption><strong>The two heads.</strong> The value head reads five
marked cells &mdash; the mover, the goal, and the candidate&rsquo;s three
cells (B, S, H) &mdash; and answers: how many steps will this plan still
cost? Output: a spread over 96 possible costs (&ldquo;probably 6, maybe
9&rdquo;). The policy head answers: which sub-goal next? It points at real
board cells, so it works with any candidate count.</figcaption>
</figure>'''


def _fig_expansion() -> str:
    return '''
<figure class="fig big">
<svg viewBox="0 0 940 140" role="img"
     aria-label="One expansion. A partial plan goes to the generator, which lists all legal sub-goals. One policy call keeps the best five. One value call prices those five. Five priced child plans result.">
  <rect x="14" y="42" width="150" height="60" rx="8" class="nbox alt"/>
  <text x="89" y="68" text-anchor="middle" class="nt">a partial plan</text>
  <text x="89" y="88" text-anchor="middle" class="ns">next open task</text>
  <rect x="200" y="42" width="176" height="60" rx="8" class="nbox alt"/>
  <text x="288" y="68" text-anchor="middle" class="nt">generator</text>
  <text x="288" y="88" text-anchor="middle" class="ns">all legal sub-goals</text>
  <rect x="412" y="42" width="176" height="60" rx="8" class="nbox"/>
  <text x="500" y="68" text-anchor="middle" class="nt">policy net ranks</text>
  <text x="500" y="88" text-anchor="middle" class="ns">keeps the best 5</text>
  <rect x="624" y="42" width="176" height="60" rx="8" class="nbox"/>
  <text x="712" y="68" text-anchor="middle" class="nt">value net estimates</text>
  <text x="712" y="88" text-anchor="middle" class="ns">one grouped call</text>
  <rect x="836" y="42" width="90" height="60" rx="8" class="nbox alt"/>
  <text x="881" y="68" text-anchor="middle" class="nt">5 child</text>
  <text x="881" y="88" text-anchor="middle" class="ns">plans</text>
  <line x1="164" y1="72" x2="192" y2="72" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="376" y1="72" x2="404" y2="72" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="588" y1="72" x2="616" y2="72" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="800" y1="72" x2="828" y2="72" class="arr ink" marker-end="url(#sa-ink)"/>
</svg>
<figcaption><strong>One expansion</strong> = one policy call plus one value
call that scores all five children together. The exam budget is 1,200
expansions per puzzle (2,400 network calls). Physics checks do not count
against the search budget. The budget counts network calls only &mdash; the
same rule the benchmark uses. A failed replay removes the plan from the
tree.</figcaption>
</figure>'''


def _journey_tree() -> str:
    return '''
<ul class="tree">
  <li><span class="what">Start: solver-taught planners + one small size-free value net</span>
    <ul>
      <li><span class="chip good">&#10003; kept</span> <span class="what">Rebuild both networks size-free</span>
          <span class="why">One pair beat the specialist 24&times;24 planner: 215 vs 205 of 232, with 5&times; less search.</span>
        <ul>
          <li><span class="chip bad">&#10007; flat</span> <span class="what">Self-play, base vocabulary</span>
              <span class="why">Two rounds changed nothing, as the wall predicted. A clean negative result.</span>
            <ul>
              <li><span class="chip good">&#10003; step</span> <span class="what">Five rounds, extended vocabulary</span>
                  <span class="why">Hard-exam solves went 133 &rarr; 144 &rarr; 139 &rarr; 158 &rarr; 153 &rarr; 153. The loop learned speed instead. By round 4 the quick A* search needed only 25.9 expansions per puzzle (A*: a standard shortest-path search). That solved what round 0&rsquo;s full tree search (~500 expansions) had reached. Round 5 held it (30.3 expansions). Moves did not improve.</span>
                <ul>
                  <li><span class="chip good">&#10003; jump</span> <span class="what">Mixed-size training: 24&times;24 + 32&times;32 + 8 robots</span>
                      <span class="why">One round ended the stall. Hard-exam solves: 170, 235 and 269 &mdash; of 218, 275 and 289 puzzles, one hard exam per board type. Later rounds gave a little back (165, 225, 268).</span>
                    <ul>
                      <li><span class="what">The experiment lab: 15 cards, most changing one thing</span>
                          <span class="why">One control (the unchanged recipe), three never run, eleven tested ideas. Two cards stack proven winners. Same start networks, same budget, a 200-puzzle new-boards exam. Every claim needs a second seed &mdash; a rerun with different random starting conditions.</span>
                        <ul>
                          <li><span class="chip bad">&#10007; killed</span> <span class="what">v01 &middot; AlphaZero&rsquo;s own policy target</span>
                              <span class="why">v01 replaced our scoring rule (score a candidate by the verified cost of the plans that used it) with AlphaZero&rsquo;s (prefer what the search visited most). Training collapsed on every exam (fluke chance below one in a million). Our rule is what makes the training work.</span></li>
                          <li><span class="chip good">&#10003; adopted</span> <span class="what">v04 &middot; keep every examined decision</span>
                              <span class="why">Twice the records from the same search. A replicated win on the hard exam.</span></li>
                          <li><span class="chip warn">~ did not repeat</span> <span class="what">v06 &middot; a different way to randomise the search&rsquo;s first choices</span>
                              <span class="why">Won at seed 7, reversed at seed 8. The two-seed rule caught it.</span></li>
                          <li><span class="chip ctl">from-scratch run</span> <span class="what">v08 &middot; cold start from random weights</span>
                              <span class="why">One round with no human or solver answers &mdash; its only labels were its own replay-checked plans: 132 of 200 new-board puzzles. The solver-taught planner: 134. A tie, from nothing (the puzzle-by-puzzle test finds no real difference).</span></li>
                          <li><span class="chip good">&#10003; adopted</span> <span class="what">v09 &middot; predict real move counts</span>
                              <span class="why">The value net now predicts real move counts &mdash; the number the project is graded on &mdash; not an internal plan-step count. Better everywhere, both seeds.</span></li>
                          <li><span class="chip bad">&#10007; killed</span> <span class="what">v12 &middot; train only on hard puzzles</span>
                              <span class="why">Flat after one round. Flat after three.</span></li>
                          <li><span class="chip good">&#10003; adopted</span> <span class="what">v14 &middot; v04 + v09 together</span>
                              <span class="why">Wins the new-boards exam at both seeds. The lab&rsquo;s recipe.</span></li>
                          <li><span class="chip bad">&#10007; killed twice</span> <span class="what">v15, v16 &middot; train on puzzles whose start was shifted by one ordinary move (what the hybrid search sees)</span>
                              <span class="why">Flat, both seeds, twice. The networks already handle post-slide positions.</span></li>
                          <li><span class="chip good">&#10003; the break</span> <span class="what">v07 &middot; the hybrid search</span>
                              <span class="why">Try one or two ordinary moves first, then plan. The only change that went below the wall. No retraining needed.</span></li>
                        </ul>
                      </li>
                      <li><span class="chip good">&#10003; flagship</span> <span class="what">= the v09 networks + the hybrid search, two ordinary moves allowed first</span>
                          <span class="why">Chosen by rule: every part survived its matched control (an identical run without that one change) and its second seed.</span></li>
                    </ul>
                  </li>
                </ul>
              </li>
            </ul>
          </li>
        </ul>
      </li>
    </ul>
  </li>
</ul>'''


def sec_story() -> str:
    out = ['<section id="story" class="story">',
           "<h2>The story &mdash; start here</h2>"]
    out.append(
        '<nav class="toc" aria-label="story chapters">'
        '<a href="#st-game">1 game</a>'
        '<a href="#st-language">2 sub-goals</a>'
        '<a href="#st-ceiling">3 the wall</a>'
        '<a href="#st-referee">4 the referee</a>'
        '<a href="#st-nets">5 networks</a>'
        '<a href="#st-search">6 search</a>'
        '<a href="#st-journey">7 journey</a>'
        '<a href="#st-scoreboard">8 scoreboard</a>'
        '<a href="#st-failed">9 failures</a>'
        '<a href="#st-limits">10 limits</a>'
        '</nav>')
    out.append(
        '<p class="lede">Two planners came before this project: the '
        '<em>solver-taught</em> planner (called &ldquo;backward '
        'supervised&rdquo; in result files), trained on an exact '
        'solver&rsquo;s answers, and a slower <em>move-by-move</em> one that '
        'is nearly perfect when it solves. Then a small pair of neural '
        'networks taught itself to solve Ricochet Robots puzzles, starting '
        'from networks copied from the solver-taught planner. No human '
        'answers were used during self-play. (A separate control that '
        'started from random weights still matched the solver-taught '
        'baseline &mdash; see &sect;7.) On 200 '
        'brand-new puzzles it solves 188. The solver-taught planner '
        'solves 134. We also measured a hard quality limit of the '
        'planner&rsquo;s vocabulary. A small search change then broke that '
        'limit: the planner may stop being a pure sub-goal planner and try '
        'an ordinary move first. '
        '<code>EXPLAINER.md</code> cites a source for every number here.</p>')

    # 1
    out.append('<h3 id="st-game"><span class="no">1</span>A game where you build your own walls</h3>')
    out.append(
        '<p>Ricochet Robots is a puzzle on a square grid. A move sends one '
        'robot in one direction. It slides until a wall or another robot '
        'stops it. The goal: stop one target robot on one target cell, in '
        'few moves.</p>')
    out.append(_fig_slide())
    out.append(
        '<p>Solutions are construction projects, up to ~30 moves. An exact '
        'solver exists, but it fails on many puzzles. Its reach defines two '
        'of the three exams: the puzzles it graded form the standard exam, '
        'and its failures form the hard exam. The third uses fresh '
        'boards:</p>')
    out.append('<div class="tw"><table class="t"><thead><tr>'
               '<th>exam</th><th>puzzles</th><th>meaning</th></tr></thead><tbody>'
        '<tr><td>standard (also called graded)</td><td>232</td><td>optimum known (perfect play averages 7.69 moves over all 232)</td></tr>'
        '<tr><td>hard (frontier)</td><td>218</td><td>the exact solver could not crack these when the exam was frozen. An optimum exists but is not known. Planners have since solved many. (24&times;24 set; other board types have their own)</td></tr>'
               '<tr><td>new boards (unseen)</td><td>200</td><td>fresh boards, never trained on (137 have a known optimum)</td></tr>'
               '</tbody></table></div>')
    out.append(
        '<p>One rule is absolute: a puzzle counts as solved only when its '
        'plan replays, move by move, in the game physics.</p>')

    # 2
    out.append('<h3 id="st-language"><span class="no">2</span>The planner thinks in sub-goals</h3>')
    out.append(
        '<p>The planner does not search move by move. It reasons backwards, '
        'in sub-goals:</p>')
    out.append(_fig_subgoal())

    # 3
    out.append('<h3 id="st-ceiling"><span class="no">3</span>First, measure the wall</h3>')
    out.append(
        '<p>First, measure the limit. How good could a perfect sub-goal '
        'planner be? Enumerate every expressible plan. Replay each one. '
        'Compare the best with perfect play.</p>')
    out.append(_fig_ceiling())
    out.append(
        '<p class="key">This measurement shaped everything after it. It '
        'predicted that base-vocabulary self-play would be useless. It was. '
        'And it set the breakthrough target: go below +1.17. The wall is a '
        'measured floor, not a mathematical proof. Deeper probes moved it '
        'only by hundredths of a move (see the '
        '<a href="#ceiling">Ceiling tab</a>).</p>')

    # 4
    out.append('<h3 id="st-referee"><span class="no">4</span>Self-play with a physics referee</h3>')
    out.append(
        '<p>AlphaZero learns from games against itself. This puzzle has no '
        'opponent and no win or loss. Physics is the referee: only plans that '
        'replay correctly become '
        'training data. Fresh random boards every round give the variety.</p>')
    out.append(_fig_loop())

    # 5
    out.append('<h3 id="st-nets"><span class="no">5</span>Two small networks, any board size</h3>')
    out.append(
        '<p>The planner is two networks with one shared encoder design: '
        '2,206,176 parameters in total.</p>')
    out.append(_fig_encoder())
    out.append(_fig_heads())
    out.append('<div class="tw"><table class="t"><thead><tr>'
               '<th>network</th><th>input</th><th>output</th><th>parameters</th></tr></thead><tbody>'
               '<tr><td>value</td><td>board + one candidate sub-goal</td><td>cost still needed (scores 0&ndash;95 plan-steps &mdash; enough for the largest boards)</td><td>982,944</td></tr>'
               '<tr><td>policy</td><td>board, no candidate shown</td><td>which sub-goal to try</td><td>1,223,232</td></tr>'
               '</tbody></table></div>')
    out.append(
        '<p>The first networks trained at 16&times;16 and 24&times;24 (the '
        'loop later added 32&times;32 and 8-robot boards). The starting '
        'value network was audited at sizes 32, 40, 48, 56 and 64. The '
        'measure: how often its top-ranked candidate is truly optimal on '
        'solver-graded decisions. It beats its teacher at every size. (The '
        'teacher = the solver-taught labelling network it was copied from '
        '&mdash; the network that wrote its training answers. The teacher '
        'was re-measured with the same tool at 56 and 64.) Details: the '
        'far-size audit table in the <a href="#res-audit">Milestone '
        'results tab</a>. Later 24&times;24-only practice trades some of '
        'this away &mdash; see that table&rsquo;s note.</p>')

    # 6
    out.append('<h3 id="st-search"><span class="no">6</span>One search step</h3>')
    out.append(_fig_expansion())

    # 7
    out.append('<h3 id="st-journey"><span class="no">7</span>The journey</h3>')
    out.append(
        '<p>Green = adopted. Red = killed. Amber = did not replicate.</p>')
    out.append(_journey_tree())
    out.append(
        '<p class="key">One pattern repeats: a big step when something '
        'structural changes, then flat. And training never bought shorter '
        'solutions. Search changes did.</p>')

    # 8
    out.append('<h3 id="st-scoreboard"><span class="no">8</span>The scoreboard</h3>')
    out.append(
        '<p>Every row: same budget, every solve replay-certified.</p>')
    out.append(_fig_break())
    out.append(
        '<p>Allowing a third ordinary move first improves the flagship&rsquo;s '
        '+0.94 to +0.86 on the standard exam (each over its own solves) '
        '&mdash; measured once, not '
        'adopted.</p>'
        '<p><strong>The new-boards exam (unseen).</strong> 200 puzzles on fresh boards. '
        'Perfect play needs 8.71 moves where the optimum is known:</p>')
    out.append('<div class="tw"><table class="t"><thead><tr>'
               '<th>planner</th><th>solved of 200</th><th>solved perfectly*</th>'
               '<th>extra moves vs perfect*</th></tr></thead><tbody>'
               '<tr class="hl"><td>flagship (self-taught + hybrid search)</td><td>188</td><td>57%</td><td class="g">+1.47</td></tr>'
               '<tr><td>same networks, unchanged search<br><span class="note">the '
               'same-budget control from the hybrid experiment (Variants '
               'v07)</span></td><td>182</td><td>50%</td><td>+1.94</td></tr>'
               '<tr><td>the frozen start networks</td><td>175</td><td>47%</td><td>+2.93</td></tr>'
               '<tr><td>label-free, from random weights</td><td>132</td><td>45%</td><td>+2.89</td></tr>'
               '<tr><td>solver-taught backward (the baseline)</td><td>134</td><td>36%</td><td>+4.84</td></tr>'
               '<tr><td>move-by-move (solver-taught)</td><td>101</td><td>90%</td><td>+0.12</td></tr>'
               '</tbody></table>'
               '<p class="note">* both starred columns count only the 137 puzzles '
               'with a known optimum, restricted to the ones that system '
               'solved. The Variants tab shows +0.90 for the same planner: '
               'that counts only the 93 puzzles every system solved &mdash; a '
               'smaller, easier set. Both are true.</p></div>')
    out.append(
        '<p>The move-by-move planner is almost perfect when it solves. But '
        'it solves barely half the exam. And it spends 688 expansions per '
        'puzzle &mdash; 23 times the solver-taught baseline&rsquo;s 30. '
        '(The flagship spends 798: quality costs search wherever it comes '
        'from.)</p>')
    out.append(
        '<p>The original goal had two halves. The first &mdash; at least as '
        'many solves as the solver-taught baseline, with fewer moves &mdash; '
        'is met. The second &mdash; matching the move-by-move '
        'planner&rsquo;s solution quality &mdash; is not. The hybrid halved '
        'the gap.</p>')
    out.append(
        '<p><strong>Transfer.</strong> The hybrid search itself was developed '
        'at 24&times;24 with 4 robots only. Run unchanged elsewhere:</p>')
    out.append('<div class="tw"><table class="t"><thead><tr>'
               '<th>new exam</th><th>hybrid: solved</th>'
               '<th>control: solved</th>'
               '<th>moves on shared solves (hybrid vs control)**</th>'
               '<th>move wins (shorter / longer)**</th></tr></thead><tbody>'
               '<tr><td>32&times;32 standard exam</td><td class="g">174/175</td>'
               '<td>172/175</td><td>9.27 vs 9.66</td><td>17 / 0</td></tr>'
               '<tr><td>8 robots, standard exam</td><td class="g">159/161</td>'
               '<td>159/161</td><td>6.85 vs 7.43</td><td>23 / 0</td></tr>'
               '<tr><td>8 robots, hard exam</td><td class="g">276/289</td>'
               '<td>279/289</td><td>14.25 vs 15.30</td><td>82 / 2</td></tr>'
               '</tbody></table>'
               '<p class="note">** counted on the puzzles both runs solved '
               '(172, 159 and 274 puzzles). On the 8-robot hard exam the '
               'control solves three more puzzles (279 vs 276) but uses more '
               'moves where both solve. The solver-taught planner there: 161 '
               'of 289 with the extended vocabulary, 154 of 289 with the '
               'base vocabulary.</p></div>')

    # 9
    out.append('<h3 id="st-failed"><span class="no">9</span>What failed, and why it mattered</h3>')
    out.append(
        '<ul class="plain">'
        '<li><strong>The textbook recipe lost.</strong> AlphaZero&rsquo;s '
        'visit-count target collapsed every exam. Scoring candidates by the '
        'certified cost of their finished plans is what makes training work.</li>'
        '<li><strong>The base vocabulary was already full.</strong> Two '
        'rounds changed nothing, as predicted.</li>'
        '<li><strong>Slide training failed twice.</strong> The search change '
        'did the work. The training change did not.</li>'
        '<li><strong>One win vanished on a second seed.</strong> Every '
        'adopted result survived that test.</li>'
        '</ul>')

    # 10
    out.append('<h3 id="st-limits"><span class="no">10</span>Limits, and what is next</h3>')
    out.append(
        '<ul class="plain">'
        '<li>Self-play training bought solve rate and speed, never shorter '
        'solutions.</li>'
        '<li>On the standard exam the flagship uses +0.79 more moves than '
        'the move-by-move planner, on the 219 puzzles both solved. (That is '
        'the paired number in <a href="#baselines">Baselines</a>.) On new boards that '
        'planner stays near-perfect (+0.12 on its own solves) but solves barely half.</li>'
        
        '</ul>')
    out.append(
        '<p><strong>Next:</strong> the same loop at 80&times;80 and '
        '96&times;96. No exact solver exists there. Physics certification '
        'is the only ground truth. The size-free networks run there '
        'unchanged.</p>')
    out.append("</section>")
    return "\n".join(out)
