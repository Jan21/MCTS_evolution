"""The Story tab — a plain-English, diagrammed account of the whole self-play
project, rendered as the FIRST panel of selfplay.html.

Unlike the data tabs (which regenerate every number from result files at page
build time), this tab is a hand-written narrative. Its numbers were verified
against the result files when `self_play_robots/EXPLAINER.md` was written — the
in-depth companion that cites a source file for every figure — and the live,
authoritative versions of every table live in the other tabs of this page. The
figures (inline SVG) were drawn from the code itself: the architecture panel
matches `nn_labeler/model.py::LoopedLayer`/`SizeFreeValueNet` and
`spr/nets.py::SizeFreePolicyNet`; the loop panel matches `spr/selfplay.py` and
`jobs/selfplay_mix_iter.slurm`; the expansion panel matches
`spr/search.py::expand`.

Kept in its own module so edits here never touch the data-tab regions of
`gen_report.py`.
"""
from __future__ import annotations

# --- CSS additions used only by this tab (appended to the page CSS) ----------
STORY_CSS = """
.story p { max-width: 46rem; }
.story .lede { font-size: 1.02rem; max-width: 46rem; }
.story h3 { margin-top: 2rem; font-size: 1.05rem; }
.story h3 .no { color: var(--mut); font-weight: 500; margin-right: .3rem; }
.story .toc { color: var(--mut); font-size: .85rem; max-width: 46rem; }
.story .toc a { white-space: nowrap; }
.story .key { background: var(--card); border-left: 3px solid var(--acc);
              padding: .6rem .9rem; margin: 1rem 0; max-width: 44rem; }
.story .fig.big svg { max-width: 56rem; }
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
.story .good-t { fill: var(--good-ink); font: 600 11px system-ui, sans-serif; }
.story .bad-t { fill: var(--bad-ink); font: 600 11px system-ui, sans-serif; }
.story .dot { stroke: var(--bg); stroke-width: 2; fill: var(--mut); }
.story .dot.acc { fill: var(--acc); }
.story .dot.hollow { fill: var(--bg); stroke: var(--ink); }
.story table.t { min-width: 0; width: 100%; }
.story td.g { color: var(--good-ink); font-weight: 600; }
.story ul.tree { list-style: none; padding-left: 0; margin: 1rem 0; }
.story ul.tree ul { list-style: none; padding-left: 1.3rem;
                    border-left: 2px solid var(--line); margin: .25rem 0 .25rem .45rem; }
.story ul.tree li { margin: .55rem 0; padding-left: .2rem; }
.story ul.tree .what { font-weight: 600; }
.story ul.tree .why { color: var(--mut); font-size: .88rem; display: block;
                      max-width: 44rem; }
.story .chip.ctl { background: var(--ctl); color: var(--mut); }
.story dl.gloss dt { font-weight: 600; margin-top: .55rem; }
.story dl.gloss dd { margin: .1rem 0 0 0; color: var(--mut); font-size: .92rem; }
"""

# The markers are defined once, in the first SVG of this panel, and referenced
# by fragment id from the panel's other SVGs (same document, same visibility
# subtree, so hidden-panel rendering quirks cannot bite).
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
<svg viewBox="0 0 560 168" role="img"
     aria-label="Two mini-boards. Left: with no blocker, the red robot slides past the target cell to the far wall. Right: after parking the blue robot one cell beyond the target, the red robot stops exactly on it.">
  {_DEFS}
  <g>
    <rect x="20" y="34" width="182" height="78" class="bgrid"/>
    <path d="M46 34V112 M72 34V112 M98 34V112 M124 34V112 M150 34V112 M176 34V112 M20 60H202 M20 86H202" class="bgrid"/>
    <line x1="202" y1="30" x2="202" y2="116" class="wallln"/>
    <rect x="98.5" y="60.5" width="25" height="25" class="cellmark"/>
    <text x="111" y="79" text-anchor="middle" class="nb">&#9678;</text>
    <circle cx="33" cy="73" r="8" fill="#c94436"/>
    <line x1="46" y1="73" x2="192" y2="73" class="arr ink" marker-end="url(#sa-ink)"/>
    <text x="111" y="26" text-anchor="middle" class="ns">nothing to stop it &mdash; Red slides past the target</text>
    <text x="111" y="132" text-anchor="middle" class="cap">a robot cannot stop halfway</text>
  </g>
  <g>
    <rect x="330" y="34" width="182" height="78" class="bgrid"/>
    <path d="M356 34V112 M382 34V112 M408 34V112 M434 34V112 M460 34V112 M486 34V112 M330 60H512 M330 86H512" class="bgrid"/>
    <line x1="512" y1="30" x2="512" y2="116" class="wallln"/>
    <rect x="408.5" y="60.5" width="25" height="25" class="cellmark"/>
    <text x="421" y="79" text-anchor="middle" class="nb">&#9678;</text>
    <circle cx="447" cy="73" r="8" fill="#3a6fc9"/>
    <circle cx="343" cy="73" r="8" fill="#c94436"/>
    <line x1="356" y1="73" x2="428" y2="73" class="arr ink" marker-end="url(#sa-ink)"/>
    <text x="421" y="26" text-anchor="middle" class="ns">park Blue one cell beyond &mdash; now Red stops on the target</text>
    <text x="421" y="132" text-anchor="middle" class="cap">the other robots are the walls you build</text>
  </g>
</svg>
<figcaption><strong>The core mechanic.</strong> You rarely <em>travel</em> to the
goal; you <em>construct</em> the thing that stops you there &mdash; often spending
moves to park a helper robot on an empty cell purely as a future wall.</figcaption>
</figure>'''


def _fig_subgoal() -> str:
    return '''
<figure class="fig big">
<svg viewBox="0 0 560 240" role="img"
     aria-label="A mini-board showing one sub-goal: send helper robot Blue to the support cell S so that the moving robot Red stops at the bottleneck cell B, from which it can finish.">
  <rect x="60" y="30" width="182" height="130" class="bgrid"/>
  <path d="M86 30V160 M112 30V160 M138 30V160 M164 30V160 M190 30V160 M216 30V160 M60 56H242 M60 82H242 M60 108H242 M60 134H242" class="bgrid"/>
  <rect x="112.5" y="108.5" width="25" height="25" class="cellmark"/>
  <text x="125" y="127" text-anchor="middle" class="nt">B</text>
  <rect x="112.5" y="82.5" width="25" height="25" fill="none" stroke="var(--acc)" stroke-width="1.6" stroke-dasharray="4 3"/>
  <text x="125" y="100" text-anchor="middle" class="lbl">S</text>
  <circle cx="203" cy="69" r="8" fill="#3a6fc9"/>
  <text x="203" y="53" text-anchor="middle" class="ns">helper H</text>
  <circle cx="203" cy="147" r="8" fill="#c94436"/>
  <text x="203" y="176" text-anchor="middle" class="ns">mover</text>
  <path d="M191 69 C 160 62, 136 72, 127 82" class="arr acc dashed" marker-end="url(#sa-acc)"/>
  <path d="M191 147 C 165 150, 142 140, 130 126" class="arr ink" marker-end="url(#sa-ink)"/>
  <text x="272" y="60" class="lbl">1 &middot; send helper H to the support cell S</text>
  <text x="272" y="76" class="cap">(how H gets there is a NEW segment &mdash; the recursion)</text>
  <text x="272" y="112" class="nt">2 &middot; the mover arrives and stops at</text>
  <text x="272" y="128" class="nt">bottleneck B &mdash; blocked by H on S</text>
  <text x="272" y="144" class="cap">(without H there, it would slide past)</text>
  <text x="272" y="182" class="cap">one decision = choosing (B, S, H) from a hand-written</text>
  <text x="272" y="197" class="cap">generator&rsquo;s list of legal triples; a plan is done</text>
  <text x="272" y="212" class="cap">when no segment is open</text>
</svg>
<figcaption><strong>One sub-goal decision:</strong> &ldquo;send helper
<strong>H</strong> to support cell <strong>S</strong>, so the mover stops at
bottleneck <strong>B</strong>, from which it can finish.&rdquo; Picking it spawns
two new segments (mover&rarr;B, H&rarr;S) &mdash; that recursion is the tree the
search explores: wide (up to ~50 choices) but only 2&ndash;6 decisions deep.
A real 12-candidate decision is walked step by step in
<code>EXPLAINER.md</code> &sect;2.3.</figcaption>
</figure>'''


def _fig_ceiling(with_flagship: bool) -> str:
    """The extra-moves number line; drawn twice — before and after the break."""
    # x = 60 + v * 131  maps 0..5 extra moves onto 60..715
    if not with_flagship:
        extra = '''
  <circle cx="60" cy="96" r="6" class="dot hollow"/>
  <text x="60" y="76" text-anchor="middle" class="ns">perfect play</text>
  <circle cx="69.2" cy="96" r="6" class="dot"/>
  <text x="97" y="63" text-anchor="middle" class="ns">move-by-move</text>
  <text x="97" y="76" text-anchor="middle" class="ns">planner +0.07</text>
  <circle cx="610.2" cy="96" r="6" class="dot"/>
  <text x="610" y="63" text-anchor="middle" class="ns">supervised sub-goal</text>
  <text x="610" y="76" text-anchor="middle" class="ns">planner +4.20</text>
  <line x1="273.5" y1="52" x2="273.5" y2="96" class="floorln"/>
  <text x="290" y="46" text-anchor="middle" class="bad-t">+1.63 &mdash; base floor</text>
  <line x1="213.3" y1="34" x2="213.3" y2="96" class="floorln"/>
  <text x="213" y="26" text-anchor="middle" class="bad-t">+1.17 &mdash; extended-language floor</text>'''
        h, axis_y = 168, 96
        aria = ("A number line of extra moves versus perfect play. Perfect play "
                "at 0, the move-by-move planner at +0.07, two dashed walls at "
                "+1.17 and +1.63 (the sub-goal language floors), and the "
                "supervised sub-goal planner far right at +4.20.")
    else:
        extra = '''
  <circle cx="60" cy="118" r="6" class="dot hollow"/>
  <text x="60" y="98" text-anchor="middle" class="ns">perfect</text>
  <circle cx="69.2" cy="118" r="6" class="dot"/>
  <text x="83" y="81" text-anchor="middle" class="ns">move-by-move +0.07</text>
  <text x="83" y="95" text-anchor="middle" class="cap">(but solves far fewer)</text>
  <circle cx="183.7" cy="118" r="8" class="dot acc"/>
  <text x="168" y="63" text-anchor="middle" class="lbl">flagship +0.944</text>
  <text x="168" y="78" text-anchor="middle" class="cap">left of the wall</text>
  <circle cx="246.1" cy="118" r="6" class="dot"/>
  <text x="292" y="81" text-anchor="middle" class="ns">same networks, standard</text>
  <text x="292" y="95" text-anchor="middle" class="ns">search, same budget +1.42</text>
  <circle cx="610.2" cy="118" r="6" class="dot"/>
  <text x="610" y="98" text-anchor="middle" class="ns">supervised +4.20</text>
  <line x1="213.3" y1="44" x2="213.3" y2="118" class="floorln"/>
  <text x="228" y="38" text-anchor="middle" class="bad-t">the +1.17 wall</text>'''
        h, axis_y = 190, 118
        aria = ("The same number line, now with the flagship at +0.944 extra "
                "moves, left of the dashed +1.17 language floor, and the "
                "matched same-networks standard search at +1.42, right of it.")
    ticks = "".join(
        f'<line x1="{60 + i * 131}" y1="{axis_y - 5}" x2="{60 + i * 131}" '
        f'y2="{axis_y + 5}" class="axis"/>' for i in range(6))
    lbls = "".join(
        f'<text x="{60 + i * 131}" y="{axis_y + 20}" text-anchor="middle" '
        f'class="cap">{"0" if i == 0 else f"+{i}"}</text>' for i in range(6))
    return f'''
<figure class="fig big">
<svg viewBox="0 0 760 {h}" role="img" aria-label="{aria}">
  <line x1="55" y1="{axis_y}" x2="715" y2="{axis_y}" class="axis"/>
  {ticks}{lbls}
  <text x="385" y="{axis_y + 44}" text-anchor="middle" class="cap">extra moves per puzzle vs perfect play (24&times;24 graded exam) &mdash; each system&rsquo;s mean over its own solved set</text>
  {extra}
</svg>
<figcaption>{"<strong>The measured wall.</strong> No planner speaking the sub-goal language &mdash; however trained, however long it searches &mdash; can average less than <strong>+1.63</strong> extra moves (base vocabulary) or <strong>+1.17</strong> (extended &ldquo;B2&rdquo; vocabulary: transient blockers, reusing robots already in the plan, shoving a nuisance robot aside first). The optimal move sequence sometimes does things a sub-goal simply cannot say. A <em>language limit, not a learning limit</em> &mdash; and the move-by-move planner&rsquo;s +0.07 shows how much is left on the table." if not with_flagship else "<strong>The breakthrough, drawn.</strong> With the same networks and the same budget, the standard search stays right of the wall (+1.42); the hybrid is the only planner to its left (+0.944, 231 of 232 solved, 68% solved perfectly; depth&nbsp;3 pushes to +0.861). Paired on identical puzzles: 40/0 move wins &mdash; fluke odds below one in a trillion. The wall applies to <em>pure</em> sub-goal planners; the hybrid steps outside the language, which is precisely the point."}</figcaption>
</figure>'''


def _fig_loop() -> str:
    return '''
<figure class="fig big">
<svg viewBox="0 0 880 268" role="img"
     aria-label="The self-play loop: fresh random boards feed a noisy tree search; every finished plan is replayed in the physics; certified plans become training records; failed plans are discarded so a wrong label cannot exist; the networks retrain warm-started and the cycle repeats on new boards; pinned exams with paired statistics gate each iteration.">
  <rect x="14" y="42" width="150" height="58" rx="8" class="nbox alt"/>
  <text x="89" y="66" text-anchor="middle" class="nt">fresh random boards</text>
  <text x="89" y="83" text-anchor="middle" class="ns">brand-new every round</text>
  <rect x="204" y="42" width="150" height="58" rx="8" class="nbox"/>
  <text x="279" y="66" text-anchor="middle" class="nt">search, with noise</text>
  <text x="279" y="83" text-anchor="middle" class="ns">the two networks + MCTS</text>
  <rect x="394" y="42" width="166" height="58" rx="8" class="nbox alt"/>
  <text x="477" y="66" text-anchor="middle" class="nt">replay every plan</text>
  <text x="477" y="83" text-anchor="middle" class="ns">the physics simulator</text>
  <rect x="608" y="42" width="160" height="58" rx="8" class="nbox"/>
  <text x="688" y="60" text-anchor="middle" class="nt">training records</text>
  <text x="688" y="76" text-anchor="middle" class="ns">label = certified plan&rsquo;s</text>
  <text x="688" y="90" text-anchor="middle" class="ns">real remaining cost</text>
  <rect x="608" y="168" width="160" height="52" rx="8" class="nbox"/>
  <text x="688" y="190" text-anchor="middle" class="nt">retrain, warm start</text>
  <text x="688" y="206" text-anchor="middle" class="ns">6 epochs, collapse alarms</text>
  <rect x="204" y="168" width="200" height="52" rx="8" class="nbox alt"/>
  <text x="304" y="190" text-anchor="middle" class="nt">exams + paired stats</text>
  <text x="304" y="206" text-anchor="middle" class="ns">gate: keep only real gains</text>
  <line x1="164" y1="71" x2="196" y2="71" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="354" y1="71" x2="386" y2="71" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="560" y1="63" x2="600" y2="63" class="arr ink" marker-end="url(#sa-ink)"/>
  <text x="580" y="52" text-anchor="middle" class="good-t">works &#10003;</text>
  <line x1="477" y1="100" x2="477" y2="140" class="arr" marker-end="url(#sa-mut)"/>
  <text x="489" y="122" class="bad-t">fails &#10007; &rarr; discarded, no record</text>
  <text x="489" y="138" class="cap">a wrong label is structurally impossible &mdash;</text>
  <text x="489" y="152" class="cap">only sub-optimality can creep in, and below 64&times;64</text>
  <text x="489" y="166" class="cap">even that is audited against the exact solver</text>
  <line x1="688" y1="100" x2="688" y2="160" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="600" y1="194" x2="412" y2="194" class="arr" marker-end="url(#sa-mut)"/>
  <path d="M608 178 C 420 118, 340 116, 296 104" class="arr acc dashed" marker-end="url(#sa-acc)"/>
  <text x="352" y="146" class="lbl">next round: better networks, new boards</text>
</svg>
<figcaption><strong>The loop&rsquo;s contract:</strong> a value backed up from an
unrealised plan is a <em>hypothesis</em>; only a replayed, certified plan sets the
record. Variety comes from the world (fresh boards each iteration; board IDs are
fenced so self-play can never touch the exam boards) plus AlphaZero&rsquo;s trick of
noise at the search root. One mixed-size iteration: ~940 puzzles attacked, ~860
solved, ~12,000 certified records, about 5&ndash;6 hours on one GPU.</figcaption>
</figure>'''


def _fig_arch() -> str:
    return '''
<figure class="fig big">
<svg viewBox="0 0 900 372" role="img"
     aria-label="Architecture: the board becomes n squared plus one tokens; walls become two attention masks; one transformer block with three attention families is applied twelve times with shared weights; its output feeds two heads: a value head that gathers five cells and outputs a 96-bin cost distribution, and a policy head of three chained pointer queries choosing bottleneck, support, then helper.">
  <rect x="14" y="60" width="158" height="92" rx="8" class="nbox alt"/>
  <text x="93" y="82" text-anchor="middle" class="nt">the board</text>
  <text x="93" y="100" text-anchor="middle" class="ns">n&sup2;+1 tokens: one per cell</text>
  <text x="93" y="114" text-anchor="middle" class="ns">+ 1 global scratchpad</text>
  <text x="93" y="130" text-anchor="middle" class="ns">7&ndash;9 yes/no channels each</text>
  <text x="93" y="144" text-anchor="middle" class="ns">(robots, goal, plan context)</text>
  <rect x="14" y="168" width="158" height="64" rx="8" class="nbox alt"/>
  <text x="93" y="188" text-anchor="middle" class="nt">walls &rarr; two masks</text>
  <text x="93" y="204" text-anchor="middle" class="ns">&ldquo;what can slide to what&rdquo;</text>
  <text x="93" y="218" text-anchor="middle" class="ns">not a channel &mdash; the wiring</text>
  <rect x="240" y="46" width="270" height="212" rx="10" class="nbox"/>
  <text x="375" y="68" text-anchor="middle" class="nt">one transformer block, d=192</text>
  <rect x="258" y="82" width="234" height="26" rx="5" class="nbox alt"/>
  <text x="375" y="99" text-anchor="middle" class="ns">attention &middot; global (unmasked)</text>
  <rect x="258" y="114" width="234" height="26" rx="5" class="nbox alt"/>
  <text x="375" y="131" text-anchor="middle" class="ns">attention &middot; along legal slides</text>
  <rect x="258" y="146" width="234" height="26" rx="5" class="nbox alt"/>
  <text x="375" y="163" text-anchor="middle" class="ns">attention &middot; helper-free slides only</text>
  <text x="375" y="192" text-anchor="middle" class="ns">concat &rarr; 192 &middot; + feed-forward 192&rarr;768&rarr;192</text>
  <text x="375" y="215" text-anchor="middle" class="lbl">applied &times;12 &mdash; the same weights every time</text>
  <text x="375" y="232" text-anchor="middle" class="ns">740,928 params (untied it would be 8,891,136)</text>
  <path d="M510 118 C 546 128, 546 176, 510 186" class="arr acc dashed" marker-end="url(#sa-acc)"/>
  <text x="536" y="156" class="lbl">&times;12</text>
  <line x1="172" y1="106" x2="232" y2="120" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="172" y1="196" x2="232" y2="170" class="arr dashed" marker-end="url(#sa-mut)"/>
  <text x="196" y="230" class="cap">masks gate</text>
  <text x="196" y="244" class="cap">the attention</text>
  <rect x="600" y="30" width="286" height="128" rx="8" class="nbox"/>
  <text x="743" y="52" text-anchor="middle" class="nt">value head &middot; 982,944 params total</text>
  <text x="743" y="72" text-anchor="middle" class="ns">gather 5 cells (start, end, B, S, H) &rarr; MLP</text>
  <text x="743" y="88" text-anchor="middle" class="ns">output: a distribution over 96 cost bins</text>
  <g>
    <rect x="672" y="100" width="9" height="10" fill="var(--mut)"/>
    <rect x="684" y="96" width="9" height="14" fill="var(--mut)"/>
    <rect x="696" y="84" width="9" height="26" fill="var(--acc)"/>
    <rect x="708" y="90" width="9" height="20" fill="var(--mut)"/>
    <rect x="720" y="98" width="9" height="12" fill="var(--mut)"/>
    <rect x="732" y="103" width="9" height="7" fill="var(--mut)"/>
    <line x1="664" y1="110" x2="800" y2="110" class="axis"/>
    <text x="758" y="106" class="cap">&ldquo;probably 6, maybe 9&rdquo;</text>
  </g>
  <text x="743" y="132" text-anchor="middle" class="ns">reported value = the distribution&rsquo;s mean;</text>
  <text x="743" y="146" text-anchor="middle" class="ns">trained on certified costs, + a ranking loss</text>
  <rect x="600" y="180" width="286" height="118" rx="8" class="nbox"/>
  <text x="743" y="202" text-anchor="middle" class="nt">policy head &middot; 1,223,232 params total</text>
  <rect x="614" y="214" width="76" height="30" rx="5" class="nbox alt"/>
  <text x="652" y="233" text-anchor="middle" class="ns">bottleneck?</text>
  <rect x="706" y="214" width="76" height="30" rx="5" class="nbox alt"/>
  <text x="744" y="233" text-anchor="middle" class="ns">support?</text>
  <rect x="798" y="214" width="76" height="30" rx="5" class="nbox alt"/>
  <text x="836" y="233" text-anchor="middle" class="ns">helper?</text>
  <line x1="690" y1="229" x2="700" y2="229" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="782" y1="229" x2="792" y2="229" class="arr ink" marker-end="url(#sa-ink)"/>
  <text x="743" y="266" text-anchor="middle" class="ns">three chained &ldquo;pointer&rdquo; queries: each scores the</text>
  <text x="743" y="280" text-anchor="middle" class="ns">actual board cells by dot product &mdash; no fixed output list</text>
  <line x1="510" y1="94" x2="592" y2="76" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="510" y1="210" x2="592" y2="228" class="arr ink" marker-end="url(#sa-ink)"/>
  <text x="450" y="300" text-anchor="middle" class="lbl">no positional embeddings anywhere &mdash; nothing in the weights is board-shaped,</text>
  <text x="450" y="317" text-anchor="middle" class="lbl">so ONE checkpoint runs at 16&times;16, 24&times;24, 32&times;32, 64&times;64 or 96&times;96 unchanged</text>
  <text x="450" y="338" text-anchor="middle" class="cap">position comes from the masks: the network is never told &ldquo;this is cell (7,12)&rdquo; &mdash; it is told what can reach what,</text>
  <text x="450" y="353" text-anchor="middle" class="cap">and twelve rounds of message-passing let information travel twelve slides across the board</text>
</svg>
<figcaption><strong>The architecture, drawn from the code
(<code>nn_labeler/model.py</code>, <code>spr/nets.py</code>).</strong> Three
unusual choices do the work. <em>Walls are wiring, not input:</em> attention is
masked so information flows only along legal slides. <em>One block, twelve
times:</em> the same 740,928 weights are reused for all twelve reasoning rounds
&mdash; a 12&times; parameter saving. <em>No positional embeddings:</em> the older
per-size networks carried a 110,784-parameter position table that locked each
checkpoint to one board size; deleting it is what makes these networks
size-free.</figcaption>
</figure>'''


def _fig_expansion() -> str:
    return '''
<figure class="fig big">
<svg viewBox="0 0 900 214" role="img"
     aria-label="One expansion: a partial plan goes to the candidate generator, one policy pass ranks all legal sub-goals, the top five are kept, one batched value pass prices them, and five priced child plans result. Physics checks and forced fixes are free and not counted.">
  <rect x="14" y="52" width="128" height="56" rx="8" class="nbox alt"/>
  <text x="78" y="76" text-anchor="middle" class="nt">one partial plan</text>
  <text x="78" y="93" text-anchor="middle" class="ns">next open segment</text>
  <rect x="176" y="52" width="150" height="56" rx="8" class="nbox alt"/>
  <text x="251" y="76" text-anchor="middle" class="nt">generator</text>
  <text x="251" y="93" text-anchor="middle" class="ns">every legal sub-goal (~12&ndash;50)</text>
  <rect x="360" y="52" width="150" height="56" rx="8" class="nbox"/>
  <text x="435" y="76" text-anchor="middle" class="nt">policy pass</text>
  <text x="435" y="93" text-anchor="middle" class="ns">1 GPU call &middot; rank &middot; keep top 5</text>
  <rect x="544" y="52" width="150" height="56" rx="8" class="nbox"/>
  <text x="619" y="76" text-anchor="middle" class="nt">value pass</text>
  <text x="619" y="93" text-anchor="middle" class="ns">1 batched call &middot; price the 5</text>
  <g>
    <rect x="728" y="30" width="150" height="22" rx="5" class="nbox alt"/>
    <rect x="728" y="58" width="150" height="22" rx="5" class="nbox alt"/>
    <rect x="728" y="86" width="150" height="22" rx="5" class="nbox alt"/>
    <text x="803" y="45" text-anchor="middle" class="ns">child plan &middot; est. 23.9</text>
    <text x="803" y="73" text-anchor="middle" class="ns">child plan &middot; est. 24.3</text>
    <text x="803" y="101" text-anchor="middle" class="ns">&hellip; 5 children, each priced</text>
  </g>
  <line x1="142" y1="80" x2="168" y2="80" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="326" y1="80" x2="352" y2="80" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="510" y1="80" x2="536" y2="80" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="694" y1="72" x2="720" y2="56" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="694" y1="80" x2="720" y2="80" class="arr ink" marker-end="url(#sa-ink)"/>
  <line x1="694" y1="88" x2="720" y2="104" class="arr ink" marker-end="url(#sa-ink)"/>
  <rect x="176" y="140" width="518" height="30" rx="6" class="nbox alt"/>
  <text x="435" y="159" text-anchor="middle" class="ns">free, never counted: segments solvable outright &middot; every physics check &middot; replay certification</text>
  <text x="435" y="196" text-anchor="middle" class="cap">exam budget: 1,200 expansions per puzzle &mdash; the same accounting as the frozen benchmark, so every row is comparable</text>
</svg>
<figcaption><strong>One expansion</strong> &mdash; the unit everything is budgeted
in: one policy call plus one batched value call, five children born
(<code>spr/search.py::expand</code>). The tree search on top is AlphaZero&rsquo;s
selection rule adapted for costs (take the <em>minimum</em> over children, not the
average &mdash; there is no opponent), with one twist that matters: a finished plan
is immediately replayed, and a plan that fails the replay is removed from the
tree entirely, its parent re-valued. Physics settles the leaves; the networks
only order the visits.</figcaption>
</figure>'''


def _journey_tree() -> str:
    return '''
<ul class="tree">
  <li><span class="what">Inherited: supervised planners + one small size-free value net</span>
      <span class="why">the value net (the &ldquo;labeller&rdquo;) had been trained on 8&times;8&ndash;16&times;16 boards only</span>
    <ul>
      <li><span class="chip good">&#10003; kept</span> <span class="what">Rebuild both networks size-free, prove they match the specialists</span>
          <span class="why">they overshot: one pair beat the purpose-built 24&times;24 planner by 10 puzzles with 5&times; less search (215/232 vs 205/232) &mdash; after discovering the training recipe needed gradient clipping (the naive recipe <em>degrades</em> from epoch 0)</span>
        <ul>
          <li><span class="chip bad">&#10007; flat</span> <span class="what">Self-play in the base vocabulary</span>
              <span class="why">two full iterations moved nothing &mdash; exactly as the wall measurement predicted: the base language was already saturated. A clean negative result that redirected the project</span>
            <ul>
              <li><span class="chip good">&#10003; step</span> <span class="what">Five self-play iterations in the extended (B2) vocabulary</span>
                  <span class="why">frontier solves 133&rarr;158; but the real lesson: the loop learned <em>efficiency</em> &mdash; a cheap 26-expansion search now reaches what needed a 500-expansion search before. Move counts did not improve: search got distilled into the networks, reach did not grow</span>
                <ul>
                  <li><span class="chip good">&#10003; jump</span> <span class="what">Mixed-size curriculum: self-play on 24&times;24 + 32&times;32 + 8-robot boards at once</span>
                      <span class="why">one iteration broke the plateau everywhere &mdash; frontier 170 / 235 / <strong>269</strong> (the 8-robot jump: 269 of 289 vs 161 for the supervised arm). Diagnosis confirmed: the plateau was a <em>data-distribution</em> limit, not capacity. Iterations 2&ndash;3: flat again &mdash; the loop&rsquo;s recurring shape</span>
                    <ul>
                      <li><span class="what">The experiment lab: 21 matched-protocol arms, one change each</span>
                          <span class="why">same frozen starting networks, same budget, a 200-puzzle unseen exam nothing trains on, two-seed replication before any claim &mdash; the full cards live in the <a href="#variants">Variants lab tab</a></span>
                        <ul>
                          <li><span class="chip bad">&#10007; killed</span> <span class="what">v01 &middot; AlphaZero&rsquo;s own policy target (visit counts)</span>
                              <span class="why">catastrophic here (every exam collapsed, p &le; 1e-7) &mdash; proof the certified-cost target is load-bearing</span></li>
                          <li><span class="chip good">&#10003; adopted</span> <span class="what">v04 &middot; keep every decision the search examined</span>
                              <span class="why">the control trained only on the winning line; certified off-line decisions are just as true &mdash; ~2&times; the records, replicated frontier gain</span></li>
                          <li><span class="chip warn">~ not replicated</span> <span class="what">v06 &middot; Gumbel-style root exploration</span>
                              <span class="why">a convincing seed-7 win that reversed at seed 8 &mdash; exactly what the two-seed rule exists to catch</span></li>
                          <li><span class="chip ctl">control</span> <span class="what">v08 &middot; cold start: random weights, no supervised anything</span>
                              <span class="why">one label-free iteration reached 132/200 unseen &mdash; statistically tied with the fully-supervised planner&rsquo;s 134. The supervised head start helps (~40 puzzles) but is not <em>needed</em></span></li>
                          <li><span class="chip good">&#10003; adopted</span> <span class="what">v09 &middot; train the value net on the number you are graded on</span>
                              <span class="why">the whole stack had trained on abstract plan cost but was graded on real moves; self-play&rsquo;s certified records carry the real count, so the target could finally be fixed &mdash; better on every exam, both seeds. The strongest network change of the project</span></li>
                          <li><span class="chip bad">&#10007; killed</span> <span class="what">v12 &middot; practice only on hard puzzles</span>
                              <span class="why">flat after one iteration and still flat after three chained ones</span></li>
                          <li><span class="chip good">&#10003; adopted</span> <span class="what">v14 &middot; v04 + v09 stacked</span>
                              <span class="why">wins the unseen exam at both seeds &mdash; the lab&rsquo;s recipe</span></li>
                          <li><span class="chip bad">&#10007; killed twice</span> <span class="what">v15, v16 &middot; teach the networks about slides</span>
                              <span class="why">both flat at both seeds; the transfer result later explained why &mdash; the networks already generalise to post-slide positions. There was no gap to teach</span></li>
                          <li><span class="chip good">&#10003; the break</span> <span class="what">v07 &middot; the hybrid search: one or two ordinary moves <em>before</em> sub-goal planning</span>
                              <span class="why">a sub-goal plan from the <em>shifted</em> position is a different plan space &mdash; sometimes a cheaper one, even after paying for the slide. The only change that ever went below the +1.17 wall, and it needed no retraining at all</span></li>
                        </ul>
                      </li>
                      <li><span class="chip good">&#10003; flagship</span> <span class="what">= the v09 networks + the depth-2 hybrid search</span>
                          <span class="why">chosen by rule, not taste: each ingredient had to survive its matched control and its replication seed</span></li>
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
    """The Story tab body."""
    out = ['<section id="story" class="story">', "<h2>The story &mdash; start here</h2>"]
    out.append(
        '<p class="toc">Chapters: '
        '<a href="#st-game">the game</a> &middot; '
        '<a href="#st-language">sub-goals</a> &middot; '
        '<a href="#st-ceiling">the wall</a> &middot; '
        '<a href="#st-referee">the referee</a> &middot; '
        '<a href="#st-nets">the networks</a> &middot; '
        '<a href="#st-search">one search step</a> &middot; '
        '<a href="#st-journey">the journey</a> &middot; '
        '<a href="#st-scoreboard">scoreboard</a> &middot; '
        '<a href="#st-failed">what failed</a> &middot; '
        '<a href="#st-limits">limits</a>. '
        'This chapter is a hand-written narrative; its numbers were verified '
        'against the result files in <code>EXPLAINER.md</code> (the in-depth, '
        'source-cited companion), and the live tables in the other tabs are '
        'the authoritative versions.</p>')
    out.append(
        '<p class="lede">Three results, up front. A planner trained '
        '<strong>only on its own physically-verified solutions</strong> solves '
        '188 of 200 brand-new puzzles, against 134 for the planner taught from '
        'a solution library. Its plan vocabulary was <strong>proved</strong> to '
        'cost at least +1.17 moves over perfect play &mdash; and a one-line change '
        'to the search went below that wall, to +0.94. And the same networks, '
        'never adjusted, do it on board sizes and robot counts they never ran '
        'on before.</p>')

    # 1 -- the game
    out.append('<h3 id="st-game"><span class="no">1</span>A game where you build your own walls</h3>')
    out.append(
        '<p><strong>Ricochet Robots</strong> is played on a square grid with a '
        'few walls. A move picks one robot and a direction; the robot then '
        '<em>slides until something stops it</em> &mdash; a wall or another robot. '
        'It cannot stop halfway. The goal: bring one specific robot to rest on '
        'one specific cell, in as few moves as possible.</p>')
    out.append(_fig_slide())
    out.append(
        '<p>That is why the game is hard: solutions are little construction '
        'projects, 8&ndash;30 moves long, where every robot is a potential wall '
        'for every other. How hard, concretely? The project ships an exact '
        'optimal solver. On the pinned 24&times;24 exam it cracked '
        '<strong>232 puzzles</strong> (&ldquo;graded&rdquo; &mdash; perfect play is '
        'known, averaging 7.59 moves) and <strong>failed on 218</strong> '
        '(&ldquo;frontier&rdquo;). On the 200-puzzle &ldquo;unseen&rdquo; exam '
        '&mdash; fresh boards nothing ever trained on &mdash; it managed only 137. '
        '<em>On a third of these puzzles, nobody knows the right answer, '
        'including the exact solver.</em></p>')
    out.append(
        '<p>One rule is absolute throughout the project: a puzzle counts as '
        'solved only when the claimed plan has been replayed move by move '
        'against the physics and observed to work. No number anywhere comes '
        'from a network&rsquo;s opinion.</p>')

    # 2 -- sub-goals
    out.append('<h3 id="st-language"><span class="no">2</span>Thinking in sub-goals, not moves</h3>')
    out.append(
        '<p>The planner does not search move by move (that tree is 8&ndash;30 '
        'levels deep). It reasons <em>backwards from the goal</em> in '
        '<strong>sub-goals</strong>. Its state is a partial plan &mdash; a to-do '
        'list of segments, &ldquo;robot X must get from A to B&rdquo; &mdash; and '
        'each decision resolves one segment by answering: <em>what would make '
        'that journey possible?</em> A sub-goal is a triple, and it is the '
        'planner&rsquo;s entire vocabulary:</p>')
    out.append(_fig_subgoal())

    # 3 -- the wall
    out.append('<h3 id="st-ceiling"><span class="no">3</span>First, prove the wall exists</h3>')
    out.append(
        '<p>Before training anything, the project asked a question most ML '
        'work skips: <em>if the ranking were perfect and the search unlimited, '
        'how good could a sub-goal planner possibly be?</em> That is answerable '
        'without any network &mdash; enumerate every expressible plan, physically '
        'verify each one, compare the best to perfect play '
        '(<a href="#ceiling">Ceiling study tab</a> holds the full data).</p>')
    out.append(_fig_ceiling(with_flagship=False))
    out.append(
        '<p class="key">Measuring the wall first is what shaped everything '
        'after. It predicted that self-play in the base vocabulary would be '
        'pointless (it was), it justified the pivot to the extended '
        'vocabulary, and it defined what a real breakthrough would have to '
        'look like: <em>go below +1.17</em>.</p>')

    # 4 -- referee
    out.append('<h3 id="st-referee"><span class="no">4</span>Self-play with a physics referee</h3>')
    out.append(
        '<p>AlphaZero improves by playing itself: search beats raw instinct, '
        'so the network trains to imitate its own search, and search on the '
        'improved network is better still. Two of its ingredients are missing '
        'here &mdash; there is <em>no opponent</em> (a one-player puzzle) and '
        '<em>no win/lose</em> (a move count to minimise). What replaces '
        'them:</p>')
    out.append(_fig_loop())

    # 5 -- networks
    out.append('<h3 id="st-nets"><span class="no">5</span>The two networks &mdash; 2.2 million parameters, any board size</h3>')
    out.append(
        '<p>The planner is a <strong>policy network</strong> (&ldquo;which '
        'sub-goal should we try?&rdquo;) and a <strong>value network</strong> '
        '(&ldquo;how many steps will this plan still cost?&rdquo;) sharing one '
        'encoder design. Together: <strong>2,206,176 parameters</strong> &mdash; '
        'tiny by modern standards, and deliberately so.</p>')
    out.append(_fig_arch())
    out.append('<div class="tw"><table class="t">'
               '<thead><tr><th>network</th><th>sees (channels)</th>'
               '<th>answers</th><th>parameters</th></tr></thead><tbody>'
               '<tr><td>value</td>'
               '<td style="text-align:left">9 &mdash; mover, goal, the candidate&rsquo;s B/S/H, all robots, and the rest of the plan&rsquo;s commitments</td>'
               '<td style="text-align:left">&ldquo;how many steps will finishing this plan still cost?&rdquo; &mdash; a 96-bin distribution</td>'
               '<td>982,944</td></tr>'
               '<tr><td>policy</td>'
               '<td style="text-align:left">7 &mdash; the same minus any candidate: it must <em>generate</em> one</td>'
               '<td style="text-align:left">&ldquo;which (B,&thinsp;S,&thinsp;H) should we try?&rdquo; &mdash; three chained pointer choices</td>'
               '<td>1,223,232</td></tr>'
               '</tbody></table></div>')
    out.append(
        '<p class="note">Proof the size-free bet works, not merely compiles: '
        'trained on 16&times;16 and 24&times;24 boards only, the value network '
        'was audited against exact ground truth at 32, 40, 48, 56 and 64 &mdash; '
        'it picks the optimal candidate 86.3% of the time at 32&times;32 and '
        '83.8% at 64&times;64, beating the network that initialised it at '
        'every single size (live table: <a href="#res-audit">far-size '
        'audits</a>).</p>')

    # 6 -- one search step
    out.append('<h3 id="st-search"><span class="no">6</span>One search step, and how the budget is counted</h3>')
    out.append(_fig_expansion())

    # 7 -- the journey
    out.append('<h3 id="st-journey"><span class="no">7</span>The journey: every decision, and what it taught</h3>')
    out.append(
        '<p>The path to the best planner was not a straight line &mdash; and the '
        'failures were as load-bearing as the wins. Green = adopted, red = '
        'tried and killed, amber = not replicated. Each verdict is backed by '
        'paired statistics at matched search budgets, and every claimed win '
        'was replicated on a second training seed before being believed.</p>')
    out.append(_journey_tree())
    out.append(
        '<p class="key">Two patterns run through the whole tree. <strong>One '
        'big step when something structural changes, then flat</strong> &mdash; '
        'iterating a good recipe never compounded. And <strong>training never '
        'bought shorter solutions; search changes did</strong> &mdash; every '
        'move-count gain on this page traces to the hybrid search, not to the '
        'training loop.</p>')

    # 8 -- scoreboard
    out.append('<h3 id="st-scoreboard"><span class="no">8</span>The scoreboard, in absolute terms</h3>')
    out.append('<p>All rows: 1,200-expansion budget, five candidates per step, '
               'every solved puzzle replay-certified.</p>')
    out.append(_fig_ceiling(with_flagship=True))
    out.append('<p><strong>Brand-new puzzles &mdash; the generalisation exam.</strong> '
               '200 puzzles on 50 fresh boards nothing ever trained on. On the '
               '137 where the exact solver could establish perfect play (8.71 '
               'moves on average):</p>')
    out.append('<div class="tw"><table class="t">'
               '<thead><tr><th>planner</th><th>solved (of 200)</th>'
               '<th>of the 137 graded</th><th>solved perfectly</th>'
               '<th>extra moves vs perfect</th></tr></thead><tbody>'
               '<tr class="hl"><td>flagship (self-taught + hybrid search)</td><td>188</td><td>134</td><td>57%</td><td class="g">+1.47</td></tr>'
               '<tr><td>same networks, standard search</td><td>182</td><td>132</td><td>50%</td><td>+1.94</td></tr>'
               '<tr><td>the frozen networks every lab arm started from</td><td>175</td><td>131</td><td>47%</td><td>+2.93</td></tr>'
               '<tr><td>label-free from <em>random</em> initialisation (one iteration)</td><td>132</td><td>109</td><td>45%</td><td>+2.89</td></tr>'
               '<tr><td>supervised backward planner (the baseline to beat)</td><td>134</td><td>115</td><td>36%</td><td>+4.84</td></tr>'
               '<tr><td>supervised move-by-move planner</td><td>101</td><td>101</td><td>90%</td><td>+0.12</td></tr>'
               '</tbody></table></div>')
    out.append(
        '<p class="note">Read the last row carefully &mdash; it frames the whole '
        'project. The move-by-move planner is nearly perfect <em>when it '
        'solves at all</em>, but it solves half the exam at 3&ndash;4&times; the '
        'search cost. The stated goal &mdash; &ldquo;solve at least as many as '
        'the supervised baseline, with fewer moves&rdquo; &mdash; is met on '
        'brand-new boards with no human labels anywhere in the loop. The '
        'stretch goal &mdash; matching the move-by-move planner&rsquo;s solution '
        '<em>lengths</em> &mdash; is not met and was provably out of reach for '
        'the sub-goal language; the hybrid cut the gap from ~2.2 moves to '
        '0.87 on the graded exam. The apples-to-apples versions of these '
        'comparisons (moves on the puzzles <em>all</em> systems solved) live '
        'in the <a href="#baselines">Baselines tab</a>.</p>')
    out.append('<p><strong>Does it survive a change of board?</strong> The '
               'hybrid search and its networks were developed entirely at '
               '24&times;24 with 4 robots. Run unchanged elsewhere:</p>')
    out.append('<div class="tw"><table class="t">'
               '<thead><tr><th>exam it never ran on</th><th>hybrid</th>'
               '<th>matched control (same nets, same budget)</th>'
               '<th>paired move wins</th></tr></thead><tbody>'
               '<tr><td>32&times;32 graded (175)</td><td class="g">174 solved, +1.60 vs perfect</td><td>172, +1.87</td><td>17 / 0</td></tr>'
               '<tr><td>24&times;24, 8 robots, graded (161)</td><td class="g">159 solved, +1.24 vs perfect</td><td>159, +1.82</td><td>23 / 0</td></tr>'
               '<tr><td>24&times;24, 8 robots, frontier (289)</td><td class="g">276 solved &mdash; the program record</td>'
               '<td colspan="2" style="text-align:left">(vs 161 for the supervised arm)</td></tr>'
               '</tbody></table></div>')
    out.append('<p>The biggest gain is on the crowded 8-robot boards &mdash; where '
               'being allowed to shove a robot out of the way first is worth '
               'the most. Nothing was retrained, retuned, or even told the '
               'board had changed.</p>')

    # 9 -- what failed
    out.append('<h3 id="st-failed"><span class="no">9</span>What failed, and why the failures mattered</h3>')
    out.append(
        '<ul>'
        '<li><strong>The textbook recipe lost.</strong> AlphaZero&rsquo;s own '
        'policy target (train on the search&rsquo;s visit counts) collapsed every '
        'exam. Its failure is what proves the alternative &mdash; training on '
        'physically-certified costs &mdash; is the load-bearing choice, not an '
        'incidental one.</li>'
        '<li><strong>The base language was saturated before self-play '
        'began.</strong> Two iterations moved nothing, exactly as the wall '
        'measurement predicted &mdash; which is why measuring the wall first was '
        'worth more than any training run.</li>'
        '<li><strong>Teaching the networks about slides failed twice</strong> '
        '(uniform nudges, then the search&rsquo;s own preferred slides) &mdash; and '
        'the transfer result explained why: the networks already handle '
        'post-slide positions. The gap the training was meant to close did '
        'not exist. The lever was the <em>search</em>.</li>'
        '<li><strong>A convincing win evaporated on a second seed.</strong> '
        'The Gumbel exploration arm won at seed 7 (p&thinsp;=&thinsp;0.015) '
        'and reversed at seed 8. Every adopted result on this page survived '
        'that test; this one is why the test exists.</li>'
        '</ul>')

    # 10 -- limits
    out.append('<h3 id="st-limits"><span class="no">10</span>Honest limits, and what&rsquo;s next</h3>')
    out.append(
        '<ul>'
        '<li>Self-play <em>training</em> bought solve rate and search '
        'efficiency &mdash; never, by itself, shorter solutions. All move-count '
        'gains came from the search change.</li>'
        '<li>The move-by-move planner still wins pure solution quality '
        '(+0.12 vs +0.86); closing that likely needs deeper hybrids or a '
        'learned slide proposer.</li>'
        '<li>Chained iterations of a good recipe do not compound &mdash; one '
        'step, then flat &mdash; and nobody yet knows why the ceiling arrives so '
        'fast.</li>'
        '<li>Most main-line results are single-seed on the network side (the '
        'lab replicates at two; the measured seed noise is small, but '
        'measured at one rung).</li>'
        '</ul>')
    out.append(
        '<p><strong>Next:</strong> the same loop at 80&times;80 and '
        '96&times;96 &mdash; board sizes where <em>no exact solver exists at '
        'all</em> and physics certification is the only ground truth there '
        'will ever be. The size-free networks run there unchanged; the '
        'fidelity audits up to 64&times;64 are the warranty. That is the '
        'experiment this whole design was built to make possible.</p>')
    out.append('<h3>Mini-glossary</h3>')
    out.append(
        '<dl class="gloss">'
        '<dt>certified / replayed</dt><dd>the plan was executed move by move '
        'in the game physics and observed to work &mdash; the project&rsquo;s only '
        'definition of &ldquo;solved&rdquo;</dd>'
        '<dt>expansion</dt><dd>the search budget unit: one policy call + one '
        'batched value call, five child plans born; exams allow 1,200 per '
        'puzzle</dd>'
        '<dt>graded / frontier / unseen</dt><dd>puzzles where perfect play is '
        'known / where the exact solver failed / fresh boards nothing trained '
        'on</dd>'
        '<dt>extra moves vs perfect (&ldquo;regret&rdquo;)</dt><dd>how many '
        'moves beyond the proven optimum a planner&rsquo;s solutions use, '
        'averaged</dd>'
        '<dt>warm start</dt><dd>initialising training from an existing '
        'checkpoint instead of random weights</dd>'
        '<dt>paired test</dt><dd>comparing two planners puzzle by puzzle on '
        'the same puzzles &mdash; the only comparison that survives this '
        'domain&rsquo;s seed noise</dd>'
        '</dl>')
    out.append('<p class="note">Depth on any chapter: '
               '<code>self_play_robots/EXPLAINER.md</code> &mdash; the same story '
               'with every number source-cited, a real worked decision, and '
               'the full training recipes.</p>')
    out.append("</section>")
    return "\n".join(out)
