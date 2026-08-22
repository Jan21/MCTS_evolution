"""Slide-aware training (wave 4, v07's training half): generate self-play data
from POST-SLIDE states, so the nets learn the distribution the hybrid search
actually queries.

v07 showed the root-slides search wins with nets that never saw a slid state:
every sub-search evaluates positions one primitive move away from the sampled
instance distribution, i.e. slightly OFF the nets' training distribution. This
variant closes that gap on the data side without touching extraction or cost
semantics: during generation, half of the sampled instances get one uniformly
random legal slide applied BEFORE the search runs. A slid instance is just a
different valid instance on the same board -- labels, certification and the
18-field record schema are untouched. The trained nets are then benched (a)
under the standard protocol by the runner, and (b) under the hybrid search in
the wave-4 flagship job, where the hypothesis actually cashes out: hybrid
search + slide-trained nets > hybrid search + v09 nets.
"""
import random as _random

from variants import Variant

P_SLIDE = 0.5

VARIANT = Variant(
    vid="v15_slide_training",
    axis="action-space",
    title="Train on post-slide states (v07's training half)",
    hypothesis="Nets trained on the post-slide state distribution rank the "
               "hybrid search's sub-problems better, widening v07's moves win "
               "at unchanged solve rate.",
    mechanism="Generation hook: with probability 0.5 apply one uniformly "
              "random legal slide to each sampled instance before the search "
              "(labels/certification untouched). Standard runner benches; the "
              "hybrid-search evaluation runs in the flagship job.",
    expected_failure="Post-slide states are 'free' data only if the slide "
                     "distribution matches what the hybrid search visits; a "
                     "UNIFORM slide is not the search's ranked slide -- the "
                     "shift may be too broad to help.",
    hooks=("selfplay",),
    plain_what="The one-move-first planner (v07) asks the networks about "
               "board states that are one robot move away from anything they "
               "were trained on. We retrained them with half of the practice "
               "puzzles nudged by one random robot move first.",
    plain_why="Networks answer best near their training data; teaching them "
              "the nudged states should make the one-move-first search even "
              "stronger.",
)


def apply_selfplay(ctx):
    import nn.generate as G
    from move_planner.state import legal_moves
    from simulate import wall_sets, _board_size
    from GridEnv import State, Robot_at

    _orig = G.random_instance
    _walls = {}

    def slid_instance(env, colors, rng):
        st = _orig(env, colors, rng)
        if rng.random() >= P_SLIDE:
            return st
        key = id(env)
        if key not in _walls:
            n = _board_size(env.grid_data, None)
            _walls[key] = (*wall_sets(env.grid_data, n), n)
        wr, wd, n = _walls[key]
        robots = [st.target_robot] + list(st.helpers)
        positions = tuple(r.position for r in robots)
        mv = legal_moves(positions, wr, wd, n)
        if not mv:
            return st
        slot, _di, newpos = mv[rng.randrange(len(mv))]
        robots2 = [Robot_at(position=tuple(newpos[j]), color=robots[j].color)
                   for j in range(len(robots))]
        return State(target=st.target, target_robot=robots2[0], helpers=robots2[1:])

    G.random_instance = slid_instance
