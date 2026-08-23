"""Search-ranked slide training (wave 5) -- v15's sharpened successor.

v15 (uniform random nudges) was flat at both seeds; its declared failure mode
-- "a uniform slide is not the search's chosen slide" -- was the likely cause.
This variant nudges half the training instances by the TOP-RANKED slide under
the hybrid search's own ranking (moved-state initial-plan cost, exactly the
criterion v07 uses to pick sub-search states), so the nets train on the
specific off-distribution states the hybrid search actually descends into.
Everything else is the matched protocol; the payoff evaluation (hybrid search
with these nets) runs in the closeout if the standard rows justify it.
"""
from variants import Variant

P_SLIDE = 0.5

VARIANT = Variant(
    vid="v16_ranked_slide_training",
    axis="action-space",
    title="Train on the search's own chosen slides",
    hypothesis="Nudging training instances by the hybrid ranking's TOP slide "
               "(not a uniform one) matches the sub-search state distribution "
               "and widens the hybrid's moves win.",
    mechanism="Generation hook: with p=0.5, rank all legal slides by moved-"
              "state initial-plan cost (the hybrid's criterion) and apply the "
              "best one before the search. Labels/certification untouched.",
    expected_failure="The ranked slide usually LOWERS plan cost, so nudged "
                     "instances are easier -- the distribution shifts easy, "
                     "diluting the buffer like v03 feared in reverse.",
    hooks=("selfplay",),
    plain_what="Same idea as v15 (practice on one-move-nudged puzzles), but "
               "nudged by the exact move the one-move-first search itself "
               "would pick, instead of a random one.",
    plain_why="The networks should learn precisely the states the new search "
              "asks them about -- not random neighbours of them.",
)


def apply_selfplay(ctx):
    import nn.generate as G
    from move_planner.state import legal_moves
    from simulate import wall_sets, _board_size
    from skeleton.astar import _initial_plan
    from GridEnv import State, Robot_at
    from spr.search import forced_fixes

    _orig = G.random_instance
    _walls = {}
    solver = ctx["solver"]

    def ranked_slid_instance(env, colors, rng):
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
        best, best_c = None, None
        for slot, _di, newpos in legal_moves(positions, wr, wd, n):
            robots2 = [Robot_at(position=tuple(newpos[j]), color=robots[j].color)
                       for j in range(len(robots))]
            st2 = State(target=st.target, target_robot=robots2[0], helpers=robots2[1:])
            try:
                c = float(forced_fixes(env, st2, solver,
                                       _initial_plan(env, st2)).cost())
            except Exception:
                continue
            if best_c is None or c < best_c:
                best, best_c = st2, c
        return best if best is not None else st

    G.random_instance = ranked_slid_instance
