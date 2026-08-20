"""Self-generated curriculum: spend the generation budget on the CURRENT net's
failure frontier instead of uniform random instances.

Structurally different from post-hoc filtering (the parked v03 knob): instead of
discarding easy records after paying for their search, this variant redirects
the search budget BEFORE it is spent. Each worker screens candidate instances
with a cheap probe (the net-guided anytime A* at 40 expansions -- the planner's
own notion of "easy") and only runs the full 300-expansion MCTS on instances the
probe FAILS. The training distribution becomes "what the current net cannot yet
do", the moving frontier that §19d identified as the missing ingredient
("6k records on the same distribution each iteration"), and that PROBLEM.md
§6.4 asked for ("sample instances the current net finds hard"). Boards stay
fresh and uniform (never pinned/exam ids); only the instance acceptance changes,
so the no-human-label constraint is untouched.

Mechanism (worker-side hook): wrap `nn.generate.random_instance`; sample up to
PROBE_TRIES candidates per call, return the first the probe cannot certify
within PROBE_EXP expansions; fall back to the hardest seen (largest first-
certified expansion count) if all pass. Probe cost ~40 expansions vs 300 for
generation: <= 15% overhead paid for a concentrated buffer.
"""
from variants import Variant

PROBE_EXP = 40
PROBE_TRIES = 6

VARIANT = Variant(
    vid="v12_frontier_curriculum",
    axis="data",
    title="Frontier-mining curriculum (probe-and-reject easy instances)",
    hypothesis="Concentrating full searches on instances the current net's own "
               "cheap probe fails moves the frontier exams (incl. unseen "
               "boards) where uniform sampling saturated.",
    mechanism="Worker-side hook wrapping nn.generate.random_instance: 40-"
              "expansion anytime-A* probe with the CURRENT nets; keep the "
              "first failing instance of <= 6 draws (else the hardest).",
    expected_failure="Frontier instances at this budget mostly stay UNSOLVED "
                     "in the full search too -> fewer certified records, "
                     "training starves (watch records/iteration in the "
                     "manifest).",
    hooks=("selfplay",),
)


def apply_selfplay(ctx):
    import nn.generate as G
    from spr.search import astar

    _orig = G.random_instance

    def frontier_instance(env, colors, rng):
        _W = ctx
        solver, ev = _W["solver"], _W["ev"]
        opts = _W["opts"]
        parks = opts.get("vocab") in ("b1", "b2")
        from simulate import _board_size
        n = _board_size(env.grid_data, None)
        hardest, hardest_score = None, -1.0
        for t in range(PROBE_TRIES):
            st = _orig(env, colors, rng)
            try:
                res = astar(env, st, solver, ev, -1, n, opts["k"], PROBE_EXP,
                            None, False, "parent", None, False,
                            anytime=True, parks=parks)
            except Exception:
                return st                      # probe crash -> just use it
            if res.strict is None:             # probe failed: frontier instance
                return st
            score = float(res.expansions)
            if score > hardest_score:
                hardest, hardest_score = st, score
        return hardest if hardest is not None else _orig(env, colors, rng)

    G.random_instance = frontier_instance
