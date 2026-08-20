"""STUB (wave 2, design only): hybrid action space -- subgoal macros + primitive
move escapes.

Why it exists: FINDINGS §3 proved the moves headline ("closer to optimal than
forward-supervised") is UNREACHABLE inside the pure subgoal language -- even a
perfect ranker with unlimited search bottoms out at ~+0.9/+1.2 mean regret,
20-50x the forward planner's. The only way to that headline is to change the
action space itself. The B2 vocabulary was step 1 (richer subgoals); this stub
is step 2: let the search interleave PRIMITIVE MOVES with subgoal decisions so
it can express the "one clever slide" the subgoal grammar cannot.

Sketch (not implemented; the two stacks disagree on node identity):
  * a decision node offers `heuristics.propose*` candidates PLUS the <= 4*R
    primitive slides applied to the plan's current certified prefix state;
  * a slide child re-roots the search on the post-slide state with the SAME
    open goals (needs a State -> PartialPlan re-projection, the hard part:
    `skeleton.astar._initial_plan(env, moved_state)` re-proposes from scratch,
    which loses the plan's committed structure);
  * cost bookkeeping: slides cost 1 strict move; subgoal fixed_g stays
    abstract -- unify by pricing everything in strict moves via incremental
    certification (Certifier already replays prefixes);
  * value/policy nets need a "moved robots" input channel -- the forward
    MoveNet Guide (spr/fwd) already encodes exactly that and could serve as
    the slide-head.
Estimated cost to first smoke: 2-3 days of work; park until wave 1 reports.
"""
from variants import Variant

VARIANT = Variant(
    vid="v07_hybrid_actions",
    axis="action-space",
    title="Hybrid subgoal + primitive-move search (design stub)",
    hypothesis="Interleaving primitive slides with subgoal decisions breaks "
               "the §3 language ceiling and closes part of the 20-50x regret "
               "gap to the forward planner.",
    mechanism="NOT IMPLEMENTED -- see module docstring for the design sketch "
              "and the State->PartialPlan re-projection blocker.",
    expected_failure="Search-space explosion: 4R slides x subgoal candidates "
                     "per node at 1200 expansions may underperform both pure "
                     "spaces.",
    status="stub",
)
