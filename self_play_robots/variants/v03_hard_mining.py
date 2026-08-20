"""Hard-instance mining: train only on decisions the search had to WORK for.

PROBLEM.md §6.4 warns flat difficulty is a real failure mode (80x80 corpora are
81% depth-0), and §19d attributes the loop's saturation to "6k records on the
same distribution each iteration". This variant keeps an instance's records only
if the search spent >= 30 expansions (MINEXP knob, already wired in
spr.selfplay: easy instances are solved in a handful), and doubles the instance
budget per board so the KEPT record volume stays comparable to control.
FINDINGS §12 found this filter flat in the saturated BASE vocabulary at
iteration 2 -- the open question is whether it helps in the unsaturated B2 +
fresh-board setting where hard instances still carry signal.
"""
from variants import Variant

VARIANT = Variant(
    vid="v03_hard_mining",
    axis="data",
    title="Hard-instance mining (MINEXP=30, 2x instance budget)",
    hypothesis="Concentrating the replay buffer on decisions that needed search "
               "raises frontier solve rate at unchanged graded performance.",
    mechanism="Generation knobs only: --min-expansions 30, per-board 16 "
              "(attempts cap scales in spr.selfplay).",
    expected_failure="Hard instances at this budget are mostly UNSOLVED -> "
                     "fewer certified records overall, noisier training, "
                     "graded regression (what §12 saw in the base vocab).",
    env={"MINEXP": 30, "PER_BOARD": 16},
    status="parked",  # parked: existing MINEXP knob = incremental; superseded by v12_frontier_curriculum
)
