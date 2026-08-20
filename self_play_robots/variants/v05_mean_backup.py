"""Visit-weighted MEAN backup during generation (control: min backup).

Min backup is the right semantics for a deterministic single agent at
INFERENCE (the subtree's best certified cost is achievable). During GENERATION
it makes Q-estimates -- and therefore visit allocation and the labels'
completion quality -- hostage to single lucky rollouts; mean backup (AlphaZero's
choice) averages over the subtree and explores more evenly. Benches stay
min-backup arena MCTS/A*: the variant only changes what the DATA looks like.
"""
from variants import Variant

VARIANT = Variant(
    vid="v05_mean_backup",
    axis="search",
    title="Mean backup in generation",
    hypothesis="Mean backup spreads generation visits more evenly over "
               "candidates, yielding better-calibrated labels and a policy "
               "that ranks second-best candidates more accurately.",
    mechanism="Generation knob only: spr.selfplay --backup mean "
              "(BACKUP env in the runner).",
    expected_failure="Mean over a min-structure objective underestimates good "
                     "branches -> worse completions, noisier labels, flat "
                     "or worse benches.",
    env={"BACKUP": "mean"},
    status="parked",  # parked: existing knob = incremental
)
