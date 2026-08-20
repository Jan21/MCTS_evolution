"""Cold-start control: how much does the supervised prior actually matter?

Owner clarification (2026-08-20): bootstrapping from the supervised stack is
allowed but not required -- making "bootstrapped vs from-scratch" itself a
comparison axis. This arm runs the SAME matched iteration but trains both nets
FROM SCRATCH on the iteration's certified records alone (no --init warm start;
value gets the cold recipe: warmup epochs + higher lr; CollapseStop guards the
known cold-value failure mode, FINDINGS §7). Generation still uses the frozen
seed nets (the data must be matched -- this isolates the TRAINING prior).
One iteration of ~3k records will not match a warm-started net; the number to
read is HOW FAR from control it lands (the size of the supervised prior's
contribution at fixed data), not whether it wins.
"""
from variants import Variant

VARIANT = Variant(
    vid="v08_cold_start",
    axis="bootstrap",
    title="From-scratch retrain (no supervised warm start)",
    hypothesis="At one iteration's data volume the supervised prior dominates: "
               "cold nets land far below control on every exam; the gap IS the "
               "measurement.",
    mechanism="Runner knob COLD=1: spr.train without --init (policy fresh; "
              "value fresh with --warmup 4, lr 3e-4, 12+20 epochs). Generation "
              "identical to control (frozen seed nets).",
    expected_failure="Value collapse (CollapseStop fires) or a policy too weak "
                     "to rank B2 candidates -> near-zero frontier solves. "
                     "Still informative: it bounds the prior's value.",
    env={"COLD": 1, "EP": 20},
)
