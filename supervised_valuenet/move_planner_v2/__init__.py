"""move_planner_v2: self-play (plain Expert Iteration) for the Ricochet-Robots move planner.

Trains move_planner.net.MoveNet from a random init (pure self-play) or warm from
move_planner/checkpoints/best.ckpt by iterating: the net's own budgeted A* (the expert)
solves generated puzzles and emits value (remaining length) + policy (committed move)
targets; unsolved puzzles are dropped; the net is retrained on them. See DESIGN.md.
"""
