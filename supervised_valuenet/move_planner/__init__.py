"""Move-based Ricochet-Robots planner: primitive-move policy + optimal-cost value.

A self-contained subpackage that reuses the parent baseline's shared code
(simulate physics, gen_grids boards, looped-transformer encoder, benchmark splits).
Run modules from the supervised_valuenet/ directory, e.g.
    PYTHONPATH=. python -m move_planner.generate ...
"""
