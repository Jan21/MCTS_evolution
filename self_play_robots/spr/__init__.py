"""self_play_robots (spr): AlphaZero-style self-play loop for Ricochet Robots.

Code here CALLS the supervised stack in `supervised_valuenet/` (on PYTHONPATH,
never forked): the physics (`simulate.py`), the plan language
(`skeleton/`), the realizer (`eval/realize.py`), the arena (`eval/compare.py`)
and the size-free value net (`nn_labeler/`). See PROBLEM.md next to this
package for the brief and README.md for the module map.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent            # self_play_robots/spr
SPR = HERE.parent                                  # self_play_robots/
REPO = SPR.parent                                  # MCTS_evolution/
SV = REPO / "supervised_valuenet"                  # the supervised stack
RESULTS = SPR / "results"
ASSETS = SPR / "assets"
