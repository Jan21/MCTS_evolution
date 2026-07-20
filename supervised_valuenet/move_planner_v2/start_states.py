"""Oracle-free start-state generation for self-play (forward-walk + goal-relabel).

The generator is DeepCubeA's recipe adapted for a **non-reversible** move space
(Ricochet Robots slides don't invert -- reverse-scramble stays local to the goal and
only makes ~1-move puzzles, so it was replaced):

    sample a random full board S0, apply `k` random forward slides, set the GOAL to the
    cell the target robot ends up on.

The forward walk is itself a witness that a length-<=`k` solution exists (replaying it
lands the target on the goal), so `k ~ U(1, walk_k_max)` is an oracle-free, genuinely
NON-local difficulty dial: small `k` = near-goal (bootstraps a random net), large `k`
approaches the full-difficulty eval distribution. No curriculum ladder, no solve-rate
gate -- unsolved instances are simply dropped, so the solvable frontier ratchets outward
on its own (value iteration is the implicit curriculum). The target always stops on the
goal via a slide, so the goal is guaranteed stoppable -- no reachability filter needed.
"""
from __future__ import annotations

from move_planner.state import legal_moves, is_goal, NUM_ROBOTS  # noqa: F401
from move_planner.oracle import relaxed_target_dist, INF
from train.encode import walls_for
from nn.gen_grids import GRID

_CELLS_ALL = [(x, y) for y in range(GRID) for x in range(GRID)]


def forward_walk_relabel(env_id: int, k: int, rng, wr, wd, target_bias: float = 0.6) -> tuple | None:
    """Random start S0 + `k` forward slides; goal = target robot's final cell.

    `target_bias`: probability of moving the TARGET robot at each step (else a random
    robot). Without bias a random-robot walk moves the target only ~k/4 times, so it
    stays near the goal (mean optimal ~2.5); biasing toward the target makes it travel.

    Returns `(positions, target_idx, goal)` (a solvable instance whose optimal cost is
    <= `k`), or `None` if the target robot did not end up displaced (trivial).
    """
    cells = rng.sample(_CELLS_ALL, NUM_ROBOTS + 1)
    start = tuple(cells[:NUM_ROBOTS])
    tidx = rng.randrange(NUM_ROBOTS)
    cur = start
    for _ in range(k):
        moves = legal_moves(cur, wr, wd, GRID)
        if not moves:
            break
        if rng.random() < target_bias:
            tmoves = [m for m in moves if m[0] == tidx]
            m = rng.choice(tmoves) if tmoves else rng.choice(moves)
        else:
            m = rng.choice(moves)
        cur = m[2]
    goal = cur[tidx]
    if goal == start[tidx]:
        return None  # target robot never displaced -> instance is already solved
    return start, tidx, goal


def sample_solvable_instance(env_id: int, rng, wr, wd, max_try: int = 200) -> tuple | None:
    """A random reachable instance from the *eval* distribution (mirrors
    `evaluate.sample_instance`'s filter WITHOUT the `oracle.solve` call). Optional mix-in
    to tighten coverage on the true test distribution."""
    for _ in range(max_try):
        cells = rng.sample(_CELLS_ALL, NUM_ROBOTS + 1)
        positions = tuple(cells[:NUM_ROBOTS])
        target = cells[NUM_ROBOTS]
        tidx = rng.randrange(NUM_ROBOTS)
        if relaxed_target_dist(target, wr, wd, GRID).get(positions[tidx], INF) < INF:
            return positions, tidx, target
    return None


def sample_start_state(env_id: int, cfg, rng) -> tuple | None:
    """Draw one training instance. With prob `cfg.rand_mix_prob` a raw random
    eval-distribution instance (covers the deep tail on the true test distribution), else
    a target-biased forward-walk puzzle at depth `k ~ U(1, cfg.walk_k_max)` (deep + cheap:
    guaranteed solvable, so the expert wastes no A* budget on failures). Returns
    `(positions, target_idx, target)` or `None`."""
    wr, wd = walls_for(env_id)
    if getattr(cfg, "rand_mix_prob", 0.0) > 0.0 and rng.random() < cfg.rand_mix_prob:
        return sample_solvable_instance(env_id, rng, wr, wd)
    k = rng.randint(1, cfg.walk_k_max)
    return forward_walk_relabel(env_id, k, rng, wr, wd,
                                target_bias=getattr(cfg, "walk_target_bias", 0.6))


def train_board_ids(cfg) -> list[int]:
    """The pkl-backed train-board pool the generator samples from."""
    return cfg.train_ids()
