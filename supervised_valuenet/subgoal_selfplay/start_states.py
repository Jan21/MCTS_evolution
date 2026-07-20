"""Oracle-free start-state generation for subgoal-planner self-play.

Instances are drawn exactly like the supervised generator draws them
(`nn.generate.random_instance`: distinct random cells for the robots and the target)
plus a cheap relaxed-reachability prefilter (`compute_relaxed_shortest_path_length`
target-robot -> target) so the expert never wastes A* budget on structurally hopeless
boards. The relaxed check is a graph reachability test, not a solver call.

An optional forward-walk-relabel dial (`cfg.walk_relabel_prob`, default OFF) replaces
the random target with the endpoint of a length-k walk on the board's slide graph
starting from the target robot's cell. Walk edges may be dependent slides (they need a
support robot), so the walk is a relaxed-reachability witness rather than a solvability
witness -- it biases instances toward targets the mover can plausibly travel to, which
concentrates early from-scratch iterations on solvable-looking geometry. Unsolved
instances are dropped by the outer loop either way (the implicit curriculum), so the
dial is a sampling bias, never a label source.
"""
from __future__ import annotations

from GridEnv import GridEnv, State
from nn.generate import random_instance


def _relaxed_reachable(env: GridEnv, state: State) -> bool:
    """Cheap prefilter: the mover can reach the target on the relaxed slide graph."""
    return env.compute_relaxed_shortest_path_length(
        state.target_robot.position, state.target, None) is not None


def forward_walk_relabel(env: GridEnv, state: State, k: int, rng) -> State | None:
    """Slide-graph walk of length `k` from the target robot's cell; the endpoint
    becomes the target. Returns None when the walk goes nowhere or ends on a robot."""
    cur = state.target_robot.position
    for _ in range(k):
        nxt = list(env.G.successors(cur))
        if not nxt:
            break
        cur = rng.choice(nxt)
    occupied = {state.target_robot.position} | {h.position for h in state.helpers}
    if cur == state.target_robot.position or cur in occupied:
        return None
    return State(target=cur, target_robot=state.target_robot, helpers=state.helpers)


def random_reachable_instance(env: GridEnv, colors, rng, max_try: int = 50) -> State | None:
    """A raw random instance passing the relaxed-reachability prefilter.

    This is the eval distribution (used by the probe and, by default, by generation)."""
    for _ in range(max_try):
        st = random_instance(env, colors, rng)
        if _relaxed_reachable(env, st):
            return st
    return None


def sample_instance(env: GridEnv, colors, rng, cfg) -> State | None:
    """Draw one training instance. With prob `cfg.walk_relabel_prob` the target is
    relabeled to a forward-walk endpoint (see module docstring); otherwise (default)
    a raw random reachable instance."""
    for _ in range(cfg.sample_max_try):
        st = random_instance(env, colors, rng)
        if cfg.walk_relabel_prob > 0.0 and rng.random() < cfg.walk_relabel_prob:
            st = forward_walk_relabel(env, st, rng.randint(1, cfg.walk_k_max), rng)
            if st is None:
                continue
        if _relaxed_reachable(env, st):
            return st
    return None
