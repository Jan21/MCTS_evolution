"""LazyGridEnv: GridEnv with on-demand caches, identical semantics.

`GridEnv.__init__` eagerly builds three caches, each entry entailing at least
one full `G.copy()` -- hours at 64x64 (DESIGN.md section 9 note). This subclass
computes exactly the same entries lazily, mirroring the `__init__` bodies per
key:

- `precomputed_final_components[goal]`      (goal -> ancestors on the
  dependent-edge-free graph),
- `_dependent_edge_cache.get((bn, sup))`    ('edges' immediately from the
  grouped scan; 'final_component' -- the expensive extended-graph ancestors --
  only on access, which only the pairs cache ever does),
- `_bottleneck_support_pairs_cache.get(key)` (collected from the final
  component on demand; a key eager would NOT have yields None, sending
  `propose_subgoal_states` down its existing recompute branch, which
  DESIGN section 5.9 documents as semantically identical).

Every computation calls the REAL GridEnv methods (`_compute_final_component`,
`get_extended_graph`, `remove_dependent_edges`,
`_collect_bottleneck_support_pairs`) with the same inputs `__init__` would
use, so results -- including set iteration order, which leaks into candidate
enumeration order -- are bit-identical to the eager build from the same
graph. Proven empirically by prove_lazy_env.py.

Only valid for datagen: max_final_component_distance is pinned to None
(DESIGN section 5.6 asserts this for every work item).
"""
from __future__ import annotations

from collections import defaultdict

from GridEnv import GridEnv

_MISS = object()


class _FinalComponents:
    """Lazy mirror of the precomputed_final_components dict."""

    def __init__(self, env):
        self._env = env
        self._memo = {}

    def __getitem__(self, goal):
        v = self._memo.get(goal, _MISS)
        if v is _MISS:
            env = self._env
            v = env._compute_final_component(env._independent_G, goal)
            self._memo[goal] = v
        return v

    def get(self, goal, default=None):
        if goal not in self._env.G.nodes():
            return default
        return self[goal]


class _LazyDepEntry:
    """One _dependent_edge_cache entry; 'final_component' computed on access."""
    __slots__ = ("_env", "_key", "_edges", "_fc")

    def __init__(self, env, key, edges):
        self._env = env
        self._key = key
        self._edges = edges
        self._fc = _MISS

    def __getitem__(self, k):
        if k == "edges":
            return self._edges
        if k == "final_component":
            if self._fc is _MISS:
                env = self._env
                bottleneck_pos, support_pos = self._key
                ext_G = env.get_extended_graph(support_pos, bottleneck_pos)
                ext_G_independent = env.remove_dependent_edges(ext_G)
                self._fc = env._compute_final_component(
                    ext_G_independent, bottleneck_pos,
                    is_extended_graph=True, blocker_pos=support_pos)
                del ext_G_independent
                del ext_G
            return self._fc
        raise KeyError(k)


class _DepCache:
    """Lazy mirror of _dependent_edge_cache (keys = grouped dependent edges)."""

    def __init__(self, env, grouped):
        self._entries = {k: _LazyDepEntry(env, k, edges)
                         for k, edges in grouped.items()}

    def get(self, key, default=None):
        return self._entries.get(key, default)

    def __getitem__(self, key):
        return self._entries[key]

    def __contains__(self, key):
        return key in self._entries


class _PairsCache:
    """Lazy mirror of _bottleneck_support_pairs_cache.

    Returns None for keys the eager cache would not contain, so callers fall
    through to their existing recompute branches exactly as with eager.
    """
    _ABSENT = object()

    def __init__(self, env):
        self._env = env
        self._memo = {}

    def get(self, key, default=None):
        v = self._memo.get(key, _MISS)
        if v is _MISS:
            v = self._compute(key)
            self._memo[key] = v
        return default if v is self._ABSENT else v

    def _compute(self, key):
        env = self._env
        goal, support_pos = key
        if support_pos is None:
            has_independent_in_edge = any(
                "dependent" not in d
                for _, _, d in env.G.in_edges(goal, data=True))
            if not has_independent_in_edge:
                return self._ABSENT
            return env._collect_bottleneck_support_pairs(
                env.precomputed_final_components[goal] | {goal})
        entry = env._dependent_edge_cache.get(key)
        if entry is None:
            return self._ABSENT
        return env._collect_bottleneck_support_pairs(
            entry["final_component"] | {goal})


class LazyGridEnv(GridEnv):
    def __init__(self, grid_graph, independent_paths, all_paths):
        # Deliberately NOT calling super().__init__ (that is the eager build).
        self.G = grid_graph
        self.reachability_matrix = independent_paths
        self.relaxed_reachability_matrix = all_paths
        self.max_final_component_distance = None

        # Same expression __init__ evaluates once for all goals.
        self._independent_G = self.remove_dependent_edges(self.G.copy())

        self._wall_nodes = frozenset(
            node for node in self.G.nodes()
            if any(d["weight"] == 1
                   for _, _, d in self.G.in_edges(node, data=True)))

        grouped = defaultdict(list)
        for u, v, d in self.G.edges(data=True):
            if "dependent" in d:
                grouped[(v, d["dependent"])].append((u, v, d))

        self.precomputed_final_components = _FinalComponents(self)
        self._dependent_edge_cache = _DepCache(self, dict(grouped))
        self._bottleneck_support_pairs_cache = _PairsCache(self)
