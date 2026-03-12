"""
Edits each env pickle:
1. Changes dependent edge weights from 100 to 2 in grid_graph
2. Recomputes all_paths using the updated graph
"""

import pickle
import networkx as nx
from pathlib import Path

ENV_DIR = Path("environments")

for i in range(128):
    path = ENV_DIR / f"env_{i}.pkl"
    with open(path, "rb") as f:
        env = pickle.load(f)

    g = env["grid_graph"]

    # Update dependent edge weights to 2
    for u, v, d in g.edges(data=True):
        if "dependent" in d:
            d["weight"] = 2

    # Recompute all_paths (shortest paths on updated graph)
    all_paths = {}
    nodes = sorted(g.nodes())
    lengths = dict(nx.all_pairs_dijkstra_path_length(g, weight="weight"))
    for src in nodes:
        src_lengths = lengths.get(src, {})
        for dst in nodes:
            all_paths[(src, dst)] = src_lengths.get(dst, None)

    env["all_paths"] = all_paths

    with open(path, "wb") as f:
        pickle.dump(env, f)

    print(f"env_{i}.pkl done")

print("All environments updated.")
