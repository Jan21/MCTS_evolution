"""Generate new Ricochet Robots boards (env_*.pkl) in the exact stored format.

This repo ships 128 precomputed boards and no generator; cross-board
generalization is the value net's real bottleneck, so more *distinct geometries*
help more than more instances on the same boards.

A board is random interior walls (matching the stock density of 48 segments) plus
the border. The slide graph is rebuilt from the walls with simulate.py physics
(independent edge = wall stop; dependent edges = each intermediate stop, helper
at the next cell) -- this builder reproduces the stock graphs edge-for-edge.
All-pairs distances are then precomputed exactly as `GridEnv.from_env` expects.

    python -m nn.gen_grids --n 200 --start 1000 --out environments
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
from pathlib import Path

import networkx as nx

from simulate import wall_sets, slide
from GridEnv import Robot_at

GRID = int(os.environ.get("RR_GRID", "16"))
# Interior wall count scales with board area to keep the stock density (48 @ 16x16).
N_INTERIOR = int(os.environ.get("RR_WALLS", str(round(48 * (GRID / 16) ** 2))))
PALETTE = ["Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Cyan", "Magenta"]
COLORS = PALETTE[:int(os.environ.get("RR_ROBOTS", "4"))]
DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def gen_walls(rng) -> list[str]:
    """Per-cell wall strings: border + N_INTERIOR random interior segments."""
    sides = [set() for _ in range(GRID * GRID)]

    def idx(x, y):
        return y * GRID + x

    # border
    for x in range(GRID):
        sides[idx(x, 0)].add("N")
        sides[idx(x, GRID - 1)].add("S")
    for y in range(GRID):
        sides[idx(0, y)].add("W")
        sides[idx(GRID - 1, y)].add("E")

    # random interior segments (mirrored onto the neighbour)
    placed = 0
    while placed < N_INTERIOR:
        x, y = rng.randrange(GRID), rng.randrange(GRID)
        d = rng.choice("NSEW")
        if d == "E" and x < GRID - 1:
            a, b = "E", "W"; nx_, ny_ = x + 1, y
        elif d == "W" and x > 0:
            a, b = "W", "E"; nx_, ny_ = x - 1, y
        elif d == "S" and y < GRID - 1:
            a, b = "S", "N"; nx_, ny_ = x, y + 1
        elif d == "N" and y > 0:
            a, b = "N", "S"; nx_, ny_ = x, y - 1
        else:
            continue
        if a in sides[idx(x, y)]:
            continue
        sides[idx(x, y)].add(a)
        sides[idx(nx_, ny_)].add(b)
        placed += 1

    return ["".join(sorted(s)) for s in sides]


def build_graph(grid_data) -> nx.DiGraph:
    wr, wd = wall_sets(grid_data, GRID)
    G = nx.DiGraph()
    G.add_nodes_from((x, y) for y in range(GRID) for x in range(GRID))
    for y in range(GRID):
        for x in range(GRID):
            c = (x, y)
            for d, (dx, dy) in DIRS.items():
                stop = slide(c, d, frozenset(), wr, wd, GRID)
                if stop == c:
                    continue
                path, cur = [], c
                while cur != stop:
                    cur = (cur[0] + dx, cur[1] + dy)
                    path.append(cur)
                G.add_edge(c, stop, weight=1)
                for v in path[:-1]:
                    G.add_edge(c, v, weight=2, dependent=(v[0] + dx, v[1] + dy))
    return G


def all_pairs(G, weight) -> dict:
    nodes = list(G.nodes())
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))
    out = {}
    for s in nodes:
        sl = lengths.get(s, {})
        for t in nodes:
            out[(s, t)] = sl.get(t)
    return out


def independent_paths(G) -> dict:
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from((u, v, d) for u, v, d in G.edges(data=True)
                     if "dependent" not in d)
    return all_pairs(H, "weight")


def random_instance(rng):
    pts = rng.sample([(x, y) for y in range(GRID) for x in range(GRID)],
                     len(COLORS) + 1)
    robots = [Robot_at(position=pts[i], color=COLORS[i]) for i in range(len(COLORS))]
    return {"target": pts[-1], "target_robot": robots[0],
            "helper_robots": robots[1:]}


def make_board(graph_idx, rng) -> dict:
    grid_data = gen_walls(rng)
    G = build_graph(grid_data)
    return {
        "graph_idx": graph_idx,
        "grid_data": grid_data,
        "grid_graph": G,
        "instances": [random_instance(rng)],
        "independent_paths": independent_paths(G),
        "all_paths": all_pairs(G, "weight"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--start", type=int, default=1000)
    p.add_argument("--out", default=os.environ.get("RR_ENV_DIR", "environments"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for k in range(a.n):
        idx = a.start + k
        rng = random.Random(a.seed * 100000 + idx)
        board = make_board(idx, rng)
        with open(out / f"env_{idx}.pkl", "wb") as f:
            pickle.dump(board, f)
        if k % 25 == 0:
            print(f"generated env_{idx} ({k + 1}/{a.n})", flush=True)
    print(f"done: {a.n} boards env_{a.start}..env_{a.start + a.n - 1}")


if __name__ == "__main__":
    main()
