#!/usr/bin/env python3
"""Golden-corpus dumper for rust_datagen gates 1-2 (graph + tables + slides).

Run from `supervised_valuenet/` with PYTHONPATH=. and the ph_main conda python
(networkx 3.3):

    PY=/home/p23131/.conda/envs/ph_main/bin/python3
    PYTHONPATH=. $PY rust_datagen/pyref/dump_reference.py \
        --stock --ids 0 1 --tables --slides 200 --seed 1 --gzip --out DIR
    PYTHONPATH=. $PY rust_datagen/pyref/dump_reference.py \
        --fresh --n 24 --count 4 --seed 1 --tables --slides 200 --gzip --out DIR

Modes
-----
--stock            dump env_{i}.pkl boards AS PICKLED (graph is NOT rebuilt).
                   Stock pickles store dependent edges at weight 100; the dump
                   keeps that raw weight and records `dependent_edge_weight`
                   (the re-weighting `GridEnv.from_env` applies) so the Rust
                   side can normalise: expected weight = dependent_edge_weight
                   for dependent edges, raw weight otherwise.
--fresh            generate boards via nn.gen_grids (gen_walls + build_graph).
                   RR_GRID is set from --n BEFORE the module import (it reads
                   the env var at import time) — one size per process.
--tables           add canonical SHA-256 hashes of the all-pairs table
                   (recomputed at --weight exactly as GridEnv.from_env does:
                   re-weight dependent edges, Dijkstra path lengths, sorted
                   node loops) and of the independent-paths table
                   (nn.gen_grids.independent_paths semantics). Hashes are
                   streamed with hashlib; full tables are never serialised.
--slides K         K random slide cases per board (pos, dir, blockers -> stop
                   via simulate.slide).

Output: one JSON file per board (gzipped with --gzip). Edges are sorted by
(ux, uy, vx, vy); grid_data is dumped verbatim.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import random
import sys
from math import isqrt
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stock", action="store_true",
                      help="dump pickled stock boards (environments/env_{i}.pkl)")
    mode.add_argument("--fresh", action="store_true",
                      help="generate boards with nn.gen_grids at size --n")
    p.add_argument("--ids", type=int, nargs="*", default=None,
                   help="stock env ids (default 0..127)")
    p.add_argument("--env-dir", default="environments",
                   help="stock board directory (relative to cwd)")
    p.add_argument("--n", type=int, default=16, help="fresh board size")
    p.add_argument("--count", type=int, default=1, help="number of fresh boards")
    p.add_argument("--start", type=int, default=0, help="fresh board index offset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tables", action="store_true", help="dump table hashes")
    p.add_argument("--tables-limit", type=int, default=None,
                   help="hash tables only for the first M boards of this run")
    p.add_argument("--slides", type=int, default=0, metavar="K",
                   help="random slide cases per board")
    p.add_argument("--weight", type=int, default=2,
                   help="dependent_edge_weight (GridEnv.from_env default 2)")
    p.add_argument("--gzip", action="store_true", help="gzip the JSON dumps")
    p.add_argument("--procs", type=int, default=1,
                   help="parallel worker processes (fresh mode; capped at 16)")
    p.add_argument("--out", required=True, help="output directory")
    return p.parse_args()


ARGS = parse_args()
if ARGS.fresh:
    # nn.gen_grids reads RR_GRID at import time — must be set BEFORE the import.
    os.environ["RR_GRID"] = str(ARGS.n)

import networkx as nx  # noqa: E402

from simulate import DIRECTIONS, slide, wall_sets  # noqa: E402

if ARGS.fresh:
    from nn.gen_grids import GRID, build_graph, gen_walls  # noqa: E402
    assert GRID == ARGS.n, f"RR_GRID not honoured: GRID={GRID} != --n {ARGS.n}"


def sorted_edge_rows(G):
    """[ux, uy, vx, vy, weight, sx|None, sy|None] rows, sorted (ux,uy,vx,vy).

    Weight is dumped as stored (stock pickles: dependent weight 100)."""
    rows = []
    for u, v, d in G.edges(data=True):
        w = d["weight"]
        assert w == int(w), f"non-integer edge weight {w!r}"
        dep = d.get("dependent")
        if dep is None:
            rows.append([u[0], u[1], v[0], v[1], int(w), None, None])
        else:
            rows.append([u[0], u[1], v[0], v[1], int(w), dep[0], dep[1]])
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    return rows


def _hash_lengths(nodes_sorted, per_source_lengths):
    """SHA-256 over the canonical text "sx,sy,tx,ty,D\\n", lexicographic (s,t).

    D is the integer distance or the literal `None` (unreachable)."""
    h = hashlib.sha256()
    for s in nodes_sorted:
        sl = per_source_lengths(s)
        for t in nodes_sorted:
            d = sl.get(t)
            if d is None:
                h.update(f"{s[0]},{s[1]},{t[0]},{t[1]},None\n".encode())
            else:
                assert d == int(d), f"non-integer distance {d!r}"
                h.update(f"{s[0]},{s[1]},{t[0]},{t[1]},{int(d)}\n".encode())
    return h.hexdigest()


def table_hashes(G, weight_value):
    """(all_pairs_sha256, independent_sha256) at dependent weight `weight_value`.

    all_pairs mirrors GridEnv.from_env: re-weight dependent edges to
    weight_value, then Dijkstra path lengths over sorted node loops.
    (from_env uses nx.all_pairs_dijkstra_path_length, which just runs
    single_source_dijkstra_path_length per node — computed per source here so
    the full table is never held in memory.)
    independent mirrors nn.gen_grids.independent_paths: Dijkstra over the
    subgraph WITHOUT dependent edges, stored weights (all 1).
    """
    for _, _, d in G.edges(data=True):
        if "dependent" in d:
            d["weight"] = weight_value
    nodes_sorted = sorted(G.nodes())
    ap = _hash_lengths(
        nodes_sorted,
        lambda s: nx.single_source_dijkstra_path_length(G, s, weight="weight"))
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from((u, v, d) for u, v, d in G.edges(data=True)
                     if "dependent" not in d)
    ind = _hash_lengths(
        nodes_sorted,
        lambda s: nx.single_source_dijkstra_path_length(H, s, weight="weight"))
    return ap, ind


def slide_cases(grid_data, n, k, rng):
    """K random cases: (pos, dir, blockers subset of cells\\{pos}) -> stop."""
    wr, wd = wall_sets(grid_data, n)
    cells = [(x, y) for y in range(n) for x in range(n)]
    cases = []
    for _ in range(k):
        pos = (rng.randrange(n), rng.randrange(n))
        d = rng.choice(DIRECTIONS)
        nb = rng.randrange(0, 9)
        blockers = rng.sample([c for c in cells if c != pos], nb)
        stop = slide(pos, d, frozenset(blockers), wr, wd, n)
        cases.append({"pos": list(pos), "dir": d,
                      "blockers": sorted([list(b) for b in blockers]),
                      "stop": list(stop)})
    return cases


def dump_board(board_id, env_id, n, grid_data, G, ordinal, args, out_dir):
    all_cells = {(x, y) for y in range(n) for x in range(n)}
    assert set(G.nodes()) == all_cells, f"{board_id}: graph nodes != all cells"
    doc = {
        "board_id": board_id,
        "env_id": env_id,
        "n": n,
        "dependent_edge_weight": args.weight,
        "grid_data": list(grid_data),
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "edges": sorted_edge_rows(G),
    }
    if args.tables and (args.tables_limit is None or ordinal < args.tables_limit):
        ap, ind = table_hashes(G, args.weight)  # mutates dependent weights (last use of G)
        doc["all_pairs_sha256"] = ap
        doc["independent_sha256"] = ind
    if args.slides > 0:
        rng = random.Random(args.seed * 1_000_003 + (env_id if env_id is not None
                                                     else args.start + ordinal))
        doc["slides"] = slide_cases(grid_data, n, args.slides, rng)
    payload = json.dumps(doc, separators=(",", ":")).encode()
    path = out_dir / (board_id + (".json.gz" if args.gzip else ".json"))
    if args.gzip:
        # mtime=0 -> byte-stable archives for identical inputs
        with open(path, "wb") as f:
            with gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as gz:
                gz.write(payload)
    else:
        path.write_bytes(payload)
    return path


def do_stock_board(ordinal_and_id, args, out_dir):
    ordinal, env_id = ordinal_and_id
    with open(Path(args.env_dir) / f"env_{env_id}.pkl", "rb") as f:
        env = pickle.load(f)
    grid_data = env["grid_data"]
    n = isqrt(len(grid_data))
    assert n * n == len(grid_data)
    G = env["grid_graph"]  # dumped as pickled — never rebuilt
    return dump_board(f"stock_{env_id}", env_id, n, grid_data, G,
                      ordinal, args, out_dir)


def do_fresh_board(ordinal, args, out_dir):
    idx = args.start + ordinal
    # Mirrors nn.gen_grids.main board seeding: random.Random(seed*100000 + idx)
    rng = random.Random(args.seed * 100000 + idx)
    grid_data = gen_walls(rng)
    G = build_graph(grid_data)
    return dump_board(f"fresh{args.n}_s{args.seed}_{idx}", None, args.n,
                      grid_data, G, ordinal, args, out_dir)


def main():
    args = ARGS
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.stock:
        ids = args.ids if args.ids else list(range(128))
        work = list(enumerate(ids))
        worker = lambda item: do_stock_board(item, args, out_dir)  # noqa: E731
    else:
        work = list(range(args.count))
        worker = lambda item: do_fresh_board(item, args, out_dir)  # noqa: E731

    procs = max(1, min(args.procs, 16))  # shared box: never above 16
    if procs == 1:
        for i, item in enumerate(work):
            path = worker(item)
            print(f"[{i + 1}/{len(work)}] {path}", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=procs) as ex:
            futs = [(item, ex.submit(_pool_worker, item, args)) for item in work]
            for i, (item, fut) in enumerate(futs):
                print(f"[{i + 1}/{len(work)}] {fut.result()}", flush=True)
    print(f"done: {len(work)} boards -> {out_dir}")


def _pool_worker(item, args):
    out_dir = Path(args.out)
    if args.stock:
        return str(do_stock_board(item, args, out_dir))
    return str(do_fresh_board(item, args, out_dir))


if __name__ == "__main__":
    sys.exit(main())
