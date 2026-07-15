"""Generate the layered backward-machinery fixtures under golden/backward/.

Five layers, converged one by one by the Rust subgoal port (Agent C):

  (a) paths.jsonl.gz            exact/relaxed shortest-path lengths, incl. the
                                supported (dependent-edge-group) variants, over
                                ~2000 random (start, end, support?) triples.
  (b) propose.jsonl.gz          heuristics.propose candidate lists for ~100
                                (goal, mover, helpers, support?) contexts.
                                Rust compares as MULTISETS of
                                (bn, sp, helper, parent_support, score) — the
                                candidate ORDER is Python-set-iteration order
                                and deliberately NOT reproduced.
  (c) final_components.jsonl.gz plain + extended-graph final components for
                                ~60 keys (compared as sets).
  (d) decisions.jsonl.gz        ~30 replay_backward_decision lines WITH
                                python_labels, produced by pyref/
                                dump_decisions.py (Agent E1) via subprocess.
  (e) rollouts.jsonl.gz         12 complete nn.generate.rollout record streams
                                (max_candidates=14, mi=4000, mf=40000).
                                Rust compares per-decision record multisets.

Boards: 2 each from environments (g16r4, stock), environments_g16r6 and
environments_g24r4. All environments are built from grid_data alone via
pyref/common.build_env (lazy mode — semantics proven identical to the eager
GridEnv by prove_lazy_env.py), so the fixtures are hermetic: the Rust tests
recompile each board from the grid_data stored inline in every fixture file.

    /home/p23131/.conda/envs/ph_main/bin/python3 \
        rust_datagen/pyref/gen_fixtures_backward.py [--skip-decisions]

Reproducibility: all sampling uses fixed seeds; Python set iteration order for
tuples of small ints is stable across runs (unsalted int hashing), so repeated
generation produces identical files.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

PYREF_DIR = Path(__file__).resolve().parent
RD_DIR = PYREF_DIR.parent                    # rust_datagen/
SV_DIR = RD_DIR.parent                       # supervised_valuenet/
for _p in (str(SV_DIR), str(PYREF_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common  # noqa: E402

OUT_DIR = RD_DIR / "golden" / "backward"

# (tag, env_dir relative to supervised_valuenet/, env_id, n, robots)
BOARDS = [
    ("g16r4", "environments", 0, 16, 4),
    ("g16r4", "environments", 1, 16, 4),
    ("g16r6", "environments_g16r6", 900, 16, 6),
    ("g16r6", "environments_g16r6", 901, 16, 6),
    ("g24r4", "environments_g24r4", 900, 24, 4),
    ("g24r4", "environments_g24r4", 901, 24, 4),
]

WEIGHT = 2


class _Timeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _Timeout()


def _grouped_keys(G):
    """Dependent-edge group keys (bottleneck, support) in G.edges order —
    the same insertion order GridEnv.__init__'s grouped dict uses."""
    grouped = defaultdict(list)
    for u, v, d in G.edges(data=True):
        if "dependent" in d:
            grouped[(v, d["dependent"])].append(u)
    return list(grouped.keys())


def _load_boards():
    out = []
    for tag, env_dir, gid, n, robots in BOARDS:
        grid_data = common.load_board_grid_data(SV_DIR / env_dir, gid)
        out.append((tag, gid, n, robots, grid_data))
    return out


def _build_env(grid_data, n):
    return common.build_env(grid_data, n, mode="lazy", weight=WEIGHT,
                            table_workers=16)


def _write_jsonl_gz(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")
    print(f"  wrote {path.name}: {len(lines)} lines, "
          f"{path.stat().st_size / 1024:.0f} KiB")


# -- (a) path lengths ---------------------------------------------------------

def gen_paths(boards, envs):
    lines = []
    for bi, (tag, gid, n, robots, grid_data) in enumerate(boards):
        env = envs[bi]
        rng = random.Random(1000 + bi)
        nodes = list(env.G.nodes())
        gkeys = _grouped_keys(env.G)
        triples = []
        # 150 unsupported pairs
        for _ in range(150):
            s, e = rng.choice(nodes), rng.choice(nodes)
            triples.append((s, e, None))
        # 150 real dependent-group supports
        for _ in range(150):
            e, sup = rng.choice(gkeys)
            s = rng.choice(nodes)
            triples.append((s, e, sup))
        # 50 fake supports (almost always no group -> None)
        for _ in range(50):
            s, e, sup = rng.choice(nodes), rng.choice(nodes), rng.choice(nodes)
            triples.append((s, e, sup))
        rows = []
        for s, e, sup in triples:
            rows.append({
                "s": common.xy(s), "e": common.xy(e), "sup": common.xy(sup),
                "exact": env.compute_exact_shortest_path_length(s, e, sup),
                "relaxed": env.compute_relaxed_shortest_path_length(s, e, sup),
            })
        lines.append({"tag": tag, "board": common.board_obj(gid, n, grid_data),
                      "triples": rows})
    _write_jsonl_gz(OUT_DIR / "paths.jsonl.gz", lines)


# -- (b) propose candidate lists ----------------------------------------------

def _ser_cand_full(c):
    return {"bn": common.xy(c.subgoal.bottleneck.position),
            "sp": common.xy(c.subgoal.support.position),
            "helper": common.ser_robot(c.subgoal.helper),
            "parent_support": common.xy(c.parent_support),
            "score": int(c.score)}


def gen_propose(boards, envs):
    from GridEnv import Robot_at
    from skeleton import heuristics

    lines = []
    for bi, (tag, gid, n, robots, grid_data) in enumerate(boards):
        env = envs[bi]
        rng = random.Random(2000 + bi)
        nodes = list(env.G.nodes())
        colors = common.PALETTE[:robots]
        # goals with no independent in-edge exercise the pairs-cache-miss
        # recompute branch (eager cache has no (goal, None) entry for them)
        goals_no_indep = [
            v for v in nodes
            if env.G.in_degree(v) > 0
            and all("dependent" in d for _, _, d in env.G.in_edges(v, data=True))
        ]
        contexts = []
        for ci in range(14):
            goal = rng.choice(nodes)
            mover_color = rng.choice(colors)
            helper_colors = [c for c in colors if c != mover_color]
            cells = rng.sample(nodes, len(helper_colors) + 1)
            mover = Robot_at(position=cells[0], color=mover_color)
            helpers = [Robot_at(position=cells[i + 1], color=helper_colors[i])
                       for i in range(len(helper_colors))]
            support = None
            if rng.random() < 0.4:
                deps = sorted({d["dependent"]
                               for _, _, d in env.G.in_edges(goal, data=True)
                               if "dependent" in d})
                if deps:
                    support = Robot_at(position=rng.choice(deps),
                                       color=rng.choice(helper_colors))
            contexts.append((goal, mover, helpers, support))
        # 3 extra contexts on no-independent-in-edge goals (recompute branch)
        for ci in range(3):
            if not goals_no_indep:
                break
            goal = rng.choice(goals_no_indep)
            mover_color = rng.choice(colors)
            helper_colors = [c for c in colors if c != mover_color]
            cells = rng.sample(nodes, len(helper_colors) + 1)
            mover = Robot_at(position=cells[0], color=mover_color)
            helpers = [Robot_at(position=cells[i + 1], color=helper_colors[i])
                       for i in range(len(helper_colors))]
            contexts.append((goal, mover, helpers, None))

        ctx_rows = []
        for goal, mover, helpers, support in contexts:
            cands = heuristics.propose(env, goal, mover, helpers, support)
            ctx_rows.append({
                "goal": common.xy(goal),
                "mover": common.ser_robot(mover),
                "helpers": [common.ser_robot(h) for h in helpers],
                "support": common.ser_robot(support) if support else None,
                "n_candidates": len(cands),
                "candidates": [_ser_cand_full(c) for c in cands],
            })
        lines.append({"tag": tag, "board": common.board_obj(gid, n, grid_data),
                      "contexts": ctx_rows})
    _write_jsonl_gz(OUT_DIR / "propose.jsonl.gz", lines)


# -- (c) final components -----------------------------------------------------

def gen_final_components(boards, envs):
    lines = []
    for bi, (tag, gid, n, robots, grid_data) in enumerate(boards):
        env = envs[bi]
        rng = random.Random(3000 + bi)
        nodes = list(env.G.nodes())
        gkeys = _grouped_keys(env.G)
        goals = rng.sample(nodes, 5)
        keys = rng.sample(gkeys, 5)
        grows = [{"goal": common.xy(g),
                  "fc": sorted(common.xy(c)
                               for c in env.precomputed_final_components[g])}
                 for g in goals]
        erows = []
        for bn, sup in keys:
            fc = env._dependent_edge_cache[(bn, sup)]["final_component"]
            erows.append({"bn": common.xy(bn), "sup": common.xy(sup),
                          "fc": sorted(common.xy(c) for c in fc)})
        lines.append({"tag": tag, "board": common.board_obj(gid, n, grid_data),
                      "goals": grows, "extended": erows})
    _write_jsonl_gz(OUT_DIR / "final_components.jsonl.gz", lines)


# -- (d) replay decision dumps (via Agent E1's dump_decisions.py) -------------

def gen_decisions(per_config=12):
    lines = []
    for cfg in ("g16r4", "g16r6", "g24r4"):
        tmp = OUT_DIR / f"_tmp_decisions_{cfg}.jsonl"
        cmd = [sys.executable, str(PYREF_DIR / "dump_decisions.py"),
               "--config", cfg, "--task", "backward",
               "--instances", "12", "--n-boards", "3", "--seed", "7",
               "--workers", "3", "--out", str(tmp)]
        print(f"  [decisions] {cfg}: running dump_decisions.py ...")
        t0 = time.time()
        subprocess.run(cmd, check=True, cwd=str(SV_DIR))
        kept = 0
        with open(tmp) as f:
            for raw in f:
                raw = raw.strip()
                if raw and kept < per_config:
                    lines.append(json.loads(raw))
                    kept += 1
        tmp.unlink()
        print(f"  [decisions] {cfg}: kept {kept} lines "
              f"({time.time() - t0:.0f}s)")
    _write_jsonl_gz(OUT_DIR / "decisions.jsonl.gz", lines)


# -- (e) complete rollouts ----------------------------------------------------

def gen_rollouts(boards, envs, per_board=2, timeout_s=300):
    from skeleton.astar import AStar
    from nn.generate import rollout, random_instance

    signal.signal(signal.SIGALRM, _raise_timeout)
    lines = []
    for bi, (tag, gid, n, robots, grid_data) in enumerate(boards):
        env = envs[bi]
        rng = random.Random(5000 + bi)
        colors = common.PALETTE[:robots]
        solver = AStar(max_iters=4000, max_frontier=40_000)
        kept = attempts = 0
        while kept < per_board and attempts < per_board * 6:
            attempts += 1
            st = random_instance(env, colors, rng)
            signal.alarm(timeout_s)
            try:
                t0 = time.time()
                recs = rollout(env, st, solver, gid, max_candidates=14)
            except _Timeout:
                print(f"  [rollouts] {tag}/e{gid} attempt {attempts}: timeout")
                recs = []
            except Exception as ex:
                print(f"  [rollouts] {tag}/e{gid} attempt {attempts}: {ex!r}")
                recs = []
            finally:
                signal.alarm(0)
            if not recs:
                continue
            kept += 1
            print(f"  [rollouts] {tag}/e{gid} attempt {attempts}: "
                  f"{len(recs)} records ({time.time() - t0:.1f}s)")
            lines.append({
                "tag": tag, "board": common.board_obj(gid, n, grid_data),
                "robots": robots, "state": common.ser_state(st),
                "max_candidates": 14, "max_iters": 4000,
                "max_frontier": 40_000, "records": recs,
            })
    _write_jsonl_gz(OUT_DIR / "rollouts.jsonl.gz", lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--skip-decisions", action="store_true",
                   help="skip layer (d) (dump_decisions.py subprocess)")
    p.add_argument("--only", default=None,
                   help="comma list of layers to build (a,b,c,d,e)")
    a = p.parse_args()
    layers = set(a.only.split(",")) if a.only else {"a", "b", "c", "d", "e"}
    if a.skip_decisions:
        layers.discard("d")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    boards = _load_boards()
    need_env = layers & {"a", "b", "c", "e"}
    envs = None
    if need_env:
        print("[envs] building GridEnvs (lazy) ...")
        t0 = time.time()
        envs = [_build_env(gd, n) for _, _, n, _, gd in boards]
        print(f"[envs] done in {time.time() - t0:.1f}s")

    if "a" in layers:
        print("[a] path lengths")
        gen_paths(boards, envs)
    if "b" in layers:
        print("[b] propose candidate lists")
        gen_propose(boards, envs)
    if "c" in layers:
        print("[c] final components")
        gen_final_components(boards, envs)
    if "d" in layers:
        print("[d] replay decision dumps")
        gen_decisions()
    if "e" in layers:
        print("[e] complete rollouts")
        gen_rollouts(boards, envs)
    print("done.")


if __name__ == "__main__":
    main()
