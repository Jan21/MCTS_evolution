"""Solve + benchmark the move-based learned planner.

Hosts the two-headed `MoveNet` inside search, mirroring `eval/end2end.py` but over
primitive moves instead of subgoals:

- POLICY head orders/prunes the <=16 legal `(robot, direction)` moves at a node.
- VALUE head scores a resulting board state (predicted optimal cost-to-go), used as
  the A\\* priority `f = g + value(child)`.
- Goal test: the target robot stands on the target cell.

Three planners are compared against the exact optimum (`nn.move_bfs.solve`):
  * NN A\\*        -- best-first `g + value`, policy top-k expansion, closed set.
  * NN greedy(pol)-- follow the policy argmax (beam 1).
  * NN greedy(val)-- step to the child with the smallest predicted value (beam 1).

    # one puzzle, printed
    python -m move_planner.evaluate --ckpt <best.ckpt> --demo --boards 2400 --seed 1
    # benchmark
    python -m move_planner.evaluate --ckpt <best.ckpt> --boards 2400-2599 --per-board 4 \
        --k 5 --astar-iters 2000
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import random
import statistics
import time

import torch

from move_planner.net import MoveNet
from move_planner.encode import x257, dest_cells, NUM_SLOTS, _ix
from train.looped_pc import _adj
from train.encode import walls_for
from nn.gen_grids import GRID
from move_planner.state import legal_moves, is_goal, COLOR_ORDER, NUM_ROBOTS
from move_planner import oracle as move_bfs

INF = 1 << 30
DIRNAMES = ("up", "down", "left", "right")


# -- net wrapper --------------------------------------------------------------

class Guide:
    """Batched value + policy queries over board states for one env."""

    def __init__(self, ckpt, device="cpu"):
        self.model = MoveNet.load_from_checkpoint(ckpt, map_location=device)
        self.model.eval().to(device)
        self.device = device
        # slide planes present iff in_channels == robots+2+8 (from hparams, so a
        # ckpt stays self-describing under any env config; legacy ckpts lack the
        # robots hparam and fall back to the process default).
        robots = self.model.hparams.get("robots", NUM_SLOTS)
        self.with_slide = self.model.hparams.in_channels == robots + 2 + 8

    def _batch(self, env_id, states, target_idx, target):
        A_all, A_ind = _adj(env_id)
        xs, rc, dc, vc = [], [], [], []
        for pos in states:
            rec = {"env_id": env_id, "robots": [list(p) for p in pos],
                   "target": list(target), "target_idx": target_idx}
            xs.append(x257(rec, self.with_slide))
            rc.append(torch.tensor([_ix(p) for p in pos], dtype=torch.long))
            dc.append(dest_cells(rec))
            vc.append(torch.tensor([_ix(target), _ix(pos[target_idx])], dtype=torch.long))
        n = len(states)
        return dict(
            x=torch.stack(xs).to(self.device),
            A_all=A_all[None].expand(n, -1, -1).to(self.device),
            A_ind=A_ind[None].expand(n, -1, -1).to(self.device),
            robot_cells=torch.stack(rc).to(self.device),
            dest_cells=torch.stack(dc).to(self.device),
            val_cells=torch.stack(vc).to(self.device),
        )

    @torch.no_grad()
    def eval_states(self, env_id, states, target_idx, target):
        """Return (values[N] float, policy_logits[N,4,4] float) for board states."""
        b = self._batch(env_id, states, target_idx, target)
        vlog, plog = self.model(b)
        return self.model._value(vlog).cpu().tolist(), plog.cpu()


# -- planners -----------------------------------------------------------------

def _ordered_moves(plog_row, succ, k):
    """Legal successors ordered by policy logit (desc); keep top-k."""
    scores = [float(plog_row[s, d]) for (s, d, _) in succ]
    order = sorted(range(len(succ)), key=lambda i: -scores[i])
    return [succ[i] for i in order[:k]]


def nn_astar(guide, env_id, start, target_idx, target, wr, wd, k=5, max_iters=2000):
    if is_goal(start, target_idx, target):
        return 0, []
    cnt = itertools.count()
    v0, _ = guide.eval_states(env_id, [start], target_idx, target)
    frontier = [(v0[0], next(cnt), 0, start, [])]
    best_g = {start: 0}
    iters = 0
    while frontier and iters < max_iters:
        f, _, g, cur, path = heapq.heappop(frontier)
        if is_goal(cur, target_idx, target):
            return g, path
        if g > best_g.get(cur, INF):
            continue
        iters += 1
        succ = legal_moves(cur, wr, wd, GRID)
        if not succ:
            continue
        _, plog = guide.eval_states(env_id, [cur], target_idx, target)
        kids = _ordered_moves(plog[0], succ, k)
        vals, _ = guide.eval_states(env_id, [c for (_, _, c) in kids], target_idx, target)
        for (s, d, child), v in zip(kids, vals):
            ng = g + 1
            if ng < best_g.get(child, INF):
                best_g[child] = ng
                heapq.heappush(frontier, (ng + v, next(cnt), ng, child, path + [(s, d)]))
    return None, None


def nn_greedy_policy(guide, env_id, start, target_idx, target, wr, wd, max_steps=40):
    cur, path = start, []
    for _ in range(max_steps):
        if is_goal(cur, target_idx, target):
            return len(path), path
        succ = legal_moves(cur, wr, wd, GRID)
        if not succ:
            return None, None
        _, plog = guide.eval_states(env_id, [cur], target_idx, target)
        s, d, child = max(succ, key=lambda m: float(plog[0][m[0], m[1]]))
        cur = child
        path.append((s, d))
    return (len(path), path) if is_goal(cur, target_idx, target) else (None, None)


def nn_greedy_value(guide, env_id, start, target_idx, target, wr, wd, max_steps=40):
    cur, path, seen = start, [], {start}
    for _ in range(max_steps):
        if is_goal(cur, target_idx, target):
            return len(path), path
        succ = legal_moves(cur, wr, wd, GRID)
        if not succ:
            return None, None
        vals, _ = guide.eval_states(env_id, [c for (_, _, c) in succ], target_idx, target)
        order = sorted(range(len(succ)), key=lambda i: vals[i])
        nxt = next((succ[i] for i in order if succ[i][2] not in seen), None)
        if nxt is None:
            return None, None
        cur = nxt[2]
        seen.add(cur)
        path.append((nxt[0], nxt[1]))
    return (len(path), path) if is_goal(cur, target_idx, target) else (None, None)


# -- instance sampling --------------------------------------------------------

def sample_instance(wr, wd, rng, max_try=200):
    """A random solvable instance on a board: returns (positions, target_idx, target, d*)."""
    cells_all = [(x, y) for y in range(GRID) for x in range(GRID)]
    for _ in range(max_try):
        cells = rng.sample(cells_all, NUM_ROBOTS + 1)
        positions = tuple(cells[:NUM_ROBOTS])
        target = cells[NUM_ROBOTS]
        tidx = rng.randrange(NUM_ROBOTS)
        hd = move_bfs.relaxed_target_dist(target, wr, wd, GRID)
        if hd.get(positions[tidx], INF) >= INF:
            continue
        d = move_bfs.solve(positions, tidx, target, wr, wd, GRID, hd, max_expansions=60_000)
        if d is not None:
            return positions, tidx, target, d
    return None


def parse_boards(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _fmt_plan(path):
    return " ".join(f"{COLOR_ORDER[s]}-{DIRNAMES[d]}" for s, d in path)


def demo(guide, boards, seed, k, astar_iters):
    rng = random.Random(seed)
    env_id = boards[0]
    wr, wd = walls_for(env_id)
    inst = sample_instance(wr, wd, rng)
    if inst is None:
        print("no solvable instance sampled"); return
    positions, tidx, target, d = inst
    print(f"board {env_id}  target={COLOR_ORDER[tidx]}->{target}")
    print(f"robots: " + ", ".join(f"{COLOR_ORDER[i]}@{p}" for i, p in enumerate(positions)))
    print(f"optimal cost (A* oracle) = {d}")
    ac, apath = nn_astar(guide, env_id, positions, tidx, target, wr, wd, k, astar_iters)
    gc, gpath = nn_greedy_policy(guide, env_id, positions, tidx, target, wr, wd)
    vc, vpath = nn_greedy_value(guide, env_id, positions, tidx, target, wr, wd)
    print(f"NN A*          : cost={ac}  regret={None if ac is None else ac-d}  plan: {_fmt_plan(apath) if apath else '-'}")
    print(f"NN greedy(pol) : cost={gc}  regret={None if gc is None else gc-d}")
    print(f"NN greedy(val) : cost={vc}  regret={None if vc is None else vc-d}")


def benchmark(guide, boards, per_board, seed, k, astar_iters):
    rng = random.Random(seed)
    reg = {"astar": [], "gp": [], "gv": []}
    fail = {"astar": 0, "gp": 0, "gv": 0}
    n = 0
    t0 = time.time()
    for bi, env_id in enumerate(boards):
        wr, wd = walls_for(env_id)
        for _ in range(per_board):
            inst = sample_instance(wr, wd, rng)
            if inst is None:
                continue
            positions, tidx, target, d = inst
            n += 1
            ac, _ = nn_astar(guide, env_id, positions, tidx, target, wr, wd, k, astar_iters)
            gc, _ = nn_greedy_policy(guide, env_id, positions, tidx, target, wr, wd)
            vc, _ = nn_greedy_value(guide, env_id, positions, tidx, target, wr, wd)
            for key, c in (("astar", ac), ("gp", gc), ("gv", vc)):
                if c is None:
                    fail[key] += 1
                else:
                    reg[key].append(c - d)
        if (bi + 1) % 10 == 0:
            print(f"[{bi+1}/{len(boards)}] instances={n} ({time.time()-t0:.0f}s)", flush=True)

    def line(key, name):
        r = reg[key]
        solved = len(r)
        mr = statistics.mean(r) if r else float("nan")
        opt = sum(x == 0 for x in r)
        print(f"  {name:<16} solved {solved}/{n}  mean_regret={mr:.3f}  "
              f"optimal={opt}/{solved}  fail={fail[key]}")
    print(f"\n=== move-based benchmark: {n} instances over {len(boards)} boards ===")
    line("astar", f"NN A* (k={k})")
    line("gp", "NN greedy(pol)")
    line("gv", "NN greedy(val)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--boards", default="2400-2449")
    p.add_argument("--per-board", type=int, default=4)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--astar-iters", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--demo", action="store_true")
    a = p.parse_args()
    guide = Guide(a.ckpt, a.device)
    boards = parse_boards(a.boards)
    if a.demo:
        demo(guide, boards, a.seed, a.k, a.astar_iters)
    else:
        benchmark(guide, boards, a.per_board, a.seed, a.k, a.astar_iters)


if __name__ == "__main__":
    main()
