"""Stage 2 driver: generate exact goal-conditioned cost data, train, evaluate.

    PYTHONPATH=supervised_valuenet:self_play_robots python -m subgoal.stage2 gen   ...
    ...                                                    subgoal.stage2 train ...
    ...                                                    subgoal.stage2 eval  ...

Cost target, architecture and the two baselines are documented in
`subgoal/costnet.py`. Board-id splits come from `scaling/configs.py::CONFIGS`
["g16r4"]: train "0-95,1000-1799", val "1800-2399", test "112-127,2400-2999",
bench "2400-2549" (the 150 boards of `eval/data/bench450.jsonl`). Training uses
train-split boards ONLY; every evaluation board is a board no training state
ever came from.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPR = HERE.parent
REPO = SPR.parent
SV = REPO / "supervised_valuenet"
for _p in (str(SV), str(SPR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simulate import wall_sets                                       # noqa: E402
from scaling.configs import CONFIGS, parse_ids                       # noqa: E402
from subgoal.costnet import (cost_row, relaxed_dist, lone_robot_cost,  # noqa: E402
                             group_metrics, UNREACHABLE)

CFG = CONFIGS["g16r4"]
ENV_DIR = SV / "environments"
SIZE = 16
ROBOTS = 4
BENCH = SV / "eval/data/bench450.jsonl"

# Baseline B2's stand-in cost for a cell the LONE robot cannot stop on (it needs
# another robot as a stopper). Larger than any lone-robot distance on a 16x16
# board; stated here so the number is auditable rather than tuned.
B2_UNDEFINED = 20.0

# --- the Stage 2 gate, fixed BEFORE any model was trained -------------------
# "beat the baseline on ranking by a clear and stated margin, on boards it never
# trained on": mean per-decision Spearman rho over the truly reachable
# candidates of a state (all four robots pooled -- the set the planner ranks)
# must exceed the BETTER of the two learning-free baselines by at least 0.05,
# and mean top-1 accuracy must not be worse than that baseline's.
GATE_RHO_MARGIN = 0.05

_BOARDS: dict = {}


def board(env_id):
    if env_id not in _BOARDS:
        with open(ENV_DIR / f"env_{env_id}.pkl", "rb") as f:
            grid_data = pickle.load(f)["grid_data"]
        _BOARDS[env_id] = wall_sets(grid_data, SIZE)
    return _BOARDS[env_id]


# ---------------------------------------------------------------------------
# data generation
# ---------------------------------------------------------------------------

def random_state(rng, n=SIZE, robots=ROBOTS):
    cells = rng.sample(range(n * n), robots)
    return [(c % n, c // n) for c in cells]


def walk(positions, steps, wr, wd, rng, n=SIZE):
    """Apply `steps` random macro subgoals (a random robot to a random cell it
    can come to rest on). Produces states a planner actually reaches, not only
    the uniform-random ones."""
    pos = list(positions)
    for _ in range(steps):
        r = rng.randrange(len(pos))
        row = cost_row(pos, r, wr, wd, n)
        cand = np.flatnonzero(row >= 0)
        if len(cand) == 0:
            continue
        c = int(rng.choice(list(cand)))
        pos[r] = (c % n, c // n)
    return pos


def gen(a):
    rng = random.Random(a.seed)
    if a.from_bench:
        rows = [json.loads(l) for l in Path(a.from_bench).read_text().splitlines() if l.strip()]
        states = [(r["env_id"], [tuple(p) for p in r["positions"]]) for r in rows]
    else:
        ids = parse_ids(a.boards)
        states = []
        for eid in ids:
            wr, wd = board(eid)
            for _ in range(a.per_board):
                pos = random_state(rng)
                k = rng.randint(0, a.walk)
                if k:
                    pos = walk(pos, k, wr, wd, rng)
                states.append((eid, pos))
        rng.shuffle(states)
    env_id = np.zeros(len(states), np.int32)
    positions = np.zeros((len(states), ROBOTS, 2), np.int16)
    cost = np.zeros((len(states), ROBOTS, SIZE * SIZE), np.int8)
    t0 = time.time()
    for i, (eid, pos) in enumerate(states):
        wr, wd = board(eid)
        env_id[i] = eid
        positions[i] = pos
        for r in range(ROBOTS):
            cost[i, r] = cost_row(pos, r, wr, wd, SIZE)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / (out.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, env_id=env_id, positions=positions, cost=cost,
                                meta=np.array(json.dumps(dict(
                                boards=a.boards, per_board=a.per_board,
                                walk=a.walk, seed=a.seed,
                                from_bench=a.from_bench, n_states=len(states),
                                size=SIZE, robots=ROBOTS,
                                seconds=round(time.time() - t0, 1),
                                date=time.strftime("%Y-%m-%dT%H:%M:%S")))))
    os.replace(tmp, out)
    reach = (cost >= 0)
    print(f"[gen] {len(states)} states -> {out}  "
          f"({reach.mean() * 100:.1f}% of the {ROBOTS * SIZE * SIZE} candidates per "
          f"state reachable, mean cost {cost[reach].mean():.2f}, "
          f"max cost {cost.max()}, {time.time() - t0:.1f}s)")
    print(f"SPR SUBGOAL STAGE2 GEN DONE {out}")


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train(a):
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from subgoal.costnet import GoalCostNet, GoalCostDataset, collate

    pl.seed_everything(a.seed, workers=True)
    torch.set_float32_matmul_precision("high")   # TF32 on the A100s
    tr = GoalCostDataset(np.load(a.train, allow_pickle=False), ENV_DIR, SIZE)
    va = GoalCostDataset(np.load(a.val, allow_pickle=False), ENV_DIR, SIZE)
    print(f"[train] {len(tr)} train groups, {len(va)} val groups")
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch, shuffle=True,
                                     collate_fn=collate, num_workers=a.workers,
                                     persistent_workers=a.workers > 0, drop_last=True)
    vdl = torch.utils.data.DataLoader(va, batch_size=a.batch, shuffle=False,
                                      collate_fn=collate, num_workers=a.workers,
                                      persistent_workers=a.workers > 0)
    net = GoalCostNet(d_model=a.d_model, recurrence=a.recurrence, lr=a.lr,
                      pe=a.pe)
    ck = ModelCheckpoint(dirpath=a.run, filename="cost-{epoch}-{val_spearman:.4f}",
                         monitor="val_spearman", mode="max", save_top_k=1,
                         save_last=True)
    tr_ = pl.Trainer(max_epochs=a.epochs, accelerator="auto", devices=1,
                     default_root_dir=a.run, callbacks=[ck],
                     log_every_n_steps=20, precision=a.precision,
                     enable_progress_bar=False,
                     max_time=dict(minutes=a.max_minutes),
                     limit_val_batches=a.limit_val)
    tr_.fit(net, dl, vdl)
    if "val_spearman" not in tr_.callback_metrics:
        raise SystemExit("STAGE2 TRAIN FAILED: validation never ran")
    print(f"[train] epochs run {tr_.current_epoch}, "
          f"final metrics {dict(tr_.callback_metrics)}")
    print(f"[train] best {ck.best_model_path} ({ck.best_model_score})")
    Path(a.run, "best.txt").write_text(str(ck.best_model_path) + "\n")
    print("SPR SUBGOAL STAGE2 TRAIN DONE")


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def _baselines(positions, r, wr, wd, eid, n=SIZE):
    """(B1, B2) predictions for every goal cell of robot r, as float arrays."""
    start = tuple(positions[r])
    b1 = np.empty(n * n, np.float64)
    # relaxed_dist is symmetric, so one BFS from the ROBOT's cell gives the
    # relaxed distance to every goal cell.
    b1[:] = relaxed_dist(start, wr, wd, n, eid)
    lone = lone_robot_cost(positions, r, wr, wd, n).astype(np.float64)
    b2 = np.where(lone >= 0, lone, B2_UNDEFINED)
    return b1, b2, lone


def evaluate(a):
    import torch
    from subgoal.costnet import GoalCostNet, state_features
    from nn_labeler import encode as nn_encode

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = GoalCostNet.load_from_checkpoint(a.ckpt, map_location=dev).to(dev).eval()
    store = np.load(a.data, allow_pickle=False)
    env_id, positions, cost = store["env_id"], store["positions"], store["cost"]
    S = len(env_id) if a.limit is None else min(a.limit, len(env_id))
    n = SIZE

    per_robot = {k: [] for k in ("net", "b1", "b2")}
    per_state = {k: [] for k in ("net", "b1", "b2")}
    reach_stats = dict(tp=0, fp=0, fn=0, tn=0)
    b2_defined = {k: [] for k in ("net", "b2")}
    dump = []
    t0 = time.time()
    with torch.no_grad():
        for s0 in range(0, S, a.batch):
            idx = list(range(s0, min(s0 + a.batch, S)))
            xs, aa, ai, meta = [], [], [], []
            for si in idx:
                eid = int(env_id[si])
                A_all, A_ind = nn_encode.adjacency(str(ENV_DIR), eid, n)
                for r in range(ROBOTS):
                    xs.append(state_features(positions[si], r, n))
                    aa.append(A_all)
                    ai.append(A_ind)
                    meta.append((si, r, eid))
            x = torch.from_numpy(np.stack(xs)).to(dev)
            A_all = torch.as_tensor(np.stack(aa)).to(dev)
            A_ind = torch.as_tensor(np.stack(ai)).to(dev)
            logits, rlogit = net(x, A_all, A_ind, n)
            chat = net.cost_hat(logits).float().cpu().numpy()
            preach = torch.sigmoid(rlogit).float().cpu().numpy()
            for k, (si, r, eid) in enumerate(meta):
                wr, wd = board(eid)
                y = cost[si, r].astype(np.float64)
                m = y >= 0
                b1, b2, lone = _baselines(positions[si], r, wr, wd, eid, n)
                reach_stats["tp"] += int(((preach[k] > .5) & m).sum())
                reach_stats["fp"] += int(((preach[k] > .5) & ~m).sum())
                reach_stats["fn"] += int(((preach[k] <= .5) & m).sum())
                reach_stats["tn"] += int(((preach[k] <= .5) & ~m).sum())
                if m.sum() >= 2:
                    per_robot["net"].append(group_metrics(chat[k][m], y[m]))
                    per_robot["b1"].append(group_metrics(b1[m], y[m]))
                    per_robot["b2"].append(group_metrics(b2[m], y[m]))
                    d = m & (lone >= 0)
                    if d.sum() >= 2:
                        b2_defined["net"].append(group_metrics(chat[k][d], y[d]))
                        b2_defined["b2"].append(group_metrics(b2[d], y[d]))
            # pooled per state: all four robots' reachable candidates together
            for j, si in enumerate(idx):
                rows = slice(j * ROBOTS, (j + 1) * ROBOTS)
                y = cost[si].astype(np.float64).reshape(-1)
                m = y >= 0
                if m.sum() < 2:
                    continue
                eid = int(env_id[si])
                wr, wd = board(eid)
                B1 = np.concatenate([_baselines(positions[si], r, wr, wd, eid, n)[0]
                                     for r in range(ROBOTS)])
                B2 = np.concatenate([_baselines(positions[si], r, wr, wd, eid, n)[1]
                                     for r in range(ROBOTS)])
                P = chat[rows].reshape(-1)
                per_state["net"].append(group_metrics(P[m], y[m]))
                per_state["b1"].append(group_metrics(B1[m], y[m]))
                per_state["b2"].append(group_metrics(B2[m], y[m]))
                if len(dump) < 5:
                    dump.append(dict(env_id=eid, positions=positions[si].tolist(),
                                     n_reachable=int(m.sum())))
    def agg(rows):
        out = {}
        for k in ("mae", "spearman", "top1"):
            v = [d[k] for d in rows if d[k] == d[k]]
            out[k] = float(np.mean(v)) if v else None
            out[f"{k}_n"] = len(v)
        out["mean_group_size"] = float(np.mean([d["n"] for d in rows])) if rows else None
        out["groups"] = len(rows)
        return out

    res = dict(
        data=str(a.data), ckpt=str(a.ckpt), n_states=S,
        per_state={k: agg(v) for k, v in per_state.items()},
        per_robot={k: agg(v) for k, v in per_robot.items()},
        b2_defined_subset={k: agg(v) for k, v in b2_defined.items()},
        reachability=reach_stats | dict(
            acc=(reach_stats["tp"] + reach_stats["tn"]) /
                max(1, sum(reach_stats.values())),
            precision=reach_stats["tp"] / max(1, reach_stats["tp"] + reach_stats["fp"]),
            recall=reach_stats["tp"] / max(1, reach_stats["tp"] + reach_stats["fn"])),
        b2_undefined_constant=B2_UNDEFINED,
        gate_rho_margin=GATE_RHO_MARGIN,
        seconds=round(time.time() - t0, 1),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        date=time.strftime("%Y-%m-%dT%H:%M:%S"), sample=dump)
    base_rho = max(res["per_state"]["b1"]["spearman"], res["per_state"]["b2"]["spearman"])
    base_top1 = max(res["per_state"]["b1"]["top1"], res["per_state"]["b2"]["top1"])
    res["gate"] = dict(
        net_spearman=res["per_state"]["net"]["spearman"],
        best_baseline_spearman=base_rho,
        margin=res["per_state"]["net"]["spearman"] - base_rho,
        net_top1=res["per_state"]["net"]["top1"], best_baseline_top1=base_top1,
        passed=bool(res["per_state"]["net"]["spearman"] >= base_rho + GATE_RHO_MARGIN
                    and res["per_state"]["net"]["top1"] >= base_top1))
    print(json.dumps({k: v for k, v in res.items() if k != "sample"}, indent=1))
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(res, indent=1) + "\n")
        os.replace(tmp, out)
        print(f"SPR SUBGOAL STAGE2 EVAL DONE {out}")


# ---------------------------------------------------------------------------
# beam survival: would a top-k beam keep an optimal subgoal alive?
# ---------------------------------------------------------------------------

KS = (1, 3, 5, 10, 20)


def _survival(score, opt_idx, ks=None):
    """Would a top-k beam over `score` keep at least one optimal candidate?

    Integer-valued scorers (true cost, the baselines) tie massively -- half the
    board can share "cost 1" -- so a single rank is not a fair summary. Three
    numbers per k, all exact:

      optimistic   ties resolved in the optimal candidate's favour  (upper bound)
      pessimistic  ties resolved against it                          (lower bound)
      expected     the probability under UNIFORM RANDOM tie-breaking, which is
                   what an implementation that does not care about ties does.
                   With L candidates strictly cheaper than the best-placed
                   optimal one, T tied with it and M of those optimal:
                   0 if L >= k, 1 if L + T <= k, else
                   1 - C(T-M, k-L) / C(T, k-L).
    """
    ks = KS if ks is None else ks
    so = min(float(score[o]) for o in opt_idx)
    L = int((score < so).sum())
    T = int((score == so).sum())
    M = sum(1 for o in opt_idx if float(score[o]) == so)
    out = {}
    for k in ks:
        opt = L + 1 <= k
        pess = L + T - M + 1 <= k
        if L >= k:
            exp = 0.0
        elif L + T <= k:
            exp = 1.0
        else:
            r = k - L
            exp = 1.0 - (math.comb(T - M, r) / math.comb(T, r) if r <= T - M else 0.0)
        out[k] = (int(opt), int(pess), exp)
    return out


def beam(a):
    import torch
    from subgoal.costnet import GoalCostNet, state_features
    from subgoal.space import relaxed_h
    from nn_labeler import encode as nn_encode

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = GoalCostNet.load_from_checkpoint(a.ckpt, map_location=dev).to(dev).eval()
    rows = json.loads(Path(a.optsets).read_text())["rows"]
    rows = [r for r in rows if r.get("ok")]
    n = SIZE
    INF = 1e9

    scorers = ("net", "true", "b1", "net+h", "true+h", "b1+h")
    tally = {s: {sc: {k: [0, 0, 0.0] for k in KS} for sc in scorers}
             for s in ("all", "root", "reach", "reach_root")}
    seen = {s: 0 for s in tally}
    random_hit = {"all": [], "root": []}
    per_instance = []
    t0 = time.time()
    with torch.no_grad():
        for r in rows:
            eid = r["env_id"]
            wr, wd = board(eid)
            H = relaxed_h(tuple(r["target"]), wr, wd)
            tidx = r["target_idx"]
            A_all, A_ind = nn_encode.adjacency(str(ENV_DIR), eid, n)
            inst_rows = []
            for di, dec in enumerate(r["path"]):
                pos = [tuple(c) for c in dec["positions"]]
                opt = {(int(rr), int(cc)) for rr, cc in dec["optimal_next"]}
                x = torch.from_numpy(np.stack([state_features(pos, rb, n)
                                               for rb in range(ROBOTS)])).to(dev)
                AA = torch.as_tensor(np.stack([A_all] * ROBOTS)).to(dev)
                AI = torch.as_tensor(np.stack([A_ind] * ROBOTS)).to(dev)
                logits, rlogit = net(x, AA, AI, n)
                netscore = net.score(logits, rlogit).float().cpu().numpy()   # [4,256]
                true = np.stack([cost_row(pos, rb, wr, wd, n) for rb in range(ROBOTS)])
                B1 = np.stack([_baselines(pos, rb, wr, wd, eid, n)[0]
                               for rb in range(ROBOTS)])
                # relaxed remaining distance of the CHILD state: only a move of
                # the target robot changes it.
                hbase = float(H.get(pos[tidx], INF))
                hchild = np.full((ROBOTS, n * n), hbase, np.float64)
                hchild[tidx] = np.array([H.get((c % n, c // n), INF)
                                         for c in range(n * n)], np.float64)
                reach = (true >= 0).reshape(-1)
                tcost = np.where(true >= 0, true, INF).astype(np.float64).reshape(-1)
                sc = {
                    "net": netscore.reshape(-1).astype(np.float64),
                    "true": tcost,
                    "b1": np.minimum(B1.reshape(-1), INF),
                    "net+h": netscore.reshape(-1).astype(np.float64) + hchild.reshape(-1),
                    "true+h": tcost + hchild.reshape(-1),
                    "b1+h": np.minimum(B1.reshape(-1), INF) + hchild.reshape(-1),
                }
                opt_idx = [rr * n * n + cc for rr, cc in opt]
                assert all(reach[i] for i in opt_idx), "optimal subgoal unreachable"
                buckets = [("all", np.ones_like(reach)), ("reach", reach)]
                rec = dict(decision=di, n_optimal=len(opt_idx),
                           n_reachable=int(reach.sum()))
                for bname, mask in buckets:
                    rootname = "root" if bname == "all" else "reach_root"
                    keep = np.flatnonzero(mask)
                    remap = {int(j): i for i, j in enumerate(keep)}
                    oi = [remap[i] for i in opt_idx]
                    for scn in scorers:
                        surv = _survival(sc[scn][keep], oi)
                        rec[f"{bname}:{scn}"] = {str(k): v for k, v in surv.items()}
                        for k in KS:
                            for c in range(3):
                                tally[bname][scn][k][c] += surv[k][c]
                                if di == 0:
                                    tally[rootname][scn][k][c] += surv[k][c]
                    seen[bname] += 1
                    if di == 0:
                        seen[rootname] += 1
                random_hit["all"].append(len(opt_idx) / (ROBOTS * n * n))
                if di == 0:
                    random_hit["root"].append(len(opt_idx) / (ROBOTS * n * n))
                inst_rows.append(rec)
            per_instance.append(dict(idx=r["idx"], env_id=eid, d_star=r["d_star"],
                                     decisions=inst_rows))

    def pct(bucket):
        d = seen[bucket]
        return {scn: {f"top{k}": dict(
            expected=100.0 * tally[bucket][scn][k][2] / max(1, d),
            optimistic=100.0 * tally[bucket][scn][k][0] / max(1, d),
            pessimistic=100.0 * tally[bucket][scn][k][1] / max(1, d))
            for k in KS} for scn in scorers}

    res = dict(optsets=str(a.optsets), ckpt=str(a.ckpt),
               instances=len(rows), decisions=seen["all"], roots=seen["root"],
               scorers=dict(
                   net="the network's expected cost over all 1024 candidates "
                       "(p_reach * cost_hat + (1-p_reach) * 50)",
                   true="the EXACT cost c(s,R,C); unreachable = +inf. The oracle "
                        "for what this network is trained to predict.",
                   b1="baseline B1, the any-stop relaxation distance",
                   **{f"{k}+h": f"{k} plus the any-stop relaxation distance of the "
                                "TARGET robot in the resulting state -- i.e. the "
                                "f = g + c + h priority Stage 1's exhaustive A* used"
                      for k in ("net", "true", "b1")}),
               buckets=dict(
                   all="ranked over all 4*256 = 1024 (robot, cell) candidates",
                   reach="ranked over only the candidates physics says are "
                         "reachable (Stage 3's expansion knows this exactly)"),
               survival=  {b: pct(b) for b in tally},
               random_top5=dict(
                   all=100.0 * 5 * float(np.mean(random_hit["all"])),
                   root=100.0 * 5 * float(np.mean(random_hit["root"]))),
               mean_optimal_set=float(np.mean([d["n_optimal"] for p in per_instance
                                               for d in p["decisions"]])),
               mean_reachable=float(np.mean([d["n_reachable"] for p in per_instance
                                             for d in p["decisions"]])),
               seconds=round(time.time() - t0, 1),
               slurm_job_id=os.environ.get("SLURM_JOB_ID"),
               date=time.strftime("%Y-%m-%dT%H:%M:%S"),
               per_instance=per_instance)
    print(json.dumps({k: v for k, v in res.items() if k != "per_instance"}, indent=1))
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(res, indent=1) + "\n")
        os.replace(tmp, out)
        print(f"SPR SUBGOAL STAGE2 BEAM DONE {out}")


# ---------------------------------------------------------------------------
# report: the markdown tables of STAGE2.md, printed from the payloads
# ---------------------------------------------------------------------------

def report(a):
    """Print STAGE2.md's tables from the eval / beam payloads. Nothing here is
    typed by hand; every cell is read out of a JSON produced by the job."""
    names = dict(net="**goal-conditioned cost net**",
                 b1="B1 any-stop relaxation", b2="B2 lone-robot exact BFS")
    for path in a.eval:
        d = json.loads(Path(path).read_text())
        print(f"\n### {Path(path).name}  ({d['n_states']} states, "
              f"ckpt {Path(d['ckpt']).name})\n")
        for scope, title in (("per_state", "pooled over all four robots "
                              "(the 1024-candidate set a planner ranks)"),
                             ("per_robot", "one group per (state, robot)")):
            g = d[scope]
            print(f"*{title}* -- {g['net']['groups']} groups, "
                  f"mean {g['net']['mean_group_size']:.1f} reachable candidates\n")
            print("| scorer | Spearman rho | top-1 acc | MAE (moves) |")
            print("|---|---|---|---|")
            for k in ("net", "b1", "b2"):
                print(f"| {names[k]} | {g[k]['spearman']:.3f} | "
                      f"{100 * g[k]['top1']:.1f}% | {g[k]['mae']:.2f} |")
            print()
        sub = d["b2_defined_subset"]
        print(f"B2-defined subset only (mean {sub['b2']['mean_group_size']:.1f} "
              f"candidates/group): net rho {sub['net']['spearman']:.3f} / "
              f"MAE {sub['net']['mae']:.2f}; B2 rho {sub['b2']['spearman']:.3f} / "
              f"MAE {sub['b2']['mae']:.2f}")
        r = d["reachability"]
        print(f"reachability head: acc {100 * r['acc']:.1f}%, "
              f"precision {100 * r['precision']:.1f}%, recall {100 * r['recall']:.1f}%")
        print(f"GATE: net rho {d['gate']['net_spearman']:.3f} vs best baseline "
              f"{d['gate']['best_baseline_spearman']:.3f} "
              f"(margin {d['gate']['margin']:+.3f}, required "
              f"+{d['gate_rho_margin']}) -> "
              f"{'PASS' if d['gate']['passed'] else 'FAIL'}")
    if a.beam:
        d = json.loads(Path(a.beam).read_text())
        print(f"\n### beam survival ({d['instances']} instances, "
              f"{d['decisions']} decisions, {d['roots']} roots; mean optimal set "
              f"{d['mean_optimal_set']:.2f} of 1024, mean reachable "
              f"{d['mean_reachable']:.1f}; a uniformly random top-5 would survive "
              f"{d['random_top5']['all']:.1f}%)\n")
        for bucket in ("all", "reach", "root", "reach_root"):
            print(f"*{bucket}* -- expected survival under uniform random "
                  "tie-breaking (optimistic / pessimistic bounds in brackets)\n")
            print("| ranking | top-1 | top-3 | top-5 | top-10 | top-20 |")
            print("|---|---|---|---|---|---|")
            for scn, row in d["survival"][bucket].items():
                cells = " | ".join(
                    f"{row[f'top{k}']['expected']:.1f}% "
                    f"[{row[f'top{k}']['pessimistic']:.0f}-{row[f'top{k}']['optimistic']:.0f}]"
                    for k in KS)
                print(f"| {scn} | {cells} |")
            print()


# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--boards", default=CFG.board_ranges["train"])
    g.add_argument("--per-board", type=int, default=8)
    g.add_argument("--walk", type=int, default=3,
                   help="apply 0..WALK random macro subgoals to each sampled "
                        "placement, so the corpus contains planner-reachable "
                        "descendants and not only uniform placements")
    g.add_argument("--from-bench", default=None,
                   help="instead of sampling, take the START states of a bench "
                        "jsonl (the exact roots a Stage 3 planner sees)")
    g.add_argument("--seed", type=int, default=11)
    g.add_argument("--out", required=True)
    g.set_defaults(fn=gen)

    t = sub.add_parser("train")
    t.add_argument("--train", required=True)
    t.add_argument("--val", required=True)
    t.add_argument("--run", required=True)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--d-model", type=int, default=192)
    t.add_argument("--recurrence", type=int, default=12)
    t.add_argument("--pe", default="none")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--precision", default="32-true",
                   help="the reference size-free nets train in fp32; the "
                        "edge-masked attention fills with -inf, which fp16 "
                        "cannot represent")
    t.add_argument("--max-minutes", type=int, default=45,
                   help="hard wall on fit() so the job always reaches eval")
    t.add_argument("--limit-val", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=11)
    t.set_defaults(fn=train)

    e = sub.add_parser("eval")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--batch", type=int, default=8, help="states per forward batch")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--out", default=None)
    e.set_defaults(fn=evaluate)

    b = sub.add_parser("beam")
    b.add_argument("--ckpt", required=True)
    b.add_argument("--optsets", required=True)
    b.add_argument("--out", default=None)
    b.set_defaults(fn=beam)

    rp = sub.add_parser("report")
    rp.add_argument("--eval", nargs="*", default=[])
    rp.add_argument("--beam", default=None)
    rp.set_defaults(fn=report)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
