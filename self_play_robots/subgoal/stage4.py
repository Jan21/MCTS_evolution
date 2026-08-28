"""Stage 4 of PLAN_SUBGOAL_DISCOVERY.md: re-baseline the planner, then run
certified self-play inside it, warm-started and from scratch.

    PYTHONPATH=supervised_valuenet:self_play_robots python -m subgoal.stage4 \
        devgen | plan | selfplay | train | beam | report | verify

Three parts, in this order (STAGE4.md holds the pre-registration):

  A  the two free wins Stage 3 measured and did not take.
     A1 re-size the beam.  An expansion costs exactly four encoder passes at
        EVERY k, because the candidate cell is read out rather than encoded, so
        the prune buys nothing in network cost and only throws children away.
        The width is chosen on a HELD-OUT dev set of fresh boards (`devgen`),
        never on bench450, by BEAM_RULE below.
     A2 strengthen the optimality bound.  Stage 3's is board-only, so even an
        exact `h` could not terminate early (STAGE3.md section 4);
        `planner.py::Search._h2` adds the only two terms that look at where the
        robots are.  Sound, so it is also used to skip nodes that cannot
        improve the incumbent.
     Then the plan's four-row table at the new headline configuration.  THAT
     number, not 60.7%, is the bar Part B must beat.

  B  self-play: fresh boards, search with noise, replay every finished plan,
     label every state on a CERTIFIED plan with that plan's true remaining
     cost, retrain `h` warm-started on a rolling buffer, repeat.

  C  the same loop from random initialisation, with no exact-engine label at
     any point, for the same number of rounds.

Nothing here ever plans on, trains on, or selects on a benchmark board.
"""
from __future__ import annotations

import argparse
import json
import os
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

from subgoal.costnet import cost_row                            # noqa: E402
from subgoal.space import board                                 # noqa: E402
from subgoal.table import BENCH, ENV_DIR, SIZE, load_bench, replay  # noqa: E402
from subgoal import planner as P                                # noqa: E402

ROBOTS = 4
OUT = SPR / "results/subgoal/stage4"

# --- pre-registered constants ----------------------------------------------
# A1. Candidate widths, and the rule that picks one. 1024 = no prune at all.
BEAM_WIDTHS = [5, 10, 20, 50, 100, 1024]
BEAM_TOL = 1.0
BEAM_RULE = ("the SMALLEST width whose dev-set optimal%% is within %.1f point "
             "of the best width's, measured on the held-out dev instances of "
             "`devgen` (fresh boards, never bench450)" % BEAM_TOL)

# Board splits. g16r4 bench450 is boards 2400-2549 and is touched by nothing
# here; Stage 3 trained on 1000-1699 and validated on 1800-1849.
DEV_BOARDS = "1900-1999"          # A1's beam choice + nothing else
SP_TRAIN_BOARDS = "1000-1799"     # Stage 4 self-play generation
SP_VAL_BOARDS = "1850-1899"       # Stage 4 per-round checkpoint selection

# B/C. Self-play, untuned, fixed before the first round was generated.
SP_NOISE = 0.5                    # sigma of the Gaussian jitter on f
SP_EXPANSIONS = 400               # per self-play search
SP_ROUNDS = 3
SP_BUFFER = 2                     # rolling buffer: this round + the previous one

GATE_STAGE3_PROTOCOL_PCT = 60.7   # Stage 3's inherited-protocol headline
BAR_FILE = OUT / "partA_bar.json"  # Part A's number, frozen before B and C run


# ---------------------------------------------------------------------------
# instance sets
# ---------------------------------------------------------------------------

def _puzzles(ids, per_board, rng, n=SIZE, walk=0):
    """Random (placement, target robot, target cell) puzzles on given boards."""
    out = []
    for eid in ids:
        wr, wd = board(eid)
        for _ in range(per_board):
            cells = rng.sample(range(n * n), ROBOTS + 1)
            pos = [(c % n, c // n) for c in cells[:ROBOTS]]
            target = (cells[ROBOTS] % n, cells[ROBOTS] // n)
            tidx = rng.randrange(ROBOTS)
            for _ in range(rng.randint(0, walk) if walk else 0):
                r = rng.randrange(ROBOTS)
                row = cost_row(pos, r, wr, wd, n)
                cand = np.flatnonzero(row >= 0)
                if len(cand):
                    c = int(rng.choice(cand))
                    pos[r] = (c % n, c // n)
            if pos[tidx] == target:
                continue
            out.append(dict(env_id=int(eid), positions=[list(p) for p in pos],
                            target=list(target), target_idx=tidx))
    rng.shuffle(out)
    return out


def devgen(a):
    """A held-out instance set with EXACT d*, for choosing the beam width.

    d* comes from the same exact engine Stage 3 labelled with. It is used only
    to score a SEARCH setting that every arm (including the network-free
    control) shares -- no network ever sees these boards or these numbers.
    """
    from subgoal import rustexact
    from scaling.configs import parse_ids
    rng = random.Random(a.seed)
    cand = _puzzles(parse_ids(a.boards), a.per_board, rng)
    cand = cand[:a.n * 2]
    vals = rustexact.ctg_batch([(c["env_id"], c["positions"], c["target_idx"],
                                 c["target"]) for c in cand],
                               a.sidecars, a.work, threads=a.threads, tag="dev")
    keep = []
    for c, v in zip(cand, vals):
        if v is None or v <= 0:
            continue
        c["d_star"] = int(v)
        keep.append(c)
        if len(keep) >= a.n:
            break
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / (out.name + ".tmp")
    tmp.write_text("".join(json.dumps(c) + "\n" for c in keep))
    os.replace(tmp, out)
    ds = np.array([c["d_star"] for c in keep])
    print(f"[devgen] {len(keep)} dev instances on boards {a.boards}, "
          f"mean d* {ds.mean():.2f}, max {ds.max()} -> {out}")
    print(f"SPR SUBGOAL STAGE4 DEVGEN DONE {out}")


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def _heuristic(a, env_ids):
    if a.arm == "relaxed":
        return P.RelaxedHeuristic(), dict(heuristic="relaxed any-stop distance "
                                          "(board only, no network)")
    if a.arm == "net":
        return (P.NetHeuristic(a.ckpt, env_ids, ENV_DIR, SIZE, batch=a.net_batch),
                dict(heuristic="learned CtgNet cost-to-go", ckpt=str(a.ckpt)))
    raise SystemExit(f"unknown arm {a.arm}")


def plan(a):
    inst = load_bench(a.instances)
    if a.limit:
        inst = inst[:a.limit]
    env_ids = sorted({int(r["env_id"]) for r in inst})
    heur, extra = _heuristic(a, env_ids)
    t0 = time.time()
    rows = P.run(inst, heur, a.k, a.expansions, concurrency=a.concurrency,
                 bound=a.bound, clamp=a.clamp)
    wall = time.time() - t0
    pay = P.payload(rows, inst, a.name or f"{a.arm}_k{a.k}_e{a.expansions}",
                    a.k, a.expansions,
                    extra=dict(**extra, arm=a.arm, bound=a.bound, clamp=a.clamp,
                               instances_file=str(a.instances),
                               wall_seconds=round(wall, 1),
                               concurrency=a.concurrency,
                               heuristic_passes=int(getattr(heur, "passes", 0)),
                               slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                               date=time.strftime("%Y-%m-%dT%H:%M:%S")))
    agg = pay["systems"][list(pay["systems"])[0]]["aggregate"]
    print(f"[plan] arm={a.arm} k={a.k} exp={a.expansions} bound={a.bound}: "
          f"solved {agg['solved']}/{len(inst)}, "
          f"optimal {agg['pct_optimal_of_n']:.1f}% of {len(inst)}, "
          f"mean extra {agg['mean_regret']}, proved {agg['proved_optimal_in_pruned_graph']}, "
          f"budget-limited {agg['budget_limited']}, {wall:.0f}s", flush=True)
    if a.out:
        _atomic_json(a.out, pay)
        print(f"SPR SUBGOAL STAGE4 PLAN DONE {a.out}")


def _atomic_json(path, obj):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / (out.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, out)


# ---------------------------------------------------------------------------
# self-play: certified labels from the planner's own finished plans
# ---------------------------------------------------------------------------

def selfplay(a):
    """Search fresh puzzles with exploration noise, replay every finished plan
    under the real rules, and label every state on a CERTIFIED plan with that
    plan's true remaining cost. No oracle is consulted anywhere.

    Two passes, because one certified plan labels only the children it walked
    through and the beam has to ORDER a whole candidate set:

      1. roots. Each finished, replay-certified plan labels every (parent,
         robot, cell) on it with that plan's remaining cost from the child.
      2. probes. For each parent the first pass reached, `--probe m` of its
         OTHER reachable children become roots of their own searches; a
         certified plan from such a child is, by the same definition, the label
         of h(parent, robot, cell). This is what turns a 1-label group into an
         ordered candidate set, and it is the only source of labels for
         children an optimal plan never takes.
    """
    from scaling.configs import parse_ids
    rng = random.Random(a.seed)
    insts = _puzzles(parse_ids(a.boards), a.per_board, rng, walk=a.walk)
    if a.n:
        insts = insts[:a.n]
    env_ids = sorted({int(r["env_id"]) for r in insts})
    heur, _extra = _heuristic(a, env_ids)

    labels = {}                    # state key -> {(robot, cell): remaining cost}
    exact = set()                  # (key, robot, cell) whose search PROVED it
    stat = dict(searches=len(insts), solved=0, plans=0, certified=0,
                replay_failures=0, proved=0, cost_sum=0,
                probe_searches=0, probe_solved=0, probe_labels=0,
                probe_replay_failures=0)

    def put(key, r, cell, val, proved):
        d = labels.setdefault(key, {})
        if d.get((r, cell), 1 << 30) > val:
            d[(r, cell)] = val
            if proved:
                exact.add((key, r, cell))
            else:
                exact.discard((key, r, cell))

    def harvest(s):
        if s.best_cost is None:
            return
        stat["solved"] += 1
        stat["cost_sum"] += s.best_cost
        proved = s.reason == "proved optimal in the pruned graph"
        if proved:
            stat["proved"] += 1
        for pl in s.plans():
            if a.harvest == "best" and pl["cost"] != s.best_cost:
                continue
            stat["plans"] += 1
            n_moves, ok, _why = replay(s.inst, pl["moves"])
            if not ok or n_moves != pl["cost"]:
                stat["replay_failures"] += 1
                continue           # a plan that fails replay labels NOTHING
            stat["certified"] += 1
            D, spent = pl["cost"], 0
            for parent, r, cell, _child, edge in pl["steps"]:
                spent += edge
                put((s.env_id, parent, s.tidx, s.goal), r, cell, D - spent,
                    proved)

    t0 = time.time()
    P.run(insts, heur, a.k, a.expansions, concurrency=a.concurrency,
          bound=a.bound, noise=a.noise, seed=a.seed, collect=True,
          on_done=harvest)
    stat["root_seconds"] = round(time.time() - t0, 1)
    stat["root_states"] = len(labels)
    stat["root_label_pairs"] = sum(len(v) for v in labels.values())

    if a.probe:
        n = SIZE
        probes = []
        for key in list(labels):
            eid, flat, tidx, goal = key
            RAY, PIR = P.rays(eid, n)
            taken = set(labels[key])
            # probe the SAME robot the certified plan moved, so the group the
            # beam has to order ends up with a real candidate set rather than
            # one labelled child scattered across four robots
            cnt = {}
            for (r, _c) in taken:
                cnt[r] = cnt.get(r, 0) + 1
            rq = max(cnt, key=lambda r: (cnt[r], -r)) if cnt else rng.randrange(ROBOTS)
            blockers = flat[:rq] + flat[rq + 1:]
            pool = [(rq, cell) for cell in P.rest_cells_fast(flat[rq], blockers,
                                                             RAY, PIR)
                    if (rq, cell) not in taken and not (rq == tidx and cell == goal)]
            for r, cell in rng.sample(pool, min(a.probe, len(pool))):
                child = flat[:r] + (cell,) + flat[r + 1:]
                probes.append((key, r, cell, dict(
                    env_id=eid, target_idx=tidx,
                    target=[goal % n, goal // n],
                    positions=[[c % n, c // n] for c in child])))
        stat["probe_searches"] = len(probes)

        def harvest_probe(s):
            key, r, cell, _inst = probes[s.idx]
            if s.best_cost is None:
                return
            seq = s.sequence()
            n_moves, ok, _why = replay(s.inst, seq)
            if not ok or n_moves != s.best_cost:
                stat["probe_replay_failures"] += 1
                return
            stat["probe_solved"] += 1
            put(key, r, cell, s.best_cost,
                s.reason == "proved optimal in the pruned graph")

        t1 = time.time()
        P.run([q[3] for q in probes], heur, a.k, a.probe_expansions,
              concurrency=a.concurrency, bound=a.bound, noise=0.0,
              on_done=harvest_probe)
        stat["probe_seconds"] = round(time.time() - t1, 1)
        stat["probe_labels"] = sum(len(v) for v in labels.values()) - \
            stat["root_label_pairs"]

    stat["seconds"] = round(time.time() - t0, 1)
    stat["states"] = len(labels)
    stat["label_pairs"] = sum(len(v) for v in labels.values())
    stat["label_pairs_proved"] = len(exact)
    _write_corpus(labels, a.out, stat, a)
    print(f"[selfplay] {stat}", flush=True)
    print(f"SPR SUBGOAL STAGE4 SELFPLAY DONE {a.out}")


def _write_corpus(labels, out, stat, a, n=SIZE):
    """The npz `subgoal/ctgnet.py::CtgDataset` reads, built from certified
    labels only. `cost` is physics (every reachable child); `ctg` is -1 except
    on the children a certified plan actually walked through, plus the ZERO
    that physics gives for free when a child IS the goal."""
    keys = sorted(labels)
    S = len(keys)
    env_id = np.zeros(S, np.int32)
    positions = np.zeros((S, ROBOTS, 2), np.int16)
    tgt_idx = np.zeros(S, np.int8)
    tgt = np.zeros((S, 2), np.int16)
    cost = np.full((S, ROBOTS, n * n), -1, np.int8)
    ctg = np.full((S, ROBOTS, n * n), -1, np.int8)
    nzero = 0
    for s, key in enumerate(keys):
        eid, flat, tidx, goal = key
        pos = [(c % n, c // n) for c in flat]
        wr, wd = board(eid)
        env_id[s] = eid
        positions[s] = pos
        tgt_idx[s] = tidx
        tgt[s] = (goal % n, goal // n)
        for r in range(ROBOTS):
            cost[s, r] = cost_row(pos, r, wr, wd, n)
        if cost[s, tidx, goal] >= 0:
            ctg[s, tidx, goal] = 0            # physics: that child IS the goal
            nzero += 1
        for (r, cell), v in labels[key].items():
            if 0 <= v < 120:
                ctg[s, r, cell] = v
    lab = ctg >= 0
    stat = dict(stat, n_states=S, n_labels=int(lab.sum()), n_goal_zeros=nzero,
                mean_ctg=float(ctg[lab].mean()) if lab.any() else None,
                max_ctg=int(ctg[lab].max()) if lab.any() else None,
                boards=a.boards, seed=a.seed, noise=a.noise, k=a.k,
                expansions=a.expansions, harvest=a.harvest, arm=a.arm,
                ckpt=str(getattr(a, "ckpt", None)),
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                date=time.strftime("%Y-%m-%dT%H:%M:%S"))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / (out.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, env_id=env_id, positions=positions,
                            target_idx=tgt_idx, target=tgt, cost=cost, ctg=ctg,
                            meta=np.array(json.dumps(stat)))
    os.replace(tmp, out)
    Path(str(out) + ".meta.json").write_text(json.dumps(stat, indent=1) + "\n")


def load_many(paths):
    """Merge self-play corpora, keeping the SMALLEST certified label for any
    (state, robot, cell) that several rounds both reached."""
    idx, rows = {}, []
    for p in paths:
        z = np.load(p, allow_pickle=False)
        for s in range(z["env_id"].shape[0]):
            key = (int(z["env_id"][s]), tuple(map(tuple, z["positions"][s].tolist())),
                   int(z["target_idx"][s]), tuple(z["target"][s].tolist()))
            j = idx.get(key)
            if j is None:
                idx[key] = len(rows)
                rows.append([z["env_id"][s].copy(), z["positions"][s].copy(),
                             z["target_idx"][s].copy(), z["target"][s].copy(),
                             z["cost"][s].copy(), z["ctg"][s].copy()])
            else:
                old = rows[j][5]
                new = z["ctg"][s]
                m = (new >= 0) & ((old < 0) | (new < old))
                old[m] = new[m]
    return dict(env_id=np.array([r[0] for r in rows], np.int32),
                positions=np.stack([r[1] for r in rows]),
                target_idx=np.array([r[2] for r in rows], np.int8),
                target=np.stack([r[3] for r in rows]),
                cost=np.stack([r[4] for r in rows]),
                ctg=np.stack([r[5] for r in rows]))


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

class CollapseGuard:
    """Stage 2's failure mode: the net parks on a constant value and the spread
    of its predictions across a state's children goes to zero. Recurrence 4 is
    the pre-registered depth that avoided it; this watches for it anyway and
    stops the run so the round reports it instead of silently training noise."""

    def __init__(self, patience=5, floor=0.05):
        self.patience = patience
        self.floor = floor
        self.bad = 0
        self.collapsed = False


def train(a):
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, Callback
    from subgoal.ctgnet import CtgNet, CtgDataset, collate

    pl.seed_everything(a.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    tr_store = load_many(a.train) if len(a.train) > 1 else \
        np.load(a.train[0], allow_pickle=False)
    va_store = load_many(a.val) if len(a.val) > 1 else \
        np.load(a.val[0], allow_pickle=False)
    tr = CtgDataset(tr_store, ENV_DIR, SIZE, min_labels=a.min_labels)
    va = CtgDataset(va_store, ENV_DIR, SIZE, min_labels=a.min_labels)
    print(f"[train] {len(tr)} train groups, {len(va)} val groups "
          f"({int((np.asarray(tr_store['ctg']) >= 0).sum())} train labels)", flush=True)
    if not len(tr) or not len(va):
        raise SystemExit("STAGE4 TRAIN FAILED: empty corpus")
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch, shuffle=True,
                                     collate_fn=collate, num_workers=a.workers,
                                     persistent_workers=a.workers > 0,
                                     drop_last=len(tr) > a.batch)
    vdl = torch.utils.data.DataLoader(va, batch_size=a.batch, shuffle=False,
                                      collate_fn=collate, num_workers=a.workers,
                                      persistent_workers=a.workers > 0)
    if a.init and a.init != "scratch":
        net = CtgNet.load_from_checkpoint(a.init, map_location="cpu", lr=a.lr,
                                          rank_weight=a.rank_weight)
        print(f"[train] warm start from {a.init}")
    else:
        net = CtgNet(d_model=a.d_model, recurrence=a.recurrence, lr=a.lr,
                     pe=a.pe, rank_weight=a.rank_weight)
        print("[train] RANDOM initialisation (no exact-engine data anywhere)")

    guard = CollapseGuard(patience=a.collapse_patience, floor=a.collapse_floor)

    class _Guard(Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            m = trainer.callback_metrics
            sp = m.get("val_spread_reach")
            if sp is None:
                return
            if float(sp) < guard.floor:
                guard.bad += 1
                if guard.bad >= guard.patience:
                    guard.collapsed = True
                    print(f"[train] COLLAPSE GUARD: val_spread_reach < "
                          f"{guard.floor} for {guard.bad} epochs -- stopping",
                          flush=True)
                    trainer.should_stop = True
            else:
                guard.bad = 0

    ck = ModelCheckpoint(dirpath=a.run, filename="ctg-{epoch}-{%s:.4f}" % a.monitor,
                         monitor=a.monitor, mode=a.mode, save_top_k=1,
                         save_last=True)
    trainer = pl.Trainer(max_epochs=a.epochs, accelerator="auto", devices=1,
                         default_root_dir=a.run, callbacks=[ck, _Guard()],
                         log_every_n_steps=20, precision=a.precision,
                         enable_progress_bar=False,
                         max_time=dict(minutes=a.max_minutes))
    trainer.fit(net, dl, vdl)
    if a.monitor not in trainer.callback_metrics:
        raise SystemExit(f"STAGE4 TRAIN FAILED: {a.monitor} never logged")
    print(f"[train] epochs run {trainer.current_epoch}, "
          f"best {ck.best_model_path} ({ck.best_model_score}), "
          f"collapsed={guard.collapsed}", flush=True)
    Path(a.run, "best.txt").write_text(str(ck.best_model_path) + "\n")
    Path(a.run, "train_meta.json").write_text(json.dumps(dict(
        init=a.init, monitor=a.monitor, mode=a.mode, epochs=a.epochs,
        train_files=[str(x) for x in a.train], val_files=[str(x) for x in a.val],
        train_groups=len(tr), val_groups=len(va),
        best=str(ck.best_model_path),
        best_score=float(ck.best_model_score) if ck.best_model_score is not None
        else None, collapsed=guard.collapsed,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        date=time.strftime("%Y-%m-%dT%H:%M:%S")), indent=1) + "\n")
    print("SPR SUBGOAL STAGE4 TRAIN DONE")


def scratchckpt(a):
    """A randomly-initialised CtgNet checkpoint: Part C's round-0 planner, and
    the only initialisation that arm ever gets. Pinned seed, so it is
    reproducible and can be committed as a provenance record."""
    import torch
    import pytorch_lightning as pl
    from subgoal.ctgnet import CtgNet
    pl.seed_everything(a.seed, workers=True)
    net = CtgNet(d_model=a.d_model, recurrence=a.recurrence, pe=a.pe)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=net.state_dict(),
                    hyper_parameters=dict(net.hparams),
                    hparams_name="hparams", epoch=0, global_step=0,
                    **{"pytorch-lightning_version": pl.__version__},
                    loops={}, callbacks={}, optimizer_states=[],
                    lr_schedulers=[]), out)
    back = CtgNet.load_from_checkpoint(out, map_location="cpu")
    same = all(torch.equal(a_, b_) for a_, b_ in
               zip(net.state_dict().values(), back.state_dict().values()))
    print(f"[scratchckpt] seed {a.seed} -> {out} (reload identical: {same})")
    if not same:
        raise SystemExit("STAGE4 SCRATCHCKPT FAILED")
    print("SPR SUBGOAL STAGE4 SCRATCHCKPT DONE")


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def beam(a):
    from subgoal import table as T
    inst = load_bench(a.instances)
    recs = []
    for f in sorted(Path(a.dir).glob(a.glob)):
        s = T.score(f, inst)
        pay = json.loads(f.read_text())
        proto = pay["protocol"]
        agg = list(pay["systems"].values())[0]["aggregate"]
        recs.append(dict(path=str(f), arm=proto.get("arm"), k=proto.get("k"),
                         expansions=proto.get("expansions"),
                         bound=proto.get("bound", "weak"),
                         optimal=s["optimal"], pct=s["pct_optimal"],
                         solved=s["solved"], n=s["n"], mean_extra=s["mean_extra"],
                         replay_failures=len(s["replay_failures"]),
                         mean_expansions=agg["expansions_mean"],
                         encoder_passes=agg["h_queries_total"],
                         proved=agg["proved_optimal_in_pruned_graph"],
                         budget_limited=agg.get("budget_limited"),
                         bound_skips=agg.get("bound_skips_total"),
                         wall_seconds=proto.get("wall_seconds")))
    recs.sort(key=lambda r: (r["arm"] or "", r["bound"], r["k"] or 0))
    print(f"| h | bound | beam k | budget | optimal % of {recs[0]['n'] if recs else 0} "
          f"| solved | extra moves | mean exp | proved | budget-limited | wall s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        me = "—" if r["mean_extra"] is None else f"{r['mean_extra']:.3f}"
        print(f"| {r['arm']} | {r['bound']} | {r['k']} | {r['expansions']} "
              f"| {r['pct']:.1f}% ({r['optimal']}/{r['n']}) | {r['solved']}/{r['n']} "
              f"| {me} | {r['mean_expansions']:.0f} | {r['proved']} "
              f"| {r['budget_limited']} | {r['wall_seconds']} |")
    print(f"\nreplay failures across {len(recs)} payloads: "
          f"{sum(r['replay_failures'] for r in recs)}")
    if a.json_out:
        _atomic_json(a.json_out, recs)
    if a.pick:
        best = max(r["pct"] for r in recs)
        ok = [r for r in recs if r["pct"] >= best - BEAM_TOL]
        chosen = min(ok, key=lambda r: r["k"])
        print(f"\nBEAM RULE: {BEAM_RULE}\n  best dev optimal% = {best:.1f} "
              f"(k={max(recs, key=lambda r: r['pct'])['k']}); within {BEAM_TOL}: "
              f"{sorted(r['k'] for r in ok)}\n  CHOSEN k = {chosen['k']}")
        _atomic_json(a.pick, dict(rule=BEAM_RULE, tolerance=BEAM_TOL,
                                  chosen_k=chosen["k"], best_pct=best,
                                  rows=recs,
                                  date=time.strftime("%Y-%m-%dT%H:%M:%S")))


def auditlabels(a):
    """Every self-play label must be the cost of a plan that exists, so it can
    never be BELOW the true optimum. Checked against the exact engine on a
    sample -- the engine is used to AUDIT the corpus, never to build it, and
    this runs after the round it audits."""
    from subgoal import rustexact
    rng = random.Random(a.seed)
    z = np.load(a.corpus, allow_pickle=False)
    ctg, cost, pos = z["ctg"], z["cost"], z["positions"]
    n = SIZE
    items = [(s, r, c) for s in range(ctg.shape[0]) for r in range(ROBOTS)
             for c in np.flatnonzero(ctg[s, r] >= 0)]
    rng.shuffle(items)
    items = items[:a.n]
    q = []
    for s, r, c in items:
        child = [list(map(int, p)) for p in pos[s]]
        child[r] = [int(c) % n, int(c) // n]
        q.append((int(z["env_id"][s]), child, int(z["target_idx"][s]),
                  [int(z["target"][s][0]), int(z["target"][s][1])]))
    vals = rustexact.ctg_batch(q, a.sidecars, a.work, threads=a.threads,
                               tag="audit4")
    below = tight = checked = 0
    slack = []
    for (s, r, c), v in zip(items, vals):
        if v is None:
            continue
        lab = int(ctg[s, r, c])
        checked += 1
        slack.append(lab - v)
        if lab < v:
            below += 1
            if below < 5:
                print(f"  IMPOSSIBLE label {lab} < true {v} at state {s} "
                      f"robot {r} cell {c}")
        tight += lab == v
    sl = np.array(slack, float)
    print(f"[auditlabels] {a.corpus}: {checked} labels audited, {below} below "
          f"the true optimum (must be 0), {tight} exactly optimal "
          f"({100 * tight / max(checked, 1):.1f}%), mean slack {sl.mean():.3f}, "
          f"max {sl.max():.0f}")
    if a.out:
        _atomic_json(a.out, dict(corpus=str(a.corpus), checked=checked,
                                 below_true=below, exactly_optimal=int(tight),
                                 mean_slack=float(sl.mean()),
                                 max_slack=float(sl.max()),
                                 date=time.strftime("%Y-%m-%dT%H:%M:%S")))
    if below:
        raise SystemExit("STAGE4 AUDITLABELS FAILED: a label is below the optimum")
    print("SPR SUBGOAL STAGE4 AUDITLABELS DONE")


def stops(a):
    """Row-by-row diff of two payloads of the SAME instance set: what the
    stronger bound converted, and whether it changed any solution."""
    from subgoal import table as T
    inst = load_bench(a.instances)
    A = list(json.loads(Path(a.before).read_text())["systems"].values())[0]["rows"]
    B = list(json.loads(Path(a.after).read_text())["systems"].values())[0]["rows"]
    sa, sb = T.score(a.before, inst), T.score(a.after, inst)
    trans, conv, worse, better = {}, 0, 0, 0
    exp_a = exp_b = 0
    for x, y in zip(A, B):
        key = (x["stop_reason"], y["stop_reason"])
        trans[key] = trans.get(key, 0) + 1
        if x["stop_reason"] == "expansion budget" and \
                y["stop_reason"] == "proved optimal in the pruned graph":
            conv += 1
        exp_a += x["expansions"]
        exp_b += y["expansions"]
        la = x["realized_strict"] if x["solved"] else None
        lb = y["realized_strict"] if y["solved"] else None
        if la is None and lb is not None:
            better += 1
        elif la is not None and lb is None:
            worse += 1
        elif la is not None and lb is not None:
            better += lb < la
            worse += lb > la
    print(f"before {a.before}\n  {sa['optimal']}/{sa['n']} optimal, "
          f"{sa['solved']} solved, {exp_a} expansions")
    print(f"after  {a.after}\n  {sb['optimal']}/{sb['n']} optimal, "
          f"{sb['solved']} solved, {exp_b} expansions")
    print(f"\nbudget-limited BEFORE: "
          f"{sum(1 for x in A if x['stop_reason'] == 'expansion budget')}; "
          f"AFTER: {sum(1 for y in B if y['stop_reason'] == 'expansion budget')}; "
          f"converted budget-limited -> proved: {conv}")
    print(f"solutions improved {better}, worsened {worse}; "
          f"expansions saved {exp_a - exp_b} ({100 * (exp_a - exp_b) / exp_a:.1f}%)")
    print("\n| stop reason before | after | n |\n|---|---|---|")
    for (x, y), n in sorted(trans.items(), key=lambda kv: -kv[1]):
        print(f"| {x} | {y} | {n} |")
    if a.json_out:
        _atomic_json(a.json_out, dict(
            before=a.before, after=a.after, converted=conv,
            budget_before=sum(1 for x in A if x["stop_reason"] == "expansion budget"),
            budget_after=sum(1 for y in B if y["stop_reason"] == "expansion budget"),
            improved=better, worsened=worse,
            expansions_before=exp_a, expansions_after=exp_b,
            optimal_before=sa["optimal"], optimal_after=sb["optimal"],
            solved_before=sa["solved"], solved_after=sb["solved"],
            transitions={f"{x} -> {y}": n for (x, y), n in trans.items()}))


def report(a):
    from subgoal import table as T
    bench = load_bench(a.instances)
    rows = [] if a.only_rows else list(T.DEFAULT_ROWS)
    for spec in a.row:
        lbl, _, path = spec.partition("=")
        rows.append((lbl, path))
    scored = [(lbl, T.score(p, bench)) for lbl, p in rows]
    print(T.markdown(scored))
    print()
    for lbl, s in scored:
        print(f"[{lbl}] {s['path']}\n    system={s['system']!r} exp={s['expansions']} "
              f"k={s['k']}: {s['optimal']}/{s['n']} optimal, {s['solved']}/{s['n']} "
              f"solved, mean extra {s['mean_extra']}\n    checks: replay failures "
              f"{len(s['replay_failures'])}, misaligned {len(s['misaligned'])}, "
              f"length disagreements {len(s['length_disagreements'])}, "
              f"below d* {len(s['below_d_star'])}")
    if a.freeze_bar:
        lbl, s = scored[-1]
        _atomic_json(BAR_FILE, dict(
            label=lbl, path=s["path"], pct_optimal=s["pct_optimal"],
            optimal=s["optimal"], solved=s["solved"], n=s["n"],
            mean_extra=s["mean_extra"],
            note="Part A's re-baselined headline. Part B and Part C must EXCEED "
                 "this to pass Stage 4's gate; frozen before any self-play job "
                 "was submitted.",
            date=time.strftime("%Y-%m-%dT%H:%M:%S")))
        print(f"\nPart A bar frozen at {s['pct_optimal']:.1f}% -> {BAR_FILE}")
    if a.gate and BAR_FILE.exists():
        bar = json.loads(BAR_FILE.read_text())
        g = [s for lbl, s in scored if s["path"] == a.gate]
        if g:
            pct = g[0]["pct_optimal"]
            v = "PASS" if pct > bar["pct_optimal"] else "FAIL"
            print(f"\nStage 4 gate (Part A bar {bar['pct_optimal']:.1f}% of "
                  f"{bar['n']}): {pct:.1f}% -> {v}")
    if a.json_out:
        _atomic_json(a.json_out, {"instances": str(a.instances),
                                  "rows": [{"label": l, **s} for l, s in scored]})


def series(a):
    """The per-round series of one self-play arm, with denominators."""
    from subgoal import table as T
    bench = load_bench(a.instances)
    bar = json.loads(BAR_FILE.read_text()) if BAR_FILE.exists() else None
    out = []
    for spec in a.row:
        lbl, _, path = spec.partition("=")
        if not Path(path).exists():
            out.append((lbl, None))
            continue
        out.append((lbl, T.score(path, bench)))
    print("| round | optimal % of 450 | solved / 450 | extra moves | proved | "
          "vs Part A bar |")
    print("|---|---|---|---|---|---|")
    for lbl, s in out:
        if s is None:
            print(f"| {lbl} | — | — | — | — | — |")
            continue
        pay = json.loads(Path(s["path"]).read_text())
        agg = list(pay["systems"].values())[0]["aggregate"]
        d = "" if bar is None else f"{s['pct_optimal'] - bar['pct_optimal']:+.1f}"
        print(f"| {lbl} | {s['pct_optimal']:.1f}% ({s['optimal']}/{s['n']}) "
              f"| {s['solved']}/{s['n']} | {s['mean_extra']:.3f} "
              f"| {agg['proved_optimal_in_pruned_graph']} | {d} |")
    if a.json_out:
        _atomic_json(a.json_out, [{"label": l, **(s or {})} for l, s in out])


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def verify(a):
    """Two checks, both of which must pass before any Stage 4 number is used.

    1. The tight bound is ADMISSIBLE: on random reachable states of bench
       boards, h_tight(s) <= the exact engine's true cost-to-go of s.
    2. The weak-bound path is unchanged: the search with bound="weak" is
       byte-identical to Stage 3's stored payload rows.
    """
    from subgoal import rustexact
    rng = random.Random(a.seed)
    bench = load_bench(a.instances)
    n = SIZE
    states, meta = [], []
    for inst in bench[:a.n]:
        eid = int(inst["env_id"])
        wr, wd = board(eid)
        tidx = int(inst["target_idx"])
        tgt = tuple(inst["target"])
        for _ in range(a.states):
            pos = [(c % n, c // n) for c in rng.sample(range(n * n), ROBOTS)]
            if pos[tidx] == tgt:
                continue
            states.append((eid, pos, tidx, tgt))
    vals = rustexact.ctg_batch(states, a.sidecars, a.work, threads=a.threads,
                               tag="verify4")
    bad = checked = 0
    gain = []
    for (eid, pos, tidx, tgt), v in zip(states, vals):
        if v is None:
            continue
        inst = dict(env_id=eid, positions=[list(p) for p in pos],
                    target=list(tgt), target_idx=tidx, d_star=int(v))
        s = P.Search(inst, 0, 5, 1, bound="tight")
        st = s.start
        h2 = s._h2(st)
        h_weak = s.H[st[tidx]]
        checked += 1
        gain.append(h2 - h_weak)
        if h2 > v:
            bad += 1
            if bad < 5:
                print(f"  INADMISSIBLE env {eid} pos {pos} tgt {tgt}: "
                      f"h_tight={h2} > d*={v}")
    g = np.array(gain, float)
    print(f"[verify] admissibility: {checked} states, {bad} violations; "
          f"tight - weak: mean {g.mean():.3f}, >0 on {(g > 0).mean() * 100:.1f}%, "
          f"max {g.max():.0f}")
    if bad:
        raise SystemExit("STAGE4 VERIFY FAILED: bound is not admissible")

    if a.reference:
        ref = json.loads(Path(a.reference).read_text())
        rrows = list(ref["systems"].values())[0]["rows"][:a.replay_n]
        sub = bench[:a.replay_n]
        heur = P.RelaxedHeuristic()
        got = P.run(sub, heur, ref["protocol"]["k"], ref["protocol"]["expansions"],
                    concurrency=32, bound="weak")
        diff = 0
        for x, y in zip(rrows, got):
            for f in ("expansions", "h_queries", "moves_seq", "realized_strict",
                      "stop_reason", "solved"):
                if x.get(f) != y.get(f):
                    diff += 1
                    if diff < 5:
                        print(f"  DIFF idx {x['idx']} field {f}: {x.get(f)} != "
                              f"{y.get(f)}")
                    break
        print(f"[verify] weak-bound reproduction of {a.reference}: "
              f"{len(rrows)} rows, {diff} differing")
        if diff:
            raise SystemExit("STAGE4 VERIFY FAILED: weak path changed")
    print("SPR SUBGOAL STAGE4 VERIFY DONE")


# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    SIDE = str(SPR / "results/subgoal/stage3/boards")
    WORK = str(SPR / "results/subgoal/stage4/work")

    d = sub.add_parser("devgen")
    d.add_argument("--boards", default=DEV_BOARDS)
    d.add_argument("--per-board", type=int, default=4)
    d.add_argument("--n", type=int, default=200)
    d.add_argument("--seed", type=int, default=41)
    d.add_argument("--threads", type=int, default=16)
    d.add_argument("--sidecars", default=SIDE)
    d.add_argument("--work", default=WORK)
    d.add_argument("--out", default=str(OUT / "dev200.jsonl"))
    d.set_defaults(fn=devgen)

    pl_ = sub.add_parser("plan")
    pl_.add_argument("--arm", required=True, choices=["relaxed", "net"])
    pl_.add_argument("--ckpt", default=None)
    pl_.add_argument("--instances", default=str(BENCH))
    pl_.add_argument("--k", type=int, default=1024)
    pl_.add_argument("--expansions", type=int, default=1200)
    pl_.add_argument("--bound", default="tight", choices=["weak", "tight"])
    pl_.add_argument("--clamp", action="store_true",
                     help="exploratory: rank by max(h, admissible bound)")
    pl_.add_argument("--limit", type=int, default=None)
    pl_.add_argument("--concurrency", type=int, default=64)
    pl_.add_argument("--net-batch", type=int, default=64)
    pl_.add_argument("--name", default=None)
    pl_.add_argument("--out", default=None)
    pl_.set_defaults(fn=plan)

    sp = sub.add_parser("selfplay")
    sp.add_argument("--arm", default="net", choices=["relaxed", "net"])
    sp.add_argument("--ckpt", default=None)
    sp.add_argument("--boards", default=SP_TRAIN_BOARDS)
    sp.add_argument("--per-board", type=int, default=3)
    sp.add_argument("--n", type=int, default=None)
    sp.add_argument("--walk", type=int, default=2)
    sp.add_argument("--k", type=int, default=1024)
    sp.add_argument("--expansions", type=int, default=SP_EXPANSIONS)
    sp.add_argument("--noise", type=float, default=SP_NOISE)
    sp.add_argument("--bound", default="tight", choices=["weak", "tight"])
    sp.add_argument("--harvest", default="best", choices=["best", "all"])
    sp.add_argument("--probe", type=int, default=3,
                    help="extra children of every reached state to certify")
    sp.add_argument("--probe-expansions", type=int, default=300)
    sp.add_argument("--concurrency", type=int, default=64)
    sp.add_argument("--net-batch", type=int, default=64)
    sp.add_argument("--seed", type=int, default=101)
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=selfplay)

    t = sub.add_parser("train")
    t.add_argument("--train", nargs="+", required=True)
    t.add_argument("--val", nargs="+", required=True)
    t.add_argument("--run", required=True)
    t.add_argument("--init", default=None, help="checkpoint to warm start from, "
                                                "or 'scratch'")
    t.add_argument("--monitor", default="val_mae_all")
    t.add_argument("--mode", default="min")
    t.add_argument("--min-labels", type=int, default=1)
    t.add_argument("--rank-weight", type=float, default=1.0)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--d-model", type=int, default=192)
    t.add_argument("--recurrence", type=int, default=4)
    t.add_argument("--pe", default="none")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--precision", default="32-true")
    t.add_argument("--max-minutes", type=int, default=25)
    t.add_argument("--collapse-patience", type=int, default=5)
    t.add_argument("--collapse-floor", type=float, default=0.05)
    t.add_argument("--seed", type=int, default=11)
    t.set_defaults(fn=train)

    sc = sub.add_parser("scratchckpt")
    sc.add_argument("--seed", type=int, default=2026)
    sc.add_argument("--d-model", type=int, default=192)
    sc.add_argument("--recurrence", type=int, default=4)
    sc.add_argument("--pe", default="none")
    sc.add_argument("--out", default=str(OUT / "scratch_init.ckpt"))
    sc.set_defaults(fn=scratchckpt)

    bm = sub.add_parser("beam")
    bm.add_argument("--dir", default=str(OUT))
    bm.add_argument("--glob", default="dev_*.json")
    bm.add_argument("--instances", default=str(OUT / "dev200.jsonl"))
    bm.add_argument("--json", dest="json_out", default=None)
    bm.add_argument("--pick", default=None, help="write the chosen width here")
    bm.set_defaults(fn=beam)

    al = sub.add_parser("auditlabels")
    al.add_argument("--corpus", required=True)
    al.add_argument("--n", type=int, default=3000)
    al.add_argument("--seed", type=int, default=5)
    al.add_argument("--threads", type=int, default=16)
    al.add_argument("--sidecars", default=SIDE)
    al.add_argument("--work", default=WORK)
    al.add_argument("--out", default=None)
    al.set_defaults(fn=auditlabels)

    st = sub.add_parser("stops")
    st.add_argument("--before", required=True)
    st.add_argument("--after", required=True)
    st.add_argument("--instances", default=str(BENCH))
    st.add_argument("--json", dest="json_out", default=None)
    st.set_defaults(fn=stops)

    r = sub.add_parser("report")
    r.add_argument("--instances", default=str(BENCH))
    r.add_argument("--row", action="append", default=[])
    r.add_argument("--only-rows", action="store_true")
    r.add_argument("--gate", default=None)
    r.add_argument("--freeze-bar", action="store_true")
    r.add_argument("--json", dest="json_out", default=None)
    r.set_defaults(fn=report)

    se = sub.add_parser("series")
    se.add_argument("--instances", default=str(BENCH))
    se.add_argument("--row", action="append", default=[])
    se.add_argument("--json", dest="json_out", default=None)
    se.set_defaults(fn=series)

    v = sub.add_parser("verify")
    v.add_argument("--instances", default=str(BENCH))
    v.add_argument("--n", type=int, default=30, help="boards")
    v.add_argument("--states", type=int, default=8)
    v.add_argument("--seed", type=int, default=7)
    v.add_argument("--threads", type=int, default=16)
    v.add_argument("--sidecars", default=SIDE)
    v.add_argument("--work", default=WORK)
    v.add_argument("--reference", default=str(SPR /
                   "results/subgoal/stage3/plan_relaxed_k5_e1200.json"))
    v.add_argument("--replay-n", type=int, default=40)
    v.set_defaults(fn=verify)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
