"""Stage 3 of PLAN_SUBGOAL_DISCOVERY.md: the planner, and the fourth table row.

    PYTHONPATH=supervised_valuenet:self_play_robots python -m subgoal.stage3 \
        gen | train | plan | report

Stage 2's verdict rewrote this stage: physics supplies the edges (it is cheaper
than a network and the search must compute it anyway) and the network supplies
only `h`, the moves still needed from the child state. `gen` labels children
with the exact engine, `train` fits `CtgNet`, `plan` runs
`subgoal/planner.py` over the pinned 450-instance benchmark, and `report`
prints the four-row table through `subgoal/table.py`.

THE GATE, fixed here before any planner was run: the learned-h planner's
percent of the 450 solved with a provably optimal move count must EXCEED the
backward supervised planner's 53.3% (240/450, FINDINGS 29). The pre-registered
headline configuration is the arena protocol, 1200 expansions and k = 5.
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

from scaling.configs import CONFIGS, parse_ids                   # noqa: E402
from subgoal.costnet import cost_row                            # noqa: E402
from subgoal.space import board                                  # noqa: E402
from subgoal.table import BENCH, ENV_DIR, SIZE, load_bench       # noqa: E402
from subgoal import planner as P                                 # noqa: E402

CFG = CONFIGS["g16r4"]
ROBOTS = 4
GATE_BACKWARD_PCT = 53.3      # backward supervised, 240/450 (FINDINGS 29)


# ---------------------------------------------------------------------------
# data generation: exact cost-to-go labels for the children of sampled states
# ---------------------------------------------------------------------------

def _walk(positions, steps, wr, wd, rng, n=SIZE):
    """`steps` random macro subgoals (a random robot to a random cell it can
    come to rest on) -- the Stage 2 recipe, so the corpus holds states a planner
    actually reaches and not only uniform placements."""
    pos = list(positions)
    for _ in range(steps):
        r = rng.randrange(len(pos))
        row = cost_row(pos, r, wr, wd, n)
        cand = np.flatnonzero(row >= 0)
        if len(cand) == 0:
            continue
        c = int(rng.choice(cand))
        pos[r] = (c % n, c // n)
    return [tuple(p) for p in pos]


def gen(a):
    from subgoal import rustexact
    rng = random.Random(a.seed)
    ids = parse_ids(a.boards)
    parents = []
    n = SIZE
    for eid in ids:
        wr, wd = board(eid)
        for _ in range(a.per_board):
            cells = rng.sample(range(n * n), ROBOTS + 1)
            pos0 = [(c % n, c // n) for c in cells[:ROBOTS]]
            target = (cells[ROBOTS] % n, cells[ROBOTS] // n)
            tidx = rng.randrange(ROBOTS)
            for _ in range(a.parents):
                pos = _walk(pos0, rng.randint(0, a.walk), wr, wd, rng)
                if pos[tidx] == target:
                    continue                    # already solved: no decision here
                parents.append((eid, pos, tidx, target))
    rng.shuffle(parents)
    parents = parents[:a.limit] if a.limit else parents

    S = len(parents)
    env_id = np.zeros(S, np.int32)
    positions = np.zeros((S, ROBOTS, 2), np.int16)
    tgt_idx = np.zeros(S, np.int8)
    tgt = np.zeros((S, 2), np.int16)
    cost = np.full((S, ROBOTS, n * n), -1, np.int8)
    ctg = np.full((S, ROBOTS, n * n), -1, np.int8)

    t0 = time.time()
    queries, slots = [], []
    for s, (eid, pos, tidx, target) in enumerate(parents):
        wr, wd = board(eid)
        env_id[s] = eid
        positions[s] = pos
        tgt_idx[s] = tidx
        tgt[s] = target
        for r in range(ROBOTS):
            row = cost_row(pos, r, wr, wd, n)
            cost[s, r] = row
            for flat in np.flatnonzero(row >= 0):
                cell = (int(flat) % n, int(flat) // n)
                if r == tidx and cell == target:
                    ctg[s, r, int(flat)] = 0     # this macro edge finishes it
                    continue
                child = tuple(pos[:r]) + (cell,) + tuple(pos[r + 1:])
                queries.append((eid, child, tidx, target))
                slots.append((s, r, int(flat)))
    print(f"[gen] {S} parent states, {len(queries)} children to label "
          f"({time.time() - t0:.1f}s of physics)", flush=True)

    vals = rustexact.ctg_batch(queries, a.sidecars, a.work, threads=a.threads,
                               tag=Path(a.out).stem)
    nnull = 0
    for (s, r, flat), v in zip(slots, vals):
        if v is None or v > 120:
            nnull += 1
            continue
        ctg[s, r, flat] = v
    lab = ctg >= 0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / (out.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, env_id=env_id, positions=positions,
                            target_idx=tgt_idx, target=tgt, cost=cost, ctg=ctg,
                            meta=np.array(json.dumps(dict(
                                boards=a.boards, per_board=a.per_board,
                                parents=a.parents, walk=a.walk, seed=a.seed,
                                n_states=S, n_labels=int(lab.sum()),
                                n_unlabelled=nnull, size=SIZE, robots=ROBOTS,
                                seconds=round(time.time() - t0, 1),
                                date=time.strftime("%Y-%m-%dT%H:%M:%S")))))
    os.replace(tmp, out)
    print(f"[gen] {int(lab.sum())} exact labels ({nnull} unlabelled), "
          f"mean cost-to-go {ctg[lab].mean():.2f}, max {ctg[lab].max()}, "
          f"{time.time() - t0:.1f}s -> {out}", flush=True)
    print(f"SPR SUBGOAL STAGE3 GEN DONE {out}")


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train(a):
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from subgoal.ctgnet import CtgNet, CtgDataset, collate

    pl.seed_everything(a.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    tr = CtgDataset(np.load(a.train, allow_pickle=False), ENV_DIR, SIZE)
    va = CtgDataset(np.load(a.val, allow_pickle=False), ENV_DIR, SIZE)
    print(f"[train] {len(tr)} train groups, {len(va)} val groups", flush=True)
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch, shuffle=True,
                                     collate_fn=collate, num_workers=a.workers,
                                     persistent_workers=a.workers > 0, drop_last=True)
    vdl = torch.utils.data.DataLoader(va, batch_size=a.batch, shuffle=False,
                                      collate_fn=collate, num_workers=a.workers,
                                      persistent_workers=a.workers > 0)
    net = CtgNet(d_model=a.d_model, recurrence=a.recurrence, lr=a.lr, pe=a.pe)
    ck = ModelCheckpoint(dirpath=a.run, filename="ctg-{epoch}-{val_top5:.4f}",
                         monitor="val_top5", mode="max", save_top_k=1,
                         save_last=True)
    trainer = pl.Trainer(max_epochs=a.epochs, accelerator="auto", devices=1,
                         default_root_dir=a.run, callbacks=[ck],
                         log_every_n_steps=20, precision=a.precision,
                         enable_progress_bar=False,
                         max_time=dict(minutes=a.max_minutes),
                         limit_val_batches=a.limit_val)
    trainer.fit(net, dl, vdl)
    if "val_top5" not in trainer.callback_metrics:
        raise SystemExit("STAGE3 TRAIN FAILED: validation never ran")
    print(f"[train] epochs run {trainer.current_epoch}, "
          f"final {dict(trainer.callback_metrics)}", flush=True)
    print(f"[train] best {ck.best_model_path} ({ck.best_model_score})")
    Path(a.run, "best.txt").write_text(str(ck.best_model_path) + "\n")
    print("SPR SUBGOAL STAGE3 TRAIN DONE")


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def plan(a):
    bench = load_bench(a.bench)
    if a.limit:
        bench = bench[:a.limit]
    env_ids = sorted({int(r["env_id"]) for r in bench})
    if a.arm == "relaxed":
        heur = P.RelaxedHeuristic()
        extra = dict(heuristic="relaxed any-stop distance of the target robot "
                               "(board only, no network)")
    elif a.arm == "net":
        heur = P.NetHeuristic(a.ckpt, env_ids, ENV_DIR, SIZE, batch=a.net_batch)
        extra = dict(heuristic="learned CtgNet cost-to-go", ckpt=str(a.ckpt))
    elif a.arm == "exact":
        heur = P.ExactHeuristic(a.sidecars, env_ids, threads=a.threads)
        extra = dict(heuristic="exact engine cost-to-go (diagnostic ceiling)")
    else:
        raise SystemExit(f"unknown arm {a.arm}")
    t0 = time.time()
    rows = P.run(bench, heur, a.k, a.expansions, concurrency=a.concurrency)
    wall = time.time() - t0
    pay = P.payload(rows, bench, a.name or f"{a.arm}_k{a.k}_e{a.expansions}",
                    a.k, a.expansions,
                    extra=dict(**extra, arm=a.arm, wall_seconds=round(wall, 1),
                               concurrency=a.concurrency,
                               heuristic_passes=int(getattr(heur, "passes", 0)),
                               slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                               date=time.strftime("%Y-%m-%dT%H:%M:%S")))
    agg = pay["systems"][list(pay["systems"])[0]]["aggregate"]
    print(f"[plan] arm={a.arm} k={a.k} exp={a.expansions}: "
          f"solved {agg['solved']}/{len(bench)}, "
          f"optimal {agg['pct_optimal_of_n']:.1f}% of {len(bench)}, "
          f"mean extra {agg['mean_regret']}, {wall:.0f}s, "
          f"{agg['h_queries_total']} heuristic queries", flush=True)
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.parent / (out.name + ".tmp")
        tmp.write_text(json.dumps(pay, indent=1) + "\n")
        os.replace(tmp, out)
        print(f"SPR SUBGOAL STAGE3 PLAN DONE {out}")
    if hasattr(heur, "close"):
        heur.close()


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def report(a):
    """The plan's one table, recomputed from move dumps by `subgoal/table.py`,
    with the Stage 3 rows appended and the gate verdict printed."""
    from subgoal import table as T
    bench = load_bench(a.bench)
    rows = [] if a.only_rows else list(T.DEFAULT_ROWS)
    for spec in a.row:
        lbl, _, path = spec.partition("=")
        rows.append((lbl, path))
    scored = [(lbl, T.score(p, bench)) for lbl, p in rows]
    print(T.markdown(scored))
    print()
    for lbl, s in scored:
        print(f"[{lbl}] {s['path']}\n    system={s['system']!r} exp={s['expansions']} "
              f"k={s['k']}: {s['optimal']}/450 optimal, {s['solved']}/450 solved, "
              f"mean extra {s['mean_extra']}\n    checks: replay failures "
              f"{len(s['replay_failures'])}, misaligned {len(s['misaligned'])}, "
              f"length disagreements {len(s['length_disagreements'])}, "
              f"below d* {len(s['below_d_star'])}")
    if a.gate:
        g = [s for lbl, s in scored if s["path"] == a.gate]
        if g:
            pct = g[0]["pct_optimal"]
            verdict = "PASS" if pct > GATE_BACKWARD_PCT else "FAIL"
            print(f"\nStage 3 gate ({GATE_BACKWARD_PCT}% of 450, the backward "
                  f"supervised planner): {pct:.1f}% -> {verdict}")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(
            {"bench": a.bench, "gate_backward_pct": GATE_BACKWARD_PCT,
             "rows": [{"label": l, **s} for l, s in scored]}, indent=1) + "\n")


def beam(a):
    """The beam-width study: every `plan_<arm>_k<k>_e<exp>.json` in a directory,
    recomputed the same way -- percent of all 450 solved move-optimally."""
    from subgoal import table as T
    bench = load_bench(a.bench)
    found = sorted(Path(a.dir).glob("plan_*_k*_e*.json"))
    recs = []
    for f in found:
        stem = f.stem.split("_")
        arm, k, exp = stem[1], int(stem[2][1:]), int(stem[3][1:])
        s = T.score(f, bench)
        pay = json.loads(f.read_text())
        proto = pay["protocol"]
        agg = list(pay["systems"].values())[0]["aggregate"]
        recs.append(dict(arm=arm, k=k, expansions=exp, path=str(f),
                         optimal=s["optimal"], pct=s["pct_optimal"],
                         solved=s["solved"], mean_extra=s["mean_extra"],
                         replay_failures=len(s["replay_failures"]),
                         mean_expansions=agg["expansions_mean"],
                         encoder_passes=agg["h_queries_total"],
                         candidates_scored=agg["h_candidates_total"],
                         proved=agg["proved_optimal_in_pruned_graph"],
                         wall_seconds=proto.get("wall_seconds")))
    recs.sort(key=lambda r: (r["arm"], r["k"]))
    print("| h | beam k | optimal % of 450 | solved / 450 | extra moves | mean expansions | encoder passes | wall s |")
    print("|---|---|---|---|---|---|---|---|")
    for r in recs:
        me = "—" if r["mean_extra"] is None else f"{r['mean_extra']:.3f}"
        print(f"| {r['arm']} | {r['k']} | {r['pct']:.1f}% ({r['optimal']}/450) "
              f"| {r['solved']}/450 | {me} | {r['mean_expansions']:.0f} "
              f"| {r['encoder_passes']} | {r['wall_seconds']} |")
    bad = [r for r in recs if r["replay_failures"]]
    print(f"\nreplay failures across all {len(recs)} payloads: "
          f"{sum(r['replay_failures'] for r in recs)}"
          + (f"  ({[r['path'] for r in bad]})" if bad else ""))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(recs, indent=1) + "\n")


# ---------------------------------------------------------------------------
# heuristic quality, independent of any search
# ---------------------------------------------------------------------------

def rank(a):
    """Learned h vs the non-learned one as RANKERS, on a labelled corpus.

    For every (state, robot) group the beam orders children by f = c + h with
    the exact edge cost c. This scores the two h's against the true f = c + ctg
    with the exact cost-to-go: Spearman, top-1, and top-5 survival (does the
    beam's five best contain a child with the truly minimal f). It isolates the
    heuristic from the search, and needs no planning run.
    """
    import torch
    from subgoal.ctgnet import CtgNet, CtgDataset, collate, group_metrics
    from subgoal.space import board as _board, relaxed_h
    store = np.load(a.data, allow_pickle=False)
    ds = CtgDataset(store, ENV_DIR, SIZE)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = CtgNet.load_from_checkpoint(a.ckpt, map_location=dev).eval().to(dev)
    n = SIZE
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=False,
                                     collate_fn=collate, num_workers=2)
    preds = []
    with torch.no_grad():
        for b in dl:
            logits = net(b["x"].to(dev), b["A_all"].to(dev), b["A_ind"].to(dev), n)
            preds.append(net.ctg_hat(logits).float().cpu().numpy())
    preds = np.concatenate(preds, 0)
    Hcache = {}
    out = {"net": [], "relaxed": [], "zero": []}
    pool = {"net": {}, "relaxed": {}, "zero": {}}
    for j, (si, r) in enumerate(ds.index):
        eid = int(ds.env_id[si])
        tidx = int(ds.tidx[si])
        tgt = (int(ds.target[si][0]), int(ds.target[si][1]))
        key = (eid, tgt)
        H = Hcache.get(key)
        if H is None:
            wr, wd = _board(eid)
            Hd = relaxed_h(tgt, wr, wd, n)
            H = Hcache[key] = [Hd.get((c % n, c // n), 1e6) for c in range(n * n)]
        y = ds.ctg[si, r].astype(np.float64)
        c = ds.cost[si, r].astype(np.float64)
        m = ds.ctg[si, r] >= 0
        cells = np.flatnonzero(m)
        if r == tidx:
            hrel = np.array([H[int(cc)] for cc in cells], np.float64)
        else:
            hrel = np.full(len(cells), float(H[int(ds.pos[si][tidx][1]) * n
                                               + int(ds.pos[si][tidx][0])]))
        for key, hh in (("net", preds[j][m]), ("relaxed", hrel),
                        ("zero", np.zeros(len(cells)))):
            out[key].append(group_metrics(hh, y[m], c[m]))
            pool[key].setdefault(si, []).append((hh, y[m], c[m]))
    summary = {}
    for k, v in out.items():
        summary[k] = {m + "_group": float(np.nanmean([d[m] for d in v]))
                      for m in ("mae", "spearman", "top1", "top5")}
        summary[k]["groups"] = len(v)
        # the real decision pools all four robots of a state
        dec = [group_metrics(np.concatenate([x[0] for x in parts]),
                             np.concatenate([x[1] for x in parts]),
                             np.concatenate([x[2] for x in parts]))
               for parts in pool[k].values()]
        summary[k]["decisions"] = len(dec)
        summary[k]["top1"] = float(np.mean([d["top1"] for d in dec]))
        summary[k]["top5"] = float(np.mean([d["top5"] for d in dec]))
        summary[k]["mae"] = summary[k]["mae_group"]
        summary[k]["spearman"] = summary[k]["spearman_group"]
    print(f"[rank] {a.data}  ({len(ds)} groups, mean "
          f"{np.mean([d['n'] for d in out['net']]):.1f} labelled children)")
    print("| scorer | Spearman rho vs true cost-to-go | top-1 of f=c+h | "
          "top-5 of f=c+h | MAE | (per (state,robot) group: top-1 / top-5) |")
    print("|---|---|---|---|---|---|")
    for k, lbl in (("net", "learned CtgNet h"),
                   ("relaxed", "any-stop relaxation h (no network)"),
                   ("zero", "h = 0 (rank by edge cost alone)")):
        d = summary[k]
        print(f"| {lbl} | {d['spearman']:.3f} | {d['top1'] * 100:.1f}% | "
              f"{d['top5'] * 100:.1f}% | {d['mae']:.2f} | "
              f"{d['top1_group'] * 100:.1f}% / {d['top5_group'] * 100:.1f}% |")
    print(f"\n(top-1 / top-5 are over the WHOLE decision: all four robots' "
          f"candidates pooled, {summary['net']['decisions']} decisions)")
    if a.out:
        Path(a.out).write_text(json.dumps(
            dict(data=a.data, ckpt=str(a.ckpt), summary=summary,
                 date=time.strftime("%Y-%m-%dT%H:%M:%S")), indent=1) + "\n")
        print(f"SPR SUBGOAL STAGE3 RANK DONE {a.out}")


# ---------------------------------------------------------------------------
# verification of the fast physics against the Stage 1 reference
# ---------------------------------------------------------------------------

def verify(a):
    """`planner.rest_cells_fast` must agree cell-for-cell and cost-for-cost with
    `space.rest_cells`, which is built on `simulate.slide` -- the physics the
    certifier uses. Checked on random joint states of bench boards."""
    from subgoal.space import rest_cells
    rng = random.Random(a.seed)
    bench = load_bench(a.bench)
    n = SIZE
    bad = checked = 0
    for inst in bench[:a.n]:
        eid = int(inst["env_id"])
        wr, wd = board(eid)
        RAY, PIR = P.rays(eid, n)
        for _ in range(a.states):
            cells = rng.sample(range(n * n), ROBOTS)
            pos = [(c % n, c // n) for c in cells]
            for r in range(ROBOTS):
                ref = rest_cells(pos[r], frozenset(p for j, p in enumerate(pos)
                                                   if j != r), wr, wd, n)
                fast = P.rest_cells_fast(cells[r], tuple(c for j, c in
                                                         enumerate(cells) if j != r),
                                         RAY, PIR)
                a_ = {(c[1] * n + c[0]): v[0] for c, v in ref.items()}
                b_ = {c: v[0] for c, v in fast.items()}
                checked += 1
                if a_ != b_:
                    bad += 1
                    if bad < 4:
                        print(f"  MISMATCH env {eid} robot {r} pos {pos}")
    print(f"[verify] fast physics vs space.rest_cells: {checked} (state, robot) "
          f"groups, {bad} mismatches")
    if bad:
        raise SystemExit("STAGE3 VERIFY FAILED")
    print("SPR SUBGOAL STAGE3 VERIFY DONE")


# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--boards", default=CFG.board_ranges["train"])
    g.add_argument("--per-board", type=int, default=2, help="puzzles per board")
    g.add_argument("--parents", type=int, default=2, help="parent states per puzzle")
    g.add_argument("--walk", type=int, default=3, help="max random macro moves")
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--seed", type=int, default=11)
    g.add_argument("--threads", type=int, default=16)
    g.add_argument("--sidecars", default=str(SPR / "results/subgoal/stage3/boards"))
    g.add_argument("--work", default=str(SPR / "results/subgoal/stage3/work"))
    g.add_argument("--out", required=True)
    g.set_defaults(fn=gen)

    t = sub.add_parser("train")
    t.add_argument("--train", required=True)
    t.add_argument("--val", required=True)
    t.add_argument("--run", required=True)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--d-model", type=int, default=192)
    t.add_argument("--recurrence", type=int, default=4)
    t.add_argument("--pe", default="none")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--precision", default="32-true")
    t.add_argument("--max-minutes", type=int, default=40)
    t.add_argument("--limit-val", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=11)
    t.set_defaults(fn=train)

    pl_ = sub.add_parser("plan")
    pl_.add_argument("--arm", required=True, choices=["relaxed", "net", "exact"])
    pl_.add_argument("--ckpt", default=None)
    pl_.add_argument("--bench", default=str(BENCH))
    pl_.add_argument("--k", type=int, default=5)
    pl_.add_argument("--expansions", type=int, default=1200)
    pl_.add_argument("--limit", type=int, default=None)
    pl_.add_argument("--concurrency", type=int, default=64)
    pl_.add_argument("--net-batch", type=int, default=64)
    pl_.add_argument("--threads", type=int, default=16)
    pl_.add_argument("--sidecars", default=str(SPR / "results/subgoal/stage3/boards"))
    pl_.add_argument("--name", default=None)
    pl_.add_argument("--out", default=None)
    pl_.set_defaults(fn=plan)

    v = sub.add_parser("verify")
    v.add_argument("--bench", default=str(BENCH))
    v.add_argument("--n", type=int, default=15, help="boards")
    v.add_argument("--states", type=int, default=8, help="random states per board")
    v.add_argument("--seed", type=int, default=3)
    v.set_defaults(fn=verify)

    r = sub.add_parser("report")
    r.add_argument("--bench", default=str(BENCH))
    r.add_argument("--row", action="append", default=[])
    r.add_argument("--only-rows", action="store_true")
    r.add_argument("--gate", default=None, help="path of the row the gate judges")
    r.add_argument("--json", dest="json_out", default=None)
    r.set_defaults(fn=report)

    rk = sub.add_parser("rank")
    rk.add_argument("--ckpt", required=True)
    rk.add_argument("--data", required=True)
    rk.add_argument("--batch", type=int, default=16)
    rk.add_argument("--out", default=None)
    rk.set_defaults(fn=rank)

    bm = sub.add_parser("beam")
    bm.add_argument("--dir", default=str(SPR / "results/subgoal/stage3"))
    bm.add_argument("--bench", default=str(BENCH))
    bm.add_argument("--json", dest="json_out", default=None)
    bm.set_defaults(fn=beam)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
