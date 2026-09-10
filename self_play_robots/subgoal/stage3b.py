"""Stage 3b of PLAN_HYBRID_ACTION_SPACE.md: the FINISH-HERE ablation.

The macro search, unchanged, with one extra action available at a popped node:
hand the position to the old sub-goal vocabulary and take whatever it certifies.
`--fin-cap 0` disables it and reproduces the recorded rows exactly, so the two
arms of the ablation differ by that one flag and nothing else.

The question: my Stage 3a portfolio leans on the old vocabulary heavily, but it
ranks with a weak guide. With a GOOD guide the macro search may already reach
those states unaided, making the finisher redundant. This is the cheap test of
that before anything bigger is built.

    python -m subgoal.stage3b --arm net --ckpt CK --fin-cap 0  --out a.json
    python -m subgoal.stage3b --arm net --ckpt CK --fin-cap 24 --out b.json
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path



def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="g16r4",
                   help="board configuration; sets the RR_* env, the board\n"
                        "directory and the board size BEFORE the planner is\n"
                        "imported, because it binds SIZE as a default argument")
    p.add_argument("--bench", default=None)
    p.add_argument("--arm", choices=["relaxed", "net"], default="net")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--expansions", type=int, default=1200)
    p.add_argument("--bound", choices=["weak", "tight"], default="tight")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--net-batch", type=int, default=64)
    p.add_argument("--fin-cap", type=int, default=0,
                   help="how many popped nodes may use the FINISH-HERE action; 0 = off")
    p.add_argument("--fin-iters", type=int, default=200)
    p.add_argument("--vocab", choices=["base", "b1", "b2"], default="b2")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shards", type=int, default=1,
                   help="split the bench into contiguous shards; the\n"
                        "FINISH-HERE arm is ~10x slower per instance than the\n"
                        "control and does not fit one qgpu_exp hour")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--name", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    # ---- configure, THEN import ------------------------------------------
    import sys, importlib
    from spr import SV
    sys.path.insert(0, str(SV))
    from scaling.configs import get, env as cfg_env
    cfg = get(a.config)
    for k in ("RR_GRID", "RR_ROBOTS", "RR_WALLS", "RR_ENV_DIR"):
        os.environ.pop(k, None)
    if not cfg.legacy:
        os.environ.update(cfg_env(cfg))

    import subgoal.table as T
    if not cfg.legacy:
        T.SIZE = int(os.environ["RR_GRID"])
        T.ENV_DIR = Path(os.environ["RR_ENV_DIR"])
    # subgoal/table.py hardcodes four robot colours, but the project derives the
    # list from RR_ROBOTS at import time, so an 8-robot configuration needs the
    # real list. Set it before space/planner import their copies from table.
    from move_planner.state import COLOR_ORDER as _CO
    T.COLOR_ORDER = list(_CO)
    import subgoal.space as SP
    SP.SIZE, SP.ENV_DIR = T.SIZE, T.ENV_DIR          # space.py holds its own copies
    SP.COLOR_ORDER = T.COLOR_ORDER
    import subgoal.planner as P                       # binds SIZE at import time
    importlib.reload(P) if getattr(P, "SIZE", None) != T.SIZE else None
    SIZE, ENV_DIR = T.SIZE, T.ENV_DIR
    print(f"[3b] config={a.config} size={SIZE} env_dir={ENV_DIR}", flush=True)

    # bench paths are quoted SV-relative in the project's docs and jobs,
    # but the driver runs from self_play_robots/, so resolve them here
    bp = Path(a.bench) if a.bench else Path(T.BENCH)
    if not bp.is_absolute() and not bp.exists():
        bp = SV / bp
    bench = T.load_bench(bp)
    if a.limit:
        bench = bench[:a.limit]
    if a.shards > 1:
        import math
        w = math.ceil(len(bench) / a.shards)
        bench = bench[a.shard * w:(a.shard + 1) * w]
    env_ids = sorted({int(r["env_id"]) for r in bench})
    if a.arm == "relaxed":
        heur = P.RelaxedHeuristic()
        extra = dict(heuristic="relaxed any-stop distance (no network)")
    else:
        heur = P.NetHeuristic(a.ckpt, env_ids, ENV_DIR, SIZE, batch=a.net_batch)
        extra = dict(heuristic="learned CtgNet cost-to-go", ckpt=str(a.ckpt))

    fin = P.make_finisher(vocab=a.vocab, iters=a.fin_iters) if a.fin_cap > 0 else None
    t0 = time.time()
    rows = P.run(bench, heur, a.k, a.expansions, concurrency=a.concurrency,
                 bound=a.bound, finisher=fin, fin_cap=a.fin_cap)
    wall = time.time() - t0

    name = a.name or f"{a.arm}_k{a.k}_fin{a.fin_cap}"
    pay = P.payload(rows, bench, name, a.k, a.expansions,
                    extra=dict(**extra, arm=a.arm, bound=a.bound,
                               fin_cap=a.fin_cap, fin_iters=a.fin_iters,
                               vocab=a.vocab, wall_seconds=round(wall, 1),
                               slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                               date=time.strftime("%Y-%m-%dT%H:%M:%S")))
    agg = pay["systems"][list(pay["systems"])[0]]["aggregate"]
    fh = sum(r.get("fin_hits", 0) or 0 for r in rows)
    fc = sum(r.get("fin_calls", 0) or 0 for r in rows)
    fa = sum(1 for r in rows if r.get("answer_from_finisher"))
    print(f"[3b] arm={a.arm} fin_cap={a.fin_cap}: solved {agg['solved']}/{len(bench)}, "
          f"optimal {agg['pct_optimal_of_n']:.1f}% of {len(bench)}, "
          f"mean extra {agg['mean_regret']}, {wall:.0f}s", flush=True)
    print(f"[3b] finisher: {fc} calls, {fh} improved the incumbent, "
          f"{fa} instances answered BY the finisher", flush=True)

    if a.out:
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.parent / (out.name + ".tmp")
        tmp.write_text(json.dumps(pay, indent=1) + "\n")
        os.replace(tmp, out)
        print(f"SPR STAGE3B DONE {out}")
    if hasattr(heur, "close"):
        heur.close()


if __name__ == "__main__":
    main()
