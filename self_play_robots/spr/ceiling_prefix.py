"""Stage 1 of PLAN_HYBRID_ACTION_SPACE.md: the ceiling of
"hand-written vocabulary + a macro-move head start".

The 24x24 and 16x16 plan decompositions both found the same thing: about one
optimal plan in five needs the TARGET robot to move at an intermediate point,
which the hand-written (bottleneck, support, helper) vocabulary cannot say. This
module measures how much of that vocabulary's ceiling gap the missing structure
accounts for, WITHOUT changing the vocabulary.

Method, per instance:
  candidates   the empty prefix, plus one macro move (a robot slides to a cell
               it can come to rest on with the others frozen, at its true slide
               cost) -- by the target robot only (--prefix target) or by any
               robot (--prefix any);
  each candidate becomes an ordinary instance with the moved robot's new
  position, and is handed to `spr.ceiling._work` UNCHANGED, so the language, the
  caps, the realization and the certification are all exactly the recorded
  probe's;
  score        prefix_cost + best_realizable_moves of that candidate;
  the instance's ceiling is the MINIMUM over candidates that resolved.

The empty-prefix arm is recomputed here under the same caps, so the comparison
is internal and does not depend on the recorded payload's caps.

    PYTHONPATH=supervised_valuenet:self_play_robots python -m spr.ceiling_prefix \
        --config g16r4 --instances eval/data/bench450.jsonl --vocab b2 \
        --prefix any --idx-list 0,1,2 --out results/ceiling/prefix_g16r4.json
"""
from __future__ import annotations

import argparse, json, multiprocessing as mp, os, sys, time
from pathlib import Path
from spr import RESULTS, SV


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--instances", required=True)
    p.add_argument("--vocab", choices=["base", "b1", "b2"], default="b2")
    p.add_argument("--prefix", choices=["none", "target", "any"], default="any")
    p.add_argument("--idx-list", default=None, help="comma-separated instance indices")
    p.add_argument("--idx-from", default=None,
                   help="comma-separated payload paths; union their instance_indices/idx_list")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--time-cap", type=float, default=30.0)
    p.add_argument("--max-iters", type=int, default=100_000)
    p.add_argument("--max-frontier", type=int, default=200_000)
    p.add_argument("--park-cap", type=int, default=2)
    p.add_argument("--slack", type=float, default=0.0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    # one config per process: RR_* before any repo import (children inherit)
    sys.path.insert(0, str(SV))
    from scaling.configs import get, env as cfg_env
    cfg = get(a.config)
    for k in ("RR_GRID", "RR_ROBOTS", "RR_WALLS", "RR_ENV_DIR"):
        os.environ.pop(k, None)
    if not cfg.legacy:
        os.environ.update(cfg_env(cfg))

    from spr.ceiling import _init_worker, _work
    from subgoal.space import rest_cells
    from GridEnv import GridEnv
    from simulate import wall_sets, _board_size

    inst_path = Path(a.instances)
    if not inst_path.is_absolute():
        inst_path = SV / inst_path
    insts = [json.loads(l) for l in inst_path.read_text().splitlines() if l.strip()]
    if a.idx_from:
        u = set()
        for q in a.idx_from.split(","):
            d = json.loads(Path(q).read_text())
            u |= set(d.get("instance_indices") or d.get("idx_list") or [])
        sel = sorted(u)
    elif a.idx_list:
        sel = [int(x) for x in a.idx_list.split(",")]
    else:
        sel = list(range(len(insts)))[::a.stride][:a.n]

    # ---- build the candidate set -------------------------------------------
    envs, items, meta = {}, [], {}
    for i in sel:
        inst = insts[i]
        env = envs.get(inst["env_id"])
        if env is None:
            env, _ = GridEnv.from_env(inst["env_id"])
            envs.clear(); envs[inst["env_id"]] = env
        size = _board_size(env.grid_data, None)
        wr, wd = wall_sets(env.grid_data, size)
        pos = [tuple(p) for p in inst["positions"]]
        cands = [(0, None, None)]                        # the empty prefix
        movers = ([inst["target_idx"]] if a.prefix == "target"
                  else list(range(len(pos))) if a.prefix == "any" else [])
        for r in movers:
            blockers = frozenset(pos[j] for j in range(len(pos)) if j != r)
            for cell, (cost, _d, _p) in rest_cells(pos[r], blockers, wr, wd, size).items():
                cands.append((cost, r, cell))
        for c, (cost, r, cell) in enumerate(cands):
            np_ = list(pos)
            if r is not None:
                np_[r] = cell
            key = i * 10_000 + c
            meta[key] = dict(inst_idx=i, prefix_cost=cost, prefix_robot=r,
                             prefix_cell=list(cell) if cell else None)
            items.append((key, dict(env_id=inst["env_id"], positions=[list(x) for x in np_],
                                    target=inst["target"], target_idx=inst["target_idx"],
                                    d_star=inst.get("d_star"))))

    items.sort(key=lambda t: t[1]["env_id"])
    caps = dict(max_iters=a.max_iters, max_frontier=a.max_frontier,
                time_cap=a.time_cap, park_cap=a.park_cap, slack=a.slack)
    print(f"[prefix] {a.config} vocab={a.vocab} prefix={a.prefix} instances={len(sel)} "
          f"candidates={len(items)} workers={a.workers} caps={caps}", flush=True)

    # ---- run ----------------------------------------------------------------
    t0 = time.time(); raw = []
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers, initializer=_init_worker,
                  initargs=(a.config, a.vocab, caps)) as pool:
        for k, res in enumerate(pool.imap_unordered(_work, items, chunksize=4)):
            raw.append(res)
            if (k + 1) % 200 == 0 or k + 1 == len(items):
                print(f"[prefix] {k+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)

    # ---- score: three arms from the same candidates ------------------------
    # An arm is a filter over prefixes. Ties are broken by the SHORTER prefix,
    # so "won by a prefix" always means strictly fewer moves, never a tie.
    ARMS = {
        "baseline": lambda m: m["prefix_robot"] is None,
        "target":   lambda m: m["prefix_robot"] is None
                              or m["prefix_robot"] == insts[m["inst_idx"]]["target_idx"],
        "any":      lambda m: True,
    }
    best = {k: {} for k in ARMS}
    for res in raw:
        m = meta[res["idx"]]
        b = res.get("best_realizable_moves")
        if b is None:
            continue
        total = m["prefix_cost"] + b
        for arm, keep in ARMS.items():
            if not keep(m):
                continue
            cur = best[arm].get(m["inst_idx"])
            if cur is None or (total, m["prefix_cost"]) < (cur["total"], cur["prefix_cost"]):
                best[arm][m["inst_idx"]] = dict(total=total, category=res["category"],
                                                capped=res["capped"], **m)

    ncand = {}
    for m in meta.values():
        ncand[m["inst_idx"]] = ncand.get(m["inst_idx"], 0) + 1

    rows = []
    for i in sel:
        row = dict(idx=i, env_id=insts[i]["env_id"], d_star=insts[i].get("d_star"),
                   n_candidates=ncand.get(i, 0))
        for arm in ARMS:
            b = best[arm].get(i)
            row[f"{arm}_total"] = (b or {}).get("total")
            row[f"{arm}_prefix_cost"] = (b or {}).get("prefix_cost")
            row[f"{arm}_robot"] = (b or {}).get("prefix_robot")
            row[f"{arm}_cell"] = (b or {}).get("prefix_cell")
        rows.append(row)

    def arm_summary(arm):
        got = [r for r in rows if r[f"{arm}_total"] is not None and r["d_star"] is not None]
        ex = [r[f"{arm}_total"] - r["d_star"] for r in got]
        base = [r for r in got if (r[f"{arm}_prefix_cost"] or 0) > 0]
        return dict(n_scored=len(got),
                    mean_extra=(sum(ex) / len(ex)) if ex else None,
                    n_optimal=sum(1 for v in ex if v == 0),
                    pct_optimal=(100.0 * sum(1 for v in ex if v == 0) / len(ex)) if ex else None,
                    n_below_dstar=sum(1 for v in ex if v < 0),
                    n_strictly_helped_by_prefix=len(base),
                    mean_winning_prefix_cost=(sum(r[f"{arm}_prefix_cost"] for r in base) / len(base)) if base else None)

    summary = dict(n_instances=len(rows),
                   arms={arm: arm_summary(arm) for arm in ARMS},
                   candidates_total=len(items),
                   wall_seconds=round(time.time() - t0, 1))
    print("\n[prefix] summary:", json.dumps(summary, indent=1), flush=True)

    out = Path(a.out)
    if not out.is_absolute():
        out = (RESULTS.parent / out) if str(out).startswith("results/") else (RESULTS / out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(config=a.config, vocab=a.vocab, prefix=a.prefix,
                   instances=str(inst_path), idx_list=sel, caps=caps,
                   slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                   date=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   summary=summary, rows=rows)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n")
    os.replace(tmp, out)
    print(f"SPR PREFIX CEILING DONE {out}")


if __name__ == "__main__":
    main()
