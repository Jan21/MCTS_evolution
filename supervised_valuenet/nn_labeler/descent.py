"""Certified-descent labeler: the NN replaces `solve_plan` in backward labeling.

The exact labeler (`nn/generate.py::rollout`) prices every candidate at every
decision with a full plan-space A* solve -- the cost wall this track exists to
remove (FINDINGS 30). Here each candidate is priced by GREEDY NN COMPLETION
instead: commit the candidate, then repeatedly take the value net's best-scored
candidate (with the exact-shortest-path fast path and, optionally, the proven
`prefix_playable` pruning) until the plan is complete. The completed plan is
REALIZED under full physics (`eval/realize.py::strict_moves`); only candidates
whose completion actually plays out are labeled. Labels are therefore certified
upper bounds on cost-to-go, in the exact labeler's own plan-cost units:

    ctg = int(completed_plan.cost()) - fixed_g(plan_before_candidate)

`is_optimal` = argmin over the decision's certified upper bounds. Every emitted
record carries the frozen 18-field schema (drop-in for the existing trainers)
plus provenance: `n`, `label_source`, `ctg_certified`, `label_model`,
`label_model_sha`. House rules kept: one vocabulary per dataset (this file
defaults to base and names outputs `backward.nnlab.jsonl`); park repairs are
not labeled; one config per process (`apply_env` before any repo import).

Instance sources: RNG sampling with the labeler's keep/attempt protocol
(`--per-graph`, attempts <= 4x), or `--instances-from <exact.jsonl>` which
replays the instance set of an existing exact corpus so descent labels can be
audited decision-by-decision at depth 0 against exact labels.

    PYTHONPATH=. python -m nn_labeler.descent --config g12r4 \
        --ckpt nn_labeler/runs/d2_mix8910_sin2d/.../epoch=*.ckpt \
        --graphs 900-1049 --per-graph 10 \
        --out scaling/data/g12r4/backward.nnlab.jsonl

Writes atomically (tmp + rename) and drops a sidecar manifest
`<out>.manifest.json` with keep/drop accounting. Final line on success:
`NNLAB DESCENT DONE <out>`.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import signal
import time
from pathlib import Path


# -- helpers shared with the exact labeler's record layout ---------------------

def _xy(p):
    return [int(p[0]), int(p[1])] if p is not None else None


class _Timeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _Timeout()


def _silent(*_a, **_k):
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True,
                   help="SizeFreeValueNet checkpoint; glob ok (exactly one match)")
    p.add_argument("--graphs", default=None,
                   help="board id spec, e.g. 900-1049; default: the config's "
                        "train+val+test ranges")
    p.add_argument("--per-graph", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-candidates", type=int, default=14)
    p.add_argument("--vocab", default="base", choices=["base", "b1", "b2"],
                   help="house rule: never mix vocabularies in one dataset")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--timeout", type=int, default=120,
                   help="per-instance wall-time cap (s), SIGALRM as in "
                        "scaling/backward_label.py")
    p.add_argument("--max-depth", type=int, default=64,
                   help="descent step cap per completion")
    p.add_argument("--no-certify", action="store_true",
                   help="skip strict_moves realization (labels then UNcertified)")
    p.add_argument("--no-prefix-check", action="store_true",
                   help="disable prefix_playable pruning inside descents")
    p.add_argument("--instances-from", default=None,
                   help="exact-corpus jsonl: replay ITS instances instead of RNG "
                        "sampling (for depth-0 label audits)")
    p.add_argument("--limit-instances", type=int, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    # One config per process: env vars must precede every repo import.
    from scaling.configs import REPO, apply_env, get
    cfg = get(a.config)
    apply_env(cfg)

    import random

    import numpy as np
    import torch

    from GridEnv import GridEnv, State, Robot_at
    from skeleton.astar import (_apply, _initial_plan, _reference_helpers,
                                _segment)
    from eval.realize import prefix_playable, strict_moves
    from nn.generate import (_context, _fixed_g, make_solver, parse_graphs,
                             random_instance)
    from nn_labeler import encode
    from nn_labeler.model import SizeFreeValueNet

    hits = sorted(glob.glob(a.ckpt))
    if len(hits) != 1:
        raise SystemExit(f"--ckpt must match exactly one file, got {len(hits)}")
    ckpt = hits[0]
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if a.device == "auto" else a.device
    model = SizeFreeValueNet.load_from_checkpoint(ckpt, map_location=device)
    model.eval().to(device)
    coord = model.hparams.pe == "coord"
    n = cfg.grid
    ckpt_sha = hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()[:16]

    solver = make_solver(a.vocab)  # propose/_expand/score machinery only; its
    # A* search budgets are irrelevant here -- solve_plan is never called.

    @torch.no_grad()
    def score_candidates(cand_recs, env_id):
        """Value-net estimated cost-to-go for candidate record dicts, batched."""
        x = torch.from_numpy(np.stack(
            [encode.node_features(r, n, coord_channels=coord)
             for r in cand_recs]))
        key = torch.tensor([encode.key_indices(r, n) for r in cand_recs])
        A_all, A_ind = encode.adjacency(str(cfg.env_dir_abs), env_id, n)
        aa = torch.as_tensor(A_all).unsqueeze(0).expand(len(cand_recs), -1, -1)
        ai = torch.as_tensor(A_ind).unsqueeze(0).expand(len(cand_recs), -1, -1)
        val = model._value(model(x.to(device), aa.to(device), ai.to(device),
                                 n, key.to(device)))
        return [float(v) for v in val]

    def cand_record(state, seg, ctx, cand):
        """Featurizer-sufficient dict for one candidate (rollout's record layout)."""
        bn_ctx, sp_ctx, open_eps = ctx
        return {
            "target": _xy(state.target),
            "target_robot": [_xy(state.target_robot.position),
                             state.target_robot.color],
            "helpers": [[_xy(h.position), h.color] for h in state.helpers],
            "seg_start": _xy(seg.start), "seg_end": _xy(seg.end),
            "seg_support": _xy(seg.fix_support), "mover_color": seg.mover.color,
            "ctx_bottlenecks": bn_ctx, "ctx_supports": sp_ctx,
            "ctx_open_endpoints": open_eps,
            "cand_bottleneck": _xy(cand.subgoal.bottleneck.position),
            "cand_support": _xy(cand.subgoal.support.position),
            "cand_helper": [_xy(cand.subgoal.helper.position),
                            cand.subgoal.helper.color],
            "cand_parent_support": _xy(cand.parent_support),
        }

    def decision(env, state, plan):
        """One open-edge decision: (seg, applied [(cand, child_plan, rec)]) or
        None when the segment pins exactly (caller fixes it via _expand)."""
        parent, child = plan.open_edges()[0]
        seg = _segment(plan, state, parent, child)
        if env.compute_exact_shortest_path_length(
                seg.start, seg.end, seg.fix_support) is not None:
            return None
        if solver.by_reference:
            seg.helpers = seg.helpers + _reference_helpers(plan, seg.mover.color)
        cands = solver.propose(env, seg.end, seg.mover, seg.helpers, seg.support)
        cands = sorted(cands, key=lambda c: solver.score(env, c))[:a.max_candidates]
        ctx = _context(plan)
        applied = []
        for cand in cands:
            cp = _apply(env, plan, parent, child, seg, cand,
                        by_reference=solver.by_reference)
            if cp is not None:
                applied.append((cand, cp, cand_record(state, seg, ctx, cand)))
        return seg, applied

    prefix_check = not a.no_prefix_check

    def nn_complete(env, state, plan, env_id):
        """Greedy net-guided completion. Returns completed plan or None."""
        for _ in range(a.max_depth):
            if plan.is_complete():
                return plan
            dec = decision(env, state, plan)
            if dec is None:
                exp = solver._expand(env, state, plan)
                if not exp:
                    return None
                plan = exp[0]
                continue
            _seg, applied = dec
            if not applied:
                return None
            ests = score_candidates([r for _, _, r in applied], env_id)
            for i in sorted(range(len(applied)), key=lambda i: ests[i]):
                cp = applied[i][1]
                if prefix_check and not prefix_playable(env, state, cp, log=None):
                    continue
                plan = cp
                break
            else:
                return None
        return plan if plan.is_complete() else None

    stats = {"attempted": 0, "kept": 0, "records": 0, "timeouts": 0,
             "decisions": 0, "cand_labeled": 0, "cand_no_complete": 0,
             "cand_uncertified": 0, "empty": 0}

    def nn_rollout(env, state, env_id):
        """Mirror of nn/generate.py::rollout with NN completion as the pricer."""
        records = []
        plan = _initial_plan(env, state)
        depth = 0
        while not plan.is_complete():
            dec = decision(env, state, plan)
            if dec is None:
                exp = solver._expand(env, state, plan)
                if not exp:
                    break
                plan = exp[0]
                continue
            seg, applied = dec
            fixed_g = _fixed_g(plan)
            stats["decisions"] += 1
            labeled = []
            for cand, cp, rec in applied:
                done = nn_complete(env, state, cp, env_id)
                if done is None:
                    stats["cand_no_complete"] += 1
                    continue
                if not a.no_certify:
                    if strict_moves(env, state, done, log=_silent) is None:
                        stats["cand_uncertified"] += 1
                        continue
                ctg = int(done.cost()) - int(fixed_g)
                labeled.append((cand, cp, rec, ctg))
            if not labeled:
                break
            best_ctg = min(c for _, _, _, c in labeled)
            for _cand, _cp, rec, ctg in labeled:
                records.append({
                    "env_id": env_id, **rec,
                    "cost_to_go": ctg, "is_optimal": ctg == best_ctg,
                    "depth": depth,
                    "n": n, "label_source": "nnlab_descent",
                    "ctg_certified": not a.no_certify,
                    "label_model": os.path.basename(ckpt),
                    "label_model_sha": ckpt_sha,
                })
            stats["cand_labeled"] += len(labeled)
            plan = next(cp for _c, cp, _r, c in labeled if c == best_ctg)
            depth += 1
        return records

    # -- instance sources ------------------------------------------------------

    def instances_from_corpus(path):
        """Unique instances of an exact corpus, in first-appearance order."""
        seen, out = set(), []
        for line in open(path):
            r = json.loads(line)
            k = (r["env_id"], tuple(r["target"]),
                 tuple(r["target_robot"][0]), r["target_robot"][1],
                 tuple((tuple(h[0]), h[1]) for h in r["helpers"]))
            if k in seen:
                continue
            seen.add(k)
            out.append((r["env_id"], State(
                target=tuple(r["target"]),
                target_robot=Robot_at(position=tuple(r["target_robot"][0]),
                                      color=r["target_robot"][1]),
                helpers=[Robot_at(position=tuple(h[0]), color=h[1])
                         for h in r["helpers"]])))
        return out

    graphs = parse_graphs(a.graphs) if a.graphs else sorted(
        {i for s in ("train", "val", "test") for i in cfg.ids(s)})

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    t0 = time.time()
    signal.signal(signal.SIGALRM, _alarm)

    def run_instance(f, env, st, env_id):
        stats["attempted"] += 1
        signal.alarm(a.timeout)
        try:
            recs = nn_rollout(env, st, env_id)
        except _Timeout:
            stats["timeouts"] += 1
            recs = []
        except Exception:
            recs = []
        finally:
            signal.alarm(0)
        if not recs:
            stats["empty"] += 1
            return False
        for r in recs:
            f.write(json.dumps(r) + "\n")
        stats["records"] += len(recs)
        stats["kept"] += 1
        return True

    with open(tmp, "w") as f:
        if a.instances_from:
            inst = instances_from_corpus(a.instances_from)
            if a.limit_instances:
                inst = inst[:a.limit_instances]
            by_env = {}
            for env_id, st in inst:
                by_env.setdefault(env_id, []).append(st)
            for env_id in sorted(by_env):
                env, _ = GridEnv.from_env(env_id)
                for st in by_env[env_id]:
                    run_instance(f, env, st, env_id)
                print(f"graph {env_id}: {stats['kept']}/{stats['attempted']} kept, "
                      f"{stats['records']} records ({time.time() - t0:.0f}s)",
                      flush=True)
        else:
            rng = random.Random(a.seed)
            _env0, s0 = GridEnv.from_env(graphs[0])
            colors = [s0.target_robot.color] + [h.color for h in s0.helpers]
            for gid in graphs:
                env, _ = GridEnv.from_env(gid)
                kept = attempts = 0
                while kept < a.per_graph and attempts < a.per_graph * 4:
                    attempts += 1
                    st = random_instance(env, colors, rng)
                    kept += run_instance(f, env, st, gid)
                print(f"graph {gid}: {kept} instances, {stats['records']} records "
                      f"so far ({time.time() - t0:.0f}s)", flush=True)
                if a.limit_instances and stats["attempted"] >= a.limit_instances:
                    break

    os.replace(tmp, out)
    manifest = {
        "config": cfg.name, "grid": n, "robots": cfg.robots,
        "vocab": a.vocab, "label_source": "nnlab_descent",
        "ckpt": ckpt, "ckpt_sha": ckpt_sha, "device": device,
        "per_graph": a.per_graph, "seed": a.seed,
        "max_candidates": a.max_candidates, "max_depth": a.max_depth,
        "certify": not a.no_certify, "prefix_check": prefix_check,
        "instances_from": a.instances_from, "graphs": a.graphs,
        "seconds": round(time.time() - t0, 1), "stats": stats,
    }
    mpath = out.with_suffix(out.suffix + ".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=1))
    print(f"[descent] stats: {stats}")
    print(f"NNLAB DESCENT DONE {out}", flush=True)


if __name__ == "__main__":
    main()
