"""M6 Phase-0 GPU smoke helpers (one subcommand per part; see jobs/m6_smoke.slurm).

Everything writes JSON lines to --out so a walltime kill keeps earlier parts.
Boards/instances live in the job's scratch dir; nothing touches pinned data.
"""
from __future__ import annotations

import argparse, json, os, pickle, random, sys, time
from pathlib import Path


def _log(out, obj):
    with open(out, "a") as f:
        f.write(json.dumps(obj) + "\n")
    print("[m6smoke]", json.dumps(obj), flush=True)


def boards(a):
    """Fresh lean boards + tiny instance files at n=64/80/96 (smoke ids 30900+)."""
    sys.path.insert(0, str(Path(a.sv)))
    from scaling.configs import get, env as cfg_env
    for name, nb, per in (("g64r4", 1, 2), ("g80r4", 2, 3), ("g96r4", 1, 2)):
        cfg = get(name)
        os.environ.update(cfg_env(cfg))
        d = Path(a.work) / f"boards_{name}"
        d.mkdir(parents=True, exist_ok=True)
        os.environ["RR_ENV_DIR"] = str(d)
        for m in list(sys.modules):
            if m.split(".")[0] in ("GridEnv", "simulate", "nn", "nn_labeler",
                                   "move_planner", "skeleton", "heuristics"):
                del sys.modules[m]
        from nn_labeler import leanboard
        from nn.generate import random_instance
        from move_planner.state import COLOR_ORDER
        t0 = time.time()
        rows = []
        for i in range(30900, 30900 + nb):
            leanboard.write_board(str(d), i, cfg.grid, walls=cfg.walls,
                                  robots=cfg.robots, seed=6001)
            env, s0 = leanboard.from_env(i, env_dir=str(d))
            rng = random.Random(100 + i)
            colors = [s0.target_robot.color] + [h.color for h in s0.helpers]
            for _ in range(per):
                st = random_instance(env, colors, rng)
                by = {r.color: tuple(r.position) for r in [st.target_robot] + list(st.helpers)}
                order = [c for c in COLOR_ORDER if c in by]
                rows.append({"env_id": i, "positions": [list(by[c]) for c in order],
                             "target": list(st.target),
                             "target_idx": order.index(st.target_robot.color), "d_star": 0})
        inst = Path(a.work) / f"inst_{name}.jsonl"
        inst.write_text("".join(json.dumps(r) + "\n" for r in rows))
        _log(a.out, {"part": "boards", "config": name, "boards": nb,
                     "instances": len(rows), "seconds": round(time.time() - t0, 1)})


def micro(a):
    """Pure net-call latency + peak CUDA memory at n=64/80/96 (policy encoder =
    the same LoopedLayer stack the value net runs; batch 1)."""
    sys.path.insert(0, str(Path(a.sv)))
    import torch
    from spr.nets import load_policy, policy_features
    from nn_labeler import encode
    pol = load_policy(a.policy, "cuda")
    for name, n in (("g64r4", 64), ("g80r4", 80), ("g96r4", 96)):
        d = Path(a.work) / f"boards_{name}"
        rec = {"seg_start": [1, 1], "seg_end": [n - 2, n - 2], "seg_support": None,
               "target_robot": [[1, 1], "Red"], "helpers": [[[2, 2], "Blue"]],
               "ctx_open_endpoints": [], "ctx_bottlenecks": [], "ctx_supports": []}
        x = torch.from_numpy(policy_features(rec, n))[None].cuda()
        A_all, A_ind = encode.adjacency(str(d), 30900, n)
        import numpy as np
        aa = torch.as_tensor(np.asarray(A_all, np.float32))[None].cuda()
        ai = torch.as_tensor(np.asarray(A_ind, np.float32))[None].cuda()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            h = pol._encode(x, aa, ai); torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3):
                h = pol._encode(x, aa, ai)
            torch.cuda.synchronize()
        _log(a.out, {"part": "micro", "n": n,
                     "encode_ms": round((time.time() - t0) / 3 * 1000),
                     "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
        del x, aa, ai, h
        torch.cuda.empty_cache()


def trainprobe(a):
    """One real optimizer step at n=80: policy (1 decision) and value (1 group,
    --records from the generation part). Reports peak memory or OOM."""
    sys.path.insert(0, str(Path(a.sv)))
    import torch
    from nn_labeler.model import SizeFreeValueNet, collate_groups
    from nn_labeler import encode, dataset
    from spr.nets import SizeFreePolicyNet, PolicyGroupDataset, collate_policy
    recs = [json.loads(l) for l in open(a.records)]
    for r in recs:
        r["_n"], r["_config"], r["_env_dir"] = 80, "g80r4", r["boards_dir"]
    groups = [g for g in dataset.group_by_decision(recs) if len(g) >= 2]
    res = {"part": "trainprobe", "n": 80, "groups": len(groups)}
    if not groups:
        res["error"] = "no groups"; _log(a.out, res); return
    # value step: 1 group, capped records
    try:
        vnet = SizeFreeValueNet.load_from_checkpoint(a.value, map_location="cuda").cuda().train()
        opt = torch.optim.AdamW(vnet.parameters(), lr=1e-4)
        import functools
        coll = functools.partial(collate_groups, featurize_fn=encode.node_features,
                                 adjacency_fn=encode.adjacency, key_fn=encode.key_indices,
                                 coord_channels=False, num_classes=96)
        g = sorted(groups, key=len)[0][: a.value_records]
        torch.cuda.reset_peak_memory_stats()
        b = coll([g])
        loss = vnet.training_step({k: (v.cuda() if hasattr(v, "cuda") else v)
                                   for k, v in b.items()}, 0)
        loss.backward(); opt.step()
        res["value_records"] = len(g)
        res["value_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        del vnet, opt, b, loss
    except torch.cuda.OutOfMemoryError:
        res["value_peak_gb"] = "OOM"
    torch.cuda.empty_cache()
    # policy step: 1 decision meta
    try:
        pnet = SizeFreePolicyNet.load_from_checkpoint(a.policy, map_location="cuda").cuda().train()
        opt = torch.optim.AdamW(pnet.parameters(), lr=1e-4)
        ds = PolicyGroupDataset(groups[:4], byref=True)
        torch.cuda.reset_peak_memory_stats()
        pb = collate_policy([ds[0]])
        loss = pnet.training_step({k: (v.cuda() if hasattr(v, "cuda") else v)
                                   for k, v in pb.items()}, 0)
        loss.backward(); opt.step()
        res["policy_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except torch.cuda.OutOfMemoryError:
        res["policy_peak_gb"] = "OOM"
    _log(a.out, res)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["boards", "micro", "trainprobe"])
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sv", default="supervised_valuenet")
    p.add_argument("--policy"); p.add_argument("--value")
    p.add_argument("--records"); p.add_argument("--value-records", type=int, default=2)
    a = p.parse_args()
    {"boards": boards, "micro": micro, "trainprobe": trainprobe}[a.cmd](a)


if __name__ == "__main__":
    main()
