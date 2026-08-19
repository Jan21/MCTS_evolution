"""Zero-shot exact audit of a (policy, value) pair on exact-labeled decision
corpora at ANY board size (PROBLEM.md M5: "net trained via self-play at <=32
benched zero-shot at 40-64 against exact ground truth").

    PYTHONPATH=supervised_valuenet:self_play_robots python -m spr.audit \
        --policy P.ckpt --value V.ckpt --split test \
        --data g40r4=scaling/data/g40r4/backward_audit.rust.jsonl \
        --data g64r4=scaling/data/g64r4/backward_audit.rust.jsonl --out audit.json

Per config (over exact decision groups, base vocabulary, same records the
labeler's fidelity curve used -- nn_labeler/results/audit_*.json):
  policy   top1_optimal, regret@1/@3/@5 (best exact ctg among the policy's
           top-k minus the group's optimum), recall@5 (an optimal candidate is
           in the top-5 = the k=5 arena filter never loses the optimum)
  value    top1_optimal, regret (argmin of the value estimate; = nn_labeler.audit)
  pair     the planner's greedy decision: value argmin over the policy's top-5
           (what one arena expansion decides), top1_optimal + regret
  paired   pair vs value-alone / policy-alone win-loss counts on the same groups
Value predictions are computed with the labeler's own featurizer/collate
(nn_labeler.model.collate_groups, nn_labeler.encode); policy log-probs with
spr.nets.heads_logp_map -- the inference arithmetic of the searches.
Depth is kept per group (`by_depth`) because the exact corpora hold all depths.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path

import numpy as np
import torch

from nn_labeler import dataset, encode
from nn_labeler.model import collate_groups
from scaling.configs import get
from spr import REPO, SV
from spr.nets import flat, load_policy, load_value, policy_meta, collate_policy


def _abs(p):
    p = Path(p)
    if p.is_absolute():
        return p
    return (REPO / p) if (REPO / p).exists() else (SV / p)


def cand_key(r, g0, n, byref):
    hidx = {tuple(h[0]): i for i, h in enumerate(g0["helpers"])}
    cidx = {h[1]: i for i, h in enumerate(g0["helpers"])}
    hi = hidx.get(tuple(r["cand_helper"][0]))
    if hi is None and byref:
        hi = cidx.get(r["cand_helper"][1])
    if hi is None:
        return None
    return (flat(r["cand_bottleneck"], n), flat(r["cand_support"], n), hi)


@torch.no_grad()
def audit_config(policy, value, groups, n, dev, byref, batch=8):
    coord = getattr(value.hparams, "pe", "none") == "coord"
    vcoll = functools.partial(collate_groups, featurize_fn=encode.node_features,
                              adjacency_fn=encode.adjacency, key_fn=encode.key_indices,
                              coord_channels=coord, num_classes=value.hparams.num_classes)
    rows = []
    for i0 in range(0, len(groups), batch):
        gs = groups[i0:i0 + batch]
        # value predictions per record (group order preserved by collate_groups)
        b = vcoll(gs)
        val = value._value(value(b["x"].to(dev), b["A_all"].to(dev), b["A_ind"].to(dev),
                                 b["n"], b["key"].to(dev))).cpu()
        # policy log-probs per group
        metas = []
        for g in gs:
            m = policy_meta(g, n, byref=byref)
            if m is not None:
                m["_env_dir"] = g[0]["_env_dir"]
            metas.append(m)
        live = [m for m in metas if m is not None]
        lps = {}
        if live:
            pb = collate_policy(live)
            h = policy._encode(pb["x"].to(dev), pb["A_all"].to(dev), pb["A_ind"].to(dev))
            for j, m in enumerate(live):
                lps[id(m)] = policy.logp_map(h[j], m)
        for gi, (g, m) in enumerate(zip(gs, metas)):
            sel = (b["group"] == gi)
            preds = val[sel].tolist()
            ctgs = [int(round(float(r["cost_to_go"]))) for r in g]
            opts = [bool(r.get("is_optimal", False)) for r in g]
            best = min(ctgs)
            row = {"depth": int(g[0].get("depth", -1)), "n_cands": len(g)}
            # value alone
            vi = int(np.argmin(preds))
            row["value_top1"] = opts[vi]; row["value_regret"] = ctgs[vi] - best
            if m is None:
                row["policy_ok"] = False
                rows.append(row)
                continue
            row["policy_ok"] = True
            lp = lps[id(m)]
            keys = [cand_key(r, g[0], n, byref) for r in g]
            scored = sorted(((lp.get(k, -1e9), i) for i, k in enumerate(keys) if k is not None),
                            reverse=True)
            for k in (1, 3, 5):
                top = [i for _, i in scored[:k]]
                row[f"policy_regret@{k}"] = min(ctgs[i] for i in top) - best
                row[f"policy_opt@{k}"] = any(opts[i] for i in top)
            row["policy_top1"] = row["policy_opt@1"]
            top5 = [i for _, i in scored[:5]]
            pi = min(top5, key=lambda i: preds[i])
            row["pair_top1"] = opts[pi]; row["pair_regret"] = ctgs[pi] - best
            rows.append(row)
    return rows


def summarize(rows):
    n = len(rows)
    pol = [r for r in rows if r.get("policy_ok")]
    out = {"n_groups": n, "n_policy_groups": len(pol),
           "value": {"top1_optimal": sum(r["value_top1"] for r in rows) / n,
                     "regret": sum(r["value_regret"] for r in rows) / n}}
    if pol:
        m = len(pol)
        out["policy"] = {"top1_optimal": sum(r["policy_top1"] for r in pol) / m,
                         **{f"regret@{k}": sum(r[f"policy_regret@{k}"] for r in pol) / m for k in (1, 3, 5)},
                         "recall@5": sum(r["policy_opt@5"] for r in pol) / m}
        out["pair"] = {"top1_optimal": sum(r["pair_top1"] for r in pol) / m,
                       "regret": sum(r["pair_regret"] for r in pol) / m}
        out["paired"] = {
            "pair_vs_value": [sum(1 for r in pol if r["pair_regret"] < r["value_regret"]),
                              sum(1 for r in pol if r["pair_regret"] > r["value_regret"])],
            "pair_vs_policy": [sum(1 for r in pol if r["pair_regret"] < r["policy_regret@1"]),
                               sum(1 for r in pol if r["pair_regret"] > r["policy_regret@1"])]}
    by = {}
    for r in rows:
        by.setdefault(r["depth"], []).append(r)
    out["by_depth"] = {str(d): {"n": len(v),
                                "value_top1": sum(r["value_top1"] for r in v) / len(v),
                                "pair_top1": (sum(r.get("pair_top1", False) for r in v)
                                              / max(1, sum(1 for r in v if r.get("policy_ok"))))}
                       for d, v in sorted(by.items())}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policy", required=True)
    p.add_argument("--value", required=True)
    p.add_argument("--data", action="append", required=True, metavar="CONFIG=PATH")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--byref", action="store_true", help="helper slot fallback by colour (B2-trained policy)")
    p.add_argument("--limit-groups", type=int, default=None)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    policy = load_policy(a.policy, dev)
    value = load_value(a.value, dev)
    out = _abs(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"policy": a.policy, "value": a.value, "split": a.split, "byref": a.byref,
              "configs": {}}
    if out.is_file():      # resume: keep configs already audited with the same nets
        old = json.loads(out.read_text())
        if old.get("policy") == a.policy and old.get("value") == a.value and old.get("split") == a.split:
            report["configs"] = {k: v for k, v in old.get("configs", {}).items() if v.get("n_groups")}
            print(f"[audit] resuming {out}: {sorted(report['configs'])} done", flush=True)

    def _write():
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=1))
        os.replace(tmp, out)

    for item in a.data:
        name, path = item.split("=", 1)
        if name in report["configs"]:
            print(f"[audit] {name}: done earlier, skipped", flush=True)
            continue
        cfg = get(name)
        recs = dataset.load_corpus(str(_abs(path)), cfg.name, cfg.grid, cfg.env_dir_abs)
        if a.split != "all":
            recs = dataset.by_split(recs, a.split, {cfg.name: {a.split: cfg.ids(a.split)}})
        groups = dataset.group_by_decision(recs)
        groups = [g for g in groups if len(g) >= 2]
        if a.limit_groups:
            groups = groups[:a.limit_groups]
        if not groups:
            report["configs"][name] = {"n_groups": 0}
            print(f"[audit] {name}: EMPTY", flush=True)
            continue
        # dense masks are (n^2+1)^2 per record: shrink the group batch above 40x40
        bs = a.batch if cfg.grid <= 40 else max(1, int(a.batch * ((40 ** 2 + 1) / (cfg.grid ** 2 + 1)) ** 1.5))
        rows = audit_config(policy, value, groups, cfg.grid, dev, a.byref, bs)
        s = summarize(rows)
        s["path"] = str(path)
        report["configs"][name] = s
        _write()
        pol = s.get("policy", {})
        print(f"[audit] {name} n={cfg.grid} groups={s['n_groups']} "
              f"value top1={s['value']['top1_optimal']:.3f} regret={s['value']['regret']:.3f} | "
              f"policy top1={pol.get('top1_optimal', float('nan')):.3f} r@1={pol.get('regret@1', float('nan')):.3f} "
              f"r@5={pol.get('regret@5', float('nan')):.3f} recall@5={pol.get('recall@5', float('nan')):.3f} | "
              f"pair top1={s.get('pair', {}).get('top1_optimal', float('nan')):.3f} "
              f"regret={s.get('pair', {}).get('regret', float('nan')):.3f}", flush=True)
    _write()
    print(f"SPR AUDIT DONE {out}", flush=True)


if __name__ == "__main__":
    main()
