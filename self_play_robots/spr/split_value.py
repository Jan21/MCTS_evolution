"""Encode-once value nets: a static geometry encoder run once per board, and a
cheap per-decision head. Three variants, one training harness, one baseline.

Why. The size-free value net re-encodes all n^2 cells for every decision, four
recurrence steps, although the walls never change during a search and the
decision is a few dozen marked cells. At 96x96 that is 9,217 tokens x 12 attention
passes per decision, twice per expansion, and it is why the trained planner costs
~5 minutes per puzzle there against ~1.6 s for the network-free search.

What the network actually receives (nn_labeler.encode.node_features): 9 binary
marker channels per cell -- robots, the current segment's start/end/support, the
target, and the partial-plan context (other open segments, committed sub-goal
cells). So the partial plan IS encoded, but as a sparse set of role-marked cells,
never as a graph. That is what makes a static/dynamic split possible: the
expensive part (geometry) is per board, the changing part (roles) is per decision
and sparse.

The three variants share the SAME static encoder and readout and differ only in
what the per-decision head attends over:

  sparse   tokens = marked cells only (~10-40).  Cheapest. Loses any information a
           far unmarked cell might have carried.
  reach    tokens = marked cells + every cell one walls-only slide away from a
           marked cell (~100-500). The middle option.
  inject   tokens = ALL cells, roles injected into the static embeddings, then ONE
           masked attention step over the slide graph + readout. Keeps global
           mixing; cost ~1/12 of the original per decision.

The stability probe (2026-09-10) showed the CURRENT net does NOT factorise this
way (far cells shift 17% when robots move), so none of these is a lossless
refactor. They are retrained from scratch on the same records that trained the
current size-free pair, and judged against it on a held-out split.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spr import REPO, SV
sys.path.insert(0, str(SV))

from nn_labeler import encode as nn_encode                       # noqa: E402
from nn_labeler.model import LoopedLayer                          # noqa: E402
from nn_labeler.dataset import load_corpus, group_by_decision, by_split  # noqa: E402

NUM_BINS = 96
ROLE_CH = 9                # nn_labeler.encode BASE_CHANNELS
GEO_CH = 6                 # degree_all, degree_ind, wall L/R/U/D  (size-free)


# ---------------------------------------------------------------------------
# static geometry features: symmetry-breaking input for the static encoder
# ---------------------------------------------------------------------------

def geometry_features(A_all, A_ind, n):
    """[n*n+1, GEO_CH] from the slide graph alone. No robots, no plan."""
    N = n * n
    f = np.zeros((N + 1, GEO_CH), np.float32)
    deg_a = A_all[:N, :N].sum(1); deg_i = A_ind[:N, :N].sum(1)
    f[:N, 0] = deg_a / max(1.0, deg_a.max())
    f[:N, 1] = deg_i / max(1.0, deg_i.max())
    # a wall on a side == the one-cell neighbour in that direction is NOT a slide target
    idx = np.arange(N); x, y = idx % n, idx // n
    for c, (dx, dy) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
        nx, ny = x + dx, y + dy
        ok = (nx >= 0) & (nx < n) & (ny >= 0) & (ny < n)
        nb = np.where(ok, ny * n + nx, 0)
        reach = A_all[idx, nb] > 0
        f[:N, 2 + c] = np.where(ok, (~reach).astype(np.float32), 1.0)
    return f


# ---------------------------------------------------------------------------
# the models
# ---------------------------------------------------------------------------

class StaticEncoder(nn.Module):
    """Looped transformer over the slide graph, geometry channels in, one
    embedding per cell out. Run ONCE per board."""
    def __init__(self, d=192, heads=4, recurrence=4):
        super().__init__()
        self.enc = nn.Linear(GEO_CH, d)
        self.layer = LoopedLayer(d, heads, use_global=True)
        self.rec = recurrence

    def forward(self, geo, A_all, A_ind):
        X = self.enc(geo)
        for _ in range(self.rec):
            X = self.layer(X, A_all, A_ind)
        return X                                                  # [B, N+1, d]


class Readout(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d),
                                 nn.GELU(), nn.Linear(d, NUM_BINS))

    def forward(self, tok, mask):
        # mean + max over valid tokens
        m = mask.unsqueeze(-1).float()
        mean = (tok * m).sum(1) / m.sum(1).clamp(min=1)
        mx = tok.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
        return self.mlp(torch.cat([mean, mx], -1))


class SmallTransformer(nn.Module):
    def __init__(self, d, heads=4, layers=2):
        super().__init__()
        blk = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0,
                                         batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(blk, layers)

    def forward(self, tok, mask):
        return self.tf(tok, src_key_padding_mask=~mask)


class SplitValueNet(nn.Module):
    """variant in {sparse, reach, inject}."""
    def __init__(self, variant, d=192, heads=4, static_rec=4, head_layers=2):
        super().__init__()
        self.variant = variant
        self.static = StaticEncoder(d, heads, static_rec)
        self.role = nn.Linear(ROLE_CH, d)
        if variant == "inject":
            self.dyn = LoopedLayer(d, heads, use_global=True)     # ONE masked step
        else:
            self.dyn = SmallTransformer(d, heads, head_layers)
        self.readout = Readout(d)
        self.register_buffer("bins", torch.arange(NUM_BINS, dtype=torch.float32))

    def forward(self, G, roles, A_all, A_ind, tok_idx=None, tok_mask=None):
        """G: static embeddings [B,N+1,d] (precomputed once per board).
        roles: [B,N+1,ROLE_CH]. tok_idx/tok_mask: [B,T] token selection for the
        sparse/reach variants."""
        H = G + self.role(roles)
        if self.variant == "inject":
            H = self.dyn(H, A_all, A_ind)
            return self.readout(H, torch.ones(H.shape[:2], dtype=torch.bool, device=H.device))
        b = torch.arange(H.shape[0], device=H.device).unsqueeze(1)
        tok = H[b, tok_idx]                                       # [B,T,d]
        tok = self.dyn(tok, tok_mask)
        return self.readout(tok, tok_mask)

    def ctg_hat(self, logits):
        return (F.softmax(logits, -1) * self.bins).sum(-1)


def hl_gauss(ctg, sigma=1.0):
    """The project's soft target over cost bins."""
    c = ctg.clamp(0, NUM_BINS - 1).unsqueeze(-1)
    b = torch.arange(NUM_BINS, dtype=torch.float32, device=ctg.device).view(1, -1)
    w = torch.exp(-0.5 * ((b - c) / sigma) ** 2)
    return w / w.sum(-1, keepdim=True)


# ---------------------------------------------------------------------------
# data: reuse the project's corpus loader; add geometry + token selection
# ---------------------------------------------------------------------------

def token_indices(roles_np, A_all, n, variant, cap=512):
    """Which cells the per-decision head sees. sparse: marked cells. reach: marked
    cells plus their one-slide walls-only neighbours. Always includes the global
    row (index n*n)."""
    N = n * n
    marked = np.flatnonzero(roles_np[:N].sum(1) > 0)
    if variant == "reach":
        nb = np.flatnonzero(A_all[marked, :N].sum(0) > 0) if len(marked) else np.array([], int)
        marked = np.unique(np.concatenate([marked, nb]))
    idx = np.concatenate([marked, [N]])[:cap]
    return idx


def load_groups(data_specs, splits_spec, split):
    """Records for one split across configs, grouped by decision. Uses the
    project's own --splits parser so the split is identical to the one that
    trained the current pair."""
    from scaling.configs import get
    from spr.train import parse_splits
    ranges = parse_splits(splits_spec)
    groups = []
    for spec in data_specs:
        cfg_name, path = spec.split("=", 1)
        cfg = get(cfg_name)
        env_dir = str(SV / cfg.env_dir)
        recs = load_corpus(path, cfg_name, cfg.grid, env_dir)
        for r in recs:
            # self-play records live on generated boards and carry their own dir
            r["_config"] = cfg_name; r["_n"] = cfg.grid
            r["_env_dir"] = r.get("boards_dir") or env_dir
        recs = by_split(recs, split, ranges)
        groups += group_by_decision(recs)
    return groups


class GeoCache:
    def __init__(self):
        self.c = {}

    def get(self, env_dir, env_id, n):
        k = (env_dir, int(env_id), n)
        if k not in self.c:
            A_all, A_ind = nn_encode.adjacency(env_dir, env_id, n)
            A_all = np.asarray(A_all, np.float32); A_ind = np.asarray(A_ind, np.float32)
            self.c[k] = (A_all, A_ind, geometry_features(A_all, A_ind, n))
            if len(self.c) > 64:
                self.c.pop(next(iter(self.c)))
        return self.c[k]


def make_batch(group, variant, geo: GeoCache, max_per_group=32, rng=None):
    """One decision = one group of candidate records sharing a board."""
    g = group if len(group) <= max_per_group else (rng.sample(group, max_per_group) if rng else group[:max_per_group])
    n = g[0]["_n"]; env_dir = g[0]["_env_dir"]; N = n * n
    g = [r for r in g if r["_env_dir"] == env_dir]          # one board per decision
    A_all, A_ind, gf = geo.get(env_dir, g[0]["env_id"], n)
    roles = np.stack([nn_encode.node_features(r, n) for r in g])            # [B,N+1,9]
    ctg = np.array([float(r["cost_to_go"]) for r in g], np.float32)
    opt = np.array([bool(r.get("is_optimal", False)) for r in g])
    out = dict(n=n, env_id=int(g[0]["env_id"]),
               geo=torch.from_numpy(gf).unsqueeze(0),                        # [1,N+1,6]
               A_all=torch.from_numpy(A_all).unsqueeze(0), A_ind=torch.from_numpy(A_ind).unsqueeze(0),
               roles=torch.from_numpy(roles), ctg=torch.from_numpy(ctg),
               opt=torch.from_numpy(opt))
    if variant != "inject":
        idxs = [token_indices(roles[i], A_all, n, variant) for i in range(len(g))]
        T = max(len(i) for i in idxs)
        ti = np.zeros((len(g), T), np.int64); tm = np.zeros((len(g), T), bool)
        for i, ix in enumerate(idxs):
            ti[i, :len(ix)] = ix; tm[i, :len(ix)] = True
        out["tok_idx"] = torch.from_numpy(ti); out["tok_mask"] = torch.from_numpy(tm)
    return out


# ---------------------------------------------------------------------------
# train / eval
# ---------------------------------------------------------------------------

def run_model(net, b, dev, G_cache):
    """Static encoder once per board (cached), head per candidate."""
    key = (b["env_id"], b["n"])
    if key not in G_cache:
        with torch.set_grad_enabled(net.training):
            G_cache[key] = net.static(b["geo"].to(dev), b["A_all"].to(dev), b["A_ind"].to(dev))
        if len(G_cache) > 8:
            G_cache.pop(next(iter(G_cache)))
    G = G_cache[key]
    B = b["roles"].shape[0]
    G = G.expand(B, -1, -1)
    A_all = b["A_all"].to(dev).expand(B, -1, -1); A_ind = b["A_ind"].to(dev).expand(B, -1, -1)
    if net.variant == "inject":
        return net(G, b["roles"].to(dev), A_all, A_ind)
    return net(G, b["roles"].to(dev), A_all, A_ind, b["tok_idx"].to(dev), b["tok_mask"].to(dev))


def group_metrics(pred, ctg, opt):
    """MAE, Spearman, top-1 (argmin lands on an optimal candidate), n."""
    pred = np.asarray(pred, float); ctg = np.asarray(ctg, float)
    mae = float(np.abs(pred - ctg).mean())
    if len(pred) > 1 and ctg.std() > 0 and pred.std() > 0:
        # Spearman = Pearson on average ranks (no scipy in the venv)
        def _rank(v):
            o = np.argsort(v); r = np.empty(len(v)); r[o] = np.arange(len(v))
            u, inv = np.unique(v, return_inverse=True)
            return np.array([r[inv == i].mean() for i in inv])
        rp, rc = _rank(pred), _rank(ctg)
        rho = float(np.corrcoef(rp, rc)[0, 1]) if rp.std() > 0 and rc.std() > 0 else float("nan")
    else:
        rho = float("nan")
    top1 = float(opt[int(np.argmin(pred))]) if opt.any() else float("nan")
    return mae, rho, top1


def evaluate(net, groups, variant, geo, dev):
    net.eval(); G_cache = {}
    maes, rhos, top1s = [], [], []
    with torch.no_grad():
        for g in groups:
            b = make_batch(g, variant, geo)
            pred = net.ctg_hat(run_model(net, b, dev, G_cache)).float().cpu().numpy()
            m, r, t = group_metrics(pred, b["ctg"].numpy(), b["opt"].numpy())
            maes.append(m); rhos.append(r); top1s.append(t)
    return dict(mae=float(np.mean(maes)), spearman=float(np.nanmean(rhos)),
                top1=float(np.nanmean(top1s)), decisions=len(groups))


def evaluate_baseline(ckpt, groups, geo, dev):
    """The current size-free value net on the same groups, same metrics."""
    from spr.nets import load_value
    from nn_labeler.model import LoopedValueNet  # noqa
    vn = load_value(ckpt, dev, "sizefree")
    net = getattr(vn, "net", vn)
    net.eval(); maes, rhos, top1s = [], [], []
    with torch.no_grad():
        for g in groups:
            n = g[0]["_n"]; A_all, A_ind, _ = geo.get(g[0]["_env_dir"], g[0]["env_id"], n)
            x = torch.from_numpy(np.stack([nn_encode.node_features(r, n) for r in g])).to(dev)
            Aa = torch.from_numpy(A_all).unsqueeze(0).expand(len(g), -1, -1).to(dev)
            Ai = torch.from_numpy(A_ind).unsqueeze(0).expand(len(g), -1, -1).to(dev)
            try:
                logits = net(x, Aa, Ai, n)
            except TypeError:
                logits = net(x, Aa, Ai)
            pred = net.ctg_hat(logits).float().cpu().numpy() if hasattr(net, "ctg_hat") else \
                (F.softmax(logits, -1) * torch.arange(logits.shape[-1], device=dev)).sum(-1).cpu().numpy()
            ctg = np.array([float(r["cost_to_go"]) for r in g]); opt = np.array([bool(r.get("is_optimal", False)) for r in g])
            m, r_, t = group_metrics(pred, ctg, opt)
            maes.append(m); rhos.append(r_); top1s.append(t)
    return dict(mae=float(np.mean(maes)), spearman=float(np.nanmean(rhos)),
                top1=float(np.nanmean(top1s)), decisions=len(groups))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variant", choices=["sparse", "reach", "inject"], required=True)
    p.add_argument("--data", action="append", required=True, metavar="CONFIG=PATH")
    p.add_argument("--splits", action="append", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d", type=int, default=192)
    p.add_argument("--static-rec", type=int, default=4)
    p.add_argument("--head-layers", type=int, default=2)
    p.add_argument("--eval-groups", type=int, default=400)
    p.add_argument("--baseline", default=None, help="current size-free value ckpt to compare against")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    import random
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    geo = GeoCache()
    train = load_groups(a.data, a.splits, "train")
    val = load_groups(a.data, a.splits, "val")
    rng.shuffle(val); val = val[:a.eval_groups]
    print(f"[split:{a.variant}] train decisions {len(train)}, val decisions {len(val)}, "
          f"sizes {sorted({g[0]['_n'] for g in train})}", flush=True)

    net = SplitValueNet(a.variant, a.d, static_rec=a.static_rec, head_layers=a.head_layers).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    t0 = time.time(); losses = []
    net.train(); G_cache = {}
    for step in range(a.steps):
        g = rng.choice(train)
        b = make_batch(g, a.variant, geo, rng=rng)
        G_cache.clear()                     # static encoder is trained, so no stale cache
        logits = run_model(net, b, dev, G_cache)
        loss = -(hl_gauss(b["ctg"].to(dev)) * F.log_softmax(logits, -1)).sum(-1).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step(); losses.append(loss.item())
        if (step + 1) % 250 == 0:
            print(f"[split:{a.variant}] step {step+1}/{a.steps} loss {np.mean(losses[-250:]):.4f} "
                  f"{time.time()-t0:.0f}s", flush=True)

    res = dict(variant=a.variant, steps=a.steps, d=a.d, static_rec=a.static_rec,
               head_layers=a.head_layers, train_decisions=len(train),
               train_seconds=round(time.time() - t0, 1),
               params=sum(p.numel() for p in net.parameters()))
    res["val"] = evaluate(net, val, a.variant, geo, dev)
    print(f"[split:{a.variant}] VAL  mae {res['val']['mae']:.3f}  spearman {res['val']['spearman']:.3f}  "
          f"top-1 {100*res['val']['top1']:.1f}%  ({res['val']['decisions']} decisions)", flush=True)
    if a.baseline:
        res["baseline"] = evaluate_baseline(a.baseline, val, geo, dev)
        bl = res["baseline"]
        print(f"[split:{a.variant}] BASELINE (current size-free value net, same decisions): "
              f"mae {bl['mae']:.3f}  spearman {bl['spearman']:.3f}  top-1 {100*bl['top1']:.1f}%", flush=True)

    # per-decision cost at the sizes that matter: head only, static encoder amortized
    net.eval(); timing = {}
    with torch.no_grad():
        for n in (16, 32, 64, 96):
            from scaling.configs import get
            cfgn = {16: "g16r4", 32: "g32r4", 64: "g64r4", 96: "g96r4"}[n]
            try:
                cfg = get(cfgn); env_dir = str(SV / cfg.env_dir)
                eid = 0 if n >= 80 else (2400 if n == 16 else 900)
                A_all, A_ind, gf = geo.get(env_dir, eid, n)
                N = n * n
                fake = [dict(target=[1, 1], target_robot=[[2, 2], "Red"], helpers=[[[3, 3], "Blue"], [[4, 4], "Green"], [[5, 5], "Yellow"]],
                             seg_start=[2, 2], seg_end=[1, 1], seg_support=None, mover_color="Red",
                             ctx_bottlenecks=[], ctx_supports=[], ctx_open_endpoints=[[1, 1], [2, 2]],
                             cand_bottleneck=[1, 1], cand_support=[1, 2], cand_helper=[[3, 3], "Blue"],
                             cand_parent_support=None, cost_to_go=5, is_optimal=True, env_id=eid,
                             _n=n, _env_dir=env_dir)] * 8
                b = make_batch(fake, a.variant, geo)
                Gc = {}
                torch.cuda.synchronize() if dev == "cuda" else None
                t0 = time.time(); run_model(net, b, dev, Gc); torch.cuda.synchronize() if dev == "cuda" else None
                t_static = time.time() - t0
                t0 = time.time()
                for _ in range(5): run_model(net, b, dev, Gc)
                torch.cuda.synchronize() if dev == "cuda" else None
                t_head = (time.time() - t0) / 5
                timing[n] = dict(static_once_s=round(t_static, 4), head_per_decision_s=round(t_head, 4),
                                 tokens=int(b["tok_mask"].sum(1).float().mean().item()) if "tok_mask" in b else N + 1)
            except Exception as e:
                timing[n] = dict(error=repr(e)[:80])
    res["timing"] = timing
    print(f"[split:{a.variant}] per-decision head cost (batch 8): " +
          ", ".join(f"{n}x{n}: {v.get('head_per_decision_s','?')}s over {v.get('tokens','?')} tokens" for n, v in timing.items()), flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    torch.save(net.state_dict(), out.with_suffix(".pt"))
    print(f"SPLIT VALUE DONE {out}")


if __name__ == "__main__":
    main()
