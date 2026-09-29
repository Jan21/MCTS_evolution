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

# node_features channels: 0 seg_start, 1 seg_end, 2 cand_bottleneck, 3 cand_support,
# 4 cand_helper, 5 all helper robots, 6-8 partial-plan context. Within ONE decision
# every candidate shares 0,1,5,6,7,8 and differs only in 2,3,4.
CAND_CH = (2, 3, 4)
SHARED_CH = (0, 1, 5, 6, 7, 8)
_shared_mismatch = [0]


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
    """The reference net's readout: gather the 5 decision-defining cells (the
    project's `encode.key_indices`: segment start/end, candidate bottleneck,
    support, helper) and concatenate. A pooled readout dilutes the candidate
    among every marked cell and was measurably worse."""
    def __init__(self, d, extra=0):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(5 * d + extra, d), nn.GELU(), nn.Linear(d, d),
                                 nn.GELU(), nn.Linear(d, NUM_BINS))
        if extra:   # the extra input columns start at zero: step 0 ignores them
            with torch.no_grad():
                self.mlp[0].weight[:, 5 * d:].zero_()

    def forward(self, five, extra=None):                          # [B,5,d], [B,extra]
        h = five.reshape(five.shape[0], -1)
        if extra is not None:
            h = torch.cat([h, extra], -1)
        return self.mlp(h)


class SmallTransformer(nn.Module):
    def __init__(self, d, heads=4, layers=2):
        super().__init__()
        blk = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0,
                                         batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(blk, layers)

    def forward(self, tok, mask):
        return self.tf(tok, src_key_padding_mask=~mask)


PAIR_H = 8   # BFS horizon in slide moves; beyond it a pair is "unreached"


def slide_pair_features(key, A, H=PAIR_H):
    """[B,5] key cells, [N+1,N+1] slide adjacency (rows: from, cols: to) ->
    [B, 40]: for the 20 ordered pairs (i != j) of key cells, the directed
    slide-graph distance from key_i to key_j as d/H (clipped) and an
    unreached flag. Exact BFS in H masked matmuls, shared by every candidate."""
    B, K = key.shape
    N1 = A.shape[0]
    Ab = (A > 0).to(A.dtype)
    reach = F.one_hot(key, N1).to(A.dtype)                        # [B,K,N1]
    dist = torch.full((B, K, N1), float(H + 1), device=A.device, dtype=A.dtype)
    dist = torch.where(reach > 0, torch.zeros_like(dist), dist)
    for h in range(1, H + 1):
        nxt = (torch.matmul(reach, Ab) > 0).to(A.dtype)
        newly = (nxt > 0) & (dist > H)
        dist = torch.where(newly, torch.full_like(dist, float(h)), dist)
        reach = nxt
    idx = key.unsqueeze(1).expand(B, K, K)                          # [B,K,K]: dist[b,i,key[b,j]]
    dp = dist.gather(2, idx)                                        # [B,K,K]
    off = ~torch.eye(K, dtype=torch.bool, device=A.device)
    dp = dp[:, off].reshape(B, K * (K - 1))                         # [B,20]
    return torch.cat([(dp / H).clamp(max=1.0), (dp > H).to(A.dtype)], -1)


class SplitValueNet(nn.Module):
    """variant in {sparse, reach, inject}."""
    def __init__(self, variant, d=192, heads=4, static_rec=4, head_layers=2, dyn_steps=1):
        super().__init__()
        self.variant = variant
        self.dyn_steps = dyn_steps            # dec1: candidate-aware masked hops after the shared encoding
        # dec0 / dec1: PER-DECISION encode once. The shared context (segment,
        # robots, plan context; the same channels the policy reads) goes through
        # the full looped encoder once per decision; each candidate then costs a
        # readout (dec0) or one masked looped step with its 3 candidate channels
        # injected (dec1). No per-board static encoder at all.
        self.static = None if variant.startswith("dec") else StaticEncoder(d, heads, static_rec)
        self.role = nn.Linear(ROLE_CH, d)
        if variant in ("mono", "dec0", "dec0f", "dec0g", "dec1"):
            # the reference net's encoder: roles in, looped layer over all cells
            self.mono_enc = nn.Linear(ROLE_CH, d)
            self.mono_layer = LoopedLayer(d, heads, use_global=True)
            self.mono_rec = static_rec
        if variant in ("inject", "dec1"):
            self.dyn = LoopedLayer(d, heads, use_global=True)     # ONE masked step
        elif variant in ("sparse", "reach"):
            self.dyn = SmallTransformer(d, heads, head_layers)
        else:
            self.dyn = None
        self.readout = Readout(d, extra=40 if variant == "dec0g" else 0)
        if variant == "dec0f":
            # candidate geometry side path: zero-init last layer, so step 0 == dec0
            self.feat = nn.Sequential(nn.Linear(40, d), nn.GELU(), nn.Linear(d, NUM_BINS))
            nn.init.zeros_(self.feat[-1].weight); nn.init.zeros_(self.feat[-1].bias)
        self.register_buffer("bins", torch.arange(NUM_BINS, dtype=torch.float32))

    def forward(self, G, roles, A_all, A_ind, key, tok_idx=None, tok_mask=None, shared=None):
        """G: static embeddings [B,N+1,d] (once per board). roles: [B,N+1,9].
        key: [B,5] the readout cells. tok_idx/tok_mask: [B,T] token selection for
        sparse/reach, whose first five tokens ARE the key cells."""
        b = torch.arange(roles.shape[0], device=roles.device).unsqueeze(1)
        if self.variant in ("dec0", "dec0f", "dec0g", "dec1"):
            X = self.mono_enc(shared)                             # [1,N+1,d], once per decision
            for _ in range(self.mono_rec):
                X = self.mono_layer(X, A_all[:1], A_ind[:1])
            X = X.expand(roles.shape[0], -1, -1)
            if self.variant == "dec1":
                H = X + self.role(roles)                          # roles: candidate channels only
                for _ in range(self.dyn_steps):
                    H = self.dyn(H, A_all, A_ind)
                return self.readout(H[b, key])
            if self.variant == "dec0f":
                return self.readout(X[b, key]) + self.feat(slide_pair_features(key, A_all[0]))
            if self.variant == "dec0g":
                return self.readout(X[b, key], slide_pair_features(key, A_all[0]))
            return self.readout(X[b, key])
        if self.variant == "mono":
            X = self.mono_enc(roles)
            for _ in range(self.mono_rec):
                X = self.mono_layer(X, A_all, A_ind)
            return self.readout(X[b, key])
        H = G + self.role(roles)
        if self.variant == "inject":
            H = self.dyn(H, A_all, A_ind)
            return self.readout(H[b, key])
        tok = H[b, tok_idx]                                       # [B,T,d]
        tok = self.dyn(tok, tok_mask)
        return self.readout(tok[:, :5])

    def warm_start(self, ckpt):
        """Copy the baseline's weights into every component with a matching shape:
        the looped layer (static encoder and the inject/mono layers), the 9->d
        role/input encoder, and the 5-cell readout head, which is the same shape.
        Cold training of this family collapses to the label marginal (the
        project's FINDINGS 50/57: 7 of 8 cold runs); its own loop always warm-starts."""
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        own = self.state_dict(); n_copied = 0
        for k, v in own.items():
            src = None
            if k.startswith("static.layer."):      src = "layer." + k[len("static.layer."):]
            elif k.startswith("mono_layer."):      src = "layer." + k[len("mono_layer."):]
            elif k.startswith("dyn.") and self.variant in ("inject", "dec1"): src = "layer." + k[len("dyn."):]
            elif k.startswith("role."):            src = "enc." + k[len("role."):]
            elif k.startswith("mono_enc."):        src = "enc." + k[len("mono_enc."):]
            elif k.startswith("readout.mlp."):     src = "head." + k[len("readout.mlp."):]
            if src in sd and sd[src].shape == v.shape:
                own[k] = sd[src].clone(); n_copied += 1
            elif src in sd and k == "readout.mlp.0.weight" and sd[src].shape[0] == v.shape[0]:
                w = torch.zeros_like(v); w[:, :sd[src].shape[1]] = sd[src]      # dec0g: widened first layer
                own[k] = w; n_copied += 1
        self.load_state_dict(own)
        return n_copied

    def ctg_hat(self, logits):
        return (F.softmax(logits, -1) * self.bins).sum(-1)


def rank_loss(cost, opt):
    """The reference net's `_rank` for ONE group: a softmax over -cost across the
    candidates, with the target mass spread over the optimal ones. This is what
    trains within-group ORDER, which the classification loss alone does not."""
    o = opt.float()
    if o.sum() == 0:
        return cost.sum() * 0
    return -(o / o.sum() * F.log_softmax(-cost, 0)).sum()


def hl_gauss(ctg, sigma=1.0):
    """The project's soft target over cost bins."""
    c = ctg.clamp(0, NUM_BINS - 1).unsqueeze(-1)
    b = torch.arange(NUM_BINS, dtype=torch.float32, device=ctg.device).view(1, -1)
    w = torch.exp(-0.5 * ((b - c) / sigma) ** 2)
    return w / w.sum(-1, keepdim=True)


# ---------------------------------------------------------------------------
# data: reuse the project's corpus loader; add geometry + token selection
# ---------------------------------------------------------------------------

def token_indices(key, roles_np, A_all, n, variant, cap=512):
    """Which cells the per-decision head sees, with the 5 readout cells FIRST so
    the readout can take tok[:, :5]. sparse: + the other marked cells. reach: +
    every cell one walls-only slide from any marked cell. Global row last."""
    N = n * n
    key = [int(k) for k in key]
    marked = [int(i) for i in np.flatnonzero(roles_np[:N].sum(1) > 0) if int(i) not in key]
    extra = []
    if variant == "reach":
        seeds = np.array(key + marked, int)
        seeds = seeds[seeds < N]
        nb = np.flatnonzero(A_all[seeds, :N].sum(0) > 0) if len(seeds) else np.array([], int)
        extra = [int(i) for i in nb if int(i) not in key and int(i) not in marked]
    idx = key + marked + extra + [N]
    return np.array(idx[:cap], np.int64)


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
    keys = np.array([[int(i) for i in nn_encode.key_indices(r, n)] for r in g], np.int64)  # [B,5]
    ctg = np.array([float(r["cost_to_go"]) for r in g], np.float32)
    opt = np.array([bool(r.get("is_optimal", False)) for r in g])
    shared = roles[:1].copy(); shared[:, :, list(CAND_CH)] = 0.0                 # [1,N+1,9]
    if (roles[:, :, list(SHARED_CH)] != roles[:1, :, list(SHARED_CH)]).any():
        _shared_mismatch[0] += 1
        if _shared_mismatch[0] in (1, 10, 100):
            print(f"[split] WARNING shared channels differ within a decision ({_shared_mismatch[0]} groups so far)", flush=True)
    roles_in = roles
    if variant.startswith("dec"):
        roles_in = roles.copy(); roles_in[:, :, list(SHARED_CH)] = 0.0             # candidate channels only
    out = dict(n=n, env_id=int(g[0]["env_id"]), shared=torch.from_numpy(shared),
               geo=torch.from_numpy(gf).unsqueeze(0),                        # [1,N+1,6]
               A_all=torch.from_numpy(A_all).unsqueeze(0), A_ind=torch.from_numpy(A_ind).unsqueeze(0),
               roles=torch.from_numpy(roles_in), roles_full=torch.from_numpy(roles),
               key=torch.from_numpy(keys), ctg=torch.from_numpy(ctg),
               opt=torch.from_numpy(opt))
    if variant in ("sparse", "reach"):
        idxs = [token_indices(keys[i], roles[i], A_all, n, variant) for i in range(len(g))]
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
    if net.variant in ("dec0", "dec0f", "dec0g", "dec1"):
        B = b["roles"].shape[0]
        A_all = b["A_all"].to(dev).expand(B, -1, -1); A_ind = b["A_ind"].to(dev).expand(B, -1, -1)
        return net(None, b["roles"].to(dev), A_all, A_ind, b["key"].to(dev), shared=b["shared"].to(dev))
    if key not in G_cache:
        with torch.set_grad_enabled(net.training):
            G_cache[key] = net.static(b["geo"].to(dev), b["A_all"].to(dev), b["A_ind"].to(dev))
        if len(G_cache) > 8:
            G_cache.pop(next(iter(G_cache)))
    G = G_cache[key]
    B = b["roles"].shape[0]
    G = G.expand(B, -1, -1)
    A_all = b["A_all"].to(dev).expand(B, -1, -1); A_ind = b["A_ind"].to(dev).expand(B, -1, -1)
    key = b["key"].to(dev)
    if net.variant in ("inject", "mono"):
        return net(G, b["roles"].to(dev), A_all, A_ind, key)
    return net(G, b["roles"].to(dev), A_all, A_ind, key, b["tok_idx"].to(dev), b["tok_mask"].to(dev))


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
    """The current size-free value net on the same groups, same metrics, built
    with the project's own collate (5-cell readout key included) so it is called
    exactly as training and the planner call it."""
    from spr.nets import load_value
    from nn_labeler.model import collate_groups
    vn = load_value(ckpt, dev, "sizefree")
    net = getattr(vn, "net", vn); net.eval()
    maes, rhos, top1s = [], [], []
    with torch.no_grad():
        for g in groups:
            n = g[0]["_n"]
            b = collate_groups([g], nn_encode.node_features, nn_encode.adjacency,
                               nn_encode.key_indices)
            logits = net(b["x"].to(dev), b["A_all"].to(dev), b["A_ind"].to(dev), n, b["key"].to(dev))
            pred = net._value(logits).float().cpu().numpy() if hasattr(net, "_value") else \
                (F.softmax(logits, -1) * torch.arange(logits.shape[-1], device=dev)).sum(-1).cpu().numpy()
            ctg = np.array([float(r["cost_to_go"]) for r in g]); opt = np.array([bool(r.get("is_optimal", False)) for r in g])
            m, r_, t = group_metrics(pred, ctg, opt)
            maes.append(m); rhos.append(r_); top1s.append(t)
    return dict(mae=float(np.mean(maes)), spearman=float(np.nanmean(rhos)),
                top1=float(np.nanmean(top1s)), decisions=len(groups))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variant", choices=["sparse", "reach", "inject", "mono", "dec0", "dec0f", "dec0g", "dec1"], required=True)
    p.add_argument("--data", action="append", required=True, metavar="CONFIG=PATH")
    p.add_argument("--splits", action="append", required=True)
    p.add_argument("--extra-train", action="append", default=[], metavar="CONFIG=PATH",
                   help="corpora added to TRAIN only (val stays the same decisions)")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d", type=int, default=192)
    p.add_argument("--static-rec", type=int, default=4)
    p.add_argument("--head-layers", type=int, default=2)
    p.add_argument("--dyn-steps", type=int, default=1, help="dec1: candidate-aware looped steps")
    p.add_argument("--eval-groups", type=int, default=400)
    p.add_argument("--baseline", default=None, help="current size-free value ckpt to compare against")
    p.add_argument("--warm", action="store_true", help="warm-start matching components from --baseline")
    p.add_argument("--rank-weight", type=float, default=1.0)
    p.add_argument("--distill", type=float, default=0.0,
                   help="weight of KL(student || baseline teacher logits) on every candidate")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    import random
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    geo = GeoCache()
    train = load_groups(a.data, a.splits, "train")
    if a.extra_train:
        extra = load_groups(a.extra_train, a.splits, "train")
        print(f"[split:{a.variant}] extra train decisions {len(extra)} from {a.extra_train}", flush=True)
        train = train + extra
    val = load_groups(a.data, a.splits, "val")
    rng.shuffle(val); val = val[:a.eval_groups]
    print(f"[split:{a.variant}] train decisions {len(train)}, val decisions {len(val)}, "
          f"sizes {sorted({g[0]['_n'] for g in train})}", flush=True)

    net = SplitValueNet(a.variant, a.d, static_rec=a.static_rec, head_layers=a.head_layers, dyn_steps=a.dyn_steps)
    if a.warm and a.baseline:
        print(f"[split:{a.variant}] warm-started {net.warm_start(a.baseline)} tensors from the baseline", flush=True)
    net = net.to(dev)
    teacher = None
    if a.distill > 0:
        from spr.nets import load_value
        teacher = load_value(a.baseline, dev, "sizefree"); teacher = getattr(teacher, "net", teacher).eval()
        for q in teacher.parameters(): q.requires_grad_(False)
        print(f"[split:{a.variant}] distilling from the baseline, weight {a.distill}", flush=True)
    # --steps 0: evaluate the (warm-started) net as built, no optimizer, no
    # schedule. The step-0 check that the control reproduces the baseline.
    if a.steps > 0:
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    t0 = time.time(); losses = []
    net.train(); G_cache = {}
    for step in range(a.steps):
        g = rng.choice(train)
        b = make_batch(g, a.variant, geo, rng=rng)
        G_cache.clear()                     # static encoder is trained, so no stale cache
        logits = run_model(net, b, dev, G_cache)
        ce = -(hl_gauss(b["ctg"].to(dev)) * F.log_softmax(logits, -1)).sum(-1).mean()
        loss = a.rank_weight * rank_loss(net.ctg_hat(logits), b["opt"].to(dev)) + ce
        if teacher is not None:
            with torch.no_grad():
                Bn = b["roles_full"].shape[0]
                t_logits = teacher(b["roles_full"].to(dev), b["A_all"].to(dev).expand(Bn, -1, -1),
                                   b["A_ind"].to(dev).expand(Bn, -1, -1), b["n"], b["key"].to(dev))
            loss = loss + a.distill * F.kl_div(F.log_softmax(logits, -1), F.softmax(t_logits, -1),
                                               reduction="batchmean")
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step(); losses.append(loss.item())
        if (step + 1) % 250 == 0:
            print(f"[split:{a.variant}] step {step+1}/{a.steps} loss {np.mean(losses[-250:]):.4f} "
                  f"{time.time()-t0:.0f}s", flush=True)

    res = dict(variant=a.variant, steps=a.steps, d=a.d, static_rec=a.static_rec, dyn_steps=a.dyn_steps,
               distill=a.distill, extra_train=a.extra_train,
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
                                 tokens=int(b["tok_mask"].sum(1).float().mean().item()) if "tok_mask" in b else (5 if a.variant in ("dec0", "dec0f", "dec0g") else N + 1))
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
