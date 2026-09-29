"""Replicate the proposer (policy) network's behaviour with a cheaper student.

The teacher (the size-free policy net) runs 12 weight-tied rounds over the
board with 7 plan marks per cell and reads candidate log-probabilities off
the cell embeddings. A student splits the rounds: `s` STATIC rounds on the
walls alone (geometry channels, once per board) then the plan marks are fused
in and `k` DYNAMIC rounds run per decision. (0, 12) is the teacher's own
shape; (6, 6) is the proposal under test.

Supervision, both online from the teacher: squared error on the final cell
embeddings, KL on the candidate distribution over the enumerated candidates.
Readout heads start as the teacher's. Metrics on held-out decisions: top-1
agreement, top-5 overlap (the search keeps five), Spearman, KL.
"""
import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_labeler import encode as nn_encode
from nn_labeler.model import LoopedLayer
from spr.nets import load_policy, policy_features, flat, POLICY_CHANNELS
from spr.split_value import geometry_features, GEO_CH, load_groups, GeoCache


# ---------------------------------------------------------------------------
# candidate metadata and the readout, size-free (no train.* imports)
# ---------------------------------------------------------------------------

def meta_n(group, n):
    g0 = group[0]
    helpers = g0["helpers"]
    hpos = {tuple(h[0]): i for i, h in enumerate(helpers)}
    hcol = {h[1]: i for i, h in enumerate(helpers)}
    keys = []
    for r in group:
        hi = hpos.get(tuple(r["cand_helper"][0]))
        if hi is None:
            hi = hcol.get(r["cand_helper"][1])
        if hi is None:
            continue
        keys.append((flat(r["cand_bottleneck"], n), flat(r["cand_support"], n), hi))
    keys = list(dict.fromkeys(keys))
    if len(keys) < 2:
        return None
    vbn = sorted({k[0] for k in keys})
    sup_by_bn = {bn: sorted({k[1] for k in keys if k[0] == bn}) for bn in vbn}
    return dict(seg_start=flat(g0["seg_start"], n), seg_end=flat(g0["seg_end"], n), valid_bn=vbn,
                sup_by_bn=sup_by_bn, helper_cells=[flat(h[0], n) for h in helpers], keys=keys)


def readout(heads, h, m):
    """[K] log-probabilities of the enumerated candidates from cell embeddings
    h [N+1, d]: the teacher's three chained softmaxes, batched."""
    seg, q_bn, q_sup, q_help = heads
    dev = h.device
    g = seg(torch.cat([h[m["seg_start"]], h[m["seg_end"]], h.mean(0)]))
    vbn = m["valid_bn"]
    pairs = [(bn, sp) for bn in vbn for sp in m["sup_by_bn"][bn]]
    Hb = h[torch.tensor(vbn, device=dev)]
    bn_lp = torch.log_softmax(Hb @ q_bn(g), 0)
    Qs = q_sup(torch.cat([g.unsqueeze(0).expand(len(vbn), -1), Hb], -1))
    sup_lp = {bn: torch.log_softmax(h[torch.tensor(m["sup_by_bn"][bn], device=dev)] @ Qs[i], 0)
              for i, bn in enumerate(vbn)}
    pb = torch.tensor([p[0] for p in pairs], device=dev); ps = torch.tensor([p[1] for p in pairs], device=dev)
    Qh = q_help(torch.cat([g.unsqueeze(0).expand(len(pairs), -1), h[pb], h[ps]], -1))
    Hh = h[torch.tensor(m["helper_cells"], device=dev)]
    hl = torch.log_softmax(Qh @ Hh.T, -1)
    pi = {p: i for i, p in enumerate(pairs)}; bi = {bn: i for i, bn in enumerate(vbn)}
    si = {bn: {sp: j for j, sp in enumerate(m["sup_by_bn"][bn])} for bn in vbn}
    return torch.stack([bn_lp[bi[b]] + sup_lp[b][si[b][sp]] + hl[pi[(b, sp)], hi] for (b, sp, hi) in m["keys"]])


# ---------------------------------------------------------------------------
# the student
# ---------------------------------------------------------------------------

class SplitPolicy(nn.Module):
    def __init__(self, s, k, d=192, heads=4, tie=True):
        super().__init__()
        self.s, self.k, self.tie = s, k, tie
        self.geo = nn.Linear(GEO_CH, d)                    # walls in (fresh)
        self.enc = nn.Linear(POLICY_CHANNELS, d)           # plan marks in (teacher's)
        self.layer = LoopedLayer(d, heads, use_global=True)
        self.layer_dyn = self.layer if tie else LoopedLayer(d, heads, use_global=True)
        self.seg = _mlp(3 * d, d); self.q_bn = _mlp(d, d); self.q_sup = _mlp(2 * d, d); self.q_help = _mlp(3 * d, d)

    def heads(self):
        return (self.seg, self.q_bn, self.q_sup, self.q_help)

    def static(self, geo, A_all, A_ind):
        if self.s == 0:
            return None
        S = self.geo(geo)
        for _ in range(self.s):
            S = self.layer(S, A_all, A_ind)
        return S

    def dynamic(self, S, x, A_all, A_ind):
        H = self.enc(x) if S is None else S + self.enc(x)
        for _ in range(self.k):
            H = self.layer_dyn(H, A_all, A_ind)
        return H

    def warm_start(self, teacher):
        t = teacher.state_dict(); own = self.state_dict(); n = 0
        for k, v in own.items():
            src = k
            if k.startswith("layer_dyn."):
                src = "layer." + k[len("layer_dyn."):]
            if src in t and t[src].shape == v.shape:
                own[k] = t[src].clone(); n += 1
        self.load_state_dict(own); return n


def _mlp(din, d):
    return nn.Sequential(nn.Linear(din, d), nn.GELU(), nn.Linear(d, d))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def make_item(group, geo: GeoCache):
    n = group[0]["_n"]; env_dir = group[0]["_env_dir"]
    group = [r for r in group if r["_env_dir"] == env_dir]
    m = meta_n(group, n)
    if m is None:
        return None
    A_all, A_ind, gf = geo.get(env_dir, group[0]["env_id"], n)
    x = torch.from_numpy(policy_features(group[0], n)).unsqueeze(0)
    return dict(n=n, env=(env_dir, int(group[0]["env_id"])), x=x, geo=torch.from_numpy(gf).unsqueeze(0),
                A_all=torch.from_numpy(A_all).unsqueeze(0), A_ind=torch.from_numpy(A_ind).unsqueeze(0), meta=m)


def metrics(lp_s, lp_t):
    K = lp_s.numel()
    ps, pt = torch.softmax(lp_s, 0), torch.softmax(lp_t, 0)
    kl = float((pt * (torch.log(pt + 1e-12) - torch.log(ps + 1e-12))).sum())
    top1 = float(int(lp_s.argmax()) == int(lp_t.argmax()))
    kk = min(5, K)
    ts = set(torch.topk(lp_s, kk).indices.tolist()); tt = set(torch.topk(lp_t, kk).indices.tolist())
    ov = len(ts & tt) / kk
    a, b = lp_s.cpu().numpy(), lp_t.cpu().numpy()
    if K > 2 and a.std() > 0 and b.std() > 0:
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        rho = float(np.corrcoef(ra, rb)[0, 1])
    else:
        rho = float("nan")
    return kl, top1, ov, rho


@torch.no_grad()
def evaluate(student, teacher, items, dev):
    student.eval(); S_cache = {}
    out = []
    for it in items:
        A_all, A_ind = it["A_all"].to(dev), it["A_ind"].to(dev)
        hT = teacher._encode(it["x"].to(dev), A_all, A_ind)[0]
        lpT = readout((teacher.seg, teacher.q_bn, teacher.q_sup, teacher.q_help), hT, it["meta"])
        S = S_cache.get(it["env"])
        if S is None and student.s > 0:
            S = S_cache[it["env"]] = student.static(it["geo"].to(dev), A_all, A_ind)
            if len(S_cache) > 16: S_cache.pop(next(iter(S_cache)))
        hS = student.dynamic(S, it["x"].to(dev), A_all, A_ind)[0]
        lpS = readout(student.heads(), hS, it["meta"])
        kl, t1, ov, rho = metrics(lpS, lpT)
        out.append((kl, t1, ov, rho, float(F.mse_loss(hS, hT))))
    a = np.array(out, float)
    return dict(kl=float(a[:, 0].mean()), top1=float(a[:, 1].mean()), top5=float(a[:, 2].mean()),
                spearman=float(np.nanmean(a[:, 3])), emb_mse=float(a[:, 4].mean()), decisions=len(out))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, help="s+k static and dynamic rounds, e.g. 6+6")
    p.add_argument("--data", action="append", required=True); p.add_argument("--splits", action="append", required=True)
    p.add_argument("--extra-train", action="append", default=[])
    p.add_argument("--teacher", required=True)
    p.add_argument("--steps", type=int, default=25000); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--w-emb", type=float, default=1.0); p.add_argument("--w-kl", type=float, default=1.0)
    p.add_argument("--untied", action="store_true", help="separate weights for the dynamic rounds")
    p.add_argument("--eval-groups", type=int, default=400); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    s, k = (int(v) for v in a.variant.replace(",", "+").split("+"))
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    geo = GeoCache()
    train = load_groups(a.data, a.splits, "train")
    if a.extra_train:
        train += load_groups(a.extra_train, a.splits, "train")
    val = load_groups(a.data, a.splits, "val"); rng.shuffle(val); val = val[:a.eval_groups]
    tr_items = [it for it in (make_item(g, geo) for g in train) if it is not None]
    va_items = [it for it in (make_item(g, geo) for g in val) if it is not None]
    print(f"[distill {s}+{k}] train decisions {len(tr_items)}, val {len(va_items)}, sizes {sorted({it['n'] for it in tr_items})}", flush=True)

    teacher = load_policy(a.teacher, dev, "sizefree"); teacher = getattr(teacher, "net", teacher).eval()
    for q in teacher.parameters(): q.requires_grad_(False)
    student = SplitPolicy(s, k, tie=not a.untied)
    print(f"[distill {s}+{k}] warm-started {student.warm_start(teacher)} tensors from the teacher", flush=True)
    student = student.to(dev)
    if a.steps > 0:
        opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    t0 = time.time(); losses = []
    student.train()
    for step in range(a.steps):
        it = rng.choice(tr_items)
        A_all, A_ind = it["A_all"].to(dev), it["A_ind"].to(dev)
        with torch.no_grad():
            hT = teacher._encode(it["x"].to(dev), A_all, A_ind)[0]
            lpT = readout((teacher.seg, teacher.q_bn, teacher.q_sup, teacher.q_help), hT, it["meta"])
        S = student.static(it["geo"].to(dev), A_all, A_ind)
        hS = student.dynamic(S, it["x"].to(dev), A_all, A_ind)[0]
        lpS = readout(student.heads(), hS, it["meta"])
        kl = F.kl_div(torch.log_softmax(lpS, 0), torch.softmax(lpT, 0), reduction="sum")
        loss = a.w_emb * F.mse_loss(hS, hT) + a.w_kl * kl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step(); sched.step(); losses.append(loss.item())
        if (step + 1) % 250 == 0:
            print(f"[distill {s}+{k}] step {step+1}/{a.steps} loss {np.mean(losses[-250:]):.4f} {time.time()-t0:.0f}s", flush=True)

    res = dict(variant=a.variant, s=s, k=k, tied=not a.untied, steps=a.steps, lr=a.lr, w_emb=a.w_emb, w_kl=a.w_kl,
               train_decisions=len(tr_items))
    res["val"] = evaluate(student, teacher, va_items, dev)
    v = res["val"]
    print(f"[distill {s}+{k}] VAL vs teacher: top-1 agree {100*v['top1']:.1f}%  top-5 overlap {v['top5']:.3f}  "
          f"spearman {v['spearman']:.3f}  KL {v['kl']:.4f}  emb mse {v['emb_mse']:.4f}  ({v['decisions']} decisions)", flush=True)
    # per-decision cost at 96x96: the dynamic rounds only (static is once per board)
    if dev == "cuda":
        with torch.no_grad():
            N = 96 * 96 + 1
            idx = torch.arange(N - 1, device=dev); y = idx // 96
            A = torch.zeros(N, N, device=dev); A[idx, idx] = 1
            for d_ in (1, -1):
                nb = idx + d_; ok = (nb >= 0) & (nb < N - 1) & ((nb // 96) == y); A[idx[ok], nb[ok]] = 1
                nb = idx + 96 * d_; ok = (nb >= 0) & (nb < N - 1); A[idx[ok], nb[ok]] = 1
            A = A.unsqueeze(0); x = torch.zeros(1, N, POLICY_CHANNELS, device=dev); g = torch.zeros(1, N, GEO_CH, device=dev)
            S = student.static(g, A, A); torch.cuda.synchronize()
            student.dynamic(S, x, A, A); torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(5): student.dynamic(S, x, A, A)
            torch.cuda.synchronize(); res["dyn_ms_96"] = 1000 * (time.perf_counter() - t) / 5
            print(f"[distill {s}+{k}] per-decision cost at 96x96: {res['dyn_ms_96']:.0f} ms (dynamic rounds; static once per board)", flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n"); torch.save(student.state_dict(), out.with_suffix(".pt"))
    print(f"DISTILL POLICY DONE {out}")


if __name__ == "__main__":
    main()
