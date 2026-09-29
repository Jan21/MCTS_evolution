"""Stage-by-stage timing of one proposer (policy) network call on real decisions.

1. Run the benchmark search for a few expansions on one ladder instance and
   capture the candidate groups handed to the proposer.
2. Time every stage of a call with card synchronisation at each boundary:
   meta, featurisation, transfer, input projection, each of the 12 rounds
   (split into the three attention families, projection+norm, feed-forward+norm),
   the batched readout, transfer back.
3. Kernel-level profile of one encode pass.
4. Batch sweep of the encode pass (1, 2, 4, 8 boards) at this size and at 64x64.
"""
import os, sys, time, json, statistics as st
import torch
import torch.nn.functional as F

REPO = "/scratch/project/open-37-42/janhula/mcts_stage5_probe"
NETS = "/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/selfplay/mix_b2mix_iter3"
POL = NETS + "/policy/lightning_logs/version_0/checkpoints/epoch=2-step=2361.ckpt"
VALS = REPO + "/self_play_robots/results/subgoal/split/dec0-sw-r12-lr1e-4-x8-distill.pt"
INST = sys.argv[1]; EXP = int(sys.argv[2]) if len(sys.argv) > 2 else 8
dev = "cuda"

# ---- 1. capture real decisions -------------------------------------------------
import eval.end2end as e2e
groups = []
_orig = e2e._policy_logp
def _cap(policy, group, d, byref=False):
    groups.append(group)
    return _orig(policy, group, d, byref)
e2e._policy_logp = _cap
import spr.bench as bench
sys.argv = ["spr.bench", "--instances", INST, "--policy", POL, "--value", VALS, "--arch", "split",
            "--search", "arena_astar", "--vocab", "b2", "--anytime", "--expansions", str(EXP), "--k", "5",
            "--boards", "lean", "--device", dev, "--out", "/tmp/prof_policy_capture.json"]
try:
    bench.main()
except SystemExit:
    pass
e2e._policy_logp = _orig
sizes = sorted(len(g) for g in groups)
print(f"captured {len(groups)} decisions; candidates per decision min/median/max = {sizes[0]}/{sizes[len(sizes)//2]}/{sizes[-1]}", flush=True)
g0 = max(groups, key=len)                                  # the most candidate-rich decision

# ---- 2. stage timing -------------------------------------------------------------
from spr.nets import load_policy
from train.policy_common import _meta, _ix
from train.policy_tf import _adj, _x257 as _pol_x
policy = load_policy(POL, dev, "sizefree"); net = getattr(policy, "net", policy); net.eval()
sync = torch.cuda.synchronize
def clock(fn, reps=1):
    sync(); t = time.perf_counter()
    for _ in range(reps): out = fn()
    sync(); return (time.perf_counter() - t) / reps, out

n = int(round((len(_pol_x(g0[0])) - 1) ** 0.5))
print(f"decision: {len(g0)} candidates, board {n}x{n}, tokens {n*n+1}", flush=True)
A_all_c, A_ind_c = _adj(g0[0]["env_id"])
t_adj_h2d, (A_all, A_ind) = clock(lambda: (A_all_c.to(dev), A_ind_c.to(dev)))
A_all1, A_ind1 = A_all.unsqueeze(0), A_ind.unsqueeze(0)

with torch.no_grad():
    # warm-up
    x = _pol_x(g0[0]).unsqueeze(0).to(dev); h = net._encode(x, A_all1, A_ind1)[0]; sync()
    R = 5
    t_meta, m = clock(lambda: _meta(g0, byref=True), R)
    t_feat, xc = clock(lambda: _pol_x(g0[0]).unsqueeze(0), R)
    t_h2d, x = clock(lambda: xc.to(dev), R)
    t_in, X0 = clock(lambda: net.enc(x), R)
    layer = net.layer
    rounds = []
    X = X0
    for r in range(12):
        parts = {}
        amap = {"g": None, "a": A_all1, "i": A_ind1}
        outs = {}
        for f in layer.fams:
            parts[f], outs[f] = clock(lambda f=f: layer._head(X, f, amap[f]), R)
        a = torch.cat([outs[f] for f in layer.fams], -1)
        parts["proj+norm"], X1 = clock(lambda: layer.norm1(X + layer.proj(a)), R)
        parts["ffn+norm"], X = clock(lambda: layer.norm2(X1 + layer.mlp(X1)), R)
        rounds.append(parts)
    hcell = X[0]
    t_mask, _ = clock(lambda: (A_all1 != 0)[:, None], R)
    # batched readout, exactly as eval/end2end._policy_logp
    def readout():
        g = net.seg(torch.cat([hcell[m["seg_start"]], hcell[m["seg_end"]], hcell.mean(0)]))
        vbn = m["valid_bn"]; pairs = [(bn, sp) for bn in vbn for sp in m["sup_by_bn"][bn]]
        keys = [k for k in m["ctg_map"]]
        bn_t = torch.tensor(vbn, device=dev); Hb = hcell[bn_t]
        bn_lp = torch.log_softmax(Hb @ net.q_bn(g), 0)
        gB = g.unsqueeze(0).expand(len(vbn), -1); Qs = net.q_sup(torch.cat([gB, Hb], -1))
        sup_lp = {bn: torch.log_softmax(hcell[torch.tensor(m["sup_by_bn"][bn], device=dev)] @ Qs[i], 0) for i, bn in enumerate(vbn)}
        pb = torch.tensor([p[0] for p in pairs], device=dev); ps = torch.tensor([p[1] for p in pairs], device=dev)
        gP = g.unsqueeze(0).expand(len(pairs), -1); Qh = net.q_help(torch.cat([gP, hcell[pb], hcell[ps]], -1))
        Hh = hcell[torch.tensor(m["helper_cells"], device=dev)]; hl = torch.log_softmax(Qh @ Hh.T, -1)
        pi = {p: i for i, p in enumerate(pairs)}; bi = {bn: i for i, bn in enumerate(vbn)}
        si = {bn: {sp: j for j, sp in enumerate(m["sup_by_bn"][bn])} for bn in vbn}
        return torch.stack([bn_lp[bi[b]] + sup_lp[b][si[b][sp]] + hl[pi[(b, sp)], hi] for (b, sp, hi) in keys])
    t_read, tot = clock(readout, R)
    t_d2h, _ = clock(lambda: tot.tolist(), R)
    t_full, _ = clock(lambda: net._encode(x, A_all1, A_ind1), R)

ms = lambda s: f"{1000*s:7.1f} ms"
enc_total = sum(sum(p.values()) for p in rounds)
print("\n=== one proposer call, stage by stage (mean of 5, card synchronised)")
print(f"  adjacency to card (once per board, cached) {ms(t_adj_h2d)}")
print(f"  candidate meta (CPU, {len(g0)} candidates)      {ms(t_meta)}")
print(f"  featurise board (CPU)                      {ms(t_feat)}")
print(f"  transfer marks to card                     {ms(t_h2d)}")
print(f"  input projection                           {ms(t_in)}")
print(f"  12 rounds, sum of stages                   {ms(enc_total)}   (one unsplit encode pass: {ms(t_full)})")
for f in layer.fams:
    print(f"      attention '{f}' x12                     {ms(sum(p[f] for p in rounds))}")
print(f"      projection + norm x12                  {ms(sum(p['proj+norm'] for p in rounds))}")
print(f"      feed-forward + norm x12                {ms(sum(p['ffn+norm'] for p in rounds))}")
print(f"      (mask build A!=0, one family, once)    {ms(t_mask)}  -> x24 per pass = {ms(24*t_mask)}")
print(f"  readout, all {len(g0)} candidates             {ms(t_read)}")
print(f"  transfer scores back                       {ms(t_d2h)}")
print(f"  TOTAL                                      {ms(t_meta+t_feat+t_h2d+t_in+enc_total+t_read+t_d2h)}")
print("  per round:", " ".join(f"{1000*sum(p.values()):.0f}" for p in rounds), "ms")

# ---- 3. kernel-level profile of one encode pass ------------------------------------
from torch.profiler import profile, ProfilerActivity
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        net._encode(x, A_all1, A_ind1); sync()
ka = prof.key_averages()
def dt(k):
    return getattr(k, "device_time_total", getattr(k, "cuda_time_total", 0)) or 0
tot_cuda = sum(dt(k) for k in ka)
print("\n=== kernel-level, one encode pass (card time by op, top 12)")
for k in sorted(ka, key=lambda k: -dt(k))[:12]:
    if dt(k) <= 0: continue
    print(f"  {k.key[:60]:60s} {dt(k)/1000:8.1f} ms  {100*dt(k)/max(1,tot_cuda):5.1f}%  calls {k.count}")

# ---- 4. batch sweep -----------------------------------------------------------------
print("\n=== encode pass vs batch size (ms per call, ms per board)")
with torch.no_grad():
    for B in (1, 2, 4, 8):
        xb = x.expand(B, -1, -1).contiguous()
        t, _ = clock(lambda: net._encode(xb, A_all1.expand(B, -1, -1), A_ind1.expand(B, -1, -1)), 3)
        print(f"  {n}x{n}  batch {B}: {1000*t:7.1f} ms   {1000*t/B:6.1f} ms/board")
    # 64x64 with a synthetic input (cost does not depend on the marks)
    from scaling.configs import get
    try:
        os.environ["RR_GRID"] = "64"; os.environ["RR_WALLS"] = str(get("g64r4").walls) if hasattr(get("g64r4"), "walls") else os.environ.get("RR_WALLS", "")
    except Exception:
        pass
    N64 = 64 * 64 + 1
    idx = torch.arange(N64 - 1, device=dev); xg, yg = idx % 64, idx // 64
    A64 = torch.zeros(N64, N64, device=dev)
    A64[idx, idx] = 1
    # row/column neighbours as a stand-in for the slide graph (same sparsity order)
    for d_ in (1, -1):
        nb = idx + d_; ok = (nb >= 0) & (nb < N64 - 1) & ((nb // 64) == yg); A64[idx[ok], nb[ok]] = 1
        nb = idx + 64 * d_; ok = (nb >= 0) & (nb < N64 - 1); A64[idx[ok], nb[ok]] = 1
    A64 = A64.unsqueeze(0)
    for B in (1, 2, 4, 8):
        xb = torch.zeros(B, N64, x.shape[-1], device=dev)
        t, _ = clock(lambda: net._encode(xb, A64.expand(B, -1, -1), A64.expand(B, -1, -1)), 3)
        print(f"  64x64  batch {B}: {1000*t:7.1f} ms   {1000*t/B:6.1f} ms/board   (synthetic sparse graph)")
print("PROFILE POLICY DONE")
