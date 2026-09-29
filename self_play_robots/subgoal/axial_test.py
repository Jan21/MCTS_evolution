"""Exactness and speed of the RR_AXIAL path against the SDPA dense-masked path,
on real boards, for the policy encoder (12 rounds) and the candidate log-probs."""
import os, sys, time, json, torch
import numpy as np
REPO = "/scratch/project/open-37-42/janhula/mcts_stage5_probe"
from nn_labeler import model as M
from nn_labeler import encode as nn_encode
from scaling.configs import get
from spr.nets import load_policy
POL = "/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/selfplay/mix_b2mix_iter3/policy/lightning_logs/version_0/checkpoints/epoch=2-step=2361.ckpt"
dev = "cuda"
policy = load_policy(POL, dev, "sizefree"); net = getattr(policy, "net", policy).eval()
sync = torch.cuda.synchronize
def clock(fn, reps):
    fn(); sync(); t = time.perf_counter()
    for _ in range(reps): out = fn()
    sync(); return (time.perf_counter() - t) / reps, out
for cfg, eid in (("g24r4", 900), ("g64r4", 900), ("g96r4", 0)):
    c = get(cfg); n = c.grid
    A_all, A_ind = nn_encode.adjacency(str(REPO + "/supervised_valuenet/" + c.env_dir), eid, n)
    A_all = torch.as_tensor(np.asarray(A_all, np.float32)).unsqueeze(0).to(dev)
    A_ind = torch.as_tensor(np.asarray(A_ind, np.float32)).unsqueeze(0).to(dev)
    N = n * n + 1
    torch.manual_seed(0)
    x = (torch.rand(1, N, 7, device=dev) < 0.02).float(); x[:, -1] = 0
    with torch.no_grad():
        M._USE_AXIAL = False; t_d, h_d = clock(lambda: net._encode(x, A_all, A_ind), 5)
        M._USE_AXIAL = True;  t_a, h_a = clock(lambda: net._encode(x, A_all, A_ind), 5)
        # one masked family, one round, direct comparison
        X = net.enc(x)
        M._USE_AXIAL = False; o_d = net.layer._head(X, "a", A_all)
        M._USE_AXIAL = True;  o_a = net.layer._head(X, "a", A_all)
    print(f"{cfg}: one masked head max|diff| {float((o_d-o_a).abs().max()):.2e}  |  12 rounds: max|diff| {float((h_d-h_a).abs().max()):.2e}, "
          f"rel {float((h_d-h_a).norm()/h_d.norm()):.2e}  |  pass dense {1000*t_d:.1f} ms -> axial {1000*t_a:.1f} ms ({t_d/t_a:.1f}x)", flush=True)
print("AXIAL TEST DONE")
