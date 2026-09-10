#!/bin/bash
set -uo pipefail
ml purge 2>/dev/null; ml Python/3.11.5-GCCcore-13.2.0 2>/dev/null
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1
python - <<'PY'
import os, time, numpy as np, torch
from scaling.configs import get, env as cfg_env
from nn_labeler.model import LoopedLayer, segment_bounds, make_flex_mask
from nn_labeler import encode as nn_encode
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
SV="/scratch/project/open-37-42/janhula/mcts_stage5_probe/supervised_valuenet/"
torch.manual_seed(0)
dev="cuda"
flex = torch.compile(flex_attention, dynamic=False)
print("%-8s %6s %11s %11s %9s %10s %12s" %
      ("config","K","dense (s)","flex (s)","speedup","density","max |diff|"))
for c in ("g16r4","g32r4","g48r4","g64r4","g96r4"):
    cfg=get(c)
    for k in ("RR_GRID","RR_ROBOTS","RR_WALLS","RR_ENV_DIR"): os.environ.pop(k,None)
    if not cfg.legacy: os.environ.update(cfg_env(cfg))
    n=cfg.grid
    ed=os.environ.get("RR_ENV_DIR") or (SV+str(cfg.env_dir))
    eid = 900 if n>16 else 2400
    if n>=80: eid = 0
    A_all,_=nn_encode.adjacency(ed, eid, n)
    A=torch.as_tensor(np.asarray(A_all,"float32")).unsqueeze(0).to(dev)
    B,K=A.shape[0],A.shape[1]; N=n*n; d,heads=192,4; dk=d//heads
    lay=LoopedLayer(d,heads,use_global=True).eval().to(dev)
    X=torch.randn(B,K,d,device=dev)
    bounds = segment_bounds(A, n)
    mm = make_flex_mask(bounds, n, N, K, dev)
    try:
        bm = create_block_mask(mm, B=None, H=None, Q_LEN=K, KV_LEN=K, device=dev)
    except Exception as e:
        print("%-8s create_block_mask failed: %s" % (c, repr(e)[:60])); continue
    q,k,v = lay.qkv["a"](X).chunk(3,-1)
    sh = lambda t: t.view(B,K,heads,dk).transpose(1,2).contiguous()
    q,k,v = sh(q),sh(k),sh(v)
    with torch.no_grad():
        try:
            o = flex(q,k,v,block_mask=bm)          # warm up / compile
        except Exception as e:
            print("%-8s flex failed: %s" % (c, repr(e)[:60])); continue
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(5): o = flex(q,k,v,block_mask=bm)
        torch.cuda.synchronize(); tf=(time.time()-t0)/5
        try:
            torch.cuda.synchronize(); t0=time.time()
            for _ in range(5): dn = torch.nan_to_num(lay._head(X,"a",A),nan=0.0)
            torch.cuda.synchronize(); td=(time.time()-t0)/5
            of = o.transpose(1,2).reshape(B,K,d)
            diff = (dn-of).abs().max().item()
        except torch.OutOfMemoryError:
            td, diff = float("nan"), float("nan")
            torch.cuda.empty_cache()
    dens = float(A.sum().item())/(K*K)
    print("%-8s %6d %11.4f %11.4f %8s %9.2f%% %12.2e" %
          (c,K,td,tf,("%.1fx"%(td/tf)) if td==td else "OOM", 100*dens, diff))
PY
