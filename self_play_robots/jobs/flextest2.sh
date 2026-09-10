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
torch.manual_seed(0); dev="cuda"
flex = torch.compile(flex_attention, dynamic=False)
print("the lockstep search batches ~4 queries x concurrency instances per call,")
print("so batch 32-128 is the realistic regime, not batch 1\n")
print("%-8s %6s %5s %11s %11s %9s %12s %12s" %
      ("config","K","B","dense (s)","flex (s)","speedup","dense GiB","max |diff|"))
for c in ("g32r4","g64r4","g96r4"):
    cfg=get(c)
    for k in ("RR_GRID","RR_ROBOTS","RR_WALLS","RR_ENV_DIR"): os.environ.pop(k,None)
    if not cfg.legacy: os.environ.update(cfg_env(cfg))
    n=cfg.grid; ed=os.environ.get("RR_ENV_DIR") or (SV+str(cfg.env_dir))
    eid = 0 if n>=80 else 900
    A1=torch.as_tensor(np.asarray(nn_encode.adjacency(ed,eid,n)[0],"float32")).unsqueeze(0).to(dev)
    K=A1.shape[1]; N=n*n; d,heads=192,4; dk=d//heads
    lay=LoopedLayer(d,heads,use_global=True).eval().to(dev)
    bm = create_block_mask(make_flex_mask(segment_bounds(A1,n),n,N,K,dev), B=None,H=None,Q_LEN=K,KV_LEN=K,device=dev)
    for B in (1, 8, 32):
        X=torch.randn(B,K,d,device=dev); A=A1.expand(B,K,K)
        q,k,v=(t.view(B,K,heads,dk).transpose(1,2).contiguous() for t in lay.qkv["a"](X).chunk(3,-1))
        with torch.no_grad():
            o=flex(q,k,v,block_mask=bm)
            torch.cuda.synchronize(); t0=time.time()
            for _ in range(3): o=flex(q,k,v,block_mask=bm)
            torch.cuda.synchronize(); tf=(time.time()-t0)/3
            of=o.transpose(1,2).reshape(B,K,d)
            try:
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0=time.time()
                for _ in range(3): dn=torch.nan_to_num(lay._head(X,"a",A),nan=0.0)
                torch.cuda.synchronize(); td=(time.time()-t0)/3
                gib=torch.cuda.max_memory_allocated()/2**30; diff=(dn-of).abs().max().item()
                print("%-8s %6d %5d %11.4f %11.4f %8.1fx %12.2f %12.2e" % (c,K,B,td,tf,td/tf,gib,diff))
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("%-8s %6d %5d %11s %11.4f %9s %12s %12s" % (c,K,B,"OOM",tf,"inf","OOM","(n/a)"))
PY
