#!/bin/bash
set -uo pipefail
ml purge 2>/dev/null; ml Python/3.11.5-GCCcore-13.2.0 2>/dev/null
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1
python - <<'PY'
import os, time, numpy as np, torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from scaling.configs import get, env as cfg_env
from nn_labeler.model import LoopedLayer
from nn_labeler import encode as nn_encode
SV="/scratch/project/open-37-42/janhula/mcts_stage5_probe/supervised_valuenet/"
torch.manual_seed(0); dev="cuda"
print("SDPA with a BOOLEAN mask [B,1,K,K]: 1 byte per entry, so 81 MB per instance")
print("at 96x96 instead of 1.27 GiB, and the memory-efficient kernel never forms the")
print("fp32 score matrix. No compilation, no fallback surprises.\n")
print("%-8s %6s %5s %11s %11s %9s %11s %11s %12s" %
      ("config","K","B","dense (s)","sdpa (s)","speedup","dense GiB","sdpa GiB","max |diff|"))
for c in ("g32r4","g64r4","g96r4"):
    cfg=get(c)
    for k in ("RR_GRID","RR_ROBOTS","RR_WALLS","RR_ENV_DIR"): os.environ.pop(k,None)
    if not cfg.legacy: os.environ.update(cfg_env(cfg))
    n=cfg.grid; ed=os.environ.get("RR_ENV_DIR") or (SV+str(cfg.env_dir))
    A1=torch.as_tensor(np.asarray(nn_encode.adjacency(ed,0 if n>=80 else 900,n)[0],"float32")).unsqueeze(0).to(dev)
    K=A1.shape[1]; d,heads=192,4; dk=d//heads
    lay=LoopedLayer(d,heads,use_global=True).eval().to(dev)
    M1=A1.bool().unsqueeze(1)                    # [1,1,K,K], broadcasts over heads
    for B in (1, 8, 32):
        X=torch.randn(B,K,d,device=dev); A=A1.expand(B,K,K); M=M1.expand(B,1,K,K)
        q,k,v=(t.view(B,K,heads,dk).transpose(1,2) for t in lay.qkv["a"](X).chunk(3,-1))
        with torch.no_grad():
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                o=F.scaled_dot_product_attention(q,k,v,attn_mask=M)
                torch.cuda.synchronize(); t0=time.time()
                for _ in range(3): o=F.scaled_dot_product_attention(q,k,v,attn_mask=M)
                torch.cuda.synchronize(); ts=(time.time()-t0)/3
            gs=torch.cuda.max_memory_allocated()/2**30
            os_=torch.nan_to_num(o,nan=0.0).transpose(1,2).reshape(B,K,d)
            try:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize(); t0=time.time()
                for _ in range(3): dn=torch.nan_to_num(lay._head(X,"a",A),nan=0.0)
                torch.cuda.synchronize(); td=(time.time()-t0)/3
                gd=torch.cuda.max_memory_allocated()/2**30; diff=(dn-os_).abs().max().item()
                print("%-8s %6d %5d %11.4f %11.4f %8.1fx %11.2f %11.2f %12.2e"%(c,K,B,td,ts,td/ts,gd,gs,diff))
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("%-8s %6d %5d %11s %11.4f %9s %11s %11.2f %12s"%(c,K,B,"OOM",ts,"inf","OOM",gs,"(n/a)"))
PY
