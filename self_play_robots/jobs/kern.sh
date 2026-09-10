#!/bin/bash
set -uo pipefail
ml purge 2>/dev/null; ml Python/3.11.5-GCCcore-13.2.0 2>/dev/null
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1
python - <<'PY'
import torch, math, time
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    print("flex_attention: AVAILABLE  <- arbitrary mask fn, block-sparse, no K^2 materialization")
    HAVE_FLEX = True
except Exception as e:
    print("flex_attention: not available:", repr(e)[:70]); HAVE_FLEX = False
import torch.nn.functional as F
print("scaled_dot_product_attention:", hasattr(F, "scaled_dot_product_attention"))
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    print("sdpa backends selectable: yes")
except Exception as e:
    print("sdpa backends selectable: no", repr(e)[:50])

# what the naive path costs vs what a fused kernel costs, same shapes, 96x96
K, h, dk = 96*96+1, 4, 48
dev = "cuda" if torch.cuda.is_available() else "cpu"
q = torch.randn(1, h, K, dk, device=dev)
k = torch.randn(1, h, K, dk, device=dev)
v = torch.randn(1, h, K, dk, device=dev)
print(f"\nK={K}, heads={h}, dk={dk}, device={dev}")
print(f"  a dense [1,{h},{K},{K}] score tensor alone is "
      f"{1*h*K*K*4/2**30:.2f} GiB in fp32")
if dev == "cuda":
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(5): o = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    print("  fused SDPA, no mask, 5 calls: %.3f s, peak %.2f GiB"
          % (time.time()-t0, torch.cuda.max_memory_allocated()/2**30))
PY
