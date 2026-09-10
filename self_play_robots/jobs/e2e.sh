#!/bin/bash
set -uo pipefail
ml purge 2>/dev/null; ml Python/3.11.5-GCCcore-13.2.0 2>/dev/null
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1
python - <<'PY'
# END-TO-END: the real checkpoint, real inputs, whole CtgNet forward, both paths.
import os, time, numpy as np, torch
import nn_labeler.model as M
from subgoal.ctgnet import CtgNet, CtgDataset, collate
from subgoal.table import ENV_DIR, SIZE
CK="/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/subgoal/stage4_c_r3/ctg-epoch=2-val_mae_all=1.5468.ckpt"
dev="cuda"
net=CtgNet.load_from_checkpoint(CK, map_location=dev).eval().to(dev)
store=np.load("results/subgoal/stage3/val_b1800-1849.npz", allow_pickle=False)
ds=CtgDataset(store, ENV_DIR, SIZE)
dl=torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate, num_workers=2)
b=next(iter(dl)); n=SIZE
x,Aa,Ai=b["x"].to(dev),b["A_all"].to(dev),b["A_ind"].to(dev)
with torch.no_grad():
    M._USE_SDPA=False; torch.cuda.synchronize(); t0=time.time()
    ref=net.ctg_hat(net(x,Aa,Ai,n)).float(); torch.cuda.synchronize(); td=time.time()-t0
    M._USE_SDPA=True;  torch.cuda.synchronize(); t0=time.time()
    new=net.ctg_hat(net(x,Aa,Ai,n)).float(); torch.cuda.synchronize(); ts=time.time()-t0
print("whole-model, real checkpoint, batch 32 at 16x16:")
print("  max |diff| in predicted cost-to-go:", "%.2e" % (ref-new).abs().max().item())
print("  dense %.4f s   sdpa %.4f s   (%.1fx)" % (td, ts, td/ts))
print("  argmin agreement per (state,robot) group: %.1f%%" %
      (100*(ref.argmin(-1)==new.argmin(-1)).float().mean().item()))
PY
