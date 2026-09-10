#!/bin/bash
set -uo pipefail
ml purge; ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
python - <<'PY'
import os, time, json, numpy as np, torch
from scaling.configs import get, env as cfg_env
cfg=get("g64r4")
for k in ("RR_GRID","RR_ROBOTS","RR_WALLS","RR_ENV_DIR"): os.environ.pop(k,None)
os.environ.update(cfg_env(cfg))
from subgoal.ctgnet import CtgNet, state_features
from nn_labeler import encode as nn_encode
CK="/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/subgoal/stage4_c_r3/ctg-epoch=2-val_mae_all=1.5468.ckpt"
dev="cuda"; net=CtgNet.load_from_checkpoint(CK,map_location=dev).eval().to(dev)
n=64; N=n*n; ed=os.environ["RR_ENV_DIR"]
A_all,A_ind=nn_encode.adjacency(ed,900,n)
Aa=torch.as_tensor(np.asarray(A_all,"float32")).unsqueeze(0).to(dev)
Ai=torch.as_tensor(np.asarray(A_ind,"float32")).unsqueeze(0).to(dev)
rng=np.random.default_rng(0)

# ---------- 1. embedding stability: encode ONE board under 30 robot placements
print("=== EMBEDDING STABILITY: one 64x64 board, 30 random robot placements ===")
embs=[]; dists=[]
with torch.no_grad():
    for _ in range(30):
        cells=rng.choice(N,5,replace=False); pos=[(int(c%n),int(c//n)) for c in cells[:4]]; goal=(int(cells[4]%n),int(cells[4]//n))
        x=torch.as_tensor(state_features(pos,0,0,goal,n)).unsqueeze(0).to(dev)
        E=net._encode(x,Aa,Ai,n)[0,:N].float().cpu().numpy()          # [N,d]
        embs.append(E)
        yy,xx=np.divmod(np.arange(N),n); act=np.array(pos+[goal])
        dists.append(np.min(np.abs(xx[:,None]-act[None,:,0])+np.abs(yy[:,None]-act[None,:,1]),1))
E=np.stack(embs); D=np.stack(dists)                                      # [30,N,d], [30,N]
mean=E.mean(0,keepdims=True); var_tot=E.var(0).mean(-1)+1e-9              # per cell
static_frac=1-(E-mean).var(0).mean(-1)/(E.var(0).mean(-1)+1e-9)          # trivially 0; use across-placement vs across-cell
# better: how much of each cell's embedding is explained by its placement-independent mean,
# relative to the spread of embeddings across the whole board
across_cells=E.reshape(-1,E.shape[-1]).var(0).mean()
per_cell_change=(E-mean).reshape(-1,E.shape[-1]).var(0).mean()
print(f"  variance across placements / variance across cells: {per_cell_change/across_cells:.3f}")
print("  (small = embeddings are mostly a fixed function of the board, not of robot positions)")
print()
print("  per-cell change vs distance to nearest robot/goal:")
chg=np.sqrt(((E-mean)**2).mean(-1))                                       # [30,N]
for lo,hi in ((0,1),(1,3),(3,6),(6,12),(12,24),(24,200)):
    m=(D>=lo)&(D<hi)
    print("    distance %3d-%-3d cells: mean change %.4f   (n=%d)" % (lo,hi,chg[m].mean(),m.sum()))
print("  overall embedding scale (rms over cells): %.4f" % np.sqrt((E**2).mean()))

# ---------- 2. half precision: same guide, same argmin?
print()
print("=== HALF PRECISION: does bf16 change what the search would pick? ===")
from subgoal.ctgnet import CtgDataset, collate
from subgoal.table import ENV_DIR, SIZE
for k in ("RR_GRID","RR_ROBOTS","RR_WALLS","RR_ENV_DIR"): os.environ.pop(k,None)
store=np.load("results/subgoal/stage3/val_b1800-1849.npz",allow_pickle=False)
ds=CtgDataset(store,ENV_DIR,SIZE)
dl=torch.utils.data.DataLoader(ds,batch_size=32,shuffle=False,collate_fn=collate,num_workers=2)
agree=tot=0; t32=t16=0.0
with torch.no_grad():
    for i,b in enumerate(dl):
        if i>=8: break
        x,A1,A2=b["x"].to(dev),b["A_all"].to(dev),b["A_ind"].to(dev)
        torch.cuda.synchronize(); t0=time.time(); r32=net.ctg_hat(net(x,A1,A2,SIZE)).float(); torch.cuda.synchronize(); t32+=time.time()-t0
        torch.cuda.synchronize(); t0=time.time()
        with torch.autocast("cuda",dtype=torch.bfloat16): r16=net.ctg_hat(net(x,A1,A2,SIZE)).float()
        torch.cuda.synchronize(); t16+=time.time()-t0
        agree+=(r32.argmin(-1)==r16.argmin(-1)).sum().item(); tot+=r32.shape[0]
print(f"  argmin agreement fp32 vs bf16: {100*agree/tot:.1f}%  ({tot} groups)")
print(f"  time fp32 {t32:.3f}s  bf16 {t16:.3f}s  ({t32/max(t16,1e-9):.2f}x)")
PY
