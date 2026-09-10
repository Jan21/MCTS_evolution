#!/bin/bash
set -uo pipefail
ml purge; ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
python - <<'PY'
import os, time, json, torch
from scaling.configs import get, env as cfg_env
for C in ("g24r4","g64r4","g96r4"):
    cfg=get(C)
    for k in ("RR_GRID","RR_ROBOTS","RR_WALLS","RR_ENV_DIR"): os.environ.pop(k,None)
    os.environ.update(cfg_env(cfg))
    import importlib, sys
    for m in [k for k in list(sys.modules) if k.split(".")[0] in ("GridEnv","skeleton","spr","nn","nn_labeler","move_planner","simulate","eval")]:
        sys.modules.pop(m,None)
    from nn_labeler import leanboard
    from spr.nets import load_policy, load_value
    from skeleton import heuristics
    from GridEnv import State, Robot_at
    from move_planner.state import COLOR_ORDER
    NETS="/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/selfplay/mix_b2mix_iter3"
    pol=load_policy(NETS+"/policy/lightning_logs/version_0/checkpoints/epoch=2-step=2361.ckpt","cuda","sizefree")
    val=load_value(NETS+"/value/lightning_logs/version_0/checkpoints/epoch=0-step=1573.ckpt","cuda","sizefree")
    n=cfg.grid
    inst=json.loads(open(f"../supervised_valuenet/scaling/data/{C}/"+("ladder.jsonl" if n>32 else "bench.solved.jsonl")).readline())
    t0=time.time(); env,_=leanboard.from_env(inst["env_id"]); tl=time.time()-t0
    pos=[tuple(p) for p in inst["positions"]]; ti=inst["target_idx"]
    st=State(target=tuple(inst["target"]),target_robot=Robot_at(position=pos[ti],color=COLOR_ORDER[ti]),
             helpers=[Robot_at(position=pos[j],color=COLOR_ORDER[j]) for j in range(4) if j!=ti])
    # CPU: one round of candidate proposal
    t0=time.time()
    try:
        cands=heuristics.propose(env, st.target, st.target_robot, st.helpers, None, b1=True); nc=len(cands)
    except Exception as e:
        cands=[]; nc=-1
    tp=time.time()-t0
    # GPU: what one forward of each net costs at this board size, small batch
    net_p=getattr(pol,"net",pol); net_v=getattr(val,"net",val)
    K=n*n+1; d=192
    x=torch.zeros(8,K,4,device="cuda")
    def fwd(net,B):
        # the encoder is the cost; call the underlying looped layer the way the net does
        X=torch.randn(B,K,d,device="cuda")
        import nn_labeler.model as M
        A=torch.ones(B,K,K,device="cuda")*0  # placeholder adjacency: dense mask path not exercised here
        lay=None
        for mod in net.modules():
            if type(mod).__name__=="LoopedLayer": lay=mod; break
        torch.cuda.synchronize(); t0=time.time()
        with torch.no_grad():
            for _ in range(4):                    # recurrence 4
                X=lay(X, None, None)              # all-to-all path (upper bound on cost)
        torch.cuda.synchronize(); return time.time()-t0
    tf8=fwd(net_v,8); tf64=fwd(net_v,64)
    print(f"{C}: K={K}  load {tl:.2f}s | propose {tp:.2f}s -> {nc} candidates | "
          f"one encoder pass (rec 4): batch 8 {tf8*1000:.0f} ms, batch 64 {tf64*1000:.0f} ms "
          f"({tf64/tf8:.1f}x for 8x the work)")
PY
