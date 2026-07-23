#!/bin/bash
set -u
SCRATCH=/tmp/claude-1010/-mnt-raid-data-Hyner-Petr-MCTS-evolution/6bc085eb-eb79-4a20-97d5-e670e4d4c89c/scratchpad
cd /mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet
export PYTHONPATH=.
G=$($SCRATCH/claim_gpu.sh b1_retrain)
echo "[b1rt] claimed GPU $G — policy first (stable trainer), then value warm-started"
CUDA_VISIBLE_DEVICES=$G python3 -m train.policy_tf --data nn/data/combined_b1.jsonl --epochs 25 > $SCRATCH/b1_policy_train.log 2>&1
echo "[b1rt] policy done ($?)"
CUDA_VISIBLE_DEVICES=$G python3 - > $SCRATCH/b1_value_train.log 2>&1 <<'PYEOF'
# value net: warm-start from the known-good value_v2 (cold retrains proved unstable
# across seeds); fine-tune on the B1-vocabulary labels
import sys, torch
torch.manual_seed(11)
import train.looped_pc as lp
_orig = lp.LoopedValueNet
class WarmValueNet(_orig):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        try:
            ck = torch.load("checkpoints_backward/value_v2.ckpt", map_location="cpu")
            self.load_state_dict(ck["state_dict"])
            print("[b1rt] warm-started from value_v2", flush=True)
        except Exception as e:
            print(f"[b1rt] warm start failed ({e}); cold init", flush=True)
lp.LoopedValueNet = WarmValueNet
sys.argv = ["looped_pc", "--data", "nn/data/combined_b1.jsonl", "--epochs", "20", "--warmup", "0"]
lp.main()
PYEOF
echo "[b1rt] value done ($?)"
rm -rf /tmp/claude-1010/gpu_claims/gpu$G
echo "[b1rt] B1 RETRAIN DONE"
