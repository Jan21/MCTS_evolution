#!/bin/bash
set -u
SCRATCH=/tmp/claude-1010/-mnt-raid-data-Hyner-Petr-MCTS-evolution/6bc085eb-eb79-4a20-97d5-e670e4d4c89c/scratchpad
cd /mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet
export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONPATH=.

python3 - <<'PYEOF'
import pickle, os
bad = 0
for cfg in ("environments_g32r4", "environments_g24r8"):
    cdir = os.path.join(cfg, "cache")
    if not os.path.isdir(cdir):
        continue
    for f in os.listdir(cdir):
        p = os.path.join(cdir, f)
        try:
            with open(p, "rb") as fh:
                pickle.load(fh)
        except Exception:
            os.remove(p); bad += 1
print("corrupt caches removed:", bad, flush=True)
PYEOF
echo "[lanes] cache scan done"

( # g32r4 lane
export RR_GRID=32 RR_ROBOTS=4 RR_WALLS=192 RR_ENV_DIR=/mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet/environments_g32r4
BP=$(ls scaling/runs/g32r4/backward-policy/lightning_logs/*/checkpoints/epoch*.ckpt | head -1)
BV=$(ls scaling/runs/g32r4/backward-value/lightning_logs/*/checkpoints/epoch*.ckpt | head -1)
python3 -m eval.compare --instances scaling/data/g32r4/bench.solved.jsonl --expansions 1200 --k 5 --backward-policy $BP --backward-value $BV --forward-ckpts lightning_logs/version_45/checkpoints/epoch=3-step=158564.ckpt --backward-prefix-check --device cpu --out scaling/results/g32r4/comparison.json --md scaling/results/g32r4/COMPARISON.md
echo "[lanes] G32R4_GRADED_DONE"
python3 -m eval.compare --instances scaling/data/g32r4/bench.unsolved.jsonl --expansions 1200 --k 5 --backward-policy $BP --backward-value $BV --forward-ckpts lightning_logs/version_45/checkpoints/epoch=3-step=158564.ckpt --backward-prefix-check --device cpu --out scaling/results/g32r4/comparison_ungraded.json --md /dev/null
echo "[lanes] G32R4_UNGRADED_DONE"
) > $SCRATCH/g32r4_results2.log 2>&1 &

( # g24r8 lane: bench first (single process, seeded), then compares
python3 -m scaling.bench --config g24r8 --per-board 3 --seed 1
echo "[lanes] BENCH24R8_DONE"
export RR_GRID=24 RR_ROBOTS=8 RR_WALLS=108 RR_ENV_DIR=/mnt/raid/data/Hyner_Petr/MCTS_evolution/supervised_valuenet/environments_g24r8
BP=$(ls scaling/runs/g24r8/backward-policy/lightning_logs/*/checkpoints/epoch*.ckpt | head -1)
BV=$(ls scaling/runs/g24r8/backward-value/lightning_logs/*/checkpoints/epoch*.ckpt | head -1)
mkdir -p scaling/results/g24r8
python3 -m eval.compare --instances scaling/data/g24r8/bench.solved.jsonl --expansions 1200 --k 5 --backward-policy $BP --backward-value $BV --forward-ckpts lightning_logs/version_47/checkpoints/epoch=3-step=115144.ckpt --backward-prefix-check --device cpu --out scaling/results/g24r8/comparison.json --md scaling/results/g24r8/COMPARISON.md
echo "[lanes] G24R8_GRADED_DONE"
python3 - <<'PYEOF'
import json
sk = set()
for line in open("scaling/data/g24r8/bench.solved.jsonl"):
    r = json.loads(line); sk.add((r["env_id"], tuple(map(tuple, r["positions"])), r["target_idx"], tuple(r["target"])))
with open("scaling/data/g24r8/bench.unsolved.jsonl", "w") as out:
    for line in open("scaling/data/g24r8/bench.jsonl"):
        r = json.loads(line)
        k = (r["env_id"], tuple(map(tuple, r["positions"])), r["target_idx"], tuple(r["target"]))
        if k not in sk:
            r["d_star"] = 0; out.write(json.dumps(r) + "\n")
PYEOF
python3 -m eval.compare --instances scaling/data/g24r8/bench.unsolved.jsonl --expansions 1200 --k 5 --backward-policy $BP --backward-value $BV --forward-ckpts lightning_logs/version_47/checkpoints/epoch=3-step=115144.ckpt --backward-prefix-check --device cpu --out scaling/results/g24r8/comparison_ungraded.json --md /dev/null
echo "[lanes] G24R8_UNGRADED_DONE"
) > $SCRATCH/g24r8_results2.log 2>&1 &
wait
echo "[lanes] ALL FINAL LANES DONE"
