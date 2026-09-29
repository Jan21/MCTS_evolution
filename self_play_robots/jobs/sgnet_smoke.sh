#!/bin/bash
set -uo pipefail
ml purge; ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
NETS=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/selfplay/mix_b2mix_iter3
POL=${POLICY:-$NETS/policy/lightning_logs/version_0/checkpoints/epoch=2-step=2361.ckpt}
VAL=${VALUE:-$NETS/value/lightning_logs/version_0/checkpoints/epoch=0-step=1573.ckpt}
ARCH=${ARCH:-sizefree}; TAG=${TAG:-}
CFG=${CFG:-g40r4}
# the config's RR_* env, exactly as the project's own jobs set it
eval "$(python -c "
from scaling.configs import get, shell_env
print(shell_env(get('$CFG')))" 2>/dev/null || python -c "
import os
from scaling.configs import get, env
for k,v in env(get('$CFG')).items(): print(f'export {k}={v}')")"
export RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR
echo "RR_GRID=$RR_GRID RR_ROBOTS=$RR_ROBOTS RR_ENV_DIR=$RR_ENV_DIR"
OFF=${OFF:-0}; sed -n "$((OFF+1)),$((OFF+LIM))p" "$REPO/supervised_valuenet/scaling/data/$CFG/ladder.jsonl" > /tmp/sg_${CFG}_${LIM}_${OFF}.jsonl
mkdir -p results/subgoal/sgnet
python -m spr.bench --instances /tmp/sg_${CFG}_${LIM}_${OFF}.jsonl --policy "$POL" --value "$VAL" \
  --search arena_astar --vocab b2 --anytime --expansions 1200 --k 5 \
  --boards lean --device cuda --dump-moves --arch "$ARCH" \
  --out results/subgoal/sgnet/${CFG}_${LIM}_${OFF}${TAG}.json 2>&1 | grep -vE "^W0|UserWarning" | tail -12
