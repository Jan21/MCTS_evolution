#!/bin/bash
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/profpol-%j.out
set -uo pipefail
ml purge; ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
CFG=${CFG:-g96r4}
eval "$(python -c "
from scaling.configs import get, env
for k,v in env(get('$CFG')).items(): print(f'export {k}={v}')")"
export RR_GRID RR_ROBOTS RR_WALLS RR_ENV_DIR
sed -n "${IDX:-2}p" "$REPO/supervised_valuenet/scaling/data/$CFG/ladder.jsonl" > /tmp/profpol_inst.jsonl
nvidia-smi -L
python subgoal/prof_policy.py /tmp/profpol_inst.jsonl ${EXP:-8} 2>&1 | grep -vE "^W0|UserWarning|warnings.warn"
