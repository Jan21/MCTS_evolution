#!/bin/bash
#SBATCH --job-name=sz-smoke
#SBATCH --account=open-37-42
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/project/open-37-42/janhula/mcts_stage5_probe/runs/sz-smoke-%j.out
set -uo pipefail
ml purge; ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/janhula/mcts_stage5_probe
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
CK=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/subgoal/stage4_c_r3/ctg-epoch=2-val_mae_all=1.5468.ckpt
O=results/subgoal/scaling; mkdir -p $O
echo "##### REGRESSION: 16x16 through the new config path, must still be 95.3% territory"
python -m subgoal.stage3b --config g16r4 --arm net --ckpt "$CK" --k 100 --bound tight \
  --fin-cap 0 --limit 40 --concurrency 40 --out $O/_sz_g16.json 2>&1 | tail -4
echo
echo "##### SIZE-AGNOSTIC TEST: does the 16x16-trained net work at 24x24 at all?"
python -m subgoal.stage3b --config g24r4 --bench scaling/data/g24r4/bench.solved.jsonl \
  --arm net --ckpt "$CK" --k 100 --bound tight \
  --fin-cap 0 --limit 40 --concurrency 40 --out $O/_sz_g24_fin0.json 2>&1 | tail -4
echo
echo "##### and with the finisher"
python -m subgoal.stage3b --config g24r4 --bench scaling/data/g24r4/bench.solved.jsonl \
  --arm net --ckpt "$CK" --k 100 --bound tight \
  --fin-cap 4 --limit 40 --concurrency 40 --out $O/_sz_g24_fin4.json 2>&1 | tail -4
echo
echo "##### control: relaxation at 24x24, no network, so the net's contribution is visible"
python -m subgoal.stage3b --config g24r4 --bench scaling/data/g24r4/bench.solved.jsonl \
  --arm relaxed --k 100 --bound tight \
  --fin-cap 4 --limit 40 --concurrency 40 --out $O/_sz_g24_rel4.json 2>&1 | tail -4
