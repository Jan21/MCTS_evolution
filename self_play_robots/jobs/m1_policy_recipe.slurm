#!/bin/bash
# M1 policy-recipe probe: the pe=none SizeFreePolicyNet peaked at epoch 0 and
# then degraded on both corpora (g16: val_regret 4.74 -> 7.75 over 25 epochs at
# lr 3e-4; g24: 4.21 -> ~5.0) while the per-size PolicyTF improves steadily.
# Four 8-epoch variants on the g16r4 legacy corpus, same seed:
#   A lr 1e-4 + clip 1.0        B pe=sin2d lr 3e-4 + clip 1.0
#   C encoder init from the labeler (prod_v1_s11 LoopedLayer) + lr 1e-4 + clip 1.0
#   D lr 3e-4 + clip 1.0 (does clipping alone stabilize?)
#   sbatch -J spr-m1-recipe self_play_robots/jobs/m1_policy_recipe.slurm
# ~10 min each on 1 GPU (qgpu_exp, 1 h) ~ 0.1 nh.
#SBATCH -A open-37-42
#SBATCH -p qgpu_exp
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
DATA="--data g16r4=nn/data/combined.jsonl"
LAB=$REPO/self_play_robots/assets/labeler_prod_v1_s11.ckpt
run() { local tag="$1"; shift; local d="$REPO/runs/spr/m1/recipe_g16_${tag}"; [ -f "$d/DONE" ] && return 0
  python -m spr.train --system policy $DATA --epochs 8 --batch-size 8 --torch-seed 21 --num-workers 4 --device cuda --out-dir "$d" "$@" \
    && touch "$d/DONE"; echo "SPR RECIPE $tag rc=$? $(date +%T)"; }
run A_lr1e4_clip   --lr 1e-4 --grad-clip 1.0
run B_sin2d_clip   --pe sin2d --lr 3e-4 --grad-clip 1.0
run C_encinit_lr1e4_clip --init-encoder "$LAB" --lr 1e-4 --grad-clip 1.0
run D_lr3e4_clip   --lr 3e-4 --grad-clip 1.0
python3 - <<'PY'
import csv, glob
for f in sorted(glob.glob("/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/m1/recipe_g16_*/lightning_logs/version_0/metrics.csv")):
    rows=[r for r in csv.DictReader(open(f)) if r.get("val_regret")]
    print("SPR RECIPE", f.split("/")[-4], " ".join(f"e{r['epoch']}:{float(r['val_regret']):.2f}/{float(r['val_recall@5']):.2f}" for r in rows))
PY
echo "SPR RECIPE DONE $(date +%F' '%T)"
