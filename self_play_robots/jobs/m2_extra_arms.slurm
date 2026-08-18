#!/bin/bash
# Audit-requested M2 controls (results/audit_claims_2026-08-18.md): the strongest A*
# control -- arena f (child) + best-at-budget -- and MCTS with a small k=5 tree
# fully enumerated is what "best-at-budget" is; run both size-free arms at both sizes.
#   sbatch -J spr-m2x self_play_robots/jobs/m2_extra_arms.slurm <POLICY> <VALUE>
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1
BP="$1"; BV="$2"
for cfg in g24r4 g16r4; do
  INST=scaling/data/${cfg}/bench.solved.jsonl; [ "$cfg" = g16r4 ] && INST=eval/data/bench450.jsonl
  out=results/m2/sizefree_mixed_warm_${cfg}_astar_child_best.json; [ -s "$out" ] && continue
  python -m spr.arena bench --arm "m2x_${cfg}_astar_child_best" --config "$cfg" --instances "$INST" --policy "$BP" --value "$BV" \
      --driver spr --flags "--search spr_astar --f-mode child --prefix-check --best-at-budget" --width 8 --threads 2 --device cuda --out "$out" \
      || echo "SPR M2X FAILED $cfg"
done
echo "SPR M2X DONE $(date +%F' '%T)"
