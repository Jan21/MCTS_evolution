#!/bin/bash
# M2 -- search beats greed at inference (PROBLEM.md section 8): same M1 nets,
# same pinned exam, same budget (1200 expansions, k=5, prefix-check,
# replay-certified), different searches:
#   greedy        the labeler's descent rule (one expansion per decision)   <- the M2 opponent
#   astar_child   spr A*, arena f (fixed_g(child)+ctg), first certified
#   astar_parent  spr A*, label-consistent f (fixed_g(parent)+ctg), first certified
#   astar_best    astar_parent + best-at-budget (anytime, cheapest certified)
#   mcts_min      PUCT c=1.5 min-backup, best-at-budget
#   mcts_mean     PUCT c=1.5 mean-backup, best-at-budget
# (arena_astar = the M1 gate row already exists in results/m1/.)
#
#   sbatch -J spr-m2 self_play_robots/jobs/m2_search_arms.slurm <TAG> <POLICY.ckpt|DIR> <VALUE.ckpt|DIR> [cfgs="g24r4 g16r4"] [device=cuda]
# Outputs results/m2/<TAG>_<cfg>_<arm>.json + paired tables. Cost basis:
# 6 arms x 2 configs, each <=30 min 8-wide on GPU -> ~4-6 h = 0.5-0.75 nh.
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1
TAG="$1"; P="$2"; V="$3"; CFGS="${4:-g24r4 g16r4}"; DEV="${5:-cuda}"
best_ckpt() { if [ -d "$1" ]; then ls -t "$1"/lightning_logs/version_*/checkpoints/epoch=*.ckpt | head -1; else echo "$1"; fi; }
BP=$(best_ckpt "$P"); BV=$(best_ckpt "$V")
[ -s "$BP" ] && [ -s "$BV" ] || { echo "FATAL: checkpoints"; exit 1; }
echo "SPR M2 $TAG start $(date +%F' '%T) job=$SLURM_JOB_ID policy=$BP value=$BV cfgs=$CFGS"
mkdir -p results/m2
declare -A FLAGS=(
  [greedy]="--search greedy --prefix-check"
  [astar_child]="--search spr_astar --f-mode child --prefix-check"
  [astar_parent]="--search spr_astar --f-mode parent --prefix-check"
  [astar_best]="--search spr_astar --f-mode parent --prefix-check --best-at-budget"
  [mcts_min]="--search mcts --mcts-backup min --prefix-check --best-at-budget"
  [mcts_mean]="--search mcts --mcts-backup mean --prefix-check --best-at-budget"
)
rc_all=0
for cfg in $CFGS; do
  INST=scaling/data/${cfg}/bench.solved.jsonl; [ "$cfg" = g16r4 ] && INST=eval/data/bench450.jsonl
  for arm in greedy astar_child astar_parent astar_best mcts_min mcts_mean; do
    out="results/m2/${TAG}_${cfg}_${arm}.json"
    [ -s "$out" ] && { echo "SPR M2 SKIP $cfg $arm"; continue; }
    t0=$SECONDS
    python -m spr.arena bench --arm "m2_${TAG}_${cfg}_${arm}" --config "$cfg" --instances "$INST" \
        --policy "$BP" --value "$BV" --driver spr --flags "${FLAGS[$arm]}" --width 8 --threads 2 \
        --device "$DEV" --out "$out" || { echo "SPR M2 BENCH FAILED $cfg $arm"; rc_all=1; continue; }
    echo "SPR M2 bench $cfg $arm took $((SECONDS-t0))s"
  done
  # paired tables vs greedy (the M2 gate) and vs the arena A*
  for arm in astar_child astar_parent astar_best mcts_min mcts_mean; do
    a="results/m2/${TAG}_${cfg}_${arm}.json"; b="results/m2/${TAG}_${cfg}_greedy.json"
    [ -s "$a" ] && [ -s "$b" ] && python -m spr.gate compare --a "$a" --b "$b" > "results/m2/${TAG}_${cfg}_${arm}.vs_greedy.json"
  done
done
echo "SPR M2 DONE $TAG rc=$rc_all $(date +%F' '%T)"
exit $rc_all
