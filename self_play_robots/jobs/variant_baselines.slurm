#!/bin/bash
# Variants lab -- baselines on the UNSEEN exam (once): the frozen seed nets
# (mix_b2mix_iter2, B2 anytime = every arm's warm start), the supervised
# per-size backward pair (its own recorded protocol: base vocab + prefix-check),
# and the supervised forward MoveNet is NOT run here (spr.fwd has no lean-board
# path yet -- wave 2; its graded g24r4 rows are on file for context).
#   sbatch -J spr-var-base self_play_robots/jobs/variant_baselines.slurm
#SBATCH -A open-37-42
#SBATCH -p qgpu
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project/open-37-42/petrhyner/MCTS_evolution/runs/spr/%x-%j.out
set -uo pipefail
ml purge
ml Python/3.11.5-GCCcore-13.2.0
REPO=/scratch/project/open-37-42/petrhyner/MCTS_evolution
source /scratch/project/open-37-42/petrhyner/venv/bin/activate
cd "$REPO/self_play_robots"
export PYTHONPATH="$REPO/supervised_valuenet:$REPO/self_play_robots" PYTHONUNBUFFERED=1
RES="$REPO/self_play_robots/results/variants/baselines"; mkdir -p "$RES"
EXAM="$REPO/self_play_robots/results/variants/exam/g24r4_unseen.jsonl"
EXAMB="$REPO/self_play_robots/results/variants/exam/boards_g24r4"
SEEDNETS="$REPO/self_play_robots/results/selfplay/mix_b2mix_iter2/nets.txt"
P0=$(grep '^policy=' "$SEEDNETS" | cut -d= -f2-); V0=$(grep '^value=' "$SEEDNETS" | cut -d= -f2-)
SUPP="$REPO/supervised_valuenet/scaling/runs/g24r4/backward-policy/lightning_logs/version_0/checkpoints"
SUPV="$REPO/supervised_valuenet/scaling/runs/g24r4/backward-value/lightning_logs/version_0/checkpoints"
BPP=$(ls -t "$SUPP"/epoch=*.ckpt 2>/dev/null | head -1); BVV=$(ls -t "$SUPV"/epoch=*.ckpt 2>/dev/null | head -1)
echo "SPR VARBASE start $(date +%F' '%T) job=$SLURM_JOB_ID"
run() { local out="$RES/$1.json"; shift; [ -s "$out" ] && { echo "SKIP $out"; return; }
  python -m spr.arena bench "$@" --width 8 --threads 2 --device cuda --out "$out" \
    || echo "SPR VARBASE FAILED $out"; }
run seed_nets_unseen --arm varbase_seed_unseen --config g24r4 --instances "$EXAM" \
    --policy "$P0" --value "$V0" --driver spr \
    --flags "--search arena_astar --vocab b2 --anytime --boards lean" --env-dir "$EXAMB"
if [ -n "$BPP" ] && [ -n "$BVV" ]; then
  run supervised_persize_unseen --arm varbase_sup_unseen --config g24r4 --instances "$EXAM" \
      --policy "$BPP" --value "$BVV" --driver spr \
      --flags "--search arena_astar --prefix-check --arch persize --boards lean" --env-dir "$EXAMB"
else
  echo "SPR VARBASE WARN: supervised per-size ckpts not found under scaling/runs/g24r4"
fi
echo "SPR VARBASE DONE $(date +%F' '%T)"
